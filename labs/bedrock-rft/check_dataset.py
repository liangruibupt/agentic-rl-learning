"""Check an RFT dataset before uploading: schema, size limits, leakage, label and difficulty mix.

Usage:
  python check_dataset.py --train train.jsonl --test test.jsonl [--dev dev.jsonl]
Exit code 1 if any ERROR is found; WARN lines are things to read, not blockers.

Limits (Bedrock user guide, "Prepare data for open-weight models", checked 2026-11-11):
training data 100-20K examples, OpenAI chat-completions "messages" plus "reference_answer";
other top-level fields are passed to the reward Lambda under "metadata".
"""
import argparse
import collections
import json
import re
import sys
import unicodedata

MIN_ROWS, MAX_ROWS = 100, 20_000
FIELDS = ("category", "priority", "order_id", "amount")
CATEGORIES = {"refund", "exchange", "shipping", "account", "other"}
PRIORITIES = {"high", "medium", "low"}
ANGRY_WORDS = ("投诉", "律师", "报警", "曝光")
_PUNCT = re.compile(r"[\s\W_]+", re.UNICODE)


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    rows.append({"__error__": f"line {n}: {exc}"})
    return rows


def user_text(row):
    for m in reversed(row.get("messages") or []):
        if m.get("role") == "user":
            c = m.get("content", "")
            return "".join(p.get("text", "") for p in c if isinstance(p, dict)) if isinstance(c, list) else str(c)
    return ""


def normalize_text(s):
    """Exact-duplicate key: NFKC, lower case, drop whitespace and punctuation."""
    return _PUNCT.sub("", unicodedata.normalize("NFKC", s or "").lower())


def template_key(s):
    """Near-duplicate key: also mask order IDs and numbers, so only the wording remains."""
    return re.sub(r"[a-z]?\d+", "#", normalize_text(s))


def validate_row(row):
    """Return a list of problems with one row (empty list = OK)."""
    if "__error__" in row:
        return [row["__error__"]]
    errs = []
    msgs = row.get("messages")
    if not isinstance(msgs, list) or not msgs:
        errs.append("messages missing or empty")
    else:
        for m in msgs:
            if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant") or "content" not in m:
                errs.append("bad message entry")
                break
        if not user_text(row).strip():
            errs.append("no user message")
        if msgs[-1].get("role") == "assistant":
            errs.append("last message is assistant: the prompt should end before the model's turn")
    ref = row.get("reference_answer")
    if not isinstance(ref, dict):
        errs.append("reference_answer missing or not an object")
        return errs
    missing = [f for f in FIELDS if f not in ref]
    if missing:
        errs.append(f"reference_answer missing {missing}")
    if ref.get("category") not in CATEGORIES:
        errs.append(f"bad category {ref.get('category')!r}")
    if ref.get("priority") not in PRIORITIES:
        errs.append(f"bad priority {ref.get('priority')!r}")
    oid = ref.get("order_id")
    if oid is not None and (not isinstance(oid, str) or oid not in user_text(row)):
        errs.append("order_id not found in the ticket text")
    amt = ref.get("amount")
    if amt is not None and (isinstance(amt, bool) or not isinstance(amt, (int, float))):
        errs.append("amount must be a number or null")
    return errs


def difficulty_tags(row):
    """Which priority rule decides the label, plus traps. Specific to the ticket task."""
    ref, text = row.get("reference_answer") or {}, user_text(row)
    amt = ref.get("amount")
    days = [int(d) for d in re.findall(r"(\d+)\s*天", text)]
    tags = []
    if any(w in text for w in ANGRY_WORDS):
        tags.append("rule1_keyword")
    elif amt is not None and amt > 1000:
        tags.append("rule1_amount")
    elif amt is not None and 200 <= amt <= 1000:
        tags.append("rule2_amount")
    elif ref.get("category") == "shipping" and days and max(days) > 7:
        tags.append("rule2_shipping_days")
    else:
        tags.append("rule3_default")
    if amt in (200, 1000):
        tags.append("boundary_amount")      # "含" boundaries of the 200-1000 range
    if ref.get("category") == "exchange" and amt is not None:
        tags.append("exchange_with_amount")  # amount is optional for exchange tickets
    if amt is not None and amt > 1000 and any(w in text for w in ANGRY_WORDS):
        tags.append("both_rule1_triggers")
    return tags


def level(tags):
    if {"boundary_amount", "exchange_with_amount"} & set(tags):
        return "hard"
    return "easy" if tags[0] == "rule3_default" else "medium"


def dist(rows, key):
    c = collections.Counter(key(r) for r in rows)
    n = max(1, len(rows))
    return {k: c[k] / n for k in sorted(c, key=str)}


def max_dist_gap(a, b):
    keys = set(a) | set(b)
    return max((abs(a.get(k, 0) - b.get(k, 0)) for k in keys), default=0.0)


def check(splits, gap_warn=0.10, min_class_rows=10):
    """splits: {"train": rows, "test": rows, ...}. Returns {"errors": [...], "warnings": [...], "stats": {...}}."""
    errors, warnings, stats = [], [], {}
    n_train = len(splits.get("train", []))
    if not MIN_ROWS <= n_train <= MAX_ROWS:
        errors.append(f"train has {n_train} rows; Bedrock RFT needs {MIN_ROWS}-{MAX_ROWS:,}")
    for name, rows in splits.items():
        bad = [(i, e) for i, r in enumerate(rows) for e in validate_row(r)]
        for i, e in bad[:10]:
            errors.append(f"{name}[{i}]: {e}")
        if len(bad) > 10:
            errors.append(f"{name}: {len(bad) - 10} more row errors")
        keys = [normalize_text(user_text(r)) for r in rows]
        dup = len(keys) - len(set(keys))
        if dup:
            warnings.append(f"{name}: {dup} duplicate tickets after normalization")
        systems = {json.dumps([m for m in r.get("messages", []) if m.get("role") == "system"], ensure_ascii=False) for r in rows}
        if len(systems) > 1:
            warnings.append(f"{name}: {len(systems)} different system prompts")
        tags = [difficulty_tags(r) for r in rows]
        stats[name] = {
            "rows": len(rows),
            "category": dist(rows, lambda r: (r.get("reference_answer") or {}).get("category")),
            "priority": dist(rows, lambda r: (r.get("reference_answer") or {}).get("priority")),
            "order_id_null": sum((r.get("reference_answer") or {}).get("order_id") is None for r in rows) / max(1, len(rows)),
            "amount_null": sum((r.get("reference_answer") or {}).get("amount") is None for r in rows) / max(1, len(rows)),
            "rule": dict(collections.Counter(t[0] for t in tags)),
            "level": dict(collections.Counter(level(t) for t in tags)),
            "traps": dict(collections.Counter(x for t in tags for x in t[1:])),
            "prompt_chars_max": max((sum(len(str(m.get("content", ""))) for m in r.get("messages", [])) for r in rows), default=0),
        }
    train = splits.get("train", [])
    train_exact = {normalize_text(user_text(r)) for r in train}
    train_tmpl = {template_key(user_text(r)) for r in train}
    for name, rows in splits.items():
        if name == "train" or not rows:
            continue
        leaks = [i for i, r in enumerate(rows) if normalize_text(user_text(r)) in train_exact]
        if leaks:
            errors.append(f"{name}: {len(leaks)} tickets also in train after normalization (first: {leaks[:5]})")
        shared = sum(template_key(user_text(r)) in train_tmpl for r in rows) / len(rows)
        stats[name]["template_overlap_with_train"] = shared
        if shared > 0.5:
            warnings.append(f"{name}: {shared:.0%} of tickets reuse a train template with different numbers/IDs; "
                            "this measures in-distribution accuracy only, add held-out phrasings for generalization")
        for field in ("category", "priority"):
            gap = max_dist_gap(stats["train"][field], stats[name][field])
            if gap > gap_warn:
                warnings.append(f"{name}: {field} mix differs from train by {gap:.0%}")
            counts = {k: round(v * len(rows)) for k, v in stats[name][field].items()}
            labels = CATEGORIES if field == "category" else PRIORITIES
            rare = {k: counts.get(k, 0) for k in sorted(labels) if counts.get(k, 0) < min_class_rows}
            if rare:
                warnings.append(f"{name}: too few rows to measure {field} {rare} (< {min_class_rows} each)")
    names = [n for n in splits if splits[n]]
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if a != "train" and b != "train":
                ka = {normalize_text(user_text(r)) for r in splits[a]}
                kb = {normalize_text(user_text(r)) for r in splits[b]}
                if ka & kb:
                    errors.append(f"{a} and {b} share {len(ka & kb)} tickets")
    return {"errors": errors, "warnings": warnings, "stats": stats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train.jsonl")
    ap.add_argument("--test", default="test.jsonl")
    ap.add_argument("--dev")
    ap.add_argument("--json", action="store_true", help="print the full report as JSON")
    args = ap.parse_args()
    splits = {"train": load(args.train), "test": load(args.test)}
    if args.dev:
        splits["dev"] = load(args.dev)
    rep = check(splits)
    if args.json:
        print(json.dumps(rep, ensure_ascii=False, indent=2))
    else:
        for name, st in rep["stats"].items():
            print(f"[{name}] rows={st['rows']} level={st['level']} rule={st['rule']} traps={st['traps']}")
            print(f"    priority={ {k: round(v, 3) for k, v in st['priority'].items()} } "
                  f"order_id_null={st['order_id_null']:.3f} amount_null={st['amount_null']:.3f}"
                  + (f" template_overlap={st['template_overlap_with_train']:.3f}" if "template_overlap_with_train" in st else ""))
        for w in rep["warnings"]:
            print("WARN ", w)
        for e in rep["errors"]:
            print("ERROR", e)
        print("OK" if not rep["errors"] else f"{len(rep['errors'])} error(s)")
    sys.exit(1 if rep["errors"] else 0)


if __name__ == "__main__":
    main()
