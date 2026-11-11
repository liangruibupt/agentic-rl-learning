"""Check an RFT dataset before uploading: schema, size limits, leakage, template overlap, label and difficulty mix.

Usage:
  python check_dataset.py --train train.jsonl --test test.jsonl [--dev dev.jsonl]
Exit code 1 if any ERROR is found; WARN lines are things to read, not blockers.

Limits (Bedrock user guide, "Prepare data for open-weight models", checked 2026-11-11):
training data 100-20K examples, OpenAI chat-completions "messages" plus "reference_answer";
other top-level fields are passed to the reward Lambda under "metadata".

Optional per-row fields written by make_dataset.py and used here when present:
  template_id  which core phrasing produced the ticket -> template overlap by ID
  slice        train | dev | new_phrasing | ood        -> per-slice summary
  formats      e.g. "amt:cn,oid:hyphen,lang:mixed"     -> which slot formats the OOD slice adds
  wait_days    waiting time in days (shipping)         -> rule-2 tag without parsing 两周 / 半个月
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
ANGRY_WORDS = ("投诉", "律师", "报警", "曝光", "complaint", "lawyer", "police", "expose")
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


def oid_key(s):
    """Order IDs compare without case, spaces, hyphens or '#', as the system prompt asks."""
    return re.sub(r"[\s\-#]", "", s or "").upper()


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
    if oid is not None and (not isinstance(oid, str) or oid_key(oid) not in oid_key(user_text(row))):
        errs.append("order_id not found in the ticket text")
    amt = ref.get("amount")
    if amt is not None and (isinstance(amt, bool) or not isinstance(amt, (int, float))):
        errs.append("amount must be a number or null")
    return errs


def difficulty_tags(row):
    """Which priority rule decides the label, plus traps. Specific to the ticket task."""
    ref, text = row.get("reference_answer") or {}, user_text(row)
    amt = ref.get("amount")
    if "wait_days" in row:
        days = [row["wait_days"]] if row["wait_days"] is not None else []
    else:
        days = [int(d) for d in re.findall(r"(\d+)\s*天", text)]
    tags = []
    if any(w in text.lower() for w in ANGRY_WORDS):
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
    if amt is not None and amt > 1000 and any(w in text.lower() for w in ANGRY_WORDS):
        tags.append("both_rule1_triggers")
    if ref.get("category") == "shipping" and days and max(days) in (7, 8):
        tags.append("boundary_days")         # "超过 7 天": 7 is low, 8 is medium
    return tags


def level(tags):
    if {"boundary_amount", "exchange_with_amount", "boundary_days"} & set(tags):
        return "hard"
    return "easy" if tags[0] == "rule3_default" else "medium"


def dist(rows, key):
    c = collections.Counter(key(r) for r in rows)
    n = max(1, len(rows))
    return {k: c[k] / n for k in sorted(c, key=str)}


def max_dist_gap(a, b):
    keys = set(a) | set(b)
    return max((abs(a.get(k, 0) - b.get(k, 0)) for k in keys), default=0.0)


def format_tokens(row):
    return {t for t in str(row.get("formats") or "").split(",") if t}


def class_counts(rows):
    c = collections.Counter((r.get("reference_answer") or {}).get("category") for r in rows)
    return {k: c.get(k, 0) for k in sorted(CATEGORIES)}


def distinct_per_class(rows):
    d = collections.defaultdict(set)
    for r in rows:
        d[(r.get("reference_answer") or {}).get("category")].add(normalize_text(user_text(r)))
    return {k: len(d.get(k, ())) for k in sorted(CATEGORIES)}


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
            "class_counts": class_counts(rows),
            "distinct_texts_per_class": distinct_per_class(rows),
            "slices": dict(collections.Counter(r.get("slice") for r in rows if "slice" in r)),
            "template_ids": len({r["template_id"] for r in rows if "template_id" in r}),
        }
    train = splits.get("train", [])
    train_exact = {normalize_text(user_text(r)) for r in train}
    train_tmpl = {template_key(user_text(r)) for r in train}
    train_ids = {r["template_id"] for r in train if "template_id" in r}
    train_formats = set().union(*(format_tokens(r) for r in train)) if train else set()
    for name, rows in splits.items():
        if name == "train" or not rows:
            continue
        leaks = [i for i, r in enumerate(rows) if normalize_text(user_text(r)) in train_exact]
        if leaks:
            errors.append(f"{name}: {len(leaks)} tickets also in train after normalization (first: {leaks[:5]})")
        shared = sum(template_key(user_text(r)) in train_tmpl for r in rows) / len(rows)
        stats[name]["template_overlap_with_train"] = shared
        with_id = [r for r in rows if "template_id" in r]
        if with_id and train_ids:
            id_shared = sum(r["template_id"] in train_ids for r in with_id) / len(with_id)
            stats[name]["template_id_overlap_with_train"] = id_shared
            if id_shared > 0.5:
                warnings.append(f"{name}: {id_shared:.0%} of tickets use a train phrasing (template_id)")
        ood = [r for r in rows if r.get("slice") == "ood"]
        if ood:
            unseen = [format_tokens(r) - train_formats for r in ood]
            stats[name]["ood"] = {
                "rows": len(ood),
                "class_counts": class_counts(ood),
                "unseen_format_rate": sum(bool(u) for u in unseen) / len(ood),
                "unseen_formats": dict(sorted(collections.Counter(t for u in unseen for t in u).items())),
                "template_id_overlap_with_train": (sum(r.get("template_id") in train_ids for r in ood) / len(ood)),
            }
            if stats[name]["ood"]["unseen_format_rate"] < 0.9:
                warnings.append(f"{name}: only {stats[name]['ood']['unseen_format_rate']:.0%} of the ood slice uses a "
                                "slot format or language not seen in train")
            rare = {k: v for k, v in stats[name]["ood"]["class_counts"].items() if v < min_class_rows}
            if rare:
                warnings.append(f"{name}: ood slice has too few rows to measure category {rare} (< {min_class_rows} each)")
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
                  + (f" template_overlap={st['template_overlap_with_train']:.3f}" if "template_overlap_with_train" in st else "")
                  + (f" template_id_overlap={st['template_id_overlap_with_train']:.3f}" if "template_id_overlap_with_train" in st else ""))
            print(f"    class_counts={st['class_counts']} distinct_texts={st['distinct_texts_per_class']} "
                  f"template_ids={st['template_ids']} slices={st['slices']}")
            if "ood" in st:
                o = st["ood"]
                print(f"    ood: rows={o['rows']} class_counts={o['class_counts']} unseen_format_rate={o['unseen_format_rate']:.3f} "
                      f"template_id_overlap={o['template_id_overlap_with_train']:.3f}")
                print(f"         unseen_formats={o['unseen_formats']}")
        for w in rep["warnings"]:
            print("WARN ", w)
        for e in rep["errors"]:
            print("ERROR", e)
        print("OK" if not rep["errors"] else f"{len(rep['errors'])} error(s)")
    sys.exit(1 if rep["errors"] else 0)


if __name__ == "__main__":
    main()
