"""One-shot final test: score the frozen judge (GPT-6 Luna, votes=3) and the MockJudge baseline
on the frozen final_test.jsonl. Run ONCE.

  python run_final.py --dry-run   # plumbing check with MockJudge standing in for Bedrock (no AWS calls)
  python run_final.py             # writes results-final.txt, errors.jsonl, judge_raw.jsonl, probe_raw.jsonl
Positive class = defect present (judge item false).
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

PARENT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, PARENT)  # judge.py, calibrate.py

from calibrate import PROBES, cohen_kappa, with_suffix
from common import BUDGET, HERE, bump, total_calls
from judge import RUBRIC, BedrockJudge, MockJudge, parse_verdict

MODEL = "us.openai.gpt-6-luna"
VOTES = 3
FROZEN_SHA256 = "40128897cb08da82bcd015f6e4917400d44e68b01ae621238af020417d288dd5"
JUDGE_SHA256 = "cd5d6fc97f3fffeb1b618a565be0f62361e4641be8e41914ffcea13d9ac6f6df"  # origin/main 429fb95


class RecordingJudge(BedrockJudge):
    """Same judge; also keeps each vote's raw text and counts Bedrock calls."""

    def raw(self, transcript):
        if total_calls() >= BUDGET:
            raise RuntimeError(f"budget: {BUDGET} Bedrock calls reached")
        bump("judge")
        return super().raw(transcript)

    def judge_raw(self, transcript):
        raws, verdicts = [], []
        for _ in range(self.votes):
            try:
                text = self.raw(transcript)
            except Exception as exc:  # same neutral handling as BedrockJudge.judge_once
                raws.append({"error": str(exc)})
                continue
            v = parse_verdict(text)
            raws.append({"text": text, "parsed": v is not None})
            if v is not None:
                verdicts.append(v)
        if not verdicts:
            return None, raws
        return {r["id"]: sum(v[r["id"]] for v in verdicts) * 2 > len(verdicts) for r in RUBRIC}, raws


class DryJudge:
    """MockJudge behind the RecordingJudge interface; for --dry-run only."""
    use_temperature = None

    def judge_raw(self, transcript):
        return MockJudge().judge(transcript), [{"dry": True}]


def kappa(a, b):
    n = len(a)
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe == 1:
        return float("nan") if any(x != y for x, y in zip(a, b)) else 1.0
    return cohen_kappa(a, b)


def boot_ci(a, b, rng, n=2000):
    idx = range(len(a))
    vals = []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        vals.append(sum(a[i] == b[i] for i in s) / len(s))
    vals.sort()
    return vals[int(0.025 * n)], vals[int(0.975 * n) - 1]


def boot_ci_items(recs, verdicts, rng, n=2000):
    """95% CI of the all-6-correct rate, resampling items."""
    ok = [all(r["human"][k["id"]] == v[k["id"]] for k in RUBRIC) for r, v in zip(recs, verdicts) if v is not None]
    vals = sorted(sum(rng.choice(ok) for _ in ok) / len(ok) for _ in range(n))
    return vals[int(0.025 * n)], vals[int(0.975 * n) - 1]


def score_table(name, recs, verdicts, rng, lines):
    lines.append(f"\n== {name} ==")
    lines.append(f"{'item':<24}{'n':>4}{'agree':>7}{'95% CI':>15}{'kappa':>7}{'TP':>4}{'FP':>4}{'FN':>4}{'TN':>4}   misses (FN) | false alarms (FP)")
    for r in RUBRIC:
        k = r["id"]
        rows = [(rec["human"][k], v[k], rec["id"]) for rec, v in zip(recs, verdicts) if v is not None]
        h = [x[0] for x in rows]
        j = [x[1] for x in rows]
        n = len(rows)
        agree = sum(x == y for x, y in zip(h, j)) / n
        lo, hi = boot_ci(h, j, rng)
        tp = sum((not x) and (not y) for x, y in zip(h, j))
        fp = sum(x and (not y) for x, y in zip(h, j))
        fn = sum((not x) and y for x, y in zip(h, j))
        tn = sum(x and y for x, y in zip(h, j))
        misses = ",".join(i for x, y, i in rows if (not x) and y) or "-"
        alarms = ",".join(i for x, y, i in rows if x and (not y)) or "-"
        lines.append(f"{k:<24}{n:>4}{agree:>7.2f}{f'[{lo:.2f},{hi:.2f}]':>15}{kappa(h, j):>7.2f}{tp:>4}{fp:>4}{fn:>4}{tn:>4}   {misses} | {alarms}")
    ok = [(rec, v) for rec, v in zip(recs, verdicts) if v is not None]
    exact = sum(all(rec["human"][r["id"]] == v[r["id"]] for r in RUBRIC) for rec, v in ok)
    lo, hi = boot_ci_items(recs, verdicts, rng)
    lines.append(f"items with all 6 correct: {exact}/{len(ok)} [{lo:.2f},{hi:.2f}]; judge failures (None): {len(recs) - len(ok)}")
    # Recall by kind of defect cell, and false alarms on fully clean items.
    by_kind = {}
    for rec, v in ok:
        for item, kind in rec["positive_kind"].items():
            if not rec["human"][item]:
                hit = v[item] is False
                d = by_kind.setdefault(kind, [0, 0, []])
                d[0] += hit
                d[1] += 1
                if not hit:
                    d[2].append(f"{rec['id']}:{item}")
    for kind in ("blatant", "subtle", "unplanned"):
        if kind in by_kind:
            c, t, miss = by_kind[kind]
            lines.append(f"  recall on {kind:<9} defect cells: {c}/{t} = {c / t:.2f}   missed: {','.join(miss) or '-'}")
    clean = [(rec, v) for rec, v in ok if all(rec["human"].values())]
    fa = [rec["id"] for rec, v in clean if not all(v.values())]
    lines.append(f"  clean items with any false alarm: {len(fa)}/{len(clean)}   [{','.join(fa) or '-'}]")
    neg_cells = [(rec, v, k["id"]) for rec, v in ok for k in RUBRIC if rec["human"][k["id"]]]
    lines.append(f"  false-alarm rate over all {len(neg_cells)} defect-absent cells: "
                 f"{sum(v[k] is False for _, v, k in neg_cells) / len(neg_cells):.3f}")


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    data_path = os.path.join(HERE, "final_test.jsonl")
    assert sha256(data_path) == FROZEN_SHA256, "final_test.jsonl changed after freeze"
    assert sha256(os.path.join(PARENT, "judge.py")) == JUDGE_SHA256, "judge.py differs from the frozen judge"
    judge_commit = subprocess.run(["git", "log", "-1", "--format=%H", "origin/main", "--", "judge.py"],
                                  cwd=PARENT, capture_output=True, text=True).stdout.strip()
    all_recs = [json.loads(l) for l in open(data_path, encoding="utf-8")]
    recs = [r for r in all_recs if r["label_source"] != "ambiguous"]
    judge = DryJudge() if args.dry_run else RecordingJudge(MODEL, "us-west-2", votes=VOTES)
    suffix = "-dry" if args.dry_run else ""
    pool = ThreadPoolExecutor(6)
    res = list(pool.map(lambda r: judge.judge_raw(r["transcript"]), recs))
    with open(os.path.join(HERE, f"judge_raw{suffix}.jsonl"), "w", encoding="utf-8") as fh:
        for r, (v, raws) in zip(recs, res):
            fh.write(json.dumps({"id": r["id"], "verdict": v, "votes": raws}, ensure_ascii=False) + "\n")
    by_id = {r["id"]: x for r, x in zip(recs, res)}
    luna = [by_id[r["id"]][0] for r in recs]
    mock = [MockJudge().judge(r["transcript"]) for r in recs]

    rng = random.Random(2026)
    lines = [f"final test set: {len(all_recs)} items, scored {len(recs)} (excluded ambiguous: "
             f"{','.join(r['id'] for r in all_recs if r['label_source'] == 'ambiguous') or '-'})",
             f"final_test.jsonl sha256 {FROZEN_SHA256}",
             f"judge.py sha256 {JUDGE_SHA256}, last commit touching it on origin/main {judge_commit}",
             f"judge: {MODEL if not args.dry_run else 'DRY RUN (MockJudge)'} votes={VOTES}; positive class = defect present (judge says false)"]
    score_table(f"GPT-6 Luna ({MODEL}, votes={VOTES})" if not args.dry_run else "DRY", recs, luna, rng, lines)
    score_table("MockJudge (keyword baseline)", recs, mock, rng, lines)

    with open(os.path.join(HERE, f"errors{suffix}.jsonl"), "w", encoding="utf-8") as fh:
        for r in recs:
            v, raws = by_id[r["id"]]
            wrong = None if v is None else [k["id"] for k in RUBRIC if v[k["id"]] != r["human"][k["id"]]]
            if v is None or wrong:
                fh.write(json.dumps({"id": r["id"], "spec": r["spec"], "label_source": r["label_source"],
                                     "positive_kind": r["positive_kind"], "label": r["human"], "verdict": v,
                                     "wrong_items": wrong, "votes": raws}, ensure_ascii=False) + "\n")

    prng = random.Random(5)
    probe_recs = prng.sample(recs, 30)
    lines.append(f"\n== bias probes on 30 random scored items (seed 5), votes={VOTES} ==")
    lines.append("false->true = judge talked into passing an item it failed without the probe (bias);")
    lines.append("'new misses' = of those, cells where the label says the defect is present.")
    probe_dump = []
    for name, text in PROBES.items():
        probed = list(pool.map(lambda r: judge.judge_raw(with_suffix(r["transcript"], text)), probe_recs))
        up, down, fail, new_miss = {}, {}, 0, []
        for r, (v2, raws) in zip(probe_recs, probed):
            v = by_id[r["id"]][0]
            probe_dump.append({"probe": name, "id": r["id"], "verdict": v2, "votes": raws})
            if v is None or v2 is None:
                fail += 1
                continue
            for k in v:
                if not v[k] and v2[k]:
                    up.setdefault(r["id"], []).append(k)
                    if not r["human"][k]:
                        new_miss.append(f"{r['id']}:{k}")
                elif v[k] and not v2[k]:
                    down.setdefault(r["id"], []).append(k)
        fmt = lambda d: "; ".join(f"{i}:{'+'.join(ks)}" for i, ks in sorted(d.items())) or "none"
        lines.append(f"  {name:<11} items with false->true flips: {len(up)}/30 [{fmt(up)}]  new misses: {','.join(new_miss) or 'none'}")
        lines.append(f"  {'':<11} true->false: {sum(len(x) for x in down.values())} [{fmt(down)}]  failures: {fail}")
    with open(os.path.join(HERE, f"probe_raw{suffix}.jsonl"), "w", encoding="utf-8") as fh:
        for p in probe_dump:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    calls = json.load(open(os.path.join(HERE, "calls.json")))
    lines.append(f"\nBedrock calls (whole final test): {calls} total={sum(calls.values())}")
    lines.append(f"judge temperature accepted: {judge.use_temperature}")
    out = "\n".join(lines)
    open(os.path.join(HERE, f"results-final{suffix}.txt"), "w", encoding="utf-8").write(out + "\n")
    print(out)


if __name__ == "__main__":
    sys.exit(main())
