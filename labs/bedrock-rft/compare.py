"""Compare base vs fine-tuned results saved by evaluate.py --save, with paired bootstrap CIs.

Usage:
  python compare.py base.jsonl ft.jsonl [--boot 10000] [--seed 0] [--slice ood]
Each input line is one (task_id, run) record written by evaluate.py.

Method (same as lesson 10 section 8.3):
  - average the R runs of each test item first, then resample ITEMS (not runs) with replacement;
  - pair the two models: every bootstrap draw uses the same item indices for both;
  - 95% CI = 2.5% and 97.5% percentiles of the resampled differences.
Also prints the within-item correlation rho (ICC) of exact match, which feeds the power
calculator in lesson 11 section 10.2.
"""
import argparse
import json
import random
import statistics

from reward_lambda import WEIGHTS

METRICS = ["all_correct", "strict_exact", "format"] + [f"field_{f}" for f in WEIGHTS] + ["score", "chars"]


def load_records(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def record_from(task_id, run, score, comps, text, model=None):
    """Build one evaluate.py record from score_one() output."""
    fields = {f: int(comps.get(f, 0.0) == w) for f, w in WEIGHTS.items()}
    all_correct = int(all(fields.values()))
    return {"model": model, "task_id": task_id, "run": run, "score": round(score, 4),
            "all_correct": all_correct,
            "strict_exact": int(all_correct and comps.get("format_penalty", 0.0) == 0.0),
            "format": int(comps.get("format", 0.0) == 1.0), "fields": fields, "chars": len(text or "")}


def _value(rec, metric):
    if metric.startswith("field_"):
        return rec["fields"][metric[6:]]
    return rec[metric]


def per_item(records, metric):
    """{task_id: mean over runs}."""
    acc = {}
    for r in records:
        acc.setdefault(r["task_id"], []).append(_value(r, metric))
    return {k: sum(v) / len(v) for k, v in acc.items()}


def summarize(records):
    return {m: statistics.mean(per_item(records, m).values()) for m in METRICS}


def paired_bootstrap(base, ft, metrics=METRICS, boot=10000, seed=0):
    """Return {metric: (base_mean, ft_mean, diff, lo, hi)} over task_ids present in both."""
    ids = sorted(set(r["task_id"] for r in base) & set(r["task_id"] for r in ft))
    if not ids:
        raise ValueError("no common task_id between the two files")
    n = len(ids)
    cols = {}
    for m in metrics:
        b, f = per_item(base, m), per_item(ft, m)
        cols[m] = ([b[i] for i in ids], [f[i] for i in ids])
    rng = random.Random(seed)
    draws = {m: [] for m in metrics}
    for _ in range(boot):
        idx = [rng.randrange(n) for _ in range(n)]
        for m, (b, f) in cols.items():
            draws[m].append(sum(f[i] - b[i] for i in idx) / n)
    out = {}
    for m, (b, f) in cols.items():
        d = sorted(draws[m])
        lo, hi = d[int(0.025 * (boot - 1))], d[int(0.975 * (boot - 1))]
        out[m] = (sum(b) / n, sum(f) / n, (sum(f) - sum(b)) / n, lo, hi)
    return out


def icc(records, metric="all_correct"):
    """rho_hat ~ (Var(m) - p(1-p)/R) / (p(1-p)(1-1/R)) from per-item means m over R runs (lesson 10 section 8)."""
    acc = {}
    for r in records:
        acc.setdefault(r["task_id"], []).append(_value(r, metric))
    runs = {len(v) for v in acc.values()}
    if len(runs) != 1 or runs == {1}:
        return None
    R = runs.pop()
    m = [sum(v) / R for v in acc.values()]
    p = sum(m) / len(m)
    if p in (0.0, 1.0):
        return None
    var = sum((x - p) ** 2 for x in m) / len(m)
    return (var - p * (1 - p) / R) / (p * (1 - p) * (1 - 1 / R))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base")
    ap.add_argument("ft")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slice", help="only items of one test slice, e.g. new_phrasing or ood")
    args = ap.parse_args()
    base, ft = load_records(args.base), load_records(args.ft)
    if args.slice:
        base, ft = ([r for r in recs if r.get("slice") == args.slice] for recs in (base, ft))
    res = paired_bootstrap(base, ft, boot=args.boot, seed=args.seed)
    print(f"{'metric':<16}{'base':>9}{'ft':>9}{'diff':>9}   95% CI (paired bootstrap, {args.boot} draws)")
    for m, (b, f, d, lo, hi) in res.items():
        fmt = "{:>9.1f}" if m == "chars" else "{:>9.3f}"
        print(f"{m:<16}" + fmt.format(b) + fmt.format(f) + fmt.format(d) + f"   [{lo:.3f}, {hi:.3f}]")
    for name, recs in (("base", base), ("ft", ft)):
        r = icc(recs)
        print(f"ICC rho of all_correct ({name}): " + ("n/a" if r is None else f"{r:.2f}"))


if __name__ == "__main__":
    main()
