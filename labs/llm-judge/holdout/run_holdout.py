"""Score GPT-6 Luna (and the MockJudge baseline) on the held-out set.

  python run_holdout.py            # writes results-holdout.txt, judge_raw.jsonl, errors.jsonl
Positive class = defect present (judge item false).
"""
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # judge.py, calibrate.py

from calibrate import PROBES, cohen_kappa, with_suffix
from common import HERE, bump, total_calls
from judge import RUBRIC, BedrockJudge, MockJudge, parse_verdict

MODEL = "us.openai.gpt-6-luna"
VOTES = 3


class RecordingJudge(BedrockJudge):
    """Same judge; also keeps each vote's raw text and counts Bedrock calls."""

    def raw(self, transcript):
        if total_calls() >= 1800:
            raise RuntimeError("budget: 1800 Bedrock calls reached")
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


def kappa(a, b):
    n = len(a)
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe == 1:
        return float("nan") if any(x != y for x, y in zip(a, b)) else 1.0
    return cohen_kappa(a, b)


def boot_ci(a, b, rng, n=1000):
    idx = range(len(a))
    vals = []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        vals.append(sum(a[i] == b[i] for i in s) / len(s))
    vals.sort()
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
    lines.append(f"items with all 6 correct: {exact}/{len(ok)}; judge failures (None): {len(recs) - len(ok)}")


def main():
    all_recs = [json.loads(l) for l in open(os.path.join(HERE, "holdout_final.jsonl"), encoding="utf-8")]
    recs = [r for r in all_recs if r["label_source"] != "ambiguous"]
    judge = RecordingJudge(MODEL, "us-west-2", votes=VOTES)
    pool = ThreadPoolExecutor(6)
    res = list(pool.map(lambda r: judge.judge_raw(r["transcript"]), all_recs))
    with open(os.path.join(HERE, "judge_raw.jsonl"), "w", encoding="utf-8") as fh:
        for r, (v, raws) in zip(all_recs, res):
            fh.write(json.dumps({"id": r["id"], "verdict": v, "votes": raws}, ensure_ascii=False) + "\n")
    by_id = {r["id"]: x for r, x in zip(all_recs, res)}
    luna = [by_id[r["id"]][0] for r in recs]
    mock = [MockJudge().judge(r["transcript"]) for r in recs]

    rng = random.Random(11)
    lines = [f"held-out set: {len(all_recs)} items, scored {len(recs)} (excluded ambiguous: "
             f"{','.join(r['id'] for r in all_recs if r['label_source'] == 'ambiguous') or '-'})",
             f"judge: {MODEL} votes={VOTES}; positive class = defect present (judge says false)"]
    score_table(f"GPT-6 Luna ({MODEL}, votes={VOTES})", recs, luna, rng, lines)
    score_table("MockJudge (keyword baseline)", recs, mock, rng, lines)

    # errors.jsonl: every item where Luna disagrees with the label or fails, with raw votes.
    with open(os.path.join(HERE, "errors.jsonl"), "w", encoding="utf-8") as fh:
        for r in recs:
            v, raws = by_id[r["id"]]
            wrong = None if v is None else [k["id"] for k in RUBRIC if v[k["id"]] != r["human"][k["id"]]]
            if v is None or wrong:
                fh.write(json.dumps({"id": r["id"], "spec": r["spec"], "label_source": r["label_source"],
                                     "label": r["human"], "verdict": v, "wrong_items": wrong,
                                     "votes": raws}, ensure_ascii=False) + "\n")

    # Bias probes on a random 30.
    prng = random.Random(5)
    probe_recs = prng.sample(recs, 30)
    lines.append(f"\n== bias probes on 30 random items (seed 5), {MODEL} votes={VOTES} ==")
    lines.append("false->true = judge talked into passing an item it failed without the probe (bias).")
    probe_dump = []
    for name, text in PROBES.items():
        probed = list(pool.map(lambda r: judge.judge_raw(with_suffix(r["transcript"], text)), probe_recs))
        up, down, fail = {}, {}, 0
        for r, (v2, raws) in zip(probe_recs, probed):
            v = by_id[r["id"]][0]
            probe_dump.append({"probe": name, "id": r["id"], "verdict": v2, "votes": raws})
            if v is None or v2 is None:
                fail += 1
                continue
            for k in v:
                if not v[k] and v2[k]:
                    up.setdefault(r["id"], []).append(k)
                elif v[k] and not v2[k]:
                    down.setdefault(r["id"], []).append(k)
        fmt = lambda d: "; ".join(f"{i}:{'+'.join(ks)}" for i, ks in sorted(d.items())) or "none"
        lines.append(f"  {name:<11} items with false->true flips: {len(up)}/30 [{fmt(up)}]")
        lines.append(f"  {'':<11} true->false: {sum(len(x) for x in down.values())} [{fmt(down)}]  failures: {fail}")
    with open(os.path.join(HERE, "probe_raw.jsonl"), "w", encoding="utf-8") as fh:
        for p in probe_dump:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    calls = json.load(open(os.path.join(HERE, "calls.json")))
    lines.append(f"\nBedrock calls so far: {calls} total={sum(calls.values())}")
    lines.append(f"judge temperature accepted: {judge.use_temperature}")
    out = "\n".join(lines)
    open(os.path.join(HERE, "results-holdout.txt"), "w", encoding="utf-8").write(out + "\n")
    print(out)


if __name__ == "__main__":
    sys.exit(main())
