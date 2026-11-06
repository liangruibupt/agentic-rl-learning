"""Score a base or fine-tuned model on the held-out test set with the same reward.

Usage:
  python evaluate.py --model openai.gpt-oss-20b --runs 3
  python evaluate.py --model <FINE_TUNED_MODEL_ID> --runs 3
Reports mean reward, exact-match rate (all four fields right) and per-field accuracy.
"""
import argparse
import json
import statistics

from aws_bedrock_token_generator import provide_token
from openai import OpenAI

from reward_lambda import WEIGHTS, score_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="us-west-2")
    ap.add_argument("--model", required=True)
    ap.add_argument("--test", default="test.jsonl")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=1.0, help="evaluate at the setting you will serve at")
    args = ap.parse_args()
    client = OpenAI(base_url=f"https://bedrock-mantle.{args.region}.api.aws/v1", api_key=provide_token(region=args.region))
    rows = [json.loads(line) for line in open(args.test, encoding="utf-8")]

    run_means, run_exact = [], []
    field_hits = {k: 0 for k in WEIGHTS}
    total = 0
    for run in range(args.runs):
        scores, exact = [], 0
        for r in rows:
            out = client.chat.completions.create(model=args.model, messages=r["messages"], temperature=args.temperature)
            text = out.choices[0].message.content or ""
            s, comps = score_one(text, r["reference_answer"])
            scores.append(s)
            exact += int(s >= 0.999)
            for k, w in WEIGHTS.items():
                field_hits[k] += int(comps.get(k, 0) == w)
            total += 1
        run_means.append(statistics.mean(scores))
        run_exact.append(exact / len(rows))
        print(f"run {run + 1}: mean_reward={run_means[-1]:.3f} exact_match={run_exact[-1]:.3f}")
    print(f"\nmodel={args.model} runs={args.runs} n={len(rows)}")
    print(f"mean_reward {statistics.mean(run_means):.3f}  exact_match {statistics.mean(run_exact):.3f}")
    print("field accuracy:", {k: round(v / total, 3) for k, v in field_hits.items()})


if __name__ == "__main__":
    main()
