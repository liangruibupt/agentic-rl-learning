"""Score a base or fine-tuned model on the held-out test set with the same reward.

Usage:
  python evaluate.py --model openai.gpt-oss-20b --runs 3 --save base.jsonl
  python evaluate.py --model <FINE_TUNED_MODEL_ID> --runs 3 --save ft.jsonl
  python compare.py base.jsonl ft.jsonl          # paired bootstrap CIs
Reports mean reward, exact match (all four fields right), strict exact match (also no
format penalty), JSON format rate and per-field accuracy, then exact match per test slice
("new_phrasing" = held-out phrasings, "ood" = held-out formats / languages; see make_dataset.py).
Saved records carry the slice, so compare.py --slice ood gives the CI for one slice.

Serving settings: evaluate both models with the SAME temperature and max_tokens, set to
what you will serve with. Training rewards come from sampled rollouts, so they are not a
substitute for this evaluation.

Inference (Bedrock user guide, "Create and manage fine-tuning jobs for open-weight models",
checked 2026-11-11): the fine-tuned model ID from jobs.retrieve(job_id).fine_tuned_model is
called directly through the OpenAI-compatible Chat Completions or Responses API on
bedrock-mantle.<region>.api.aws; no endpoint deployment is needed. Checkpoints listed by
jobs.checkpoints.list() expose fine_tuned_model_checkpoint, which can be evaluated the same way.
"""
import argparse
import json
import statistics

from compare import record_from, summarize
from reward_lambda import WEIGHTS, score_one


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="us-west-2")
    ap.add_argument("--model", required=True)
    ap.add_argument("--test", default="test.jsonl")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=1.0, help="evaluate at the setting you will serve at")
    ap.add_argument("--max-tokens", type=int, default=None, help="same value for base and fine-tuned runs")
    ap.add_argument("--save", help="write one JSON line per (item, run) for compare.py")
    args = ap.parse_args()

    from aws_bedrock_token_generator import provide_token
    from openai import OpenAI

    rows = [json.loads(line) for line in open(args.test, encoding="utf-8")]
    records = []
    for run in range(args.runs):
        # short-term Bedrock API keys expire; refresh once per run
        client = OpenAI(base_url=f"https://bedrock-mantle.{args.region}.api.aws/v1", api_key=provide_token(region=args.region))
        scores = []
        for r in rows:
            kw = {"model": args.model, "messages": r["messages"], "temperature": args.temperature}
            if args.max_tokens:
                kw["max_tokens"] = args.max_tokens
            out = client.chat.completions.create(**kw)
            text = out.choices[0].message.content or ""
            s, comps = score_one(text, r["reference_answer"])
            scores.append(s)
            rec = record_from(r.get("task_id"), run, s, comps, text, model=args.model)
            rec["slice"] = r.get("slice")
            records.append(rec)
        run_recs = [x for x in records if x["run"] == run]
        print(f"run {run + 1}: mean_reward={statistics.mean(scores):.3f} "
              f"exact_match={statistics.mean(x['all_correct'] for x in run_recs):.3f}")
    summ = summarize(records)
    print(f"\nmodel={args.model} runs={args.runs} n={len(rows)} temperature={args.temperature} max_tokens={args.max_tokens}")
    print(f"mean_reward {summ['score']:.3f}  exact_match {summ['all_correct']:.3f}  "
          f"strict_exact {summ['strict_exact']:.3f}  format {summ['format']:.3f}  mean_chars {summ['chars']:.0f}")
    print("field accuracy:", {k: round(summ['field_' + k], 3) for k in WEIGHTS})
    for sl in sorted({x["slice"] for x in records if x.get("slice")}):
        sub = summarize([x for x in records if x.get("slice") == sl])
        n_sl = sum(r.get("slice") == sl for r in rows)
        print(f"slice {sl:<13} n={n_sl:<4} exact_match {sub['all_correct']:.3f}  mean_reward {sub['score']:.3f}")
    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            for x in records:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print("saved", len(records), "records to", args.save)


if __name__ == "__main__":
    main()
