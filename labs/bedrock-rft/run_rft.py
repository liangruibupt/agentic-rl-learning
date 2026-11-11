"""Upload data and launch a Bedrock RFT job through the OpenAI-compatible API.

Prerequisites: pip install openai boto3 aws-bedrock-token-generator
Usage:
  python run_rft.py --lambda-arn arn:aws:lambda:us-west-2:<ACCOUNT>:function:ticket-reward
  python run_rft.py --job-id <JOB_ID> --save-events events.json   # only poll an existing job

Facts checked on 2026-11-11 against the Bedrock user guide ("Fine-tune open-weight models
using OpenAI-compatible APIs", "Create and manage fine-tuning jobs for open-weight models")
and the AWS walkthrough blog (2026):
  - models: openai.gpt-oss-20b, qwen.qwen3-32b; region us-west-2; endpoint
    https://bedrock-mantle.us-west-2.api.aws/v1; auth with a Bedrock API key (short-term
    keys recommended, so this script refreshes the token on every poll).
  - documented hyperparameters for this API: n_epochs, batch_size, learning_rate_multiplier
    (guide example 1 / 4 / 1.0; the blog recommends a multiplier below 1.0). The number of
    responses sampled per prompt is not a documented parameter here.
  - "suffix" is not supported (comment in the guide's own example).
  - metric events have type "metrics" and data keys step, total_steps, critic_rewards_mean,
    critic_advantages_mean, actor_entropy, actor_pg_loss, actor_grad_norm,
    response_length_mean (blog event example).
  - pricing: $80 per training hour, $1.95 per trained model per month (Bedrock pricing page).
"""
import argparse
import json
import time

METRIC_KEYS = ("step", "total_steps", "critic_rewards_mean", "critic_advantages_mean", "actor_entropy",
               "actor_pg_loss", "actor_grad_norm", "response_length_mean")


def client_for(region):
    from aws_bedrock_token_generator import provide_token
    from openai import OpenAI
    return OpenAI(base_url=f"https://bedrock-mantle.{region}.api.aws/v1", api_key=provide_token(region=region))


def all_events(client, job_id, page=100):
    """Every event of a job, oldest first (the guide paginates with `after`)."""
    out, after = [], None
    while True:
        kw = {"fine_tuning_job_id": job_id, "limit": page}
        if after:
            kw["after"] = after
        resp = client.fine_tuning.jobs.list_events(**kw)
        out.extend(resp.data)
        if not getattr(resp, "has_more", False) or not resp.data:
            break
        after = resp.data[-1].id
    return sorted(out, key=lambda e: getattr(e, "created_at", 0) or 0)


def metric_line(data):
    return (f"step {data.get('step')}/{data.get('total_steps')}  reward={data['critic_rewards_mean']:.3f}  "
            f"entropy={data.get('actor_entropy', 0):.3f}  len={data.get('response_length_mean', 0):.0f}  "
            f"grad_norm={data.get('actor_grad_norm', 0):.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="us-west-2")
    ap.add_argument("--model", default="openai.gpt-oss-20b", help="or qwen.qwen3-32b")
    ap.add_argument("--train", default="train.jsonl")
    ap.add_argument("--lambda-arn")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr-mult", type=float, default=0.5, help="the AWS blog recommends < 1.0 for stability")
    ap.add_argument("--job-id", help="skip creation and only poll this job")
    ap.add_argument("--save-events", help="write metric events as JSON (paste into the lesson 11 chart)")
    ap.add_argument("--interval", type=int, default=60)
    args = ap.parse_args()
    client = client_for(args.region)

    job_id = args.job_id
    if not job_id:
        if not args.lambda_arn:
            ap.error("--lambda-arn is required when creating a job")
        with open(args.train, "rb") as f:
            file_id = client.files.create(file=f, purpose="fine-tune").id
        print("uploaded training file:", file_id)
        job = client.fine_tuning.jobs.create(
            model=args.model,
            training_file=file_id,
            extra_body={"method": {"type": "reinforcement", "reinforcement": {
                "grader": {"type": "lambda", "lambda": {"function": args.lambda_arn}},
                "hyperparameters": {"n_epochs": args.epochs, "batch_size": args.batch_size,
                                    "learning_rate_multiplier": args.lr_mult},
            }}},
        )
        job_id = job.id
        print("created job:", job_id)

    seen, metrics = set(), []
    while True:
        client = client_for(args.region)  # refresh short-term token each loop
        for ev in all_events(client, job_id):
            if ev.id in seen:
                continue
            seen.add(ev.id)
            data = getattr(ev, "data", None) or {}
            if isinstance(data, dict) and "critic_rewards_mean" in data:
                metrics.append({k: data.get(k) for k in METRIC_KEYS})
                print(metric_line(data))
            else:
                print(ev.message)
        if args.save_events:
            with open(args.save_events, "w", encoding="utf-8") as f:
                json.dump({"data": [{"type": "metrics", "data": m} for m in metrics]}, f, indent=1)
        job = client.fine_tuning.jobs.retrieve(job_id)
        if job.status in ("succeeded", "failed", "cancelled"):
            print("status:", job.status, "model:", job.fine_tuned_model)
            try:
                for ck in client.fine_tuning.jobs.checkpoints.list(fine_tuning_job_id=job_id).data:
                    print("checkpoint step", ck.step_number, ck.fine_tuned_model_checkpoint)
            except Exception as exc:  # checkpoints are optional for this script
                print("could not list checkpoints:", exc)
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
