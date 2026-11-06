"""Upload data and launch a Bedrock RFT job through the OpenAI-compatible API.

Prerequisites: pip install openai boto3 aws-bedrock-token-generator
Usage:
  python run_rft.py --lambda-arn arn:aws:lambda:us-west-2:<ACCOUNT>:function:ticket-reward
  python run_rft.py --job-id <JOB_ID>          # only poll an existing job
API shape follows the AWS blog "Reinforcement fine-tuning on Amazon Bedrock with
OpenAI-Compatible APIs: a technical walkthrough" (2026-03). Check the Bedrock
user guide for the current parameters before running.
"""
import argparse
import time

from aws_bedrock_token_generator import provide_token
from openai import OpenAI


def client_for(region):
    return OpenAI(base_url=f"https://bedrock-mantle.{region}.api.aws/v1", api_key=provide_token(region=region))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="us-west-2")
    ap.add_argument("--model", default="openai.gpt-oss-20b", help="or qwen.qwen3-32b")
    ap.add_argument("--train", default="train.jsonl")
    ap.add_argument("--lambda-arn")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr-mult", type=float, default=0.5, help="the blog recommends < 1.0 for stability")
    ap.add_argument("--job-id", help="skip creation and only poll this job")
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

    seen = set()
    while True:
        client = client_for(args.region)  # refresh short-term token each loop
        for ev in reversed(client.fine_tuning.jobs.list_events(fine_tuning_job_id=job_id, limit=50).data):
            if ev.id in seen:
                continue
            seen.add(ev.id)
            data = getattr(ev, "data", None) or {}
            if isinstance(data, dict) and "critic_rewards_mean" in data:
                print(f"step {data.get('step')}/{data.get('total_steps')}  reward={data['critic_rewards_mean']:.3f}  "
                      f"entropy={data.get('actor_entropy', 0):.3f}  len={data.get('response_length_mean', 0):.0f}")
            else:
                print(ev.message)
        job = client.fine_tuning.jobs.retrieve(job_id)
        if job.status in ("succeeded", "failed", "cancelled"):
            print("status:", job.status, "model:", job.fine_tuned_model)
            break
        time.sleep(60)


if __name__ == "__main__":
    main()
