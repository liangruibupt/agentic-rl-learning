"""Shared helpers for the held-out set: Bedrock Converse call with a call counter."""
import json
import os
import threading

import boto3
from botocore.config import Config

REGION = "us-west-2"
HERE = os.path.dirname(os.path.abspath(__file__))
CALLS_FILE = os.path.join(HERE, "calls.json")
_lock = threading.Lock()
_client = None

# Defect letter -> judge rubric item it should break (judge item ids only, never its wording).
DEFECT_ITEM = {
    "a": "confirm_before_write",
    "b": "cite_policy_on_refusal",
    "c": "offer_alternative",
    "d": "no_fabrication",
    "e": "no_false_self_claim",
    "f": "polite_tone",
}
ITEMS = list(DEFECT_ITEM.values())


def client():
    global _client
    if _client is None:
        _client = boto3.client("bedrock-runtime", region_name=REGION,
                               config=Config(read_timeout=120, retries={"max_attempts": 3}))
    return _client


def bump(role, n=1):
    with _lock:
        d = json.load(open(CALLS_FILE)) if os.path.exists(CALLS_FILE) else {}
        d[role] = d.get(role, 0) + n
        json.dump(d, open(CALLS_FILE, "w"), indent=1)
        return sum(d.values())


def total_calls():
    with _lock:
        return sum(json.load(open(CALLS_FILE)).values()) if os.path.exists(CALLS_FILE) else 0


def converse(model_id, system, user, role, max_tokens=3000, temperature=None):
    if total_calls() >= 1800:
        raise SystemExit("budget: 1800 Bedrock calls reached")
    cfg = {"maxTokens": max_tokens}
    if temperature is not None:
        cfg["temperature"] = temperature
    bump(role)
    resp = client().converse(
        modelId=model_id,
        system=[{"text": system}],
        messages=[{"role": "user", "content": [{"text": user}]}],
        inferenceConfig=cfg,
    )
    return "".join(c.get("text", "") for c in resp["output"]["message"]["content"])


def extract_json(text):
    s, e = text.find("{"), text.rfind("}")
    if s < 0 or e <= s:
        return None
    try:
        return json.loads(text[s:e + 1])
    except json.JSONDecodeError:
        return None
