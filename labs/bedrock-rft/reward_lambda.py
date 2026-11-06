"""Reward function for the ticket-extraction RFT lab.

Deploy as an AWS Lambda function and pass its ARN as the grader when creating
the Amazon Bedrock reinforcement fine-tuning job.

Contract (from the aws-samples Bedrock RFT reward function sample):
  input : a list of items, each with "id", "messages" (including the model's
          assistant reply), "reference_answer", and optional "metadata".
  output: a list of {"id", "aggregate_reward_score", "reward_components"}.

Scoring (field-level, so partial answers still give GRPO a gradient):
  category 0.3 + priority 0.2 + order_id 0.3 + amount 0.2 = 1.0
  unparseable JSON -> 0.0; extra keys -> -0.1 each; final score clipped to [0, 1].
"""
import json
import re

WEIGHTS = {"category": 0.3, "priority": 0.2, "order_id": 0.3, "amount": 0.2}
EXTRA_KEY_PENALTY = 0.1
MAX_CHARS = 4000  # keep regex work bounded so the Lambda stays fast


def _last_assistant_text(messages):
    text = ""
    for msg in messages or []:
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, list):  # content-parts format
                content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
            text = content or ""
    return text


def _reference(item):
    """reference_answer may sit at top level or under metadata; may be a dict or JSON string."""
    ref = item.get("reference_answer")
    if ref is None:
        ref = (item.get("metadata") or {}).get("reference_answer")
    if isinstance(ref, str):
        try:
            ref = json.loads(ref)
        except json.JSONDecodeError:
            return None
    return ref if isinstance(ref, dict) else None


def extract_json(text):
    """Return the last top-level JSON object in the text, or None."""
    text = (text or "")[-MAX_CHARS:]
    # Drop reasoning blocks some models emit before the answer.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    candidates = re.findall(r"\{[^{}]*\}", text)
    for cand in reversed(candidates):
        try:
            obj = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


def _norm(v):
    if v is None:
        return None
    if isinstance(v, str):
        s = v.strip()
        if s.lower() in ("", "null", "none"):
            return None
        return s.upper() if re.fullmatch(r"[A-Za-z]\d+", s) else s.lower()
    return v


def _amount_equal(pred, gold):
    if gold is None:
        return pred is None or _norm(pred) is None
    try:
        return abs(float(pred) - float(gold)) < 0.01
    except (TypeError, ValueError):
        return False


def score_one(response_text, reference):
    components = {k: 0.0 for k in WEIGHTS}
    components["format"] = 0.0
    components["extra_keys"] = 0.0
    if not reference:
        return 0.0, components
    pred = extract_json(response_text)
    if pred is None:
        return 0.0, components
    components["format"] = 1.0
    for field, w in WEIGHTS.items():
        if field == "amount":
            ok = _amount_equal(pred.get("amount"), reference.get("amount"))
        else:
            ok = _norm(pred.get(field)) == _norm(reference.get(field))
        components[field] = w if ok else 0.0
    extra = [k for k in pred if k not in WEIGHTS]
    components["extra_keys"] = -EXTRA_KEY_PENALTY * len(extra)
    total = sum(components[f] for f in WEIGHTS) + components["extra_keys"]
    return max(0.0, min(1.0, total)), components


def lambda_handler(event, context):
    items = event if isinstance(event, list) else event.get("trajectories", [])
    results = []
    for item in items:
        item_id = item.get("id") or item.get("task_id", "unknown")
        try:
            score, comps = score_one(_last_assistant_text(item.get("messages")), _reference(item))
        except Exception as exc:  # never fail the batch; a bad item scores 0
            print(f"scoring error for {item_id}: {exc}")
            score, comps = 0.0, {}
        results.append({"id": item_id, "aggregate_reward_score": score, "reward_components": comps})
    return results
