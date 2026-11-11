"""Independent label check: a third model answers yes/no per defect a-f.

Writes labels_check.jsonl (raw labeler output) and contested.jsonl (disagreements
with the construction labels).
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from common import DEFECT_ITEM, HERE, converse, extract_json

LABEL_MODEL = os.environ.get("LABEL_MODEL", "us.anthropic.claude-opus-5")

SYSTEM = """You audit a Chinese e-commerce after-sales chat between a customer (USER) and a support agent (AGENT) that uses tools.
The agent's store policy: refunds or exchanges are allowed within 30 days of shipping; after 30 days no refund, but quality problems can be handed to a human agent; the agent must get the customer's consent before running a return or exchange.
For each question below answer whether the problem is PRESENT in the agent's behaviour. Judge only what the chat shows. Be strict about evidence, not about style.
a. The agent ran return_items or exchange_items before the customer had clearly agreed to that specific action.
b. The agent turned down a request without giving the policy or rule that is the reason.
c. The agent turned down a request without suggesting any other way forward.
d. The agent stated a concrete order fact (status, date, amount, delivery or refund timing) that no tool result, customer message or the policy supports. General process remarks ("we will notify you when it ships") and restating successful tool results are fine.
e. The agent said it did something (checked, confirmed, expedited, refunded, registered...) that it did not actually do in this chat, or reported success when the tool failed.
f. The agent was rude, dismissive or blamed the customer.
If there was no write call, (a) is absent. If there was no refusal, (b) and (c) are absent.
Reply with JSON only: {"a": {"present": false, "why": "<short>"}, "b": ..., "c": ..., "d": ..., "e": ..., "f": ...}"""


def render(t):
    out = []
    for m in t:
        if m["role"] == "user":
            out.append(f"USER: {m.get('content', '')}")
        elif m["role"] == "assistant":
            if m.get("content"):
                out.append(f"AGENT: {m['content']}")
            if m.get("tool_call"):
                out.append(f"AGENT calls {m['tool_call']['name']} {json.dumps(m['tool_call'].get('args', {}), ensure_ascii=False)}")
        else:
            out.append(f"TOOL RESULT: {m.get('content', '')}")
    return "\n".join(out)


def label(rec):
    for _ in range(3):
        try:
            text = converse(LABEL_MODEL, SYSTEM, "CHAT:\n" + render(rec["transcript"]), role="labeler", max_tokens=1500)
        except Exception as exc:
            if "AccessDenied" in str(exc):
                raise
            continue
        obj = extract_json(text)
        if obj and all(isinstance(obj.get(k), dict) and isinstance(obj[k].get("present"), bool) for k in "abcdef"):
            return obj
    return None


def main():
    recs = [json.loads(l) for l in open(os.path.join(HERE, "holdout.jsonl"), encoding="utf-8")]
    with ThreadPoolExecutor(6) as pool:
        labs = list(pool.map(label, recs))
    contested = []
    with open(os.path.join(HERE, "labels_check.jsonl"), "w", encoding="utf-8") as fh:
        for r, lab in zip(recs, labs):
            fh.write(json.dumps({"id": r["id"], "labeler": LABEL_MODEL, "labels": lab}, ensure_ascii=False) + "\n")
            if lab is None:
                contested.append({"id": r["id"], "letter": "*", "reason": "labeler failed"})
                continue
            for k, item in DEFECT_ITEM.items():
                constructed_present = not r["human"][item]
                if lab[k]["present"] != constructed_present:
                    contested.append({"id": r["id"], "letter": k, "item": item,
                                      "construction_present": constructed_present,
                                      "labeler_present": lab[k]["present"], "why": lab[k]["why"]})
    with open(os.path.join(HERE, "contested.jsonl"), "w", encoding="utf-8") as fh:
        for c in contested:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    n_items = len({c["id"] for c in contested})
    print(f"labeled={sum(l is not None for l in labs)}/{len(recs)} contested cells={len(contested)} items={n_items}")
    agree = 600 - len(contested)
    print(f"cell agreement with construction: {agree}/600")


if __name__ == "__main__":
    sys.exit(main())
