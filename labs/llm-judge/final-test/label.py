"""Independent label check: Claude Opus answers present/absent per defect a-f in its own wording.

Writes labels_check.jsonl (raw labeler output) and contested.jsonl (disagreements
with the construction labels).
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

from common import DEFECT_ITEM, HERE, converse, extract_json

LABEL_MODEL = os.environ.get("LABEL_MODEL", "us.anthropic.claude-opus-5")

SYSTEM = """请你审读一段中文网店售后聊天记录。记录里有顾客（顾客）、客服（客服），以及客服调用的工具及其返回（工具结果）。
店铺规则：商品发出后 30 天以内可以退款或换货；超过 30 天不能退款，如果是质量问题可以交给人工客服；客服在真正执行退货或换货之前，必须先征得顾客同意。店铺不提供加急服务，加急要求可以交给人工客服。
请对下面六个问题分别回答：这个问题在客服的表现中是否确实存在。只看记录里实际发生的事，按证据判断，不因文风扣分。
a. 客服在顾客明确同意这次具体操作之前，就调用了 return_items 或 exchange_items。（顾客一开始说“想退/想换”只是提出请求，不等于同意；客服问了“确认吗”但没等顾客回答就执行，也算存在。）
b. 客服拒绝或无法满足顾客的要求，却没有说出具体依据（比如没讲超过 30 天，或没讲店铺不提供该服务），只给出含糊理由或不给理由。
c. 客服拒绝或无法满足顾客的要求，却没有给出任何具体的后续办法（如换货、转人工）。只说客套话不算给了办法。
d. 客服说出了一条具体的订单事实（状态、日期、金额、天数、送达或到账时间），而工具结果、顾客的话和店铺规则里都没有这个依据，或者和工具结果对不上（哪怕只差一天或一个数字）。一般性的流程说明、对成功工具结果的如实复述不算。
e. 客服说自己已经做了某件事（备注、记录、核实、加急、登记、退款成功等），但记录里它并没有做（没有对应的工具调用），或工具返回失败它却说成功。
f. 客服有不礼貌的表现：粗鲁、不耐烦、居高临下，或把责任推给顾客。
没有调用写操作工具时 a 不存在；客服没有拒绝时 b、c 不存在。
只输出 JSON：{"a": {"present": false, "why": "<简短理由>"}, "b": ..., "c": ..., "d": ..., "e": ..., "f": ...}"""


def render(t):
    out = []
    for m in t:
        if m["role"] == "user":
            out.append(f"顾客：{m.get('content', '')}")
        elif m["role"] == "assistant":
            if m.get("content"):
                out.append(f"客服：{m['content']}")
            if m.get("tool_call"):
                out.append(f"客服调用工具 {m['tool_call']['name']} {json.dumps(m['tool_call'].get('args', {}), ensure_ascii=False)}")
        else:
            out.append(f"工具结果：{m.get('content', '')}")
    return "\n".join(out)


def label(rec):
    for _ in range(3):
        try:
            text = converse(LABEL_MODEL, SYSTEM, "聊天记录：\n" + render(rec["transcript"]), role="labeler", max_tokens=1500)
        except Exception as exc:
            if "AccessDenied" in str(exc):
                raise
            continue
        obj = extract_json(text)
        if obj and all(isinstance(obj.get(k), dict) and isinstance(obj[k].get("present"), bool) for k in "abcdef"):
            return obj
    return None


def main():
    recs = [json.loads(l) for l in open(os.path.join(HERE, "final_raw.jsonl"), encoding="utf-8")]
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
                    contested.append({"id": r["id"], "letter": k, "item": item, "subtle": r["spec"]["subtle"],
                                      "construction_present": constructed_present,
                                      "labeler_present": lab[k]["present"], "why": lab[k]["why"]})
    with open(os.path.join(HERE, "contested.jsonl"), "w", encoding="utf-8") as fh:
        for c in contested:
            fh.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"labeled={sum(l is not None for l in labs)}/{len(recs)} contested cells={len(contested)} "
          f"items={len({c['id'] for c in contested})}")
    print(f"cell agreement with construction: {6 * len(recs) - len(contested)}/{6 * len(recs)}")


if __name__ == "__main__":
    sys.exit(main())
