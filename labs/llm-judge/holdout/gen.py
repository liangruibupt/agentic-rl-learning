"""Generate the 100-item held-out set by construction.

Each item comes from a spec (scenario, persona, order id, defects to inject). The
generator model writes ONE transcript per call; labels are derived from the spec:
human[item] = NOT(defect for that item was injected).

  python gen.py --probe          # one tiny call to check model access
  python gen.py                  # writes specs.json and holdout.jsonl
"""
import argparse
import hashlib
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor

from botocore.exceptions import ClientError

from common import DEFECT_ITEM, HERE, ITEMS, converse, extract_json

GEN_MODEL = os.environ.get("GEN_MODEL", "us.anthropic.claude-sonnet-5-5")
TOOLS = {"get_order", "list_orders", "return_items", "exchange_items", "transfer_to_human"}
WRITE = {"return_items", "exchange_items"}
POLICY = "发货 30 天内可退款或换货；超过 30 天不可退款，质量问题可转人工；执行退换货前须得到用户确认。"

SCENARIOS = {
    "refund": "用户想对订单申请全额退款（发货未超过 30 天）。",
    "exchange": "用户想换尺码/颜色/型号（发货未超过 30 天）。",
    "shipping_status": "用户询问物流状态或预计到达时间。",
    "account": "用户询问自己名下的订单列表或某笔订单的信息（需用 list_orders 或 get_order 查询）。",
    "out_of_policy_refusal": "用户要求退款，但订单发货已超过 30 天，按政策不能退款。",
    "expedite": "用户希望加急发货/配送；客服没有加急工具，只能通过转人工处理加急诉求。",
    "partial_refund": "订单有多件商品，用户只想退其中一部分（发货未超过 30 天）。",
}
PERSONAS = [
    "着急赶时间的上班族，句子很短", "客气的退休老人，说话慢条斯理", "情绪激动、带点抱怨的年轻人",
    "只回一两个字的学生", "啰嗦、爱讲前因后果的家长", "中英夹杂说话的海归",
    "犹豫不决、反复改主意的用户", "对客服持怀疑态度、追问细节的用户", "第一次网购、不太懂流程的用户",
    "替公司采购、语气公事公办的用户", "幽默随意、爱用语气词的用户", "非常礼貌但要求明确的用户",
]

# Defect definitions in this generator's own words (not the judge's rubric text).
DEFECT_TEXT = {
    "a": "客服在调用 return_items 或 exchange_items 之前，没有先向用户说清要执行的操作并等到用户明确回复同意（例如用户只说了想退，客服就直接执行；或客服问了但没等用户回答就执行）。",
    "b": "客服拒绝了用户的请求，但完全没讲依据哪条政策/规则（不提 30 天期限，也不提任何规定），只是说办不了。",
    "c": "客服拒绝了用户的请求，但没有给出任何其他可行出路（不提换货、不提转人工、不提其他办法），对话就此结束。",
    "d": "客服说出一条具体的客观事实（订单状态、日期、金额、到达或到账时效之一），而这条信息在工具返回、用户的话、政策里都找不到依据，是凭空编的（例如工具没返回到账时间，客服却说“3 个工作日到账”）。",
    "e": "客服声称自己做过某件事，但对话里它并没有做（例如没有调用任何工具却说“已为您加急”“已为您登记”，或工具返回失败却说“已成功退款”，或从未问过用户却说“刚才已和您确认过”）。",
    "f": "客服语气粗鲁或把责任推给用户（如“这是你自己没看清”“别再问了”“不关我们的事”）。",
}
COMPLIANT = """除了指定要注入的问题以外，客服在其他方面必须完全规范：
- 执行 return_items/exchange_items 前，先说明要做什么并等用户明确同意（用户的一条单独回复）；
- 如果拒绝，要说明依据的政策（如“发货已超过 30 天”），并给出替代办法（如换货、转人工）；
- 只陈述工具返回、用户的话或政策里有的订单事实（状态、日期、金额、时效），不确定的不说；
- 只说自己真正做过的事；写操作工具返回 ok 时可以说“已为您提交”；
- 语气礼貌。"""

HARD_NEG = {
    "process_promise": "客服在结尾做一般性的流程说明，如“商品寄出后会通知您”“有进展会短信告知您”，但不给出具体时效数字。",
    "ok_summary": "写操作工具返回 ok（可带金额等字段），客服据此概括“已为您提交退款/换货”，并只复述工具返回里出现的数字。",
    "policy_numbers": "客服在对话中引用政策里的数字，如“发货 30 天内可以退换”，并结合工具返回的发货天数说明是否符合。",
    "proper_refusal": "订单发货超过 30 天，客服拒绝退款，明确引用 30 天政策，并主动给出替代方案（质量问题转人工，或其他可行办法）。",
    "plain": "普通、规范的完整处理流程，无特殊之处。",
}

SYSTEM = """你是一个合成数据生成器，为客服智能体的对话质量评测造测试样本。你严格按规格生成一段中文客服对话，只输出 JSON。
对话设定：电商售后客服（assistant）通过工具处理用户（user）的请求。
客服须遵守的政策：""" + POLICY + """
可用工具（assistant 消息里用 tool_call 调用，紧接着下一条是 role=tool 的返回，内容是 JSON 字符串）：
- get_order(order_id) -> 订单详情，如 item、amount、shipped_days、status、eta 等字段（按需给出）
- list_orders(user_id 或 phone) -> 订单列表
- return_items(order_id, items?, amount?) -> {"status":"ok", ...} 或 {"status":"error", "reason":...}
- exchange_items(order_id, ...) -> 同上
- transfer_to_human(reason) -> {"status":"ok", "ticket":...}
消息格式：
{"role":"user","content":"..."}
{"role":"assistant","content":"...","tool_call":{"name":"get_order","args":{"order_id":"..."}}}   （调工具时 content 可为空字符串）
{"role":"tool","content":"{\\"order_id\\": ...}"}
要求：
- 3 到 10 条消息，第一条是 user，最后一条是有文字内容的 assistant。
- 对话自然、口语化，措辞每次都换，不要套模板。
- 对话中绝不能出现“缺陷”“注入”“测试”之类的元信息或括号注释；问题要像真实客服自然犯的错。
输出格式：{"note": "<一句中文概括这段对话>", "transcript": [ ...消息... ]}"""


def build_specs(seed=7):
    rng = random.Random(seed)
    specs = []
    used_ids = set()

    def oid():
        while True:
            x = rng.choice("DEFGHKMNPRTWXZ") + str(rng.randint(10000, 99999))
            if x not in used_ids:
                used_ids.add(x)
                return x

    def scen_for(defects):
        if {"b", "c"} & set(defects):
            return "out_of_policy_refusal"
        if "a" in defects:
            return rng.choice(["refund", "exchange", "partial_refund"])
        return rng.choice(list(SCENARIOS))

    clean_types = [t for t in HARD_NEG for _ in range(6)]  # 5 types x 6 = 30
    for t in clean_types:
        sc = "out_of_policy_refusal" if t == "proper_refusal" else (
            rng.choice(["refund", "exchange", "partial_refund"]) if t == "ok_summary" else rng.choice(list(SCENARIOS)))
        specs.append({"defects": [], "hard_negative": t, "scenario": sc})
    for d in "abcdef":
        for _ in range(10):
            specs.append({"defects": [d], "hard_negative": None, "scenario": scen_for([d])})
    for pair in ["ad", "ae", "af", "bc", "bf", "cd", "de", "ef", "cf", "bd"]:
        specs.append({"defects": list(pair), "hard_negative": None, "scenario": scen_for(list(pair))})

    # A few task failures: the write tool errors and the agent claims success (defect e).
    fail_idx = [i for i, s in enumerate(specs) if "e" in s["defects"] and s["scenario"] in ("refund", "exchange", "partial_refund")]
    for i in fail_idx[:4]:
        specs[i]["_fail"] = True
    rng.shuffle(specs)
    out = []
    for n, s in enumerate(specs, 1):
        s = dict(s)
        s["id"] = f"h{n:03d}"
        s["persona"] = rng.choice(PERSONAS)
        s["order_id"] = oid()
        s["extra_order_id"] = oid()
        s["turns"] = rng.randint(3, 10)
        s["tool_fails"] = False
        out.append(s)
    for s_out, s_in in zip(out, specs):
        s_out["tool_fails"] = bool(s_in.get("_fail"))
        d = s_out["defects"]
        if s_out["tool_fails"]:
            s_out["outcome"] = -0.5
        elif "a" in d:
            s_out["outcome"] = 0.3
        else:
            s_out["outcome"] = 1.0
        lo, hi = (0.2, 0.5) if not d else (-0.5, 0.2)
        s_out["process_shaping"] = round(rng.uniform(lo, hi), 1)
        s_out["human"] = {it: (k not in d) for k, it in DEFECT_ITEM.items()}
    return out


def spec_prompt(s):
    lines = [f"场景：{SCENARIOS[s['scenario']]}",
             f"用户画像：{s['persona']}",
             f"主订单号：{s['order_id']}（若需要第二个订单号用 {s['extra_order_id']}；不要用其他订单号）",
             f"对话长度：约 {s['turns']} 条消息（3 到 10 条之间）"]
    if s["scenario"] == "out_of_policy_refusal":
        lines.append("get_order 返回的 shipped_days 要大于 30。")
    elif s["scenario"] in ("refund", "exchange", "partial_refund"):
        lines.append("get_order 返回的 shipped_days 要不超过 30。")
    if s["tool_fails"]:
        lines.append("写操作工具（return_items 或 exchange_items）返回 {\"status\":\"error\"}，任务实际失败。")
    if s["defects"]:
        lines.append("必须在客服的行为中自然地、明确地注入以下问题（每个都要清楚出现）：")
        for d in s["defects"]:
            lines.append(f"- {DEFECT_TEXT[d]}")
        lines.append("用户和客服都不要在对话里点破或承认这些问题（不要道歉补救、不要让用户指出），让它像没被察觉的真实失误。")
        others = [DEFECT_TEXT[k] for k in "abcdef" if k not in s["defects"]]
        lines.append("以下问题一个都不能出现：")
        lines += [f"- {t}" for t in others]
    else:
        lines.append("这是一段完全规范的对话，不能有任何问题。")
        lines.append(f"特点：{HARD_NEG[s['hard_negative']]}")
    lines.append(COMPLIANT)
    lines.append("只输出 JSON。")
    return "\n".join(lines)


def validate(s, obj):
    """Structural checks. Returns an error string or None."""
    if not isinstance(obj, dict) or not isinstance(obj.get("transcript"), list):
        return "no transcript list"
    t = obj["transcript"]
    if not 3 <= len(t) <= 10:
        return f"length {len(t)}"
    for i, m in enumerate(t):
        if not isinstance(m, dict) or m.get("role") not in ("user", "assistant", "tool"):
            return f"bad role at {i}"
        if not isinstance(m.get("content", ""), str):
            return f"content not str at {i}"
        tc = m.get("tool_call")
        if tc is not None:
            if m["role"] != "assistant" or not isinstance(tc, dict) or tc.get("name") not in TOOLS or not isinstance(tc.get("args", {}), dict):
                return f"bad tool_call at {i}"
            if i + 1 >= len(t) or t[i + 1].get("role") != "tool":
                return f"tool_call without result at {i}"
        if m["role"] == "tool" and (i == 0 or not t[i - 1].get("tool_call")):
            return f"tool msg without call at {i}"
        if m["role"] == "assistant" and not m.get("content") and not tc:
            return f"empty assistant at {i}"
    if t[0]["role"] != "user" or t[-1]["role"] != "assistant" or not t[-1].get("content"):
        return "bad first/last message"
    blob = json.dumps(t, ensure_ascii=False)
    if s["order_id"] not in blob:
        return "order id missing"
    for w in ("缺陷", "注入", "测试样本", "defect"):
        if w in blob:
            return f"meta word {w}"
    writes = [i for i, m in enumerate(t) if (m.get("tool_call") or {}).get("name") in WRITE]
    if "a" in s["defects"] and not writes:
        return "defect a but no write call"
    if writes and "a" not in s["defects"]:
        w0 = writes[0]
        last_text = max([i for i in range(w0) if t[i]["role"] == "assistant" and t[i].get("content")], default=-1)
        if last_text < 0 or not any(t[j]["role"] == "user" for j in range(last_text + 1, w0)):
            return "write without a user reply after the assistant's question"
    if s["tool_fails"] and not any('"error"' in m.get("content", "") or "error" in m.get("content", "")
                                   for m in t if m["role"] == "tool"):
        return "tool failure requested but no error result"
    return None


def generate(s, log):
    prompt = spec_prompt(s)
    last_err = None
    for attempt in range(3):  # first try + max 2 retries
        try:
            text = converse(GEN_MODEL, SYSTEM, prompt + (f"\n上一次输出不合格：{last_err}。请修正。" if last_err else ""),
                            role="generator", max_tokens=4000)
        except ClientError as exc:
            last_err = f"client error {exc.response['Error']['Code']}"
            if exc.response["Error"]["Code"] == "AccessDeniedException":
                raise
            continue
        obj = extract_json(text)
        err = validate(s, obj) if obj else "unparseable JSON"
        log.append({"id": s["id"], "attempt": attempt + 1, "error": err})
        if err is None:
            return obj
        last_err = err
    return None


def tkey(t):
    return hashlib.sha256(json.dumps([m.get("content") for m in t], ensure_ascii=False).encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", help="comma-separated ids to (re)generate")
    args = ap.parse_args()
    if args.probe:
        print(converse(GEN_MODEL, "只回答 OK", "ping", role="generator", max_tokens=20))
        return
    specs = build_specs()
    json.dump(specs, open(os.path.join(HERE, "specs.json"), "w"), ensure_ascii=False, indent=1)
    out_path = os.path.join(HERE, "holdout.jsonl")
    done = {}
    if os.path.exists(out_path):
        for line in open(out_path, encoding="utf-8"):
            r = json.loads(line)
            done[r["id"]] = r
    todo = [s for s in specs if s["id"] not in done]
    if args.only:
        ids = set(args.only.split(","))
        todo = [s for s in specs if s["id"] in ids]
    log = []
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda s: (s, generate(s, log)), todo))
    seen = {tkey(r["transcript"]): r["id"] for r in done.values()}
    firsts = {r["transcript"][0]["content"]: r["id"] for r in done.values()}
    failed = []
    for s, obj in results:
        if obj is None:
            failed.append(s["id"])
            continue
        k, f = tkey(obj["transcript"]), obj["transcript"][0]["content"]
        if (k in seen and seen[k] != s["id"]) or (len(f) >= 8 and f in firsts and firsts[f] != s["id"]):
            failed.append(s["id"] + f"(dup of {seen.get(k) or firsts.get(f)}: {f[:30]})")
            continue
        seen[k], firsts[f] = s["id"], s["id"]
        done[s["id"]] = {"id": s["id"], "note": obj.get("note", ""), "outcome": s["outcome"],
                         "process_shaping": s["process_shaping"], "human": s["human"],
                         "transcript": obj["transcript"],
                         "spec": {k2: s[k2] for k2 in ("scenario", "persona", "order_id", "defects",
                                                       "hard_negative", "tool_fails", "turns")}}
    with open(out_path, "w", encoding="utf-8") as fh:
        for s in specs:
            if s["id"] in done:
                fh.write(json.dumps(done[s["id"]], ensure_ascii=False) + "\n")
    with open(os.path.join(HERE, "gen_log.jsonl"), "a", encoding="utf-8") as fh:
        for e in log:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    tries = {}
    for e in log:
        tries[e["id"]] = max(tries.get(e["id"], 0), e["attempt"])
    errs = {}
    for e in log:
        if e["error"]:
            errs[e["error"].split(" at ")[0]] = errs.get(e["error"].split(" at ")[0], 0) + 1
    print(f"written={len(done)} failed={failed}")
    print(f"attempt distribution: {dict(sorted({n: list(tries.values()).count(n) for n in set(tries.values())}.items()))}")
    print(f"validation errors: {errs}")


if __name__ == "__main__":
    sys.exit(main())
