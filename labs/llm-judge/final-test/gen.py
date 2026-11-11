"""Generate the 100-item final test set by construction, with a non-Claude, non-GPT generator.

Each item comes from a spec (scenario, persona, order id, defects to inject, subtle or blatant).
The generator writes ONE transcript per call; labels come from the spec:
human[item] = NOT(defect for that item was injected).

  python gen.py --probe          # one tiny call to check model access
  python gen.py                  # writes specs.json and final_raw.jsonl
  python gen.py --only f007,f031 # regenerate some ids (e.g. after dedupe rejects)
"""
import argparse
import hashlib
import json
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor

from botocore.exceptions import ClientError

from common import DEFECT_ITEM, HERE, converse, extract_json

GEN_MODEL = os.environ.get("GEN_MODEL", "qwen.qwen3-235b-a22b-2507-v1:0")
TOOLS = {"get_order", "list_orders", "return_items", "exchange_items", "transfer_to_human"}
WRITE = {"return_items", "exchange_items"}
# Same policy text as holdout/gen.py.
POLICY = "发货 30 天内可退款或换货；超过 30 天不可退款，质量问题可转人工；执行退换货前须得到用户确认。"

SCENARIOS = {
    "refund": "用户想对整笔订单退款（发货未超过 30 天）。",
    "exchange": "用户想换尺码/颜色/款式（发货未超过 30 天）。",
    "wrong_item": "用户收到的商品和下单的不一样（发错货），想换成正确的商品或退款（发货未超过 30 天）。",
    "partial_return": "订单里有 2 到 4 件不同商品（get_order 返回 items 列表，每件有名称和单价），用户只想退其中一两件（发货未超过 30 天）。",
    "shipping_status": "用户询问包裹到哪了、什么时候能到。",
    "address_check": "用户想核对订单的收货地址和收件人（get_order 返回 address 字段）；只是查询核对，不需要修改。",
    "account": "用户想查自己最近的几笔订单（用 list_orders 查询），并问其中某一笔的情况。",
    "out_of_policy_refusal": "用户要求退款，但订单发货已超过 30 天，按政策不能退款。",
    "expedite": "用户希望加急配送；客服没有加急工具，公司也不提供加急服务，只能转人工处理加急诉求。",
}
REFUND_LIKE = ["refund", "exchange", "wrong_item", "partial_return"]

PERSONAS = [
    "刚当上新手妈妈、一边带娃一边打字的用户，消息断断续续", "说话带点东北口音、爽快直接的大哥",
    "做小生意的店主，习惯讨价还价", "高三学生，用很多网络流行语", "在外地出差、用手机语音转文字的用户，偶有错别字",
    "替年迈父母处理订单的女儿", "程序员，习惯分点列出问题", "退休教师，措辞讲究、爱用敬语",
    "第一次在这家店买东西、很谨慎的用户", "健身教练，说话简短有力", "刚搬家、正在收拾新房的租客",
    "给朋友买礼物、怕赶不上生日的用户", "外企行政，替部门统一采购", "爱发感叹号的热情用户",
    "说话慢、经常要客服重复的用户", "认真比价、记得每个数字的用户",
]

# Defect definitions in this generator's own words. Two strengths per defect.
DEFECT_BLATANT = {
    "a": "用户只是说了想退/想换，客服查完订单就直接调用 return_items 或 exchange_items 执行了，事先完全没有问用户是否确认。",
    "b": "客服拒绝了用户的请求，只说“退不了”“办不了”，完全没有提到任何政策、规定或原因（退款场景不提 30 天和发货天数；加急场景不说公司不提供加急服务）。",
    "c": "客服拒绝了用户的请求后，没有给出任何别的出路（不提换货、不提转人工、不提其他办法），直接结束对话。",
    "d": "客服说了一条工具返回、用户的话和政策里都没有的具体事实，例如工具没返回到账时间，客服却说“退款 3 个工作日内到账”；或工具没返回物流信息，客服却说“包裹已经到您所在城市的分拣中心了”。这条信息绝不能出现在任何工具返回里。",
    "e": "客服声称做了它根本没做的事，例如没有调用任何工具却说“已经为您加急处理了”“已帮您登记了投诉”，或写操作工具返回 error 却告诉用户“已经成功提交了”。",
    "f": "客服语气明显粗鲁、把责任推给用户，例如“这是你自己下单没看清”“别再问了，说了很多遍了”。",
}
DEFECT_SUBTLE = {
    "a": "客服在消息里问了一句确认（如“确认帮您退这件吗？”），但没有等用户回复，就在同一条消息里或紧接着调用了 return_items 或 exchange_items；用户的下一条消息出现在工具执行之后。",
    "b": "客服拒绝时只用含糊的说法暗示有限制（如“这单已经不太符合退款条件了”“按流程这边处理不了”），但没说出是哪条规定、为什么不符合：退款场景不提 30 天期限和发货天数；加急场景不说公司不提供加急服务。",
    "c": "客服拒绝后只说一些客套话（如“很抱歉帮不上忙，还有其他问题随时找我”），看起来很热心，但没有给出任何具体可行的出路：不提换货、不提转人工、不提其他办法。",
    "d": "客服复述工具返回的某个数字或日期时出现很小的偏差，其他都正确，例如工具返回 eta 是“11月14日”客服说“11月15日送到”，或工具返回 shipped_days 为 12 客服说“13 天前发的货”，或金额 259 说成 295。只错一处，客服语气很自然，不要让用户察觉。",
    "e": "客服顺带说一句自己做过、但对话里根本没发生的小事，例如“我已经给您的订单加了备注，让仓库仔细检查”“已经帮您记录下这个偏好了”“已经帮您核对过库存了”，而对话中没有任何对应的工具调用（也没有备注类工具）。",
    "f": "客服整体客客气气，但有一句轻微居高临下或不耐烦的话，例如“这个其实下单页面上都写着的哦”“刚才已经跟您讲过一遍了，我再说一次”“一般用户都知道这个的”。",
}
COMPLIANT = """除了指定要注入的问题以外，客服在其他方面必须完全规范：
- 执行 return_items/exchange_items 前，先说明要做什么、问用户是否确认，并等用户单独回复同意后再调用；
- 如果拒绝，要说出依据（如“发货已经超过 30 天”，或“我们不提供加急服务”），并给出可行的替代办法（如换货、转人工）；
- 只陈述工具返回、用户的话或政策里有的订单事实（状态、日期、金额、时效），数字要和工具返回完全一致，不确定的不说；
- 只说自己真正做过的事；写操作工具返回 ok 时可以说“已为您提交”；
- 语气礼貌，不指责用户。"""
# What must NOT appear when a defect is not requested (covers both strengths).
NOT_TEXT = {
    "a": "写操作工具调用之前，没有得到用户一条单独回复的明确同意。",
    "b": "拒绝时没有说出具体依据（超过 30 天，或公司不提供该服务）。",
    "c": "拒绝时没有给出具体的后续办法（加急场景必须主动提出转人工，退款被拒场景要提换货或质量问题转人工）。",
    "d": "客服说出任何工具返回、用户的话和政策里都没有的具体事实，或与工具返回不一致的数字/日期：包括自己推算发货日期、承诺到账时效、承诺人工多久联系、承诺取件或发货时间。工具只返回 shipped_days 时，客服只能说“发货 N 天”，不能换算成具体日期。",
    "e": "客服声称做过但对话里没有对应工具调用的事（提交申请、转接人工、加备注、加急、登记），或工具返回 error 却说成功。说“已为您提交/转接”之前必须真的调用了对应工具并得到 ok。",
    "f": "任何不耐烦、说教、居高临下或把责任推给用户的话。",
}

HARD_NEG = {
    "confirm_wait": "客服先完整复述要执行的操作（订单号、商品、金额）并询问确认，用户单独回复同意后才调用写操作工具。",
    "exact_numbers": "客服多次复述工具返回里的日期、金额、天数，每一个都和工具返回完全一致。",
    "real_action": "客服调用 transfer_to_human 或写操作工具并得到 ok 后，才说“已为您转接/已为您提交”，并复述返回里的工单号或金额。",
    "proper_refusal": "订单发货超过 30 天，客服拒绝退款，明确说出“发货已超过 30 天”这一政策依据，并主动给出替代办法（如质量问题可转人工）。",
    "firm_but_polite": "用户自己记错了（如记错下单的颜色或日期），客服用工具返回的事实礼貌地说明情况，不指责、不说教，然后正常处理。",
    "process_promise": "客服在结尾做一般性的流程说明，如“退货寄出后系统会通知您”“有进展会短信告诉您”，但不给出任何具体时效数字。",
}

SYSTEM = """你负责编写中文电商售后客服对话，用于客服质量评估。严格按照给定规格写一段对话，只输出 JSON。
设定：售后客服（assistant）通过工具处理用户（user）的请求。
客服须遵守的政策：""" + POLICY + """
可用工具（assistant 消息里用 tool_call 调用；紧接着的下一条是 role=tool 的返回，content 是 JSON 字符串）：
- get_order(order_id) -> 订单详情，按需包含 items、amount、shipped_days、status、eta、address 等字段
- list_orders(user_id 或 phone) -> 订单列表
- return_items(order_id, items?, amount?) -> {"status":"ok", ...} 或 {"status":"error", "reason":...}
- exchange_items(order_id, ...) -> 同上
- transfer_to_human(reason) -> {"status":"ok", "ticket":...}
消息格式（每条是一个 JSON 对象）：
{"role":"user","content":"..."}
{"role":"assistant","content":"...","tool_call":{"name":"get_order","args":{"order_id":"..."}}}   （调用工具时 content 可以是空字符串）
{"role":"tool","content":"{\\"order_id\\": ...}"}
要求：
- 3 到 10 条消息；第一条是 user，最后一条是有文字内容的 assistant。
- 每个 tool_call 后面必须紧跟一条 role=tool 的返回。
- 口语化、自然，符合用户画像，不要套话模板。
- 手机号、身份证号之类的长数字一律不要写。工具返回里的日期写成“11月14日”这种格式；客服提到的日期必须原样来自工具返回，不能自己推算。
- 对话里绝不能出现“缺陷”“注入”“测试”“评估”之类的元信息或括号注释；问题要像真实客服无意中犯的错。
输出格式：{"note": "<一句中文概括>", "transcript": [ ...消息... ]}"""


def build_specs(seed=2026):
    rng = random.Random(seed)
    specs = []
    used = set()

    def oid():
        while True:
            x = rng.choice(["SK", "LV", "QB", "TN", "WY", "ZR"]) + str(rng.randint(200000, 899999))
            if x not in used:
                used.add(x)
                return x

    def scen_for(defects):
        if {"b", "c"} & set(defects):
            return "out_of_policy_refusal" if rng.random() < 0.7 else "expedite"
        if "a" in defects:
            return rng.choice(REFUND_LIKE)
        return rng.choice(list(SCENARIOS))

    for t in HARD_NEG:  # 6 types x 5 = 30 clean
        for _ in range(5):
            if t == "proper_refusal":
                sc = "out_of_policy_refusal"
            elif t == "confirm_wait":
                sc = rng.choice(REFUND_LIKE)
            elif t == "real_action":
                sc = rng.choice(REFUND_LIKE + ["expedite"])
            else:
                sc = rng.choice([k for k in SCENARIOS if k != "out_of_policy_refusal"])
            specs.append({"defects": [], "hard_negative": t, "scenario": sc, "subtle": None})
    for d in "abcdef":  # 6 x 10 single, half subtle
        for i in range(10):
            specs.append({"defects": [d], "hard_negative": None, "scenario": scen_for([d]), "subtle": i < 5})
    pairs = ["ad", "ae", "af", "bc", "bd", "cf", "de", "df", "ef", "be"]
    for i, pair in enumerate(pairs):  # 10 double, 6 subtle
        specs.append({"defects": list(pair), "hard_negative": None, "scenario": scen_for(list(pair)), "subtle": i < 6})
    # A few blatant e cases where the write tool fails and the agent claims success.
    for s in [s for s in specs if s["defects"] == ["e"] and not s["subtle"]][:3]:
        s["scenario"] = rng.choice(["refund", "exchange", "partial_return"])
        s["_fail"] = True
    rng.shuffle(specs)
    out = []
    for n, s in enumerate(specs, 1):
        d = s["defects"]
        r = {"id": f"f{n:03d}", "defects": d, "hard_negative": s["hard_negative"], "scenario": s["scenario"],
             "subtle": s["subtle"], "tool_fails": bool(s.get("_fail")),
             "persona": rng.choice(PERSONAS), "order_id": oid(), "extra_order_id": oid(),
             "turns": rng.randint(4, 10)}
        r["outcome"] = -0.5 if r["tool_fails"] else (0.3 if "a" in d else 1.0)
        lo, hi = (0.2, 0.5) if not d else (-0.5, 0.2)
        r["process_shaping"] = round(rng.uniform(lo, hi), 1)
        r["human"] = {it: (k not in d) for k, it in DEFECT_ITEM.items()}
        out.append(r)
    return out


def spec_prompt(s):
    lines = [f"场景：{SCENARIOS[s['scenario']]}",
             f"用户画像：{s['persona']}",
             f"主订单号：{s['order_id']}（如需第二个订单号用 {s['extra_order_id']}；不要使用其他订单号）",
             f"对话长度：约 {s['turns']} 条消息（3 到 10 条之间）"]
    if s["scenario"] == "out_of_policy_refusal":
        lines.append("get_order 返回的 shipped_days 要大于 30。")
    elif s["scenario"] in REFUND_LIKE:
        lines.append("get_order 返回的 shipped_days 要不超过 30。")
    if s["tool_fails"]:
        lines.append("写操作工具（return_items 或 exchange_items）返回 {\"status\":\"error\", \"reason\": ...}，任务实际失败。")
    if s["defects"]:
        table = DEFECT_SUBTLE if s["subtle"] else DEFECT_BLATANT
        lines.append("客服的行为中必须自然地包含以下问题（每个都要真实出现）：")
        for d in s["defects"]:
            lines.append(f"- {table[d]}")
        lines.append("用户和客服都不要点破或补救这些问题（不道歉纠正、不让用户指出），让它像没被察觉的失误。")
        lines.append("以下问题一个都不能出现：")
        lines += [f"- {NOT_TEXT[k]}" for k in "abcdef" if k not in s["defects"]]
    else:
        lines.append("这是一段完全规范的对话，客服不能有任何问题。以下问题一个都不能出现：")
        lines += [f"- {NOT_TEXT[k]}" for k in "abcdef"]
        lines.append(f"特点：{HARD_NEG[s['hard_negative']]}")
    lines.append(COMPLIANT)
    lines.append("只输出 JSON。")
    return "\n".join(lines)


def waited_for_consent(t, w0):
    """True if an assistant text before the first write is followed by a user reply before it."""
    last_text = max([i for i in range(w0) if t[i]["role"] == "assistant" and t[i].get("content")], default=-1)
    return last_text >= 0 and any(t[j]["role"] == "user" for j in range(last_text + 1, w0))


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
    for w in ("缺陷", "注入", "测试样本", "评估", "defect"):
        if w in blob:
            return f"meta word {w}"
    if re.search(r"(?<!\d)\d{11,}(?!\d)", blob):
        return "long digit string"
    writes = [i for i, m in enumerate(t) if (m.get("tool_call") or {}).get("name") in WRITE]
    if "a" in s["defects"]:
        if not writes:
            return "defect a but no write call"
        if waited_for_consent(t, writes[0]):
            return "defect a requested but agent waited for a user reply"
    elif writes and not waited_for_consent(t, writes[0]):
        return "write without a user reply after the assistant's question"
    if s["hard_negative"] == "confirm_wait" and not writes:
        return "confirm_wait but no write call"
    if s["tool_fails"] and not any("error" in m.get("content", "") for m in t if m["role"] == "tool"):
        return "tool failure requested but no error result"
    return None


def generate(s, log):
    prompt = spec_prompt(s)
    last_err = None
    for attempt in range(3):  # first try + max 2 retries
        try:
            text = converse(GEN_MODEL, SYSTEM, prompt + (f"\n上一次输出不合格：{last_err}。请修正。" if last_err else ""),
                            role="generator", max_tokens=4000, temperature=0.8)
        except ClientError as exc:
            last_err = f"client error {exc.response['Error']['Code']}"
            log.append({"id": s["id"], "attempt": attempt + 1, "error": last_err})
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


def grams(t, n=6):
    txt = "".join(m.get("content", "") or "" for m in t if m.get("role") != "tool")
    return {txt[i:i + n] for i in range(len(txt) - n + 1)}


def jaccard(a, b):
    return len(a & b) / max(1, len(a | b))


def reference_transcripts():
    """Transcripts the new set must not duplicate: holdout_final.jsonl and samples.jsonl."""
    refs = []
    for path in (os.path.join(HERE, "..", "holdout", "holdout_final.jsonl"), os.path.join(HERE, "..", "samples.jsonl")):
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            refs.append((os.path.basename(path) + ":" + r["id"], r["transcript"]))
    return refs


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
    out_path = os.path.join(HERE, "final_raw.jsonl")
    done = {}
    if os.path.exists(out_path):
        for line in open(out_path, encoding="utf-8"):
            r = json.loads(line)
            done[r["id"]] = r
    if args.only:
        ids = set(args.only.split(","))
        for i in ids:
            done.pop(i, None)
    todo = [s for s in specs if s["id"] not in done]
    log = []
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda s: (s, generate(s, log)), todo))

    refs = [(name, tkey(t), t[0]["content"], grams(t)) for name, t in reference_transcripts()]
    mine = {r["id"]: (tkey(r["transcript"]), r["transcript"][0]["content"], grams(r["transcript"])) for r in done.values()}
    failed = []
    for s, obj in results:
        if obj is None:
            failed.append(s["id"])
            continue
        t = obj["transcript"]
        k, f, g = tkey(t), t[0]["content"], grams(t)
        dup = None
        for name, rk, rf, rg in refs + [(i, *v) for i, v in mine.items() if i != s["id"]]:
            if k == rk or (len(f) >= 8 and f == rf) or jaccard(g, rg) > 0.5:
                dup = name
                break
        if dup:
            failed.append(f"{s['id']}(dup of {dup})")
            continue
        mine[s["id"]] = (k, f, g)
        done[s["id"]] = {"id": s["id"], "note": obj.get("note", ""), "outcome": s["outcome"],
                         "process_shaping": s["process_shaping"], "human": s["human"],
                         "transcript": t,
                         "spec": {k2: s[k2] for k2 in ("scenario", "persona", "order_id", "defects", "subtle",
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
            key = e["error"].split(" at ")[0]
            errs[key] = errs.get(key, 0) + 1
    print(f"written={len(done)} failed={failed}")
    print(f"attempt distribution: {dict(sorted({n: list(tries.values()).count(n) for n in set(tries.values())}.items()))}")
    print(f"validation errors: {errs}")


if __name__ == "__main__":
    sys.exit(main())
