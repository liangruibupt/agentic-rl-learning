"""Generate a synthetic customer-ticket extraction dataset for Bedrock RFT.

Each JSONL line follows the Bedrock RFT open-weight format:
  {"messages": [...], "reference_answer": {...}, "task_id": ..., "template_id": ..., "slice": ..., "formats": ..., "wait_days": ...}
The extra top-level fields are passed to the reward Lambda under "metadata"; the reward
ignores them, check_dataset.py uses them to report template overlap and the OOD slice.

Tickets are composed, not filled in from a few fixed sentences:
  opener + core phrasing (per intent) + optional price clause + extra context + closing marker
with separate pools for slot fillers (order IDs, amounts, waiting times, items, reasons),
politeness / anger / impatience markers and noise (typos, mixed Chinese-English).

Split by construction, not by random rows:
  train     train phrasings (7 per class), train slot formats and train markers
  dev       2 other phrasings per class, train formats (for checkpoint selection)
  test      3 held-out phrasings per class, held-out anger markers, in two slices:
            "new_phrasing"  train slot formats            (2/3 of test)
            "ood"           held-out slot formats, mixed-language noise or English
                            tickets, held-out items       (1/3 of test)
Every split has the same number of tickets per category. Each split uses its own random
stream, so adding --dev does not change train.jsonl or test.jsonl.

The labeling rules are stated in the system prompt, so the task is learnable and the
reward is fully verifiable. Run:
  python make_dataset.py --train 400 --test 150 --seed 7            # the committed files
  python make_dataset.py --train 400 --test 150 --dev 100 --seed 7  # same train/test plus dev.jsonl
  python make_dataset.py --with-reply --prefix reply_               # adds a "reply" field for the judge path
Then run check_dataset.py on the output before uploading.
"""
import argparse
import json
import random
import re

SYSTEM_PROMPT = """你是电商客服工单分拣助手。阅读工单，只输出一个 JSON 对象，不要输出其他内容。
工单可能是中文、英文或中英混合；字段取值始终用下面的写法。
字段：
- category: refund（退款）| exchange（换货）| shipping（物流）| account（账号）| other
- priority: high | medium | low
- order_id: 订单号字符串，统一写成大写字母加数字，去掉空格、连字符和 #（工单写 a-10234 或 #A10234 都输出 "A10234"），没有则为 null
- amount: 涉及金额，换算成数字（单位元；“一千二百元”“¥1,200.00”“RMB 1200”都输出 1200），没有则为 null
priority 规则（按顺序判断，命中即停）：
1. 提到投诉、律师、报警、曝光（英文 complaint、lawyer、police、expose 同样算），或 amount > 1000 → high
2. amount 在 200 到 1000 之间（含），或物流类工单等待超过 7 天（一周按 7 天、半个月按 15 天、一个月按 30 天算）→ medium
3. 其他 → low"""

REPLY_FIELD = """
- reply: 给用户的一句简短回复（不超过 80 字），回应诉求并说明下一步，不要承诺工单里没有的金额或时间"""

CATEGORIES = ("refund", "exchange", "shipping", "account", "other")

# Core phrasings. Slots: {oid} order ID, {amt} amount, {days} waiting time, {item}, {reason},
# {phone4} last 4 phone digits, {n} a small count. Which split a phrasing belongs to is fixed here.
CORE = {
    "refund": {
        "train": ["我想把订单{oid}退了，申请退款{amt}。", "订单{oid}的{item}有质量问题，请退我{amt}。",
                  "上周买的{item}不想要了，订单号{oid}，退款金额{amt}。", "订单{oid}重复扣款了，多扣的{amt}请退给我。",
                  "申请退款：订单{oid}，金额{amt}，原因是{reason}。", "订单{oid}申请全额退款，{item}一次都没用过。",
                  "{item}还在7天无理由期内，订单{oid}，申请退款{amt}。"],
        "dev": ["订单{oid}我已经寄回去了，退款{amt}什么时候到账？", "{item}不想要了，怎么申请退款？订单号{oid}。"],
        "test": ["{item}和描述不符，要求退款，单号{oid}，一共{amt}。", "订单{oid}取消了，但是{amt}到现在还没退回来。",
                 "这单{oid}拍错了，还没发货的话直接退款吧，{item}不要了。"],
        "english": ["Please refund order {oid}, the {item} arrived broken. I paid {amt}.",
                    "I cancelled order {oid} but the refund of {amt} has not arrived."],
    },
    "exchange": {
        "train": ["订单{oid}的{item}尺码不对，想换大一码。", "收到的{item}颜色发错了，订单{oid}，帮我换一件。",
                  "订单{oid}里的{item}有瑕疵，换个新的。", "{item}型号发错了，订单号{oid}，请换成我下单的那款。",
                  "申请换货：订单{oid}，{item}，原因是{reason}。", "能不能把订单{oid}的{item}换成黑色？",
                  "{item}用了两天就坏了，订单{oid}，能换新吗？"],
        "dev": ["订单{oid}的{item}跟图片不一样，我想换一个款式。", "我买的{item}大了，订单{oid}，换小一号可以吗？"],
        "test": ["订单{oid}：{item}的包装破了，里面也有划痕，请安排换货。", "{item}发成旧款了，单号{oid}，麻烦换成新款。",
                 "{item}少了配件，订单{oid}，不退，直接给我换一个完整的。"],
        "english": ["Wrong size for the {item} in order {oid}, can I exchange it?",
                    "Order {oid}: wrong colour {item}, please send me a replacement."],
    },
    "shipping": {
        "train": ["订单{oid}已经{days}没有物流更新了。", "我的包裹{oid}显示派送中，可是{days}了还没到。",
                  "订单{oid}什么时候发货？已经等了{days}。", "快递一直停在中转站，订单{oid}，{days}没动了。",
                  "{item}下单{days}了还没发出，订单{oid}。", "查一下订单{oid}的快递到哪了，等了{days}。",
                  "我的快递{days}没到，订单号找不到了。"],
        "dev": ["订单{oid}的快递员电话打不通，包裹{days}没送来。", "催一下发货，已经{days}了，订单{oid}。"],
        "test": ["订单号{oid}，物流显示已签收，但我没收到，下单到现在{days}了。",
                 "订单{oid}的{item}说好次日达，结果{days}了还没到。", "{item}的物流信息停了{days}，一直没人派送。"],
        "english": ["Order {oid} has had no tracking update for {days}.",
                    "Where is my {item}? I have been waiting {days} for order {oid}."],
    },
    "account": {
        "train": ["我的账号登录不上，一直提示密码错误。", "手机号换了，原来尾号{phone4}的号码不用了，怎么修改绑定？",
                  "账号好像被别人登录了，收到了陌生设备的登录提醒。", "收不到验证码，绑定的手机尾号是{phone4}。",
                  "我想注销账号，要怎么操作？", "账号被冻结了，订单{oid}没法付款，帮我解冻一下。",
                  "实名认证一直审核不通过，已经提交{n}次了。"],
        "dev": ["会员积分突然清零了，账号是尾号{phone4}的手机号。", "忘记密码了，用尾号{phone4}的手机找回时提示号码未注册。"],
        "test": ["账号里的收货地址被人改了，是不是被盗号了？", "换了新手机以后一直提示账号异常，登录不了。",
                 "下单{oid}的时候提示账号存在风险，付不了款，手机尾号{phone4}。"],
        "english": ["I can't log in to my account, it keeps saying wrong password.",
                    "Please change the phone number on my account, the old one ends in {phone4}."],
    },
    "other": {
        "train": ["你们的发票怎么开？订单{oid}。", "请问周末有客服吗？", "{item}有没有现货？想买{n}件。",
                  "订单{oid}的发票抬头写错了，能重开吗？", "{item}怎么使用？说明书看不懂。",
                  "能开电子发票吗？金额{amt}，订单{oid}。", "{item}的保修期是多久？"],
        "dev": ["请问{item}支持分期付款吗？", "你们的门店在哪里？可以自提吗？"],
        "test": ["你们有没有企业采购的渠道？{item}想买{n}个。", "订单{oid}需要开{amt}的增值税专用发票，抬头怎么填？",
                 "提个建议：{item}的详情页图片太少了。"],
        "english": ["Do you have customer service on weekends?", "Can I get an invoice for order {oid}? The total was {amt}."],
    },
}

# Price clause appended to some exchange tickets (amount is then the price).
PRICE_CLAUSE = ["这件{item}是{amt}买的。", "当时付了{amt}。", "商品价格{amt}。"]
PRICE_CLAUSE_EN = ["It cost {amt}."]

ITEMS = ["蓝牙耳机", "运动鞋", "羽绒服", "电饭煲", "机械键盘", "行李箱", "保温杯", "台灯", "双肩包", "咖啡机"]
ITEMS_OOD = ["扫地机器人", "儿童绘本", "空气炸锅", "瑜伽垫"]
ITEMS_EN = {"蓝牙耳机": "earbuds", "运动鞋": "sneakers", "羽绒服": "down jacket", "电饭煲": "rice cooker",
            "机械键盘": "keyboard", "行李箱": "suitcase", "保温杯": "thermos", "台灯": "desk lamp", "双肩包": "backpack",
            "咖啡机": "coffee maker", "扫地机器人": "robot vacuum", "儿童绘本": "picture book", "空气炸锅": "air fryer",
            "瑜伽垫": "yoga mat"}
REASONS = {"refund": ["质量问题", "发错货", "和描述不符", "不想要了"],
           "exchange": ["尺码不合适", "颜色发错", "有瑕疵", "型号不对"]}

OPENERS = ["", "", "你好，", "客服你好，", "在吗？", "您好，打扰一下，"]
CONTEXT = ["我是老客户了。", "昨天已经问过一次了。", "这是第一次在你们家买东西。", "我平时都在App上下单。"]
POLITE = ["", "麻烦尽快处理，谢谢。", "辛苦了。", "谢谢！", "等你们回复。"]
# Impatience WITHOUT a rule-1 keyword: must not raise the priority (a trap).
IMPATIENT = ["怎么这么慢！", "太失望了。", "你们到底管不管？"]
ANGRY = {"train": ["再不处理我就投诉！", "我要找律师。", "我会去网上曝光你们。", "再这样我就报警了。"],
         "test": ["已经准备向消费者协会投诉了。", "我已经联系律师了。", "不解决就发到网上曝光。", "实在不行我只能报警处理。"],
         "english": ["I will file a complaint.", "My lawyer will contact you.", "I'm going to call the police.",
                     "I will expose you on social media."]}
POLITE_EN = ["", "Thanks.", "Please help ASAP."]

TYPOS = {"订单": "定单", "麻烦": "麻凡", "包裹": "包过", "密码": "密吗", "发票": "发漂", "客服": "客福"}
MIXED = {"退款": "refund", "换货": "exchange", "物流": "tracking", "快递": "express", "账号": "account",
         "发票": "invoice", "验证码": "verification code", "订单号": "order no.", "谢谢": "thx"}
MIXED_TAIL = ["pls 尽快", "ASAP 谢谢", "thx"]

AMOUNTS = [39, 89, 159, 199, 200, 368, 450, 599, 760, 999, 1000, 1001, 1299, 2499, 3888]
AMOUNTS_DECIMAL = [58.5, 199.9, 1000.5]
DAYS = [2, 3, 5, 6, 7, 8, 9, 10, 12, 14, 20]
OID_LETTERS = "ABCDEHK"

# Slot formats: the first group is used in train/dev/test-new_phrasing, the second only in the OOD slice.
FORMATS = {
    "oid": (("plain",), ("lower", "hyphen", "hash")),
    "amt": (("yuan", "kuai", "yen"), ("cn", "comma", "rmb")),
    "days": (("digit",), ("cn", "week", "halfmonth", "month")),
}

_CN_DIGITS = "零一二三四五六七八九"
_EN_NUM = {2: "two", 3: "three", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
           12: "twelve", 14: "fourteen", 15: "fifteen", 20: "twenty", 21: "twenty-one", 30: "thirty"}


def cn_number(n, liang=False):
    """Integer 1..9999 written in Chinese numerals, e.g. 1299 -> 一千二百九十九, 1001 -> 一千零一."""
    if not 0 < n < 10000 or int(n) != n:
        raise ValueError(n)
    n = int(n)
    parts, zero = [], False
    for value, unit in ((1000, "千"), (100, "百"), (10, "十"), (1, "")):
        d = n // value % 10
        if d == 0:
            zero = bool(parts)
            continue
        if zero:
            parts.append("零")
            zero = False
        digit = "两" if (liang and d == 2 and unit in ("千", "百") and not parts) else _CN_DIGITS[d]
        parts.append(digit + unit)
    s = "".join(parts)
    return s[1:] if s.startswith("一十") else s


def _num(v):
    return int(v) if float(v).is_integer() else v


def render_oid(raw, fmt):
    letter, digits = raw[0], raw[1:]
    return {"plain": raw, "lower": raw.lower(), "hyphen": f"{letter}-{digits}", "hash": f"#{raw}"}[fmt]


def render_amount(v, fmt, rng, english=False):
    v = _num(v)
    if english:
        return {"yuan": f"{v} yuan", "kuai": f"{v} yuan", "yen": f"¥{v}", "cn": f"RMB {v:,}",
                "comma": f"{v:,.2f} yuan", "rmb": f"RMB {v}"}[fmt]
    if fmt == "yuan":
        return f"{v}元"
    if fmt == "kuai":
        return f"{v}块"
    if fmt == "yen":
        return f"¥{v}"
    if fmt == "cn":
        return cn_number(v, liang=rng.random() < 0.5) + rng.choice(["元", "块"])
    if fmt == "comma":
        return f"{v:,.2f}元"
    return f"RMB {v}"


def pick_days(fmt, rng):
    """Return (days, text) for a waiting-time format."""
    if fmt == "week":
        w = rng.choice([1, 2, 3])
        return 7 * w, rng.choice(["{}周", "{}个星期"]).format("一两三"[w - 1])
    if fmt == "halfmonth":
        return 15, "半个月"
    if fmt == "month":
        return 30, "一个月"
    d = rng.choice(DAYS)
    if fmt == "cn":
        return d, ("两" if d == 2 else cn_number(d)) + "天"
    return d, f"{d}天"


def render_days_en(d):
    return {7: "a week", 14: "two weeks", 21: "three weeks", 15: "half a month", 30: "a month"}.get(d) or f"{_EN_NUM.get(d, d)} days"


def priority_of(cat, amount, days, angry):
    if angry or (amount is not None and amount > 1000):
        return "high"
    if (amount is not None and 200 <= amount <= 1000) or (cat == "shipping" and days is not None and days > 7):
        return "medium"
    return "low"


def make_one(rng, cat, group, ood_kind=None, system_prompt=SYSTEM_PROMPT):
    """One ticket. group: train | dev | test. ood_kind: None | format | mixed | english."""
    english = ood_kind == "english"
    pool = CORE[cat]["english" if english else group]
    t_idx = rng.randrange(len(pool))
    tmpl = pool[t_idx]
    template_id = f"{cat}/{'en' if english else group}{t_idx}"
    ood = ood_kind is not None
    items = ITEMS + ITEMS_OOD if ood else ITEMS
    item_cn = rng.choice(items)
    formats = {}

    def fmt_of(slot):
        seen, held = FORMATS[slot]
        f = rng.choice(held if ood_kind in ("format", "english") else seen)
        formats[slot] = f
        return f

    price = None
    if cat == "exchange" and rng.random() < 0.4:
        price = rng.choice(PRICE_CLAUSE_EN if english else PRICE_CLAUSE)
    has_amt = "{amt}" in tmpl or price is not None
    slots = {"item": ITEMS_EN[item_cn] if english else item_cn,
             "reason": rng.choice(REASONS.get(cat, ["其他"])),
             "phone4": f"{rng.randint(0, 9999):04d}", "n": str(rng.choice([2, 3, 5, 10]))}
    order_id = None
    if "{oid}" in tmpl:
        order_id = f"{rng.choice(OID_LETTERS)}{rng.randint(10000, 99999)}"
        slots["oid"] = render_oid(order_id, fmt_of("oid"))
    amount = None
    if has_amt:
        f = fmt_of("amt")
        amount = rng.choice(AMOUNTS_DECIMAL if f in ("yuan", "yen", "comma") and rng.random() < 0.15 else AMOUNTS)
        slots["amt"] = render_amount(amount, f, rng, english)
        amount = _num(amount)
    days = None
    if "{days}" in tmpl:
        if english:
            f = fmt_of("days")
            days, _ = pick_days(f if f != "cn" else "digit", rng)
            slots["days"] = render_days_en(days)
        else:
            days, slots["days"] = pick_days(fmt_of("days"), rng)

    core = tmpl.format(**slots)
    if price:
        core += (" " if english else "") + price.format(**slots)
    if not english and rng.random() < 0.15:  # typo noise, never inside slot values or rule keywords
        word = rng.choice([w for w in TYPOS if w in tmpl] or [None])
        if word:
            core = tmpl.replace(word, TYPOS[word]).format(**slots) + (price.format(**slots) if price else "")
            formats["noise"] = "typo"
    tail = ""
    if ood_kind == "mixed":
        hits = [w for w in MIXED if w in core]
        for w in rng.sample(hits, k=min(len(hits), 2)):
            core = core.replace(w, " " + MIXED[w] + " ", 1)
        core = re.sub(r" {2,}", " ", core).strip()
        tail = "" if hits and rng.random() < 0.5 else rng.choice(MIXED_TAIL)
        formats["lang"] = "mixed"
    elif english:
        formats["lang"] = "en"

    r = rng.random()
    angry = r < 0.22
    if english:
        opener, context = rng.choice(["", "Hi, ", "Hello, "]), ""
        if opener and not core.startswith("I "):
            core = core[0].lower() + core[1:]
        closer = rng.choice(ANGRY["english"]) if angry else rng.choice(POLITE_EN)
        text = " ".join(x for x in (opener + core, closer) if x)
    else:
        opener = rng.choice(OPENERS)
        context = rng.choice(CONTEXT) if rng.random() < 0.25 else ""
        if angry:
            closer = rng.choice(ANGRY["test" if group == "test" else "train"])
        elif r < 0.32:
            closer = rng.choice(IMPATIENT)
        else:
            closer = rng.choice(POLITE)
        if ood_kind == "mixed":
            closer += tail
        text = opener + context + core + closer
    ref = {"category": cat, "priority": priority_of(cat, amount, days, angry), "order_id": order_id, "amount": amount}
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"工单内容：{text}"},
        ],
        "reference_answer": ref,
        "template_id": template_id,
        "slice": group if group != "test" else ("ood" if ood else "new_phrasing"),
        "formats": ",".join(f"{k}:{v}" for k, v in sorted(formats.items())),
        "wait_days": days,
    }


HELD_OUT = {f"{slot}:{f}" for slot, (_, held) in FORMATS.items() for f in held} | {"lang:mixed", "lang:en"}


def has_held_out_format(row):
    return bool(HELD_OUT & set(filter(None, row["formats"].split(","))))


def _norm_key(s):
    return re.sub(r"[\s\W_]+", "", s.lower())


def make_split(n, group, seed, system_prompt, seen, ood_frac=0.0):
    """n tickets, equal count per category (remainder spread over the first categories)."""
    rng = random.Random(f"{seed}-{group}")
    n_ood = round(n * ood_frac)
    plan = []
    for kind_n, ood in ((n - n_ood, False), (n_ood, True)):
        per = [kind_n // len(CATEGORIES) + (i < kind_n % len(CATEGORIES)) for i in range(len(CATEGORIES))]
        plan += [(c, ood) for c, k in zip(CATEGORIES, per) for _ in range(k)]
    rng.shuffle(plan)
    rows = []
    for cat, ood in plan:
        for _attempt in range(1000):
            kind = rng.choices(["format", "mixed", "english"], weights=[2, 1, 1])[0] if ood else None
            row = make_one(rng, cat, group, kind, system_prompt)
            key = _norm_key(row["messages"][1]["content"])
            if ood and not has_held_out_format(row):
                continue
            if key not in seen:
                seen.add(key)
                break
        else:
            raise RuntimeError(f"could not draw a new {cat} ticket for {group}")
        row["task_id"] = f"ticket-{group}-{len(rows):05d}"
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=400)
    ap.add_argument("--test", type=int, default=150)
    ap.add_argument("--dev", type=int, default=0,
                    help="extra dev rows for checkpoint selection; own phrasings and random stream, so train/test stay identical")
    ap.add_argument("--ood-frac", type=float, default=1 / 3, help="share of the test set in the out-of-distribution slice")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--with-reply", action="store_true",
                    help="also ask for a 'reply' field (for the optional LLM-judge reward path)")
    ap.add_argument("--prefix", default="", help="output file prefix, e.g. reply_")
    args = ap.parse_args()
    system_prompt = SYSTEM_PROMPT.replace("\npriority 规则", REPLY_FIELD + "\npriority 规则", 1) if args.with_reply else SYSTEM_PROMPT
    seen = set()
    parts = [("train.jsonl", make_split(args.train, "train", args.seed, system_prompt, seen)),
             ("test.jsonl", make_split(args.test, "test", args.seed, system_prompt, seen, args.ood_frac))]
    if args.dev:
        parts.append(("dev.jsonl", make_split(args.dev, "dev", args.seed, system_prompt, seen)))
    for name, part in parts:
        name = args.prefix + name
        with open(name, "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(part)} rows to {name}")


if __name__ == "__main__":
    main()
