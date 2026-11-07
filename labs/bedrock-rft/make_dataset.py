"""Generate a synthetic customer-ticket extraction dataset for Bedrock RFT.

Each JSONL line follows the Bedrock RFT open-weight format:
  {"messages": [...], "reference_answer": {...}, "task_id": ...}

The labeling rules are stated in the system prompt, so the task is learnable
and the reward is fully verifiable. Run:
  python make_dataset.py --train 400 --test 150 --seed 7
"""
import argparse
import json
import random

SYSTEM_PROMPT = """你是电商客服工单分拣助手。阅读工单，只输出一个 JSON 对象，不要输出其他内容。
字段：
- category: refund（退款）| exchange（换货）| shipping（物流）| account（账号）| other
- priority: high | medium | low
- order_id: 订单号字符串（如 "A10234"），没有则为 null
- amount: 涉及金额（数字，单位元），没有则为 null
priority 规则（按顺序判断，命中即停）：
1. 提到投诉、律师、报警、曝光，或 amount > 1000 → high
2. amount 在 200 到 1000 之间（含），或物流类工单等待超过 7 天 → medium
3. 其他 → low"""

CATS = {
    "refund": ["我想把订单{oid}退了，申请退款{amt}元。", "订单{oid}的商品有质量问题，请退我{amt}块钱。", "上周买的东西不想要了，订单号{oid}，退款金额{amt}元。"],
    "exchange": ["订单{oid}的鞋码不对，想换大一码。", "收到的衣服颜色发错了，订单{oid}，帮我换一件。", "订单{oid}里的耳机左耳没声音，换个新的。"],
    "shipping": ["订单{oid}已经{days}天没有物流更新了。", "我的包裹{oid}显示派送中，可是{days}天了还没到。", "订单{oid}什么时候发货？已经等了{days}天。"],
    "account": ["我的账号登录不上，一直提示密码错误。", "手机号换了，怎么修改绑定？", "账号好像被别人登录了，帮我看看。"],
    "other": ["你们的发票怎么开？", "请问周末有客服吗？", "能推荐一款适合送礼的商品吗？"],
}
ANGRY = ["再不处理我就投诉！", "我要找律师。", "我会去网上曝光你们。", "再这样我就报警了。"]
POLITE = ["麻烦尽快处理，谢谢。", "辛苦了。", "", "谢谢！"]


def make_one(rng, idx):
    cat = rng.choice(list(CATS))
    oid = f"{rng.choice('ABCDK')}{rng.randint(10000, 99999)}"
    amt = rng.choice([39, 89, 159, 199, 200, 368, 599, 1000, 1299, 2499, 3888])
    days = rng.choice([2, 3, 5, 8, 10, 14])
    text = rng.choice(CATS[cat]).format(oid=oid, amt=amt, days=days)
    order_id = oid if oid in text else None
    amount = amt if cat == "refund" else None
    # Occasionally mention a price on exchange tickets to test the amount field.
    if cat == "exchange" and rng.random() < 0.4:
        text += f"这件商品{amt}元买的。"
        amount = amt
    angry = rng.random() < 0.2
    text += rng.choice(ANGRY) if angry else rng.choice(POLITE)

    if angry or (amount is not None and amount > 1000):
        pri = "high"
    elif (amount is not None and 200 <= amount <= 1000) or (cat == "shipping" and days > 7):
        pri = "medium"
    else:
        pri = "low"
    ref = {"category": cat, "priority": pri, "order_id": order_id, "amount": amount}
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"工单内容：{text}"},
        ],
        "reference_answer": ref,
        "task_id": f"ticket-{idx:05d}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=400)
    ap.add_argument("--test", type=int, default=150)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    seen, rows = set(), []
    while len(rows) < args.train + args.test:
        row = make_one(rng, len(rows))
        key = row["messages"][1]["content"]
        if key in seen:
            continue
        seen.add(key)
        rows.append(row)
    for name, part in (("train.jsonl", rows[: args.train]), ("test.jsonl", rows[args.train:])):
        with open(name, "w", encoding="utf-8") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(part)} rows to {name}")


if __name__ == "__main__":
    main()
