"""Tests for make_dataset.py (no AWS, no pytest needed)."""
import collections
import json
import random
import re

import make_dataset as md
from check_dataset import oid_key, user_text

_CN = {c: i for i, c in enumerate("零一二三四五六七八九")} | {"两": 2}
_UNIT = {"十": 10, "百": 100, "千": 1000}


def parse_cn(s):
    """Independent Chinese-numeral parser used only to check cn_number."""
    total, digit = 0, None
    for ch in s:
        if ch in _CN:
            digit = _CN[ch]
        else:
            total += (1 if digit is None else digit) * _UNIT[ch]
            digit = None
    return total + (digit or 0)


def splits(seed=7, dev=0):
    seen = set()
    out = {"train": md.make_split(400, "train", seed, md.SYSTEM_PROMPT, seen),
           "test": md.make_split(150, "test", seed, md.SYSTEM_PROMPT, seen, 1 / 3)}
    if dev:
        out["dev"] = md.make_split(dev, "dev", seed, md.SYSTEM_PROMPT, seen)
    return out


def test_cn_number():
    cases = {1: "一", 10: "十", 12: "十二", 20: "二十", 200: "二百", 205: "二百零五", 1000: "一千", 1001: "一千零一",
             1010: "一千零一十", 1299: "一千二百九十九", 3888: "三千八百八十八", 450: "四百五十"}
    for n, s in cases.items():
        assert md.cn_number(n) == s, (n, md.cn_number(n))
    assert md.cn_number(2499, liang=True) == "两千四百九十九" and md.cn_number(200, liang=True) == "两百"
    for n in range(1, 10000):
        assert parse_cn(md.cn_number(n)) == n, n


def test_rendered_slots_round_trip():
    rng = random.Random(0)
    for v in md.AMOUNTS:
        for fmt in ("yuan", "kuai", "yen", "cn", "comma", "rmb"):
            s = md.render_amount(v, fmt, rng)
            num = parse_cn(re.sub(r"[元块]", "", s)) if fmt == "cn" else float(re.sub(r"[^\d.]", "", s))
            assert num == v, (v, fmt, s)
    for fmt in ("plain", "lower", "hyphen", "hash"):
        assert oid_key(md.render_oid("A10234", fmt)) == "A10234"
    for fmt, expect in (("week", {7, 14, 21}), ("halfmonth", {15}), ("month", {30})):
        assert md.pick_days(fmt, rng)[0] in expect


def test_priority_rules():
    assert md.priority_of("refund", 1001, None, False) == "high"
    assert md.priority_of("refund", 1000, None, False) == "medium"
    assert md.priority_of("refund", 200, None, False) == "medium"
    assert md.priority_of("refund", 199.9, None, False) == "low"
    assert md.priority_of("shipping", None, 7, False) == "low"
    assert md.priority_of("shipping", None, 8, False) == "medium"
    assert md.priority_of("account", None, None, True) == "high"
    assert md.priority_of("exchange", None, 30, False) == "low"  # waiting time only counts for shipping


def test_deterministic_and_dev_does_not_change_train_test():
    a, b = splits(), splits(dev=100)
    for name in ("train", "test"):
        assert json.dumps(a[name], ensure_ascii=False) == json.dumps(b[name], ensure_ascii=False)
    assert json.dumps(splits(seed=8)["train"]) != json.dumps(a["train"])


def test_split_by_construction():
    s = splits(dev=100)
    ids = {k: {r["template_id"] for r in v} for k, v in s.items()}
    assert not ids["train"] & ids["test"] and not ids["train"] & ids["dev"] and not ids["dev"] & ids["test"]
    for name, rows in s.items():
        counts = collections.Counter(r["reference_answer"]["category"] for r in rows)
        assert len(set(counts.values())) == 1, (name, counts)  # balanced classes
    held = md.HELD_OUT
    for name in ("train", "dev"):
        assert not any(held & set(r["formats"].split(",")) for r in s[name]), name
    test = s["test"]
    assert collections.Counter(r["slice"] for r in test) == {"new_phrasing": 100, "ood": 50}
    for r in test:
        assert (r["slice"] == "ood") == md.has_held_out_format(r), r
    angry_train = set(md.ANGRY["train"])
    assert not any(user_text(r).endswith(tuple(angry_train)) for r in test)


def test_labels_are_consistent_with_text():
    for rows in splits(dev=100).values():
        for r in rows:
            ref, text = r["reference_answer"], user_text(r)
            if ref["order_id"]:
                assert oid_key(ref["order_id"]) in oid_key(text)
            if r["wait_days"] is not None:
                assert ref["category"] == "shipping"
            if ref["category"] in ("account",):
                assert ref["amount"] is None
            fmt = dict(t.split(":") for t in r["formats"].split(",") if t)
            if ref["amount"] is not None and fmt.get("amt") in ("yuan", "kuai", "yen", "rmb", "comma"):
                nums = [float(x.replace(",", "")) for x in re.findall(r"\d[\d,]*(?:\.\d+)?", text)]
                assert ref["amount"] in nums, (ref, text)


def test_committed_files_match_generator():
    s = splits()
    for name in ("train", "test"):
        with open(f"{name}.jsonl", encoding="utf-8") as f:
            assert [json.loads(line) for line in f] == s[name], f"{name}.jsonl is stale: rerun make_dataset.py"
