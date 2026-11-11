"""Tests for check_dataset.py (no AWS, no pytest needed)."""
import copy

import check_dataset as cd

SYS = {"role": "system", "content": "sys"}


def row(text, cat="refund", pri="low", oid=None, amt=None):
    return {"messages": [SYS, {"role": "user", "content": f"工单内容：{text}"}],
            "reference_answer": {"category": cat, "priority": pri, "order_id": oid, "amount": amt}}


def test_normalize_and_template_keys():
    a, b = "订单A10234 退款 1299 元！", "订单a10234退款1299元"
    assert cd.normalize_text(a) == cd.normalize_text(b)
    assert cd.template_key("订单A10234退款1299元") == cd.template_key("订单K55555退款39元")
    assert cd.normalize_text("订单A1退款") != cd.normalize_text("订单A2退款")


def test_validate_row_catches_schema_errors():
    assert cd.validate_row(row("订单A1 退款", oid="A1")) == []
    bad = row("没有订单号", oid="A1", pri="urgent")
    errs = cd.validate_row(bad)
    assert any("priority" in e for e in errs) and any("order_id" in e for e in errs)
    no_ref = copy.deepcopy(bad)
    del no_ref["reference_answer"]
    assert cd.validate_row(no_ref) == ["reference_answer missing or not an object"]
    ends_with_answer = row("x")
    ends_with_answer["messages"].append({"role": "assistant", "content": "{}"})
    assert any("assistant" in e for e in cd.validate_row(ends_with_answer))
    assert cd.validate_row({"__error__": "line 3: bad json"}) == ["line 3: bad json"]


def test_size_limits():
    small = {"train": [row(f"订单A{i} 退款", oid=f"A{i}") for i in range(99)], "test": []}
    assert any("100-20,000" in e for e in cd.check(small)["errors"])
    ok = {"train": [row(f"订单A{i} 退款", oid=f"A{i}") for i in range(100)], "test": []}
    assert not cd.check(ok)["errors"]


def test_leakage_after_normalization_is_an_error():
    train = [row(f"订单A{i} 退款", oid=f"A{i}") for i in range(100)]
    test = [row("订单A5，退款！", oid="A5"), row("订单B1 换货", cat="exchange", oid="B1")]
    rep = cd.check({"train": train, "test": test})
    assert any("also in train" in e for e in rep["errors"])


def test_template_overlap_is_a_warning_not_an_error():
    train = [row(f"订单A{i} 退款", oid=f"A{i}") for i in range(100)]
    test = [row(f"订单B{i} 退款", oid=f"B{i}") for i in range(20)]
    rep = cd.check({"train": train, "test": test})
    assert not rep["errors"]
    assert rep["stats"]["test"]["template_overlap_with_train"] == 1.0
    assert any("template" in w for w in rep["warnings"])


def test_dev_test_overlap_is_an_error():
    train = [row(f"订单A{i} 退款", oid=f"A{i}") for i in range(100)]
    shared = row("订单C1 换货", cat="exchange", oid="C1")
    rep = cd.check({"train": train, "test": [shared], "dev": [shared]})
    assert any("share" in e for e in rep["errors"])


def test_difficulty_tags_follow_priority_rules():
    assert cd.difficulty_tags(row("退款 1299 元，我要投诉", amt=1299, pri="high"))[:1] == ["rule1_keyword"]
    assert "both_rule1_triggers" in cd.difficulty_tags(row("退款 1299 元，我要投诉", amt=1299, pri="high"))
    assert cd.difficulty_tags(row("退款 1000 元", amt=1000, pri="medium")) == ["rule2_amount", "boundary_amount"]
    assert cd.difficulty_tags(row("等了 8 天", cat="shipping", pri="medium")) == ["rule2_shipping_days"]
    assert cd.difficulty_tags(row("等了 5 天", cat="shipping")) == ["rule3_default"]
    assert cd.level(["rule3_default"]) == "easy" and cd.level(["rule2_amount", "boundary_amount"]) == "hard"


def test_committed_data_is_clean_and_tags_match_labels():
    splits = {"train": cd.load("train.jsonl"), "test": cd.load("test.jsonl")}
    rep = cd.check(splits)
    assert rep["errors"] == [], rep["errors"]
    expect = {"rule1_keyword": "high", "rule1_amount": "high", "rule2_amount": "medium",
              "rule2_shipping_days": "medium", "rule3_default": "low"}
    for rows in splits.values():
        for r in rows:
            assert expect[cd.difficulty_tags(r)[0]] == r["reference_answer"]["priority"], r
    # Known weakness of the synthetic generator, surfaced by the checker (see lesson 11 section 4).
    assert rep["stats"]["test"]["template_overlap_with_train"] > 0.9
    assert any("too few rows" in w and "account" in w for w in rep["warnings"])
