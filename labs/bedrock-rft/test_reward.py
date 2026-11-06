"""Unit tests for the reward function: run `python -m pytest -q` before deploying."""
import json

from reward_lambda import lambda_handler, score_one

REF = {"category": "refund", "priority": "medium", "order_id": "A12345", "amount": 368}


def s(obj_or_text):
    text = obj_or_text if isinstance(obj_or_text, str) else json.dumps(obj_or_text, ensure_ascii=False)
    return score_one(text, REF)[0]


def test_perfect():
    assert s(REF) == 1.0


def test_unparseable_is_zero():
    assert s("这是退款工单，优先级中") == 0.0


def test_partial_credit_is_monotonic():
    one_wrong = dict(REF, priority="low")
    two_wrong = dict(REF, priority="low", amount=None)
    assert 1.0 > s(one_wrong) > s(two_wrong) > 0.0


def test_extra_keys_penalized():
    assert s(dict(REF, reason="因为用户要退款")) < 1.0


def test_case_and_number_formats():
    assert s(dict(REF, order_id="a12345", amount="368.0", category="Refund")) == 1.0


def test_reasoning_then_json():
    assert s("<think>先看金额</think>答案：" + json.dumps(REF)) == 1.0


def test_null_fields():
    ref = {"category": "account", "priority": "low", "order_id": None, "amount": None}
    assert score_one(json.dumps(ref), ref)[0] == 1.0
    assert score_one(json.dumps(dict(ref, amount=0)), ref)[0] < 1.0


def test_handler_contract():
    event = [{"id": "t1", "messages": [{"role": "user", "content": "x"},
                                       {"role": "assistant", "content": json.dumps(REF)}],
              "reference_answer": REF},
             {"id": "t2", "messages": [], "metadata": {"reference_answer": json.dumps(REF)}}]
    out = lambda_handler(event, None)
    assert [o["id"] for o in out] == ["t1", "t2"]
    assert out[0]["aggregate_reward_score"] == 1.0
    assert out[1]["aggregate_reward_score"] == 0.0
    assert all(0.0 <= o["aggregate_reward_score"] <= 1.0 for o in out)
