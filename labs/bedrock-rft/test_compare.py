"""Tests for compare.py and the evaluate.py record format (no AWS, no pytest needed)."""
import json
import random

import compare as cp
import evaluate  # noqa: F401  (must import without openai installed: SDK imports are lazy)
from reward_lambda import score_one

REF = {"category": "refund", "priority": "high", "order_id": "A10234", "amount": 1299}


def rec(task, run, ok, fmt=1, model="m"):
    fields = {"category": 1, "priority": ok, "order_id": 1, "amount": 1}
    return {"model": model, "task_id": task, "run": run, "score": 0.8 + 0.2 * ok, "all_correct": ok,
            "strict_exact": ok, "format": fmt, "fields": fields, "chars": 60}


def test_record_from_score_one():
    s, c = score_one(json.dumps(REF), REF)
    r = cp.record_from("t1", 0, s, c, json.dumps(REF))
    assert r["all_correct"] == 1 and r["strict_exact"] == 1 and r["format"] == 1
    s, c = score_one("好的，这是一张需要优先处理的退款工单，结果如下：" + json.dumps(REF), REF)
    r = cp.record_from("t1", 0, s, c, "")
    assert r["all_correct"] == 1 and r["strict_exact"] == 0 and abs(r["score"] - 0.9) < 1e-9
    s, c = score_one("无法解析", REF)
    r = cp.record_from("t1", 0, s, c, "")
    assert r["format"] == 0 and r["all_correct"] == 0


def test_runs_are_averaged_per_item():
    recs = [rec("a", 0, 1), rec("a", 1, 0), rec("b", 0, 1), rec("b", 1, 1)]
    assert cp.per_item(recs, "all_correct") == {"a": 0.5, "b": 1.0}
    assert cp.summarize(recs)["all_correct"] == 0.75


def test_paired_bootstrap_recovers_known_difference():
    rng = random.Random(1)
    base, ft = [], []
    for i in range(150):
        p = rng.random()                         # item difficulty shared by both models
        for run in range(3):
            base.append(rec(f"t{i}", run, int(rng.random() < p * 0.6)))
            ft.append(rec(f"t{i}", run, int(rng.random() < min(1, p * 0.6 + 0.15))))
    res = cp.paired_bootstrap(base, ft, metrics=["all_correct"], boot=1000, seed=0)
    b, f, d, lo, hi = res["all_correct"]
    assert lo < d < hi and lo > 0.0               # a real +~0.13 gain is detected
    assert abs(d - (f - b)) < 1e-12


def test_bootstrap_no_effect_ci_covers_zero():
    rng = random.Random(2)
    base = [rec(f"t{i}", r, int(rng.random() < 0.4)) for i in range(150) for r in range(3)]
    ft = [rec(f"t{i}", r, int(rng.random() < 0.4)) for i in range(150) for r in range(3)]
    _, _, d, lo, hi = cp.paired_bootstrap(base, ft, metrics=["all_correct"], boot=1000, seed=0)["all_correct"]
    assert lo < 0.0 < hi


def test_icc_extremes():
    same = [rec(f"t{i}", r, i % 2) for i in range(40) for r in range(3)]   # runs are copies
    assert abs(cp.icc(same) - 1.0) < 1e-9
    rng = random.Random(3)
    indep = [rec(f"t{i}", r, int(rng.random() < 0.5)) for i in range(2000) for r in range(3)]
    assert abs(cp.icc(indep)) < 0.06
    assert cp.icc([rec("a", 0, 1)]) is None


def test_no_common_ids_is_an_error():
    try:
        cp.paired_bootstrap([rec("a", 0, 1)], [rec("b", 0, 1)], boot=10)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
