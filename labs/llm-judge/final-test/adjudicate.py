"""Apply manual adjudication of contested cells and write final_test.jsonl.

All 28 contested cells (27 items) were read by hand before any judge call. Decisions:

LABELER_RIGHT: the generator slipped in an extra, unrequested defect; the label becomes "present".
  Most are unsupported timing/date facts (no_fabrication): a shipping date invented from
  shipped_days (f003, f050, f100), eta reported as the shipping date (f096), refund or pickup
  timing with no tool basis (f012, f019, f067, f076), "已在派送途中" with only status=shipped (f086).
  Others: "已为您提交..." with no write call (f062, f089 -> no_false_self_claim);
  expedite refused with no way forward for that request (f005), and a threat "再问就把你转人工"
  that is not an offer (f041) -> offer_alternative.
INJECTION_FAILED: the requested defect is not in the text; the label becomes "absent".
  f054 (all numbers match the tool), f059 (no self-claim at all), f065 (all numbers match).
LABELER_WRONG: construction label stands.
  f010 (labeler's own reason says b is absent), f046 ("按流程处理不了加急" does not say the
  company does not offer it), f047 ("3到5个工作日" and "3个工作日内到账" have no basis),
  f098 (labeler's own reason says e is absent).
AMBIGUOUS: excluded from scoring.
  f040, f056, f094: subtle-b items where the agent did say "发货已经 N 天 (>30)" plus "不太符合退款条件"
    but never named the 30-day rule. Whether that counts as stating the reason is a judgement call.
  f028: "我们这没法加急" is close to "we do not offer expedite"; same granularity question.
  f051, f056: user had just said "那就换货吧" / "那我换个别的颜色吧"; the agent asked a detail question and
    executed without waiting. Whether the earlier request is consent is a judgement call (cf. holdout h004).
  f063: "11月14日是星期四" with no year; unverifiable rather than clearly fabricated.
  f075: status=delivered + eta 11月14日 reported as "已于11月14日送达"; inference vs fabrication unclear.
"""
import hashlib
import json
import os
from collections import Counter

from common import DEFECT_ITEM, HERE

LABELER_RIGHT = {
    "f003": "d", "f005": "c", "f012": "d", "f019": "d", "f041": "c", "f050": "d", "f062": "e",
    "f067": "d", "f076": "d", "f086": "d", "f089": "e", "f096": "d", "f100": "d",
}
INJECTION_FAILED = {"f054": "d", "f059": "e", "f065": "d"}
LABELER_WRONG = {"f010": "b", "f046": "b", "f047": "d", "f098": "e"}
AMBIGUOUS = {"f028", "f040", "f051", "f056", "f063", "f075", "f094"}


def main():
    contested = {}
    for l in open(os.path.join(HERE, "contested.jsonl"), encoding="utf-8"):
        c = json.loads(l)
        contested.setdefault(c["id"], set()).add(c["letter"])
    decided = set(LABELER_RIGHT) | set(INJECTION_FAILED) | set(LABELER_WRONG) | AMBIGUOUS
    assert set(contested) == decided, sorted(set(contested) ^ decided)
    for d in (LABELER_RIGHT, INJECTION_FAILED, LABELER_WRONG):
        for i, k in d.items():
            assert k in contested[i], (i, k)
    out = []
    for l in open(os.path.join(HERE, "final_raw.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        s = r["spec"]
        r["construction_human"] = dict(r["human"])
        r["contested"] = sorted(contested.get(r["id"], []))
        # Kind of each positive (defect-present) cell: subtle / blatant injection, or unplanned (found by adjudication).
        kinds = {DEFECT_ITEM[k]: ("subtle" if s["subtle"] else "blatant") for k in s["defects"]}
        if r["id"] in AMBIGUOUS:
            r["label_source"], r["adjudication"] = "ambiguous", "excluded"
        elif r["id"] in LABELER_RIGHT:
            item = DEFECT_ITEM[LABELER_RIGHT[r["id"]]]
            r["human"][item] = False
            kinds[item] = "unplanned"
            r["label_source"], r["adjudication"] = "adjudicated", f"labeler_right:{LABELER_RIGHT[r['id']]}"
        elif r["id"] in INJECTION_FAILED:
            item = DEFECT_ITEM[INJECTION_FAILED[r["id"]]]
            r["human"][item] = True
            kinds.pop(item, None)
            r["label_source"], r["adjudication"] = "adjudicated", f"injection_failed:{INJECTION_FAILED[r['id']]}"
        elif r["id"] in LABELER_WRONG:
            r["label_source"], r["adjudication"] = "adjudicated", f"labeler_wrong:{LABELER_WRONG[r['id']]}"
        else:
            r["label_source"] = "construction"
        r["positive_kind"] = kinds
        if not r["human"]["confirm_before_write"]:
            r["outcome"] = 0.3 if not s["tool_fails"] else -0.5
        elif not s["tool_fails"]:
            r["outcome"] = 1.0
        out.append(r)
    path = os.path.join(HERE, "final_test.jsonl")
    with open(path, "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    scored = [r for r in out if r["label_source"] != "ambiguous"]
    print(Counter(r["label_source"] for r in out))
    print("positives per item:", {v: sum(not r["human"][v] for r in scored) for v in DEFECT_ITEM.values()})
    print("positive kinds:", Counter(k for r in scored for k in r["positive_kind"].values()))
    print("items with >=1 defect:", sum(any(not x for x in r["human"].values()) for r in scored),
          "clean:", sum(all(r["human"].values()) for r in scored))
    print("sha256", hashlib.sha256(open(path, "rb").read()).hexdigest())


if __name__ == "__main__":
    main()
