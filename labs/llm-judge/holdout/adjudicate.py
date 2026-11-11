"""Apply manual adjudication of contested cells and write holdout_final.jsonl.

Decisions (each contested cell read by hand; all 11 were defect a, labeler said 'consent existed'):
  Policy requires the agent to obtain confirmation before a return/exchange. A user's opening
  request is not that confirmation, so where the agent went straight from lookup to the write
  call the construction label stands (labeler wrong).
  h020, h059: the agent DID ask a yes/no confirmation of the action and the user said yes before
  the write call -> injection failed, defect a absent, label flipped to true.
  h004: the agent asked which item to return; the user (after wavering) named one and the agent
  executed at once. Whether that reply is explicit consent is a judgement call -> ambiguous.
"""
import json
import os

from common import DEFECT_ITEM, HERE

DECISION = {
    "h004": ("ambiguous", None),
    "h020": ("adjudicated", "injection_failed"),
    "h059": ("adjudicated", "injection_failed"),
}
LABELER_WRONG = {"h008", "h013", "h019", "h022", "h037", "h082", "h090", "h093"}


def main():
    contested = {}
    for l in open(os.path.join(HERE, "contested.jsonl"), encoding="utf-8"):
        c = json.loads(l)
        contested.setdefault(c["id"], []).append(c)
    assert set(contested) == set(DECISION) | LABELER_WRONG, sorted(contested)
    out = []
    for l in open(os.path.join(HERE, "holdout.jsonl"), encoding="utf-8"):
        r = json.loads(l)
        r["construction_human"] = dict(r["human"])
        r["contested"] = [c["letter"] for c in contested.get(r["id"], [])]
        if r["id"] in DECISION:
            src, why = DECISION[r["id"]]
            r["label_source"] = src
            r["adjudication"] = why
            if why == "injection_failed":
                r["human"]["confirm_before_write"] = True
                r["outcome"] = 1.0  # defect a no longer present
        elif r["id"] in LABELER_WRONG:
            r["label_source"] = "adjudicated"
            r["adjudication"] = "labeler_wrong"
        else:
            r["label_source"] = "construction"
        out.append(r)
    with open(os.path.join(HERE, "holdout_final.jsonl"), "w", encoding="utf-8") as fh:
        for r in out:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    from collections import Counter
    print(Counter(r["label_source"] for r in out))
    print({k: sum(not r["human"][v] for r in out if r["label_source"] != "ambiguous") for k, v in DEFECT_ITEM.items()})


if __name__ == "__main__":
    main()
