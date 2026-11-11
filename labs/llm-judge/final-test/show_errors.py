"""Print, for each judge error, the wrong items and the evidence/reason from each vote."""
import json
import os

from common import HERE

for l in open(os.path.join(HERE, "errors.jsonl"), encoding="utf-8"):
    e = json.loads(l)
    print(f"== {e['id']} {e['spec']['scenario']} defects={e['spec']['defects']} subtle={e['spec']['subtle']} "
          f"hn={e['spec']['hard_negative']} src={e['label_source']} wrong={e['wrong_items']}")
    for v in e["votes"]:
        if "text" not in v:
            continue
        s, t = v["text"].find("{"), v["text"].rfind("}")
        try:
            items = json.loads(v["text"][s:t + 1])["items"]
        except Exception:
            continue
        for it in items:
            if it.get("id") in e["wrong_items"]:
                print(f"   [{it['id']}] pass={it.get('pass')} ev={it.get('evidence', '')[:80]} | {it.get('reason', '')[:110]}")
