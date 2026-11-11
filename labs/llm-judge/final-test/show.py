"""Print transcripts by id for manual adjudication: python show.py f003 f005 ..."""
import json
import os
import sys

from common import HERE
from label import render

recs = {json.loads(l)["id"]: json.loads(l) for l in open(os.path.join(HERE, "final_raw.jsonl"), encoding="utf-8")}
for i in sys.argv[1:]:
    r = recs[i]
    s = r["spec"]
    print(f"===== {i} defects={s['defects']} subtle={s['subtle']} hn={s['hard_negative']} scen={s['scenario']}")
    print(render(r["transcript"]))
