"""Read the `steps` score back out of Langfuse, the L10 method, one lesson on.

L10's read_scores.py proved `abstained` left the process. This does the same for
`steps`, and then does the thing that only makes sense once BOTH scores are on
the same span: it joins them.

    abstained=1, steps=1   retrieval starved it and it knew immediately
    abstained=1, steps=3   it looked twice and still declined  <- a real corpus gap
    abstained=0, steps=3   it looked twice and talked itself into an answer

That third row is the one this lesson was written to find, and it is invisible to
either score alone.

Usage:  read_trajectory.py <trace-id> [<trace-id> ...]
"""
import base64
import json
import os
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1] / "lesson-001-chunking"
load_dotenv(ROOT / ".env")

HOST = os.environ["LANGFUSE_HOST"].rstrip("/")
AUTH = base64.b64encode(
    f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()
).decode()


def get(path):
    req = urllib.request.Request(f"{HOST}{path}", headers={"Authorization": f"Basic {AUTH}"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


for trace_id in sys.argv[1:]:
    trace = get(f"/api/public/traces/{trace_id}")
    scores = trace.get("scores", [])
    # Both scores hang on the same observation (the per-question span), so the
    # observation id is the join key. This is the payoff for putting them there.
    by_obs = defaultdict(dict)
    for s in scores:
        if s["name"] in ("abstained", "steps"):
            by_obs[s.get("observationId")][s["name"]] = s
            by_obs[s.get("observationId")].setdefault(
                "question", s.get("comment") if s["name"] == "abstained" else None)

    rows = [v for v in by_obs.values() if "steps" in v or "abstained" in v]
    print(f"\n===== trace {trace_id[:12]}… · {trace.get('name')} "
          f"· tags={trace.get('tags')} =====")
    print(f"  {len(rows)} question spans · "
          f"{sum('steps' in r for r in rows)} `steps` · "
          f"{sum('abstained' in r for r in rows)} `abstained`")

    steps_vals, abst_vals = [], []
    for r in sorted(rows, key=lambda r: (-(r.get("steps", {}).get("value", 0)),
                                         -(r.get("abstained", {}).get("value", 0)))):
        st = r.get("steps", {})
        ab = r.get("abstained", {})
        steps_vals += [st["value"]] if "value" in st else []
        abst_vals += [ab["value"]] if "value" in ab else []
        q = (r.get("question") or ab.get("comment") or "")[:62]
        mark = "ABSTAINED" if ab.get("value") == 1 else "answered "
        print(f"  steps={st.get('value', '?')}  {mark}  {st.get('comment', '')[:26]:<26} {q}")

    if steps_vals:
        cap = sum(1 for r in rows if "cap" in (r.get("steps", {}).get("comment") or ""))
        print(f"  mean steps     = {sum(steps_vals) / len(steps_vals):.2f}"
              f"   max = {int(max(steps_vals))}"
              f"   terminated_by=cap: {cap}/{len(steps_vals)} ({cap / len(steps_vals):.0%})")
    if abst_vals:
        print(f"  mean abstained = {sum(abst_vals) / len(abst_vals):.3f}  "
              f"({int(sum(abst_vals))}/{len(abst_vals)})")
