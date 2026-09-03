"""Read the `abstained` scores back out of Langfuse, the diagnosis step.

Proves the score landed on the trace (not just in local memory) and shows the
per-question verdict, which is where the disagreements live.
"""
import base64, json, os, sys, urllib.request
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path("/Users/apple/Documents/AI Bootcamp/The Great Learning/practice/lesson-001-chunking")
load_dotenv(ROOT / ".env")

HOST = os.environ["LANGFUSE_HOST"].rstrip("/")
AUTH = base64.b64encode(
    f"{os.environ['LANGFUSE_PUBLIC_KEY']}:{os.environ['LANGFUSE_SECRET_KEY']}".encode()
).decode()


def get(path):
    req = urllib.request.Request(f"{HOST}{path}", headers={"Authorization": f"Basic {AUTH}"})
    with urllib.request.urlopen(req) as r:
        return json.load(r)


for label, trace_id in [("GOLDEN", sys.argv[1]), ("TRAP", sys.argv[2])]:
    trace = get(f"/api/public/traces/{trace_id}")
    scores = [s for s in trace.get("scores", []) if s["name"] == "abstained"]
    vals = [s["value"] for s in scores]
    print(f"\n===== {label}, {len(scores)} `abstained` scores on trace {trace_id[:12]}… =====")
    for s in sorted(scores, key=lambda s: -s["value"]):
        mark = "ABSTAINED" if s["value"] == 1 else "answered "
        print(f"  [{int(s['value'])}] {mark}  {(s['comment'] or '')[:80]}")
    if vals:
        print(f"  mean abstained = {sum(vals)/len(vals):.3f}  ({int(sum(vals))}/{len(vals)})")
