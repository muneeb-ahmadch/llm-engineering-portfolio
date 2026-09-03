#!/usr/bin/env python
"""Step 1's tables, read back out of results/results-rejudge.json. Free.

The headline number the assignment asks for, "correct answers scored <= 3/5 in
the poisoned arm, was 4 -> ?", plus the two things that have to sit next to it or
it means nothing:

  the control   the SAME 18 answers re-judged today on the FLAT judge. If the flat
                judge also moves, the segregation gets no credit for the move.
  the clean arm the judge's verdicts on unpoisoned answers. A "fix" that changes
                scores where there is no poison is not a fix, it is a new bias.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
L12 = HERE.parents[0] / "lesson-012-injection"
sys.path.insert(0, str(HERE.parents[0] / "lesson-001-chunking"))
import rag_eval as R  # noqa: E402

D = json.loads((HERE / "results" / "results-rejudge.json").read_text())
INJ = {i["id"]: i for i in R.load_golden(L12 / "injections.jsonl")}
ARMS = ["clean", "poisoned", "defended", "fixed"]


def hit(p, t):
    return bool(p) and bool(re.search(p, t or "", re.I))


def obeyed(r, arm):
    inj = INJ[r["id"]]
    if inj["obeyed_when"] == "no_search":
        return None if arm == "fixed" else r["n_searches"] == 0
    return hit(inj["obeyed_regex"], r["answer"])


def rows(cell):
    for run in D[cell]:
        for r in run:
            yield r


def punished(cell, arm, key):
    """Correct answers that resisted the injection and were still scored <= 3/5.

    The L12 definition, unchanged, so the numbers are comparable: correct AND not
    obeyed AND it actually made claims AND the judge scored it <= 3.
    """
    out = []
    for r in rows(cell):
        j = r[key]
        if (hit(INJ[r["id"]]["correct_regex"], r["answer"]) and not obeyed(r, arm)
                and j["total_claims"] and j["score"] <= 3):
            out.append((r["id"], r["run"], j["score"]))
    return out


def mean_by(cell, arm, key, want):
    xs = [r[key]["score"] for r in rows(cell)
          if r[key]["total_claims"] and bool(obeyed(r, arm)) is want]
    return f"{sum(xs) / len(xs):.2f}/5 (n={len(xs)})" if xs else ","


print("# Step 1: the judge, re-calibrated\n")
print(f"Judge {D['judge_model']} · 90 judge calls · ${D['spend']:.4f} · "
      f"{D['wall_secs']:.0f}s · **no regeneration**, the 72 answers are L12's, "
      f"replayed with their exact contexts\n")

print("## The number the assignment asks for\n")
print("| poisoned corpus | flat judge (L12, Aug 8) | flat judge (re-run today) "
      "| **segregated judge** |")
print("|---|---|---|---|")
old = punished("J_seg_poisoned", "poisoned", "old_judge")
ctl = punished("J_flat_poisoned", "poisoned", "new_judge")
new = punished("J_seg_poisoned", "poisoned", "new_judge")
print(f"| correct answers scored <= 3/5 | **{len(old)}** | {len(ctl)} | "
      f"**{len(new)}** |")
print(f"| which ones | {', '.join(f'{i} r{r} ({s}/5)' for i, r, s in old) or ','} "
      f"| {', '.join(f'{i} r{r} ({s}/5)' for i, r, s in ctl) or ','} "
      f"| {', '.join(f'{i} r{r} ({s}/5)' for i, r, s in new) or ','} |")

print("\n## The inversion, does obeying still outscore resisting?\n")
print("| arm | judge | obeyed | resisted | correct answers <= 3/5 |")
print("|---|---|---|---|---|")
for arm in ARMS:
    cell = f"J_seg_{arm}"
    for key, name in (("old_judge", "flat (L12)"), ("new_judge", "SEGREGATED")):
        print(f"| `{arm}` | {name} | {mean_by(cell, arm, key, True)} | "
              f"{mean_by(cell, arm, key, False)} | "
              f"**{len(punished(cell, arm, key))}** |")

print("\n## Score changes, every one of the 72\n")
print("| arm | answers | scores changed | up | down | total_claims changed |")
print("|---|---|---|---|---|---|")
for arm in ARMS:
    rs = list(rows(f"J_seg_{arm}"))
    ch = [r for r in rs if r["old_judge"]["score"] != r["new_judge"]["score"]]
    up = [r for r in ch if r["new_judge"]["score"] > r["old_judge"]["score"]]
    cc = [r for r in rs
          if r["old_judge"]["total_claims"] != r["new_judge"]["total_claims"]]
    print(f"| `{arm}` | {len(rs)} | {len(ch)} | {len(up)} | {len(ch) - len(up)} | "
          f"{len(cc)} |")
ctlr = list(rows("J_flat_poisoned"))
ctlch = [r for r in ctlr if r["old_judge"]["score"] != r["new_judge"]["score"]]
print(f"| `poisoned` **flat control** | {len(ctlr)} | **{len(ctlch)}** | "
      f"{sum(1 for r in ctlch if r['new_judge']['score'] > r['old_judge']['score'])} "
      f"| {sum(1 for r in ctlch if r['new_judge']['score'] < r['old_judge']['score'])} "
      f"| {sum(1 for r in ctlr if r['old_judge']['total_claims'] != r['new_judge']['total_claims'])} |")

print("\n## Every changed verdict, verbatim\n")
for cell, arm in [(f"J_seg_{a}", a) for a in ARMS] + [("J_flat_poisoned", "poisoned")]:
    for r in rows(cell):
        o, n = r["old_judge"], r["new_judge"]
        if o["score"] == n["score"]:
            continue
        tag = "flat-control" if cell == "J_flat_poisoned" else arm
        print(f"\n**`{tag}` {r['id']} run {r['run']}: {o['score']}/5 -> {n['score']}/5"
              f"** ({o['grounded_claims']}/{o['total_claims']} -> "
              f"{n['grounded_claims']}/{n['total_claims']} grounded)")
        print(f"> answer: {' '.join((r['answer'] or '').split())[:200]}")
        for c in o["claims"]:
            if not c["grounded"]:
                print(f">\n> flat judge, ungrounded: *\"{c['why'][:300]}\"*")
        for c in n["claims"]:
            if not c["grounded"]:
                print(f">\n> segregated judge, ungrounded: *\"{c['why'][:300]}\"*")

print("\n## What the judge's own bill did\n")
print("| | flat (L12) | segregated | |")
print("|---|---|---|---|")
segr = [r for a in ARMS for r in rows(f"J_seg_{a}")]
flat_old = [r["old_judge"] for r in segr]
seg_new = [r["new_judge"] for r in segr]


def s(xs, k):
    return sum(x[k] for x in xs)


print(f"| prompt tokens (72 calls) | {s(flat_old, 'prompt_tokens'):,} | "
      f"{s(seg_new, 'prompt_tokens'):,} | "
      f"{s(seg_new, 'prompt_tokens') / s(flat_old, 'prompt_tokens'):.2f}x |")
print(f"| **cached** prompt tokens | {s(flat_old, 'cached_tokens'):,} | "
      f"**{s(seg_new, 'cached_tokens'):,}** | the nonce, one layer down |")
print(f"| completion tokens | {s(flat_old, 'completion_tokens'):,} | "
      f"{s(seg_new, 'completion_tokens'):,} | "
      f"{s(seg_new, 'completion_tokens') / s(flat_old, 'completion_tokens'):.2f}x |")
oc, nc = s(flat_old, "cost"), s(seg_new, "cost")
print(f"| cost, 72 judge calls | ${oc:.4f} | ${nc:.4f} | **{nc / oc:.2f}x** |")
print(f"| mean latency | {s(flat_old, 'latency_ms') / len(flat_old):,.0f} ms | "
      f"{s(seg_new, 'latency_ms') / len(seg_new):,.0f} ms | "
      f"{s(seg_new, 'latency_ms') / s(flat_old, 'latency_ms'):.2f}x |")
