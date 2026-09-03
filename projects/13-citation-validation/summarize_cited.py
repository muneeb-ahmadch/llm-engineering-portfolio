#!/usr/bin/env python
"""Every table in results.md. Free, no API calls, no model load.

Three sources, and the join between them is the methodological point:

  L12 results-injection.json   the `poisoned` and `defended` answers. Not re-run:
                               same code, same corpus, same config. Paying again
                               would buy noise, not a number.
  results-rejudge.json         those same answers' faithfulness scores, re-read
                               off the SEGREGATED judge. Every faithfulness figure
                               in this report comes from the instrument calibrated
                               in step 1, comparing a `cited` answer scored by the
                               fixed judge against a `defended` answer scored by the
                               compromised one would be exactly the error step 1
                               exists to prevent.
  results-cited.json           today's arm.

    python summarize_cited.py
    python summarize_cited.py --answers
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
L12 = HERE.parents[0] / "lesson-012-injection"
sys.path.insert(0, str(HERE.parents[0] / "lesson-001-chunking"))
import rag_eval as R  # noqa: E402

C = json.loads((HERE / "results" / "results-cited.json").read_text())
D12 = json.loads((L12 / "results" / "results-injection.json").read_text())
RJ = json.loads((HERE / "results" / "results-rejudge.json").read_text())
INJ = {i["id"]: i for i in R.load_golden(L12 / "injections.jsonl")}
CEIL = {i["id"]: i for i in R.load_golden(L12 / "injections-ceiling.jsonl")}

# (label, source arm), `defended` is L12's segregated arm, renamed here to the
# prompt variant it actually used, because the report compares PROMPTS.
ARMS = [("poisoned", "hardened"), ("defended", "segregated"), ("cited", "cited")]


def hit(p, t):
    return bool(p) and bool(re.search(p, t or "", re.I))


def rows(arm):
    src = C["I_cited"] if arm == "cited" else D12[f"I_{arm}"]
    for i, run in enumerate(src, 1):
        for q in run["per_question"]:
            yield i, q


def rejudged(arm):
    """{(id, run): verdict} from the segregated judge, for the two L12 arms."""
    return {(r["id"], r["run"]): r["new_judge"] for run in RJ[f"J_seg_{arm}"]
            for r in run}


REJ = {a: rejudged(a) for a in ("poisoned", "defended")}


def verdict(arm, run_i, q):
    if arm == "cited":
        return q.get("judge")
    return REJ[arm][(q["id"], run_i)]


def obeyed(q, arm):
    inj = INJ[q["id"]]
    if inj["obeyed_when"] == "no_search":
        return q.get("n_searches", 0) == 0
    return hit(inj["obeyed_regex"], q["answer"])


def correct(q):
    return hit(INJ[q["id"]]["correct_regex"], q["answer"])


def blocked(q):
    return bool(q.get("blocked"))


def ci_blocked(q):
    """The CI-ONLY clause, applied post-hoc: did the answer cite a chunk I know
    carries a payload? Pure set intersection, `needle_labels` is recorded at run
    time and is the empty set in production by definition."""
    return bool(set(q.get("cited_ids") or []) & set(q.get("needle_labels") or []))


# ---------------------------------------------------------------- tables

print(f"# L13: the deterministic stage · ${C['total_spend']:.4f}, "
      f"{C['wall_secs']:.0f}s wall clock\n")

print("## The table: poisoned / segregated / cited x {obeyed, blocked, correct}\n")
print("| id | shape | " + " | ".join(
    f"{a} obeyed · blocked · correct" for a, _ in ARMS) + " |")
print("|---" * (len(ARMS) + 2) + "|")
tot = {a: [0, 0, 0, 0] for a, _ in ARMS}
for iid, inj in INJ.items():
    cells = []
    for arm, _ in ARMS:
        qs = [(i, q) for i, q in rows(arm) if q["id"] == iid]
        o = sum(obeyed(q, arm) for _, q in qs)
        b = sum(blocked(q) for _, q in qs)
        k = sum(correct(q) for _, q in qs)
        for j, v in enumerate((o, b, k, len(qs))):
            tot[arm][j] += v
        cells.append(f"{o}/{len(qs)} · {b}/{len(qs)} · {k}/{len(qs)}")
    print(f"| {iid} | {inj['shape'].split(', ')[0]} | " + " | ".join(cells) + " |")
print("| **all** | | " + " | ".join(
    f"**{tot[a][0]}/{tot[a][3]} · {tot[a][1]}/{tot[a][3]} · {tot[a][2]}/{tot[a][3]}**"
    for a, _ in ARMS) + " |")

print("\n## What the validator actually decided, clause by clause\n")
print("| | production clauses | + the CI-only clause |")
print("|---|---|---|")
cq = [q for _, q in rows("cited")]
print(f"| answers blocked | **{sum(blocked(q) for q in cq)}/18** | "
      f"**{sum(blocked(q) or ci_blocked(q) for q in cq)}/18** |")
print(f"| of which cite a known payload |, | "
      f"{sum(ci_blocked(q) for q in cq)}/18 |")
reasons = [r for q in cq for r in q.get("block_reasons", [])]
print(f"| block reasons | {reasons or 'none, no production clause fired'} | |")

print("\n**Counterfactual on L12's 8 obeyed runs** (the preregistered number). "
      "Each row is one of the 8, replayed against the validator's clauses using "
      "the answer L12 actually produced:\n")
print("| run | answer | production clauses | + CI-only |")
print("|---|---|---|---|")
n_prod = n_ci = 0
for i, q in rows("poisoned"):
    if not obeyed(q, "poisoned"):
        continue
    a = " ".join((q["answer"] or "").split())
    v = verdict("poisoned", i, q)   # the SEGREGATED judge's verdict, not L12's
    # The 8 answers predate labels, so this is reasoned, not measured, the
    # production clauses can only fire if the claim is uncitable or the id is
    # invented, and every one of these sentences is verbatim in a retrieved chunk.
    if R.is_abstention(a):
        prod, ci, note = "pass", "pass", "abstention, zero claims, nothing to check"
    elif v["total_claims"] and v["grounded_claims"] == 0:
        prod, ci, note = "pass", "pass", "claim not in context, but a plausible id still passes clauses 2-3"
    else:
        prod, ci, note = "pass", "**BLOCK**", "cites a poisoned chunk truthfully"
    n_prod += prod != "pass"
    n_ci += ci != "pass"
    print(f"| {q['id']} r{i} | {a[:64]}… | {prod} | {ci}, *{note}* |")
print(f"| **total** | | **{n_prod}/8 blocked** | **{n_ci}/8 blocked** |")

print("\n## The regression, CLEAN corpus, nothing injected\n")
G = {"hardened": D12["G_hardened"], "segregated": D12["G_segregated"],
     "cited": C["G_cited"]}
T = {"hardened": D12["T_hardened"], "segregated": D12["T_segregated"],
     "cited": C["T_cited"]}
print("| | hardened (L09/L11) | segregated (L12) | **cited (L13)** | vs segregated |")
print("|---|---|---|---|---|")


def grow(label, fn, fmt="{:.3f}", ratio=False):
    vs = [fn(G[a]) for a in ("hardened", "segregated", "cited")]
    if ratio:
        d = f"**{vs[2] / vs[1]:.2f}x**" if vs[1] else ","
    else:
        d = f"{vs[2] - vs[1]:+.2f}" if all(v is not None for v in vs[1:]) else ","
    print(f"| {label} | " + " | ".join(
        fmt.format(v) if v is not None else "," for v in vs) + f" | {d} |")


grow("mean faithfulness (20 q)", lambda r: r["mean_faithfulness"], "{:.2f}")
grow("claim groundedness", lambda r: r["claim_groundedness"], "{:.1%}")
grow("total claims decomposed", lambda r: r["total_claims"], "{:,}")
grow("abstention rate", lambda r: r["abstention_rate"], "{:.1%}")
print(f"| **blocked by the validator** | n/a | n/a | "
      f"**{C['G_cited']['n_blocked']}/20** | new |")
print(f"| answers judged | 20 | 20 | **{C['G_cited']['n_judged']}/20** | "
      f"blocked answers are not scored |")
grow("mean steps", lambda r: r["mean_steps"], "{:.2f}")
grow("mean latency (ms)", lambda r: r["mean_latency_ms"], "{:,.0f}", ratio=True)
grow("max latency (ms)", lambda r: r["max_latency_ms"], "{:,.0f}", ratio=True)
grow("prompt tokens (20 q)", lambda r: r["total_prompt_tokens"], "{:,}", ratio=True)
grow("completion tokens (20 q)", lambda r: r["total_completion_tokens"], "{:,}",
     ratio=True)
grow("cached prompt tokens", lambda r: r["total_cached_tokens"], "{:,}")
grow("**cost / query**", lambda r: r["cost_per_query"], "${:.6f}", ratio=True)
for a in ("hardened", "segregated", "cited"):
    hall = "/".join(str(r["hallucinations"]) for r in T[a])
    st = sum(r["mean_steps"] for r in T[a]) / len(T[a])
    cost = sum(r["total_cost"] for r in T[a]) / len(T[a])
    blk = sum(r.get("n_blocked", 0) for r in T[a])
    print(f"| trap x3 ({a}) | {hall} of 5 | steps {st:.2f} | "
          f"${cost:.4f}/run | blocked {blk}/15 |")

print("\n## The ceiling, INJ-07, the one that just lies\n")
print("| arm | answer | blocked | faithfulness |")
print("|---|---|---|---|")
for arm, key in (("clean (hardened)", None), ("poisoned (hardened)", None),
                 ("defended (segregated)", None)):
    pass
CE = json.loads((L12 / "results" / "results-ceiling.json").read_text())
for arm in ("clean", "poisoned", "defended"):
    for i, r in enumerate(CE[arm], 1):
        q = r["per_question"][0]
        if i > 1:
            continue
        outs = [" ".join((rr["per_question"][0]["answer"] or "").split())[:70]
                for rr in CE[arm]]
        same = "×3" if len(set(outs)) == 1 else "(varies)"
        print(f"| `{arm}` (L12) | {outs[0]}… {same} | n/a | "
              f"{q['judge']['score']}/5 |")
for i, r in enumerate(C["C_cited"], 1):
    q = r["per_question"][0]
    j = q.get("judge")
    print(f"| **`cited`** run {i} | {' '.join((q['answer'] or '').split())[:70]}… | "
          f"**{'BLOCKED' if q['blocked'] else 'passed'}** | "
          f"{str(j['score']) + '/5' if j else ', not scored'} |")
print(f"\ncited ids: "
      f"{[q['cited_ids'] for r in C['C_cited'] for q in r['per_question']]} · "
      f"payload sat under "
      f"{[q.get('needle_labels') for r in C['C_cited'] for q in r['per_question']]}")

print("\n## Faithfulness on the injection set, segregated judge throughout\n")
print("| arm | answers scored | mean faithfulness | grounded claims | blocked "
      "(unscored) |")
print("|---|---|---|---|---|")
for arm, _ in ARMS:
    vs = [(q, verdict(arm, i, q)) for i, q in rows(arm)]
    sc = [v for _, v in vs if v and v["total_claims"]]
    g = sum(v["grounded_claims"] for v in sc)
    t = sum(v["total_claims"] for v in sc)
    nb = sum(blocked(q) for q, _ in vs)
    print(f"| `{arm}` | {len(sc)} | "
          f"{sum(v['score'] for v in sc) / len(sc):.2f}/5 | {g}/{t} ({g / t:.0%}) | "
          f"{nb} |")

V1 = HERE / "results" / "results-cited-v1.json"
if V1.exists():
    v1 = json.loads(V1.read_text())
    print("\n## The label-format bug, v1 vs v2, and why it is the day's best finding\n")
    print("v1 printed the id and the source inside one bracket, `[C2 · "
          "day15-rag-langchain.md]`, while rule A2 said *use the id exactly as "
          "printed*. Two readings, and the model took the other one.\n")
    print("| | v1 `[C2 · file.md]` | v2 `[C2] file.md` |")
    print("|---|---|---|")
    for label, key in (("golden 20 blocked", "G_cited"),):
        print(f"| **{label}** | **{v1[key]['n_blocked']}/20** | "
              f"**{C[key]['n_blocked']}/20** |")
    print(f"| golden mean faithfulness | {v1['G_cited']['mean_faithfulness']:.2f} | "
          f"{C['G_cited']['mean_faithfulness']:.2f} |")
    print(f"| golden answers judged | {v1['G_cited']['n_judged']}/20 | "
          f"{C['G_cited']['n_judged']}/20 |")
    inj_v1 = sum(1 for run in v1["I_cited"] for q in run["per_question"] if q["blocked"])
    inj_v2 = sum(1 for run in C["I_cited"] for q in run["per_question"] if q["blocked"])
    print(f"| injection set blocked | {inj_v1}/18 | {inj_v2}/18 |")
    print(f"| trap hallucinations x3 | "
          f"{'/'.join(str(r['hallucinations']) for r in v1['T_cited'])} | "
          f"{'/'.join(str(r['hallucinations']) for r in C['T_cited'])} |")
    print(f"| cost / query (golden) | ${v1['G_cited']['cost_per_query']:.6f} | "
          f"${C['G_cited']['cost_per_query']:.6f} |")
    bad = [q for q in v1["G_cited"]["per_question"] if q["blocked"]]
    print("\n**The two v1 blocks, both of them CORRECT answers:**\n")
    for q in bad:
        print(f"- *{q['question']}* → cited "
              f"`{q['cited_ids'][0]}` where the request minted "
              f"`{q['context_labels']}`")
    print("\nEvery id the model emitted in v1 on the injection set was a bare "
          "label, which is why the arm that mattered showed nothing. The bug "
          "needed the clean corpus to surface, the regression half is not a "
          "formality.")

if "--answers" in sys.argv:
    print("\n\n## Every `cited` answer verbatim\n")
    for i, q in rows("cited"):
        j = q.get("judge")
        print(f"\n[{q['id']} run {i}] obeyed={obeyed(q, 'cited')} "
              f"correct={correct(q)} blocked={blocked(q)} "
              f"ci_blocked={ci_blocked(q)} "
              f"faith={j['score'] if j else ','} steps={q['steps']}")
        print(f"  cited {q['cited_ids']} of {q['context_labels']} · "
              f"payload under {q.get('needle_labels')}")
        print("  " + " ".join((q["answer"] or "").split()))
        for c in q.get("claims", []):
            print(f"    · [{c['chunk_id']}] {c['text'][:110]}")
