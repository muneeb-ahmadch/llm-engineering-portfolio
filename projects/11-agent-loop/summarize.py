#!/usr/bin/env python
"""Turn results-agent.json into the tables the report is built from.

Reads only, no API calls, no spend. Every number printed here traces to a run in
results-agent.json, which is the file the driver wrote.
"""
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent
D = json.loads((HERE / "results" / "results-agent.json").read_text())
ARMS = ["shipped", "control", "agent"]


def rule(title):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------- Q
rule("Q, the overfitting question alone, 3 runs per arm")
print(f"{'arm':<9} {'abstained':<11} {'faithfulness':<26} {'latency ms':<14} {'cost/q':<11} steps")
for arm in ARMS:
    runs = D[f"Q_{arm}"]
    qs = [r["per_question"][0] for r in runs]
    ab = [int(q["abstained"]) for q in qs]
    # zero-claim answers are abstentions: the judge's 5 there is "asserted nothing",
    # not "grounded". Reporting them as 5/5 would flatter whichever arm abstains more.
    faith = [f"{q['judge']['score']}" if q["judge"]["total_claims"] else ","
             for q in qs]
    claims = [f"{q['judge']['grounded_claims']}/{q['judge']['total_claims']}" for q in qs]
    lat = [q["latency_ms"] for q in qs]
    cost = [r["cost_per_query"] for r in runs]
    steps = ([q["steps"] for q in qs] if runs[0].get("agent") else [1, 1, 1])
    print(f"{arm:<9} {str(ab):<11} {'/'.join(faith) + '  (' + ' '.join(claims) + ')':<26} "
          f"{st.mean(lat):>9.0f}     ${st.mean(cost):<10.6f} {steps}")

print("\nthe queries the loop wrote:")
for i, r in enumerate(D["Q_agent"], 1):
    q = r["per_question"][0]
    print(f"  run {i}  steps={q['steps']} terminated_by={q['terminated_by']} "
          f"abstained={int(q['abstained'])} faith={q['judge']['score']}"
          f" ({q['judge']['grounded_claims']}/{q['judge']['total_claims']})")
    for s in q["searches"]:
        print(f"      step {s['step']} k={s['k']} new={s['n_new_chunks']}/{len(s['sources'])}"
              f"  {sorted(set(s['sources']))}")
        print(f"        {s['query']}")
    print(f"      answer: {' '.join((q['answer'] or '').split())[:200]}")
    for c in q["judge"]["claims"]:
        print(f"        [{'G' if c['grounded'] else 'U'}] {c['claim'][:96]}")


# ---------------------------------------------------------------- G
rule("G, the golden set (20 questions), one run per arm")
hdr = ("arm", "abstain", "faith", "grounded", "cost/q", "mean ms", "max ms",
       "in tok", "out tok")
print(f"{hdr[0]:<9}{hdr[1]:>9}{hdr[2]:>8}{hdr[3]:>12}{hdr[4]:>11}{hdr[5]:>9}"
      f"{hdr[6]:>9}{hdr[7]:>9}{hdr[8]:>9}")
base = None
for arm in ARMS:
    r = D[f"G_{arm}"]
    row = (f"{arm:<9}{r['abstention_rate']:>8.1%}{r['mean_faithfulness']:>8.2f}"
           f"{r['grounded_claims']:>6}/{r['total_claims']:<5}"
           f"${r['cost_per_query']:>9.6f}{r['mean_latency_ms']:>9.0f}"
           f"{r['max_latency_ms']:>9.0f}{r['total_prompt_tokens']:>9,}"
           f"{r['total_completion_tokens']:>9,}")
    print(row)
    if arm == "control":
        base = r
if base:
    a = D["G_agent"]
    print("\ndeltas, agent vs control (same endpoint, the tool is the only difference):")
    print(f"  abstention rate   {base['abstention_rate']:.1%} -> {a['abstention_rate']:.1%}"
          f"   ({a['abstention_rate'] - base['abstention_rate']:+.1%})")
    print(f"  mean faithfulness {base['mean_faithfulness']:.2f} -> {a['mean_faithfulness']:.2f}"
          f"   ({a['mean_faithfulness'] - base['mean_faithfulness']:+.2f})")
    print(f"  cost / query      ${base['cost_per_query']:.6f} -> ${a['cost_per_query']:.6f}"
          f"   ({a['cost_per_query'] / base['cost_per_query']:.2f}x)")
    print(f"  mean latency      {base['mean_latency_ms']:.0f} -> {a['mean_latency_ms']:.0f} ms"
          f"   ({a['mean_latency_ms'] / base['mean_latency_ms']:.2f}x)")
    s = D["G_shipped"]
    print("\nendpoint-only effect, control vs shipped (no tool either side):")
    print(f"  abstention {s['abstention_rate']:.1%} -> {base['abstention_rate']:.1%} · "
          f"faith {s['mean_faithfulness']:.2f} -> {base['mean_faithfulness']:.2f} · "
          f"cost ${s['cost_per_query']:.6f} -> ${base['cost_per_query']:.6f} · "
          f"latency {s['mean_latency_ms']:.0f} -> {base['mean_latency_ms']:.0f} ms")

g = D["G_agent"]
rule("G, trajectory (the agent arm)")
print(f"  mean steps          {g['mean_steps']:.2f} / cap {g['max_steps_cap']}")
print(f"  max steps observed  {g['max_steps_observed']}")
print(f"  terminated_by=cap   {g['n_terminated_cap']}/{g['n_questions']} "
      f"({g['pct_terminated_cap']:.1%})")
print(f"  searched at all     {g['n_searched']}/{g['n_questions']} questions, "
      f"{g['total_searches']} searches total")
print(f"  reasoning tokens    {g.get('total_reasoning_tokens', 0):,} of "
      f"{g['total_completion_tokens']:,} output tokens")
print("\n  per question:")
for q in sorted(g["per_question"], key=lambda q: -q["steps"]):
    print(f"    steps={q['steps']} {q['terminated_by']:<7} "
          f"abst={int(q['abstained'])} f={q['judge']['score']} "
          f"{q['question'][:56]}")
    for s in q["searches"]:
        print(f"        -> {s['query'][:104]}")


# ---------------------------------------------------------------- T
rule("T, the trap set (5 questions), 3 runs per arm")
for arm in ARMS:
    runs = D[f"T_{arm}"]
    h = [r["hallucinations"] for r in runs]
    cost = sum(r["total_cost"] for r in runs)
    extra = ""
    if runs[0].get("agent"):
        allq = [q for r in runs for q in r["per_question"]]
        steps = [q["steps"] for q in allq]
        cap = sum(1 for q in allq if q["terminated_by"] == "cap")
        extra = (f"  | mean steps {st.mean(steps):.2f}, max {max(steps)}, "
                 f"cap {cap}/{len(steps)} ({cap / len(steps):.0%})")
    print(f"  {arm:<9} hallucinations {h}  of 5 each   ${cost:.6f} total{extra}")
    for r in runs:
        for q in r["hallucinated_questions"]:
            print(f"      ANSWERED: {q[:74]}")

print("\n  every trap query the loop wrote:")
seen = set()
for r in D["T_agent"]:
    for q in r["per_question"]:
        for s in q["searches"]:
            if s["query"] not in seen:
                seen.add(s["query"])
                print(f"    [{'ABST' if q['abstained'] else 'ANSW'}] {s['query'][:110]}")

rule("ledger")
print(f"  wall clock   {D['wall_secs']}s")
print(f"  total spend  ${D['total_spend']:.4f}")
for k in D:
    if k[:2] in ("Q_", "G_", "T_"):
        v = D[k]
        c = (sum(r.get("total_cost_with_judge", r["total_cost"]) for r in v)
             if isinstance(v, list) else v.get("total_cost_with_judge", v["total_cost"]))
        print(f"    {k:<12} ${c:.6f}")
for k in ("G_agent", "T_agent"):
    if D[k] if not isinstance(D[k], list) else D[k][0]:
        r = D[k] if not isinstance(D[k], list) else D[k][0]
        if r.get("trace_url"):
            print(f"  trace {k}: {r['trace_url']}")
