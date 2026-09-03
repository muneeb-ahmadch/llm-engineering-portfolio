#!/usr/bin/env python
"""Lesson 11 driver: the agent loop, measured against a matched control.

The L06/L07/L09 config is held fixed throughout (bge-small · chunk 1000 · fixed ·
top-5 · rerank pool=20 · gpt-5.6-luna · hardened prompt). The only thing that
moves is whether the model gets a search_corpus tool.

THREE arms, not two, and the third one is the whole reason this file exists:

    shipped   single-shot on /v1/chat/completions, what L09/L10 measured and
                                                      what is in the CI gate today
    control   single-shot on /v1/responses, same endpoint as the agent,
                                                      no tool
    agent     the MAX_STEPS=3 loop on /v1/responses, same endpoint, plus the tool

gpt-5.6-luna refuses function tools on /v1/chat/completions at any reasoning
effort above 'none', so the agent HAS to run on /v1/responses. Comparing it
straight against `shipped` would confound "the loop helped/hurt" with "the
endpoint changed". `control` is the fix: control vs agent isolates the tool,
shipped vs control prices the endpoint switch on its own.

Three scopes, because they answer different questions:

    Q  the overfitting question alone, x3 per arm, read the trajectory
    G  the golden set (20), once per arm + judge, the deltas
    T  the trap set (5), x3 per arm, the 0/5 that is at risk

Hypotheses were written down BEFORE any of this ran, see hypothesis.md.

Equivalent CLI for any single cell, if you'd rather watch it live:

    python rag_eval.py run --encoder bge-small --chunk-size 1000 --top-k 5 \
        --rerank --pool 20 --generate --prompt hardened \
        [--agent | --api responses] [--only overfitting] [--trap] [--judge] -v
"""
import json
import sys
import time
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parents[1] / "lesson-001-chunking"
sys.path.insert(0, str(HARNESS_DIR))
import rag_eval as R  # noqa: E402

OUT = Path(__file__).resolve().parent / "results" / "results-agent.json"
OUT.parent.mkdir(exist_ok=True)

CONFIG = dict(encoder="bge-small", chunk_size=1000, top_k=5, overlap=50, mode="all",
              chunker="fixed", rerank=True, pool=20, generate=True,
              model="gpt-5.6-luna", prompt_variant="hardened")
JUDGE = "gpt-5.6-terra"
REPEATS = 3

# arm name -> the two kwargs that define it. Everything else is CONFIG.
ARMS = {
    "shipped": dict(agent=False, api="chat"),
    "control": dict(agent=False, api="responses"),
    "agent":   dict(agent=True,  api="responses"),
}


def _traj(r):
    if not r.get("agent"):
        return ""
    return (f" · steps {r['mean_steps']:.2f} (max {r['max_steps_observed']}, "
            f"cap {r['pct_terminated_cap']:.0%})")


def q_runs(arm):
    """The overfitting question alone, REPEATS times. temp=1 means one run is an
    anecdote; three is the smallest thing that can show a coin flip."""
    out = []
    for i in range(REPEATS):
        r = R.evaluate(**CONFIG, **ARMS[arm], only="overfitting", judge=True,
                       judge_model=JUDGE)
        q = r["per_question"][0]
        print(f"    [{arm}] Q run {i + 1}/{REPEATS}: "
              f"abstained={int(q['abstained'])} "
              f"faith={q['judge']['score']}/5 "
              f"({q['judge']['grounded_claims']}/{q['judge']['total_claims']} claims) "
              f"{q['latency_ms']:.0f}ms ${r['total_cost']:.6f}{_traj(r)}")
        for s in q.get("searches", []):
            print(f"        step {s['step']} query: {s['query']}")
        out.append(r)
    return out


def golden_run(arm, trace=False):
    r = R.evaluate(**CONFIG, **ARMS[arm], judge=True, judge_model=JUDGE, trace=trace)
    print(f"    [{arm}] golden: faith {r['mean_faithfulness']:.2f}/5 · "
          f"abstention {r['abstention_rate']:.1%} · "
          f"${r['cost_per_query']:.6f}/q · {r['mean_latency_ms']:.0f}ms{_traj(r)}")
    if r.get("trace_url"):
        print(f"        trace: {r['trace_url']}")
    return r


def trap_runs(arm, trace=False):
    out = []
    for i in range(REPEATS):
        r = R.evaluate(**CONFIG, **ARMS[arm], trap=True,
                       trace=trace and i == 0)  # trace one of the three
        print(f"    [{arm}] trap run {i + 1}/{REPEATS}: "
              f"{r['hallucinations']}/{r['n_questions']} hallucinations "
              f"${r['total_cost']:.6f}{_traj(r)}")
        for q in r["hallucinated_questions"]:
            print(f"        HALLUCINATED: {q[:78]}")
        if r.get("trace_url"):
            print(f"        trace: {r['trace_url']}")
        out.append(r)
    return out


def _spend(runs):
    total = 0.0
    for r in runs if isinstance(runs, list) else [runs]:
        total += r.get("total_cost_with_judge", r["total_cost"])
    return total


def save(out, t0):
    """Checkpoint after every cell.

    The first attempt at this experiment wrote results ONCE, at the end, and a
    hung API call 12 questions into the agent golden pass threw away every paid
    call before it. A run that can only be read if it finishes is a run you pay
    for twice. Now each cell lands on disk as it completes, and CELLS below lets
    a re-run skip whatever already survived.
    """
    out["wall_secs"] = round(time.time() - t0, 1)
    out["total_spend"] = round(sum(_spend(v) for k, v in out.items()
                                   if k[:2] in ("Q_", "G_", "T_")), 6)
    OUT.write_text(json.dumps(out, indent=2, default=str))


# cell name -> how to run it. Ordered cheapest-first within each scope so a
# failure late in the run still leaves the cheap evidence on disk.
CELLS = {}
for _arm in ARMS:
    CELLS[f"Q_{_arm}"] = (lambda a=_arm: q_runs(a))
for _arm in ARMS:
    CELLS[f"G_{_arm}"] = (lambda a=_arm: golden_run(a, trace=(a == "agent")))
for _arm in ARMS:
    CELLS[f"T_{_arm}"] = (lambda a=_arm: trap_runs(a, trace=(a == "agent")))


if __name__ == "__main__":
    t0 = time.time()
    # --resume keeps whatever is already in results-agent.json and runs only the
    # missing cells. Re-running a cell that already succeeded costs real money
    # and buys nothing but a different sample of a temperature-1 distribution.
    resume = "--resume" in sys.argv and OUT.exists()
    out = json.loads(OUT.read_text()) if resume else {}
    out.update({"config": CONFIG, "arms": ARMS, "judge_model": JUDGE,
                "repeats": REPEATS, "max_steps": R.MAX_STEPS,
                "request_timeout_s": R.REQUEST_TIMEOUT_S,
                "prices_verified": "see PRICES in rag_eval.py, re-verified on run day"})

    only = [a for a in sys.argv[1:] if a in CELLS]
    scope = None
    for cell, run in CELLS.items():
        if only and cell not in only:
            continue
        if resume and cell in out:
            print(f"  [skip] {cell}, already on disk")
            continue
        if cell[0] != scope:
            scope = cell[0]
            print({"Q": "\nQ · the overfitting question alone, x3 per arm (+judge)",
                   "G": "\nG · the golden set (20 questions), once per arm (+judge)",
                   "T": "\nT · the trap set (5 questions), x3 per arm, the hypothesis test",
                   }[scope])
        out[cell] = run()
        save(out, t0)   # <- the whole point

    save(out, t0)
    print(f"\nWrote {OUT}  ({out['wall_secs']}s, ${out['total_spend']:.4f} of API spend)")
