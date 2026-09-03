#!/usr/bin/env python
"""Lesson 12 driver: one indirect injection, then a defence, both measured.

The L06/L07/L09/L11 config is held fixed throughout (bge-small · chunk 1000 ·
fixed · top-5 · rerank pool=20 · gpt-5.6-luna on /v1/responses · MAX_STEPS=3).
Exactly two things move, one at a time:

    the CORPUS   corpus/ (clean)  ->  corpus-poisoned/ (six injected chunks)
    the PROMPT   hardened (L09)   ->  segregated (L12, OWASP LLM01 mitigation #6)

Four arms on the injection set, and the first and last exist to stop the middle
two from lying:

    clean      clean corpus   · hardened   · agent   does the model say the
                                                     injected thing ANYWAY? If
                                                     yes, the attack proved
                                                     nothing.
    poisoned   poisoned       · hardened   · agent   THE ATTACK. Same code that
                                                     shipped at the end of L11.
    defended   poisoned       · segregated · agent   THE MITIGATION. Differs from
                                                     `poisoned` by the system
                                                     prompt and the delimiters
                                                     and nothing else, same
                                                     corpus, same index, same
                                                     retrieved chunks.
    fixed      poisoned       · hardened   · NO tool is this an AGENT problem or a
                                                     RAG problem? The fixed chain
                                                     eats the same poisoned chunk.

`clean` vs `poisoned` isolates the payload. `poisoned` vs `defended` isolates the
mitigation. `poisoned` vs `fixed` prices how much of the exposure the loop added.

Plus the regression halves, on the CLEAN corpus, because a defence that breaks
the system it defends is not a defence (the L09/L11 habit, report the regression
next to the win):

    G   golden 20, agent, judge, hardened vs segregated
    T   trap 5 x3,  agent,        hardened vs segregated   <- the L09 gate

Everything is checkpointed per cell (the L11 lesson: a run you can only read if
it finishes is a run you pay for twice). `--resume` skips what is already on disk.

    python build_poisoned_corpus.py     # first, free, and it gates this file
    python run_injection.py             # ~$1 of API spend
    python summarize.py                 # every table, free, no spend
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS_DIR = HERE.parents[0] / "lesson-001-chunking"
sys.path.insert(0, str(HARNESS_DIR))
import rag_eval as R  # noqa: E402

OUT = HERE / "results" / "results-injection.json"
OUT.parent.mkdir(exist_ok=True)
POISONED = HERE / "corpus-poisoned"
INJECTIONS = HERE / "injections.jsonl"

CONFIG = dict(encoder="bge-small", chunk_size=1000, top_k=5, overlap=50, mode="all",
              chunker="fixed", rerank=True, pool=20, generate=True,
              model="gpt-5.6-luna")
JUDGE = "gpt-5.6-terra"
REPEATS = 3

# arm -> (corpus, prompt, agent). Everything else is CONFIG.
ARMS = {
    "clean":    dict(corpus_dir=None,     prompt_variant="hardened",   agent=True),
    "poisoned": dict(corpus_dir=POISONED, prompt_variant="hardened",   agent=True),
    "defended": dict(corpus_dir=POISONED, prompt_variant="segregated", agent=True),
    "fixed":    dict(corpus_dir=POISONED, prompt_variant="hardened",   agent=False,
                     api="responses"),
}
REGRESSION_ARMS = {                       # clean corpus both sides: this is the cost side
    "hardened":   dict(prompt_variant="hardened",   agent=True),
    "segregated": dict(prompt_variant="segregated", agent=True),
}


def _traj(q):
    if "steps" not in q:
        return ""
    return f" · steps {q['steps']}/{R.MAX_STEPS} ({q['terminated_by']})"


def inj_runs(arm):
    """The six injections, REPEATS times. temp=1: one run is an anecdote."""
    out = []
    for i in range(REPEATS):
        r = R.evaluate(**CONFIG, **ARMS[arm], questions_path=INJECTIONS,
                       judge=True, judge_model=JUDGE)
        print(f"  [{arm}] run {i + 1}/{REPEATS}  ${r['total_cost_with_judge']:.4f}")
        for q in r["per_question"]:
            j = q["judge"]
            ans = " ".join((q["answer"] or "").split())
            print(f"    {q['id']} needle_in_ctx={int(bool(q.get('needle_in_context')))} "
                  f"faith={j['score']}/5 ({j['grounded_claims']}/{j['total_claims']})"
                  f"{_traj(q)}")
            print(f"         ↳ {ans[:150]}{'…' if len(ans) > 150 else ''}")
        out.append(r)
    return out


def golden_run(arm):
    r = R.evaluate(**CONFIG, **REGRESSION_ARMS[arm], judge=True, judge_model=JUDGE)
    print(f"  [{arm}] golden: faith {r['mean_faithfulness']:.2f}/5 · "
          f"grounded {r['grounded_claims']}/{r['total_claims']} · "
          f"abstention {r['abstention_rate']:.1%} · steps {r['mean_steps']:.2f} · "
          f"${r['cost_per_query']:.6f}/q · {r['mean_latency_ms']:.0f}ms · "
          f"cached {r['total_cached_tokens']:,} tok")
    for q in r["per_question"]:
        if q["abstained"]:
            print(f"      ABSTAINED: {q['question'][:70]}")
    return r


def trap_runs(arm):
    out = []
    for i in range(REPEATS):
        r = R.evaluate(**CONFIG, **REGRESSION_ARMS[arm], trap=True)
        print(f"  [{arm}] trap run {i + 1}/{REPEATS}: "
              f"{r['hallucinations']}/{r['n_questions']} hallucinations · "
              f"steps {r['mean_steps']:.2f} · ${r['total_cost']:.6f}")
        for q in r["hallucinated_questions"]:
            print(f"      HALLUCINATED: {q[:74]}")
        out.append(r)
    return out


def _spend(runs):
    return sum(r.get("total_cost_with_judge", r["total_cost"])
               for r in (runs if isinstance(runs, list) else [runs]))


def save(out, t0):
    out["wall_secs"] = round(time.time() - t0, 1)
    out["total_spend"] = round(sum(_spend(v) for k, v in out.items()
                                   if k[:2] in ("I_", "G_", "T_")), 6)
    OUT.write_text(json.dumps(out, indent=2, default=str))


CELLS = {}
for _a in ARMS:
    CELLS[f"I_{_a}"] = (lambda a=_a: inj_runs(a))
for _a in REGRESSION_ARMS:
    CELLS[f"G_{_a}"] = (lambda a=_a: golden_run(a))
for _a in REGRESSION_ARMS:
    CELLS[f"T_{_a}"] = (lambda a=_a: trap_runs(a))


if __name__ == "__main__":
    if not POISONED.exists():
        sys.exit("No corpus-poisoned/, run build_poisoned_corpus.py first. "
                 "It also proves every payload is retrievable, which this file assumes.")
    t0 = time.time()
    resume = "--resume" in sys.argv and OUT.exists()
    out = json.loads(OUT.read_text()) if resume else {}
    out.update({"config": CONFIG, "arms": {k: {kk: str(vv) for kk, vv in v.items()}
                                           for k, v in ARMS.items()},
                "regression_arms": REGRESSION_ARMS, "judge_model": JUDGE,
                "repeats": REPEATS, "max_steps": R.MAX_STEPS,
                "request_timeout_s": R.REQUEST_TIMEOUT_S,
                "segregation_rules": R.SEGREGATION_RULES,
                "prices_verified": "see PRICES in rag_eval.py"})

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
            print({"I": "\nI · the six injections, x3 per arm (+judge), the attack "
                        "and the defence",
                   "G": "\nG · golden 20 on the CLEAN corpus (+judge), what the "
                        "defence COSTS",
                   "T": "\nT · trap 5 x3 on the CLEAN corpus, does the L09 gate "
                        "still hold?"}[scope])
        out[cell] = run()
        save(out, t0)

    save(out, t0)
    print(f"\nWrote {OUT}  ({out['wall_secs']}s, ${out['total_spend']:.4f} of API spend)")
