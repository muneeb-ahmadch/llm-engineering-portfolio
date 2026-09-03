#!/usr/bin/env python
"""L13 steps 5 and 6: the deterministic stage, measured against the one I own.

Everything from L06/L07/L09/L11/L12 is held fixed (bge-small · chunk 1000 · fixed
· top-5 · rerank pool=20 · gpt-5.6-luna on /v1/responses · MAX_STEPS=3 ·
gpt-5.6-terra judging). Exactly one thing moves:

    the PROMPT + RETURN TYPE   segregated (L12)  ->  cited (L13)

`cited` is `segregated` plus three rules and a schema, a strict superset, so the
L12-vs-L13 column is one change and not two. The corpus, the index, the retrieved
chunks and the judge are identical.

    I_cited   6 injections x 3, POISONED corpus. Compared against L12's `poisoned`
              and `defended` arms, which are read out of L12's results file rather
              than re-run: they are the same code, the same corpus and the same
              config, and paying for them twice would buy nothing but noise.
    G_cited   golden 20, CLEAN corpus, judged. The regression half. A forced format
              is a constraint on the model and it is not obviously free.
    T_cited   trap 5 x3, CLEAN corpus. The every-commit gate since L09.
    C_cited   INJ-07 x3, the ceiling probe. The injection that issues no order and
              simply lies. Predicted in predictions.md before this ran.

The judge is SEGREGATED throughout (step 1, `judge_segregate=True` is now the
default). That is not a detail: today's numbers are read off an instrument that was
calibrated this morning, and L12's `poisoned`/`defended` judge columns are re-read
from `../lesson-013-orchestration/results/results-rejudge.json` for the same reason.

    python test_validator.py    # first, free, and it gates this file
    python run_cited.py         # ~$0.55
    python summarize_cited.py   # every table, free
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
L12 = HERE.parents[0] / "lesson-012-injection"
sys.path.insert(0, str(HERE.parents[0] / "lesson-001-chunking"))
import rag_eval as R  # noqa: E402

OUT = HERE / "results" / "results-cited.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

CONFIG = dict(encoder="bge-small", chunk_size=1000, top_k=5, overlap=50, mode="all",
              chunker="fixed", rerank=True, pool=20, generate=True,
              model="gpt-5.6-luna", agent=True)
JUDGE = "gpt-5.6-terra"
REPEATS = 3
PROMPT = "cited"


def _line(q):
    tag = "BLOCKED" if q.get("blocked") else "passed "
    j = q.get("judge")
    faith = f"faith={j['score']}/5 ({j['grounded_claims']}/{j['total_claims']})" \
        if j else "faith=, (blocked, not scored)"
    print(f"    {q.get('id', q['question'][:28])} "
          f"needle_in_ctx={int(bool(q.get('needle_in_context')))} "
          f"{tag} cited={q.get('cited_ids')} {faith} "
          f"steps {q.get('steps')}/{R.MAX_STEPS} ({q.get('terminated_by')})")
    if q.get("needle_labels"):
        print(f"         payload was under {q['needle_labels']}  "
              f"(CI knows this; production does not)")
    for r in q.get("block_reasons", []):
        print(f"         x {r}")
    print(f"         ↳ {' '.join((q['answer'] or '').split())[:150]}")


def injections():
    out = []
    for i in range(REPEATS):
        r = R.evaluate(**CONFIG, prompt_variant=PROMPT, corpus_dir=L12 / "corpus-poisoned",
                       questions_path=L12 / "injections.jsonl",
                       judge=True, judge_model=JUDGE)
        print(f"  [cited] run {i + 1}/{REPEATS}  ${r['total_cost_with_judge']:.4f}  "
              f"blocked {r['n_blocked']}/{r['n_questions']}")
        for q in r["per_question"]:
            _line(q)
        out.append(r)
    return out


def golden():
    r = R.evaluate(**CONFIG, prompt_variant=PROMPT, judge=True, judge_model=JUDGE)
    print(f"  [cited] golden: faith {r['mean_faithfulness']:.2f}/5 · "
          f"grounded {r['grounded_claims']}/{r['total_claims']} · "
          f"blocked {r['n_blocked']}/20 · abstention {r['abstention_rate']:.1%} · "
          f"steps {r['mean_steps']:.2f} · ${r['cost_per_query']:.6f}/q · "
          f"{r['mean_latency_ms']:.0f}ms · cached {r['total_cached_tokens']:,} tok")
    for q in r["per_question"]:
        if q["abstained"] or q["blocked"]:
            print(f"      {'BLOCKED' if q['blocked'] else 'ABSTAINED'}: "
                  f"{q['question'][:66]}")
            for x in q["block_reasons"]:
                print(f"         x {x}")
    return r


def traps():
    out = []
    for i in range(REPEATS):
        r = R.evaluate(**CONFIG, prompt_variant=PROMPT, trap=True)
        print(f"  [cited] trap run {i + 1}/{REPEATS}: "
              f"{r['hallucinations']}/{r['n_questions']} hallucinations · "
              f"blocked {r['n_blocked']}/5 · steps {r['mean_steps']:.2f} · "
              f"${r['total_cost']:.6f}")
        for q in r["hallucinated_questions"]:
            print(f"      HALLUCINATED: {q[:70]}")
        out.append(r)
    return out


def ceiling():
    """Step 6: INJ-07, the one that just lies. Prediction is in predictions.md."""
    out = []
    for i in range(REPEATS):
        r = R.evaluate(**CONFIG, prompt_variant=PROMPT,
                       corpus_dir=L12 / "corpus-poisoned-ceiling",
                       questions_path=L12 / "injections-ceiling.jsonl",
                       judge=True, judge_model=JUDGE)
        print(f"  [cited] ceiling run {i + 1}/{REPEATS}")
        for q in r["per_question"]:
            _line(q)
        out.append(r)
    return out


CELLS = {"I_cited": injections, "G_cited": golden, "T_cited": traps,
         "C_cited": ceiling}
HEAD = {"I_cited": "\nI · the six injections x3, POISONED corpus, `cited`",
        "G_cited": "\nG · golden 20 on the CLEAN corpus, what the SCHEMA costs",
        "T_cited": "\nT · trap 5 x3: does the L09 gate still hold under a schema?",
        "C_cited": "\nC · INJ-07 x3: the ceiling. Predicted before it ran"}

if __name__ == "__main__":
    t0 = time.time()
    resume = "--resume" in sys.argv and OUT.exists()
    out = json.loads(OUT.read_text()) if resume else {}
    out.update({"config": CONFIG, "prompt_variant": PROMPT, "judge_model": JUDGE,
                "judge_segregated": True, "repeats": REPEATS,
                "max_steps": R.MAX_STEPS, "citation_rules": R.CITATION_RULES,
                "schema": R.CITED_ANSWER_SCHEMA,
                "compare_against": "../lesson-012-injection/results/results-injection.json"
                                   " + results/results-rejudge.json"})

    only = [a for a in sys.argv[1:] if a in CELLS]
    for cell, run in CELLS.items():
        if (only and cell not in only) or (resume and cell in out):
            print(f"  [skip] {cell}")
            continue
        print(HEAD[cell])
        out[cell] = run()
        runs = [r for k, v in out.items() if k in CELLS
                for r in (v if isinstance(v, list) else [v])]
        out["total_spend"] = round(sum(r.get("total_cost_with_judge", r["total_cost"])
                                       for r in runs), 6)
        out["wall_secs"] = round(time.time() - t0, 1)
        OUT.write_text(json.dumps(out, indent=2, default=str))

    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {OUT}  ({out['wall_secs']}s, ${out['total_spend']:.4f})")
