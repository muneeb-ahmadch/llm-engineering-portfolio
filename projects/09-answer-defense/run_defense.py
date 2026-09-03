#!/usr/bin/env python
"""Lesson 09 driver: run the defense layer and save every number to JSON.

Four runs, one experiment. The L06/L07 config is held fixed throughout
(bge-small · chunk 1000 · fixed · top-5 · rerank pool=20 · gpt-5.6-luna); the
ONLY thing that moves is the system prompt:

    A  trap set  x3   baseline prompt   -> hallucination count, and its flicker
    B  golden    +judge  baseline       -> mean faithfulness, abstention rate
    C  trap set  x3   hardened prompt   -> did prompt-hardening move the count?
    D  golden    +judge  hardened       -> ...at what cost in FALSE abstention?

C and D are the pair. L08 Scenario 4 said the abstention rate has to climb
"without raising the false-abstention rate on answerable questions", one number
alone is gameable (a model that abstains on everything scores a perfect trap set
and is useless), so the trap set and the golden set have to be read together.

Equivalent CLI, if you'd rather watch it live:

    python rag_eval.py run --encoder bge-small --chunk-size 1000 --chunker fixed \
        --top-k 5 --rerank --pool 20 --generate --trap --repeat 3 [--prompt hardened]
    python rag_eval.py run ... --generate --judge [--prompt hardened]
"""
import json
import sys
import time
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parents[1] / "lesson-001-chunking"
sys.path.insert(0, str(HARNESS_DIR))
import rag_eval as R  # noqa: E402

OUT = Path(__file__).resolve().parent / "results-defense.json"

CONFIG = dict(encoder="bge-small", chunk_size=1000, top_k=5, overlap=50, mode="all",
              chunker="fixed", rerank=True, pool=20, generate=True,
              model="gpt-5.6-luna")
JUDGE = "gpt-5.6-terra"
REPEATS = 3


def trap_runs(prompt_variant):
    out = []
    for i in range(REPEATS):
        r = R.evaluate(**CONFIG, trap=True, prompt_variant=prompt_variant)
        print(f"    [{prompt_variant}] trap run {i + 1}/{REPEATS}: "
              f"{r['hallucinations']}/{r['n_questions']} hallucinations  "
              f"(${r['total_cost']:.6f})")
        out.append(r)
    return out


def golden_run(prompt_variant):
    r = R.evaluate(**CONFIG, judge=True, judge_model=JUDGE, prompt_variant=prompt_variant)
    print(f"    [{prompt_variant}] golden: MRR {r['mrr']:.3f} · "
          f"faithfulness {r['mean_faithfulness']}/5 over {r['n_judged_scored']} scored · "
          f"abstention {r['abstention_rate']:.1%} · ${r['total_cost_with_judge']:.6f}")
    return r


if __name__ == "__main__":
    t0 = time.time()
    out = {"config": CONFIG, "judge_model": JUDGE, "repeats": REPEATS,
           "prices_verified": "2026-07-25 via developers.openai.com/api/docs/pricing"}

    print("A · trap set x3, baseline prompt")
    out["A_trap_baseline"] = trap_runs("baseline")
    print("B · golden set + judge, baseline prompt")
    out["B_golden_baseline"] = golden_run("baseline")
    print("C · trap set x3, hardened prompt")
    out["C_trap_hardened"] = trap_runs("hardened")
    print("D · golden set + judge, hardened prompt")
    out["D_golden_hardened"] = golden_run("hardened")

    out["wall_secs"] = round(time.time() - t0, 1)
    out["total_spend"] = round(
        sum(r["total_cost"] for r in out["A_trap_baseline"] + out["C_trap_hardened"])
        + out["B_golden_baseline"]["total_cost_with_judge"]
        + out["D_golden_hardened"]["total_cost_with_judge"], 6)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT}  ({out['wall_secs']}s, ${out['total_spend']:.4f} of API spend)")
