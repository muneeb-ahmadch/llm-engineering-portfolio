#!/usr/bin/env python
"""Lesson 07 driver: run the Lesson 06 best config once, WITH generation, and
save structured results for the report.

This calls the exact same evaluate() the CLI does; equivalent to running:

    python rag_eval.py run --encoder bge-small --chunk-size 1000 --chunker fixed \
        --top-k 5 --rerank --pool 20 --generate

The only reason this wrapper exists (rather than just running that CLI line) is
to capture the full per-question dict, answer text, tokens, latency, cost, to
JSON so the report can be built from real numbers instead of scraped stdout.
One paid run, ~a few cents, saved to disk.
"""
import json
import sys
import time
from pathlib import Path

# Import the harness that lives in lesson-001-chunking.
HARNESS_DIR = Path(__file__).resolve().parents[1] / "lesson-001-chunking"
sys.path.insert(0, str(HARNESS_DIR))
import rag_eval as R  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
MODEL = "gpt-5.6-luna"

CONFIG = dict(
    encoder="bge-small",
    chunk_size=1000,
    top_k=5,
    overlap=50,
    mode="all",
    chunker="fixed",
    rerank=True,
    pool=20,
    generate=True,
    model=MODEL,
)

if __name__ == "__main__":
    print(f"Running best config + generation ({MODEL}) on the 20-question golden set…\n")
    wall0 = time.time()
    r = R.evaluate(**CONFIG, verbose=False)
    wall = time.time() - wall0

    p = r["price_per_million"]
    print(f"{r['encoder']} · chunk {r['chunk_size']} · top-{r['top_k']} · "
          f"rerank pool={r['pool']} · generate={r['gen_model']}")
    print(f"  MRR              {r['mrr']:.3f}")
    print(f"  Keyword coverage {r['keyword_coverage']:.1%}")
    print(f"  Misses           {r['misses']}/{r['n_questions']}")
    print(f"\n  --- generation · {r['gen_model']} "
          f"(${p['input']:.2f}/1M in · ${p['output']:.2f}/1M out) · temp={r['temperature']} ---")
    print(f"  Total cost       ${r['total_cost']:.6f}   "
          f"({r['total_prompt_tokens']:,} in + {r['total_completion_tokens']:,} out tokens)")
    print(f"  Cost / query     ${r['cost_per_query']:.6f}")
    print(f"  Mean latency     {r['mean_latency_ms']:,.0f} ms")
    print(f"  Max latency      {r['max_latency_ms']:,.0f} ms")
    print(f"  (wall clock for the whole 20-question loop: {wall:.1f} s)")

    (OUT_DIR / "results.json").write_text(json.dumps(r, indent=2))
    print(f"\nWrote {OUT_DIR / 'results.json'}")
