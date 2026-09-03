#!/usr/bin/env python
"""Lesson 06, step 1, full-corpus rank for the two known misses, BEFORE any
reranking runs and BEFORE a pool size is chosen.

Same technique Lesson 05's compare_chunkers.py used to surface day26's rank
#8: embed once, take the full cosine-ranked order for a question (not just
top-k), and walk it until the first chunk that satisfies is_relevant(). That
walk is exactly what --pool would have to reach for --rerank to have any
chance of fixing the miss, a cross-encoder can only re-sort candidates that
cosine already put in the pool.

Reuses the Lesson-01 harness as a library (no re-embedding: this config's
matrix is already cached at results/.cache/), so this is nearly free, the
whole point is to know the number before spending anything on a cross-encoder
run.

Run with the Lesson-01 venv:
    ../lesson-001-chunking/.venv/bin/python find_full_rank.py
"""

import sys
from pathlib import Path

import numpy as np

LESSON1 = Path(__file__).parent.parent / "lesson-001-chunking"
if not (LESSON1 / "rag_eval.py").exists():
    sys.exit(f"Can't find the Lesson-01 harness at {LESSON1}")
sys.path.insert(0, str(LESSON1))
import rag_eval as R  # noqa: E402

ENCODER = "bge-small"
CHUNK_SIZE = 1000
OVERLAP = 50
CHUNKER = "fixed"
RELEVANCE = "all"
TARGETS = ["lstm", "overfitting"]  # substrings of the question text


def main():
    docs = R.load_documents()
    tests = R.load_golden()
    chunks = R.build_chunks(docs, CHUNK_SIZE, OVERLAP, CHUNKER)
    model = R.get_model(ENCODER)
    matrix, cached = R.embed_chunks(model, chunks, ENCODER, CHUNK_SIZE, OVERLAP, docs, CHUNKER)
    query_vecs = R.embed_queries(model, [t["question"] for t in tests], ENCODER)
    scores = query_vecs @ matrix.T

    print(f"config: {ENCODER} · chunk {CHUNK_SIZE} · chunker={CHUNKER} · "
          f"{len(chunks)} chunks · embeddings {'(cached)' if cached else '(just computed)'}\n")

    for target in TARGETS:
        ti = next(i for i, t in enumerate(tests) if target in t["question"].lower())
        test = tests[ti]
        row = scores[ti]
        order = np.argsort(-row)

        full_rank = None
        for rank, idx in enumerate(order, 1):
            if R.is_relevant(chunks[idx]["text"].lower(), test["keywords"], RELEVANCE):
                full_rank = rank
                break

        top5_idx = order[:5]
        top5 = [chunks[i] for i in top5_idx]
        rr = R.reciprocal_rank(test, top5, RELEVANCE)
        verdict = f"HIT @{round(1/rr)}" if rr else "MISS (outside top-5)"

        print(f"Q: {test['question']}")
        print(f"   keywords: {test['keywords']}")
        print(f"   top-5 verdict: {verdict}")
        print(f"   full-corpus rank of first relevant chunk: #{full_rank} of {len(chunks)}")
        print()


if __name__ == "__main__":
    main()
