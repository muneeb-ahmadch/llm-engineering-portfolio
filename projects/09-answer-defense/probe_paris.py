#!/usr/bin/env python
"""The Lesson 07 Paris probe, re-run now that the guardrail exists.

L07 fed an LSTM chunk as context, asked for a capital city, and got "Paris, the
capital of France" instead of the instructed abstention. That was ONE anecdote, which is exactly why L08 Scenario 4 refused to fine-tune on it. This re-runs the
same probe against both prompts and hands each answer to the faithfulness judge,
so the anecdote becomes three readings:

    does the model abstain?              (the string gate)
    does the judge catch it if it doesn't? (the judge, is it worth its cost?)
    does the judge cry wolf on a good answer? (the control, a judge that fails
                                               everything is not a judge)

The third run is the control and matters as much as the other two: a guardrail
you have only ever seen fire is indistinguishable from a guardrail that is stuck on.
"""
import sys
from pathlib import Path

import numpy as np

HARNESS_DIR = Path(__file__).resolve().parents[1] / "lesson-001-chunking"
sys.path.insert(0, str(HARNESS_DIR))
import rag_eval as R  # noqa: E402

OFF_TOPIC = "What is the capital of France?"
ON_TOPIC = "How does an LSTM solve the vanishing gradient problem?"


def retrieve(question, chunks, matrix, encoder_model, ce, top_k=5, pool=20):
    qv = R.embed_queries(encoder_model, [question], "bge-small")
    row = (qv @ matrix.T)[0]
    cand = [chunks[i] for i in np.argsort(-row)[:pool]]
    scores = ce.predict([(question, c["text"]) for c in cand])
    return [cand[i] for i in np.argsort(-np.asarray(scores))[:top_k]]


if __name__ == "__main__":
    docs = R.load_documents()
    chunks = R.build_chunks(docs, 1000, 50, "fixed")
    em = R.get_model("bge-small")
    matrix, _ = R.embed_chunks(em, chunks, "bge-small", 1000, 50, docs, "fixed")
    ce = R.get_cross_encoder()
    client = R.get_openai_client()
    temp_state = {"supports_temp0": True}

    # Context is retrieved for the LSTM question, exactly as in L07, then a
    # question that context cannot possibly answer is asked against it.
    lstm_ctx = retrieve(ON_TOPIC, chunks, matrix, em, ce)
    cases = [
        ("Paris probe · baseline prompt", OFF_TOPIC, lstm_ctx, "baseline"),
        ("Paris probe · hardened prompt", OFF_TOPIC, lstm_ctx, "hardened"),
        ("CONTROL: on-topic · hardened", ON_TOPIC,
         retrieve(ON_TOPIC, chunks, matrix, em, ce), "hardened"),
    ]

    print(f"\ncontext = top-5 for {ON_TOPIC!r}")
    print(f"sources: {[c['source'] for c in lstm_ctx]}\n")
    for label, q, ctx, variant in cases:
        gen = R.generate_answer(client, "gpt-5.6-luna", q, ctx, temp_state,
                                prompt_variant=variant)
        v = R.judge_answer(client, R.DEFAULT_JUDGE_MODEL, gen["answer"], ctx)
        print(f"── {label}")
        print(f"   question     {q}")
        print(f"   answer       {' '.join(gen['answer'].split())[:150]}")
        print(f"   abstained    {R.is_abstention(gen['answer'])}")
        print(f"   faithfulness {v['score']}/5  "
              f"({v['grounded_claims']}/{v['total_claims']} claims grounded)")
        for c in v["claims"]:
            print(f"      {'✓' if c['grounded'] else '✗'} {c['claim'][:78]}")
            if not c["grounded"]:
                print(f"        why: {c['why'][:100]}")
        print()
