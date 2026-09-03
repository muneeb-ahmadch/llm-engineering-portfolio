#!/usr/bin/env python
"""Step 2: run the ported graph on three of L11's golden questions.

The three are chosen so both edges out of `agent` get exercised, and so the
answers are comparable to a run I already have numbers for:

    overfitting   L11: 2 steps, one search, agent -> tools -> agent -> END
    reranking     L11: 2 steps, one search, the only other question of the 20 that searched
    prompt caching L11: 1 step, no search, agent -> END, the edge that ends the run

    python run_graph.py [--fresh]      # --fresh wipes checkpoints + counters first
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph as G                                                     # noqa: E402

OUT = G.HERE / "results-step2.json"
QUESTIONS = [
    ("l15-overfitting", "What is overfitting in a neural network?"),
    ("l15-reranking", "What does reranking do in a RAG pipeline?"),
    ("l15-prompt-caching", "What is prompt caching?"),
]

if __name__ == "__main__":
    if "--fresh" in sys.argv:
        for p in (G.DB_PATH, G.API_CALLS_LOG, G.TOOL_RUNS_LOG):
            p.unlink(missing_ok=True)
        print("  wiped checkpoints + counters")

    G.build_index()
    saver = G.sqlite_saver()
    app = G.make_graph(checkpointer=saver)

    import importlib.metadata as md
    out = {"langgraph": md.version("langgraph"),
           "langgraph_checkpoint_sqlite": md.version("langgraph-checkpoint-sqlite"),
           "model": G.MODEL, "max_steps": G.MAX_STEPS, "config": G.CONFIG,
           "prompt_variant": G.PROMPT_VARIANT, "questions": []}

    for thread_id, question in QUESTIONS:
        config = {"configurable": {"thread_id": thread_id}}
        print(f"\n  [{thread_id}] {question}")
        final = app.invoke(G.initial_state(question), config)
        t = G.totals(final["usage"])
        n_ckpt = len(list(app.get_state_history(config)))
        print(f"    steps {final['steps']} · {final['terminated_by']} · "
              f"{len(final['searches'])} search(es) · {t['api_calls']} api calls · "
              f"${t['cost']:.6f} · {t['latency_ms']:.0f}ms · {n_ckpt} checkpoints")
        for s in final["searches"]:
            print(f"      step {s['step']} query: {s['query']}")
            print(f"        -> {s['sources']}  ({s['n_new_chunks']} new chunks)")
        print(f"    answer: {final['answer'][:300]}")
        out["questions"].append({
            "thread_id": thread_id, "question": question, "answer": final["answer"],
            "steps": final["steps"], "terminated_by": final["terminated_by"],
            "searches": final["searches"], "n_checkpoints": n_ckpt,
            "sources_shown": [c["source"] for c in final["seen"]], **t,
        })

    out["total_cost"] = round(sum(q["cost"] for q in out["questions"]), 6)
    out["api_calls_log_lines"] = G.count_lines(G.API_CALLS_LOG)
    out["tool_runs_log_lines"] = G.count_lines(G.TOOL_RUNS_LOG)
    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\n  wrote {OUT.name} · ${out['total_cost']:.6f} · "
          f"{out['api_calls_log_lines']} api calls, {out['tool_runs_log_lines']} tool runs")
