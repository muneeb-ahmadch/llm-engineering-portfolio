#!/usr/bin/env python
"""Step 4: the preregistered measurement: what does a side effect placed BEFORE
`interrupt()` cost you?

The `tools` node here is L15's tools node with two lines inserted, in this order:

    1.  append a timestamped line to side_effects.log   <- the counted side effect
    2.  interrupt("approve this search?")               <- the pause
    3.  ...then the search runs, exactly as before

Stand-in for the real thing: line 1 is the charge, the INSERT, the email, the
"reserve the inventory" call. The number predicted in predictions.md (P1) was
written before this file existed.

    python interrupt_run.py pause      # process A: run until the interrupt, then exit
    python interrupt_run.py approve    # process B: resume the same thread with an answer

Two processes again, because a pause you can only resume from the process that
started it is not a pause, it is a blocked while loop with extra steps.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph as G                                                     # noqa: E402
from langgraph.types import Command, interrupt                        # noqa: E402

# --thread <suffix> starts the experiment over on a clean thread. Needed because at
# temperature 1 the agent node sometimes answers (or abstains) without calling the
# tool at all, and a run that never enters `tools` never reaches the interrupt.
_sfx = (sys.argv[sys.argv.index("--thread") + 1] if "--thread" in sys.argv else "")
THREAD = f"l15-interrupt-overfitting{('-' + _sfx) if _sfx else ''}"
QUESTION = "What is overfitting in a neural network?"
CONFIG = {"configurable": {"thread_id": THREAD}}
G.API_CALLS_LOG = G.HERE / "api_calls-interrupt.log"
G.TOOL_RUNS_LOG = G.HERE / "tool_runs-interrupt.log"
SIDE_EFFECTS = G.SIDE_EFFECTS_LOG


def approving_tools(state):
    """The side effect, then the request for permission. In that order, which is
    the order a careless implementation naturally lands on, because you write the
    code that does the work first and bolt the approval on afterwards."""
    calls = [m for m in state["messages"] if m.get("type") == "function_call"]
    done = {m["call_id"] for m in state["messages"]
            if m.get("type") == "function_call_output"}
    pending = [c for c in calls if c["call_id"] not in done]
    query = json.loads(pending[0]["arguments"])["query"] if pending else "?"

    # ---- line 1: the side effect. A number at the end, as the assignment asks.
    G.note(SIDE_EFFECTS, f"CHARGE for search: {query[:70]!r}")
    n = G.count_lines(SIDE_EFFECTS)
    print(f"    side effect recorded, side_effects.log is now {n} line(s)")

    # ---- line 2: the pause. Raises GraphInterrupt on the first pass through.
    decision = interrupt({"question": "approve this search?", "query": query})
    print(f"    approval received: {decision!r}")

    # ---- line 3 onward: the original node, unchanged.
    return G.tools(state)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "pause"
    print(f"\n=== {mode} ===")
    G.build_index()
    app = G.make_graph(checkpointer=G.sqlite_saver(), tools_node=approving_tools)

    if mode == "pause":
        for p in (SIDE_EFFECTS, G.API_CALLS_LOG, G.TOOL_RUNS_LOG):
            p.write_text("")
        print(f"  side_effects.log starts at {G.count_lines(SIDE_EFFECTS)} lines")
        out = app.invoke(G.initial_state(QUESTION), CONFIG, durability="sync")
        intr = out.get("__interrupt__")
        print(f"  invoke() RETURNED (the process is free to exit) with: "
              f"{intr[0].value if intr else None}")
        print(f"  state.next = {app.get_state(CONFIG).next}")
        print(f"\n  side_effects.log: {G.count_lines(SIDE_EFFECTS)} line(s), "
              f"before approval")
        print(f"  api_calls-interrupt.log: {G.count_lines(G.API_CALLS_LOG)} line(s)")

    elif mode == "approve":
        before = G.count_lines(SIDE_EFFECTS)
        api_before = G.count_lines(G.API_CALLS_LOG)
        print(f"  side_effects.log before approving: {before} line(s)")
        final = app.invoke(Command(resume="approved"), CONFIG, durability="sync")
        after = G.count_lines(SIDE_EFFECTS)
        api_after = G.count_lines(G.API_CALLS_LOG)
        t = G.totals(final["usage"])
        print(f"\n  finished: steps {final['steps']} · {final['terminated_by']} · "
              f"{len(final['searches'])} search(es) · ${t['cost']:.6f}")
        print(f"  answer: {final['answer'][:240]}")
        print(f"\n  ---- THE MEASUREMENT ----")
        print(f"  side effects BEFORE the interrupt fired : {before}")
        print(f"  side effects AFTER  one approval        : {after}")
        print(f"  approvals given                         : 1")
        print(f"  agent-node API calls: {api_before} -> {api_after} "
              f"(3 would mean the pause also re-ran the paid call)")
        print("\n  side_effects.log:")
        print("    " + "\n    ".join(SIDE_EFFECTS.read_text().splitlines()))
        n_ckpt = len(list(app.get_state_history(CONFIG)))
        print(f"\n  checkpoints on this thread: {n_ckpt}")
        (G.HERE / "results-step4.json").write_text(json.dumps({
            "predicted_side_effects": 2, "measured_side_effects": after,
            "approvals": 1, "side_effects_before_approval": before,
            "agent_api_calls_before": api_before, "agent_api_calls_after": api_after,
            "n_checkpoints": n_ckpt, "cost_in_state": t["cost"],
            "answer": final["answer"],
        }, indent=2))
