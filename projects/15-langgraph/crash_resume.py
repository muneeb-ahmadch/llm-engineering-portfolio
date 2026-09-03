#!/usr/bin/env python
"""Step 3: kill it, then resume it. Three separate processes, on purpose.

    python crash_resume.py crash   [--durability sync] # process A: SIGKILL inside `tools`
    python crash_resume.py inspect [--durability sync] # process B: reads the corpse
    python crash_resume.py resume  [--durability sync] # process C: continues the thread

Process A really does die, os.kill(getpid(), SIGKILL), not an exception, not
sys.exit. Nothing is flushed, no finally block runs, no atexit handler fires.
Whatever is in checkpoints.sqlite afterwards got there because the checkpointer
committed it, which is the only claim this lesson makes.

TWO ARMS, and the second one exists because the first one surprised me.
`Pregel.invoke(durability=...)` defaults to **"async"**, "Changes are persisted
asynchronously WHILE THE NEXT STEP EXECUTES". So under the default, a kill -9
during node N+1 can land before node N's checkpoint does, and the resume re-runs
node N. `durability="sync"` persists before the next step starts. The arms are
the same crash at the same instant; only that flag moves.

The receipt is `api_calls.log`. Every line carries a step number and a pid, so
"what got re-run" is a grep, not a feeling: if `step=1` appears twice, the paid
model call was repeated; if it appears once across all three processes, it was not.
"""
import os
import signal
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import graph as G                                                     # noqa: E402

DURABILITY = "sync" if "--durability" in sys.argv and "sync" in sys.argv else "async"
THREAD = f"l15-crash-overfitting-{DURABILITY}"
QUESTION = "What is overfitting in a neural network?"
CONFIG = {"configurable": {"thread_id": THREAD}}

# One pair of counters per arm, so arm B cannot overwrite arm A's evidence.
G.API_CALLS_LOG = G.HERE / f"api_calls-crash-{DURABILITY}.log"
G.TOOL_RUNS_LOG = G.HERE / f"tool_runs-crash-{DURABILITY}.log"


def crashing_tools(state):
    """The tools node, sabotaged. It records that it was entered, then the process
    dies before a single search runs, the worst moment to die, because the model
    has already been paid for and its tool call is pending."""
    G.note(G.TOOL_RUNS_LOG, f"ENTERED tools (steps={state['steps']}), about to SIGKILL")
    print(f"    tools node entered at steps={state['steps']}, kill -9 now")
    sys.stdout.flush()
    os.kill(os.getpid(), signal.SIGKILL)


def show_history(app, label):
    hist = list(app.get_state_history(CONFIG))
    print(f"\n  {label}: {len(hist)} checkpoints (newest first)")
    print(f"    {'#':>2}  {'next':<14} {'source':<10} {'msgs':<5} {'steps':<6} "
          f"{'pending tasks':<16} answer?")
    for i, s in enumerate(hist):
        pend = ",".join(t.name for t in s.tasks) or "-"
        print(f"    {len(hist) - i:>2}  {str(s.next):<14} "
              f"{str(s.metadata.get('source')):<10} "
              f"{len(s.values.get('messages', [])):<5} "
              f"{str(s.values.get('steps')):<6} {pend[:16]:<16} "
              f"{'yes' if s.values.get('answer') else 'no'}")
    return hist


def counters(label):
    api = G.API_CALLS_LOG.read_text().splitlines() if G.API_CALLS_LOG.exists() else []
    tool = G.TOOL_RUNS_LOG.read_text().splitlines() if G.TOOL_RUNS_LOG.exists() else []
    mine = [l for l in api if QUESTION[:40] in l]
    print(f"  {label}: {len(mine)} agent-node API calls for this thread, "
          f"{len(tool)} tool-node entries (all threads)")
    for l in mine:
        print(f"    {l}")
    return mine


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "inspect"
    print(f"\n=== {mode} · pid {os.getpid()} ===")

    if mode == "crash":
        for p in (G.API_CALLS_LOG, G.TOOL_RUNS_LOG):
            # counters start clean for this experiment; checkpoints.sqlite does NOT
            # get wiped: step 2's threads stay in it, which is the point of a file
            p.write_text("")
        G.build_index()
        app = G.make_graph(checkpointer=G.sqlite_saver(), tools_node=crashing_tools)
        print(f"    invoking thread {THREAD!r} durability={DURABILITY!r} "
              f", this process will not return")
        app.invoke(G.initial_state(QUESTION), CONFIG, durability=DURABILITY)
        print("    UNREACHABLE, if you can read this, the kill did not work")

    elif mode == "inspect":
        app = G.make_graph(checkpointer=G.sqlite_saver())
        show_history(app, "after the crash, read from a different process")
        st = app.get_state(CONFIG)
        print(f"\n  state.next = {st.next}   <- the node a resume will run")
        pend = [m for m in st.values["messages"] if m.get("type") == "function_call"]
        for m in pend:
            print(f"  pending tool call: search_corpus({m['arguments'][:80]}...)")
        counters("counters at crash time")

    elif mode == "resume":
        G.build_index()
        app = G.make_graph(checkpointer=G.sqlite_saver())   # real tools node this time
        before = counters("before resume")
        st = app.get_state(CONFIG)
        print(f"  resuming thread {THREAD!r} at node {st.next} "
              f"(steps so far: {st.values['steps']})")
        # invoke(None, ...), no new input. Continue whatever is in the checkpoint.
        final = app.invoke(None, CONFIG, durability=DURABILITY)
        after = counters("after resume")
        t = G.totals(final["usage"])
        print(f"\n  finished: steps {final['steps']} · {final['terminated_by']} · "
              f"{len(final['searches'])} search(es) · {t['api_calls']} api calls billed "
              f"in state · ${t['cost']:.6f}")
        print(f"  answer: {final['answer'][:280]}")
        show_history(app, "after the resume")
        print(f"\n  RE-RUN CHECK, step=1 agent calls across all processes: "
              f"{len([l for l in after if 'step=1 ' in l])} "
              f"(1 = the paid call was not repeated)")
        print(f"  new API calls made by the resuming process: {len(after) - len(before)}")
