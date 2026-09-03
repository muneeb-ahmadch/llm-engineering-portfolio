# 15. Kill it, then resume it

**Question:** LangGraph promises durable execution. Is that worth a dependency for an agent
that runs for a minute in front of one person?

**Answer:** not yet, and the headline promise is true only on a flag that is not the default.

## The finding

**With the default settings, `kill -9` cost me a model call I had already paid for.**

`durability` defaults to `"async"`, which writes checkpoints *while the next node runs*. The
crash beat the checkpoint to disk, so the resume re-ran the paid `agent` node. Setting
`durability="sync"` fixes it.

This is the kind of thing you only learn by actually killing the process. The library's
documentation is not wrong; the default just does not mean what the pitch implies, and the
gap between them is a real bill.

## Method

Three properties tested against my own loop from experiment 11, with the pipeline
configuration frozen at the values established in experiments 06, 09 and 11 so that nothing
but the orchestration changed. Predictions in [`predictions.md`](predictions.md), committed
first. Crash, interrupt and resume runs each logged separately, so the API call logs can be
diffed against the tool run logs to prove what was replayed:

```
api_calls.log · api_calls-crash-sync.log · api_calls-crash-async.log · api_calls-interrupt.log
tool_runs.log · tool_runs-crash-sync.log · tool_runs-crash-async.log · tool_runs-interrupt.log
side_effects.log · checkpoints.sqlite
```

## Verdict

**Build on my own loop.** None of the three properties clears the bar for an agent that runs
for a minute in front of one person, and the two that come closest are cheaper to obtain
without the dependency. The condition that would flip the decision is written at the bottom
of the report, and it is property 2, human in the loop interrupts, not property 1.

Versions, because these results are version specific: `langgraph 1.2.11`,
`langgraph-checkpoint-sqlite 3.1.1`, `langgraph-checkpoint 4.2.0`, `langchain-core 1.5.4`,
Python 3.12.12.

Measured spend: about $0.023, including three calls that were killed or unbilled to state.

## Run it

```bash
python run_graph.py             # the baseline run
python crash_resume.py          # kill -9 mid-run, then resume from the checkpoint
python interrupt_run.py         # pause for a human, resume from a different process
```

Full write-up: [`results.md`](results.md) · Predictions: [`predictions.md`](predictions.md)
