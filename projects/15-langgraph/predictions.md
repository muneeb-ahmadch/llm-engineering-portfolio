# Preregistration: L15

**P0 written 2026-08-13 02:27 PKT**, before a line of `graph.py` existed. State at time of
writing: `langgraph==1.2.11` and `langgraph-checkpoint-sqlite==3.1.1` installed (that is the
whole of step 1's build work); no graph file created; no node written; nothing run.

L13 said my predictions always cover the *design* and never the *code*. So P0 is deliberately
not about whether checkpointing works. It is about which line of **my** build throws first.

---

## P0: the implementation failure (the one line I owe from L13)

**The way my build of this will break is: `Annotated[list, add_messages]` will reject my
message list, because my L11 messages are raw `/v1/responses` input items (`reasoning`,
`function_call`, `function_call_output` dicts with no `role` and no `id`) and `add_messages`
is a LangChain-message reducer that will try to coerce them into `BaseMessage` and raise
before a single API call goes out.**

Confidence 70%. The fix I expect to make: drop `add_messages` and use `operator.add` as the
reducer. Appending is all I ever needed from it, and the LangChain message model is the part
of LangGraph I am specifically not adopting.

Two more implementation failures, ranked, so this is a real ordering and not one lucky guess:

| # | the line | what breaks | conf |
|---|---|---|---|
| 2 | `SqliteSaver.from_conn_string(path)` used **without** `with` | it is a `@contextmanager`, so I get a `_GeneratorContextManager` (or a connection closed the moment the block exits) and `compile()` either raises on a missing `.put` or the DB is empty when step 3 tries to resume from a *new process* | 55% |
| 3 | the checkpointer serialising OpenAI SDK objects | if I put `resp.output` items in state as SDK objects rather than dicts, msgpack serialisation fails or silently pickles something that will not survive a process boundary, the exact thing step 3 tests | 40% |

**What would falsify all three:** the graph runs first try. If it does, I record that the port
was mechanical and that L13's criticism does not get a win here.

---

## P1: the step-4 number (a side effect placed before `interrupt()`)

**Written 2026-08-13 02:36 PKT.** Honest about what I already knew when I wrote it: `graph.py`,
`run_graph.py` and both arms of step 3 had already run, so I had *just* watched a resume re-run
a node. `interrupt_run.py` did not exist, not a line of it, and no interrupt had been fired.
This is weaker than P0, which predates every line of code, and it is marked as such rather than
presented as an equal.

Design under test: the `tools` node appends one timestamped line to `side_effects.log`
(the counter = number of lines), **then** calls `interrupt("approve this search?")`, then runs
the search. One question, one pause, one approval, one `Command(resume=...)`.

**Prediction: 2 lines. Confidence 90%.**

Checkpoints are written *between* nodes, so there is no checkpoint between my append and the
`interrupt()`. Resuming re-enters `tools` at line one: the append happens a second time, and
this time `interrupt()` returns the approval value instead of raising. Append, interrupt,
(pause), append, resume → **2**.

Where the 10% sits: if LangGraph deduplicated writes within a superstep, or replayed the node
from some finer-grained record, it would be 1, and the docs' idempotence warning would be
about something narrower than I think. I do not expect this; *"interrupts work by re-running
the nodes they were called from"* only parses if the answer is 2.

**Secondary, same run, no extra cost:** the `agent` node's API call must NOT be repeated by the
approval, because the pause is in `tools` and `agent` completed a whole node earlier.
`api_calls-interrupt.log` should end at **2** lines (step 1 before the pause, step 2 after),
never 3. If that one comes out at 3, then step 3's async-durability finding also applies to
interrupts and the pause costs a model call every time.

**The number that matters is not 2, it is the ratio: 2 side effects per 1 approval. Every
side effect placed before an `interrupt()` happens N+1 times for N pauses.**
