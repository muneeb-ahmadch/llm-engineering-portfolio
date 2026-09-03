# Experiment 15 report: kill it, then resume it

**Run date:** 2026-08-13 · **Installed:** `langgraph 1.2.11`, `langgraph-checkpoint-sqlite 3.1.1`, `langgraph-checkpoint 4.2.0`, `langchain-core 1.5.4` (Python 3.12.12) · **Model:** `gpt-5.6-luna` on `/v1/responses`, config frozen at L06/L09/L11 values (`bge-small · chunk 1000 · fixed · top-5 · rerank pool=20 · hardened prompt · MAX_STEPS=3`) · **API spend:** $0.0176 priced + ~$0.005 in three calls that were killed or unbilled to state ≈ **$0.023**

**The verdict, up front: build Project 2 on my own loop.** None of the three properties clears the bar for a career/research agent that runs for a minute in front of one person, and the two that come closest are cheaper to get without the dependency. The one condition that flips it is written at the bottom, and it is property 2, not property 1.

**And the finding I did not expect, which is the reason this report is worth reading:** with the default settings, **a `kill -9` cost me the model call I had already paid for.** LangGraph's `durability` defaults to `"async"`, meaning checkpoints are written *while the next node runs*, so the crash beat the checkpoint to disk and the resume re-ran the paid `agent` node. `durability="sync"` fixes it. The headline promise of the library is true only on a flag that is not the default.

---

## The two lines (step 1)

**Version:** `langgraph 1.2.11` + `langgraph-checkpoint-sqlite 3.1.1`, installed 2026-08-13 (the lesson said 1.2.10; 1.2.11 is what `pip install -U` gave me).

**The interview sentence:** *"LangGraph writes the run's state to a checkpoint after every node, keyed by a thread_id, so the run can be killed, paused for a human, or inspected step by step and then continued by a different process hours later, which a `while` loop cannot do, because a `while` loop's state dies with its process."*

The sentence I'd add if they push, because it is the part I measured: *"and the checkpoint boundary is the node, not the line, so anything a node does before it pauses happens again when it resumes."*

---

## Step 5 first, because that was the deal: the preregistered implementation failure

[`predictions.md`](predictions.md), P0, written 2026-08-13 02:27 PKT, before `graph.py` existed:

> **the way my build of this will break is: `Annotated[list, add_messages]` will reject my message list, because my L11 messages are raw `/v1/responses` input items (`reasoning`, `function_call`, `function_call_output` dicts with no `role` and no `id`) and `add_messages` is a LangChain-message reducer that will try to coerce them into `BaseMessage` and raise before a single API call goes out.** Confidence 70%.

**Confirmed, all four shapes, at zero cost** ([`probe_p0.py`](probe_p0.py)):

| message shape my graph puts in state | `add_messages` verdict |
|---|---|
| `{"role": "system"...}`, `{"role": "user"...}` | ok, coerced to `SystemMessage` / `HumanMessage` |
| `{"type": "reasoning", "id": ..., "summary": []}` | **`ValueError: Message dict must contain 'role' and 'content' keys`** |
| `{"type": "function_call", "call_id": ..., "arguments": ...}` | **`ValueError`**, same |
| `{"type": "function_call_output", "call_id": ..., "output": ...}` | **`ValueError`**, same |

And the sting is in the row that *passed*: my system message comes back as `SystemMessage(content=..., id='71531f68-…')`, a `BaseMessage` object, which `client.responses.create(input=...)` would reject on the way back out. So the "working" path is a second failure two lines later. The fix was the one I predicted: `Annotated[list, operator.add]`, which is all I ever wanted from a reducer.

**Prediction #2 also confirmed:** `SqliteSaver.from_conn_string(path)` without a `with` block returns a `_GeneratorContextManager` with **no `.put` method**, so not a checkpointer. `SqliteSaver(sqlite3.connect(path, check_same_thread=False))` is what [`graph.py`](graph.py) uses.

**Prediction #3 (serialising OpenAI SDK objects) never fired**, because L11's `_as_input_item()` already converts every output item to a plain dict, so there was nothing un-serialisable to put in state. I am counting that as *avoided by inherited code*, not as a prediction I got right.

**What this says about the L13 criticism:** the gap was real and the correction worked. Both failures I named were in the framework's *seams*, the reducer's type contract and a factory's return type, and neither had anything to do with whether the design was sound. That is the shape of every implementation bug I have hit today, including the one in step 3 that I did *not* predict.

---

## Step 2: the port

[`graph.py`](graph.py) · [`run_graph.py`](run_graph.py) · [`results-step2.json`](results-step2.json)

```
START ──▶ agent ──has tool calls?──▶ tools ──┐
            │ no                             │
            ▼                                └──▶ back to agent
           END                       checkpoint after every node → checkpoints.sqlite
```

The port rule was that **the OpenAI call is byte-identical to L11's**: same model, endpoint, `SEARCH_TOOL`, system prompt + `AGENT_ADDENDUM`, `tool_choice="none"` on the last step, same retriever behind `search_corpus`. `MAX_STEPS` is *imported* from `rag_eval.py` rather than re-declared, so it cannot drift.

What actually changed, and it is a short list:

| L11 `run_agent()` | L15 graph |
|---|---|
| `messages` local variable | `AgentState["messages"]`, reducer `operator.add` |
| `while steps < MAX_STEPS:` | `state["steps"]` + the router; no `while` anywhere |
| `if tool_calls: … else: break` | `route()` on a conditional edge → `"tools"` or `END` |
| bottom half of the loop body | the `tools` node |
| nothing | `SqliteSaver` + `thread_id`, the only new capability |

**Three questions, chosen so both edges out of `agent` fire and so the numbers are comparable to a run I already have:**

| question | L11 steps/searches | **L15 steps/searches** | L11 cost | **L15 cost** | prompt tok (cached) L11 → L15 |
|---|---|---|---|---|---|
| overfitting | 2 / 1 | **2 / 1** | $0.002326 | **$0.004232** | 4947 (3626) → 4946 (1813) |
| reranking | 2 / 1 | **2 / 1** | $0.004051 | **$0.004091** | 4806 (1666) → 4846 (1666) |
| prompt caching | 1 / 0 | **1 / 0** | $0.001956 | **$0.001848** | 1698 (0) → 1698 (0) |

Same trajectory on all three, and the tool queries the model wrote are near-identical to L11's:

> L11: *"Overfitting is when a neural network learns the training data too well and performs poorly on unseen validation or test data"*
> L15: *"Overfitting is when a neural network learns the training data too well and performs poorly on unseen data"*

**The one cost gap is not the framework, it is prompt caching, and I checked before blaming anything.** The overfitting question is 1.82× dearer on the graph, but its prompt is the same size (4946 vs 4947 tokens): what moved is `cached_tokens`, 3626 → 1813. L11 ran twenty questions back to back through the same prefix and was warm; L15 ran three, cold. The reranking question, which had identical cache warmth in both (1666), came out at **$0.004091 vs $0.004051, a gap of 1.0%.** That is the honest measurement of what the graph costs per query: nothing.

**Checkpoints written:** 5 for a two-step question, 3 for a one-step question. For the two-step case: `__start__`, then one before `agent` runs, then one after each of the three node executions.

---

## Step 3: kill it, then resume it

[`crash_resume.py`](crash_resume.py). Three separate processes: **A** dies with `os.kill(getpid(), SIGKILL)` inside the `tools` node, with no exception, no `finally`, no `atexit` and nothing flushed. **B** reads the corpse from a different process. **C** resumes.

The receipt is `api_calls-crash-*.log`: the `agent` node appends `pid=… step=N` *before* every OpenAI call. So "what got re-run" is a grep. If `step=1` appears twice, I paid twice.

**The first run refused to behave, and that is the finding.** Under the default settings the crash left only **2** checkpoints and `state.next = ('agent',)`, so the agent's completed work was *not* on disk. `Pregel.invoke(durability=…)` in the installed source:

> `durability`: The durability mode for the graph execution, defaults to **`"async"`**. `"sync"`: Changes are persisted synchronously before the next step starts. `"async"`: **Changes are persisted asynchronously while the next step executes.** `"exit"`: Changes are persisted only when the graph exits.

So I ran it as two arms, same crash, same instant, one flag:

| | **`durability="async"` (the default)** | **`durability="sync"`** |
|---|---|---|
| checkpoints on disk at crash | **2** | **3** |
| `state.next` after the kill | `('agent',)` | `('tools',)` |
| pending tool call visible in state | no | **yes**, `search_corpus({"query":"Overfitting occurs when…"})` |
| resumed at node | **`agent`** | **`tools`** |
| `step=1` agent calls across all processes | **2** | **1** |
| API calls made by the resuming process | 2 | 1 |
| **total model calls for one 2-step question** | **3** | **2** |
| cost recorded in the final state | $0.002566 | $0.002352 |

Checkpoint history after the `sync` resume, which is the picture the lesson describes:

```
 #  next           source     msgs  steps  pending tasks    answer?
 5  ()             loop       6     2      -                yes
 4  ('agent',)     loop       5     1      agent            no
 3  ('tools',)     loop       4     1      tools            no      <- where it resumed
 2  ('agent',)     loop       2     0      agent            no
 1  ('__start__',) input      0     None   __start__        no
```

**What got re-run, precisely:**

- **`sync`:** nothing. The resuming process ran the `tools` node (the search, free and local) and then one new `agent` call for step 2. The paid step-1 call was not repeated. This is the promise, kept.
- **`async` (default):** the whole `agent` node. The resuming process re-issued step 1, paid for it again, got a fresh query out of the model, and only then went on. **50% waste on a two-step run, and the failure is silent**, because the final state reports `$0.002566` and two calls, since the killed call's usage was never checkpointed. *The state's own cost accounting cannot see the money you lost.* If I had trusted it I would have under-reported the bill by a third and never known.

Both arms produced a correct answer, which is the trap: the run recovers either way, and the only visible difference is on the invoice.

---

## Step 4: the preregistered number

[`interrupt_run.py`](interrupt_run.py) · [`results-step4.json`](results-step4.json). The `tools` node, in this order: **(1)** append a timestamped line to `side_effects.log`, the stand-in for the charge, the `INSERT`, the email, **(2)** `interrupt("approve this search?")` **(3)** the search. Two processes: one pauses and exits, a different one approves.

**Predicted: 2 lines** (P1, written 02:36 PKT before `interrupt_run.py` existed, confidence 90%).
**Measured: 2 lines.**

```
2026-08-12T21:38:31Z pid=8102 CHARGE for search: 'Overfitting is when a neural network learns…'
2026-08-12T21:38:52Z pid=8154 CHARGE for search: 'Overfitting is when a neural network learns…'
```

One approval. Two charges. **Different pids.** The second one happened in the process that granted permission, twenty seconds later, which is exactly the shape of a human approving something the next morning.

The secondary prediction held too: the `agent` node's API calls went **1 → 2**, not 3. The pause is in `tools`, and `agent` had completed a whole node earlier, so the model call was not repeated. The blast radius of an interrupt is precisely one node, the one it is written in. The installed docstring says the same thing without hedging: *"The graph resumes from the start of the node, **re-executing** all logic."*

**One sentence on what the number means for an agent whose node spends money before asking permission:** it charges the card once to ask you whether it may charge the card, and once more when you say yes, so *"pause for approval"* is not a safety feature until the side effect is moved to after the `interrupt()` or into its own node, and the ratio is not a one-off but **N+1 side effects for N pauses**, which for a run that escalates twice is three charges for two approvals.

**An honest note about this run:** the first thread I tried never reached the interrupt. At temperature 1 the model answered *"I don't have that information."* on the first pass without calling the tool at all, so `tools` never ran and there was nothing to pause. I started the experiment again on a clean `thread_id` rather than nudging the prompt, and the abstention is left in `checkpoints.sqlite` (`l15-interrupt-overfitting`, 3 checkpoints) rather than deleted. It is also a small L11 result restated: the agent's trajectory is a distribution, not a property.

---

## What the port cost, since the whole lesson is that machinery is a price

- **20 new packages** into a venv that already had `openai`, `pydantic`, `httpx` and friends: `aiosqlite`, `jsonpatch`, `jsonpointer`, `langchain-core`, `langchain-protocol`, `langgraph`, `langgraph-checkpoint`, `langgraph-checkpoint-sqlite`, `langgraph-prebuilt`, `langgraph-sdk`, `langsmith`, `orjson`, `ormsgpack`, `requests-toolbelt`, `sqlite-vec`, `tenacity`, `uuid-utils`, `websockets`, `xxhash`, `zstandard` (plus forced upgrades of `packaging` and `charset-normalizer`). Into an *empty* 3.12 venv the same two lines install **38 packages**, checked, because "one dependency" is the kind of thing that is technically true and practically false.
- **`langchain-core<2,>=1.4.7` is a hard requirement of `langgraph 1.2.11`**, and `langgraph-checkpoint` requires it too. The docs' *"you don't need to use LangChain to use LangGraph"* is true about the programming model, since my nodes are plain functions calling the OpenAI SDK and no LangChain abstraction appears anywhere in [`graph.py`](graph.py), but it is **not** true about the dependency tree. Worth knowing before saying it in an interview.
- **~40 lines of graph scaffolding** replacing ~15 lines of `while`/`if`. The node bodies are lifted from `run_agent()` almost unchanged.
- **823 KB of sqlite** for 8 threads / 33 checkpoints. Every checkpoint stores the whole `messages` list, so a long agent run stores its context O(n²) times. Not a problem at this size; a real one for a 40-step agent.

---

## Step 6: the verdict for Project 2

**Project 2 is one built agent, the career/research agent, made mine and defensible for an hour, visibly defending against the L12 guardrail failure.** Against the three properties, on today's evidence:

| property | does Project 2 need it? | evidence |
|---|---|---|
| **1. Resume after failure** | **No.** A run is 1 to 4 model calls over a minute, in front of one person. If it dies, I run it again for ~$0.005. And the default `durability="async"` would have re-run the paid node anyway, so the property does not even arrive switched on | step 3, both arms |
| **2. A human in the middle** | **Not yet, and this is the one that could flip it.** The agent has no action today that spends money or writes to the world. An approval gate on a *read* is theatre | step 4 |
| **3. Replay and time travel** | **No.** I already have this, and better. Langfuse has traced every span since L10 with the prompt, the completion, the tool call, the token counts and a URL I can put in a report. `get_state_history()` gives me graph state, not the model's inputs and outputs, and it is local-only | L10–L13 traces |

**Decision: my own loop, and the property that decides it is #1, resume after failure, because it is the only one of the three that the loop genuinely cannot fake, and Project 2 has no run long enough or expensive enough to need it.** Properties 2 and 3 are not "no" because a loop does them well; they are "no" because Project 2 does not do the thing that needs them.

**The condition that flips this, written down now so it is a decision and not a preference:** if Project 2 grows an action that **spends money or writes to the world**, such as sending the email, filing the application or posting the PR, then property 2 decides it and I port, because "pause, persist, resume in another process" is exactly what I would otherwise be building by hand and getting wrong. And I would port *knowing* what step 4 measured: the approval gate goes in its own node, with the side effect strictly after the `interrupt()`, or the gate is worse than no gate.

**What I would not say in an interview:** that I "use LangGraph". What I would say is: *"I ported my agent loop to LangGraph, measured what checkpointing actually guarantees, and found that the default durability mode re-ran the paid model call after a kill -9, and that a side effect placed before `interrupt()` fires twice for one approval. Then I kept my loop, because that project needed none of the three things the library sells."*

---

## Reproduce

```bash
cd projects/15-langgraph
P=../01-chunking/.venv/bin/python

$P probe_p0.py                                  # step 5: the preregistered failures, $0
$P run_graph.py --fresh                         # step 2: three questions, ~$0.010

$P crash_resume.py crash                        # step 3 arm A: dies with SIGKILL
$P crash_resume.py inspect                      #             read from a new process
$P crash_resume.py resume                       #             re-runs the paid agent node
$P crash_resume.py crash   --durability sync    # step 3 arm B
$P crash_resume.py inspect --durability sync
$P crash_resume.py resume  --durability sync    #             resumes at tools, no re-run

$P interrupt_run.py pause   --thread b          # step 4: pause, then the process exits
$P interrupt_run.py approve --thread b          #         a different process approves
```

Receipts on disk: `checkpoints.sqlite` (8 threads, 33 checkpoints), `api_calls-crash-async.log` (3 lines, where the `step=1` duplicate is the finding), `api_calls-crash-sync.log` (2), `side_effects.log` (2 lines, 1 approval).

One bookkeeping note, since it would otherwise look like a discrepancy: step 2's counters (`api_calls.log` / `tool_runs.log`) were truncated by the first `crash_resume.py crash` run, which wiped them before I split the counters per arm. Step 2's real totals, **5 API calls and 2 tool runs**, are preserved in `results-step2.json` (`api_calls_log_lines`, `tool_runs_log_lines`), recorded at the time. The crash and interrupt arms each write their own pair of files and are unaffected.
