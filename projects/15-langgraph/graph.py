#!/usr/bin/env python
"""Lesson 15: the L11 agent loop, ported to a LangGraph state graph.

The port rule: the OpenAI call is IDENTICAL to L11's. Same model, same endpoint,
same system prompt + AGENT_ADDENDUM, same SEARCH_TOOL schema, same
tool_choice="none" on the last step, same retriever behind search_corpus, same
MAX_STEPS=3. If the answers differ from L11's, it is the framework or the dice, not a second change I made while I was in there.

    START ──▶ agent ──tool calls?──▶ tools ──┐
                │  no                        │
                ▼                            └──▶ back to agent
               END

What is new, and it is the only thing that is new: a file-backed SqliteSaver
writes the state after every node under a thread_id, so a run can be killed and
picked up by a different process (step 3) or paused for a human (step 4).

Two side-effect counters exist for the measurement, and they are the point:

    api_calls.log   one line per OpenAI call the agent node makes
    tool_runs.log   one line per search the tools node runs

Both are appended to BEFORE the work they count, because the question this
lesson asks is "what got re-run", and a counter that only records completions
cannot answer it.
"""
import json
import operator
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, TypedDict

HERE = Path(__file__).resolve().parent
HARNESS_DIR = HERE.parents[1] / "practice" / "lesson-001-chunking"
if not HARNESS_DIR.exists():                      # running from inside practice/
    HARNESS_DIR = HERE.parent / "lesson-001-chunking"
sys.path.insert(0, str(HARNESS_DIR))
import numpy as np                                                    # noqa: E402
import rag_eval as R                                                  # noqa: E402

from langgraph.graph import END, START, StateGraph                    # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver                   # noqa: E402

# ---------------------------------------------------------------- config, held at L11's values
CONFIG = dict(encoder="bge-small", chunk_size=1000, overlap=50, top_k=5,
              chunker="fixed", rerank=True, pool=20)
MODEL = "gpt-5.6-luna"
PROMPT_VARIANT = "hardened"
MAX_STEPS = R.MAX_STEPS          # 3: imported, not re-declared, so it cannot drift

DB_PATH = HERE / "checkpoints.sqlite"
API_CALLS_LOG = HERE / "api_calls.log"
TOOL_RUNS_LOG = HERE / "tool_runs.log"
SIDE_EFFECTS_LOG = HERE / "side_effects.log"      # step 4's counter, kept separate


def count_lines(path):
    return len(path.read_text().splitlines()) if path.exists() else 0


def note(path, text):
    """Append one timestamped line. THIS is the side effect being measured."""
    with path.open("a") as f:
        f.write(f"{datetime.now(timezone.utc).isoformat()} pid={os.getpid()} {text}\n")


# ---------------------------------------------------------------- state
#
# `messages` is the /v1/responses INPUT list: role dicts for system/user, then the
# model's own output items (reasoning, function_call) fed back verbatim, then
# function_call_output dicts. It is the same list L11's run_agent() carried in a
# local variable: the only difference is that a reducer now says how a node's
# update merges into it, and that it gets written to sqlite between nodes.
class AgentState(TypedDict):
    question: str
    messages: Annotated[list, operator.add]
    searches: Annotated[list, operator.add]
    usage: Annotated[list, operator.add]     # one dict per API call; summed at the end
    steps: int
    answer: str | None
    terminated_by: str | None
    seen: Annotated[list, operator.add]      # every chunk shown, for the judge/report


# ---------------------------------------------------------------- the index (built once)
_INDEX = {}


def build_index(verbose=True):
    """Load corpus, chunk, embed, and bind search_corpus, byte-identical to the
    path evaluate() takes, so the tool the graph calls is the tool L11 called."""
    if _INDEX:
        return _INDEX
    t0 = time.time()
    docs = R.load_documents()
    chunks = R.build_chunks(docs, CONFIG["chunk_size"], CONFIG["overlap"],
                            CONFIG["chunker"])
    encoder_model = R.get_model(CONFIG["encoder"])
    matrix, cached = R.embed_chunks(encoder_model, chunks, CONFIG["encoder"],
                                    CONFIG["chunk_size"], CONFIG["overlap"], docs,
                                    CONFIG["chunker"])
    cross_encoder = R.get_cross_encoder() if CONFIG["rerank"] else None
    _INDEX.update(
        docs=docs, chunks=chunks, encoder_model=encoder_model, matrix=matrix,
        cross_encoder=cross_encoder,
        search_corpus=R.make_search_corpus(encoder_model, CONFIG["encoder"], chunks,
                                           matrix, cross_encoder, CONFIG["pool"]),
    )
    if verbose:
        print(f"  index: {len(chunks)} chunks, embeddings {'cached' if cached else 'built'}"
              f" ({time.time() - t0:.1f}s)")
    return _INDEX


def retrieve(question):
    """The initial top-k, the same as every lesson since L06."""
    ix = build_index()
    qv = R.embed_queries(ix["encoder_model"], [question], CONFIG["encoder"])[0]
    row = qv @ ix["matrix"].T
    n_candidates = CONFIG["pool"] if CONFIG["rerank"] else CONFIG["top_k"]
    idx = np.argsort(-row)[:n_candidates]
    candidates = [ix["chunks"][i] for i in idx]
    if CONFIG["rerank"]:
        pairs = [(question, c["text"]) for c in candidates]
        ce = ix["cross_encoder"].predict(pairs)
        order = np.argsort(-np.asarray(ce))[:CONFIG["top_k"]]
        candidates = [candidates[i] for i in order]
    return candidates[:CONFIG["top_k"]]


def initial_state(question):
    retrieved = retrieve(question)
    return {
        "question": question,
        "messages": [
            {"role": "system",
             "content": R.system_prompt(PROMPT_VARIANT) + R.AGENT_ADDENDUM},
            {"role": "user",
             "content": f"Context:\n{R.build_context(retrieved)}\n\nQuestion: {question}"},
        ],
        "searches": [], "usage": [], "seen": list(retrieved),
        "steps": 0, "answer": None, "terminated_by": None,
    }


# ---------------------------------------------------------------- nodes
_CLIENT = None


def client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = R.get_openai_client()
    return _CLIENT


def agent(state: AgentState) -> dict:
    """L11's OpenAI call, unchanged, inside a function.

    The only lines that are not lifted from run_agent() are the two that read and
    write `steps`, in the loop that was `while steps < MAX_STEPS`, and a graph has
    no while, so the cap lives in the state and in the router.
    """
    step = state["steps"] + 1
    last = step == MAX_STEPS          # budget spent: force an answer, exactly as L11

    note(API_CALLS_LOG, f"step={step} last={last} q={state['question'][:40]!r}")

    t0 = time.perf_counter()
    resp = client().responses.create(
        model=MODEL,
        input=state["messages"],
        tools=R.SEARCH_TOOL,
        tool_choice="none" if last else "auto",
        store=False,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    usage = R._responses_usage(resp)
    usage["latency_ms"] = latency_ms
    out = [R._as_input_item(i) for i in resp.output]
    tool_calls = [i for i in out if i["type"] == "function_call"]

    update = {"messages": out, "steps": step, "usage": [usage]}
    if not tool_calls:
        update["answer"] = resp.output_text
        update["terminated_by"] = "cap" if last else "answer"
    return update


def tools(state: AgentState) -> dict:
    """Runs search_corpus for every pending call and appends the results.

    In L11 this was the bottom half of the while body. Here it is a node, which
    means the state is checkpointed on both sides of it, and means that when a
    run dies or pauses, THIS is the function that starts again at line one.
    """
    ix = build_index(verbose=False)
    calls = [m for m in state["messages"] if m.get("type") == "function_call"]
    done = {m["call_id"] for m in state["messages"]
            if m.get("type") == "function_call_output"}
    pending = [c for c in calls if c["call_id"] not in done]

    seen_keys = {c["text"] for c in state["seen"]}
    msgs, searches, fresh_chunks = [], [], []
    for tc in pending:
        args = json.loads(tc["arguments"])
        query, k = args["query"], args.get("k", 5)
        note(TOOL_RUNS_LOG, f"step={state['steps']} query={query[:60]!r}")
        hits = ix["search_corpus"](query, k)
        fresh = [c for c in hits if c["text"] not in seen_keys]
        seen_keys.update(c["text"] for c in fresh)
        fresh_chunks.extend(fresh)
        searches.append({"step": state["steps"], "query": query, "k": k,
                         "sources": [c["source"] for c in hits],
                         "n_new_chunks": len(fresh)})
        msgs.append({"type": "function_call_output", "call_id": tc["call_id"],
                     "output": R.build_context(hits) or "No results."})
    return {"messages": msgs, "searches": searches, "seen": fresh_chunks}


def route(state: AgentState) -> str:
    """The conditional edge. L11 spelled this `if tool_calls: … else: break`."""
    done = {m["call_id"] for m in state["messages"]
            if m.get("type") == "function_call_output"}
    pending = [m for m in state["messages"]
               if m.get("type") == "function_call" and m["call_id"] not in done]
    return "tools" if pending else END


# ---------------------------------------------------------------- the graph
def make_graph(checkpointer=None, tools_node=tools):
    """tools_node is swappable so steps 3 and 4 can wrap it (crash / interrupt)
    without forking the graph definition itself."""
    g = StateGraph(AgentState)
    g.add_node("agent", agent)
    g.add_node("tools", tools_node)
    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    g.add_edge("tools", "agent")
    return g.compile(checkpointer=checkpointer)


def sqlite_saver(path=DB_PATH):
    """File-backed, not InMemorySaver, step 3 needs the state to outlive the
    process. NOT via SqliteSaver.from_conn_string(): that is a @contextmanager, so
    without a `with` block it hands back a _GeneratorContextManager with no .put()
    and is not a checkpointer at all (predicted in predictions.md, confirmed in
    probe_p0.py). An explicit connection avoids the question."""
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn)


def totals(usage):
    t = {"prompt_tokens": 0, "completion_tokens": 0, "cached_tokens": 0,
         "reasoning_tokens": 0, "latency_ms": 0.0}
    for u in usage:
        for k in t:
            t[k] += u.get(k, 0)
    t["cost"] = R.compute_cost(t["prompt_tokens"], t["completion_tokens"],
                               t["cached_tokens"], R.PRICES[MODEL])
    t["api_calls"] = len(usage)
    return t
