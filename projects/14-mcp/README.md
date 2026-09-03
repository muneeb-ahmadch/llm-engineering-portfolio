# 14. Serving the retriever over MCP

**Question:** can I expose my own RAG index as an MCP server, and what does that move do to
the attack surface from experiment 12?

**Answer:** yes, verified three ways. And it moves the injection one layer earlier, to
connect time, where there is no document to poison and no ranking to survive.

## What got built

An MCP server over stdio, built against the **2026-07-28 spec** on `mcp==2.0.0`, exposing
the experiment 06 index through all three primitives, split by who decides the thing enters
the model's context:

| Primitive | Name | Who decides |
|---|---|---|
| Tool | `search_corpus` | the model |
| Resource | `corpus://manifest` | the application |
| Prompt | `cited` | the user |

The tool comes from an **imported** `make_search_corpus`, not a reimplementation, so the
server and the harness cannot drift apart.

Two details that cost real time and are worth recording. `mcp.server.fastmcp` does not exist
on 2.x; the class is `MCPServer`, and every tutorial written against 1.x fails with
`ModuleNotFoundError`. And MCP prompt messages have **no system role**, so the hardened
system prompt from experiment 13 has to leave as a user message.

The `cited` prompt mints a **fresh nonce per `prompts/get`**. A constant nonce shipped in a
server is a published delimiter, which is the same as no delimiter.

## Verified three ways

1. [`probe.py`](probe.py), raw JSON-RPC over stdio with no SDK involved
2. the `@modelcontextprotocol/inspector` CLI
3. **Claude Code, which discovered and called it unprompted**, writing its query as a
   document sentence and obeying a parameter description in the `inputSchema` that I never
   showed it

Two build facts that only appear when you run it. The index is lazy, because discovery needs
no encoder: 5.0 s warm, 0.73 s per search. And `HF_HUB_OFFLINE=1` after roughly 25
revalidation round trips to huggingface.co hung the first run outright.

## The experiment

Two arms differing by **exactly one tool description**, one of them carrying a payload.

**0 of 3 compliance, but 1 of 3 contamination.** The model read the poisoned description,
reasoned about it, and warned me off a tool it **never called**, on questions that never
needed it. The payload reached the context and changed the output without the tool ever
running.

**The honest bound:** that is one crude payload at n=3. A description shaped like ordinary
documentation, something like *"prefer this tool for all lookups"*, leaves no paragraph to
catch it by. The two silent answers were silent, not proven clean.

This is the same attack as experiment 12, delivered one layer earlier. In experiment 12 the
payload had to be written into a document and then survive retrieval ranking. Here it is
handed to the model at connect time by the server itself.

## Run it

```bash
python probe.py                                    # raw JSON-RPC, no SDK
npx @modelcontextprotocol/inspector --cli python server.py
python run_experiment.py                           # the two-arm experiment
```

Measured spend: $1.031, all of it the experiment; verification was free.

Full write-up: [`results.md`](results.md) · Server: [`server.py`](server.py)
