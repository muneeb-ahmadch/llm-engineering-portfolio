#!/usr/bin/env python
"""L14 — the L06 retrieval index, served over MCP.

No new capability. Every one of the three primitives below already existed in
`rag_eval.py` as a Python object, and this file adds nothing to what they do:

    tool      search_corpus(query, k)   <- make_search_corpus(), imported, not reimplemented
    resource  corpus://manifest         <- load_documents() + build_chunks()
    prompt    cited                     <- SYSTEM_PROMPTS["cited"]

What it adds is the WIRE. `rag_eval.py` could only ever be used by a process that
imported `rag_eval`. Anything that speaks MCP can use this — Claude Code, the
Inspector, Cursor — none of which has seen a line of the harness.

Read against the 2026-07-28 spec, which matters because it changed things that
older tutorials still teach: the class is `MCPServer` (FastMCP is gone — the
import `mcp.server.fastmcp` raises ModuleNotFoundError on mcp 2.0.0), the
protocol is stateless (per-request `_meta`, plus a `server/discover` request in
place of the old initialize handshake), and sampling is deprecated.

    uv pip install --python ../lesson-001-chunking/.venv/bin/python "mcp[cli]"
    ../lesson-001-chunking/.venv/bin/python server.py       # speaks JSON-RPC on stdin/stdout
    npx @modelcontextprotocol/inspector -- <that command>   # to poke it by hand

The venv is the harness's existing one, deliberately: `search_corpus` needs
torch + sentence-transformers + the embedding cache that are already sitting in
`lesson-001-chunking/.venv`, and a fresh `uv add` project would re-download ~2GB
of wheels to end up in the same place.
"""
import contextlib
import json
import os
import sys
from pathlib import Path
from typing import Annotated

from pydantic import Field

HERE = Path(__file__).resolve().parent
HARNESS_DIR = HERE.parents[0] / "lesson-001-chunking"
sys.path.insert(0, str(HARNESS_DIR))

# Before torch is anywhere near the import graph. HF's progress bars and warnings
# are the classic way to break a stdio server: this process's stdout is a
# JSON-RPC frame stream, and one stray "Downloading..." line on it is a parse
# error at the client, not a cosmetic blemish. Belt and braces — _index() also
# redirects stdout to stderr for the whole build.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# OFFLINE, and this one was measured, not assumed. Both encoders have been in
# ~/.cache/huggingface since L02, but sentence-transformers still revalidates
# every config file over the network on load: ~25 HTTP round-trips to
# huggingface.co, which on the first run of this server stalled long enough that
# the client gave up. Same weights, no network: 5.0s to a warm index, 0.7s per
# search. A local retrieval server that can be hung by someone else's CDN is a
# server with an availability bug, so the default is off and the escape hatch is
# `HF_HUB_OFFLINE=0` for the day a model isn't cached yet.
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import rag_eval as R  # noqa: E402
from mcp.server import MCPServer  # noqa: E402
from mcp.server.mcpserver.prompts.base import UserMessage  # noqa: E402

# The L06 winning config, byte-identical to the dict in rejudge.py. Changing a
# number here silently makes this server a different retriever from the one every
# report since L06 was measured on, so it is stated once and read from.
CONFIG = dict(encoder="bge-small", chunk_size=1000, overlap=50, chunker="fixed",
              top_k=5, rerank=True, pool=20)

mcp = MCPServer(
    name="corpus",
    version="0.1.0",
    instructions=(
        "Retrieval over a fixed local corpus of 21 markdown notes from an AI "
        "engineering bootcamp (LLM APIs, RAG, evals, agents, fine-tuning, "
        "computer vision). It answers questions ABOUT THOSE NOTES only, and "
        "returns raw chunks — it does not summarise, rank by truth, or verify."
    ),
)


# ---------------------------------------------------------------- the index

_INDEX = None


def _index():
    """Build the L06 index once, on first use, and hand back the same objects after.

    Lazy on purpose. A client starts this process and then talks to it; loading
    bge-small and the cross-encoder costs ~10s of wall clock, and doing that at
    import time spends it before the server can answer `server/discover` or
    `tools/list` — neither of which needs an index. So discovery is instant and
    the cost lands on the first call that actually retrieves.

    Not expensive twice: embed_chunks() keys its .npy cache on a hash of every
    document's content, so this reads the same matrix the L06-L13 runs used
    rather than re-encoding 21 documents.
    """
    global _INDEX
    if _INDEX is None:
        with contextlib.redirect_stdout(sys.stderr):     # see the os.environ note
            docs = R.load_documents()
            chunks = R.build_chunks(docs, CONFIG["chunk_size"], CONFIG["overlap"],
                                    CONFIG["chunker"])
            model = R.get_model(CONFIG["encoder"])
            matrix, cached = R.embed_chunks(model, chunks, CONFIG["encoder"],
                                            CONFIG["chunk_size"], CONFIG["overlap"],
                                            docs, CONFIG["chunker"])
            ce = R.get_cross_encoder() if CONFIG["rerank"] else None
            search = R.make_search_corpus(model, CONFIG["encoder"], chunks, matrix,
                                          ce, CONFIG["pool"])
            print(f"[corpus] index ready: {len(docs)} docs, {len(chunks)} chunks, "
                  f"embeddings {'from cache' if cached else 'freshly encoded'}",
                  file=sys.stderr)
        _INDEX = {"docs": docs, "chunks": chunks, "search": search}
    return _INDEX


# ---------------------------------------------------------------- tool (model-controlled)

@mcp.tool(title="Search the bootcamp corpus")
def search_corpus(
    query: Annotated[str, Field(description=(
        "The search query. This is matched against the text of the documents "
        "themselves, so phrase it the way the answer would be written in a "
        "document, not the way a question is asked."
    ))],
    k: Annotated[int, Field(ge=1, le=10, description="How many chunks to return (1-10).")] = 5,
) -> str:
    """Search a fixed local corpus of 21 markdown notes from an AI engineering
    bootcamp — LLM APIs and pricing, RAG, chunking, retrieval evaluation,
    agents, fine-tuning, and computer vision — and get back the most similar
    verbatim chunks, each labelled with the file it came from.

    Use this when a question is about the contents of those notes. The results
    are raw retrieved text, not answers: they may be off-topic, may contradict
    each other, and are not evidence of anything beyond "this is what the corpus
    says". If the returned chunks do not state the fact being asked for, the
    corpus does not cover it — say so rather than filling the gap in from
    elsewhere.

    Retrieval is bge-small embeddings over 1000-character chunks, reranked by a
    ms-marco cross-encoder over the top 20 candidates. Local and free; no model
    is called.
    """
    hits = _index()["search"](query, k)
    if not hits:
        return f"No chunks matched {query!r}."
    return "\n\n".join(f"[{c['source']}]\n{c['text']}" for c in hits)


# The docstring above is the `description` field on the wire. That is the whole
# point of opening this file in the Inspector: what you read in tools/list is
# prose, written by whoever wrote the server, delivered into the client's model
# context at connect time and before any tool is called. Here that author is me.
# On a server from a public directory it is a stranger, which is why the spec
# tells clients to treat tool annotations as untrusted unless the server is.
#
# `experiment_server.py` is the same channel pointed at myself — see results.md.


# ---------------------------------------------------------------- resource (application-driven)

@mcp.resource(
    "corpus://manifest",
    name="corpus-manifest",
    title="Indexed documents",
    description=("The documents in the search index, with their sizes and chunk "
                 "counts, plus the retrieval config search_corpus runs."),
    mime_type="application/json",
)
def manifest() -> str:
    """What is actually in the index — not what is in the corpus directory.

    The distinction is the reason this is worth serving. `ls corpus/` is a
    directory listing; this is a report from the same objects `search_corpus`
    queries, so if the chunker dropped a file or the config drifted, the manifest
    says so and the tool's silence doesn't have to be trusted.

    A resource, not a tool, because the model does not decide it enters the
    context — the application does, by attaching it.
    """
    idx = _index()
    per_doc = {}
    for c in idx["chunks"]:
        per_doc[c["source"]] = per_doc.get(c["source"], 0) + 1
    return json.dumps({
        "corpus_dir": str(R.CORPUS_DIR.relative_to(HERE.parents[1])),
        "config": CONFIG,
        "n_documents": len(idx["docs"]),
        "n_chunks": len(idx["chunks"]),
        "documents": [
            {"source": d["source"], "chars": len(d["content"]),
             "chunks": per_doc.get(d["source"], 0)}
            for d in idx["docs"]
        ],
    }, indent=2)


# ---------------------------------------------------------------- prompt (user-controlled)

@mcp.prompt(
    name="cited",
    title="Answer with citations (L13 template)",
    description=("The L13 generation prompt: answer strictly from retrieved "
                 "context, treat that context as untrusted data, and attribute "
                 "every claim to the chunk it came from."),
)
def cited(question: str = "") -> list[UserMessage]:
    """Hand back SYSTEM_PROMPTS["cited"] — hardened + segregated + citation rules.

    Two things the port had to decide:

    Nonce. The template carries `{nonce}` placeholders, and the whole security
    property of S1-S3 is that the marker id is unpredictable and fresh per
    request. So one is minted here, per prompts/get, rather than baked in — a
    constant nonce shipped in a server's prompt would be a nonce the attacker can
    read off the wire and forge.

    Role. MCP prompt messages are only ever "user" or "assistant"; there is no
    system role in the primitive. So this arrives as a user message, and where it
    lands in the final prompt is the client's decision, not mine. Worth knowing
    before assuming a shipped "system prompt" is going to sit where you put it.
    """
    text = R.system_prompt("cited", R.new_nonce("cited"))
    if question:
        text += f"\n\nQuestion: {question}"
    return [UserMessage(text)]


if __name__ == "__main__":
    mcp.run(transport="stdio")
