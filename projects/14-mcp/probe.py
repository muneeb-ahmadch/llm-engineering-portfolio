#!/usr/bin/env python
"""Talk to server.py the way a client does, raw JSON-RPC 2.0 over stdio.

The Inspector does exactly this behind a web UI. Doing it in 60 lines here is
worth the trouble twice over: it is the artefact that can be checked into the
repo and re-run, and it makes "discovery" concrete, nothing below knows a
single thing about `search_corpus` until the server says the word on line 2.

    python probe.py            # newline-delimited JSON-RPC, no SDK on this side

Every request carries `_meta` with the protocol version and client identity,
because 2026-07-28 is stateless: there is no initialize handshake establishing a
session, so each frame has to say who is asking and which version they speak.
"""
import json
import select
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PY = HERE.parents[0] / "lesson-001-chunking" / ".venv" / "bin" / "python"
PROTOCOL = "2026-07-28"

META = {
    "io.modelcontextprotocol/protocolVersion": PROTOCOL,
    "io.modelcontextprotocol/clientInfo": {"name": "l14-probe", "version": "0.1.0"},
    "io.modelcontextprotocol/clientCapabilities": {},
}


def call(proc, i, method, params=None, timeout=90):
    """One request, one response. `timeout` because the first version of this file
    had none, and a stalled server looked exactly like a slow one for five
    minutes before anything said so."""
    frame = {"jsonrpc": "2.0", "id": i, "method": method,
             "params": {**(params or {}), "_meta": META}}
    print(f"--> {method}", flush=True)
    proc.stdin.write(json.dumps(frame) + "\n")
    proc.stdin.flush()
    deadline = time.time() + timeout
    while True:
        if not select.select([proc.stdout], [], [], max(0, deadline - time.time()))[0]:
            proc.kill()
            sys.exit(f"TIMEOUT, no response to {method} in {timeout}s")
        line = proc.stdout.readline()
        if not line:
            sys.exit("server closed stdout, check its stderr above")
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            # The stdio failure mode worth naming: anything the server prints to
            # stdout lands in the frame stream and is not JSON.
            sys.exit(f"NON-JSON ON STDOUT, something printed past the guard:\n{line!r}")
        if msg.get("id") == i:                 # skip notifications
            if "error" in msg:
                sys.exit(f"{method} -> ERROR {json.dumps(msg['error'], indent=2)}")
            return msg["result"]


def show(title, obj, limit=1400):
    body = json.dumps(obj, indent=2)
    print(f"\n=== {title} " + "=" * max(0, 60 - len(title)))
    print(body if len(body) <= limit else body[:limit] + f"\n... [{len(body)} chars]",
          flush=True)


if __name__ == "__main__":
    proc = subprocess.Popen([str(PY), str(HERE / "server.py")], text=True,
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    show("server/discover", call(proc, 1, "server/discover"))

    tools = call(proc, 2, "tools/list")
    show("tools/list", tools)
    for t in tools["tools"]:
        print(f"\n  --- description field for {t['name']!r}, verbatim off the wire ---")
        print("  " + t["description"].replace("\n", "\n  "))

    show("tools/call search_corpus",
         call(proc, 3, "tools/call",
              {"name": "search_corpus",
               "arguments": {"query": "a cross-encoder reranker scores query-chunk pairs",
                             "k": 2}}))

    show("resources/list", call(proc, 4, "resources/list"))
    show("resources/read corpus://manifest",
         call(proc, 5, "resources/read", {"uri": "corpus://manifest"}))

    show("prompts/list", call(proc, 6, "prompts/list"))
    show("prompts/get cited",
         call(proc, 7, "prompts/get",
              {"name": "cited", "arguments": {"question": "What is MRR?"}}))

    proc.stdin.close()
    proc.wait(timeout=10)
    print("\nOK, three primitives answered over stdio.")
