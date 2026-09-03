#!/usr/bin/env python
"""Zero-cost probes for the three implementation failures preregistered in
predictions.md. No API calls: these are the reducer and the checkpointer factory,
tested on the exact shapes my build uses.

    python probe_p0.py
"""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from langgraph.graph.message import add_messages                      # noqa: E402
from langgraph.checkpoint.sqlite import SqliteSaver                   # noqa: E402

# The four message shapes my graph actually puts in state, in the order it does.
ROLE_MSGS = [
    {"role": "system", "content": "You answer from the context."},
    {"role": "user", "content": "Context:\n[day16.md]\n...\n\nQuestion: What is overfitting?"},
]
REASONING_ITEM = {"type": "reasoning", "id": "rs_abc", "summary": []}
FUNCTION_CALL = {"type": "function_call", "id": "fc_abc", "call_id": "call_1",
                 "name": "search_corpus",
                 "arguments": '{"query": "overfitting", "k": 5}'}
FUNCTION_CALL_OUTPUT = {"type": "function_call_output", "call_id": "call_1",
                        "output": "[day22.md]\nOverfitting is..."}


def probe(label, fn):
    try:
        out = fn()
    except Exception as e:                    # noqa: BLE001: the outcome IS the result
        print(f"  {label}\n     RAISED {type(e).__name__}: {str(e)[:220]}")
        return None
    print(f"  {label}\n     ok -> {out}")
    return out


print("\nP0, does add_messages accept the /v1/responses input items?")
probe("1. role dicts (system + user)",
      lambda: [type(m).__name__ for m in add_messages([], ROLE_MSGS)])
probe("2. a reasoning item fed back verbatim",
      lambda: [type(m).__name__ for m in add_messages([], [REASONING_ITEM])])
probe("3. a function_call item fed back verbatim",
      lambda: [type(m).__name__ for m in add_messages([], [FUNCTION_CALL])])
probe("4. a function_call_output item",
      lambda: [type(m).__name__ for m in add_messages([], [FUNCTION_CALL_OUTPUT])])
probe("5. what a coerced role dict looks like going BACK to the API",
      lambda: add_messages([], ROLE_MSGS)[0])

print("\nP0#2, SqliteSaver.from_conn_string() without a `with` block")
probe("6. type of the returned object",
      lambda: type(SqliteSaver.from_conn_string(str(Path(__file__).parent / "probe.sqlite"))).__name__)
probe("7. does it have .put (the checkpointer protocol)?",
      lambda: hasattr(SqliteSaver.from_conn_string(str(Path(__file__).parent / "probe.sqlite")), "put"))
probe("8. explicit sqlite3 connection instead",
      lambda: hasattr(SqliteSaver(sqlite3.connect(":memory:", check_same_thread=False)), "put"))
print()
