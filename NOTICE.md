# On the corpus

`projects/01-chunking/corpus/` holds 21 markdown files of my own study notes, written up
from an AI engineering course I took in 2026. They are here because a retrieval evaluation
is meaningless without the documents it retrieves from, and because I needed a corpus I had
actually read well enough to write golden questions from memory.

The MIT license above covers the code and the write-ups. The notes are included so the
experiments reproduce, not as course material for redistribution.

`projects/12-injection/corpus-poisoned/` and `corpus-poisoned-ceiling/` are copies of that
corpus with six and one adversarial paragraphs added respectively. They exist to be attacked
and are clearly marked in [`projects/12-injection/injections.jsonl`](projects/12-injection/injections.jsonl).
Do not use them as reference material: several of their sentences are deliberate falsehoods.

# On the em dashes that remain

The prose in this repository does not use em dashes. Thirty of them survive, and every
one is deliberate, because changing them would change the evidence:

- **9 in `projects/01-chunking/rag_eval.py`** sit inside the system prompts, the
  segregation rules and the judge rules. Those exact bytes produced every measured
  number in these reports. Editing them for style would silently invalidate the results.
- **19 in `projects/14-mcp/server.py` and `experiment_server.py`** are tool descriptions.
  Under MCP a tool's docstring *is* the description sent to the model, and experiment 14
  is an experiment whose two arms differ by exactly one tool description.
- **2 in `projects/14-mcp/results.md`** are verbatim quotations: the wire payload as it
  appeared in the `tools/list` response, and what the model said back.

The lecture notes under `corpus/` are untouched for the same reason. They are the input
the measurements were taken on, and their byte count (712,484) and chunk boundaries are
load-bearing throughout.
