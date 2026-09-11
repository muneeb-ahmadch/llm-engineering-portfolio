# On the corpus

`projects/01-chunking/corpus/` holds 21 markdown files, 712,484 bytes. They are the lecture
notes distributed with an AI engineering course I took in 2026, reproduced here byte for
byte. They are not my writing and I am not presenting them as such.

They are here for one reason. A retrieval evaluation means nothing without the documents it
retrieves from, and every figure in this repository was measured against these exact bytes.
Replace the corpus and none of the numbers hold. I kept these files rather than substitute a
public corpus because I had read all 21 closely enough to write golden questions from memory
and check the answers by eye, which is the part of a golden set a machine cannot do for you.

The MIT license above covers my code and my write-ups. It does not extend to these notes.
They are included so the experiments reproduce, not for reuse, and I will take them down on
request from whoever holds the rights.

`projects/12-injection/corpus-poisoned/` and `corpus-poisoned-ceiling/` are copies of that
corpus with six and one adversarial paragraphs added respectively. They exist to be attacked
and are clearly marked in [`projects/12-injection/injections.jsonl`](projects/12-injection/injections.jsonl).
Do not use them as reference material: several of their sentences are deliberate falsehoods.

# On the em dashes that remain

My own written prose here avoids em dashes, and the reports and this file follow that.
They have not been stripped everywhere, and the places they remain fall into five
groups, because in each one the bytes are evidence rather than style:

- **System prompts, segregation rules and judge rules** in
  `projects/01-chunking/rag_eval.py`. Those exact bytes produced every measured number in
  these reports. Editing them for style would silently invalidate the results.
- **Tool descriptions** in `projects/14-mcp/server.py` and `experiment_server.py`. Under
  MCP a tool's docstring *is* the description sent to the model, and experiment 14 is an
  experiment whose two arms differ by exactly one tool description.
- **Attack payloads** in `projects/12-injection/injections.jsonl` and
  `injections-ceiling.jsonl`. The payload is the independent variable.
- **Captured model output**: the `.log` files under `projects/*/results/`, and the
  quotations in `projects/14-mcp/results.md`. Those are transcripts, not prose.
- **The golden set and the trap set**, `projects/01-chunking/golden.jsonl` and
  `golden-trap.jsonl`. The commentary in those two files predates the rule. I would
  rather leave files the harness reads alone than edit them for punctuation.

The count is **105**, outside the corpus copies, the generated `docs/` and
`site/snapshots/`, and this file. As of 9 September 2026. An earlier version of this file
said thirty, which was wrong: it named three of the five groups and undercounted each of
those by one. Reproduce the real number with:

```bash
git ls-files | grep -v -e '^docs/' -e '^site/snapshots/' -e corpus -e NOTICE.md \
  | xargs grep -o '—' | wc -l
```

The lecture notes under `corpus/` are untouched for the same reason as everything above.
They are the input the measurements were taken on, and their byte count (712,484) and
chunk boundaries are load-bearing throughout.
