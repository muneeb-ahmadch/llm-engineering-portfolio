# LLM Engineering Portfolio

Thirteen experiments on a single RAG pipeline, each one asking a question that could be
answered with a number, and each one recording the answer it actually got.

Every figure in this repository was produced by code in this repository, run against the
corpus in this repository, on the date printed beside it. Nothing is quoted from a lecture
slide, estimated, or rounded up. Where a result contradicted what I predicted, the
prediction is still in the repo next to the result.

**Read it as a site:** [muneeb-ahmadch.github.io/llm-engineering-portfolio](https://muneeb-ahmadch.github.io/llm-engineering-portfolio/)

**Muneeb Ahmad** · [github.com/muneeb-ahmadch](https://github.com/muneeb-ahmadch) · muneebahmad.ch1@gmail.com

---

## The short version

I built one evaluation harness ([`rag_eval.py`](projects/01-chunking/rag_eval.py), 2,187 lines)
and then spent thirteen sessions trying to break the pipeline it measures. The interesting
results are mostly the negative ones:

| Experiment | The question | What the numbers said |
|---|---|---|
| [01 Chunking](projects/01-chunking/) | Chunk size or encoder: which lever matters? | Encoder. `bge-small` over `minilm` bought **+0.128 MRR** (0.627 to 0.754) at zero extra cost. Halving chunk size **hurt both** encoders. |
| [05 Heading chunker](projects/05-heading-chunker/) | Does splitting on headings rescue a known miss? | No. The target chunk fell from full corpus rank **#8 to #67**. Cleaner chunks helped 66 competitors more than they helped the target. |
| [06 Reranking](projects/06-reranking/) | Can a cross encoder rescue the two misses? | One of two. LSTM flipped to a hit at pool 20. Overfitting stayed a miss at **every pool size from 5 to 30**, so it was a scoring decision, not a recall problem. |
| [07 Generation](projects/07-generation/) | What does an answer cost and how long does it take? | **$0.002174 per query, 2.1 s mean latency.** A live price check overruled my own memory, which had the model down as fiction. |
| [08 RAG vs fine-tune](projects/08-rag-vs-finetune/) | Which of the three tools fits which problem? | Four scenarios diagnosed on knowledge versus behaviour, with the break-even written out: ~$72.47/day saved means a small-model fine-tune pays back in days. |
| [09 Answer defense](projects/09-answer-defense/) | Will it hallucinate on questions the corpus cannot answer? | The prompt I had been shipping hallucinated on **15 of 15** trap runs. A better prompt alone took it to **0 of 15**, faithfulness 3.90 to 4.80, groundedness 66.9% to 95.6%. |
| [10 Abstention tracing](projects/10-abstention-trace/) | Can I see refusals in production telemetry? | Boolean score on every span. Traps **5/5**, golden **1/20**. The single false abstention traced back to one chunk in 734, so it was a chunking defect rather than a timid model. |
| [11 Agent loop](projects/11-agent-loop/) | Is a search loop worth the money? | **Four of five preregistered hypotheses were wrong.** The loop fired on 2 questions in 20, zero false fires, **1.32x cost**. Verdict: ship it behind a trigger, not by default. |
| [12 Prompt injection](projects/12-injection/) | What happens if someone writes into my corpus? | **1,820 bytes, 0.255% of the corpus, and the agent obeyed 8 of 18 runs.** My own faithfulness judge scored the most successful attack **5/5, fully grounded, three times out of three.** |
| [13 Citation validation](projects/13-citation-validation/) | Does forcing citations stop the attack? | No. It catches **none of the eight attacks that landed**, because the attacker's method is to make the citation true. Segregating the judge did work: correct answers wrongly punished went **4 to 0**. |
| [14 MCP server](projects/14-mcp/) | Can I serve the retriever over MCP? | Built to the 2026-07-28 spec and verified three ways, including Claude Code discovering and calling it unprompted. A poisoned tool description got **0/3 compliance but 1/3 contamination**. |
| [15 Stateful graphs](projects/15-langgraph/) | Does LangGraph durability survive a crash? | Only on a flag that is not the default. With `durability="async"`, a `kill -9` **lost a model call I had already paid for**. `sync` fixes it. |

The two experiments not in that table are [04 Answer quality](projects/04-answer-quality/),
a by-hand audit that overturned an earlier verdict of mine, and the numbering gaps, which
are sessions that produced no separate codebase.

---

## What this is evidence of

**Measurement before opinion.** Every design decision in the pipeline traces to a run.
The encoder choice, the chunk size, the rerank pool, the prompt, and the decision to gate
the agent loop behind a trigger are each attached to a table produced on this corpus.

**Adversarial thinking about my own work.** Experiments 12 through 14 are attacks on
systems I had just built and was pleased with. The most useful finding in the repository
is that the guardrail I trusted most, an LLM faithfulness judge, endorsed the attack that
beat it.

**Honest negative results.** Five of the thirteen sessions ended by disproving the thing I
set out to prove. Those write-ups are the longest ones, and the predictions that failed are
still committed next to them ([11](projects/11-agent-loop/hypothesis.md),
[13](projects/13-citation-validation/predictions.md), [15](projects/15-langgraph/predictions.md)).

**Production instincts.** Cost and latency measured per query, tracing wired into Langfuse
with custom scores, a CI gate that exits non-zero, unit tests on the deterministic
validator, and a threat model written before the attack code.

---

## The pipeline

```
corpus (21 markdown files, 712 KB)
  -> chunker            fixed 1000 / 50 overlap, or heading aware
  -> encoder            bge-small-en-v1.5, local, CPU, no API cost
  -> retrieval          cosine top-k over an in-memory index
  -> reranker           cross-encoder/ms-marco-MiniLM-L6-v2, pool 20
  -> context builder    per-request nonce delimiter around every chunk
  -> generator          gpt-5.6-luna, hardened system prompt
  -> agent loop         one search_corpus tool, MAX_STEPS 3, trigger gated
  -> judge              gpt-5.6-terra, segregated context
  -> validator          pure Python citation check, 19 unit tests
  -> tracing            Langfuse spans with abstained / needle_in_context scores
```

Shipping configuration, unchanged since experiment 06 because nothing since has beaten it:
`bge-small · chunk 1000 · fixed · top-5 · rerank pool=20 · hardened prompt`.

## The corpus and the golden set

21 markdown files of AI engineering lecture notes, 712,484 bytes. I chose them because I
had read all of them, which means I can write golden questions from memory and check
answers by eye. They also have heavy topic overlap between adjacent files, which makes
retrieval work harder than a corpus of unrelated documents would.

- `golden.jsonl`, 20 hand written questions with keyword relevance rules
- `golden-trap.jsonl`, 5 questions the corpus provably cannot answer, used to measure hallucination
- `check` validates both before any run, because a question matching zero chunks scores 0 forever and a question matching fifty chunks pins MRR near 1.0 while measuring nothing

## Running it

```bash
git clone https://github.com/muneeb-ahmadch/llm-engineering-portfolio
cd llm-engineering-portfolio/projects/01-chunking

uv venv --python 3.12 .venv          # torch has no 3.14 wheels
uv pip install --python .venv/bin/python -r ../../requirements.txt

.venv/bin/python rag_eval.py check                        # validate the golden set, $0
.venv/bin/python rag_eval.py run --encoder bge-small --chunk-size 1000 --top-k 5 -v
.venv/bin/python rag_eval.py sweep                        # the 2x2 matrix, $0
```

Everything through experiment 06 is local, CPU only, and costs nothing. Generation,
judging, and the agent loop need `OPENAI_API_KEY`; tracing needs Langfuse keys. See
[`.env.example`](.env.example). Total measured API spend across every experiment in this
repository is **$4.95**, summed from the totals each report records.

## Reading order

If you have five minutes, read [12 Prompt injection](projects/12-injection/results.md).
It contains the threat model, the attack, the defense, the cost of the defense, and the
attack that still beats it.

If you have twenty, add [09 Answer defense](projects/09-answer-defense/report.md) for how
the guardrail was built and [13 Citation validation](projects/13-citation-validation/results.md)
for why the obvious fix does not work.

The [docs/reference/](docs/reference/) directory holds thirteen pages compressing each topic to what
is worth remembering: chunking, embeddings and retrieval, RAG metrics, reranking, the
pipeline and its cost, answer defense, tracing and scores, agents versus workflows, prompt
injection and guardrails, orchestration and validation, MCP, stateful graphs, and the
RAG versus fine-tune versus prompt decision.

## A note on scope

This is applied LLM and RAG engineering, built on four years of Python, FastAPI, Docker and
CI/CD backend work. It is not deep learning research, reinforcement learning, or MLOps at
scale, and I have not claimed otherwise anywhere in it.

One piece is deliberately unfinished and labelled as such: task 2 of
[04 Answer quality](projects/04-answer-quality/), the false positive hunt, was left blank
rather than filled in after the fact.
