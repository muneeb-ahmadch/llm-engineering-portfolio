# Experiment 07 report: retrieve, then answer (cost and latency)

**Run date:** 2026-07-22 · **Config:** `bge-small · chunk 1000 · fixed · top-5 · rerank pool=20` (Lesson 06 best) **+ `--generate`** · **Model:** `gpt-5.6-luna` · **Golden set:** 20 questions

The one-line command this whole report is built on:

```bash
python rag_eval.py run --encoder bge-small --chunk-size 1000 --chunker fixed \
    --top-k 5 --rerank --pool 20 --generate
```

(`run_generation.py` in this folder calls the identical `evaluate()` and dumps `results.json`; the numbers below come from that file.)

---

## Step 1: the price, verified before a line of code (the actual point of the lesson)

| Model | Input / 1M | Cached input / 1M | Output / 1M | Source | Date |
|---|---|---|---|---|---|
| **gpt-5.6-luna** | **$1.00** | $0.10 | **$6.00** | [OpenAI pricing page](https://platform.openai.com/docs/pricing) → `developers.openai.com/api/docs/pricing` | 2026-07-22 |

Verified **twice, independently**: once by hand off the pricing page, once by a live `WebFetch` of the same page today. The two agreed to the cent.

**A mistake I made, kept in on purpose because it *is* the lesson.** My reference knowledge stops in January 2026, so when I first saw `gpt-5.6-luna` I asserted it was course fiction, a made-up model. It is not. Listing the models the key can actually reach (`client.models.list()`) returned `gpt-5.6-luna`, `gpt-5.5`, `gpt-5.4-mini`, and dated variants like `gpt-5.4-2026-03-05`, all real releases that postdate that memory. **The live check overruled memory, exactly as the lesson demands.** Pricing and model availability are the two facts that must never be answered from memory; I did, and I was wrong, and the two-second verification caught it. That is the whole muscle this assignment is training.

---

## Steps 2 and 3: what got wired into `rag_eval.py`

All changes are additive; with `--generate` off, the harness is byte-for-byte the retrieval-only tool it was through Lesson 06, and still costs $0.

| Piece | What it does |
|---|---|
| `PRICES` dict | Verified USD-per-1M rates, dated, with the source URL in a comment. **`--model` is restricted to keys in this dict**, so the code physically cannot price a model whose rate has not been recorded. That's the "no remembered numbers" rule enforced in code, not just prose. |
| `SYSTEM_PROMPT` + `generate_answer()` | Builds the `system` + `user` messages in the exact shape of Day 16's `answer_question()`: a grounding system role, then `Context:\n…\n\nQuestion: …`. Reuses the **same `retrieved` top-k** the metrics already computed (no second retrieval), and times **only** the `client.chat.completions.create()` call with `time.perf_counter()`. |
| `compute_cost()` | `uncached_input×$1 + cached_input×$0.10 + output×$6`, per 1M. Reduces to the lesson's `prompt×input + completion×output` whenever nothing is cached, which was the case for all 20, since each question carries a different context and the prompt prefix never repeats enough for OpenAI's cache to engage. |
| `evaluate()` | Now returns `total_cost`, `cost_per_query`, `mean_latency_ms`, `max_latency_ms` (plus token totals and the per-question answer/latency/cost) whenever `generate=True`. |
| `cmd_run` | Prints the cost/latency block under the existing MRR / coverage / misses summary. |

### Two live API surprises worth writing down

1. **`gpt-5.6-luna` rejects `temperature=0.0`**, returning `"Only the default (1) value is supported."` Day 16 hardcoded `temperature=0.0` for determinism; the entire GPT-5 tier refuses it. `generate_answer()` handles this generally: it tries `0.0`, and on that specific error flips a per-run flag and retries without it, so the harness stays correct whether `--model` is `gpt-5.6-luna` (temp=1) or an older `gpt-4o-mini` (temp=0.0). The practical consequence: **generated answers are non-deterministic**, which matters for the CI decision below.
2. **A name collision I introduced and caught.** My new `model` parameter (generation model) clashed with `evaluate()`'s local `model = get_model(encoder)` (the SentenceTransformer): the encoder object shadowed the string and `PRICES[model]` blew up with a `KeyError` whose "key" was a whole SentenceTransformer repr. Renamed the local to `encoder_model`. First run failed; caught and fixed before any of the 20 paid calls went out.

---

## Step 4: results (20 questions, one run)

```
MRR              0.785          ← identical to Lesson 06: generation does not touch retrieval
Keyword coverage 93.3%
Misses           2/20

--- generation · gpt-5.6-luna ($1.00/1M in · $6.00/1M out) · temp=default(1) ---
Total cost       $0.043471     (25,627 in + 2,974 out tokens)
Cost / query     $0.002174
Mean latency     2,101 ms
Max latency      4,344 ms
Wall clock       63.4 s for the whole 20-question loop (serial)
```

**Cost breakdown:** input is **59%** of the bill ($0.0256), output **41%** ($0.0178). Even though output costs 6× per token, there were ~8.6× fewer output tokens, so the long retrieved contexts (≈1,280 prompt tokens each) dominate. To cut cost, trimming context (fewer or smaller chunks) beats capping answer length.

**Latency distribution (ms):** min 955 · median 1,916 · mean 2,101 · p95 3,560 · **max 4,344**. The max is 2.1× the mean, and it tracks output length: the slowest question ("chat vs reasoning models", 216 output tokens) generated the most text, and the fastest (955 ms, 15 output tokens) barely wrote a sentence. **Generation latency scales with tokens produced, and the tail is wide.**

### The finding that matters most: retrieval misses ≠ answer misses

Both retrieval "misses" got **correct, complete answers:**

> **"What does reranking do in a RAG pipeline?"** *(retrieval MISS)* → *"Reranking reorders the initially retrieved document chunks so the most relevant ones appear first… A retriever quickly finds candidates using embeddings… a reranker evaluates them more carefully, often using a cross-encoder…"* Correct.

> **"What is overfitting in a neural network?"** *(retrieval MISS)* → *"Overfitting occurs when a neural network learns the training data too closely, including noise, rather than patterns that generalize… signs: high training accuracy, lower validation accuracy, a growing gap…"* Textbook-correct.

These are misses only under the **keyword proxy** (the required keywords weren't all present in the top-5). The LLM answered them anyway, **from its own parametric knowledge**, because these are generic ML concepts luna already knows cold. Two things follow, and they're the interview-grade insight of the day:

- **This is exactly why Lesson 04 kept the two metrics separate.** MRR measures *did retrieval surface the right chunk*; it is blind to the answer. Answer quality here is *higher* than MRR suggests, but only because the model happens to know the subject. On a **private corpus** (InsureElm's proprietary policies, or my own documents) the model has no parametric fallback: a retrieval miss becomes an *abstention or a hallucination*, and it would never be visible from watching answer quality alone. Keep measuring retrieval on its own.
- **Generation adds a failure mode retrieval-only never had: hallucination.** A quick out-of-band probe, *"What is the capital of a country never mentioned in the context?"* with an LSTM chunk as context, got **"Paris, the capital of France"** instead of the instructed *"I don't have that information."* The grounding instruction is not a guarantee; the model will reach past the context when it thinks it knows. That is the risk being bought by letting an LLM into the loop.

---

## The verdict: every commit, or nightly?

**Nightly (or on-demand), not every commit, and the deciding factor is *not* cost.** At $0.043 per 20-question run, even 200 commits a day is under $9/day; money is a rounding error here. The real costs are **latency**, at 63 s of serial wall-clock added to every push and growing to ~2.5 min once the golden set hits the roadmap's 50 questions, and **non-determinism**: at forced `temperature=1`, generated answers vary run-to-run, so any commit gate on answer *content* would flake. So the pipeline splits by what each half actually needs. The **retrieval metrics (MRR / coverage / misses) are free, deterministic, and cached-fast, so those run on every commit**; the **paid, slow, non-deterministic generation and cost/latency pass runs nightly**, where a 2-minute run and some answer variance are fine.

---

## Step 5: Langfuse tracing (built, run, and self-audited)

This went past the "one sentence" and was actually wired, following the official [Langfuse agent skill](https://github.com/langfuse/skills), whose first rule is *never implement from memory, fetch current docs*. So I pulled the OpenAI-Python integration and best-practices pages fresh and introspected the installed SDK, `langfuse==4.14.1`, directly.

**What got added** (all behind a new `--trace` flag; a no-op when off):

```bash
python rag_eval.py run --encoder bge-small --chunk-size 1000 --chunker fixed \
    --top-k 5 --rerank --pool 20 --generate --trace
```

- The OpenAI client is swapped for Langfuse's **drop-in wrapper** (`from langfuse.openai import OpenAI`) only when `--trace` is set, so each `chat.completions.create` is auto-captured (model, prompt, completion, token usage, latency) with zero extra call code. The skill is explicit: *prefer the framework integration over manual instrumentation.*
- The run is shaped like the **RAG pipeline it is**, not one flat blob. Per question: an `answer-question` span containing a `retrieve-context` **retriever** observation and a `generate-answer` **generation**, all under one `rag-eval` root span. Verb-first names, no model names or dynamic values in the names (those go in tags and metadata), straight from the best-practices page.
- Trace-level **tags** (`bge-small`, `gpt-5.6-luna`, `lesson-07`, `rerank`) and `environment=development` via `propagate_attributes`, so runs are filterable in the dashboard.
- Our authoritative, verified-price **cost/latency totals are stamped on the root span** metadata.
- `langfuse.flush()` before exit (the skill's #1 "common mistake": traces silently never send without it).

**A design detail worth noting.** `gpt-5.6-luna` rejects `temperature=0.0`. The retrieval-only fallback (try 0.0, catch, retry) would record a *failed-then-retried* generation on question 1 inside the trace. So when tracing, the harness first settles temperature support with **one tiny untraced probe** (a plain, un-wrapped client), keeping the trace free of spurious ERROR spans.

**Self-audit (the skill requires running the path and fetching the trace back, not just compiling).** Fetched the live 20-question trace via the SDK and checked it against the baseline table:

| Baseline requirement | Result |
|---|---|
| 61 observations, correct shape | ✓ 1 root + 20×(span → retriever + generation) |
| Every level DEFAULT (no errors) | ✓ zero ERROR spans |
| Model name + token usage on generations | ✓ all 20, incl. `reasoning_tokens` and `cached_tokens` |
| Descriptive verb-first names | ✓ `retrieve-context`, `generate-answer` |
| Span hierarchy + correct types | ✓ RETRIEVER / GENERATION typed correctly |
| Trace input/output readable | ✓ root in = config, out = `{mrr, misses}`; per-Q in = question, out = answer |
| Tags / environment | ✓ set at creation |

**Live trace:** https://cloud.langfuse.com/project/cmrvxaxqt01kxad0jvhtb8dlm/traces/c77719c54845808a655a5acb6de2f715

### Two things the trace itself taught

1. **Prompt caching is real and visible.** The traced run cost **$0.020893**, *half* the cold run's $0.043471 on the same 25,627 input tokens, because re-running identical prompts inside OpenAI's cache window billed much of the input at the cached $0.10/1M instead of $1.00/1M (`input_cached_tokens` shows right in the trace). This is why `compute_cost()` is cache-aware, and it is a genuine production cost lever: cache stable prefixes (system prompt, shared context).
2. **Dashboard cost ≠ verified cost, for a brand-new model.** Langfuse auto-computes a per-generation cost from its own model-price table; for a model as new as `gpt-5.6-luna` that estimate will not match a vendor-verified, cache-aware number. The fix is either a custom model-price entry in Langfuse settings, or what I did here: treat the **verified-price total stamped on the root span as source of truth** and use the dashboard for structure and latency exploration.

---

## Files

- [`../01-chunking/rag_eval.py`](../01-chunking/rag_eval.py), the harness, now with `--generate` / `--model` / `--trace`
- [`run_generation.py`](run_generation.py), the driver that produced the cold-run numbers
- [`results.json`](results.json), full per-question data from the cold run (answers, tokens, latency, cost)
- `../01-chunking/.env`, gitignored, holds `OPENAI_API_KEY` + `LANGFUSE_*` keys
- deps added to the venv: `openai`, `python-dotenv`, `langfuse`

**Headline numbers in this report are the cold run** (`results.json`, $0.043471). The `--trace` run cost less ($0.020893) purely because of prompt caching on repeated identical prompts, as covered in Step 5.
