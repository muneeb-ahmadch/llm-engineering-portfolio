# Experiment 10 report: catch abstention in the trace

**Run date:** 2026-07-28 · **Config:** `bge-small · chunk 1000 · fixed · top-5 · rerank pool=20` (L06 best, unchanged) · **Generator:** `gpt-5.6-luna` · **Prompt:** `hardened` (the L09 fix, which is what ships) · **Sets:** golden 20 + trap 5 · **Spend:** $0.040330

The two commands the whole report is built on:

```bash
python rag_eval.py run --encoder bge-small --chunk-size 1000 --top-k 5 \
    --rerank --pool 20 --generate --trace --prompt hardened
python rag_eval.py run ... --generate --trace --prompt hardened --trap
```

**Traces:**
[golden `61dc2ad0`](https://cloud.langfuse.com/project/cmrvxaxqt01kxad0jvhtb8dlm/traces/61dc2ad0214a6e9cc940eb455a70d493) ·
[trap `55fbb76b`](https://cloud.langfuse.com/project/cmrvxaxqt01kxad0jvhtb8dlm/traces/55fbb76b4f94a42fbccdbb15fd6b891d)

---

## Step 1: the score, emitted

Five lines into `evaluate()` ([rag_eval.py:694](../01-chunking/rag_eval.py#L694)), inside the existing `if qspan is not None` guard so untraced runs are byte-for-byte unchanged:

```python
if qspan is not None:
    qspan.update(output=gen["answer"])
    # Control #6: the abstention verdict rides the trace as a
    # SCORE, not a tag. A score is the only primitive that
    # averages, and the number worth watching is the rate.
    qspan.score(name="abstained",
                value=1 if gen["abstained"] else 0,
                data_type="BOOLEAN",
                comment=test["question"])
```

`gen["abstained"]` was already being computed one block up (line 686) and thrown away after the local summary. It now leaves the process.

**Signature verified against the installed SDK, not from memory:** `langfuse 4.14.1`, `LangfuseSpan.score(*, name, value, score_id, data_type, comment, config_id, timestamp, metadata)`. The `score` / `score_trace` pair is exactly the batch-vs-live distinction: this harness is one trace with a span per question, so the verdict hangs on the **span**. A live app is one trace per request, so it would hang on the **trace**.

Why a score and not a tag: I want a **rate**, and a rate is a mean. Tags filter, metadata searches; neither averages. The primitive was picked by the number I need to read later.

---

## Steps 2 and 3: read as a diagnosis

Both runs completed and the scores were read **back out of Langfuse** (via `GET /api/public/traces/{id}`) rather than trusted from local memory: 20 scores on the golden trace, 5 on the trap trace, all `name=abstained`, `data_type=BOOLEAN`.

| Set | mean `abstained` | Verdict |
|---|---|---|
| **trap** (5) | **1.000** (5/5) | Every trap scored 1. **No landmine.** Nothing answered that shouldn't have been. |
| **golden** (20) | **0.050** (1/20) | One false abstention. **This is the finding.** |

### The disagreement: `abstained` vs MRR, in both directions

Retrieval has two `rr = 0` misses on this golden set, unchanged since L06, at `MRR 0.785` and `misses 2/20`. The abstention score disagrees with **both** of them, and in opposite directions. That is the whole value of the number:

| Question | `rr` | `abstained` | What actually happened |
|---|---|---|---|
| *What is overfitting in a neural network?* | **0** | **1** | Retrieval returned `day26-deep-learning.md` **three times in the top-5**, the *right file*, the only file in the corpus containing "validation accuracy". It never surfaced the *right chunk*. Model got on-topic context with the asked-for fact missing, and declined. |
| *What does reranking do in a RAG pipeline?* | **0** | **0** | Retrieval returned `day16-rag-evals.md` ×5 and the model answered: *"Reranking reorders retrieved chunks by relevance, moving the most useful information to the top before generating an answer."* Correct and grounded. |

Two conclusions the checkmark alone would have hidden:

1. **The false abstention is not a timid model. It is a chunking defect.** `check` says exactly **1 chunk of 734** holds all three keywords (`Overfitting`, `Training accuracy`, `Validation accuracy`); the concept appears 32× across 6 files. The evidence exists, is in the retrieved *document*, and is not in the retrieved *window*. The hardened prompt's rule 2, "check that the specific fact asked for is actually stated in the context", fired correctly on starved context. **The system behaved right; retrieval fed it wrong.** This is the same `overfitting` question that survived the L05 header-chunker and the L06 rerank as a MISS. It is now a work-queue item with a name, at **rung 2**.
2. **The other `rr = 0` is a metric artifact, not a failure.** Strict `all`-keyword matching scores it a miss; the answer is fine. L07 found "retrieval-miss ≠ answer-miss" by reading answers by hand. The abstention score now says it **automatically, on every run**: a question that scores `rr=0, abstained=0` and reads correctly is my ruler being bent, not my system being broken.

### One honest discrepancy with L09

L09 measured abstention on the golden set at **0.0%** with this same hardened prompt. Today: **5.0%**. Same config, same corpus, same prompt. The generator runs at `temp=default(1)` (luna refuses `temperature=0`, probed at startup), so **the overfitting question is sitting on the decision boundary**, starved enough that abstain-vs-answer is a coin the model flips. That is not noise to wave away; it is the measurement telling me the band has to be wide enough to contain a single question flipping, on n=20.

---

## Step 4: the rate, and the band

> **Abstention rate (golden set) = mean `abstained` = 0.050 (1/20).**

**The band I'd alert outside of: `0.00 < mean abstained ≤ 0.10`, on a rolling window of 100 live questions.**

Tied to the L09 numbers, both edges are a real failure and not a round number:

| Edge | Value | Why that number | What it means |
|---|---|---|---|
| **upper** | **0.10** | L09/L06 retrieval misses = **2/20 = 10%** | A legitimate abstention is downstream of a retrieval miss. Abstention **above** the miss rate means declines the retrieval can't account for, so either the prompt has gone timid or retrieval degraded past its measured floor. |
| **lower** | **> 0.00** on a window containing traps | L09 trap baseline = **5/5 hallucinations** on the un-hardened prompt | A rate of exactly 0 across a window that contains unanswerable questions is the L09 failure mode returning: the sentinel broke, the prompt regressed, or a swapped model started answering things it shouldn't. **Zero is not a pass.** |
| **trap gate** | **must stay 1.000** | L09 gate rule: hallucinations must be 0 | Unchanged from CI. Any trap scoring 0 fails the commit. |

Today's 0.050 sits mid-band, and its single unit is explained (a chunk-level retrieval defect). Nothing to alert on.

**The caveat that keeps this honest:** 0.050 is measured on *my exam*, 20 questions I wrote. It is the **baseline**, not the production rate. The production rate is what this score will accumulate on traffic nobody wrote a test for, and until real traffic flows the band above is a threshold waiting for its population. That distinction is the entire reason this lesson exists separately from L09: **CI tests the questions I wrote; the trace watches the questions I didn't.**

---

## Step 5: the (c), automatic

> This production abstention rate would justify climbing to fine-tune when **mean `abstained` stays above 0.10 across a rolling 100-question window for 7 consecutive days on live traffic, measured after the L09 `hardened` prompt**, and until then it tells me to stay at prompt+RAG because **it reads 0.050, and 1 of that 1 abstention resolves to a rung-2 retrieval defect (1 chunk of 734 holds the evidence, never retrieved): rung-3 behaviour defects = 0**.

Both blanks are numbers. Rung 3 is unearned, and I can now point at the exact figure that would earn it.

---

## What this unlocked, concretely

- **Control #6 is real.** Filter the trace to `abstained = 1` → the corpus gaps, by name. Today that list has one entry and it is actionable: re-chunk `day26-deep-learning.md` so the overfitting definition and the train/validation accuracy gap land in one window.
- **A regression CI cannot see.** Swap the generator and the golden-set gate stays green while the live abstention rate collapses toward 0 on traffic the trap set never covered. The commit gate would not blink. The trace would.
- **The next identical call.** `qspan.score(name="faithfulness", value=<judge 1-5>, data_type="NUMERIC")`, the L09 judge's verdict on the same span, same method, one type change.

---

## Files

| File | What |
|---|---|
| [`../01-chunking/rag_eval.py`](../01-chunking/rag_eval.py#L694) | the 5-line score emission in `evaluate()` |
| `read_scores.py` | reads `abstained` back out of Langfuse per question, the diagnosis view, and the proof the score left the process |
