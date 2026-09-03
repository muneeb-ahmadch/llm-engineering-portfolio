# Run report: the first full sweep, 2026-07-18

**Date:** 2026-07-18
**What this is:** A step-by-step log of the first full run of the harness, after the
golden set (`golden.jsonl`) reached all 20 questions. Every command I ran, exactly what
came back, and what each number meant. Kept verbatim rather than tidied up afterwards.

---

## The 30-second summary

I ran the three commands the README tells you to run, in order:

1. `check` → validate the questions. **All 20 passed. Golden set looks sound.**
2. `run` → evaluate one config to prove the loop works. **MRR 0.627.**
3. `sweep` → the 2×2 experiment (2 chunk sizes × 2 encoders). **Best = `bge-small` at
   chunk 1000, MRR 0.754.** It wrote `results/results.md` for you.

Nothing errored. The one thing still left for *you* is the "What I conclude" section at
the bottom of `results/results.md`, the assignment deliberately leaves that blank, and
I did not fill it in (more on that at the very end).

---

## Step 0: Check the Python environment

Before running anything, I confirmed the pre-built virtual environment is really Python
3.12 (the README warns that torch has no 3.14 wheels yet, so the wrong Python would
break the imports).

**Command:**
```bash
.venv/bin/python --version
```

**Output:**
```
Python 3.12.12
```

**What it means:** Good: the `.venv` is Python 3.12 as expected. Every command below
uses `.venv/bin/python` (the Python *inside* the virtual environment) rather than the
system `python`, so all the installed libraries like `sentence-transformers` are
available. That's why nothing needed installing.

---

## Step 1: `check`: validate the golden set

This is the step the README says people skip and shouldn't. It doesn't do any retrieval
or scoring: it just reads your 20 questions and, for each one, counts how many chunks
of the corpus contain your keywords. It's catching two silent failures:

- **Zero matching chunks** → the question can never score, no matter how good retrieval
  gets. A broken question, not a finding.
- **Too many matching chunks (>25)** → the keywords are so generic that everything looks
  "relevant" and the score pins near-perfect. Looks great, measures nothing.

**Command:**
```bash
.venv/bin/python rag_eval.py check
```

**Output:**
```
20 questions · 734 chunks @ 1000 chars

  [ok] What does NDCG measure that MRR does not?
       all:   1  any:   7
  [ok] How is IoU calculated for a predicted bounding box?
       all:   2  any:  55
  [ok] What does LoRA stand for?
       all:   1  any:  51
  [ok] Which tool in the LangChain ecosystem handles monitoring a
       all:   1  any:  35
  [ok] Why is RAG optimization described as whack-a-mole?
       all:   1  any:  21
  [ok] What is the Chat Completions API?
       all:   1  any: 158
  [ok] How are chat models different from reasoning models?
       all:   3  any: 215
  [ok] How are LangChain and LiteLLM different?
       all:   5  any:  13
  [ok] What is prompt caching?
       all:   2  any:  57
  [ok] What is computer vision?
       all:   1  any: 153
  [ok] How does RecursiveCharacterTextSplitter differ from a plai
       all:   1  any:   7
  [ok] How does an LSTM solve the vanishing gradient problem?
       all:   1  any:   6
  [ok] What problem do skip connections in ResNet solve?
       all:   5  any:  13
  [ok] What does mAP measure in object detection?
       all:   2  any:  27
  [ok] What does reranking do in a RAG pipeline?
       all:   2  any:  20
  [ok] What is Gradio used for?
       all:   3  any: 292
  [ok] What is Async IO in Python?
       all:   1  any:  14
  [ok] What is overfitting in a neural network?
       all:   1  any:  27
  [ok] What is backpropagation?
       all:   4  any:  46
  [ok] What is Non-Maximum Suppression in object detection?
       all:   3  any:  11

Golden set looks sound. Run the sweep.
```

**What it means:**
- **`all` column** = how many chunks contain *every* keyword for that question (this is
  the scoring rule the harness uses by default). **`any` column** = how many contain *at
  least one* keyword (the looser Day-16 rule, shown only for comparison).
- Every line is `[ok]`. No question was flagged. That means, under the strict `all` rule,
  each question has between **1 and 5** chunks that count as the "right" answer, enough
  to be hittable, few enough to actually discriminate. None hit the ">25 = too generic"
  warning.
- Notice how large some `any` numbers are, "What is Gradio used for?" matches **292**
  chunks under `any`. If the harness scored with `any`, that question would be trivially
  "solved" and tell you nothing. This is exactly why the harness defaults to `all`, and
  why your keyword choices pass: you picked 2–3 terms that co-occur in one real passage.
- Final line: **"Golden set looks sound. Run the sweep."**, the green light. (Earlier,
  with only 5 questions, it would have nagged you to write the other 15. With 20, that
  note is gone.)

---

## Step 2: `run`: evaluate one configuration (with details)

Before the full experiment, I ran a single config with the `-v` (verbose) flag so you
can see the per-question breakdown, not just the summary number. This is the exact
command from the README. It downloads the `minilm` model on first use (it was already
cached here, so it was instant).

**Command:**
```bash
.venv/bin/python rag_eval.py run --encoder minilm --chunk-size 1000 --top-k 5 -v
```

**Output:**
```
   MISS  cov 50%  What does LoRA stand for?
   MISS  cov 50%  Why is RAG optimization described as whack-a-mole?
   MISS  cov 0%   What does reranking do in a RAG pipeline?
   MISS  cov 33%  What is overfitting in a neural network?
     @5  cov 100%  What is Non-Maximum Suppression in object detection?
     @3  cov 100%  What does NDCG measure that MRR does not?
     @2  cov 100%  How are chat models different from reasoning models?
     @2  cov 100%  How are LangChain and LiteLLM different?
     @2  cov 100%  How does RecursiveCharacterTextSplitter differ from a plain ch
     @2  cov 100%  How does an LSTM solve the vanishing gradient problem?
     @1  cov 100%  How is IoU calculated for a predicted bounding box?
     @1  cov 100%  Which tool in the LangChain ecosystem handles monitoring and d
     @1  cov 100%  What is the Chat Completions API?
     @1  cov 100%  What is prompt caching?
     @1  cov 100%  What is computer vision?
     @1  cov 100%  What problem do skip connections in ResNet solve?
     @1  cov 100%  What does mAP measure in object detection?
     @1  cov 100%  What is Gradio used for?
     @1  cov 100%  What is Async IO in Python?
     @1  cov 100%  What is backpropagation?

minilm · chunk 1000 · top-5 · relevance=all · 734 chunks
  MRR              0.627
  Keyword coverage 86.7%
  Misses           4/20
```

**How to read the per-question lines** (sorted worst-to-best):
- **`@1`** means the first correct chunk showed up at rank 1, a perfect hit. `@2` = rank
  2, `@3` = rank 3, etc. The bigger the number, the lower down the right chunk was.
- **`MISS`** means no relevant chunk appeared anywhere in the top 5, that question scored
  0 for this config.
- **`cov`** (coverage) = what fraction of the question's keywords showed up *somewhere* in
  the retrieved chunks. `cov 0%` on "reranking" means none of its keywords were retrieved
  at all; `cov 50%` means half showed up but never all-in-one-chunk (so still a MISS).

**What the summary means:**
- **MRR 0.627**, Mean Reciprocal Rank. For each question you take 1/(rank of first
  correct chunk), then average. A `@1` contributes 1.0, a `@2` contributes 0.5, a MISS
  contributes 0. 0.627 says "on average the first right answer sits a bit below rank 2".
- **Keyword coverage 86.7%**, across all questions, ~87% of keywords were retrieved
  somewhere in the top-k.
- **Misses 4/20**: four questions found nothing relevant in the top 5: LoRA,
  whack-a-mole, reranking, and overfitting. Those are the weak spots for this particular
  model+chunk-size, and they're the interesting cases to dig into later.

---

## Step 3: `sweep`: the 2×2 experiment

This is the actual assignment: run the same 20 questions across **two encoders**
(`minilm` vs `bge-small`) and **two chunk configurations** (1000 chars / top-5, and 500
chars / top-10), then compare. It reuses cached embeddings where it can, so it's fast.

**Command:**
```bash
.venv/bin/python rag_eval.py sweep
```

**Output:**
```
=== minilm · chunk 1000 · top-5 ===
  MRR 0.627 · coverage 86.7% · misses 4/20

=== minilm · chunk 500 · top-10 ===
  MRR 0.529 · coverage 92.5% · misses 4/20

=== bge-small · chunk 1000 · top-5 ===
  MRR 0.754 · coverage 95.0% · misses 2/20

=== bge-small · chunk 500 · top-10 ===
  MRR 0.642 · coverage 93.3% · misses 4/20

Wrote .../results/results.md, go fill in 'What I conclude'.
```

**What it means:** All four configs ran. Note the first row (`minilm · 1000 · top-5`) is
identical to Step 2's run: that's the same configuration, so it's a nice sanity check
that the numbers are reproducible (0.627 both times). The sweep then wrote a formatted
table to `results/results.md`.

---

## What the sweep wrote to `results/results.md`

```
# Sweep: 2026-07-18 16:57

Corpus: 21 files · golden set: 20 questions · relevance rule: `all` · overlap: 50

| Encoder | Chunk | top-k | Chunks | MRR | vs baseline | Coverage | Misses |
|---|---|---|---|---|---|---|---|
| `minilm`    | 1000 | 5  | 734  | **0.627** | +0.000 | 86.7% | 4/20 |
| `minilm`    | 500  | 10 | 1539 | **0.529** | -0.097 | 92.5% | 4/20 |
| `bge-small` | 1000 | 5  | 734  | **0.754** | +0.128 | 95.0% | 2/20 |
| `bge-small` | 500  | 10 | 1539 | **0.642** | +0.015 | 93.3% | 4/20 |

Baseline is row 1 (minilm · 1000 · top-5).
```

**Reading the table (just the facts, the interpretation is yours to write):**
- **Baseline** is row 1, `minilm · 1000`, at MRR 0.627. The **vs baseline** column shows
  every other row's gap from it.
- The **best MRR is 0.754**, `bge-small` at chunk 1000 / top-5, +0.128 over baseline,
  and it also had the fewest misses (2/20).
- Switching encoder (minilm → bge-small) while holding chunk size at 1000 moved MRR from
  0.627 → 0.754. Shrinking the chunk to 500/top-10 *hurt* both encoders here (minilm fell
  to 0.529; bge-small fell to 0.642).
- Coverage and MRR don't move together in lockstep, e.g. `minilm · 500` has *higher*
  coverage (92.5%) than baseline but *lower* MRR (0.529). Coverage asks "was the keyword
  in there somewhere", MRR asks "was the right chunk near the top". Worth noticing.

---

## What's left for you (and what I deliberately did NOT do)

The harness left a `## What I conclude` section at the bottom of `results/results.md`
with three prompts:

1. Which single change bought the most MRR?
2. Is the gap between best and second-best bigger than noise at n=20? (If you can't tell,
   say so.)
3. What would you actually ship, and why?

**I left that blank on purpose.** The README is explicit that writing this yourself, *before you look anything up*: is the whole point of the exercise, and it's exactly what
an interviewer probes when they ask "how did you know it worked?". Me filling it in would
defeat the assignment. The numbers above are everything you need to answer it.

One honest caveat the lesson wants you to state out loud: **relevance here is a keyword
proxy, not real relevance**, a chunk can contain all your keywords and still be useless.
And **n=20 is small**, so treat small MRR gaps with suspicion.

### Optional five-minute experiment the README suggests
Re-run the sweep with the looser relevance rule and watch the numbers inflate:
```bash
.venv/bin/python rag_eval.py sweep --relevance any
```
The README calls out that "the relevance definition moved MRR more than the chunk size
did" is a genuinely senior observation worth having. (I did not run this, it would
overwrite `results/results.md` with `any`-rule numbers, and your real result should stay
the `all`-rule one. Run it yourself if you want to see the effect, then re-run the plain
`sweep` to restore the real table.)
