# Worksheet: being the judge, by hand

> The deliverable is **not a number**. It's a documented gap between what the
> keyword ruler *measured* and what was *true*. Fill in the blanks below with your
> own judgement: the whole point is that a machine can't do this part for you.
>
> Generate the evidence you're judging with:
> ```
> ../01-chunking/.venv/bin/python show_chunks.py --list
> ../01-chunking/.venv/bin/python show_chunks.py -q overfitting -q "<pass 1>" -q "<pass 2>"
> ```

## The three questions I'm judging

Required: **overfitting** (a keyword MISS). Then pick **two that currently pass**
(status `@1`–`@4` in `--list`).

| # | Question | Ruler says | I'll be checking… |
|---|---|---|---|
| 1 | What is overfitting in a neural network? | MISS · cov 33% | is the miss *real* or a false negative? |
| 2 | What is prompt caching? | @1 · cov 100% | is the pass *real* or a false positive? → **real (true positive)** |
| 3 | _(not picked)_ | @? · cov 100% | is the pass *real* or a false positive? |

---

## Task 1: Find the false negative

For **overfitting**, coverage is 33% and the rule scores it a MISS. Read the top-5.
Does any retrieved chunk *actually answer* "what is overfitting?", in whatever
words, keywords be damned?

- **Which chunk (rank + source):** **None of the top-5 truly answers it.** The five
  retrieved are #1 `day23-oss-finetuning.md`, #2 `day26-deep-learning.md`,
  #3 & #4 `day27-deep-learning-2.md`, #5 `day26-deep-learning.md`.
- **The words that answer it** (quote the sentence): **There are none.** Every mention
  of *overfitting* is incidental: #1 lists dropout/weight decay as things that
  "Prevents overfitting", #3 notes "deeper layers are more prone to overfitting", and
  #4 actually *negates* it: "This wasn't overfitting, it was a fundamental training
  problem." No chunk defines what overfitting *is* or contrasts training vs.
  validation accuracy.
- **Why the keyword rule missed it** (which required keyword is absent, and why its
  absence doesn't mean the answer is absent): The required keywords *Training accuracy*
  and *Validation accuracy* appear in no single chunk (only *Overfitting* does, in 3 of
  5). Here, though, their absence is **honest**, the definitional answer (train acc ≫
  val acc, model memorises the noise) genuinely isn't in the top-5.
- If **no** chunk in the top-5 truly answers it, say that instead, that's a
  *different* finding (a real retrieval/chunking miss, not a bent ruler). Which is it
  here? **This is the case, a real retrieval/chunking miss.** The keyword ruler and my
  read agree: the answer simply wasn't retrieved. It is *not* a bent ruler / false
  negative.

## Task 2: Construct a false positive

Describe a chunk that would hit **all three** of a question's keywords yet a human
would **reject** as the answer. Either find a real one in the corpus, or invent a
plausible one and say so.

- **Question it targets:** _____
- **The keywords it would satisfy:** _____
- **The chunk (real or invented):** _____
- **Why a human rejects it**: negation? wrong context? keywords co-occur but the
  meaning is opposite/irrelevant? _____
  *(Hint from the lesson: for overfitting, a chunk saying "ignore validation
  accuracy" still matches all the words.)*

## Task 3: Name the metric

For **each disagreement** between the keyword score and your human judgement, which
answer-quality metric would have caught it? Use this crib (see
[../../reference/rag-metrics.html](../../reference/rag-metrics.html)):

| Metric | The question it asks | Catches |
|---|---|---|
| **Context precision / recall** | did retrieval bring back chunks that *contain* the answer, judged by meaning? | the overfitting **false negative** |
| **Faithfulness** | is every claim in the generated answer *grounded* in the retrieved context? | a true-but-ungrounded (hallucinated) answer |
| **Answer relevance / correctness** | does the answer *address the question* and match ground truth? | fluent, on-topic, but wrong |

- Disagreement A (the false negative): caught by → **Context precision / recall**, it
  asks whether retrieval returned chunks that *contain* the answer by meaning; here it
  flags overfitting because none do. (Note: ruler and human *agree* it's a miss, so
  this is a **confirmed real miss**, not a bent-ruler disagreement.)
- Disagreement B (the false positive): caught by → **none needed**, the passing
  question I checked (*What is prompt caching?*) is a genuine **true positive**
  (chunk #5 literally answers "What is Prompt Caching?"), so there's no disagreement to
  catch.

---

## The write-up (~5 sentences): this is the deliverable

_Documented gap between the ruler and the truth. Then the decision it forces._

> The keyword ruler scored *overfitting* a MISS at 33% coverage, and reading the top-5
> I agree: none of the five chunks actually answers "what is overfitting?"; they only
> mention the word while explaining dropout, weight decay, or (in #4) explicitly ruling
> it out. So for this question there is **no gap** between the ruler and the truth: both
> say the answer wasn't retrieved, which makes it a genuine retrieval/chunking miss, not
> a false negative. On the other side, the @1 pass for *What is prompt caching?* holds
> up: chunk #5's "What is Prompt Caching? … the provider reuses some of the processing
> work" is a real, direct answer, so that's a true positive, not a bent ruler inflating
> the score. The disagreement the lesson expected (a keyword false negative hiding a
> good chunk) didn't appear; what appeared instead is a **content gap** the corpus
> can't cover for overfitting. That points the fix upstream, not at the measurement.

**So what's the next move?** (circle the one your evidence supports)

- [ ] **Wire an LLM-judge into `rag_eval.py`**, because the retrieval was fine and
      only the *measurement* was blind.
- [x] **Re-chunk / fix the golden set first**, because the chunks genuinely don't
      contain a clean answer, or the keywords were the wrong ones to demand.

**Why:** For overfitting the retrieval itself failed, the definitional answer (training
accuracy ≫ validation accuracy) is absent from all top-5 chunks, so an LLM-judge would
have nothing better to grade; the fix is upstream in chunking / corpus / golden set. The
prompt-caching true positive confirms the ruler isn't the bottleneck here.
