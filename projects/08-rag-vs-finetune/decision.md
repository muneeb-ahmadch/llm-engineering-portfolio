# Experiment 08 decision doc: RAG vs fine-tune vs prompt

A decision doc, not a build. Graded on the diagnosis, not the tool name. The axis
that settles it: **is the gap knowledge or behaviour?** Prompting changes what I say,
RAG changes what the model can see, fine-tuning changes what the model is.

Status: **done**. Four scenarios diagnosed, break-even framed on the L07 cost number, claim written.

---

## Part 1: four scenarios, one call each

For each: **(a)** prompt / RAG / fine-tune / composition · **(b)** the gap in one sentence
(knowledge, behaviour, or both) · **(c)** the single measurement that says the call was right.

### Scenario 1: InsureElm's 400-page policy handbook, revised every quarter

> Answer employee questions about the handbook.

- **(a) Call:** **RAG.**
- **(b) Gap:** **Knowledge.** The LLM doesn't know InsureElm's private policy; RAG lets it *see* the relevant chunks to answer from, no facts written into weights.
- **(c) Measurement:** Retrieval **MRR ≈ 1.0**, then check the LLM half for low **hallucination** and correct **abstention** when the chunk isn't there. Discriminating test for RAG *over fine-tune*: after a quarterly revision, **swap the index and confirm answer-correctness on the golden set still tracks the *new* policy with zero retraining.** That's the number that proves an updatable index beat stale weights.

**Why not the others:** Not prompt-only, because 400 pages blows the context window and you would pay for all of it, diluted, every query. Not fine-tune, because the handbook is revised quarterly, so memorized facts go stale and training is re-paid every cycle.

### Scenario 2: correct but chatty; clinicians need terse, fixed-structure output

> The answers are correct but too long and chatty; clinicians need terse, fixed-structure output every time.

- **(a) Call:** **Composition, prompt then fine-tune.** Prompt first (few-shot template plus enforced output schema); fine-tune only when terseness or format still drifts above the length budget after the prompt's ceiling is measured.
- **(b) Gap:** **Behaviour.** "Correct" says the facts are already right, so there is nothing to retrieve and it is not knowledge; the *form* (verbosity, structure) is wrong, which is a behaviour gap.
- **(c) Measurement:** **Structural-conformance rate on a held-out set**, the percentage of outputs matching the required template *and* under the length budget. Measure prompt-only first to establish the ceiling; the fine-tune call is right iff prompt plateaus below bar (~90%) and fine-tune clears it (~99%+). Secondary: tokens and latency per query drop once the format lives in the weights instead of the prompt.

**Why not the others:** Not RAG, because no fact is missing, so retrieval adds nothing and the form is wrong rather than the knowledge. Not prompt-*only* if it plateaus, because a prompt has a silent ceiling and "every time" is a high-reliability bar it may not clear for tone. (Note: temperature=0 buys determinism, not terseness. Verbosity isn't a randomness knob, which is a second reason RAG plus temperature is the wrong instrument.)

### Scenario 3: support-ticket classifier, 12 stable categories, 50k tickets/day, latency-sensitive

> A support-ticket classifier: 12 stable categories, 50,000 tickets/day, latency-sensitive.

- **(a) Call:** **Fine-tune a small model.** The three words settle it: *stable* (label set won't move → weights won't go stale, unlike Scenario 1), *50k/day* (volume amortizes a one-time training bill), *latency-sensitive* (a small fine-tuned model beats a big prompted one on speed).
- **(b) Gap:** **Behaviour.** The behaviour installed is the **classification mapping itself**, ticket → one of 12 fixed labels, learned from the abundant labeled data 50k/day provides. Not knowledge: nothing private or changing to retrieve, and the whole "knowledge" is 12 fixed category names, which belong in the weights rather than an index.
- **(c) Measurement:** **Macro-F1 on a held-out labeled set.** The fine-tuned small model must **match or beat the big prompted baseline's F1** (macro, so rare categories don't hide behind common ones) **while p95 latency and cost/query drop.** Right call = accuracy held, latency and cost fell. F1 slips → model too small; latency and cost flat → fine-tuning bought nothing.

**Why not the others:** Not RAG, because it adds an embedding and vector-search hop to *every* query (more latency, more cost, a hallucination and abstention surface) with nothing to retrieve. Not prompt-only, because it works as the baseline but a long few-shot prompt on a big model at 50k/day is slower and costlier than a small fine-tuned model.

### Scenario 4: the Lesson 07 "Paris" probe

> The model answered instead of saying "I don't have that information."
> (L07: fed an LSTM chunk as context, asked for the capital of a country never mentioned → returned "Paris, the capital of France" despite the grounding instruction.)

- **(a) Call:** **Composition. Prompt-harden plus a groundedness guardrail first; fine-tune for abstention only if a *measured* out-of-context failure rate survives that.** The ceiling to exhaust first is the **prompt** ceiling (stronger grounding plus a "is this supported by the retrieved context? if not, abstain" verification step), **not** RAG, because RAG is a knowledge lever and has no bearing on abstention.
- **(b) Gap:** **Behaviour.** Knowledge isn't missing, it's over-present: the model knew "Paris" too well and reached past the context. Nothing to retrieve; the broken thing is the *form of behaviour*, staying grounded and abstaining when unsupported.
- **(c) Measurement:** On a **held-out set of unanswerable or out-of-context probes** (Paris, scaled to dozens), the **abstention rate** (percentage correctly saying "I don't have that information") must climb toward ~99% **without raising the false-abstention rate on *answerable* questions.** The pair is the number, because a model that abstains on everything aces abstention and is useless.

**Why fine-tune is the wrong *first* move (the hype trap):** the L07 evidence is **one** out-of-band probe, not a measured failure rate. Fine-tuning off one anecdote is the "fine-tune it!" reflex misfiring. Prompt-harden + guardrail is the correct stopping point unless a measured residual clears your pain threshold.

---

## Part 2: one real number (break-even on Scenario 3)

Inputs: L07 cost/query = **$0.002174**; fine-tuning drops the retrieved context → per-query cost ≈ **a third**; volume = **50,000 tickets/day**.

- Current daily cost: `0.002174 × 50,000` = **$108.70/day**
- Fine-tuned daily cost (÷3): **$36.23/day**
- **Daily saving: $108.70 − $36.23 = ~$72.47/day** (≈ $2,174/month ≈ $26k/year)

**Break-even framing.** Training is a *one-time* cost; the saving is *ongoing*. A training bill of $X pays back in `X ÷ 72.47` days:

| One-time training bill | Pays back in |
|---|---|
| $72 | ~1 day |
| $725 | ~10 days |
| $2,174 | ~30 days (1 month) |
| $6,522 | ~90 days (1 quarter) |
| $26,450 | ~365 days (1 year) |

At ~$72/day saved, a realistic small-model fine-tune (tens to low-hundreds of dollars, maybe low-thousands with iteration) is recovered in **days to a couple of weeks**. It stops being worth it only if the training-plus-upkeep bill exceeds what is saved over the model's usable lifetime, an enormous ceiling at this volume.

**Why "stable" makes the math forgiving:** break-even is `one-time cost ÷ lifetime days of saving`. Stable categories → pay training **once**, collect $72/day **indefinitely** (huge denominator, almost any bill clears). Contrast Scenario 1's quarterly revision: training **recurs** ~4×/year and the break-even clock **resets each quarter**, which is exactly why fine-tune was wrong there. **Stability lets a one-time cost amortize; churn kills it.**

---

## Part 3: the claim, or "when would you fine-tune instead of RAG?"

> "I'd fine-tune instead of RAG only when the gap is **behaviour, not knowledge**, where the model
> already has the facts but won't *act* right (fixed format, tone, reliable abstention), and only
> after a good prompt has hit a **measured ceiling** on a task **stable and high-volume enough to
> earn the training bill**; my Scenario-3 ticket classifier clears that bar, where ~$72/day saved
> pays a small-model fine-tune back in days. My own RAG system **correctly stops at prompt+RAG**
> because its gap is **knowledge**: a private, quarterly-changing corpus whose facts belong in an
> index I can update and cite, not in weights that go stale. The one behaviour wobble I actually
> saw, the L07 'Paris' abstention miss, was a **single unmeasured probe** I'd first fix with a
> stronger grounding prompt and a groundedness guardrail; until I've **measured** an abstention
> failure rate that survives that, fine-tuning would be cost and staleness I haven't earned."

**The rule underneath it:** gap = knowledge or behaviour? Knowledge (private / fresh / changing)
→ RAG, because facts belong in an index I can update and cite. Behaviour that *survives a measured
prompt ceiling* AND is *stable + high-volume* → fine-tune, because the training bill amortizes.
Prompt+RAG is a correct **stopping point**, not a failure to reach rung 3.
