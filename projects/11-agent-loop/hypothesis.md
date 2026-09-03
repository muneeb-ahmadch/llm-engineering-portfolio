# Preregistration, written before a single agent call was made

Written 2026-07-29, after the `--agent` code was finished and before it was run
even once. The point of this file is that the report can't quietly become a
story about what I already saw. Every prediction below is falsifiable and dated.

If a prediction is wrong, the report says so in the same line as the number.

---

## H1: the step-2 query (the diagnosis)

The overfitting question fails today because the evidence is in the retrieved
*document* (`day26-deep-learning.md`, returned 3× in the top-5) and not in the
retrieved *window*: exactly **1 chunk of 734** holds all three golden keywords.

So there are two possible rewrites, and they mean opposite things:

- **A rephrase.** Something like `"overfitting neural network definition"`,
  synonyms of the original question. This tells me the loop did **not** understand
  why the context was starved. Cosine on a paraphrase lands in the same
  neighbourhood, returns the same three day26 chunks, and step 3 is step 1 with a
  bigger bill.
- **A diagnosis.** Something naming the *missing mechanism* rather than the topic,
  `"training accuracy much higher than validation accuracy"`, `"validation loss
  increases while training loss decreases"`, `"model memorises noise generalises
  poorly"`. This is the query that could actually move to a different chunk,
  because the one chunk that holds the evidence is the one that contains those
  words.

**I predict a rephrase, not a diagnosis**, maybe 65/35. The model is not told
*why* its context is insufficient; it is only told the context is what it has.
Query rewriting without a failure signal is, in my experience, topic-preserving.

**If it does write a diagnostic query, that is the strongest result in this
lesson**, and it also means the loop can rescue the L09 residual.

## H2: the overfitting question, both ways

- Non-agent (today's shipping config): `abstained = 1`, faithfulness **n/a**
  (zero-claim), ~1 call, ~$0.002.
- Agent: I expect **it answers** (`abstained = 0`), because a model that has just
  spent a tool call looking is in "make progress" mode.

The number that decides whether this is a win: **faithfulness on that answer.**
- **≥ 4** → the second search actually found the evidence. A real rescue.
- **≤ 2** → the loop searched, found nothing new, and answered from pre-training
  anyway. That is L09's *parametric rescue* wearing a costume, and it is strictly
  worse than the abstention it replaced, because the abstention was honest.

I predict **1/5 or 2/5**, parametric rescue.

## H3: the golden set (20 questions)

| Metric | Prediction | Reason |
|---|---|---|
| abstention rate | 5.0% → **0.0%** | the one abstention was the overfitting question; the loop will talk itself out of it |
| mean faithfulness | 4.80 → **down, not up** (4.2–4.6) | more retrieved context = more surface for ungrounded synthesis, and the judge grades against everything the model was shown |
| cost / query | **2–3×** | prompt tokens are re-sent every step and the context grows monotonically |
| mean latency | **1.6–2.2×** | serial calls, no parallelism |
| mean steps | **1.2–1.5** | most golden questions are answerable at rank 1; the loop should not fire |

The mean-steps prediction is the one I care about most. **A loop that fires on
every question is a loop that has learned nothing.** If mean steps ≈ 1.1 and only
the genuinely starved questions search, the mechanism is discriminating. If mean
steps ≈ 2+, it is searching reflexively and I am paying for a habit.

## H4: the trap set (THE trap), stated before the run

> **A second chance at searching helps the model talk itself into an answer.**

Trap hallucinations: **0/5 → 2/5 or worse.** I'll call it a confirmed failure at
≥ 1 and a surprise at 0.

Three reasons, in order of how much I believe them:

1. **The traps are built to survive a second search.** Each one sits *just* outside
   the corpus boundary with maximally on-topic neighbours (`golden-trap.jsonl`
   `adjacent_to`). A rewritten query lands in the *same* on-topic neighbourhood
   and returns *more* plausible-looking context with the fact still absent. The
   model ends step 2 with twice the context and zero more evidence, and context
   volume is exactly what reads as "I have enough to answer."
2. **Sunk cost is a real prior in an agent transcript.** The transcript now
   contains: I searched. I got results. Abstaining after visibly working looks, to
   a next-token predictor, like the wrong continuation.
3. **The forced-answer step is an instruction to answer.** At the cap I send
   `tool_choice="none"`, which removes the option to keep looking. The hardened
   prompt's rule 3 still permits abstention, but the loop's own shape pushes the
   other way.

The counter-case, which is why this is worth running rather than assuming:
the hardened prompt's rule 1 ("your own knowledge is not admissible") is
*evidence-conditioned*, not effort-conditioned. A model that actually applies it
should abstain **harder** after searching, because it now has direct evidence that two
different queries failed to surface the fact. If trap stays 0/5, that is the
prompt doing something I did not think it could do.

**`0/5` is the L09 result I am putting at risk, and it is the only number in this
project that has ever been part of a per-commit CI gate.**

## H5: the verdict I expect to write

Ship it **off** by default. I expect the agent loop to buy nothing on the golden
set, cost 2–3×, and regress the one gate rule I actually enforce. My prior is
that this is a control that needs a *trigger* (fire only on a starved context),
not a control that should run on every query.

Written before the first run. M.
