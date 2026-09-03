# Step 6: what the check can decide, clause by clause

The point of the exercise is not "I added a validator." It is knowing, for each line
of it, whether the thing it decides is available at 3am on live traffic or only in a
test I wrote. Four clauses ship; a fifth exists and cannot.

| # | clause | decides a property of… | production? | catches | misses |
|---|---|---|---|---|---|
| 1 | `parsed` is `{answer: str, claims: [...]}` | **the wire format**, my own decode | **yes** | a reply the code cannot read: a truncated stream, a cap-terminated step, a model that emitted prose anyway | nothing about content. The schema makes this near-vacuous, which is why it costs one line |
| 2 | every claim carries a non-empty `chunk_id` | **presence of a field I demanded** | **yes** | uncited assertion: parametric rescue, the L07 *"Paris"* wound, and `INJ-05` run 1 where the model answered from pre-training with the search suppressed | anything the model is willing to cite, which after A3 is nearly everything |
| 3 | every `chunk_id` ∈ the ids this request minted | **a relation I control**: `Labeller.ids` | **yes** | invented ids; a forged id written into a payload; an id copied from another request. Not probabilistically: a 64-bit-nonce-scoped label set is a set-membership test | **a truthful citation of a poisoned chunk.** This is the miss that matters and it is the attacker's entire method |
| 4 | a non-abstaining answer carries ≥1 claim | **an invariant between two fields I defined** | **yes** | the hole through 2 and 3: emit `claims: []`, assert freely, pass. Same shape as L09's zero-claim answer scoring a free 5/5 | an answer that makes one citable claim and six uncitable ones folded into prose |
| n/a | *the cited chunk's text supports the claim* | the **content**, a keyword/overlap proxy | **not built** | citation drift: right id, claim not in the block | paraphrase. And it is a heuristic, so it has a false-positive rate that would have to be measured before it could gate anything |
| 5 | no cited chunk carries a known payload | **a list that does not exist at runtime** | **CI only** | every injection I planted, deterministically | **everything, in production.** If I had the poison list I would have deleted the poison, not cited it |

## The line, stated plainly

Clauses 1–4 decide **whether the answer is attributable**. Clause 5 decides **whether
the source is trustworthy**. Only the first question has an answer inside the request.

Attributability is a closed property: the request minted the ids, the request assembled
the context, and asking "is this id one of mine" is arithmetic on a set I built four
lines earlier. There is no sentence an attacker can write into a markdown file that
changes the answer, because the answer does not depend on anything the file says.

Trustworthiness is not a property of the request at all. It is a property of **how the
document got into the corpus**, and that happened before the process started.

## What would have to exist upstream

For clause 5 to run in production, `retrieved[i]` would have to arrive carrying
provenance a checker can evaluate without reading the prose:

1. **A signed corpus.** Documents signed at ingest by an authenticated author, signature
   verified at index build. Then "is this chunk from a trusted author" is decidable, and
   it is decidable the same way clause 3 is, a key check rather than a judgement.
2. **A content diff between index builds.** Nothing in this repo diffs `corpus/` between
   runs. The L12 attack was **1,820 bytes on 712,484** appearing between one build and
   the next, and a diff would have printed all six payloads without a model in the loop.
   This is the cheapest of the three by a wide margin and it is why it goes first.
3. **A review gate on the diff.** A human or a policy that has to approve new corpus
   text before it is indexed. This is where "0.255% of the corpus" stops being invisible.

All three are **stage 0**, upstream of retrieve and upstream of every guardrail in this
repository. That is the honest shape of the ceiling: today's check moved the boundary
between "provable" and "measurable" one stage to the right. It did not move the boundary
between "checkable" and "unknowable", because that one is not inside the pipeline.

## The sentence in the L12 report that this corrects

> *"…it is the one that would have caught INJ-04, because a Python check 'does every
> claim trace to a chunk that is not on the poison list' does not care how persuasive
> the prose was."*

True, and it quietly assumes the poison list. In the adversarial suite I have it, because
I wrote the payloads. In production the list is empty, not because nothing is poisoned,
but because nothing is **known** to be. The clause that would have caught INJ-04 is
exactly the clause that does not ship.
