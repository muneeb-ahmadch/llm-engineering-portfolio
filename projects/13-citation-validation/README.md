# 13. The stage that cannot be argued with

**Question:** experiment 12 ended by predicting that structured citations plus a
deterministic Python check would have caught the attacks. Would they?

**Answer:** no. On the clauses that can run in production it catches **none of the eight
attacks that landed**, and it is not close.

## Why the prediction was wrong

The check works exactly as specified, passes all 19 unit tests, runs in microseconds at $0.
It is the claim that was wrong, and the reason is worth more than the fix would have been.

**The attacker's method is to make the citation true.** The winning payload wrote a sentence
into a chunk my retriever ranked first. The model read it, said it, and cited it, correctly.
Every clause passes, because every clause is asking whether the answer is **attributable**,
and the answer is impeccably attributable to a document that lies.

## Three findings, in ascending order of surprise

**1. Segregating the judge worked, and the control proves it.** `judge_answer()` got the same
nonce-delimited context and its own rule block. Correct answers punished at 3/5 or below in
the poisoned arm went from **4 to 0**. The three answers that scored **1/5** because a forged
`SYSTEM:` line told the judge they were superseded now score **5/5, three runs out of three**.
The control that makes this readable: re-judging the same 18 answers on the old flat judge
today changed **0 of 18 scores**. The improvement is the segregation, not the dice.

**2. The validator's real product is not blocking, it is a change in the kind of statement I
can make.** Zero blocks on the injection suite is the honest headline. But `is_abstention()`
is a regex I measured at 0/18, while `validate_citations()` is a branch I can read: every id
in the answer is in the set this request minted, or it is not, and no sentence in any markdown
file changes that. The stage moved from measurable to **provable**. What it proves is narrower
than experiment 12 promised, and naming the narrower thing is the exercise.

**3. My own validator blocked two correct answers on the clean corpus.** Version 1 printed
each chunk as `[C2 · day15-rag-langchain.md]` while the rule said to use the id exactly as
printed. The model read the whole bracket as the id and the check correctly reported it was
not in the minted set. **A 10% false positive rate on clean, correct answers, caused by a
two-reading contract rather than by the model failing.**

A deterministic stage is deterministic about the property it checks. Its **soundness** still
depends on an agreement with the probabilistic stage feeding it, and that agreement is prose.
Fixing the format took blocks from 2/20 to 0/20 with everything else held.

The injection suite would never have found this: every id the model emitted on the poisoned
arm was a bare `C3`. It took the **clean regression half**, the part that exists to check the
defense does not break the system, to surface a bug in the defense itself.

## What is in here

- [`stage-map.md`](stage-map.md), all six pipeline stages classified as code or model,
  written first
- [`predictions.md`](predictions.md), committed before `validate_citations()` existed
- [`test_validator.py`](test_validator.py), 19 cases, no API, no model load, gates everything
- [`decidability.md`](decidability.md), the ceiling split into what is decidable in production
  versus what is only decidable in CI

## Run it

```bash
python test_validator.py        # free, 19 cases, gates everything below
python rejudge.py               # judge-only calibration, no regeneration, ~$0.62
python run_cited.py             # the cited arm, regression and ceiling, ~$0.55
python summarize_cited.py --answers
```

Measured spend: $1.708 across 329 calls, nothing lost.

Full write-up: [`results.md`](results.md)
