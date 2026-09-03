# 09. Closing the hallucination gap

**Question:** the pipeline answered a question the corpus could not support. How often does
that happen, and what is the cheapest thing that fixes it?

**Answer:** every single time. And a prompt fixed it.

## The headline

Three numbers, measured twice. **The only thing that differs between the columns is the
system prompt.**

| | baseline prompt | hardened prompt | gate rule |
|---|---|---|---|
| Trap hallucinations (3 runs) | **5/5 · 5/5 · 5/5** | **0/5 · 0/5 · 0/5** | must be 0 |
| Mean faithfulness (golden, 1 to 5) | 3.90 | **4.80** | >= 4.0 |
| Claim groundedness | 95/142 = 66.9% | 43/45 = **95.6%** | |
| Abstention rate (golden) | 0.0% | 0.0% | <= 30% |
| MRR | 0.785 | 0.785 | retrieval untouched |

The system I had been shipping since experiment 07 failed two of three gate rules and failed
the trap set completely: **15 hallucinations out of 15 runs.** Not a flicker, a flatline.

## Building a trap set that is actually a trap

Five off-topic questions would measure nothing. An off-topic question retrieves junk, the
model sees junk, and abstaining is easy.

A real trap sits **just outside the corpus boundary**: the retriever returns chunks that look
maximally on topic, the model is handed plausible context on exactly the right subject, and
the one specific fact asked for is not in it. That is where a model reaches past its context.

So the five probe different shapes of boundary, each naming the covered ground it sits next
to. A metric-shaped hole, where the corpus mentions MRR 32 times and NDCG 11 times but never
defines faithfulness. A mechanism listed as a bare bullet and never explained. And so on. Two
candidate traps died under `check --trap` because the corpus turned out to answer them, which
is exactly what that command is for.

## What got built

- `is_abstention()`, so a refusal is detected rather than scored as a bad answer
- an LLM faithfulness judge returning a 1 to 5 score plus per claim groundedness
- `gate`, the CI entry point, which exits 1 on regression and encodes the three rules above

## The caveat I would raise in an interview

This fix was validated by a judge that is itself a language model reading the same context.
Experiment 12 attacks exactly that assumption and finds this judge endorsing an attack it
should have caught, and experiment 13 fixes the judge by segregating its context. The defense
in this experiment is real, and the instrument that certified it was not yet trustworthy.

## Run it

```bash
cd ../01-chunking
.venv/bin/python rag_eval.py check --trap                                    # $0
.venv/bin/python rag_eval.py run ... --generate --trap --repeat 3 --prompt hardened
.venv/bin/python rag_eval.py run ... --generate --judge --prompt hardened
.venv/bin/python rag_eval.py gate --nightly                                  # exits 1 on failure
```

Measured spend for the whole experiment: $0.49.

Full write-up: [`report.md`](report.md) · Raw runs: [`results-defense.json`](results-defense.json) ·
The original probe: [`probe_paris.py`](probe_paris.py)
