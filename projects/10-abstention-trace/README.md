# 10. Seeing refusals in production telemetry

**Question:** experiment 09 proved the model abstains correctly on a test set. In production
there is no test set. How would I know?

**Answer:** emit abstention as a scored dimension on the trace, then set a band on it.

## What got built

Five lines inside the existing tracing guard, so untraced runs are byte for byte unchanged:

```python
qspan.score(name="abstained", data_type="BOOLEAN", value=...)
```

Abstention stops being a property of a test run and becomes a filterable dimension of live
traffic, alongside the retrieval and generation spans already emitted in experiment 07.

## Measured

| Set | Abstention rate |
|---|---|
| Trap (5) | **5/5 = 1.000** |
| Golden (20) | **1/20 = 0.050** |

Spend: $0.040330.

**The band: `0 < rate <= 0.10`.** Not zero, because a system that never abstains is the
system from experiment 09 that hallucinated 15 times out of 15. The upper edge is set at the
known 2/20 miss rate, so the alarm fires when the model starts refusing more often than
retrieval actually fails it.

## The one false abstention

The golden question it refused was `overfitting`, the same question that has now failed in
experiments 01, 05 and 06. The trace makes the diagnosis concrete: the right file was
retrieved three times in the top 5, but the wrong chunk each time. Exactly **one chunk of
734** carries the evidence the golden question asks for.

That rules out a timid model and rules out shallow retrieval. It is a chunking defect, one
level down the stack from where the symptom appeared. Experiment 11 eventually dissolves it
from a third direction, by reformulating the query.

## Run it

```bash
cd ../01-chunking
.venv/bin/python rag_eval.py run ... --generate --trace --prompt hardened
.venv/bin/python rag_eval.py run ... --generate --trace --prompt hardened --trap
cd ../10-abstention-trace && python read_scores.py
```

Full write-up: [`report.md`](report.md)
