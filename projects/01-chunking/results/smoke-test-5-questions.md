# Sweep, 2026-07-17 19:15

Corpus: 21 files · golden set: 5 questions · relevance rule: `all` · overlap: 50

| Encoder | Chunk | top-k | Chunks | MRR | vs baseline | Coverage | Misses |
|---|---|---|---|---|---|---|---|
| `minilm` | 1000 | 5 | 734 | **0.467** | +0.000 | 80.0% | 2/5 |
| `minilm` | 500 | 10 | 1539 | **0.550** | +0.083 | 90.0% | 1/5 |
| `bge-small` | 1000 | 5 | 734 | **0.867** | +0.400 | 100.0% | 0/5 |
| `bge-small` | 500 | 10 | 1539 | **0.800** | +0.333 | 100.0% | 0/5 |

Baseline is row 1 (minilm · 1000 · top-5).

## What I conclude

<!-- Write this yourself, before you look anything up. Three questions:
     1. Which single change bought the most MRR?
     2. Is the gap between the best and second-best bigger than noise
        at this sample size? If you can't tell, say so.
     3. What would you actually ship, and why? -->
