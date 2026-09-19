# Sweep, 2026-09-18 01:45

Corpus: 21 files · golden set: 50 questions · relevance rule: `all` · overlap: 50

| Encoder | Chunk | top-k | Chunks | MRR | vs baseline | Coverage | Misses |
|---|---|---|---|---|---|---|---|
| `minilm` | 1000 | 5 | 734 | **0.707** | +0.000 | 91.7% | 7/50 |
| `minilm` | 500 | 10 | 1539 | **0.476** | -0.231 | 92.7% | 11/50 |
| `bge-small` | 1000 | 5 | 734 | **0.745** | +0.037 | 98.0% | 2/50 |
| `bge-small` | 500 | 10 | 1539 | **0.631** | -0.077 | 96.3% | 9/50 |

Baseline is row 1 (minilm · 1000 · top-5).

## What I conclude

<!-- Write this yourself, before you look anything up. Three questions:
     1. Which single change bought the most MRR?
     2. Is the gap between the best and second-best bigger than noise
        at this sample size? If you can't tell, say so.
     3. What would you actually ship, and why? -->
