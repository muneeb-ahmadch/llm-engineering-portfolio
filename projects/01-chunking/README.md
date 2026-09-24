# 01. The evaluation harness

**Question:** on my own corpus, which lever moves retrieval quality more, chunk size or encoder?

**Answer:** at n=20, the encoder, by a wide margin. **Re-run at n=50 (18 September 2026), the
encoder gain shrank to +0.037 MRR, which no longer clears the noise bar this page applies, so
the n=20 answer does not stand.** Halving the chunk size made things worse in both runs.

## What I built

`rag_eval.py`, which every later experiment extends rather than replaces. By the end of the
portfolio it is 2,183 lines with four commands and the flag surface built up across thirteen
sessions:

| Command | What it does |
|---|---|
| `check` | validates the golden set against the corpus before anything runs, at $0 |
| `run` | evaluates one configuration end to end |
| `sweep` | the 2x2 matrix of chunk size against encoder |
| `gate` | the CI entry point, exits 1 on regression (added in experiment 09) |

The corpus is 21 markdown files, 712,484 bytes, with heavy topic overlap between adjacent
files. That overlap is the point: retrieval has to discriminate between genuinely similar
documents rather than pick the only one on the subject.

## Why `check` exists

It is the command people skip. Both failure modes of a hand written golden set are silent:

- **A question matching zero chunks** scores 0 forever, no matter how good retrieval gets.
  That is a broken question, not a finding, and it will drag MRR down while you spend an
  evening tuning chunk size against a typo.
- **A question matching fifty chunks** means the keywords are so generic that everything
  counts as relevant. MRR pins near 1.0 and you have measured nothing. This is the more
  dangerous one, because the number looks excellent.

## What I found

Sweep of 2026-07-18, 20 golden questions, relevance rule `all`, overlap 50:

| Encoder | Chunk | top-k | Chunks | MRR | vs baseline | Coverage | Misses |
|---|---|---|---|---|---|---|---|
| `minilm` | 1000 | 5 | 734 | 0.627 | +0.000 | 86.7% | 4/20 |
| `minilm` | 500 | 10 | 1539 | 0.529 | -0.097 | 92.5% | 4/20 |
| **`bge-small`** | **1000** | **5** | **734** | **0.754** | **+0.128** | **95.0%** | **2/20** |
| `bge-small` | 500 | 10 | 1539 | 0.642 | +0.015 | 93.3% | 4/20 |

The encoder swap gained +0.128 MRR at chunk 1000 and +0.113 at chunk 500, so it helped in
both settings. Chunk size never bought anything: going from 1000 to 500 hurt `minilm` by
0.098 and `bge-small` by 0.112.

**On whether that is noise (the n=20 reasoning; the n=50 re-run below overturns it).** At n=20, one question flipping between a miss and a rank-1 hit
moves MRR by 0.05, so I treat 0.05 as one question's worth of noise. The encoder gain is
more than two questions' worth and points the same direction under both chunk sizes, which
is why I acted on it at n=20. The 0.015 gap between `bge-small`/500 and `minilm`/1000 is inside the
noise floor and I do not read anything into it. With n=20 and a keyword proxy standing in
for real relevance, the encoder result is confident enough to ship, not proven.

**Shipped at n=20:** `bge-small` at chunk 1000, top-5. It costs the same as `minilm`, since both
run locally on CPU with no API key, so the upgrade is free. It is still the pipeline default.
The n=50 re-run below no longer shows it is better at retrieval.

## Re-run at n=50 (18 September 2026)

Thirty questions were added on 9 September to cover ten corpus files the original twenty never
asked about. Same sweep, 50 golden questions:

| Encoder | Chunk | top-k | Chunks | MRR | vs baseline | Coverage | Misses |
|---|---|---|---|---|---|---|---|
| `minilm` | 1000 | 5 | 734 | 0.707 | +0.000 | 91.7% | 7/50 |
| `minilm` | 500 | 10 | 1539 | 0.476 | -0.231 | 92.7% | 11/50 |
| `bge-small` | 1000 | 5 | 734 | 0.745 | +0.037 | 98.0% | 2/50 |
| `bge-small` | 500 | 10 | 1539 | 0.631 | -0.077 | 96.3% | 9/50 |

At n=50 one question is worth 0.02 MRR, so +0.037 is 1.87 questions' worth. That is under the
two-question bar the n=20 write-up used to justify acting, so **the encoder result that
shipped at n=20 did not replicate at that size.** Coverage rose 91.7% to 98.0% and misses fell
7/50 to 2/50. Halving chunk size hurt both encoders again, and hurt `minilm` markedly harder
(-0.231, against -0.098 at n=20). Full table and write-up: [`results/results.md`](results/results.md).

The two questions it still misses are the subject of experiments 05, 06, 10 and 11.

## Run it

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r ../../requirements.txt

.venv/bin/python rag_eval.py check
.venv/bin/python rag_eval.py run --encoder bge-small --chunk-size 1000 --top-k 5 -v
.venv/bin/python rag_eval.py sweep
```

Free, offline, no API key. Embeddings cache under `results/.cache/`, so re-running a
configuration is instant.

Full sweep write-up: [`results/results.md`](results/results.md) ·
Encoder comparison: [`results/lesson-02-encoder-comparison.md`](results/lesson-02-encoder-comparison.md)
