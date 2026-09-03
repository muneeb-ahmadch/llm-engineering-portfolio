# 01. The evaluation harness

**Question:** on my own corpus, which lever moves retrieval quality more, chunk size or encoder?

**Answer:** the encoder, by a wide margin. Halving the chunk size made things worse.

## What I built

`rag_eval.py`, which every later experiment extends rather than replaces. By the end of the
portfolio it is 2,187 lines with four commands and the flag surface built up across thirteen
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

**On whether that is noise.** At n=20, one question flipping between a miss and a rank-1 hit
moves MRR by 0.05, so I treat 0.05 as one question's worth of noise. The encoder gain is
more than two questions' worth and points the same direction under both chunk sizes, which
is why I act on it. The 0.015 gap between `bge-small`/500 and `minilm`/1000 is inside the
noise floor and I do not read anything into it. With n=20 and a keyword proxy standing in
for real relevance, the encoder result is confident enough to ship, not proven.

**Shipped:** `bge-small` at chunk 1000, top-5. It costs the same as `minilm`, since both run
locally on CPU with no API key, so the upgrade is free.

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
