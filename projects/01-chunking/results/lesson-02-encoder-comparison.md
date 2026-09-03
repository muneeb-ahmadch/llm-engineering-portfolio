# Experiment 02: encoder miss-list comparison

**Configs compared** (both chunk 1000 / top-5 / relevance=all, 734 chunks):

```bash
.venv/bin/python rag_eval.py run --encoder minilm    --chunk-size 1000 --top-k 5 -v
.venv/bin/python rag_eval.py run --encoder bge-small --chunk-size 1000 --top-k 5 -v
```

- `minilm`   → MRR 0.627, coverage 86.7%, **4 misses**: LoRA, whack-a-mole, reranking, overfitting
- `bge-small`→ MRR 0.754, coverage 95.0%, **2 misses**: LSTM, overfitting

## Diff of the miss lists

| Question | minilm | bge-small | Verdict |
|---|---|---|---|
| What does LoRA stand for? | MISS | @1 | **rescued** |
| Why is RAG optimization described as whack-a-mole? | MISS | @1 | **rescued** |
| What does reranking do in a RAG pipeline? | MISS (cov 0%) | @4 | **rescued** |
| What is overfitting in a neural network? | MISS | MISS | **survived** |
| How does an LSTM solve the vanishing gradient problem? | @2 | MISS (cov 67%) | **new regression** |

**The honest count:** the assignment predicted "rescues two, two survive." On my golden
set bge-small actually **rescued three** (LoRA, whack-a-mole, reranking), **one survived**
(overfitting), and it **broke one that minilm had gotten right** (LSTM, @2 → MISS). Net
misses still went 4 → 2, but the composition is +3 −1, not a clean two-for-two. You only
see the LSTM regression by diffing the verbose lists. The MRR summary hides it.

## Why bge-small put a rescued query nearer its answer (one sentence)

**LoRA:** bge-small is trained specifically for retrieval, on contrastive query→passage
training with hard negatives, plus an asymmetric query prefix ("Represent this sentence
for searching relevant passages:"), so it maps the short question to the one chunk that
*defines* the acronym instead of getting pulled toward the ~51 chunks that merely mention
"LoRA" in passing, which is where minilm's general-purpose, topic-overlap embedding lands.

## The survivor: what I'd try next (one sentence)

**Overfitting:** I would NOT reach for a still-stronger encoder first. Coverage is stuck
at 33% (only one of the three keywords ever surfaces) and just one chunk in the whole
corpus holds all three, which smells like a chunk-boundary / keyword-phrasing problem
rather than a semantic-mapping one, so I'd try smaller overlapping chunks (or a reranker,
or rethinking the keywords) before assuming a bigger model fixes it.

## Bonus finding I'd raise in an interview

The LSTM regression means "upgrade the encoder" is not free. bge-small is better *on
average* but it is not strictly better per-question. If LSTM-type questions mattered to
the product I'd want an ensemble or a reranker rather than blindly swapping encoders, and
at n=20 a single-question flip is only 0.05 of MRR, so I'd confirm the regression is real
before acting on it.
