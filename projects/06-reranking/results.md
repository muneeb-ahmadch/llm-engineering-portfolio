# Experiment 06 results: cross-encoder rerank vs cosine-only baseline

Command: `run --encoder bge-small --chunk-size 1000 --chunker fixed --top-k 5 --rerank --pool 20 -v` vs the same command without `--rerank`. overlap=50, relevance=`all`. Cross-encoder: `cross-encoder/ms-marco-MiniLM-L6-v2`.

## 0. Full-corpus ranks, found before picking a pool size

| Question | Full-corpus cosine rank | Of chunks |
|---|---|---|
| How does an LSTM solve the vanishing gradient problem? | #14 | 734 |
| What is overfitting in a neural network? | #8 | 734 |

Pool size chosen: **20**, a 4× widen over top-5 and the standard retrieve-many-then-rerank-down default in production RAG pipelines. Chosen for that reason alone, before looking at the ranks above; both ranks happen to fall inside 20 (#14 and #8), so this pool size is at least structurally capable of reaching both misses.

## 1. Does LSTM flip?

**YES. MISS → hit**

- baseline (cosine only) → MISS
- reranked (pool=20) → @2
- cosine full-corpus rank was #14 (inside the pool of 20); the cross-encoder promoted it to cross-encoder-rank #2 within that pool, which lands inside top-5.

## 2. Does overfitting flip?

**NO. Still a MISS**

- baseline (cosine only) → MISS
- reranked (pool=20) → MISS
- cosine full-corpus rank was #8, which is **inside** the pool of 20, so the cross-encoder did see it as a candidate. It re-scored it and placed it at cross-encoder-rank #7 within the pool of 20, outside top-5. Being in the pool was necessary but not sufficient: the cross-encoder itself judged 6 other candidates more relevant to the literal question wording than the chunk holding all three required keywords (its own score was negative: -9.01, meaning the cross-encoder actively considers it a poor match, not a marginal one).

### Pool-size sweep: does pool size alone decide this?

| Pool | LSTM | Overfitting |
|---|---|---|
| 5 | not in pool | not in pool |
| 8 | not in pool | HIT |
| 10 | not in pool | HIT |
| 14 | HIT | HIT |
| 20 | HIT | in-pool, not rescued |
| 30 | HIT | in-pool, not rescued |

Overfitting never becomes a hit at any pool size tried, including pools far larger than its cosine rank of #8. That confirms the miss is a cross-encoder scoring choice, not a pool-size problem. LSTM (cosine rank #14) needs pool ≥ 14 to even be a candidate, and is rescued at every pool size at or above that.

## 3. What happens to overall MRR across all 20?

| Run | MRR | Δ vs baseline | Coverage | Misses |
|---|---|---|---|---|
| baseline (cosine only) | **0.754** | +0.000 | 95.0% | 2/20 |
| reranked (pool=20) | **0.785** | +0.031 | 93.3% | 2/20 |

### Per-question diff (the honest part: who got helped, who got hurt)

| Question | baseline | reranked | Verdict |
|---|---|---|---|
| What does NDCG measure that MRR does not? | @3 | @5 | worsened |
| How is IoU calculated for a predicted bounding box? | @1 | @1 | unchanged |
| What does LoRA stand for? | @1 | @1 | unchanged |
| Which tool in the LangChain ecosystem handles monitoring and debugging? | @1 | @1 | unchanged |
| Why is RAG optimization described as whack-a-mole? | @1 | @1 | unchanged |
| What is the Chat Completions API? | @1 | @1 | unchanged |
| How are chat models different from reasoning models? | @2 | @2 | unchanged |
| How are LangChain and LiteLLM different? | @1 | @1 | unchanged |
| What is prompt caching? | @1 | @1 | unchanged |
| What is computer vision? | @1 | @1 | unchanged |
| How does RecursiveCharacterTextSplitter differ from a plain character splitter? | @2 | @2 | unchanged |
| How does an LSTM solve the vanishing gradient problem? | MISS | @2 | **rescued** |
| What problem do skip connections in ResNet solve? | @1 | @1 | unchanged |
| What does mAP measure in object detection? | @1 | @1 | unchanged |
| What does reranking do in a RAG pipeline? | @4 | MISS | **regressed** |
| What is Gradio used for? | @1 | @1 | unchanged |
| What is Async IO in Python? | @1 | @1 | unchanged |
| What is overfitting in a neural network? | MISS | MISS | unchanged |
| What is backpropagation? | @1 | @1 | unchanged |
| What is Non-Maximum Suppression in object detection? | @2 | @1 | improved |

**Composition of the +0.031 MRR move:** 1 question(s) rescued from MISS, 1 broken into a new MISS, 1 moved to a better rank while already hitting, 1 moved to a worse rank while still hitting. Net misses went 2 → 2 (flat).

## Deliverable

| | MRR | Misses/20 | LSTM | Overfitting |
|---|---|---|---|---|
| Before (cosine only) | 0.754 | 2 | MISS (full-corpus #14) | MISS (full-corpus #8) |
| After (rerank, pool=20) | 0.785 | 2 | **hit** (@2) | MISS (cross-encoder rank #7 of 20) |

**Would I ship this? Not as-is.** The rerank buys a real +0.031 MRR and a genuine rescue (LSTM: MISS to hit, and a confident one at cross-encoder rank #2 in its pool), for a fixed extra cost of one cross-encoder forward pass per candidate (20 pairs/question here). But it also introduces a *new* miss elsewhere ("What does reranking do in a RAG pipeline?", @4 → MISS) that cosine-only retrieval did not have, so the net-misses count doesn't move (2 → 2) even though MRR looks better. That is the same "MRR up, misses flat-or-worse" pattern Lesson 05 found with the heading chunker. And **the pool size I picked for a structural reason (4× widen, a common default) reaches only one of the two misses it was chosen to address**: both answer chunks are inside the pool of 20 by cosine, but the cross-encoder only promotes LSTM into the top-5. It actively scores the overfitting chunk as a poor match (-9.01), and the pool-size sweep shows this is not a threshold effect fixable by widening further (pool=30 doesn't rescue it either, and pool=8-14 rescue it while a wider pool of 20-30 does not, so rerank quality on this question is non-monotonic in pool size rather than a simple 'make it wider' problem). Shipping this specific configuration means keeping the rerank's average lift while quietly accepting that it fixes the miss it happened to be good at and not the other one.
