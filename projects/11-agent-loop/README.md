# 11. Is a search loop worth the money?

**Question:** if the model can call the retriever itself and search again, does it fix
anything that a single retrieval pass could not, and what does that cost?

**Answer:** yes, on exactly the defect that had survived five experiments. Ship it behind a
trigger, not on every query.

## Preregistered, then wrong

[`hypothesis.md`](hypothesis.md) was committed before a single agent call was made. **Four
of its five hypotheses turned out to be wrong**, all in the direction that made the loop look
better than I expected. That is an uncomfortable thing to publish, which is why it is at the
top of the report rather than buried.

## What got built

`--agent`, giving the model one tool, `search_corpus(query, k)`, with `MAX_STEPS = 3` and
`steps` / `terminated_by` recorded per question. Plus a matched `control` arm, because the
model refuses tools on `chat.completions` and setting `reasoning_effort=none` degrades it
badly, so the comparison had to be built rather than assumed.

## What I found

**The five-experiment residual closed.** The `overfitting` question had failed under fixed
chunking (01), heading chunking (05), reranking at every pool size (06), the hardened prompt
(09) and tracing (10). The loop answered it correctly **3 times out of 3 at 5/5 faithfulness,
every claim grounded**.

The mechanism is the useful part. Its rewritten query **abandoned the file everyone had
assumed held the answer** and found the fact in two entirely different files, which I then
verified by grep. So the defect was never chunking, and never ranking. It was **query
formulation**, and five experiments looking one layer too low never had a chance of finding
that.

**The traps held at 0 of 15**, and the mechanism there is the opposite of what I predicted.
The model wrote its own hallucination into the search query, got nothing back, and treated
the empty result as evidence to decline. A hallucination turned into a hypothesis, then
falsified by the corpus.

**Cost, which is why it ships gated.** Mean steps 1.10. The loop fired on 2 questions in 20,
both of them the known retrieval misses, with **zero false fires**. Total cost 1.32x the
single-pass baseline. On 18 of 20 questions it does nothing except add latency, which is both
the reason to gate it and the reason it is cheap enough to keep.

**Verdict:** ship behind the trigger `abstained AND steps == 1`.

Measured spend: $0.58, plus about $0.35 lost to a hung first attempt, recorded rather than
quietly dropped.

## Run it

```bash
cd ../01-chunking
.venv/bin/python rag_eval.py run --encoder bge-small --chunk-size 1000 --top-k 5 \
    --rerank --pool 20 --generate --prompt hardened --agent --only overfitting -v
```

Full write-up: [`results.md`](results.md) · Preregistration: [`hypothesis.md`](hypothesis.md) ·
Trajectory reader: [`read_trajectory.py`](read_trajectory.py)
