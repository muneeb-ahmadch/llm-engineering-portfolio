# 07. What an answer costs

**Question:** now that the pipeline generates answers, what do they cost, how long do they
take, and what new failure mode did I just buy?

**Answer:** $0.002174 per query, 2.1 s mean latency, and hallucination.

## Verifying the price before writing the code

| Model | Input /1M | Cached input /1M | Output /1M | Verified |
|---|---|---|---|---|
| `gpt-5.6-luna` | $1.00 | $0.10 | $6.00 | 2026-07-22, live from the pricing page |

I got this wrong first. My own knowledge had `gpt-5.6-luna` down as a fictional model, so I
asserted it did not exist. Listing what the key could actually reach returned it, alongside
`gpt-5.5`, `gpt-5.4-mini` and dated variants. The live check overruled my memory, and I left
that mistake in the report because pricing and model availability are the two facts you can
never answer from memory.

That rule is enforced in code, not prose: `--model` is restricted to keys in the `PRICES`
dict, so the harness physically cannot price a model whose verified rate is not recorded.

## Measured, 20 questions, one run

```
MRR              0.785        identical to experiment 06; generation does not touch retrieval
Coverage         93.3%
Misses           2/20

Total cost       $0.043471    (25,627 in + 2,974 out tokens)
Cost per query   $0.002174
Mean latency     2,101 ms     min 955 · median 1,916 · p95 3,560 · max 4,344
Wall clock       63.4 s for the serial 20-question loop
```

**Input is 59% of the bill.** Output costs 6x per token, but there were 8.6x fewer output
tokens, so the retrieved contexts at roughly 1,280 prompt tokens each dominate. To cut cost,
trim context before capping answer length.

**The latency tail is wide and tracks output length.** Max is 2.1x the mean. The slowest
question wrote 216 output tokens, the fastest 15.

## Two findings

**A retrieval miss is not an answer miss.** Both retrieval misses got correct, complete
answers, because `overfitting` and `reranking` are generic ML concepts the model knows cold
and answered from parametric knowledge. Answer quality here is better than MRR suggests, but
only by luck of subject matter. On a private corpus the model has no fallback, and the same
retrieval miss becomes an abstention or a hallucination that you would never see coming if
you only watched answer quality. This is the argument for keeping the two metrics separate.

**Generation adds a failure mode retrieval never had.** An out of band probe asking for the
capital of a country never mentioned in the context, with an unrelated chunk supplied,
returned "Paris, the capital of France" instead of the instructed refusal. The grounding
instruction is not a guarantee. That probe is what experiment 09 was built to close.

## The CI verdict

Nightly, not per commit, and cost is not the deciding factor. At $0.043 per run, 200 commits
a day is under $9. The real costs are 63 s of serial wall clock on every push, growing past
two minutes as the golden set grows, and non-determinism: the model refuses `temperature=0`,
so any commit gate on answer content would flake. So the pipeline splits by what each half
needs. **Retrieval metrics are free, deterministic and cache-fast, so they run on every
commit. The paid, slow, non-deterministic generation pass runs nightly.**

## Tracing

Langfuse wired in behind a `--trace` flag that is a no-op when off, using the drop-in
`langfuse.openai` wrapper rather than manual instrumentation. The run is shaped like the
pipeline it is: a `rag-eval` root span, then per question an `answer-question` span holding a
`retrieve-context` retriever observation and a `generate-answer` generation. Verified by
fetching the live trace back through the SDK and checking its shape against the expected
61 observations, not by assuming it worked.

One detail worth keeping: because the model rejects `temperature=0`, the naive
try-then-retry fallback would record a failed-then-retried generation inside the trace. When
tracing, the harness settles temperature support with one tiny untraced probe first, so the
trace carries no spurious error spans.

## Run it

```bash
cd ../01-chunking
.venv/bin/python rag_eval.py run --encoder bge-small --chunk-size 1000 --chunker fixed \
    --top-k 5 --rerank --pool 20 --generate [--trace]
```

Full write-up: [`report.md`](report.md) · Raw run: [`results.json`](results.json)
