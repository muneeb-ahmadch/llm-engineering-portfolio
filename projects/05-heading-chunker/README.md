# 05. Heading aware chunking

**Question:** if I split on markdown headings instead of every 1000 characters, does the
known `overfitting` miss become a hit?

**Answer:** no. It got substantially worse, and the reason is more useful than a fix would
have been.

## The hypothesis

Isolate the section that actually answers the question, and its vector stops being polluted
by the CNN training code it was fused to. A cleaner vector should rank higher.

## What happened

| | Chunker | Full corpus rank of the target chunk |
|---|---|---|
| Before | `fixed` | #8 of 734 |
| After | `headings` | **#67 of 1499** |

Half the hypothesis was correct. The `Monitoring Training` chunk really is cleaner: 540
characters of metrics and warning signs instead of a block fused to unrelated code. But
heading splitting does not only clean up the target. It fragments the entire corpus, from
734 chunks to 1499, mean length 985 down to 464.

This corpus was built with deliberate topic overlap, and `overfitting` is discussed in at
least five files. Under `fixed`, those mentions were diluted inside larger mixed topic chunks
that did not compete hard on cosine similarity. Under `headings`, every one of them becomes
its own small, topically pure chunk: `Preventing Overfitting`, `Pitfall 3: Overfitting`,
`Dropout`, `Weight Decay`. All of them now outrank the target, because each is a tighter
match to the literal words "what is overfitting" even though none contains the evidence the
golden question asks for.

**Concentrating signal helped the target chunk. It helped 66 competitors more.**

## The aggregate, which hides all of that

| Chunker | Chunks | MRR | vs fixed | Coverage | Misses |
|---|---|---|---|---|---|
| `fixed` | 734 | 0.754 | +0.000 | 95.0% | 2/20 |
| `headings` | 1499 | 0.767 | +0.013 | 95.0% | 3/20 |

MRR went up by 0.013, inside the noise floor, while the specific question I was trying to fix
went backwards by 59 positions and the miss count went up. An aggregate metric that moves
0.013 in the right direction while the thing you care about degrades is the argument for
per-question diffs, which is why [`results.md`](results.md) carries one.

One more thing worth noting from the top 5 under `headings`: rank #5 is a passage saying
"training loss decreases, validation loss increases". That is a correct answer in different
words, scored a miss only because the golden keywords say "accuracy". The measurement problem
from experiment 04 reappearing, this time in the runner up.

## Run it

```bash
cd ../01-chunking
.venv/bin/python rag_eval.py run --encoder bge-small --chunk-size 1000 --top-k 5 --chunker fixed -v
.venv/bin/python rag_eval.py run --encoder bge-small --chunk-size 1000 --top-k 5 --chunker headings -v
```

Full write-up: [`results.md`](results.md) · Top 5 under each chunker: [`report.md`](report.md)
