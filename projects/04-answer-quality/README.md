# 04. Reading the chunks by hand

**Question:** where does the keyword ruler disagree with a human reader?

**Answer:** on the question I had already written off. Reading the retrieved text overturned
my own earlier verdict.

## Why do this by hand

The harness reports `MISS  cov 33%  What is overfitting in a neural network?` and nothing
else. That is the ruler's verdict, not what the ruler read. Before automating a judge, I
wanted to know what a judge would be judging, so I built the one tool the harness was
missing and used my eyes.

`show_chunks.py` prints the text of the retrieved chunks. It reuses experiment 01's chunker,
encoders and embedding cache, so every number matches `rag_eval.py run` exactly and nothing
re-embeds.

## What I found

I read the top 5 for three questions and wrote the disagreements down in
[`report.md`](report.md). The important one: the `overfitting` miss is not a chunking failure
in the way I had assumed. The correct explanation was being retrieved, ranked outside the
top 5. That reclassified it from a coverage problem to a ranking problem, which is what sent
experiment 06 at a cross encoder rather than at a smaller chunk size.

Two related metrics fall out of the exercise. Keyword coverage measures whether the right
words are present. It cannot see a passage that says the right thing in different words,
which is exactly the failure mode in experiment 05, where a chunk explaining that validation
loss rises while training loss falls scored a miss because the golden keywords said
"accuracy".

## Honestly incomplete

Task 2 of [`worksheet.md`](worksheet.md), constructing a false positive, is blank. I ran out
of session and chose to leave it visibly unfinished rather than fill it in later and present
it as contemporaneous work.

## Run it

```bash
../01-chunking/.venv/bin/python show_chunks.py --list
../01-chunking/.venv/bin/python show_chunks.py -q overfitting -q "prompt caching" -q reranking
```

Full write-up: [`report.md`](report.md)
