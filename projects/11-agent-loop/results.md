# Experiment 11 report: when is a loop worth it?

**Run date:** 2026-07-29 · **Config:** `bge-small · chunk 1000 · fixed · top-5 · rerank pool=20` (L06 best, unchanged) · **Generator:** `gpt-5.6-luna` · **Judge:** `gpt-5.6-terra` · **Prompt:** `hardened` (the L09 fix, which is what ships) · **Cap:** `MAX_STEPS = 3` · **Sets:** golden 20 + trap 5 · **API spend:** $0.58 measured + ~$0.35 lost to a hung first attempt

**The verdict, up front: I would ship it, behind a trigger rather than on every query.** It closed the one defect that survived five lessons, at 5/5 grounded, three times out of three. It regressed nothing. And the reason I'd still gate it is that on 18 of 20 questions it does nothing at all except cost 1.32×, which is also the reason it's cheap enough to be worth having.

**Four of my five preregistered hypotheses were wrong, and they were wrong in the direction that makes the loop look better than I expected.** That is an uncomfortable thing to write, so it's written first. [`hypothesis.md`](hypothesis.md) was committed before a single agent call was made.

The commands:

```bash
# one question, one loop: the 15-minute version
python rag_eval.py run --encoder bge-small --chunk-size 1000 --top-k 5 --rerank --pool 20 \
    --generate --prompt hardened --agent --only overfitting -v

# the full three-arm experiment
python run_agent_loop.py          # writes results/results-agent.json, checkpointed per cell
python summarize.py               # every table below
python read_trajectory.py <golden-trace-id> <trap-trace-id>
```

**Traces:**
[golden `4f6da1a3`](https://cloud.langfuse.com/project/cmrvxaxqt01kxad0jvhtb8dlm/traces/4f6da1a37a12161b48bafa4bcdfc624b) ·
[trap `97742625`](https://cloud.langfuse.com/project/cmrvxaxqt01kxad0jvhtb8dlm/traces/97742625d82f0079c2339f5342c9d698)

---

## Before anything else: there are three arms, because two would have lied

`gpt-5.6-luna` **refuses function tools on `/v1/chat/completions`**:

```
400: "Function tools with reasoning_effort are not supported for gpt-5.6-luna in
       /v1/chat/completions. To use function tools, use /v1/responses or set
       reasoning_effort to 'none'."
```

Two escapes. I probed both before choosing, and the choice mattered more than it looks.

| Escape | What happened on the probe | Verdict |
|---|---|---|
| `reasoning_effort="none"` on chat.completions | Given context reading only *"Cats are mammals"* and asked the boiling point of water, the model **never considered the tool** and answered **"212°F at standard atmospheric pressure"**, straight from pre-training | **A trap.** This is the exact failure the lesson exists to measure, injected by my own harness |
| `/v1/responses` at full reasoning | Called `search_corpus("boiling point of water in Fahrenheit", k=5)` | Correct behaviour |

So the loop runs on `/v1/responses`. Which creates a confound: any difference between "agent" and "what I ship" would be **the tool *and* the endpoint**, tangled. Hence three arms:

| Arm | Endpoint | Tool | What it is |
|---|---|---|---|
| `shipped` | chat.completions | n/a | what L09/L10 measured and what the CI gate runs today |
| `control` | responses | n/a | **the matched control** |
| `agent` | responses | `search_corpus` | the treatment |

`control` vs `agent` isolates the tool. `shipped` vs `control` prices the endpoint switch on its own, and it turns out to be nearly free (cost −0.8%, latency +27%, faithfulness within noise), which is what licenses reading the third column against the first.

**If I had skipped the control arm I would have had a clean-looking table and no idea which column caused what.**

---

## Steps 1 and 2: the query it wrote, which is the diagnosis

One tool, bound to the retriever already in memory ([rag_eval.py](../01-chunking/rag_eval.py)):

```python
def search_corpus(query, k):
    qv = embed_queries(encoder_model, [query], encoder)[0]
    row = qv @ matrix.T
    ...  # identical path to the one that built the initial context, reranker included
```

Local encoder, in-memory index: **the tool costs $0.** The loop's entire bill is extra reasoning turns, which makes this an unusually clean experiment. Every dollar below is autonomy, not infrastructure.

The step-2 query on the overfitting question, across six runs (three in a lost first attempt, three in the recorded one):

> `Overfitting is when a neural network learns the training data too well and performs poorly on unseen data`
> `Overfitting occurs when a neural network memorizes the training data and performs poorly on new unseen data.`
> `Overfitting is when a neural network learns the training data too well, including noise, and performs poorly on unseen validation or test data.`

**This is a rephrase, not a diagnosis**, and my H1 called that correctly. The model did not name the missing mechanism (the train/validation accuracy gap L10 identified as the starved fact). It wrote out **the answer it already believed** and went looking for it.

By the reasoning in `hypothesis.md`, that should have failed. It didn't, and the reason is the interesting part.

### Where the rewrite actually landed

| | file | result |
|---|---|---|
| step 1 (the question, as asked) | `day26-deep-learning.md` ×3 in top-5 | the file L10 blamed. Right file, wrong window, with 1 chunk of 734 holding the evidence |
| step 2 (the model's rewrite) | `day22-finetuning.md`, `day23-oss-finetuning.md`, `day27-deep-learning-2.md` | **3 of 5 chunks new.** Never returned to day26 at all |

The rewrite **abandoned the file everyone assumed was the answer** and found the fact somewhere else entirely, in the *fine-tuning* docs, where the sentence is written out literally:

```
day23-oss-finetuning.md:645   Validation loss increasing while training loss decreases → Overfitting!
day22-finetuning.md:634       Training loss decreases, validation loss increases
```

I confirmed those by `grep`, **independently of the judge**, so the 5/5 below has a receipt that doesn't require trusting the judge.

**The honest caveat.** My tool description tells the model: *"phrase it the way the answer would be written in a document, not the way a question is asked."* That instruction is doing real work here. It's what turns a question-shaped query into an answer-shaped one, which is what reached the prose in day22/day23. **Part of the credit for that query belongs to a prompt I wrote, not to the loop's autonomy.** A fair reading is that the loop supplied the *second attempt* and my tool description supplied the *strategy*.

---

## Step 3: did it close? The overfitting question, both ways, 3 runs each

| arm | abstained | faithfulness | grounded claims | mean latency | cost/query | steps |
|---|---|---|---|---|---|---|
| `shipped` | **1, 1, 1** | n/a (zero-claim) | n/a | 1,367 ms | $0.000875 | 1, 1, 1 |
| `control` | **1, 1, 1** | n/a (zero-claim) | n/a | 1,051 ms | $0.000747 | 1, 1, 1 |
| **`agent`** | **0, 0, 0** | **5/5 · 5/5 · 5/5** | **1/1 · 3/3 · 2/2** | 4,010 ms | $0.003427 | 2, 2, 2 |

> **It closed. 3/3, at perfect faithfulness, every claim grounded.**

The three answers, and the judge's claim-level verdicts:

- *"Overfitting occurs when a neural network memorizes the training data rather than learning patterns that generalize to new data."* 1/1 grounded
- *"…Its training loss decreases while its validation loss increases, indicating poorer performance on unseen data."* 3/3 grounded
- *"Overfitting occurs when training loss decreases while validation loss increases…"* 2/2 grounded

**H2 was wrong, and it was the prediction I was most confident about.** I predicted 1/5 or 2/5: parametric rescue wearing a costume, an answer that is right because the model knows it rather than because the context said it. I wrote that a rescue that isn't grounded is worse than the abstention it replaces, because the abstention was honest.

That reasoning still stands. It just doesn't apply, because **this rescue is grounded.** The evidence was in the corpus the whole time, in a file no query derived from the original question had ever reached. What L05, L06, L09 and L10 all read as "the evidence is in one chunk of day26 and retrieval can't reach it" was **half the picture**: the evidence is also in day22 and day23, and one rewritten query walks straight to it.

That reframes the L10 work-queue item. It was filed at **rung 2, a chunking defect**. It is really a **query-formulation defect**: the corpus was never the problem, the single fixed query was.

---

## Step 4a: the other nineteen

Golden set, 20 questions, one run per arm, judge on every answer:

| | `shipped` | `control` | **`agent`** | agent vs control |
|---|---|---|---|---|
| **abstention rate** | 5.0% | 0.0% | **0.0%** | **+0.0%** |
| **mean faithfulness** | 4.95 | 4.84 | **5.00** | **+0.16** |
| claim groundedness | 40/41 (97.6%) | 37/39 (94.9%) | **39/39 (100%)** | +5.1pp |
| **cost / query** | $0.001600 | $0.001588 | **$0.002090** | **1.32×** |
| **mean latency** | 1,146 ms | 1,455 ms | **1,776 ms** | **1.22×** |
| max latency | 1,849 ms | 4,238 ms | 4,337 ms | 1.02× |
| prompt tokens (20 q) | 27,867 | 27,867 | **40,591** | 1.46× |
| output tokens | 913 | 872 | 994 | 1.14× |

**No regression anywhere on the golden set.** Faithfulness up, groundedness perfect, abstention unchanged, cost up 32%, latency up 22%.

H3 predicted faithfulness would go *down* (more retrieved context means more surface for ungrounded synthesis) and cost would go up 2 to 3×. Both wrong, and for the same reason, which is the number in the next section.

### One measurement-integrity note I have to flag

`shipped` measured 5.0% abstention and `control` measured 0.0%, with **no tool in either.** That single unit is the overfitting question flipping at `temperature=1`, precisely the decision-boundary behaviour L10 diagnosed and set its band width for. Across today's runs that question abstained 6/6 in isolation and answered 1/1 inside the 20-question set.

**So "abstention 0.0%, unchanged" is a delta measured against a baseline that is itself ±5% at n=20.** The agent arm's 0.0% is not evidence it avoided false abstention; it's consistent with that, and the honest statement is *no detectable change on an instrument that cannot resolve one question*. The faithfulness and trap numbers below are the load-bearing ones.

---

## Step 4b: the traps, and the hypothesis I got wrong

**Stated before the run, in [`hypothesis.md`](hypothesis.md):**

> **H4. A second chance at searching helps the model talk itself into an answer.**
> Trap hallucinations: **0/5 → 2/5 or worse.** I'll call it a confirmed failure at ≥ 1 and a surprise at 0.

Measured:

| arm | run 1 | run 2 | run 3 | mean steps | max | cap rate | cost/q |
|---|---|---|---|---|---|---|---|
| `shipped` | 0/5 | 0/5 | 0/5 | 1.00 | 1 | n/a | $0.000674 |
| `control` | 0/5 | 0/5 | 0/5 | 1.00 | 1 | n/a | $0.000654 |
| **`agent`** | **0/5** | **0/5** | **0/5** | **2.20** | **3** | **20%** | $0.003386 |

> **0/15. The surprise case. H4 is wrong, and not marginally.**

The loop searched **twice as hard on the traps as on the golden set** (2.20 steps vs 1.10) and abstained every single time. My three stated reasons, that more plausible context lowers the bar, that sunk cost reads as a reason to answer, and that the forced-answer step pushes toward answering, were all wrong.

**Why, and this is the finding of the lesson.** Look at what it searched for:

```
[ABSTAINED] "Chroma uses HNSW approximate nearest neighbour index under the hood
             and HNSW trades recall for speed"
[ABSTAINED] "An MCP server exposes resources, prompts, and tools to clients, and
             the protocol uses JSON-RPC over a transport"
[ABSTAINED] "Faithfulness measures whether the generated answer is supported by the
             retrieved context, calculated as the proportion of..."
```

**The model wrote its own parametric belief into the query, the exact hallucination L09 caught it producing, searched the corpus for it, got nothing back, and treated that as evidence to abstain.**

In L09, that same belief came out as a confident answer: *"Chroma uses an HNSW (Hierarchical Navigable Small World) graph index, commonly implemented through hnswlib."* True, verifiable, and absent from all 1,294 tokens of its context. Today the identical belief became a **search query instead of an answer**, the search failed, and the failure was read correctly.

That is the counter-case I wrote down as the reason to run this rather than assume it: the hardened prompt's rule 1 is **evidence-conditioned**, not effort-conditioned. A model that actually applies it should abstain *harder* after searching, because it now has direct evidence that a targeted query for the fact came back empty. **It does.** The loop converted a hallucination into a falsifiable hypothesis and then falsified it.

The single strongest data point: the **BM25 trap hit the cap**, with two searches, `terminated_by="cap"`, forced to answer with `tool_choice="none"`, **and still abstained.** Maximum effort, option to keep looking removed, and it declined anyway.

---

## Step 5: the trajectory numbers

Emitted next to L10's `abstained` on the same span ([rag_eval.py:1047](../01-chunking/rag_eval.py#L1047)):

```python
qspan.score(name="steps", value=gen["steps"], data_type="NUMERIC",
            comment=f"{gen['terminated_by']} · {gen['n_searches']} search(es)")
```

`NUMERIC`, not `BOOLEAN`, because the questions worth asking are *"what's the mean"* and *"which ones hit the cap"*, and both need a number that averages. Read **back out of Langfuse** via `GET /api/public/traces/{id}`, not trusted from local memory ([`read_trajectory.py`](read_trajectory.py)):

| | golden (20) | trap (5) |
|---|---|---|
| **mean steps** | **1.10** | **2.20** |
| **max steps** | **2** | **3** |
| **% `terminated_by = cap`** | **0%** (0/20) | **20%** (1/5) |
| searched at all | **2/20** | 5/5 |
| mean `abstained` | 0.000 | 1.000 |

### Mean steps 1.10 is the number the verdict turns on

**18 of 20 questions never called the tool.** The two that did:

```
steps=2  answered  What is overfitting in a neural network?
steps=2  answered  What does reranking do in a RAG pipeline?
steps=1  answered  (the other eighteen)
```

Those two are **exactly** the two `rr = 0` retrieval misses that have been on the books since L06, and the two questions L09's judge split apart. The loop fired on precisely the questions retrieval failed on, and on nothing else. **Zero false fires in 18 opportunities.**

That is why cost came in at 1.32× instead of the 2–3× I predicted: **you only pay for the loop where the loop does something.** A reflexive loop searching every question would have cost ~2× and bought nothing on 18 of them. I predicted mean steps 1.2–1.5 and wrote that ">2 means it's searching reflexively and I'm paying for a habit." At 1.10 it is more discriminating than my optimistic case.

### The join is the payoff for putting both scores on one span

Neither score says this alone:

| `steps` | `abstained` | reading | seen today |
|---|---|---|---|
| 1 | 1 | retrieval starved it and it knew immediately | n/a |
| **3** | **1** | **looked twice, still declined → a real corpus gap** | the BM25 trap |
| 2–3 | 0 | looked, found it, answered → **a rescue** (check faithfulness) | overfitting, reranking |
| 2–3 | 0 | looked, found nothing, answered anyway → **parametric rescue** | **0 occurrences** |

That last row is the one this lesson was built to find. It is empty today, and now it has a query I can run every night.

---

## The verdict: would I ship it?

**Yes, gated behind a trigger. Not on by default, and not off.**

**What earns the yes:**

1. It closed a defect that survived the header chunker (L05), the cross-encoder rerank (L06), prompt hardening (L09) and diagnosis (L10), at **3/3, 5/5 faithfulness, every claim grounded, verified by grep independently of the judge.**
2. **It did not cost me the 0/5.** Three runs, fifteen traps, zero hallucinations, while searching twice as hard as on answerable questions. The regression I preregistered did not happen.
3. Faithfulness went **up** (4.84 → 5.00, groundedness 100%).
4. It is **discriminating**: 2 fires in 20 questions, both correct, zero false fires.

**What earns the gate:**

1. **On 18 of 20 questions it is pure overhead**, at 1.32× cost and 1.22× latency for a call that changes nothing. The lesson's own rule is *"add complexity only when it demonstrably improves outcomes."* It demonstrably improves outcomes on **10% of my golden set**, and I know which 10%.
2. **Latency is now a distribution, not a constant.** The p50 barely moved; the tail is 4.3 s. For a per-commit gate that's irrelevant. For a user waiting on an answer it is the number that matters, and I now have to quote a p95 where I used to quote a mean.
3. **n = 20 and n = 5, one corpus, one model, one day.** Three trap runs are enough to say "not a flicker" (L09's standard) and nowhere near enough to say "never." The 0/5 is a **measured absence of the failure, not proof of its impossibility**, and the traps are public-corpus traps where the model has pre-training to fall back on. On InsureElm's private PDFs (the [`test-corpus/`](../09-answer-defense/test-corpus/) still sitting unbuilt from L09) the same query-writing behaviour has nothing to retrieve *and* nothing to hallucinate from, and I do not know what it does there.
4. **The tool description is load-bearing and untested as a variable.** "Phrase it the way the answer would be written in a document" plausibly did as much work as the loop. I changed two things at once, a loop and a query-formulation instruction, and only measured the pair.

**The shape I'd actually ship:**

> Run the fixed chain. If the answer is the abstention sentinel **and** `steps == 1`, re-run once with the tool. Everything else stays a workflow.

That buys the entire measured win (both fires today were on questions where the chain was starved) at ~10% of the added cost, keeps the median path a testable fixed chain, and leaves `terminated_by` and `steps` on the trace as the health metrics. **The autonomy is worth paying for exactly where I couldn't write the path down, and I can write down 18 of 20 of them.**

### What would change my mind, in numbers

| Reading | Threshold | Action |
|---|---|---|
| `steps ≥ 2 AND abstained = 0 AND faithfulness ≤ 2` | **any occurrence** on a nightly judge pass | parametric rescue is real → turn the loop off, it is L09's failure with more steps |
| trap hallucinations | **> 0 on any run** | off immediately; this is the CI gate and it does not negotiate |
| `% terminated_by = cap` on live traffic | **> 20% sustained** | the cap is doing the terminating, not the model → raise it or fix retrieval, don't ship a truncation |
| mean steps on golden | **> 1.5** | it has become reflexive; the 1.32× stops being a bargain |

---

## What it cost, including the part I wasted

| Cell | Calls | Cost |
|---|---|---|
| Q. overfitting ×3 × 3 arms (+judge) | 21 gen + 9 judge | $0.0502 |
| G. golden 20 × 3 arms (+judge) | 62 gen + 60 judge | $0.4598 |
| T. trap 5 ×3 × 3 arms | 56 gen | $0.0707 |
| **Total (`results/results-agent.json`)** | **208 calls** | **$0.5807** |

Wall clock 654.6 s. Plus **~$0.35 lost** to the first attempt, which is the next section.

The loop's own share: on the golden set the agent arm cost **$0.0418** of generation against the control's **$0.0318**, so **$0.010 for the whole 20-question pass**, of which effectively all of it bought two questions. **Roughly half a cent per rescue.** The faithfulness judge that grades it still costs $0.115 per pass, so **the guardrail remains 3× the thing it guards**, unchanged from L09's finding.

---

## Two things the harness didn't have, and now does

Both came out of a failure, not a design review.

**1. `REQUEST_TIMEOUT_S = 240` / `max_retries=3`.** The first full run **wedged 12 questions into the agent golden pass**, sat at 0% CPU for ~50 minutes holding three ESTABLISHED sockets, blocked on a socket read that was never coming back. The SDK default is a 600 s timeout with retries: up to half an hour of silence per call, and no bound at all on a run.

> `MAX_STEPS` caps how many calls a question can make. It says nothing about how long one call may take. **They are the same guardrail, unbounded consumption, and the step cap only covers the half that shows up on the invoice.** The half that shows up as a wedged pipeline needs a client timeout, and I did not have one.

That is OWASP LLM10 territory that the lesson's guardrail table doesn't spell out, and I found it by having it happen.

**2. Checkpoint after every cell.** The first driver wrote `results-agent.json` once, at the end. The hang meant **every paid call before it was thrown away**: the Q phase and two golden arms, about $0.35, gone. Results now land on disk per cell and `--resume` skips what survived. **A run you can only read if it finishes is a run you pay for twice.**

Neither is about agents specifically. Both got materially worse *because* of the loop: a fixed chain makes one call per question, so a hang costs you one question and the blast radius of "write at the end" is small. A loop makes an unknown number of calls over an unknown wall clock, and both failure modes scale with it.

---

## Where my hypotheses landed

| | Predicted | Measured | |
|---|---|---|---|
| **H1** query is a rephrase, not a diagnosis | rephrase (65/35) | **rephrase**, since it searched for the answer it already believed | ✅ |
| **H2** the rescue is parametric | 1/5 or 2/5 | **5/5, three times, every claim grounded** | ❌ |
| **H3** faithfulness down, cost 2–3×, steps 1.2–1.5 | regression | **faithfulness +0.16, cost 1.32×, steps 1.10** | ❌ |
| **H4** the traps break | 2/5 or worse | **0/5, 0/5, 0/5** | ❌ |
| **H5** ship it off by default | "buys nothing, regresses the gate" | **buys the L09 residual, regresses nothing** | ❌ |

**One of five.** The one I got right is the one about the model's behaviour; the four I got wrong were all predictions that the mechanism would fail. I was primed by L08's fine-tune trap and L09's parametric-rescue finding to expect the sophisticated-sounding thing to lose, and I wrote that expectation down as four separate hypotheses. It won.

The instruments are why that's a finding rather than an embarrassment: **every one of those was falsifiable before it was tested, and the judge, the trap set and the trace are what falsified them.** L08's lesson was don't reach for the expensive mechanism on a hunch. The symmetric lesson, which I needed today, is **don't reject it on one either.**

---

## Files

| File | What |
|---|---|
| [`hypothesis.md`](hypothesis.md) | the five predictions, written before the first agent call |
| [`../01-chunking/rag_eval.py`](../01-chunking/rag_eval.py) | `MAX_STEPS`, `SEARCH_TOOL`, `AGENT_ADDENDUM`, `make_search_corpus()`, `run_agent()`, `--agent/--only/--api/--out`, the `steps` score, `REQUEST_TIMEOUT_S` |
| [`run_agent_loop.py`](run_agent_loop.py) | the 9-cell three-arm driver, checkpointed per cell |
| [`summarize.py`](summarize.py) | every table above, read-only, no spend |
| [`read_trajectory.py`](read_trajectory.py) | reads `steps` × `abstained` back out of Langfuse and joins them |
| [`results/results-agent.json`](results/results-agent.json) | every answer, query, claim verdict, token count, latency and cost |
| `results/run-attempt1-hung.log` | the hung run, kept, because it's the evidence for the timeout guardrail |

Retrieval-only runs are unchanged and still free: MRR **0.785**, 2/20 misses, identical to L06–L10.

---

## What's still open

1. **The two changes are entangled.** The loop and the "phrase it as a document sentence" tool description shipped together. The clean follow-up is a third query-formulation variant (a bare *"search again with a better query"*) to see how much of the win is the instruction. One golden pass, ~$0.16.
2. **The private-corpus test is still unbuilt**, being three InsureElm PDFs from L09 that still need a PDF loader. It's now more interesting than it was: the whole trap result rests on the model *having* a parametric belief to turn into a query, and on that corpus it has none.
3. **The trigger isn't implemented.** The verdict above proposes `abstained AND steps == 1 → retry with tool`, and I measured the always-on version. Shipping the gate means measuring the gate.
4. **`n = 5` traps, three runs.** Enough for "not a flicker," not enough for "never." The cap-terminated BM25 trap is the one to watch, because it is the closest any question came to the failure mode, and it is one question.
5. **The judge is still unvalidated at scale** (L09's open item, unchanged). Today it is loaded even more heavily: the entire "grounded rescue, not parametric rescue" claim rests on it. The grep receipt at `day23:645` covers the overfitting question specifically, and nothing else has that backup.
