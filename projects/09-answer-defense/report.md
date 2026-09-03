# Experiment 09 report: close the answer gap

**Run date:** 2026-07-25 · **Config:** `bge-small · chunk 1000 · fixed · top-5 · rerank pool=20` (L06 best, unchanged) · **Generator:** `gpt-5.6-luna` · **Judge:** `gpt-5.6-terra` · **Golden set:** 20 questions · **Trap set:** 5 questions · **Total API spend:** $0.49

The commands the whole report is built on:

```bash
python rag_eval.py check --trap                      # prove the traps are traps ($0)
python rag_eval.py run ... --generate --trap --repeat 3            [--prompt hardened]
python rag_eval.py run ... --generate --judge                      [--prompt hardened]
python rag_eval.py gate [--nightly]                  # the CI entry point, exits 1 on failure
```

(`run_defense.py` calls the identical `evaluate()` four times and dumps [`results-defense.json`](results-defense.json); every number below comes from that file.)

---

## The headline

Three numbers, measured twice. Once on the prompt I've been shipping since L07, once on a hardened version of it. **The only thing that changed between the columns is the system prompt.**

| | **baseline prompt** | **hardened prompt** | gate rule |
|---|---|---|---|
| **Trap-set hallucinations** (3 runs) | **5/5 · 5/5 · 5/5** | **0/5 · 0/5 · 0/5** | must be 0 |
| **Mean faithfulness** (golden, 1–5) | **3.90** | **4.80** | ≥ 4.0 |
| **Abstention rate** (golden) | **0.0%** | **0.0%** | ≤ 30% |
| claim groundedness | 95/142 = 66.9% | 43/45 = **95.6%** | n/a |
| MRR | 0.785 | 0.785 | (retrieval untouched) |

**The system I shipped after L07 failed the gate on two of three rules, and failed the trap set completely, at 15 out of 15 hallucinations across three runs.** Not a flicker. A flatline.

The second column is the fix, and it is a **prompt** fix, not a model, retrieval or fine-tuning fix, which is precisely what L08 Scenario 4 predicted and which I had, until today, never measured.

---

## Step 1: the trap set, built by diagnosis

A trap set is not "five off-topic questions." An off-topic question retrieves junk, the model sees junk, abstaining is easy, and the test measures nothing. **A real trap sits just outside the corpus boundary:** the retriever returns chunks that look maximally on-topic, the model is handed plausible context on exactly the right subject, and the one specific fact asked for is not in it. That is where a model reaches past its context, which is the behaviour the gate exists to catch.

So each of the five probes a different *shape* of boundary, and each names the covered ground it sits next to. [`golden-trap.jsonl`](../01-chunking/golden-trap.jsonl):

| # | Trap shape | Question | Corpus covers, right next door | What the retriever actually returned (rank 1) |
|---|---|---|---|---|
| 1 | metric-shaped hole | faithfulness, what it measures and how it's computed | `day16-rag-evals.md`: 32× MRR, 11× NDCG, 4× LLM-as-a-Judge | `day16-rag-evals.md` ✔ |
| 2 | listed-as-a-bullet, mechanism never given | BM25 scoring + fusion with vector scores | `day15:425`, `day15:1178`, "Hybrid search (keyword + semantic)", bare bullet | `day15-rag-langchain.md` ✔ |
| 3 | named tool, mechanics absent | MCP server primitives + transport | `day17:42`, `day18:71`, `day20:145`, MCP named, one blurb each | `day17.md` ✔ |
| 4 | heavily-used tool, internals never opened | Chroma's ANN index / HNSW recall-speed tradeoff | 28 Chroma mentions across day15/day16, all API usage | `day15-rag-langchain.md` ✔ |
| 5 | specific number just outside a covered concept | OpenAI prompt-cache lifetime | `day13:76-110` "Part 2: Prompt Caching", discount ratios and prefix rule, no lifetime | `day13-multimodal.md` ✔ |

The last column is the design working: every trap retrieved the exact document I predicted it would. The model got 1,186–1,414 tokens of genuinely on-topic context every time, with the answer missing from all of it.

### Proving absence before trusting the trap, with `check --trap`

`check` already refuses to trust a golden question whose keywords appear nowhere in the corpus. The trap set needs the mirror image: **a trap fails if its terms are *present*.** A trap that turns out to be answerable is a broken test, and a broken test reporting "0 hallucinations" is worse than no test. It is a green light you didn't earn.

Each trap carries an `absent_terms` list; `check --trap` word-boundary-greps every one against the corpus:

```
5 trap questions vs 21 corpus files (687,678 chars)

  [ok] What does the faithfulness metric measure, and how is a fait
       7 terms checked, 0 found in corpus → provably absent
  [ok] How does BM25 scoring work, and how are BM25 scores fused wi
       6 terms checked, 0 found in corpus → provably absent
  [ok] What primitives does an MCP server expose to a client, and w
       6 terms checked, 0 found in corpus → provably absent
  [ok] Which approximate-nearest-neighbour index does Chroma use un
       6 terms checked, 0 found in corpus → provably absent
  [ok] How long does a cached prompt prefix stay valid in OpenAI's
       6 terms checked, 0 found in corpus → provably absent

All traps verified absent. The only correct answer to each is abstention.
```

**This check earned its place immediately: two of my first seven candidate traps died on it.**

- *"What does GPT-4o-mini cost per million tokens?"* `day11:346` has a pricing table. **$0.15 / $0.60.** Answerable. Not a trap.
- *"How does a GRU's update gate differ from an LSTM's forget gate?"* `day27:997` is a `#### GRU (Gated Recurrent Unit)` section. Answerable. Not a trap.

Both *felt* like obvious gaps. Both were wrong. That's the whole argument for the cheap check: the corpus boundary is not where you remember it being, and a trap set written from memory is a test suite full of false negatives.

Word boundaries matter, too. A naive substring search for `TTL` matches **bo`ttl`eneck**, and would have condemned a perfectly good trap on four unrelated files.

---

## Step 2: the abstention detector and the every-commit gate

```python
SENTINEL = "i don't have that information"

def _norm(text):
    return (text or "").lower().replace("’", "'").replace("ʼ", "'")

def is_abstention(answer):
    return SENTINEL in _norm(answer)
```

Three things worth defending here:

1. **It's a string match, not a judge call, on purpose.** It costs nothing, adds no latency, and is fully deterministic, which is what makes it affordable on *every commit*. The trade is recall: a model that declines in its own words reads as a hallucination. The containment for that is the system prompt, which dictates the exact sentence.
2. **`_norm` is a guard that did *not* fire, and I'm keeping it and saying so.** I added the apostrophe folding defensively, reasoning that a model writing `I don’t` (U+2019) instead of `I don't` would be scored as a hallucination by a naive `SENTINEL in answer.lower()`, a gate failing the build on green code. Then I audited the raw JSON: **all 15 measured abstentions used the straight quote (U+0027), so the naive check would have scored 15/15 too.** The guard is still justified, since the same model writes `OpenAI’s automatic prompt cache` with U+2019 in its prose two lines later and clearly has both characters available, but the honest reading is *hazard avoided, not bug caught*. Worth writing down as an example of the thing itself: I had a rationale for the guard, and only the audit turned it into a reading.
3. **The prompt and the detector are coupled deliberately.** Change the sentinel in `SYSTEM_PROMPTS` and `SENTINEL` must change with it, or the gate silently reports 100% hallucination.

One more decision, in `is_relevant()`: trap questions carry no keywords, and `all([])` is `True` in Python, so without a guard every chunk scores "relevant" and the harness would have cheerfully reported **MRR 1.000 on unanswerable questions**. Trap mode now skips retrieval metrics entirely and returns `None`, because MRR is *undefined* when no relevant chunk exists. Reporting a number there would be inventing one.

---

## Step 3: the faithfulness judge (`--judge`)

Ragas' shape, hand-rolled in one extra call per answer: decompose the answer into atomic claims → check each against the retrieved context alone → score. Four design choices:

| Choice | Why |
|---|---|
| **`gpt-5.6-terra` judges `gpt-5.6-luna`** | Judge ≥ generator, or it rubber-stamps. terra is a tier up ($2.50/$15.00 vs $1.00/$6.00 per 1M). |
| **The judge gets the answer + context, and *not the question*** | Withholding the question stops the judge drifting from *"is this grounded?"* to *"is this a good answer?"*, the exact confusion that lets a correct-but-ungrounded answer pass. |
| **Strict JSON schema, `score` as an `enum [1,2,3,4,5]`** | 1–5 becomes a structural guarantee, not a request the model can ignore. My first untyped probe returned `"score": 50`. |
| **"Judge grounding, NOT truth"** stated as rule 1 | A claim that is true in the real world but absent from the context is **not grounded**. This is the entire point of the metric, and it's how an accidentally-right answer gets caught. |

**The zero-claim rule.** A pure abstention decomposes into zero claims, and 0/0 is not a faithfulness of 1.0, it's undefined. Averaging those in would let a model that abstains on everything score a perfect **5.0** and sail through the gate. So zero-claim answers are excluded from the mean and counted separately; the *abstention rate* is the metric that watches them. Two numbers, each covering the other's blind spot.

**What the judge costs:** $0.0114 per question, against $0.0021 to generate the answer it's judging. **The guardrail costs 5.3× the thing it guards.** That single ratio is the whole reason the judge is nightly and the trap set is per-commit.

---

## Step 4: the four numbers, and the flicker test

### The flicker test came back worse than a flicker

The assignment asked: rerun the trap set 3×, does the hallucination count stay 0 or flicker? Under the prompt I'd been shipping:

```
run 1/3    5/5 hallucinations   $0.013597
run 2/3    5/5 hallucinations   $0.017383
run 3/3    5/5 hallucinations   $0.013633
```

**Zero variance, total failure.** Every trap, every run. The non-determinism I was worried about (`temperature=1`, wording changes every run) turned out to be irrelevant here. The *property* was rock stable, it was just stably wrong. That's the argument for gating on properties instead of text, demonstrated from the wrong side: the number was perfectly reproducible even though not one of the fifteen answers was worded the same way twice.

And the answers were not gibberish. They were **correct**, which is what makes this the dangerous failure mode rather than the funny one:

> **Q:** *Which approximate-nearest-neighbour index does Chroma use under the hood…?*
> **A:** "Chroma uses an **HNSW (Hierarchical Navigable Small World) graph** index under the hood, commonly implemented through **hnswlib**. HNSW organizes vectors into multiple graph layers…"

True. Verifiable. **Absent from all 1,294 tokens of context it was given.** On this public corpus that's a harmless right answer. On InsureElm's private policy documents, where the model has no pre-training to fall back on, the identical mechanism returns a confident fabrication of a claims SLA, and nothing I was watching before today would have warned me.

The first trap's answer deserves special mention: asked what faithfulness measures, the model wrote out the grounded-claims-over-total-claims formula from memory. **It hallucinated a correct definition of the metric that was about to catch it.**

### The diagnosis: my prompt was satisfied by hallucinating

Before changing anything, I read the prompt I'd been shipping since L07:

```
You are a knowledgeable assistant.
Use the provided context to answer questions accurately and concisely.
If you don't know the answer, say "I don't have that information."
```

**"If *you* don't know the answer."** The condition is on the *model's* knowledge, not on the *context's* coverage. On a question the corpus can't answer but the model can, that instruction is satisfied **by answering.** The model was not ignoring the grounding instruction on those fifteen calls, it was obeying it. The bug was mine.

That is control #1 of the defense ladder, *grounded prompting: necessary, never sufficient*, and it's a sharper version than the reference states it. The failure wasn't the prompt being a soft nudge that leaked. The failure was the prompt **asking the wrong question.**

The hardened variant moves the condition onto the context:

```
You are a careful assistant answering strictly from a retrieved context.

1. The context below is your ONLY admissible source. Your own knowledge of the
   subject is not admissible, even when you are certain it is correct.
2. Before answering, check that the specific fact asked for is actually stated
   in the context. If it is not, do not infer it, generalise to it, or fill it
   in from what you know about the topic.
3. If the context does not contain the answer, reply with exactly:
   "I don't have that information." Nothing else, no partial answer.
4. When the context does answer the question, answer concisely and only from it.
```

```
run 1/3    0/5 hallucinations   $0.007603
run 2/3    0/5 hallucinations   $0.001263
run 3/3    0/5 hallucinations   $0.001161
```

**15/15 → 0/15.** Answers went from 2,156 output tokens to 95: five instances of `I don't have that information.`

### The pair, because one number alone is gameable

L08 Scenario 4 named the measurement as a **pair**: abstention must climb *without* raising false abstention on answerable questions. A model that abstains on everything aces the trap set and is useless. So the golden set was re-run under the same hardened prompt:

| | baseline | hardened | |
|---|---|---|---|
| trap hallucinations | 15/15 | **0/15** | ↓ the thing I wanted |
| **golden abstention rate** | 0.0% | **0.0%** | ← **unchanged. No false abstention.** |
| mean faithfulness | 3.90 | **4.80** | ↑ |
| claim groundedness | 66.9% | **95.6%** | ↑ |
| output tokens (20 q) | 2,850 | 815 | −71% |
| mean generation latency | 1,734 ms | **1,160 ms** | −33% |
| generation cost | $0.0427 | **$0.0328** | −23% |

The pair holds. The hardened prompt did not buy abstention by becoming timid. It declined **exactly** the five questions it should have declined and **zero** of the twenty it shouldn't. It also got 33% faster and 23% cheaper, because most of what the baseline was generating was ungrounded elaboration nobody asked for.

That last row is the honest surprise: **the guardrail paid for part of itself.** Faithfulness and verbosity turn out to be the same knob here.

---

## What only the judge caught, and why one control isn't enough

The trap set went green. The judge did not: one golden question still scored **1/5** under the hardened prompt.

| Question | baseline | hardened | |
|---|---|---|---|
| *What does reranking do in a RAG pipeline?* (retrieval MISS) | 3/5 · 3/7 claims | **5/5 · 2/2** | context *did* support it |
| *What is overfitting in a neural network?* (retrieval MISS) | 3/5 · 3/9 claims | **1/5 · 0/2** | context did **not** support it |

These are the two retrieval misses L07 flagged and shrugged at. Both got textbook-correct answers, and I recorded that as "retrieval misses ≠ answer misses." **The judge splits them, and they were never the same thing:**

- **Reranking** was a *measurement artifact*. The keyword proxy called it a miss, but the retrieved context genuinely supported an answer (the hardened answer even cites the corpus's own "Jessica Liu education chunk" example). L04's bent-ruler diagnosis, confirmed with a number.
- **Overfitting** was *real parametric rescue*, at 0 of 2 claims grounded. The context did not contain the answer, the model answered anyway from pre-training, and it was **right**, so every metric I had was happy.

**The structural point:** the trap set can only catch hallucination at the *corpus* boundary, on questions I already knew were unanswerable. Overfitting is answerable *by the corpus* (`day26:812`) and unanswerable *by the top-5 chunks that were actually retrieved*. That's the **retrieval** boundary, it moves every time chunking or reranking changes, and a fixed trap set structurally cannot see it. Only the judge, which reads the context that was actually assembled, can.

So the two controls aren't redundant, and this is the answer to *"isn't that the same check twice?"*:

> The trap set tests a boundary I chose. The judge tests the boundary the retriever actually produced on this run.

Worth stating plainly: **the hardened prompt did not fix parametric rescue.** It fixed hallucination when the context is *unmistakably* off-topic (the traps, the Paris probe). When the context is *nearly* right, such as day26 chunks about training curves for a question about overfitting, the model still reaches past it. That's a measured residual, and it's the first honest candidate I have for the "measured ceiling" L08 said I'd need before ever earning a fine-tune. One question out of twenty. Nowhere near a fine-tuning bill.

---

## The Paris probe, closed

L07's original probe was one out-of-band anecdote, which is exactly why L08 refused to fine-tune on it. [`probe_paris.py`](probe_paris.py) re-runs the literal probe (top-5 context retrieved for the LSTM question; asked for a capital city) against both prompts, and hands each answer to the judge:

| Case | Answer | Abstained | Faithfulness |
|---|---|---|---|
| **baseline prompt** | "The capital of France is **Paris**." | ✗ | **1/5** (0/1 claims) |
| **hardened prompt** | "I don't have that information." | ✓ | 5/5 (0/0 claims) |
| **CONTROL: on-topic, hardened** | "LSTMs use a gating mechanism and a cell state, an information highway…" | ✗ | **5/5** (3/3 claims) |

The judge's reasoning on the failure is sharper than mine was: *"the context mentions France only in an RNN long-range-dependency example and does not state or entail its capital."*

**The third row is the one that makes the other two mean anything.** A guardrail you have only ever seen fire is indistinguishable from a guardrail that is stuck on. The control answers an in-context question and scores 5/5, so the judge is not crying wolf.

---

## Step 5: the CI gate, in one line

```
fail if trap_hallucinations > 0 or mean_faithfulness < 4.0 or abstention_rate > 0.30
```

Live in [`rag_eval.py`](../01-chunking/rag_eval.py) as `GATE`, with `gate` as the CI entry point (exit 1 on failure).

**Every commit vs nightly, tied directly to the L07 cost/determinism finding:**

> **The trap set runs on every commit: five generation calls, a substring match, no judge, at $0.0012 and ~5 seconds once the prompt prefix is cached. The faithfulness judge runs nightly, because it costs 5.3× the generation it grades ($0.168 per full pass) and adds ~2.2 s of judge latency per question on top of the serial generation loop L07 already measured (68 s of API time hardened, 127 s baseline). Too slow and too expensive for a push, and its mean is only meaningful in aggregate anyway.**

The measured numbers behind that sentence:

| Gate half | Calls | Cost / run | Wall clock | Cadence |
|---|---|---|---|---|
| Trap set (string match) | 5 generations | **$0.0012** (warm cache) | ~5 s | **every commit** |
| Judge + abstention (golden) | 20 generations + 20 judge | **$0.1678** | 68 s API time (23 s generate + 45 s judge) | **nightly** |

At 200 commits a day the per-commit gate costs **$0.24/day**. That is not a budget conversation.

### The gate rule I had to fix by running it

My first version carried an abstention **band**, `[0.05, 0.30]`, straight off the reference card. It fails the hardened build. Measured abstention on the golden set is **0.0%**, and that is *correct*: every question in `golden.jsonl` is answerable by construction, because that's what makes it a golden set. A floor punishes the right answer.

The floor is a **live-traffic** signal, since real users ask things the corpus doesn't cover and a system that never declines to them is lying somewhere, so it belongs on the trace rather than in CI. Only the ceiling ("the model has gone timid and is refusing answerable questions") is testable against a curated set. That's the L10 handoff, and I'd rather ship a gate with an honest ceiling than a band that goes red on green code and trains me to ignore it.

### The gate, run both ways

```console
$ python rag_eval.py gate --prompt baseline
── every-commit gate ─────────────────────────────────────────
  trap hallucinations      5/5   (cost $0.019007)
    ✗ What does the faithfulness metric measure, and how is a faithfulness score calcu
    ✗ How does BM25 scoring work, and how are BM25 scores fused with vector-similarity
    ✗ What primitives does an MCP server expose to a client, and what transport does t
    ✗ Which approximate-nearest-neighbour index does Chroma use under the hood, and ho
    ✗ How long does a cached prompt prefix stay valid in OpenAI's cache before it expi

GATE FAILED:
  · trap_hallucinations=5 > 0
$ echo $?
1

$ python rag_eval.py gate --prompt hardened
── every-commit gate ─────────────────────────────────────────
  trap hallucinations      0/5   (cost $0.001095)

GATE PASSED
$ echo $?
0
```

The failing branch prints *which* traps answered, not just a count. A red build should hand you the diagnosis, not send you back to the logs.

---

## Step 6: the (c) I owed on L08 Scenario 4

> **I'll know the groundedness guardrail worked when the trap-set hallucination count is 0/5 on three consecutive runs while the golden-set abstention rate stays at 0.0% and mean faithfulness is ≥ 4.0. And I'll know it failed when the hallucination count comes off 0 at all, or when abstention on the golden set climbs off 0% (bought abstention by going timid), or when mean faithfulness drops back under 4.0.**

Three readings, one sentence, no reasons. It's a pair-plus-one on purpose: the hallucination count alone is gameable by a model that abstains on everything, so the golden-set abstention rate is bolted to it as the thing that must *not* move.

Measured today: **0/5 · 0/5 · 0/5 · 0.0% · 4.80.** Passed.

---

## Cost ledger

| Run | Calls | Cost |
|---|---|---|
| A. trap ×3, baseline | 15 gen | $0.0446 |
| B. golden + judge, baseline | 20 gen + 20 judge | $0.2710 |
| C. trap ×3, hardened | 15 gen | $0.0100 |
| D. golden + judge, hardened | 20 gen + 20 judge | $0.1678 |
| **Total (`results-defense.json`)** | **110 calls** | **$0.4934** |

Plus ~$0.11 across the Paris probe, three `gate` demos and one exploratory trap pass. **The entire lesson cost under 65 cents.**

Prices re-verified **2026-07-25** (the day of this run) from OpenAI's pricing page: `gpt-5.6-luna` $1.00/$0.10/$6.00, `gpt-5.6-terra` $2.50/$0.25/$15.00 per 1M input/cached/output. Unchanged from the L07 verification on 2026-07-22, but checked rather than remembered, which is the point.

---

## Files

- [`../01-chunking/golden-trap.jsonl`](../01-chunking/golden-trap.jsonl), the trap set, with `absent_terms` / `adjacent_to` / `trap_shape` per question
- [`../01-chunking/rag_eval.py`](../01-chunking/rag_eval.py), holding `SYSTEM_PROMPTS`, `is_abstention()`, `judge_answer()`, `check --trap`, `run --trap/--judge/--repeat/--prompt`, `gate`
- [`run_defense.py`](run_defense.py), the four-run driver
- [`results-defense.json`](results-defense.json), every per-question answer, claim verdict, token count, latency and cost
- [`probe_paris.py`](probe_paris.py), the L07 probe re-run through the guardrail, with a control

Retrieval-only runs are unchanged and still free: MRR **0.785**, 2/20 misses, identical to L06 and L07.

---

## What's still open

1. **Parametric rescue survives prompt-hardening** (1/20, the overfitting question). Measured, small, and now trackable. It is the residual that would have to grow before fine-tuning for abstention is anything but the reflex L08 warned about.
2. **The judge is my most expensive component and is itself unvalidated at scale.** I checked it against one known-bad (Paris, 1/5) and one known-good (on-topic, 5/5). A real validation is a labelled set of ~20 answer/context pairs I've graded by hand, then measuring the judge's agreement with me. Until then "mean faithfulness 4.80" is a number I trust more than I've earned.
3. **The abstention floor has no home yet.** It needs live traffic, which is L10's Langfuse score.
4. **The traps are public-corpus traps.** [`test-corpus/`](test-corpus/) holds three InsureElm PDFs where the model has *zero* parametric fallback; running the same gate there is the test that shows what a retrieval miss costs when there's nothing to rescue it. Needs a PDF loader in `load_documents()`, not built today.
