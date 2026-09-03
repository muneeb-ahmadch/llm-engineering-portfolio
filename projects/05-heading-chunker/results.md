# Experiment 05 results: heading-aware chunker vs fixed-size baseline

Command (both runs): `run --encoder bge-small --chunk-size 1000 --top-k 5 -v` with `--chunker fixed` then `--chunker headings`. overlap=50, relevance=`all`.

## 1. Does overfitting flip MISS → hit?

**NO. Still a MISS**

- `fixed`    → MISS (full-corpus rank #8)
- `headings` → MISS (full-corpus rank #67)

## 2. What rank does the day26 section land at now?

- `fixed`: the chunk holding all three keywords sits at full-corpus rank **#8** of 734 chunks (outside top-5).
- `headings`: the isolated `#### Monitoring Training` section sits at full-corpus rank **#67** of 1499 chunks.
- Movement: rank #8 → #67 (**-59 positions. It got worse, not better.**)

**Why it got worse, not better.** The hypothesis was "isolate the section, its vector points
somewhere honest, it ranks higher." Half of that happened: the `Monitoring Training` chunk
*is* cleaner now, at 540 chars of pure metrics-and-signs bullets instead of being fused to
CNN training code. But heading-splitting doesn't just clean up *that* section, it fragments
the *whole* corpus: 734 chunks → 1499 (mean length 985 → 464 chars). This corpus was built
with heavy topic overlap across days on purpose (see the Lesson-01 README), and "overfitting"
is discussed in at least five files (day14, day22, day23, day26, day27). Under `fixed`,
those mentions were mostly diluted into larger, mixed-topic chunks that didn't compete hard
on cosine similarity. Under `headings`, every one of those mentions becomes its own small,
topically-pure chunk: `#### **Preventing Overfitting**` (day26), `### Pitfall 3: Overfitting`
(day23), `### Dropout` (day23), `### Weight Decay` (day23), and *all of them* now outrank
the `Monitoring Training` chunk, because each is a tighter semantic match to the words
"what is overfitting" even though none contains the literal phrase "Training accuracy" /
"Validation accuracy." Concentrating signal helped the target chunk's own vector, but it
helped 66 *other* chunks more. See [`report.md`](report.md) for the actual top-5 under each
chunker. Worth noting: `headings` rank #5 (`day23`, "Training loss decreases, validation
loss increases") is arguably a *correct* answer in different words, scored a miss only
because it says "loss" where the golden keywords say "accuracy". That is the bent-ruler problem
from Lesson 04 reappearing in the runner-up, not the target chunk.

## 3. What happens to overall MRR across all 20?

| Chunker | Chunks | MRR | Δ vs fixed | Coverage | Misses |
|---|---|---|---|---|---|
| `fixed` (baseline) | 734 | **0.754** | +0.000 | 95.0% | 2/20 |
| `headings` | 1499 | **0.767** | +0.013 | 95.0% | 3/20 |

### Per-question diff (the honest part: who got helped, who got hurt)

| Question | fixed | headings | Verdict |
|---|---|---|---|
| What does NDCG measure that MRR does not? | @3 | @1 | improved |
| How is IoU calculated for a predicted bounding box? | @1 | @1 | unchanged |
| What does LoRA stand for? | @1 | @2 | worsened |
| Which tool in the LangChain ecosystem handles monitoring and debugging? | @1 | @1 | unchanged |
| Why is RAG optimization described as whack-a-mole? | @1 | @1 | unchanged |
| What is the Chat Completions API? | @1 | @1 | unchanged |
| How are chat models different from reasoning models? | @2 | MISS | **regressed** |
| How are LangChain and LiteLLM different? | @1 | MISS | **regressed** |
| What is prompt caching? | @1 | @1 | unchanged |
| What is computer vision? | @1 | @2 | worsened |
| How does RecursiveCharacterTextSplitter differ from a plain character splitter? | @2 | @1 | improved |
| How does an LSTM solve the vanishing gradient problem? | MISS | @1 | **rescued** |
| What problem do skip connections in ResNet solve? | @1 | @1 | unchanged |
| What does mAP measure in object detection? | @1 | @1 | unchanged |
| What does reranking do in a RAG pipeline? | @4 | @1 | improved |
| What is Gradio used for? | @1 | @1 | unchanged |
| What is Async IO in Python? | @1 | @1 | unchanged |
| What is overfitting in a neural network? | MISS | MISS | unchanged |
| What is backpropagation? | @1 | @1 | unchanged |
| What is Non-Maximum Suppression in object detection? | @2 | @3 | worsened |

**Composition of the +0.013 MRR move:** 1 question(s) rescued from MISS, 2 broken into a new MISS, 3 moved to a better rank while already hitting, 3 moved to a worse rank while still hitting. Net misses went 2 → 3 (worse), while MRR moved up. A new splitter helping some questions and hurting others, exactly as advertised.

The MRR move is a coincidence of magnitudes, not a sign the swap is safe: the one rescue
(LSTM, a genuine MISS → @1) is worth +0.05 on its own, more than enough to paper over two
new regressions (chat-vs-reasoning, LangChain-vs-LiteLLM) worth about −0.04 combined. A
different golden set of the same size could easily land the opposite way. This is the
"whack-a-mole" the corpus README warned about, now measured instead of anecdotal.

## Deliverable

**Before / after, one row per question, is the table above.** Summary:

| | MRR | Misses/20 | Overfitting question |
|---|---|---|---|
| Before (`fixed`) | 0.754 | 2 | MISS, answer chunk at full-corpus rank #8 |
| After (`headings`) | 0.767 | 3 | MISS, answer chunk at full-corpus rank #67 (worse) |

**Would I ship this splitter? No, not as a drop-in replacement.** It buys a marginal
+0.013 MRR and rescues one real failure (LSTM), but it breaks two questions that fixed-size
chunking got right, nearly doubles the index (734 → 1499 chunks, i.e. more to embed, store,
and search), and on the exact case it was built to fix, overfitting, it makes things
*worse*: the answer chunk is cleaner but drops from rank #8 to rank #67 because fragmenting
the whole corpus hands 66 other small, topically-adjacent "overfitting" chunks a chance to
outrank it. The fix isn't wrong in principle (the target chunk genuinely got less diluted),
it's just insufficient alone on a corpus with this much cross-day topic overlap: the section
is in the net but still out-ranked, which per the lesson means the next lever is a reranker
(re-score the wider net by relevance instead of relying on cosine rank alone), not a finer
splitter, and not yet a full swap of the production chunker.

See [`report.md`](report.md) for the full top-5 chunk text under both chunkers, side by side.
