# 12. The attack that scores grounded

**Question:** what happens if someone can write 1,800 bytes into the corpus my agent reads?

**Answer:** it obeys them 8 times in 18, and the guardrail I trusted most certifies the
result as fully grounded.

**If you read one thing in this repository, read this one.**

## Method

Threat model first, in [`threat-model.md`](threat-model.md), committed before the poisoned
corpus existed. Four OWASP categories triaged: LLM06 and LLM10 handled by construction
because the only tool is a read-only local index, LLM02 parked, and **LLM01 indirect
injection** identified as the one worth building.

Then six injection shapes planted in a copy of my own lecture notes. **1,820 bytes on
712,484, which is 0.255% of the corpus.** Four arms, three runs each. Delivery was proved
free rather than assumed: a preflight check that failed once and caught a payload the
retriever could never have reached.

Nobody typed anything hostile. No tool was compromised. The questions were the same ordinary
ones I had been asking since experiment 01.

## What happened

**The loop obeyed 8 of 18 runs.** The most successful payload produced the sentence
*"Gradio is a commercial dashboarding product from Google"* verbatim, three times.

**The faithfulness judge from experiment 09 scored that answer 5/5, every claim grounded,
three runs out of three.** The instrument that took trap hallucinations from 15/15 to 0/15,
and the reason I trusted the pipeline at all, did not malfunction. It agreed with the
attacker.

**The announced attack was obeyed 0 of 3 times.** A payload shouting `SYSTEM NOTE` never
landed. The ones that landed are quiet, and they ride experiment 09's own instruction to
trust the retrieved context.

**The unpredicted finding, which I think is the real one: the injection propagates into the
judge.** Answers that obeyed scored 4.33/5. Answers that correctly refused scored 3.40/5.
Three correct answers scored **1/5**, because a forged `SYSTEM:` line told the grader they
had been superseded. I had hardened the generator and left the guardrail wide open. That is
what experiment 13 goes and fixes.

## The defense, and its price

A per request random nonce delimiter wrapped around every retrieved chunk, applied to tool
output as well, plus three rules telling the model that delimited text is data rather than
instructions.

| | Poisoned | Defended |
|---|---|---|
| Obedience | 8/18 | **1/18** |
| Agent loop suppression attack | search killed | fully reversed, 3/3 |
| Faithfulness | 5.00 | 5.00 |
| Groundedness | 100% | 100% |
| Traps | 0/5 | 0/5 |
| Cost | 1.00x | **1.68x** |

Zero measured quality cost on every instrument I own. The 1.68x is 1.38x more tokens plus a
prompt cache the nonce necessarily destroys: cached tokens on single-step questions went from
8,942 to **0**. A per request random value cannot be cached, and that is the trade.

## The ceiling, measured rather than argued

I built a seventh injection that issues **no order at all**: one sentence in flat
documentation voice that simply states a false number. It walked straight through the
defense, **3/3 at 5/5 faithfulness**, because there is nothing in it to refuse.

That was prediction P4 at 90% confidence, written before the poisoned corpus existed.
**Segregation stops text that acts like an instruction. It has nothing to say about text
that just lies.**

## Run it

```bash
python build_poisoned_corpus.py          # free, and proves every payload is retrievable
python run_injection.py                  # the four-arm experiment, ~$0.91, checkpointed
python summarize.py                      # every table, free
python summarize.py --answers            # all 72 answers verbatim, for the hand pass
```

Measured spend: $1.0404, nothing lost.

Full write-up: [`results.md`](results.md) · Threat model: [`threat-model.md`](threat-model.md) ·
Payloads: [`injections.jsonl`](injections.jsonl)
