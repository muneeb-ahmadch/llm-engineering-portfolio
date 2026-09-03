# 08. RAG, fine-tune, or prompt

**Question:** given a problem, which of the three tools is the right one, and can I defend
the call with a number?

This is a decision document rather than a codebase. It exists because "when would you
fine-tune instead of RAG" is asked in every interview and is usually answered with a slogan.

## The diagnostic

Sort the problem on **knowledge versus behaviour**. RAG adds knowledge the model does not
have. Fine-tuning changes behaviour the model will not adopt from instructions. Prompting is
the thing you try first, because it is free and reversible.

Four scenarios, one call each:

| Scenario | Call | Why |
|---|---|---|
| 400-page policy handbook, revised quarterly | **RAG** | knowledge, and it churns; fine-tuning would have to be redone every quarter |
| Correct but chatty output, clinicians need terse fixed structure | **prompt, then fine-tune** | behaviour, so try the free lever first and escalate if it will not hold |
| Ticket classifier, 12 stable categories, 50k/day, latency sensitive | **fine-tune** | behaviour, stable, and volume makes a small model pay for itself |
| The experiment 07 "Paris" hallucination | **prompt plus guardrail** | neither knowledge nor behaviour, it is a missing refusal contract |

## The number

Scenario 3, priced off the real per query cost measured in experiment 07:

**Daily saving: $108.70 - $36.23 = ~$72.47/day**, about $2,174/month or $26k/year. Training
is one time, the saving is ongoing, so a training bill of $X pays back in `X / 72.47` days.
At those numbers almost any realistic training bill clears in under a week.

The point is not the arithmetic, it is what stability does to it. Break-even is a one-time
cost divided by the lifetime days of saving. Stable categories mean you pay once and collect
indefinitely, so the denominator is enormous and nearly any bill clears. Scenario 1's
quarterly revision makes training recur four times a year and **resets the break-even clock
every quarter**, which is precisely why fine-tuning was the wrong call there. Stability lets
a one-time cost amortise. Churn kills it.

## The claim I will defend

That my hands-on work stops at prompting and RAG, that I can diagnose which of the three a
problem needs and price the decision, and that I have not fine-tuned a model in production.
Written out explicitly in part 3 of the document, because a portfolio should say where it
ends.

Full document: [`decision.md`](decision.md)
