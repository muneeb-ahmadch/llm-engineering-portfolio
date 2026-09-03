#!/usr/bin/env python
"""Lesson 05: before/after: does a Markdown-header splitter surface the overfitting answer?

Turns the lesson's hypothesis into a number: re-run the "what is overfitting"
question under the Lesson-01 fixed-size chunker and the new heading-aware
chunker (rag_eval.chunk_by_headings), same encoder / chunk-size / top-k, and
report (1) whether the miss flips, (2) what rank the day26 Monitoring
Training section lands at now, even if it's still outside the top-5, and
(3) what the swap costs across the full 20-question golden set.

Reuses the Lesson-01 harness as a library (same trick as Lesson 04's
show_chunks.py), so numbers match `rag_eval.py run` exactly and nothing
re-embeds if the cache already has it.

Run with the Lesson-01 venv:
    ../lesson-001-chunking/.venv/bin/python compare_chunkers.py
"""

import sys
from pathlib import Path

import numpy as np

LESSON1 = Path(__file__).parent.parent / "lesson-001-chunking"
if not (LESSON1 / "rag_eval.py").exists():
    sys.exit(f"Can't find the Lesson-01 harness at {LESSON1}")
sys.path.insert(0, str(LESSON1))
import rag_eval as R  # noqa: E402

OUT_DIR = Path(__file__).parent

ENCODER = "bge-small"
CHUNK_SIZE = 1000
TOP_K = 5
OVERLAP = 50
RELEVANCE = "all"
TARGET_SUBSTR = "overfitting"
CHUNKERS = ["fixed", "headings"]


def full_ranking(chunker):
    """(tests, chunks, scores) for every golden question against one chunker's index."""
    docs = R.load_documents()
    tests = R.load_golden()
    chunks = R.build_chunks(docs, CHUNK_SIZE, OVERLAP, chunker)
    model = R.get_model(ENCODER)
    matrix, _ = R.embed_chunks(model, chunks, ENCODER, CHUNK_SIZE, OVERLAP, docs, chunker)
    query_vecs = R.embed_queries(model, [t["question"] for t in tests], ENCODER)
    scores = query_vecs @ matrix.T  # both sides normalized -> cosine similarity
    return tests, chunks, scores


def analyze_target(tests, chunks, scores):
    """Full-corpus rank + top-k detail for the overfitting question, one chunker."""
    ti = next(i for i, t in enumerate(tests) if TARGET_SUBSTR in t["question"].lower())
    test = tests[ti]
    row = scores[ti]
    order = np.argsort(-row)  # every chunk, ranked, not just top-k

    day26_rank = None
    for rank, idx in enumerate(order, 1):
        if R.is_relevant(chunks[idx]["text"].lower(), test["keywords"], RELEVANCE):
            day26_rank = rank
            break

    top_k_idx = order[:TOP_K]
    top_k = [(rank, chunks[i], float(row[i])) for rank, i in enumerate(top_k_idx, 1)]
    retrieved = [chunks[i] for i in top_k_idx]
    rr = R.reciprocal_rank(test, retrieved, RELEVANCE)
    cov = R.keyword_coverage(test, retrieved)
    return {
        "test": test,
        "n_chunks": len(chunks),
        "day26_rank": day26_rank,
        "top_k": top_k,
        "rr": rr,
        "hit": rr > 0,
        "coverage": cov,
    }


def kw_hits(text_lower, keywords):
    return [kw for kw in keywords if kw.lower() in text_lower]


def chunk_block(rank, chunk, sim, keywords, in_topk_hit):
    text = chunk["text"].strip()
    hits = kw_hits(text.lower(), keywords)
    marks = " · ".join(f"{kw} {'✓' if kw in hits else '✗'}" for kw in keywords)
    all_here = " **◀ ALL keywords, this is the answer chunk**" if len(hits) == len(keywords) else ""
    lines = [
        f"<details{' open' if in_topk_hit or rank <= 5 else ''}>",
        f"<summary><b>#{rank}</b> · sim {sim:.3f} · <code>{chunk['source']}</code> · "
        f"{len(text)} chars · {marks}{all_here}</summary>",
        "",
        "```text",
        text,
        "```",
        "</details>",
        "",
    ]
    return "\n".join(lines)


def build_report(analyses):
    test = analyses["fixed"]["test"]
    kws = test["keywords"]
    lines = [
        "# Lesson 05, top-k chunks, fixed vs headings",
        "",
        f"Question: **{test['question']}**  ",
        f"Keywords required (relevance=`{RELEVANCE}`, all must be in one chunk): "
        + " · ".join(f"`{k}`" for k in kws) + "  ",
        f"Config: `{ENCODER}` · chunk-size {CHUNK_SIZE} · top-{TOP_K} · overlap {OVERLAP}",
        "",
        "Open this file in the Markdown **preview**, each retrieved chunk is a "
        "collapsible block so you can scan ten of them without scrolling forever.",
        "",
        "---",
        "",
        "## Rank summary",
        "",
        "| Chunker | Chunks in index | Top-5 verdict | Rank in top-5 | Rank across "
        "*all* chunks (if not in top-5) | Coverage |",
        "|---|---|---|---|---|---|",
    ]
    for name in CHUNKERS:
        a = analyses[name]
        verdict = "✅ HIT" if a["hit"] else "❌ MISS"
        rank5 = f"@{round(1 / a['rr'])}" if a["hit"] else ","
        beyond = "," if a["hit"] else f"#{a['day26_rank']}"
        lines.append(
            f"| `{name}` | {a['n_chunks']} | {verdict} | {rank5} | {beyond} | "
            f"{a['coverage']:.0%} |"
        )
    lines += ["", "---", ""]

    for name in CHUNKERS:
        a = analyses[name]
        lines.append(f"## `{name}` chunker, top-{TOP_K} retrieved\n")
        for rank, chunk, sim in a["top_k"]:
            lines.append(chunk_block(rank, chunk, sim, kws, a["hit"]))
        if not a["hit"]:
            lines.append(
                f"> The answer chunk is **not** in the top-{TOP_K} above. It exists in "
                f"the index at full-corpus rank **#{a['day26_rank']}** of {a['n_chunks']} "
                f", in the net, but out-ranked by {a['day26_rank'] - 1} other chunks.\n"
            )
        lines.append("---\n")

    (OUT_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def build_results_md(analyses, full_results, diffs):
    fixed, heads = analyses["fixed"], analyses["headings"]
    fr, hr = full_results["fixed"], full_results["headings"]

    flip = "YES, MISS → hit" if (not fixed["hit"] and heads["hit"]) else (
        "NO, still a MISS" if not heads["hit"] else "already a hit before the change"
    )
    fixed_rank_desc = f"@{round(1/fixed['rr'])}" if fixed["hit"] else f"MISS (full-corpus rank #{fixed['day26_rank']})"
    heads_rank_desc = f"@{round(1/heads['rr'])}" if heads["hit"] else f"MISS (full-corpus rank #{heads['day26_rank']})"

    lines = [
        "# Lesson 05 results, heading-aware chunker vs fixed-size baseline",
        "",
        f"Command (both runs): `run --encoder {ENCODER} --chunk-size {CHUNK_SIZE} "
        f"--top-k {TOP_K} -v` with `--chunker fixed` then `--chunker headings`. "
        f"overlap={OVERLAP}, relevance=`{RELEVANCE}`.",
        "",
        "## 1. Does overfitting flip MISS → hit?",
        "",
        f"**{flip}**",
        "",
        f"- `fixed`    → {fixed_rank_desc}",
        f"- `headings` → {heads_rank_desc}",
        "",
        "## 2. What rank does the day26 section land at now?",
        "",
        f"- `fixed`: the chunk holding all three keywords sits at full-corpus rank "
        f"**#{fixed['day26_rank']}** of {fixed['n_chunks']} chunks (outside top-{TOP_K}).",
        f"- `headings`: the isolated `#### Monitoring Training` section sits at "
        f"full-corpus rank **#{heads['day26_rank']}** of {heads['n_chunks']} chunks.",
        f"- Movement: rank #{fixed['day26_rank']} → #{heads['day26_rank']} "
        f"({fixed['day26_rank'] - heads['day26_rank']:+d} positions"
        f"{', but still short of top-'+str(TOP_K) if not heads['hit'] else ', now inside top-'+str(TOP_K)}).",
        "",
        "## 3. What happens to overall MRR across all 20?",
        "",
        "| Chunker | Chunks | MRR | Δ vs fixed | Coverage | Misses |",
        "|---|---|---|---|---|---|",
        f"| `fixed` (baseline) | {fr['n_chunks']} | **{fr['mrr']:.3f}** | +0.000 | "
        f"{fr['keyword_coverage']:.1%} | {fr['misses']}/{fr['n_questions']} |",
        f"| `headings` | {hr['n_chunks']} | **{hr['mrr']:.3f}** | "
        f"{hr['mrr']-fr['mrr']:+.3f} | {hr['keyword_coverage']:.1%} | "
        f"{hr['misses']}/{hr['n_questions']} |",
        "",
        "### Per-question diff (the honest part, who got helped, who got hurt)",
        "",
        "| Question | fixed | headings | Verdict |",
        "|---|---|---|---|",
    ]
    for q, f_rr, h_rr in diffs:
        f_s = f"@{round(1/f_rr)}" if f_rr else "MISS"
        h_s = f"@{round(1/h_rr)}" if h_rr else "MISS"
        if f_rr == 0 and h_rr > 0:
            v = "**rescued**"
        elif f_rr > 0 and h_rr == 0:
            v = "**regressed**"
        elif f_rr == h_rr:
            v = "unchanged"
        elif h_rr > f_rr:
            v = "improved"
        else:
            v = "worsened"
        lines.append(f"| {q} | {f_s} | {h_s} | {v} |")

    rescued = sum(1 for _, f, h in diffs if f == 0 and h > 0)
    regressed = sum(1 for _, f, h in diffs if f > 0 and h == 0)
    worsened_ranked = sum(1 for _, f, h in diffs if f > 0 and h > 0 and h < f)
    improved_ranked = sum(1 for _, f, h in diffs if f > 0 and h > 0 and h > f)

    lines += [
        "",
        f"**Composition of the +{hr['mrr']-fr['mrr']:.3f} MRR move:** {rescued} question(s) "
        f"rescued from MISS, {regressed} broken into a new MISS, {improved_ranked} moved to "
        f"a better rank while already hitting, {worsened_ranked} moved to a worse rank while "
        f"still hitting. Net misses went {fr['misses']} → {hr['misses']} "
        f"({'better' if hr['misses'] < fr['misses'] else 'worse' if hr['misses'] > fr['misses'] else 'flat'}), "
        f"while MRR moved {'up' if hr['mrr']>fr['mrr'] else 'down' if hr['mrr']<fr['mrr'] else 'flat'} "
        f", a new splitter helping some questions and hurting others, exactly as advertised.",
        "",
        "## Deliverable",
        "",
    ]
    (OUT_DIR / "results.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    analyses = {}
    full_results = {}
    per_chunker_rr = {}
    for chunker in CHUNKERS:
        print(f"=== {chunker} ===")
        tests, chunks, scores = full_ranking(chunker)
        analyses[chunker] = analyze_target(tests, chunks, scores)
        r = R.evaluate(ENCODER, CHUNK_SIZE, TOP_K, OVERLAP, RELEVANCE, chunker=chunker)
        full_results[chunker] = r
        per_chunker_rr[chunker] = {q["question"]: q["rr"] for q in r["per_question"]}
        a = analyses[chunker]
        status = f"@{round(1/a['rr'])}" if a["hit"] else f"MISS (full-corpus #{a['day26_rank']})"
        print(f"  overfitting: {status}   |   MRR {r['mrr']:.3f}   misses {r['misses']}/20")

    questions = [q["question"] for q in full_results["fixed"]["per_question"]]
    diffs = [
        (q, per_chunker_rr["fixed"][q], per_chunker_rr["headings"][q]) for q in questions
    ]

    build_report(analyses)
    build_results_md(analyses, full_results, diffs)
    print(f"\nWrote {OUT_DIR / 'report.md'} and {OUT_DIR / 'results.md'}")


if __name__ == "__main__":
    main()
