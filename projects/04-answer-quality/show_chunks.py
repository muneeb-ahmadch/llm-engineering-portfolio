#!/usr/bin/env python
"""Lesson 04: be the human judge, by hand.

The Lesson-01 harness (`rag_eval.py`) scores retrieval but never shows you the
*text* it retrieved. You can't judge an answer you can't read. This tool does the
one missing thing: for a question, it pulls the top-k chunks and lays them out so
you can decide, with your own eyes, whether they answer it, the judgement the
keyword rule can't make.

It reuses the Lesson-01 machinery (chunking, encoders, the cached embeddings), so
numbers here match `rag_eval.py run` exactly and nothing re-embeds.

Run it with the Lesson-01 venv (it has the deps + model weights):

    ../lesson-001-chunking/.venv/bin/python show_chunks.py --list
    ../lesson-001-chunking/.venv/bin/python show_chunks.py -q overfitting
    ../lesson-001-chunking/.venv/bin/python show_chunks.py -q overfitting -q "prompt caching" -q reranking

Defaults match the assignment: bge-small, chunk 1000, top-5, relevance=all.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Reuse the Lesson-01 harness as a library. Because rag_eval.py sets its own
# ROOT = Path(__file__).parent, importing it here still points CORPUS_DIR,
# GOLDEN_PATH and the embedding CACHE at the lesson-001 folder, so we read the
# same corpus, the same golden set, and hit the same cached vectors.
LESSON1 = Path(__file__).parent.parent / "lesson-001-chunking"
if not (LESSON1 / "rag_eval.py").exists():
    sys.exit(f"Can't find the Lesson-01 harness at {LESSON1}")
sys.path.insert(0, str(LESSON1))
import rag_eval as R  # noqa: E402


def retrieve(encoder, chunk_size, top_k, overlap):
    """Return (tests, chunks, top_idx_per_test, sims_per_test), no re-embedding."""
    docs = R.load_documents()
    tests = R.load_golden()
    chunks = R.build_chunks(docs, chunk_size, overlap)
    model = R.get_model(encoder)
    matrix, _ = R.embed_chunks(model, chunks, encoder, chunk_size, overlap, docs)
    query_vecs = R.embed_queries(model, [t["question"] for t in tests], encoder)
    scores = query_vecs @ matrix.T  # both normalized -> cosine similarity
    top_idx, sims = [], []
    for row in scores:
        idx = np.argsort(-row)[:top_k]
        top_idx.append(idx)
        sims.append(row[idx])
    return tests, chunks, top_idx, sims


def kw_hits(chunk_text_lower, keywords):
    return [kw for kw in keywords if kw.lower() in chunk_text_lower]


def cmd_list(tests, chunks, top_idx, mode):
    """One line per question: does the keyword ruler call it a pass or a miss?"""
    print(f"\n{len(tests)} golden questions, as the KEYWORD RULER scores them:\n")
    print("  status  cov   question")
    print("  " + "-" * 70)
    rows = []
    for t, idx in zip(tests, top_idx):
        retrieved = [chunks[i] for i in idx]
        rr = R.reciprocal_rank(t, retrieved, mode)
        cov = R.keyword_coverage(t, retrieved)
        rows.append((rr, cov, t["question"]))
    for rr, cov, q in sorted(rows, key=lambda r: (r[0], r[1])):
        status = f"@{round(1/rr)}" if rr else "MISS"
        print(f"  {status:>5}  {cov:>4.0%}  {q}")
    print(
        "\n  'MISS' = the keyword rule found no single chunk with ALL its keywords.\n"
        "  Pick OVERFITTING (a miss) + two that PASS, then: -q <substring> for each.\n"
    )


def show_question(t, chunks, idx, sims, mode):
    kws = t["keywords"]
    retrieved = [chunks[i] for i in idx]
    cov = R.keyword_coverage(t, retrieved)
    rr = R.reciprocal_rank(t, retrieved, mode)
    first_rel = round(1 / rr) if rr else None

    print("\n" + "=" * 78)
    print(f"Q: {t['question']}")
    print(f"   category: {t.get('category', '?')}")
    print(f"   keywords: {' · '.join(kws)}")
    print("-" * 78)
    print(f"   KEYWORD-RULER VERDICT (what rag_eval.py scores):")
    print(f"     coverage (keywords found anywhere in top-{len(idx)}): {cov:.0%}"
          f"  [{sum(kw.lower() in ' '.join(c['text'].lower() for c in retrieved) for kw in kws)}/{len(kws)}]")
    if first_rel:
        print(f"     counted RELEVANT ('{mode}' rule): yes, first at rank {first_rel}")
    else:
        print(f"     counted RELEVANT ('{mode}' rule): NO, scored 0 / a MISS")
    print(f"   reference answer (independent ground truth, NOT scored):")
    print(f"     {t.get('reference_answer', '(none written)')}")
    print("-" * 78)
    print(f"   YOUR JOB: read the {len(idx)} chunks below. Ignore the keyword marks,"
          f"\n   does any chunk actually ANSWER the question, in whatever words?\n")

    for rank, (i, sim) in enumerate(zip(idx, sims), 1):
        c = chunks[i]
        text = c["text"].strip()
        hit = kw_hits(text.lower(), kws)
        marks = "  ".join(
            f"[{kw} {'✓' if kw in hit else '✗'}]" for kw in kws
        )
        all_here = " ◀ ALL keywords here" if len(hit) == len(kws) else ""
        print(f"  ── #{rank}  sim {sim:.3f}  {c['source']}{all_here}")
        print(f"     {marks}")
        # Indent the chunk body so it's visually distinct from the metadata.
        for line in text.splitlines():
            print(f"       {line}")
        print()


def question_md(t, chunks, idx, sims, mode):
    """Same content as show_question, but as clean Markdown for the preview pane."""
    kws = t["keywords"]
    retrieved = [chunks[i] for i in idx]
    cov = R.keyword_coverage(t, retrieved)
    rr = R.reciprocal_rank(t, retrieved, mode)
    verdict = f"✅ PASS, first relevant at rank {round(1/rr)}" if rr else "❌ MISS, scored 0"

    out = [
        f"## {t['question']}",
        "",
        f"**Ruler verdict:** {verdict} · coverage {cov:.0%} · category `{t.get('category','?')}`  ",
        f"**Keywords demanded:** {' · '.join(f'`{k}`' for k in kws)}  ",
        f"**Reference answer** (ground truth, not scored): {t.get('reference_answer','(none)')}",
        "",
        f"> **Your job:** ignore the ✓/✗ below. Does any chunk actually *answer* the "
        f"question, in whatever words? Note the rank of the one that does (or that none does).",
        "",
    ]
    for rank, (i, sim) in enumerate(zip(idx, sims), 1):
        c = chunks[i]
        text = c["text"].strip()
        marks = " · ".join(
            f"{kw} {'✓' if kw.lower() in text.lower() else '✗'}" for kw in kws
        )
        all_here = " · **◀ all keywords**" if all(k.lower() in text.lower() for k in kws) else ""
        out += [
            f"<details open>",
            f"<summary><b>#{rank}</b> · sim {sim:.3f} · <code>{c['source']}</code> · "
            f"{marks}{all_here}</summary>",
            "",
            "```text",
            text,
            "```",
            "</details>",
            "",
        ]
    out.append("\n---\n")
    return "\n".join(out)


def write_report(path, tests, chunks, top_idx, sims, questions, encoder, chunk_size,
                 top_k, mode):
    parts = [
        "# Experiment 04: chunks to judge",
        "",
        f"Config: `{encoder}` · chunk {chunk_size} · top-{top_k} · relevance=`{mode}`. "
        "Regenerate any time by re-running `show_chunks.py`.",
        "",
        "Open this in the Markdown **preview** (right-click → *Open Preview*, or ⌘K V). "
        "Each chunk is a collapsible block, click a `#N` heading to fold it away once "
        "you've judged it. Then record your verdicts in "
        "[`worksheet.md`](worksheet.md).",
        "",
        "---",
        "",
    ]
    for substr in questions:
        for i, t in enumerate(tests):
            if substr.lower() in t["question"].lower():
                parts.append(question_md(t, chunks, top_idx[i], sims[i], mode))
    Path(path).write_text("\n".join(parts), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("-q", "--question", action="append", default=[], metavar="SUBSTR",
                   help="case-insensitive substring of a golden question; repeatable")
    p.add_argument("--list", action="store_true",
                   help="list every question with its keyword-ruler pass/miss")
    p.add_argument("--encoder", choices=list(R.ENCODERS), default="bge-small")
    p.add_argument("--chunk-size", type=int, default=1000)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--overlap", type=int, default=50)
    p.add_argument("--relevance", choices=["all", "any"], default="all")
    p.add_argument("--out", nargs="?", const="report.md", metavar="FILE",
                   help="write a clean Markdown report (default report.md) to open in "
                        "the preview pane, instead of dumping to the terminal")
    args = p.parse_args()

    tests, chunks, top_idx, sims = retrieve(
        args.encoder, args.chunk_size, args.top_k, args.overlap
    )

    if args.out:
        qs = args.question or ["overfitting"]
        write_report(args.out, tests, chunks, top_idx, sims, qs,
                     args.encoder, args.chunk_size, args.top_k, args.relevance)
        print(f"Wrote {args.out}, open it in the Markdown preview (⌘K V) and read there.")
        return

    if args.list or not args.question:
        cmd_list(tests, chunks, top_idx, args.relevance)
        if not args.question:
            return

    for substr in args.question:
        matches = [i for i, t in enumerate(tests)
                   if substr.lower() in t["question"].lower()]
        if not matches:
            print(f"\n!! no question matches '{substr}', try --list", file=sys.stderr)
            continue
        for i in matches:
            show_question(tests[i], chunks, top_idx[i], sims[i], args.relevance)


if __name__ == "__main__":
    main()
