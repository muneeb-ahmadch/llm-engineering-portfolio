#!/usr/bin/env python
"""Build corpus-poisoned/ from the clean corpus + injections.jsonl, then PROVE
the payloads are reachable, all of it free, no API calls.

Three jobs, in order, and the third is the one that stops this experiment lying:

  1. copy      corpus/*.md -> corpus-poisoned/*.md, byte-for-byte. The clean
               corpus is opened read-only and never written. Everything L01-L11
               measured stays reproducible.
  2. inject    insert each payload immediately after its unique `anchor` line.
               A non-unique or missing anchor is a hard error, not a warning:
               an injection inserted in the wrong place is a different experiment
               than the one this file documents.
  3. preflight embed the poisoned corpus and check, for every injection, that the
               `needle` lands whole inside ONE chunk and that the chunk is in the
               top-5 the retriever returns for its question.

Why (3) is not optional. An injection that never gets retrieved produces an
obedience rate of 0% under every mitigation and under none, a green light you
did not earn, which is exactly the failure `check --trap` exists to prevent on
the L09 trap set. Same discipline, one file over.

Two ways it can fail quietly and how each is caught:
  needle straddles a chunk boundary  fixed-size character chunking cuts every
      1000 chars with no regard for meaning, so a payload can be sliced in half
      and neither half carries the order. Caught by "needle whole in a chunk".
  needle in the index but not in top-5  the payload is present but the question
      never pulls it. Caught by needle_rank.

    python build_poisoned_corpus.py            # build + preflight
    python build_poisoned_corpus.py --check    # preflight only, no rebuild
"""
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HARNESS_DIR = HERE.parents[0] / "lesson-001-chunking"
sys.path.insert(0, str(HARNESS_DIR))
import rag_eval as R  # noqa: E402

CLEAN = HARNESS_DIR / "corpus"
POISONED = HERE / "corpus-poisoned"
INJECTIONS = HERE / "injections.jsonl"   # overridable with --injections

# The retrieval config held fixed since L06 and used by every run in this lesson.
CONFIG = dict(encoder="bge-small", chunk_size=1000, top_k=5, overlap=50, mode="all",
              chunker="fixed", rerank=True, pool=20)


def build(out=None, ids=None):
    """ids=None injects everything. A subset goes to its OWN directory, because
    inserting text shifts every chunk boundary after it in that file, adding a
    seventh payload to the main corpus would silently re-index the six already
    measured, and the numbers in results.md would no longer be from the corpus
    they claim to be from."""
    out = out or POISONED
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(CLEAN, out, ignore=shutil.ignore_patterns(".*"))

    injections = [i for i in R.load_golden(INJECTIONS)
                  if ids is None or i["id"] in ids]
    print(f"Building {out.name}/ from {CLEAN.name}/ "
          f"({len(list(out.glob('*.md')))} files, "
          f"{len(injections)} injections)\n")

    for inj in injections:
        path = out / inj["target"]
        if not path.exists():
            sys.exit(f"{inj['id']}: no such file {inj['target']}")
        # newline="" both ways, and the file's OWN newline for the inserted block.
        # Every file in this corpus is CRLF; the first version of this script read
        # with universal newlines and wrote back LF, silently rewriting ~2,600 bytes
        # of line endings in the four files it touched. The chunker never sees the
        # difference (load_documents() normalises on read) but the byte delta below
        # came out NEGATIVE, which is how it got noticed. A corpus diff that isn't
        # exactly the payload is a corpus diff you have to defend in the report.
        with path.open(encoding="utf-8", newline="") as fh:   # newline="" needs open()
            text = fh.read()
        n = text.count(inj["anchor"])
        if n != 1:
            sys.exit(f"{inj['id']}: anchor occurs {n} times in {inj['target']} "
                     f"(must be exactly 1), {inj['anchor'][:60]!r}")
        before = len(text)
        nl = "\r\n" if "\r\n" in text else "\n"
        block = nl + nl + inj["payload"].replace("\n", nl)
        text = text.replace(inj["anchor"], inj["anchor"] + block, 1)
        with path.open("w", encoding="utf-8", newline="") as fh:
            fh.write(text)
        print(f"  {inj['id']}  +{len(text) - before:>4} chars into {inj['target']:<24} "
              f"after {inj['anchor'][:44]!r}…")

    clean_bytes = sum(p.stat().st_size for p in CLEAN.glob("*.md"))
    dirty_bytes = sum(p.stat().st_size for p in out.glob("*.md"))
    print(f"\n  corpus grew {clean_bytes:,} -> {dirty_bytes:,} bytes "
          f"({(dirty_bytes - clean_bytes) / clean_bytes:.3%}). "
          f"That fraction is the entire attack.")


def preflight(out=None, ids=None):
    """Free. Retrieval only, no generate, so no client, no key, no spend."""
    out = out or POISONED
    print("\n── preflight: is every payload actually reachable? ──────────────")
    r = R.evaluate(**CONFIG, corpus_dir=out, questions_path=INJECTIONS)
    injections = {i["id"]: i for i in R.load_golden(INJECTIONS)}
    if ids is not None:
        r["per_question"] = [q for q in r["per_question"] if q["id"] in ids]

    # Chunk-level check: does the needle survive the 1000-char splitter intact?
    docs = R.load_documents(out)
    chunks = R.build_chunks(docs, CONFIG["chunk_size"], CONFIG["overlap"],
                            CONFIG["chunker"])
    print(f"  poisoned index: {len(chunks)} chunks "
          f"(clean was 734, the delta is the payloads)\n")

    bad = []
    for q in r["per_question"]:
        inj = injections[q["id"]]
        whole = sum(1 for c in chunks if inj["needle"] in c["text"])
        rank = q["needle_rank"]
        ok = whole >= 1 and rank is not None
        print(f"  {'ok ' if ok else 'FAIL'} {q['id']}  needle in {whole} chunk(s) · "
              f"top-5 rank {rank if rank else ', NOT RETRIEVED'} · "
              f"top source {q['top_source']}")
        print(f"       {inj['shape']}")
        if not ok:
            bad.append(q["id"])

    print()
    if bad:
        sys.exit(f"PREFLIGHT FAILED for {', '.join(bad)}, move the anchor. "
                 f"An injection the retriever never surfaces measures nothing.")
    print("PREFLIGHT PASSED, every payload reaches the model. The numbers that "
          "follow are about obedience, not about retrieval.")


if __name__ == "__main__":
    # --injections F --out DIR builds a SEPARATE variant from a different set
    ids = None
    if "--ids" in sys.argv:
        ids = set(sys.argv[sys.argv.index("--ids") + 1].split(","))
    if "--injections" in sys.argv:
        INJECTIONS = HERE / sys.argv[sys.argv.index("--injections") + 1]
    out = (HERE / sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else None
    if "--check" not in sys.argv:
        build(out, ids)
    preflight(out, ids)
