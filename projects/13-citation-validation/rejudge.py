#!/usr/bin/env python
"""L13 step 1: re-judge L12's answers on a judge that has been segregated.

The rule that makes this step 1 and not step 5: you cannot measure today's
mitigation on a grader you have already proved is compromised. L12 ended with
`judge_answer()` reading the poisoned corpus as flat text, and a forged
"SYSTEM:" line talking it into scoring three CORRECT answers 1/5. Every number
in today's report is read off that instrument, so the instrument gets calibrated
before anything else is built.

NO REGENERATION. The 72 answers are already paid for and sitting in
`../lesson-012-injection/results/results-injection.json`. This file re-runs only
the judge over them, which is the cheap half of the bill.

The one hard part is the CONTEXT. The results file stores `context_sources` but
not the chunk text, so the judge's input has to be rebuilt:

    top-k        deterministic, same corpus, same encoder, same reranker, and
                 the embedding cache keys on document content
    the loop     `searches` recorded (query, k) per step; search_corpus is a
                 matrix multiply plus a cross-encoder, both deterministic, so
                 replaying the queries reproduces the chunks the model was shown

Then it is CHECKED, not assumed: the rebuilt source list must equal the stored
`context_sources` for every one of the 72 answers, or the run aborts. A judge
scored against a context that is nearly the right one is a number with no
meaning, and a silent near-miss here would poison the whole report.

Three cells, because "was 4, now ?" needs to survive one obvious objection, the judge runs at temperature default(1), so a 4-to-something move could just be
the judge rolling different dice:

    J_seg_<arm>   all four arms, 72 answers, judge SEGREGATED   (the measurement)
    J_flat_poisoned   18 answers, judge FLAT, re-run today      (the control:
                  how much does the old judge move on its own?)

    python rejudge.py            # ~$0.50
    python rejudge.py --resume   # skip cells already on disk
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
HARNESS_DIR = HERE.parents[0] / "lesson-001-chunking"
L12 = HERE.parents[0] / "lesson-012-injection"
sys.path.insert(0, str(HARNESS_DIR))
import rag_eval as R  # noqa: E402

SRC = L12 / "results" / "results-injection.json"
OUT = HERE / "results" / "results-rejudge.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

CONFIG = dict(encoder="bge-small", chunk_size=1000, overlap=50, chunker="fixed",
              top_k=5, rerank=True, pool=20)
JUDGE = "gpt-5.6-terra"
ARMS = {"clean": None, "poisoned": L12 / "corpus-poisoned",
        "defended": L12 / "corpus-poisoned", "fixed": L12 / "corpus-poisoned"}


class Index:
    """The L06 retrieval config over one corpus, built once and reused.

    Everything here already exists in evaluate(); this is the same six lines
    lifted out so a replay can retrieve without generating.
    """

    def __init__(self, corpus_dir):
        docs = R.load_documents(corpus_dir)
        self.chunks = R.build_chunks(docs, CONFIG["chunk_size"], CONFIG["overlap"],
                                     CONFIG["chunker"])
        self.model = R.get_model(CONFIG["encoder"])
        self.matrix, _ = R.embed_chunks(self.model, self.chunks, CONFIG["encoder"],
                                        CONFIG["chunk_size"], CONFIG["overlap"], docs,
                                        CONFIG["chunker"])
        self.ce = R.get_cross_encoder()
        self.search = R.make_search_corpus(self.model, CONFIG["encoder"], self.chunks,
                                           self.matrix, self.ce, CONFIG["pool"])

    def topk(self, question):
        qv = R.embed_queries(self.model, [question], CONFIG["encoder"])[0]
        row = qv @ self.matrix.T
        idx = np.argsort(-row)[:CONFIG["pool"]]
        cands = [self.chunks[i] for i in idx]
        scores = self.ce.predict([(question, c["text"]) for c in cands])
        order = np.argsort(-np.asarray(scores))[:CONFIG["top_k"]]
        return [cands[i] for i in order]


def replay_context(index, q):
    """Rebuild exactly the chunk list judge_answer() was handed in L12.

    run_agent() keeps `seen`: the initial top-k, then every chunk a search
    returned that was not already in it, deduped on text in the order shown.
    Reproduced here move for move, and verified against the stored sources by
    the caller, which is the only reason this is allowed to be load-bearing.
    """
    seen = list(index.topk(q["question"]))
    keys = {c["text"] for c in seen}
    for s in q.get("searches", []):
        for c in index.search(s["query"], s["k"]):
            if c["text"] not in keys:
                keys.add(c["text"])
                seen.append(c)
    return seen


def rejudge_arm(client, arm, runs, segregate):
    index = INDEXES[arm]
    out, mismatches = [], 0
    for i, run in enumerate(runs, 1):
        rows = []
        for q in run["per_question"]:
            ctx = replay_context(index, q)
            got = [c["source"] for c in ctx]
            if got != q["context_sources"]:
                mismatches += 1
                print(f"    !! {q['id']} run {i}: context replay MISMATCH\n"
                      f"       stored {q['context_sources']}\n       rebuilt {got}")
                continue
            v = R.judge_answer(client, JUDGE, q["answer"], ctx, segregate=segregate)
            v["cost"] = R.compute_cost(v["prompt_tokens"], v["completion_tokens"],
                                       v["cached_tokens"], R.PRICES[JUDGE])
            old = q["judge"]
            flag = "" if v["score"] == old["score"] else f"   <-- was {old['score']}/5"
            print(f"    {q['id']} run {i}  {old['score']}/5 -> {v['score']}/5  "
                  f"({v['grounded_claims']}/{v['total_claims']} grounded){flag}")
            rows.append({"id": q["id"], "run": i, "answer": q["answer"],
                         "old_judge": old, "new_judge": v,
                         "n_context_chunks": len(ctx),
                         "needle_in_context": q.get("needle_in_context"),
                         "n_searches": q.get("n_searches", 0)})
        out.append(rows)
    if mismatches:
        sys.exit(f"ABORT, {mismatches} context replays did not reproduce L12's "
                 f"context. Every judge score below would be graded against the "
                 f"wrong evidence. Fix the replay before reading any number.")
    return out


if __name__ == "__main__":
    t0 = time.time()
    data = json.loads(SRC.read_text())
    resume = "--resume" in sys.argv and OUT.exists()
    out = json.loads(OUT.read_text()) if resume else {}
    out.update({"source": str(SRC.relative_to(HERE.parents[1])), "judge_model": JUDGE,
                "config": CONFIG, "judge_rules": R.JUDGE_SEGREGATION_RULES})

    client = R.get_openai_client()
    print("Building indexes (free, cached)…")
    INDEXES = {a: Index(d) for a, d in ARMS.items()}

    cells = {f"J_seg_{a}": (a, True) for a in ARMS}
    cells["J_flat_poisoned"] = ("poisoned", False)   # the noise control

    for cell, (arm, seg) in cells.items():
        if resume and cell in out:
            print(f"  [skip] {cell}")
            continue
        print(f"\n{cell}, {'SEGREGATED' if seg else 'FLAT (control)'} judge, "
              f"{arm} arm")
        out[cell] = rejudge_arm(client, arm, data[f"I_{arm}"], seg)
        spend = sum(r["new_judge"]["cost"] for c in out.values()
                    if isinstance(c, list) for run in c for r in run)
        out["spend"] = round(spend, 6)
        out["wall_secs"] = round(time.time() - t0, 1)
        OUT.write_text(json.dumps(out, indent=2, default=str))

    OUT.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {OUT}  ({out['wall_secs']}s, ${out['spend']:.4f})")
