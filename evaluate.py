"""Step 2: measure RETRIEVAL quality on eval/questions.csv.

CSV columns: question, expected, language, type
  expected = one or more acceptable pages:  FILE.pdf:7  or  FILE.pdf:7|OTHER.pdf:8
  type     = fact (has an answer in the manuals) | unanswerable (nothing to retrieve)

Metrics (answerable questions only):
  Hit@k  share of questions where an acceptable page is among the top-k results
  MRR    mean of 1/rank of the first acceptable page (0 if not found)

Run:  python evaluate.py                 vector search only
      python evaluate.py --hybrid        vector + keyword (BM25)
      python evaluate.py --k 3           top-3 instead of config.TOP_K
NOTE: in --hybrid mode rag.retrieve() returns up to k+2 results (k vector + keyword extras), so hybrid
sees slightly more results than vector-only. Step 2 replaces this with Reciprocal Rank Fusion at exactly k.
Every run is appended to eval/results.jsonl so settings can be compared later.
"""
import csv, sys, json, time, argparse
from collections import defaultdict


def parse_expected(cell):
    """'A.pdf:7|B.pdf:8' -> {('A.pdf', 7), ('B.pdf', 8)}"""
    out = set()
    for part in filter(None, (p.strip() for p in cell.split("|"))):
        name, page = part.rsplit(":", 1)
        out.add((name.strip(), int(page)))
    return out


def first_rank(hits, accepted):
    """1-based rank of the first retrieved chunk whose (source, page) is accepted, else None."""
    for rank, h in enumerate(hits, 1):
        if (h["source"], int(h["page"])) in accepted:
            return rank
    return None


def score(rows, retrieve_fn, k):
    per_q, groups = [], defaultdict(list)
    for r in rows:
        if r["type"] != "fact":
            continue
        accepted = parse_expected(r["expected"])
        hits = retrieve_fn(r["question"], k)  # hybrid returns up to k+2 results (see note)
        rank = first_rank(hits, accepted)
        rec = {"question": r["question"], "language": r["language"], "rank": rank,
               "top": [(h["source"], h["page"]) for h in hits[:3]], "expected": sorted(accepted)}
        per_q.append(rec)
        groups["all"].append(rec)
        groups[r["language"]].append(rec)
    summary = {}
    for g, recs in groups.items():
        n = len(recs)
        summary[g] = {"n": n,
                      "hit": sum(x["rank"] is not None for x in recs) / n,
                      "mrr": sum(1 / x["rank"] for x in recs if x["rank"]) / n}
    return per_q, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--csv", default="eval/questions.csv")
    ap.add_argument("--note", default="")
    a = ap.parse_args()

    import config as C
    from rag import retrieve
    k = a.k or C.TOP_K
    rows = list(csv.DictReader(open(a.csv, encoding="utf-8")))
    per_q, summary = score(rows, lambda q, kk: retrieve(q, k=kk, hybrid=a.hybrid), k)

    for r in per_q:
        mark = "✔" if r["rank"] else "✘"
        print(f"{mark} [{r['language']}] rank={r['rank']}  {r['question']}")
        if not r["rank"]:
            print(f"      expected: {r['expected']}\n      got top3: {r['top']}")
    n_un = sum(r["type"] == "unanswerable" for r in rows)
    print(f"\n(k={k}, hybrid={a.hybrid}, chunk_words={C.CHUNK_WORDS}; {n_un} unanswerable questions skipped here - they test the LLM refusal, Step 3)")
    print(f"{'group':6s} {'n':>3s} {'Hit@'+str(k):>7s} {'MRR':>6s}")
    for g in ["all"] + sorted(x for x in summary if x != "all"):
        s = summary[g]
        print(f"{g:6s} {s['n']:3d} {100*s['hit']:6.0f}% {s['mrr']:6.2f}")

    with open("eval/results.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M"), "k": k, "hybrid": a.hybrid,
                             "chunk_words": C.CHUNK_WORDS, "embed_model": C.EMBED_MODEL,
                             "note": a.note, "summary": summary}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
