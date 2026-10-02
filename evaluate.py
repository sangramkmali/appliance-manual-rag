"""Step 5+6: measure retrieval hit rate on your test questions.
eval/questions.csv columns: question,expected_source,expected_page
Run: python evaluate.py            (vector only)
     python evaluate.py --hybrid   (vector + keyword)"""
import csv, sys
from rag import retrieve

hybrid = "--hybrid" in sys.argv
rows = list(csv.DictReader(open("eval/questions.csv", encoding="utf-8")))
hits = 0
for r in rows:
    got = retrieve(r["question"], hybrid=hybrid)
    ok = any(h["source"] == r["expected_source"] and str(h["page"]) == r["expected_page"] for h in got)
    hits += ok
    print(("✔" if ok else "✘"), r["question"])
print(f"\nRetrieval hit rate: {hits}/{len(rows)} = {100 * hits / max(len(rows), 1):.0f}%  (hybrid={hybrid})")
