"""Step 3+4: retrieve relevant chunks and let the LLM answer only from them, with citations."""
from functools import lru_cache
from sentence_transformers import SentenceTransformer
import chromadb, anthropic
from rank_bm25 import BM25Okapi
import config as C

SYSTEM = ("You answer questions about appliance manuals. Use ONLY the numbered sources provided. "
          "Cite sources as [n] after each statement. If the answer is not in the sources, say "
          "'This is not covered in the documents.' Answer in the language of the question.")

@lru_cache
def _model():
    return SentenceTransformer(C.EMBED_MODEL)

@lru_cache
def _col():
    return chromadb.PersistentClient(path=C.DB_DIR).get_collection(C.COLLECTION)

@lru_cache
def _bm25():
    data = _col().get(include=["documents", "metadatas"])
    return BM25Okapi([d.lower().split() for d in data["documents"]]), data

def retrieve(question, k=C.TOP_K, hybrid=False):
    q = _model().encode(["query: " + question], normalize_embeddings=True)[0].tolist()
    r = _col().query(query_embeddings=[q], n_results=k)
    hits = [{"text": t, **m} for t, m in zip(r["documents"][0], r["metadatas"][0])]
    if hybrid:  # Step 6 option: add keyword (BM25) hits, merge without duplicates
        bm, data = _bm25()
        scores = bm.get_scores(question.lower().split())
        top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        seen = {(h["source"], h["page"], h["text"][:40]) for h in hits}
        for i in top:
            h = {"text": data["documents"][i], **data["metadatas"][i]}
            if (h["source"], h["page"], h["text"][:40]) not in seen:
                hits.append(h)
        hits = hits[:k + 2]
    return hits

def answer(question, hybrid=False):
    hits = retrieve(question, hybrid=hybrid)
    ctx = "\n\n".join(f"[{i}] ({h['source']}, p.{h['page']})\n{h['text']}" for i, h in enumerate(hits, 1))
    msg = anthropic.Anthropic().messages.create(
        model=C.LLM_MODEL, max_tokens=600, system=SYSTEM,
        messages=[{"role": "user", "content": f"Sources:\n{ctx}\n\nQuestion: {question}"}])
    return msg.content[0].text, hits

if __name__ == "__main__":
    import sys
    text, hits = answer(" ".join(sys.argv[1:]) or "How do I defrost the freezer?")
    print(text, "\n\nSources:")
    for i, h in enumerate(hits, 1):
        print(f"[{i}] {h['source']} p.{h['page']}")
