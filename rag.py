"""Step 3+4 (+Step 2 retrieval upgrade): retrieve relevant chunks and let the LLM answer only from them, with citations."""
import re
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
    return BM25Okapi([tokenize(d) for d in data["documents"]]), data


def tokenize(text):
    """Lower-case word tokens; punctuation is stripped so 'KF96N,' still matches 'KF96N'."""
    return re.findall(r"\w+", text.lower())


def rrf_fuse(rank_lists, k, c=60):
    """Reciprocal Rank Fusion: score(d) = sum over lists of 1/(c + rank). Returns the best k ids.
    Needs no score normalisation, so vector distances and BM25 scores can be mixed safely."""
    score = {}
    for ranking in rank_lists:
        for rank, doc_id in enumerate(ranking, 1):
            score[doc_id] = score.get(doc_id, 0.0) + 1.0 / (c + rank)
    return sorted(score, key=lambda d: -score[d])[:k]


def retrieve(question, k=C.TOP_K, hybrid=False, fusion="rrf", pool=20):
    """fusion='rrf'    : vector + BM25 merged with Reciprocal Rank Fusion, exactly k results (fair).
       fusion='append' : legacy Step-1 merge (k vector hits + extra BM25 hits, up to k+2) - kept for comparison."""
    q = _model().encode(["query: " + question], normalize_embeddings=True)[0].tolist()
    if not hybrid or fusion == "append":
        r = _col().query(query_embeddings=[q], n_results=k)
        hits = [{"text": t, **m} for t, m in zip(r["documents"][0], r["metadatas"][0])]
        if hybrid:
            bm, data = _bm25_legacy()
            scores = bm.get_scores(question.lower().split())
            top = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
            seen = {(h["source"], h["page"], h["text"][:40]) for h in hits}
            for i in top:
                h = {"text": data["documents"][i], **data["metadatas"][i]}
                if (h["source"], h["page"], h["text"][:40]) not in seen:
                    hits.append(h)
            hits = hits[:k + 2]
        return hits
    r = _col().query(query_embeddings=[q], n_results=pool, include=["documents", "metadatas"])
    by_id = {i: {"text": t, **m} for i, t, m in zip(r["ids"][0], r["documents"][0], r["metadatas"][0])}
    vec_rank = list(r["ids"][0])
    bm, data = _bm25()
    scores = bm.get_scores(tokenize(question))
    top = sorted(range(len(scores)), key=lambda i: -scores[i])[:pool]
    kw_rank = []
    for i in top:
        if scores[i] <= 0:
            break
        by_id.setdefault(data["ids"][i], {"text": data["documents"][i], **data["metadatas"][i]})
        kw_rank.append(data["ids"][i])
    return [by_id[i] for i in rrf_fuse([vec_rank, kw_rank], k)]


@lru_cache
def _bm25_legacy():
    data = _col().get(include=["documents", "metadatas"])
    return BM25Okapi([d.lower().split() for d in data["documents"]]), data


def answer(question, hybrid=True):
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
