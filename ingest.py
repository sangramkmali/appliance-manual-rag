"""Step 1+2: read PDFs, split into chunks (keeping file + page), embed, store in Chroma."""
import os, glob
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import config as C

def chunks_from_pdf(path):
    name = os.path.basename(path)
    for page_no, page in enumerate(PdfReader(path).pages, start=1):
        words = (page.extract_text() or "").split()
        step = C.CHUNK_WORDS - C.OVERLAP_WORDS
        for i in range(0, max(len(words), 1), step):
            text = " ".join(words[i:i + C.CHUNK_WORDS])
            if len(text) > 50:
                yield {"id": f"{name}-p{page_no}-{i}", "text": text, "source": name, "page": page_no}

def main():
    pdfs = glob.glob(os.path.join(C.DATA_DIR, "*.pdf"))
    if not pdfs:
        raise SystemExit(f"No PDFs in {C.DATA_DIR}/ – add some manuals first.")
    docs = [c for p in pdfs for c in chunks_from_pdf(p)]
    model = SentenceTransformer(C.EMBED_MODEL)
    # e5 models expect a "passage:" / "query:" prefix
    vecs = model.encode(["passage: " + d["text"] for d in docs], show_progress_bar=True, normalize_embeddings=True)
    client = chromadb.PersistentClient(path=C.DB_DIR)
    try:
        client.delete_collection(C.COLLECTION)
    except Exception:
        pass
    col = client.create_collection(C.COLLECTION, metadata={"hnsw:space": "cosine"})
    for i in range(0, len(docs), 500):
        b = docs[i:i + 500]
        col.add(ids=[d["id"] for d in b], documents=[d["text"] for d in b],
                embeddings=[v.tolist() for v in vecs[i:i + 500]],
                metadatas=[{"source": d["source"], "page": d["page"]} for d in b])
    print(f"Indexed {len(docs)} chunks from {len(pdfs)} PDFs.")

if __name__ == "__main__":
    main()
