"""Step 1+2: read PDFs, clean, split into chunks (keeping file + page), embed, store in Chroma."""
import os, re, glob
import config as C

DOT_LEADER = re.compile(r"(?:\.\s?){4,}")          # '..........' or '. . . . .' (table-of-contents leaders)
TOC_LINE = re.compile(r"(?:\.\s?){5,}\s*\d+")      # '...... 12'  = one table-of-contents entry


def clean_text(text):
    """Remove dot leaders and collapse whitespace."""
    return " ".join(DOT_LEADER.sub(" ", text).split())


TOC_STRONG = re.compile(r"inhaltsverzeichnis|table of contents|sommaire|inhoudsopgave|indice", re.I)
TOC_WEAK = re.compile(r"(?:inhalt|contents|inhoud)\b", re.I)   # only counts right at the top of the page


def is_front_matter(page_no, raw_text):
    """Cover page (page 1) and table-of-contents pages (early pages only): they match many
    queries but answer none."""
    if page_no == 1:
        return True
    if page_no > 6:
        return False
    flat = " ".join(raw_text.split())
    weak = TOC_WEAK.search(flat[:25])
    return bool(TOC_STRONG.search(flat[:150])) or bool(weak) or len(TOC_LINE.findall(raw_text)) >= 2


def context_header(filename):
    """BOSCH_KGN_DE-EN-FR-IT-NL.pdf -> 'Manual: BOSCH KGN | languages: DE EN FR IT NL'.
    Content pages rarely repeat the brand/model, so we add it to every chunk."""
    parts = os.path.splitext(filename)[0].split("_")
    brand = parts[0]
    model = " ".join(parts[1:-1]) if len(parts) > 2 else (parts[1] if len(parts) > 1 else "")
    langs = parts[-1].replace("-", " ") if len(parts) > 2 else ""
    return f"Manual: {brand} {model} | languages: {langs}".strip()


def chunks_from_pdf(path, clean=None, header=None):
    from pypdf import PdfReader
    clean = C.CLEAN if clean is None else clean
    header = C.CONTEXT_HEADER if header is None else header
    name = os.path.basename(path)
    prefix = context_header(name) + "\n" if header else ""
    for page_no, page in enumerate(PdfReader(path).pages, start=1):
        raw = page.extract_text() or ""
        if clean:
            if is_front_matter(page_no, raw):
                continue
            raw = clean_text(raw)
        words = raw.split()
        step = C.CHUNK_WORDS - C.OVERLAP_WORDS
        for i in range(0, max(len(words), 1), step):
            text = " ".join(words[i:i + C.CHUNK_WORDS])
            if len(text) > 50:
                yield {"id": f"{name}-p{page_no}-{i}", "text": prefix + text, "source": name, "page": page_no}


def main():
    from sentence_transformers import SentenceTransformer
    import chromadb
    pdfs = sorted(glob.glob(os.path.join(C.DATA_DIR, "*.pdf")))
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
    print(f"Indexed {len(docs)} chunks from {len(pdfs)} PDFs "
          f"(clean={C.CLEAN}, header={C.CONTEXT_HEADER}, chunk_words={C.CHUNK_WORDS}, db={C.DB_DIR}).")


if __name__ == "__main__":
    main()
