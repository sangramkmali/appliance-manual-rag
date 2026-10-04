"""Central settings – change here, or override per experiment with RAG_* environment variables."""
import os

def _env(name, default, cast=str):
    return cast(os.environ.get(name, default))

DATA_DIR = "data"                                   # put your PDF manuals here
DB_DIR = _env("RAG_DB_DIR", "chroma_db")            # local vector store (created automatically)
COLLECTION = "manuals"
EMBED_MODEL = _env("RAG_EMBED_MODEL", "intfloat/multilingual-e5-small")   # German + English, runs locally
CHUNK_WORDS = _env("RAG_CHUNK_WORDS", 150, int)     # chunk size in words (Step 2 winner: 150; tested 150 / 300 / 500)
OVERLAP_WORDS = _env("RAG_OVERLAP_WORDS", 25, int)
TOP_K = _env("RAG_TOP_K", 5, int)
LLM_MODEL = "claude-haiku-4-5-20251001"             # needs ANTHROPIC_API_KEY in the environment

# Step 2 retrieval improvements - defaults = best experiment (E4); override with RAG_* env vars to compare
CLEAN = _env("RAG_CLEAN", 1, int)                   # 1 = drop cover/TOC pages, remove dot leaders
CONTEXT_HEADER = _env("RAG_CONTEXT_HEADER", 1, int) # 1 = prefix every chunk with brand/model/language

# Step 3: cross-encoder re-ranking (second-stage model that reads question + chunk together)
RERANK_MODEL = _env("RAG_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")   # multilingual; alt: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
RERANK_POOL = _env("RAG_RERANK_POOL", 20, int)      # candidates passed to the re-ranker (then top-k are kept)

# Step 3: light stemming for BM25 (0 = off). N>0 truncates purely alphabetic words to their first N letters,
# so French 'rangez'/'ranger' or 'alimentaires'/'aliments' match. Language-agnostic; model numbers (with digits) are untouched.
BM25_STEM = _env("RAG_BM25_STEM", 0, int)
