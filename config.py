"""Central settings – change here, or override per experiment with RAG_* environment variables."""
import os

def _env(name, default, cast=str):
    return cast(os.environ.get(name, default))

DATA_DIR = "data"                                   # put your PDF manuals here
DB_DIR = _env("RAG_DB_DIR", "chroma_db")            # local vector store (created automatically)
COLLECTION = "manuals"
EMBED_MODEL = _env("RAG_EMBED_MODEL", "intfloat/multilingual-e5-small")   # German + English, runs locally
CHUNK_WORDS = _env("RAG_CHUNK_WORDS", 300, int)     # chunk size in words (experiments: 150 / 300 / 500)
OVERLAP_WORDS = _env("RAG_OVERLAP_WORDS", 50, int)
TOP_K = _env("RAG_TOP_K", 5, int)
LLM_MODEL = "claude-haiku-4-5-20251001"             # needs ANTHROPIC_API_KEY in the environment

# Step 2 retrieval improvements (each can be switched on/off to measure its effect)
CLEAN = _env("RAG_CLEAN", 0, int)                   # 1 = drop cover/TOC pages, remove dot leaders
CONTEXT_HEADER = _env("RAG_CONTEXT_HEADER", 0, int) # 1 = prefix every chunk with brand/model/language
