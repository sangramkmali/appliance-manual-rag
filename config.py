"""Central settings – change here, not in the code."""
DATA_DIR = "data"              # put your PDF manuals here
DB_DIR = "chroma_db"           # local vector store (created automatically)
COLLECTION = "manuals"
EMBED_MODEL = "intfloat/multilingual-e5-small"   # German + English, runs locally
CHUNK_WORDS = 300              # chunk size in words (try 150 / 300 / 500 in the evaluation)
OVERLAP_WORDS = 50
TOP_K = 5
LLM_MODEL = "claude-sonnet-5-5" # needs ANTHROPIC_API_KEY in the environment
