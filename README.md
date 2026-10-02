# Appliance Manual Assistant – RAG demo

Ask questions about fridge/freezer user manuals (German or English) and get answers **only from the documents, with file + page citations**.

## Architecture
```
PDF manuals → chunking (file+page kept) → multilingual embeddings → Chroma vector DB
question → embedding → top-k retrieval (+ optional BM25 keyword search) → Claude with "answer only from sources" prompt → answer + citations
```

## Setup (Windows, Python 3.11)
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
set ANTHROPIC_API_KEY=your_key_here
```
1. Put 5–10 **publicly available** appliance manuals (PDF) into `data/`.
2. `python ingest.py` – builds the vector database.
3. `python rag.py "Wie taue ich das Gefrierfach ab?"` – quick test.
4. `streamlit run app.py` – chat UI.

## Evaluation
1. Replace the examples in `eval/questions.csv` with 15–20 real questions and where the answer is (file, page).
2. `python evaluate.py` and `python evaluate.py --hybrid`.
3. Change `CHUNK_WORDS` in `config.py` (150 / 300 / 500), re-run `ingest.py` and `evaluate.py`, and record results:

| Setting | Hit rate |
|---|---|
| 300 words, vector only | |
| 300 words, hybrid | |
| … | |

## Design decisions (talking points)
- Local multilingual embeddings – no data leaves the machine except the final prompt.
- Citations + "not covered" rule as a guardrail against hallucination.
- Hybrid search because manuals contain exact terms (error codes, model numbers) that pure vector search can miss.
- Measured, not guessed: chunk size and retrieval mode chosen by hit rate.

## Next steps
Re-ranking, answer-quality evaluation (LLM-as-judge), Databricks Vector Search / Azure AI Search deployment, cost and latency logging.
