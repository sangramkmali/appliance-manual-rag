# Appliance Manual Assistant – RAG demo

Ask questions about fridge/freezer user manuals (German or English) and get answers **only from the documents, with file + page citations**.

## Architecture
```
PDF manuals → cleaning (drop cover/TOC) → chunking (file+page kept) + context header (brand/model/language) → multilingual embeddings → Chroma vector DB
question → vector search + BM25 keyword search → Reciprocal Rank Fusion (top-5) → Claude with "answer only from sources" prompt → answer + citations
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

## Evaluation (Step 2)
Test set: `eval/questions.csv` – 23 questions on 18 manuals (DE/EN/FR/NL); 18 answerable ones with the expected page (used for retrieval metrics) and 5 unanswerable ones (refusal tests, Step 3).
Metrics: **Hit@5** (right page among the top 5) and **MRR** (rewards a higher rank). Reproduce everything with `python experiments.py` (local, no API cost); each run is logged to `eval/results.jsonl`.

| Exp | Change | Vector Hit@5 / MRR | Hybrid (RRF) Hit@5 / MRR |
|---|---|---|---|
| E0 | baseline (300-word chunks) | 50 % / 0.26 | 83 % / 0.60 |
| E1 | + drop cover/TOC pages, clean dot leaders | 56 % / 0.31 | 83 % / 0.64 |
| E2 | + context header (brand/model/language) per chunk | 83 % / 0.65 | 89 % / 0.79 |
| E3 | E1 + E2 | 83 % / 0.65 | 89 % / 0.79 |
| **E4** | **E3 with 150-word chunks (default)** | **83 % / 0.77** | **89 % / 0.84** |
| E5 | E3 with 500-word chunks | 78 % / 0.68 | 89 % / 0.71 |

Held-out check (`eval/heldout.csv`, 13 answerable + 3 unanswerable questions written after the setup was chosen and not used to tune it):

| Setup (E4: clean + header + 150-word chunks) | Hit@5 | MRR |
|---|---|---|
| Vector only | 100 % (13/13) | 0.79 |
| Hybrid (RRF) | 100 % (13/13) | 0.90 |

Both modes find the right page on all 13 held-out questions; hybrid ranks it higher (MRR 0.90 vs 0.79). With 13 questions this only says the true hit rate is very likely above ~75 %, not that it is 100 %. Hard cases ranked 3rd or 4th: Siemens GIV vs its twin manuals (same "60 hours" text), Siemens KIN holiday mode and door alarm.

### Step 3 - re-ranking and light stemming

Two further changes, each measured on the main set and on the held-out set (hybrid search, top 5):

| Setup | Main set Hit@5 / MRR | Held-out Hit@5 / MRR |
|---|---|---|
| E4 hybrid (Step 2 result) | 89 % / 0.84 | 100 % / 0.90 |
| + cross-encoder re-ranker (`BAAI/bge-reranker-v2-m3`, 20 candidates -> top 5) | 94 % / 0.92 | 100 % / 0.94 |
| **+ light stemming for BM25 (first 5 letters) (default)** | **100 % / 0.93** | **100 % / 0.94** |

- The re-ranker reads question and passage together. It fixed the Liebherr "F1 to F5" question and moved the Siemens GIV/KIN questions from rank 3 to rank 1. One side effect: a Dutch question slipped from rank 1 to rank 2.
- The remaining French miss (page 53, *"ne rangez pas de produits alimentaires"*) was never in the 20 candidates: the question says *ranger* / *aliments*, the page says *rangez* / *alimentaires*, and BM25 without stemming treats them as different words. Cutting alphabetic words to their first 5 letters (language-agnostic, model numbers untouched) brought the page into the candidates (rank 4 after re-ranking). On the held-out set it changed nothing, so it did not hurt.
- Caveat: the stemming rule was chosen after seeing that one failure; the held-out result only shows it does no harm there. The sets are still small (1 question = 5.6 points on the main set).
- Reproduce the Step 2 numbers with `RAG_BM25_STEM=0 RAG_RERANK=0`. Cost of re-ranking: a 2.3 GB model on CPU - `evaluate.py` now prints the mean search time per question so the accuracy gain can be weighed against latency.

The Step 2 notes on the two main-set misses (kept for the record; both are resolved by the changes above), E4 with hybrid: the Liebherr HC 2090G "F1 to F5" question (the troubleshooting page is retrieved twice instead of page 11, the only page that lists the codes) and a French Bosch KGN question ("combien de temps ... attendre avant de ranger des aliments", answer on page 53, which says "plusieurs heures"; retrieval returns other French pages). Both are wording mismatches between question and manual.

What the numbers say
- **Context header was the biggest single gain** (vector 50 → 83 %): content pages rarely mention brand/model, so brand-specific questions could not find them.
- Hybrid search beat vector search at baseline (83 vs 50 %). Note: E0 hybrid already uses the new RRF fusion and tokenizer, so it is not identical to the Step 1 hybrid.
- Cleaning adds little once headers exist (E3 = E2).
- Smaller chunks rank the right page higher (MRR 0.65 → 0.77 vector, 0.79 → 0.84 hybrid); 500 words is worse.
- Caveat: only 18 answerable questions (1 question = 5.6 points) and the fixes were designed after seeing the failures, so a held-out question set is needed (planned for Step 3).

## Design decisions (talking points)
- Local multilingual embeddings – no data leaves the machine except the final prompt.
- Citations + "not covered" rule as a guardrail against hallucination.
- Hybrid search (vector + BM25, Reciprocal Rank Fusion) because manuals contain exact terms (error codes, model numbers) that pure vector search can miss.
- Context header on every chunk, because content pages do not repeat brand/model.
- Measured, not guessed: chunk size and retrieval mode chosen by hit rate.

## Next steps
Answer-quality evaluation (LLM-as-judge: faithfulness, citation correctness, refusals; prompt-injection test), Databricks Vector Search / Azure AI Search deployment, cost and latency logging.
