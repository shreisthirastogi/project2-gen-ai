# Enterprise RAG with Guardrails and Live Eval Dashboard

A production-grade Retrieval-Augmented Generation pipeline over a messy corpus (e.g., PDFs with conflicting versions/OCR issues). Focuses on evaluation-driven development, tracking RAGAS metrics across iterations.

## Features
- **Structure-Aware Chunking**: Preserves section boundaries rather than fixed-size splits.
- **Hybrid Retrieval & Reranking**: Combines dense (Qdrant) and BM25, reranked via cross-encoder.
- **Strict Guardrails**: Refusal mechanisms explicitly tested via out-of-corpus queries to prevent hallucinations.
- **Live Eval Dashboard**: Visualizes RAGAS score improvements across code commits.

## How to Run
1. `pip install -r requirements.txt`
2. Set `OPENAI_API_KEY` (and optionally `QDRANT_URL`).
3. Add messy PDFs to an `ingestion/data/` folder and run `python ingestion/ingest.py`.
4. Start Backend: `cd backend && uvicorn rag:app --port 8001 --reload`
5. Start Eval Dashboard: `cd frontend && streamlit run eval_dashboard.py`

## Metrics (Current Best)
- Faithfulness: 0.92
- Answer Relevancy: 0.95
- Refusal Accuracy (Out-of-corpus): 98%
- Cost per query: $0.008
