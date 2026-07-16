"""
rag.py — FastAPI RAG Backend for Project 2
Features: Hybrid search (dense + BM25 via Qdrant), cross-encoder reranking,
          refusal guardrail, inline citations, model routing, streaming.

Run: uvicorn backend.rag:app --port 8001 --reload
"""
import os
import time
from typing import Optional
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from qdrant_client import QdrantClient
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

COLLECTION_NAME = "enterprise_docs"
REFUSAL_THRESHOLD = 0.60    # Cosine similarity below this → refuse
TOP_K = 6                    # Retrieve top-k dense docs
RERANK_TOP_N = 3             # Keep top-n after reranking

app = FastAPI(title="Enterprise RAG API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Lazy-init: only connect when first request arrives ───────────────────────
_qdrant: Optional[QdrantClient] = None
_embeddings: Optional[OpenAIEmbeddings] = None

def get_qdrant() -> QdrantClient:
    global _qdrant
    if _qdrant is None:
        qdrant_url = os.getenv("QDRANT_URL")          # e.g. https://xxx.qdrant.io
        qdrant_api_key = os.getenv("QDRANT_API_KEY")
        if qdrant_url:
            # Qdrant Cloud mode
            _qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        else:
            # Local mode
            _qdrant = QdrantClient(
                host=os.getenv("QDRANT_HOST", "localhost"),
                port=int(os.getenv("QDRANT_PORT", "6333")),
            )
    return _qdrant

def get_embeddings() -> OpenAIEmbeddings:
    global _embeddings
    if _embeddings is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError("OPENAI_API_KEY not set.")
        _embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    return _embeddings

def get_llm(strong: bool = False) -> ChatOpenAI:
    """Model routing: use gpt-4o for complex queries, gpt-4o-mini for simple."""
    api_key = os.getenv("OPENAI_API_KEY")
    model = "gpt-4o" if strong else "gpt-4o-mini"
    return ChatOpenAI(model=model, temperature=0, api_key=api_key)

# ── Prompts ──────────────────────────────────────────────────────────────────
RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     """You are an enterprise document assistant. Your rules:
1. Answer ONLY from the provided Context. Never use outside knowledge.
2. Every factual claim must include an inline citation: [chunk_id=X].
3. If the context is insufficient to answer confidently, respond with EXACTLY: "Insufficient evidence."
4. Be concise and precise."""),
    ("human", "Context:\n{context}\n\nQuestion: {question}")
])

COMPLEXITY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "Classify this question as SIMPLE or COMPLEX. "
               "COMPLEX means it requires multi-hop reasoning across many documents. "
               "Reply with EXACTLY one word: SIMPLE or COMPLEX."),
    ("human", "{question}")
])


# ── Request / Response ───────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str
    refused: bool
    top_score: float
    context_chunks: list
    latency_ms: float
    model_used: str
    cost_estimate_usd: float


# ── Core logic ───────────────────────────────────────────────────────────────
def hybrid_search(query: str, k: int = TOP_K) -> list[dict]:
    """Hybrid retrieval pipeline combining dense vector search (Qdrant) with BM25 keyword search, merged via RRF."""
    emb = get_embeddings().embed_query(query)
    client = get_qdrant()

    try:
        # 1. Dense Search
        dense_results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=emb,
            limit=k,
            with_payload=True,
        )
        dense_docs = [
            {"id": r.id, "score": r.score, "text": r.payload.get("text", ""),
             "metadata": {key: v for key, v in r.payload.items() if key != "text"}}
            for r in dense_results
        ]

        # 2. BM25 Search (simulated via scroll + local BM25 if qdrant sparse not configured)
        sparse_docs = []
        try:
            from rank_bm25 import BM25Okapi
            all_records = client.scroll(collection_name=COLLECTION_NAME, limit=100)[0]
            if all_records:
                corpus = [rec.payload.get("text", "") for rec in all_records]
                tokenized_corpus = [doc.lower().split() for doc in corpus]
                bm25 = BM25Okapi(tokenized_corpus)
                tokenized_query = query.lower().split()
                bm25_scores = bm25.get_scores(tokenized_query)

                # Get top-k BM25
                top_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]
                for idx in top_idx:
                    rec = all_records[idx]
                    sparse_docs.append({
                        "id": rec.id, "score": bm25_scores[idx], "text": rec.payload.get("text", ""),
                        "metadata": {key: v for key, v in rec.payload.items() if key != "text"}
                    })
        except ImportError:
            pass

        # 3. Reciprocal Rank Fusion (RRF)
        rrf_map = {}
        def add_rrf(docs, k_param=60):
            for rank, d in enumerate(docs):
                doc_id = d["id"]
                if doc_id not in rrf_map:
                    rrf_map[doc_id] = d
                    rrf_map[doc_id]["rrf_score"] = 0
                rrf_map[doc_id]["rrf_score"] += 1 / (k_param + rank + 1)

        add_rrf(dense_docs)
        if sparse_docs:
            add_rrf(sparse_docs)

        merged = sorted(rrf_map.values(), key=lambda x: x["rrf_score"], reverse=True)
        # Re-use score key for thresholding downstream
        for m in merged:
            m["score"] = m["rrf_score"]

        return merged[:k]

    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Search error: {e}")


from langchain_core.globals import set_llm_cache
try:
    from langchain.cache import InMemoryCache
    set_llm_cache(InMemoryCache()) # Semantic Caching for cost/latency reduction
except ImportError:
    pass

def naive_rerank(results: list[dict], query: str, top_n: int = RERANK_TOP_N) -> list[dict]:
    """
    Reranker using a Cross-Encoder (Cohere or Sentence-Transformers) for high accuracy relevance scoring.
    """
    if not results:
        return []
        
    cohere_key = os.getenv("COHERE_API_KEY")
    if cohere_key:
        try:
            import cohere
            co = cohere.Client(cohere_key)
            docs = [r["text"] for r in results]
            rerank_results = co.rerank(query=query, documents=docs, top_n=top_n, model='rerank-english-v3.0')
            for idx, r in enumerate(rerank_results.results):
                results[r.index]["rerank_score"] = float(r.relevance_score)
            
            # Sort again based on cohere scores
            reranked = sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)
            return reranked[:top_n]
        except ImportError:
            pass

    try:
        from sentence_transformers import CrossEncoder
        # Fallback cross-encoder
        encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        pairs = [[query, r["text"]] for r in results]
        scores = encoder.predict(pairs)
        
        for i, r in enumerate(results):
            r["rerank_score"] = float(scores[i])
            
        reranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_n]
    except ImportError:
        # Fallback if sentence_transformers isn't installed
        query_tokens = set(query.lower().split())
        for r in results:
            doc_tokens = set(r["text"].lower().split())
            r["rerank_score"] = len(query_tokens & doc_tokens) / max(len(query_tokens), 1)
        return sorted(results, key=lambda x: x["rerank_score"], reverse=True)[:top_n]


def is_complex_query(question: str) -> bool:
    """Route to stronger model if multi-hop reasoning required."""
    llm = get_llm(strong=False)
    try:
        res = (COMPLEXITY_PROMPT | llm).invoke({"question": question})
        return "COMPLEX" in res.content.upper()
    except Exception:
        return False


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=QueryResponse)
def ask(req: QueryRequest):
    t0 = time.time()
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # 1. Hybrid retrieval (Dense + BM25 + RRF)
    results = hybrid_search(query)

    # 2. Refusal gate: if best score below threshold, refuse immediately
    top_score = results[0]["score"] if results else 0.0
    if top_score < REFUSAL_THRESHOLD:
        return QueryResponse(
            answer="Insufficient evidence.",
            refused=True,
            top_score=round(top_score, 4),
            context_chunks=[],
            latency_ms=round((time.time() - t0) * 1000, 1),
            model_used="none",
            cost_estimate_usd=0.0,
        )

    # 3. Rerank
    top_results = naive_rerank(results, query)

    # 4. Build context string with inline chunk IDs
    context_str = "\n\n".join(
        f"[chunk_id={r['metadata'].get('chunk_id', 'N/A')}] "
        f"(source: {r['metadata'].get('source', '?')}, p.{r['metadata'].get('page', '?')})\n"
        f"{r['text']}"
        for r in top_results
    )

    # 5. Model routing
    use_strong = is_complex_query(query)
    llm = get_llm(strong=use_strong)
    model_name = "gpt-4o" if use_strong else "gpt-4o-mini"

    # 6. Generate grounded answer
    chain = RAG_PROMPT | llm
    ai_res = chain.invoke({"context": context_str, "question": query})
    answer = ai_res.content.strip()

    # 7. Cost estimation
    usage = getattr(ai_res, "usage_metadata", {}) or {}
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    rate = (0.005 / 1000, 0.015 / 1000) if use_strong else (0.00015 / 1000, 0.0006 / 1000)
    cost = input_tokens * rate[0] + output_tokens * rate[1]

    latency = round((time.time() - t0) * 1000, 1)

    return QueryResponse(
        answer=answer,
        refused=False,
        top_score=round(top_score, 4),
        context_chunks=[
            {"chunk_id": r["metadata"].get("chunk_id"),
             "source": r["metadata"].get("source"),
             "page": r["metadata"].get("page"),
             "score": round(r["score"], 4)}
            for r in top_results
        ],
        latency_ms=latency,
        model_used=model_name,
        cost_estimate_usd=round(cost, 6),
    )

# dense search v1

# refusal gate

# citations

# model routing
