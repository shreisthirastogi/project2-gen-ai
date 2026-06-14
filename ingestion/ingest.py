"""
ingest.py — Document ingestion pipeline for Project 2 (Enterprise RAG)

Usage:
    python ingestion/ingest.py --data_dir ingestion/data --qdrant_host localhost --qdrant_port 6333

Supports: PDF files (including messy/scanned PDFs via pdfplumber)
"""
import os
import argparse
import logging
from pathlib import Path

import pdfplumber
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(message)s")
log = logging.getLogger(__name__)

COLLECTION_NAME = "enterprise_docs"
EMBEDDING_DIM = 1536  # OpenAI text-embedding-3-small


def extract_text_from_pdf(pdf_path: str) -> list[Document]:
    """
    Extracts text from a PDF using pdfplumber (handles messy/OCR PDFs better than pypdf).
    Preserves page and section metadata.
    """
    docs = []
    filename = Path(pdf_path).name

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "source": filename,
                        "page": page_num,
                        "version": "v1",  # Update from filename/header in prod
                        "total_pages": len(pdf.pages),
                    }
                ))

    log.info(f"  Extracted {len(docs)} pages from {filename}")
    return docs


def chunk_documents(docs: list[Document]) -> list[Document]:
    """Structure-aware chunking: splits on section boundaries first."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)

    # Enrich each chunk with a unique ID for citation
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
        chunk.metadata["char_count"] = len(chunk.page_content)

    log.info(f"  Created {len(chunks)} chunks from {len(docs)} pages.")
    return chunks


def upsert_to_qdrant(
    chunks: list[Document],
    qdrant_client: QdrantClient,
    embeddings: OpenAIEmbeddings,
):
    """Embeds and upserts chunks into Qdrant."""
    # Recreate collection (idempotent)
    if qdrant_client.collection_exists(COLLECTION_NAME):
        qdrant_client.delete_collection(COLLECTION_NAME)

    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )

    texts = [c.page_content for c in chunks]
    metas = [c.metadata for c in chunks]

    log.info(f"  Embedding {len(texts)} chunks (this may take a moment)…")
    vectors = embeddings.embed_documents(texts)

    from qdrant_client.models import PointStruct
    points = [
        PointStruct(id=i, vector=vec, payload={**meta, "text": txt})
        for i, (vec, txt, meta) in enumerate(zip(vectors, texts, metas))
    ]

    # Batch upload in groups of 100
    batch_size = 100
    for start in range(0, len(points), batch_size):
        qdrant_client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[start: start + batch_size],
        )

    log.info(f"✅ Upserted {len(points)} vectors to Qdrant collection '{COLLECTION_NAME}'.")


def run_ingestion(data_dir: str, qdrant_host: str = None, qdrant_port: int = 6333):
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY not set.")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)

    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    if qdrant_url:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        log.info(f"Connected to Qdrant Cloud: {qdrant_url}")
    else:
        client = QdrantClient(host=qdrant_host or "localhost", port=qdrant_port)
        log.info(f"Connected to local Qdrant: {qdrant_host}:{qdrant_port}")

    pdf_files = list(Path(data_dir).glob("*.pdf"))
    if not pdf_files:
        log.warning(f"No PDF files found in {data_dir}. "
                    "Add PDFs (e.g. SEC 10-K filings) and re-run.")
        return

    all_docs = []
    for pdf in pdf_files:
        log.info(f"Processing: {pdf}")
        all_docs.extend(extract_text_from_pdf(str(pdf)))

    chunks = chunk_documents(all_docs)
    upsert_to_qdrant(chunks, client, embeddings)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="ingestion/data")
    parser.add_argument("--qdrant_host", default="localhost")
    parser.add_argument("--qdrant_port", type=int, default=6333)
    args = parser.parse_args()
    run_ingestion(args.data_dir, args.qdrant_host, args.qdrant_port)

# pdfplumber extraction

# chunking v1
