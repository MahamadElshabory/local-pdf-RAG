import datetime
import logging
import os
import uuid
from pathlib import Path

import inngest
import inngest.fast_api
from dotenv import load_dotenv
from fastapi import FastAPI
from ollama import AsyncClient

from custom_types import (
    RAGChunkAndSrc,
    RAGQueryResult,
    RAGSearchResult,
    RAGUpsertResult,
)
from data_loader import EMBED_DIM, embed_documents , load_and_chunk_pdf , embed_query
from vector_db import QdrantStorage


# ---------------------------------------------------------
# Load environment variables
# ---------------------------------------------------------

load_dotenv()


OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)

OLLAMA_LLM_MODEL = os.getenv(
    "OLLAMA_LLM_MODEL",
    "gemma2:2b",
)

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://localhost:6333",
)

QDRANT_COLLECTION = os.getenv(
    "QDRANT_COLLECTION",
    "docs_nomic",
)


# ---------------------------------------------------------
# Create reusable clients
# ---------------------------------------------------------

ollama_client = AsyncClient(
    host=OLLAMA_BASE_URL,
)


def get_vector_store() -> QdrantStorage:
    """
    Create a QdrantStorage object using the local Qdrant settings.

    EMBED_DIM should be 768 when using nomic-embed-text.
    """
    return QdrantStorage(
        url=QDRANT_URL,
        collection=QDRANT_COLLECTION,
        dim=EMBED_DIM,
    )


# ---------------------------------------------------------
# Create Inngest client
# ---------------------------------------------------------

inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)


# ---------------------------------------------------------
# Workflow 1: Load, chunk, embed and store a PDF
# ---------------------------------------------------------

@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(
        event="rag/ingest_pdf",
    ),
    retries=2,
)
async def rag_ingest_pdf(
    ctx: inngest.Context,
) -> dict:
    """
    Expected event data:

    {
        "pdf_path": "C:/documents/example.pdf",
        "source_id": "example.pdf"
    }

    source_id is optional.
    """

    def load_pdf() -> RAGChunkAndSrc:
        event_data = ctx.event.data

        pdf_path = str(
            event_data.get("pdf_path", "")
        ).strip()

        if not pdf_path:
            raise ValueError(
                "The event must contain a non-empty 'pdf_path'."
            )

        if not Path(pdf_path).is_file():
            raise FileNotFoundError(
                f"PDF file was not found: {pdf_path}"
            )

        source_id = str(
            event_data.get("source_id") or pdf_path
        )

        chunks = load_and_chunk_pdf(pdf_path)

        if not chunks:
            raise ValueError(
                "The PDF did not contain readable text."
            )

        return RAGChunkAndSrc(
            chunks=chunks,
            source_id=source_id,
        )

    def embed_and_upsert(
        chunks_and_source: RAGChunkAndSrc,
    ) -> RAGUpsertResult:
        chunks = chunks_and_source.chunks
        source_id = chunks_and_source.source_id

        # Convert every PDF chunk into an embedding vector.
        vectors = embed_documents(chunks)

        if len(vectors) != len(chunks):
            raise ValueError(
                "The number of embeddings does not match "
                "the number of chunks."
            )

        # Generate a repeatable UUID for each chunk.
        ids = [
            str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{source_id}:{index}",
                )
            )
            for index in range(len(chunks))
        ]

        # Store the readable text beside each vector.
        payloads = [
            {
                "source": source_id,
                "text": chunk,
                "chunk_index": index,
            }
            for index, chunk in enumerate(chunks)
        ]

        vector_store = get_vector_store()

        vector_store.upsert(
            ids=ids,
            vectors=vectors,
            payloads=payloads,
        )

        return RAGUpsertResult(
            ingested=len(chunks),
        )

    chunks_and_source = await ctx.step.run(
        "load-and-chunk",
        load_pdf,
        output_type=RAGChunkAndSrc,
    )

    ingestion_result = await ctx.step.run(
        "embed-and-upsert",
        lambda: embed_and_upsert(chunks_and_source),
        output_type=RAGUpsertResult,
    )

    return ingestion_result.model_dump()


# ---------------------------------------------------------
# Workflow 2: Search the PDF data and answer with Ollama
# ---------------------------------------------------------

@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(
        event="rag/query_pdf_ai",
    ),
    retries=2,
)
async def rag_query_pdf_ai(
    ctx: inngest.Context,
) -> dict:
    """
    Expected event data:

    {
        "question": "What does the document say about FastAPI?",
        "top_k": 5
    }

    top_k is optional.
    """

    def search_documents(
    question: str,
    top_k: int,
) -> RAGSearchResult:
        query_vector = embed_query(question)

        vector_store = get_vector_store()

        found = vector_store.search(
        query_vector=query_vector,
        top_k=top_k,
    )

        return RAGSearchResult(
        contexts=found["contexts"],
        sources=found["sources"],
        )

    event_data = ctx.event.data

    question = str(
        event_data.get("question", "")
    ).strip()

    if not question:
        raise ValueError(
            "The event must contain a non-empty 'question'."
        )

    try:
        top_k = int(
            event_data.get("top_k", 5)
        )
    except (TypeError, ValueError):
        raise ValueError(
            "'top_k' must be an integer."
        )

    # Prevent invalid or excessively large retrieval values.
    top_k = max(1, min(top_k, 10))

    search_result = await ctx.step.run(
        "embed-and-search",
        lambda: search_documents(
            question,
            top_k,
        ),
        output_type=RAGSearchResult,
    )

    # Do not ask the model to invent an answer when Qdrant
    # did not return relevant contexts.
    if not search_result.contexts:
        result = RAGQueryResult(
            answer=(
                "I could not find relevant information "
                "in the uploaded documents."
            ),
            sources=[],
            num_contexts=0,
        )

        return result.model_dump()

    context_block = "\n\n".join(
        f"[Context {index}]\n{context}"
        for index, context in enumerate(
            search_result.contexts,
            start=1,
        )
    )

    async def generate_answer() -> RAGQueryResult:
        response = await ollama_client.chat(
            model=OLLAMA_LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a document question-answering assistant. "
                        "Answer using only the supplied context. "
                        "If the context does not contain the answer, "
                        "say that the information was not found."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Context:\n{context_block}\n\n"
                        f"Question:\n{question}\n\n"
                        "Give a clear and concise answer."
                    ),
                },
            ],
            options={
                "temperature": 0.2,
                "num_predict": 1024,
            },
            stream=False,
        )

        answer = (
            response.message.content or ""
        ).strip()

        if not answer:
            raise ValueError(
                "Ollama returned an empty answer."
            )

        return RAGQueryResult(
            answer=answer,
            sources=search_result.sources,
            num_contexts=len(
                search_result.contexts
            ),
        )

    final_result = await ctx.step.run(
        "ollama-answer",
        generate_answer,
        output_type=RAGQueryResult,
    )

    return final_result.model_dump()


# ---------------------------------------------------------
# Create FastAPI application
# ---------------------------------------------------------

app = FastAPI(
    title="Local RAG Application",
    description="RAG backend using Ollama, Qdrant and Inngest.",
)


@app.get("/")
def root() -> dict:
    return {
        "message": "Local RAG backend is running.",
        "llm_model": OLLAMA_LLM_MODEL,
        "embedding_dimension": EMBED_DIM,
        "qdrant_collection": QDRANT_COLLECTION,
    }


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "ollama_url": OLLAMA_BASE_URL,
        "qdrant_url": QDRANT_URL,
    }


# Register the Inngest functions with FastAPI.
inngest.fast_api.serve(
    app,
    inngest_client,
    functions=[
        rag_ingest_pdf,
        rag_query_pdf_ai,
    ],
)