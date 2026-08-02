import os

from dotenv import load_dotenv
from llama_index.core.node_parser import SentenceSplitter
from llama_index.readers.file import PDFReader
from ollama import Client


load_dotenv()

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
)

OLLAMA_EMBEDDING_MODEL = os.getenv(
    "OLLAMA_EMBEDDING_MODEL",
    "nomic-embed-text",
)

EMBED_DIM = 768

ollama_client = Client(host=OLLAMA_BASE_URL)


# Smaller chunks generally make specific facts easier to retrieve.
splitter = SentenceSplitter(
    chunk_size=500,
    chunk_overlap=100,
)


def load_and_chunk_pdf(path: str) -> list[str]:
    docs = PDFReader().load_data(file=path)

    texts = [
        document.text
        for document in docs
        if getattr(document, "text", None)
    ]

    chunks = []

    for text in texts:
        chunks.extend(splitter.split_text(text))

    return chunks


def embed_documents(
    texts: list[str],
) -> list[list[float]]:
    """Create embeddings for PDF chunks."""

    if not texts:
        return []

    prepared_texts = [
        f"search_document: {text}"
        for text in texts
    ]

    response = ollama_client.embed(
        model=OLLAMA_EMBEDDING_MODEL,
        input=prepared_texts,
    )

    return response["embeddings"]


def embed_query(
    question: str,
) -> list[float]:
    """Create one embedding for a user's question."""

    clean_question = question.strip()

    if not clean_question:
        raise ValueError(
            "The question cannot be empty."
        )

    response = ollama_client.embed(
        model=OLLAMA_EMBEDDING_MODEL,
        input=f"search_query: {clean_question}",
    )

    return response["embeddings"][0]