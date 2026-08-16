# Local PDF RAG

A local Retrieval-Augmented Generation application that allows users to upload PDF files and ask questions about their content.



## Overview

This is a Retrieval-Augmented Generation application that lets a user upload a PDF and ask questions about its content — but built to run entirely locally, with no paid API key required. Every part of the pipeline that would normally call an external service (embeddings, the LLM itself) runs through Ollama instead, which meant handling model loading, embedding generation, and inference orchestration myself rather than relying on a hosted API to abstract it away.

The core RAG pattern here — chunk a document, embed the chunks, store them in a vector database, retrieve the most relevant chunks for a given question, and pass that context to an LLM to generate a grounded answer — is the foundation of any AI search or document Q&A system. I also wired in Inngest to monitor the ingestion workflow as a background job rather than blocking the UI while a PDF is processed, which mattered once PDFs got long enough that embedding generation took real time.

## Features

- Upload and process PDF files
- Split PDF text into chunks
- Generate embeddings locally (no external API calls)
- Store and search vectors using Qdrant
- Generate answers using a local Ollama model
- Display the source PDF alongside the answer
- Monitor ingestion workflows using Inngest

## Technologies

- **FastAPI** — backend
- **Streamlit** — user interface
- **Ollama** — local LLM and embeddings
- **Qdrant** — vector database
- **Inngest** — background workflows
- **LlamaIndex** — PDF reading and chunking
- **Pydantic** — data validation

## Application Flow

```text
PDF Upload
   ↓
Extract and split text
   ↓
Generate embeddings with nomic-embed-text
   ↓
Store vectors in Qdrant

User Question
   ↓
Generate question embedding
   ↓
Search Qdrant
   ↓
Send retrieved context to Gemma
   ↓
Display answer and source
```

## Project Structure

```text
main.py             FastAPI and Inngest workflows
streamlit_app.py    Streamlit user interface
data_loader.py      PDF loading, chunking and embeddings
vector_db.py        Qdrant storage and search
custom_types.py     Pydantic models
```

## Environment Variables

Create a `.env` file:

```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=gemma2:2b
OLLAMA_EMBEDDING_MODEL=nomic-embed-text

QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=docs_nomic

INNGEST_DEV=1
INNGEST_API_BASE=http://127.0.0.1:8288/v1
```

## Install Ollama Models

```powershell
ollama pull gemma2:2b
ollama pull nomic-embed-text
```

## Start Qdrant

```powershell
docker run -d `
  --name rag-qdrant `
  -p 6333:6333 `
  -p 6334:6334 `
  -v rag_qdrant_storage:/qdrant/storage `
  qdrant/qdrant
```

# Run locally

## 1. Install Ollama models:

bash
ollama pull gemma2:2b
ollama pull nomic-embed-text

## 2. Start Qdrant:

bash
docker run -d --name rag-qdrant -p 6333:6333 -p 6334:6334 -v rag_qdrant_storage:/qdrant/storage qdrant/qdrant

## 3. Run each service:

bash
python -m uvicorn main:app --reload --port 8000        # FastAPI
npx inngest-cli@latest dev --no-discovery -u http://127.0.0.1:8000/api/inngest   # Inngest
python -m streamlit run streamlit_app.py                # Streamlit UI


## Service	URL

Streamlit	http://localhost:8501
FastAPI   	http://localhost:8000
Inngest  	http://localhost:8288
Qdrant	   http://localhost:6333/dashboard
Ollama	   http://localhost:11434


## Usage
1- Open the Streamlit app
2- Upload a text-based PDF
3- Click Ingest PDF and wait for the ingestion workflow to complete
4- Ask a question about the PDF
5- View the generated answer and its source

## Honest limitations
Scanned PDFs require OCR, which isn't implemented
Uses local file storage, not a persistent document store
No authentication or multi-user document management yet

## Possible next steps
Add OCR support for scanned PDFs
Add multi-document support with per-document filtering
Add authentication if extended to multi-user use

