# Local PDF RAG

A local Retrieval-Augmented Generation application that allows users to upload PDF files and ask questions about their content.

The project runs locally using **Ollama**, so no paid AI API key is required.

## Features

- Upload and process PDF files
- Split PDF text into chunks
- Generate embeddings locally
- Store and search vectors using Qdrant
- Generate answers using a local Ollama model
- Display the source PDF
- Monitor workflows using Inngest

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

## Run the Application

### FastAPI

```powershell
python -m uvicorn main:app --reload --port 8000
```

### Inngest

```powershell
npx --ignore-scripts=false inngest-cli@latest dev `
  --no-discovery `
  -u http://127.0.0.1:8000/api/inngest
```

### Streamlit

```powershell
python -m streamlit run streamlit_app.py
```

## Local URLs

| Service | URL |
|---|---|
| Streamlit | `http://localhost:8501` |
| FastAPI | `http://localhost:8000` |
| Inngest | `http://localhost:8288` |
| Qdrant | `http://localhost:6333/dashboard` |
| Ollama | `http://localhost:11434` |

## Usage

1. Open the Streamlit application.
2. Upload a text-based PDF.
3. Click **Ingest PDF**.
4. Wait for the ingestion workflow to complete.
5. Enter a question related to the PDF.
6. View the generated answer and its source.

## Limitations

- Scanned PDFs require OCR, which is not included.
- The project currently uses local file storage.
- Authentication and document management are not implemented.

