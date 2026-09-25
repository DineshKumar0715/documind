# 🧠 DocuMind: Enterprise Document Intelligence & RAG Platform

**DocuMind** is a production-grade, AI-powered Document Intelligence and Retrieval-Augmented Generation (RAG) platform powered by **Google Gemini** (`gemini-1.5-flash`, `gemini-1.5-pro`, `gemini-2.0`) and **ChromaDB**.

---

## 🌟 Key Capabilities

- 📄 **Multi-Format Ingestion**: Ingest and process PDF documents (with page tracking), Markdown (`.md`), Plain Text (`.txt`), JSON (`.json`), and CSV files.
- ✂️ **Semantic Text Chunking**: Recursive character chunking with configurable overlap, sentence preservation, and rich metadata extraction.
- 🔍 **Vector Retrieval**: Vector similarity search and score normalization powered by ChromaDB.
- 🤖 **Google Gemini AI Integration**:
  - Embedding Generation via `models/text-embedding-004`.
  - Context Synthesis via `gemini-1.5-flash` / `gemini-1.5-pro` with grounded factual citations (`[Source: filename, Page: X]`).
  - Offline deterministic fallback mode for testing without requiring an immediate API key.
- 💬 **Multi-Turn Conversational Memory**: Threaded chat sessions tracked in an embedded SQLite database.
- 🌊 **Real-Time Streaming**: Server-Sent Events (SSE) streaming endpoint (`/api/v1/chat/stream`) for ultra-low time-to-first-token.
- 🖥️ **Interactive Built-in Dashboard**: Clean, modern web UI served directly at `http://localhost:8000/` for uploading documents and testing RAG conversations.
- 🧪 **Comprehensive Evaluation Benchmark**: Automated evaluation harness measuring Retrieval Hit Rate, MRR, and Latency.
- 🐳 **Production Docker Ready**: Multi-stage `Dockerfile` and `docker-compose.yml`.

---

## 🏗️ Architecture Overview

```
                          ┌────────────────────────┐
                          │   Client / Frontend    │
                          │   (Browser / cURL)     │
                          └───────────┬────────────┘
                                      │ REST / SSE
                                      ▼
                          ┌────────────────────────┐
                          │     FastAPI App        │
                          │   (app.main:app)       │
                          └─────┬────────────┬─────┘
                                │            │
            ┌───────────────────┘            └────────────────────┐
            ▼                                                     ▼
┌─────────────────────────┐                             ┌─────────────────────────┐
│  Document Ingestion     │                             │  Conversational RAG     │
│  (app/services/ingest)  │                             │  (app/services/rag)     │
└───────────┬─────────────┘                             └───────────┬─────────────┘
            │                                                       │
            ▼                                                       ▼
┌─────────────────────────┐                             ┌─────────────────────────┐
│ Gemini Embedding        │                             │  Chroma Vector Store    │
│ (text-embedding-004)    │                             │  (data/chroma)          │
└───────────┬─────────────┘                             └───────────┬─────────────┘
            │                                                       │
            ▼                                                       ▼
┌─────────────────────────┐                             ┌─────────────────────────┐
│ SQLite Metadata & Chat  │◄────────────────────────────┤  Gemini 1.5/2.0 LLM     │
│ (data/documind.db)      │                             │  (Synthesis + Citation) │
└─────────────────────────┘                             └─────────────────────────┘
```

---

## 🚀 Quickstart & Run Commands

### 1. Environment Configuration

Copy the example environment file and add your Google Gemini API key:

```bash
cp .env.example .env
```

Edit `.env`:
```ini
GEMINI_API_KEY=your_actual_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash
EMBEDDING_MODEL=models/text-embedding-004
```
> *Get your free Gemini API Key at [Google AI Studio](https://aistudio.google.com/).*

---

### 2. Start Local Development Server

**Windows PowerShell**:
```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

**Windows CMD**:
```cmd
.\.venv\Scripts\activate.bat
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- 🌐 **Interactive Dashboard**: `http://127.0.0.1:8000/`
- 📚 **Swagger API Docs**: `http://127.0.0.1:8000/docs`
- 🩺 **Health Check**: `http://127.0.0.1:8000/health`

---

### 3. CLI Batch Ingestion

Ingest a directory of documents directly from the terminal:

```bash
.\.venv\Scripts\python.exe scripts/ingest.py --path ./data/documents
```

---

### 4. Run Automated Test Suite

```bash
.\.venv\Scripts\pytest.exe -v
```

---

### 5. Run RAG Evaluation Benchmark

```bash
.\.venv\Scripts\python.exe evaluation/evaluate.py
```

---

### 6. Run with Docker Compose

```bash
docker-compose up --build
```

---

## 📡 API Reference Summary

### Documents API (`/api/v1/documents`)
- `POST /api/v1/documents/upload` - Upload and index document (PDF, TXT, MD, JSON, CSV).
- `GET /api/v1/documents` - List all indexed documents with chunk counts and status.
- `GET /api/v1/documents/{id}` - Get document details and all chunk excerpts.
- `DELETE /api/v1/documents/{id}` - Delete document, its embeddings from Chroma, and metadata.

### Chat & RAG API (`/api/v1/chat`)
- `POST /api/v1/chat` - Perform grounded RAG query with source citations.
- `POST /api/v1/chat/stream` - Stream RAG response tokens in real-time (SSE).
- `POST /api/v1/chat/sessions` - Create a new chat session.
- `GET /api/v1/chat/sessions` - List chat sessions.
- `GET /api/v1/chat/sessions/{id}` - Retrieve chat history with citations.
- `DELETE /api/v1/chat/sessions/{id}` - Delete chat session.
