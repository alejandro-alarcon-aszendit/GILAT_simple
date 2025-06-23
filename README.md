# LangGraph Document Service – Documentation (v1.0)

---

## 1 · Overview

A micro‑service that ingests arbitrary documents (PDF, DOCX, TXT, …) and exposes:

- **Semantic Q&A** (`GET /ask`)
- **Multi‑document summarisation** (`GET /summary`)
- **Document catalogue & life‑cycle management**

Key traits ▶︎

- **FastAPI** REST API
- **LangChain + LangGraph** orchestration
- **OpenAI** GPT‑4o‑mini + text‑embedding‑ada‑002
- **Persistent vector store** – ChromaDB (per document directory)
- **Relational metadata** – SQLModel (SQLite / PostgreSQL)
- **Background ingestion** – Celery + Redis

---

## 2 · High‑level Architecture

```
┌──────────────┐   1 HTTP POST    ┌────────────────────┐
│   Client     │ ───────────────▶ │  FastAPI / Gunicorn │
└──────────────┘                  │  (web pod)         │
        ▲                         └────────┬───────────┘
        │2 status polls / queries            │enqueue task
        │                                    ▼
┌──────────────┐  4 store metadata   ┌────────────────────┐
│ PostgreSQL   │◄─────────────────── │    Celery worker   │
└──────────────┘                    │  parse → embed     │
                                    │  persist vectors   │
┌──────────────┐  3 vector persist   └────────────────────┘
│ ChromaDB dir │◄──────────────────────────────────────────┘
└──────────────┘
```

1. **Upload** → API saves a *processing* record and enqueues `ingest_task`.
2. Client polls `/documents/{id}` until `status: ready`.
3. Worker writes embeddings to `vector_db/<doc_id>/` (Chroma) + `chunks.json`.
4. Worker updates DB (`status: ready`, `n_chunks`).

---

## 3 · Installation (local dev)

```bash
# clone repo
pip install -r requirements.txt  # or copy the list below
export OPENAI_API_KEY=sk‑…
# optional: run postgres; fallback to SQLite works out of the box
redis-server &                    # needs Redis ≥ 6
uvicorn agent_app:app --reload    # web pod
celery -A agent_app.celery_app worker -l info  # worker pod
```

**Dependencies**

```
fastapi "uvicorn[standard]" langchain langchain-openai langgraph
sqlmodel sqlalchemy psycopg2-binary  # (or aiosqlite)
docling chromadb tiktoken numpy
celery redis
```

---

## 4 · Environment variables

| Var              | Default                    | Purpose                 |
| ---------------- | -------------------------- | ----------------------- |
| `OPENAI_API_KEY` | —                          | calls GPT & embeddings  |
| `DATABASE_URL`   | `sqlite:///./app.db`       | SQLModel engine         |
| `REDIS_URL`      | `redis://localhost:6379/0` | Celery broker / backend |

---

## 5 · Directory layout

```
.
├─ agent_app.py          # FastAPI + Celery codebase (see canvas)
├─ vector_db/            # one sub‑dir per document (Chroma index)
│   └─ <doc_id>/
│       ├─ index.sqlite  # Chroma DB
│       └─ chunks.json   # raw chunk texts
└─ app.db                # SQLite (if PostgreSQL not configured)
```

---

## 6 · API Reference

### 6.1 Upload Document

`POST /documents`

| Field                      | Type                | Notes                |
| -------------------------- | ------------------- | -------------------- |
| file (🗋)                  | multipart/form‑data | any supported format |
| **Returns** `202 Accepted` |                     |                      |

```json
{ "doc_id": "<uuid>", "status": "processing" }
```

### 6.2 List Documents

`GET /documents`

```json
[
  {"id":"…","name":"report.pdf","status":"ready","n_chunks":28,"created_at":"…"},
  …
]
```

### 6.3 Document Detail / Status

`GET /documents/{id}` → same schema as above.

### 6.4 Delete Document

`DELETE /documents/{id}` → `{ "status": "deleted", "doc_id": "…" }`

### 6.5 Multi‑Document Summary

`GET /summary?doc_id=id1&doc_id=id2&length=medium&query=<topic>&top_k=10`

| Query                                 | Description                                                       |
| ------------------------------------- | ----------------------------------------------------------------- |
| `doc_id`                              | repeatable – one or more UUIDs                                   |
| `length`                              | `short` ≈ 3 sent., `medium` ≈ 8, `long` ≈ 15                      |
| `query` (optional)                    | topic/query to focus summary on using vector similarity search   |
| `top_k` (optional)                    | max relevant chunks when using query-focused mode (default: 10)  |
| **409** if any document not `ready`. |                                                                   |

**Query-Focused Summarization:** When `query` parameter is provided, the system performs vector similarity search to find the most relevant content chunks across all specified documents, then summarizes only those relevant portions. This enables focused summaries on specific topics rather than general document summaries.

**Multi-Topic Parallel Processing:** Use comma-separated topics (e.g., `query=machine learning,financial performance,project timeline`) to generate separate focused summaries for each topic processed in parallel using LangGraph's map-reduce pattern. Each topic gets its own vector similarity search and summary generation.

### 6.6 Semantic Q&A

`GET /ask?q=<question>&doc_id=<id>&doc_id=<id>&top_k=3`

- `q` – free‑text question
- `top_k` – retrieved chunks per document (default 3)

---

## 7 · Database Schema

```sql
CREATE TABLE doc (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  status TEXT NOT NULL,       -- processing | ready | failed
  n_chunks INTEGER,
  created_at TIMESTAMP NOT NULL
);
```

---

## 8 · Background Ingestion Flow

1. **Celery task **``
   - parse → `docling` → Markdown
   - split → `RecursiveCharacterTextSplitter`
   - embed & persist → `Chroma.from_documents(..., persist_directory=…)`
   - save chunks → `chunks.json`
   - update `doc` row (`ready` + `n_chunks`)
2. Failure sets `status: failed` and keeps logs.

---

## 9 · How to Deploy

- **Docker / Compose** – single‑host dev:
  - `api` image (FastAPI + Gunicorn)
  - `worker` image (Celery)
  - `redis`
  - `postgres`
- **Kubernetes** – 2 deployments (`api`, `worker`) + 2 stateful services (`redis`, `postgres`).
- Mount `vector_db` on persistent volume claim (PVC) or switch to S3‑backed Chroma.

---

## 10 · Testing

```bash
pytest tests/            # unit + httpx integration tests
pytest -m e2e            # requires OPENAI_API_KEY set
```

Mock LLM via `langchain.chat_models.fake.FakeListChatModel` for CI.

---

## 11 · Extension Points

- Swap Chroma for Weaviate / Pinecone by replacing `langchain.vectorstores.Chroma` wrapper.
- Add topic segmentation endpoint (`/segments`) using embeddings + K‑Means.
- Enable SSE / WebSocket to stream token responses or ingest task progress.
- Implement OAuth / JWT to secure endpoints per user.

---

© 2025 LangGraph Doc Service • MIT

