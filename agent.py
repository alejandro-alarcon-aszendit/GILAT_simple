"""LangGraph Document Service (v1.0)
=================================
*Production‑oriented* refactor: **metadata in SQLModel + async FastAPI**
+ **Celery worker** for heavy ingestion.

Highlights
----------
1. **PostgreSQL / SQLite** stores document metadata (id, name, status, n_chunks).
2. **Celery** parses, splits, embeds chunks and persists Chroma vectors; main API
   returns immediately with *202 Processing*.
3. Endpoints enforce status (`ready`) before summary / ask.
4. Chunks are persisted as `chunks.json` beside the Chroma directory to rebuild
   `langchain.Document` objects.

Install / run (local dev)
------------------------
```bash
pip install fastapi "uvicorn[standard]" langchain langchain-openai \
    langgraph docling chromadb tiktoken numpy sqlmodel celery redis

# run API
uvicorn agent_app:app --reload
# run Celery worker (in another terminal)
celery -A agent_app.celery_app worker -l info
```
Environment
-----------
* `OPENAI_API_KEY`
* `DATABASE_URL`  (default `sqlite:///./app.db`)
* `REDIS_URL`     (default `redis://localhost:6379/0`)
"""

from __future__ import annotations

import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from celery import Celery
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langgraph.graph import Graph
from pydantic import BaseModel
from sqlmodel import Field, Session, SQLModel, create_engine, select

# -------------------- Config ----------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
BASE_DIR = Path("vector_db")
BASE_DIR.mkdir(exist_ok=True)
CHUNK_FILE = "chunks.json"

LLM = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)
EMBEDDER = OpenAIEmbeddings(model="text-embedding-ada-002")
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

# -------------------- DB models -------------------------------------
class Doc(SQLModel, table=True):
    id: str = Field(primary_key=True)
    name: str
    status: str = "processing"  # processing | ready | failed
    n_chunks: int | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)


engine = create_engine(DATABASE_URL, echo=False)
SQLModel.metadata.create_all(engine)

# helper
def db_session() -> Session:
    return Session(engine)

# -------------------- Celery ----------------------------------------
celery_app = Celery("agent_service", broker=REDIS_URL, backend=REDIS_URL)

# -------------------- LangGraph mini‑pipelines ----------------------

def build_ingest_graph():
    g = Graph()

    def _parse(file_path: str):
        try:
            import docling  # local import (may be heavy)
        except ImportError:
            raise RuntimeError("Install docling: `pip install docling`.")
        text = docling.Parsers.auto(file_path).parse().to_markdown()
        return {"text": text}

    def _split(state):
        docs = [Document(page_content=c) for c in SPLITTER.split_text(state["text"])]
        return {"docs": docs}

    g.add_node("parse", _parse)
    g.add_node("split", _split)
    g.set_entry_point("parse")
    g.add_edge("parse", "split")
    return g.compile()

INGEST_GRAPH = build_ingest_graph()

# -------------------- Celery task -----------------------------------
@celery_app.task(name="agent_service.ingest")
def ingest_task(doc_id: str, tmp_path: str, filename: str):
    """Background ingestion: parse → split → embed → persist → update DB."""
    try:
        # Simple text reading for markdown/text files
        with open(tmp_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 1. split text into chunks
        chunks = SPLITTER.split_text(text)
        docs = [Document(page_content=chunk) for chunk in chunks]

        # 2. embed & persist
        vs_dir = BASE_DIR / doc_id
        vs_dir.mkdir(exist_ok=True)
        vs = Chroma.from_documents(docs, EMBEDDER, persist_directory=str(vs_dir))

        # 3. save chunk texts
        with open(vs_dir / CHUNK_FILE, "w", encoding="utf-8") as f:
            json.dump([d.page_content for d in docs], f)

        # 4. update DB with proper session management
        engine_local = create_engine(DATABASE_URL, echo=False)
        with Session(engine_local) as s:
            doc = s.get(Doc, doc_id)
            if doc:
                doc.status = "ready"
                doc.n_chunks = len(docs)
                s.add(doc)
                s.commit()
                
    except Exception as exc:
        # Handle errors with proper session management
        engine_local = create_engine(DATABASE_URL, echo=False)
        with Session(engine_local) as s:
            doc = s.get(Doc, doc_id)
            if doc:
                doc.status = "failed"
                s.add(doc)
                s.commit()
        raise exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

# -------------------- Helpers ---------------------------------------

def load_vectorstore(doc_id: str) -> Chroma:
    vs_dir = BASE_DIR / doc_id
    if not vs_dir.exists():
        raise FileNotFoundError
    return Chroma(persist_directory=str(vs_dir), embedding_function=EMBEDDER)


def load_chunks(doc_id: str) -> List[Document]:
    file_path = BASE_DIR / doc_id / CHUNK_FILE
    if not file_path.exists():
        raise FileNotFoundError
    texts = json.loads(file_path.read_text(encoding="utf-8"))
    return [Document(page_content=t) for t in texts]


# -------------------- Summary graph ---------------------------------

def build_summary_graph():
    g = Graph()

    def _summarise(state):
        docs = state.get("docs", [])
        
        if not docs:
            return {"summary": "No content to summarize."}
        
        try:
            # For large document sets, use map-reduce approach
            chain = load_summarize_chain(LLM, chain_type="map_reduce")
            result = chain.invoke({"input_documents": docs})
            
            # Extract the summary text from the result
            if isinstance(result, dict) and "output_text" in result:
                summary_text = result["output_text"]
            elif isinstance(result, str):
                summary_text = result
            else:
                summary_text = str(result) if result else "Unable to generate summary."
                
            return {"summary": summary_text}
        except Exception as e:
            return {"summary": f"Error generating summary: {str(e)}"}

    g.add_node("summarise", _summarise)
    g.set_entry_point("summarise")
    g.set_finish_point("summarise")
    return g.compile()

SUMMARY_GRAPH = build_summary_graph()

# -------------------- FastAPI ---------------------------------------
app = FastAPI(title="LangGraph Doc Service", version="1.0")

# -------- Upload (async) -------------------------------------------
@app.post("/documents", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(file: UploadFile = File(...)):
    # 1. create tmp copy
    with NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    doc_id = str(uuid.uuid4())
    # 2. insert DB row status=processing
    with db_session() as s:
        s.add(Doc(id=doc_id, name=file.filename))
        s.commit()

    # 3. enqueue background job
    ingest_task.apply_async(args=[doc_id, tmp_path, file.filename])
    return {"doc_id": doc_id, "status": "processing"}

# -------- Document list / detail -----------------------------------
class DocOut(BaseModel):
    id: str
    name: str
    status: str
    n_chunks: int | None
    created_at: datetime

@app.get("/documents", response_model=list[DocOut])
async def list_documents():
    with db_session() as s:
        docs = s.exec(select(Doc).order_by(Doc.created_at.desc())).all()
        return docs

@app.get("/documents/{doc_id}", response_model=DocOut)
async def get_document(doc_id: str):
    with db_session() as s:
        doc = s.get(Doc, doc_id)
        if not doc:
            raise HTTPException(404, "Document not found")
        return doc

# -------- Delete ----------------------------------------------------
@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    with db_session() as s:
        doc = s.get(Doc, doc_id)
        if not doc:
            raise HTTPException(404, "Document not found")
        s.delete(doc)
        s.commit()
    # remove on‑disk data
    vs_dir = BASE_DIR / doc_id
    if vs_dir.exists():
        shutil.rmtree(vs_dir, ignore_errors=True)
    return {"status": "deleted", "doc_id": doc_id}

# -------- Summary (multiple docs) ----------------------------------
@app.get("/summary")
async def multi_summary(
    doc_id: List[str] = Query(...),
    length: str = Query("medium", enum=["short", "medium", "long"]),
):
    with db_session() as s:
        docs_meta = s.exec(select(Doc).where(Doc.id.in_(doc_id))).all()
    not_ready = [d.id for d in docs_meta if d.status != "ready"]
    if not_ready:
        raise HTTPException(409, f"Documents not ready: {not_ready}")

    all_docs: List[Document] = []
    for d in doc_id:
        all_docs.extend(load_chunks(d))

    summary_state = SUMMARY_GRAPH.invoke({"docs": all_docs})
    summary = summary_state.get("summary", "Unable to generate summary.") if summary_state else "Unable to generate summary."
    
    if not summary or summary == "Unable to generate summary." or summary == "No content to summarize.":
        return {"summary": "No content available for summarization.", "documents": doc_id}
    
    summary = summary.strip()
    target = {"short": "≈3 sentences", "medium": "≈8 sentences", "long": "≈15 sentences"}[length]
    refined = LLM.invoke(
        f"Rewrite this summary so it fits {target} while preserving key info:\n\n{summary}"
    )
    
    final_summary = refined.content.strip() if hasattr(refined, 'content') else str(refined).strip()
    return {"summary": final_summary, "documents": doc_id}

# -------- Ask -------------------------------------------------------
@app.get("/ask")
async def ask_docs(
    q: str = Query(...),
    doc_id: List[str] = Query(...),
    top_k: int = 3,
):
    with db_session() as s:
        docs_meta = s.exec(select(Doc).where(Doc.id.in_(doc_id))).all()
    not_ready = [d.id for d in docs_meta if d.status != "ready"]
    if not_ready:
        raise HTTPException(409, f"Documents not ready: {not_ready}")

    retrieved: List[Document] = []
    for d in doc_id:
        vs = load_vectorstore(d)
        retrieved.extend(vs.similarity_search(q, k=top_k))

    if not retrieved:
        return {"answer": "No relevant passages found.", "snippets": []}

    answer = LLM.invoke(
        "Answer the user's question based only on the excerpts below. "
        "If the answer is not contained, say so.\n\n" +
        "\n---\n".join(doc.page_content for doc in retrieved) + f"\n\nQuestion: {q}"
    )
    snippets = [{"content": d.page_content} for d in retrieved]
    return JSONResponse({"answer": answer.content.strip(), "snippets": snippets, "documents": doc_id})
