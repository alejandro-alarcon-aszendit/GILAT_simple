"""LangGraph Document Service (v1.0)
=================================
*Simplified* version: **metadata in SQLModel + async FastAPI**
+ **Synchronous ingestion** for immediate processing.

Highlights
----------
1. **PostgreSQL / SQLite** stores document metadata (id, name, status, n_chunks).
2. **Synchronous processing** parses, splits, embeds chunks and persists Chroma vectors.
3. Endpoints enforce status (`ready`) before summary / ask.
4. Chunks are persisted as `chunks.json` beside the Chroma directory to rebuild
   `langchain.Document` objects.

Install / run (local dev)
------------------------
```bash
pip install fastapi "uvicorn[standard]" langchain langchain-openai \
    langgraph docling chromadb tiktoken numpy sqlmodel

# run API
uvicorn agent:app --reload
```
Environment
-----------
* `OPENAI_API_KEY`
* `DATABASE_URL`  (default `sqlite:///./app.db`)
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

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langgraph.graph import Graph
from pydantic import BaseModel
from sqlmodel import Field, Session, SQLModel, create_engine, select

# Document processing
from docling.document_converter import DocumentConverter

# -------------------- Config ----------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")
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
    status: str = "ready"  # ready | failed
    n_chunks: int | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)


engine = create_engine(DATABASE_URL, echo=False)
SQLModel.metadata.create_all(engine)

# helper
def db_session() -> Session:
    return Session(engine)

# -------------------- LangGraph mini‑pipelines ----------------------

def build_ingest_graph():
    g = Graph()

    def _parse(file_path: str):
        """Parse document content based on file type"""
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension in ['.txt']:
                # Simple text reading for plain text files
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            elif file_extension in ['.pdf', '.md', '.docx', '.pptx', '.html']:
                # Use docling for supported document formats
                converter = DocumentConverter()
                result = converter.convert(file_path)
                
                # Check if conversion was successful
                if result is None or result.document is None:
                    raise ValueError("Document conversion returned None")
                
                # Try to extract text content
                try:
                    text = result.document.export_to_text()
                except:
                    # Fallback to markdown export
                    try:
                        text = result.document.export_to_markdown()
                    except:
                        raise ValueError("Failed to extract text from document")
                
                # Ensure we got some text
                if not text or not text.strip():
                    raise ValueError("No text content extracted from document")
                    
            else:
                # Try to read as text for other file types
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                except UnicodeDecodeError:
                    # If UTF-8 fails, try with error handling
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
            
            # Final check to ensure we have content
            if not text or not text.strip():
                raise ValueError("No text content found in document")
            
            return {"text": text}
        except Exception as e:
            raise ValueError(f"Failed to parse document '{file_path.name}': {str(e)}")

    def _split(state):
        docs = [Document(page_content=c) for c in SPLITTER.split_text(state["text"])]
        return {"docs": docs}

    g.add_node("parse", _parse)
    g.add_node("split", _split)
    g.set_entry_point("parse")
    g.add_edge("parse", "split")
    g.set_finish_point("split")
    return g.compile()

INGEST_GRAPH = build_ingest_graph()

# -------------------- Synchronous ingestion function ---------------
def ingest_document(doc_id: str, tmp_path: str, filename: str):
    """Synchronous ingestion: parse → split → embed → persist → update DB."""
    try:
        # 1. Process document using LangGraph
        result = INGEST_GRAPH.invoke(tmp_path)
        docs = result["docs"]

        # 2. embed & persist
        vs_dir = BASE_DIR / doc_id
        vs_dir.mkdir(exist_ok=True)
        vs = Chroma.from_documents(docs, EMBEDDER, persist_directory=str(vs_dir))

        # 3. save chunk texts
        with open(vs_dir / CHUNK_FILE, "w", encoding="utf-8") as f:
            json.dump([d.page_content for d in docs], f)

        # 4. update DB
        with db_session() as s:
            doc = s.get(Doc, doc_id)
            if doc:
                doc.status = "ready"
                doc.n_chunks = len(docs)
                s.add(doc)
                s.commit()
                
        return len(docs)
                
    except Exception as exc:
        # Handle errors
        with db_session() as s:
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
        query_context = state.get("query_context", "")
        
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
            
            # If we have query context, enhance the summary with it
            if query_context and summary_text and summary_text != "Unable to generate summary.":
                enhanced_prompt = f"""
                Based on the following summary, provide a refined version that emphasizes aspects related to: {query_context}
                
                Original summary:
                {summary_text}
                
                Please ensure the refined summary maintains accuracy while highlighting relevant information about the specified topic.
                """
                enhanced_result = LLM.invoke(enhanced_prompt)
                if hasattr(enhanced_result, 'content'):
                    summary_text = enhanced_result.content.strip()
                
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

# -------- Upload (synchronous) --------------------------------------
@app.post("/documents", status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    # 1. create tmp copy
    with NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    doc_id = str(uuid.uuid4())
    
    # 2. insert DB row
    with db_session() as s:
        s.add(Doc(id=doc_id, name=file.filename, status="ready"))
        s.commit()

    try:
        # 3. process document immediately
        n_chunks = ingest_document(doc_id, tmp_path, file.filename)
        return {"doc_id": doc_id, "status": "ready", "n_chunks": n_chunks}
    except Exception as e:
        # If processing fails, update status and raise error
        with db_session() as s:
            doc = s.get(Doc, doc_id)
            if doc:
                doc.status = "failed"
                s.add(doc)
                s.commit()
        raise HTTPException(500, f"Document processing failed: {str(e)}")

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
    query: str = Query(None, description="Optional query/topic to focus the summary on specific aspects"),
    top_k: int = Query(10, description="Number of most relevant chunks to include when using query-focused summarization"),
):
    with db_session() as s:
        docs_meta = s.exec(select(Doc).where(Doc.id.in_(doc_id))).all()
    not_ready = [d.id for d in docs_meta if d.status != "ready"]
    if not_ready:
        raise HTTPException(409, f"Documents not ready: {not_ready}")

    # If query is provided, use vector similarity search to get relevant chunks
    if query and query.strip():
        relevant_docs: List[Document] = []
        for d in doc_id:
            vs = load_vectorstore(d)
            retrieved = vs.similarity_search(query.strip(), k=top_k)
            relevant_docs.extend(retrieved)
        
        if not relevant_docs:
            return {"summary": f"No relevant content found for the query: '{query}'", "documents": doc_id, "query": query}
        
        all_docs = relevant_docs
        summary_context = f"focusing on aspects related to: {query}"
    else:
        # Original behavior: load all chunks
        all_docs: List[Document] = []
        for d in doc_id:
            all_docs.extend(load_chunks(d))
        summary_context = "covering all content"

    if not all_docs:
        return {"summary": "No content available for summarization.", "documents": doc_id, "query": query}

    # Prepare state for summary graph
    summary_input = {"docs": all_docs}
    if query and query.strip():
        summary_input["query_context"] = query.strip()
    
    summary_state = SUMMARY_GRAPH.invoke(summary_input)
    summary = summary_state.get("summary", "Unable to generate summary.") if summary_state else "Unable to generate summary."
    
    if not summary or summary == "Unable to generate summary." or summary == "No content to summarize.":
        return {"summary": "No content available for summarization.", "documents": doc_id, "query": query}
    
    summary = summary.strip()
    target = {"short": "≈3 sentences", "medium": "≈8 sentences", "long": "≈15 sentences"}[length]
    
    # Include query context in the refinement prompt if a query was provided
    if query and query.strip():
        refinement_prompt = f"Rewrite this summary so it fits {target} while preserving key info, {summary_context}:\n\n{summary}"
    else:
        refinement_prompt = f"Rewrite this summary so it fits {target} while preserving key info:\n\n{summary}"
    
    refined = LLM.invoke(refinement_prompt)
    
    final_summary = refined.content.strip() if hasattr(refined, 'content') else str(refined).strip()
    
    result = {
        "summary": final_summary, 
        "documents": doc_id,
        "chunks_processed": len(all_docs)
    }
    
    if query and query.strip():
        result["query"] = query
        result["search_method"] = "vector_similarity"
    else:
        result["search_method"] = "full_document"
        
    return result

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
