"""FastAPI endpoints for the Document Service.

Modular API endpoints with clear separation of concerns.
"""

import shutil
from typing import List
from fastapi import File, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse
from sqlmodel import select

from src.models.database import Doc, get_db_session
from src.models.schemas import DocOut, QAResponse
from src.services.document_service import DocumentService
from src.core.config import llm


class DocumentEndpoints:
    """Endpoints for document management operations."""
    
    @staticmethod
    async def upload_document(file: UploadFile = File(...)):
        """Upload and process a document.
        
        **Processing Pipeline:**
        1. Save uploaded file to temporary location
        2. Create document record in database
        3. Process document synchronously (parse → split → embed → persist)
        4. Update database with results
        
        Returns:
            Document metadata with processing results
        """
        # 1. Save uploaded file
        tmp_path = await DocumentService.save_uploaded_file(file)
        
        # 2. Create document record
        doc_id = await DocumentService.create_document_record(file.filename)
        
        try:
            # 3. Process document immediately (synchronous pipeline)
            n_chunks = DocumentService.ingest_document(doc_id, tmp_path, file.filename)
            return {"doc_id": doc_id, "status": "ready", "n_chunks": n_chunks}
        except Exception as e:
            # If processing fails, update status and raise error
            with get_db_session() as session:
                doc = session.get(Doc, doc_id)
                if doc:
                    doc.status = "failed"
                    session.add(doc)
                    session.commit()
            raise HTTPException(500, f"Document processing failed: {str(e)}")
    
    @staticmethod
    async def list_documents():
        """List all documents in the database."""
        with get_db_session() as session:
            docs = session.exec(select(Doc).order_by(Doc.created_at.desc())).all()
            return docs
    
    @staticmethod
    async def get_document(doc_id: str):
        """Get a specific document by ID."""
        with get_db_session() as session:
            doc = session.get(Doc, doc_id)
            if not doc:
                raise HTTPException(404, "Document not found")
            return doc
    
    @staticmethod
    async def delete_document(doc_id: str):
        """Delete a document and its associated vector data."""
        with get_db_session() as session:
            doc = session.get(Doc, doc_id)
            if not doc:
                raise HTTPException(404, "Document not found")
            session.delete(doc)
            session.commit()
        
        # Remove vector store directory
        from src.core.config import BASE_DIR
        vs_dir = BASE_DIR / doc_id
        if vs_dir.exists():
            shutil.rmtree(vs_dir, ignore_errors=True)
        
        return {"status": "deleted", "doc_id": doc_id}


class SummaryEndpoints:
    """Endpoints for document summarization operations."""
    
    @staticmethod
    async def multi_summary(
        doc_id: List[str] = Query(...),
        length: str = Query("medium", enum=["short", "medium", "long"]),
        query: str = Query(None, description="Optional query/topic(s) to focus the summary on. Use commas to separate multiple topics for parallel processing."),
        top_k: int = Query(10, description="Number of most relevant chunks to include when using query-focused summarization"),
        enable_reflection: bool = Query(False, description="Enable AI reflection to review and improve summary quality, accuracy, and length compliance"),
    ):
        """Generate summaries for multiple documents.
        
        **Processing Modes:**
        1. **Full Document**: Summarizes all content if no query provided
        2. **Single Topic**: Uses vector similarity search for focused summary
        3. **Multi-Topic Parallel**: Processes multiple comma-separated topics in parallel with optional reflection
        
        **Parallel Workloads:**
        - Multi-topic processing uses ThreadPoolExecutor for true parallelism
        - Each topic processed independently with performance monitoring
        - Optional reflection system for quality improvement
        """
        # Validate document readiness
        with get_db_session() as session:
            docs_meta = session.exec(select(Doc).where(Doc.id.in_(doc_id))).all()
        not_ready = [d.id for d in docs_meta if d.status != "ready"]
        if not_ready:
            raise HTTPException(409, f"Documents not ready: {not_ready}")

        # Parse query for topics (single or multiple)
        if query and query.strip():
            topics = [topic.strip() for topic in query.split(",") if topic.strip()]
            
            # Always use parallel processing for topic-based queries (single or multiple)
            # This ensures consistent processing and parallel workflow visibility
            return await SummaryEndpoints._process_multi_topic(
                topics=topics,
                doc_ids=doc_id,
                top_k=top_k,
                length=length,
                enable_reflection=enable_reflection
            )
        
        # Full document processing
        else:
            return await SummaryEndpoints._process_full_documents(
                doc_ids=doc_id,
                length=length
            )
    
    @staticmethod
    async def _process_multi_topic(topics, doc_ids, top_k, length, enable_reflection):
        """Process multiple topics in parallel."""
        from src.graphs.summary import MULTI_TOPIC_SUMMARY_GRAPH
        from src.graphs.reflection import MULTI_TOPIC_SUMMARY_WITH_REFLECTION_GRAPH
        
        multi_topic_input = {
            "topics": topics,
            "doc_ids": doc_ids,
            "top_k": top_k,
            "length": length,
            "enable_reflection": enable_reflection
        }
        
        # Choose graph based on reflection setting
        if enable_reflection:
            result_state = MULTI_TOPIC_SUMMARY_WITH_REFLECTION_GRAPH.invoke(multi_topic_input)
        else:
            result_state = MULTI_TOPIC_SUMMARY_GRAPH.invoke(multi_topic_input)
        
        summaries = result_state.get("summaries", [])
        parallel_metadata = result_state.get("parallel_processing", {})
        
        if not summaries:
            return {
                "type": "multi_topic",
                "summaries": [],
                "message": "No summaries could be generated for the provided topics.",
                "documents": doc_ids,
                "topics": topics,
                "parallel_processing": parallel_metadata
            }
        
        # Calculate metrics
        total_chunks = sum(s.get("chunks_processed", 0) for s in summaries)
        successful_summaries = [s for s in summaries if s.get("status") == "success"]
        
        # Performance calculations
        individual_times = [s.get("processing_time", 0) for s in summaries if s.get("processing_time")]
        max_individual_time = max(individual_times) if individual_times else 0
        total_sequential_time = sum(individual_times) if individual_times else 0
        parallel_speedup = total_sequential_time / parallel_metadata.get("total_time", 1) if parallel_metadata.get("total_time") else 1
        
        # Reflection statistics
        reflection_applied_count = sum(1 for s in summaries if s.get("reflection_applied", False))
        
        # Determine response type based on number of topics
        response_type = "single_topic" if len(topics) == 1 else "multi_topic"
        search_method = "vector_similarity" if len(topics) == 1 else "vector_similarity_multi_topic"
        
        response = {
            "type": response_type,
            "documents": doc_ids,
            "topics": topics,
            "total_chunks_processed": total_chunks,
            "successful_topics": len(successful_summaries),
            "total_topics": len(topics),
            "search_method": search_method,
            "parallel_processing": parallel_metadata,
            "performance": {
                "parallel_time": parallel_metadata.get("total_time", 0),
                "estimated_sequential_time": total_sequential_time,
                "speedup_factor": round(parallel_speedup, 2),
                "longest_individual_task": max_individual_time,
                "efficiency": round((total_sequential_time / (parallel_metadata.get("total_time", 1) * len(topics))) * 100, 1) if parallel_metadata.get("total_time") and topics else 0,
                "parallel_method": parallel_metadata.get("method", "ThreadPoolExecutor"),
                "max_workers": min(len(topics), 5) if topics else 0
            },
            "reflection_enabled": enable_reflection
        }
        
        # Add summaries - for single topic, also include a flattened summary field
        if len(topics) == 1 and successful_summaries:
            response["summary"] = successful_summaries[0].get("summary", "")
            response["summaries"] = summaries
            response["query"] = topics[0]
            response["chunks_processed"] = successful_summaries[0].get("chunks_processed", 0)
        else:
            response["summaries"] = summaries
        
        # Add reflection statistics
        if enable_reflection:
            response["reflection_statistics"] = {
                "total_topics": len(summaries),
                "reflection_applied": reflection_applied_count,
                "reflection_skipped": len(summaries) - reflection_applied_count
            }
        
        return response
    

    
    @staticmethod
    async def _process_full_documents(doc_ids, length):
        """Process full documents without query filtering."""
        from langchain.docstore.document import Document
        
        all_docs: List[Document] = []
        for d in doc_ids:
            all_docs.extend(DocumentService.load_document_chunks(d))
        
        if not all_docs:
            return {
                "type": "single",
                "summary": "No content available for summarization.", 
                "documents": doc_ids, 
                "query": None
            }
        
        return await SummaryEndpoints._generate_summary(
            docs=all_docs,
            doc_ids=doc_ids,
            query=None,
            length=length,
            search_method="full_document"
        )
    
    @staticmethod
    async def _generate_summary(docs, doc_ids, query, length, search_method):
        """Generate summary using the summary graph."""
        from src.graphs.summary import SUMMARY_GRAPH
        
        # Prepare state for summary graph
        summary_input = {"docs": docs}
        if query:
            summary_input["query_context"] = query
        
        summary_state = SUMMARY_GRAPH.invoke(summary_input)
        summary = summary_state.get("summary", "Unable to generate summary.") if summary_state else "Unable to generate summary."
        
        if not summary or summary == "Unable to generate summary." or summary == "No content to summarize.":
            return {
                "type": "single",
                "summary": "No content available for summarization.", 
                "documents": doc_ids, 
                "query": query
            }
        
        # Refine summary to meet length requirement
        summary = summary.strip()
        target = {"short": "≈3 sentences", "medium": "≈8 sentences", "long": "≈15 sentences"}[length]
        
        if query:
            refinement_prompt = f"Rewrite this summary so it fits {target} while preserving key info, focusing on aspects related to: {query}:\n\n{summary}"
        else:
            refinement_prompt = f"Rewrite this summary so it fits {target} while preserving key info:\n\n{summary}"
        
        refined = llm.invoke(refinement_prompt)
        final_summary = refined.content.strip() if hasattr(refined, 'content') else str(refined).strip()
        
        result = {
            "type": "single",
            "summary": final_summary, 
            "documents": doc_ids,
            "chunks_processed": len(docs),
            "search_method": search_method
        }
        
        if query:
            result["query"] = query
            
        return result


class QAEndpoints:
    """Endpoints for question answering operations."""
    
    @staticmethod
    async def ask_docs(
        q: str = Query(...),
        doc_id: List[str] = Query(...),
        top_k: int = 3,
    ):
        """Ask questions about documents using vector similarity search.
        
        **Processing Pipeline:**
        1. Validate document readiness
        2. Retrieve relevant chunks using vector similarity
        3. Generate answer using LLM with retrieved context
        """
        # Validate document readiness
        with get_db_session() as session:
            docs_meta = session.exec(select(Doc).where(Doc.id.in_(doc_id))).all()
        not_ready = [d.id for d in docs_meta if d.status != "ready"]
        if not_ready:
            raise HTTPException(409, f"Documents not ready: {not_ready}")

        # Retrieve relevant passages
        from langchain.docstore.document import Document
        retrieved: List[Document] = []
        for d in doc_id:
            vs = DocumentService.load_vector_store(d)
            retrieved.extend(vs.similarity_search(q, k=top_k))

        if not retrieved:
            return {"answer": "No relevant passages found.", "snippets": []}

        # Generate answer
        answer = llm.invoke(
            "Answer the user's question based only on the excerpts below. "
            "If the answer is not contained, say so.\n\n" +
            "\n---\n".join(doc.page_content for doc in retrieved) + f"\n\nQuestion: {q}"
        )
        
        snippets = [{"content": d.page_content} for d in retrieved]
        return JSONResponse({
            "answer": answer.content.strip(), 
            "snippets": snippets, 
            "documents": doc_id
        }) 