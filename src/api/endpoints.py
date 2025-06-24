"""FastAPI endpoints for the Document Service.

Modular API endpoints with clear separation of concerns.
Supports both file uploads and URL content fetching.
"""

import shutil
from typing import List
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi import File, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse
from sqlmodel import select
from pydantic import BaseModel, HttpUrl

from src.models.database import Doc, get_db_session
from src.models.schemas import DocOut, QAResponse
from src.services.document_service import DocumentService
from src.services.web_content_service import WebContentService
from src.core.config import llm


class URLIngestRequest(BaseModel):
    """Request model for URL ingestion."""
    url: HttpUrl
    name: str = None  # Optional custom name for the document


class DocumentEndpoints:
    """Endpoints for document management operations."""
    
    @staticmethod
    async def get_supported_formats():
        """Get all supported document formats.
        
        Returns:
            Dictionary with supported formats organized by category
        """
        return DocumentService.get_supported_formats()
    
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
    async def ingest_url(request: URLIngestRequest):
        """Fetch content from a URL and process it as a document.
        
        **URL Processing Pipeline:**
        1. Validate URL format
        2. Create document record in database
        3. Fetch content from URL using WebContentService
        4. Process content (split → embed → persist)
        5. Update database with results
        
        Args:
            request: URLIngestRequest containing URL and optional name
        
        Returns:
            Document metadata with processing results
        """
        url = str(request.url)
        document_name = request.name or url
        
        # 1. Validate URL
        if not WebContentService.is_valid_url(url):
            raise HTTPException(400, f"Invalid URL format: {url}")
        
        # 2. Create document record
        doc_id = await DocumentService.create_document_record(document_name)
        
        try:
            # 3. Process URL content (fetch → split → embed → persist)
            n_chunks = DocumentService.ingest_url(doc_id, url)
            return {
                "doc_id": doc_id, 
                "status": "ready", 
                "n_chunks": n_chunks,
                "source_type": "url",
                "source": url
            }
        except Exception as e:
            # If processing fails, update status and raise error
            with get_db_session() as session:
                doc = session.get(Doc, doc_id)
                if doc:
                    doc.status = "failed"
                    session.add(doc)
                    session.commit()
            raise HTTPException(500, f"URL processing failed: {str(e)}")
    
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
    
    @staticmethod
    async def get_document_chunks(doc_id: str):
        """Get document chunks as JSON format.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Document chunks with metadata
        """
        try:
            chunks = DocumentService.load_document_chunks(doc_id)
            return {
                "doc_id": doc_id,
                "chunks": [{"content": chunk.page_content} for chunk in chunks],
                "total_chunks": len(chunks)
            }
        except Exception as e:
            raise HTTPException(404, f"Document chunks not found: {str(e)}")


class SummaryEndpoints:
    """Endpoints for document summarization operations."""
    
    @staticmethod
    async def multi_summary(
        doc_id: List[str] = Query(None, description="Document IDs to summarize. If not provided with a query, all ready documents will be used for topic-focused search."),
        length: int = Query(8, ge=1, le=30, description="Number of sentences for the summary (1-30)"),
        strategy: str = Query("abstractive", enum=["abstractive", "extractive", "hybrid"], description="Summarization strategy: abstractive (generates new sentences), extractive (selects key sentences), hybrid (combines both approaches)"),
        query: str = Query(None, description="Optional query/topic(s) to focus the summary on. Use commas to separate multiple topics for parallel processing. When provided without doc_id, searches across all documents."),
        top_k: int = Query(10, description="Number of most relevant chunks to include when using query-focused summarization"),
        enable_reflection: bool = Query(False, description="Enable AI reflection to review and improve summary quality, accuracy, and length compliance"),
    ):
        """Generate summaries for multiple documents.
        
        **Processing Modes:**
        1. **Full Document**: Summarizes all content if no query provided (requires doc_id)
        2. **Single Topic**: Uses vector similarity search for focused summary
        3. **Multi-Topic Parallel**: Processes multiple comma-separated topics in parallel with optional reflection
        4. **All-Document Topic Search**: When query provided without doc_id, searches across all ready documents
        
        **Parallel Workloads:**
        - Multi-topic processing uses ThreadPoolExecutor for true parallelism
        - Each topic processed independently with performance monitoring
        - Optional reflection system for quality improvement
        """
        
        # Handle different input scenarios
        if query and query.strip():
            # Topic-focused processing (with or without specific documents)
            topics = [topic.strip() for topic in query.split(",") if topic.strip()]
            
            # If no specific documents provided, use all ready documents
            if not doc_id:
                with get_db_session() as session:
                    all_docs = session.exec(select(Doc).where(Doc.status == "ready")).all()
                    if not all_docs:
                        raise HTTPException(404, "No ready documents found for topic search")
                    doc_id = [doc.id for doc in all_docs]
                    print(f"🌍 Using all {len(doc_id)} ready documents for topic search: {topics}")
            else:
                # Validate selected documents
                with get_db_session() as session:
                    docs_meta = session.exec(select(Doc).where(Doc.id.in_(doc_id))).all()
                not_ready = [d.id for d in docs_meta if d.status != "ready"]
                if not_ready:
                    raise HTTPException(409, f"Documents not ready: {not_ready}")
            
            # Always use parallel processing for topic-based queries (single or multiple)
            return await SummaryEndpoints._process_multi_topic(
                topics=topics,
                doc_ids=doc_id,
                top_k=top_k,
                length=length,
                strategy=strategy,
                enable_reflection=enable_reflection
            )
        
        # Full document processing (requires specific documents)
        else:
            if not doc_id:
                raise HTTPException(400, "doc_id is required when no query is provided for full document summarization")
            
            # Validate document readiness
            with get_db_session() as session:
                docs_meta = session.exec(select(Doc).where(Doc.id.in_(doc_id))).all()
            not_ready = [d.id for d in docs_meta if d.status != "ready"]
            if not_ready:
                raise HTTPException(409, f"Documents not ready: {not_ready}")
            
            return await SummaryEndpoints._process_full_documents(
                doc_ids=doc_id,
                length=length,
                strategy=strategy
            )
    
    @staticmethod
    async def _process_multi_topic(topics, doc_ids, top_k, length, strategy, enable_reflection):
        """Process multiple topics in parallel using the unified LangGraph Send API."""
        from src.graphs.unified_summary_reflection import UNIFIED_SUMMARY_REFLECTION_GRAPH
        
        multi_topic_input = {
            "topics": topics,
            "doc_ids": doc_ids,
            "top_k": top_k,
            "length": length,
            "strategy": strategy,
            "enable_reflection": enable_reflection
        }
        
        # Use the unified graph that handles both summary and reflection with Send API
        result_state = UNIFIED_SUMMARY_REFLECTION_GRAPH.invoke(multi_topic_input)
        
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
            "strategy": strategy,
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
    async def _process_full_documents(doc_ids, length, strategy):
        """Process full documents using the existing parallel topic processing system."""
        from src.graphs.unified_summary_reflection import UNIFIED_SUMMARY_REFLECTION_GRAPH
        
        if len(doc_ids) == 1:
            # Single document - use existing logic for backwards compatibility
            from langchain.docstore.document import Document
            
            all_docs: List[Document] = []
            all_docs.extend(DocumentService.load_document_chunks(doc_ids[0]))
            
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
                search_method="full_document",
                strategy=strategy
            )
        
        # Multiple documents - process each document separately in parallel
        print(f"🚀 Processing {len(doc_ids)} documents separately in parallel")
        
        # Process each document individually to avoid mixing chunks
        summaries = []
        
        # Import here to avoid circular imports
        from langchain.docstore.document import Document
        
        for i, doc_id in enumerate(doc_ids):
            print(f"📄 Processing document {i+1}/{len(doc_ids)}: {doc_id}")
            
            # Load chunks for this specific document only
            doc_chunks = DocumentService.load_document_chunks(doc_id)
            
            if not doc_chunks:
                print(f"⚠️ No chunks found for document {doc_id}")
                continue
            
            # Generate summary for this document
            try:
                doc_summary = await SummaryEndpoints._generate_summary(
                    docs=doc_chunks,
                    doc_ids=[doc_id],  # Single document
                    query=None,
                    length=length,
                    search_method=f"full_document_{i+1}",
                    strategy=strategy
                )
                
                # Extract just the summary text
                summary_text = doc_summary.get("summary", "")
                if summary_text and summary_text != "No content available for summarization.":
                    summaries.append({
                        "document_id": doc_id,
                        "document_index": i + 1,
                        "summary": summary_text,
                        "chunks_processed": len(doc_chunks)
                    })
                    print(f"✅ Generated summary for document {i+1} ({len(doc_chunks)} chunks)")
                else:
                    print(f"⚠️ No summary generated for document {i+1}")
                    
            except Exception as e:
                print(f"❌ Error processing document {doc_id}: {str(e)}")
                continue
        
        # Return results based on number of successful summaries
        if not summaries:
            return {
                "type": "single",
                "summary": "No content available for summarization.",
                "documents": doc_ids,
                "query": None
            }
        
        # Now aggregate if multiple summaries exist
        if len(summaries) == 1:
            # Single successful summary
            return {
                "type": "single",
                "summary": summaries[0]["summary"],
                "documents": doc_ids,
                "chunks_processed": summaries[0]["chunks_processed"],
                "search_method": "single_document_processing",
                "strategy": strategy,
                "query": None
            }
        
        # Multiple summaries - aggregate them
        print(f"🔄 Aggregating {len(summaries)} document summaries...")
        
        from src.core.config import LLMConfig
        llm = LLMConfig.MAIN_LLM
        
        # Create individual summary texts
        individual_summaries = []
        total_chunks = 0
        
        for summary in summaries:
            summary_text = summary["summary"]
            if summary_text:
                individual_summaries.append(f"Document {summary['document_index']}: {summary_text}")
                total_chunks += summary.get("chunks_processed", 0)
        
        if not individual_summaries:
            summary_text = "No content available for summarization."
        elif len(individual_summaries) == 1:
            summary_text = individual_summaries[0].replace("Document 1: ", "")
        else:
            combined = "\n\n".join(individual_summaries)
            
            aggregation_prompt = f"""Based on the following individual document summaries, create a comprehensive summary in exactly {length} sentences that captures the key information from all documents:

{combined}

Comprehensive Summary:"""
            
            try:
                response = llm.invoke(aggregation_prompt)
                summary_text = response.content.strip() if hasattr(response, 'content') else str(response).strip()
                
                if not summary_text:
                    summary_text = " ".join(s.get("summary", "") for s in summaries if s.get("summary"))
                    
                print(f"✅ Successfully aggregated {len(summaries)} document summaries")
            except Exception as e:
                print(f"❌ Error aggregating summaries: {str(e)}")
                summary_text = " ".join(s.get("summary", "") for s in summaries if s.get("summary"))
        
        return {
            "type": "single",
            "summary": summary_text,
            "documents": doc_ids,
            "chunks_processed": total_chunks,
            "search_method": "parallel_document_processing",
            "strategy": strategy,
            "query": None,
            "individual_summaries": len(summaries)
        }
    
    @staticmethod
    async def _process_full_documents_fallback(doc_ids, length, strategy):
        """Fallback to sequential processing if parallel processing fails."""
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
            search_method="full_document_fallback",
            strategy=strategy
        )
    
    @staticmethod
    async def _generate_summary(docs, doc_ids, query, length, search_method, strategy):
        """Generate summary using strategy functions directly."""
        # Use the strategy functions from the utility module
        from src.utils.summarization_strategies import get_strategy_function
        
        # Generate summary using strategy
        strategy_function = get_strategy_function(strategy)
        summary_result = strategy_function(docs, query or "")
        
        summary = summary_result.get("summary", "Unable to generate summary.")
        
        if not summary or summary == "Unable to generate summary." or summary == "No content to summarize.":
            return {
                "type": "single",
                "summary": "No content available for summarization.", 
                "documents": doc_ids,
                "strategy": strategy,
                "query": query
            }
        
        # For non-extractive strategies, refine summary to meet length requirement
        if strategy != "extractive":
            summary = summary.strip()
            target = f"exactly {length} sentences"
            
            if query:
                refinement_prompt = f"Rewrite this summary so it fits {target} while preserving key info, focusing on aspects related to: {query}:\n\n{summary}"
            else:
                refinement_prompt = f"Rewrite this summary so it fits {target} while preserving key info:\n\n{summary}"
            
            refined = llm.invoke(refinement_prompt)
            final_summary = refined.content.strip() if hasattr(refined, 'content') else str(refined).strip()
        else:
            final_summary = summary
        
        result = {
            "type": "single",
            "summary": final_summary, 
            "documents": doc_ids,
            "chunks_processed": len(docs),
            "strategy": strategy,
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
        doc_id: List[str] = Query(None, description="Document IDs to search. If not provided, searches across all ready documents."),
        top_k: int = 3,
    ):
        """Ask questions about documents using vector similarity search.
        
        **Processing Pipeline:**
        1. Determine document scope (specific docs or all ready docs)
        2. Validate document readiness
        3. Retrieve relevant chunks using cross-document similarity ranking
        4. Generate answer using LLM with retrieved context
        """
        # If no specific documents provided, use all ready documents
        if not doc_id:
            with get_db_session() as session:
                all_docs = session.exec(select(Doc).where(Doc.status == "ready")).all()
                if not all_docs:
                    raise HTTPException(404, "No ready documents found for question answering")
                doc_id = [doc.id for doc in all_docs]
                print(f"🌍 Searching across all {len(doc_id)} ready documents for question: '{q}'")
        else:
            # Validate selected documents
            with get_db_session() as session:
                docs_meta = session.exec(select(Doc).where(Doc.id.in_(doc_id))).all()
            not_ready = [d.id for d in docs_meta if d.status != "ready"]
            if not_ready:
                raise HTTPException(409, f"Documents not ready: {not_ready}")

        # Retrieve relevant passages with concurrent cross-document similarity ranking
        from langchain.docstore.document import Document
        all_candidates = []
        retrieval_stats = {}
        
        print(f"🔍 Searching across {len(doc_id)} documents with concurrent cross-document ranking")
        
        # Get more candidates per document for better global ranking
        candidates_per_doc = min(top_k * 2, 10)  # Get 2x requested or max 10 per doc
        
        # Execute queries concurrently using ThreadPoolExecutor
        from src.core.config import ParallelConfig
        with ThreadPoolExecutor(max_workers=min(len(doc_id), ParallelConfig.MAX_DB_QUERY_WORKERS)) as executor:
            # Submit all document queries concurrently
            future_to_doc_id = {
                executor.submit(_query_single_document_qa, d, q, candidates_per_doc): d 
                for d in doc_id
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_doc_id):
                doc_id_key = future_to_doc_id[future]
                try:
                    candidates, returned_doc_id, retrieval_count = future.result()
                    all_candidates.extend(candidates)
                    retrieval_stats[returned_doc_id] = retrieval_count
                except Exception as e:
                    print(f"Error processing results for doc {doc_id_key}: {e}")
                    retrieval_stats[doc_id_key] = 0
        
        # Sort by similarity score and take top-k globally
        if all_candidates:
            all_candidates.sort(key=lambda x: x[1])  # Sort by score (ascending = most similar first)
            top_candidates = all_candidates[:top_k]
            retrieved = [doc for doc, score, doc_id in top_candidates]
            
            print(f"✅ Selected {len(retrieved)} globally top-ranked chunks (concurrent retrieval)")
            print(f"📊 Score range: {top_candidates[0][1]:.4f} to {top_candidates[-1][1]:.4f}")
            print(f"🚀 Queried {len(doc_id)} documents concurrently")
        else:
            retrieved = []

        if not retrieved:
            return {
                "answer": "No relevant passages found for your question. Try rephrasing your query or check if the documents contain information related to your topic.",
                "snippets": [],
                "retrieval_stats": retrieval_stats,
                "total_chunks_retrieved": 0
            }

        # Generate answer with improved prompt
        context_text = "\n---\n".join(doc.page_content for doc in retrieved)
        
        prompt = f"""You are a helpful assistant answering questions based on the provided document excerpts. 

INSTRUCTIONS:
1. Answer the user's question using ONLY the information provided in the excerpts below
2. If the exact information isn't available, explain what related information IS available
3. Be specific about what the documents contain vs. what they don't contain  
4. If you can provide partial information or context, do so while being clear about limitations
5. If no relevant information is found, say so clearly but helpfully

DOCUMENT EXCERPTS:
{context_text}

QUESTION: {q}

ANSWER:"""

        try:
            answer = llm.invoke(prompt)
            answer_text = answer.content.strip() if hasattr(answer, 'content') else str(answer).strip()
            
            # Ensure we have a meaningful answer
            if not answer_text or len(answer_text) < 10:
                answer_text = f"I found {len(retrieved)} relevant passages in the documents, but I cannot provide a specific answer to your question. Please review the snippets below to see what information is available."
                
        except Exception as e:
            print(f"❌ Error generating answer: {str(e)}")
            answer_text = f"I apologize, but I encountered an error while generating an answer. The retrieved documents contain information that may be relevant to your question. Please check the snippets below for details."
        
        snippets = [{"content": d.page_content} for d in retrieved]
        
        # Debug information
        print(f"🔍 Question: {q}")
        print(f"📄 Retrieved {len(retrieved)} total chunks")
        print(f"📊 Retrieval stats: {retrieval_stats}")
        
        return JSONResponse({
            "answer": answer_text, 
            "snippets": snippets, 
            "documents": doc_id,
            "retrieval_stats": retrieval_stats,
            "total_chunks_retrieved": len(retrieved),
            "query": q
        }) 

def _query_single_document_qa(doc_id: str, query: str, candidates_per_doc: int):
    """Query a single document for QA. Helper function for concurrent execution.
    
    Uses the unified query function from topic_processing utilities.
    """
    from src.utils.topic_processing import query_single_document
    return query_single_document(doc_id, query, candidates_per_doc, include_count=True) 