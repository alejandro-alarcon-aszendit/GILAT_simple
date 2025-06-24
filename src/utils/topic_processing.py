"""Topic processing utilities for document retrieval and content preparation.

This module handles the mapping of topics to relevant documents and prepares
content for summarization and reflection processes.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Tuple
from langchain.docstore.document import Document

from src.core.config import ParallelConfig, LLMConfig
from src.services.document_service import DocumentService
from src.utils.summarization_strategies import get_strategy_function
from src.utils.reflection_utils import apply_reflection_to_summary


def query_single_document(doc_id: str, query: str, candidates_per_doc: int, include_count: bool = False) -> Tuple[List[Tuple[Document, float, str]], str, int]:
    """Query a single document for relevant chunks. Unified helper function for concurrent execution.
    
    Args:
        doc_id: Document ID to query
        query: Query/topic to search for
        candidates_per_doc: Number of candidates to retrieve per document
        include_count: Whether to include retrieval count in return
        
    Returns:
        Tuple of (candidates_list, doc_id, retrieval_count)
    """
    try:
        vs = DocumentService.load_vector_store(doc_id)
        retrieved_with_scores = vs.similarity_search_with_score(query.strip(), k=candidates_per_doc)
        
        candidates = []
        if retrieved_with_scores:
            # Add document source metadata to each chunk
            for doc, score in retrieved_with_scores:
                doc.metadata = doc.metadata or {}
                doc.metadata["source_doc_id"] = doc_id
                doc.metadata["similarity_score"] = score
                candidates.append((doc, score, doc_id))
            
            print(f"    📄 Query '{query}' found {len(retrieved_with_scores)} candidate chunks in document {doc_id}")
        
        return candidates, doc_id, len(retrieved_with_scores)
        
    except Exception as e:
        print(f"Error retrieving docs for query '{query}' from doc {doc_id}: {e}")
        return [], doc_id, 0


# Backward compatibility wrapper
def _query_single_document(doc_id: str, topic: str, candidates_per_doc: int) -> Tuple[List[Tuple[Document, float, str]], str]:
    """Backward compatibility wrapper for the unified query function."""
    candidates, doc_id_result, _ = query_single_document(doc_id, topic, candidates_per_doc)
    return candidates, doc_id_result


def retrieve_documents_for_topic(topic: str, doc_ids: List[str], top_k: int = 10) -> Tuple[List[Document], List[str]]:
    """Retrieve relevant documents for a specific topic with concurrent cross-document similarity ranking.
    
    Args:
        topic: Topic to search for
        doc_ids: List of document IDs to search in
        top_k: Number of top-ranked chunks to retrieve globally
        
    Returns:
        Tuple of (relevant_documents, contributing_doc_ids)
    """
    all_candidates = []
    doc_sources = []
    
    print(f"    🔍 Retrieving documents for topic '{topic}' with concurrent cross-document ranking")
    
    # Get more candidates per document for better global ranking
    candidates_per_doc = min(top_k * 2, 20)  # Get 2x requested or max 20 per doc
    
    # Execute queries concurrently using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=min(len(doc_ids), ParallelConfig.MAX_DB_QUERY_WORKERS)) as executor:
        # Submit all document queries concurrently
        future_to_doc_id = {
            executor.submit(_query_single_document, doc_id, topic, candidates_per_doc): doc_id 
            for doc_id in doc_ids
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_doc_id):
            doc_id = future_to_doc_id[future]
            try:
                candidates, returned_doc_id = future.result()
                if candidates:
                    all_candidates.extend(candidates)
                    doc_sources.append(returned_doc_id)
            except Exception as e:
                print(f"Error processing results for topic '{topic}' from doc {doc_id}: {e}")
                continue
    
    if not all_candidates:
        return [], []
    
    # Sort by similarity score and take top-k globally
    all_candidates.sort(key=lambda x: x[1])  # Sort by score (ascending = most similar first)
    top_candidates = all_candidates[:top_k]
    
    # Extract documents and contributing doc IDs
    topic_relevant_docs = [doc for doc, score, doc_id in top_candidates]
    final_doc_sources = list(set(doc_id for doc, score, doc_id in top_candidates))
    
    print(f"    ✅ Selected {len(topic_relevant_docs)} globally top-ranked chunks for topic '{topic}' (concurrent retrieval)")
    print(f"    📊 Score range: {top_candidates[0][1]:.4f} to {top_candidates[-1][1]:.4f}")
    print(f"    🚀 Queried {len(doc_ids)} documents concurrently")
    
    return topic_relevant_docs, final_doc_sources


def prepare_source_content(docs: List[Document]) -> str:
    """Prepare source content for reflection by combining and truncating documents.
    
    Args:
        docs: List of documents to combine
        
    Returns:
        Combined and truncated source content string
    """
    # Limit documents for performance
    limited_docs = docs[:min(ParallelConfig.MAX_CHUNKS_PER_TOPIC, len(docs))]
    source_content = "\n\n".join([doc.page_content for doc in limited_docs])
    
    # Truncate if too long
    if len(source_content) > ParallelConfig.MAX_SOURCE_CONTENT_LENGTH:
        source_content = source_content[:ParallelConfig.MAX_SOURCE_CONTENT_LENGTH] + "\n\n[Content truncated for reflection...]"
    
    return source_content


def enhance_summary_for_topic(summary: str, topic: str, length: int, strategy: str) -> str:
    """Enhance summary with topic-specific context for non-extractive strategies.
    
    Args:
        summary: Initial summary text
        topic: Topic context
        length: Target length (number of sentences)
        strategy: Summarization strategy used
        
    Returns:
        Enhanced summary text
    """
    # Skip enhancement for extractive strategy to preserve original sentences
    if strategy == "extractive" or not summary or summary == "Unable to generate summary.":
        return summary
    
    target = f"exactly {length} sentences"
    enhanced_prompt = f"""
    Create a focused summary about "{topic}" based on the following content. 
    Keep it to {target} while emphasizing information most relevant to this specific topic.
    
    Content:
    {summary}
    """
    
    try:
        enhanced_result = LLMConfig.MAIN_LLM.invoke(enhanced_prompt)
        if hasattr(enhanced_result, 'content'):
            return enhanced_result.content.strip()
    except Exception as e:
        print(f"    ⚠️ Topic enhancement failed: {e}")
    
    return summary  # Return original if enhancement fails


def process_single_topic_complete(
    topic_id: int,
    topic: str, 
    docs: List[Document],
    source_content: str,
    length: int,
    strategy: str,
    enable_reflection: bool
) -> Dict[str, Any]:
    """Process a single topic with summarization and optional reflection.
    
    This is the complete processing pipeline for a single topic including:
    1. Strategy-based summarization
    2. Topic enhancement (for non-extractive strategies)
    3. Optional reflection and improvement
    
    Args:
        topic_id: Unique identifier for the topic
        topic: Topic text
        docs: Relevant documents for this topic
        source_content: Combined source content for reflection
        length: Target summary length
        strategy: Summarization strategy to use
        enable_reflection: Whether to apply reflection
        
    Returns:
        Dict containing summary result with metadata
    """
    print(f"  📝 Processing topic {topic_id}: '{topic}' with {len(docs)} docs...")
    start_time = time.time()
    
    if not docs:
        return {
            "topic": topic,
            "topic_id": topic_id,
            "summary": f"No relevant content found for topic: '{topic}'",
            "chunks_processed": 0,
            "status": "no_content",
            "processing_time": time.time() - start_time,
            "strategy": strategy,
            "reflection_applied": False
        }
    
    try:
        # Step 1: Generate initial summary using strategy
        strategy_function = get_strategy_function(strategy)
        summary_result = strategy_function(docs, topic)
        initial_summary = summary_result.get("summary", "Unable to generate summary.")
        
        # Step 2: Enhance with topic context (for non-extractive strategies)
        enhanced_summary = enhance_summary_for_topic(initial_summary, topic, length, strategy)
        
        # Step 3: Apply reflection if enabled
        final_summary = enhanced_summary
        reflection_metadata = {"reflection_applied": False}
        
        if enable_reflection and enhanced_summary and enhanced_summary != "Unable to generate summary.":
            print(f"    🔍 Applying reflection to topic {topic_id}: '{topic}'...")
            
            try:
                reflection_result = apply_reflection_to_summary(
                    summary_text=enhanced_summary,
                    topic=topic,
                    length_requirement=length,
                    source_content=source_content
                )
                
                improved_summary = reflection_result.get("improved_summary")
                
                if improved_summary and hasattr(improved_summary, 'improved_text') and improved_summary.improved_text:
                    final_summary = improved_summary.improved_text
                    reflection_metadata = {
                        "reflection_applied": True,
                        "changes_made": improved_summary.changes_made if hasattr(improved_summary, 'changes_made') else [],
                        "initial_evaluation": reflection_result.get("evaluation").__dict__ if reflection_result.get("evaluation") else None,
                        "final_evaluation": improved_summary.final_evaluation.__dict__ if hasattr(improved_summary, 'final_evaluation') else None
                    }
                    print(f"    ✨ Reflection complete for topic {topic_id}: '{topic}'")
                else:
                    reflection_metadata = {
                        "reflection_applied": False,
                        "error": reflection_result.get("error", "Unknown reflection error")
                    }
                    print(f"    ⚠️ Reflection failed for topic {topic_id}: '{topic}', using original summary")
                    
            except Exception as e:
                print(f"    ❌ Reflection error for topic {topic_id}: '{topic}' - {str(e)}")
                reflection_metadata = {
                    "reflection_applied": False,
                    "error": f"Reflection failed: {str(e)}"
                }
        
        processing_time = time.time() - start_time
        print(f"  ✅ Completed topic {topic_id}: '{topic}' in {processing_time:.2f}s")
        
        return {
            "topic": topic,
            "topic_id": topic_id,
            "summary": final_summary,
            "chunks_processed": len(docs),
            "status": "success",
            "processing_time": processing_time,
            "strategy": strategy,
            **reflection_metadata
        }
        
    except Exception as e:
        processing_time = time.time() - start_time
        print(f"  ❌ Failed topic {topic_id}: '{topic}' in {processing_time:.2f}s - {str(e)}")
        
        return {
            "topic": topic,
            "topic_id": topic_id,
            "summary": f"Error generating summary for '{topic}': {str(e)}",
            "chunks_processed": len(docs),
            "status": "error",
            "processing_time": processing_time,
            "strategy": strategy,
            "reflection_applied": False
        } 