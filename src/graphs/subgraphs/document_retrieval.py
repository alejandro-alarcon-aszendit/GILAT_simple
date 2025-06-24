"""Document Retrieval Subgraph for LangGraph workflows.

Simple, focused subgraph that extracts the document retrieval logic from the
existing topic processing utilities into a reusable component with cross-document
similarity ranking using concurrent queries.
"""

from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.docstore.document import Document
from langgraph.graph import StateGraph, START, END

from src.utils.graph_schemas import DocumentRetrievalState
from src.services.document_service import DocumentService


def _query_single_document_subgraph(doc_id: str, topic: str, candidates_per_doc: int) -> Tuple[List[Tuple[Document, float, str]], str]:
    """Query a single document for relevant chunks. Helper function for concurrent execution in subgraph.
    
    Uses the unified query function from topic_processing utilities.
    """
    from src.utils.topic_processing import query_single_document
    candidates, doc_id_result, _ = query_single_document(doc_id, topic, candidates_per_doc)
    return candidates, doc_id_result


def build_document_retrieval_subgraph() -> StateGraph:
    """Build document retrieval subgraph with concurrent database queries.
    
    **Pipeline:**
    1. Retrieve relevant documents for topic from specified document IDs (CONCURRENTLY)
    2. Rank by similarity score across all documents
    3. Return top-k globally ranked documents
    
    Returns:
        Compiled StateGraph for concurrent document retrieval
    """
    
    def retrieve_documents_for_topic(state: DocumentRetrievalState) -> DocumentRetrievalState:
        """Retrieve relevant documents for a specific topic with concurrent cross-document similarity ranking.
        
        Enhanced to collect similarity scores from all documents using concurrent queries
        and return the globally highest-scoring chunks rather than top-k per document.
        """
        topic = state["topic"]
        doc_ids = state["doc_ids"]
        top_k = state["top_k"]
        
        # Collect all candidates with scores from all documents
        all_candidates = []
        doc_sources = []
        
        print(f"🔍 Retrieving documents for topic: '{topic}' with concurrent cross-document ranking")
        
        # Get more candidates per document for better global ranking
        candidates_per_doc = min(top_k * 2, 20)  # Get 2x requested or max 20 per doc
        
        # Execute queries concurrently using ThreadPoolExecutor
        from src.core.config import ParallelConfig
        with ThreadPoolExecutor(max_workers=min(len(doc_ids), ParallelConfig.MAX_DB_QUERY_WORKERS)) as executor:
            # Submit all document queries concurrently
            future_to_doc_id = {
                executor.submit(_query_single_document_subgraph, doc_id, topic, candidates_per_doc): doc_id 
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
            print(f"⚠️ No documents found for topic '{topic}'")
            return {
                **state,
                "relevant_documents": [],
                "contributing_docs": []
            }
        
        # Sort all candidates by similarity score (lower scores = higher similarity in some embeddings)
        # Note: Chroma typically returns distance scores where lower = more similar
        all_candidates.sort(key=lambda x: x[1])  # Sort by score (ascending = most similar first)
        
        # Take the top-k globally ranked documents
        top_global_candidates = all_candidates[:top_k]
        
        # Extract just the documents for the final result
        topic_relevant_docs = [doc for doc, score, doc_id in top_global_candidates]
        
        # Get unique contributing document IDs from the final selection
        final_doc_sources = list(set(doc_id for doc, score, doc_id in top_global_candidates))
        
        print(f"✅ Selected {len(topic_relevant_docs)} globally top-ranked documents for topic '{topic}' (concurrent)")
        print(f"📊 Score range: {top_global_candidates[0][1]:.4f} to {top_global_candidates[-1][1]:.4f}")
        print(f"📋 Contributing documents: {final_doc_sources}")
        print(f"🚀 Queried {len(doc_ids)} documents concurrently")
        
        return {
            **state,
            "relevant_documents": topic_relevant_docs,
            "contributing_docs": final_doc_sources
        }
    
    # Build the simple subgraph
    graph = StateGraph(DocumentRetrievalState)
    
    # Add single node for retrieval
    graph.add_node("retrieve_documents", retrieve_documents_for_topic)
    
    # Simple linear flow
    graph.add_edge(START, "retrieve_documents")
    graph.add_edge("retrieve_documents", END)
    
    return graph.compile()


# Create the compiled subgraph instance
DOCUMENT_RETRIEVAL_SUBGRAPH = build_document_retrieval_subgraph()