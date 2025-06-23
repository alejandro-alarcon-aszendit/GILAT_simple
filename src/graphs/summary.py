"""Summarization LangGraphs.

Contains graphs for single document and multi-topic summarization.
"""

import time
from typing import List
from langgraph.graph import Graph
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document

from src.core.config import LLMConfig, ParallelConfig
from src.services.document_service import DocumentService
from src.services.parallel_service import ParallelProcessingService, ParallelWorkload


def build_summary_graph():
    """Build basic document summarization graph.
    
    **Pipeline:**
    - Single node that summarizes a list of documents
    
    Returns:
        Compiled LangGraph for document summarization
    """
    g = Graph()

    def _summarise(state):
        """Summarize documents with optional query context."""
        docs = state.get("docs", [])
        query_context = state.get("query_context", "")
        
        if not docs:
            return {"summary": "No content to summarize."}
        
        try:
            # Use map-reduce summarization for large document sets
            chain = load_summarize_chain(LLMConfig.MAIN_LLM, chain_type="map_reduce")
            result = chain.invoke({"input_documents": docs})
            
            # Extract summary text
            if isinstance(result, dict) and "output_text" in result:
                summary_text = result["output_text"]
            elif isinstance(result, str):
                summary_text = result
            else:
                summary_text = str(result) if result else "Unable to generate summary."
            
            # Enhance with query context if provided
            if query_context and summary_text and summary_text != "Unable to generate summary.":
                enhanced_prompt = f"""
                Based on the following summary, provide a refined version that emphasizes aspects related to: {query_context}
                
                Original summary:
                {summary_text}
                
                Please ensure the refined summary maintains accuracy while highlighting relevant information about the specified topic.
                """
                enhanced_result = LLMConfig.MAIN_LLM.invoke(enhanced_prompt)
                if hasattr(enhanced_result, 'content'):
                    summary_text = enhanced_result.content.strip()
                
            return {"summary": summary_text}
        except Exception as e:
            return {"summary": f"Error generating summary: {str(e)}"}

    g.add_node("summarise", _summarise)
    g.set_entry_point("summarise")
    g.set_finish_point("summarise")
    return g.compile()


def build_multi_topic_summary_graph():
    """Build multi-topic parallel summarization graph.
    
    **Pipeline:**
    1. Map topics to relevant documents
    2. Process topics in parallel using ThreadPoolExecutor
    
    Returns:
        Compiled LangGraph for multi-topic summarization
    """
    g = Graph()
    
    def _map_topics(state):
        """Map step: Find relevant documents for each topic."""
        topics = state.get("topics", [])
        doc_ids = state.get("doc_ids", [])
        top_k = state.get("top_k", 10)
        
        if not topics or not doc_ids:
            return {"topic_docs": []}
        
        print(f"📋 Mapping {len(topics)} topics across {len(doc_ids)} documents...")
        
        topic_docs = []
        for topic in topics:
            topic_relevant_docs = []
            for doc_id in doc_ids:
                try:
                    vs = DocumentService.load_vector_store(doc_id)
                    retrieved = vs.similarity_search(topic.strip(), k=top_k)
                    topic_relevant_docs.extend(retrieved)
                except Exception as e:
                    print(f"Error retrieving docs for topic '{topic}' from doc {doc_id}: {e}")
                    continue
            
            topic_docs.append({
                "topic": topic.strip(),
                "docs": topic_relevant_docs,
                "doc_count": len(topic_relevant_docs)
            })
            print(f"  📄 Topic '{topic}' mapped to {len(topic_relevant_docs)} chunks")
        
        return {"topic_docs": topic_docs}
    
    def _reduce_summaries(state):
        """Reduce step: Generate summaries for each topic in parallel."""
        topic_docs = state.get("topic_docs", [])
        length = state.get("length", "medium")
        
        if not topic_docs:
            return {"summaries": []}
        
        def process_single_topic(topic_data):
            """Process a single topic - designed for parallel execution."""
            topic = topic_data["topic"]
            docs = topic_data["docs"]
            
            if not docs:
                return {
                    "topic": topic,
                    "summary": f"No relevant content found for topic: '{topic}'",
                    "chunks_processed": 0,
                    "status": "no_content",
                    "processing_time": 0
                }
            
            start_time = time.time()
            
            try:
                # Generate summary using map-reduce
                chain = load_summarize_chain(LLMConfig.MAIN_LLM, chain_type="map_reduce")
                result = chain.invoke({"input_documents": docs})
                
                # Extract summary text
                if isinstance(result, dict) and "output_text" in result:
                    summary_text = result["output_text"]
                elif isinstance(result, str):
                    summary_text = result
                else:
                    summary_text = str(result) if result else "Unable to generate summary."
                
                # Enhance with topic context
                if summary_text and summary_text != "Unable to generate summary.":
                    target = {"short": "≈3 sentences", "medium": "≈8 sentences", "long": "≈15 sentences"}[length]
                    enhanced_prompt = f"""
                    Create a focused summary about "{topic}" based on the following content. 
                    Keep it to {target} while emphasizing information most relevant to this specific topic.
                    
                    Content:
                    {summary_text}
                    """
                    enhanced_result = LLMConfig.MAIN_LLM.invoke(enhanced_prompt)
                    if hasattr(enhanced_result, 'content'):
                        summary_text = enhanced_result.content.strip()
                
                processing_time = time.time() - start_time
                
                return {
                    "topic": topic,
                    "summary": summary_text,
                    "chunks_processed": len(docs),
                    "status": "success",
                    "processing_time": processing_time
                }
                
            except Exception as e:
                processing_time = time.time() - start_time
                return {
                    "topic": topic,
                    "summary": f"Error generating summary for '{topic}': {str(e)}",
                    "chunks_processed": len(docs),
                    "status": "error",
                    "processing_time": processing_time
                }
        
        # Create parallel workloads
        workloads = []
        for i, topic_data in enumerate(topic_docs):
            workloads.append(ParallelWorkload(
                id=f"topic_{i}",
                name=f"Topic: {topic_data['topic']}",
                function=process_single_topic,
                args=(topic_data,),
                kwargs={},
                estimated_duration=30.0
            ))
        
        # Execute in parallel
        parallel_result = ParallelProcessingService.execute_workloads(
            workloads=workloads,
            max_workers=ParallelConfig.MAX_TOPIC_WORKERS
        )
        
        # Extract results and add performance metadata
        summaries = [r.result for r in parallel_result["results"] if r.success and r.result]
        performance = parallel_result["performance"]
        
        return {
            "summaries": summaries,
            "parallel_processing": {
                "total_time": performance["total_time"],
                "topics_count": performance["workloads_count"],
                "average_time_per_topic": performance["total_time"] / performance["workloads_count"] if performance["workloads_count"] else 0,
                "method": "ParallelProcessingService_ThreadPoolExecutor"
            }
        }
    
    g.add_node("map_topics", _map_topics)
    g.add_node("reduce_summaries", _reduce_summaries)
    g.set_entry_point("map_topics")
    g.add_edge("map_topics", "reduce_summaries")
    g.set_finish_point("reduce_summaries")
    return g.compile()


# Create compiled graph instances
SUMMARY_GRAPH = build_summary_graph()
MULTI_TOPIC_SUMMARY_GRAPH = build_multi_topic_summary_graph() 