"""Unified summarization and reflection graph using LangGraph's native Send command.

This module provides a clean, modular graph that orchestrates summarization and 
reflection using LangGraph's Send API for true parallel processing. All utility
functions have been extracted to separate modules for better maintainability.
"""

import time
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

# Import utility modules
from src.utils.graph_schemas import UnifiedState, TopicState
from src.utils.topic_processing import (
    retrieve_documents_for_topic, 
    prepare_source_content,
    process_single_topic_complete
)


def build_unified_summary_reflection_graph():
    """Build unified graph that uses LangGraph's Send for parallel processing.
    
    **Pipeline:**
    1. Map topics to relevant documents
    2. Use Send API to process each topic in parallel with optional reflection
    3. Collect results using state reducers (operator.add)
    
    Returns:
        Compiled StateGraph for unified summarization with reflection
    """
    
    def map_topics_to_sends(state: UnifiedState):
        """Map step: Create Send objects for parallel topic processing."""
        topics = state.get("topics", [])
        doc_ids = state.get("doc_ids", [])
        top_k = state.get("top_k", 10)
        length = state.get("length", "medium")
        strategy = state.get("strategy", "abstractive")
        enable_reflection = state.get("enable_reflection", True)
        
        if not topics or not doc_ids:
            return {"topic_results": []}
        
        print(f"📋 Mapping {len(topics)} topics for parallel processing using LangGraph Send...")
        
        sends = []
        for i, topic in enumerate(topics):
            # Retrieve relevant documents for this topic
            topic_relevant_docs, doc_sources = retrieve_documents_for_topic(topic, doc_ids, top_k)
            
            # Prepare source content for reflection
            source_content = prepare_source_content(topic_relevant_docs)
            
            # Create individual state for this topic
            topic_state = {
                "topic_id": i,
                "topic": topic.strip(),
                "docs": topic_relevant_docs,
                "source_content": source_content,
                "length": length,
                "strategy": strategy,
                "enable_reflection": enable_reflection,
                "contributing_docs": doc_sources,
                "doc_count": len(topic_relevant_docs)
            }
            
            # Create Send object for parallel processing
            sends.append(Send("process_topic_with_reflection", topic_state))
            print(f"  📋 Created Send for topic '{topic}' with {len(topic_relevant_docs)} chunks")
        
        print(f"🚀 Created {len(sends)} Send objects for parallel processing")
        # Store sends in state for conditional routing
        return {"sends": sends}
    
    def route_to_parallel_processing(state: UnifiedState):
        """Route to parallel processing using Send objects."""
        sends = state.get("sends", [])
        if not sends:
            # No sends means no topics to process, go directly to collect
            return "collect_results"
        return sends  # Return list of Send objects for parallel execution
    
    def process_topic_with_reflection(state: TopicState):
        """Process a single topic with summary generation and optional reflection.
        
        This function is called in parallel for each topic via Send API.
        Returns partial state that will be automatically aggregated.
        """
        # Extract state parameters
        topic_id = state.get("topic_id", 0)
        topic = state.get("topic", "")
        docs = state.get("docs", [])
        source_content = state.get("source_content", "")
        length = state.get("length", "medium")
        strategy = state.get("strategy", "abstractive")
        enable_reflection = state.get("enable_reflection", True)
        
        # Use the complete topic processing utility
        result = process_single_topic_complete(
            topic_id=topic_id,
            topic=topic,
            docs=docs,
            source_content=source_content,
            length=length,
            strategy=strategy,
            enable_reflection=enable_reflection
        )
        
        # Return single-item list - will be automatically aggregated by operator.add
        return {"topic_results": [result]}
    
    def collect_results(state: UnifiedState):
        """Collect and aggregate all topic results (automatically aggregated by reducer)."""
        topic_results = state.get("topic_results", [])
        
        if not topic_results:
            return {
                "summaries": [],
                "parallel_processing": {
                    "total_time": 0.0,
                    "topics_count": 0,
                    "method": "LangGraph_Send_API",
                    "reflection_statistics": {
                        "total_topics": 0,
                        "reflection_applied": 0,
                        "reflection_skipped": 0
                    }
                }
            }
        
        # Sort results by topic_id to maintain order
        sorted_results = sorted(topic_results, key=lambda x: x.get("topic_id", 0))
        
        # Calculate statistics
        total_time = max([r.get("processing_time", 0) for r in sorted_results]) if sorted_results else 0
        reflection_applied_count = sum(1 for r in sorted_results if r.get("reflection_applied", False))
        
        print(f"🎉 LangGraph Send parallel processing completed")
        print(f"    📊 Processed {len(sorted_results)} topics")
        print(f"    🔍 Reflection applied to {reflection_applied_count} topics")
        print(f"    ⏱️  Max processing time: {total_time:.2f}s")
        
        return {
            "summaries": sorted_results,
            "parallel_processing": {
                "total_time": total_time,
                "topics_count": len(sorted_results),
                "average_time_per_topic": sum(r.get("processing_time", 0) for r in sorted_results) / len(sorted_results) if sorted_results else 0,
                "method": "LangGraph_Send_API",
                "reflection_statistics": {
                    "total_topics": len(sorted_results),
                    "reflection_applied": reflection_applied_count,
                    "reflection_skipped": len(sorted_results) - reflection_applied_count
                }
            }
        }
    
    # Build the graph using StateGraph
    graph = StateGraph(UnifiedState)
    
    # Add nodes
    graph.add_node("map_topics_to_sends", map_topics_to_sends)
    graph.add_node("process_topic_with_reflection", process_topic_with_reflection)
    graph.add_node("collect_results", collect_results)
    
    # Add edges
    graph.add_edge(START, "map_topics_to_sends")
    
    # Add conditional edge that creates Send objects for parallel processing
    graph.add_conditional_edges(
        "map_topics_to_sends",
        route_to_parallel_processing,  # Function that returns Send objects
        ["process_topic_with_reflection"]  # Target nodes for Send objects
    )
    
    # Edge from parallel processing to collection (automatic via state reducer)
    graph.add_edge("process_topic_with_reflection", "collect_results")
    graph.add_edge("collect_results", END)
    
    return graph.compile()


# Create the compiled unified graph instance
UNIFIED_SUMMARY_REFLECTION_GRAPH = build_unified_summary_reflection_graph() 