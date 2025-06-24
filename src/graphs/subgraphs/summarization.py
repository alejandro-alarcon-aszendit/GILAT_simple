"""Summarization Subgraph for LangGraph workflows.

Modular subgraph that handles topic-based summarization with parallel processing
using LangGraph's Send API. Extracts and enhances existing summarization logic.
"""

import time
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send

from src.utils.graph_schemas import SummarizationState, SummarizationTaskState
from src.utils.topic_processing import retrieve_documents_for_topic, prepare_source_content, process_single_topic_complete


def build_summarization_subgraph():
    """Build summarization subgraph with parallel processing.
    
    **Pipeline:**
    1. Map topics to Send objects for parallel processing
    2. Process each topic using existing logic (summarization + optional reflection)
    3. Collect and aggregate results
    
    Returns:
        Compiled StateGraph for parallel summarization
    """
    
    def map_topics_to_sends(state: SummarizationState) -> SummarizationState:
        """Create Send objects for parallel topic processing.
        
        Enhanced version of the existing mapping logic.
        """
        topics = state["topics"]
        doc_ids = state["doc_ids"]
        top_k = state["top_k"]
        length = state["length"]
        strategy = state["strategy"]
        enable_reflection = state["enable_reflection"]
        
        if not topics or not doc_ids:
            return {**state, "summarization_sends": []}
        
        print(f"📋 Mapping {len(topics)} topics for parallel processing...")
        
        sends = []
        for i, topic in enumerate(topics):
            # Use existing retrieval logic
            topic_relevant_docs, doc_sources = retrieve_documents_for_topic(topic, doc_ids, top_k)
            source_content = prepare_source_content(topic_relevant_docs)
            
            # Create task state for Send API
            task_state = SummarizationTaskState(
                topic_id=i,
                topic=topic.strip(),
                docs=topic_relevant_docs,
                source_content=source_content,
                length=length,
                strategy=strategy,
                enable_reflection=enable_reflection
            )
            
            sends.append(Send("process_topic", task_state))
            print(f"  📋 Created Send for topic '{topic}' with {len(topic_relevant_docs)} chunks")
        
        print(f"🚀 Created {len(sends)} Send objects for parallel processing")
        
        return {
            **state,
            "summarization_sends": sends
        }
    
    def route_to_parallel_processing(state: SummarizationState):
        """Route to parallel processing using Send objects."""
        sends = state.get("summarization_sends", [])
        if not sends:
            return "collect_results"
        return sends
    
    def process_topic(task_state: SummarizationTaskState) -> SummarizationState:
        """Process a single topic using existing complete processing logic."""
        # Use existing complete processing function
        result = process_single_topic_complete(
            topic_id=task_state["topic_id"],
            topic=task_state["topic"],
            docs=task_state["docs"],
            source_content=task_state["source_content"],
            length=task_state["length"],
            strategy=task_state["strategy"],
            enable_reflection=task_state["enable_reflection"]
        )
        
        # Return in format expected by reducer
        return {"topic_results": [result]}
    
    def collect_results(state: SummarizationState) -> SummarizationState:
        """Collect and aggregate parallel processing results.
        
        Enhanced version of existing collection logic.
        """
        topic_results = state.get("topic_results", [])
        
        if not topic_results:
            return {
                **state,
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
        
        # Calculate statistics (existing logic)
        total_time = max([r.get("processing_time", 0) for r in sorted_results]) if sorted_results else 0
        reflection_applied_count = sum(1 for r in sorted_results if r.get("reflection_applied", False))
        
        print(f"🎉 Summarization parallel processing completed")
        print(f"    📊 Processed {len(sorted_results)} topics")
        print(f"    🔍 Reflection applied to {reflection_applied_count} topics")
        print(f"    ⏱️  Max processing time: {total_time:.2f}s")
        
        parallel_stats = {
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
        
        return {
            **state,
            "summaries": sorted_results,
            "parallel_processing": parallel_stats
        }
    
    # Build the subgraph
    graph = StateGraph(SummarizationState)
    
    # Add nodes
    graph.add_node("map_topics_to_sends", map_topics_to_sends)
    graph.add_node("process_topic", process_topic)
    graph.add_node("collect_results", collect_results)
    
    # Add edges
    graph.add_edge(START, "map_topics_to_sends")
    
    # Conditional routing for parallel execution
    graph.add_conditional_edges(
        "map_topics_to_sends",
        route_to_parallel_processing,
        ["process_topic", "collect_results"]
    )
    
    # Parallel execution flows to collection
    graph.add_edge("process_topic", "collect_results")
    graph.add_edge("collect_results", END)
    
    return graph.compile()


# Create the compiled subgraph instance
SUMMARIZATION_SUBGRAPH = build_summarization_subgraph()