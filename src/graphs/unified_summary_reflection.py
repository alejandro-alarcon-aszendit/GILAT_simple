"""Unified summarization and reflection graph using modular LangGraph subgraphs.

This module demonstrates LangGraph subgraph composition by orchestrating 
document retrieval, summarization, and reflection using reusable subgraphs
with proper state management and Send API for parallel processing.

Supports both multi-topic processing and parallel document processing for
full document summaries.
"""

import time
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from typing import List

# Import enhanced state schemas and utilities
from src.utils.graph_schemas import UnifiedState, SummarizationState, DocumentRetrievalState
from src.utils.state_transformers import unified_to_summarization_state, summarization_to_unified_state

# Import modular subgraphs
from src.graphs.subgraphs import SUMMARIZATION_SUBGRAPH, DOCUMENT_RETRIEVAL_SUBGRAPH, REFLECTION_SUBGRAPH


def build_unified_summary_reflection_graph():
    """Build unified graph using modular subgraphs with proper state management.
    
    **LangGraph Subgraph Composition Pipeline:**
    1. Transform UnifiedState → SummarizationState
    2. Delegate to SummarizationSubgraph (handles Send API parallel processing)
    3. Transform SummarizationState → UnifiedState
    4. Collect final results with statistics
    
    **Benefits:**
    - Reusable subgraphs for different workflows
    - Clean separation of concerns
    - Maintains existing functionality patterns
    - Enhanced modularity and testability
    
    Returns:
        Compiled StateGraph for unified summarization with reflection
    """
    
    def prepare_for_summarization(state: UnifiedState) -> UnifiedState:
        """Transform UnifiedState to SummarizationState for subgraph delegation."""
        topics = state.get("topics", [])
        doc_ids = state.get("doc_ids", [])
        
        if not topics or not doc_ids:
            return {
                **state,
                "summaries": [],
                "parallel_processing": {
                    "total_time": 0.0,
                    "topics_count": 0,
                    "method": "LangGraph_Subgraph_Composition",
                    "reflection_statistics": {
                        "total_topics": 0,
                        "reflection_applied": 0,
                        "reflection_skipped": 0
                    }
                }
            }
        
        print(f"🏗️  Preparing {len(topics)} topics for modular subgraph processing...")
        return state
    
    def delegate_to_summarization_subgraph(state: UnifiedState) -> UnifiedState:
        """Delegate to SummarizationSubgraph using state transformation.
        
        This demonstrates proper LangGraph subgraph composition patterns.
        """
        # Transform state for subgraph
        summarization_state = unified_to_summarization_state(state)
        
        print("🔀 Delegating to SummarizationSubgraph...")
        
        # Execute the summarization subgraph
        result = SUMMARIZATION_SUBGRAPH.invoke(summarization_state)
        
        # Transform results back to unified state
        updated_state = summarization_to_unified_state(result, state)
        
        print("✅ SummarizationSubgraph delegation completed")
        
        return updated_state
    
    def finalize_results(state: UnifiedState) -> UnifiedState:
        """Finalize results and update metadata for subgraph composition."""
        summaries = state.get("summaries", [])
        parallel_processing = state.get("parallel_processing", {})
        
        # Update processing method to reflect subgraph composition
        if parallel_processing:
            parallel_processing["method"] = "LangGraph_Subgraph_Composition"
            parallel_processing["subgraphs_used"] = [
                "SummarizationSubgraph",
                "DocumentRetrievalSubgraph", 
                "ReflectionSubgraph"
            ]
        
        print(f"🎯 Finalized {len(summaries)} topic summaries using modular subgraphs")
        
        return {
            **state,
            "summaries": summaries,
            "parallel_processing": parallel_processing
        }
    
    # Build the graph using StateGraph with modular subgraph composition
    graph = StateGraph(UnifiedState)
    
    # Add nodes for subgraph orchestration
    graph.add_node("prepare_for_summarization", prepare_for_summarization)
    graph.add_node("delegate_to_summarization_subgraph", delegate_to_summarization_subgraph)
    graph.add_node("finalize_results", finalize_results)
    
    # Add linear edges for subgraph composition
    graph.add_edge(START, "prepare_for_summarization")
    graph.add_edge("prepare_for_summarization", "delegate_to_summarization_subgraph")
    graph.add_edge("delegate_to_summarization_subgraph", "finalize_results")
    graph.add_edge("finalize_results", END)
    
    return graph.compile()


# Create the compiled unified graph instance
UNIFIED_SUMMARY_REFLECTION_GRAPH = build_unified_summary_reflection_graph() 