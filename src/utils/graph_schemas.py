"""State schemas for LangGraph workflows with enhanced modularity.

This module defines TypedDict schemas for modular LangGraph workflows
while maintaining backward compatibility with existing patterns.
"""

import operator
from typing import List, Dict, Any, Optional, Annotated
from typing_extensions import TypedDict
from langchain.docstore.document import Document
from langgraph.types import Send


# ==================== Core States (Enhanced from existing) ====================

class UnifiedState(TypedDict):
    """Main graph state with automatic result aggregation using reducers.
    
    Enhanced version maintaining backward compatibility while supporting subgraphs.
    """
    # Input parameters
    topics: List[str]
    doc_ids: List[str]
    top_k: int
    length: int
    strategy: str
    enable_reflection: bool
    
    # Intermediate state for Send routing
    sends: List[Send]
    
    # Results aggregated automatically using operator.add
    topic_results: Annotated[List[Dict[str, Any]], operator.add]
    
    # Final outputs
    summaries: List[Dict[str, Any]]
    parallel_processing: Dict[str, Any]


class TopicState(TypedDict):
    """Individual topic state for parallel processing via Send API."""
    topic_id: int
    topic: str
    docs: List[Document]
    source_content: str
    length: int
    strategy: str
    enable_reflection: bool
    contributing_docs: List[str]
    doc_count: int


# ==================== Subgraph States ====================

class DocumentRetrievalState(TypedDict):
    """State for document retrieval subgraph."""
    # Input
    topic: str
    doc_ids: List[str]
    top_k: int
    
    # Output
    relevant_documents: List[Document]
    contributing_docs: List[str]


class SummarizationTaskState(TypedDict):
    """State for individual summarization task (used with Send API)."""
    topic_id: int
    topic: str
    docs: List[Document]
    source_content: str
    length: int
    strategy: str
    enable_reflection: bool


class SummarizationState(TypedDict):
    """State for summarization subgraph with parallel processing."""
    # Input parameters
    topics: List[str]
    doc_ids: List[str]
    top_k: int
    length: int
    strategy: str
    enable_reflection: bool
    
    # Send API coordination
    summarization_sends: List[Send]
    
    # Results aggregated automatically using operator.add
    topic_results: Annotated[List[Dict[str, Any]], operator.add]
    
    # Output
    summaries: List[Dict[str, Any]]
    parallel_processing: Dict[str, Any]


class ReflectionState(TypedDict):
    """State for reflection subgraph."""
    # Input
    summary_text: str
    topic: str
    length_requirement: int
    source_content: str
    enable_reflection: bool
    
    # Output
    final_summary: str
    reflection_applied: bool
    reflection_metadata: Dict[str, Any]