"""State schemas for the unified graph using LangGraph's Send API.

This module defines the TypedDict schemas used for state management
in the unified summarization and reflection graph.
"""

import operator
from typing import List, Dict, Any, Annotated
from typing_extensions import TypedDict
from langchain.docstore.document import Document
from langgraph.types import Send


class UnifiedState(TypedDict):
    """Main graph state with automatic result aggregation using reducers.
    
    This state schema supports LangGraph's Send API for parallel processing
    with automatic result aggregation using the operator.add reducer.
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
    """Individual topic state for parallel processing via Send API.
    
    This schema defines the state passed to each parallel topic processing
    task when using LangGraph's Send API.
    """
    topic_id: int
    topic: str
    docs: List[Document]
    source_content: str
    length: int
    strategy: str
    enable_reflection: bool
    contributing_docs: List[str]
    doc_count: int 