"""Simple state transformation utilities for LangGraph workflows.

Provides basic utilities for transforming state between subgraphs while
maintaining compatibility with existing patterns.
"""

from typing import Dict, Any, List
from langchain.docstore.document import Document

from .graph_schemas import (
    DocumentRetrievalState,
    SummarizationState,
    SummarizationTaskState,
    ReflectionState,
    UnifiedState,
    TopicState,
)


# ==================== Simple State Creators ====================

def create_retrieval_state(topic: str, doc_ids: List[str], top_k: int = 10) -> DocumentRetrievalState:
    """Create document retrieval state."""
    return DocumentRetrievalState(
        topic=topic,
        doc_ids=doc_ids,
        top_k=top_k,
        relevant_documents=[],
        contributing_docs=[]
    )


def create_summarization_state(topics: List[str], doc_ids: List[str], 
                              strategy: str = 'abstractive', length: int = 5,
                              enable_reflection: bool = True, top_k: int = 10) -> SummarizationState:
    """Create summarization state from existing parameters."""
    return SummarizationState(
        topics=topics,
        doc_ids=doc_ids,
        top_k=top_k,
        length=length,
        strategy=strategy,
        enable_reflection=enable_reflection,
        summarization_sends=[],
        topic_results=[],
        summaries=[],
        parallel_processing={}
    )


def create_reflection_state(summary_text: str, topic: str, length_requirement: int,
                          source_content: str, enable_reflection: bool = True) -> ReflectionState:
    """Create reflection state."""
    return ReflectionState(
        summary_text=summary_text,
        topic=topic,
        length_requirement=length_requirement,
        source_content=source_content,
        enable_reflection=enable_reflection,
        final_summary='',
        reflection_applied=False,
        reflection_metadata={}
    )


# ==================== State Conversion Utilities ====================

def unified_to_summarization_state(unified_state: Dict[str, Any]) -> SummarizationState:
    """Convert UnifiedState to SummarizationState for subgraph usage."""
    return create_summarization_state(
        topics=unified_state.get('topics', []),
        doc_ids=unified_state.get('doc_ids', []),
        strategy=unified_state.get('strategy', 'abstractive'),
        length=unified_state.get('length', 5),
        enable_reflection=unified_state.get('enable_reflection', True),
        top_k=unified_state.get('top_k', 10)
    )


def summarization_to_unified_state(summarization_state: Dict[str, Any], 
                                 original_unified_state: Dict[str, Any]) -> Dict[str, Any]:
    """Update UnifiedState with results from SummarizationState."""
    return {
        **original_unified_state,
        'summaries': summarization_state.get('summaries', []),
        'parallel_processing': summarization_state.get('parallel_processing', {}),
        'topic_results': summarization_state.get('topic_results', [])
    }


def topic_state_to_task_state(topic_state: Dict[str, Any]) -> SummarizationTaskState:
    """Convert legacy TopicState to SummarizationTaskState."""
    return SummarizationTaskState(
        topic_id=topic_state.get('topic_id', 0),
        topic=topic_state.get('topic', ''),
        docs=topic_state.get('docs', []),
        source_content=topic_state.get('source_content', ''),
        length=topic_state.get('length', 5),
        strategy=topic_state.get('strategy', 'abstractive'),
        enable_reflection=topic_state.get('enable_reflection', True)
    )


# ==================== Simple State Utilities ====================

def extract_fields(state: Dict[str, Any], fields: List[str]) -> Dict[str, Any]:
    """Extract specific fields from state."""
    return {field: state.get(field) for field in fields if field in state}


def merge_state_updates(original_state: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Merge state updates into original state."""
    return {**original_state, **updates}