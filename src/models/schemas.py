"""Pydantic schemas for structured outputs and API responses."""

from datetime import datetime
from typing import List, Literal, Optional
from pydantic import BaseModel, Field

# -------------------- Reflection System Schemas -----------------
class SummaryEvaluation(BaseModel):
    """Structured evaluation of a summary's quality and accuracy."""
    factual_accuracy: Literal["excellent", "good", "fair", "poor"] = Field(
        description="Assessment of factual accuracy based on source content"
    )
    length_compliance: Literal["perfect", "slightly_over", "slightly_under", "significantly_off"] = Field(
        description="How well the summary meets the specified length requirement"
    )
    topic_relevance: Literal["highly_relevant", "mostly_relevant", "somewhat_relevant", "off_topic"] = Field(
        description="How well the summary addresses the specified topic"
    )
    clarity_readability: Literal["excellent", "good", "fair", "poor"] = Field(
        description="Overall clarity and readability of the summary"
    )
    improvement_needed: bool = Field(
        description="Whether the summary needs improvement"
    )
    specific_issues: List[str] = Field(
        description="List of specific issues found (empty if none)",
        default=[]
    )
    confidence_score: float = Field(
        description="Confidence in the evaluation (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )

class ImprovedSummary(BaseModel):
    """Improved version of a summary with reflection metadata."""
    improved_text: str = Field(
        description="The improved version of the summary"
    )
    changes_made: List[str] = Field(
        description="List of specific changes made during improvement"
    )
    final_evaluation: SummaryEvaluation = Field(
        description="Final evaluation of the improved summary"
    )

# -------------------- API Response Schemas -----------------------
class DocOut(BaseModel):
    """Document output schema for API responses."""
    id: str
    name: str
    status: str
    n_chunks: Optional[int]
    created_at: datetime

class TopicSummary(BaseModel):
    """Schema for individual topic summaries in multi-topic responses."""
    topic: str
    summary: str
    chunks_processed: int
    status: str
    processing_time: float
    strategy: str = Field(description="Summarization strategy used (abstractive, extractive, hybrid)")
    reflection_applied: Optional[bool] = False
    changes_made: Optional[List[str]] = None
    initial_evaluation: Optional[dict] = None
    final_evaluation: Optional[dict] = None

class ParallelProcessingMetadata(BaseModel):
    """Metadata about parallel processing performance."""
    total_time: float
    topics_count: int
    average_time_per_topic: float
    method: str
    reflection_statistics: Optional[dict] = None

class MultiTopicSummaryResponse(BaseModel):
    """Response schema for multi-topic summarization."""
    type: str = "multi_topic"
    summaries: List[TopicSummary]
    documents: List[str]
    topics: List[str]
    strategy: str = Field(description="Summarization strategy used")
    total_chunks_processed: int
    successful_topics: int
    total_topics: int
    search_method: str
    parallel_processing: ParallelProcessingMetadata
    performance: dict
    reflection_enabled: bool
    reflection_statistics: Optional[dict] = None

class SingleSummaryResponse(BaseModel):
    """Response schema for single document/topic summarization."""
    type: str = "single"
    summary: str
    documents: List[str]
    chunks_processed: int
    strategy: str = Field(description="Summarization strategy used")
    query: Optional[str] = None
    search_method: str

class QAResponse(BaseModel):
    """Response schema for question answering."""
    answer: str
    snippets: List[dict]
    documents: List[str] 