"""Unified summarization and reflection graph using LangGraph's native Send command.

This module replaces the separate summary and reflection graphs with a single unified
graph that uses LangGraph's Send API for true parallel processing without ThreadPoolExecutor.
"""

import time
import operator
import re
from typing import List, Dict, Any, Annotated
from typing_extensions import TypedDict
from collections import Counter
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain.chains.summarize import load_summarize_chain
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.docstore.document import Document

from src.core.config import LLMConfig, ParallelConfig
from src.services.document_service import DocumentService
from src.models.schemas import SummaryEvaluation, ImprovedSummary


# State schemas for proper Send API usage with automatic result aggregation
class UnifiedState(TypedDict):
    """Main graph state with automatic result aggregation using reducers."""
    # Input parameters
    topics: List[str]
    doc_ids: List[str]
    top_k: int
    length: str
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
    length: str
    strategy: str
    enable_reflection: bool
    contributing_docs: List[str]
    doc_count: int


def _extractive_summarization(docs: List[Document], query_context: str = ""):
    """Extract key sentences from documents using frequency-based scoring.
    
    Args:
        docs: List of documents to summarize
        query_context: Optional query to boost relevant sentences
        
    Returns:
        dict with 'summary' key containing extracted sentences
    """
    if not docs:
        return {"summary": "No content to summarize."}
    
    # Combine all document content
    combined_text = "\n".join([doc.page_content for doc in docs])
    
    # Split into sentences
    sentences = re.split(r'[.!?]+', combined_text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 20]
    
    if not sentences:
        return {"summary": "No extractable sentences found."}
    
    # Remove stop words and calculate word frequencies
    stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
    words = re.findall(r'\b[a-zA-Z]+\b', combined_text.lower())
    word_freq = Counter([word for word in words if word not in stop_words])
    
    # Score sentences based on word frequency
    sentence_scores = []
    for sentence in sentences:
        sentence_words = re.findall(r'\b[a-zA-Z]+\b', sentence.lower())
        score = sum(word_freq.get(word, 0) for word in sentence_words if word not in stop_words)
        
        # Boost score if query context matches
        if query_context and query_context.lower() in sentence.lower():
            score *= 1.5
            
        sentence_scores.append((sentence, score))
    
    # Sort by score and select top sentences
    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    top_sentences = [sentence for sentence, _ in sentence_scores[:5]]  # Top 5 sentences
    
    # Join sentences for final summary
    summary = ". ".join(top_sentences)
    if summary and not summary.endswith('.'):
        summary += "."
    
    return {"summary": summary}


def _abstractive_summarization(docs: List[Document], query_context: str = ""):
    """Generate new summary sentences using LLM abstractive capabilities.
    
    Args:
        docs: List of documents to summarize
        query_context: Optional query for focused summarization
        
    Returns:
        dict with 'summary' key containing generated summary
    """
    if not docs:
        return {"summary": "No content to summarize."}
    
    try:
        # Use existing map-reduce chain for abstractive summarization
        llm = LLMConfig.MAIN_LLM
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        
        # Add query context if provided
        if query_context:
            # Create a simple prompt-based summary for query-focused abstractive
            combined_content = "\n".join([doc.page_content for doc in docs])
            prompt = f"""
            Create a comprehensive summary of the following content with special focus on: {query_context}
            
            Content:
            {combined_content}
            
            Summary:
            """
            result = llm.invoke(prompt)
            summary_text = result.content if hasattr(result, 'content') else str(result)
        else:
            # Use standard map-reduce chain
            result = chain.run(docs)
            summary_text = result
        
        return {"summary": summary_text.strip()}
        
    except Exception as e:
        return {"summary": f"Error in abstractive summarization: {str(e)}"}


def _hybrid_summarization(docs: List[Document], query_context: str = ""):
    """Combine extractive and abstractive approaches for hybrid summarization.
    
    Args:
        docs: List of documents to summarize
        query_context: Optional query for focused summarization
        
    Returns:
        dict with 'summary' key containing hybrid summary
    """
    if not docs:
        return {"summary": "No content to summarize."}
    
    try:
        # Step 1: Extract key sentences using extractive method
        extractive_result = _extractive_summarization(docs, query_context)
        extracted_content = extractive_result.get("summary", "")
        
        if not extracted_content or extracted_content == "No extractable sentences found.":
            # Fallback to pure abstractive if extraction fails
            return _abstractive_summarization(docs, query_context)
        
        # Step 2: Use LLM to refine and paraphrase extracted content
        llm = LLMConfig.MAIN_LLM
        
        hybrid_prompt = f"""
        The following text contains key sentences extracted from documents. Please refine this into a more cohesive and fluent summary while preserving the important information.
        {f"Focus especially on aspects related to: {query_context}" if query_context else ""}
        
        Extracted content:
        {extracted_content}
        
        Refined summary:
        """
        
        result = llm.invoke(hybrid_prompt)
        refined_summary = result.content if hasattr(result, 'content') else str(result)
        
        return {"summary": refined_summary.strip()}
        
    except Exception as e:
        # Fallback to extractive if hybrid processing fails
        return _extractive_summarization(docs, query_context)


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
            topic_relevant_docs = []
            doc_sources = []
            
            # Collect relevant documents for this topic
            for doc_id in doc_ids:
                try:
                    vs = DocumentService.load_vector_store(doc_id)
                    retrieved = vs.similarity_search(topic.strip(), k=top_k)
                    if retrieved:
                        topic_relevant_docs.extend(retrieved)
                        doc_sources.append(doc_id)
                        print(f"    📄 Topic '{topic}' found {len(retrieved)} chunks in document {doc_id}")
                except Exception as e:
                    print(f"Error retrieving docs for topic '{topic}' from doc {doc_id}: {e}")
                    continue
            
            # Prepare source content for reflection
            limited_docs = topic_relevant_docs[:min(ParallelConfig.MAX_CHUNKS_PER_TOPIC, len(topic_relevant_docs))]
            source_content = "\n\n".join([doc.page_content for doc in limited_docs])
            
            if len(source_content) > ParallelConfig.MAX_SOURCE_CONTENT_LENGTH:
                source_content = source_content[:ParallelConfig.MAX_SOURCE_CONTENT_LENGTH] + "\n\n[Content truncated for reflection...]"
            
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
        topic = state.get("topic", "")
        topic_id = state.get("topic_id", 0)
        docs = state.get("docs", [])
        source_content = state.get("source_content", "")
        length = state.get("length", "medium")
        strategy = state.get("strategy", "abstractive")
        enable_reflection = state.get("enable_reflection", True)
        
        print(f"  📝 Processing topic {topic_id}: '{topic}' with {len(docs)} docs...")
        start_time = time.time()
        
        if not docs:
            result = {
                "topic": topic,
                "topic_id": topic_id,
                "summary": f"No relevant content found for topic: '{topic}'",
                "chunks_processed": 0,
                "status": "no_content",
                "processing_time": time.time() - start_time,
                "strategy": strategy,
                "reflection_applied": False
            }
            # Return single-item list - will be automatically aggregated by operator.add
            return {"topic_results": [result]}
        
        try:
            # Step 1: Generate initial summary using strategy
            if strategy == "extractive":
                summary_result = _extractive_summarization(docs, topic)
            elif strategy == "hybrid":
                summary_result = _hybrid_summarization(docs, topic)
            else:  # default to abstractive
                summary_result = _abstractive_summarization(docs, topic)
            
            initial_summary = summary_result.get("summary", "Unable to generate summary.")
            
            # Enhance with topic context for abstractive/hybrid strategies
            if strategy != "extractive" and initial_summary and initial_summary != "Unable to generate summary.":
                target = {"short": "≈3 sentences", "medium": "≈8 sentences", "long": "≈15 sentences"}[length]
                enhanced_prompt = f"""
                Create a focused summary about "{topic}" based on the following content. 
                Keep it to {target} while emphasizing information most relevant to this specific topic.
                
                Content:
                {initial_summary}
                """
                enhanced_result = LLMConfig.MAIN_LLM.invoke(enhanced_prompt)
                if hasattr(enhanced_result, 'content'):
                    initial_summary = enhanced_result.content.strip()
            
            # Step 2: Apply reflection if enabled
            final_summary = initial_summary
            reflection_metadata = {"reflection_applied": False}
            
            if enable_reflection and initial_summary and initial_summary != "Unable to generate summary.":
                print(f"    🔍 Applying reflection to topic {topic_id}: '{topic}'...")
                
                try:
                    # Create reflection input state
                    reflection_state = {
                        "summary": initial_summary,
                        "topic": topic,
                        "length": length,
                        "source_content": source_content
                    }
                    
                    # Apply reflection subgraph
                    reflection_result = apply_reflection_subgraph(reflection_state)
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
            
            result = {
                "topic": topic,
                "topic_id": topic_id,
                "summary": final_summary,
                "chunks_processed": len(docs),
                "status": "success",
                "processing_time": processing_time,
                "strategy": strategy,
                **reflection_metadata
            }
            
            # Return single-item list - will be automatically aggregated by operator.add
            return {"topic_results": [result]}
            
        except Exception as e:
            processing_time = time.time() - start_time
            print(f"  ❌ Failed topic {topic_id}: '{topic}' in {processing_time:.2f}s - {str(e)}")
            
            result = {
                "topic": topic,
                "topic_id": topic_id,
                "summary": f"Error generating summary for '{topic}': {str(e)}",
                "chunks_processed": len(docs),
                "status": "error",
                "processing_time": processing_time,
                "strategy": strategy,
                "reflection_applied": False
            }
            
            # Return single-item list - will be automatically aggregated by operator.add
            return {"topic_results": [result]}
    
    def apply_reflection_subgraph(state):
        """Apply reflection as a subgraph - evaluate and improve summary."""
        summary_text = state.get("summary", "")
        topic = state.get("topic", "")
        length_requirement = state.get("length", "medium")
        source_content = state.get("source_content", "")
        
        if not summary_text:
            return {"evaluation": None, "improved_summary": None, "error": "No summary provided"}
        
        try:
            # Step 1: Evaluate summary
            evaluation_parser = PydanticOutputParser(pydantic_object=SummaryEvaluation)
            evaluation_prompt = PromptTemplate(
                template="""You are an expert content reviewer. Evaluate the following summary based on the specified criteria.

**Topic**: {topic}
**Length Requirement**: {length_requirement} (short ≈ 3 sentences, medium ≈ 8 sentences, long ≈ 15 sentences)

**Summary to Evaluate**:
{summary_text}

**Source Content** (for factual verification):
{source_content}

Evaluate the summary on these dimensions:
1. **Factual Accuracy**: How well does the summary reflect the source content?
2. **Length Compliance**: How well does it meet the length requirement?
3. **Topic Relevance**: How well does it address the specified topic?
4. **Clarity & Readability**: Is it clear and well-written?

Be thorough in your evaluation and specific about any issues found.

{format_instructions}""",
                input_variables=["topic", "length_requirement", "summary_text", "source_content"],
                partial_variables={"format_instructions": evaluation_parser.get_format_instructions()}
            )
            
            # Get structured evaluation
            chain = evaluation_prompt | LLMConfig.REFLECTION_LLM | evaluation_parser
            evaluation = chain.invoke({
                "topic": topic or "general content",
                "length_requirement": length_requirement,
                "summary_text": summary_text,
                "source_content": source_content[:2000] if source_content else "No source content provided"
            })
            
            # Step 2: Improve if needed
            if not evaluation.improvement_needed:
                return {
                    "evaluation": evaluation,
                    "improved_summary": ImprovedSummary(
                        improved_text=summary_text,
                        changes_made=["No changes needed - summary meets quality standards"],
                        final_evaluation=evaluation
                    )
                }
            
            # Create improvement
            improvement_parser = PydanticOutputParser(pydantic_object=ImprovedSummary)
            improvement_prompt = PromptTemplate(
                template="""You are a conservative content editor. Your job is to make MINIMAL improvements to the summary while staying strictly within the bounds of the provided source content.

**CRITICAL RULES:**
1. ONLY use information that is EXPLICITLY stated in the source content
2. DO NOT add any new facts, details, or interpretations not present in the source
3. DO NOT make connections or inferences beyond what is directly stated
4. If the source content is limited, keep the summary correspondingly brief
5. When in doubt, err on the side of being conservative rather than comprehensive

**Topic**: {topic}
**Length Requirement**: {length_requirement} (short ≈ 3 sentences, medium ≈ 8 sentences, long ≈ 15 sentences)

**Original Summary**:
{summary_text}

**Issues to Address**:
{specific_issues}

**Source Content** (ONLY use what is explicitly stated here):
{source_content}

Improve the summary by addressing the specific issues while following these conservative guidelines:
- Fix factual errors by referring ONLY to the source content
- Adjust length by removing/condensing content, not by adding new information
- Improve clarity by reorganizing existing information, not by elaborating beyond the source
- If the source content is insufficient for the desired length, explain this rather than fabricating content

REMEMBER: Better to have a shorter, accurate summary than a longer one with made-up information.

{format_instructions}""",
                input_variables=["topic", "length_requirement", "summary_text", "specific_issues", "source_content"],
                partial_variables={"format_instructions": improvement_parser.get_format_instructions()}
            )
            
            # Get improved summary
            chain = improvement_prompt | LLMConfig.IMPROVEMENT_LLM | improvement_parser
            improved = chain.invoke({
                "topic": topic,
                "length_requirement": length_requirement,
                "summary_text": summary_text,
                "specific_issues": ", ".join(evaluation.specific_issues) if evaluation.specific_issues else "No specific issues identified",
                "source_content": source_content[:2000] if source_content else "No source content provided"
            })
            
            return {
                "evaluation": evaluation,
                "improved_summary": improved
            }
            
        except Exception as e:
            return {
                "evaluation": None,
                "improved_summary": None,
                "error": f"Reflection failed: {str(e)}"
            }
    
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