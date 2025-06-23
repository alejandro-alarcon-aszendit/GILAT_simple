"""Reflection system LangGraphs.

Contains graphs for evaluating and improving summaries using structured output.
"""

from langgraph.graph import Graph
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from src.core.config import LLMConfig
from src.models.schemas import SummaryEvaluation, ImprovedSummary


def build_reflection_graph():
    """Build reflection graph for evaluating and improving summaries.
    
    **Pipeline:**
    1. Evaluate summary quality using structured output
    2. Improve summary based on evaluation feedback
    
    Returns:
        Compiled LangGraph for reflection
    """
    g = Graph()
    
    def _evaluate_summary(state):
        """Evaluate a summary for quality, accuracy, and compliance."""
        summary_text = state.get("summary", "")
        topic = state.get("topic", "")
        length_requirement = state.get("length", "medium")
        source_content = state.get("source_content", "")
        
        if not summary_text:
            return {"evaluation": None, "error": "No summary provided"}
        
        print(f"🔍 Evaluating summary for topic: '{topic}'...")
        
        # Create structured output parser
        evaluation_parser = PydanticOutputParser(pydantic_object=SummaryEvaluation)
        
        # Create evaluation prompt
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
        
        try:
            # Get structured evaluation
            chain = evaluation_prompt | LLMConfig.REFLECTION_LLM | evaluation_parser
            evaluation = chain.invoke({
                "topic": topic or "general content",
                "length_requirement": length_requirement,
                "summary_text": summary_text,
                "source_content": source_content[:2000] if source_content else "No source content provided"
            })
            
            print(f"  📊 Evaluation complete - Improvement needed: {evaluation.improvement_needed}")
            print(f"      Factual accuracy: {evaluation.factual_accuracy}")
            print(f"      Length compliance: {evaluation.length_compliance}")
            print(f"      Topic relevance: {evaluation.topic_relevance}")
            
            return {"evaluation": evaluation}
            
        except Exception as e:
            print(f"  ❌ Evaluation failed: {str(e)}")
            return {"evaluation": None, "error": f"Evaluation failed: {str(e)}"}
    
    def _improve_summary(state):
        """Improve a summary based on the evaluation feedback."""
        summary_text = state.get("summary", "")
        evaluation = state.get("evaluation")
        topic = state.get("topic", "")
        length_requirement = state.get("length", "medium")
        source_content = state.get("source_content", "")
        
        if not evaluation or not evaluation.improvement_needed:
            print(f"  ✅ No improvement needed for topic: '{topic}'")
            return {
                "improved_summary": {
                    "improved_text": summary_text,
                    "changes_made": ["No changes needed - summary meets quality standards"],
                    "final_evaluation": evaluation
                }
            }
        
        print(f"🔧 Improving summary for topic: '{topic}'...")
        print(f"    Issues found: {', '.join(evaluation.specific_issues)}")
        
        # Create structured output parser for improvement
        improvement_parser = PydanticOutputParser(pydantic_object=ImprovedSummary)
        
        # Create improvement prompt
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
        
        try:
            # Get improved summary
            chain = improvement_prompt | LLMConfig.IMPROVEMENT_LLM | improvement_parser
            improved = chain.invoke({
                "topic": topic,
                "length_requirement": length_requirement,
                "summary_text": summary_text,
                "specific_issues": ", ".join(evaluation.specific_issues) if evaluation.specific_issues else "No specific issues identified",
                "source_content": source_content[:2000] if source_content else "No source content provided"
            })
            
            print(f"  ✨ Improvement complete - Changes: {len(improved.changes_made)}")
            print(f"      Final quality: {improved.final_evaluation.factual_accuracy} factual accuracy")
            
            return {"improved_summary": improved}
            
        except Exception as e:
            print(f"  ❌ Improvement failed: {str(e)}")
            return {"improved_summary": None, "error": f"Improvement failed: {str(e)}"}
    
    g.add_node("evaluate", _evaluate_summary)
    g.add_node("improve", _improve_summary)
    g.set_entry_point("evaluate")
    g.add_edge("evaluate", "improve")
    g.set_finish_point("improve")
    return g.compile()


def build_multi_topic_summary_with_reflection_graph():
    """Build enhanced multi-topic graph that includes reflection and improvement.
    
    **Pipeline:**
    1. Map topics to relevant documents (with source content preservation)
    2. Generate summaries and apply reflection in parallel
    
    Returns:
        Compiled LangGraph for multi-topic summarization with reflection
    """
    from src.graphs.summary import build_multi_topic_summary_graph
    from src.services.parallel_service import ParallelProcessingService, ParallelWorkload
    from src.core.config import ParallelConfig
    import concurrent.futures
    import time
    
    g = Graph()
    
    def _map_topics_enhanced(state):
        """Enhanced map step that preserves source content for reflection."""
        topics = state.get("topics", [])
        doc_ids = state.get("doc_ids", [])
        top_k = state.get("top_k", 10)
        
        if not topics or not doc_ids:
            return {"topic_docs": []}
        
        from src.services.document_service import DocumentService
        
        topic_docs = []
        for topic in topics:
            topic_relevant_docs = []
            doc_sources = []  # Track which docs contributed content
            
            for doc_id in doc_ids:
                try:
                    vs = DocumentService.load_vector_store(doc_id)
                    retrieved = vs.similarity_search(topic.strip(), k=top_k)
                    if retrieved:  # Only include if we found relevant content
                        topic_relevant_docs.extend(retrieved)
                        doc_sources.append(doc_id)
                        print(f"    📄 Topic '{topic}' found {len(retrieved)} chunks in document {doc_id}")
                except Exception as e:
                    print(f"Error retrieving docs for topic '{topic}' from doc {doc_id}: {e}")
                    continue
            
            # Preserve source content for reflection, but limit to avoid overwhelming the reflection
            # Take only the most relevant chunks (already sorted by similarity)
            limited_docs = topic_relevant_docs[:min(ParallelConfig.MAX_CHUNKS_PER_TOPIC, len(topic_relevant_docs))]
            source_content = "\n\n".join([doc.page_content for doc in limited_docs])
            
            # Truncate source content if too long to avoid token limits
            if len(source_content) > ParallelConfig.MAX_SOURCE_CONTENT_LENGTH:
                source_content = source_content[:ParallelConfig.MAX_SOURCE_CONTENT_LENGTH] + "\n\n[Content truncated for reflection...]"
            
            topic_docs.append({
                "topic": topic.strip(),
                "docs": topic_relevant_docs,
                "doc_count": len(topic_relevant_docs),
                "source_content": source_content,
                "contributing_docs": doc_sources,
                "limited_chunks_for_reflection": len(limited_docs)
            })
            
            print(f"  📋 Topic '{topic}' mapped to {len(topic_relevant_docs)} total chunks from {len(doc_sources)} documents")
        
        return {"topic_docs": topic_docs}
    
    def _reduce_and_reflect(state):
        """Generate summaries and apply reflection in parallel."""
        topic_docs = state.get("topic_docs", [])
        length = state.get("length", "medium")
        enable_reflection = state.get("enable_reflection", True)
        
        if not topic_docs:
            return {"summaries": []}
        
        print(f"🚀 Starting parallel processing with reflection for {len(topic_docs)} topics...")
        start_time = time.time()
        
        def process_with_reflection(topic_data):
            """Process a topic with summary generation and reflection."""
            from langchain.chains.summarize import load_summarize_chain
            
            topic = topic_data["topic"]
            docs = topic_data["docs"]
            source_content = topic_data.get("source_content", "")
            topic_start = time.time()
            
            print(f"  📝 Processing with reflection: '{topic}' with {len(docs)} docs...")
            
            if not docs:
                return {
                    "topic": topic,
                    "summary": f"No relevant content found for topic: '{topic}'",
                    "chunks_processed": 0,
                    "status": "no_content",
                    "processing_time": time.time() - topic_start,
                    "reflection_applied": False
                }
            
            try:
                # Step 1: Generate initial summary
                chain = load_summarize_chain(LLMConfig.MAIN_LLM, chain_type="map_reduce")
                result = chain.invoke({"input_documents": docs})
                
                if isinstance(result, dict) and "output_text" in result:
                    initial_summary = result["output_text"]
                elif isinstance(result, str):
                    initial_summary = result
                else:
                    initial_summary = str(result) if result else "Unable to generate summary."
                
                # Step 2: Apply reflection if enabled and we have a valid summary
                if enable_reflection and initial_summary and initial_summary != "Unable to generate summary.":
                    print(f"    🔍 Applying reflection to '{topic}'...")
                    
                    reflection_input = {
                        "summary": initial_summary,
                        "topic": topic,
                        "length": length,
                        "source_content": source_content
                    }
                    
                    reflection_graph = build_reflection_graph()
                    reflection_result = reflection_graph.invoke(reflection_input)
                    improved_summary = reflection_result.get("improved_summary")
                    
                    if improved_summary and improved_summary.improved_text:
                        final_summary = improved_summary.improved_text
                        reflection_metadata = {
                            "reflection_applied": True,
                            "changes_made": improved_summary.changes_made,
                            "initial_evaluation": reflection_result.get("evaluation").__dict__ if reflection_result.get("evaluation") else None,
                            "final_evaluation": improved_summary.final_evaluation.__dict__
                        }
                        print(f"    ✨ Reflection complete for '{topic}' - Improved: {len(improved_summary.changes_made) > 1}")
                    else:
                        final_summary = initial_summary
                        reflection_metadata = {
                            "reflection_applied": False,
                            "error": reflection_result.get("error", "Unknown reflection error")
                        }
                        print(f"    ⚠️ Reflection failed for '{topic}', using original summary")
                else:
                    final_summary = initial_summary
                    reflection_metadata = {"reflection_applied": False, "reason": "Reflection disabled or invalid summary"}
                
                total_time = time.time() - topic_start
                print(f"  ✅ Completed '{topic}' with reflection in {total_time:.2f}s")
                
                return {
                    "topic": topic,
                    "summary": final_summary,
                    "chunks_processed": len(docs),
                    "status": "success",
                    "processing_time": total_time,
                    **reflection_metadata
                }
                
            except Exception as e:
                total_time = time.time() - topic_start
                print(f"  ❌ Failed '{topic}' in {total_time:.2f}s - {str(e)}")
                return {
                    "topic": topic,
                    "summary": f"Error generating summary for '{topic}': {str(e)}",
                    "chunks_processed": len(docs),
                    "status": "error",
                    "processing_time": total_time,
                    "reflection_applied": False
                }
        
        # Process all topics in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(topic_docs), ParallelConfig.MAX_REFLECTION_WORKERS)) as executor:
            futures = [executor.submit(process_with_reflection, topic_data) for topic_data in topic_docs]
            summaries = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        total_time = time.time() - start_time
        print(f"🎉 Parallel processing with reflection completed in {total_time:.2f}s")
        
        # Calculate reflection statistics
        reflection_applied_count = sum(1 for s in summaries if s.get("reflection_applied", False))
        
        result = {"summaries": summaries}
        result["parallel_processing"] = {
            "total_time": total_time,
            "topics_count": len(topic_docs),
            "average_time_per_topic": total_time / len(topic_docs) if topic_docs else 0,
            "method": "ThreadPoolExecutor_with_Reflection",
            "reflection_statistics": {
                "total_topics": len(summaries),
                "reflection_applied": reflection_applied_count,
                "reflection_skipped": len(summaries) - reflection_applied_count
            }
        }
        
        return result
    
    g.add_node("map_topics_enhanced", _map_topics_enhanced)
    g.add_node("reduce_and_reflect", _reduce_and_reflect)
    g.set_entry_point("map_topics_enhanced")
    g.add_edge("map_topics_enhanced", "reduce_and_reflect")
    g.set_finish_point("reduce_and_reflect")
    return g.compile()


# Create compiled graph instances
REFLECTION_GRAPH = build_reflection_graph()
MULTI_TOPIC_SUMMARY_WITH_REFLECTION_GRAPH = build_multi_topic_summary_with_reflection_graph() 