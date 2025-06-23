"""Reflection utilities for summary quality improvement.

This module handles the evaluation and improvement of summaries using
structured LLM feedback and conservative editing approaches.
"""

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate

from src.core.config import LLMConfig
from src.models.schemas import SummaryEvaluation, ImprovedSummary


def apply_reflection_to_summary(summary_text: str, topic: str, length_requirement: str, source_content: str) -> dict:
    """Apply reflection to improve summary quality.
    
    This function evaluates a summary and potentially improves it using a two-step process:
    1. Evaluate the summary against quality criteria
    2. Generate improvements if needed (conservative approach)
    
    Args:
        summary_text: Original summary to evaluate and improve
        topic: Topic the summary should address
        length_requirement: Length target (short/medium/long)
        source_content: Original source content for factual verification
        
    Returns:
        dict with evaluation, improved_summary, and optional error keys
    """
    if not summary_text:
        return {"evaluation": None, "improved_summary": None, "error": "No summary provided"}
    
    try:
        # Step 1: Evaluate summary
        evaluation = _evaluate_summary(summary_text, topic, length_requirement, source_content)
        
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
        improved_summary = _improve_summary(summary_text, topic, length_requirement, source_content, evaluation)
        
        return {
            "evaluation": evaluation,
            "improved_summary": improved_summary
        }
        
    except Exception as e:
        return {
            "evaluation": None,
            "improved_summary": None,
            "error": f"Reflection failed: {str(e)}"
        }


def _evaluate_summary(summary_text: str, topic: str, length_requirement: str, source_content: str) -> SummaryEvaluation:
    """Evaluate summary quality against multiple criteria.
    
    Args:
        summary_text: Summary to evaluate
        topic: Expected topic
        length_requirement: Expected length
        source_content: Source content for verification
        
    Returns:
        SummaryEvaluation object with scores and feedback
    """
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
    
    return evaluation


def _improve_summary(summary_text: str, topic: str, length_requirement: str, source_content: str, evaluation: SummaryEvaluation) -> ImprovedSummary:
    """Generate improved version of summary using conservative editing.
    
    Args:
        summary_text: Original summary
        topic: Topic context
        length_requirement: Length target
        source_content: Source for fact-checking
        evaluation: Evaluation results with specific issues
        
    Returns:
        ImprovedSummary object with refined text and change notes
    """
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
    
    return improved 