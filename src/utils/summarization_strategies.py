"""Summarization strategy implementations.

This module contains the three core summarization strategies:
- Extractive: Key sentence extraction using frequency scoring
- Abstractive: LLM-generated paraphrased summaries  
- Hybrid: Combination of extractive + abstractive refinement
"""

import re
from typing import List
from collections import Counter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain

from src.core.config import LLMConfig


def extractive_summarization(docs: List[Document], query_context: str = "") -> dict:
    """Extract key quotes and sentences using LLM-based identification.
    
    This strategy uses an LLM to intelligently identify and extract the most
    important quotes and key sentences directly from the source documents,
    preserving the original language while ensuring coherence.
    
    Args:
        docs: List of documents to summarize
        query_context: Optional query to focus extraction
        
    Returns:
        dict with 'summary' key containing extracted key quotes
    """
    if not docs:
        return {"summary": "No content to summarize."}
    
    try:
        # Combine all document content
        combined_text = "\n".join([doc.page_content for doc in docs])
        
        # Truncate if too long for processing
        max_length = 8000  # Reasonable limit for LLM processing
        if len(combined_text) > max_length:
            combined_text = combined_text[:max_length] + "\n\n[Content truncated...]"
        
        # Create focused extraction prompt
        focus_instruction = f"with special focus on: {query_context}" if query_context else ""
        
        extraction_prompt = f"""
You are an expert at extracting key quotes and important sentences from documents. Your task is to identify and extract the most significant quotes, statements, and key sentences that capture the essential information {focus_instruction}.

INSTRUCTIONS:
1. Extract 5-8 key quotes or sentences that represent the most important information
2. Preserve the EXACT original wording - do not paraphrase or modify
3. Select quotes that are complete, coherent, and meaningful on their own
4. Prioritize quotes that contain specific facts, key insights, or important statements
5. Arrange the extracted quotes in a logical flow
6. Each quote should be a complete sentence or meaningful phrase
7. Focus on substantive content, avoid filler or transitional text

CONTENT TO ANALYZE:
{combined_text}

EXTRACTED KEY QUOTES:
Please provide the key quotes below, each on a separate line, maintaining their exact original wording:
"""
        
        # Use LLM to extract key quotes
        llm = LLMConfig.MAIN_LLM
        result = llm.invoke(extraction_prompt)
        extracted_text = result.content if hasattr(result, 'content') else str(result)
        
        # Clean up the extracted text
        extracted_text = extracted_text.strip()
        
        # Remove any instruction echoing or formatting
        lines = extracted_text.split('\n')
        clean_lines = []
        for line in lines:
            line = line.strip()
            # Skip empty lines, instruction echoing, or formatting
            if (line and 
                not line.startswith('EXTRACTED') and 
                not line.startswith('Please provide') and
                not line.startswith('Key quotes:') and
                not line.startswith('**') and
                len(line) > 10):  # Ensure meaningful content
                # Remove bullet points or numbering
                line = re.sub(r'^[\d\-\*\•\◦]+\.?\s*', '', line)
                line = re.sub(r'^[\"\'""]', '', line)  # Remove leading quotes
                line = re.sub(r'[\"\'""]$', '', line)  # Remove trailing quotes
                clean_lines.append(line.strip())
        
        if clean_lines:
            # Join with appropriate punctuation
            summary = ". ".join(clean_lines)
            # Ensure proper ending punctuation
            if summary and not summary.endswith(('.', '!', '?')):
                summary += "."
            return {"summary": summary}
        else:
            return {"summary": "Unable to extract meaningful quotes from the content."}
        
    except Exception as e:
        print(f"Error in LLM-based extractive summarization: {str(e)}")
        return {"summary": f"Error in extractive summarization: {str(e)}"}





def abstractive_summarization(docs: List[Document], query_context: str = "") -> dict:
    """Generate new summary sentences using LLM abstractive capabilities.
    
    This strategy uses AI to generate new sentences that paraphrase and 
    synthesize the source content, incorporating key insights and quotes
    in a coherent narrative format.
    
    Args:
        docs: List of documents to summarize
        query_context: Optional query for focused summarization
        
    Returns:
        dict with 'summary' key containing generated summary
    """
    if not docs:
        return {"summary": "No content to summarize."}
    
    try:
        llm = LLMConfig.MAIN_LLM
        
        # Combine content for processing
        combined_content = "\n".join([doc.page_content for doc in docs])
        
        # Truncate if too long
        max_length = 8000
        if len(combined_content) > max_length:
            combined_content = combined_content[:max_length] + "\n\n[Content truncated...]"
        
        # Create focused abstractive prompt
        focus_instruction = f"Pay special attention to information related to: {query_context}" if query_context else ""
        
        abstractive_prompt = f"""
You are an expert summarizer. Your task is to create a comprehensive, well-structured summary that synthesizes the key information from the provided content. {focus_instruction}

INSTRUCTIONS:
1. Create a coherent narrative summary in your own words
2. Include the most important facts, concepts, and insights
3. When appropriate, incorporate impactful direct quotes or key phrases from the original text
4. Organize information logically with smooth transitions
5. Maintain accuracy while making the content accessible
6. Aim for 6-10 sentences that capture the essential information
7. Balance paraphrasing with selective use of original phrasing for key concepts

CONTENT TO SUMMARIZE:
{combined_content}

SYNTHESIZED SUMMARY:
"""
        
        result = llm.invoke(abstractive_prompt)
        summary_text = result.content if hasattr(result, 'content') else str(result)
        
        return {"summary": summary_text.strip()}
        
    except Exception as e:
        print(f"Error in enhanced abstractive summarization: {str(e)}")
        # Fallback to basic map-reduce chain
        try:
            chain = load_summarize_chain(LLMConfig.MAIN_LLM, chain_type="map_reduce")
            result = chain.run(docs)
            return {"summary": result.strip()}
        except Exception as fallback_error:
            return {"summary": f"Error in abstractive summarization: {str(e)}"}


def hybrid_summarization(docs: List[Document], query_context: str = "") -> dict:
    """Combine extractive and abstractive approaches for hybrid summarization.
    
    This strategy first extracts key quotes and insights, then uses AI to 
    weave them into a coherent narrative that combines direct quotes with
    synthesized explanations.
    
    Args:
        docs: List of documents to summarize
        query_context: Optional query for focused summarization
        
    Returns:
        dict with 'summary' key containing hybrid summary
    """
    if not docs:
        return {"summary": "No content to summarize."}
    
    try:
        # Step 1: Extract key quotes using improved extractive method
        extractive_result = extractive_summarization(docs, query_context)
        extracted_quotes = extractive_result.get("summary", "")
        
        if not extracted_quotes or extracted_quotes in ["No extractable sentences found.", "No content to summarize."]:
            # Fallback to pure abstractive if extraction fails
            return abstractive_summarization(docs, query_context)
        
        # Step 2: Get original content for additional context
        combined_content = "\n".join([doc.page_content for doc in docs])
        max_context_length = 4000  # Limit context for the hybrid processing
        if len(combined_content) > max_context_length:
            combined_content = combined_content[:max_context_length] + "\n\n[Additional content available...]"
        
        # Step 3: Use LLM to create hybrid summary combining quotes with synthesis
        llm = LLMConfig.MAIN_LLM
        
        focus_instruction = f"with particular emphasis on: {query_context}" if query_context else ""
        
        hybrid_prompt = f"""
You are creating a hybrid summary that combines direct quotes with synthesized explanations. You have been provided with key quotes extracted from documents, and you need to create a comprehensive summary that weaves these quotes together with your own explanatory text.

TASK: Create a coherent summary {focus_instruction} that:
1. Incorporates the extracted key quotes naturally into the narrative
2. Adds explanatory context and connections between the quotes
3. Synthesizes additional insights from the source material
4. Maintains a balance between direct quotes and paraphrased content
5. Creates smooth transitions and logical flow
6. Ensures the final summary is comprehensive yet concise

EXTRACTED KEY QUOTES:
{extracted_quotes}

ADDITIONAL SOURCE CONTEXT:
{combined_content}

HYBRID SUMMARY:
Create a well-structured summary that thoughtfully combines these quotes with explanatory synthesis:
"""
        
        result = llm.invoke(hybrid_prompt)
        hybrid_summary = result.content if hasattr(result, 'content') else str(result)
        
        return {"summary": hybrid_summary.strip()}
        
    except Exception as e:
        print(f"Error in hybrid summarization: {str(e)}")
        # Fallback to extractive if hybrid processing fails
        return extractive_summarization(docs, query_context)


def get_strategy_function(strategy: str):
    """Get the appropriate strategy function based on name.
    
    Args:
        strategy: Name of strategy ('extractive', 'abstractive', 'hybrid')
        
    Returns:
        Strategy function
    """
    strategy_map = {
        "extractive": extractive_summarization,
        "abstractive": abstractive_summarization,
        "hybrid": hybrid_summarization
    }
    
    return strategy_map.get(strategy, abstractive_summarization)  # Default to abstractive 