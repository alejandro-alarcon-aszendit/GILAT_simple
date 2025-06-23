"""LangGraph Document Service (v1.0)
=================================
*Simplified* version: **metadata in SQLModel + async FastAPI**
+ **Synchronous ingestion** for immediate processing.

Highlights
----------
1. **PostgreSQL / SQLite** stores document metadata (id, name, status, n_chunks).
2. **Synchronous processing** parses, splits, embeds chunks and persists Chroma vectors.
3. Endpoints enforce status (`ready`) before summary / ask.
4. Chunks are persisted as `chunks.json` beside the Chroma directory to rebuild
   `langchain.Document` objects.

Install / run (local dev)
------------------------
```bash
pip install fastapi "uvicorn[standard]" langchain langchain-openai \
    langgraph docling chromadb tiktoken numpy sqlmodel

# run API
uvicorn agent:app --reload
```
Environment
-----------
* `OPENAI_API_KEY`
* `DATABASE_URL`  (default `sqlite:///./app.db`)
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langgraph.graph import Graph
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field as PydanticField
from sqlmodel import Field, Session, SQLModel, create_engine, select
from typing import Optional, Literal

# Document processing
from docling.document_converter import DocumentConverter

# -------------------- Config ----------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")
BASE_DIR = Path("vector_db")
BASE_DIR.mkdir(exist_ok=True)
CHUNK_FILE = "chunks.json"

LLM = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)
REFLECTION_LLM = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)  # Lower temp for more consistent evaluation
IMPROVEMENT_LLM = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)  # Slightly higher temp for creative improvements
EMBEDDER = OpenAIEmbeddings(model="text-embedding-ada-002")
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

# -------------------- Structured Output Models for Reflection -----
class SummaryEvaluation(BaseModel):
    """Structured evaluation of a summary's quality and accuracy."""
    factual_accuracy: Literal["excellent", "good", "fair", "poor"] = PydanticField(
        description="Assessment of factual accuracy based on source content"
    )
    length_compliance: Literal["perfect", "slightly_over", "slightly_under", "significantly_off"] = PydanticField(
        description="How well the summary meets the specified length requirement"
    )
    topic_relevance: Literal["highly_relevant", "mostly_relevant", "somewhat_relevant", "off_topic"] = PydanticField(
        description="How well the summary addresses the specified topic"
    )
    clarity_readability: Literal["excellent", "good", "fair", "poor"] = PydanticField(
        description="Overall clarity and readability of the summary"
    )
    improvement_needed: bool = PydanticField(
        description="Whether the summary needs improvement"
    )
    specific_issues: List[str] = PydanticField(
        description="List of specific issues found (empty if none)",
        default=[]
    )
    confidence_score: float = PydanticField(
        description="Confidence in the evaluation (0.0 to 1.0)",
        ge=0.0,
        le=1.0
    )

class ImprovedSummary(BaseModel):
    """Improved version of a summary with reflection metadata."""
    improved_text: str = PydanticField(
        description="The improved version of the summary"
    )
    changes_made: List[str] = PydanticField(
        description="List of specific changes made during improvement"
    )
    final_evaluation: SummaryEvaluation = PydanticField(
        description="Final evaluation of the improved summary"
    )

# -------------------- DB models -------------------------------------
class Doc(SQLModel, table=True):
    id: str = Field(primary_key=True)
    name: str
    status: str = "ready"  # ready | failed
    n_chunks: int | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)


engine = create_engine(DATABASE_URL, echo=False)
SQLModel.metadata.create_all(engine)

# helper
def db_session() -> Session:
    return Session(engine)

# -------------------- LangGraph mini‑pipelines ----------------------

def build_ingest_graph():
    g = Graph()

    def _parse(file_path: str):
        """Parse document content based on file type"""
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()
        
        try:
            if file_extension in ['.txt']:
                # Simple text reading for plain text files
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            elif file_extension in ['.pdf', '.md', '.docx', '.pptx', '.html']:
                # Use docling for supported document formats
                converter = DocumentConverter()
                result = converter.convert(file_path)
                
                # Check if conversion was successful
                if result is None or result.document is None:
                    raise ValueError("Document conversion returned None")
                
                # Try to extract text content
                try:
                    text = result.document.export_to_text()
                except:
                    # Fallback to markdown export
                    try:
                        text = result.document.export_to_markdown()
                    except:
                        raise ValueError("Failed to extract text from document")
                
                # Ensure we got some text
                if not text or not text.strip():
                    raise ValueError("No text content extracted from document")
                    
            else:
                # Try to read as text for other file types
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                except UnicodeDecodeError:
                    # If UTF-8 fails, try with error handling
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
            
            # Final check to ensure we have content
            if not text or not text.strip():
                raise ValueError("No text content found in document")
            
            return {"text": text}
        except Exception as e:
            raise ValueError(f"Failed to parse document '{file_path.name}': {str(e)}")

    def _split(state):
        docs = [Document(page_content=c) for c in SPLITTER.split_text(state["text"])]
        return {"docs": docs}

    g.add_node("parse", _parse)
    g.add_node("split", _split)
    g.set_entry_point("parse")
    g.add_edge("parse", "split")
    g.set_finish_point("split")
    return g.compile()

INGEST_GRAPH = build_ingest_graph()

# -------------------- Synchronous ingestion function ---------------
def ingest_document(doc_id: str, tmp_path: str, filename: str):
    """Synchronous ingestion: parse → split → embed → persist → update DB."""
    try:
        # 1. Process document using LangGraph
        result = INGEST_GRAPH.invoke(tmp_path)
        docs = result["docs"]

        # 2. embed & persist
        vs_dir = BASE_DIR / doc_id
        vs_dir.mkdir(exist_ok=True)
        vs = Chroma.from_documents(docs, EMBEDDER, persist_directory=str(vs_dir))

        # 3. save chunk texts
        with open(vs_dir / CHUNK_FILE, "w", encoding="utf-8") as f:
            json.dump([d.page_content for d in docs], f)

        # 4. update DB
        with db_session() as s:
            doc = s.get(Doc, doc_id)
            if doc:
                doc.status = "ready"
                doc.n_chunks = len(docs)
                s.add(doc)
                s.commit()
                
        return len(docs)
                
    except Exception as exc:
        # Handle errors
        with db_session() as s:
            doc = s.get(Doc, doc_id)
            if doc:
                doc.status = "failed"
                s.add(doc)
                s.commit()
        raise exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

# -------------------- Helpers ---------------------------------------

def load_vectorstore(doc_id: str) -> Chroma:
    vs_dir = BASE_DIR / doc_id
    if not vs_dir.exists():
        raise FileNotFoundError
    return Chroma(persist_directory=str(vs_dir), embedding_function=EMBEDDER)


def load_chunks(doc_id: str) -> List[Document]:
    file_path = BASE_DIR / doc_id / CHUNK_FILE
    if not file_path.exists():
        raise FileNotFoundError
    texts = json.loads(file_path.read_text(encoding="utf-8"))
    return [Document(page_content=t) for t in texts]


# -------------------- Summary graph ---------------------------------

def build_summary_graph():
    g = Graph()

    def _summarise(state):
        docs = state.get("docs", [])
        query_context = state.get("query_context", "")
        
        if not docs:
            return {"summary": "No content to summarize."}
        
        try:
            # For large document sets, use map-reduce approach
            chain = load_summarize_chain(LLM, chain_type="map_reduce")
            result = chain.invoke({"input_documents": docs})
            
            # Extract the summary text from the result
            if isinstance(result, dict) and "output_text" in result:
                summary_text = result["output_text"]
            elif isinstance(result, str):
                summary_text = result
            else:
                summary_text = str(result) if result else "Unable to generate summary."
            
            # If we have query context, enhance the summary with it
            if query_context and summary_text and summary_text != "Unable to generate summary.":
                enhanced_prompt = f"""
                Based on the following summary, provide a refined version that emphasizes aspects related to: {query_context}
                
                Original summary:
                {summary_text}
                
                Please ensure the refined summary maintains accuracy while highlighting relevant information about the specified topic.
                """
                enhanced_result = LLM.invoke(enhanced_prompt)
                if hasattr(enhanced_result, 'content'):
                    summary_text = enhanced_result.content.strip()
                
            return {"summary": summary_text}
        except Exception as e:
            return {"summary": f"Error generating summary: {str(e)}"}

    g.add_node("summarise", _summarise)
    g.set_entry_point("summarise")
    g.set_finish_point("summarise")
    return g.compile()

def build_multi_topic_summary_graph():
    """LangGraph for processing multiple topics in parallel using map-reduce pattern"""
    g = Graph()
    
    def _map_topics(state):
        """Map step: For each topic, find relevant documents and prepare for summarization"""
        topics = state.get("topics", [])
        doc_ids = state.get("doc_ids", [])
        top_k = state.get("top_k", 10)
        
        if not topics or not doc_ids:
            return {"topic_docs": []}
        
        topic_docs = []
        for topic in topics:
            topic_relevant_docs = []
            for doc_id in doc_ids:
                try:
                    vs = load_vectorstore(doc_id)
                    retrieved = vs.similarity_search(topic.strip(), k=top_k)
                    topic_relevant_docs.extend(retrieved)
                except Exception as e:
                    print(f"Error retrieving docs for topic '{topic}' from doc {doc_id}: {e}")
                    continue
            
            topic_docs.append({
                "topic": topic.strip(),
                "docs": topic_relevant_docs,
                "doc_count": len(topic_relevant_docs)
            })
        
        return {"topic_docs": topic_docs}
    

    
    def _reduce_summaries(state):
        """Reduce step: Generate summary for each topic in TRUE parallel using ThreadPoolExecutor"""
        import concurrent.futures
        
        topic_docs = state.get("topic_docs", [])
        length = state.get("length", "medium")
        
        if not topic_docs:
            return {"summaries": []}
        
        print(f"🚀 Starting parallel processing of {len(topic_docs)} topics...")
        start_time = time.time()
        
        def process_single_topic(topic_data):
            """Process a single topic synchronously"""
            topic = topic_data["topic"]
            docs = topic_data["docs"]
            topic_start = time.time()
            
            print(f"  📝 Processing topic: '{topic}' with {len(docs)} docs...")
            
            if not docs:
                return {
                    "topic": topic,
                    "summary": f"No relevant content found for topic: '{topic}'",
                    "chunks_processed": 0,
                    "status": "no_content",
                    "processing_time": time.time() - topic_start
                }
            
            try:
                # Use the existing summary chain for each topic
                chain = load_summarize_chain(LLM, chain_type="map_reduce")
                
                chain_start = time.time()
                result = chain.invoke({"input_documents": docs})
                chain_time = time.time() - chain_start
                print(f"    ⚡ Chain processing for '{topic}' took {chain_time:.2f}s")
                
                # Extract summary text
                if isinstance(result, dict) and "output_text" in result:
                    summary_text = result["output_text"]
                elif isinstance(result, str):
                    summary_text = result
                else:
                    summary_text = str(result) if result else "Unable to generate summary."
                
                # Enhance with topic context
                if summary_text and summary_text != "Unable to generate summary.":
                    target = {"short": "≈3 sentences", "medium": "≈8 sentences", "long": "≈15 sentences"}[length]
                    enhanced_prompt = f"""
                    Create a focused summary about "{topic}" based on the following content. 
                    Keep it to {target} while emphasizing information most relevant to this specific topic.
                    
                    Content:
                    {summary_text}
                    """
                    enhance_start = time.time()
                    enhanced_result = LLM.invoke(enhanced_prompt)
                    enhance_time = time.time() - enhance_start
                    print(f"    ✨ Enhancement for '{topic}' took {enhance_time:.2f}s")
                    
                    if hasattr(enhanced_result, 'content'):
                        summary_text = enhanced_result.content.strip()
                
                total_time = time.time() - topic_start
                print(f"  ✅ Completed topic: '{topic}' in {total_time:.2f}s")
                
                return {
                    "topic": topic,
                    "summary": summary_text,
                    "chunks_processed": len(docs),
                    "status": "success",
                    "processing_time": total_time
                }
                
            except Exception as e:
                total_time = time.time() - topic_start
                print(f"  ❌ Failed topic: '{topic}' in {total_time:.2f}s - {str(e)}")
                return {
                    "topic": topic,
                    "summary": f"Error generating summary for '{topic}': {str(e)}",
                    "chunks_processed": len(docs),
                    "status": "error",
                    "processing_time": total_time
                }
        
        # Process all topics in parallel using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(topic_docs), 5)) as executor:
            # Submit all tasks
            futures = [executor.submit(process_single_topic, topic_data) for topic_data in topic_docs]
            
            # Collect results as they complete
            summaries = []
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    summaries.append(result)
                except Exception as e:
                    print(f"  ❌ Task failed with exception: {str(e)}")
                    summaries.append({
                        "topic": "unknown",
                        "summary": f"Task failed: {str(e)}",
                        "chunks_processed": 0,
                        "status": "error",
                        "processing_time": 0
                    })
        
        total_time = time.time() - start_time
        print(f"🎉 Parallel processing completed in {total_time:.2f}s for {len(topic_docs)} topics")
        
        # Add timing metadata to the result
        result = {"summaries": summaries}
        result["parallel_processing"] = {
            "total_time": total_time,
            "topics_count": len(topic_docs),
            "average_time_per_topic": total_time / len(topic_docs) if topic_docs else 0,
            "method": "ThreadPoolExecutor"
        }
        
        return result
    
    g.add_node("map_topics", _map_topics)
    g.add_node("reduce_summaries", _reduce_summaries)
    g.set_entry_point("map_topics")
    g.add_edge("map_topics", "reduce_summaries")
    g.set_finish_point("reduce_summaries")
    return g.compile()

# -------------------- Reflection System -----------------------------

def build_reflection_graph():
    """LangGraph for evaluating and improving summaries with structured output"""
    g = Graph()
    
    def _evaluate_summary(state):
        """Evaluate a summary for quality, accuracy, and compliance"""
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
            chain = evaluation_prompt | REFLECTION_LLM | evaluation_parser
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
        """Improve a summary based on the evaluation feedback"""
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
            template="""You are an expert content editor. Improve the following summary based on the evaluation feedback.

**FOCUS**: Improve this summary about "{topic}" ONLY. Do not try to connect it to other unrelated topics that may appear in the source content.

**Topic**: {topic}
**Length Requirement**: {length_requirement} (short ≈ 3 sentences, medium ≈ 8 sentences, long ≈ 15 sentences)

**Original Summary**:
{summary_text}

**Evaluation Feedback**:
- Factual Accuracy: {factual_accuracy}
- Length Compliance: {length_compliance}
- Topic Relevance: {topic_relevance}
- Clarity & Readability: {clarity_readability}
- Specific Issues: {specific_issues}

**Source Content** (for reference - only use content relevant to "{topic}"):
{source_content}

Please provide an improved version that addresses all the identified issues. Your improved summary should:
1. Be factually accurate to the source content that relates to "{topic}"
2. Meet the specified length requirement
3. Stay focused ONLY on the topic "{topic}" - do not mention unrelated topics
4. Be clear and well-written

IMPORTANT: Keep the summary focused solely on "{topic}". Do not suggest connections to other topics.

{format_instructions}""",
            input_variables=["topic", "length_requirement", "summary_text", "factual_accuracy", 
                           "length_compliance", "topic_relevance", "clarity_readability", 
                           "specific_issues", "source_content"],
            partial_variables={"format_instructions": improvement_parser.get_format_instructions()}
        )
        
        try:
            # Get improved summary
            chain = improvement_prompt | IMPROVEMENT_LLM | improvement_parser
            improved = chain.invoke({
                "topic": topic or "general content",
                "length_requirement": length_requirement,
                "summary_text": summary_text,
                "factual_accuracy": evaluation.factual_accuracy,
                "length_compliance": evaluation.length_compliance,
                "topic_relevance": evaluation.topic_relevance,
                "clarity_readability": evaluation.clarity_readability,
                "specific_issues": ", ".join(evaluation.specific_issues) if evaluation.specific_issues else "None",
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

SUMMARY_GRAPH = build_summary_graph()
MULTI_TOPIC_SUMMARY_GRAPH = build_multi_topic_summary_graph()
REFLECTION_GRAPH = build_reflection_graph()

# -------------------- Enhanced Multi-Topic Summary with Reflection ---

def build_multi_topic_summary_with_reflection_graph():
    """Enhanced multi-topic graph that includes reflection and improvement"""
    g = Graph()
    
    def _map_topics_enhanced(state):
        """Enhanced map step that preserves source content for reflection"""
        topics = state.get("topics", [])
        doc_ids = state.get("doc_ids", [])
        top_k = state.get("top_k", 10)
        
        if not topics or not doc_ids:
            return {"topic_docs": []}
        
        topic_docs = []
        for topic in topics:
            topic_relevant_docs = []
            doc_sources = []  # Track which docs contributed content
            
            for doc_id in doc_ids:
                try:
                    vs = load_vectorstore(doc_id)
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
            limited_docs = topic_relevant_docs[:min(20, len(topic_relevant_docs))]
            source_content = "\n\n".join([doc.page_content for doc in limited_docs])
            
            # Truncate source content if too long to avoid token limits
            if len(source_content) > 4000:
                source_content = source_content[:4000] + "\n\n[Content truncated for reflection...]"
            
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
        """Generate summaries and apply reflection in parallel"""
        import concurrent.futures
        
        topic_docs = state.get("topic_docs", [])
        length = state.get("length", "medium")
        enable_reflection = state.get("enable_reflection", True)
        
        if not topic_docs:
            return {"summaries": []}
        
        print(f"🚀 Starting parallel processing with reflection for {len(topic_docs)} topics...")
        start_time = time.time()
        
        def process_with_reflection(topic_data):
            """Process a topic with summary generation and reflection"""
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
                chain = load_summarize_chain(LLM, chain_type="map_reduce")
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
                    
                    reflection_result = REFLECTION_GRAPH.invoke(reflection_input)
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
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(topic_docs), 3)) as executor:
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

MULTI_TOPIC_SUMMARY_WITH_REFLECTION_GRAPH = build_multi_topic_summary_with_reflection_graph()

# -------------------- FastAPI ---------------------------------------
app = FastAPI(title="LangGraph Doc Service", version="1.0")

# -------- Upload (synchronous) --------------------------------------
@app.post("/documents", status_code=status.HTTP_201_CREATED)
async def upload_document(file: UploadFile = File(...)):
    # 1. create tmp copy
    with NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    doc_id = str(uuid.uuid4())
    
    # 2. insert DB row
    with db_session() as s:
        s.add(Doc(id=doc_id, name=file.filename, status="ready"))
        s.commit()

    try:
        # 3. process document immediately
        n_chunks = ingest_document(doc_id, tmp_path, file.filename)
        return {"doc_id": doc_id, "status": "ready", "n_chunks": n_chunks}
    except Exception as e:
        # If processing fails, update status and raise error
        with db_session() as s:
            doc = s.get(Doc, doc_id)
            if doc:
                doc.status = "failed"
                s.add(doc)
                s.commit()
        raise HTTPException(500, f"Document processing failed: {str(e)}")

# -------- Document list / detail -----------------------------------
class DocOut(BaseModel):
    id: str
    name: str
    status: str
    n_chunks: int | None
    created_at: datetime

@app.get("/documents", response_model=list[DocOut])
async def list_documents():
    with db_session() as s:
        docs = s.exec(select(Doc).order_by(Doc.created_at.desc())).all()
        return docs

@app.get("/documents/{doc_id}", response_model=DocOut)
async def get_document(doc_id: str):
    with db_session() as s:
        doc = s.get(Doc, doc_id)
        if not doc:
            raise HTTPException(404, "Document not found")
        return doc

# -------- Delete ----------------------------------------------------
@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    with db_session() as s:
        doc = s.get(Doc, doc_id)
        if not doc:
            raise HTTPException(404, "Document not found")
        s.delete(doc)
        s.commit()
    # remove on‑disk data
    vs_dir = BASE_DIR / doc_id
    if vs_dir.exists():
        shutil.rmtree(vs_dir, ignore_errors=True)
    return {"status": "deleted", "doc_id": doc_id}

# -------- Summary (multiple docs) ----------------------------------
@app.get("/summary")
async def multi_summary(
    doc_id: List[str] = Query(...),
    length: str = Query("medium", enum=["short", "medium", "long"]),
    query: str = Query(None, description="Optional query/topic(s) to focus the summary on. Use commas to separate multiple topics for parallel processing."),
    top_k: int = Query(10, description="Number of most relevant chunks to include when using query-focused summarization"),
    enable_reflection: bool = Query(True, description="Enable AI reflection to review and improve summary quality, accuracy, and length compliance"),
):
    with db_session() as s:
        docs_meta = s.exec(select(Doc).where(Doc.id.in_(doc_id))).all()
    not_ready = [d.id for d in docs_meta if d.status != "ready"]
    if not_ready:
        raise HTTPException(409, f"Documents not ready: {not_ready}")

    # If query is provided, check if it contains multiple topics (comma-separated)
    if query and query.strip():
        topics = [topic.strip() for topic in query.split(",") if topic.strip()]
        
        # If multiple topics, use multi-topic graph (with or without reflection)
        if len(topics) > 1:
            multi_topic_input = {
                "topics": topics,
                "doc_ids": doc_id,
                "top_k": top_k,
                "length": length,
                "enable_reflection": enable_reflection
            }
            
            # Choose graph based on reflection setting
            if enable_reflection:
                result_state = MULTI_TOPIC_SUMMARY_WITH_REFLECTION_GRAPH.invoke(multi_topic_input)
            else:
                result_state = MULTI_TOPIC_SUMMARY_GRAPH.invoke(multi_topic_input)
            summaries = result_state.get("summaries", [])
            parallel_metadata = result_state.get("parallel_processing", {})
            
            if not summaries:
                return {
                    "type": "multi_topic",
                    "summaries": [],
                    "message": "No summaries could be generated for the provided topics.",
                    "documents": doc_id,
                    "topics": topics,
                    "parallel_processing": parallel_metadata
                }
            
            total_chunks = sum(s.get("chunks_processed", 0) for s in summaries)
            successful_summaries = [s for s in summaries if s.get("status") == "success"]
            
            # Calculate performance metrics
            individual_times = [s.get("processing_time", 0) for s in summaries if s.get("processing_time")]
            max_individual_time = max(individual_times) if individual_times else 0
            total_sequential_time = sum(individual_times) if individual_times else 0
            parallel_speedup = total_sequential_time / parallel_metadata.get("total_time", 1) if parallel_metadata.get("total_time") else 1
            
            # Calculate reflection statistics if available
            reflection_stats = parallel_metadata.get("reflection_statistics", {})
            reflection_applied_count = sum(1 for s in summaries if s.get("reflection_applied", False))
            
            response = {
                "type": "multi_topic",
                "summaries": summaries,
                "documents": doc_id,
                "topics": topics,
                "total_chunks_processed": total_chunks,
                "successful_topics": len(successful_summaries),
                "total_topics": len(topics),
                "search_method": "vector_similarity_multi_topic",
                "parallel_processing": parallel_metadata,
                "performance": {
                    "parallel_time": parallel_metadata.get("total_time", 0),
                    "estimated_sequential_time": total_sequential_time,
                    "speedup_factor": round(parallel_speedup, 2),
                    "longest_individual_task": max_individual_time,
                    "efficiency": round((total_sequential_time / (parallel_metadata.get("total_time", 1) * len(topics))) * 100, 1) if parallel_metadata.get("total_time") and topics else 0,
                    "parallel_method": parallel_metadata.get("method", "ThreadPoolExecutor"),
                    "max_workers": min(len(topics), 5) if topics else 0
                },
                "reflection_enabled": enable_reflection
            }
            
            # Add reflection statistics if reflection was used
            if enable_reflection and reflection_stats:
                response["reflection_statistics"] = reflection_stats
            elif enable_reflection:
                response["reflection_statistics"] = {
                    "total_topics": len(summaries),
                    "reflection_applied": reflection_applied_count,
                    "reflection_skipped": len(summaries) - reflection_applied_count
                }
            
            return response
        
        # Single topic - use existing logic
        else:
            single_topic = topics[0]
            relevant_docs: List[Document] = []
            for d in doc_id:
                vs = load_vectorstore(d)
                retrieved = vs.similarity_search(single_topic, k=top_k)
                relevant_docs.extend(retrieved)
            
            if not relevant_docs:
                return {
                    "type": "single_topic",
                    "summary": f"No relevant content found for the query: '{single_topic}'", 
                    "documents": doc_id, 
                    "query": single_topic
                }
            
            all_docs = relevant_docs
            summary_context = f"focusing on aspects related to: {single_topic}"
    else:
        # Original behavior: load all chunks
        all_docs: List[Document] = []
        for d in doc_id:
            all_docs.extend(load_chunks(d))
        summary_context = "covering all content"

    if not all_docs:
        return {
            "type": "single",
            "summary": "No content available for summarization.", 
            "documents": doc_id, 
            "query": query
        }

    # Prepare state for summary graph
    summary_input = {"docs": all_docs}
    if query and query.strip() and "," not in query:
        summary_input["query_context"] = query.strip()
    
    summary_state = SUMMARY_GRAPH.invoke(summary_input)
    summary = summary_state.get("summary", "Unable to generate summary.") if summary_state else "Unable to generate summary."
    
    if not summary or summary == "Unable to generate summary." or summary == "No content to summarize.":
        return {
            "type": "single",
            "summary": "No content available for summarization.", 
            "documents": doc_id, 
            "query": query
        }
    
    summary = summary.strip()
    target = {"short": "≈3 sentences", "medium": "≈8 sentences", "long": "≈15 sentences"}[length]
    
    # Include query context in the refinement prompt if a query was provided
    if query and query.strip() and "," not in query:
        refinement_prompt = f"Rewrite this summary so it fits {target} while preserving key info, {summary_context}:\n\n{summary}"
    else:
        refinement_prompt = f"Rewrite this summary so it fits {target} while preserving key info:\n\n{summary}"
    
    refined = LLM.invoke(refinement_prompt)
    
    final_summary = refined.content.strip() if hasattr(refined, 'content') else str(refined).strip()
    
    result = {
        "type": "single",
        "summary": final_summary, 
        "documents": doc_id,
        "chunks_processed": len(all_docs)
    }
    
    if query and query.strip():
        result["query"] = query
        result["search_method"] = "vector_similarity"
    else:
        result["search_method"] = "full_document"
        
    return result


# -------- Ask -------------------------------------------------------
@app.get("/ask")
async def ask_docs(
    q: str = Query(...),
    doc_id: List[str] = Query(...),
    top_k: int = 3,
):
    with db_session() as s:
        docs_meta = s.exec(select(Doc).where(Doc.id.in_(doc_id))).all()
    not_ready = [d.id for d in docs_meta if d.status != "ready"]
    if not_ready:
        raise HTTPException(409, f"Documents not ready: {not_ready}")

    retrieved: List[Document] = []
    for d in doc_id:
        vs = load_vectorstore(d)
        retrieved.extend(vs.similarity_search(q, k=top_k))

    if not retrieved:
        return {"answer": "No relevant passages found.", "snippets": []}

    answer = LLM.invoke(
        "Answer the user's question based only on the excerpts below. "
        "If the answer is not contained, say so.\n\n" +
        "\n---\n".join(doc.page_content for doc in retrieved) + f"\n\nQuestion: {q}"
    )
    snippets = [{"content": d.page_content} for d in retrieved]
    return JSONResponse({"answer": answer.content.strip(), "snippets": snippets, "documents": doc_id})
