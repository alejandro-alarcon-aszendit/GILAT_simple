"""Configuration module for the Document Service.

Centralizes all configuration, environment variables, and LLM instances.
"""

import os
from pathlib import Path
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# -------------------- Directory Setup (Auto-creation) -------------------
# Ensure all required directories exist
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

VECTOR_DB_DIR = Path("vector_db")
VECTOR_DB_DIR.mkdir(exist_ok=True)

# -------------------- Database Configuration --------------------
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR}/app.db")

# -------------------- File Storage Configuration ----------------
BASE_DIR = VECTOR_DB_DIR
CHUNK_FILE = "chunks.json"

# -------------------- LLM Configuration -------------------------
class LLMConfig:
    """Configuration for different LLM instances with specific purposes."""
    
    # Main LLM for general tasks
    MAIN_LLM = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)
    
    # Specialized LLMs for reflection system
    REFLECTION_LLM = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)  # Lower temp for consistency
    IMPROVEMENT_LLM = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)  # Higher temp for creativity
    
    # Embeddings
    EMBEDDER = OpenAIEmbeddings(model="text-embedding-ada-002")
    
    # Text splitter
    SPLITTER = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

# -------------------- Parallel Processing Configuration ---------
class ParallelConfig:
    """Configuration for parallel processing workloads."""
    
    # Maximum concurrent workers for LangGraph parallel processing
    MAX_TOPIC_WORKERS = 5  # For multi-topic summary processing via LangGraph Send API
    
    # Maximum concurrent workers for database queries
    MAX_DB_QUERY_WORKERS = 8  # For concurrent vector store queries across documents
    
    # Timeouts and limits
    PROCESSING_TIMEOUT = 300  # 5 minutes
    MAX_CHUNKS_PER_TOPIC = 20  # Limit chunks for reflection to avoid token limits
    MAX_SOURCE_CONTENT_LENGTH = 4000  # Truncate source content for reflection

# -------------------- API Configuration -------------------------
class APIConfig:
    """Configuration for the FastAPI application."""
    
    TITLE = "LangGraph Document Service"
    VERSION = "2.0"
    DESCRIPTION = """
    Modular document processing service with parallel workloads.
    
    Features:
    - Async document ingestion with vector storage
    - Multi-topic parallel summarization
    - AI reflection for quality improvement
    - Vector similarity search and Q&A
    """

# Export commonly used instances
llm = LLMConfig.MAIN_LLM
embedder = LLMConfig.EMBEDDER
splitter = LLMConfig.SPLITTER 