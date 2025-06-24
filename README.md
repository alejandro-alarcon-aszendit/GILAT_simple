# LangGraph Document Processing Service

A sophisticated **modular document processing service** built around **LangGraph workflows** with a focus on **parallel processing**, **reusable subgraphs**, and **proper state management**. Features advanced AI-powered document analysis, cross-document similarity search, and production-ready deployment.

---

## 🎯 Overview

This service provides enterprise-grade document processing capabilities using cutting-edge LangGraph architecture:

- **📄 Multi-Modal Document Processing**: 20+ formats including PDFs, images (OCR), web content
- **🧠 LangGraph Workflows**: Native Send API for true parallel processing
- **🔍 Cross-Document Search**: Concurrent vector similarity ranking across multiple documents
- **📝 AI-Powered Summarization**: Three strategies (extractive, abstractive, hybrid) with reflection
- **🤖 Intelligent Q&A**: Context-aware question answering with source attribution
- **🔐 Enterprise Security**: Optional JWT authentication with fine-grained access control
- **🎨 Modern UI**: Streamlit interface with real-time processing updates

### 🚀 Key Architectural Features

- **LangGraph Send API**: True parallel processing with automatic state aggregation
- **Modular Subgraph Design**: Reusable workflow components (Document Retrieval, Summarization, Reflection)
- **Concurrent Processing**: ThreadPoolExecutor for database queries + Send API for topic processing
- **Cross-Document Ranking**: Global similarity scoring across multiple vector stores
- **AI Reflection System**: Conservative quality improvement with factual accuracy preservation
- **Strategy Pattern**: Pluggable summarization algorithms with consistent interfaces
- **State Management**: TypedDict schemas with annotated reducers for automatic result merging

---

## 📁 Project Structure

```
src/
├── core/                    # Configuration and authentication
│   ├── config.py           # LLM config, parallel settings, API config
│   └── auth.py             # JWT authentication with conditional security
├── models/                  # Data models and schemas
│   ├── database.py         # SQLModel database models
│   └── schemas.py          # Pydantic API response schemas
├── services/                # Business logic services
│   ├── document_service.py # Document parsing, chunking, embedding
│   └── web_content_service.py # URL content fetching
├── graphs/                  # LangGraph workflow definitions
│   ├── unified_summary_reflection.py # Main orchestrator graph
│   └── subgraphs/          # Modular reusable subgraphs
│       ├── __init__.py     # Subgraph exports
│       ├── document_retrieval.py # Concurrent document retrieval
│       ├── summarization.py # Send API parallel processing
│       └── reflection.py   # Quality improvement workflow
├── api/                     # FastAPI endpoints
│   ├── endpoints.py        # Document, summary, and Q&A endpoints
│   └── auth_endpoints.py   # Login and token verification
├── utils/                   # Modular utilities
│   ├── graph_schemas.py    # TypedDict schemas for LangGraph
│   ├── state_transformers.py # State transformation utilities
│   ├── summarization_strategies.py # Strategy pattern implementations
│   ├── reflection_utils.py # Summary evaluation and improvement
│   └── topic_processing.py # Cross-document retrieval and processing
└── main.py                 # FastAPI application factory
```

---

## 🔧 Installation & Setup

### Prerequisites
- Python 3.9+
- OpenAI API key
- UV (recommended)
- Docker (optional, for deployment)

### Quick Start

1. **Clone and install dependencies:**
   ```bash
   git clone <repository>
   cd GILAT_simple
   
   # Using uv (recommended)
   uv venv
   source .venv/bin/activate
   uv sync
   
   # Or using pip
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```

2. **Configure environment:**
   ```bash
   cp env.example .env
   # Edit .env and set your OPENAI_API_KEY
   ```

3. **Run the services:**
   ```bash
   # Start API server
   python -m src.main
   
   # Start UI (in another terminal)
   streamlit run streamlit_app.py
   ```

4. **Access the application:**
   - **Streamlit UI**: http://localhost:8501
   - **FastAPI Backend**: http://localhost:8000
   - **API Documentation**: http://localhost:8000/docs

---

## 🌐 Supported Document Formats

### File Upload Formats
- **Text**: `.txt`, `.md`, `.adoc`
- **Office**: `.pdf`, `.docx`, `.xlsx`, `.pptx`
- **Web**: `.html`, `.xhtml`
- **Data**: `.csv`
- **Images**: `.png`, `.jpg`, `.jpeg`, `.tiff`, `.tif`, `.bmp`, `.webp` (with OCR)
- **XML**: `.xml` (including USPTO and JATS formats)

### URL Content Fetching
- Web pages with automatic content extraction
- HTML parsing with text extraction
- Metadata preservation for source tracking

---

## 🔐 Authentication System

### JWT Token Authentication
The service supports **conditional authentication** that can be enabled or disabled:

**Disabled Authentication** (Default):
```bash
# Leave API_AUTH_KEY empty or use placeholder
API_AUTH_KEY=your_secure_api_auth_key_here  # Placeholder = disabled
```

**Enabled Authentication**:
```bash
# Set a real API key to enable authentication
API_AUTH_KEY=my_secure_key_123
JWT_SECRET_KEY=your_jwt_secret_key_change_in_production
```

### Authentication Flow
1. **Login**: `POST /auth/login` with API key → Receives JWT token
2. **Protected Requests**: Include `Authorization: Bearer <token>` header
3. **Token Expiration**: Tokens expire after 30 minutes

### Security Features
- **Conditional Security**: Automatically enables/disables based on configuration
- **Public Endpoints**: Health checks and documentation always accessible
- **CORS Protection**: Configured for frontend integration
- **Token Validation**: Server-side JWT verification with proper error handling

---

## 📡 API Reference

### Authentication Endpoints
```http
POST /auth/login          # Login with API key → JWT token
GET /auth/verify          # Verify current token (protected)
```

### Document Management
```http
GET /formats              # Get supported file formats
POST /documents           # Upload and process file
POST /documents/url       # Fetch and process URL content
GET /documents            # List all documents
GET /documents/{id}       # Get document details
DELETE /documents/{id}    # Delete document and vectors
```

### Summarization (Parallel Processing)
```http
GET /summary?doc_id=123&doc_id=456&query=topic1,topic2&length=medium&enable_reflection=true
```

**Parameters:**
- `doc_id`: Document IDs (multiple allowed)
- `query`: Topic(s) for focused summarization (comma-separated for parallel processing)
- `length`: `short` | `medium` | `long`
- `strategy`: `abstractive` | `extractive` | `hybrid`
- `top_k`: Number of relevant chunks per topic (default: 10)
- `enable_reflection`: AI quality improvement (default: false)

### Question Answering
```http
GET /ask?q=question&doc_id=123&doc_id=456&top_k=3
```

---

## ⚡ LangGraph Parallel Processing Architecture

### Send API Multi-Topic Processing
Process multiple topics using LangGraph's native Send API for true parallelism:

```bash
curl "localhost:8000/summary?doc_id=123&query=AI,machine learning,neural networks&enable_reflection=true"
```

**Response includes comprehensive parallel processing metadata:**
```json
{
  "type": "multi_topic",
  "summaries": [
    {
      "topic": "AI",
      "topic_id": 0,
      "summary": "Artificial intelligence encompasses...",
      "chunks_processed": 15,
      "status": "success",
      "processing_time": 3.45,
      "strategy": "hybrid",
      "reflection_applied": true,
      "changes_made": ["Improved clarity", "Added context"]
    }
  ],
  "parallel_processing": {
    "total_time": 8.45,
    "estimated_sequential_time": 23.12,
    "speedup_factor": 2.74,
    "efficiency": 54.8,
    "method": "LangGraph_Send_API",
    "max_workers": 5,
    "subgraphs_used": ["SummarizationSubgraph", "DocumentRetrievalSubgraph", "ReflectionSubgraph"]
  },
  "reflection_statistics": {
    "total_topics": 3,
    "reflection_applied": 2,
    "reflection_skipped": 1
  }
}
```

### Concurrent Architecture Benefits
- **LangGraph Send API**: Superstep-based execution with automatic checkpointing
- **ThreadPoolExecutor**: Concurrent vector store queries across multiple documents  
- **Automatic State Aggregation**: Results merged using `Annotated[List, operator.add]` reducers
- **Error Isolation**: Individual topic/document failures don't affect other operations
- **Cross-Document Ranking**: Global similarity scoring with concurrent retrieval
- **Performance Monitoring**: Detailed timing, speedup calculations, and efficiency metrics

---

## 📝 Summarization Strategies

### Available Strategies
1. **Abstractive** (Default): AI generates new sentences by paraphrasing content
2. **Extractive**: Selects and preserves key sentences from original text
3. **Hybrid**: Combines extraction with AI refinement for best of both approaches

### AI Reflection System
Optional quality improvement process that:
- **Evaluates** summary accuracy, completeness, and length compliance
- **Improves** content while preserving factual accuracy
- **Conservative Editing**: Only uses information explicitly stated in source
- **Strategy Aware**: Preserves extractive integrity while enhancing others

---

## 🎨 Streamlit UI Features

### Document Management
- **Drag & Drop Upload**: Support for all document formats
- **URL Content Fetching**: Direct web content ingestion
- **Document Library**: Search, filter, and context management
- **Real-time Status**: Processing progress and chunk counts

### Intelligent Summarization
- **Multi-Document Processing**: Summarize across document collections
- **Topic Filtering**: Focus summaries on specific subjects
- **Length Control**: Adjust summary detail level
- **Context Awareness**: Only process documents in active context

### Q&A Interface
- **Natural Language Queries**: Ask questions in plain English
- **Source Attribution**: See relevant passages used for answers
- **Configurable Retrieval**: Adjust chunk retrieval parameters

### System Monitoring
- **Connection Status**: Real-time API health checking
- **Document Statistics**: Overview of processing status
- **Error Handling**: Clear feedback on failures

---

## 🐳 Docker Deployment

### Quick Deploy
```bash
# Set up environment
cp env.example .env
# Edit .env with your OPENAI_API_KEY

# Start all services
docker-compose up -d
```

### Services
- **API**: http://localhost:8000 (FastAPI backend)
- **UI**: http://localhost:8501 (Streamlit frontend)

### Data Persistence
- `./vector_db/`: Document embeddings and chunks
- `./app.db`: SQLite database

### Environment Variables
```bash
# Required
OPENAI_API_KEY=sk-...

# Authentication (optional)
API_AUTH_KEY=your_secure_api_key    # Leave empty to disable auth
JWT_SECRET_KEY=secure_random_string # Required if using auth
```

---

## 🔍 Advanced Configuration

### LLM Configuration (`src/core/config.py`)
```python
class LLMConfig:
    # Role-specific LLM instances for different tasks
    MAIN_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=2000)
    REFLECTION_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.1, max_tokens=1500)  # Consistency
    IMPROVEMENT_LLM = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, max_tokens=2000)  # Creativity
    EMBEDDER = OpenAIEmbeddings(model="text-embedding-ada-002")
    SPLITTER = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
```

### Parallel Processing Configuration
```python
class ParallelConfig:
    # LangGraph Send API parallelism
    MAX_TOPIC_WORKERS = 5           # Send API parallel topic processing
    
    # ThreadPoolExecutor for database queries
    MAX_DB_QUERY_WORKERS = 8        # Concurrent vector store queries
    
    # Processing limits and timeouts
    PROCESSING_TIMEOUT = 300        # 5 minutes
    MAX_CHUNKS_PER_TOPIC = 20       # Memory limit for reflection
    MAX_SOURCE_CONTENT_LENGTH = 4000 # Token limit for reflection
```

---

## 🛠️ Development

### Adding New Features

**1. New Service:**
```python
# src/services/new_service.py
class NewService:
    @staticmethod
    def process_data(data):
        # Business logic here
        pass
```

**2. New API Endpoints:**
```python
# src/api/endpoints.py
class NewEndpoints:
    @staticmethod
    async def new_endpoint():
        # Endpoint logic here
        pass
```

**3. New LangGraph Subgraph:**
```python
# src/graphs/subgraphs/new_subgraph.py
from langgraph.graph import StateGraph, START, END
from src.utils.graph_schemas import NewState

def build_new_subgraph():
    def process_node(state: NewState) -> NewState:
        # Processing logic here
        return state
    
    graph = StateGraph(NewState)
    graph.add_node("process", process_node)
    graph.add_edge(START, "process")
    graph.add_edge("process", END)
    
    return graph.compile()

NEW_SUBGRAPH = build_new_subgraph()
```

**4. State Schema for New Subgraph:**
```python
# src/utils/graph_schemas.py
from typing import TypedDict, Annotated, List
import operator

class NewState(TypedDict):
    input_data: List[str]
    results: Annotated[List[dict], operator.add]  # Auto-aggregation
    processing_metadata: dict
```

### Testing

```bash
# Test summary length compliance across strategies
python test_summary_length.py

# Test API endpoints
curl http://localhost:8000/health
curl http://localhost:8000/docs  # Swagger UI

# Test LangGraph workflows
python -c "
from src.graphs.unified_summary_reflection import UNIFIED_SUMMARY_REFLECTION_GRAPH
result = UNIFIED_SUMMARY_REFLECTION_GRAPH.invoke({
    'topics': ['test'], 'doc_ids': [], 'length': 5, 'strategy': 'abstractive'
})
print('Graph test completed')
"
```

---

## 📊 Performance Monitoring & Metrics

The system provides comprehensive performance tracking across all parallel operations:

### LangGraph Send API Metrics
- **Total Execution Time**: Wall clock time for entire workflow
- **Speedup Factor**: `sequential_time / parallel_time` 
- **Efficiency**: `(speedup_factor / max_workers) * 100`
- **Superstep Performance**: Automatic checkpointing and error isolation

### ThreadPoolExecutor Database Metrics  
- **Concurrent Queries**: Parallel vector store operations
- **Cross-Document Ranking**: Global similarity score distribution
- **Database Query Times**: Per-document retrieval performance
- **Connection Pool Usage**: Resource utilization tracking

### Memory and Resource Management
- **Automatic Content Truncation**: Configurable limits for reflection processing
- **Chunk Count Limits**: Prevent memory overflow with large documents  
- **Token Management**: Intelligent truncation for LLM context windows
- **Vector Store Optimization**: Per-document stores for optimal retrieval

### Sample Performance Output
```json
{
  "performance": {
    "parallel_time": 8.45,
    "estimated_sequential_time": 23.12,
    "speedup_factor": 2.74,
    "efficiency": 54.8,
    "longest_individual_task": 7.89,
    "parallel_method": "LangGraph_Send_API",
    "max_workers": 5,
    "database_queries": {
      "concurrent_documents": 8,
      "query_completion_times": [1.2, 1.5, 0.9, 2.1],
      "global_ranking_time": 0.15
    }
  }
}
```

---

## 🔧 Troubleshooting

### Common Issues

**Authentication Errors:**
- Verify `API_AUTH_KEY` is set correctly
- Check JWT token expiration (30 minutes)
- Ensure proper `Authorization: Bearer <token>` header format

**Document Processing Failures:**
- Confirm `OPENAI_API_KEY` is valid and has credits
- Check file format is supported
- Review API logs for detailed error messages

**UI Connection Issues:**
- Verify FastAPI server is running on port 8000
- Check for port conflicts
- Test API health endpoint: `curl localhost:8000/health`

### Performance Tips
- **Large Documents**: Automatically chunked for optimal processing
- **Multiple Topics**: Use parallel processing for better performance
- **Memory Usage**: Configured limits prevent token overflow
- **Batch Operations**: Process multiple documents simultaneously

---

## 📄 Database Schema

```sql
CREATE TABLE doc (
    id TEXT PRIMARY KEY,           -- UUID
    name TEXT NOT NULL,            -- Original filename or URL
    status TEXT NOT NULL,          -- processing | ready | failed
    n_chunks INTEGER,              -- Number of vector chunks
    created_at TIMESTAMP NOT NULL
);
```

Vector storage structure:
```
vector_db/
└── <doc_id>/
    ├── chroma.sqlite3      # ChromaDB vector index
    ├── chunks.json         # Raw chunk texts for retrieval
    └── <collection_id>/    # HNSW vector data
        ├── data_level0.bin
        ├── header.bin
        └── link_lists.bin
```

---

## 🤝 Contributing

1. **Fork** the repository
2. **Create** a feature branch
3. **Add** tests for new functionality
4. **Ensure** type safety with proper annotations
5. **Update** documentation for API changes
6. **Submit** a pull request

---

## 📋 Changelog

- ✅ **LangGraph Architecture**: Send API for true parallel processing
- ✅ **Modular Subgraphs**: Document Retrieval, Summarization, Reflection subgraphs  
- ✅ **Cross-Document Ranking**: Concurrent vector queries with global similarity scoring
- ✅ **State Management**: TypedDict schemas with annotated reducers
- ✅ **Dual Concurrency**: Send API (topics) + ThreadPoolExecutor (database queries)
- ✅ **AI Reflection System**: Conservative quality improvement with factual accuracy
- ✅ **Strategy Pattern**: Pluggable summarization algorithms (extractive/abstractive/hybrid)
- ✅ **Enterprise Security**: Optional JWT authentication with conditional activation
- ✅ **Performance Monitoring**: Comprehensive metrics and speedup calculations
- ✅ **Production Deployment**: Docker with persistent storage and configuration


## 🔗 Related Documentation

- **[📐 Complete Architecture Documentation](./ARCHITECTURE.md)** - Comprehensive technical overview with diagrams
- **[🔍 API Reference](http://localhost:8000/docs)** - Interactive Swagger documentation (when running)
- **[🎨 UI Interface](http://localhost:8501)** - Streamlit user interface (when running)

