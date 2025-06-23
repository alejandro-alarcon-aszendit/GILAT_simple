# LangGraph Document Service v2.0

A comprehensive document processing service with parallel workloads, AI reflection, and JWT authentication. Built with FastAPI, LangGraph, and modern async architecture.

---

## 🎯 Overview

This service ingests documents from multiple sources and provides intelligent document analysis capabilities:

- **📄 Document Processing**: Upload files or fetch from URLs with auto-parsing
- **🤖 Semantic Q&A**: Natural language question answering over your documents  
- **📝 Multi-Document Summarization**: Parallel processing with AI reflection
- **🔍 Vector Search**: Similarity-based content retrieval
- **🔐 JWT Authentication**: Secure API access with token-based auth
- **🎨 Modern UI**: Beautiful Streamlit interface with real-time updates

### 🚀 Key Features

- **Multiple Input Sources**: File uploads + URL content fetching
- **Parallel Processing**: Multi-topic summarization with ThreadPoolExecutor
- **AI Reflection System**: Quality improvement for generated summaries
- **Modular Architecture**: Clean separation of concerns with service layers
- **Type Safety**: Full type hints and Pydantic validation
- **Production Ready**: Docker deployment with authentication

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
│   ├── parallel_service.py # Parallel workload orchestration
│   └── web_content_service.py # URL content fetching
├── graphs/                  # LangGraph workflow definitions
│   ├── ingestion.py        # Document ingestion pipeline
│   └── unified_summary_reflection.py # Parallel summarization with Send API
├── api/                     # FastAPI endpoints
│   ├── endpoints.py        # Document, summary, and Q&A endpoints
│   └── auth_endpoints.py   # Login and token verification
├── utils/                   # Modular utilities
│   ├── summarization_strategies.py # Abstractive/extractive/hybrid strategies
│   ├── reflection_utils.py # Summary evaluation and improvement
│   ├── topic_processing.py # Document retrieval and topic handling
│   └── graph_schemas.py    # TypedDict schemas for LangGraph
└── main.py                 # FastAPI application factory
```

---

## 🔧 Installation & Setup

### Prerequisites
- Python 3.9+
- OpenAI API key
- Docker (optional, for deployment)

### Quick Start

1. **Clone and install dependencies:**
   ```bash
   git clone <repository>
   cd GILAT_simple
   pip install -r requirements.txt
   ```

2. **Configure environment:**
   ```bash
   cp env.example .env
   # Edit .env and set your OPENAI_API_KEY
   ```

3. **Run the services:**
   ```bash
   # Start API server
   python src/main.py
   
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

## ⚡ Parallel Processing Features

### Multi-Topic Summarization
Process multiple topics simultaneously using ThreadPoolExecutor:

```bash
curl "localhost:8000/summary?doc_id=123&query=AI,machine learning,neural networks&enable_reflection=true"
```

**Response includes parallel processing metadata:**
```json
{
  "type": "multi_topic",
  "summaries": [...],
  "parallel_processing": {
    "total_time": 15.2,
    "topics_count": 3,
    "method": "ThreadPoolExecutor",
    "speedup_factor": 2.8
  },
  "reflection_statistics": {
    "total_topics": 3,
    "reflection_applied": 2,
    "reflection_skipped": 1
  }
}
```

### LangGraph Send API Integration
- **True Parallel Processing**: Uses LangGraph's Send API for concurrent operations
- **Automatic Aggregation**: Results collected with `operator.add` reducers
- **Error Resilience**: Graceful handling of failed workloads
- **Performance Monitoring**: Detailed timing and speedup calculations

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
    MAIN_LLM = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)
    REFLECTION_LLM = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)
    IMPROVEMENT_LLM = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
    EMBEDDER = OpenAIEmbeddings(model="text-embedding-ada-002")
```

### Parallel Processing Limits
```python
class ParallelConfig:
    MAX_TOPIC_WORKERS = 5           # Multi-topic processing
    PROCESSING_TIMEOUT = 300        # 5 minutes
    MAX_CHUNKS_PER_TOPIC = 20       # Limit for reflection
    MAX_SOURCE_CONTENT_LENGTH = 4000 # Content truncation
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

**2. New Endpoints:**
```python
# src/api/endpoints.py
class NewEndpoints:
    @staticmethod
    async def new_endpoint():
        # Endpoint logic here
        pass
```

**3. New LangGraph:**
```python
# src/graphs/new_graph.py
def build_new_graph():
    g = Graph()
    # Define nodes and edges
    return g.compile()
```

### Testing
```bash
# Run tests (if available)
pytest

# Test API endpoints
curl http://localhost:8000/health
curl http://localhost:8000/docs  # Swagger UI
```

---

## 📊 Performance Monitoring

The system provides detailed metrics for all parallel operations:

- **Execution Time**: Total and per-workload timing
- **Speedup Calculations**: Sequential vs parallel performance comparison
- **Worker Utilization**: Efficiency metrics and resource usage
- **Error Tracking**: Failed workloads with detailed error messages
- **Memory Management**: Automatic content truncation and chunking limits

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
    ├── index.sqlite    # ChromaDB vector index
    └── chunks.json     # Raw chunk texts for retrieval
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

### v2.0 (Current)
- ✅ Modular architecture with service layers
- ✅ JWT authentication with conditional security
- ✅ URL content fetching and processing
- ✅ Parallel multi-topic summarization
- ✅ AI reflection system for quality improvement
- ✅ LangGraph Send API integration
- ✅ Docker deployment support
- ✅ Comprehensive Streamlit UI
- ✅ Three summarization strategies (abstractive/extractive/hybrid)

### v1.0 (Legacy)
- Basic document upload and processing
- Simple summarization without parallel processing
- No authentication system
- Limited UI capabilities

---

## 📄 License

MIT License - See LICENSE file for details.

---

**© 2025 LangGraph Document Service** • Built with FastAPI, LangGraph, and OpenAI

