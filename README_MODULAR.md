# Document Service v2.0 - Modular Architecture

A modular document processing service with parallel workloads and AI reflection, built with FastAPI and LangGraph.

## 📁 Project Structure

```
src/
├── core/                    # Core configuration and settings
│   ├── __init__.py
│   └── config.py           # Centralized configuration (LLMs, parallel settings)
├── models/                  # Data models and schemas
│   ├── __init__.py
│   ├── database.py         # SQLModel database models
│   └── schemas.py          # Pydantic schemas for API responses
├── services/                # Business logic services
│   ├── __init__.py
│   ├── document_service.py # Document processing operations
│   └── parallel_service.py # Parallel workload management
├── graphs/                  # LangGraph workflow definitions
│   ├── __init__.py
│   ├── ingestion.py        # Document ingestion pipeline
│   ├── summary.py          # Summarization workflows
│   └── reflection.py       # AI reflection system
├── api/                     # FastAPI endpoints
│   ├── __init__.py
│   └── endpoints.py        # Modular endpoint classes
└── main.py                 # FastAPI application factory
```

## 🚀 Key Features

### **Modular Architecture**
- **Clear Separation of Concerns**: Each module has a specific responsibility
- **Service Layer**: Business logic separated from API endpoints
- **Configuration Management**: Centralized configuration for easy maintenance
- **Type Safety**: Full type hints and Pydantic schemas

### **Parallel Workloads** 
The system clearly identifies and optimizes parallel processing:

1. **Multi-Topic Summarization**
   - Uses `ThreadPoolExecutor` for true parallelism
   - Each topic processed independently
   - Performance monitoring and speedup calculations

2. **AI Reflection System**
   - Parallel evaluation and improvement of summaries
   - Structured output parsing for consistent quality
   - Configurable reflection workers

3. **Document Retrieval**
   - Concurrent vector similarity searches
   - Parallel chunk processing during ingestion

### **Processing Pipelines**

#### Document Ingestion (Synchronous)
```
Upload → Parse → Split → Embed → Store → Update DB
```

#### Multi-Topic Summarization (Parallel)
```
Topics → Map to Docs → Parallel Processing → Aggregate Results
```

#### Reflection System (Parallel)
```
Summary → Evaluate → Improve → Quality Check
```

## 🔧 Configuration

### LLM Configuration (`src/core/config.py`)
```python
class LLMConfig:
    MAIN_LLM = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)
    REFLECTION_LLM = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.1)
    IMPROVEMENT_LLM = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)
```

### Parallel Processing Configuration
```python
class ParallelConfig:
    MAX_TOPIC_WORKERS = 5      # Multi-topic processing
    MAX_REFLECTION_WORKERS = 3  # Reflection system
    PROCESSING_TIMEOUT = 300    # 5 minutes
```

## 🚦 Running the Service

### Development
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-api-key"

# Run with uvicorn
uvicorn src.main:app --reload

# Or run directly
python -m src.main
```

### Production
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## 📡 API Endpoints

### Document Management
- `POST /documents` - Upload and process documents
- `GET /documents` - List all documents
- `GET /documents/{id}` - Get document details
- `DELETE /documents/{id}` - Delete document

### Summarization
- `GET /summary` - Multi-document summarization with parallel processing
  - **Single Topic**: `?query=machine learning`
  - **Multi-Topic**: `?query=machine learning,neural networks,AI ethics`
  - **Length Control**: `?length=short|medium|long`
  - **Reflection**: `?enable_reflection=true`

### Question Answering
- `GET /ask` - Ask questions about documents

## 🔄 Parallel Workload Examples

### Multi-Topic Parallel Processing
```bash
curl "http://localhost:8000/summary?doc_id=123&query=AI,ML,NLP&enable_reflection=true"
```

Response includes parallel processing metadata:
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

## 🧩 Adding New Features

### 1. New Service
```python
# src/services/new_service.py
class NewService:
    @staticmethod
    def process_data(data):
        # Business logic here
        pass
```

### 2. New LangGraph
```python
# src/graphs/new_graph.py
def build_new_graph():
    g = Graph()
    # Define nodes and edges
    return g.compile()
```

### 3. New Endpoints
```python
# src/api/endpoints.py
class NewEndpoints:
    @staticmethod
    async def new_endpoint():
        # Endpoint logic here
        pass
```

## 🔍 Monitoring Parallel Workloads

The system provides detailed monitoring for all parallel operations:

- **Execution Time**: Total and per-workload timing
- **Speedup Calculations**: Sequential vs parallel performance
- **Worker Utilization**: Efficiency metrics
- **Error Tracking**: Failed workloads and reasons
- **Resource Usage**: Memory and CPU considerations

## 🌟 Benefits of Modular Structure

1. **Maintainability**: Easy to locate and modify specific functionality
2. **Testability**: Each module can be tested independently
3. **Scalability**: Easy to add new features without affecting existing code
4. **Clarity**: Clear understanding of parallel vs sequential operations
5. **Performance**: Optimized parallel processing with monitoring
6. **Type Safety**: Full type hints for better development experience

## 🔧 Migration from Monolithic

The original `agent.py` has been refactored into:

- **Configuration**: `src/core/config.py`
- **Database Models**: `src/models/database.py`
- **Document Processing**: `src/services/document_service.py`
- **Parallel Processing**: `src/services/parallel_service.py`
- **LangGraphs**: `src/graphs/`
- **API Endpoints**: `src/api/endpoints.py`
- **Application**: `src/main.py`

The functionality remains the same, but the code is now much more organized and maintainable. 