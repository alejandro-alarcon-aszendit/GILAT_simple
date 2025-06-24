# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Start Development Environment:**
```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
cp env.example .env
# Edit .env and set your OPENAI_API_KEY

# Start API server
python src/main.py

# Start UI (in separate terminal)
streamlit run streamlit_app.py
```

**Running Tests:**
```bash
# Run the main test file
python test_summary_length.py

# Test API endpoints manually
curl http://localhost:8000/health
curl http://localhost:8000/docs  # Swagger UI
```

**Docker Deployment:**
```bash
docker-compose up -d
```

## Core Architecture & LangGraph Best Practices

This is a **modular document processing service** built around **LangGraph workflows** with a focus on **parallel processing**, **reusable subgraphs**, and **proper state management**. The architecture follows current LangGraph best practices for building resilient, scalable AI workflows.

### LangGraph Implementation Patterns

**1. Send API for Dynamic Parallel Processing** (`src/graphs/unified_summary_reflection.py:36-187`)
- **Current Implementation**: Uses LangGraph's `Send` API for true parallel processing
- **Pattern**: Map-reduce with dynamic Send object creation
```python
# Create Send objects for parallel processing
sends.append(Send("process_topic_with_reflection", topic_state))

# Route to parallel execution via conditional edges
graph.add_conditional_edges(
    "map_topics_to_sends",
    route_to_parallel_processing,  # Returns Send objects  
    ["process_topic_with_reflection"]  # Target nodes
)
```
- **Best Practice**: Use Send API instead of ThreadPoolExecutor for LangGraph-native parallelism
- **Error Recovery**: Superstep-based execution ensures transactional behavior - if any Send fails, entire superstep errors but successful results are checkpointed

**2. State Management with TypedDict & Reducers** (`src/utils/graph_schemas.py:14-54`)
- **Current Pattern**: Proper TypedDict schemas with Annotated reducers
```python
class UnifiedState(TypedDict):
    topic_results: Annotated[List[Dict[str, Any]], operator.add]
```
- **Best Practice**: Always use `Annotated` with reducers for parallel execution
- **Reducer Patterns**: 
  - `operator.add` for list concatenation
  - `add_messages` for message handling
  - Custom reducers for specific merging logic

**3. Modular Graph Construction & Subgraphs** (`src/graphs/subgraphs/`)
- **Current**: Compiled graphs as module-level constants
- **Subgraph Structure**: 
  - `DocumentRetrievalSubgraph`: Document fetching and preparation
  - `SummarizationSubgraph`: Core summarization logic
  - `ReflectionSubgraph`: Quality improvement workflow
- **Communication Patterns**: 
  - Shared state schemas for seamless integration
  - Transform functions for different state schemas (`src/utils/state_transformers.py`)

### Service Layer Architecture

**Configuration Management** (`src/core/config.py:24-71`)
- **Current**: Centralized LLM configuration with role-specific instances
- **Pattern**: Separate classes for different concerns: `LLMConfig`, `ParallelConfig`, `APIConfig`
- **Best Practice**: Configure different LLM instances for different tasks (main, reflection, improvement)

**Strategy Pattern Implementation** (`src/utils/summarization_strategies.py`)
- **Current**: Pluggable algorithms (extractive, abstractive, hybrid)
- **Extension Opportunity**: Apply pattern to other processing types
- **Benefit**: Easy addition of new strategies without modifying core logic

**Utility Modules** (`src/utils/`)
- **Current Structure**: Focused, single-responsibility modules
- **Pattern**: Extract complex logic into focused utility modules
- **Modules**: Topic processing, reflection utils, graph schemas, summarization strategies, state transformers

### API & Service Integration

**Modular Endpoints** (`src/api/endpoints.py`)
- **Pattern**: Class-based endpoint organization
- **Authentication**: Conditional JWT via dependency injection
- **Service Integration**: Thin controllers delegating to service classes

**LangGraph Execution** (`src/services/`)
- **Pattern**: Services handle graph execution and result processing
- **State Management**: Clean separation between API logic and business logic

### Key LangGraph Concepts in Use

**Multi-Agent Workflows**: Used for complex document processing pipelines
**Shared State Communication**: Leveraged for agent coordination via proper reducers
**Conditional Logic**: Used for dynamic routing based on content analysis
**Send API**: Implemented for true parallel processing of topics and documents

### Environment Variables

Required:
- `OPENAI_API_KEY`: OpenAI API key for LLM operations

Optional (Authentication):
- `API_AUTH_KEY`: API authentication key (leave empty to disable auth)
- `JWT_SECRET_KEY`: JWT signing key (required if auth enabled)

### File Format Support

Documents: `.txt`, `.md`, `.pdf`, `.docx`, `.html`, `.csv`, `.xml`
Images: `.png`, `.jpg`, `.jpeg`, `.tiff`, `.bmp`, `.webp` (with OCR)
URLs: Web content fetching with automatic extraction

### Testing & Validation

Run `python test_summary_length.py` to validate length parameter functionality across different strategies and reflection systems.