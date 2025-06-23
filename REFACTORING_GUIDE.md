# LangGraph Send API Refactoring Guide

## Overview

This document explains the architectural refactoring that replaces ThreadPoolExecutor-based parallel processing with LangGraph's native `Send` command for a more integrated, LangGraph-native approach.

## 🔄 What Changed

### Before: Multiple Graphs + ThreadPoolExecutor
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ SUMMARY_GRAPH   │    │ REFLECTION_GRAPH│    │ ThreadPoolExecutor│
│                 │    │                 │    │                 │
│ • Single doc    │    │ • Evaluate      │    │ • External      │
│   summarization │    │ • Improve       │    │   parallel      │
│                 │    │                 │    │   processing    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
        ┌─────────────────────────▼─────────────────────────┐
        │ MULTI_TOPIC_SUMMARY_WITH_REFLECTION_GRAPH        │
        │                                                  │
        │ • Combines summary + reflection                  │
        │ • Uses ThreadPoolExecutor for parallel topics   │
        │ • Separate graph invocations                     │
        │ • Complex state management                       │
        └──────────────────────────────────────────────────┘
```

### After: Unified Graph + Send API
```
┌──────────────────────────────────────────────────────────────┐
│ UNIFIED_SUMMARY_REFLECTION_GRAPH                             │
│                                                              │
│ ┌─────────────┐  Send API  ┌─────────────────────────────┐   │
│ │Map Topics   │──────────▶│ Process Topic + Reflection  │   │
│ │to Sends     │            │ (Parallel via Send)         │   │
│ └─────────────┘            │                             │   │
│                            │ • Summary generation        │   │
│                            │ • Integrated reflection     │   │
│                            │ • Native LangGraph parallel │   │
│                            └─────────────────────────────┘   │
│                                          │                   │
│                            ┌─────────────▼─────────────┐     │
│                            │ Collect Results           │     │
│                            │ (State Reducers)          │     │
│                            └───────────────────────────┘     │
└──────────────────────────────────────────────────────────────┘
```

## 🚀 Key Improvements

### 1. Native LangGraph Parallel Processing
- **Before**: Used external `ThreadPoolExecutor` with `concurrent.futures`
- **After**: Uses LangGraph's `Send` API for dynamic parallel routing
- **Benefit**: True LangGraph-native architecture with built-in state management

### 2. Unified Graph Architecture
- **Before**: 4 separate graphs (`SUMMARY_GRAPH`, `MULTI_TOPIC_SUMMARY_GRAPH`, `REFLECTION_GRAPH`, `MULTI_TOPIC_SUMMARY_WITH_REFLECTION_GRAPH`)
- **After**: 1 unified graph (`UNIFIED_SUMMARY_REFLECTION_GRAPH`)
- **Benefit**: Simplified architecture, reduced complexity, single source of truth

### 3. Integrated Reflection System
- **Before**: Reflection as separate graph invocations within thread workers
- **After**: Reflection integrated as subgraph within each parallel topic processing
- **Benefit**: Atomic processing, consistent state, reduced overhead

### 4. Better State Management
- **Before**: Manual state passing between ThreadPoolExecutor workers
- **After**: Built-in LangGraph state reducers (`operator.add`)
- **Benefit**: Automatic result aggregation, better error handling

## 📁 File Structure

```
src/
├── graphs/
│   ├── summary.py                      # Original summary graphs (kept for comparison)
│   ├── reflection.py                   # Original reflection graphs (kept for comparison)
│   ├── unified_summary_reflection.py   # NEW: Unified Send API implementation
│   └── ingestion.py                    # Unchanged
├── api/
│   └── endpoints.py                    # Updated to use unified graph
├── services/
│   └── parallel_service.py             # Original ThreadPoolExecutor service (kept)
└── main.py                            # Updated root endpoint descriptions
```

## 🔧 Technical Implementation

### Send API Usage

The key innovation is using LangGraph's `Send` command for dynamic parallel processing:

```python
def route_to_parallel_processing(state):
    """Route to parallel processing using Send objects."""
    sends = state.get("sends", [])
    if not sends:
        return "collect_results"
    return sends  # Return list of Send objects for parallel execution

# Create Send objects for each topic
sends = []
for i, topic in enumerate(topics):
    topic_state = {
        "topic_id": i,
        "topic": topic,
        "docs": relevant_docs,
        # ... other state
    }
    sends.append(Send("process_topic_with_reflection", topic_state))
```

### State Reducers

Results are automatically aggregated using LangGraph's state reducers:

```python
# Add reducer to accumulate topic results
graph.add_reducer("topic_results", operator.add)

# Each parallel node returns results that get automatically merged
return {"topic_results": [result]}
```

### Integrated Reflection

Reflection is now a subgraph within each parallel topic processing:

```python
def _process_topic_with_reflection(state):
    # Step 1: Generate summary
    initial_summary = generate_summary(docs)
    
    # Step 2: Apply reflection if enabled (integrated)
    if enable_reflection:
        reflection_result = _apply_reflection_subgraph({
            "summary": initial_summary,
            "topic": topic,
            "source_content": source_content
        })
        final_summary = reflection_result.get("improved_summary", {}).get("improved_text", initial_summary)
```

## 🎯 Usage Examples

### Basic Multi-Topic Summarization
```python
from src.graphs.unified_summary_reflection import UNIFIED_SUMMARY_REFLECTION_GRAPH

input_state = {
    "topics": ["machine learning", "data science", "AI ethics"],
    "doc_ids": ["doc_1", "doc_2"],
    "top_k": 10,
    "length": "medium",
    "enable_reflection": True
}

result = UNIFIED_SUMMARY_REFLECTION_GRAPH.invoke(input_state)
summaries = result["summaries"]
```

### API Integration
```python
# Updated endpoint now uses unified graph
from src.graphs.unified_summary_reflection import UNIFIED_SUMMARY_REFLECTION_GRAPH

async def _process_multi_topic(topics, doc_ids, top_k, length, enable_reflection):
    """Process multiple topics using unified LangGraph Send API."""
    result_state = UNIFIED_SUMMARY_REFLECTION_GRAPH.invoke({
        "topics": topics,
        "doc_ids": doc_ids,
        "top_k": top_k,
        "length": length,
        "enable_reflection": enable_reflection
    })
    return result_state
```

## 🧪 Testing & Comparison

Run the demo script to compare old vs new approaches:

```bash
python demo_unified_graph.py
```

The demo provides:
- Side-by-side performance comparison
- Architecture analysis
- Send API feature demonstration
- Reflection integration testing

## 📊 Performance Benefits

### Reduced Overhead
- **Before**: Separate graph compilation and invocation overhead
- **After**: Single graph execution with native parallel processing
- **Impact**: Lower memory usage, faster startup

### Better Resource Management
- **Before**: Fixed ThreadPoolExecutor size, manual worker management
- **After**: Dynamic Send API routing, automatic resource allocation
- **Impact**: More efficient resource utilization

### Simplified Error Handling
- **Before**: Manual exception handling across thread boundaries
- **After**: Built-in LangGraph error handling and state consistency
- **Impact**: More robust error recovery

## 🔍 State Schema

### Input State
```python
{
    "topics": List[str],           # Topics to process in parallel
    "doc_ids": List[str],          # Document IDs to search
    "top_k": int,                  # Chunks per topic
    "length": str,                 # "short" | "medium" | "long"
    "enable_reflection": bool      # Enable reflection system
}
```

### Output State
```python
{
    "summaries": [                 # Results from parallel processing
        {
            "topic": str,
            "topic_id": int,
            "summary": str,
            "status": str,
            "processing_time": float,
            "reflection_applied": bool,
            "changes_made": List[str],     # If reflection applied
            # ... more metadata
        }
    ],
    "parallel_processing": {
        "method": "LangGraph_Send_API",
        "total_time": float,
        "topics_count": int,
        "reflection_statistics": {...}
    }
}
```

## 🎨 Architecture Patterns

### Map-Reduce with Send API
1. **Map**: Create `Send` objects for each topic
2. **Parallel Process**: Each Send routes to topic processing node
3. **Reduce**: State reducers automatically aggregate results

### Subgraph Integration
- Reflection logic embedded as internal function within parallel processing
- No separate graph invocations
- Atomic processing ensures consistency

### Dynamic Routing
- Conditional edges return `Send` objects for parallel execution
- LangGraph handles the parallel routing automatically
- No manual thread pool management required

## 🚨 Migration Notes

### Breaking Changes
- API endpoints now use `UNIFIED_SUMMARY_REFLECTION_GRAPH` instead of separate graphs
- Parallel processing metadata format changed (now includes "LangGraph_Send_API" method)

### Backwards Compatibility
- Original graphs are preserved for comparison and fallback
- API interface remains the same for client applications
- Configuration and settings unchanged

### Performance Considerations
- First invocation may be slightly slower due to graph compilation
- Subsequent calls benefit from cached compilation
- Memory usage reduced due to single graph instance

## 🔮 Future Enhancements

### Potential Improvements
1. **Subgraph Nodes**: Convert reflection to proper LangGraph subgraph node
2. **Conditional Reflection**: Smart reflection triggering based on summary quality
3. **Streaming Results**: Real-time result streaming as topics complete
4. **Error Recovery**: Advanced error handling with retry mechanisms

### Extensibility
The unified architecture makes it easier to:
- Add new processing steps to each topic
- Implement different parallel strategies
- Integrate additional quality checks
- Add monitoring and observability

## 📚 References

- [LangGraph Send API Documentation](https://langchain-ai.github.io/langgraph/concepts/low_level/#send)
- [LangGraph Parallel Processing](https://langchain-ai.github.io/langgraph/how-tos/map-reduce/)
- [State Reducers in LangGraph](https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers)

---

*This refactoring demonstrates the power of LangGraph's native parallel processing capabilities and shows how to eliminate external dependencies like ThreadPoolExecutor in favor of more integrated, graph-native solutions.* 