"""Document ingestion LangGraph.

Handles the document processing pipeline using LangGraph.
Supports both file uploads and URL content fetching.
"""

from langgraph.graph import Graph
from langchain.docstore.document import Document
from src.services.document_service import DocumentService
from src.services.web_content_service import WebContentService


def build_ingestion_graph():
    """Build the document ingestion LangGraph.
    
    **Pipeline Steps:**
    1. Parse document content from file or URL
    2. Split text into chunks
    
    Returns:
        Compiled LangGraph for document ingestion
    """
    g = Graph()

    def _parse(input_source: str):
        """Parse document content from file path or URL."""
        # Check if input is a URL or file path
        if WebContentService.is_valid_url(input_source):
            # Fetch content from URL
            text = WebContentService.fetch_url_content(input_source)
            return {"text": text, "source_type": "url", "source": input_source}
        else:
            # Parse document file
            text = DocumentService.parse_document(input_source)
            return {"text": text, "source_type": "file", "source": input_source}

    def _split(state):
        """Split text into Document chunks with metadata."""
        text = state["text"]
        source_type = state.get("source_type", "file")
        source = state.get("source", "unknown")
        
        docs = DocumentService.split_text(text)
        
        # Add metadata to chunks
        for doc in docs:
            doc.metadata.update({
                "source_type": source_type,
                "source": source
            })
        
        return {"docs": docs}

    # Build graph
    g.add_node("parse", _parse)
    g.add_node("split", _split)
    g.set_entry_point("parse")
    g.add_edge("parse", "split")
    g.set_finish_point("split")
    
    return g.compile()


# Create the compiled graph instance
INGESTION_GRAPH = build_ingestion_graph() 