"""Document ingestion LangGraph.

Handles the document processing pipeline using LangGraph.
"""

from langgraph.graph import Graph
from langchain.docstore.document import Document
from src.services.document_service import DocumentService


def build_ingestion_graph():
    """Build the document ingestion LangGraph.
    
    **Pipeline Steps:**
    1. Parse document content from file
    2. Split text into chunks
    
    Returns:
        Compiled LangGraph for document ingestion
    """
    g = Graph()

    def _parse(file_path: str):
        """Parse document content based on file type."""
        text = DocumentService.parse_document(file_path)
        return {"text": text}

    def _split(state):
        """Split text into Document chunks."""
        text = state["text"]
        docs = DocumentService.split_text(text)
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