"""Main FastAPI application for the Document Service.

Modular FastAPI application with clear structure and parallel workload visibility.
"""

from fastapi import FastAPI
from src.core.config import APIConfig
from src.models.schemas import DocOut
from src.api.endpoints import DocumentEndpoints, SummaryEndpoints, QAEndpoints


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title=APIConfig.TITLE,
        version=APIConfig.VERSION,
        description=APIConfig.DESCRIPTION
    )
    
    # Document management endpoints
    app.post("/documents", status_code=201, response_model=dict)(
        DocumentEndpoints.upload_document
    )
    app.get("/documents", response_model=list[DocOut])(
        DocumentEndpoints.list_documents
    )
    app.get("/documents/{doc_id}", response_model=DocOut)(
        DocumentEndpoints.get_document
    )
    app.delete("/documents/{doc_id}")(
        DocumentEndpoints.delete_document
    )
    
    # Summarization endpoints
    app.get("/summary")(
        SummaryEndpoints.multi_summary
    )
    
    # Question answering endpoints
    app.get("/ask")(
        QAEndpoints.ask_docs
    )
    
    @app.get("/")
    async def root():
        """Root endpoint with service information."""
        return {
            "service": APIConfig.TITLE,
            "version": APIConfig.VERSION,
            "status": "running",
            "features": [
                "Document upload and processing",
                "Multi-topic parallel summarization",
                "AI reflection for quality improvement", 
                "Vector similarity search",
                "Question answering"
            ],
            "parallel_workloads": [
                "Multi-topic summarization (LangGraph Send API)",
                "Reflection system (integrated with Send API)",
                "Document retrieval (concurrent vector searches)"
            ]
        }
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "service": APIConfig.TITLE}
    
    return app


# Create the application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 