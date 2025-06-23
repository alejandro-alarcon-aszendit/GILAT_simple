"""Main FastAPI application for the Document Service.

Modular FastAPI application with clear structure and parallel workload visibility.
Supports both file uploads and URL content fetching with authentication and CORS.
"""

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from src.core.config import APIConfig
from src.core.auth import jwt_auth
from src.models.schemas import DocOut, QAResponse
from src.api.endpoints import DocumentEndpoints, SummaryEndpoints, QAEndpoints
from src.api.auth_endpoints import AuthEndpoints, LoginRequest, LoginResponse


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title=APIConfig.TITLE,
        version=APIConfig.VERSION,
        description=APIConfig.DESCRIPTION
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:8501", "http://localhost:3000", "*"],  # Add your frontend URLs
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
    )
    
    # Authentication endpoints (public)
    app.post("/auth/login", response_model=LoginResponse)(
        AuthEndpoints.login
    )
    app.get("/auth/verify", dependencies=[Depends(jwt_auth)])(
        AuthEndpoints.verify_token
    )
    
    # Document management endpoints (with authentication)
    app.get("/formats", dependencies=[Depends(jwt_auth)])(
        DocumentEndpoints.get_supported_formats
    )
    app.post("/documents", status_code=201, response_model=dict, dependencies=[Depends(jwt_auth)])(
        DocumentEndpoints.upload_document
    )
    app.post("/documents/url", status_code=201, response_model=dict, dependencies=[Depends(jwt_auth)])(
        DocumentEndpoints.ingest_url
    )
    app.get("/documents", response_model=list[DocOut], dependencies=[Depends(jwt_auth)])(
        DocumentEndpoints.list_documents
    )
    app.get("/documents/{doc_id}", response_model=DocOut, dependencies=[Depends(jwt_auth)])(
        DocumentEndpoints.get_document
    )
    app.delete("/documents/{doc_id}", dependencies=[Depends(jwt_auth)])(
        DocumentEndpoints.delete_document
    )
    
    # Summarization endpoints (with authentication)
    app.get("/summary", dependencies=[Depends(jwt_auth)])(
        SummaryEndpoints.multi_summary
    )
    
    # Question answering endpoints (with authentication)
    app.get("/ask", response_model=QAResponse, dependencies=[Depends(jwt_auth)])(
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
                "URL content fetching and processing",
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