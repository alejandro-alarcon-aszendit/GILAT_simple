"""Main FastAPI application for the Document Service.

Modular FastAPI application with clear structure and parallel workload visibility.
Supports both file uploads and URL content fetching with authentication and CORS.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from src.core.config import APIConfig
from src.core.auth import jwt_auth
from src.models.schemas import DocOut, QAResponse
from src.api.endpoints import DocumentEndpoints, SummaryEndpoints, QAEndpoints
from src.api.auth_endpoints import AuthEndpoints, LoginRequest, LoginResponse
from src.models.database import init_database
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables to store initialized heavy dependencies
_document_converter = None

def get_document_converter():
    """Get the pre-initialized document converter."""
    global _document_converter
    if _document_converter is None:
        logger.warning("Document converter not initialized during startup, creating new instance")
        from docling.document_converter import DocumentConverter
        _document_converter = DocumentConverter()
    return _document_converter

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    global _document_converter
    
    # Startup
    logger.info("Starting application initialization...")
    
    # 1. Initialize database
    logger.info("Initializing database...")
    try:
        init_database()
        logger.info("✅ Database initialization completed.")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        # Don't fail the app startup for database issues in development
    
    # 2. Initialize heavy dependencies
    logger.info("Initializing document processing dependencies...")
    try:
        from docling.document_converter import DocumentConverter
        logger.info("Loading docling DocumentConverter (downloading models if needed)...")
        _document_converter = DocumentConverter()
        logger.info("✅ DocumentConverter initialized successfully.")
    except Exception as e:
        logger.error(f"❌ DocumentConverter initialization failed: {e}")
        logger.warning("Document processing may be slower on first use.")
    
    # 3. Pre-warm other components if needed
    try:
        # Pre-import heavy dependencies to cache them
        logger.info("Pre-loading other dependencies...")
        from src.core.config import LLMConfig  # This loads OpenAI models
        from langchain_chroma import Chroma  # Pre-load Chroma
        logger.info("✅ Dependencies pre-loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Dependency pre-loading failed: {e}")
    
    logger.info("🚀 Application startup completed!")
    
    yield
    
    # Shutdown (if needed)
    logger.info("Application shutting down...")

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    
    app = FastAPI(
        title=APIConfig.TITLE,
        version=APIConfig.VERSION,
        description=APIConfig.DESCRIPTION,
        lifespan=lifespan
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
    app.get("/documents/{doc_id}/chunks", dependencies=[Depends(jwt_auth)])(
        DocumentEndpoints.get_document_chunks
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
                "Document retrieval (concurrent vector database queries)",
                "Question answering (concurrent cross-document search)"
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