"""Document processing service.

Handles document parsing, chunking, embedding, and vector storage.
Supports both file uploads and URL content fetching.
"""

import json
import uuid
from pathlib import Path
from typing import List
from tempfile import NamedTemporaryFile

from langchain.docstore.document import Document
from langchain_chroma import Chroma
from docling.document_converter import DocumentConverter

from src.core.config import BASE_DIR, CHUNK_FILE, embedder, splitter
from src.models.database import Doc, get_db_session
from src.services.web_content_service import WebContentService


class DocumentService:
    """Service for handling document operations."""
    
    @staticmethod
    def get_supported_formats() -> dict:
        """Get all supported document formats organized by category.
        
        Returns:
            Dictionary with format categories and their supported extensions
        """
        return {
            "text_formats": [".txt", ".md", ".adoc"],
            "office_documents": [".pdf", ".docx", ".xlsx", ".pptx"], 
            "web_formats": [".html", ".xhtml"],
            "data_formats": [".csv"],
            "image_formats": [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"],
            "xml_formats": [".xml"],
            "description": {
                "text_formats": "Plain text, Markdown, and AsciiDoc files",
                "office_documents": "PDF and Microsoft Office documents",
                "web_formats": "HTML and XHTML web pages", 
                "data_formats": "Comma-separated value files",
                "image_formats": "Common image formats with OCR capabilities",
                "xml_formats": "XML files including USPTO and JATS formats"
            },
            "all_extensions": [
                ".txt", ".md", ".adoc", ".pdf", ".docx", ".xlsx", ".pptx",
                ".html", ".xhtml", ".csv", ".png", ".jpg", ".jpeg", 
                ".tiff", ".tif", ".bmp", ".webp", ".xml"
            ]
        }
    
    @staticmethod
    def parse_document(file_path: str) -> str:
        """Parse document content based on file type.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Extracted text content
            
        Raises:
            ValueError: If document parsing fails
        """
        file_path = Path(file_path)
        file_extension = file_path.suffix.lower()
        
        # Get supported formats (excluding plain text which we handle separately)
        supported_formats = DocumentService.get_supported_formats()
        docling_formats = set(supported_formats["all_extensions"]) - {".txt"}
        
        try:
            if file_extension in ['.txt']:
                # Simple text reading for plain text files
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            elif file_extension in docling_formats:
                # Use docling for all supported document formats
                converter = DocumentConverter()
                result = converter.convert(file_path)
                
                # Check if conversion was successful
                if result is None or result.document is None:
                    raise ValueError("Document conversion returned None")
                
                # Try to extract text content
                try:
                    text = result.document.export_to_text()
                except:
                    # Fallback to markdown export
                    try:
                        text = result.document.export_to_markdown()
                    except:
                        raise ValueError("Failed to extract text from document")
                
                # Ensure we got some text
                if not text or not text.strip():
                    raise ValueError("No text content extracted from document")
                    
            else:
                # Try to read as text for other file types
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read()
                except UnicodeDecodeError:
                    # If UTF-8 fails, try with error handling
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        text = f.read()
            
            # Final check to ensure we have content
            if not text or not text.strip():
                raise ValueError("No text content found in document")
            
            return text
        except Exception as e:
            raise ValueError(f"Failed to parse document '{file_path.name}': {str(e)}")
    
    @staticmethod
    def split_text(text: str) -> List[Document]:
        """Split text into chunks using the configured text splitter.
        
        Args:
            text: Text content to split
            
        Returns:
            List of Document objects with chunked content
        """
        chunks = splitter.split_text(text)
        return [Document(page_content=chunk) for chunk in chunks]
    
    @staticmethod
    def create_vector_store(docs: List[Document], doc_id: str) -> Chroma:
        """Create and persist vector store for documents.
        
        Args:
            docs: List of Document objects to embed
            doc_id: Unique document identifier
            
        Returns:
            Chroma vector store instance
        """
        vs_dir = BASE_DIR / doc_id
        vs_dir.mkdir(exist_ok=True)
        
        # Create vector store
        vs = Chroma.from_documents(docs, embedder, persist_directory=str(vs_dir))
        
        # Save chunk texts for later retrieval
        with open(vs_dir / CHUNK_FILE, "w", encoding="utf-8") as f:
            json.dump([doc.page_content for doc in docs], f)
        
        return vs
    
    @staticmethod
    def load_vector_store(doc_id: str) -> Chroma:
        """Load existing vector store for a document.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Chroma vector store instance
            
        Raises:
            FileNotFoundError: If vector store doesn't exist
        """
        vs_dir = BASE_DIR / doc_id
        if not vs_dir.exists():
            raise FileNotFoundError(f"Vector store not found for document {doc_id}")
        return Chroma(persist_directory=str(vs_dir), embedding_function=embedder)
    
    @staticmethod
    def load_document_chunks(doc_id: str) -> List[Document]:
        """Load document chunks from stored JSON file.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            List of Document objects (empty list if file doesn't exist)
        """
        chunk_file = BASE_DIR / doc_id / CHUNK_FILE
        
        if not chunk_file.exists():
            return []
        
        try:
            with open(chunk_file, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
            return [Document(page_content=chunk) for chunk in chunk_data]
        except Exception:
            return []
    
    @staticmethod
    def ingest_document(doc_id: str, tmp_path: str, filename: str) -> int:
        """Complete document ingestion pipeline.
        
        **Synchronous Processing Pipeline:**
        1. Parse document content
        2. Split into chunks
        3. Create embeddings and vector store
        4. Update database status
        
        Args:
            doc_id: Unique document identifier
            tmp_path: Path to temporary uploaded file
            filename: Original filename
            
        Returns:
            Number of chunks created
            
        Raises:
            Exception: If any step in the pipeline fails
        """
        try:
            # 1. Parse document
            text = DocumentService.parse_document(tmp_path)
            
            # 2. Split into chunks
            docs = DocumentService.split_text(text)
            
            # 3. Create vector store and persist
            DocumentService.create_vector_store(docs, doc_id)
            
            # 4. Update database
            with get_db_session() as session:
                doc = session.get(Doc, doc_id)
                if doc:
                    doc.status = "ready"
                    doc.n_chunks = len(docs)
                    session.add(doc)
                    session.commit()
            
            return len(docs)
            
        except Exception as exc:
            # Handle errors and update status
            with get_db_session() as session:
                doc = session.get(Doc, doc_id)
                if doc:
                    doc.status = "failed"
                    session.add(doc)
                    session.commit()
            raise exc
        finally:
            # Clean up temporary file
            Path(tmp_path).unlink(missing_ok=True)
    
    @staticmethod
    def ingest_url(doc_id: str, url: str) -> int:
        """Complete URL ingestion pipeline.
        
        **URL Processing Pipeline:**
        1. Fetch content from URL
        2. Split into chunks
        3. Create embeddings and vector store
        4. Update database status
        
        Args:
            doc_id: Unique document identifier
            url: URL to fetch content from
            
        Returns:
            Number of chunks created
            
        Raises:
            Exception: If any step in the pipeline fails
        """
        try:
            # 1. Fetch content from URL
            text = WebContentService.fetch_url_content(url)
            
            # 2. Split into chunks with metadata
            docs = DocumentService.split_text(text)
            
            # Add URL metadata to chunks
            for doc in docs:
                doc.metadata.update({
                    "source_type": "url",
                    "source": url
                })
            
            # 3. Create vector store and persist
            DocumentService.create_vector_store(docs, doc_id)
            
            # 4. Update database
            with get_db_session() as session:
                doc = session.get(Doc, doc_id)
                if doc:
                    doc.status = "ready"
                    doc.n_chunks = len(docs)
                    session.add(doc)
                    session.commit()
            
            return len(docs)
            
        except Exception as exc:
            # Handle errors and update status
            with get_db_session() as session:
                doc = session.get(Doc, doc_id)
                if doc:
                    doc.status = "failed"
                    session.add(doc)
                    session.commit()
            raise exc
    
    @staticmethod
    async def create_document_record(filename: str) -> str:
        """Create a new document record in the database.
        
        Args:
            filename: Original filename or URL
            
        Returns:
            Generated document ID
        """
        doc_id = str(uuid.uuid4())
        
        with get_db_session() as session:
            session.add(Doc(id=doc_id, name=filename, status="processing"))
            session.commit()
        
        return doc_id
    
    @staticmethod
    async def save_uploaded_file(file) -> str:
        """Save uploaded file to temporary location.
        
        Args:
            file: UploadFile object from FastAPI
            
        Returns:
            Path to temporary file
        """
        with NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            tmp.write(await file.read())
            return tmp.name 