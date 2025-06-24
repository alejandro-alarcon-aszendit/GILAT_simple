"""Service layer for document processing operations.

Provides services for document ingestion, processing, and content management.
"""

from .document_service import DocumentService
from .web_content_service import WebContentService

__all__ = [
    'DocumentService',
    'WebContentService',
] 