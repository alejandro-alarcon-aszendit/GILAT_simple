"""Web content fetching service using LangChain WebBaseLoader."""

import re
from typing import List
from urllib.parse import urlparse
from langchain_community.document_loaders import WebBaseLoader
from langchain.docstore.document import Document


class WebContentService:
    """Service for fetching content from URLs."""
    
    @staticmethod
    def is_valid_url(url: str) -> bool:
        """Check if the provided string is a valid HTTP/HTTPS URL.
        
        Args:
            url: String to validate as URL
            
        Returns:
            True if valid HTTP/HTTPS URL, False otherwise
        """
        try:
            result = urlparse(url)
            return all([
                result.scheme in ['http', 'https'],  # Only allow HTTP/HTTPS
                result.netloc
            ])
        except Exception:
            return False
    
    @staticmethod
    def fetch_url_content(url: str) -> str:
        """Fetch content from a single URL.
        
        Args:
            url: URL to fetch content from
            
        Returns:
            Extracted text content from the webpage
            
        Raises:
            ValueError: If URL is invalid or content fetching fails
        """
        if not WebContentService.is_valid_url(url):
            raise ValueError(f"Invalid URL format: {url}")
        
        try:
            # Use WebBaseLoader to fetch and parse content
            loader = WebBaseLoader(url)
            documents = loader.load()
            
            if not documents:
                raise ValueError(f"No content found at URL: {url}")
            
            # Combine all document content
            content = "\n\n".join([doc.page_content for doc in documents])
            
            if not content.strip():
                raise ValueError(f"Empty content retrieved from URL: {url}")
            
            return content
            
        except Exception as e:
            raise ValueError(f"Failed to fetch content from {url}: {str(e)}")
    
    @staticmethod
    def fetch_multiple_urls(urls: List[str]) -> List[Document]:
        """Fetch content from multiple URLs.
        
        Args:
            urls: List of URLs to fetch content from
            
        Returns:
            List of Document objects with content and metadata
        """
        documents = []
        
        for url in urls:
            try:
                content = WebContentService.fetch_url_content(url)
                doc = Document(
                    page_content=content,
                    metadata={
                        "source": url,
                        "type": "web_content"
                    }
                )
                documents.append(doc)
            except ValueError as e:
                print(f"Warning: {e}")
                continue
        
        return documents 