# docubot/src/__init__.py

"""
Mock classes for DocuBot development environment.
Used when actual components cannot be imported.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional

from .document_processor import MockDocumentProcessor
from .llm_client import MockLLMClient
from .database_client import MockDatabaseClient

__all__ = ['MockDocumentProcessor', 'MockLLMClient', 'MockDatabaseClient']


class MockDocumentProcessor:
    """Mock document processor for development."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.supported_formats = ['.txt', '.pdf', '.docx']
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """Mock document processing."""
        file_size = Path(file_path).stat().st_size if Path(file_path).exists() else 0
        return {
            'status': 'success',
            'document_id': os.path.basename(file_path),
            'chunks_processed': 3,
            'processing_time': 1.5,
            'file_size': file_size
        }
    
    def get_supported_formats(self) -> List[str]:
        return self.supported_formats


class MockLLMClient:
    """Mock LLM client for development."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.models = ['llama2:7b', 'mistral:7b', 'neural-chat:7b']
    
    def generate(self, query: str, context: Optional[str] = None, **kwargs) -> str:
        """Mock response generation."""
        responses = [
            "This is a mock response from DocuBot.",
            "Based on the provided documents, the answer is...",
            "I cannot find relevant information in your documents.",
            "The document analysis suggests that..."
        ]
        import random
        return random.choice(responses)
    
    def list_models(self) -> List[str]:
        return self.models


class MockSQLiteClient:
    """Mock database client for development."""
    
    def __init__(self, db_path=None):
        self.db_path = db_path or "data/database/docubot.db"
        self.documents = []
    
    def add_document(self, file_path: str, metadata: Dict[str, Any]) -> str:
        """Mock document addition."""
        doc_id = f"doc_{len(self.documents) + 1}"
        self.documents.append({
            'id': doc_id,
            'file_path': file_path,
            'metadata': metadata
        })
        return doc_id
    
    def list_documents(self) -> List[Dict[str, Any]]:
        return self.documents
    
    def get_document_count(self) -> int:
        return len(self.documents)