# docubot/src/tests/conftest.py

"""
Pytest configuration and fixtures for DocuBot
Includes database fixtures for integration tests
"""

import pytest
import sys
import os
import tempfile
from pathlib import Path
from datetime import datetime
import json

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Now we can import project modules
from src.database.sqlite_client import SQLiteClient
from src.database.models import Base

# ============================================
# PYTEST CONFIGURATION
# ============================================

def pytest_configure(config):
    """Configure pytest"""
    # Add custom markers
    config.addinivalue_line(
        "markers", "integration: integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: unit tests"
    )
    config.addinivalue_line(
        "markers", "e2e: end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "database: database tests"
    )
    config.addinivalue_line(
        "markers", "slow: slow tests (skip with -m 'not slow')"
    )

# ============================================
# BASIC FIXTURES
# ============================================

@pytest.fixture
def project_root_path():
    """Return project root path"""
    return Path(__file__).parent.parent

@pytest.fixture
def test_data_dir():
    """Return test data directory"""
    return Path(__file__).parent / "test_data"

@pytest.fixture
def temp_dir():
    """Create temporary directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)

@pytest.fixture
def sample_text():
    """Sample text for testing"""
    return """This is a sample document for testing purposes.
It contains multiple paragraphs and sentences.
The quick brown fox jumps over the lazy dog.
Machine learning and artificial intelligence are transforming the world.
Natural language processing enables computers to understand human language.
This document will be used for chunking, embedding, and retrieval tests."""

# ============================================
# DATABASE FIXTURES
# ============================================

@pytest.fixture
def temp_db_path(temp_dir):
    """Create temporary database file path"""
    return temp_dir / "test_database.db"

@pytest.fixture
def sqlite_client(temp_db_path):
    """Create SQLiteClient with temporary database"""
    client = SQLiteClient(str(temp_db_path))
    client.connect()
    
    # Create tables
    with client.session_scope() as session:
        Base.metadata.create_all(client.engine)
    
    yield client
    
    # Cleanup
    client.close()

@pytest.fixture
def sample_document_data():
    """Sample document data for testing"""
    return {
        "id": "test_doc_001",
        "file_path": "/test/path/document.pdf",
        "file_name": "Test Document",
        "file_type": "pdf",
        "file_size": 1024,
        "processing_status": "completed",
        "word_count": 500,
        "language": "en",
        "summary": "This is a test document summary."
    }

@pytest.fixture
def sample_chunk_data():
    """Sample chunk data for testing"""
    return [
        {
            "id": "chunk_001_001",
            "chunk_index": 0,
            "text_content": "First chunk of text content.",
            "cleaned_text": "First chunk of text content.",
            "token_count": 10,
            "vector_id": "vec_001"
        },
        {
            "id": "chunk_001_002",
            "chunk_index": 1,
            "text_content": "Second chunk of text content.",
            "cleaned_text": "Second chunk of text content.",
            "token_count": 8,
            "vector_id": "vec_002"
        }
    ]

@pytest.fixture
def populated_database(sqlite_client, sample_document_data, sample_chunk_data):
    """Database populated with test data"""
    with sqlite_client.session_scope() as session:
        # Import here to avoid circular imports
        from src.database.models import Document, Chunk
        
        # Create document
        doc = Document(**sample_document_data)
        session.add(doc)
        
        # Create chunks
        for chunk_data in sample_chunk_data:
            chunk = Chunk(
                document_id=sample_document_data["id"],
                **chunk_data
            )
            session.add(chunk)
        
        session.commit()
    
    return sqlite_client

# ============================================
# CONFIGURATION FIXTURES
# ============================================

@pytest.fixture
def sample_config():
    """Sample configuration for testing"""
    return {
        "app": {
            "name": "DocuBot Test",
            "version": "1.0.0"
        },
        "document_processing": {
            "chunk_size": 500,
            "chunk_overlap": 50,
            "max_file_size_mb": 10
        },
        "ai": {
            "llm": {
                "model": "llama2:7b",
                "temperature": 0.1
            },
            "embeddings": {
                "model": "all-MiniLM-L6-v2"
            }
        }
    }

# ============================================
# UTILITY FUNCTIONS
# ============================================

@pytest.fixture
def current_timestamp():
    """Get current timestamp"""
    return datetime.now().isoformat()

@pytest.fixture
def json_encoder():
    """JSON encoder that handles datetime"""
    import json
    from datetime import datetime
    
    class DateTimeEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            return super().default(obj)
    
    return DateTimeEncoder