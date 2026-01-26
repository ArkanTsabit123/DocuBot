# docubot/tests/integration/conftest.py

"""
Test configuration and fixtures for integration tests.
"""

import sqlite3
import pytest
import tempfile
import os
from pathlib import Path
import sys
import json
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.database.sqlite_client import SQLiteClient


@pytest.fixture(scope="session")
def project_root_path():
    """Get project root directory."""
    return project_root


@pytest.fixture(scope="function")
def temp_db_file():
    """Create a temporary database file."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    yield db_path
    
    # Cleanup
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture(scope="function")
def sqlite_client(temp_db_file):
    """Create SQLiteClient instance."""
    client = SQLiteClient(temp_db_file)
    yield client
    client.close()


@pytest.fixture(scope="function")
def populated_database(sqlite_client):
    """Create database with populated tables."""
    # Create all standard tables
    sqlite_client._init_database()
    
    # Add some test documents
    for i in range(3):
        sqlite_client.add_document(
            file_path=f"/test/docs/doc{i}.pdf",
            file_name=f"test_document_{i}.pdf",
            file_type=".pdf",
            file_size=1024 * (i + 1),
            chunk_count=i * 2 + 1,
            metadata={
                "title": f"Test Document {i}",
                "author": f"Author {i}",
                "year": 2024
            }
        )
    
    # Add a test conversation
    conv_id = sqlite_client.add_conversation(
        title="Test Conversation",
        tags=["test", "integration", "database"]
    )
    
    # Add some messages
    sqlite_client.add_message(
        conversation_id=conv_id,
        role="user",
        content="What is the meaning of life?",
        tokens=7,
        model_used="llama2:7b"
    )
    
    sqlite_client.add_message(
        conversation_id=conv_id,
        role="assistant",
        content="The meaning of life is to learn, grow, and contribute.",
        tokens=12,
        model_used="llama2:7b",
        sources=[{"document_id": "test123", "relevance": 0.85}]
    )
    
    return sqlite_client


@pytest.fixture(scope="function")
def mock_sqlite_connection():
    """Mock SQLite connection for testing."""
    import sqlite3
    from unittest.mock import Mock, patch
    
    with patch('sqlite3.connect') as mock_connect:
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # Setup default mock behavior
        mock_row = Mock()
        mock_row.keys.return_value = ['id', 'name']
        mock_cursor.fetchall.return_value = [mock_row]
        mock_cursor.fetchone.return_value = mock_row
        mock_cursor.description = [('id',), ('name',)]
        
        yield {
            'connect': mock_connect,
            'connection': mock_conn,
            'cursor': mock_cursor
        }


@pytest.fixture(scope="function")
def sample_document_data():
    """Sample document data for testing."""
    return {
        "file_path": "/test/path/sample.pdf",
        "file_name": "sample_document.pdf",
        "file_type": ".pdf",
        "file_size": 2048,
        "chunk_count": 5,
        "metadata": {
            "title": "Sample Document",
            "author": "Test Author",
            "pages": 10,
            "language": "en"
        }
    }


@pytest.fixture(scope="session")
def test_config():
    """Test configuration."""
    return {
        "database": {
            "test_prefix": "test_",
            "cleanup": True
        },
        "performance": {
            "timeout_seconds": 30,
            "bulk_insert_count": 100
        }
    }

@pytest.fixture
def temp_database():
    """Alias for temp_db_file for compatibility with validators."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    yield db_path
    
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def database_client():
    """Alias for sqlite_client for compatibility with validators."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    client = SQLiteClient(db_path)
    yield client
    client.close()
    
    if os.path.exists(db_path):
        os.unlink(db_path)


@pytest.fixture
def test_database_schema():
    """Create database with basic schema for testing."""
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    client = SQLiteClient(db_path)
    
    # Create basic tables
    client.execute_query("""
        CREATE TABLE test_table (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            value INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    client.execute_query("""
        CREATE TABLE test_users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE NOT NULL,
            email TEXT
        )
    """)
    
    # Insert sample data
    client.execute_query(
        "INSERT INTO test_users (username, email) VALUES (?, ?)",
        ("testuser", "test@example.com")
    )
    
    yield client
    
    client.close()
    if os.path.exists(db_path):
        os.unlink(db_path)

def pytest_configure(config):
    """Configure pytest."""
    # Add custom markers
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "database: marks tests as database tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    for item in items:
        # Mark all tests in this directory as integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        if "database" in str(item.fspath) or "test_database" in str(item.fspath):
            item.add_marker(pytest.mark.database)