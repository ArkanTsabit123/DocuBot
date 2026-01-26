# Docubot/tests/unit/test_database.py

"""
Unit tests for SQLiteClient
"""
import pytest
import tempfile
import json
from unittest.mock import Mock, patch
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.database.sqlite_client import SQLiteClient


class TestSQLiteClientUnit:
    """Unit tests for SQLiteClient"""
    
    def test_initialization_default_path(self):
        """Test client initialization with default path"""
        with patch('src.database.sqlite_client.Path') as mock_path:
            mock_path.return_value.parent.mkdir = Mock()
            mock_path.return_value.exists.return_value = True
            
            client = SQLiteClient()
            
            assert client.db_path is not None
            mock_path.return_value.parent.mkdir.assert_called_once_with(parents=True, exist_ok=True)
    
    def test_initialization_custom_path(self):
        """Test client initialization with custom path"""
        test_path = "/custom/path/test.db"
        client = SQLiteClient(test_path)
        
        assert str(client.db_path) == test_path
    
    def test_get_table_definitions(self):
        """Test table definition generation"""
        client = SQLiteClient(":memory:")
        tables = client._get_table_definitions()
        
        expected_tables = [
            'documents',
            'chunks', 
            'conversations',
            'messages',
            'settings',
            'tags',
            'document_tags'
        ]
        
        for table in expected_tables:
            assert table in tables
            assert 'CREATE TABLE' in tables[table]
    
    @patch('sqlite3.connect')
    def test_init_database(self, mock_connect):
        """Test database initialization"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.fetchone.return_value = None
        
        client = SQLiteClient(":memory:")
        
        # Verify connect was called
        mock_connect.assert_called_once()
        
        # Verify table creation attempts
        assert mock_cursor.execute.call_count >= 7  # For 7 tables
    
    def test_row_to_dict_json_parsing(self):
        """Test JSON field parsing in row_to_dict"""
        client = SQLiteClient(":memory:")
        
        # Create mock row with JSON fields
        mock_row = {
            'id': 'test-id',
            'metadata_json': '{"author": "Test", "pages": 10}',
            'vector_ids_json': '["vec1", "vec2"]',
            'tags_json': '["tag1", "tag2"]',
            'sources_json': '[{"doc": "test"}]',
            'regular_field': 'regular_value'
        }
        
        # Mock sqlite3.Row behavior
        class MockRow:
            def keys(self):
                return mock_row.keys()
            
            def __getitem__(self, key):
                return mock_row[key]
        
        result = client._row_to_dict(MockRow())
        
        # Test JSON parsing
        assert 'metadata' in result
        assert result['metadata']['author'] == "Test"
        assert result['metadata']['pages'] == 10
        
        assert 'vector_ids' in result
        assert result['vector_ids'] == ["vec1", "vec2"]
        
        assert 'tags' in result
        assert result['tags'] == ["tag1", "tag2"]
        
        assert 'sources' in result
        assert result['sources'][0]['doc'] == "test"
        
        # Test regular field preservation
        assert 'regular_field' in result
        assert result['regular_field'] == 'regular_value'
        
        # Test JSON fields removed
        assert 'metadata_json' not in result
        assert 'vector_ids_json' not in result
    
    def test_row_to_dict_invalid_json(self):
        """Test row_to_dict with invalid JSON"""
        client = SQLiteClient(":memory:")
        
        mock_row = {
            'metadata_json': 'invalid json',
            'vector_ids_json': None,
            'tags_json': ''
        }
        
        class MockRow:
            def keys(self):
                return mock_row.keys()
            
            def __getitem__(self, key):
                return mock_row[key]
        
        result = client._row_to_dict(MockRow())
        
        # Should handle invalid JSON gracefully
        assert 'metadata' in result
        assert result['metadata'] == []  # Default for invalid
        
        assert 'vector_ids' in result
        assert result['vector_ids'] == []  # Default for None
        
        assert 'tags' in result
        assert result['tags'] == []  # Default for empty
    
    @patch('sqlite3.connect')
    def test_add_document(self, mock_connect):
        """Test document addition"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        client = SQLiteClient(":memory:")
        
        # Mock uuid generation
        with patch('uuid.uuid4') as mock_uuid:
            mock_uuid.return_value = 'test-uuid-1234'
            
            doc_id = client.add_document(
                file_path="/test/path.pdf",
                file_name="test.pdf",
                file_type=".pdf",
                file_size=1024
            )
            
            assert doc_id == "test-uuid-1234"
            
            # Verify SQL execution
            mock_cursor.execute.assert_called_once()
            mock_conn.commit.assert_called_once()
    
    @patch('sqlite3.connect')
    def test_update_document_success(self, mock_connect):
        """Test successful document update"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        mock_cursor.rowcount = 1  # Simulate successful update
        
        client = SQLiteClient(":memory:")
        
        result = client.update_document(
            "test-id",
            {"processing_status": "completed", "word_count": 1000}
        )
        
        assert result is True
        mock_conn.commit.assert_called_once()
    
    @patch('sqlite3.connect')
    def test_update_document_failure(self, mock_connect):
        """Test failed document update"""
        mock_conn = Mock()
        mock_cursor = Mock()
        mock_connect.return_value = mock_conn
        mock_conn.cursor.return_value = mock_cursor
        
        # Simulate SQL error
        mock_cursor.execute.side_effect = Exception("SQL Error")
        
        client = SQLiteClient(":memory:")
        
        result = client.update_document("test-id", {"status": "error"})
        
        assert result is False
        mock_conn.rollback.assert_called_once()
    
    def test_get_database_client_singleton(self):
        """Test singleton pattern for get_database_client"""
        from src.database.sqlite_client import get_database_client
        
        # First call should create instance
        client1 = get_database_client()
        
        # Second call should return same instance
        client2 = get_database_client()
        
        assert client1 is client2
        
        # Call with different path should create new instance
        client3 = get_database_client("/different/path.db")
        assert client3 is not client1


def test_transaction_handling():
    """Test transaction handling"""
    # Mock test for transaction context
    assert True, "Transaction handling test placeholder"


def test_error_recovery():
    """Test database error recovery"""
    assert True, "Error recovery test placeholder"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])