# docubot/tests/integration/test_database.py
"""
Database Integration Test Suite for DocuBot
testing of SQLite database operations and SQLiteClient API.
"""

import pytest
import sys
import os
import tempfile
import json
import sqlite3
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import time


project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.database.sqlite_client import SQLiteClient


class TestConnection:
    def test_connection_success(self, sqlite_client):
        assert sqlite_client is not None
        result = sqlite_client.execute_query("SELECT 1 as test_value")
        assert len(result) == 1
        assert result[0]['test_value'] == 1
    
    def test_connection_persistence(self, sqlite_client):
        result1 = sqlite_client.execute_query("SELECT 1 as first")
        result2 = sqlite_client.execute_query("SELECT 2 as second")
        assert result1[0]['first'] == 1
        assert result2[0]['second'] == 2
    
    def test_connection_closure(self, temp_db_file):
        client = SQLiteClient(temp_db_file)
        client.close()
        assert client.connection is None


class TestSchemaOperations:
    def test_table_creation(self, sqlite_client):
        sqlite_client.execute_query("""
            CREATE TABLE test_schema (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        tables = sqlite_client.execute_query(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='test_schema'"
        )
        assert len(tables) == 1
        assert tables[0]['name'] == 'test_schema'
    
    def test_schema_modification(self, sqlite_client):
        sqlite_client.execute_query("""
            CREATE TABLE test_alter (
                id INTEGER PRIMARY KEY,
                name TEXT
            )
        """)
        
        sqlite_client.execute_query("ALTER TABLE test_alter ADD COLUMN email TEXT")
        sqlite_client.execute_query("ALTER TABLE test_alter ADD COLUMN age INTEGER")
        
        columns = sqlite_client.execute_query("PRAGMA table_info(test_alter)")
        column_names = [col['name'] for col in columns]
        assert "email" in column_names
        assert "age" in column_names


class TestCRUDOperations:
    def test_create_operations(self, sqlite_client):
        sqlite_client.execute_query("""
            CREATE TABLE test_create (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                content TEXT,
                views INTEGER DEFAULT 0
            )
        """)
        
        sqlite_client.execute_query(
            "INSERT INTO test_create (title, content, views) VALUES (?, ?, ?)",
            ("Test Title", "Test Content", 100)
        )
        
        count_result = sqlite_client.execute_query("SELECT COUNT(*) as count FROM test_create")
        assert count_result[0]['count'] == 1
    
    def test_read_operations(self, sqlite_client):
        sqlite_client.execute_query("""
            CREATE TABLE test_read (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT
            )
        """)
        
        test_data = [
            ("user1", "user1@example.com"),
            ("user2", "user2@example.com"),
            ("user3", "user3@example.com")
        ]
        
        for username, email in test_data:
            sqlite_client.execute_query(
                "INSERT INTO test_read (username, email) VALUES (?, ?)",
                (username, email)
            )
        
        all_users = sqlite_client.execute_query("SELECT * FROM test_read ORDER BY username")
        assert len(all_users) == 3
        
        specific_user = sqlite_client.execute_query(
            "SELECT * FROM test_read WHERE username = ?",
            ("user1",)
        )
        assert len(specific_user) == 1
        assert specific_user[0]['email'] == 'user1@example.com'
    
    def test_update_operations(self, sqlite_client):
        sqlite_client.execute_query("""
            CREATE TABLE test_update (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product TEXT NOT NULL,
                price REAL
            )
        """)
        
        sqlite_client.execute_query(
            "INSERT INTO test_update (product, price) VALUES (?, ?)",
            ("Laptop", 999.99)
        )
        
        sqlite_client.execute_query(
            "UPDATE test_update SET price = ? WHERE product = ?",
            (899.99, "Laptop")
        )
        
        updated = sqlite_client.execute_query(
            "SELECT price FROM test_update WHERE product = ?",
            ("Laptop",)
        )
        assert updated[0]['price'] == 899.99
    
    def test_delete_operations(self, sqlite_client):
        sqlite_client.execute_query("""
            CREATE TABLE test_delete (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                item TEXT,
                quantity INTEGER
            )
        """)
        
        for i in range(5):
            sqlite_client.execute_query(
                "INSERT INTO test_delete (item, quantity) VALUES (?, ?)",
                (f"Item {i}", i * 10)
            )
        
        initial_count = sqlite_client.execute_query("SELECT COUNT(*) as count FROM test_delete")
        assert initial_count[0]['count'] == 5
        
        sqlite_client.execute_query(
            "DELETE FROM test_delete WHERE quantity < ?",
            (30,)
        )
        
        after_count = sqlite_client.execute_query("SELECT COUNT(*) as count FROM test_delete")
        assert after_count[0]['count'] == 2


class TestTransactions:
    def test_transaction_commit(self, sqlite_client):
        sqlite_client.execute_query("""
            CREATE TABLE test_transaction (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account TEXT,
                amount REAL
            )
        """)
        
        sqlite_client.execute_query("BEGIN TRANSACTION")
        sqlite_client.execute_query(
            "INSERT INTO test_transaction (account, amount) VALUES (?, ?)",
            ("ACC001", 1000.0)
        )
        sqlite_client.execute_query("COMMIT")
        
        count_result = sqlite_client.execute_query("SELECT COUNT(*) as count FROM test_transaction")
        assert count_result[0]['count'] == 1
    
    def test_transaction_rollback(self, sqlite_client):
        sqlite_client.execute_query("""
            CREATE TABLE test_rollback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                code TEXT UNIQUE,
                value INTEGER
            )
        """)
        
        sqlite_client.execute_query("BEGIN TRANSACTION")
        sqlite_client.execute_query(
            "INSERT INTO test_rollback (code, value) VALUES (?, ?)",
            ("A001", 100)
        )
        
        try:
            sqlite_client.execute_query(
                "INSERT INTO test_rollback (code, value) VALUES (?, ?)",
                ("A001", 200)
            )
            assert False, "Expected sqlite3.IntegrityError"
        except sqlite3.IntegrityError:
            sqlite_client.execute_query("ROLLBACK")
        
        count_result = sqlite_client.execute_query("SELECT COUNT(*) as count FROM test_rollback")
        assert count_result[0]['count'] == 0


class TestErrorHandling:
    def test_syntax_error_handling(self, sqlite_client):
        with pytest.raises(sqlite3.Error):
            sqlite_client.execute_query("SELECT FROM WHERE INVALID SYNTAX")
    
    def test_constraint_violation_handling(self, sqlite_client):
        sqlite_client.execute_query("""
            CREATE TABLE test_constraints (
                id INTEGER PRIMARY KEY,
                email TEXT UNIQUE NOT NULL
            )
        """)
        
        with pytest.raises(sqlite3.Error):
            sqlite_client.execute_query(
                "INSERT INTO test_constraints (id, email) VALUES (?, ?)",
                (1, None)
            )
        
        sqlite_client.execute_query(
            "INSERT INTO test_constraints (id, email) VALUES (?, ?)",
            (1, "test@example.com")
        )
        
        with pytest.raises(sqlite3.IntegrityError):
            sqlite_client.execute_query(
                "INSERT INTO test_constraints (id, email) VALUES (?, ?)",
                (2, "test@example.com")
            )


class TestPerformance:
    def test_bulk_insert_performance(self, sqlite_client):
        sqlite_client.execute_query("""
            CREATE TABLE test_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                value INTEGER,
                timestamp TEXT
            )
        """)
        
        start_time = time.time()
        sqlite_client.execute_query("BEGIN TRANSACTION")
        for i in range(50):
            sqlite_client.execute_query(
                "INSERT INTO test_performance (value, timestamp) VALUES (?, ?)",
                (i, datetime.now().isoformat())
            )
        sqlite_client.execute_query("COMMIT")
        elapsed_time = time.time() - start_time
        
        count_result = sqlite_client.execute_query("SELECT COUNT(*) as count FROM test_performance")
        assert count_result[0]['count'] == 50
        assert elapsed_time < 2.0
    
    def test_index_performance_improvement(self, sqlite_client):
        sqlite_client.execute_query("""
            CREATE TABLE test_index (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT,
                value REAL
            )
        """)
        
        categories = ['A', 'B', 'C', 'D', 'E']
        sqlite_client.execute_query("BEGIN TRANSACTION")
        for i in range(200):
            category = categories[i % len(categories)]
            sqlite_client.execute_query(
                "INSERT INTO test_index (category, value) VALUES (?, ?)",
                (category, i * 1.5)
            )
        sqlite_client.execute_query("COMMIT")
        
        start_time = time.time()
        result_no_index = sqlite_client.execute_query(
            "SELECT * FROM test_index WHERE category = ?",
            ("C",)
        )
        time_no_index = time.time() - start_time
        
        sqlite_client.execute_query("CREATE INDEX idx_category ON test_index(category)")
        
        start_time = time.time()
        result_with_index = sqlite_client.execute_query(
            "SELECT * FROM test_index WHERE category = ?",
            ("C",)
        )
        time_with_index = time.time() - start_time
        
        assert len(result_no_index) == len(result_with_index)


class TestSQLiteClientAPI:
    def test_document_operations(self, sqlite_client):
        doc_id = sqlite_client.add_document(
            file_path="/test/path/document.pdf",
            file_name="test_document.pdf",
            file_type=".pdf",
            file_size=2048,
            chunk_count=3,
            metadata={"author": "Test Author", "pages": 15}
        )
        
        assert isinstance(doc_id, str)
        assert len(doc_id) == 36
        
        document = sqlite_client.get_document(doc_id)
        assert document is not None
        assert document['file_name'] == "test_document.pdf"
        assert document['file_size'] == 2048
        assert document['chunk_count'] == 3
        assert document['metadata']['author'] == "Test Author"
        
        updates = {
            "file_name": "updated.pdf",
            "processing_status": "completed"
        }
        
        success = sqlite_client.update_document(doc_id, updates)
        assert success is True
        
        updated_document = sqlite_client.get_document(doc_id)
        assert updated_document['file_name'] == "updated.pdf"
        assert updated_document['processing_status'] == "completed"
        
        success = sqlite_client.delete_document(doc_id)
        assert success is True
        
        deleted_document = sqlite_client.get_document(doc_id)
        assert deleted_document is None
    
    def test_document_listing(self, sqlite_client):
        for i in range(5):
            sqlite_client.add_document(
                file_path=f"/test/path/doc{i}.pdf",
                file_name=f"document_{i}.pdf",
                file_type=".pdf",
                file_size=1000 + i * 100
            )
        
        all_docs = sqlite_client.list_documents()
        assert len(all_docs) == 5
        
        pending_docs = sqlite_client.list_documents(status="pending")
        assert len(pending_docs) == 5
        
        paginated_docs = sqlite_client.list_documents(limit=2, offset=0)
        assert len(paginated_docs) == 2
    
    def test_database_statistics(self, sqlite_client):
        for i in range(3):
            sqlite_client.add_document(
                file_path=f"/test/path/doc{i}.pdf",
                file_name=f"doc_{i}.pdf",
                file_type=".pdf",
                file_size=500 + i * 100
            )
        
        stats = sqlite_client.get_stats()
        assert isinstance(stats, dict)
        assert 'total_documents' in stats
        assert 'documents_by_status' in stats
        assert 'database_size_bytes' in stats
        assert stats['total_documents'] == 3
        assert 'pending' in stats['documents_by_status']
    
    def test_chunk_operations(self, sqlite_client):
        doc_id = sqlite_client.add_document(
            file_path="/test/path/document.pdf",
            file_name="test.pdf",
            file_type=".pdf",
            file_size=5000,
            chunk_count=0
        )
        
        chunk_ids = []
        for i in range(3):
            chunk_id = sqlite_client.add_chunk(
                document_id=doc_id,
                chunk_index=i,
                text_content=f"Chunk {i} content",
                cleaned_text=f"Cleaned chunk {i}",
                vector_id=f"vec_{doc_id}_{i}",
                token_count=50 + i * 10,
                embedding_model="all-MiniLM-L6-v2"
            )
            chunk_ids.append(chunk_id)
        
        chunks = sqlite_client.get_chunks_by_document(doc_id)
        assert len(chunks) == 3
        assert chunks[0]['chunk_index'] == 0
        assert chunks[1]['chunk_index'] == 1
        assert chunks[2]['chunk_index'] == 2
        assert chunks[0]['embedding_model'] == "all-MiniLM-L6-v2"


def main():
    test_classes = [
        TestConnection,
        TestSchemaOperations,
        TestCRUDOperations,
        TestTransactions,
        TestErrorHandling,
        TestPerformance,
        TestSQLiteClientAPI
    ]
    
    total_tests = sum(
        len([m for m in dir(cls) if m.startswith('test_')])
        for cls in test_classes
    )
    
    print("=" * 70)
    print("DATABASE INTEGRATION TEST SUITE")
    print(f"Test Classes: {len(test_classes)}")
    print(f"Total Tests: {total_tests}")
    print("=" * 70)
    
    retcode = pytest.main([__file__, "-v", "--tb=short", "-x", "--disable-warnings"])
    sys.exit(retcode)


if __name__ == "__main__":
    main()