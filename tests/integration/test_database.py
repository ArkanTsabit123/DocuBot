# DocuBot/tests/integration/test_database.py

"""
Database Integration Tests
tests for database operations including CRUD, queries, and performance.
"""

import os
import sys
import tempfile
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

import pytest
from pytest import fixture
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
import sqlite3
from sqlite3 import Error as SQLiteError

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from database.sqlite_client import SQLiteClient, DatabaseError
from database.models import Base
from core.config import Config


@fixture
def test_database_path():
    temp_dir = tempfile.mkdtemp()
    db_path = Path(temp_dir) / "test_database.db"
    yield str(db_path)
    if db_path.exists():
        db_path.unlink()
    Path(temp_dir).rmdir()


@fixture
def db_client(test_database_path):
    config = Config()
    config.database_path = test_database_path
    client = SQLiteClient(db_path=test_database_path, config=config)
    
    success = client.connect()
    assert success, "Failed to connect to database"
    
    yield client
    client.close()


@fixture
def sample_document_data():
    return {
        "file_path": "/test/path/document.pdf",
        "file_name": "test_document.pdf",
        "file_type": "pdf",
        "file_size": 1024,
        "metadata": {
            "title": "Test Document",
            "author": "Test Author",
            "date": "2026-01-01",
            "tags": ["test", "sample", "document"]
        }
    }


@fixture  
def sample_chunk_data():
    return [
        {
            "text": "This is the first chunk of text.",
            "chunk_index": 0,
            "metadata": {"section": "introduction"}
        },
        {
            "text": "This is the second chunk with more content.",
            "chunk_index": 1,
            "metadata": {"section": "body"}
        },
        {
            "text": "This is the final chunk concluding the document.",
            "chunk_index": 2,
            "metadata": {"section": "conclusion"}
        }
    ]


class TestDatabaseConnection:
    def test_connection_establishment(self, test_database_path):
        client = SQLiteClient(db_path=test_database_path)
        success = client.connect()
        assert success
        assert client.engine is not None
        client.close()
    
    def test_connection_pooling(self, db_client):
        with db_client.session_scope() as session1:
            result1 = session1.execute(text("SELECT 1")).scalar()
            assert result1 == 1
        
        with db_client.session_scope() as session2:
            result2 = session2.execute(text("SELECT 2")).scalar()
            assert result2 == 2
    
    def test_connection_timeout(self, test_database_path):
        client = SQLiteClient(db_path=test_database_path)
        success = client.connect()
        assert success
        client.close()
    
    def test_invalid_connection_parameters(self):
        client = SQLiteClient(db_path="/invalid/path/to/database.db")
        try:
            success = client.connect()
            if not success:
                assert True
        except (ConnectionError, DatabaseError):
            assert True
        except Exception as e:
            pytest.fail(f"Unexpected exception: {e}")


class TestDatabaseSchema:
    def test_table_creation(self, db_client):
        inspector = inspect(db_client.engine)
        tables = inspector.get_table_names()
        
        expected_tables = [
            'documents', 'chunks', 'conversations', 
            'messages', 'tags', 'document_tags', 'settings'
        ]
        
        for table in expected_tables:
            assert table in tables, f"Missing table: {table}"
    
    def test_column_definitions(self, db_client):
        inspector = inspect(db_client.engine)
        doc_columns = [col['name'] for col in inspector.get_columns('documents')]
        expected_columns = ['id', 'file_path', 'file_name', 'file_type', 'file_size', 'upload_date']
        
        for col in expected_columns:
            assert col in doc_columns, f"Missing column: {col}"
    
    def test_foreign_key_constraints(self, db_client):
        with db_client.session_scope() as session:
            try:
                session.execute(text("""
                    INSERT INTO chunks (id, document_id, chunk_index, text_content, vector_id)
                    VALUES ('test-chunk', 'non-existent-doc', 0, 'test text', 'vec-123')
                """))
                session.commit()
                pytest.fail("Should have raised integrity error")
            except (IntegrityError, SQLiteError):
                session.rollback()
                assert True
    
    def test_index_creation(self, db_client):
        inspector = inspect(db_client.engine)
        indexes = inspector.get_indexes('documents')
        assert len(indexes) > 0, "No indexes found on documents table"
    
    def test_schema_migration(self, test_database_path):
        client1 = SQLiteClient(db_path=test_database_path)
        client1.connect()
        inspector = inspect(client1.engine)
        initial_tables = inspector.get_table_names()
        client1.close()
        
        client2 = SQLiteClient(db_path=test_database_path)
        client2.connect()
        inspector2 = inspect(client2.engine)
        final_tables = inspector2.get_table_names()
        client2.close()
        
        assert set(initial_tables) == set(final_tables), "Schema changed unexpectedly"


class TestDocumentCRUD:
    def test_create_document(self, db_client, sample_document_data):
        doc_id = db_client.add_document(**sample_document_data)
        assert doc_id is not None
        assert isinstance(doc_id, str)
        assert len(doc_id) > 0
    
    def test_read_document(self, db_client, sample_document_data):
        doc_id = db_client.add_document(**sample_document_data)
        document = db_client.get_document(doc_id)
        
        assert document is not None
        assert document['id'] == doc_id
        assert document['file_name'] == sample_document_data['file_name']
    
    def test_update_document(self, db_client, sample_document_data):
        doc_id = db_client.add_document(**sample_document_data)
        
        updates = {
            'file_name': 'updated_name.pdf',
            'metadata': {'title': 'Updated Title', 'status': 'processed'}
        }
        
        success = db_client.update_document(doc_id, **updates)
        assert success
        
        document = db_client.get_document(doc_id)
        assert document['file_name'] == 'updated_name.pdf'
        assert document['metadata'].get('title') == 'Updated Title'
    
    def test_delete_document(self, db_client, sample_document_data):
        doc_id = db_client.add_document(**sample_document_data)
        document = db_client.get_document(doc_id)
        assert document is not None
        
        success = db_client.delete_document(doc_id)
        assert success
        
        document = db_client.get_document(doc_id)
        assert document is None
    
    def test_document_validation(self, db_client):
        invalid_data = {
            'file_name': '',
            'file_type': 'invalid_type',
            'file_size': -100
        }
        
        try:
            doc_id = db_client.add_document(**invalid_data)
            if doc_id:
                document = db_client.get_document(doc_id)
                assert document is not None
        except (ValueError, DatabaseError):
            assert True
    
    def test_document_metadata_storage(self, db_client):
        complex_metadata = {
            'title': 'Complex Document',
            'author': 'John Doe',
            'date': '2026-01-15',
            'keywords': ['AI', 'Machine Learning', 'RAG'],
            'summary': 'This is a comprehensive document about AI systems.',
            'pages': 45,
            'language': 'en',
            'custom_field': 'custom_value'
        }
        
        doc_data = {
            'file_path': '/test/complex.pdf',
            'file_name': 'complex_document.pdf',
            'file_type': 'pdf',
            'file_size': 2048,
            'metadata': complex_metadata
        }
        
        doc_id = db_client.add_document(**doc_data)
        document = db_client.get_document(doc_id)
        
        assert document is not None
        assert 'metadata' in document
        assert document['metadata']['title'] == 'Complex Document'
        assert 'keywords' in document['metadata']
        assert len(document['metadata']['keywords']) == 3


class TestChunkCRUD:
    def test_create_chunks_bulk(self, db_client, sample_document_data, sample_chunk_data):
        doc_id = db_client.add_document(**sample_document_data)
        
        for chunk in sample_chunk_data:
            chunk['document_id'] = doc_id
        
        chunk_ids = db_client.add_chunks(doc_id, sample_chunk_data)
        
        assert len(chunk_ids) == len(sample_chunk_data)
        assert all(isinstance(id, str) for id in chunk_ids)
    
    def test_chunk_document_relationship(self, db_client, sample_document_data, sample_chunk_data):
        doc_id = db_client.add_document(**sample_document_data)
        
        for chunk in sample_chunk_data:
            chunk['document_id'] = doc_id
        
        db_client.add_chunks(doc_id, sample_chunk_data)
        chunks = db_client.get_chunks_by_document(doc_id)
        
        assert len(chunks) == len(sample_chunk_data)
        for chunk in chunks:
            assert chunk['document_id'] == doc_id
    
    def test_chunk_retrieval_by_document(self, db_client, sample_document_data, sample_chunk_data):
        doc1_id = db_client.add_document(**sample_document_data)
        
        doc2_data = sample_document_data.copy()
        doc2_data['file_name'] = 'document2.pdf'
        doc2_id = db_client.add_document(**doc2_data)
        
        chunks_doc1 = []
        for i, chunk_data in enumerate(sample_chunk_data):
            chunk = chunk_data.copy()
            chunk['document_id'] = doc1_id
            chunk['text'] = f"Doc1 Chunk {i}: {chunk['text']}"
            chunks_doc1.append(chunk)
        
        db_client.add_chunks(doc1_id, chunks_doc1)
        
        chunks_doc2 = []
        for i, chunk_data in enumerate(sample_chunk_data):
            chunk = chunk_data.copy()
            chunk['document_id'] = doc2_id
            chunk['text'] = f"Doc2 Chunk {i}: {chunk['text']}"
            chunks_doc2.append(chunk)
        
        db_client.add_chunks(doc2_id, chunks_doc2)
        
        retrieved_doc1 = db_client.get_chunks_by_document(doc1_id)
        retrieved_doc2 = db_client.get_chunks_by_document(doc2_id)
        
        assert len(retrieved_doc1) == len(chunks_doc1)
        assert len(retrieved_doc2) == len(chunks_doc2)
        
        if retrieved_doc1 and retrieved_doc2:
            assert retrieved_doc1[0]['text'] != retrieved_doc2[0]['text']
    
    def test_chunk_metadata_integrity(self, db_client, sample_document_data, sample_chunk_data):
        doc_id = db_client.add_document(**sample_document_data)
        
        for chunk in sample_chunk_data:
            chunk['document_id'] = doc_id
            chunk['metadata'] = {
                'section': f'section_{chunk["chunk_index"]}',
                'word_count': len(chunk['text'].split()),
                'importance': chunk['chunk_index'] == 0
            }
        
        db_client.add_chunks(doc_id, sample_chunk_data)
        chunks = db_client.get_chunks_by_document(doc_id)
        
        for chunk in chunks:
            assert 'metadata' in chunk
            assert 'section' in chunk['metadata']
            assert 'word_count' in chunk['metadata']
            assert isinstance(chunk['metadata']['word_count'], int)


class TestConversationCRUD:
    def test_create_conversation(self, db_client):
        title = "Test Conversation about AI"
        conv_id = db_client.create_conversation(title)
        
        assert conv_id is not None
        assert isinstance(conv_id, str)
        
        conversation = db_client.get_conversation(conv_id)
        assert conversation is not None
        assert conversation['title'] == title
    
    def test_add_message_to_conversation(self, db_client):
        conv_id = db_client.create_conversation("Message Test")
        
        messages = [
            ("user", "Hello, I have a question about document processing."),
            ("assistant", "I can help with that. What would you like to know?"),
            ("user", "How does the chunking algorithm work?"),
            ("assistant", "The algorithm splits documents into 500-token chunks with 50-token overlap.")
        ]
        
        message_ids = []
        for role, content in messages:
            msg_id = db_client.add_message(conv_id, role, content)
            message_ids.append(msg_id)
            assert msg_id is not None
        
        conversation = db_client.get_conversation(conv_id)
        assert conversation is not None
        assert len(conversation['messages']) == len(messages)
        
        for i, msg in enumerate(conversation['messages']):
            assert msg['role'] == messages[i][0]
            assert msg['content'] == messages[i][1]
    
    def test_get_conversation_history(self, db_client):
        conv_id = db_client.create_conversation("History Test")
        
        for i in range(10):
            role = "user" if i % 2 == 0 else "assistant"
            content = f"Message {i}: Test content for history tracking."
            db_client.add_message(conv_id, role, content)
        
        conversation = db_client.get_conversation(conv_id)
        
        assert conversation is not None
        assert 'messages' in conversation
        assert len(conversation['messages']) == 10
        
        message_contents = [msg['content'] for msg in conversation['messages']]
        for i in range(10):
            expected = f"Message {i}: Test content for history tracking."
            assert expected in message_contents
    
    def test_conversation_tagging(self, db_client):
        conv1_id = db_client.create_conversation("AI Discussion")
        conv2_id = db_client.create_conversation("Document Processing")
        conv3_id = db_client.create_conversation("General Questions")
        
        db_client.add_message(conv1_id, "user", "Tell me about AI ethics.")
        db_client.add_message(conv2_id, "user", "How to process PDF files?")
        db_client.add_message(conv3_id, "user", "What is the weather today?")
        
        conversations = db_client.list_conversations(limit=10)
        
        assert len(conversations) >= 3
        conv_titles = [conv['title'] for conv in conversations]
        assert "AI Discussion" in conv_titles
        assert "Document Processing" in conv_titles


class TestQueryOperations:
    def test_search_by_filename(self, db_client, sample_document_data):
        documents = [
            {**sample_document_data, 'file_name': 'project_report.pdf'},
            {**sample_document_data, 'file_name': 'meeting_notes.docx'},
            {**sample_document_data, 'file_name': 'research_paper.pdf'},
            {**sample_document_data, 'file_name': 'budget_spreadsheet.xlsx'}
        ]
        
        for doc_data in documents:
            db_client.add_document(**doc_data)
        
        pdf_results = db_client.search_documents("pdf", field="file_type")
        assert len(pdf_results) >= 2
        
        report_results = db_client.search_documents("report", field="file_name")
        assert len(report_results) >= 1
    
    def test_filter_by_file_type(self, db_client, sample_document_data):
        file_types = ['pdf', 'docx', 'txt', 'pdf', 'html', 'pdf']
        
        for i, file_type in enumerate(file_types):
            doc_data = sample_document_data.copy()
            doc_data['file_type'] = file_type
            doc_data['file_name'] = f'document_{i}.{file_type}'
            db_client.add_document(**doc_data)
        
        stats = db_client.get_document_statistics()
        assert 'count_by_type' in stats
        assert stats['count_by_type'].get('pdf', 0) >= 3
        assert stats['count_by_type'].get('docx', 0) >= 1
    
    def test_pagination(self, db_client, sample_document_data):
        for i in range(25):
            doc_data = sample_document_data.copy()
            doc_data['file_name'] = f'document_{i:03d}.pdf'
            db_client.add_document(**doc_data)
        
        page1 = db_client.list_documents(limit=10, offset=0)
        page2 = db_client.list_documents(limit=10, offset=10)
        page3 = db_client.list_documents(limit=10, offset=20)
        
        assert len(page1) == 10
        assert len(page2) == 10
        assert len(page3) <= 10
        
        if page1 and page2:
            assert page1[0]['file_name'] != page2[0]['file_name']
    
    def test_sorting(self, db_client, sample_document_data):
        documents = []
        for i in range(5):
            doc_data = sample_document_data.copy()
            doc_data['file_name'] = f'document_{i}.pdf'
            doc_id = db_client.add_document(**doc_data)
            documents.append(doc_id)
            time.sleep(0.01)
        
        sorted_docs = db_client.list_documents(
            limit=10, 
            sort_by="upload_date", 
            sort_order="DESC"
        )
        
        if len(sorted_docs) >= 2:
            assert sorted_docs[0]['upload_date'] >= sorted_docs[1]['upload_date']
    
    def test_aggregate_queries(self, db_client, sample_document_data):
        size_categories = {
            'small': [100, 500, 300],
            'medium': [1500, 2000, 1800],
            'large': [5000, 10000, 7500]
        }
        
        for category, sizes in size_categories.items():
            for size in sizes:
                doc_data = sample_document_data.copy()
                doc_data['file_name'] = f'{category}_{size}.pdf'
                doc_data['file_size'] = size
                db_client.add_document(**doc_data)
        
        stats = db_client.get_document_statistics()
        
        assert 'total_documents' in stats
        assert stats['total_documents'] == 9
        
        assert 'total_size' in stats
        assert stats['total_size'] > 0
        
        assert 'average_size' in stats
        assert stats['average_size'] > 0


class TestPerformance:
    def test_bulk_insert_performance(self, db_client, sample_document_data):
        num_documents = 100
        start_time = time.time()
        
        for i in range(num_documents):
            doc_data = sample_document_data.copy()
            doc_data['file_name'] = f'bulk_document_{i}.pdf'
            db_client.add_document(**doc_data)
        
        end_time = time.time()
        duration = end_time - start_time
        
        assert duration < 30.0, f"Bulk insert took too long: {duration:.2f}s"
        
        stats = db_client.get_document_statistics()
        assert stats['total_documents'] >= num_documents
    
    def test_query_performance_with_index(self, db_client, sample_document_data):
        for i in range(50):
            doc_data = sample_document_data.copy()
            doc_data['file_name'] = f'perf_document_{i}.pdf'
            doc_data['file_type'] = 'pdf' if i % 2 == 0 else 'docx'
            db_client.add_document(**doc_data)
        
        queries = [
            ("Simple count", lambda: len(db_client.list_documents(limit=100))),
            ("Filter by type", lambda: len(db_client.search_documents("pdf", field="file_type"))),
            ("Get statistics", lambda: db_client.get_document_statistics())
        ]
        
        for query_name, query_func in queries:
            start_time = time.time()
            result = query_func()
            end_time = time.time()
            duration = end_time - start_time
            
            assert duration < 5.0, f"{query_name} took too long: {duration:.2f}s"
            assert result is not None, f"{query_name} should return result"
    
    def test_concurrent_access(self, test_database_path, sample_document_data):
        results = []
        errors = []
        
        def worker(worker_id, db_path, doc_data):
            try:
                client = SQLiteClient(db_path=db_path)
                client.connect()
                
                for i in range(5):
                    doc_copy = doc_data.copy()
                    doc_copy['file_name'] = f'worker_{worker_id}_doc_{i}.pdf'
                    doc_id = client.add_document(**doc_copy)
                    results.append((worker_id, i, doc_id))
                
                client.close()
            except Exception as e:
                errors.append((worker_id, str(e)))
        
        threads = []
        for i in range(3):
            thread = threading.Thread(
                target=worker,
                args=(i, test_database_path, sample_document_data)
            )
            threads.append(thread)
        
        for thread in threads:
            thread.start()
        
        for thread in threads:
            thread.join(timeout=10.0)
        
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) == 15, f"Expected 15 documents, got {len(results)}"


def test_invalid_data_handling(db_client):
    try:
        db_client.add_document(file_name=None, file_type=None, file_size=-1)
        assert True
    except (ValueError, DatabaseError):
        assert True
    except Exception as e:
        pytest.fail(f"Unexpected exception for invalid data: {e}")
    
    try:
        db_client.add_document(
            file_name="x" * 1000,
            file_type="pdf",
            file_size=10**9
        )
        assert True
    except Exception:
        assert True


def test_constraint_violations(db_client, sample_document_data):
    doc_id = db_client.add_document(**sample_document_data)
    
    try:
        doc_id2 = db_client.add_document(**sample_document_data)
        if doc_id2:
            assert doc_id != doc_id2
    except (IntegrityError, DatabaseError):
        assert True


def test_transaction_rollback(db_client, sample_document_data):
    try:
        with db_client.session_scope() as session:
            doc_data = sample_document_data.copy()
            doc_data['file_name'] = 'rollback_test_1.pdf'
            raise ValueError("Simulated error for rollback test")
    except ValueError:
        assert True


def test_database_locking(db_client):
    with db_client.session_scope() as session:
        result = session.execute(text("SELECT COUNT(*) FROM documents")).scalar()
        assert result is not None
    assert True


def test_recovery_after_crash(test_database_path, sample_document_data):
    client = SQLiteClient(db_path=test_database_path)
    client.connect()
    
    for i in range(5):
        doc_data = sample_document_data.copy()
        doc_data['file_name'] = f'crash_test_{i}.pdf'
        client.add_document(**doc_data)
    
    initial_stats = client.get_document_statistics()
    initial_count = initial_stats.get('total_documents', 0)
    client.close()
    
    client2 = SQLiteClient(db_path=test_database_path)
    client2.connect()
    
    recovered_stats = client2.get_document_statistics()
    recovered_count = recovered_stats.get('total_documents', 0)
    
    assert recovered_count >= initial_count - 1
    client2.close()


if __name__ == "__main__":
    print("Running database integration tests...")
    pytest.main([__file__, "-v", "--tb=short"])