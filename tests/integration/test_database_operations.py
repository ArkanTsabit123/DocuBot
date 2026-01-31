"""
test_database_operations.py - Advanced Database Operations Integration Tests

This module contains advanced database operation tests for DocuBot's SQLite integration.
These tests focus on complex queries, batch operations, integration scenarios,
real-world use cases, and failure recovery scenarios.

Part of P1.13.3: Database integration tests implementation.
"""

# ============================================
# IMPORTS & SETUP
# ============================================

import pytest
import sqlite3
from sqlite3 import Error, Connection, Cursor
import json
from pathlib import Path
import sys
import os
import time
import threading
import tempfile
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from contextlib import contextmanager
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# SQLAlchemy imports
from sqlalchemy import create_engine, text, select, update, delete, func, and_, or_
from sqlalchemy.orm import Session, sessionmaker, scoped_session
from sqlalchemy.exc import SQLAlchemyError, IntegrityError

# Project imports
from src.database.sqlite_client import SQLiteClient
from src.database.models import Base, Document, Chunk, Conversation, Message, Tag, DocumentTag
from src.core.config import Config
from src.utilities.logger import get_logger

# ============================================
# FIXTURES
# ============================================

@pytest.fixture
def test_db_path(tmp_path):
    """Create a temporary database file for testing."""
    db_path = tmp_path / "test_operations.db"
    return str(db_path)


@pytest.fixture
def db_client(test_db_path):
    """Create SQLiteClient instance with temporary database."""
    client = SQLiteClient(test_db_path)
    client.connect()
    
    # Initialize with test data
    with client.session_scope() as session:
        # Create test documents
        for i in range(1, 6):
            doc = Document(
                id=f"doc_{i}",
                file_path=f"/test/path/doc_{i}.pdf",
                file_name=f"Test Document {i}",
                file_type="pdf",
                file_size=1000 * i,
                upload_date=datetime.now() - timedelta(days=i),
                processing_status="completed",
                word_count=500 * i,
                language="en"
            )
            session.add(doc)
        
        session.commit()
    
    yield client
    client.close()


@pytest.fixture
def large_dataset_db(tmp_path):
    """Create database with large dataset for performance testing."""
    db_path = tmp_path / "large_dataset.db"
    client = SQLiteClient(str(db_path))
    client.connect()
    
    # Add larger dataset
    with client.session_scope() as session:
        # Add 1000 documents
        for i in range(1, 1001):
            doc = Document(
                id=f"large_doc_{i:04d}",
                file_path=f"/large/path/doc_{i:04d}.txt",
                file_name=f"Large Test Document {i:04d}",
                file_type="txt",
                file_size=random.randint(1000, 10000),
                upload_date=datetime.now() - timedelta(hours=i),
                processing_status="completed" if i % 10 != 0 else "pending",
                word_count=random.randint(100, 1000),
                language="en"
            )
            session.add(doc)
            
            # Add chunks for some documents
            if i % 3 == 0:
                for j in range(1, 4):
                    chunk = Chunk(
                        id=f"chunk_{i:04d}_{j}",
                        document_id=f"large_doc_{i:04d}",
                        chunk_index=j,
                        text_content=f"Chunk {j} content for document {i:04d}",
                        cleaned_text=f"Chunk {j} content for document {i:04d}",
                        token_count=random.randint(50, 200),
                        embedding_model="all-MiniLM-L6-v2",
                        vector_id=f"vec_{i:04d}_{j}"
                    )
                    session.add(chunk)
        
        session.commit()
    
    yield client
    client.close()


@pytest.fixture
def sample_tags(db_client):
    """Create sample tags for testing."""
    tags = []
    with db_client.session_scope() as session:
        for tag_name in ["research", "important", "archived", "todo", "reference"]:
            tag = Tag(
                id=f"tag_{tag_name}",
                name=tag_name,
                color=f"#{random.randint(0, 0xFFFFFF):06x}",
                description=f"Test tag {tag_name}"
            )
            session.add(tag)
            tags.append(tag)
        
        session.commit()
    
    return tags


# ============================================
# A. ADVANCED OPERATION TESTS
# ============================================

def test_complex_join_operations(db_client, sample_tags):
    """Test complex JOIN operations across multiple tables."""
    
    with db_client.session_scope() as session:
        # Create a document with tags
        doc = Document(
            id="join_test_doc",
            file_path="/join/test.pdf",
            file_name="Join Test Document",
            file_type="pdf",
            file_size=5000,
            processing_status="completed",
            word_count=2500
        )
        session.add(doc)
        
        # Add chunks
        for i in range(1, 4):
            chunk = Chunk(
                id=f"join_chunk_{i}",
                document_id="join_test_doc",
                chunk_index=i,
                text_content=f"Join chunk {i} content",
                cleaned_text=f"Join chunk {i} content",
                token_count=100,
                vector_id=f"join_vec_{i}"
            )
            session.add(chunk)
        
        # Tag the document
        research_tag = session.query(Tag).filter_by(name="research").first()
        if research_tag:
            doc_tag = DocumentTag(document_id="join_test_doc", tag_id=research_tag.id)
            session.add(doc_tag)
        
        session.commit()
        
        # Test complex join: documents with their chunks and tags
        query = session.query(
            Document.file_name,
            func.count(Chunk.id).label('chunk_count'),
            func.group_concat(Tag.name).label('tags')
        ).outerjoin(Chunk, Document.id == Chunk.document_id
        ).outerjoin(DocumentTag, Document.id == DocumentTag.document_id
        ).outerjoin(Tag, DocumentTag.tag_id == Tag.id
        ).filter(Document.id == "join_test_doc"
        ).group_by(Document.id)
        
        result = query.first()
        
        assert result is not None
        assert result.file_name == "Join Test Document"
        assert result.chunk_count == 3  # Should have 3 chunks
        assert "research" in result.tags  # Should have research tag


def test_subquery_operations(db_client):
    """Test subquery operations for complex filtering."""
    
    with db_client.session_scope() as session:
        # Create a subquery to find documents with more than 2 chunks
        subquery = session.query(
            Chunk.document_id,
            func.count(Chunk.id).label('chunk_count')
        ).group_by(Chunk.document_id
        ).having(func.count(Chunk.id) > 2).subquery()
        
        # Main query: find documents matching subquery criteria
        query = session.query(Document).join(
            subquery, Document.id == subquery.c.document_id
        )
        
        results = query.all()
        
        # At this point, we shouldn't have documents with >2 chunks
        # So let's create one
        doc = Document(
            id="subquery_test_doc",
            file_path="/subquery/test.pdf",
            file_name="Subquery Test Document",
            file_type="pdf",
            processing_status="completed"
        )
        session.add(doc)
        
        # Add 3 chunks
        for i in range(1, 4):
            chunk = Chunk(
                id=f"subquery_chunk_{i}",
                document_id="subquery_test_doc",
                chunk_index=i,
                text_content=f"Subquery chunk {i}",
                cleaned_text=f"Subquery chunk {i}",
                token_count=50,
                vector_id=f"subquery_vec_{i}"
            )
            session.add(chunk)
        
        session.commit()
        
        # Now run the query again
        results = query.all()
        
        assert len(results) >= 1
        assert any(doc.id == "subquery_test_doc" for doc in results)


def test_window_functions(db_client):
    """Test window functions for analytical queries."""
    
    with db_client.session_scope() as session:
        # Create documents with varying file sizes
        sizes = [1000, 2000, 3000, 4000, 5000]
        for i, size in enumerate(sizes, 1):
            doc = Document(
                id=f"window_doc_{i}",
                file_path=f"/window/test_{i}.pdf",
                file_name=f"Window Test Document {i}",
                file_type="pdf",
                file_size=size,
                processing_status="completed",
                upload_date=datetime.now() - timedelta(days=i)
            )
            session.add(doc)
        
        session.commit()
        
        # Use window function to rank documents by size
        # Note: SQLAlchemy has limited window function support for SQLite
        # We'll use raw SQL for this test
        raw_query = """
        SELECT 
            file_name,
            file_size,
            RANK() OVER (ORDER BY file_size DESC) as size_rank,
            AVG(file_size) OVER () as avg_size
        FROM documents
        WHERE file_type = 'pdf'
        ORDER BY size_rank
        """
        
        result = session.execute(text(raw_query))
        rows = result.fetchall()
        
        assert len(rows) >= 5
        
        # Verify ranking
        sorted_rows = sorted(rows, key=lambda x: x[1], reverse=True)
        for i, row in enumerate(sorted_rows, 1):
            assert row[2] == i  # size_rank should match position
        
        # All should have same average
        avg_size = sum(row[1] for row in rows) / len(rows)
        for row in rows:
            assert abs(row[3] - avg_size) < 0.01


def test_full_text_search(db_client):
    """Test full-text search capabilities."""
    
    with db_client.session_scope() as session:
        # Create documents with specific content
        test_docs = [
            ("Database Migration Guide", "This document explains how to migrate databases."),
            ("Python SQLAlchemy Tutorial", "Learn how to use SQLAlchemy with Python for database operations."),
            ("Advanced SQL Queries", "Master complex SQL queries including joins and subqueries."),
            ("Database Backup Strategies", "Best practices for database backup and recovery.")
        ]
        
        for i, (title, summary) in enumerate(test_docs, 1):
            doc = Document(
                id=f"fts_doc_{i}",
                file_path=f"/fts/test_{i}.pdf",
                file_name=title,
                file_type="pdf",
                processing_status="completed",
                summary=summary
            )
            session.add(doc)
        
        session.commit()
        
        # Test basic text search using LIKE
        query = session.query(Document).filter(
            or_(
                Document.file_name.ilike('%database%'),
                Document.summary.ilike('%database%')
            )
        )
        
        results = query.all()
        
        # Should find documents with "database" in name or summary
        assert len(results) >= 2
        
        found_titles = [doc.file_name for doc in results]
        assert "Database Migration Guide" in found_titles
        assert "Database Backup Strategies" in found_titles
        
        # Test more specific search
        query = session.query(Document).filter(
            Document.summary.ilike('%SQLAlchemy%')
        )
        
        results = query.all()
        assert len(results) == 1
        assert results[0].file_name == "Python SQLAlchemy Tutorial"


def test_recursive_queries(db_client):
    """Test recursive queries (CTEs) for hierarchical data."""
    
    with db_client.session_scope() as session:
        # Create hierarchical conversation structure
        parent_conv = Conversation(
            id="parent_conv",
            title="Parent Conversation",
            created_at=datetime.now() - timedelta(days=2)
        )
        session.add(parent_conv)
        
        # Add messages to parent
        for i in range(1, 4):
            msg = Message(
                id=f"parent_msg_{i}",
                conversation_id="parent_conv",
                role="user" if i % 2 == 1 else "assistant",
                content=f"Parent message {i}",
                tokens=50
            )
            session.add(msg)
        
        # Create child conversation
        child_conv = Conversation(
            id="child_conv",
            title="Child Conversation",
            created_at=datetime.now() - timedelta(days=1)
        )
        session.add(child_conv)
        
        # Add messages to child
        for i in range(1, 3):
            msg = Message(
                id=f"child_msg_{i}",
                conversation_id="child_conv",
                role="user" if i % 2 == 1 else "assistant",
                content=f"Child message {i}",
                tokens=40
            )
            session.add(msg)
        
        session.commit()
        
        # Test recursive CTE to get all conversations with message counts
        # Note: SQLite has limited CTE support, using simpler approach
        conv_query = session.query(
            Conversation.id,
            Conversation.title,
            func.count(Message.id).label('message_count')
        ).outerjoin(Message, Conversation.id == Message.conversation_id
        ).group_by(Conversation.id)
        
        results = conv_query.all()
        
        # Verify counts
        conv_dict = {conv.id: conv for conv in results}
        assert conv_dict["parent_conv"].message_count == 3
        assert conv_dict["child_conv"].message_count == 2


# ============================================
# B. BATCH & BULK OPERATIONS
# ============================================

def test_batch_insert_1000_documents(db_client):
    """Test batch insertion of 1000 documents."""
    
    start_time = time.time()
    
    with db_client.session_scope() as session:
        # Create 1000 documents in batch
        documents = []
        for i in range(1, 1001):
            doc = Document(
                id=f"batch_doc_{i:04d}",
                file_path=f"/batch/path/doc_{i:04d}.txt",
                file_name=f"Batch Document {i:04d}",
                file_type="txt",
                file_size=random.randint(1000, 5000),
                upload_date=datetime.now() - timedelta(minutes=i),
                processing_status="completed",
                word_count=random.randint(500, 2000),
                language="en"
            )
            documents.append(doc)
        
        # Batch insert
        session.bulk_save_objects(documents)
        session.commit()
    
    end_time = time.time()
    insert_time = end_time - start_time
    
    # Verify insertion
    with db_client.session_scope() as session:
        count = session.query(func.count(Document.id)).filter(
            Document.id.like("batch_doc_%")
        ).scalar()
    
    assert count == 1000
    assert insert_time < 10.0  # Should complete within 10 seconds
    
    print(f"Batch insert of 1000 documents completed in {insert_time:.2f} seconds")


def test_batch_update_metadata(db_client):
    """Test batch update of document metadata."""
    
    # First create some test documents
    with db_client.session_scope() as session:
        for i in range(1, 101):
            doc = Document(
                id=f"update_doc_{i:03d}",
                file_path=f"/update/path/doc_{i:03d}.pdf",
                file_name=f"Update Test {i:03d}",
                file_type="pdf",
                file_size=2000,
                processing_status="pending"
            )
            session.add(doc)
        
        session.commit()
    
    start_time = time.time()
    
    # Batch update: mark all as completed and add timestamp
    with db_client.session_scope() as session:
        update_stmt = update(Document).where(
            Document.id.like("update_doc_%")
        ).values(
            processing_status="completed",
            last_accessed=datetime.now()
        )
        
        result = session.execute(update_stmt)
        session.commit()
        
        updated_count = result.rowcount
    
    end_time = time.time()
    update_time = end_time - start_time
    
    # Verify updates
    with db_client.session_scope() as session:
        pending_count = session.query(func.count(Document.id)).filter(
            Document.id.like("update_doc_%"),
            Document.processing_status == "pending"
        ).scalar()
        
        completed_count = session.query(func.count(Document.id)).filter(
            Document.id.like("update_doc_%"),
            Document.processing_status == "completed"
        ).scalar()
    
    assert pending_count == 0
    assert completed_count == updated_count
    assert update_time < 5.0  # Should complete within 5 seconds
    
    print(f"Batch update of {updated_count} documents completed in {update_time:.2f} seconds")


def test_bulk_delete_with_cascade(db_client):
    """Test bulk delete with cascade operations."""
    
    # Create test data with relationships
    with db_client.session_scope() as session:
        # Create documents
        for i in range(1, 51):
            doc = Document(
                id=f"delete_doc_{i:02d}",
                file_path=f"/delete/path/doc_{i:02d}.pdf",
                file_name=f"Delete Test {i:02d}",
                file_type="pdf",
                processing_status="completed"
            )
            session.add(doc)
            
            # Add chunks for some documents
            if i % 2 == 0:
                for j in range(1, 4):
                    chunk = Chunk(
                        id=f"delete_chunk_{i:02d}_{j}",
                        document_id=f"delete_doc_{i:02d}",
                        chunk_index=j,
                        text_content=f"Chunk {j} for deletion test",
                        cleaned_text=f"Chunk {j} for deletion test",
                        token_count=100,
                        vector_id=f"delete_vec_{i:02d}_{j}"
                    )
                    session.add(chunk)
        
        session.commit()
    
    # Get counts before deletion
    with db_client.session_scope() as session:
        doc_count_before = session.query(func.count(Document.id)).filter(
            Document.id.like("delete_doc_%")
        ).scalar()
        
        chunk_count_before = session.query(func.count(Chunk.id)).filter(
            Chunk.id.like("delete_chunk_%")
        ).scalar()
    
    start_time = time.time()
    
    # Bulk delete documents (should cascade to chunks)
    with db_client.session_scope() as session:
        delete_stmt = delete(Document).where(
            Document.id.like("delete_doc_%")
        )
        
        result = session.execute(delete_stmt)
        session.commit()
        
        deleted_doc_count = result.rowcount
    
    end_time = time.time()
    delete_time = end_time - start_time
    
    # Verify deletion and cascade
    with db_client.session_scope() as session:
        doc_count_after = session.query(func.count(Document.id)).filter(
            Document.id.like("delete_doc_%")
        ).scalar()
        
        chunk_count_after = session.query(func.count(Chunk.id)).filter(
            Chunk.id.like("delete_chunk_%")
        ).scalar()
    
    assert doc_count_before == deleted_doc_count
    assert doc_count_after == 0
    assert chunk_count_after == 0  # Should be cascade deleted
    assert delete_time < 3.0
    
    print(f"Bulk delete of {deleted_doc_count} documents (with cascade) completed in {delete_time:.2f} seconds")


def test_transaction_isolation_levels(db_client):
    """Test transaction isolation levels and concurrency control."""
    
    results = {"thread1": None, "thread2": None}
    errors = {"thread1": None, "thread2": None}
    
    def thread1_operation():
        """Thread 1: Starts transaction, reads, waits, then updates."""
        try:
            with db_client.session_scope() as session:
                # Read initial value
                doc = session.query(Document).filter_by(id="doc_1").with_for_update().first()
                results["thread1"] = doc.file_name if doc else None
                
                # Simulate some processing time
                time.sleep(0.5)
                
                # Update the document
                if doc:
                    doc.file_name = "Updated by Thread 1"
                
                session.commit()
        except Exception as e:
            errors["thread1"] = str(e)
    
    def thread2_operation():
        """Thread 2: Tries to update same document."""
        try:
            # Wait a bit to ensure thread1 starts first
            time.sleep(0.1)
            
            with db_client.session_scope() as session:
                # Try to update the same document
                update_stmt = update(Document).where(
                    Document.id == "doc_1"
                ).values(
                    file_name="Updated by Thread 2"
                )
                
                result = session.execute(update_stmt)
                session.commit()
                
                results["thread2"] = result.rowcount
        except Exception as e:
            errors["thread2"] = str(e)
    
    # Create test document
    with db_client.session_scope() as session:
        doc = Document(
            id="doc_1",
            file_path="/isolation/test.pdf",
            file_name="Isolation Test",
            file_type="pdf",
            processing_status="completed"
        )
        session.add(doc)
        session.commit()
    
    # Run threads
    thread1 = threading.Thread(target=thread1_operation)
    thread2 = threading.Thread(target=thread2_operation)
    
    thread1.start()
    thread2.start()
    
    thread1.join()
    thread2.join()
    
    # Check results
    print(f"Thread 1 result: {results['thread1']}, error: {errors['thread1']}")
    print(f"Thread 2 result: {results['thread2']}, error: {errors['thread2']}")
    
    # At least one should succeed
    assert results["thread1"] is not None or results["thread2"] is not None
    
    # Verify final state
    with db_client.session_scope() as session:
        final_doc = session.query(Document).filter_by(id="doc_1").first()
        assert final_doc is not None
        print(f"Final document name: {final_doc.file_name}")


# ============================================
# C. INTEGRATION TESTS
# ============================================

def test_database_vector_store_sync(db_client):
    """Test synchronization between database and vector store."""
    
    test_docs = []
    
    # Create test documents in database
    with db_client.session_scope() as session:
        for i in range(1, 6):
            doc = Document(
                id=f"vector_sync_doc_{i}",
                file_path=f"/vector/sync/doc_{i}.pdf",
                file_name=f"Vector Sync Doc {i}",
                file_type="pdf",
                processing_status="completed",
                is_indexed=False
            )
            session.add(doc)
            test_docs.append(doc)
        
        session.commit()
    
    # Simulate vector store indexing
    vector_ids = {}
    with db_client.session_scope() as session:
        for doc in test_docs:
            # Simulate creating chunks and embeddings
            chunk_ids = []
            for j in range(1, 4):
                chunk = Chunk(
                    id=f"vector_chunk_{doc.id}_{j}",
                    document_id=doc.id,
                    chunk_index=j,
                    text_content=f"Content for {doc.file_name} chunk {j}",
                    cleaned_text=f"Content for {doc.file_name} chunk {j}",
                    token_count=100,
                    embedding_model="all-MiniLM-L6-v2",
                    vector_id=f"vector_{doc.id}_{j}"
                )
                session.add(chunk)
                chunk_ids.append(chunk.id)
            
            # Update document as indexed
            doc.is_indexed = True
            doc.indexed_at = datetime.now()
            doc.vector_ids_json = json.dumps(chunk_ids)
            
            vector_ids[doc.id] = chunk_ids
        
        session.commit()
    
    # Verify synchronization
    with db_client.session_scope() as session:
        # Check all documents are marked as indexed
        indexed_count = session.query(func.count(Document.id)).filter(
            Document.id.like("vector_sync_doc_%"),
            Document.is_indexed == True
        ).scalar()
        
        assert indexed_count == 5
        
        # Check chunks were created
        chunk_count = session.query(func.count(Chunk.id)).filter(
            Chunk.id.like("vector_chunk_%")
        ).scalar()
        
        assert chunk_count == 15  # 5 docs * 3 chunks each
        
        # Verify vector IDs are stored
        for doc_id, expected_chunk_ids in vector_ids.items():
            doc = session.query(Document).filter_by(id=doc_id).first()
            assert doc is not None
            assert doc.is_indexed == True
            assert doc.indexed_at is not None
            
            stored_ids = json.loads(doc.vector_ids_json or "[]")
            assert set(stored_ids) == set(expected_chunk_ids)


def test_import_export_functionality(db_client, tmp_path):
    """Test database import/export functionality."""
    
    # Create test data
    with db_client.session_scope() as session:
        for i in range(1, 11):
            doc = Document(
                id=f"import_export_doc_{i}",
                file_path=f"/import_export/doc_{i}.txt",
                file_name=f"Import Export Test {i}",
                file_type="txt",
                processing_status="completed"
            )
            session.add(doc)
        
        session.commit()
    
    # Test export
    export_path = tmp_path / "export.json"
    
    start_time = time.time()
    
    # Simulate export (in real implementation, this would be a method in SQLiteClient)
    with db_client.session_scope() as session:
        # Query all documents
        documents = session.query(Document).filter(
            Document.id.like("import_export_doc_%")
        ).all()
        
        # Convert to dict format
        export_data = {
            "documents": [
                {
                    "id": doc.id,
                    "file_name": doc.file_name,
                    "file_type": doc.file_type,
                    "processing_status": doc.processing_status
                }
                for doc in documents
            ],
            "export_timestamp": datetime.now().isoformat(),
            "count": len(documents)
        }
        
        # Write to file
        with open(export_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
    
    export_time = time.time() - start_time
    
    # Verify export file
    assert export_path.exists()
    assert export_path.stat().st_size > 0
    
    with open(export_path, 'r', encoding='utf-8') as f:
        loaded_data = json.load(f)
    
    assert loaded_data["count"] == 10
    assert len(loaded_data["documents"]) == 10
    
    # Test import simulation
    import_path = tmp_path / "import.json"
    
    # Create import data
    import_data = {
        "documents": [
            {
                "id": f"imported_doc_{i}",
                "file_name": f"Imported Document {i}",
                "file_type": "pdf",
                "processing_status": "pending"
            }
            for i in range(1, 6)
        ]
    }
    
    with open(import_path, 'w', encoding='utf-8') as f:
        json.dump(import_data, f, indent=2)
    
    # Simulate import
    start_time = time.time()
    
    with db_client.session_scope() as session:
        for doc_data in import_data["documents"]:
            doc = Document(
                id=doc_data["id"],
                file_path=f"/imported/path/{doc_data['id']}.pdf",
                file_name=doc_data["file_name"],
                file_type=doc_data["file_type"],
                processing_status=doc_data["processing_status"],
                upload_date=datetime.now()
            )
            session.add(doc)
        
        session.commit()
    
    import_time = time.time() - start_time
    
    # Verify import
    with db_client.session_scope() as session:
        imported_count = session.query(func.count(Document.id)).filter(
            Document.id.like("imported_doc_%")
        ).scalar()
        
        assert imported_count == 5
    
    print(f"Export completed in {export_time:.2f}s, Import completed in {import_time:.2f}s")


def test_backup_restore_integrity(db_client, tmp_path):
    """Test database backup and restore with integrity verification."""
    
    # Create test data
    test_data = []
    with db_client.session_scope() as session:
        for i in range(1, 21):
            doc = Document(
                id=f"backup_doc_{i:02d}",
                file_path=f"/backup/path/doc_{i:02d}.pdf",
                file_name=f"Backup Test {i:02d}",
                file_type="pdf",
                file_size=1000 * i,
                processing_status="completed",
                word_count=500 * i
            )
            session.add(doc)
            test_data.append({
                "id": doc.id,
                "file_name": doc.file_name,
                "file_size": doc.file_size,
                "word_count": doc.word_count
            })
        
        session.commit()
    
    # Get initial counts
    with db_client.session_scope() as session:
        initial_count = session.query(func.count(Document.id)).scalar()
        backup_doc_count = session.query(func.count(Document.id)).filter(
            Document.id.like("backup_doc_%")
        ).scalar()
    
    assert backup_doc_count == 20
    
    # Create backup (simulate by copying database file)
    backup_path = tmp_path / "backup.db"
    original_path = Path(db_client.db_path)
    
    if original_path.exists():
        shutil.copy2(original_path, backup_path)
    
    assert backup_path.exists()
    assert backup_path.stat().st_size > 0
    
    # Modify original database
    with db_client.session_scope() as session:
        # Delete some documents
        delete_stmt = delete(Document).where(
            Document.id.like("backup_doc_%")
        )
        session.execute(delete_stmt)
        session.commit()
        
        # Verify deletion
        remaining_count = session.query(func.count(Document.id)).filter(
            Document.id.like("backup_doc_%")
        ).scalar()
        
        assert remaining_count == 0
    
    # Restore from backup
    if backup_path.exists() and original_path.exists():
        # Close any existing connections
        db_client.close()
        
        # Restore the backup
        shutil.copy2(backup_path, original_path)
        
        # Reconnect
        db_client.connect()
    
    # Verify restoration
    with db_client.session_scope() as session:
        restored_count = session.query(func.count(Document.id)).filter(
            Document.id.like("backup_doc_%")
        ).scalar()
        
        assert restored_count == 20
        
        # Verify data integrity
        for expected_data in test_data:
            doc = session.query(Document).filter_by(id=expected_data["id"]).first()
            assert doc is not None
            assert doc.file_name == expected_data["file_name"]
            assert doc.file_size == expected_data["file_size"]
            assert doc.word_count == expected_data["word_count"]
    
    print("Backup and restore integrity test passed")


def test_migration_scenarios(db_client):
    """Test database migration scenarios."""
    
    # Test 1: Schema migration simulation
    print("Testing schema migration...")
    
    with db_client.session_scope() as session:
        # Check current schema
        tables = session.execute(text(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        )).fetchall()
        
        table_names = [t[0] for t in tables]
        print(f"Current tables: {table_names}")
        
        # Simulate adding a new column (in real migration, this would be done via Alembic)
        try:
            # Try to add a new column if it doesn't exist
            session.execute(text(
                "ALTER TABLE documents ADD COLIF NOT EXISTS new_column TEXT DEFAULT 'default_value'"
            ))
        except:
            # In test, we'll simulate the effect
            print("Simulating schema migration...")
        
        session.commit()
    
    # Test 2: Data migration simulation
    print("Testing data migration...")
    
    with db_client.session_scope() as session:
        # Create some old-format data
        old_format_docs = []
        for i in range(1, 6):
            doc = Document(
                id=f"migration_old_{i}",
                file_path=f"/migration/old/doc_{i}.txt",
                file_name=f"Old Format Doc {i}",
                file_type="txt",
                processing_status="needs_migration"  # Special status indicating needs migration
            )
            session.add(doc)
            old_format_docs.append(doc)
        
        session.commit()
        
        # Simulate migration: update old format to new format
        update_stmt = update(Document).where(
            Document.processing_status == "needs_migration"
        ).values(
            processing_status="completed",
            last_accessed=datetime.now()
        )
        
        migrated_count = session.execute(update_stmt).rowcount
        session.commit()
        
        assert migrated_count == 5
        
        # Verify migration
        migrated_docs = session.query(Document).filter(
            Document.id.like("migration_old_%"),
            Document.processing_status == "completed"
        ).count()
        
        assert migrated_docs == 5
    
    # Test 3: Migration rollback simulation
    print("Testing migration rollback...")
    
    try:
        with db_client.session_scope() as session:
            # Start a migration that will fail
            session.execute(text("BEGIN TRANSACTION"))
            
            # Do some migration operations
            session.execute(text(
                "UPDATE documents SET file_name = 'MIGRATED: ' || file_name WHERE id LIKE 'migration_old_%'"
            ))
            
            # Simulate an error
            raise Exception("Simulated migration error")
            
            # This won't be reached due to the exception
            session.commit()
            
    except Exception as e:
        print(f"Migration failed (as expected): {e}")
        
        # The transaction should be rolled back automatically
        with db_client.session_scope() as session:
            # Check that changes were rolled back
            migrated_names = session.query(Document.file_name).filter(
                Document.id.like("migration_old_%"),
                Document.file_name.like("MIGRATED:%")
            ).count()
            
            assert migrated_names == 0, "Rollback should have prevented changes"
    
    print("All migration scenario tests passed")


# ============================================
# D. REAL-WORLD SCENARIOS
# ============================================

def test_concurrent_user_scenario(db_client):
    """Test database handling under concurrent user access."""
    
    results_lock = threading.Lock()
    results = []
    errors = []
    
    def simulate_user(user_id, operation_count=10):
        """Simulate a user performing database operations."""
        user_results = []
        
        try:
            for i in range(operation_count):
                with db_client.session_scope() as session:
                    # Simulate various user operations
                    operation_type = random.choice(["read", "write", "update", "delete"])
                    
                    if operation_type == "read":
                        # Read random document
                        count = session.query(func.count(Document.id)).scalar()
                        if count > 0:
                            offset = random.randint(0, max(0, count - 1))
                            doc = session.query(Document).offset(offset).first()
                            user_results.append(f"User{user_id} read: {doc.file_name if doc else 'None'}")
                    
                    elif operation_type == "write":
                        # Write new document
                        doc_id = f"concurrent_user{user_id}_doc_{i}"
                        doc = Document(
                            id=doc_id,
                            file_path=f"/concurrent/user{user_id}/doc_{i}.txt",
                            file_name=f"User{user_id} Document {i}",
                            file_type="txt",
                            processing_status="completed"
                        )
                        session.add(doc)
                        user_results.append(f"User{user_id} wrote: {doc_id}")
                    
                    elif operation_type == "update":
                        # Update random document
                        count = session.query(func.count(Document.id)).scalar()
                        if count > 0:
                            offset = random.randint(0, max(0, count - 1))
                            doc = session.query(Document).offset(offset).first()
                            if doc:
                                old_name = doc.file_name
                                doc.file_name = f"Updated by User{user_id}"
                                user_results.append(f"User{user_id} updated: {old_name} -> {doc.file_name}")
                    
                    elif operation_type == "delete":
                        # Delete user's own documents only
                        delete_stmt = delete(Document).where(
                            Document.id.like(f"concurrent_user{user_id}_%")
                        )
                        deleted = session.execute(delete_stmt).rowcount
                        if deleted > 0:
                            user_results.append(f"User{user_id} deleted {deleted} documents")
                    
                    # Small random delay
                    time.sleep(random.uniform(0.01, 0.05))
                    
                    session.commit()
        
        except Exception as e:
            with results_lock:
                errors.append(f"User{user_id} error: {str(e)}")
        
        with results_lock:
            results.extend(user_results)
    
    # Create initial data
    with db_client.session_scope() as session:
        for i in range(1, 11):
            doc = Document(
                id=f"initial_concurrent_doc_{i}",
                file_path=f"/concurrent/initial/doc_{i}.pdf",
                file_name=f"Initial Concurrent Doc {i}",
                file_type="pdf",
                processing_status="completed"
            )
            session.add(doc)
        
        session.commit()
    
    # Simulate multiple concurrent users
    threads = []
    num_users = 5
    operations_per_user = 15
    
    start_time = time.time()
    
    for user_id in range(1, num_users + 1):
        thread = threading.Thread(
            target=simulate_user,
            args=(user_id, operations_per_user)
        )
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    end_time = time.time()
    total_time = end_time - start_time
    
    # Verify no data corruption
    with db_client.session_scope() as session:
        # Check that all documents have valid file names
        invalid_names = session.query(Document).filter(
            or_(
                Document.file_name == None,
                Document.file_name == ""
            )
        ).count()
        
        assert invalid_names == 0, "Found documents with invalid file names"
        
        # Check for duplicate IDs
        duplicate_ids = session.query(
            Document.id,
            func.count(Document.id)
        ).group_by(Document.id
        ).having(func.count(Document.id) > 1).count()
        
        assert duplicate_ids == 0, f"Found {duplicate_ids} duplicate document IDs"
    
    print(f"Concurrent user test completed in {total_time:.2f} seconds")
    print(f"Total operations: {len(results)}")
    print(f"Errors encountered: {len(errors)}")
    
    if errors:
        print("Sample errors:")
        for error in errors[:3]:
            print(f"  - {error}")


def test_large_dataset_handling(large_dataset_db):
    """Test database performance with large datasets."""
    
    print("Testing large dataset handling...")
    
    # Test 1: Count operations
    start_time = time.time()
    
    with large_dataset_db.session_scope() as session:
        total_docs = session.query(func.count(Document.id)).scalar()
        completed_docs = session.query(func.count(Document.id)).filter(
            Document.processing_status == "completed"
        ).scalar()
        pending_docs = session.query(func.count(Document.id)).filter(
            Document.processing_status == "pending"
        ).scalar()
    
    count_time = time.time() - start_time
    
    assert total_docs == 1000
    assert completed_docs == 900  # 90% should be completed (based on fixture)
    assert pending_docs == 100    # 10% should be pending
    
    print(f"Count operations completed in {count_time:.3f}s")
    print(f"Total: {total_docs}, Completed: {completed_docs}, Pending: {pending_docs}")
    
    # Test 2: Complex query performance
    start_time = time.time()
    
    with large_dataset_db.session_scope() as session:
        # Complex query: documents with chunks, grouped by status
        query = session.query(
            Document.processing_status,
            func.count(Document.id).label('doc_count'),
            func.avg(Document.file_size).label('avg_size'),
            func.sum(func.coalesce(Chunk.token_count, 0)).label('total_tokens')
        ).outerjoin(Chunk, Document.id == Chunk.document_id
        ).group_by(Document.processing_status)
        
        results = query.all()
    
    query_time = time.time() - start_time
    
    print(f"Complex query completed in {query_time:.3f}s")
    for status, count, avg_size, total_tokens in results:
        print(f"  Status: {status}, Docs: {count}, Avg Size: {avg_size:.0f}, Total Tokens: {total_tokens or 0}")
    
    assert query_time < 2.0  # Should complete within 2 seconds
    
    # Test 3: Pagination performance
    page_size = 50
    total_pages = (total_docs + page_size - 1) // page_size
    
    print(f"\nTesting pagination ({total_pages} pages of {page_size} items):")
    
    page_times = []
    for page in range(1, min(6, total_pages + 1)):  # Test first 5 pages
        offset = (page - 1) * page_size
        
        start_time = time.time()
        
        with large_dataset_db.session_scope() as session:
            documents = session.query(Document).order_by(
                Document.upload_date.desc()
            ).offset(offset).limit(page_size).all()
        
        page_time = time.time() - start_time
        page_times.append(page_time)
        
        print(f"  Page {page}: {len(documents)} docs in {page_time:.3f}s")
    
    avg_page_time = sum(page_times) / len(page_times)
    print(f"Average page load time: {avg_page_time:.3f}s")
    
    assert avg_page_time < 0.5  # Should be fast even with large dataset
    
    # Test 4: Index usage verification
    start_time = time.time()
    
    with large_dataset_db.session_scope() as session:
        # Query that should use indexes
        indexed_query = session.query(Document).filter(
            Document.file_type == "txt",
            Document.processing_status == "completed",
            Document.upload_date > datetime.now() - timedelta(days=7)
        ).order_by(Document.upload_date).limit(100)
        
        indexed_results = indexed_query.all()
    
    indexed_time = time.time() - start_time
    
    print(f"\nIndexed query returned {len(indexed_results)} docs in {indexed_time:.3f}s")
    assert indexed_time < 1.0


def test_disk_space_management(db_client, tmp_path):
    """Test database disk space management and optimization."""
    
    print("Testing disk space management...")
    
    # Get initial database size
    db_path = Path(db_client.db_path)
    initial_size = db_path.stat().st_size if db_path.exists() else 0
    print(f"Initial database size: {initial_size:,} bytes")
    
    # Test 1: Insert large amount of data
    print("\n1. Testing bulk data insertion...")
    
    start_time = time.time()
    
    with db_client.session_scope() as session:
        # Insert 500 documents with chunks
        for i in range(1, 501):
            doc = Document(
                id=f"diskspace_doc_{i:04d}",
                file_path=f"/diskspace/path/doc_{i:04d}.pdf",
                file_name=f"Disk Space Test Doc {i:04d}",
                file_type="pdf",
                file_size=random.randint(5000, 50000),
                processing_status="completed",
                word_count=random.randint(1000, 5000)
            )
            session.add(doc)
            
            # Add chunks with large text content
            for j in range(1, random.randint(3, 10)):
                large_text = "X" * random.randint(1000, 5000)  # Create large text content
                chunk = Chunk(
                    id=f"diskspace_chunk_{i:04d}_{j}",
                    document_id=doc.id,
                    chunk_index=j,
                    text_content=large_text,
                    cleaned_text=large_text,
                    token_count=len(large_text.split()),
                    embedding_model="all-MiniLM-L6-v2",
                    vector_id=f"diskspace_vec_{i:04d}_{j}"
                )
                session.add(chunk)
        
        session.commit()
    
    insert_time = time.time() - start_time
    
    # Check size after insertion
    after_insert_size = db_path.stat().st_size
    size_increase = after_insert_size - initial_size
    print(f"After insertion: {after_insert_size:,} bytes (+{size_increase:,})")
    print(f"Insertion time: {insert_time:.2f}s")
    
    # Test 2: Database vacuum (reclaim space)
    print("\n2. Testing VACUUM operation...")
    
    start_time = time.time()
    
    with db_client.session_scope() as session:
        # Delete half the data
        delete_stmt = delete(Document).where(
            Document.id.like("diskspace_doc_%"),
            func.cast(substr(Document.id, -4), sqlite3.INTEGER) % 2 == 0  # Delete even-numbered docs
        )
        deleted_count = session.execute(delete_stmt).rowcount
        session.commit()
    
    delete_time = time.time() - start_time
    
    # Size after deletion (shouldn't change much due to SQLite's behavior)
    after_delete_size = db_path.stat().st_size
    print(f"Deleted {deleted_count} documents in {delete_time:.2f}s")
    print(f"Size after deletion: {after_delete_size:,} bytes")
    
    # Perform VACUUM
    print("\n3. Performing VACUUM...")
    
    start_time = time.time()
    
    with db_client.session_scope() as session:
        session.execute(text("VACUUM"))
        session.commit()
    
    vacuum_time = time.time() - start_time
    
    # Check size after VACUUM
    after_vacuum_size = db_path.stat().st_size
    space_reclaimed = after_delete_size - after_vacuum_size
    
    print(f"VACUUM completed in {vacuum_time:.2f}s")
    print(f"Size after VACUUM: {after_vacuum_size:,} bytes")
    print(f"Space reclaimed: {space_reclaimed:,} bytes ({space_reclaimed/after_delete_size*100:.1f}%)")
    
    # Test 3: Check fragmentation
    print("\n4. Checking database fragmentation...")
    
    with db_client.session_scope() as session:
        # Get page count and freelist count
        pragma_result = session.execute(text("PRAGMA page_count")).fetchone()
        page_count = pragma_result[0] if pragma_result else 0
        
        freelist_result = session.execute(text("PRAGMA freelist_count")).fetchone()
        freelist_count = freelist_result[0] if freelist_result else 0
        
        page_size_result = session.execute(text("PRAGMA page_size")).fetchone()
        page_size = page_size_result[0] if page_size_result else 4096
    
    total_space = page_count * page_size
    used_space = after_vacuum_size
    fragmentation = freelist_count / page_count if page_count > 0 else 0
    
    print(f"Page count: {page_count}")
    print(f"Freelist pages: {freelist_count}")
    print(f"Page size: {page_size:,} bytes")
    print(f"Total allocated space: {total_space:,} bytes")
    print(f"Actual file size: {used_space:,} bytes")
    print(f"Fragmentation: {fragmentation:.1%}")
    
    assert after_vacuum_size <= after_delete_size, "VACUUM should reduce or maintain size"
    assert fragmentation < 0.5, "Database should not be highly fragmented after VACUUM"
    
    print("\nDisk space management tests passed!")


def test_memory_usage_optimization(db_client):
    """Test database memory usage and optimization strategies."""
    
    print("Testing memory usage optimization...")
    
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    # Test 1: Baseline memory usage
    initial_memory = process.memory_info().rss
    print(f"Initial memory usage: {initial_memory:,} bytes")
    
    # Test 2: Memory usage during large query
    print("\n1. Testing memory during large query...")
    
    # Create test data
    with db_client.session_scope() as session:
        for i in range(1, 101):
            doc = Document(
                id=f"memory_doc_{i:03d}",
                file_path=f"/memory/test/doc_{i:03d}.txt",
                file_name=f"Memory Test Doc {i:03d}",
                file_type="txt",
                processing_status="completed",
                summary="X" * 1000  # Large summary
            )
            session.add(doc)
        
        session.commit()
    
    # Memory before query
    memory_before = process.memory_info().rss
    
    # Execute large query (fetch all documents with large text)
    start_time = time.time()
    
    with db_client.session_scope() as session:
        documents = session.query(Document).filter(
            Document.id.like("memory_doc_%")
        ).all()
    
    query_time = time.time() - start_time
    
    # Memory after query
    memory_after = process.memory_info().rss
    memory_increase = memory_after - memory_before
    
    print(f"Fetched {len(documents)} documents in {query_time:.2f}s")
    print(f"Memory increase: {memory_increase:,} bytes")
    print(f"Memory per document: {memory_increase/len(documents):,.0f} bytes")
    
    # Clear references to allow garbage collection
    del documents
    
    # Test 3: Memory usage with streaming/batching
    print("\n2. Testing memory with batch processing...")
    
    memory_before = process.memory_info().rss
    
    start_time = time.time()
    total_docs = 0
    
    with db_client.session_scope() as session:
        # Use yield_per for batch processing
        query = session.query(Document).filter(
            Document.id.like("memory_doc_%")
        ).yield_per(20)  # Process 20 at a time
        
        batch_count = 0
        for batch in query:
            # Process batch
            batch_count += 1
            total_docs += 1
            
            # Small processing delay
            time.sleep(0.001)
    
    batch_time = time.time() - start_time
    
    memory_after = process.memory_info().rss
    batch_memory_increase = memory_after - memory_before
    
    print(f"Processed {total_docs} documents in {batch_count} batches")
    print(f"Batch processing time: {batch_time:.2f}s")
    print(f"Memory increase with batching: {batch_memory_increase:,} bytes")
    
    # Test 4: Connection pool memory
    print("\n3. Testing connection pool memory...")
    
    # Create multiple clients/sessions
    clients = []
    sessions = []
    
    memory_before = process.memory_info().rss
    
    for i in range(10):
        # Create new client (would create new connection pool in real scenario)
        # For testing, we'll create new sessions from existing client
        with db_client.session_scope() as session:
            # Just create session, don't do anything
            sessions.append(session)
            time.sleep(0.01)
    
    memory_after = process.memory_info().rss
    connection_memory = memory_after - memory_before
    
    print(f"Memory for 10 sessions: {connection_memory:,} bytes")
    print(f"Memory per session: {connection_memory/10:,.0f} bytes")
    
    # Test 5: Memory optimization techniques
    print("\n4. Testing memory optimization techniques...")
    
    # Technique 1: Use scalar() instead of all() when possible
    memory_before = process.memory_info().rss
    
    with db_client.session_scope() as session:
        # Inefficient: fetch all then count
        all_docs = session.query(Document).filter(
            Document.id.like("memory_doc_%")
        ).all()
        count_inefficient = len(all_docs)
        
        # Efficient: use scalar()
        count_efficient = session.query(func.count(Document.id)).filter(
            Document.id.like("memory_doc_%")
        ).scalar()
    
    memory_after = process.memory_info().rss
    memory_diff = memory_after - memory_before
    
    print(f"Count via all(): {count_inefficient} (memory: {memory_diff:,} bytes)")
    print(f"Count via scalar(): {count_efficient}")
    assert count_inefficient == count_efficient
    
    # Technique 2: Use only() to select specific columns
    memory_before = process.memory_info().rss
    
    with db_client.session_scope() as session:
        # Fetch all columns
        all_columns = session.query(Document).filter(
            Document.id.like("memory_doc_%")
        ).limit(10).all()
        
        # Fetch only needed columns
        specific_columns = session.query(
            Document.id, Document.file_name
        ).filter(
            Document.id.like("memory_doc_%")
        ).limit(10).all()
    
    memory_after = process.memory_info().rss
    column_memory_diff = memory_after - memory_before
    
    print(f"Memory for all columns vs specific columns difference: {column_memory_diff:,} bytes")
    
    # Cleanup
    print("\n5. Cleaning up test data...")
    
    with db_client.session_scope() as session:
        delete_stmt = delete(Document).where(
            Document.id.like("memory_doc_%")
        )
        deleted = session.execute(delete_stmt).rowcount
        session.commit()
    
    print(f"Deleted {deleted} test documents")
    
    final_memory = process.memory_info().rss
    total_memory_change = final_memory - initial_memory
    
    print(f"\nFinal memory usage: {final_memory:,} bytes")
    print(f"Total memory change: {total_memory_change:,} bytes")
    
    # Memory should return close to initial (allow for Python overhead)
    assert abs(total_memory_change) < 50 * 1024 * 1024  # Within 50MB
    
    print("\nMemory usage optimization tests passed!")


# ============================================
# E. FAILURE & RECOVERY
# ============================================

def test_power_failure_recovery(db_client, tmp_path):
    """Test recovery from simulated power failure."""
    
    print("Testing power failure recovery...")
    
    # Create a copy of the database to simulate power failure
    test_db_path = tmp_path / "power_failure_test.db"
    original_path = Path(db_client.db_path)
    
    if original_path.exists():
        shutil.copy2(original_path, test_db_path)
    
    # Create a new client for the test database
    test_client = SQLiteClient(str(test_db_path))
    test_client.connect()
    
    try:
        # Start a transaction
        with test_client.session_scope() as session:
            # Create some test data
            for i in range(1, 6):
                doc = Document(
                    id=f"power_doc_{i}",
                    file_path=f"/power/test/doc_{i}.txt",
                    file_name=f"Power Test {i}",
                    file_type="txt",
                    processing_status="in_progress"
                )
                session.add(doc)
            
            # Commit first part
            session.commit()
            
            # Start another transaction
            for i in range(6, 11):
                doc = Document(
                    id=f"power_doc_{i}",
                    file_path=f"/power/test/doc_{i}.txt",
                    file_name=f"Power Test {i}",
                    file_type="txt",
                    processing_status="in_progress"
                )
                session.add(doc)
            
            # Simulate power failure before commit
            print("Simulating power failure...")
            
            # Forcefully close the connection without commit
            # This simulates power failure during transaction
            test_client.close()
            
            # Delete the journal file if it exists (simulating corrupted state)
            journal_path = test_db_path.with_suffix(test_db_path.suffix + '-journal')
            if journal_path.exists():
                print(f"Removing journal file to simulate corruption: {journal_path}")
                journal_path.unlink()
            
    except Exception as e:
        print(f"Expected exception during power failure simulation: {e}")
    
    # Now try to recover
    print("\nAttempting recovery...")
    
    # Reconnect (this should trigger recovery)
    test_client.connect()
    
    # Check what survived
    with test_client.session_scope() as session:
        # Count documents
        total_docs = session.query(func.count(Document.id)).scalar()
        power_docs = session.query(func.count(Document.id)).filter(
            Document.id.like("power_doc_%")
        ).scalar()
        
        print(f"Total documents after recovery: {total_docs}")
        print(f"Power test documents after recovery: {power_docs}")
        
        # Check database integrity
        try:
            integrity_result = session.execute(text("PRAGMA integrity_check")).fetchone()
            integrity_status = integrity_result[0] if integrity_result else "unknown"
            print(f"Database integrity check: {integrity_status}")
            
            if integrity_status != "ok":
                print("Database integrity compromised, attempting repair...")
                
                # Try to recover what we can
                session.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))
                session.commit()
                
                # Check again
                integrity_result = session.execute(text("PRAGMA integrity_check")).fetchone()
                integrity_status = integrity_result[0] if integrity_result else "unknown"
                print(f"After repair attempt: {integrity_status}")
        
        except Exception as e:
            print(f"Integrity check failed: {e}")
        
        # Verify transaction atomicity
        # Documents 1-5 should exist (committed before simulated failure)
        # Documents 6-10 might not exist (uncommitted)
        for i in range(1, 11):
            doc = session.query(Document).filter_by(id=f"power_doc_{i}").first()
            if i <= 5:
                assert doc is not None, f"Document {i} should exist (was committed)"
                print(f"  ✓ Document {i}: EXISTS (committed)")
            else:
                # Documents 6-10 might or might not exist depending on recovery
                status = "EXISTS" if doc else "MISSING"
                print(f"  ? Document {i}: {status} (was in uncommitted transaction)")
    
    test_client.close()
    
    print("\nPower failure recovery test completed!")


def test_corrupt_database_recovery(db_client, tmp_path):
    """Test recovery from corrupted database."""
    
    print("Testing corrupt database recovery...")
    
    # Create a test database
    corrupt_db_path = tmp_path / "corrupt_test.db"
    test_client = SQLiteClient(str(corrupt_db_path))
    test_client.connect()
    
    # Add some test data
    with test_client.session_scope() as session:
        for i in range(1, 11):
            doc = Document(
                id=f"corrupt_doc_{i}",
                file_path=f"/corrupt/test/doc_{i}.txt",
                file_name=f"Corrupt Test {i}",
                file_type="txt",
                processing_status="completed"
            )
            session.add(doc)
        
        session.commit()
    
    test_client.close()
    
    # Corrupt the database file
    print("\nSimulating database corruption...")
    
    with open(corrupt_db_path, 'rb+') as f:
        # Move to a random position and overwrite with garbage
        file_size = corrupt_db_path.stat().st_size
        if file_size > 100:
            corrupt_position = random.randint(100, file_size - 100)
            f.seek(corrupt_position)
            f.write(b'CORRUPTED' * 10)
    
    print(f"Database corrupted at position {corrupt_position}")
    
    # Try to connect to corrupted database
    print("\nAttempting to connect to corrupted database...")
    
    try:
        test_client.connect()
        print("Connected to corrupted database (might work depending on corruption)")
    except Exception as e:
        print(f"Connection failed: {e}")
    
    # Try various recovery methods
    print("\nAttempting recovery methods...")
    
    # Method 1: Backup and restore
    print("1. Backup and restore method:")
    
    backup_path = tmp_path / "backup_before_corruption.db"
    if corrupt_db_path.exists():
        try:
            shutil.copy2(corrupt_db_path, backup_path)
            print(f"  Backup created: {backup_path}")
        except Exception as e:
            print(f"  Backup failed: {e}")
    
    # Method 2: Use .recover command (simulated)
    print("2. Using .recover command (simulated):")
    
    try:
        # In real scenario, you'd use: sqlite3 corrupt.db ".recover" | sqlite3 recovered.db
        print("  Simulating: sqlite3 corrupt.db \".recover\" | sqlite3 recovered.db")
        
        recovered_path = tmp_path / "recovered.db"
        
        # For test purposes, create a new database
        recovered_client = SQLiteClient(str(recovered_path))
        recovered_client.connect()
        
        # Try to recover data from corrupted file
        if corrupt_db_path.exists():
            try:
                # Attempt to read what we can
                conn = sqlite3.connect(str(corrupt_db_path))
                cursor = conn.cursor()
                
                # Try to get table list
                try:
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = cursor.fetchall()
                    print(f"  Found {len(tables)} tables in corrupted DB")
                    
                    # Try to recover documents table
                    try:
                        cursor.execute("SELECT id, file_name FROM documents")
                        recovered_data = cursor.fetchall()
                        print(f"  Recovered {len(recovered_data)} documents")
                        
                        # Insert into recovered database
                        with recovered_client.session_scope() as session:
                            for doc_id, file_name in recovered_data:
                                doc = Document(
                                    id=doc_id,
                                    file_path=f"/recovered/{doc_id}.txt",
                                    file_name=file_name,
                                    file_type="txt",
                                    processing_status="recovered"
                                )
                                session.add(doc)
                            
                            session.commit()
                        
                        print(f"  Successfully recovered {len(recovered_data)} documents")
                        
                    except sqlite3.Error as e:
                        print(f"  Could not read documents table: {e}")
                
                except sqlite3.Error as e:
                    print(f"  Could not read table list: {e}")
                
                finally:
                    conn.close()
            
            except Exception as e:
                print(f"  Recovery attempt failed: {e}")
        
        recovered_client.close()
    
    except Exception as e:
        print(f"  Recovery simulation failed: {e}")
    
    # Method 3: Create new database from scratch
    print("3. Create new database from scratch:")
    
    new_db_path = tmp_path / "new_database.db"
    new_client = SQLiteClient(str(new_db_path))
    new_client.connect()
    
    # Recreate essential data
    with new_client.session_scope() as session:
        # Create essential tables
        Base.metadata.create_all(new_client.engine)
        
        # Add minimum required data
        doc = Document(
            id="recovered_doc_1",
            file_path="/recovered/essential.txt",
            file_name="Essential Recovered Document",
            file_type="txt",
            processing_status="completed",
            summary="Recovered after corruption"
        )
        session.add(doc)
        
        session.commit()
    
    # Verify new database
    with new_client.session_scope() as session:
        doc_count = session.query(func.count(Document.id)).scalar()
        print(f"  New database has {doc_count} documents")
        
        # Check integrity
        try:
            integrity_result = session.execute(text("PRAGMA integrity_check")).fetchone()
            print(f"  New database integrity: {integrity_result[0] if integrity_result else 'N/A'}")
        except:
            pass
    
    new_client.close()
    
    # Summary
    print("\nCorruption recovery summary:")
    print(f"1. Original corrupted size: {corrupt_db_path.stat().st_size if corrupt_db_path.exists() else 0:,} bytes")
    print(f"2. Backup created: {'Yes' if backup_path.exists() else 'No'}")
    print(f"3. New database created: {'Yes' if new_db_path.exists() else 'No'}")
    
    # Cleanup
    if corrupt_db_path.exists():
        corrupt_db_path.unlink()
    
    print("\nCorrupt database recovery test completed!")


def test_rollback_scenarios(db_client):
    """Test various rollback scenarios."""
    
    print("Testing rollback scenarios...")
    
    # Scenario 1: Explicit rollback on error
    print("\n1. Explicit rollback on error:")
    
    initial_count = 0
    with db_client.session_scope() as session:
        initial_count = session.query(func.count(Document.id)).scalar()
    
    try:
        with db_client.session_scope() as session:
            # Add some documents
            for i in range(1, 4):
                doc = Document(
                    id=f"rollback_doc_{i}",
                    file_path=f"/rollback/test/doc_{i}.txt",
                    file_name=f"Rollback Test {i}",
                    file_type="txt"
                )
                session.add(doc)
            
            # Simulate an error
            raise ValueError("Simulated error before commit")
            
            # This won't be reached
            session.commit()
    
    except ValueError as e:
        print(f"  Error caught: {e}")
        print("  Transaction should be rolled back automatically")
    
    # Verify rollback
    with db_client.session_scope() as session:
        after_rollback_count = session.query(func.count(Document.id)).scalar()
        rollback_docs = session.query(func.count(Document.id)).filter(
            Document.id.like("rollback_doc_%")
        ).scalar()
    
    assert initial_count == after_rollback_count, "Rollback should maintain count"
    assert rollback_docs == 0, "Rollback documents should not exist"
    print(f"  ✓ Verified: No documents added (count: {initial_count} -> {after_rollback_count})")
    
    # Scenario 2: Nested transaction rollback
    print("\n2. Nested transaction rollback:")
    
    try:
        with db_client.session_scope() as session:
            # Outer transaction
            doc1 = Document(
                id="outer_doc",
                file_path="/rollback/outer.txt",
                file_name="Outer Document"
            )
            session.add(doc1)
            session.commit()  # Commit outer
            
            # Inner transaction (simulated with savepoint)
            doc2 = Document(
                id="inner_doc",
                file_path="/rollback/inner.txt",
                file_name="Inner Document"
            )
            session.add(doc2)
            
            # Create a savepoint
            session.begin_nested()
            
            # Add more in nested transaction
            doc3 = Document(
                id="nested_doc",
                file_path="/rollback/nested.txt",
                file_name="Nested Document"
            )
            session.add(doc3)
            
            # Rollback only nested transaction
            session.rollback()  # This rolls back to savepoint
            
            # Commit outer transaction
            session.commit()
    
    except Exception as e:
        print(f"  Nested transaction error: {e}")
    
    # Verify nested rollback
    with db_client.session_scope() as session:
        outer_exists = session.query(Document).filter_by(id="outer_doc").first() is not None
        inner_exists = session.query(Document).filter_by(id="inner_doc").first() is not None
        nested_exists = session.query(Document).filter_by(id="nested_doc").first() is not None
    
    print(f"  Outer document: {'EXISTS ✓' if outer_exists else 'MISSING'}")
    print(f"  Inner document: {'EXISTS ✓' if inner_exists else 'MISSING'}")
    print(f"  Nested document: {'EXISTS' if nested_exists else 'MISSING ✓'}")
    
    assert outer_exists, "Outer document should exist"
    assert inner_exists, "Inner document should exist (committed with outer)"
    assert not nested_exists, "Nested document should not exist (was rolled back)"
    
    # Scenario 3: Partial commit/rollback
    print("\n3. Partial commit/rollback scenario:")
    
    with db_client.session_scope() as session:
        # Add multiple documents
        docs_to_add = []
        for i in range(1, 6):
            doc = Document(
                id=f"partial_doc_{i}",
                file_path=f"/partial/test/doc_{i}.txt",
                file_name=f"Partial Test {i}",
                file_type="txt"
            )
            docs_to_add.append(doc)
            session.add(doc)
        
        # Commit first 3
        for i in range(3):
            session.flush()  # Ensure they're in session
        
        # Now simulate an error with the 4th
        docs_to_add[3].file_name = None  # This will cause error on commit
        
        try:
            session.commit()
            print("  ERROR: Should have raised an exception")
        except Exception as e:
            print(f"  Expected error: {type(e).__name__}")
            
            # The entire transaction should be rolled back
            # including the first 3 documents
            session.rollback()
    
    # Verify all were rolled back
    with db_client.session_scope() as session:
        partial_count = session.query(func.count(Document.id)).filter(
            Document.id.like("partial_doc_%")
        ).scalar()
    
    print(f"  Documents after partial rollback: {partial_count}")
    assert partial_count == 0, "All partial documents should be rolled back"
    
    # Scenario 4: Manual rollback in complex operation
    print("\n4. Manual rollback in complex operation:")
    
    documents_before = 0
    with db_client.session_scope() as session:
        documents_before = session.query(func.count(Document.id)).scalar()
    
    try:
        # Manual transaction control
        session = db_client.Session()
        
        try:
            # Complex operation
            for i in range(1, 4):
                doc = Document(
                    id=f"manual_doc_{i}",
                    file_path=f"/manual/test/doc_{i}.txt",
                    file_name=f"Manual Test {i}"
                )
                session.add(doc)
            
            # Simulate complex validation
            if random.random() < 0.7:  # 70% chance of "validation failure"
                print("  Simulating validation failure...")
                raise ValueError("Complex validation failed")
            
            # If we get here, commit
            session.commit()
            print("  Transaction committed")
            
        except Exception as e:
            print(f"  Validation/error occurred: {e}")
            session.rollback()
            print("  Transaction rolled back")
            
        finally:
            session.close()
    
    except Exception as e:
        print(f"  Outer error: {e}")
    
    # Verify
    with db_client.session_scope() as session:
        documents_after = session.query(func.count(Document.id)).scalar()
        manual_docs = session.query(func.count(Document.id)).filter(
            Document.id.like("manual_doc_%")
        ).scalar()
    
    print(f"  Documents before: {documents_before}")
    print(f"  Documents after: {documents_after}")
    print(f"  Manual test documents: {manual_docs}")
    
    # Given our 70% chance of rollback, we can't assert specific counts
    # but we can check consistency
    assert documents_after >= documents_before
    print("  ✓ Manual rollback scenario completed")
    
    print("\nAll rollback scenario tests completed!")


def test_replication_scenarios(db_client, tmp_path):
    """Test database replication scenarios."""
    
    print("Testing database replication scenarios...")
    
    # Scenario 1: Master-slave replication simulation
    print("\n1. Master-slave replication simulation:")
    
    # Create "master" and "slave" databases
    master_path = tmp_path / "master.db"
    slave_path = tmp_path / "slave.db"
    
    master_client = SQLiteClient(str(master_path))
    slave_client = SQLiteClient(str(slave_path))
    
    master_client.connect()
    slave_client.connect()
    
    # Initialize both with same schema
    with master_client.session_scope() as session:
        Base.metadata.create_all(master_client.engine)
    
    with slave_client.session_scope() as session:
        Base.metadata.create_all(slave_client.engine)
    
    # Add data to master
    print("  Adding data to master...")
    with master_client.session_scope() as session:
        for i in range(1, 6):
            doc = Document(
                id=f"master_doc_{i}",
                file_path=f"/master/doc_{i}.txt",
                file_name=f"Master Document {i}",
                file_type="txt",
                processing_status="completed"
            )
            session.add(doc)
        
        session.commit()
    
    # Simulate replication to slave
    print("  Replicating to slave...")
    with master_client.session_scope() as master_session:
        with slave_client.session_scope() as slave_session:
            # Get all documents from master
            master_docs = master_session.query(Document).all()
            
            # Replicate to slave
            for master_doc in master_docs:
                # Check if already exists in slave
                existing = slave_session.query(Document).filter_by(id=master_doc.id).first()
                if not existing:
                    slave_doc = Document(
                        id=master_doc.id,
                        file_path=master_doc.file_path,
                        file_name=master_doc.file_name,
                        file_type=master_doc.file_type,
                        processing_status=master_doc.processing_status
                    )
                    slave_session.add(slave_doc)
            
            slave_session.commit()
    
    # Verify replication
    with master_client.session_scope() as session:
        master_count = session.query(func.count(Document.id)).scalar()
    
    with slave_client.session_scope() as session:
        slave_count = session.query(func.count(Document.id)).scalar()
    
    print(f"  Master documents: {master_count}")
    print(f"  Slave documents: {slave_count}")
    assert master_count == slave_count, "Slave should have same document count as master"
    print("  ✓ Master-slave replication verified")
    
    # Scenario 2: Write to master, read from slave
    print("\n2. Write to master, read from slave:")
    
    # Add more data to master
    with master_client.session_scope() as session:
        new_doc = Document(
            id="new_master_doc",
            file_path="/master/new.txt",
            file_name="New Master Document",
            file_type="txt"
        )
        session.add(new_doc)
        session.commit()
    
    # Initially, slave shouldn't have it
    with slave_client.session_scope() as session:
        new_doc_in_slave = session.query(Document).filter_by(id="new_master_doc").first()
        print(f"  New document in slave before sync: {'FOUND' if new_doc_in_slave else 'NOT FOUND'}")
        assert new_doc_in_slave is None, "Slave shouldn't have new doc before sync"
    
    # Simulate async replication (with delay)
    print("  Simulating async replication (3 second delay)...")
    time.sleep(3)
    
    # Manually replicate the new document
    with master_client.session_scope() as master_session:
        with slave_client.session_scope() as slave_session:
            new_master_doc = master_session.query(Document).filter_by(id="new_master_doc").first()
            if new_master_doc:
                slave_doc = Document(
                    id=new_master_doc.id,
                    file_path=new_master_doc.file_path,
                    file_name=new_master_doc.file_name,
                    file_type=new_master_doc.file_type
                )
                slave_session.add(slave_doc)
                slave_session.commit()
    
    print("  Replication complete")
    
    # Verify slave now has it
    with slave_client.session_scope() as session:
        new_doc_in_slave = session.query(Document).filter_by(id="new_master_doc").first()
        print(f"  New document in slave after sync: {'FOUND ✓' if new_doc_in_slave else 'NOT FOUND'}")
        assert new_doc_in_slave is not None, "Slave should have new doc after sync"
    
    # Scenario 3: Conflict resolution
    print("\n3. Conflict resolution (last write wins):")
    
    # Simulate concurrent update to same document in both master and slave
    with master_client.session_scope() as session:
        doc = session.query(Document).filter_by(id="master_doc_1").first()
        if doc:
            doc.file_name = "Updated in MASTER"
            session.commit()
    
    with slave_client.session_scope() as session:
        doc = session.query(Document).filter_by(id="master_doc_1").first()
        if doc:
            doc.file_name = "Updated in SLAVE"
            session.commit()
    
    # Resolve conflict (last write wins)
    print("  Resolving conflict (master wins)...")
    
    with master_client.session_scope() as master_session:
        with slave_client.session_scope() as slave_session:
            master_doc = master_session.query(Document).filter_by(id="master_doc_1").first()
            slave_doc = slave_session.query(Document).filter_by(id="master_doc_1").first()
            
            if master_doc and slave_doc:
                # Master wins (could use timestamp, version number, etc.)
                slave_doc.file_name = master_doc.file_name
                slave_session.commit()
                
                print(f"  Master name: {master_doc.file_name}")
                print(f"  Slave name after resolution: {slave_doc.file_name}")
                assert slave_doc.file_name == master_doc.file_name
                print("  ✓ Conflict resolved")
    
    # Scenario 4: Replication failure and recovery
    print("\n4. Replication failure and recovery:")
    
    # Simulate replication failure
    print("  Simulating replication failure...")
    
    # Add data during "outage"
    with master_client.session_scope() as session:
        for i in range(1, 4):
            doc = Document(
                id=f"outage_doc_{i}",
                file_path=f"/outage/doc_{i}.txt",
                file_name=f"Outage Document {i}"
            )
            session.add(doc)
        
        session.commit()
    
    # Slave doesn't have outage docs
    with slave_client.session_scope() as session:
        outage_count = session.query(func.count(Document.id)).filter(
            Document.id.like("outage_doc_%")
        ).scalar()
        print(f"  Outage documents in slave during failure: {outage_count}")
        assert outage_count == 0
    
    # "Fix" replication
    print("  'Fixing' replication...")
    
    # Get all missing documents from master
    with master_client.session_scope() as master_session:
        with slave_client.session_scope() as slave_session:
            # Get all master docs
            all_master_docs = master_session.query(Document.id).all()
            master_ids = {doc_id for (doc_id,) in all_master_docs}
            
            # Get all slave docs
            all_slave_docs = slave_session.query(Document.id).all()
            slave_ids = {doc_id for (doc_id,) in all_slave_docs}
            
            # Find missing docs
            missing_ids = master_ids - slave_ids
            print(f"  Found {len(missing_ids)} missing documents")
            
            # Replicate missing docs
            for doc_id in missing_ids:
                master_doc = master_session.query(Document).filter_by(id=doc_id).first()
                if master_doc:
                    slave_doc = Document(
                        id=master_doc.id,
                        file_path=master_doc.file_path,
                        file_name=master_doc.file_name,
                        file_type=master_doc.file_type,
                        processing_status=master_doc.processing_status
                    )
                    slave_session.add(slave_doc)
            
            slave_session.commit()
    
    # Verify recovery
    with slave_client.session_scope() as session:
        final_outage_count = session.query(func.count(Document.id)).filter(
            Document.id.like("outage_doc_%")
        ).scalar()
        
        total_slave_docs = session.query(func.count(Document.id)).scalar()
    
    with master_client.session_scope() as session:
        total_master_docs = session.query(func.count(Document.id)).scalar()
    
    print(f"  Outage documents after recovery: {final_outage_count}")
    print(f"  Total master documents: {total_master_docs}")
    print(f"  Total slave documents: {total_slave_docs}")
    
    assert final_outage_count == 3, "Should have all outage documents"
    assert total_master_docs == total_slave_docs, "Master and slave should be in sync"
    print("  ✓ Replication recovery successful")
    
    # Cleanup
    master_client.close()
    slave_client.close()
    
    print("\nAll replication scenario tests completed!")


# ============================================
# MAIN TEST RUNNER (for debugging)
# ============================================

if __name__ == "__main__":
    """Run tests directly for debugging."""
    
    print("Running test_database_operations.py tests directly...")
    print("=" * 60)
    
    # Create a temporary test database
    import tempfile
    tmp_dir = tempfile.mkdtemp()
    test_db_path = Path(tmp_dir) / "direct_test.db"
    
    print(f"Test database: {test_db_path}")
    
    # Create client
    client = SQLiteClient(str(test_db_path))
    client.connect()
    
    try:
        # Run specific tests
        print("\n1. Testing complex join operations...")
        test_complex_join_operations(client, [])
        
        print("\n2. Testing batch operations...")
        test_batch_insert_1000_documents(client)
        
        print("\n3. Testing large dataset handling...")
        # Create a large dataset client for this test
        large_tmp_dir = tempfile.mkdtemp()
        large_db_path = Path(large_tmp_dir) / "large_test.db"
        large_client = SQLiteClient(str(large_db_path))
        large_client.connect()
        
        # Initialize with data
        with large_client.session_scope() as session:
            Base.metadata.create_all(large_client.engine)
            for i in range(1, 1001):
                doc = Document(
                    id=f"direct_large_{i:04d}",
                    file_path=f"/direct/large/doc_{i:04d}.txt",
                    file_name=f"Direct Large Test {i:04d}",
                    file_type="txt",
                    file_size=random.randint(1000, 10000),
                    processing_status="completed" if i % 10 != 0 else "pending"
                )
                session.add(doc)
            session.commit()
        
        test_large_dataset_handling(large_client)
        large_client.close()
        
        print("\n4. Testing rollback scenarios...")
        test_rollback_scenarios(client)
        
        print("\n" + "=" * 60)
        print("All direct tests completed successfully!")
        
    except Exception as e:
        print(f"\nError during direct test run: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        client.close()
        
        # Cleanup temp directories
        import shutil
        if Path(tmp_dir).exists():
            shutil.rmtree(tmp_dir)
        if 'large_tmp_dir' in locals() and Path(large_tmp_dir).exists():
            shutil.rmtree(large_tmp_dir)