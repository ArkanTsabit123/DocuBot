"""
SQLite Database Client with CRUD Operations
"""

import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
import json

logger = logging.getLogger(__name__)


class SQLiteClient:
    """SQLite database client"""
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize SQLite client.
        
        Args:
            db_path: Path to SQLite database file
        """
        if db_path is None:
            from ...core.constants import DATABASE_DIR, DATABASE_NAME
            db_path = DATABASE_DIR / DATABASE_NAME
        
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
        
        logger.info(f"SQLite client initialized: {self.db_path}")
    
    def _init_database(self):
        """Initialize database schema"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        tables_sql = self._get_table_definitions()
        
        for table_name, create_sql in tables_sql.items():
            try:
                cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
                if not cursor.fetchone():
                    cursor.execute(create_sql)
                    logger.debug(f"Created table: {table_name}")
            except Exception as e:
                logger.error(f"Error creating table {table_name}: {e}")
        
        conn.commit()
        conn.close()
    
    def _get_table_definitions(self) -> Dict[str, str]:
        """Get SQL table definitions"""
        return {
            'documents': """
                CREATE TABLE documents (
                    id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size INTEGER,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processing_status TEXT DEFAULT 'pending',
                    processing_error TEXT,
                    metadata_json TEXT,
                    vector_ids_json TEXT,
                    chunk_count INTEGER DEFAULT 0,
                    word_count INTEGER DEFAULT 0,
                    language TEXT,
                    tags_json TEXT,
                    summary TEXT,
                    is_indexed BOOLEAN DEFAULT FALSE,
                    indexed_at TIMESTAMP,
                    last_accessed TIMESTAMP,
                    access_count INTEGER DEFAULT 0
                )
            """,
            'chunks': """
                CREATE TABLE chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text_content TEXT NOT NULL,
                    cleaned_text TEXT NOT NULL,
                    token_count INTEGER,
                    embedding_model TEXT,
                    vector_id TEXT NOT NULL,
                    metadata_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
            """,
            'conversations': """
                CREATE TABLE conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    message_count INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    tags_json TEXT,
                    is_archived BOOLEAN DEFAULT FALSE,
                    export_path TEXT
                )
            """,
            'messages': """
                CREATE TABLE messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tokens INTEGER,
                    model_used TEXT,
                    sources_json TEXT,
                    processing_time_ms INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
            """,
            'settings': """
                CREATE TABLE settings (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
        }
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def add_document(self, 
                    file_path: str,
                    file_name: str,
                    file_type: str,
                    file_size: int,
                    chunk_count: int = 0,
                    vector_ids: Optional[List[str]] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a new document.
        
        Args:
            file_path: Full path to document
            file_name: Document file name
            file_type: File extension/type
            file_size: File size in bytes
            chunk_count: Number of chunks
            vector_ids: List of vector IDs
            metadata: Additional metadata
            
        Returns:
            Document ID
        """
        import uuid
        
        doc_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO documents 
            (id, file_path, file_name, file_type, file_size, chunk_count, 
             vector_ids_json, metadata_json, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            doc_id,
            file_path,
            file_name,
            file_type,
            file_size,
            chunk_count,
            json.dumps(vector_ids or []),
            json.dumps(metadata or {}),
            now
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Added document: {file_name} (ID: {doc_id})")
        return doc_id
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document data or None
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
        row = cursor.fetchone()
        
        conn.close()
        
        if row:
            return self._row_to_dict(row)
        return None
    
    def update_document(self, doc_id: str, updates: Dict[str, Any]) -> bool:
        """
        Update document.
        
        Args:
            doc_id: Document ID
            updates: Fields to update
            
        Returns:
            True if successful
        """
        if not updates:
            return False
        
        set_clause = []
        values = []
        
        for key, value in updates.items():
            if key in ['vector_ids_json', 'metadata_json', 'tags_json']:
                value = json.dumps(value) if value else '[]'
            
            set_clause.append(f"{key} = ?")
            values.append(value)
        
        values.append(doc_id)
        
        sql = f"UPDATE documents SET {', '.join(set_clause)} WHERE id = ?"
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(sql, values)
            conn.commit()
            success = cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            success = False
        finally:
            conn.close()
        
        return success
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete document.
        
        Args:
            doc_id: Document ID
            
        Returns:
            True if successful
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            conn.commit()
            success = cursor.rowcount > 0
            
            if success:
                logger.info(f"Deleted document: {doc_id}")
        except Exception as e:
            logger.error(f"Error deleting document {doc_id}: {e}")
            success = False
        finally:
            conn.close()
        
        return success
    
    def list_documents(self, 
                      limit: int = 100,
                      offset: int = 0,
                      status: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List documents.
        
        Args:
            limit: Maximum number of documents
            offset: Offset for pagination
            status: Filter by processing status
            
        Returns:
            List of documents
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        where_clause = "WHERE 1=1"
        params = []
        
        if status:
            where_clause += " AND processing_status = ?"
            params.append(status)
        
        sql = f"""
            SELECT * FROM documents 
            {where_clause}
            ORDER BY upload_date DESC
            LIMIT ? OFFSET ?
        """
        
        params.extend([limit, offset])
        
        cursor.execute(sql, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_dict(row) for row in rows]
    
    def add_chunk(self,
                 document_id: str,
                 chunk_index: int,
                 text_content: str,
                 cleaned_text: str,
                 vector_id: str,
                 token_count: Optional[int] = None,
                 embedding_model: Optional[str] = None,
                 metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a chunk.
        
        Args:
            document_id: Parent document ID
            chunk_index: Chunk index in document
            text_content: Original text content
            cleaned_text: Cleaned text content
            vector_id: Vector store ID
            token_count: Number of tokens
            embedding_model: Embedding model used
            metadata: Additional metadata
            
        Returns:
            Chunk ID
        """
        import uuid
        
        chunk_id = str(uuid.uuid4())
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO chunks 
            (id, document_id, chunk_index, text_content, cleaned_text, 
             token_count, embedding_model, vector_id, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            chunk_id,
            document_id,
            chunk_index,
            text_content,
            cleaned_text,
            token_count,
            embedding_model,
            vector_id,
            json.dumps(metadata or {})
        ))
        
        conn.commit()
        conn.close()
        
        return chunk_id
    
    def get_chunks_by_document(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            List of chunks
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM chunks 
            WHERE document_id = ?
            ORDER BY chunk_index
        """, (document_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_dict(row) for row in rows]
    
    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert SQLite row to dictionary"""
        result = dict(row)
        
        json_fields = ['vector_ids_json', 'metadata_json', 'tags_json', 'sources_json']
        
        for field in json_fields:
            if field in result and result[field]:
                try:
                    result[field.replace('_json', '')] = json.loads(result[field])
                except (json.JSONDecodeError, TypeError):
                    result[field.replace('_json', '')] = []
                finally:
                    del result[field]
        
        return result
    
    def execute_query(self, sql: str, params: Tuple = ()) -> List[Dict[str, Any]]:
        """
        Execute custom SQL query.
        
        Args:
            sql: SQL query
            params: Query parameters
            
        Returns:
            Query results
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            results = [self._row_to_dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Query execution error: {e}")
            results = []
        finally:
            conn.close()
        
        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.
        
        Returns:
            Statistics dictionary
        """
        stats = {}
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT processing_status, COUNT(*) as count
                FROM documents
                GROUP BY processing_status
            """)
            stats['documents_by_status'] = dict(cursor.fetchall())
            
            cursor.execute("SELECT COUNT(*) FROM documents")
            stats['total_documents'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM chunks")
            stats['total_chunks'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM conversations")
            stats['total_conversations'] = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM messages")
            stats['total_messages'] = cursor.fetchone()[0]
            
            db_size = self.db_path.stat().st_size if self.db_path.exists() else 0
            stats['database_size_bytes'] = db_size
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error getting stats: {e}")
            stats['error'] = str(e)
        
        return stats


_db_instance = None

def get_database_client(db_path: Optional[Path] = None) -> SQLiteClient:
    """
    Get or create SQLiteClient instance.
    
    Args:
        db_path: Optional database path
        
    Returns:
        SQLiteClient instance
    """
    global _db_instance
    
    if _db_instance is None:
        _db_instance = SQLiteClient(db_path)
    
    return _db_instance


if __name__ == "__main__":
    client = SQLiteClient()
    
    doc_id = client.add_document(
        file_path="/test/document.pdf",
        file_name="test.pdf",
        file_type=".pdf",
        file_size=1024,
        chunk_count=5,
        metadata={"test": "data"}
    )
    
    print(f"Added document with ID: {doc_id}")
    
    doc = client.get_document(doc_id)
    print(f"Retrieved document: {doc['file_name']}")
    
    docs = client.list_documents(limit=5)
    print(f"Total documents: {len(docs)}")
    
    stats = client.get_stats()
    print(f"Database stats: {stats}")
