# docubot/src/database/sqlite_client.py

"""
SQLite Database Client for DocuBot
"""

import sqlite3
import json
import uuid
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

logger = logging.getLogger(__name__)


class SQLiteClient:
    """SQLite database client for DocuBot application."""
    
    def __init__(self, db_path: Optional[Union[str, Path]] = None):
        """
        Initialize SQLite client.
        
        Args:
            db_path: Path to SQLite database file. If None, uses default location.
        """
        if db_path is None:
            try:
                from ...core.constants import DATABASE_DIR, DATABASE_NAME
                db_path = DATABASE_DIR / DATABASE_NAME
            except ImportError:
                db_path = Path("data/database/sqlite.db")
        
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = None
        
        self._init_database()
        logger.info(f"SQLite client initialized: {self.db_path}")
    
    def close(self):
        """Close database connection."""
        if self.connection:
            try:
                self.connection.close()
                self.connection = None
                logger.debug(f"Closed database connection: {self.db_path}")
            except Exception as e:
                logger.error(f"Error closing database connection: {e}")
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get or create database connection with row factory."""
        if self.connection is None:
            self.connection = sqlite3.connect(str(self.db_path))
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA foreign_keys = ON")
        
        return self.connection
    
    def _init_database(self) -> None:
        """Initialize database schema with required tables."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        tables_sql = self._get_table_definitions()
        
        for table_name, create_sql in tables_sql.items():
            try:
                cursor.execute(
                    f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
                )
                if not cursor.fetchone():
                    cursor.execute(create_sql)
                    logger.debug(f"Created table: {table_name}")
            except sqlite3.Error as e:
                logger.error(f"Error creating table {table_name}: {e}")
                raise
        
        conn.commit()
    
    def _get_table_definitions(self) -> Dict[str, str]:
        """Get SQL table definitions."""
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
            """,
            'tags': """
                CREATE TABLE tags (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    color TEXT,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    usage_count INTEGER DEFAULT 0
                )
            """,
            'document_tags': """
                CREATE TABLE document_tags (
                    document_id TEXT NOT NULL,
                    tag_id TEXT NOT NULL,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (document_id, tag_id),
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                    FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
                )
            """
        }
    
    def add_document(
        self,
        file_path: str,
        file_name: str,
        file_type: str,
        file_size: int,
        chunk_count: int = 0,
        vector_ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a new document to the database."""
        doc_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO documents 
                (id, file_path, file_name, file_type, file_size, chunk_count, 
                 vector_ids_json, metadata_json, last_accessed)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                doc_id,
                str(file_path),
                file_name,
                file_type,
                file_size,
                chunk_count,
                json.dumps(vector_ids or []),
                json.dumps(metadata or {}),
                now
            ))
            
            conn.commit()
            logger.info(f"Added document: {file_name} (ID: {doc_id})")
            return doc_id
        except sqlite3.Error as e:
            conn.rollback()
            logger.error(f"Error adding document {file_name}: {e}")
            raise
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve document by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT * FROM documents WHERE id = ?", (doc_id,))
            row = cursor.fetchone()
            
            if row:
                return self._row_to_dict(row)
            return None
        finally:
            cursor.close()
    
    def update_document(self, doc_id: str, updates: Dict[str, Any]) -> bool:
        """Update document fields."""
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
            return cursor.rowcount > 0
        except sqlite3.Error as e:
            conn.rollback()
            logger.error(f"Error updating document {doc_id}: {e}")
            return False
        finally:
            cursor.close()
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete document from database."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            conn.commit()
            success = cursor.rowcount > 0
            
            if success:
                logger.info(f"Deleted document: {doc_id}")
            return success
        except sqlite3.Error as e:
            conn.rollback()
            logger.error(f"Error deleting document {doc_id}: {e}")
            return False
        finally:
            cursor.close()
    
    def list_documents(
        self,
        limit: int = 100,
        offset: int = 0,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List documents with pagination and filtering."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
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
            
            return [self._row_to_dict(row) for row in rows]
        finally:
            cursor.close()
    
    def add_chunk(
        self,
        document_id: str,
        chunk_index: int,
        text_content: str,
        cleaned_text: str,
        vector_id: str,
        token_count: Optional[int] = None,
        embedding_model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a text chunk to the database."""
        chunk_id = str(uuid.uuid4())
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
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
            return chunk_id
        except sqlite3.Error as e:
            conn.rollback()
            logger.error(f"Error adding chunk for document {document_id}: {e}")
            raise
        finally:
            cursor.close()
    
    def get_chunks_by_document(self, document_id: str) -> List[Dict[str, Any]]:
        """Retrieve all chunks for a document."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT * FROM chunks 
                WHERE document_id = ?
                ORDER BY chunk_index
            """, (document_id,))
            
            rows = cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]
        finally:
            cursor.close()
    
    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert SQLite row to dictionary with JSON parsing."""
        result = dict(row)
        
        json_fields = [
            'vector_ids_json',
            'metadata_json',
            'tags_json',
            'sources_json'
        ]
        
        for field in json_fields:
            if field in result and result[field]:
                try:
                    parsed_value = json.loads(result[field])
                    result[field.replace('_json', '')] = parsed_value
                except (json.JSONDecodeError, TypeError):
                    result[field.replace('_json', '')] = []
                finally:
                    del result[field]
        
        return result
    
    def execute_query(
        self,
        sql: str,
        params: Tuple = ()
    ) -> List[Dict[str, Any]]:
        """Execute custom SQL query."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]
        except sqlite3.Error as e:
            logger.error(f"Query execution error: {e}")
            raise
        finally:
            cursor.close()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        stats = {}
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
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
            
            return stats
        except sqlite3.Error as e:
            logger.error(f"Error getting stats: {e}")
            raise
        finally:
            cursor.close()
    
    def add_conversation(
        self,
        title: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """Add a new conversation."""
        conversation_id = str(uuid.uuid4())
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO conversations 
                (id, title, tags_json)
                VALUES (?, ?, ?)
            """, (
                conversation_id,
                title,
                json.dumps(tags or [])
            ))
            
            conn.commit()
            logger.info(f"Added conversation: {title or conversation_id}")
            return conversation_id
        except sqlite3.Error as e:
            conn.rollback()
            logger.error(f"Error adding conversation: {e}")
            raise
        finally:
            cursor.close()
    
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        tokens: Optional[int] = None,
        model_used: Optional[str] = None,
        sources: Optional[List[Dict]] = None,
        processing_time_ms: Optional[int] = None
    ) -> str:
        """Add a message to a conversation."""
        message_id = str(uuid.uuid4())
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO messages 
                (id, conversation_id, role, content, tokens, 
                 model_used, sources_json, processing_time_ms)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                message_id,
                conversation_id,
                role,
                content,
                tokens,
                model_used,
                json.dumps(sources or []),
                processing_time_ms
            ))
            
            cursor.execute("""
                UPDATE conversations 
                SET message_count = message_count + 1,
                    total_tokens = total_tokens + ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (tokens or 0, conversation_id))
            
            conn.commit()
            return message_id
        except sqlite3.Error as e:
            conn.rollback()
            logger.error(f"Error adding message to conversation {conversation_id}: {e}")
            raise
        finally:
            cursor.close()
    
    def get_conversation_messages(
        self,
        conversation_id: str,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Get messages for a conversation."""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT * FROM messages 
                WHERE conversation_id = ?
                ORDER BY created_at
                LIMIT ? OFFSET ?
            """, (conversation_id, limit, offset))
            
            rows = cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]
        finally:
            cursor.close()


_db_instance: Optional[SQLiteClient] = None


def get_database_client(db_path: Optional[Union[str, Path]] = None) -> SQLiteClient:
    """Get or create SQLiteClient instance."""
    global _db_instance
    
    if _db_instance is None or (db_path is not None and 
                                _db_instance.db_path != Path(db_path)):
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
    
    client.close()