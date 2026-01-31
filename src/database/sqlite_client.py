"""
SQLite Database Client for DocuBot
Implements comprehensive database operations for document management, conversations,
and metadata storage using SQLAlchemy ORM.
"""

import sqlite3
from sqlite3 import Error, Connection, Cursor
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Generator
from datetime import datetime, timedelta
import logging
from contextlib import contextmanager
import hashlib
import uuid
import gzip
import shutil

# SQLAlchemy imports
from sqlalchemy import create_engine, text, select, update, delete, func, and_, or_, not_
from sqlalchemy.orm import Session, sessionmaker, scoped_session, joinedload
from sqlalchemy.exc import SQLAlchemyError, IntegrityError as SQLAIntegrityError
from sqlalchemy.pool import QueuePool

# Project imports
from .models import Base, Document, Chunk, Conversation, Message, Tag, DocumentTag, Setting
from ..core.config import AppConfig as Config
from ..utilities.logger import get_logger
from ..core.exceptions import DatabaseException, ValidationError

# ============================================================================
# CUSTOM EXCEPTIONS
# ============================================================================

class DatabaseError(Exception):
    """Base exception for database errors."""
    pass

class ConnectionError(DatabaseError):
    """Connection-related errors."""
    pass

class QueryError(DatabaseError):
    """Query execution errors."""
    pass

class IntegrityError(DatabaseError):
    """Data integrity violation errors."""
    pass

class NotFoundError(DatabaseError):
    """Resource not found errors."""
    pass

class ValidationError(DatabaseError):
    """Data validation errors."""
    pass

# ============================================================================
# MAIN SQLiteClient CLASS
# ============================================================================

class SQLiteClient:
    """
    Main SQLite database client for DocuBot.
    
    Provides comprehensive CRUD operations for documents, chunks, conversations,
    tags, and system settings with proper transaction management and error handling.
    """
    
    def __init__(self, db_path: Optional[str] = None, config: Optional[Config] = None):
        """
        Initialize SQLiteClient.
        
        Args:
            db_path: Path to SQLite database file. Defaults to data/database/docubot.db
            config: Configuration object for database settings
        """
        self.config = config or Config()
        
        # Set database path
        if db_path:
            self.db_path = Path(db_path)
        else:
            self.db_path = Path(self.config.get('storage.database.path', 'data/database/docubot.db'))
        
        # Ensure directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize SQLAlchemy components
        self.engine = None
        self.SessionFactory = None
        self.session = None
        
        # Initialize logger
        self.logger = get_logger(__name__)
        
        # Connection parameters
        self.pool_size = self.config.get('storage.database.pool_size', 5)
        self.max_overflow = self.config.get('storage.database.max_overflow', 10)
        self.pool_timeout = self.config.get('storage.database.pool_timeout', 30)
        self.pool_recycle = self.config.get('storage.database.pool_recycle', 3600)
        
        # Performance settings
        self.batch_size = self.config.get('storage.database.batch_size', 100)
        self.query_timeout = self.config.get('storage.database.query_timeout', 30)
        
        # Track connection state
        self._is_connected = False
        self._connection_count = 0
        
        self.logger.info(f"SQLiteClient initialized for database: {self.db_path}")
    
    def connect(self) -> bool:
        """
        Establish database connection and create tables if they don't exist.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Create SQLAlchemy engine with connection pooling
            database_url = f"sqlite:///{self.db_path}"
            
            self.engine = create_engine(
                database_url,
                poolclass=QueuePool,
                pool_size=self.pool_size,
                max_overflow=self.max_overflow,
                pool_timeout=self.pool_timeout,
                pool_recycle=self.pool_recycle,
                echo=self.config.get('storage.database.debug', False)
            )
            
            # Create session factory
            self.SessionFactory = sessionmaker(
                bind=self.engine,
                expire_on_commit=False,
                autoflush=False
            )
            
            # Create scoped session for thread safety
            self.Session = scoped_session(self.SessionFactory)
            
            # Configure SQLite pragmas for better performance and reliability
            with self.engine.connect() as conn:
                # Enable foreign keys
                conn.execute(text("PRAGMA foreign_keys = ON"))
                
                # Enable Write-Ahead Logging for better concurrency
                conn.execute(text("PRAGMA journal_mode = WAL"))
                
                # Set synchronous mode for balance between safety and performance
                conn.execute(text("PRAGMA synchronous = NORMAL"))
                
                # Increase cache size
                conn.execute(text("PRAGMA cache_size = -2000"))  # 2MB cache
                
                # Set temp store to memory
                conn.execute(text("PRAGMA temp_store = MEMORY"))
                
                # Set page size
                conn.execute(text("PRAGMA page_size = 4096"))
                
                conn.commit()
            
            # Create all tables if they don't exist
            Base.metadata.create_all(self.engine)
            
            # Initialize database with default settings if empty
            self._initialize_database()
            
            self._is_connected = True
            self.logger.info("Database connection established successfully")
            
            # Perform health check
            health = self.health_check()
            if health['status'] == 'healthy':
                self.logger.info(f"Database health check passed: {health}")
                return True
            else:
                self.logger.warning(f"Database health check issues: {health}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            self._is_connected = False
            raise ConnectionError(f"Database connection failed: {e}")
    
    def close(self) -> None:
        """
        Close database connection and cleanup resources.
        """
        try:
            if self.session:
                self.session.remove()
            
            if self.engine:
                self.engine.dispose()
                self.engine = None
            
            self.SessionFactory = None
            self.Session = None
            self._is_connected = False
            
            self.logger.info("Database connection closed")
            
        except Exception as e:
            self.logger.error(f"Error closing database connection: {e}")
    
    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        """
        Context manager for database sessions with automatic commit/rollback.
        
        Yields:
            Session: SQLAlchemy session object
            
        Example:
            with db.session_scope() as session:
                document = Document(file_name="test.pdf")
                session.add(document)
        """
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"Session error: {e}")
            raise
        finally:
            session.close()
    
    def is_connected(self) -> bool:
        """
        Check if database is connected.
        
        Returns:
            bool: True if connected, False otherwise
        """
        return self._is_connected
    
    # ========================================================================
    # DOCUMENT OPERATIONS
    # ========================================================================
    
    def add_document(self, **kwargs) -> str:
        """
        Add a new document to the database.
        
        Args:
            **kwargs: Document attributes (file_path, file_name, file_type, etc.)
            
        Returns:
            str: Document ID
            
        Raises:
            ValidationError: If required fields are missing
            IntegrityError: If document already exists
        """
        required_fields = ['file_path', 'file_name', 'file_type']
        for field in required_fields:
            if field not in kwargs:
                raise ValidationError(f"Missing required field: {field}")
        
        # Generate unique document ID
        document_id = kwargs.get('id') or self._generate_id(kwargs['file_path'])
        
        # Prepare document data
        document_data = {
            'id': document_id,
            'file_path': str(kwargs['file_path']),
            'file_name': kwargs['file_name'],
            'file_type': kwargs['file_type'],
            'file_size': kwargs.get('file_size'),
            'processing_status': kwargs.get('processing_status', 'pending'),
            'metadata_json': json.dumps(kwargs.get('metadata', {})),
            'chunk_count': kwargs.get('chunk_count', 0),
            'word_count': kwargs.get('word_count', 0),
            'language': kwargs.get('language'),
            'tags_json': json.dumps(kwargs.get('tags', [])),
            'summary': kwargs.get('summary'),
            'is_indexed': kwargs.get('is_indexed', False),
            'indexed_at': kwargs.get('indexed_at'),
            'last_accessed': datetime.now(),
            'access_count': 0
        }
        
        # Remove None values
        document_data = {k: v for k, v in document_data.items() if v is not None}
        
        try:
            with self.session_scope() as session:
                document = Document(**document_data)
                session.add(document)
                
                # Add tags if provided
                tags = kwargs.get('tags', [])
                if tags:
                    for tag_name in tags:
                        tag = self._get_or_create_tag(session, tag_name)
                        document.tags.append(tag)
                
                self.logger.info(f"Document added: {document_id} ({kwargs['file_name']})")
                
            return document_id
            
        except SQLAIntegrityError as e:
            raise IntegrityError(f"Document already exists or integrity violation: {e}")
        except Exception as e:
            self.logger.error(f"Error adding document: {e}")
            raise DatabaseError(f"Failed to add document: {e}")
    
    def get_document(self, document_id: str, include_chunks: bool = False) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by ID.
        
        Args:
            document_id: Document ID
            include_chunks: Whether to include chunk data
            
        Returns:
            Optional[Dict]: Document data as dictionary, or None if not found
        """
        try:
            with self.session_scope() as session:
                # Build query
                query = session.query(Document).filter(Document.id == document_id)
                
                if include_chunks:
                    query = query.options(joinedload(Document.chunks))
                
                document = query.first()
                
                if not document:
                    raise NotFoundError(f"Document not found: {document_id}")
                
                # Update access statistics
                document.last_accessed = datetime.now()
                document.access_count += 1
                session.commit()
                
                return self._document_to_dict(document, include_chunks)
                
        except NotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Error retrieving document {document_id}: {e}")
            raise DatabaseError(f"Failed to retrieve document: {e}")
    
    def update_document(self, document_id: str, **updates) -> bool:
        """
        Update document metadata.
        
        Args:
            document_id: Document ID
            **updates: Fields to update
            
        Returns:
            bool: True if update successful
            
        Raises:
            NotFoundError: If document not found
        """
        try:
            with self.session_scope() as session:
                document = session.query(Document).filter(Document.id == document_id).first()
                
                if not document:
                    raise NotFoundError(f"Document not found: {document_id}")
                
                # Update fields
                for key, value in updates.items():
                    if hasattr(document, key):
                        if key in ['metadata', 'tags']:
                            # Handle JSON fields
                            setattr(document, f"{key}_json", json.dumps(value))
                        else:
                            setattr(document, key, value)
                
                document.updated_at = datetime.now()
                
                self.logger.info(f"Document updated: {document_id}")
                return True
                
        except NotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Error updating document {document_id}: {e}")
            raise DatabaseError(f"Failed to update document: {e}")
    
    def delete_document(self, document_id: str, cascade: bool = True) -> bool:
        """
        Delete a document and optionally its chunks.
        
        Args:
            document_id: Document ID
            cascade: Whether to delete associated chunks
            
        Returns:
            bool: True if deletion successful
            
        Raises:
            NotFoundError: If document not found
        """
        try:
            with self.session_scope() as session:
                document = session.query(Document).filter(Document.id == document_id).first()
                
                if not document:
                    raise NotFoundError(f"Document not found: {document_id}")
                
                # Delete chunks first if cascade is True
                if cascade:
                    chunks = session.query(Chunk).filter(Chunk.document_id == document_id).all()
                    for chunk in chunks:
                        session.delete(chunk)
                
                # Delete document tags associations
                session.query(DocumentTag).filter(DocumentTag.document_id == document_id).delete()
                
                # Delete document
                session.delete(document)
                
                self.logger.info(f"Document deleted: {document_id}")
                return True
                
        except NotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Error deleting document {document_id}: {e}")
            raise DatabaseError(f"Failed to delete document: {e}")
    
    def list_documents(
        self,
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        sort_by: str = "upload_date",
        sort_order: str = "DESC"
    ) -> List[Dict[str, Any]]:
        """
        List documents with filtering, sorting, and pagination.
        
        Args:
            limit: Maximum number of documents to return
            offset: Number of documents to skip
            filters: Dictionary of filter conditions
            sort_by: Field to sort by
            sort_order: Sort order ("ASC" or "DESC")
            
        Returns:
            List[Dict]: List of document dictionaries
        """
        try:
            with self.session_scope() as session:
                # Build query
                query = session.query(Document)
                
                # Apply filters
                if filters:
                    query = self._apply_filters(query, Document, filters)
                
                # Apply sorting
                sort_column = getattr(Document, sort_by, Document.upload_date)
                if sort_order.upper() == "DESC":
                    query = query.order_by(sort_column.desc())
                else:
                    query = query.order_by(sort_column.asc())
                
                # Apply pagination
                query = query.limit(limit).offset(offset)
                
                # Execute query
                documents = query.all()
                
                return [self._document_to_dict(doc) for doc in documents]
                
        except Exception as e:
            self.logger.error(f"Error listing documents: {e}")
            raise DatabaseError(f"Failed to list documents: {e}")
    
    # ========================================================================
    # CHUNK OPERATIONS
    # ========================================================================
    
    def add_chunks(self, document_id: str, chunks: List[Dict[str, Any]]) -> List[str]:
        """
        Add multiple chunks for a document.
        
        Args:
            document_id: Document ID
            chunks: List of chunk dictionaries
            
        Returns:
            List[str]: List of chunk IDs
            
        Raises:
            NotFoundError: If document not found
        """
        chunk_ids = []
        
        try:
            with self.session_scope() as session:
                # Verify document exists
                document = session.query(Document).filter(Document.id == document_id).first()
                if not document:
                    raise NotFoundError(f"Document not found: {document_id}")
                
                # Add chunks in batches
                for i in range(0, len(chunks), self.batch_size):
                    batch = chunks[i:i + self.batch_size]
                    
                    for chunk_data in batch:
                        chunk_id = chunk_data.get('id') or self._generate_id(f"{document_id}_{chunk_data['chunk_index']}")
                        
                        chunk = Chunk(
                            id=chunk_id,
                            document_id=document_id,
                            chunk_index=chunk_data['chunk_index'],
                            text_content=chunk_data.get('text', ''),
                            cleaned_text=chunk_data.get('cleaned_text', ''),
                            token_count=chunk_data.get('token_count'),
                            embedding_model=chunk_data.get('embedding_model'),
                            vector_id=chunk_data.get('vector_id', ''),
                            metadata_json=json.dumps(chunk_data.get('metadata', {}))
                        )
                        
                        session.add(chunk)
                        chunk_ids.append(chunk_id)
                    
                    # Update document chunk count
                    document.chunk_count = len(chunks)
                    document.updated_at = datetime.now()
                
                self.logger.info(f"Added {len(chunks)} chunks for document {document_id}")
                
            return chunk_ids
            
        except NotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Error adding chunks for document {document_id}: {e}")
            raise DatabaseError(f"Failed to add chunks: {e}")
    
    def get_chunks_by_document(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get all chunks for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            List[Dict]: List of chunk dictionaries
            
        Raises:
            NotFoundError: If document not found
        """
        try:
            with self.session_scope() as session:
                # Verify document exists
                document = session.query(Document).filter(Document.id == document_id).first()
                if not document:
                    raise NotFoundError(f"Document not found: {document_id}")
                
                chunks = session.query(Chunk).filter(
                    Chunk.document_id == document_id
                ).order_by(Chunk.chunk_index).all()
                
                return [self._chunk_to_dict(chunk) for chunk in chunks]
                
        except NotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Error getting chunks for document {document_id}: {e}")
            raise DatabaseError(f"Failed to get chunks: {e}")
    
    def get_chunk_by_vector_id(self, vector_id: str) -> Optional[Dict[str, Any]]:
        """
        Find chunk by ChromaDB vector ID.
        
        Args:
            vector_id: Vector database ID
            
        Returns:
            Optional[Dict]: Chunk data or None if not found
        """
        try:
            with self.session_scope() as session:
                chunk = session.query(Chunk).filter(Chunk.vector_id == vector_id).first()
                
                if chunk:
                    return self._chunk_to_dict(chunk)
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting chunk by vector_id {vector_id}: {e}")
            raise DatabaseError(f"Failed to get chunk: {e}")
    
    def update_chunk_metadata(self, chunk_id: str, **updates) -> bool:
        """
        Update chunk metadata.
        
        Args:
            chunk_id: Chunk ID
            **updates: Fields to update
            
        Returns:
            bool: True if update successful
            
        Raises:
            NotFoundError: If chunk not found
        """
        try:
            with self.session_scope() as session:
                chunk = session.query(Chunk).filter(Chunk.id == chunk_id).first()
                
                if not chunk:
                    raise NotFoundError(f"Chunk not found: {chunk_id}")
                
                # Update fields
                for key, value in updates.items():
                    if hasattr(chunk, key):
                        if key == 'metadata':
                            chunk.metadata_json = json.dumps(value)
                        else:
                            setattr(chunk, key, value)
                
                chunk.updated_at = datetime.now()
                
                self.logger.info(f"Chunk updated: {chunk_id}")
                return True
                
        except NotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Error updating chunk {chunk_id}: {e}")
            raise DatabaseError(f"Failed to update chunk: {e}")
    
    # ========================================================================
    # CONVERSATION OPERATIONS
    # ========================================================================
    
    def create_conversation(self, title: str = "", **metadata) -> str:
        """
        Create a new conversation.
        
        Args:
            title: Conversation title
            **metadata: Additional metadata
            
        Returns:
            str: Conversation ID
        """
        conversation_id = self._generate_id(f"conv_{datetime.now().timestamp()}")
        
        try:
            with self.session_scope() as session:
                conversation = Conversation(
                    id=conversation_id,
                    title=title or f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    tags_json=json.dumps(metadata.get('tags', [])),
                    metadata_json=json.dumps(metadata)
                )
                
                session.add(conversation)
                
                self.logger.info(f"Conversation created: {conversation_id}")
                
            return conversation_id
            
        except Exception as e:
            self.logger.error(f"Error creating conversation: {e}")
            raise DatabaseError(f"Failed to create conversation: {e}")
    
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        sources: Optional[List[Dict]] = None,
        model_used: Optional[str] = None,
        tokens: Optional[int] = None,
        **metadata
    ) -> str:
        """
        Add a message to a conversation.
        
        Args:
            conversation_id: Conversation ID
            role: Message role (user, assistant, system)
            content: Message content
            sources: List of source documents
            model_used: Model used to generate response
            tokens: Number of tokens in message
            **metadata: Additional metadata
            
        Returns:
            str: Message ID
            
        Raises:
            NotFoundError: If conversation not found
        """
        message_id = self._generate_id(f"msg_{conversation_id}_{datetime.now().timestamp()}")
        
        try:
            with self.session_scope() as session:
                # Verify conversation exists
                conversation = session.query(Conversation).filter(
                    Conversation.id == conversation_id
                ).first()
                
                if not conversation:
                    raise NotFoundError(f"Conversation not found: {conversation_id}")
                
                # Create message
                message = Message(
                    id=message_id,
                    conversation_id=conversation_id,
                    role=role,
                    content=content,
                    tokens=tokens,
                    model_used=model_used,
                    sources_json=json.dumps(sources or []),
                    metadata_json=json.dumps(metadata),
                    processing_time_ms=metadata.get('processing_time_ms')
                )
                
                session.add(message)
                
                # Update conversation statistics
                conversation.message_count += 1
                conversation.total_tokens = (conversation.total_tokens or 0) + (tokens or 0)
                conversation.updated_at = datetime.now()
                
                self.logger.info(f"Message added to conversation {conversation_id}: {role}")
                
            return message_id
            
        except NotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Error adding message to conversation {conversation_id}: {e}")
            raise DatabaseError(f"Failed to add message: {e}")
    
    def get_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get conversation with all messages.
        
        Args:
            conversation_id: Conversation ID
            
        Returns:
            Dict: Conversation data with messages
            
        Raises:
            NotFoundError: If conversation not found
        """
        try:
            with self.session_scope() as session:
                conversation = session.query(Conversation).filter(
                    Conversation.id == conversation_id
                ).first()
                
                if not conversation:
                    raise NotFoundError(f"Conversation not found: {conversation_id}")
                
                # Get messages
                messages = session.query(Message).filter(
                    Message.conversation_id == conversation_id
                ).order_by(Message.created_at).all()
                
                # Convert to dictionary
                conversation_dict = self._conversation_to_dict(conversation)
                conversation_dict['messages'] = [self._message_to_dict(msg) for msg in messages]
                
                return conversation_dict
                
        except NotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Error getting conversation {conversation_id}: {e}")
            raise DatabaseError(f"Failed to get conversation: {e}")
    
    def list_conversations(
        self,
        limit: int = 50,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None,
        include_archived: bool = False
    ) -> List[Dict[str, Any]]:
        """
        List conversations with optional filters.
        
        Args:
            limit: Maximum number of conversations
            offset: Number of conversations to skip
            filters: Filter conditions
            include_archived: Whether to include archived conversations
            
        Returns:
            List[Dict]: List of conversation dictionaries
        """
        try:
            with self.session_scope() as session:
                # Build query
                query = session.query(Conversation)
                
                # Exclude archived unless specified
                if not include_archived:
                    query = query.filter(Conversation.is_archived == False)
                
                # Apply filters
                if filters:
                    query = self._apply_filters(query, Conversation, filters)
                
                # Sort by most recent
                query = query.order_by(Conversation.updated_at.desc())
                
                # Apply pagination
                query = query.limit(limit).offset(offset)
                
                # Execute query
                conversations = query.all()
                
                return [self._conversation_to_dict(conv) for conv in conversations]
                
        except Exception as e:
            self.logger.error(f"Error listing conversations: {e}")
            raise DatabaseError(f"Failed to list conversations: {e}")
    
    # ========================================================================
    # TAG OPERATIONS
    # ========================================================================
    
    def create_tag(self, name: str, color: str = "", description: str = "") -> str:
        """
        Create a new tag.
        
        Args:
            name: Tag name
            color: Tag color (hex code)
            description: Tag description
            
        Returns:
            str: Tag ID
        """
        tag_id = self._generate_id(f"tag_{name.lower()}")
        
        try:
            with self.session_scope() as session:
                # Check if tag already exists
                existing_tag = session.query(Tag).filter(Tag.name == name).first()
                if existing_tag:
                    return existing_tag.id
                
                # Create new tag
                tag = Tag(
                    id=tag_id,
                    name=name,
                    color=color,
                    description=description
                )
                
                session.add(tag)
                
                self.logger.info(f"Tag created: {name} ({tag_id})")
                
            return tag_id
            
        except Exception as e:
            self.logger.error(f"Error creating tag {name}: {e}")
            raise DatabaseError(f"Failed to create tag: {e}")
    
    def tag_document(self, document_id: str, tag_name: str) -> bool:
        """
        Associate a tag with a document.
        
        Args:
            document_id: Document ID
            tag_name: Tag name
            
        Returns:
            bool: True if association successful
            
        Raises:
            NotFoundError: If document or tag not found
        """
        try:
            with self.session_scope() as session:
                # Get document
                document = session.query(Document).filter(Document.id == document_id).first()
                if not document:
                    raise NotFoundError(f"Document not found: {document_id}")
                
                # Get or create tag
                tag = self._get_or_create_tag(session, tag_name)
                
                # Check if already tagged
                existing_association = session.query(DocumentTag).filter(
                    DocumentTag.document_id == document_id,
                    DocumentTag.tag_id == tag.id
                ).first()
                
                if existing_association:
                    return True  # Already tagged
                
                # Create association
                association = DocumentTag(
                    document_id=document_id,
                    tag_id=tag.id
                )
                
                session.add(association)
                
                # Update tag usage count
                tag.usage_count += 1
                
                # Update document tags
                doc_tags = json.loads(document.tags_json or '[]')
                if tag_name not in doc_tags:
                    doc_tags.append(tag_name)
                    document.tags_json = json.dumps(doc_tags)
                
                self.logger.info(f"Document {document_id} tagged with '{tag_name}'")
                return True
                
        except NotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Error tagging document {document_id} with '{tag_name}': {e}")
            raise DatabaseError(f"Failed to tag document: {e}")
    
    def get_tags_for_document(self, document_id: str) -> List[Dict[str, Any]]:
        """
        Get all tags for a document.
        
        Args:
            document_id: Document ID
            
        Returns:
            List[Dict]: List of tag dictionaries
            
        Raises:
            NotFoundError: If document not found
        """
        try:
            with self.session_scope() as session:
                # Verify document exists
                document = session.query(Document).filter(Document.id == document_id).first()
                if not document:
                    raise NotFoundError(f"Document not found: {document_id}")
                
                # Get tags through association
                tags = session.query(Tag).join(DocumentTag).filter(
                    DocumentTag.document_id == document_id
                ).all()
                
                return [self._tag_to_dict(tag) for tag in tags]
                
        except NotFoundError:
            raise
        except Exception as e:
            self.logger.error(f"Error getting tags for document {document_id}: {e}")
            raise DatabaseError(f"Failed to get tags: {e}")
    
    def search_by_tag(self, tag_name: str) -> List[Dict[str, Any]]:
        """
        Find documents by tag name.
        
        Args:
            tag_name: Tag name to search for
            
        Returns:
            List[Dict]: List of document dictionaries
        """
        try:
            with self.session_scope() as session:
                # Get tag
                tag = session.query(Tag).filter(Tag.name == tag_name).first()
                
                if not tag:
                    return []  # No documents with this tag
                
                # Get documents with this tag
                documents = session.query(Document).join(DocumentTag).filter(
                    DocumentTag.tag_id == tag.id
                ).all()
                
                return [self._document_to_dict(doc) for doc in documents]
                
        except Exception as e:
            self.logger.error(f"Error searching by tag '{tag_name}': {e}")
            raise DatabaseError(f"Failed to search by tag: {e}")
    
    # ========================================================================
    # QUERY & SEARCH METHODS
    # ========================================================================
    
    def search_documents(
        self,
        query: str,
        field: str = "file_name",
        exact_match: bool = False,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search documents in specified field.
        
        Args:
            query: Search query
            field: Field to search in
            exact_match: Whether to perform exact match
            limit: Maximum results
            
        Returns:
            List[Dict]: List of matching documents
        """
        try:
            with self.session_scope() as session:
                # Build query
                db_query = session.query(Document)
                
                if field == "file_name":
                    if exact_match:
                        db_query = db_query.filter(Document.file_name == query)
                    else:
                        db_query = db_query.filter(Document.file_name.ilike(f"%{query}%"))
                
                elif field == "content":
                    # Search in chunks
                    chunks = session.query(Chunk).filter(
                        Chunk.text_content.ilike(f"%{query}%")
                    ).subquery()
                    
                    db_query = db_query.join(
                        chunks,
                        Document.id == chunks.c.document_id
                    ).distinct()
                
                # Apply limit and execute
                documents = db_query.limit(limit).all()
                
                return [self._document_to_dict(doc) for doc in documents]
                
        except Exception as e:
            self.logger.error(f"Error searching documents for '{query}' in {field}: {e}")
            raise DatabaseError(f"Failed to search documents: {e}")
    
    def get_document_statistics(self) -> Dict[str, Any]:
        """
        Get document statistics.
        
        Returns:
            Dict: Statistics including counts by file type, storage usage, etc.
        """
        try:
            with self.session_scope() as session:
                stats = {}
                
                # Total document count
                stats['total_documents'] = session.query(func.count(Document.id)).scalar() or 0
                
                # Count by file type
                file_type_counts = session.query(
                    Document.file_type,
                    func.count(Document.id)
                ).group_by(Document.file_type).all()
                
                stats['by_file_type'] = {ftype: count for ftype, count in file_type_counts}
                
                # Count by processing status
                status_counts = session.query(
                    Document.processing_status,
                    func.count(Document.id)
                ).group_by(Document.processing_status).all()
                
                stats['by_status'] = {status: count for status, count in status_counts}
                
                # Total storage usage
                total_size = session.query(func.sum(Document.file_size)).scalar() or 0
                stats['total_storage_bytes'] = total_size
                stats['total_storage_mb'] = total_size / (1024 * 1024)
                
                # Chunk statistics
                total_chunks = session.query(func.count(Chunk.id)).scalar() or 0
                stats['total_chunks'] = total_chunks
                
                # Average chunks per document
                if stats['total_documents'] > 0:
                    stats['avg_chunks_per_document'] = total_chunks / stats['total_documents']
                else:
                    stats['avg_chunks_per_document'] = 0
                
                # Recent activity
                week_ago = datetime.now() - timedelta(days=7)
                recent_docs = session.query(func.count(Document.id)).filter(
                    Document.upload_date >= week_ago
                ).scalar() or 0
                
                stats['recent_documents_7d'] = recent_docs
                
                return stats
                
        except Exception as e:
            self.logger.error(f"Error getting document statistics: {e}")
            raise DatabaseError(f"Failed to get statistics: {e}")
    
    def get_recent_documents(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Get documents uploaded in last N days.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List[Dict]: List of recent documents
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days)
            
            with self.session_scope() as session:
                documents = session.query(Document).filter(
                    Document.upload_date >= cutoff_date
                ).order_by(Document.upload_date.desc()).all()
                
                return [self._document_to_dict(doc) for doc in documents]
                
        except Exception as e:
            self.logger.error(f"Error getting recent documents ({days} days): {e}")
            raise DatabaseError(f"Failed to get recent documents: {e}")
    
    def get_processing_queue(self) -> List[Dict[str, Any]]:
        """
        Get documents with pending/in_progress status.
        
        Returns:
            List[Dict]: List of documents in processing queue
        """
        try:
            with self.session_scope() as session:
                documents = session.query(Document).filter(
                    Document.processing_status.in_(['pending', 'in_progress'])
                ).order_by(Document.upload_date).all()
                
                return [self._document_to_dict(doc) for doc in documents]
                
        except Exception as e:
            self.logger.error(f"Error getting processing queue: {e}")
            raise DatabaseError(f"Failed to get processing queue: {e}")
    
    # ========================================================================
    # ADMINISTRATION METHODS
    # ========================================================================
    
    def backup_database(self, backup_path: Optional[str] = None) -> str:
        """
        Create a backup of the database.
        
        Args:
            backup_path: Path to save backup (optional)
            
        Returns:
            str: Path to backup file
        """
        try:
            if not self.db_path.exists():
                raise FileNotFoundError(f"Database file not found: {self.db_path}")
            
            # Generate backup path
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if backup_path:
                backup_file = Path(backup_path)
            else:
                backup_dir = self.db_path.parent / 'backups'
                backup_dir.mkdir(exist_ok=True)
                backup_file = backup_dir / f"docubot_backup_{timestamp}.db"
            
            # Close connections before backup
            self.close()
            
            # Copy database file
            shutil.copy2(self.db_path, backup_file)
            
            # Optional: compress backup
            if self.config.get('storage.database.compress_backups', True):
                compressed_file = backup_file.with_suffix('.db.gz')
                with open(backup_file, 'rb') as f_in:
                    with gzip.open(compressed_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                # Remove uncompressed backup
                backup_file.unlink()
                backup_file = compressed_file
            
            # Reconnect
            self.connect()
            
            self.logger.info(f"Database backup created: {backup_file}")
            return str(backup_file)
            
        except Exception as e:
            self.logger.error(f"Error creating database backup: {e}")
            raise DatabaseError(f"Failed to create backup: {e}")
    
    def restore_database(self, backup_path: str) -> bool:
        """
        Restore database from backup.
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            bool: True if restore successful
        """
        try:
            backup_file = Path(backup_path)
            if not backup_file.exists():
                raise FileNotFoundError(f"Backup file not found: {backup_path}")
            
            # Close connections
            self.close()
            
            # Handle compressed backups
            if backup_file.suffix == '.gz':
                # Decompress
                decompressed_file = backup_file.with_suffix('')
                with gzip.open(backup_file, 'rb') as f_in:
                    with open(decompressed_file, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                backup_file = decompressed_file
            
            # Create backup of current database
            current_backup = self.db_path.with_suffix(f".pre_restore_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db")
            if self.db_path.exists():
                shutil.copy2(self.db_path, current_backup)
            
            # Replace database
            shutil.copy2(backup_file, self.db_path)
            
            # Clean up temporary files
            if backup_file.suffix == '' and backup_file != self.db_path:
                backup_file.unlink()
            
            # Reconnect
            self.connect()
            
            self.logger.info(f"Database restored from: {backup_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error restoring database from {backup_path}: {e}")
            raise DatabaseError(f"Failed to restore database: {e}")
    
    def vacuum_database(self) -> bool:
        """
        Optimize database and reclaim space.
        
        Returns:
            bool: True if vacuum successful
        """
        try:
            with self.session_scope() as session:
                session.execute(text("VACUUM"))
                session.commit()
            
            self.logger.info("Database vacuum completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Error vacuuming database: {e}")
            raise DatabaseError(f"Failed to vacuum database: {e}")
    
    def export_to_json(self, output_path: str) -> bool:
        """
        Export database data to JSON format.
        
        Args:
            output_path: Path to save JSON file
            
        Returns:
            bool: True if export successful
        """
        try:
            export_data = {
                'exported_at': datetime.now().isoformat(),
                'database': str(self.db_path),
                'documents': [],
                'conversations': [],
                'tags': []
            }
            
            with self.session_scope() as session:
                # Export documents
                documents = session.query(Document).all()
                for doc in documents:
                    doc_dict = self._document_to_dict(doc, include_chunks=True)
                    export_data['documents'].append(doc_dict)
                
                # Export conversations
                conversations = session.query(Conversation).all()
                for conv in conversations:
                    conv_dict = self._conversation_to_dict(conv)
                    
                    # Get messages
                    messages = session.query(Message).filter(
                        Message.conversation_id == conv.id
                    ).all()
                    
                    conv_dict['messages'] = [self._message_to_dict(msg) for msg in messages]
                    export_data['conversations'].append(conv_dict)
                
                # Export tags
                tags = session.query(Tag).all()
                export_data['tags'] = [self._tag_to_dict(tag) for tag in tags]
            
            # Write to file
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Database exported to JSON: {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting database to JSON: {e}")
            raise DatabaseError(f"Failed to export database: {e}")
    
    def import_from_json(self, json_path: str) -> bool:
        """
        Import data from JSON file.
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            bool: True if import successful
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
            
            with self.session_scope() as session:
                # Import documents
                for doc_data in import_data.get('documents', []):
                    # Check if document already exists
                    existing = session.query(Document).filter(
                        Document.id == doc_data['id']
                    ).first()
                    
                    if not existing:
                        document = Document(**self._prepare_document_data(doc_data))
                        session.add(document)
                
                # Import conversations
                for conv_data in import_data.get('conversations', []):
                    # Check if conversation already exists
                    existing = session.query(Conversation).filter(
                        Conversation.id == conv_data['id']
                    ).first()
                    
                    if not existing:
                        conversation = Conversation(**self._prepare_conversation_data(conv_data))
                        session.add(conversation)
                        
                        # Import messages
                        for msg_data in conv_data.get('messages', []):
                            message = Message(**self._prepare_message_data(msg_data))
                            session.add(message)
                
                # Import tags
                for tag_data in import_data.get('tags', []):
                    existing = session.query(Tag).filter(Tag.id == tag_data['id']).first()
                    if not existing:
                        tag = Tag(**self._prepare_tag_data(tag_data))
                        session.add(tag)
            
            self.logger.info(f"Database imported from JSON: {json_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error importing database from JSON: {e}")
            raise DatabaseError(f"Failed to import database: {e}")
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform database health check.
        
        Returns:
            Dict: Health check results
        """
        health = {
            'status': 'unknown',
            'timestamp': datetime.now().isoformat(),
            'database': str(self.db_path),
            'connected': self._is_connected,
            'metrics': {},
            'issues': []
        }
        
        try:
            if not self._is_connected:
                health['status'] = 'disconnected'
                health['issues'].append('Database not connected')
                return health
            
            with self.session_scope() as session:
                # Check connection
                try:
                    session.execute(text("SELECT 1")).scalar()
                    health['metrics']['connection_test'] = 'passed'
                except Exception as e:
                    health['issues'].append(f"Connection test failed: {e}")
                    health['status'] = 'unhealthy'
                    return health
                
                # Get table counts
                tables = ['documents', 'chunks', 'conversations', 'messages', 'tags']
                for table in tables:
                    try:
                        count = session.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
                        health['metrics'][f'{table}_count'] = count
                    except Exception as e:
                        health['issues'].append(f"Failed to count {table}: {e}")
                
                # Check disk usage
                if self.db_path.exists():
                    size_bytes = self.db_path.stat().st_size
                    health['metrics']['size_bytes'] = size_bytes
                    health['metrics']['size_mb'] = size_bytes / (1024 * 1024)
                
                # Check for integrity issues
                try:
                    integrity_check = session.execute(text("PRAGMA integrity_check")).fetchone()
                    if integrity_check and integrity_check[0] == 'ok':
                        health['metrics']['integrity'] = 'ok'
                    else:
                        health['issues'].append(f"Integrity check failed: {integrity_check}")
                        health['status'] = 'unhealthy'
                except Exception as e:
                    health['issues'].append(f"Integrity check error: {e}")
                
                # Check WAL file
                wal_file = self.db_path.with_suffix('.db-wal')
                if wal_file.exists():
                    wal_size = wal_file.stat().st_size
                    health['metrics']['wal_size_mb'] = wal_size / (1024 * 1024)
                    
                    if wal_size > 100 * 1024 * 1024:  # 100MB
                        health['issues'].append(f"Large WAL file: {wal_size/(1024*1024):.1f}MB")
            
            # Determine overall status
            if health['issues']:
                health['status'] = 'degraded'
            else:
                health['status'] = 'healthy'
            
            return health
            
        except Exception as e:
            health['status'] = 'error'
            health['issues'].append(f"Health check error: {e}")
            return health
    
    def execute_raw_sql(self, sql: str, params: Optional[tuple] = None) -> List[Any]:
        """
        Execute raw SQL query (for administrative purposes).
        
        Args:
            sql: SQL query string
            params: Query parameters
            
        Returns:
            List: Query results
            
        Warning: Use with caution! This bypasses ORM safeguards.
        """
        try:
            with self.session_scope() as session:
                result = session.execute(text(sql), params or {})
                
                if sql.strip().upper().startswith('SELECT'):
                    return [dict(row._mapping) for row in result]
                else:
                    session.commit()
                    return [{'rows_affected': result.rowcount}]
                
        except Exception as e:
            self.logger.error(f"Error executing raw SQL: {e}")
            raise DatabaseError(f"Failed to execute SQL: {e}")
    
    def batch_operation(self, operation: str, data: List[Dict[str, Any]]) -> bool:
        """
        Perform batch insert/update operation.
        
        Args:
            operation: Operation type ('insert', 'update', 'delete')
            data: List of data dictionaries
            
        Returns:
            bool: True if operation successful
        """
        try:
            with self.session_scope() as session:
                if operation == 'insert':
                    for item in data:
                        if 'table' in item:
                            # Dynamic table insertion
                            table_class = self._get_table_class(item['table'])
                            if table_class:
                                obj = table_class(**item['data'])
                                session.add(obj)
                
                elif operation == 'update':
                    for item in data:
                        if 'table' in item and 'id' in item:
                            table_class = self._get_table_class(item['table'])
                            if table_class:
                                obj = session.query(table_class).get(item['id'])
                                if obj:
                                    for key, value in item['data'].items():
                                        if hasattr(obj, key):
                                            setattr(obj, key, value)
                
                elif operation == 'delete':
                    for item in data:
                        if 'table' in item and 'id' in item:
                            table_class = self._get_table_class(item['table'])
                            if table_class:
                                obj = session.query(table_class).get(item['id'])
                                if obj:
                                    session.delete(obj)
                
                else:
                    raise ValueError(f"Unknown operation: {operation}")
            
            self.logger.info(f"Batch {operation} completed for {len(data)} items")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in batch {operation}: {e}")
            raise DatabaseError(f"Failed to perform batch operation: {e}")
    
    def cleanup_old_data(self, days_old: int = 90) -> int:
        """
        Remove old data from database.
        
        Args:
            days_old: Remove data older than this many days
            
        Returns:
            int: Number of items deleted
        """
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            deleted_count = 0
            
            with self.session_scope() as session:
                # Delete old conversations
                old_conversations = session.query(Conversation).filter(
                    Conversation.updated_at < cutoff_date,
                    Conversation.is_archived == True
                ).all()
                
                for conv in old_conversations:
                    # Delete messages first
                    session.query(Message).filter(
                        Message.conversation_id == conv.id
                    ).delete()
                    
                    # Delete conversation
                    session.delete(conv)
                    deleted_count += 1
                
                # Delete documents with error status older than cutoff
                old_error_docs = session.query(Document).filter(
                    Document.upload_date < cutoff_date,
                    Document.processing_status == 'error'
                ).all()
                
                for doc in old_error_docs:
                    # Delete chunks first
                    session.query(Chunk).filter(
                        Chunk.document_id == doc.id
                    ).delete()
                    
                    # Delete document tags
                    session.query(DocumentTag).filter(
                        DocumentTag.document_id == doc.id
                    ).delete()
                    
                    # Delete document
                    session.delete(doc)
                    deleted_count += 1
            
            self.logger.info(f"Cleanup completed: deleted {deleted_count} items older than {days_old} days")
            return deleted_count
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
            raise DatabaseError(f"Failed to cleanup old data: {e}")
    
    # ========================================================================
    # ERROR HANDLING & LOGGING
    # ========================================================================
    
    def _handle_error(self, error: Exception, operation: str) -> None:
        """
        Handle database errors with consistent logging and exception raising.
        
        Args:
            error: Exception object
            operation: Operation that failed
            
        Raises:
            Appropriate DatabaseError subclass
        """
        error_msg = f"Database error in {operation}: {error}"
        self.logger.error(error_msg)
        
        # Map SQLAlchemy errors to our custom exceptions
        if isinstance(error, SQLAIntegrityError):
            raise IntegrityError(error_msg) from error
        elif isinstance(error, sqlite3.Error):
            if 'no such table' in str(error).lower():
                raise ConnectionError(error_msg) from error
            else:
                raise QueryError(error_msg) from error
        elif isinstance(error, SQLAlchemyError):
            raise DatabaseError(error_msg) from error
        else:
            raise DatabaseError(error_msg) from error
    
    def _retry_operation(self, func, max_retries: int = 3, **kwargs):
        """
        Retry operation with exponential backoff.
        
        Args:
            func: Function to retry
            max_retries: Maximum number of retries
            **kwargs: Function arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: If all retries fail
        """
        import time
        
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                return func(**kwargs)
            except (sqlite3.OperationalError, SQLAlchemyError) as e:
                last_exception = e
                
                # Check if error is retryable
                error_str = str(e).lower()
                if any(keyword in error_str for keyword in ['locked', 'busy', 'timeout']):
                    # Exponential backoff
                    wait_time = 0.1 * (2 ** attempt)
                    self.logger.warning(f"Retryable error (attempt {attempt + 1}/{max_retries}): {e}. Waiting {wait_time:.2f}s")
                    time.sleep(wait_time)
                else:
                    # Non-retryable error
                    break
            except Exception as e:
                last_exception = e
                break
        
        # All retries failed
        raise DatabaseError(f"Operation failed after {max_retries} attempts: {last_exception}") from last_exception
    
    # ========================================================================
    # CONFIGURATION & SETTINGS
    # ========================================================================
    
    def get_settings(self, key: str) -> Optional[str]:
        """
        Get setting from settings table.
        
        Args:
            key: Setting key
            
        Returns:
            Optional[str]: Setting value or None if not found
        """
        try:
            with self.session_scope() as session:
                setting = session.query(Setting).filter(Setting.key == key).first()
                return setting.value if setting else None
                
        except Exception as e:
            self.logger.error(f"Error getting setting '{key}': {e}")
            raise DatabaseError(f"Failed to get setting: {e}")
    
    def set_settings(self, key: str, value: str) -> bool:
        """
        Update or create setting.
        
        Args:
            key: Setting key
            value: Setting value
            
        Returns:
            bool: True if successful
        """
        try:
            with self.session_scope() as session:
                setting = session.query(Setting).filter(Setting.key == key).first()
                
                if setting:
                    setting.value = value
                    setting.updated_at = datetime.now()
                else:
                    setting = Setting(key=key, value=value)
                    session.add(setting)
                
                self.logger.info(f"Setting updated: {key} = {value}")
                return True
                
        except Exception as e:
            self.logger.error(f"Error setting '{key}': {e}")
            raise DatabaseError(f"Failed to set setting: {e}")
    
    def initialize_database(self) -> bool:
        """
        Initialize database with default data and indexes.
        
        Returns:
            bool: True if initialization successful
        """
        try:
            # Create tables (already done in connect())
            
            # Create indexes for performance
            with self.session_scope() as session:
                # Document indexes
                indexes = [
                    "CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(processing_status)",
                    "CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(file_type)",
                    "CREATE INDEX IF NOT EXISTS idx_documents_upload_date ON documents(upload_date)",
                    "CREATE INDEX IF NOT EXISTS idx_documents_file_name ON documents(file_name)",
                    
                    # Chunk indexes
                    "CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)",
                    "CREATE INDEX IF NOT EXISTS idx_chunks_vector_id ON chunks(vector_id)",
                    
                    # Message indexes
                    "CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id)",
                    "CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at)",
                    
                    # Tag indexes
                    "CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name)",
                    "CREATE INDEX IF NOT EXISTS idx_document_tags_document ON document_tags(document_id)",
                    "CREATE INDEX IF NOT EXISTS idx_document_tags_tag ON document_tags(tag_id)"
                ]
                
                for index_sql in indexes:
                    session.execute(text(index_sql))
                
                # Insert default settings
                default_settings = [
                    ('database_version', '1.0.0'),
                    ('last_maintenance', datetime.now().isoformat()),
                    ('auto_backup_enabled', 'true'),
                    ('backup_retention_days', '30'),
                    ('cleanup_enabled', 'true'),
                    ('cleanup_days_threshold', '90')
                ]
                
                for key, value in default_settings:
                    existing = session.query(Setting).filter(Setting.key == key).first()
                    if not existing:
                        setting = Setting(key=key, value=value)
                        session.add(setting)
            
            self.logger.info("Database initialized with default data and indexes")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing database: {e}")
            raise DatabaseError(f"Failed to initialize database: {e}")
    
    # ========================================================================
    # PRIVATE HELPER METHODS
    # ========================================================================
    
    def _generate_id(self, seed: str) -> str:
        """Generate unique ID from seed string."""
        return hashlib.md5(seed.encode()).hexdigest()[:16]
    
    def _get_or_create_tag(self, session: Session, tag_name: str) -> Tag:
        """Get existing tag or create new one."""
        tag = session.query(Tag).filter(Tag.name == tag_name).first()
        
        if not tag:
            tag_id = self._generate_id(f"tag_{tag_name.lower()}")
            tag = Tag(
                id=tag_id,
                name=tag_name,
                usage_count=0
            )
            session.add(tag)
        
        return tag
    
    def _apply_filters(self, query, model_class, filters: Dict[str, Any]):
        """Apply filters to query."""
        for field, value in filters.items():
            if hasattr(model_class, field):
                column = getattr(model_class, field)
                
                if isinstance(value, dict):
                    # Complex filter (gt, lt, like, etc.)
                    for op, op_value in value.items():
                        if op == 'gt':
                            query = query.filter(column > op_value)
                        elif op == 'gte':
                            query = query.filter(column >= op_value)
                        elif op == 'lt':
                            query = query.filter(column < op_value)
                        elif op == 'lte':
                            query = query.filter(column <= op_value)
                        elif op == 'like':
                            query = query.filter(column.ilike(f"%{op_value}%"))
                        elif op == 'in':
                            query = query.filter(column.in_(op_value))
                        elif op == 'not':
                            query = query.filter(column != op_value)
                elif value is None:
                    query = query.filter(column.is_(None))
                else:
                    query = query.filter(column == value)
        
        return query
    
    def _document_to_dict(self, document: Document, include_chunks: bool = False) -> Dict[str, Any]:
        """Convert Document object to dictionary."""
        doc_dict = {
            'id': document.id,
            'file_path': document.file_path,
            'file_name': document.file_name,
            'file_type': document.file_type,
            'file_size': document.file_size,
            'upload_date': document.upload_date.isoformat() if document.upload_date else None,
            'processing_status': document.processing_status,
            'processing_error': document.processing_error,
            'metadata': json.loads(document.metadata_json) if document.metadata_json else {},
            'vector_ids': json.loads(document.vector_ids_json) if document.vector_ids_json else [],
            'chunk_count': document.chunk_count,
            'word_count': document.word_count,
            'language': document.language,
            'tags': json.loads(document.tags_json) if document.tags_json else [],
            'summary': document.summary,
            'is_indexed': document.is_indexed,
            'indexed_at': document.indexed_at.isoformat() if document.indexed_at else None,
            'last_accessed': document.last_accessed.isoformat() if document.last_accessed else None,
            'access_count': document.access_count,
            'created_at': document.created_at.isoformat() if document.created_at else None,
            'updated_at': document.updated_at.isoformat() if document.updated_at else None
        }
        
        if include_chunks and hasattr(document, 'chunks'):
            doc_dict['chunks'] = [self._chunk_to_dict(chunk) for chunk in document.chunks]
        
        # Remove None values
        doc_dict = {k: v for k, v in doc_dict.items() if v is not None}
        
        return doc_dict
    
    def _chunk_to_dict(self, chunk: Chunk) -> Dict[str, Any]:
        """Convert Chunk object to dictionary."""
        chunk_dict = {
            'id': chunk.id,
            'document_id': chunk.document_id,
            'chunk_index': chunk.chunk_index,
            'text_content': chunk.text_content,
            'cleaned_text': chunk.cleaned_text,
            'token_count': chunk.token_count,
            'embedding_model': chunk.embedding_model,
            'vector_id': chunk.vector_id,
            'metadata': json.loads(chunk.metadata_json) if chunk.metadata_json else {},
            'created_at': chunk.created_at.isoformat() if chunk.created_at else None
        }
        
        return {k: v for k, v in chunk_dict.items() if v is not None}
    
    def _conversation_to_dict(self, conversation: Conversation) -> Dict[str, Any]:
        """Convert Conversation object to dictionary."""
        conv_dict = {
            'id': conversation.id,
            'title': conversation.title,
            'created_at': conversation.created_at.isoformat() if conversation.created_at else None,
            'updated_at': conversation.updated_at.isoformat() if conversation.updated_at else None,
            'message_count': conversation.message_count,
            'total_tokens': conversation.total_tokens,
            'tags': json.loads(conversation.tags_json) if conversation.tags_json else [],
            'is_archived': conversation.is_archived,
            'export_path': conversation.export_path,
            'metadata': json.loads(conversation.metadata_json) if conversation.metadata_json else {}
        }
        
        return {k: v for k, v in conv_dict.items() if v is not None}
    
    def _message_to_dict(self, message: Message) -> Dict[str, Any]:
        """Convert Message object to dictionary."""
        msg_dict = {
            'id': message.id,
            'conversation_id': message.conversation_id,
            'role': message.role,
            'content': message.content,
            'tokens': message.tokens,
            'model_used': message.model_used,
            'sources': json.loads(message.sources_json) if message.sources_json else [],
            'processing_time_ms': message.processing_time_ms,
            'created_at': message.created_at.isoformat() if message.created_at else None,
            'metadata': json.loads(message.metadata_json) if message.metadata_json else {}
        }
        
        return {k: v for k, v in msg_dict.items() if v is not None}
    
    def _tag_to_dict(self, tag: Tag) -> Dict[str, Any]:
        """Convert Tag object to dictionary."""
        tag_dict = {
            'id': tag.id,
            'name': tag.name,
            'color': tag.color,
            'description': tag.description,
            'created_at': tag.created_at.isoformat() if tag.created_at else None,
            'usage_count': tag.usage_count
        }
        
        return {k: v for k, v in tag_dict.items() if v is not None}
    
    def _prepare_document_data(self, data: Dict) -> Dict:
        """Prepare document data for database insertion."""
        prepared = {}
        
        for key, value in data.items():
            if key in ['metadata', 'vector_ids', 'tags']:
                prepared[f'{key}_json'] = json.dumps(value)
            elif key in ['upload_date', 'indexed_at', 'last_accessed', 'created_at', 'updated_at']:
                if value:
                    prepared[key] = datetime.fromisoformat(value.replace('Z', '+00:00'))
            else:
                prepared[key] = value
        
        return prepared
    
    def _prepare_conversation_data(self, data: Dict) -> Dict:
        """Prepare conversation data for database insertion."""
        prepared = {}
        
        for key, value in data.items():
            if key in ['tags', 'metadata']:
                prepared[f'{key}_json'] = json.dumps(value)
            elif key in ['created_at', 'updated_at']:
                if value:
                    prepared[key] = datetime.fromisoformat(value.replace('Z', '+00:00'))
            else:
                prepared[key] = value
        
        return prepared
    
    def _prepare_message_data(self, data: Dict) -> Dict:
        """Prepare message data for database insertion."""
        prepared = {}
        
        for key, value in data.items():
            if key in ['sources', 'metadata']:
                prepared[f'{key}_json'] = json.dumps(value)
            elif key == 'created_at':
                if value:
                    prepared[key] = datetime.fromisoformat(value.replace('Z', '+00:00'))
            else:
                prepared[key] = value
        
        return prepared
    
    def _prepare_tag_data(self, data: Dict) -> Dict:
        """Prepare tag data for database insertion."""
        prepared = {}
        
        for key, value in data.items():
            if key == 'created_at':
                if value:
                    prepared[key] = datetime.fromisoformat(value.replace('Z', '+00:00'))
            else:
                prepared[key] = value
        
        return prepared
    
    def _get_table_class(self, table_name: str):
        """Get SQLAlchemy model class by table name."""
        table_map = {
            'documents': Document,
            'chunks': Chunk,
            'conversations': Conversation,
            'messages': Message,
            'tags': Tag,
            'document_tags': DocumentTag,
            'settings': Setting
        }
        
        return table_map.get(table_name.lower())
    
    def _initialize_database(self):
        """Initialize database on first connect."""
        try:
            with self.session_scope() as session:
                # Check if database is empty
                table_count = session.execute(text(
                    "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
                )).scalar()
                
                if table_count == 0:
                    self.logger.info("Initializing empty database")
                    self.initialize_database()
                else:
                    # Check if settings table has default settings
                    setting_count = session.query(func.count(Setting.key)).scalar()
                    if setting_count == 0:
                        self.logger.info("Adding default settings to existing database")
                        self.initialize_database()
        
        except Exception as e:
            self.logger.warning(f"Database initialization check failed: {e}")

# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def create_sqlite_client(config: Optional[Config] = None) -> SQLiteClient:
    """
    Factory function to create and connect SQLiteClient.
    
    Args:
        config: Configuration object
        
    Returns:
        SQLiteClient: Connected database client
    """
    client = SQLiteClient(config=config)
    client.connect()
    return client

# ============================================================================
# TESTING SUPPORT
# ============================================================================

class TestSQLiteClient(SQLiteClient):
    """
    SQLiteClient subclass for testing with in-memory database.
    """
    
    def __init__(self, config: Optional[Config] = None):
        # Use in-memory database for testing
        super().__init__(":memory:", config)
        
        # Override configuration for testing
        self.pool_size = 1
        self.max_overflow = 0
        self.config.set('storage.database.debug', True)
    
    def cleanup_test_data(self):
        """Clean up all test data."""
        try:
            with self.session_scope() as session:
                # Delete all data (preserve schema)
                session.query(Message).delete()
                session.query(Conversation).delete()
                session.query(Chunk).delete()
                session.query(DocumentTag).delete()
                session.query(Document).delete()
                session.query(Tag).delete()
                session.query(Setting).filter(
                    Setting.key.notin_(['database_version', 'last_maintenance'])
                ).delete()
                
                self.logger.info("Test data cleaned up")
                
        except Exception as e:
            self.logger.error(f"Error cleaning test data: {e}")
            raise
        
# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    # Simple test to verify the module works
    print("Testing SQLiteClient module...")
    
    try:
        # Create test client with in-memory database
        client = TestSQLiteClient()
        client.connect()
        
        # Test health check
        health = client.health_check()
        print(f"Health check: {health['status']}")
        
        # Test adding a document
        doc_id = client.add_document(
            file_path="test.pdf",
            file_name="test.pdf",
            file_type="pdf",
            file_size=1024,
            metadata={"author": "Test Author"}
        )
        print(f"Added document: {doc_id}")
        
        # Test retrieving document
        doc = client.get_document(doc_id)
        print(f"Retrieved document: {doc['file_name']}")
        
        # Test statistics
        stats = client.get_document_statistics()
        print(f"Document statistics: {stats}")
        
        # Clean up
        client.cleanup_test_data()
        client.close()
        
        print("All tests passed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# ALIAS FOR BACKWARD COMPATIBILITY
# ============================================================================

DatabaseClient = SQLiteClient

# ============================================================================
# EXPORTS - INI HARUS DI LUAR if __name__ == "__main__":
# ============================================================================

__all__ = [
    'SQLiteClient', 
    'DatabaseClient',  
    'TestSQLiteClient',
    'create_sqlite_client',
    'DatabaseError',
    'ConnectionError', 
    'QueryError',
    'IntegrityError',
    'NotFoundError',
    'ValidationError'
]

# ============================================================================
# MAIN (for testing)
# ============================================================================

if __name__ == "__main__":
    # Simple test to verify the module works
    print("Testing SQLiteClient module...")
    
    try:
        # Create test client with in-memory database
        client = TestSQLiteClient()
        client.connect()
        
        # Test health check
        health = client.health_check()
        print(f"Health check: {health['status']}")
        
        # Test adding a document
        doc_id = client.add_document(
            file_path="test.pdf",
            file_name="test.pdf",
            file_type="pdf",
            file_size=1024,
            metadata={"author": "Test Author"}
        )
        print(f"Added document: {doc_id}")
        
        # Test retrieving document
        doc = client.get_document(doc_id)
        print(f"Retrieved document: {doc['file_name']}")
        
        # Test statistics
        stats = client.get_document_statistics()
        print(f"Document statistics: {stats}")
        
        # Clean up
        client.cleanup_test_data()
        client.close()
        
        print("All tests passed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()