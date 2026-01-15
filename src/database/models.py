"""
SQLAlchemy ORM Models for DocuBot Database
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from sqlalchemy import create_engine, Column, Integer, String, Text, Boolean, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, Session
from sqlalchemy import JSON
import uuid

from ..core.constants import DATABASE_DIR, DATABASE_NAME

Base = declarative_base()


class Document(Base):
    """Document model representing uploaded files"""
    
    __tablename__ = 'documents'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    file_path = Column(Text, nullable=False)
    file_name = Column(String(255), nullable=False)
    file_type = Column(String(10), nullable=False)
    file_size = Column(Integer)
    
    upload_date = Column(DateTime, default=datetime.utcnow)
    processing_status = Column(String(20), default='pending')
    processing_error = Column(Text)
    
    metadata_json = Column(JSON, default=dict)
    vector_ids_json = Column(JSON, default=list)
    
    chunk_count = Column(Integer, default=0)
    word_count = Column(Integer, default=0)
    language = Column(String(10))
    
    tags_json = Column(JSON, default=list)
    summary = Column(Text)
    
    is_indexed = Column(Boolean, default=False)
    indexed_at = Column(DateTime)
    
    last_accessed = Column(DateTime, default=datetime.utcnow)
    access_count = Column(Integer, default=0)
    
    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'file_path': self.file_path,
            'file_name': self.file_name,
            'file_type': self.file_type,
            'file_size': self.file_size,
            'upload_date': self.upload_date.isoformat() if self.upload_date else None,
            'processing_status': self.processing_status,
            'processing_error': self.processing_error,
            'metadata': self.metadata_json,
            'vector_ids': self.vector_ids_json,
            'chunk_count': self.chunk_count,
            'word_count': self.word_count,
            'language': self.language,
            'tags': self.tags_json,
            'summary': self.summary,
            'is_indexed': self.is_indexed,
            'indexed_at': self.indexed_at.isoformat() if self.indexed_at else None,
            'last_accessed': self.last_accessed.isoformat() if self.last_accessed else None,
            'access_count': self.access_count
        }
    
    def update_last_accessed(self):
        """Update last accessed timestamp"""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1


class Chunk(Base):
    """Text chunk model"""
    
    __tablename__ = 'chunks'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    document_id = Column(String(36), ForeignKey('documents.id', ondelete='CASCADE'), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    
    text_content = Column(Text, nullable=False)
    cleaned_text = Column(Text, nullable=False)
    token_count = Column(Integer)
    
    embedding_model = Column(String(50))
    vector_id = Column(String(255), nullable=False)
    
    metadata_json = Column(JSON, default=dict)
    created_at = Column(DateTime, default=datetime.utcnow)
    
    document = relationship("Document", back_populates="chunks")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'document_id': self.document_id,
            'chunk_index': self.chunk_index,
            'text_content': self.text_content,
            'cleaned_text': self.cleaned_text,
            'token_count': self.token_count,
            'embedding_model': self.embedding_model,
            'vector_id': self.vector_id,
            'metadata': self.metadata_json,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class Conversation(Base):
    """Conversation model"""
    
    __tablename__ = 'conversations'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(255))
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    message_count = Column(Integer, default=0)
    total_tokens = Column(Integer, default=0)
    
    tags_json = Column(JSON, default=list)
    is_archived = Column(Boolean, default=False)
    export_path = Column(Text)
    
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'title': self.title,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'message_count': self.message_count,
            'total_tokens': self.total_tokens,
            'tags': self.tags_json,
            'is_archived': self.is_archived,
            'export_path': self.export_path
        }
    
    def update_message_count(self):
        """Update message count"""
        self.message_count = len(self.messages)
        self.total_tokens = sum(msg.tokens or 0 for msg in self.messages)


class Message(Base):
    """Chat message model"""
    
    __tablename__ = 'messages'
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String(36), ForeignKey('conversations.id', ondelete='CASCADE'), nullable=False)
    
    role = Column(String(20), nullable=False)
    content = Column(Text, nullable=False)
    tokens = Column(Integer)
    
    model_used = Column(String(50))
    sources_json = Column(JSON, default=list)
    processing_time_ms = Column(Integer)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    conversation = relationship("Conversation", back_populates="messages")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'conversation_id': self.conversation_id,
            'role': self.role,
            'content': self.content,
            'tokens': self.tokens,
            'model_used': self.model_used,
            'sources': self.sources_json,
            'processing_time_ms': self.processing_time_ms,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class Setting(Base):
    """Application settings model"""
    
    __tablename__ = 'settings'
    
    key = Column(String(100), primary_key=True)
    value = Column(Text)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            'key': self.key,
            'value': self.value,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }


_engine = None
_SessionLocal = None

def get_engine():
    """Get SQLAlchemy engine"""
    global _engine
    
    if _engine is None:
        db_url = f"sqlite:///{DATABASE_DIR / DATABASE_NAME}"
        _engine = create_engine(db_url, connect_args={"check_same_thread": False})
    
    return _engine

def get_session_local():
    """Get session factory"""
    global _SessionLocal
    
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    
    return _SessionLocal

def get_session() -> Session:
    """Get database session"""
    SessionLocal = get_session_local()
    return SessionLocal()

def create_tables():
    """Create all database tables"""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)

def init_database():
    """Initialize database with tables"""
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    create_tables()
    
    session = get_session()
    try:
        existing = session.query(Setting).count()
        
        if existing == 0:
            default_settings = [
                Setting(key="app.version", value="1.0.0"),
                Setting(key="app.default_chunk_size", value="500"),
                Setting(key="app.default_chunk_overlap", value="50"),
                Setting(key="app.default_llm_model", value="llama2:7b"),
                Setting(key="app.default_embedding_model", value="all-MiniLM-L6-v2"),
            ]
            
            session.add_all(default_settings)
            session.commit()
            print("Database initialized with default settings")
        else:
            print(f"Database already initialized with {existing} settings")
            
    except Exception as e:
        session.rollback()
        print(f"Error initializing database: {e}")
    finally:
        session.close()


if __name__ == "__main__":
    init_database()
    
    session = get_session()
    
    try:
        test_doc = Document(
            file_path="/test/example.pdf",
            file_name="example.pdf",
            file_type=".pdf",
            file_size=1024,
            processing_status="completed",
            chunk_count=5,
            word_count=1000
        )
        
        session.add(test_doc)
        session.commit()
        
        print(f"Created test document: {test_doc.id}")
        
        doc = session.query(Document).filter_by(file_name="example.pdf").first()
        print(f"Retrieved document: {doc.to_dict()}")
        
        chunk = Chunk(
            document_id=doc.id,
            chunk_index=0,
            text_content="This is a test chunk.",
            cleaned_text="This is a test chunk.",
            token_count=5,
            vector_id="test_vector_123"
        )
        
        session.add(chunk)
        session.commit()
        
        print(f"Created test chunk: {chunk.id}")
        
        doc_chunks = doc.chunks
        print(f"Document has {len(doc_chunks)} chunks")
        
    except Exception as e:
        print(f"Error: {e}")
        session.rollback()
    finally:
        session.close()
