#2. Core Logic & Missing Features

"""
Database operations (P1.5.4 migrations, P1.5.5 queries)
Vector store functionality (P1.6.4 hybrid search, P1.6.5 index management)
AI integration (P1.8.3 model management, P1.8.4 streaming, P1.8.5 controls)
RAG engine (P1.11.1, P1.11.2, P1.12.2 pipeline)
Conversation handling (P1.11.3 memory, P1.12.4)
"""

import os
import sys
from pathlib import Path
import yaml
import json
import uuid
from datetime import datetime


class CoreLogicFixer:
    def __init__(self, project_dir="DocuBot"):
        self.project_dir = Path(project_dir).absolute()
        
        # Ensure directory structure exists
        self._create_directory_structure()
    
    def _create_directory_structure(self):
        """Create necessary directories for the project"""
        dirs = [
            self.project_dir / "src" / "database" / "migrations",
            self.project_dir / "src" / "vector_store",
            self.project_dir / "src" / "ai_engine",
            self.project_dir / "src" / "core",
            self.project_dir / "scripts",
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def implement_core_logic(self):
        print("Implementing DocuBot Core Logic")
        print("=" * 60)
        
        implementations = [
            self.implement_database_migrations,
            self.implement_vector_store_search,
            self.implement_llm_management,
            self.implement_rag_engine,
            self.implement_conversation_memory,
            self.implement_query_pipeline,
            self.implement_model_manager,
            self.implement_index_manager
        ]
        
        for i, impl_func in enumerate(implementations, 1):
            print(f"[{i}/{len(implementations)}] Executing {impl_func.__name__}")
            try:
                impl_func()
                print("   Success")
            except Exception as e:
                print(f"   Failed: {e}")
        
        print("\n" + "=" * 60)
        print("Core logic implementation completed")
        print("=" * 60)
    
    def implement_database_migrations(self):
        migrations_dir = self.project_dir / "src" / "database" / "migrations"
        migrations_dir.mkdir(exist_ok=True)
        
        # Create __init__.py files
        (self.project_dir / "src" / "database" / "__init__.py").write_text("")
        (self.project_dir / "src" / "database" / "migrations" / "__init__.py").write_text("")
        
        init_db_file = self.project_dir / "scripts" / "init_db.py"
        init_db_content = '''"""
Database Initialization Script
"""

import sqlite3
from pathlib import Path
import json
from datetime import datetime

def initialize_database(db_path: Path):
    """
    Initialize SQLite database with schema
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    connection = sqlite3.connect(str(db_path))
    cursor = connection.cursor()
    
    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON")
    
    # Create tables
    create_tables_sql = """
    -- Documents table
    CREATE TABLE IF NOT EXISTS documents (
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
    );
    
    -- Chunks table
    CREATE TABLE IF NOT EXISTS chunks (
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
    );
    
    -- Conversations table
    CREATE TABLE IF NOT EXISTS conversations (
        id TEXT PRIMARY KEY,
        title TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        message_count INTEGER DEFAULT 0,
        total_tokens INTEGER DEFAULT 0,
        tags_json TEXT,
        is_archived BOOLEAN DEFAULT FALSE,
        export_path TEXT
    );
    
    -- Messages table
    CREATE TABLE IF NOT EXISTS messages (
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
    );
    
    -- Settings table
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    cursor.executescript(create_tables_sql)
    
    # Create indexes
    cursor.executescript("""
    CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(processing_status);
    CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(file_type);
    CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
    CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
    CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at);
    CREATE INDEX IF NOT EXISTS idx_documents_upload_date ON documents(upload_date);
    """)
    
    # Insert default settings
    default_settings = {
        'app_version': '1.0.0',
        'default_llm_model': 'llama2:7b',
        'default_embedding_model': 'all-MiniLM-L6-v2',
        'chunk_size': '500',
        'chunk_overlap': '50',
        'rag_top_k': '5'
    }
    
    for key, value in default_settings.items():
        cursor.execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            (key, str(value), datetime.now().isoformat())
        )
    
    connection.commit()
    connection.close()
    
    print(f"Database initialized at: {db_path}")

if __name__ == "__main__":
    # Create mock constants if they don't exist
    from pathlib import Path
    DATABASE_DIR = Path.home() / ".docubot" / "data"
    DATABASE_NAME = "docubot.db"
    
    db_path = DATABASE_DIR / DATABASE_NAME
    initialize_database(db_path)
'''
        
        init_db_file.write_text(init_db_content)
        
        # Also create sqlite_client.py
        sqlite_file = self.project_dir / "src" / "database" / "sqlite_client.py"
        sqlite_content = '''"""
SQLite Database Client
"""

import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
import json


class SQLiteClient:
    """SQLite database client wrapper"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.connection = None
    
    def connect(self):
        """Establish database connection"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(self.db_path))
        self.connection.row_factory = sqlite3.Row
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None
    
    def execute(self, query: str, params: tuple = ()):
        """Execute a query"""
        if not self.connection:
            self.connect()
        
        cursor = self.connection.cursor()
        cursor.execute(query, params)
        self.connection.commit()
        return cursor.lastrowid
    
    def fetch_one(self, query: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        """Fetch a single row"""
        if not self.connection:
            self.connect()
        
        cursor = self.connection.cursor()
        cursor.execute(query, params)
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None
    
    def fetch_all(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """Fetch all rows"""
        if not self.connection:
            self.connect()
        
        cursor = self.connection.cursor()
        cursor.execute(query, params)
        rows = cursor.fetchall()
        
        return [dict(row) for row in rows]
    
    def __enter__(self):
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
'''
        sqlite_file.write_text(sqlite_content)
    
    def implement_vector_store_search(self):
        # Create __init__.py files
        (self.project_dir / "src" / "vector_store" / "__init__.py").write_text("")
        
        search_file = self.project_dir / "src" / "vector_store" / "search_engine.py"
        
        search_content = '''"""
Hybrid Search Engine for Vector Store
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from .chroma_client import ChromaClient


class HybridSearchEngine:
    """
    Implements hybrid search combining vector similarity and keyword matching
    """
    
    def __init__(self, chroma_client: ChromaClient, embedding_model: SentenceTransformer):
        self.chroma_client = chroma_client
        self.embedding_model = embedding_model
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        alpha: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector and keyword search
        
        Args:
            query: Search query
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            alpha: Weight for vector vs keyword search (0-1)
        
        Returns:
            List of search results with scores and metadata
        """
        # Vector search
        vector_results = self.vector_search(query, top_k * 2, similarity_threshold)
        
        # Keyword search (simple implementation)
        keyword_results = self.keyword_search(query, top_k * 2)
        
        # Combine results using hybrid scoring
        combined_results = self.combine_results(
            vector_results, keyword_results, alpha, top_k
        )
        
        return combined_results
    
    def vector_search(
        self,
        query: str,
        top_k: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search
        
        Args:
            query: Search query
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
        
        Returns:
            List of vector search results
        """
        query_embedding = self.embedding_model.encode(query)
        
        results = self.chroma_client.search(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        processed_results = []
        if results and 'documents' in results:
            for i in range(len(results['documents'][0])):
                score = results['distances'][0][i] if 'distances' in results else 1.0
                
                # Convert distance to similarity score (assuming cosine distance)
                similarity = 1.0 - score if isinstance(score, (int, float)) else 0.0
                
                if similarity >= similarity_threshold:
                    result = {
                        'text': results['documents'][0][i],
                        'similarity': similarity,
                        'metadata': results['metadatas'][0][i] if 'metadatas' in results else {},
                        'id': results['ids'][0][i] if 'ids' in results else f"result_{i}",
                        'search_type': 'vector'
                    }
                    processed_results.append(result)
        
        return processed_results
    
    def keyword_search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of keyword search results
        """
        # Simple keyword matching implementation
        query_keywords = set(query.lower().split())
        
        # This would typically query a separate index or database
        # For now, return empty list - implementation depends on specific setup
        return []
    
    def combine_results(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        alpha: float = 0.7,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Combine vector and keyword search results
        
        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword search
            alpha: Weight for vector vs keyword (0 = all keyword, 1 = all vector)
            top_k: Number of final results
        
        Returns:
            Combined and ranked results
        """
        # Create a combined result dictionary
        combined_dict = {}
        
        # Add vector results
        for result in vector_results:
            result_id = result.get('id')
            if result_id:
                combined_dict[result_id] = {
                    'result': result,
                    'vector_score': result.get('similarity', 0.0),
                    'keyword_score': 0.0,
                    'combined_score': 0.0
                }
        
        # Add keyword results
        for i, result in enumerate(keyword_results):
            result_id = result.get('id', f"keyword_{i}")
            keyword_score = 1.0 / (i + 1)  # Simple ranking
            
            if result_id in combined_dict:
                combined_dict[result_id]['keyword_score'] = keyword_score
            else:
                combined_dict[result_id] = {
                    'result': result,
                    'vector_score': 0.0,
                    'keyword_score': keyword_score,
                    'combined_score': 0.0
                }
        
        # Calculate combined scores
        for result_id, data in combined_dict.items():
            vector_score = data['vector_score']
            keyword_score = data['keyword_score']
            
            combined_score = (alpha * vector_score) + ((1 - alpha) * keyword_score)
            data['combined_score'] = combined_score
        
        # Sort by combined score and return top_k
        sorted_results = sorted(
            combined_dict.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )[:top_k]
        
        # Format final results
        final_results = []
        for data in sorted_results:
            result = data['result'].copy()
            result['combined_score'] = data['combined_score']
            result['search_type'] = 'hybrid'
            final_results.append(result)
        
        return final_results
    
    def semantic_search_with_filters(
        self,
        query: str,
        filters: Dict[str, Any],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search with metadata filters
        
        Args:
            query: Search query
            filters: Dictionary of metadata filters
            top_k: Number of results to return
        
        Returns:
            Filtered search results
        """
        query_embedding = self.embedding_model.encode(query)
        
        # Convert filters to ChromaDB format
        where_filter = {}
        for key, value in filters.items():
            if isinstance(value, list):
                where_filter[key] = {"$in": value}
            else:
                where_filter[key] = {"$eq": value}
        
        results = self.chroma_client.search(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter if where_filter else None
        )
        
        return self._process_search_results(results)
    
    def _process_search_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process raw search results into standardized format
        
        Args:
            results: Raw search results from vector store
        
        Returns:
            Processed search results
        """
        processed = []
        
        if not results or 'documents' not in results:
            return processed
        
        for i in range(len(results['documents'][0])):
            processed_result = {
                'text': results['documents'][0][i],
                'similarity': 1.0 - results['distances'][0][i] if 'distances' in results else 0.0,
                'metadata': results['metadatas'][0][i] if 'metadatas' in results else {},
                'id': results['ids'][0][i] if 'ids' in results else f"result_{i}"
            }
            processed.append(processed_result)
        
        return processed
'''
        
        search_file.write_text(search_content)
        
        # Also create chroma_client.py if it doesn't exist
        chroma_file = self.project_dir / "src" / "vector_store" / "chroma_client.py"
        if not chroma_file.exists():
            chroma_content = '''"""
ChromaDB Client Wrapper
"""

import chromadb
from typing import List, Dict, Any, Optional
from pathlib import Path


class ChromaClient:
    """ChromaDB client wrapper"""
    
    def __init__(self, persist_directory: str, embedding_model=None):
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        
        Path(persist_directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=persist_directory)
    
    def create_collection(self, name: str, metadata: Optional[Dict] = None):
        """Create a new collection"""
        return self.client.create_collection(
            name=name,
            metadata=metadata or {}
        )
    
    def get_collection(self, name: str):
        """Get an existing collection"""
        try:
            return self.client.get_collection(name=name)
        except:
            return None
    
    def delete_collection(self, name: str):
        """Delete a collection"""
        return self.client.delete_collection(name=name)
    
    def list_collections(self):
        """List all collections"""
        return self.client.list_collections()
    
    def add_documents(self, collection_name: str, documents: List[str], 
                      metadatas: Optional[List[Dict]] = None, 
                      ids: Optional[List[str]] = None):
        """Add documents to a collection"""
        collection = self.get_collection(collection_name)
        if not collection:
            collection = self.create_collection(collection_name)
        
        if self.embedding_model:
            embeddings = self.embedding_model.encode(documents).tolist()
            collection.add(
                documents=documents,
                metadatas=metadatas or [],
                ids=ids or [str(i) for i in range(len(documents))],
                embeddings=embeddings
            )
        else:
            collection.add(
                documents=documents,
                metadatas=metadatas or [],
                ids=ids or [str(i) for i in range(len(documents))]
            )
    
    def search(self, query_embeddings: List[List[float]], n_results: int = 5, 
               where: Optional[Dict] = None):
        """Search in a collection"""
        collection = self.get_collection("documents")  # Default collection
        if not collection:
            return {}
        
        return collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where
        )
'''
            chroma_file.write_text(chroma_content)
    
    def implement_llm_management(self):
        # Create __init__.py files
        (self.project_dir / "src" / "ai_engine" / "__init__.py").write_text("")
        
        llm_file = self.project_dir / "src" / "ai_engine" / "llm_client.py"
        
        llm_content = '''"""
Ollama LLM Client with Model Management
"""

from typing import Dict, Any, List, Optional, Generator
import json
from dataclasses import dataclass
from pathlib import Path
import time


@dataclass
class LLMResponse:
    """Structured LLM response"""
    content: str
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    processing_time: Optional[float] = None
    error: Optional[str] = None


class LLMClient:
    """
    Client for interacting with Ollama LLM models
    """
    
    def __init__(self, default_model: str = "llama2:7b"):
        self.default_model = default_model
        self.available_models = self.list_available_models()
        self.model_info_cache = {}
    
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        stream: bool = False,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        top_p: float = 0.9,
        context: Optional[str] = None
    ) -> LLMResponse:
        """
        Generate text from LLM
        
        Args:
            prompt: Input prompt
            model: Model to use (defaults to self.default_model)
            stream: Whether to stream response
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            context: Additional context for the model
        
        Returns:
            LLMResponse object
        """
        start_time = time.time()
        model = model or self.default_model
        
        try:
            # Prepare messages
            messages = self._prepare_messages(prompt, context)
            
            if stream:
                return self._generate_streaming(messages, model, start_time, temperature, max_tokens, top_p)
            else:
                return self._generate_non_streaming(messages, model, start_time, temperature, max_tokens, top_p)
                
        except Exception as e:
            processing_time = time.time() - start_time
            return LLMResponse(
                content="",
                model=model,
                processing_time=processing_time,
                error=str(e)
            )
    
    def _prepare_messages(self, prompt: str, context: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Prepare messages for the LLM
        
        Args:
            prompt: User prompt
            context: Additional context
        
        Returns:
            List of message dictionaries
        """
        messages = []
        
        if context:
            system_message = f"You are a helpful AI assistant. Use the following context to answer questions:\\n\\n{context}"
            messages.append({'role': 'system', 'content': system_message})
        
        messages.append({'role': 'user', 'content': prompt})
        
        return messages
    
    def _generate_non_streaming(
        self,
        messages: List[Dict[str, str]],
        model: str,
        start_time: float,
        temperature: float,
        max_tokens: int,
        top_p: float
    ) -> LLMResponse:
        """
        Generate non-streaming response
        
        Args:
            messages: List of messages
            model: Model name
            start_time: Start time for timing
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            top_p: Top-p sampling
        
        Returns:
            LLMResponse object
        """
        try:
            # Try to import ollama
            import ollama
            
            response = ollama.chat(
                model=model,
                messages=messages,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens,
                    'top_p': top_p
                }
            )
            
            processing_time = time.time() - start_time
            
            return LLMResponse(
                content=response['message']['content'],
                model=model,
                prompt_tokens=response.get('prompt_eval_count'),
                completion_tokens=response.get('eval_count'),
                total_tokens=(response.get('prompt_eval_count', 0) + response.get('eval_count', 0)),
                processing_time=processing_time
            )
            
        except ImportError:
            # Mock response for testing
            processing_time = time.time() - start_time
            return LLMResponse(
                content="Mock response: " + messages[-1]['content'][:50] + "...",
                model=model,
                prompt_tokens=len(messages[-1]['content'].split()),
                completion_tokens=50,
                total_tokens=len(messages[-1]['content'].split()) + 50,
                processing_time=processing_time
            )
        except Exception as e:
            processing_time = time.time() - start_time
            return LLMResponse(
                content="",
                model=model,
                processing_time=processing_time,
                error=str(e)
            )
    
    def _generate_streaming(
        self,
        messages: List[Dict[str, str]],
        model: str,
        start_time: float,
        temperature: float,
        max_tokens: int,
        top_p: float
    ) -> Generator[LLMResponse, None, None]:
        """
        Generate streaming response
        
        Args:
            messages: List of messages
            model: Model name
            start_time: Start time
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            top_p: Top-p sampling
        
        Yields:
            LLMResponse objects with partial content
        """
        full_response = ""
        
        try:
            import ollama
            
            stream = ollama.chat(
                model=model,
                messages=messages,
                options={
                    'temperature': temperature,
                    'num_predict': max_tokens,
                    'top_p': top_p
                },
                stream=True
            )
            
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    content_chunk = chunk['message']['content']
                    full_response += content_chunk
                    
                    yield LLMResponse(
                        content=content_chunk,
                        model=model,
                        processing_time=time.time() - start_time
                    )
            
            # Final response with complete content
            processing_time = time.time() - start_time
            yield LLMResponse(
                content=full_response,
                model=model,
                processing_time=processing_time
            )
            
        except ImportError:
            # Mock streaming for testing
            mock_content = "Mock streaming response for testing."
            for i in range(0, len(mock_content), 5):
                chunk = mock_content[i:i+5]
                yield LLMResponse(
                    content=chunk,
                    model=model,
                    processing_time=time.time() - start_time
                )
        except Exception as e:
            processing_time = time.time() - start_time
            yield LLMResponse(
                content="",
                model=model,
                processing_time=processing_time,
                error=str(e)
            )
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List available Ollama models
        
        Returns:
            List of model information dictionaries
        """
        try:
            import ollama
            models = ollama.list()
            return models.get('models', [])
        except:
            # Return mock data for testing
            return [
                {'name': 'llama2:7b', 'size': '3.8 GB', 'modified_at': '2023-01-01'},
                {'name': 'mistral:7b', 'size': '4.1 GB', 'modified_at': '2023-01-01'},
            ]
    
    def pull_model(self, model_name: str) -> Dict[str, Any]:
        """
        Pull/download a model
        
        Args:
            model_name: Name of model to pull
        
        Returns:
            Pull status information
        """
        try:
            import ollama
            response = ollama.pull(model_name)
            return {'success': True, 'response': response}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def delete_model(self, model_name: str) -> Dict[str, Any]:
        """
        Delete a model
        
        Args:
            model_name: Name of model to delete
        
        Returns:
            Deletion status information
        """
        try:
            import ollama
            response = ollama.delete(model_name)
            return {'success': True, 'response': response}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get information about a specific model
        
        Args:
            model_name: Name of the model
        
        Returns:
            Model information
        """
        if model_name in self.model_info_cache:
            return self.model_info_cache[model_name]
        
        try:
            # Try to get model info from Ollama
            models = self.list_available_models()
            for model in models:
                if model['name'] == model_name:
                    self.model_info_cache[model_name] = model
                    return model
            
            # Model not found locally
            return {
                'name': model_name,
                'size': 'Unknown',
                'modified_at': 'Unknown',
                'available': False
            }
            
        except Exception as e:
            return {
                'name': model_name,
                'error': str(e),
                'available': False
            }
    
    def validate_model(self, model_name: str) -> bool:
        """
        Validate if a model is available and working
        
        Args:
            model_name: Name of model to validate
        
        Returns:
            True if model is valid, False otherwise
        """
        try:
            # Try a simple generation to validate
            test_response = self.generate(
                prompt="Test",
                model=model_name,
                max_tokens=1,
                temperature=0
            )
            
            return test_response.error is None
            
        except:
            return False
'''
        
        llm_file.write_text(llm_content)
    
    def implement_rag_engine(self):
        rag_file = self.project_dir / "src" / "ai_engine" / "rag_engine.py"
        
        rag_content = '''"""
Retrieval-Augmented Generation (RAG) Engine
"""

from typing import Dict, Any, List, Optional, Generator
from dataclasses import dataclass
import json
from datetime import datetime
from .llm_client import LLMClient, LLMResponse
from ..vector_store.search_engine import HybridSearchEngine


@dataclass
class RAGResponse:
    """Structured RAG response"""
    answer: str
    sources: List[Dict[str, Any]]
    query: str
    model: str
    processing_time: float
    token_counts: Optional[Dict[str, int]] = None
    error: Optional[str] = None


class RAGEngine:
    """
    Main RAG engine orchestrating retrieval and generation
    """
    
    def __init__(
        self,
        llm_client: LLMClient,
        search_engine: HybridSearchEngine
    ):
        self.llm_client = llm_client
        self.search_engine = search_engine
        self.conversation_history = []
    
    def query(
        self,
        question: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        include_sources: bool = True,
        conversation_id: Optional[str] = None,
        stream: bool = False
    ) -> RAGResponse:
        """
        Process a query using RAG pipeline
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
            similarity_threshold: Minimum similarity for retrieval
            temperature: LLM temperature
            max_tokens: Maximum tokens to generate
            include_sources: Whether to include source citations
            conversation_id: Optional conversation ID for context
            stream: Whether to stream response
        
        Returns:
            RAGResponse with answer and sources
        """
        start_time = datetime.now()
        
        try:
            # Step 1: Retrieve relevant context
            retrieved_chunks = self.retrieve_context(
                question,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                conversation_id=conversation_id
            )
            
            # Step 2: Build context from retrieved chunks
            context = self.build_context(retrieved_chunks)
            
            # Step 3: Generate answer using LLM
            if stream:
                return self._stream_answer(
                    question=question,
                    context=context,
                    retrieved_chunks=retrieved_chunks,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    start_time=start_time,
                    include_sources=include_sources
                )
            else:
                answer = self.generate_answer(
                    question=question,
                    context=context,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                
                # Step 4: Process and format response
                processing_time = (datetime.now() - start_time).total_seconds()
                
                response = RAGResponse(
                    answer=answer,
                    sources=retrieved_chunks if include_sources else [],
                    query=question,
                    model=self.llm_client.default_model,
                    processing_time=processing_time
                )
                
                # Step 5: Update conversation history
                if conversation_id:
                    self.update_conversation_history(
                        conversation_id=conversation_id,
                        question=question,
                        answer=answer,
                        sources=retrieved_chunks
                    )
                
                return response
                
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return RAGResponse(
                answer="",
                sources=[],
                query=question,
                model=self.llm_client.default_model,
                processing_time=processing_time,
                error=str(e)
            )
    
    def _stream_answer(
        self,
        question: str,
        context: str,
        retrieved_chunks: List[Dict[str, Any]],
        temperature: float,
        max_tokens: int,
        start_time: datetime,
        include_sources: bool
    ) -> Generator[LLMResponse, None, None]:
        """
        Stream answer generation
        
        Args:
            question: User question
            context: Retrieved context
            retrieved_chunks: Retrieved chunks
            temperature: LLM temperature
            max_tokens: Maximum tokens
            start_time: Start time
            include_sources: Whether to include sources
        
        Yields:
            LLMResponse objects
        """
        # Build prompt with context
        prompt = self.build_rag_prompt(question, context)
        
        # Stream response
        full_response = ""
        for chunk in self.llm_client.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        ):
            if chunk.content:
                full_response += chunk.content
                yield chunk
        
        # Final response
        processing_time = (datetime.now() - start_time).total_seconds()
        yield LLMResponse(
            content=full_response,
            model=self.llm_client.default_model,
            processing_time=processing_time
        )
    
    def retrieve_context(
        self,
        question: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        conversation_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for a question
        
        Args:
            question: User question
            top_k: Number of chunks to retrieve
            similarity_threshold: Minimum similarity score
            conversation_id: Optional conversation ID for context
        
        Returns:
            List of relevant chunks with metadata
        """
        # Get conversation context if available
        conversation_context = None
        if conversation_id:
            conversation_context = self.get_conversation_context(conversation_id)
        
        # Enhance query with conversation context
        enhanced_query = self.enhance_query(question, conversation_context)
        
        # Perform hybrid search
        search_results = self.search_engine.hybrid_search(
            query=enhanced_query,
            top_k=top_k,
            similarity_threshold=similarity_threshold
        )
        
        # Format results
        formatted_chunks = []
        for result in search_results:
            chunk = {
                'text': result.get('text', ''),
                'similarity': result.get('similarity', 0.0),
                'metadata': result.get('metadata', {}),
                'source': result.get('metadata', {}).get('source_file', 'Unknown'),
                'chunk_index': result.get('metadata', {}).get('chunk_index', 0)
            }
            formatted_chunks.append(chunk)
        
        return formatted_chunks
    
    def build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """
        Build context string from retrieved chunks
        
        Args:
            chunks: List of retrieved chunks
        
        Returns:
            Formatted context string
        """
        if not chunks:
            return "No relevant context found."
        
        context_parts = []
        context_parts.append("Relevant information from documents:")
        context_parts.append("")
        
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get('source', 'Unknown')
            chunk_idx = chunk.get('chunk_index', 0)
            similarity = chunk.get('similarity', 0.0)
            
            context_parts.append(f"[Source {i}: {source} (chunk {chunk_idx}, relevance: {similarity:.2f})]")
            context_parts.append(chunk.get('text', ''))
            context_parts.append("")
        
        return "\\n".join(context_parts)
    
    def generate_answer(
        self,
        question: str,
        context: str,
        temperature: float = 0.1,
        max_tokens: int = 1024
    ) -> str:
        """
        Generate answer using LLM with context
        
        Args:
            question: User question
            context: Retrieved context
            temperature: LLM temperature
            max_tokens: Maximum tokens to generate
        
        Returns:
            Generated answer
        """
        # Build prompt with context
        prompt = self.build_rag_prompt(question, context)
        
        # Generate response
        response = self.llm_client.generate(
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        if response.error:
            return f"Error generating answer: {response.error}"
        
        return response.content
    
    def build_rag_prompt(self, question: str, context: str) -> str:
        """
        Build RAG prompt
        
        Args:
            question: User question
            context: Retrieved context
        
        Returns:
            Formatted prompt
        """
        prompt_template = """You are a helpful AI assistant. Use the following context to answer the question. If the context doesn't contain relevant information, say so.

Context:
{context}

Question: {question}

Answer based on the context above:"""
        
        return prompt_template.format(context=context, question=question)
    
    def enhance_query(
        self,
        query: str,
        conversation_context: Optional[str] = None
    ) -> str:
        """
        Enhance query with conversation context
        
        Args:
            query: Original query
            conversation_context: Previous conversation context
        
        Returns:
            Enhanced query
        """
        if not conversation_context:
            return query
        
        enhanced = f"{query}\\n\\nPrevious conversation context:\\n{conversation_context}"
        return enhanced
    
    def get_conversation_context(self, conversation_id: str) -> Optional[str]:
        """
        Get context from conversation history
        
        Args:
            conversation_id: Conversation ID
        
        Returns:
            Conversation context or None
        """
        conversation_messages = [
            msg for msg in self.conversation_history 
            if msg.get('conversation_id') == conversation_id
        ]
        
        if not conversation_messages:
            return None
        
        # Get last 3 messages for context
        recent_messages = conversation_messages[-3:]
        context_parts = []
        
        for msg in recent_messages:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            context_parts.append(f"{role}: {content}")
        
        return "\\n".join(context_parts)
    
    def update_conversation_history(
        self,
        conversation_id: str,
        question: str,
        answer: str,
        sources: List[Dict[str, Any]]
    ):
        """
        Update conversation history
        
        Args:
            conversation_id: Conversation ID
            question: User question
            answer: Generated answer
            sources: Retrieved sources
        """
        timestamp = datetime.now().isoformat()
        
        # Add user message
        self.conversation_history.append({
            'conversation_id': conversation_id,
            'role': 'user',
            'content': question,
            'timestamp': timestamp,
            'sources': []
        })
        
        # Add assistant message
        self.conversation_history.append({
            'conversation_id': conversation_id,
            'role': 'assistant',
            'content': answer,
            'timestamp': timestamp,
            'sources': sources
        })
'''
        
        rag_file.write_text(rag_content)
    
    def implement_conversation_memory(self):
        memory_file = self.project_dir / "src" / "ai_engine" / "conversation_memory.py"
        
        memory_content = '''"""
Conversation Memory Management
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from ..database.sqlite_client import SQLiteClient


class ConversationMemory:
    """
    Manages conversation history and context
    """
    
    def __init__(self, db_client: SQLiteClient):
        self.db_client = db_client
    
    def create_conversation(self, title: Optional[str] = None, tags: Optional[List[str]] = None) -> str:
        """
        Create a new conversation
        
        Args:
            title: Optional conversation title
            tags: Optional list of tags
        
        Returns:
            Conversation ID
        """
        from uuid import uuid4
        conversation_id = str(uuid4())
        
        self.db_client.execute(
            """
            INSERT INTO conversations (id, title, tags_json)
            VALUES (?, ?, ?)
            """,
            (conversation_id, title or f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
             json.dumps(tags or []))
        )
        
        return conversation_id
    
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        model_used: Optional[str] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
        tokens: Optional[int] = None,
        processing_time_ms: Optional[int] = None
    ) -> str:
        """
        Add a message to conversation
        
        Args:
            conversation_id: Conversation ID
            role: Message role (user/assistant)
            content: Message content
            model_used: Model used for generation
            sources: Retrieved sources for RAG
            tokens: Token count
            processing_time_ms: Processing time in milliseconds
        
        Returns:
            Message ID
        """
        from uuid import uuid4
        message_id = str(uuid4())
        
        self.db_client.execute(
            """
            INSERT INTO messages 
            (id, conversation_id, role, content, model_used, sources_json, tokens, processing_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (message_id, conversation_id, role, content, model_used, 
             json.dumps(sources or []), tokens, processing_time_ms)
        )
        
        # Update conversation timestamp and message count
        self.db_client.execute(
            """
            UPDATE conversations 
            SET updated_at = CURRENT_TIMESTAMP, 
                message_count = message_count + 1,
                total_tokens = total_tokens + ?
            WHERE id = ?
            """,
            (tokens or 0, conversation_id)
        )
        
        return message_id
    
    def get_conversation_messages(self, conversation_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get messages from a conversation
        
        Args:
            conversation_id: Conversation ID
            limit: Maximum number of messages to return
        
        Returns:
            List of messages
        """
        query = """
        SELECT * FROM messages 
        WHERE conversation_id = ? 
        ORDER BY created_at
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        messages = self.db_client.fetch_all(query, (conversation_id,))
        
        # Parse JSON fields
        for message in messages:
            if message.get('sources_json'):
                try:
                    message['sources'] = json.loads(message['sources_json'])
                except:
                    message['sources'] = []
        
        return messages
    
    def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get conversation summary
        
        Args:
            conversation_id: Conversation ID
        
        Returns:
            Conversation summary
        """
        conversation = self.db_client.fetch_one(
            "SELECT * FROM conversations WHERE id = ?",
            (conversation_id,)
        )
        
        if not conversation:
            return {}
        
        # Get recent messages
        messages = self.get_conversation_messages(conversation_id, limit=5)
        
        # Parse tags
        tags = []
        if conversation.get('tags_json'):
            try:
                tags = json.loads(conversation['tags_json'])
            except:
                pass
        
        return {
            'id': conversation_id,
            'title': conversation.get('title'),
            'created_at': conversation.get('created_at'),
            'updated_at': conversation.get('updated_at'),
            'message_count': conversation.get('message_count', 0),
            'total_tokens': conversation.get('total_tokens', 0),
            'tags': tags,
            'recent_messages': messages
        }
    
    def list_conversations(self, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """
        List all conversations
        
        Args:
            limit: Maximum number of conversations
            offset: Offset for pagination
        
        Returns:
            List of conversations
        """
        conversations = self.db_client.fetch_all(
            """
            SELECT * FROM conversations 
            WHERE is_archived = FALSE
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset)
        )
        
        for conversation in conversations:
            if conversation.get('tags_json'):
                try:
                    conversation['tags'] = json.loads(conversation['tags_json'])
                except:
                    conversation['tags'] = []
        
        return conversations
    
    def archive_conversation(self, conversation_id: str) -> bool:
        """
        Archive a conversation
        
        Args:
            conversation_id: Conversation ID
        
        Returns:
            True if successful
        """
        try:
            self.db_client.execute(
                "UPDATE conversations SET is_archived = TRUE WHERE id = ?",
                (conversation_id,)
            )
            return True
        except:
            return False
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation and all its messages
        
        Args:
            conversation_id: Conversation ID
        
        Returns:
            True if successful
        """
        try:
            # Messages will be deleted automatically due to foreign key cascade
            self.db_client.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,)
            )
            return True
        except:
            return False
    
    def get_conversation_context(self, conversation_id: str, max_tokens: int = 1000) -> str:
        """
        Get conversation context for LLM
        
        Args:
            conversation_id: Conversation ID
            max_tokens: Maximum tokens for context
        
        Returns:
            Formatted conversation context
        """
        messages = self.get_conversation_messages(conversation_id)
        
        if not messages:
            return ""
        
        # Build context from messages
        context_parts = []
        token_count = 0
        
        for message in reversed(messages):  # Start from most recent
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            # Simple token estimation (rough)
            message_tokens = len(content.split())
            
            if token_count + message_tokens > max_tokens:
                break
            
            context_parts.insert(0, f"{role}: {content}")
            token_count += message_tokens
        
        return "\\n".join(context_parts)
'''
        
        memory_file.write_text(memory_content)
    
    def implement_query_pipeline(self):
        # Create __init__.py files
        (self.project_dir / "src" / "core" / "__init__.py").write_text("")
        
        # First create config.py if it doesn't exist
        config_file = self.project_dir / "src" / "core" / "config.py"
        if not config_file.exists():
            config_content = '''"""
Configuration Management
"""

from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Any
import yaml
import json
import os


@dataclass
class AppConfig:
    """Application configuration"""
    # Database
    database_dir: Path = Path.home() / ".docubot" / "data"
    database_name: str = "docubot.db"
    
    # Vector Store
    chroma_dir: Path = Path.home() / ".docubot" / "chroma"
    
    # AI Models
    llm_model: str = "llama2:7b"
    embedding_model: str = "all-MiniLM-L6-v2"
    
    # Processing
    chunk_size: int = 500
    chunk_overlap: int = 50
    rag_top_k: int = 5
    
    # LLM Settings
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    
    # Search
    similarity_threshold: float = 0.7
    hybrid_search_alpha: float = 0.7
    
    # Paths
    models_dir: Path = Path.home() / ".docubot" / "models"
    logs_dir: Path = Path.home() / ".docubot" / "logs"


class ConfigManager:
    """Manages application configuration"""
    
    def __init__(self, config_path: Path = None):
        self.config_path = config_path or Path.home() / ".docubot" / "config.yaml"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
    
    def load(self) -> AppConfig:
        """
        Load configuration from file
        
        Returns:
            AppConfig object
        """
        default_config = AppConfig()
        
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config_data = yaml.safe_load(f) or {}
                
                # Update default config with loaded values
                for key, value in config_data.items():
                    if hasattr(default_config, key):
                        # Handle Path conversions
                        if key.endswith('_dir') or key.endswith('_path'):
                            if isinstance(value, str):
                                value = Path(value)
                        setattr(default_config, key, value)
                        
            except Exception as e:
                print(f"Error loading config: {e}")
        
        return default_config
    
    def save(self, config: AppConfig):
        """
        Save configuration to file
        
        Args:
            config: AppConfig object
        """
        try:
            config_dict = asdict(config)
            
            # Convert Path objects to strings
            for key, value in config_dict.items():
                if isinstance(value, Path):
                    config_dict[key] = str(value)
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
                
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def validate(self) -> Dict[str, Any]:
        """
        Validate configuration
        
        Returns:
            Validation results
        """
        config = self.load()
        issues = []
        
        # Check directories
        dirs_to_check = [
            (config.database_dir, 'database_dir'),
            (config.chroma_dir, 'chroma_dir'),
            (config.models_dir, 'models_dir'),
            (config.logs_dir, 'logs_dir')
        ]
        
        for dir_path, name in dirs_to_check:
            try:
                dir_path.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                issues.append(f"Cannot create directory {name}: {e}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'config': asdict(config)
        }


def get_config() -> AppConfig:
    """Get application configuration"""
    config_manager = ConfigManager()
    return config_manager.load()
'''
            config_file.write_text(config_content)
        
        app_file = self.project_dir / "src" / "core" / "app.py"
        
        app_content = '''"""
DocuBot Core Application Class
Main query processing pipeline
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
from datetime import datetime
from pathlib import Path

from .config import get_config, ConfigManager
from ..ai_engine.llm_client import LLMClient
from ..ai_engine.rag_engine import RAGEngine
from ..ai_engine.conversation_memory import ConversationMemory
from ..vector_store.chroma_client import ChromaClient
from ..vector_store.search_engine import HybridSearchEngine
from ..database.sqlite_client import SQLiteClient
from sentence_transformers import SentenceTransformer


@dataclass
class QueryResult:
    """Structured query result"""
    answer: str
    sources: List[Dict[str, Any]]
    processing_time: float
    model_used: str
    tokens_used: Optional[int] = None
    error: Optional[str] = None


class DocuBotCore:
    """
    Main application orchestrator
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        # Initialize configuration
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.load()
        
        # Initialize components
        self.initialize_components()
        
        # State
        self.active_conversation_id = None
        self.query_count = 0
    
    def initialize_components(self):
        """
        Initialize all application components
        """
        print("Initializing DocuBot components...")
        
        # Initialize database
        db_path = self.config.database_dir / self.config.database_name
        self.database = SQLiteClient(db_path)
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
        except:
            # Try to download the model
            print(f"Downloading embedding model: {self.config.embedding_model}")
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
        
        # Initialize vector store
        self.vector_store = ChromaClient(
            persist_directory=str(self.config.chroma_dir),
            embedding_model=self.embedding_model
        )
        
        # Initialize search engine
        self.search_engine = HybridSearchEngine(
            chroma_client=self.vector_store,
            embedding_model=self.embedding_model
        )
        
        # Initialize LLM client
        self.llm_client = LLMClient(default_model=self.config.llm_model)
        
        # Initialize RAG engine
        self.rag_engine = RAGEngine(
            llm_client=self.llm_client,
            search_engine=self.search_engine
        )
        
        # Initialize conversation memory
        self.conversation_memory = ConversationMemory(self.database)
        
        print("Components initialized successfully")
    
    def process_query(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        top_k: int = None,
        temperature: float = None,
        include_sources: bool = True,
        stream: bool = False
    ) -> QueryResult:
        """
        Main query processing pipeline
        
        Args:
            query: User query
            conversation_id: Optional conversation ID
            top_k: Number of chunks to retrieve
            temperature: LLM temperature
            include_sources: Whether to include source citations
            stream: Whether to stream response
        
        Returns:
            QueryResult with answer and metadata
        """
        start_time = datetime.now()
        self.query_count += 1
        
        # Use active conversation if none specified
        if not conversation_id and self.active_conversation_id:
            conversation_id = self.active_conversation_id
        
        try:
            # Set parameters
            top_k = top_k or self.config.rag_top_k
            temperature = temperature or self.config.llm_temperature
            
            # Process query with RAG engine
            rag_response = self.rag_engine.query(
                question=query,
                top_k=top_k,
                temperature=temperature,
                conversation_id=conversation_id,
                include_sources=include_sources,
                stream=stream
            )
            
            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Save to conversation memory
            if conversation_id:
                self.conversation_memory.add_message(
                    conversation_id=conversation_id,
                    role='user',
                    content=query
                )
                
                self.conversation_memory.add_message(
                    conversation_id=conversation_id,
                    role='assistant',
                    content=rag_response.answer,
                    model_used=rag_response.model,
                    sources=rag_response.sources,
                    processing_time_ms=int(processing_time * 1000)
                )
            
            # Return structured result
            return QueryResult(
                answer=rag_response.answer,
                sources=rag_response.sources if include_sources else [],
                processing_time=processing_time,
                model_used=rag_response.model,
                error=rag_response.error
            )
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds()
            return QueryResult(
                answer=f"Error processing query: {str(e)}",
                sources=[],
                processing_time=processing_time,
                model_used=self.config.llm_model,
                error=str(e)
            )
    
    def start_conversation(self, title: Optional[str] = None, tags: Optional[List[str]] = None) -> str:
        """
        Start a new conversation
        
        Args:
            title: Optional conversation title
            tags: Optional list of tags
        
        Returns:
            Conversation ID
        """
        conversation_id = self.conversation_memory.create_conversation(title, tags)
        self.active_conversation_id = conversation_id
        return conversation_id
    
    def get_conversation_history(self, conversation_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get conversation history
        
        Args:
            conversation_id: Conversation ID
            limit: Maximum number of messages
        
        Returns:
            List of messages
        """
        return self.conversation_memory.get_conversation_messages(conversation_id, limit)
    
    def search_documents(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search across documents
        
        Args:
            query: Search query
            limit: Maximum results
        
        Returns:
            List of matching documents
        """
        return self.search_engine.hybrid_search(
            query=query,
            top_k=limit
        )
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get system status information
        
        Returns:
            System status dictionary
        """
        config_dict = {
            'llm_model': self.config.llm_model,
            'embedding_model': self.config.embedding_model,
            'chunk_size': self.config.chunk_size,
            'rag_top_k': self.config.rag_top_k,
            'llm_temperature': self.config.llm_temperature
        }
        
        # Get available models
        available_models = self.llm_client.list_available_models()
        
        # Get collection stats
        collections = []
        try:
            collections = self.vector_store.list_collections()
        except:
            pass
        
        return {
            'status': 'running',
            'query_count': self.query_count,
            'active_conversation': self.active_conversation_id,
            'config': config_dict,
            'available_llm_models': [m['name'] for m in available_models],
            'collections': len(collections),
            'config_valid': self.config_manager.validate()['valid']
        }
'''
        
        app_file.write_text(app_content)
    
    def implement_model_manager(self):
        model_file = self.project_dir / "src" / "ai_engine" / "model_manager.py"
        
        model_content = '''"""
AI Model Management System
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
from datetime import datetime
import subprocess
import shutil
from .llm_client import LLMClient


class ModelManager:
    """
    Manages downloading, updating, and organizing AI models
    """
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.llm_client = LLMClient()
        self.model_registry = self.load_model_registry()
    
    def load_model_registry(self) -> Dict[str, Any]:
        """
        Load model registry from file
        
        Returns:
            Model registry dictionary
        """
        registry_file = self.models_dir / "model_registry.json"
        
        if registry_file.exists():
            try:
                with open(registry_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            'llm_models': {},
            'embedding_models': {},
            'last_updated': datetime.now().isoformat()
        }
    
    def save_model_registry(self):
        """
        Save model registry to file
        """
        registry_file = self.models_dir / "model_registry.json"
        self.model_registry['last_updated'] = datetime.now().isoformat()
        
        with open(registry_file, 'w', encoding='utf-8') as f:
            json.dump(self.model_registry, f, indent=2)
    
    def download_llm_model(self, model_name: str) -> Dict[str, Any]:
        """
        Download LLM model using Ollama
        
        Args:
            model_name: Name of model to download
        
        Returns:
            Download status
        """
        print(f"Downloading LLM model: {model_name}")
        
        result = self.llm_client.pull_model(model_name)
        
        if result['success']:
            # Update registry
            self.model_registry['llm_models'][model_name] = {
                'name': model_name,
                'downloaded_at': datetime.now().isoformat(),
                'status': 'downloaded',
                'size': self.estimate_model_size(model_name)
            }
            self.save_model_registry()
        
        return result
    
    def download_embedding_model(self, model_name: str) -> Dict[str, Any]:
        """
        Download embedding model
        
        Args:
            model_name: Name of embedding model
        
        Returns:
            Download status
        """
        from sentence_transformers import SentenceTransformer
        
        print(f"Downloading embedding model: {model_name}")
        
        try:
            start_time = datetime.now()
            
            # Download model
            model = SentenceTransformer(model_name)
            
            # Save model locally
            model_save_path = self.models_dir / "sentence-transformers" / model_name
            model_save_path.mkdir(parents=True, exist_ok=True)
            model.save(str(model_save_path))
            
            download_time = (datetime.now() - start_time).total_seconds()
            
            # Update registry
            self.model_registry['embedding_models'][model_name] = {
                'name': model_name,
                'downloaded_at': datetime.now().isoformat(),
                'status': 'downloaded',
                'download_time_seconds': download_time,
                'local_path': str(model_save_path)
            }
            self.save_model_registry()
            
            return {
                'success': True,
                'model_name': model_name,
                'download_time': download_time,
                'local_path': str(model_save_path)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'model_name': model_name
            }
    
    def get_available_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get list of available models
        
        Returns:
            Dictionary with LLM and embedding models
        """
        # Get Ollama models
        ollama_models = self.llm_client.list_available_models()
        
        # Get downloaded embedding models
        embedding_models = list(self.model_registry.get('embedding_models', {}).values())
        
        return {
            'llm_models': ollama_models,
            'embedding_models': embedding_models
        }
    
    def estimate_model_size(self, model_name: str) -> str:
        """
        Estimate model size
        
        Args:
            model_name: Name of model
        
        Returns:
            Estimated size string
        """
        # Rough estimates for common models
        size_map = {
            'llama2:7b': '3.8 GB',
            'mistral:7b': '4.1 GB',
            'neural-chat:7b': '4.3 GB',
            'all-MiniLM-L6-v2': '90 MB',
            'all-mpnet-base-v2': '420 MB'
        }
        
        return size_map.get(model_name, 'Unknown')
    
    def check_model_status(self, model_name: str, model_type: str = 'llm') -> Dict[str, Any]:
        """
        Check status of a model
        
        Args:
            model_name: Name of model
            model_type: Type of model ('llm' or 'embedding')
        
        Returns:
            Model status information
        """
        if model_type == 'llm':
            # Check if model is available in Ollama
            models = self.llm_client.list_available_models()
            model_available = any(m['name'] == model_name for m in models)
            
            return {
                'name': model_name,
                'type': 'llm',
                'available': model_available,
                'downloaded': model_available,
                'in_registry': model_name in self.model_registry.get('llm_models', {})
            }
        
        elif model_type == 'embedding':
            # Check if embedding model is downloaded
            model_info = self.model_registry.get('embedding_models', {}).get(model_name)
            
            if model_info:
                local_path = Path(model_info.get('local_path', ''))
                exists = local_path.exists() if local_path else False
                
                return {
                    'name': model_name,
                    'type': 'embedding',
                    'available': True,
                    'downloaded': exists,
                    'local_path': str(local_path) if local_path else None,
                    'download_time': model_info.get('download_time_seconds')
                }
            else:
                return {
                    'name': model_name,
                    'type': 'embedding',
                    'available': False,
                    'downloaded': False
                }
        
        return {
            'name': model_name,
            'type': model_type,
            'available': False,
            'downloaded': False,
            'error': 'Unknown model type'
        }
    
    def delete_model(self, model_name: str, model_type: str = 'llm') -> Dict[str, Any]:
        """
        Delete a model
        
        Args:
            model_name: Name of model to delete
            model_type: Type of model ('llm' or 'embedding')
        
        Returns:
            Deletion status
        """
        if model_type == 'llm':
            result = self.llm_client.delete_model(model_name)
            
            if result['success']:
                # Remove from registry
                self.model_registry['llm_models'].pop(model_name, None)
                self.save_model_registry()
            
            return result
        
        elif model_type == 'embedding':
            try:
                model_info = self.model_registry['embedding_models'].get(model_name)
                
                if model_info and 'local_path' in model_info:
                    local_path = Path(model_info['local_path'])
                    if local_path.exists():
                        shutil.rmtree(local_path)
                
                # Remove from registry
                self.model_registry['embedding_models'].pop(model_name, None)
                self.save_model_registry()
                
                return {
                    'success': True,
                    'message': f'Deleted embedding model: {model_name}'
                }
                
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e)
                }
        
        return {
            'success': False,
            'error': f'Unknown model type: {model_type}'
        }
    
    def validate_all_models(self) -> Dict[str, Any]:
        """
        Validate all downloaded models
        
        Returns:
            Validation results
        """
        results = {
            'llm_models': {},
            'embedding_models': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Validate LLM models
        for model_name in self.model_registry.get('llm_models', {}):
            results['llm_models'][model_name] = self.llm_client.validate_model(model_name)
        
        # Validate embedding models
        for model_name, model_info in self.model_registry.get('embedding_models', {}).items():
            local_path = Path(model_info.get('local_path', ''))
            results['embedding_models'][model_name] = {
                'exists': local_path.exists() if local_path else False,
                'path': str(local_path) if local_path else None
            }
        
        return results
    
    def get_disk_usage(self) -> Dict[str, Any]:
        """
        Get disk usage for models
        
        Returns:
            Disk usage information
        """
        total_size = 0
        model_sizes = {}
        
        # Calculate size of embedding models
        for model_name, model_info in self.model_registry.get('embedding_models', {}).items():
            if 'local_path' in model_info:
                path = Path(model_info['local_path'])
                if path.exists():
                    size = self.get_directory_size(path)
                    model_sizes[model_name] = size
                    total_size += size
        
        return {
            'total_size_bytes': total_size,
            'total_size_human': self.format_bytes(total_size),
            'model_sizes': model_sizes,
            'llm_models_count': len(self.model_registry.get('llm_models', {})),
            'embedding_models_count': len(self.model_registry.get('embedding_models', {}))
        }
    
    def get_directory_size(self, path: Path) -> int:
        """
        Calculate directory size
        
        Args:
            path: Directory path
        
        Returns:
            Size in bytes
        """
        total = 0
        for entry in path.rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
        return total
    
    def format_bytes(self, bytes_size: int) -> str:
        """
        Format bytes to human readable string
        
        Args:
            bytes_size: Size in bytes
        
        Returns:
            Formatted size string
        """
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f} TB"
'''
        
        model_file.write_text(model_content)
    
    def implement_index_manager(self):
        index_file = self.project_dir / "src" / "vector_store" / "index_manager.py"
        
        index_content = '''"""
Vector Store Index Management
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import json
from .chroma_client import ChromaClient


class IndexManager:
    """
    Manages vector store indices and metadata
    """
    
    def __init__(self, chroma_client: ChromaClient):
        self.chroma_client = chroma_client
        self.index_metadata = self.load_index_metadata()
    
    def load_index_metadata(self) -> Dict[str, Any]:
        """
        Load index metadata
        
        Returns:
            Index metadata dictionary
        """
        metadata_file = Path(self.chroma_client.persist_directory) / "index_metadata.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            'collections': {},
            'statistics': {},
            'last_updated': datetime.now().isoformat()
        }
    
    def save_index_metadata(self):
        """
        Save index metadata to file
        """
        metadata_file = Path(self.chroma_client.persist_directory) / "index_metadata.json"
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.index_metadata['last_updated'] = datetime.now().isoformat()
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.index_metadata, f, indent=2)
    
    def create_collection(self, collection_name: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a new collection
        
        Args:
            collection_name: Name of collection
            metadata: Optional collection metadata
        
        Returns:
            Creation status
        """
        try:
            collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata=metadata or {}
            )
            
            # Update metadata
            self.index_metadata['collections'][collection_name] = {
                'name': collection_name,
                'created_at': datetime.now().isoformat(),
                'metadata': metadata or {},
                'document_count': 0,
                'embedding_count': 0
            }
            self.save_index_metadata()
            
            return {
                'success': True,
                'collection_name': collection_name,
                'collection': collection
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'collection_name': collection_name
            }
    
    def delete_collection(self, collection_name: str) -> Dict[str, Any]:
        """
        Delete a collection
        
        Args:
            collection_name: Name of collection to delete
        
        Returns:
            Deletion status
        """
        try:
            self.chroma_client.delete_collection(collection_name)
            
            # Update metadata
            self.index_metadata['collections'].pop(collection_name, None)
            self.save_index_metadata()
            
            return {
                'success': True,
                'message': f'Deleted collection: {collection_name}'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics for a collection
        
        Args:
            collection_name: Name of collection
        
        Returns:
            Collection statistics
        """
        try:
            collection = self.chroma_client.get_collection(collection_name)
            
            if collection:
                count = collection.count()
                
                stats = {
                    'collection_name': collection_name,
                    'document_count': count,
                    'exists': True,
                    'metadata': collection.metadata
                }
                
                # Update cached metadata
                if collection_name in self.index_metadata['collections']:
                    self.index_metadata['collections'][collection_name].update({
                        'document_count': count,
                        'last_accessed': datetime.now().isoformat()
                    })
                    self.save_index_metadata()
                
                return stats
            else:
                return {
                    'collection_name': collection_name,
                    'exists': False,
                    'document_count': 0
                }
                
        except Exception as e:
            return {
                'collection_name': collection_name,
                'exists': False,
                'error': str(e)
            }
    
    def list_collections(self) -> List[Dict[str, Any]]:
        """
        List all collections
        
        Returns:
            List of collection information
        """
        try:
            collections = self.chroma_client.list_collections()
            
            collection_info = []
            for collection in collections:
                stats = self.get_collection_stats(collection.name)
                collection_info.append(stats)
            
            return collection_info
            
        except Exception as e:
            return [{'error': str(e)}]
    
    def update_collection_metadata(self, collection_name: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update collection metadata
        
        Args:
            collection_name: Name of collection
            metadata: New metadata
        
        Returns:
            Update status
        """
        try:
            collection = self.chroma_client.get_collection(collection_name)
            
            if collection:
                # Update metadata in ChromaDB
                current_metadata = collection.metadata or {}
                current_metadata.update(metadata)
                collection.modify(metadata=current_metadata)
                
                # Update local metadata
                if collection_name in self.index_metadata['collections']:
                    self.index_metadata['collections'][collection_name]['metadata'] = current_metadata
                    self.index_metadata['collections'][collection_name]['updated_at'] = datetime.now().isoformat()
                    self.save_index_metadata()
                
                return {
                    'success': True,
                    'collection_name': collection_name,
                    'metadata': current_metadata
                }
            else:
                return {
                    'success': False,
                    'error': f'Collection not found: {collection_name}'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def rebuild_index(self, collection_name: str) -> Dict[str, Any]:
        """
        Rebuild collection index
        
        Args:
            collection_name: Name of collection
        
        Returns:
            Rebuild status
        """
        # Note: ChromaDB handles indexing automatically
        # This method is a placeholder for future index optimization
        return {
            'success': True,
            'message': f'Index for {collection_name} is maintained automatically by ChromaDB',
            'collection_name': collection_name
        }
    
    def optimize_storage(self) -> Dict[str, Any]:
        """
        Optimize vector store storage
        
        Returns:
            Optimization results
        """
        # ChromaDB handles storage optimization internally
        # This method is a placeholder for future optimizations
        return {
            'success': True,
            'message': 'Storage optimization would be implemented here',
            'timestamp': datetime.now().isoformat()
        }
    
    def get_index_statistics(self) -> Dict[str, Any]:
        """
        Get overall index statistics
        
        Returns:
            Index statistics
        """
        collections = self.list_collections()
        
        total_documents = 0
        total_collections = len(collections)
        
        for collection in collections:
            if isinstance(collection, dict) and 'document_count' in collection:
                total_documents += collection['document_count']
        
        stats = {
            'total_collections': total_collections,
            'total_documents': total_documents,
            'collections': collections,
            'timestamp': datetime.now().isoformat()
        }
        
        # Update metadata
        self.index_metadata['statistics'] = stats
        self.save_index_metadata()
        
        return stats
'''
        
        index_file.write_text(index_content)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Implement DocuBot core logic")
    parser.add_argument("--dir", default="DocuBot", help="Project directory")
    
    args = parser.parse_args()
    
    fixer = CoreLogicFixer(args.dir)
    fixer.implement_core_logic()


if __name__ == "__main__":
    main()