# DocuBot/src/core/app.py

"""
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
