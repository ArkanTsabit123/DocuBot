# File: src/ai_engine/__init__.py
"""
AI Engine module for DocuBot.
"""

from .embedding_service import EmbeddingService, get_embedding_service, create_embedding_service
from .llm_client import LLMClient, create_llm_client, get_global_llm_client
from .model_manager import ModelManager, get_model_manager
from .rag_engine import RAGEngine
from .prompt_templates import PromptTemplates
from .summarizer import Summarizer
from .tagging import Tagger

__all__ = [
    'EmbeddingService',
    'get_embedding_service',
    'create_embedding_service',  # <-- INI YANG BENAR
    'LLMClient',
    'create_llm_client',
    'get_global_llm_client',
    'ModelManager',
    'get_model_manager',
    'RAGEngine',
    'PromptTemplates',
    'Summarizer',
    'Tagger'
]