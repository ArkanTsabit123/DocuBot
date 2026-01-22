# docubot/src/ai_engine/llm_client.py

"""
Complete LLM Client for DocuBot with Ollama Integration
Supports multiple models, streaming, configuration management, error handling,
performance tracking, and model management.
"""

import os
import json
import time
import logging
import re
from datetime import datetime
from typing import Dict, List, Any, Optional, Generator, Union, Tuple
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, field
import subprocess
import threading

try:
    import ollama
    from ollama import Client
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("Ollama Python client not installed. Install with: pip install ollama")

try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

from ..core.config import get_config
from ..core.exceptions import ModelError, ConnectionError, ConfigurationError


logger = logging.getLogger(__name__)


class LLMModel(str, Enum):
    """Supported LLM models enumeration."""
    LLAMA2_7B = "llama2:7b"
    MISTRAL_7B = "mistral:7b"
    NEURAL_CHAT_7B = "neural-chat:7b"
    LLAMA3_8B = "llama3:8b"
    CODEDLLAMA_7B = "codellama:7b"
    PHI_2_7B = "phi:2.7b"
    MIXTRAL_8X7B = "mixtral:8x7b"
    GEMMA_2B = "gemma:2b"
    GEMMA_7B = "gemma:7b"


@dataclass
class LLMConfig:
    """Complete LLM configuration with validation."""
    model: str = "llama2:7b"
    host: str = "http://localhost:11434"
    temperature: float = 0.1
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 1024
    context_window: int = 4096
    repeat_penalty: float = 1.1
    seed: Optional[int] = None
    timeout: int = 300
    max_retries: int = 3
    retry_delay: int = 2
    streaming: bool = False
    system_prompt: Optional[str] = None
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate configuration values."""
        errors = []
        
        # Validate temperature
        if not 0.0 <= self.temperature <= 2.0:
            errors.append(f"Temperature must be between 0.0 and 2.0, got {self.temperature}")
        
        # Validate top_p
        if not 0.0 < self.top_p <= 1.0:
            errors.append(f"top_p must be between 0.0 and 1.0, got {self.top_p}")
        
        # Validate max_tokens
        if self.max_tokens <= 0:
            errors.append(f"max_tokens must be positive, got {self.max_tokens}")
        
        # Validate timeout
        if self.timeout <= 0:
            errors.append(f"timeout must be positive, got {self.timeout}")
        
        # Validate host URL
        if not self.host.startswith(('http://', 'https://')):
            errors.append(f"Host must be a valid URL starting with http:// or https://, got {self.host}")
        
        # Validate retry settings
        if self.max_retries < 0:
            errors.append(f"max_retries must be non-negative, got {self.max_retries}")
        
        if self.retry_delay < 0:
            errors.append(f"retry_delay must be non-negative, got {self.retry_delay}")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'model': self.model,
            'host': self.host,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'top_k': self.top_k,
            'max_tokens': self.max_tokens,
            'context_window': self.context_window,
            'repeat_penalty': self.repeat_penalty,
            'seed': self.seed,
            'timeout': self.timeout,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'streaming': self.streaming,
            'system_prompt': self.system_prompt
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'LLMConfig':
        """Create LLMConfig from dictionary."""
        return cls(**config_dict)


@dataclass
class LLMResponse:
    """Complete LLM response container with metadata."""
    content: str
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    processing_time_ms: float = 0.0
    finish_reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    raw_response: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            'content': self.content,
            'model': self.model,
            'prompt_tokens': self.prompt_tokens,
            'completion_tokens': self.completion_tokens,
            'total_tokens': self.total_tokens,
            'processing_time_ms': self.processing_time_ms,
            'finish_reason': self.finish_reason,
            'metadata': self.metadata,
            'raw_response': self.raw_response
        }


@dataclass
class ModelInfo:
    """Complete model information container."""
    name: str
    display_name: str
    description: str
    context_window: int
    requirements: Dict[str, str]
    default_parameters: Dict[str, Any]
    tags: List[str]
    size_gb: float = 0.0
    modified_at: Optional[str] = None
    digest: Optional[str] = None
    is_available: bool = False
    is_current: bool = False
    license: Optional[str] = None
    modelfile: Optional[str] = None
    template: Optional[str] = None


class LLMClient:
    """
    Complete LLM Client implementation for DocuBot.
    Combines robust configuration management, multi-model support,
    error handling, performance tracking, and model management.
    """
    
    # Complete model specifications database
    MODEL_DATABASE = {
        "llama2:7b": {
            "display_name": "Llama 2 7B",
            "description": "Meta's Llama 2 7B parameter model - Balanced performance for general tasks",
            "context_window": 4096,
            "requirements": {
                "ram": "8GB",
                "storage": "4.2GB",
                "download_size": "3.8GB"
            },
            "default_parameters": {
                "temperature": 0.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": 1024
            },
            "tags": ["general", "balanced", "english", "recommended"],
            "license": "Llama 2 Community License"
        },
        "mistral:7b": {
            "display_name": "Mistral 7B",
            "description": "Mistral AI's 7B parameter model - Fast and efficient with larger context",
            "context_window": 8192,
            "requirements": {
                "ram": "8GB",
                "storage": "4.1GB",
                "download_size": "4.1GB"
            },
            "default_parameters": {
                "temperature": 0.2,
                "top_p": 0.95,
                "top_k": 50,
                "num_predict": 2048
            },
            "tags": ["fast", "efficient", "large-context"],
            "license": "Apache 2.0"
        },
        "neural-chat:7b": {
            "display_name": "Neural Chat 7B",
            "description": "Intel's fine-tuned neural chat model - High accuracy for conversational tasks",
            "context_window": 4096,
            "requirements": {
                "ram": "8GB",
                "storage": "4.3GB",
                "download_size": "4.3GB"
            },
            "default_parameters": {
                "temperature": 0.05,
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": 1024
            },
            "tags": ["chat", "conversational", "accurate"],
            "license": "MIT"
        },
        "llama3:8b": {
            "display_name": "Llama 3 8B",
            "description": "Meta's latest Llama 3 model - Improved performance and capabilities",
            "context_window": 8192,
            "requirements": {
                "ram": "8GB",
                "storage": "4.7GB",
                "download_size": "4.7GB"
            },
            "default_parameters": {
                "temperature": 0.1,
                "top_p": 0.9,
                "top_k": 40,
                "num_predict": 2048
            },
            "tags": ["latest", "improved", "multilingual"],
            "license": "Llama 3 Community License"
        },
        "codellama:7b": {
            "display_name": "CodeLlama 7B",
            "description": "Code generation model based on Llama 2 - Specialized for programming tasks",
            "context_window": 16384,
            "requirements": {
                "ram": "8GB",
                "storage": "3.8GB",
                "download_size": "3.8GB"
            },
            "default_parameters": {
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 50,
                "num_predict": 2048
            },
            "tags": ["code", "programming", "technical"],
            "license": "Llama 2 Community License"
        },
        "phi:2.7b": {
            "display_name": "Phi 2.7B",
            "description": "Microsoft's small but capable model - Efficient for limited resources",
            "context_window": 2048,
            "requirements": {
                "ram": "4GB",
                "storage": "1.7GB",
                "download_size": "1.7GB"
            },
            "default_parameters": {
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 50,
                "num_predict": 1024
            },
            "tags": ["small", "efficient", "low-resource"],
            "license": "MIT"
        },
        "gemma:2b": {
            "display_name": "Gemma 2B",
            "description": "Google's lightweight Gemma model - Fast and efficient for basic tasks",
            "context_window": 8192,
            "requirements": {
                "ram": "4GB",
                "storage": "1.6GB",
                "download_size": "1.6GB"
            },
            "default_parameters": {
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 50,
                "num_predict": 2048
            },
            "tags": ["lightweight", "fast", "google"],
            "license": "Gemma Terms"
        }
    }
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize complete LLM client.
        
        Args:
            config: LLM configuration (uses app config if None)
        
        Raises:
            ImportError: If Ollama is not installed
            ConfigurationError: If configuration is invalid
        """
        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "Ollama Python client is not installed. "
                "Please install with: pip install ollama"
            )
        
        # Load configuration
        if config is None:
            self.config = self._load_config_from_app()
        else:
            self.config = config
        
        # Validate configuration
        is_valid, errors = self.config.validate()
        if not is_valid:
            error_msg = f"Invalid LLM configuration: {', '.join(errors)}"
            logger.error(error_msg)
            raise ConfigurationError(error_msg)
        
        # Initialize Ollama client
        try:
            self.client = Client(
                host=self.config.host,
                timeout=self.config.timeout
            )
        except Exception as e:
            logger.error(f"Failed to initialize Ollama client: {e}")
            raise ConnectionError(f"Failed to initialize Ollama client: {e}")
        
        # Connection state management
        self.connected = False
        self.connection_attempts = 0
        self.last_successful_connection = None
        
        # Performance tracking
        self.request_count = 0
        self.total_tokens_processed = 0
        self.total_processing_time_ms = 0.0
        self.error_count = 0
        
        # Model cache
        self._available_models_cache = None
        self._cache_timestamp = None
        self._cache_ttl = 30  # seconds
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Test connection
        self._test_connection()
        
        logger.info(f"LLMClient initialized with model: {self.config.model}")
        logger.debug(f"Configuration: {self.config.to_dict()}")
    
    def _load_config_from_app(self) -> LLMConfig:
        """
        Load configuration from DocuBot application configuration.
        
        Returns:
            LLMConfig instance
        """
        try:
            app_config = get_config()
            
            # Get LLM configuration from app config
            llm_config_dict = app_config.ai.get('llm', {})
            
            # Map configuration keys
            config_dict = {
                'model': llm_config_dict.get('model', 'llama2:7b'),
                'host': llm_config_dict.get('host', 'http://localhost:11434'),
                'temperature': llm_config_dict.get('temperature', 0.1),
                'top_p': llm_config_dict.get('top_p', 0.9),
                'top_k': llm_config_dict.get('top_k', 40),
                'max_tokens': llm_config_dict.get('max_tokens', 1024),
                'context_window': llm_config_dict.get('context_window', 4096),
                'repeat_penalty': llm_config_dict.get('repeat_penalty', 1.1),
                'seed': llm_config_dict.get('seed'),
                'timeout': llm_config_dict.get('timeout', 300),
                'max_retries': llm_config_dict.get('max_retries', 3),
                'retry_delay': llm_config_dict.get('retry_delay', 2),
                'streaming': llm_config_dict.get('streaming', False),
                'system_prompt': llm_config_dict.get('system_prompt')
            }
            
            return LLMConfig(**config_dict)
        
        except Exception as e:
            logger.error(f"Error loading app configuration: {e}")
            logger.warning("Using default LLM configuration")
            return LLMConfig()
    
    def _test_connection(self, retry: bool = True) -> bool:
        """
        Test connection to Ollama server with robust retry logic.
        
        Args:
            retry: Whether to retry on failure
            
        Returns:
            True if connection successful
        """
        max_attempts = self.config.max_retries if retry else 1
        
        for attempt in range(max_attempts):
            self.connection_attempts += 1
            
            try:
                with self._lock:
                    response = self.client.list()
                
                self.connected = True
                self.last_successful_connection = datetime.now()
                
                model_count = len(response.get('models', []))
                logger.info(f"Connected to Ollama server at {self.config.host}")
                logger.info(f"Found {model_count} available model(s)")
                
                # Update cache
                self._available_models_cache = response.get('models', [])
                self._cache_timestamp = datetime.now()
                
                return True
            
            except Exception as e:
                error_msg = str(e)
                
                if attempt < max_attempts - 1:
                    wait_time = self.config.retry_delay * (attempt + 1)
                    logger.warning(
                        f"Connection attempt {attempt + 1}/{max_attempts} failed: {error_msg}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    self.connected = False
                    logger.error(f"Failed to connect to Ollama server after {max_attempts} attempts: {error_msg}")
        
        return False
    
    def ensure_connection(self) -> bool:
        """
        Ensure connection to Ollama server, with reconnection if needed.
        
        Returns:
            True if connected or reconnection successful
        """
        if self.connected:
            # Verify connection is still alive
            try:
                with self._lock:
                    self.client.list()
                return True
            except Exception:
                self.connected = False
                logger.warning("Connection lost, attempting to reconnect...")
        
        return self._test_connection(retry=True)
    
    def _get_cached_models(self) -> List[Dict[str, Any]]:
        """Get cached available models with TTL."""
        current_time = datetime.now()
        
        if (self._available_models_cache is not None and 
            self._cache_timestamp is not None and
            (current_time - self._cache_timestamp).total_seconds() < self._cache_ttl):
            return self._available_models_cache
        
        # Refresh cache
        if self.ensure_connection():
            try:
                with self._lock:
                    response = self.client.list()
                self._available_models_cache = response.get('models', [])
                self._cache_timestamp = current_time
                return self._available_models_cache
            except Exception as e:
                logger.error(f"Failed to refresh model cache: {e}")
        
        return []
    
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        stream: Optional[bool] = None,
        options: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        max_retries: Optional[int] = None
    ) -> Union[LLMResponse, Generator[str, None, None]]:
        """
        Generate a response from the LLM with error handling.
        
        Args:
            prompt: Input prompt
            model: Model to use (overrides default)
            stream: Whether to stream response (overrides config)
            options: Additional generation options
            system_prompt: System prompt for context
            conversation_history: Previous conversation messages
            max_retries: Maximum retries for this request
            
        Returns:
            LLMResponse for non-streaming, Generator for streaming
        
        Raises:
            ConnectionError: If not connected to Ollama
            ModelError: If model generation fails after retries
        """
        # Determine parameters
        target_model = model or self.config.model
        should_stream = stream if stream is not None else self.config.streaming
        retries = max_retries if max_retries is not None else self.config.max_retries
        
        # Ensure connection
        if not self.ensure_connection():
            raise ConnectionError("Not connected to Ollama server")
        
        # Prepare generation options
        gen_options = self._prepare_generation_options(options, target_model)
        
        # Prepare messages for chat API
        if system_prompt or conversation_history:
            messages = self._prepare_messages(prompt, system_prompt, conversation_history)
            use_chat_api = True
        else:
            messages = None
            use_chat_api = False
        
        # Track performance
        start_time = time.time()
        
        # Attempt generation with retries
        for attempt in range(retries + 1):
            try:
                if should_stream:
                    if use_chat_api:
                        return self._chat_stream(messages, target_model, gen_options, start_time)
                    else:
                        return self._generate_stream(prompt, target_model, gen_options, start_time)
                else:
                    if use_chat_api:
                        return self._chat_complete(messages, target_model, gen_options, start_time)
                    else:
                        return self._generate_complete(prompt, target_model, gen_options, start_time)
            
            except Exception as e:
                self.error_count += 1
                
                if attempt < retries:
                    wait_time = self.config.retry_delay * (attempt + 1)
                    logger.warning(
                        f"Generation attempt {attempt + 1}/{retries + 1} failed: {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                    
                    # Try to reconnect
                    self.ensure_connection()
                else:
                    error_msg = f"Generation failed after {retries + 1} attempts: {e}"
                    logger.error(error_msg)
                    
                    # Return error response
                    processing_time = (time.time() - start_time) * 1000
                    
                    if should_stream:
                        def error_generator():
                            yield f"Error: {error_msg}"
                        return error_generator()
                    else:
                        return LLMResponse(
                            content=f"Error: {error_msg}",
                            model=target_model,
                            processing_time_ms=processing_time,
                            finish_reason="error",
                            metadata={"error": str(e), "attempts": attempt + 1}
                        )
    
    def _prepare_generation_options(self, options: Optional[Dict[str, Any]], model_name: str) -> Dict[str, Any]:
        """Prepare generation options with model-specific defaults."""
        # Get model-specific defaults
        model_defaults = self.MODEL_DATABASE.get(model_name, {}).get("default_parameters", {})
        
        # Start with config defaults
        base_options = {
            'temperature': self.config.temperature,
            'top_p': self.config.top_p,
            'top_k': self.config.top_k,
            'num_predict': self.config.max_tokens,
            'repeat_penalty': self.config.repeat_penalty,
            'seed': self.config.seed
        }
        
        # Apply model-specific defaults (overriding config)
        for key, value in model_defaults.items():
            if key not in ['num_predict']:  # Don't override max_tokens from model defaults
                base_options[key] = value
        
        # Apply user options (overriding everything)
        if options:
            base_options.update(options)
        
        # Remove None values
        final_options = {k: v for k, v in base_options.items() if v is not None}
        
        logger.debug(f"Generation options for {model_name}: {final_options}")
        return final_options
    
    def _prepare_messages(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """Prepare messages list for chat API with validation."""
        messages = []
        
        # Use configured system prompt if none provided
        if system_prompt is None and self.config.system_prompt:
            system_prompt = self.config.system_prompt
        
        # Add system prompt if available
        if system_prompt:
            messages.append({
                'role': 'system',
                'content': system_prompt.strip()
            })
        
        # Add conversation history
        if conversation_history:
            for msg in conversation_history:
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    # Validate role
                    if msg['role'] not in ['system', 'user', 'assistant']:
                        logger.warning(f"Invalid message role: {msg['role']}, defaulting to 'user'")
                        msg['role'] = 'user'
                    
                    messages.append({
                        'role': msg['role'],
                        'content': str(msg['content']).strip()
                    })
                else:
                    logger.warning(f"Invalid message format in conversation history: {msg}")
        
        # Add current user prompt
        messages.append({
            'role': 'user',
            'content': prompt.strip()
        })
        
        # Log token estimate
        total_chars = sum(len(msg['content']) for msg in messages)
        estimated_tokens = total_chars // 4  # Rough estimate
        
        logger.debug(f"Prepared {len(messages)} messages, estimated {estimated_tokens} tokens")
        return messages
    
    def _generate_complete(
        self,
        prompt: str,
        model: str,
        options: Dict[str, Any],
        start_time: float
    ) -> LLMResponse:
        """Non-streaming generation with performance tracking."""
        try:
            with self._lock:
                response = self.client.generate(
                    model=model,
                    prompt=prompt,
                    options=options,
                    stream=False
                )
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Update performance metrics
            with self._lock:
                self.request_count += 1
                self.total_processing_time_ms += processing_time_ms
                self.total_tokens_processed += response.get('eval_count', 0)
            
            return LLMResponse(
                content=response['response'],
                model=model,
                prompt_tokens=response.get('prompt_eval_count'),
                completion_tokens=response.get('eval_count'),
                total_tokens=response.get('prompt_eval_count', 0) + response.get('eval_count', 0),
                processing_time_ms=processing_time_ms,
                finish_reason=response.get('done_reason', 'stop'),
                metadata={
                    'total_duration': response.get('total_duration', 0),
                    'load_duration': response.get('load_duration', 0),
                    'prompt_eval_duration': response.get('prompt_eval_duration', 0),
                    'eval_duration': response.get('eval_duration', 0)
                },
                raw_response=response
            )
        
        except Exception as e:
            raise ModelError(f"Generation failed: {str(e)}") from e
    
    def _generate_stream(
        self,
        prompt: str,
        model: str,
        options: Dict[str, Any],
        start_time: float
    ) -> Generator[str, None, None]:
        """Streaming generation with error handling."""
        try:
            with self._lock:
                stream = self.client.generate(
                    model=model,
                    prompt=prompt,
                    options=options,
                    stream=True
                )
            
            for chunk in stream:
                if 'response' in chunk:
                    yield chunk['response']
            
            # Update performance metrics after streaming completes
            processing_time_ms = (time.time() - start_time) * 1000
            with self._lock:
                self.request_count += 1
                self.total_processing_time_ms += processing_time_ms
                # Note: We can't get token count from streaming without final response
        
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield f"Error: {str(e)}"
    
    def _chat_complete(
        self,
        messages: List[Dict[str, str]],
        model: str,
        options: Dict[str, Any],
        start_time: float
    ) -> LLMResponse:
        """Non-streaming chat completion."""
        try:
            with self._lock:
                response = self.client.chat(
                    model=model,
                    messages=messages,
                    options=options,
                    stream=False
                )
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Update performance metrics
            with self._lock:
                self.request_count += 1
                self.total_processing_time_ms += processing_time_ms
                self.total_tokens_processed += response.get('eval_count', 0)
            
            return LLMResponse(
                content=response['message']['content'],
                model=model,
                prompt_tokens=response.get('prompt_eval_count'),
                completion_tokens=response.get('eval_count'),
                total_tokens=response.get('prompt_eval_count', 0) + response.get('eval_count', 0),
                processing_time_ms=processing_time_ms,
                finish_reason=response.get('done_reason', 'stop'),
                metadata={
                    'total_duration': response.get('total_duration', 0),
                    'load_duration': response.get('load_duration', 0),
                    'prompt_eval_duration': response.get('prompt_eval_duration', 0),
                    'eval_duration': response.get('eval_duration', 0)
                },
                raw_response=response
            )
        
        except Exception as e:
            raise ModelError(f"Chat completion failed: {str(e)}") from e
    
    def _chat_stream(
        self,
        messages: List[Dict[str, str]],
        model: str,
        options: Dict[str, Any],
        start_time: float
    ) -> Generator[str, None, None]:
        """Streaming chat completion."""
        try:
            with self._lock:
                stream = self.client.chat(
                    model=model,
                    messages=messages,
                    options=options,
                    stream=True
                )
            
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
            
            # Update performance metrics after streaming completes
            processing_time_ms = (time.time() - start_time) * 1000
            with self._lock:
                self.request_count += 1
                self.total_processing_time_ms += processing_time_ms
        
        except Exception as e:
            logger.error(f"Streaming chat failed: {e}")
            yield f"Error: {str(e)}"
    
    def switch_model(self, model_name: str) -> bool:
        """
        Switch to a different model with validation.
        
        Args:
            model_name: Name of model to switch to
            
        Returns:
            True if switch successful
        
        Raises:
            ValueError: If model is not supported
        """
        # Check if model is supported
        if model_name not in self.MODEL_DATABASE:
            # Check if it's available from Ollama
            available_models = self._get_cached_models()
            available_model_names = [m.get('name', '') for m in available_models]
            
            if model_name not in available_model_names:
                raise ValueError(
                    f"Model '{model_name}' is not supported and not available from Ollama. "
                    f"Supported models: {', '.join(self.MODEL_DATABASE.keys())}"
                )
        
        # Check if model is available locally
        if not self.is_model_available(model_name):
            logger.warning(f"Model '{model_name}' is not available locally. It may need to be downloaded.")
        
        # Update configuration
        old_model = self.config.model
        self.config.model = model_name
        
        logger.info(f"Switched model from {old_model} to {model_name}")
        return True
    
    def get_available_models(self, refresh_cache: bool = False) -> List[Dict[str, Any]]:
        """
        Get list of available models from Ollama server.
        
        Args:
            refresh_cache: Force refresh of model cache
            
        Returns:
            List of model information dictionaries
        """
        if refresh_cache:
            self._available_models_cache = None
            self._cache_timestamp = None
        
        cached_models = self._get_cached_models()
        
        # Format models with additional information
        formatted_models = []
        for model in cached_models:
            model_name = model.get('name', '')
            model_info = self.MODEL_DATABASE.get(model_name, {})
            
            # Parse size
            size_bytes = model.get('size', 0)
            size_gb = round(size_bytes / 1e9, 2) if size_bytes else 0.0
            
            formatted_models.append({
                'name': model_name,
                'display_name': model_info.get('display_name', model_name),
                'description': model_info.get('description', ''),
                'size_bytes': size_bytes,
                'size_gb': size_gb,
                'modified_at': model.get('modified_at', ''),
                'digest': model.get('digest', ''),
                'is_supported': model_name in self.MODEL_DATABASE,
                'is_current': model_name == self.config.model,
                'context_window': model_info.get('context_window', 4096),
                'requirements': model_info.get('requirements', {}),
                'tags': model_info.get('tags', []),
                'license': model_info.get('license')
            })
        
        return formatted_models
    
    def is_model_available(self, model_name: str) -> bool:
        """
        Check if a specific model is available locally.
        
        Args:
            model_name: Name of the model to check
            
        Returns:
            True if model is available
        """
        available_models = self._get_cached_models()
        available_model_names = [m.get('name', '') for m in available_models]
        return model_name in available_model_names
    
    def get_supported_models(self) -> List[ModelInfo]:
        """
        Get detailed information about all supported models.
        
        Returns:
            List of ModelInfo objects
        """
        available_models = self._get_cached_models()
        available_model_map = {m.get('name', ''): m for m in available_models}
        
        model_infos = []
        for model_id, model_data in self.MODEL_DATABASE.items():
            available_model = available_model_map.get(model_id, {})
            
            model_infos.append(ModelInfo(
                name=model_id,
                display_name=model_data['display_name'],
                description=model_data['description'],
                context_window=model_data['context_window'],
                requirements=model_data['requirements'],
                default_parameters=model_data['default_parameters'],
                tags=model_data['tags'],
                size_gb=round(available_model.get('size', 0) / 1e9, 2) if available_model else 0.0,
                modified_at=available_model.get('modified_at'),
                digest=available_model.get('digest'),
                is_available=model_id in available_model_map,
                is_current=model_id == self.config.model,
                license=model_data.get('license'),
                modelfile=available_model.get('modelfile'),
                template=available_model.get('template')
            ))
        
        return model_infos
    
    def pull_model(
        self,
        model_name: str,
        insecure: bool = False,
        show_progress: bool = True
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Pull/download a model from Ollama repository with progress tracking.
        
        Args:
            model_name: Name of model to pull
            insecure: Allow insecure connections
            show_progress: Whether to show download progress
            
        Yields:
            Progress updates during download
        
        Raises:
            ConnectionError: If not connected to Ollama
            ValueError: If model name is invalid
        """
        if not self.ensure_connection():
            raise ConnectionError("Not connected to Ollama server")
        
        if not model_name or not isinstance(model_name, str):
            raise ValueError("Invalid model name")
        
        logger.info(f"Starting download of model: {model_name}")
        
        try:
            for progress in self.client.pull(
                model=model_name,
                stream=True,
                insecure=insecure
            ):
                yield progress
                
                if show_progress:
                    status = progress.get('status', '')
                    if status:
                        logger.info(f"Download progress: {status}")
                
                if progress.get('status') == 'success':
                    logger.info(f"Successfully downloaded model: {model_name}")
                    break
        
        except Exception as e:
            error_msg = f"Failed to download model {model_name}: {e}"
            logger.error(error_msg)
            yield {'error': str(e), 'status': 'error'}
            raise ModelError(error_msg) from e
    
    def delete_model(self, model_name: str) -> Dict[str, Any]:
        """
        Delete a model from local storage.
        
        Args:
            model_name: Name of model to delete
            
        Returns:
            Dictionary with deletion results
        
        Raises:
            ConnectionError: If not connected to Ollama
            ValueError: If model not found
        """
        if not self.ensure_connection():
            raise ConnectionError("Not connected to Ollama server")
        
        # Check if model exists
        if not self.is_model_available(model_name):
            raise ValueError(f"Model '{model_name}' not found locally")
        
        try:
            with self._lock:
                self.client.delete(model_name)
            
            # Clear cache
            self._available_models_cache = None
            self._cache_timestamp = None
            
            # If deleted model was current, switch to default
            if model_name == self.config.model:
                self.config.model = 'llama2:7b'
                logger.info(f"Switched to default model after deletion: {self.config.model}")
            
            logger.info(f"Deleted model: {model_name}")
            
            return {
                'success': True,
                'model': model_name,
                'message': f"Model '{model_name}' deleted successfully",
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            error_msg = f"Failed to delete model {model_name}: {e}"
            logger.error(error_msg)
            return {
                'success': False,
                'model': model_name,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_name: Model name (uses current if None)
            
        Returns:
            Dictionary with model information
        
        Raises:
            ConnectionError: If not connected to Ollama
        """
        target_model = model_name or self.config.model
        
        if not self.ensure_connection():
            raise ConnectionError("Not connected to Ollama server")
        
        try:
            with self._lock:
                response = self.client.show(target_model)
            
            model_data = self.MODEL_DATABASE.get(target_model, {})
            
            return {
                'success': True,
                'model': target_model,
                'display_name': model_data.get('display_name', target_model),
                'description': model_data.get('description', ''),
                'license': response.get('license', model_data.get('license')),
                'modelfile': response.get('modelfile', ''),
                'parameters': response.get('parameters', ''),
                'template': response.get('template', ''),
                'system': response.get('system', ''),
                'details': response.get('details', {}),
                'context_window': model_data.get('context_window', 4096),
                'requirements': model_data.get('requirements', {}),
                'tags': model_data.get('tags', []),
                'is_supported': target_model in self.MODEL_DATABASE,
                'is_current': target_model == self.config.model,
                'size_bytes': response.get('size', 0),
                'digest': response.get('digest', ''),
                'modified_at': response.get('modified_at', ''),
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            error_msg = f"Failed to get model info for {target_model}: {e}"
            logger.error(error_msg)
            return {
                'success': False,
                'model': target_model,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of LLM client.
        
        Returns:
            Dictionary with health check results
        """
        start_time = time.time()
        
        try:
            # Test connection
            connection_ok = self._test_connection(retry=False)
            
            # Get available models
            available_models = []
            if connection_ok:
                available_models = self.get_available_models(refresh_cache=True)
            
            # Calculate health score (0-100)
            health_score = 0
            if connection_ok:
                health_score = 70  # Base score for connection
                
                # Bonus for having models
                if available_models:
                    health_score += 20
                
                # Bonus if current model is available
                if self.is_model_available(self.config.model):
                    health_score += 10
            
            # Performance metrics
            avg_processing_time = (
                self.total_processing_time_ms / self.request_count 
                if self.request_count > 0 else 0
            )
            
            result = {
                'success': True,
                'status': 'healthy' if health_score >= 80 else 'degraded' if health_score >= 50 else 'unhealthy',
                'health_score': health_score,
                'connected': connection_ok,
                'host': self.config.host,
                'current_model': self.config.model,
                'current_model_available': self.is_model_available(self.config.model),
                'available_models': len(available_models),
                'supported_models': len(self.MODEL_DATABASE),
                'request_count': self.request_count,
                'error_count': self.error_count,
                'total_tokens_processed': self.total_tokens_processed,
                'average_processing_time_ms': avg_processing_time,
                'connection_attempts': self.connection_attempts,
                'last_successful_connection': (
                    self.last_successful_connection.isoformat() 
                    if self.last_successful_connection else None
                ),
                'check_duration_ms': (time.time() - start_time) * 1000,
                'timestamp': datetime.now().isoformat(),
                'recommendations': []
            }
            
            # Generate recommendations
            if not connection_ok:
                result['recommendations'].append("Check Ollama server is running and accessible")
            
            if not result['current_model_available']:
                result['recommendations'].append(
                    f"Download model '{self.config.model}' or switch to an available model"
                )
            
            if result['error_rate'] > 0.1:  # More than 10% errors
                result['recommendations'].append("High error rate detected, check server stability")
            
            return result
        
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'success': False,
                'status': 'unhealthy',
                'health_score': 0,
                'connected': False,
                'error': str(e),
                'check_duration_ms': (time.time() - start_time) * 1000,
                'timestamp': datetime.now().isoformat()
            }
    
    @property
    def error_rate(self) -> float:
        """Calculate current error rate."""
        if self.request_count == 0:
            return 0.0
        return self.error_count / self.request_count
    
    def validate_model_compatibility(
        self, 
        model_name: str, 
        available_ram_gb: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Validate if a model is compatible with available system resources.
        
        Args:
            model_name: Model to validate
            available_ram_gb: Available RAM in gigabytes (auto-detected if None)
            
        Returns:
            Dictionary with compatibility assessment
        """
        if model_name not in self.MODEL_DATABASE:
            return {
                'compatible': False,
                'model': model_name,
                'error': 'Model not supported',
                'timestamp': datetime.now().isoformat()
            }
        
        model_data = self.MODEL_DATABASE[model_name]
        requirements = model_data.get('requirements', {})
        
        # Parse RAM requirement
        ram_required_str = requirements.get('ram', '8GB')
        ram_required_gb = 8.0  # Default
        
        try:
            ram_match = re.search(r'(\d+(\.\d+)?)\s*GB?', ram_required_str, re.IGNORECASE)
            if ram_match:
                ram_required_gb = float(ram_match.group(1))
        except (ValueError, AttributeError):
            logger.warning(f"Could not parse RAM requirement: {ram_required_str}")
        
        # Auto-detect available RAM if not provided
        if available_ram_gb is None:
            try:
                import psutil
                available_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            except ImportError:
                logger.warning("psutil not available, using default 8GB for compatibility check")
                available_ram_gb = 8.0
            except Exception as e:
                logger.warning(f"Failed to detect available RAM: {e}, using 8GB")
                available_ram_gb = 8.0
        
        # Check compatibility
        compatible = available_ram_gb >= ram_required_gb
        ram_utilization = (ram_required_gb / available_ram_gb) * 100 if available_ram_gb > 0 else 100
        
        result = {
            'compatible': compatible,
            'model': model_name,
            'ram_required_gb': ram_required_gb,
            'ram_available_gb': round(available_ram_gb, 2),
            'ram_utilization_percent': round(ram_utilization, 1),
            'requirements': requirements,
            'recommendation': (
                'Compatible' if compatible 
                else f'Requires at least {ram_required_gb}GB RAM, only {available_ram_gb:.1f}GB available'
            ),
            'timestamp': datetime.now().isoformat()
        }
        
        if not compatible:
            logger.warning(
                f"Model {model_name} requires {ram_required_gb}GB RAM, "
                f"only {available_ram_gb:.1f}GB available"
            )
        
        return result
    
    def get_system_prompt(self, prompt_type: str = 'default') -> str:
        """
        Get system prompt for specific use case.
        
        Args:
            prompt_type: Type of system prompt
            
        Returns:
            System prompt text
        """
        system_prompts = {
            'default': """You are DocuBot, a helpful AI assistant that helps users manage and query their documents.
You have access to the user's documents through RAG (Retrieval Augmented Generation).
Always cite your sources when providing information from documents.
Be concise and accurate in your responses.""",
            
            'concise': """You are DocuBot. Provide concise, direct answers based on the documents.
Focus on key information and avoid unnecessary details.""",
            
            'detailed': """You are DocuBot. Provide detailed, answers based on the documents.
Include explanations, context, and relevant details when applicable.""",
            
            'code': """You are DocuBot, specialized in code analysis and programming questions.
Provide technical explanations with code examples when applicable.
Focus on accuracy and best practices.""",
            
            'research': """You are DocuBot in research mode. Provide detailed responses with references, 
citations, and thorough explanations. Include source attribution and methodological details.""",
            
            'summarization': """You are DocuBot in summarization mode. Provide clear, concise summaries 
of documents focusing on key points, main ideas, and important findings.""",
            
            'creative': """You are DocuBot in creative mode. You can be more expressive and imaginative 
while still being factually accurate based on the documents."""
        }
        
        return system_prompts.get(prompt_type, system_prompts['default'])
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration with state."""
        return {
            'config': self.config.to_dict(),
            'state': {
                'connected': self.connected,
                'request_count': self.request_count,
                'error_count': self.error_count,
                'total_tokens_processed': self.total_tokens_processed,
                'total_processing_time_ms': self.total_processing_time_ms,
                'connection_attempts': self.connection_attempts,
                'last_successful_connection': (
                    self.last_successful_connection.isoformat() 
                    if self.last_successful_connection else None
                )
            }
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        avg_time_ms = (
            self.total_processing_time_ms / self.request_count 
            if self.request_count > 0 else 0
        )
        
        return {
            'requests': {
                'total': self.request_count,
                'successful': self.request_count - self.error_count,
                'failed': self.error_count,
                'error_rate': self.error_rate
            },
            'tokens': {
                'total_processed': self.total_tokens_processed,
                'average_per_request': (
                    self.total_tokens_processed / self.request_count 
                    if self.request_count > 0 else 0
                )
            },
            'timing': {
                'total_processing_time_ms': self.total_processing_time_ms,
                'average_processing_time_ms': avg_time_ms,
                'requests_per_second': (
                    1000 / avg_time_ms if avg_time_ms > 0 else 0
                )
            },
            'connection': {
                'connected': self.connected,
                'attempts': self.connection_attempts,
                'last_successful': (
                    self.last_successful_connection.isoformat() 
                    if self.last_successful_connection else None
                )
            },
            'models': {
                'current': self.config.model,
                'available': len(self._get_cached_models()),
                'supported': len(self.MODEL_DATABASE)
            },
            'timestamp': datetime.now().isoformat()
        }
    
    def reset_stats(self) -> None:
        """Reset performance statistics."""
        with self._lock:
            self.request_count = 0
            self.total_tokens_processed = 0
            self.total_processing_time_ms = 0.0
            self.error_count = 0
            self.connection_attempts = 0
        
        logger.info("Performance statistics reset")


# Factory functions for different use cases
def create_llm_client(config: Optional[Dict[str, Any]] = None) -> LLMClient:
    """
    Create LLM client with optional configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        LLMClient instance
    """
    try:
        if config:
            llm_config = LLMConfig.from_dict(config)
            return LLMClient(llm_config)
        else:
            return LLMClient()
    
    except Exception as e:
        logger.error(f"Error creating LLMClient: {e}")
        raise


def get_llm_client_from_app_config() -> LLMClient:
    """
    Get LLM client configured from DocuBot application config.
    
    Returns:
        LLMClient instance
    """
    return LLMClient()


# Global singleton instance with lazy initialization
_global_llm_instance = None
_global_llm_lock = threading.Lock()

def get_global_llm_client() -> LLMClient:
    """
    Get global singleton LLM client instance.
    
    Returns:
        LLMClient instance
    """
    global _global_llm_instance
    
    with _global_llm_lock:
        if _global_llm_instance is None:
            _global_llm_instance = get_llm_client_from_app_config()
    
    return _global_llm_instance


def validate_llm_environment() -> Dict[str, Any]:
    """
    Validate LLM environment and dependencies.
    
    Returns:
        Dictionary with validation results
    """
    results = {
        'ollama_available': OLLAMA_AVAILABLE,
        'httpx_available': HTTPX_AVAILABLE,
        'core_config_available': False,
        'environment_valid': False,
        'errors': [],
        'warnings': [],
        'timestamp': datetime.now().isoformat()
    }
    
    # Check core config
    try:
        from core.config import get_config
        results['core_config_available'] = True
    except ImportError:
        results['warnings'].append("core.config module not available")
    except Exception as e:
        results['warnings'].append(f"Error checking core.config: {e}")
    
    # Check Ollama version
    if OLLAMA_AVAILABLE:
        try:
            results['ollama_version'] = getattr(ollama, '__version__', 'unknown')
        except Exception as e:
            results['warnings'].append(f"Error getting ollama version: {e}")
    
    # Check HTTPX version
    if HTTPX_AVAILABLE:
        try:
            results['httpx_version'] = getattr(httpx, '__version__', 'unknown')
        except Exception as e:
            results['warnings'].append(f"Error getting httpx version: {e}")
    
    # Overall validation
    results['environment_valid'] = OLLAMA_AVAILABLE
    
    # Check Ollama server
    if OLLAMA_AVAILABLE:
        try:
            client = Client()
            response = client.list()
            results['ollama_server_running'] = True
            results['available_models'] = len(response.get('models', []))
        except Exception as e:
            results['errors'].append(f"Ollama server not accessible: {e}")
            results['ollama_server_running'] = False
    
    return results


def check_ollama_installation() -> Dict[str, Any]:
    """
    Check Ollama installation and version.
    
    Returns:
        Dictionary with installation check results
    """
    result = {
        'installed': False,
        'version': None,
        'path': None,
        'error': None,
        'timestamp': datetime.now().isoformat()
    }
    
    try:
        # Try to run ollama command
        process = subprocess.run(
            ['ollama', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if process.returncode == 0:
            result['installed'] = True
            result['version'] = process.stdout.strip()
            
            # Try to find ollama path
            try:
                ollama_path = subprocess.check_output(
                    ['which', 'ollama'] if os.name != 'nt' else ['where', 'ollama'],
                    text=True
                ).strip()
                result['path'] = ollama_path
            except Exception:
                pass  # Path detection is optional
        else:
            result['error'] = process.stderr.strip()
    
    except FileNotFoundError:
        result['error'] = "Ollama command not found"
    except subprocess.TimeoutExpired:
        result['error'] = "Ollama command timed out"
    except Exception as e:
        result['error'] = str(e)
    
    return result


# Test module
if __name__ == "__main__":
    """
    test for the LLM client implementation.
    """
    import sys
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 80)
    print("DOCUBOT LLM CLIENT - TEST")
    print("=" * 80)
    
    # 1. Environment validation
    print("\n1. ENVIRONMENT VALIDATION")
    print("-" * 40)
    env = validate_llm_environment()
    
    for key, value in env.items():
        if key not in ['errors', 'warnings', 'timestamp']:
            status = "✓" if value else "✗"
            if isinstance(value, bool):
                print(f"  {status} {key}")
            else:
                print(f"  {status} {key}: {value}")
    
    if env.get('errors'):
        print("\n  Errors:")
        for error in env['errors']:
            print(f"    ✗ {error}")
    
    if env.get('warnings'):
        print("\n  Warnings:")
        for warning in env['warnings']:
            print(f"    ⚠ {warning}")
    
    if not env['environment_valid']:
        print("\n  Environment validation failed!")
        sys.exit(1)
    
    # 2. Ollama installation check
    print("\n2. OLLAMA INSTALLATION CHECK")
    print("-" * 40)
    ollama_check = check_ollama_installation()
    
    if ollama_check['installed']:
        print(f"  ✓ Ollama installed: {ollama_check['version']}")
        if ollama_check['path']:
            print(f"  Path: {ollama_check['path']}")
    else:
        print(f"  ✗ Ollama not installed: {ollama_check.get('error', 'Unknown error')}")
        print("  Please install Ollama from: https://ollama.ai/")
        sys.exit(1)
    
    # 3. Client creation
    print("\n3. CLIENT INITIALIZATION")
    print("-" * 40)
    
    try:
        client = get_global_llm_client()
        config = client.get_config()
        
        print(f"  Model: {config['config']['model']}")
        print(f"  Host: {config['config']['host']}")
        print(f"  Connected: {config['state']['connected']}")
        print(f"  Temperature: {config['config']['temperature']}")
        print(f"  Max Tokens: {config['config']['max_tokens']}")
        
    except Exception as e:
        print(f"  ✗ Client initialization failed: {e}")
        sys.exit(1)
    
    # 4. Health check
    print("\n4. HEALTH CHECK")
    print("-" * 40)
    
    try:
        health = client.health_check()
        
        print(f"  Status: {health['status'].upper()}")
        print(f"  Health Score: {health['health_score']}/100")
        print(f"  Available Models: {health['available_models']}")
        print(f"  Current Model Available: {health['current_model_available']}")
        
        if health['recommendations']:
            print("\n  Recommendations:")
            for rec in health['recommendations']:
                print(f"    • {rec}")
    
    except Exception as e:
        print(f"  ✗ Health check failed: {e}")
    
    # 5. Model information
    print("\n5. MODEL INFORMATION")
    print("-" * 40)
    
    try:
        models = client.get_supported_models()
        available_models = [m for m in models if m.is_available]
        
        print(f"  Supported Models: {len(models)}")
        print(f"  Available Models: {len(available_models)}")
        
        print("\n  Available Models:")
        for model in available_models[:5]:  # Show first 5
            current = " [CURRENT]" if model.is_current else ""
            print(f"    • {model.display_name}{current} ({model.size_gb}GB)")
        
        if len(available_models) > 5:
            print(f"    ... and {len(available_models) - 5} more")
    
    except Exception as e:
        print(f"  ✗ Failed to get model information: {e}")
    
    # 6. Test generation (if connected and model available)
    print("\n6. TEST GENERATION")
    print("-" * 40)
    
    if client.connected and client.is_model_available(client.config.model):
        test_prompts = [
            "What is artificial intelligence in one sentence?",
            "Explain the concept of machine learning briefly."
        ]
        
        for i, prompt in enumerate(test_prompts[:2], 1):
            print(f"\n  Test {i}:")
            print(f"    Prompt: {prompt}")
            
            try:
                response = client.generate(prompt, stream=False, max_retries=1)
                
                if isinstance(response, LLMResponse):
                    print(f"    Response: {response.content[:100]}...")
                    print(f"    Tokens: {response.total_tokens}")
                    print(f"    Time: {response.processing_time_ms:.0f}ms")
                else:
                    print(f"    Unexpected response type: {type(response)}")
            
            except Exception as e:
                print(f"    Error: {e}")
    else:
        print("  Skipping generation tests (not connected or model not available)")
    
    # 7. Performance metrics
    print("\n7. PERFORMANCE METRICS")
    print("-" * 40)
    
    try:
        metrics = client.get_performance_metrics()
        
        print(f"  Total Requests: {metrics['requests']['total']}")
        print(f"  Successful: {metrics['requests']['successful']}")
        print(f"  Failed: {metrics['requests']['failed']}")
        print(f"  Error Rate: {metrics['requests']['error_rate']:.1%}")
        print(f"  Total Tokens: {metrics['tokens']['total_processed']}")
        print(f"  Avg Processing Time: {metrics['timing']['average_processing_time_ms']:.0f}ms")
    
    except Exception as e:
        print(f"  ✗ Failed to get performance metrics: {e}")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)