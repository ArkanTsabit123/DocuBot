"""
DocuBot Custom Exceptions and Error Handling System
"""

import logging
import sys
import uuid
import socket
from typing import Dict, Any, Optional, Callable, TypeVar, Union
from pathlib import Path
from datetime import datetime

T = TypeVar('T')
R = TypeVar('R')

class ErrorCodes:
    """Error code constants for consistent error reporting."""
    
    GENERIC = "DB_ERROR_000"
    
    CONFIG_GENERIC = "DB_ERROR_100"
    
    DOCUMENT_GENERIC = "DB_ERROR_200"
    EXTRACTION = "DB_ERROR_210"
    UNSUPPORTED_FORMAT = "DB_ERROR_211"
    PROCESSING = "DB_ERROR_220"
    CHUNKING = "DB_ERROR_221"
    
    DATABASE_GENERIC = "DB_ERROR_300"
    
    VECTOR_STORE = "DB_ERROR_400"
    
    AI_GENERIC = "DB_ERROR_500"
    LLM = "DB_ERROR_510"
    EMBEDDING = "DB_ERROR_520"
    RAG = "DB_ERROR_530"
    MODEL_MANAGEMENT = "DB_ERROR_540"
    RATE_LIMIT = "DB_ERROR_511"
    CONTEXT_LENGTH = "DB_ERROR_512"
    
    UI = "DB_ERROR_600"
    
    VALIDATION = "DB_ERROR_700"
    
    RESOURCE = "DB_ERROR_800"
    
    DIAGNOSTIC = "DB_ERROR_900"
    
    AUTHENTICATION = "DB_ERROR_1000"


class DocuBotError(Exception):
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}
        self.timestamp = datetime.now().isoformat()
        self.error_code = ErrorCodes.GENERIC

    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': type(self).__name__,
            'message': str(self),
            'error_code': self.error_code,
            'context': self.context,
            'timestamp': self.timestamp
        }


class ConfigurationError(DocuBotError):
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(message, {"config_key": config_key})
        self.error_code = ErrorCodes.CONFIG_GENERIC


class DocumentError(DocuBotError):
    def __init__(self, message: str, file_path: Optional[str] = None):
        context = {"file_path": file_path} if file_path else {}
        super().__init__(message, context)
        self.error_code = ErrorCodes.DOCUMENT_GENERIC


class ExtractionError(DocumentError):
    def __init__(self, message: str, file_path: Optional[str] = None, 
                 format_type: Optional[str] = None):
        context = {}
        if file_path is not None:
            context['file_path'] = file_path
        if format_type is not None:
            context['format_type'] = format_type
        super().__init__(message, context)
        self.error_code = ErrorCodes.EXTRACTION


class UnsupportedFormatError(ExtractionError):
    def __init__(self, format_type: str, file_path: Optional[str] = None):
        message = f"Unsupported document format: {format_type}"
        context = {"format_type": format_type}
        if file_path is not None:
            context['file_path'] = file_path
        super().__init__(message, file_path, format_type)
        self.error_code = ErrorCodes.UNSUPPORTED_FORMAT


class ProcessingError(DocumentError):
    def __init__(self, message: str, file_path: Optional[str] = None, 
                 stage: Optional[str] = None):
        context = {}
        if file_path is not None:
            context['file_path'] = file_path
        if stage is not None:
            context['stage'] = stage
        super().__init__(message, context)
        self.error_code = ErrorCodes.PROCESSING


class ChunkingError(ProcessingError):
    def __init__(self, message: str, file_path: Optional[str] = None):
        super().__init__(message, file_path, "chunking")
        self.error_code = ErrorCodes.CHUNKING


class DatabaseError(DocuBotError):
    def __init__(self, message: str, operation: Optional[str] = None,
                 query: Optional[str] = None):
        context = {}
        if operation is not None:
            context['operation'] = operation
        if query is not None:
            context['query'] = query
        super().__init__(message, context)
        self.error_code = ErrorCodes.DATABASE_GENERIC


class VectorStoreError(DocuBotError):
    def __init__(self, message: str, operation: Optional[str] = None):
        context = {"operation": operation} if operation is not None else {}
        super().__init__(message, context)
        self.error_code = ErrorCodes.VECTOR_STORE


class AIError(DocuBotError):
    def __init__(self, message: str, model: Optional[str] = None):
        context = {"model": model} if model is not None else {}
        super().__init__(message, context)
        self.error_code = ErrorCodes.AI_GENERIC


class LLMError(AIError):
    def __init__(self, message: str, model: Optional[str] = None,
                 prompt: Optional[str] = None):
        context = {}
        if model is not None:
            context['model'] = model
        if prompt is not None:
            context['prompt'] = prompt
        super().__init__(message, context)
        self.error_code = ErrorCodes.LLM


class EmbeddingError(AIError):
    def __init__(self, message: str, model: Optional[str] = None,
                 text_length: Optional[int] = None):
        context = {}
        if model is not None:
            context['model'] = model
        if text_length is not None:
            context['text_length'] = text_length
        super().__init__(message, context)
        self.error_code = ErrorCodes.EMBEDDING


class RAGError(AIError):
    def __init__(self, message: str, stage: Optional[str] = None,
                 context_size: Optional[int] = None):
        context = {}
        if stage is not None:
            context['stage'] = stage
        if context_size is not None:
            context['context_size'] = context_size
        super().__init__(message, context)
        self.error_code = ErrorCodes.RAG


class ModelManagementError(AIError):
    def __init__(self, message: str, model_name: Optional[str] = None,
                 action: Optional[str] = None):
        context = {}
        if model_name is not None:
            context['model_name'] = model_name
        if action is not None:
            context['action'] = action
        super().__init__(message, context)
        self.error_code = ErrorCodes.MODEL_MANAGEMENT


class RateLimitError(AIError):
    def __init__(self, message: str, model: Optional[str] = None,
                 reset_time: Optional[float] = None):
        context = {}
        if model is not None:
            context['model'] = model
        if reset_time is not None:
            context['reset_time'] = reset_time
        super().__init__(message, context)
        self.error_code = ErrorCodes.RATE_LIMIT


class ContextLengthError(LLMError):
    def __init__(self, message: str, model: Optional[str] = None,
                 current_length: Optional[int] = None,
                 max_length: Optional[int] = None):
        context = {}
        if model is not None:
            context['model'] = model
        if current_length is not None:
            context['current_length'] = current_length
        if max_length is not None:
            context['max_length'] = max_length
        super().__init__(message, context)
        self.error_code = ErrorCodes.CONTEXT_LENGTH


class UIError(DocuBotError):
    def __init__(self, message: str, component: Optional[str] = None):
        context = {"component": component} if component is not None else {}
        super().__init__(message, context)
        self.error_code = ErrorCodes.UI


class ValidationError(DocuBotError):
    def __init__(self, message: str, field: Optional[str] = None,
                 value: Optional[Any] = None):
        context = {}
        if field is not None:
            context['field'] = field
        if value is not None:
            context['value'] = value
        super().__init__(message, context)
        self.error_code = ErrorCodes.VALIDATION


class ResourceError(DocuBotError):
    def __init__(self, message: str, resource_type: Optional[str] = None,
                 required: Optional[Any] = None, available: Optional[Any] = None):
        context = {}
        if resource_type is not None:
            context['resource_type'] = resource_type
        if required is not None:
            context['required'] = required
        if available is not None:
            context['available'] = available
        super().__init__(message, context)
        self.error_code = ErrorCodes.RESOURCE


class AuthenticationError(DocuBotError):
    def __init__(self, message: str, resource: Optional[str] = None):
        context = {"resource": resource} if resource else {}
        super().__init__(message, context)
        self.error_code = ErrorCodes.AUTHENTICATION


class DiagnosticError(DocuBotError):
    def __init__(self, message: str, check_name: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        context = {}
        if check_name is not None:
            context['check_name'] = check_name
        if details:
            context.update(details)
        super().__init__(message, context)
        self.error_code = ErrorCodes.DIAGNOSTIC


def handle_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    error_type = type(error).__name__
    error_context = context or {}
    
    user_messages = {
        'FileNotFoundError': "The requested file was not found. Please verify the file path exists.",
        'PermissionError': "Permission denied. Please check file permissions and access rights.",
        'UnsupportedFormatError': "The document format is not supported. Please convert to a supported format.",
        'LLMError': "Unable to generate response from AI model. Please try again or check model availability.",
        'DatabaseError': "Database operation failed. Please check database connection and integrity.",
        'ValidationError': "Input validation failed. Please check the provided data.",
        'ResourceError': "Insufficient system resources. Please check available memory and disk space.",
        'ConfigurationError': "Configuration error detected. Please verify configuration files.",
        'DiagnosticError': "System diagnostic check failed. Please run the diagnostic tool for details.",
        'ValueError': "Invalid value provided. Please check your input.",
        'TypeError': "Type error occurred. Please check your input types.",
        'KeyError': "Required key not found. Please check your data structure.",
        'AttributeError': "Attribute not found. Please check your object structure.",
        'ImportError': "Module import failed. Please check your installation.",
        'OSError': "Operating system error occurred. Please check file paths and permissions.",
        'MemoryError': "Insufficient memory. Please close other applications and try again.",
        'TimeoutError': "Operation timed out. Please try again or check your connection.",
        'ConnectionError': "Connection failed. Please check your network connection.",
        'RuntimeError': "Runtime error occurred. Please try again or contact support.",
        'AuthenticationError': "Authentication failed. Please check your credentials and permissions.",
        'RateLimitError': "Rate limit exceeded. Please wait before making additional requests.",
        'ContextLengthError': "Context length exceeded. Please reduce the input size or use a model with larger context window.",
    }
    
    if isinstance(error, DocuBotError):
        error_dict = {
            'success': False,
            'error_type': error_type,
            'error_code': error.error_code,
            'error_message': str(error),
            'user_message': user_messages.get(error_type, str(error)),
            'context': {**error_context, **error.context},
            'timestamp': error.timestamp,
            'is_docubot_error': True
        }
    else:
        error_dict = {
            'success': False,
            'error_type': error_type,
            'error_code': ErrorCodes.GENERIC,
            'error_message': str(error),
            'user_message': user_messages.get(error_type, "An unexpected error occurred. Please try again."),
            'context': error_context,
            'timestamp': datetime.now().isoformat(),
            'is_docubot_error': False
        }
    
    return error_dict


def safe_execute(func: Callable[..., T], *args: Any, **kwargs: Any) -> tuple[bool, Union[T, Dict[str, Any]]]:
    try:
        result = func(*args, **kwargs)
        return True, result
    except Exception as error:
        error_context = {
            'function': func.__name__,
            'module': func.__module__ if hasattr(func, '__module__') else 'unknown',
            'args': str(args)[:200],
            'kwargs': str(kwargs)[:200]
        }
        error_result = handle_error(error, error_context)
        return False, error_result


class ErrorContext:
    def __init__(self, operation: str, logger: Optional[logging.Logger] = None,
                 raise_on_error: bool = True, collect_metrics: bool = True):
        self.operation = operation
        self.logger = logger or logging.getLogger(__name__)
        self.raise_on_error = raise_on_error
        self.collect_metrics = collect_metrics
        self.error_occurred = False
        self.error_info = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_val is not None:
            self.error_occurred = True
            error_info = handle_error(exc_val, {"operation": self.operation})
            self.error_info = error_info
            
            log_error_with_context(
                self.logger, 
                exc_val,
                {"operation": self.operation},
                level="ERROR" if self.raise_on_error else "WARNING"
            )
            
            if self.collect_metrics and hasattr(self.logger, 'metrics'):
                self.logger.metrics.record_error(exc_val)
            
            return not self.raise_on_error
        return False


class GracefulDegradation:
    @staticmethod
    def fallback_document_processing(error: Exception, file_path: str, **kwargs) -> Dict[str, Any]:
        return {
            'success': False,
            'text': f"Document processing failed: {str(error)[:100]}",
            'metadata': {
                'file_name': Path(file_path).name,
                'file_path': file_path,
                'error': str(error),
                'error_type': type(error).__name__,
                'degraded_mode': True,
                'fallback_timestamp': datetime.now().isoformat()
            },
            'chunks': [],
            'fallback_used': True,
            'original_error': str(error)
        }
    
    @staticmethod
    def fallback_llm_response(error: Exception, query: str, **kwargs) -> Dict[str, Any]:
        return {
            'answer': f"I apologize, but I'm unable to process your query at the moment. The system encountered an error: {type(error).__name__}",
            'sources': [],
            'confidence': 0.0,
            'degraded_mode': True,
            'fallback_used': True,
            'error': str(error),
            'query': query[:500],
            'timestamp': datetime.now().isoformat()
        }
    
    @staticmethod
    def fallback_embedding(error: Exception, text: str, dimensions: int = 384, **kwargs) -> list[float]:
        return [0.0] * dimensions
    
    @staticmethod
    def fallback_database_operation(error: Exception, operation: str, **kwargs) -> Dict[str, Any]:
        return {
            'success': False,
            'operation': operation,
            'error': str(error),
            'error_type': type(error).__name__,
            'degraded_mode': True,
            'fallback_used': True,
            'data': None,
            'timestamp': datetime.now().isoformat()
        }
    
    @staticmethod
    def fallback_vector_store_operation(error: Exception, operation: str, **kwargs) -> Dict[str, Any]:
        return {
            'success': False,
            'operation': operation,
            'error': str(error),
            'degraded_mode': True,
            'results': [],
            'fallback_used': True,
            'timestamp': datetime.now().isoformat()
        }


class FallbackRegistry:
    _strategies = {}
    
    @classmethod
    def register(cls, error_type: type, fallback_func: Callable):
        cls._strategies[error_type] = fallback_func
    
    @classmethod
    def get_fallback(cls, error: Exception):
        fallback = cls._strategies.get(type(error))
        if fallback:
            return fallback
        
        for error_type, func in cls._strategies.items():
            if isinstance(error, error_type):
                return func
        
        return GracefulDegradation.fallback_llm_response


FallbackRegistry.register(LLMError, GracefulDegradation.fallback_llm_response)
FallbackRegistry.register(EmbeddingError, GracefulDegradation.fallback_embedding)
FallbackRegistry.register(DatabaseError, GracefulDegradation.fallback_database_operation)
FallbackRegistry.register(VectorStoreError, GracefulDegradation.fallback_vector_store_operation)
FallbackRegistry.register(ExtractionError, GracefulDegradation.fallback_document_processing)


def graceful_execute(
    primary_func: Callable[..., R], 
    fallback_func: Optional[Callable[..., R]] = None, 
    *args: Any, 
    **kwargs: Any
) -> R:
    logger = logging.getLogger(__name__)
    
    try:
        result = primary_func(*args, **kwargs)
        return result
    except Exception as error:
        if fallback_func is None:
            fallback_func = FallbackRegistry.get_fallback(error)
        
        logger.warning(
            f"Primary function {primary_func.__name__} failed, using fallback: {error}",
            extra={
                'function': primary_func.__name__,
                'error_type': type(error).__name__,
                'error_message': str(error)
            }
        )
        
        try:
            return fallback_func(error, *args, **kwargs)
        except Exception as fallback_error:
            logger.error(
                f"Fallback function also failed: {fallback_error}",
                extra={
                    'function': fallback_func.__name__ if hasattr(fallback_func, '__name__') else 'unknown',
                    'error_type': type(fallback_error).__name__,
                    'error_message': str(fallback_error)
                }
            )
            
            return {
                'success': False,
                'error': str(error),
                'fallback_error': str(fallback_error),
                'degraded_mode': True,
                'critical_failure': True,
                'timestamp': datetime.now().isoformat()
            }


class ConfigurationValidator:
    @staticmethod
    def validate_app_config(config: Dict[str, Any]) -> tuple[bool, list[str]]:
        if not config:
            return False, ["Configuration is empty"]
        
        required_keys = ['app', 'document_processing', 'ai', 'ui', 'storage']
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            return False, [f"Missing required section: {key}" for key in missing_keys]
        
        issues = []
        
        if 'ai' in config:
            ai_config = config['ai']
            if not isinstance(ai_config, dict):
                issues.append("'ai' section must be a dictionary")
            else:
                if 'llm' not in ai_config:
                    issues.append("Missing 'ai.llm' configuration")
                if 'embeddings' not in ai_config:
                    issues.append("Missing 'ai.embeddings' configuration")
        
        if 'document_processing' in config:
            doc_config = config['document_processing']
            if not isinstance(doc_config, dict):
                issues.append("'document_processing' section must be a dictionary")
            else:
                if 'chunk_size' not in doc_config:
                    issues.append("Missing 'document_processing.chunk_size'")
                elif not isinstance(doc_config['chunk_size'], int) or doc_config['chunk_size'] <= 0:
                    issues.append("'document_processing.chunk_size' must be a positive integer")
                
                if 'supported_formats' not in doc_config:
                    issues.append("Missing 'document_processing.supported_formats'")
                elif not isinstance(doc_config['supported_formats'], list):
                    issues.append("'document_processing.supported_formats' must be a list")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def validate_file_path(path: Path) -> tuple[bool, Optional[str]]:
        if not isinstance(path, Path):
            path = Path(str(path))
        
        if not path.exists():
            return False, f"File does not exist: {path}"
        
        if not path.is_file():
            return False, f"Path is not a file: {path}"
        
        try:
            with open(path, 'rb') as f:
                f.read(1)
            return True, None
        except PermissionError:
            return False, f"Permission denied: {path}"
        except Exception as error:
            return False, f"Cannot access file: {error}"
    
    @staticmethod
    def validate_directory(path: Path) -> tuple[bool, Optional[str]]:
        if not isinstance(path, Path):
            path = Path(str(path))
        
        if not path.exists():
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception as error:
                return False, f"Cannot create directory: {error}"
        
        if not path.is_dir():
            return False, f"Path is not a directory: {path}"
        
        test_file = path / ".write_test"
        try:
            test_file.write_text("test")
            test_file.unlink()
            return True, None
        except Exception as error:
            return False, f"Cannot write to directory: {error}"
    
    @staticmethod
    def validate_model_config(config: Dict[str, Any]) -> tuple[bool, list[str]]:
        if not config:
            return False, ["Model configuration is empty"]
        
        issues = []
        
        if 'models' not in config:
            issues.append("Missing 'models' section")
        elif not isinstance(config['models'], dict):
            issues.append("'models' must be a dictionary")
        else:
            for model_name, model_config in config.get('models', {}).items():
                if not isinstance(model_config, dict):
                    issues.append(f"Model '{model_name}' configuration must be a dictionary")
                    continue
                
                required_fields = ['name', 'display_name', 'context_window']
                for field in required_fields:
                    if field not in model_config:
                        issues.append(f"Model '{model_name}' missing required field: '{field}'")
        
        return len(issues) == 0, issues


def setup_error_logging(log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("docubot.errors")
    
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        logger.addHandler(console_handler)
    
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    logger.propagate = False
    
    return logger


def log_error_with_context(
    logger: logging.Logger,
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    level: str = "ERROR"
) -> None:
    log_context = context or {}
    error_info = {
        'error_type': type(error).__name__,
        'error_message': str(error),
        'timestamp': datetime.now().isoformat(),
        **log_context
    }
    
    if isinstance(error, DocuBotError):
        error_info.update({
            'error_code': error.error_code,
            'docubot_context': error.context
        })
    
    log_method = getattr(logger, level.lower(), logger.error)
    log_method(f"Error occurred: {type(error).__name__}", extra=error_info)


class ErrorRecovery:
    @staticmethod
    def recover_from_database_error(error: Exception, max_retries: int = 3) -> tuple[bool, str]:
        logger = logging.getLogger(__name__)
        
        error_msg = str(error).lower()
        
        if "disk full" in error_msg or "readonly" in error_msg:
            logger.error("Cannot recover from disk full or readonly error")
            return False, "Disk full or readonly database"
        
        if "connection" in error_msg or "locked" in error_msg:
            logger.warning("Attempting to recover from connection/locked error")
            return True, "Connection issue detected, attempting recovery"
        
        if "syntax" in error_msg or "table" in error_msg:
            logger.error("Cannot recover from SQL syntax or table error")
            return False, "SQL syntax or table error requires manual fix"
        
        return False, "No recovery strategy available for this error type"
    
    @staticmethod
    def recover_from_resource_error(error: ResourceError) -> tuple[bool, str]:
        resource_type = error.context.get('resource_type', 'unknown')
        
        if resource_type == 'memory':
            return True, "Attempting memory cleanup and retry"
        elif resource_type == 'disk':
            return False, "Disk space requires manual intervention"
        elif resource_type == 'gpu':
            return True, "Falling back to CPU mode"
        elif resource_type == 'cpu':
            return True, "Reducing parallel processing"
        
        return False, "No recovery strategy available"
    
    @staticmethod
    def should_retry(error: Exception, attempt: int, max_attempts: int = 3) -> bool:
        if attempt >= max_attempts:
            return False
        
        no_retry_errors = [
            'UnsupportedFormatError',
            'ValidationError',
            'ConfigurationError',
            'MemoryError',
            'SyntaxError',
            'AttributeError',
            'TypeError',
            'KeyError'
        ]
        
        error_type = type(error).__name__
        if error_type in no_retry_errors:
            return False
        
        if isinstance(error, PermissionError):
            return False
        
        error_msg = str(error).lower()
        if "disk full" in error_msg:
            return False
        
        if "unsupported" in error_msg or "invalid format" in error_msg:
            return False
        
        return True
    
    @staticmethod
    def get_retry_delay(attempt: int, base_delay: float = 1.0) -> float:
        return base_delay * (2 ** (attempt - 1))


class ErrorMetrics:
    def __init__(self):
        self.error_counts = {}
        self.error_timestamps = []
        self.recovery_attempts = 0
        self.recovery_successes = 0
        
    def record_error(self, error: Exception) -> None:
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        self.error_timestamps.append(datetime.now().isoformat())
    
    def record_recovery(self, success: bool) -> None:
        self.recovery_attempts += 1
        if success:
            self.recovery_successes += 1
    
    def get_error_stats(self) -> Dict[str, Any]:
        total_errors = sum(self.error_counts.values())
        
        return {
            'total_errors': total_errors,
            'error_counts': self.error_counts,
            'recovery_rate': (self.recovery_successes / self.recovery_attempts * 100) 
                            if self.recovery_attempts > 0 else 0,
            'recovery_attempts': self.recovery_attempts,
            'recovery_successes': self.recovery_successes,
            'error_timestamps_count': len(self.error_timestamps),
            'most_common_error': max(self.error_counts.items(), key=lambda x: x[1]) 
                                if self.error_counts else None
        }
    
    def get_error_frequency(self, window_minutes: int = 60) -> float:
        if not self.error_timestamps:
            return 0.0
        
        window_start = datetime.now().timestamp() - (window_minutes * 60)
        recent_errors = 0
        
        for timestamp in self.error_timestamps:
            try:
                error_time = datetime.fromisoformat(timestamp.replace('Z', '+00:00')).timestamp()
                if error_time > window_start:
                    recent_errors += 1
            except:
                continue
        
        return recent_errors / window_minutes if window_minutes > 0 else 0.0


class ErrorClassifier:
    @staticmethod
    def classify_error(error: Exception) -> Dict[str, bool]:
        error_type = type(error).__name__
        error_msg = str(error).lower()
        
        classification = {
            'is_connection_error': False,
            'is_resource_error': False,
            'is_configuration_error': False,
            'is_user_error': False,
            'is_system_error': False,
            'is_temporary': False,
            'is_permanent': False,
            'requires_user_intervention': False,
            'can_retry': True,
            'severity': 'medium'
        }
        
        if any(term in error_msg for term in ['connection', 'timeout', 'network', 'socket']):
            classification['is_connection_error'] = True
            classification['is_temporary'] = True
            classification['can_retry'] = True
            classification['severity'] = 'medium'
        
        elif any(term in error_msg for term in ['memory', 'disk', 'space', 'resource', 'capacity']):
            classification['is_resource_error'] = True
            classification['requires_user_intervention'] = True
            classification['can_retry'] = False
            classification['severity'] = 'high'
        
        elif any(term in error_msg for term in ['config', 'setting', 'parameter', 'option']):
            classification['is_configuration_error'] = True
            classification['requires_user_intervention'] = True
            classification['can_retry'] = False
            classification['severity'] = 'medium'
        
        elif any(term in error_msg for term in ['invalid', 'unsupported', 'missing', 'required']):
            classification['is_user_error'] = True
            classification['requires_user_intervention'] = True
            classification['can_retry'] = False
            classification['severity'] = 'low'
        
        elif error_type in ['OSError', 'IOError', 'MemoryError', 'RuntimeError']:
            classification['is_system_error'] = True
            classification['is_permanent'] = True
            classification['requires_user_intervention'] = True
            classification['can_retry'] = False
            classification['severity'] = 'critical'
        
        elif error_type == 'PermissionError':
            classification['is_system_error'] = True
            classification['requires_user_intervention'] = True
            classification['can_retry'] = False
            classification['severity'] = 'high'
        
        return classification
    
    @staticmethod
    def get_recommended_action(error: Exception) -> str:
        classification = ErrorClassifier.classify_error(error)
        
        if classification['is_connection_error']:
            return "Check your network connection and try again"
        elif classification['is_resource_error']:
            return "Free up system resources (memory/disk) and try again"
        elif classification['is_configuration_error']:
            return "Check and update your configuration settings"
        elif classification['is_user_error']:
            return "Check your input and try again with valid data"
        elif classification['requires_user_intervention']:
            return "Manual intervention required. Please contact support if problem persists"
        elif classification['can_retry']:
            return "Temporary issue. Please try again in a few moments"
        else:
            return "Please try again or contact support if the problem persists"


class ErrorReporter:
    @staticmethod
    def report_to_sentry(error: Exception, context: Dict[str, Any]):
        try:
            import sentry_sdk
            with sentry_sdk.push_scope() as scope:
                for key, value in context.items():
                    scope.set_extra(key, value)
                sentry_sdk.capture_exception(error)
        except ImportError:
            pass
    
    @staticmethod
    def create_error_report(error_info: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "error_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "application": "DocuBot",
            "version": "1.0.0",
            "environment": "production",
            "error_details": error_info,
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform,
                "hostname": socket.gethostname() if hasattr(socket, 'gethostname') else "unknown"
            }
        }


if __name__ == "__main__":
    test_logger = setup_error_logging("DEBUG")
    
    print("Testing DocuBot Exception System")
    print("=" * 50)
    
    print("\n1. Testing custom exceptions:")
    try:
        test_exception = UnsupportedFormatError("xyz", "/path/to/file.xyz")
        print(f"   Exception created: {test_exception}")
        print(f"   Error code: {test_exception.error_code}")
        print(f"   Context: {test_exception.context}")
        print(f"   Message: {str(test_exception)}")
    except Exception as e:
        print(f"   Exception creation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n2. Testing error handling:")
    try:
        success, result = safe_execute(lambda x: 1 / x, 0)
        print(f"   Error handling worked: success={success}")
        print(f"   Error result type: {type(result)}")
        print(f"   Error keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
    except Exception as e:
        print(f"   Error handling failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n3. Testing graceful degradation:")
    try:
        def failing_function():
            raise ValueError("This function always fails")
        
        result = graceful_execute(
            failing_function,
            GracefulDegradation.fallback_llm_response,
            "test query"
        )
        print(f"   Graceful execution result type: {type(result)}")
        print(f"   Fallback used: {result.get('fallback_used', 'unknown')}")
        print(f"   Degraded mode: {result.get('degraded_mode', 'unknown')}")
    except Exception as e:
        print(f"   Graceful degradation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n4. Testing validation:")
    try:
        validator = ConfigurationValidator()
        
        import tempfile
        with tempfile.NamedTemporaryFile() as tmp:
            valid, message = validator.validate_file_path(Path(tmp.name))
            print(f"   File validation: {valid} - {message or 'OK'}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            valid, message = validator.validate_directory(Path(tmpdir))
            print(f"   Directory validation: {valid} - {message or 'OK'}")
        
        config = {
            "app": {"name": "test"},
            "document_processing": {
                "chunk_size": 500,
                "supported_formats": [".pdf", ".txt"]
            },
            "ai": {
                "llm": {"model": "test"},
                "embeddings": {"model": "test"}
            },
            "ui": {"theme": "dark"},
            "storage": {"path": "/tmp"}
        }
        valid, issues = validator.validate_app_config(config)
        print(f"   Config validation: {valid} - Issues: {issues}")
        
    except Exception as e:
        print(f"   Validation failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n5. Testing error classification:")
    try:
        classifier = ErrorClassifier()
        test_error = ConnectionError("Connection refused")
        classification = classifier.classify_error(test_error)
        print(f"   Error classification: {classification}")
        print(f"   Recommended action: {classifier.get_recommended_action(test_error)}")
    except Exception as e:
        print(f"   Error classification failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("Exception system test complete.")
    print("=" * 50)