# docubot/src/core/exceptions.py

"""
DocuBot Custom Exceptions and Error Handling System
Provides error handling, classification, and recovery mechanisms.

Version: 2.0
Author: DocuBot Team
"""

import logging
import sys
import uuid
import socket
import traceback
from typing import Dict, Any, Optional, Callable, TypeVar, Union, Tuple, List
from pathlib import Path
from datetime import datetime
from contextlib import contextmanager

T = TypeVar('T')
R = TypeVar('R')

__all__ = [
    # Error Codes
    'ErrorCodes',
    
    # Base Exception
    'DocuBotError',
    
    # Configuration & Validation
    'ConfigurationError',
    'ValidationError',
    
    # Document Processing
    'DocumentError',
    'ExtractionError',
    'UnsupportedFormatError',
    'ProcessingError',
    'ChunkingError',
    
    # Database & Storage
    'DatabaseError',
    'VectorStoreError',
    
    # Database Exceptions
    'DatabaseException',
    'DatabaseConnectionError',
    'DatabaseQueryError',
    'DatabaseConstraintError',
    
    # AI & ML
    'AIError',
    'LLMError',
    'EmbeddingError',
    'RAGError',
    'ModelManagementError',
    'RateLimitError',
    'ContextLengthError',
    
    # UI & Resources
    'UIError',
    'ResourceError',
    
    # Network & Connection
    'ConnectionError',
    'NetworkError',
    'TimeoutError',
    'ServiceUnavailableError',
    
    # Authentication & Permission
    'AuthenticationError',
    'PermissionError',
    
    # Model Management
    'ModelError',
    'ModelDownloadError',
    'ModelLoadError',
    'ModelNotFoundError',
    
    # Diagnostic
    'DiagnosticError',
    
    # File & I/O
    'FileSystemError',
    'FileNotFoundError',
    'FileIOError',
    
    # Utility Functions
    'handle_error',
    'safe_execute',
    'graceful_execute',
    'setup_error_logging',
    'log_error_with_context',
    
    # Classes
    'ErrorContext',
    'GracefulDegradation',
    'FallbackRegistry',
    'ConfigurationValidator',
    'ErrorRecovery',
    'ErrorMetrics',
    'ErrorClassifier',
    'ErrorReporter',
    'ErrorSafeContext',
    
    # Helper Functions
    'handle_batch_errors',
    'test_exception_system',
]


class ErrorCodes:
    """Standardized error codes for consistent error reporting."""
    
    # Generic (000-099)
    GENERIC = "DB_ERROR_000"
    UNKNOWN = "DB_ERROR_001"
    
    # Configuration & Settings (100-199)
    CONFIGURATION = "DB_ERROR_100"
    SETTINGS = "DB_ERROR_101"
    
    # Connection & Network (120-149)
    CONNECTION = "DB_ERROR_120"
    NETWORK = "DB_ERROR_121"
    TIMEOUT = "DB_ERROR_122"
    SERVICE_UNAVAILABLE = "DB_ERROR_123"
    
    # Permission & Security (130-139)
    PERMISSION = "DB_ERROR_130"
    AUTHENTICATION = "DB_ERROR_131"
    AUTHORIZATION = "DB_ERROR_132"
    
    # Document Processing (200-299)
    DOCUMENT = "DB_ERROR_200"
    EXTRACTION = "DB_ERROR_210"
    UNSUPPORTED_FORMAT = "DB_ERROR_211"
    PROCESSING = "DB_ERROR_220"
    CHUNKING = "DB_ERROR_221"
    CLEANING = "DB_ERROR_222"
    
    # Database (300-399)
    DATABASE = "DB_ERROR_300"
    QUERY = "DB_ERROR_310"
    CONNECTION_POOL = "DB_ERROR_311"
    MIGRATION = "DB_ERROR_320"
    
    # Database Exceptions (330-349)
    DATABASE_EXCEPTION = "DB_ERROR_330"
    DATABASE_CONNECTION = "DB_ERROR_331"
    DATABASE_QUERY = "DB_ERROR_332"
    DATABASE_CONSTRAINT = "DB_ERROR_333"
    
    # Vector Store (400-499)
    VECTOR_STORE = "DB_ERROR_400"
    INDEX = "DB_ERROR_410"
    EMBEDDING_STORE = "DB_ERROR_420"
    
    # AI & ML (500-599)
    AI = "DB_ERROR_500"
    LLM = "DB_ERROR_510"
    EMBEDDING = "DB_ERROR_520"
    RAG = "DB_ERROR_530"
    MODEL_MANAGEMENT = "DB_ERROR_540"
    RATE_LIMIT = "DB_ERROR_511"
    CONTEXT_LENGTH = "DB_ERROR_512"
    TEMPERATURE = "DB_ERROR_513"
    
    # UI & UX (600-699)
    UI = "DB_ERROR_600"
    RENDERING = "DB_ERROR_610"
    THEME = "DB_ERROR_620"
    LAYOUT = "DB_ERROR_630"
    
    # Validation (700-799)
    VALIDATION = "DB_ERROR_700"
    INPUT_VALIDATION = "DB_ERROR_710"
    DATA_VALIDATION = "DB_ERROR_720"
    
    # Resources (800-899)
    RESOURCE = "DB_ERROR_800"
    MEMORY = "DB_ERROR_810"
    DISK_SPACE = "DB_ERROR_820"
    GPU = "DB_ERROR_830"
    CPU = "DB_ERROR_840"
    
    # Diagnostic (900-999)
    DIAGNOSTIC = "DB_ERROR_900"
    HEALTH_CHECK = "DB_ERROR_910"
    
    # Model Management (1100-1199)
    MODEL = "DB_ERROR_1100"
    MODEL_DOWNLOAD = "DB_ERROR_1110"
    MODEL_LOAD = "DB_ERROR_1120"
    MODEL_NOT_FOUND = "DB_ERROR_1130"
    MODEL_COMPATIBILITY = "DB_ERROR_1140"
    
    # File & I/O (1500-1599)
    FILE_SYSTEM = "DB_ERROR_1500"
    FILE_NOT_FOUND = "DB_ERROR_1510"
    FILE_IO = "DB_ERROR_1520"
    FILE_PERMISSION = "DB_ERROR_1530"


class DocuBotError(Exception):
    """Base exception class for all DocuBot errors."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}
        self.timestamp = datetime.now().isoformat()
        self.error_code = ErrorCodes.GENERIC
        self._traceback = traceback.format_exc()
    
    def __str__(self) -> str:
        """String representation with error code."""
        return f"[{self.error_code}] {super().__str__()}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for serialization."""
        return {
            'error_type': type(self).__name__,
            'error_message': str(self),
            'error_code': self.error_code,
            'context': self.context,
            'timestamp': self.timestamp,
            'traceback': self._traceback
        }
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get debug information for troubleshooting."""
        return {
            'error_type': self.__class__.__name__,
            'message': str(self),
            'error_code': self.error_code,
            'context': self.context,
            'timestamp': self.timestamp,
            'traceback': self._traceback,
            'python_version': sys.version,
            'platform': sys.platform
        }
    
    def format_for_user(self) -> str:
        """Format error message for end-user display."""
        return str(self)


# ============================================================================
# CONFIGURATION & VALIDATION ERRORS
# ============================================================================

class ConfigurationError(DocuBotError):
    """Raised when configuration is invalid or missing."""
    
    def __init__(self, message: str, config_key: Optional[str] = None):
        context = {"config_key": config_key} if config_key else {}
        super().__init__(message, context)
        self.error_code = ErrorCodes.CONFIGURATION


class ValidationError(DocuBotError):
    """Raised when validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None,
                 value: Optional[Any] = None):
        context = {}
        if field:
            context['field'] = field
        if value:
            context['value'] = str(value)[:200]
        super().__init__(message, context)
        self.error_code = ErrorCodes.VALIDATION


# ============================================================================
# DOCUMENT PROCESSING ERRORS
# ============================================================================

class DocumentError(DocuBotError):
    """Base exception for document-related errors."""
    
    def __init__(self, message: str, file_path: Optional[str] = None):
        context = {"file_path": file_path} if file_path else {}
        super().__init__(message, context)
        self.error_code = ErrorCodes.DOCUMENT


class ExtractionError(DocumentError):
    """Raised when document extraction fails."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, 
                 format_type: Optional[str] = None):
        context = {}
        if file_path:
            context['file_path'] = file_path
        if format_type:
            context['format_type'] = format_type
        super().__init__(message, context)
        self.error_code = ErrorCodes.EXTRACTION


class UnsupportedFormatError(ExtractionError):
    """Raised when document format is not supported."""
    
    def __init__(self, format_type: str, file_path: Optional[str] = None):
        message = f"Unsupported document format: {format_type}"
        context = {"format_type": format_type}
        if file_path:
            context['file_path'] = file_path
        super().__init__(message, file_path, format_type)
        self.error_code = ErrorCodes.UNSUPPORTED_FORMAT
    
    def format_for_user(self) -> str:
        return f"Format dokumen tidak didukung: {self.context.get('format_type', 'unknown')}"


class ProcessingError(DocumentError):
    """Raised when document processing fails."""
    
    def __init__(self, message: str, file_path: Optional[str] = None, 
                 stage: Optional[str] = None):
        context = {}
        if file_path:
            context['file_path'] = file_path
        if stage:
            context['stage'] = stage
        super().__init__(message, context)
        self.error_code = ErrorCodes.PROCESSING


class ChunkingError(ProcessingError):
    """Raised when document chunking fails."""
    
    def __init__(self, message: str, file_path: Optional[str] = None):
        super().__init__(message, file_path, "chunking")
        self.error_code = ErrorCodes.CHUNKING


# ============================================================================
# DATABASE & STORAGE ERRORS
# ============================================================================

class DatabaseError(DocuBotError):
    """Raised when database operations fail."""
    
    def __init__(self, message: str, operation: Optional[str] = None,
                 query: Optional[str] = None):
        context = {}
        if operation:
            context['operation'] = operation
        if query:
            context['query'] = query[:500]
        super().__init__(message, context)
        self.error_code = ErrorCodes.DATABASE


class DatabaseException(DocuBotError):
    """Base exception class for specific database-related errors."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message, context)
        self.error_code = ErrorCodes.DATABASE_EXCEPTION


class DatabaseConnectionError(DatabaseException):
    """Exception raised when database connection fails."""
    
    def __init__(self, message: str, host: Optional[str] = None, 
                 port: Optional[int] = None, database: Optional[str] = None):
        context = {}
        if host:
            context['host'] = host
        if port:
            context['port'] = port
        if database:
            context['database'] = database
        super().__init__(message, context)
        self.error_code = ErrorCodes.DATABASE_CONNECTION
    
    def format_for_user(self) -> str:
        return "Koneksi database gagal. Periksa pengaturan database Anda."


class DatabaseQueryError(DatabaseException):
    """Exception raised when database query fails."""
    
    def __init__(self, message: str, query: Optional[str] = None, 
                 parameters: Optional[Dict[str, Any]] = None):
        context = {}
        if query:
            context['query'] = query[:500]
        if parameters:
            context['parameters'] = str(parameters)[:200]
        super().__init__(message, context)
        self.error_code = ErrorCodes.DATABASE_QUERY
    
    def format_for_user(self) -> str:
        return "Operasi query database gagal. Periksa sintaks query dan parameter."


class DatabaseConstraintError(DatabaseException):
    """Exception raised when database constraint is violated."""
    
    def __init__(self, message: str, constraint_name: Optional[str] = None,
                 table_name: Optional[str] = None, column_name: Optional[str] = None):
        context = {}
        if constraint_name:
            context['constraint_name'] = constraint_name
        if table_name:
            context['table_name'] = table_name
        if column_name:
            context['column_name'] = column_name
        super().__init__(message, context)
        self.error_code = ErrorCodes.DATABASE_CONSTRAINT
    
    def format_for_user(self) -> str:
        return "Pelanggaran constraint database. Data tidak memenuhi aturan integritas."


class VectorStoreError(DocuBotError):
    """Raised when vector store operations fail."""
    
    def __init__(self, message: str, operation: Optional[str] = None):
        context = {"operation": operation} if operation else {}
        super().__init__(message, context)
        self.error_code = ErrorCodes.VECTOR_STORE


# ============================================================================
# AI & ML ERRORS
# ============================================================================

class AIError(DocuBotError):
    """Base exception for AI-related errors."""
    
    def __init__(self, message: str, model: Optional[str] = None):
        context = {"model": model} if model else {}
        super().__init__(message, context)
        self.error_code = ErrorCodes.AI


class LLMError(AIError):
    """Raised when LLM operations fail."""
    
    def __init__(self, message: str, model: Optional[str] = None,
                 prompt: Optional[str] = None):
        context = {}
        if model:
            context['model'] = model
        if prompt:
            context['prompt'] = prompt[:200]
        super().__init__(message, context)
        self.error_code = ErrorCodes.LLM
    
    def format_for_user(self) -> str:
        return "Gagal memproses permintaan AI. Silakan coba lagi."


class EmbeddingError(AIError):
    """Raised when embedding operations fail."""
    
    def __init__(self, message: str, model: Optional[str] = None,
                 text_length: Optional[int] = None):
        context = {}
        if model:
            context['model'] = model
        if text_length:
            context['text_length'] = text_length
        super().__init__(message, context)
        self.error_code = ErrorCodes.EMBEDDING


class RAGError(AIError):
    """Raised when RAG operations fail."""
    
    def __init__(self, message: str, stage: Optional[str] = None,
                 context_size: Optional[int] = None):
        context = {}
        if stage:
            context['stage'] = stage
        if context_size:
            context['context_size'] = context_size
        super().__init__(message, context)
        self.error_code = ErrorCodes.RAG


class ModelManagementError(AIError):
    """Raised when model management operations fail."""
    
    def __init__(self, message: str, model_name: Optional[str] = None,
                 action: Optional[str] = None):
        context = {}
        if model_name:
            context['model_name'] = model_name
        if action:
            context['action'] = action
        super().__init__(message, context)
        self.error_code = ErrorCodes.MODEL_MANAGEMENT


class RateLimitError(AIError):
    """Raised when rate limits are exceeded."""
    
    def __init__(self, message: str, model: Optional[str] = None,
                 reset_time: Optional[float] = None):
        context = {}
        if model:
            context['model'] = model
        if reset_time:
            context['reset_time'] = reset_time
        super().__init__(message, context)
        self.error_code = ErrorCodes.RATE_LIMIT


class ContextLengthError(LLMError):
    """Raised when context length is exceeded."""
    
    def __init__(self, message: str, model: Optional[str] = None,
                 current_length: Optional[int] = None,
                 max_length: Optional[int] = None):
        context = {}
        if model:
            context['model'] = model
        if current_length:
            context['current_length'] = current_length
        if max_length:
            context['max_length'] = max_length
        super().__init__(message, context)
        self.error_code = ErrorCodes.CONTEXT_LENGTH
    
    def format_for_user(self) -> str:
        return "Teks terlalu panjang. Silakan perpendek teks Anda."


# ============================================================================
# UI & RESOURCE ERRORS
# ============================================================================

class UIError(DocuBotError):
    """Raised when UI operations fail."""
    
    def __init__(self, message: str, component: Optional[str] = None):
        context = {"component": component} if component else {}
        super().__init__(message, context)
        self.error_code = ErrorCodes.UI


class ResourceError(DocuBotError):
    """Raised when system resources are insufficient."""
    
    def __init__(self, message: str, resource_type: Optional[str] = None,
                 required: Optional[Any] = None, available: Optional[Any] = None):
        context = {}
        if resource_type:
            context['resource_type'] = resource_type
        if required:
            context['required'] = str(required)
        if available:
            context['available'] = str(available)
        super().__init__(message, context)
        self.error_code = ErrorCodes.RESOURCE


# ============================================================================
# NETWORK & CONNECTION ERRORS
# ============================================================================

class ConnectionError(DocuBotError):
    """Raised when connection to external service fails."""
    
    def __init__(self, message: str, service: Optional[str] = None, 
                 endpoint: Optional[str] = None):
        context = {}
        if service:
            context['service'] = service
        if endpoint:
            context['endpoint'] = endpoint
        super().__init__(message, context)
        self.error_code = ErrorCodes.CONNECTION
    
    def format_for_user(self) -> str:
        return "Koneksi gagal. Periksa koneksi jaringan Anda."


class NetworkError(ConnectionError):
    """Raised for general network failures."""
    
    def __init__(self, message: str, service: Optional[str] = None):
        super().__init__(message, service, None)
        self.error_code = ErrorCodes.NETWORK


class TimeoutError(ConnectionError):
    """Raised when an operation times out."""
    
    def __init__(self, message: str, service: Optional[str] = None,
                 timeout_seconds: Optional[float] = None):
        context = {"service": service} if service else {}
        if timeout_seconds:
            context['timeout_seconds'] = timeout_seconds
        super().__init__(message, context)
        self.error_code = ErrorCodes.TIMEOUT
    
    def format_for_user(self) -> str:
        return "Operasi timeout. Silakan coba lagi."


class ServiceUnavailableError(ConnectionError):
    """Raised when a service is unavailable."""
    
    def __init__(self, message: str, service: Optional[str] = None):
        context = {"service": service} if service else {}
        super().__init__(message, context)
        self.error_code = ErrorCodes.SERVICE_UNAVAILABLE
    
    def format_for_user(self) -> str:
        return "Layanan tidak tersedia. Silakan coba lagi nanti."


# ============================================================================
# AUTHENTICATION & PERMISSION ERRORS
# ============================================================================

class AuthenticationError(DocuBotError):
    """Raised when authentication fails."""
    
    def __init__(self, message: str, resource: Optional[str] = None):
        context = {"resource": resource} if resource else {}
        super().__init__(message, context)
        self.error_code = ErrorCodes.AUTHENTICATION
    
    def format_for_user(self) -> str:
        return "Autentikasi gagal. Periksa kredensial Anda."


class PermissionError(DocuBotError):
    """Raised when permission is denied."""
    
    def __init__(self, message: str, resource: Optional[str] = None):
        context = {"resource": resource} if resource else {}
        super().__init__(message, context)
        self.error_code = ErrorCodes.PERMISSION
    
    def format_for_user(self) -> str:
        return "Izin ditolak. Periksa hak akses Anda."


# ============================================================================
# MODEL MANAGEMENT ERRORS
# ============================================================================

class ModelError(DocuBotError):
    """Base exception for model-related errors."""
    
    def __init__(self, message: str, model_name: Optional[str] = None,
                 model_path: Optional[str] = None):
        context = {}
        if model_name:
            context['model_name'] = model_name
        if model_path:
            context['model_path'] = model_path
        super().__init__(message, context)
        self.error_code = ErrorCodes.MODEL


class ModelDownloadError(ModelError):
    """Raised when model download fails."""
    
    def __init__(self, message: str, model_name: Optional[str] = None,
                 download_url: Optional[str] = None, http_status: Optional[int] = None):
        context = {}
        if model_name:
            context['model_name'] = model_name
        if download_url:
            context['download_url'] = download_url
        if http_status:
            context['http_status'] = http_status
        super().__init__(message, model_name=model_name, model_path=None)
        self.error_code = ErrorCodes.MODEL_DOWNLOAD
    
    def format_for_user(self) -> str:
        return "Download model gagal. Periksa koneksi internet Anda."


class ModelLoadError(ModelError):
    """Raised when model loading fails."""
    
    def __init__(self, message: str, model_name: Optional[str] = None,
                 model_path: Optional[str] = None, load_stage: Optional[str] = None):
        context = {}
        if model_name:
            context['model_name'] = model_name
        if model_path:
            context['model_path'] = model_path
        if load_stage:
            context['load_stage'] = load_stage
        super().__init__(message, model_name=model_name, model_path=model_path)
        self.error_code = ErrorCodes.MODEL_LOAD


class ModelNotFoundError(ModelError):
    """Raised when requested model is not found."""
    
    def __init__(self, message: str, model_name: Optional[str] = None,
                 model_path: Optional[str] = None, search_paths: Optional[List[str]] = None):
        context = {}
        if model_name:
            context['model_name'] = model_name
        if model_path:
            context['model_path'] = model_path
        if search_paths:
            context['search_paths'] = search_paths[:5]
        super().__init__(message, model_name=model_name, model_path=model_path)
        self.error_code = ErrorCodes.MODEL_NOT_FOUND
    
    def format_for_user(self) -> str:
        return "Model tidak ditemukan. Silakan instal model yang diperlukan."


# ============================================================================
# DIAGNOSTIC ERRORS
# ============================================================================

class DiagnosticError(DocuBotError):
    """Raised when diagnostic checks fail."""
    
    def __init__(self, message: str, check_name: Optional[str] = None,
                 details: Optional[Dict[str, Any]] = None):
        context = {}
        if check_name:
            context['check_name'] = check_name
        if details:
            context.update(details)
        super().__init__(message, context)
        self.error_code = ErrorCodes.DIAGNOSTIC


# ============================================================================
# FILE & I/O ERRORS
# ============================================================================

class FileSystemError(DocuBotError):
    """Base exception for file system operations."""
    
    def __init__(self, message: str, file_path: Optional[str] = None):
        context = {"file_path": file_path} if file_path else {}
        super().__init__(message, context)
        self.error_code = ErrorCodes.FILE_SYSTEM


class FileNotFoundError(FileSystemError):
    """Raised when file is not found."""
    
    def __init__(self, message: str, file_path: Optional[str] = None):
        super().__init__(message, file_path)
        self.error_code = ErrorCodes.FILE_NOT_FOUND
    
    def format_for_user(self) -> str:
        return "File tidak ditemukan. Periksa lokasi file."


class FileIOError(FileSystemError):
    """Raised when file I/O operations fail."""
    
    def __init__(self, message: str, file_path: Optional[str] = None,
                 operation: Optional[str] = None):
        context = {}
        if file_path:
            context['file_path'] = file_path
        if operation:
            context['operation'] = operation
        super().__init__(message, context)
        self.error_code = ErrorCodes.FILE_IO
    
    def format_for_user(self) -> str:
        return "Operasi file gagal. Periksa file dan izin akses."


# ============================================================================
# ERROR HANDLING UTILITIES
# ============================================================================

def handle_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Handle an exception and return standardized error information."""
    error_type = type(error).__name__
    error_context = context or {}
    
    user_messages = {
        'FileNotFoundError': "File tidak ditemukan. Periksa lokasi file.",
        'PermissionError': "Izin ditolak. Periksa hak akses file.",
        'UnsupportedFormatError': "Format dokumen tidak didukung. Konversi ke format yang didukung.",
        'LLMError': "Gagal memproses permintaan AI. Silakan coba lagi.",
        'DatabaseError': "Operasi database gagal. Periksa koneksi database.",
        'DatabaseConnectionError': "Koneksi database gagal. Periksa pengaturan koneksi.",
        'DatabaseQueryError': "Query database gagal. Periksa sintaks query.",
        'DatabaseConstraintError': "Constraint database dilanggar. Periksa aturan integritas data.",
        'ValidationError': "Validasi input gagal. Periksa data yang diberikan.",
        'ResourceError': "Sumber daya sistem tidak cukup. Periksa memori dan ruang disk.",
        'ConfigurationError': "Error konfigurasi terdeteksi. Periksa file konfigurasi.",
        'DiagnosticError': "Cek diagnostik sistem gagal. Jalankan alat diagnostik.",
        'ValueError': "Nilai tidak valid. Periksa input Anda.",
        'TypeError': "Error tipe data. Periksa tipe data input.",
        'KeyError': "Key yang diperlukan tidak ditemukan. Periksa struktur data.",
        'AttributeError': "Atribut tidak ditemukan. Periksa struktur objek.",
        'ImportError': "Import module gagal. Periksa instalasi Anda.",
        'OSError': "Error sistem operasi. Periksa path file dan izin.",
        'MemoryError': "Memori tidak cukup. Tutup aplikasi lain dan coba lagi.",
        'TimeoutError': "Operasi timeout. Silakan coba lagi atau periksa koneksi.",
        'ConnectionError': "Koneksi gagal. Periksa koneksi jaringan Anda.",
        'RuntimeError': "Runtime error terjadi. Silakan coba lagi atau hubungi support.",
        'AuthenticationError': "Autentikasi gagal. Periksa kredensial dan izin Anda.",
        'RateLimitError': "Limit rate terlampaui. Tunggu sebelum membuat permintaan tambahan.",
        'ContextLengthError': "Panjang konteks terlampaui. Perkecil ukuran input.",
        'ModelError': "Operasi model gagal. Periksa konfigurasi model.",
        'ModelDownloadError': "Download model gagal. Periksa koneksi internet.",
        'ModelLoadError': "Loading model gagal. Periksa file model dan kompatibilitas.",
        'ModelNotFoundError': "Model tidak ditemukan. Verifikasi nama model dan instalasi.",
    }
    
    if isinstance(error, DocuBotError):
        error_dict = {
            'success': False,
            'error_type': error_type,
            'error_code': error.error_code,
            'error_message': str(error),
            'user_message': error.format_for_user(),
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
            'user_message': user_messages.get(error_type, "Error tak terduga terjadi. Silakan coba lagi."),
            'context': error_context,
            'timestamp': datetime.now().isoformat(),
            'is_docubot_error': False
        }
    
    return error_dict


def safe_execute(func: Callable[..., T], *args: Any, **kwargs: Any) -> Tuple[bool, Union[T, Dict[str, Any]]]:
    """Execute a function safely, catching any exceptions."""
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


def handle_batch_errors(
    errors: List[Exception],
    operation: str = "batch_operation"
) -> Dict[str, Any]:
    """Handle multiple errors from batch operations."""
    successful = 0
    failed = 0
    error_details = []
    
    for error in errors:
        error_info = handle_error(error, {"batch_operation": operation})
        error_details.append(error_info)
        if error_info.get('is_docubot_error', False):
            failed += 1
        else:
            classifier = ErrorClassifier()
            classification = classifier.classify_error(error)
            if classification.get('can_retry', False):
                failed += 1
            else:
                failed += 1
    
    total = len(errors)
    return {
        'total': total,
        'successful': successful,
        'failed': failed,
        'success_rate': (successful / total * 100) if total > 0 else 0,
        'errors': error_details[:10],
        'has_more_errors': len(error_details) > 10
    }


# ============================================================================
# ERROR CONTEXT MANAGER
# ============================================================================

class ErrorContext:
    """Context manager for error handling within a specific operation."""
    
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


# ============================================================================
# GRACEFUL DEGRADATION
# ============================================================================

class GracefulDegradation:
    """Provides fallback strategies for various types of failures."""
    
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
            'answer': f"Maaf, saya tidak dapat memproses permintaan Anda saat ini. Sistem mengalami error: {type(error).__name__}",
            'sources': [],
            'confidence': 0.0,
            'degraded_mode': True,
            'fallback_used': True,
            'error': str(error),
            'query': query[:500],
            'timestamp': datetime.now().isoformat()
        }
    
    @staticmethod
    def fallback_embedding(error: Exception, text: str, dimensions: int = 384, **kwargs) -> List[float]:
        """Return zero vector as fallback embedding."""
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
    
    @staticmethod
    def fallback_model_operation(error: Exception, model_name: str, **kwargs) -> Dict[str, Any]:
        return {
            'success': False,
            'model_name': model_name,
            'error': str(error),
            'error_type': type(error).__name__,
            'degraded_mode': True,
            'fallback_used': True,
            'alternative_model': 'default',
            'timestamp': datetime.now().isoformat()
        }


class FallbackRegistry:
    """Registry for fallback strategies."""
    
    _strategies = {}
    
    @classmethod
    def register(cls, error_type: type, fallback_func: Callable):
        cls._strategies[error_type] = fallback_func
    
    @classmethod
    def get_fallback(cls, error: Exception) -> Callable:
        """Get appropriate fallback function for the error."""
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
FallbackRegistry.register(DatabaseConnectionError, GracefulDegradation.fallback_database_operation)
FallbackRegistry.register(DatabaseQueryError, GracefulDegradation.fallback_database_operation)
FallbackRegistry.register(DatabaseConstraintError, GracefulDegradation.fallback_database_operation)
FallbackRegistry.register(VectorStoreError, GracefulDegradation.fallback_vector_store_operation)
FallbackRegistry.register(ExtractionError, GracefulDegradation.fallback_document_processing)
FallbackRegistry.register(ModelError, GracefulDegradation.fallback_model_operation)
FallbackRegistry.register(ConnectionError, GracefulDegradation.fallback_llm_response)
FallbackRegistry.register(TimeoutError, GracefulDegradation.fallback_llm_response)


def graceful_execute(
    primary_func: Callable[..., R], 
    fallback_func: Optional[Callable[..., R]] = None, 
    *args: Any, 
    **kwargs: Any
) -> R:
    """Execute a function with fallback on failure."""
    logger = logging.getLogger(__name__)
    
    try:
        result = primary_func(*args, **kwargs)
        return result
    except Exception as error:
        if fallback_func is None:
            fallback_func = FallbackRegistry.get_fallback(error)
        
        logger.warning(
            f"Primary function {primary_func.__name__} failed, using fallback",
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
                f"Fallback function also failed",
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


# ============================================================================
# CONFIGURATION VALIDATION
# ============================================================================

class ConfigurationValidator:
    """Validates various types of configurations."""
    
    @staticmethod
    def validate_app_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate application configuration."""
        if not config:
            return False, ["Configuration is empty"]
        
        required_keys = ['app', 'document_processing', 'ai', 'ui', 'storage']
        missing_keys = [key for key in required_keys if key not in config]
        
        if missing_keys:
            return False, [f"Missing required section: {key}" for key in missing_keys]
        
        issues: List[str] = []
        
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
    def validate_file_path(path: Union[str, Path]) -> Tuple[bool, Optional[str]]:
        """Validate file path existence and accessibility."""
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
    def validate_directory(path: Union[str, Path]) -> Tuple[bool, Optional[str]]:
        """Validate directory existence and write permission."""
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
    def validate_model_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate model configuration."""
        if not config:
            return False, ["Model configuration is empty"]
        
        issues: List[str] = []
        
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


# ============================================================================
# LOGGING UTILITIES
# ============================================================================

def setup_error_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure error logging for the application."""
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
    """Log an error with additional context."""
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


# ============================================================================
# ERROR RECOVERY
# ============================================================================

class ErrorRecovery:
    """Provides recovery strategies for different error types."""
    
    @staticmethod
    def recover_from_database_error(error: Exception, max_retries: int = 3) -> Tuple[bool, str]:
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
    def recover_from_resource_error(error: ResourceError) -> Tuple[bool, str]:
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
    def recover_from_model_error(error: ModelError) -> Tuple[bool, str]:
        if isinstance(error, ModelNotFoundError):
            return True, "Attempting to download missing model"
        elif isinstance(error, ModelDownloadError):
            return True, "Retrying with alternative download mirror"
        elif isinstance(error, ModelLoadError):
            return True, "Attempting to load with reduced precision or alternative format"
        
        return False, "No recovery strategy available"
    
    @staticmethod
    def should_retry(error: Exception, attempt: int, max_attempts: int = 3) -> bool:
        """Determine if an operation should be retried."""
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
            'KeyError',
            'PermissionError',
            'FileNotFoundError'
        ]
        
        error_type = type(error).__name__
        if error_type in no_retry_errors:
            return False
        
        error_msg = str(error).lower()
        if "disk full" in error_msg:
            return False
        
        if "unsupported" in error_msg or "invalid format" in error_msg:
            return False
        
        if isinstance(error, ModelNotFoundError) and attempt < 2:
            return True
        
        if isinstance(error, (ConnectionError, TimeoutError, NetworkError)):
            return True
        
        return True
    
    @staticmethod
    def get_retry_delay(attempt: int, base_delay: float = 1.0) -> float:
        """Calculate exponential backoff delay."""
        return base_delay * (2 ** (attempt - 1))


# ============================================================================
# ERROR METRICS
# ============================================================================

class ErrorMetrics:
    """Collects and analyzes error metrics."""
    
    def __init__(self):
        self.error_counts = {}
        self.error_timestamps = []
        self.recovery_attempts = 0
        self.recovery_successes = 0
        
    def record_error(self, error: Exception) -> None:
        """Record an error occurrence."""
        error_type = type(error).__name__
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        self.error_timestamps.append(datetime.now().isoformat())
    
    def record_recovery(self, success: bool) -> None:
        """Record a recovery attempt."""
        self.recovery_attempts += 1
        if success:
            self.recovery_successes += 1
    
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics."""
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
        """Calculate error frequency in errors per minute."""
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


# ============================================================================
# ERROR CLASSIFICATION
# ============================================================================

class ErrorClassifier:
    """Classifies errors for appropriate handling strategies."""
    
    @staticmethod
    def classify_error(error: Exception) -> Dict[str, Any]:
        """Classify an error for handling strategy."""
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
        
        elif isinstance(error, ModelError):
            classification['is_system_error'] = True
            classification['severity'] = 'high'
            if isinstance(error, ModelNotFoundError):
                classification['requires_user_intervention'] = False
                classification['can_retry'] = True
            elif isinstance(error, ModelDownloadError):
                classification['is_temporary'] = True
                classification['can_retry'] = True
            elif isinstance(error, ModelLoadError):
                classification['requires_user_intervention'] = True
                classification['can_retry'] = False
        
        elif isinstance(error, DatabaseException):
            classification['is_system_error'] = True
            classification['severity'] = 'high'
            if isinstance(error, DatabaseConnectionError):
                classification['is_temporary'] = True
                classification['can_retry'] = True
            elif isinstance(error, DatabaseQueryError):
                classification['requires_user_intervention'] = True
                classification['can_retry'] = False
            elif isinstance(error, DatabaseConstraintError):
                classification['is_user_error'] = True
                classification['can_retry'] = False
        
        return classification
    
    @staticmethod
    def get_recommended_action(error: Exception) -> str:
        """Get recommended action for an error."""
        classification = ErrorClassifier.classify_error(error)
        
        if classification['is_connection_error']:
            return "Periksa koneksi jaringan Anda dan coba lagi"
        elif classification['is_resource_error']:
            return "Bebaskan sumber daya sistem (memori/disk) dan coba lagi"
        elif classification['is_configuration_error']:
            return "Periksa dan perbarui pengaturan konfigurasi Anda"
        elif classification['is_user_error']:
            return "Periksa input Anda dan coba lagi dengan data yang valid"
        elif classification['requires_user_intervention']:
            return "Intervensi manual diperlukan. Hubungi support jika masalah berlanjut"
        elif classification['can_retry']:
            return "Masalah sementara. Silakan coba lagi dalam beberapa saat"
        elif isinstance(error, ModelError):
            if isinstance(error, ModelNotFoundError):
                return "Model tidak ditemukan. Silakan instal model yang diperlukan atau periksa konfigurasi model"
            elif isinstance(error, ModelDownloadError):
                return "Download model gagal. Periksa koneksi internet Anda dan coba lagi"
            elif isinstance(error, ModelLoadError):
                return "Loading model gagal. Periksa kompatibilitas dan integritas file model"
            else:
                return "Error model terjadi. Periksa konfigurasi model dan coba lagi"
        elif isinstance(error, DatabaseException):
            if isinstance(error, DatabaseConnectionError):
                return "Koneksi database gagal. Periksa pengaturan koneksi dan status server database"
            elif isinstance(error, DatabaseQueryError):
                return "Query database gagal. Periksa sintaks query dan parameter yang diberikan"
            elif isinstance(error, DatabaseConstraintError):
                return "Constraint database dilanggar. Periksa aturan integritas data Anda"
            else:
                return "Error database terjadi. Periksa log database untuk detail lebih lanjut"
        else:
            return "Silakan coba lagi atau hubungi support jika masalah berlanjut"


# ============================================================================
# ERROR REPORTING
# ============================================================================

class ErrorReporter:
    """Reports errors to external monitoring services."""
    
    @staticmethod
    def report_to_sentry(error: Exception, context: Dict[str, Any]) -> None:
        """Report error to Sentry.io."""
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
        """Create a comprehensive error report."""
        return {
            "error_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "application": "DocuBot",
            "version": "1.0.0",
            "environment": "development",
            "error_details": error_info,
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform,
                "hostname": socket.gethostname() if hasattr(socket, 'gethostname') else "unknown"
            }
        }


# ============================================================================
# ERROR SAFE CONTEXT
# ============================================================================

class ErrorSafeContext:
    """Context manager that ensures cleanup even on errors."""
    
    def __init__(self, cleanup_funcs: Optional[List[Tuple[Callable, tuple, dict]]] = None):
        self.cleanup_funcs = cleanup_funcs or []
        self.cleanup_executed = False
    
    def add_cleanup(self, func: Callable, *args, **kwargs) -> None:
        """Add a cleanup function to be executed on exit."""
        self.cleanup_funcs.append((func, args, kwargs))
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if not self.cleanup_executed:
            self._execute_cleanup()
            self.cleanup_executed = True
        
        if exc_val:
            error_info = handle_error(exc_val)
            logger = logging.getLogger(__name__)
            log_error_with_context(logger, exc_val)
            
            if isinstance(exc_val, (KeyboardInterrupt, SystemExit)):
                return False
            return True
        
        return False
    
    def _execute_cleanup(self) -> None:
        """Execute all registered cleanup functions."""
        for func, args, kwargs in self.cleanup_funcs:
            try:
                func(*args, **kwargs)
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.warning(f"Cleanup function failed: {e}")


@contextmanager
def error_safe_context(*cleanup_funcs: Callable):
    """Context manager for safe execution with cleanup."""
    context = ErrorSafeContext()
    for func in cleanup_funcs:
        context.add_cleanup(func)
    
    try:
        yield context
    finally:
        if not context.cleanup_executed:
            context._execute_cleanup()


# ============================================================================
# TEST FUNCTION
# ============================================================================

def test_exception_system() -> None:
    """Test the exception system functionality."""
    import logging
    logging.basicConfig(level=logging.DEBUG, format='%(message)s')
    test_logger = logging.getLogger("test_exceptions")
    
    print("\n" + "=" * 60)
    print("Testing DocuBot Exception System")
    print("=" * 60)
    
    print("\n1. Testing all exception types:")
    test_cases = [
        (ConfigurationError, ["Missing config"], {"config_key": "api_key"}),
        (UnsupportedFormatError, ["xyz", "/test/file.xyz"], {}),
        (LLMError, ["Model failed"], {"model": "llama2", "prompt": "test"}),
        (ModelNotFoundError, ["Model missing"], {"model_name": "test-model"}),
        (DatabaseError, ["Query failed"], {"operation": "SELECT", "query": "SELECT * FROM users"}),
        (DatabaseConnectionError, ["Connection failed"], {"host": "localhost", "port": 5432}),
        (DatabaseQueryError, ["Query syntax error"], {"query": "SELECT FROM table"}),
        (DatabaseConstraintError, ["Unique constraint violated"], {"constraint_name": "unique_email"}),
        (ConnectionError, ["Connection refused"], {"service": "database"}),
        (PermissionError, ["Access denied"], {"resource": "/secure/file"}),
        (FileNotFoundError, ["File not found"], {"file_path": "/nonexistent/file.txt"}),
    ]
    
    for exc_class, args, kwargs in test_cases:
        try:
            exc = exc_class(*args, **kwargs)
            print(f"   ✓ {exc_class.__name__}: {exc.error_code}")
            print(f"      User message: {exc.format_for_user()}")
        except Exception as e:
            print(f"   ✗ {exc_class.__name__} failed: {e}")
    
    print("\n2. Testing error handling:")
    try:
        success, result = safe_execute(lambda x: 1 / x, 0)
        print(f"   ✓ Error handling worked: success={success}")
        if isinstance(result, dict):
            print(f"   Error code: {result.get('error_code')}")
            print(f"   User message: {result.get('user_message')}")
    except Exception as e:
        print(f"   ✗ Error handling failed: {e}")
    
    print("\n3. Testing graceful degradation:")
    try:
        def failing_function():
            raise LLMError("LLM service unavailable", model="llama2")
        
        result = graceful_execute(
            failing_function,
            GracefulDegradation.fallback_llm_response,
            "test query"
        )
        print(f"   ✓ Graceful execution successful")
        print(f"   Fallback used: {result.get('fallback_used', 'unknown')}")
        print(f"   Answer: {result.get('answer', 'N/A')[:50]}...")
    except Exception as e:
        print(f"   ✗ Graceful degradation failed: {e}")
    
    print("\n4. Testing error classification:")
    try:
        classifier = ErrorClassifier()
        test_error = ConnectionError("Connection refused", service="database")
        classification = classifier.classify_error(test_error)
        print(f"   ✓ Error classification successful")
        print(f"   Is connection error: {classification.get('is_connection_error')}")
        print(f"   Can retry: {classification.get('can_retry')}")
        print(f"   Recommended action: {classifier.get_recommended_action(test_error)}")
    except Exception as e:
        print(f"   ✗ Error classification failed: {e}")
    
    print("\n5. Testing validation:")
    try:
        validator = ConfigurationValidator()
        
        import tempfile
        with tempfile.NamedTemporaryFile() as tmp:
            valid, message = validator.validate_file_path(Path(tmp.name))
            print(f"   ✓ File validation: {valid} - {message or 'OK'}")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            valid, message = validator.validate_directory(Path(tmpdir))
            print(f"   ✓ Directory validation: {valid} - {message or 'OK'}")
        
    except Exception as e:
        print(f"   ✗ Validation failed: {e}")
    
    print("\n6. Testing error context manager:")
    try:
        with ErrorContext("test_operation", raise_on_error=False) as ctx:
            raise ValueError("Test error in context")
        
        print(f"   ✓ Error context successful")
        print(f"   Error occurred: {ctx.error_occurred}")
        if ctx.error_info:
            print(f"   Error type: {ctx.error_info.get('error_type')}")
    except Exception as e:
        print(f"   ✗ Error context failed: {e}")
    
    print("\n" + "=" * 60)
    print("Exception system test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_exception_system()