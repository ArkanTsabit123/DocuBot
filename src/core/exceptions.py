"""
DocuBot Custom Exceptions
"""


class DocuBotError(Exception):
    pass


class ConfigurationError(DocuBotError):
    pass


class DocumentError(DocuBotError):
    pass


class ExtractionError(DocumentError):
    pass


class UnsupportedFormatError(ExtractionError):
    pass


class ProcessingError(DocumentError):
    pass


class ChunkingError(ProcessingError):
    pass


class DatabaseError(DocuBotError):
    pass


class VectorStoreError(DocuBotError):
    pass


class AIError(DocuBotError):
    pass


class LLMError(AIError):
    pass


class EmbeddingError(AIError):
    pass


class RAGError(AIError):
    pass


class UIError(DocuBotError):
    pass


class ValidationError(DocuBotError):
    pass


class ResourceError(DocuBotError):
    pass


def handle_error(error: Exception, context: str = "") -> Dict:
    error_type = type(error).__name__
    
    user_messages = {
        'FileNotFoundError': "The file was not found. Please check the file path.",
        'PermissionError': "Permission denied. Please check file permissions.",
        'UnsupportedFormatError': "This file format is not supported.",
        'LLMError': "Unable to get response from AI model. Please try again.",
        'DatabaseError': "Database operation failed. Please check database connection.",
        'ValidationError': "Invalid input provided.",
    }
    
    user_message = user_messages.get(error_type, str(error))
    
    return {
        'success': False,
        'error_type': error_type,
        'error_message': str(error),
        'user_message': user_message,
        'context': context,
        'timestamp': __import__('datetime').datetime.now().isoformat()
    }


def safe_execute(func, *args, **kwargs):
    try:
        result = func(*args, **kwargs)
        return True, result
    except Exception as e:
        return False, handle_error(e, f"Function: {func.__name__}")


class GracefulDegradation:
    """Implement graceful degradation for failing components"""
    
    @staticmethod
    def fallback_document_processing(file_path, error):
        """Fallback for document processing failures"""
        from pathlib import Path
        return {
            'success': False,
            'text': f"Document processing failed: {error}",
            'metadata': {
                'file_name': Path(file_path).name,
                'error': str(error),
                'degraded_mode': True
            },
            'chunks': []
        }
    
    @staticmethod
    def fallback_llm_response(query, error):
        """Fallback for LLM failures"""
        return {
            'answer': f"I apologize, but I'm unable to process your query at the moment. Error: {error}",
            'sources': [],
            'degraded_mode': True,
            'error': str(error)
        }
    
    @staticmethod
    def fallback_embedding(text, error):
        """Fallback for embedding failures"""
        # Return a simple zero vector as fallback
        return [0.0] * 384
    
    @staticmethod
    def fallback_database_operation(operation, error):
        """Fallback for database failures"""
        return {
            'success': False,
            'operation': operation,
            'error': str(error),
            'degraded_mode': True,
            'data': None
        }


def get_logger(name):
    """Get logger for module"""
    import logging
    return logging.getLogger(name)


def graceful_execute(func, fallback_func, *args, **kwargs):
    """Execute function with graceful degradation fallback"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger = get_logger(__name__)
        logger.warning(f"Primary function failed, using fallback: {e}")
        return fallback_func(*args, e, **kwargs)
