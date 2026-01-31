# docubot/src/document_processing/__init__.py

"""
DocuBot Document Processing Module
Main package for document processing functionality.
"""

# Import core classes
from .processor import DocumentProcessor
from .chunking import ChunkingStrategy, SemanticChunker, FixedSizeChunker
from .cleaning import (
    TextCleaner,
    clean_text_pipeline,
    clean_text_basic,
    clean_text_advanced,
    create_language_specific_cleaner,
    detect_language,
    fix_encoding_issues,
    normalize_unicode,
    normalize_whitespace,
    remove_special_characters,
    strip_html_tags,
    remove_excessive_newlines
)
from .metadata import MetadataExtractor

# Import from extractors submodule
from .extractors import (
    BaseExtractor,
    get_extractor,
    create_extractor,
    get_supported_extensions,
    can_process_file,
    process_document,
    get_module_status,
    log_module_status
)

__all__ = [
    # Main classes
    'DocumentProcessor',
    'ChunkingStrategy',
    'SemanticChunker',
    'FixedSizeChunker',
    'MetadataExtractor',
    
    # Cleaning classes and functions
    'TextCleaner',
    'clean_text_pipeline',
    'clean_text_basic',
    'clean_text_advanced',
    'create_language_specific_cleaner',
    'detect_language',
    'fix_encoding_issues',
    'normalize_unicode',
    'normalize_whitespace',
    'remove_special_characters',
    'strip_html_tags',
    'remove_excessive_newlines',
    
    # Extractors factory functions
    'BaseExtractor',
    'get_extractor',
    'create_extractor',
    'get_supported_extensions',
    'can_process_file',
    'process_document',
    'get_module_status',
    'log_module_status'
]

# Initialize logging
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())