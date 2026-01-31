# docubot/src/document_processing/extractors/__init__.py

"""
Document Processing Extractors Module - DocuBot
Complete extractors factory, registry, and document processing pipeline.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Union, List

logger = logging.getLogger(__name__)

# Component registry and availability tracking
_COMPONENTS: Dict[str, Any] = {}
_COMPONENT_AVAILABILITY: Dict[str, bool] = {}
_EXTRACTOR_REGISTRY: Dict[str, Callable] = {}


class PlaceholderBaseExtractor:
    """Placeholder when BaseExtractor is not available."""
    def __init__(self):
        self.supported_formats = []
    
    def extract(self, file_path):
        raise NotImplementedError("BaseExtractor module is not loaded")
    
    def can_extract(self, file_path):
        return False


try:
    from .base_extractor import BaseExtractor
    _COMPONENTS['BaseExtractor'] = BaseExtractor
    _COMPONENT_AVAILABILITY['base_extractor'] = True
    logger.debug("BaseExtractor imported successfully")
except ImportError as e:
    _COMPONENTS['BaseExtractor'] = PlaceholderBaseExtractor
    _COMPONENT_AVAILABILITY['base_extractor'] = False
    logger.debug(f"BaseExtractor not available: {e}")


try:
    from .txt_extractor import TXTExtractor
    _COMPONENTS['TXTExtractor'] = TXTExtractor
    _COMPONENT_AVAILABILITY['txt_extractor'] = True
    logger.debug("TXTExtractor imported successfully")
except ImportError as e:
    _COMPONENTS['TXTExtractor'] = None
    _COMPONENT_AVAILABILITY['txt_extractor'] = False
    logger.debug(f"TXTExtractor not available: {e}")


try:
    from .docx_extractor import DOCXExtractor
    _COMPONENTS['DOCXExtractor'] = DOCXExtractor
    _COMPONENT_AVAILABILITY['docx_extractor'] = True
    logger.debug("DOCXExtractor imported successfully")
except ImportError as e:
    _COMPONENTS['DOCXExtractor'] = None
    _COMPONENT_AVAILABILITY['docx_extractor'] = False
    logger.debug(f"DOCXExtractor not available: {e}")


try:
    from .pdf_extractor import PDFExtractor
    _COMPONENTS['PDFExtractor'] = PDFExtractor
    _COMPONENT_AVAILABILITY['pdf_extractor'] = True
    logger.debug("PDFExtractor imported successfully")
except ImportError as e:
    _COMPONENTS['PDFExtractor'] = None
    _COMPONENT_AVAILABILITY['pdf_extractor'] = False
    logger.debug(f"PDFExtractor not available: {e}")


try:
    from .epub_extractor import EPUBExtractor
    _COMPONENTS['EPUBExtractor'] = EPUBExtractor
    _COMPONENT_AVAILABILITY['epub_extractor'] = True
    logger.debug("EPUBExtractor imported successfully")
except ImportError:
    _COMPONENTS['EPUBExtractor'] = None
    _COMPONENT_AVAILABILITY['epub_extractor'] = False


try:
    from .markdown_extractor import MarkdownExtractor
    _COMPONENTS['MarkdownExtractor'] = MarkdownExtractor
    _COMPONENT_AVAILABILITY['markdown_extractor'] = True
    logger.debug("MarkdownExtractor imported successfully")
except ImportError:
    _COMPONENTS['MarkdownExtractor'] = None
    _COMPONENT_AVAILABILITY['markdown_extractor'] = False


try:
    from .html_extractor import HTMLExtractor
    _COMPONENTS['HTMLExtractor'] = HTMLExtractor
    _COMPONENT_AVAILABILITY['html_extractor'] = True
    logger.debug("HTMLExtractor imported successfully")
except ImportError:
    _COMPONENTS['HTMLExtractor'] = None
    _COMPONENT_AVAILABILITY['html_extractor'] = False


try:
    from .chunking import ChunkingStrategy, SmartChunker
    _COMPONENTS['ChunkingStrategy'] = ChunkingStrategy
    _COMPONENTS['SmartChunker'] = SmartChunker
    _COMPONENT_AVAILABILITY['chunking'] = True
    logger.debug("Chunking module imported successfully")
except ImportError as e:
    _COMPONENTS['ChunkingStrategy'] = None
    _COMPONENTS['SmartChunker'] = None
    _COMPONENT_AVAILABILITY['chunking'] = False
    logger.debug(f"Chunking module not available: {e}")


try:
    from .cleaning import TextCleaner, CleaningPipeline
    _COMPONENTS['TextCleaner'] = TextCleaner
    _COMPONENTS['CleaningPipeline'] = CleaningPipeline
    _COMPONENT_AVAILABILITY['cleaning'] = True
    logger.debug("Text cleaning module imported successfully")
except ImportError as e:
    _COMPONENTS['TextCleaner'] = None
    _COMPONENTS['CleaningPipeline'] = None
    _COMPONENT_AVAILABILITY['cleaning'] = False
    logger.debug(f"Text cleaning module not available: {e}")


try:
    from .metadata import MetadataExtractor, extract_document_metadata
    _COMPONENTS['MetadataExtractor'] = MetadataExtractor
    _COMPONENTS['extract_document_metadata'] = extract_document_metadata
    _COMPONENT_AVAILABILITY['metadata'] = True
    logger.debug("Metadata module imported successfully")
except ImportError as e:
    _COMPONENTS['MetadataExtractor'] = None
    _COMPONENTS['extract_document_metadata'] = None
    _COMPONENT_AVAILABILITY['metadata'] = False
    logger.debug(f"Metadata module not available: {e}")


if _COMPONENT_AVAILABILITY.get('txt_extractor') and _COMPONENTS['TXTExtractor']:
    text_extensions = ['.txt', '.text', '.md', '.json', '.csv', '.xml', '.html']
    for ext in text_extensions:
        _EXTRACTOR_REGISTRY[ext] = lambda: _COMPONENTS['TXTExtractor']()
    logger.debug(f"Registered TXTExtractor for extensions: {text_extensions}")


if _COMPONENT_AVAILABILITY.get('docx_extractor') and _COMPONENTS['DOCXExtractor']:
    docx_extensions = ['.docx', '.docm', '.dotx', '.dotm', '.doc']
    for ext in docx_extensions:
        _EXTRACTOR_REGISTRY[ext] = lambda: _COMPONENTS['DOCXExtractor']()
    logger.debug(f"Registered DOCXExtractor for extensions: {docx_extensions}")


if _COMPONENT_AVAILABILITY.get('pdf_extractor') and _COMPONENTS['PDFExtractor']:
    _EXTRACTOR_REGISTRY['.pdf'] = lambda: _COMPONENTS['PDFExtractor']()
    logger.debug("Registered PDFExtractor for .pdf extension")


if _COMPONENT_AVAILABILITY.get('epub_extractor') and _COMPONENTS['EPUBExtractor']:
    _EXTRACTOR_REGISTRY['.epub'] = lambda: _COMPONENTS['EPUBExtractor']()
    logger.debug("Registered EPUBExtractor for .epub extension")


if _COMPONENT_AVAILABILITY.get('markdown_extractor') and _COMPONENTS['MarkdownExtractor']:
    _EXTRACTOR_REGISTRY['.md'] = lambda: _COMPONENTS['MarkdownExtractor']()
    _EXTRACTOR_REGISTRY['.markdown'] = lambda: _COMPONENTS['MarkdownExtractor']()
    logger.debug("Registered MarkdownExtractor for .md and .markdown extensions")


if _COMPONENT_AVAILABILITY.get('html_extractor') and _COMPONENTS['HTMLExtractor']:
    _EXTRACTOR_REGISTRY['.html'] = lambda: _COMPONENTS['HTMLExtractor']()
    _EXTRACTOR_REGISTRY['.htm'] = lambda: _COMPONENTS['HTMLExtractor']()
    logger.debug("Registered HTMLExtractor for .html and .htm extensions")


def get_extractor(extension: str) -> Optional[Any]:
    """
    Factory function to get appropriate extractor for file extension.
    
    Args:
        extension: File extension including dot (e.g., '.docx', '.pdf')
        
    Returns:
        Extractor instance or None if no extractor is available
    """
    extension = extension.lower()
    
    if extension not in _EXTRACTOR_REGISTRY:
        logger.warning(f"No extractor registered for extension: {extension}")
        return None
    
    extractor_factory = _EXTRACTOR_REGISTRY[extension]
    
    try:
        if callable(extractor_factory):
            return extractor_factory()
        else:
            return extractor_factory
    except Exception as e:
        logger.error(f"Failed to create extractor for {extension}: {e}")
        return None


def create_extractor(file_path_or_extension: Union[str, Path], 
                    extractor_type: Optional[str] = None) -> Optional[Any]:
    """
    Create an appropriate extractor for a file or extension.
    
    Args:
        file_path_or_extension: Either a file path or file extension (e.g., '.pdf')
        extractor_type: Optional specific extractor type to create
        
    Returns:
        Extractor instance or None if creation fails
    """
    if isinstance(file_path_or_extension, Path):
        file_path_or_extension = str(file_path_or_extension)
    
    if extractor_type:
        extractor_type = extractor_type.lower()
        
        if extractor_type == 'pdfextractor' and _COMPONENTS.get('PDFExtractor'):
            try:
                return _COMPONENTS['PDFExtractor']()
            except Exception as e:
                logger.error(f"Failed to create PDFExtractor: {e}")
        
        elif extractor_type == 'docxextractor' and _COMPONENTS.get('DOCXExtractor'):
            try:
                return _COMPONENTS['DOCXExtractor']()
            except Exception as e:
                logger.error(f"Failed to create DOCXExtractor: {e}")
        
        elif extractor_type == 'textextractor' and _COMPONENTS.get('TXTExtractor'):
            try:
                return _COMPONENTS['TXTExtractor']()
            except Exception as e:
                logger.error(f"Failed to create TXTExtractor: {e}")
        
        else:
            logger.warning(f"Unknown or unavailable extractor type: {extractor_type}")
    
    if file_path_or_extension.startswith('.'):
        extension = file_path_or_extension.lower()
    else:
        path = Path(file_path_or_extension)
        extension = path.suffix.lower()
    
    return get_extractor(extension)


def get_supported_extensions() -> List[str]:
    """
    Get list of file extensions that have extractors available.
    
    Returns:
        List of supported file extensions
    """
    return list(sorted(_EXTRACTOR_REGISTRY.keys()))


def can_process_file(file_path: Union[str, Path]) -> bool:
    """
    Check if a file can be processed based on its extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file can be processed
    """
    if isinstance(file_path, Path):
        file_path = str(file_path)
    
    path = Path(file_path)
    extension = path.suffix.lower()
    return extension in _EXTRACTOR_REGISTRY


def process_document(file_path: Union[str, Path], **kwargs) -> Dict[str, Any]:
    """
    High-level function to process a document.
    
    Args:
        file_path: Path to document file
        **kwargs: Additional processing options
        
    Returns:
        Dictionary with processing results
    """
    path = Path(file_path) if isinstance(file_path, str) else file_path
    extension = path.suffix.lower()
    
    result = {
        'success': False,
        'file_path': str(path),
        'file_name': path.name,
        'extension': extension,
        'error': None,
        'text': '',
        'metadata': {},
        'chunks': [],
        'processing_time': 0,
        'extractor_used': None
    }
    
    start_time = time.time()
    
    try:
        extractor = create_extractor(file_path)
        if not extractor:
            result['error'] = f"No extractor available for {extension} files"
            return result
        
        result['extractor_used'] = type(extractor).__name__
        
        extraction_result = extractor.extract(path)
        
        if hasattr(extraction_result, 'success') and not extraction_result.success:
            result['error'] = getattr(extraction_result, 'error', 'Extraction failed')
            return result
        
        if isinstance(extraction_result, dict):
            if 'success' in extraction_result and not extraction_result['success']:
                result['error'] = extraction_result.get('error', 'Extraction failed')
                return result
            result['text'] = extraction_result.get('text', '')
            result['metadata'] = extraction_result.get('metadata', {})
        else:
            result['text'] = str(extraction_result)
        
        if (_COMPONENT_AVAILABILITY.get('cleaning') and 
            result['text'] and 
            _COMPONENTS['TextCleaner']):
            try:
                cleaner = _COMPONENTS['TextCleaner']()
                result['text'] = cleaner.clean(result['text'])
                result['cleaning_applied'] = True
            except Exception as e:
                logger.warning(f"Text cleaning failed: {e}")
                result['cleaning_applied'] = False
        
        if (_COMPONENT_AVAILABILITY.get('chunking') and 
            result['text'] and 
            _COMPONENTS['SmartChunker']):
            try:
                chunker = _COMPONENTS['SmartChunker']()
                result['chunks'] = chunker.chunk(result['text'])
                result['chunking_applied'] = True
            except Exception as e:
                logger.warning(f"Chunking failed: {e}")
                result['chunking_applied'] = False
        
        if (_COMPONENT_AVAILABILITY.get('metadata') and 
            _COMPONENTS['extract_document_metadata']):
            try:
                additional_metadata = _COMPONENTS['extract_document_metadata'](result['text'])
                result['metadata'].update(additional_metadata)
            except Exception as e:
                logger.warning(f"Additional metadata extraction failed: {e}")
        
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"Document processing failed for {file_path}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
    
    finally:
        result['processing_time'] = time.time() - start_time
    
    return result


def get_module_status() -> Dict[str, Any]:
    """
    Get status of all document processing components.
    
    Returns:
        Dictionary with component availability and status
    """
    return {
        'base_extractor': _COMPONENT_AVAILABILITY.get('base_extractor', False),
        'docx_extractor': _COMPONENT_AVAILABILITY.get('docx_extractor', False),
        'txt_extractor': _COMPONENT_AVAILABILITY.get('txt_extractor', False),
        'pdf_extractor': _COMPONENT_AVAILABILITY.get('pdf_extractor', False),
        'epub_extractor': _COMPONENT_AVAILABILITY.get('epub_extractor', False),
        'markdown_extractor': _COMPONENT_AVAILABILITY.get('markdown_extractor', False),
        'html_extractor': _COMPONENT_AVAILABILITY.get('html_extractor', False),
        'chunking': _COMPONENT_AVAILABILITY.get('chunking', False),
        'cleaning': _COMPONENT_AVAILABILITY.get('cleaning', False),
        'metadata': _COMPONENT_AVAILABILITY.get('metadata', False),
        'supported_extensions': get_supported_extensions(),
        'extractor_registry_size': len(_EXTRACTOR_REGISTRY),
        'module_ready': all([
            _COMPONENT_AVAILABILITY.get('base_extractor', False),
            _COMPONENT_AVAILABILITY.get('pdf_extractor', False),
            len(_EXTRACTOR_REGISTRY) > 0
        ])
    }


def log_module_status():
    """Log the current status of the document processing module."""
    status = get_module_status()
    logger.info("Document Processing Module Status:")
    
    for component in [
        'base_extractor', 'docx_extractor', 'txt_extractor', 
        'pdf_extractor', 'epub_extractor', 'markdown_extractor',
        'html_extractor', 'chunking', 'cleaning', 'metadata'
    ]:
        available = status.get(component, False)
        status_text = "Available" if available else "Not available"
        logger.info(f"  {component}: {status_text}")
    
    extensions = status['supported_extensions']
    if extensions:
        logger.info(f"  Supported extensions: {', '.join(extensions)}")
    else:
        logger.warning("  Supported extensions: None")
    
    logger.info(f"  Registry size: {status['extractor_registry_size']} extractors")
    logger.info(f"  Module ready: {'Yes' if status['module_ready'] else 'No'}")


__all__ = [
    'get_extractor',
    'create_extractor',
    'get_supported_extensions',
    'can_process_file',
    'process_document',
    'get_module_status',
    'log_module_status'
]

if _COMPONENT_AVAILABILITY.get('base_extractor', False):
    __all__.append('BaseExtractor')

if _COMPONENT_AVAILABILITY.get('txt_extractor', False) and _COMPONENTS['TXTExtractor']:
    __all__.append('TXTExtractor')

if _COMPONENT_AVAILABILITY.get('docx_extractor', False) and _COMPONENTS['DOCXExtractor']:
    __all__.append('DOCXExtractor')

if _COMPONENT_AVAILABILITY.get('pdf_extractor', False) and _COMPONENTS['PDFExtractor']:
    __all__.append('PDFExtractor')

if _COMPONENT_AVAILABILITY.get('epub_extractor', False) and _COMPONENTS['EPUBExtractor']:
    __all__.append('EPUBExtractor')

if _COMPONENT_AVAILABILITY.get('markdown_extractor', False) and _COMPONENTS['MarkdownExtractor']:
    __all__.append('MarkdownExtractor')

if _COMPONENT_AVAILABILITY.get('html_extractor', False) and _COMPONENTS['HTMLExtractor']:
    __all__.append('HTMLExtractor')

if _COMPONENT_AVAILABILITY.get('chunking', False) and _COMPONENTS['ChunkingStrategy']:
    __all__.extend(['ChunkingStrategy', 'SmartChunker'])

if _COMPONENT_AVAILABILITY.get('cleaning', False) and _COMPONENTS['TextCleaner']:
    __all__.extend(['TextCleaner', 'CleaningPipeline'])

if _COMPONENT_AVAILABILITY.get('metadata', False) and _COMPONENTS['MetadataExtractor']:
    __all__.extend(['MetadataExtractor', 'extract_document_metadata'])

for component in __all__:
    if component in _COMPONENTS:
        globals()[component] = _COMPONENTS[component]