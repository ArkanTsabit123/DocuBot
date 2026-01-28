# docubot/src/document_processing/extractors/__init__.py

"""
Document Processing Module - DocuBot

Main entry point for document processing functionality.
Only completed and verified components are imported to avoid circular dependencies.
"""

import sys
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class PlaceholderBaseExtractor:
    """Placeholder BaseExtractor when the actual module is not available."""
    def __init__(self):
        self.supported_formats = []
    
    def extract(self, file_path):
        raise NotImplementedError("BaseExtractor module is not loaded")
    
    def can_extract(self, file_path):
        return False


_COMPONENT_AVAILABILITY = {}
_EXTRACTOR_REGISTRY = {}
_COMPONENTS = {}


try:
    from .extractors.base_extractor import BaseExtractor
    _COMPONENTS['BaseExtractor'] = BaseExtractor
    _COMPONENT_AVAILABILITY['base_extractor'] = True
    logger.debug("BaseExtractor imported successfully")
except ImportError as e:
    _COMPONENTS['BaseExtractor'] = PlaceholderBaseExtractor
    _COMPONENT_AVAILABILITY['base_extractor'] = False
    logger.warning(f"BaseExtractor not available: {e}")

try:
    from .extractors.docx_extractor import DOCXExtractor, create_docx_extractor
    _COMPONENTS['DOCXExtractor'] = DOCXExtractor
    _COMPONENTS['create_docx_extractor'] = create_docx_extractor
    _COMPONENT_AVAILABILITY['docx_extractor'] = True
    logger.debug("DOCXExtractor imported successfully")
except ImportError as e:
    _COMPONENTS['DOCXExtractor'] = None
    _COMPONENTS['create_docx_extractor'] = None
    _COMPONENT_AVAILABILITY['docx_extractor'] = False
    logger.warning(f"DOCXExtractor not available: {e}")

try:
    from .extractors.text_extractor import TextExtractor
    _COMPONENTS['TextExtractor'] = TextExtractor
    _COMPONENT_AVAILABILITY['text_extractor'] = True
    logger.debug("TextExtractor imported successfully")
except ImportError:
    _COMPONENTS['TextExtractor'] = None
    _COMPONENT_AVAILABILITY['text_extractor'] = False

try:
    from .extractors.pdf_extractor import PDFExtractor
    _COMPONENTS['PDFExtractor'] = PDFExtractor
    _COMPONENT_AVAILABILITY['pdf_extractor'] = True
    logger.debug("PDFExtractor imported successfully")
except ImportError:
    _COMPONENTS['PDFExtractor'] = None
    _COMPONENT_AVAILABILITY['pdf_extractor'] = False

try:
    from .processor import DocumentProcessor
    _COMPONENTS['DocumentProcessor'] = DocumentProcessor
    _COMPONENT_AVAILABILITY['processor'] = True
    logger.debug("DocumentProcessor imported successfully")
except ImportError:
    _COMPONENTS['DocumentProcessor'] = None
    _COMPONENT_AVAILABILITY['processor'] = False

try:
    from .chunking import ChunkingStrategy, SmartChunker
    _COMPONENTS['ChunkingStrategy'] = ChunkingStrategy
    _COMPONENTS['SmartChunker'] = SmartChunker
    _COMPONENT_AVAILABILITY['chunking'] = True
    logger.debug("Chunking module imported successfully")
except ImportError:
    _COMPONENTS['ChunkingStrategy'] = None
    _COMPONENTS['SmartChunker'] = None
    _COMPONENT_AVAILABILITY['chunking'] = False

try:
    from .cleaning import TextCleaner, CleaningPipeline
    _COMPONENTS['TextCleaner'] = TextCleaner
    _COMPONENTS['CleaningPipeline'] = CleaningPipeline
    _COMPONENT_AVAILABILITY['cleaning'] = True
    logger.debug("Text cleaning module imported successfully")
except ImportError:
    _COMPONENTS['TextCleaner'] = None
    _COMPONENTS['CleaningPipeline'] = None
    _COMPONENT_AVAILABILITY['cleaning'] = False

try:
    from .metadata import MetadataExtractor, extract_document_metadata
    _COMPONENTS['MetadataExtractor'] = MetadataExtractor
    _COMPONENTS['extract_document_metadata'] = extract_document_metadata
    _COMPONENT_AVAILABILITY['metadata'] = True
    logger.debug("Metadata module imported successfully")
except ImportError:
    _COMPONENTS['MetadataExtractor'] = None
    _COMPONENTS['extract_document_metadata'] = None
    _COMPONENT_AVAILABILITY['metadata'] = False


if _COMPONENT_AVAILABILITY.get('docx_extractor') and _COMPONENTS['create_docx_extractor']:
    docx_extensions = ['.docx', '.docm', '.dotx', '.dotm']
    for ext in docx_extensions:
        _EXTRACTOR_REGISTRY[ext] = _COMPONENTS['create_docx_extractor']

if _COMPONENT_AVAILABILITY.get('text_extractor') and _COMPONENTS['TextExtractor']:
    text_extensions = ['.txt', '.md', '.json']
    for ext in text_extensions:
        _EXTRACTOR_REGISTRY[ext] = _COMPONENTS['TextExtractor']

if _COMPONENT_AVAILABILITY.get('pdf_extractor') and _COMPONENTS['PDFExtractor']:
    _EXTRACTOR_REGISTRY['.pdf'] = _COMPONENTS['PDFExtractor']


def get_extractor(extension: str) -> Optional[PlaceholderBaseExtractor]:
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
    
    extractor_class_or_factory = _EXTRACTOR_REGISTRY[extension]
    
    try:
        if callable(extractor_class_or_factory):
            return extractor_class_or_factory()
        else:
            return extractor_class_or_factory()
    except Exception as e:
        logger.error(f"Failed to create extractor for {extension}: {e}")
        return None


def get_supported_extensions() -> list:
    """
    Get list of file extensions that have extractors available.
    
    Returns:
        List of supported file extensions
    """
    return list(_EXTRACTOR_REGISTRY.keys())


def can_process_file(file_path: str) -> bool:
    """
    Check if a file can be processed based on its extension.
    
    Args:
        file_path: Path to the file
        
    Returns:
        True if the file can be processed
    """
    from pathlib import Path
    path = Path(file_path)
    extension = path.suffix.lower()
    return extension in _EXTRACTOR_REGISTRY


def process_document(file_path: str, **kwargs) -> Dict[str, Any]:
    """
    High-level function to process a document.
    Uses appropriate extractor and processor if available.
    
    Args:
        file_path: Path to document file
        **kwargs: Additional processing options
        
    Returns:
        Dictionary with processing results
    """
    from pathlib import Path
    
    path = Path(file_path)
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
        'processing_time': 0
    }
    
    import time
    start_time = time.time()
    
    try:
        extractor = get_extractor(extension)
        if not extractor:
            result['error'] = f"No extractor available for {extension} files"
            return result
        
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
            except Exception as e:
                logger.warning(f"Text cleaning failed: {e}")
        
        if (_COMPONENT_AVAILABILITY.get('chunking') and 
            result['text'] and 
            _COMPONENTS['SmartChunker']):
            try:
                chunker = _COMPONENTS['SmartChunker']()
                result['chunks'] = chunker.chunk(result['text'])
            except Exception as e:
                logger.warning(f"Chunking failed: {e}")
        
        result['success'] = True
        
    except Exception as e:
        result['error'] = str(e)
        logger.error(f"Document processing failed for {file_path}: {e}")
    
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
        'text_extractor': _COMPONENT_AVAILABILITY.get('text_extractor', False),
        'pdf_extractor': _COMPONENT_AVAILABILITY.get('pdf_extractor', False),
        'processor': _COMPONENT_AVAILABILITY.get('processor', False),
        'chunking': _COMPONENT_AVAILABILITY.get('chunking', False),
        'cleaning': _COMPONENT_AVAILABILITY.get('cleaning', False),
        'metadata': _COMPONENT_AVAILABILITY.get('metadata', False),
        'supported_extensions': get_supported_extensions(),
        'extractor_registry_size': len(_EXTRACTOR_REGISTRY)
    }


def log_module_status():
    """Log the current status of the document processing module."""
    status = get_module_status()
    logger.info("Document Processing Module Status:")
    
    for component in [
        'base_extractor', 'docx_extractor', 'text_extractor', 
        'pdf_extractor', 'processor', 'chunking', 'cleaning', 'metadata'
    ]:
        available = status.get(component, False)
        status_text = "Available" if available else "Not available"
        logger.info(f"  {component}: {status_text}")
    
    logger.info(f"  Supported extensions: {', '.join(status['supported_extensions']) or 'None'}")
    logger.info(f"  Registry size: {status['extractor_registry_size']} extractors")


__all__ = [
    'BaseExtractor',
    'get_extractor',
    'get_supported_extensions',
    'can_process_file',
    'process_document',
    'get_module_status',
    'log_module_status'
]

if _COMPONENT_AVAILABILITY.get('docx_extractor'):
    __all__.append('DOCXExtractor')
    __all__.append('create_docx_extractor')

if _COMPONENT_AVAILABILITY.get('text_extractor') and _COMPONENTS['TextExtractor']:
    __all__.append('TextExtractor')

if _COMPONENT_AVAILABILITY.get('pdf_extractor') and _COMPONENTS['PDFExtractor']:
    __all__.append('PDFExtractor')

if _COMPONENT_AVAILABILITY.get('processor') and _COMPONENTS['DocumentProcessor']:
    __all__.append('DocumentProcessor')

if _COMPONENT_AVAILABILITY.get('chunking'):
    __all__.extend(['ChunkingStrategy', 'SmartChunker'])

if _COMPONENT_AVAILABILITY.get('cleaning'):
    __all__.extend(['TextCleaner', 'CleaningPipeline'])

if _COMPONENT_AVAILABILITY.get('metadata'):
    __all__.extend(['MetadataExtractor', 'extract_document_metadata'])

for component in __all__:
    if component in _COMPONENTS:
        globals()[component] = _COMPONENTS[component]

if __name__ != "__main__":
    if logger.handlers:
        log_module_status()
    else:
        status = get_module_status()
        print(f"Document Processing Module loaded")
        print(f"  Supported formats: {', '.join(status['supported_extensions']) or 'None'}")
        print(f"  DOCX Extractor: {'Available' if status['docx_extractor'] else 'Not available'}")
        print(f"  Base Extractor: {'Available' if status['base_extractor'] else 'Not available'}")

del sys