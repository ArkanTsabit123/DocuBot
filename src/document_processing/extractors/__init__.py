"""
Document Extractors Package
"""

from .base_extractor import BaseExtractor, TextExtractor

_EXTRACTOR_REGISTRY = {}


def register_extractor(extractor_class, extensions):
    global _EXTRACTOR_REGISTRY
    for ext in extensions:
        _EXTRACTOR_REGISTRY[ext.lower()] = extractor_class


def get_extractor(file_extension):
    return _EXTRACTOR_REGISTRY.get(file_extension.lower())


def get_supported_extensions():
    return list(_EXTRACTOR_REGISTRY.keys())


def create_extractor(file_path):
    from pathlib import Path
    
    file_path = Path(file_path)
    extension = file_path.suffix.lower()
    
    extractor_class = get_extractor(extension)
    if extractor_class:
        return extractor_class()
    
    return None


try:
    from .txt_extractor import TXTExtractor
    register_extractor(TXTExtractor, ['.txt', '.text'])
except ImportError:
    pass

try:
    from .pdf_extractor import PDFExtractor
    register_extractor(PDFExtractor, ['.pdf'])
except ImportError:
    pass

try:
    from .docx_extractor import DOCXExtractor
    register_extractor(DOCXExtractor, ['.docx', '.doc'])
except ImportError:
    pass

__all__ = [
    'BaseExtractor',
    'TextExtractor',
    'register_extractor',
    'get_extractor',
    'get_supported_extensions',
    'create_extractor'
]
