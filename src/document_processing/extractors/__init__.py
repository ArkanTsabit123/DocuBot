# docubot/src/document_processing/__init__.py
"""
Document Processing Module - Corrected __init__.py
"""

# Import extractors dari folder yang benar
from .extractors.base_extractor import BaseExtractor
from .extractors.text_extractor import TextExtractor
from .extractors.pdf_extractor import PDFExtractor
from .extractors.docx_extractor import DOCXExtractor

# Import module utama
from .processor import DocumentProcessor
from .chunking import ChunkingStrategy
from .cleaning import TextCleaner
from .metadata import MetadataExtractor

__all__ = [
    'BaseExtractor',
    'TextExtractor',
    'PDFExtractor',
    'DOCXExtractor',
    'DocumentProcessor',
    'ChunkingStrategy',
    'TextCleaner',
    'MetadataExtractor'
]