# docubase/src/document_processing/extractors/base_extractor.py

"""
Base Document Extractor Interface
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BaseExtractor(ABC):
    def __init__(self):
        self.supported_extensions = []
        self.name = self.__class__.__name__
    
    @abstractmethod
    def extract(self, file_path: Path) -> str:
        pass
    
    def can_extract(self, file_path: Path) -> bool:
        extension = file_path.suffix.lower()
        return extension in self.supported_extensions
    
    def validate_file(self, file_path: Path) -> bool:
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_path.is_file():
            raise ValueError(f"Not a file: {file_path}")
        
        file_size = file_path.stat().st_size
        if file_size == 0:
            raise ValueError(f"File is empty: {file_path}")
        
        if file_size > 100 * 1024 * 1024:
            raise ValueError(f"File too large: {file_size} bytes")
        
        return True
    
    def get_metadata(self, file_path: Path) -> Dict[str, Any]:
        stat = file_path.stat()
        return {
            'file_name': file_path.name,
            'file_path': str(file_path),
            'file_size': stat.st_size,
            'created_time': stat.st_ctime,
            'modified_time': stat.st_mtime,
            'file_extension': file_path.suffix.lower(),
            'file_type': self.name.replace('Extractor', '')
        }
    
    def extract_with_metadata(self, file_path: Path) -> Dict[str, Any]:
        try:
            self.validate_file(file_path)
            
            metadata = self.get_metadata(file_path)
            text = self.extract(file_path)
            
            return {
                'success': True,
                'text': text,
                'metadata': metadata,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Extraction failed for {file_path}: {e}")
            return {
                'success': False,
                'text': '',
                'metadata': self.get_metadata(file_path),
                'error': str(e)
            }


class TextExtractor(BaseExtractor):
    def __init__(self):
        super().__init__()
        self.encoding = 'utf-8'
    
    def extract(self, file_path: Path) -> str:
        with open(file_path, 'r', encoding=self.encoding) as f:
            return f.read()
    
    def detect_encoding(self, file_path: Path) -> str:
        import chardet
        
        with open(file_path, 'rb') as f:
            raw_data = f.read(10000)
        
        result = chardet.detect(raw_data)
        return result.get('encoding', 'utf-8')
