"""
Base document extractor class for DocuBot.
All format-specific extractors should inherit from this class.

Version: 1.0.0
"""

import os
import re
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List
from pathlib import Path


class BaseExtractor(ABC):
    """Abstract base class for all document extractors."""
    
    def __init__(self, file_path: str):
        """
        Initialize the extractor.
        
        Args:
            file_path: Path to the document file
            
        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        self.file_path = file_path
        self.file_size = os.path.getsize(file_path)
        self.file_extension = Path(file_path).suffix.lower()
        
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the extractor (e.g., 'PDF Extractor', 'DOCX Extractor')."""
        pass
    
    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """List of supported file extensions (including dot, e.g., ['.pdf', '.txt'])."""
        pass
    
    @abstractmethod
    def extract(self, **kwargs) -> str:
        """Extract text content from the document.
        
        Args:
            **kwargs: Additional extraction parameters
            
        Returns:
            Extracted text as string
            
        Raises:
            ValueError: If file format is not supported
            IOError: If file cannot be read
            ExtractionError: If extraction fails
        """
        pass
    
    @abstractmethod
    def extract_metadata(self) -> Dict[str, Any]:
        """Extract document metadata.
        
        Returns:
            Dictionary containing metadata
        """
        pass
    
    def validate(self) -> bool:
        """Validate if the file can be processed by this extractor.
        
        Returns:
            True if file is valid and can be processed
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is empty, too large, or format not supported
        """
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        
        if self.file_size == 0:
            raise ValueError(f"File is empty: {self.file_path}")
        
        # 100MB limit by default
        MAX_FILE_SIZE = 100 * 1024 * 1024
        if self.file_size > MAX_FILE_SIZE:
            raise ValueError(f"File too large: {self.file_size:,} bytes (max: {MAX_FILE_SIZE:,})")
        
        # Check if extension is supported
        if self.file_extension not in self.supported_extensions:
            raise ValueError(
                f"File extension '{self.file_extension}' not supported by {self.name}. "
                f"Supported: {self.supported_extensions}"
            )
        
        return True
    
    def get_file_info(self) -> Dict[str, Any]:
        """Get basic file information.
        
        Returns:
            Dictionary with file information
        """
        path = Path(self.file_path)
        stat = path.stat()
        
        return {
            'file_path': str(path),
            'file_name': path.name,
            'file_extension': self.file_extension,
            'file_size_bytes': self.file_size,
            'file_size_human': self._format_size(self.file_size),
            'directory': str(path.parent),
            'last_modified': stat.st_mtime,
            'created': stat.st_ctime,
            'is_readable': os.access(self.file_path, os.R_OK),
            'is_writable': os.access(self.file_path, os.W_OK),
        }
    
    def _format_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"
    
    def __str__(self) -> str:
        """String representation of the extractor."""
        return f"{self.name} for {self.supported_extensions}"
    
    def __repr__(self) -> str:
        """Official string representation."""
        return f"{self.__class__.__name__}(file_path='{self.file_path}')"


class ExtractionError(Exception):
    """Custom exception for extraction errors."""
    
    def __init__(self, message: str, file_path: str = None, extractor: str = None):
        self.message = message
        self.file_path = file_path
        self.extractor = extractor
        super().__init__(self._format_message())
    
    def _format_message(self) -> str:
        """Format the error message."""
        parts = ["Document extraction failed"]
        if self.extractor:
            parts.append(f"using {self.extractor}")
        if self.file_path:
            parts.append(f"for file: {self.file_path}")
        parts.append(f"- Error: {self.message}")
        return " ".join(parts)


class UnsupportedFormatError(ExtractionError):
    """Exception for unsupported file formats."""
    pass


class CorruptFileError(ExtractionError):
    """Exception for corrupt or unreadable files."""
    pass
