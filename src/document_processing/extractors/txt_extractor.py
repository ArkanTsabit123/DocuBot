"""
Plain Text Document Extractor
"""

from pathlib import Path
from typing import Dict, Any
import logging

from .base_extractor import TextExtractor

logger = logging.getLogger(__name__)


class TXTExtractor(TextExtractor):
    """Plain text document extractor"""
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.txt', '.text', '.log', '.csv', '.tsv']
        self.name = 'TXTExtractor'
    
    def extract(self, file_path: Path) -> str:
        """
        Extract text from plain text file.
        
        Args:
            file_path: Path to text file
            
        Returns:
            Extracted text
        """
        self.validate_file(file_path)
        
        encoding = self.detect_encoding(file_path)
        
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                text = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text = f.read()
        
        logger.info(f"Extracted {len(text)} characters from text file: {file_path.name}")
        return text
    
    def get_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract text file metadata.
        
        Args:
            file_path: Path to text file
            
        Returns:
            Dictionary with metadata
        """
        metadata = super().get_metadata(file_path)
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                lines = content.split('
')
                words = content.split()
                
                metadata.update({
                    'line_count': len(lines),
                    'word_count': len(words),
                    'character_count': len(content),
                    'encoding': self.detect_encoding(file_path)
                })
                
                preview_lines = lines[:5]
                metadata['preview'] = '
'.join(preview_lines)
                
        except Exception as e:
            logger.warning(f"Could not analyze text file: {e}")
        
        return metadata


from ..extractors import register_extractor
register_extractor(TXTExtractor, ['.txt', '.text', '.log'])


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        extractor = TXTExtractor()
        result = extractor.extract_with_metadata(Path(sys.argv[1]))
        print("Extraction result:", {
            'success': result['success'],
            'text_length': len(result['text']),
            'metadata': result['metadata']
        })
    else:
        print("Usage: python txt_extractor.py <text_file>")
