from pathlib import Path
from typing import Dict, Any, List
import os

class MockDocumentProcessor:
    def __init__(self, config=None):
        self.config = config or {}
        self.supported_formats = ['.txt', '.pdf', '.docx', '.md', '.html']
        
    def process_document(self, file_path: str) -> Dict[str, Any]:
        path = Path(file_path)
        if not path.exists():
            return {
                'status': 'error',
                'error': f'File not found: {file_path}'
            }
            
        return {
            'status': 'success',
            'document_id': f"mock_{path.stem}",
            'chunks_processed': 5,
            'processing_time': 2.5,
            'file_size': path.stat().st_size,
            'file_type': path.suffix,
            'extracted_text': f"Mock text from {path.name}...",
            'metadata': {
                'title': path.stem,
                'author': 'Unknown',
                'pages': 10,
                'language': 'en'
            }
        }
    
    def get_supported_formats(self) -> List[str]:
        return self.supported_formats
    
    def validate_file(self, file_path: str) -> bool:
        return Path(file_path).exists() and Path(file_path).suffix in self.supported_formats