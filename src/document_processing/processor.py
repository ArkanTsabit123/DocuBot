"""
Main Document Processing Pipeline
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

from .extractors import create_extractor, get_supported_extensions
from .cleaning import clean_text_pipeline
from .chunking import chunk_text
from ..ai_engine.embedding_service import EmbeddingService
from ..vector_store.chroma_client import ChromaClient
from ..database.sqlite_client import SQLiteClient


logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Main document processing orchestrator"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize document processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {
            'chunk_size': 500,
            'chunk_overlap': 50,
            'max_file_size_mb': 100,
            'supported_formats': ['.pdf', '.docx', '.txt', '.epub', '.md']
        }
        
        self.embedding_service = EmbeddingService()
        self.vector_store = ChromaClient()
        self.database = SQLiteClient()
        
        logger.info("DocumentProcessor initialized")
    
    def process_document(self, file_path: Path) -> Dict[str, Any]:
        """
        Complete document processing pipeline.
        
        Args:
            file_path: Path to document file
            
        Returns:
            Dictionary with processing results
        """
        start_time = datetime.now()
        results = {
            'success': False,
            'document_id': str(file_path),
            'error': None,
            'processing_time': 0,
            'chunks_created': 0
        }
        
        try:
            self._validate_file(file_path)
            logger.info(f"Processing document: {file_path.name}")
            
            extractor = create_extractor(file_path)
            if not extractor:
                raise ValueError(f"No extractor found for file: {file_path}")
            
            extraction_result = extractor.extract_with_metadata(file_path)
            if not extraction_result['success']:
                raise ValueError(f"Extraction failed: {extraction_result['error']}")
            
            raw_text = extraction_result['text']
            metadata = extraction_result['metadata']
            
            logger.info(f"Extracted {len(raw_text)} characters from {file_path.name}")
            
            cleaned_text = clean_text_pipeline(raw_text)
            logger.debug(f"Text cleaned: {len(cleaned_text)} characters remaining")
            
            chunks = chunk_text(
                text=cleaned_text,
                chunk_size=self.config['chunk_size'],
                chunk_overlap=self.config['chunk_overlap']
            )
            
            logger.info(f"Created {len(chunks)} chunks from document")
            
            chunk_texts = [chunk['text'] for chunk in chunks]
            embeddings = self.embedding_service.generate_embeddings(chunk_texts)
            
            vector_ids = self.vector_store.add_documents(
                texts=chunk_texts,
                embeddings=embeddings,
                metadatas=[{
                    **metadata,
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'original_file': str(file_path),
                    'processed_at': datetime.now().isoformat()
                } for i in range(len(chunks))]
            )
            
            doc_id = self.database.add_document(
                file_path=str(file_path),
                file_name=file_path.name,
                file_type=metadata['file_extension'],
                file_size=metadata['file_size'],
                chunk_count=len(chunks),
                vector_ids=vector_ids,
                metadata=metadata
            )
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            results.update({
                'success': True,
                'document_id': doc_id,
                'chunks_created': len(chunks),
                'processing_time': processing_time,
                'metadata': metadata,
                'vector_ids': vector_ids
            })
            
            logger.info(f"Successfully processed {file_path.name} in {processing_time:.2f}s")
            
        except Exception as e:
            error_msg = f"Failed to process {file_path}: {str(e)}"
            logger.error(error_msg)
            results['error'] = error_msg
        
        return results
    
    def process_batch(self, file_paths: List[Path]) -> List[Dict[str, Any]]:
        """
        Process multiple documents.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            List of processing results
        """
        results = []
        
        for file_path in file_paths:
            try:
                result = self.process_document(file_path)
                results.append(result)
            except Exception as e:
                logger.error(f"Batch processing failed for {file_path}: {e}")
                results.append({
                    'success': False,
                    'document_id': str(file_path),
                    'error': str(e)
                })
        
        return results
    
    def _validate_file(self, file_path: Path) -> bool:
        """
        Validate file before processing.
        
        Args:
            file_path: Path to validate
            
        Returns:
            True if valid
            
        Raises:
            ValueError: If file is invalid
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if not file_path.is_file():
            raise ValueError(f"Not a file: {file_path}")
        
        file_size = file_path.stat().st_size
        max_size = self.config.get('max_file_size_mb', 100) * 1024 * 1024
        
        if file_size > max_size:
            raise ValueError(f"File too large: {file_size} bytes (max: {max_size} bytes)")
        
        extension = file_path.suffix.lower()
        supported = self.config.get('supported_formats', get_supported_extensions())
        
        if extension not in supported:
            raise ValueError(f"Unsupported file format: {extension}")
        
        return True
    
    def get_supported_formats(self) -> List[str]:
        """
        Get list of supported file formats.
        
        Returns:
            List of supported file extensions
        """
        return self.config.get('supported_formats', get_supported_extensions())
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with statistics
        """
        return {
            'supported_formats': self.get_supported_formats(),
            'chunk_size': self.config.get('chunk_size', 500),
            'chunk_overlap': self.config.get('chunk_overlap', 50),
            'max_file_size_mb': self.config.get('max_file_size_mb', 100)
        }


_processor_instance = None

def get_processor(config: Optional[Dict[str, Any]] = None) -> DocumentProcessor:
    """
    Get or create DocumentProcessor instance.
    
    Args:
        config: Optional configuration
        
    Returns:
        DocumentProcessor instance
    """
    global _processor_instance
    
    if _processor_instance is None:
        _processor_instance = DocumentProcessor(config)
    
    return _processor_instance


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        test_file = Path(sys.argv[1])
        if test_file.exists():
            processor = DocumentProcessor()
            result = processor.process_document(test_file)
            print("Processing result:", result)
        else:
            print(f"File not found: {test_file}")
    else:
        print("Usage: python processor.py <file_path>")
        print("Supported formats:", get_supported_extensions())
