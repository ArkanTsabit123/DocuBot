# src/document_processing/chunking.py
"""
Text chunking strategies for document processing.
"""

import re
from typing import List, Dict, Any, Optional, Union
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Enum for chunking strategies."""
    RECURSIVE = "recursive"
    SEPARATOR = "separator" 
    SENTENCE = "sentence"
    FIXED = "fixed"


@dataclass
class Chunk:
    """Represents a text chunk with metadata."""
    text: str
    chunk_index: int
    start_position: int
    end_position: int
    token_count: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TextChunker:
    """Main text chunking class with multiple strategies."""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk_text(
        self, 
        text: str, 
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separator: str = "\n\n"
    ) -> List[Dict[str, Any]]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: Input text to chunk
            chunk_size: Size of each chunk in tokens (default: self.chunk_size)
            chunk_overlap: Overlap between chunks in tokens (default: self.chunk_overlap)
            separator: Text separator to preserve boundaries
            
        Returns:
            List of chunk dictionaries
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
        if chunk_overlap is None:
            chunk_overlap = self.chunk_overlap
        
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        
        # Simple token estimation (words)
        words = text.split()
        total_tokens = len(words)
        
        if total_tokens <= chunk_size:
            return [{
                'text': text,
                'token_count': total_tokens,
                'chunk_index': 0,
                'is_last': True
            }]
        
        chunks = []
        start_idx = 0
        
        while start_idx < total_tokens:
            end_idx = min(start_idx + chunk_size, total_tokens)
            
            # Get chunk text
            chunk_words = words[start_idx:end_idx]
            chunk_text = ' '.join(chunk_words)
            
            # Clean up whitespace
            chunk_text = re.sub(r'\s+', ' ', chunk_text).strip()
            
            if chunk_text:  # Only add non-empty chunks
                chunks.append({
                    'text': chunk_text,
                    'token_count': len(chunk_words),
                    'chunk_index': len(chunks),
                    'is_last': end_idx >= total_tokens,
                    'start_token': start_idx,
                    'end_token': end_idx
                })
            
            # Move start index, accounting for overlap
            if end_idx >= total_tokens:
                break
            start_idx = end_idx - chunk_overlap
        
        return chunks
    
    def chunk_by_separator(
        self, 
        text: str, 
        separator: str = "\n\n",
        max_chunk_size: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Chunk text by natural separators (paragraphs, sentences).
        
        Args:
            text: Input text
            separator: Separator to split on
            max_chunk_size: Maximum chunk size in tokens
            
        Returns:
            List of chunk dictionaries
        """
        if not text.strip():
            return []
        
        # Split by separator
        segments = [seg.strip() for seg in text.split(separator) if seg.strip()]
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for segment in segments:
            segment_tokens = len(segment.split())
            
            # If segment itself is too large, split it
            if segment_tokens > max_chunk_size:
                if current_chunk:
                    chunks.append(self._create_chunk_from_segments(current_chunk, len(chunks)))
                    current_chunk = []
                    current_size = 0
                
                # Split the large segment
                sub_chunks = self.chunk_text(segment, max_chunk_size, 0)
                chunks.extend(sub_chunks)
                continue
            
            # Add segment to current chunk if it fits
            if current_size + segment_tokens <= max_chunk_size:
                current_chunk.append(segment)
                current_size += segment_tokens
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(self._create_chunk_from_segments(current_chunk, len(chunks)))
                
                current_chunk = [segment]
                current_size = segment_tokens
        
        # Add last chunk if any
        if current_chunk:
            chunks.append(self._create_chunk_from_segments(current_chunk, len(chunks)))
        
        return chunks
    
    def _create_chunk_from_segments(self, segments: List[str], index: int) -> Dict[str, Any]:
        """Helper to create chunk from segments."""
        text = "\n\n".join(segments)
        return {
            'text': text,
            'token_count': len(text.split()),
            'chunk_index': index,
            'segment_count': len(segments)
        }
    
    def chunk_by_sentences(
        self, 
        text: str, 
        sentences_per_chunk: int = 10,
        max_chunk_size: int = 500
    ) -> List[Dict[str, Any]]:
        """
        Chunk text by sentences.
        
        Args:
            text: Input text
            sentences_per_chunk: Target sentences per chunk
            max_chunk_size: Maximum chunk size in tokens
            
        Returns:
            List of chunk dictionaries
        """
        # Simple sentence splitting (improve with nltk if available)
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_sentences = 0
        
        for sentence in sentences:
            sentence_tokens = len(sentence.split())
            
            if sentence_tokens > max_chunk_size:
                # Sentence is too large, split it
                if current_chunk:
                    chunks.append(self._create_chunk_from_segments(current_chunk, len(chunks)))
                    current_chunk = []
                    current_sentences = 0
                
                sub_chunks = self.chunk_text(sentence, max_chunk_size, 0)
                chunks.extend(sub_chunks)
                continue
            
            if current_sentences < sentences_per_chunk and (
                sum(len(s.split()) for s in current_chunk) + sentence_tokens <= max_chunk_size
            ):
                current_chunk.append(sentence)
                current_sentences += 1
            else:
                if current_chunk:
                    chunks.append(self._create_chunk_from_segments(current_chunk, len(chunks)))
                
                current_chunk = [sentence]
                current_sentences = 1
        
        if current_chunk:
            chunks.append(self._create_chunk_from_segments(current_chunk, len(chunks)))
        
        return chunks


# Factory function for creating chunkers
def create_chunker(
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separator: str = "\n\n"
) -> TextChunker:
    """
    Factory function to create chunker with specific strategy.
    
    Args:
        strategy: Chunking strategy
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        separator: Separator for boundary preservation
        
    Returns:
        Configured TextChunker instance
    """
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return chunker


def chunk_text(
    text: str, 
    chunk_size: int = 500, 
    chunk_overlap: int = 50,
    separator: str = "\n\n",
    strategy: Union[str, ChunkingStrategy] = "recursive"
) -> List[Dict[str, Any]]:
    """
    Main function to chunk text (exported for import).
    
    Args:
        text: Text to chunk
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        separator: Separator for boundary preservation
        strategy: Chunking strategy ('recursive', 'separator', 'sentence')
        
    Returns:
        List of chunk dictionaries
    """
    # Convert string to enum if needed
    if isinstance(strategy, str):
        try:
            strategy = ChunkingStrategy(strategy.lower())
        except ValueError:
            strategy = ChunkingStrategy.RECURSIVE
    
    chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    if strategy == ChunkingStrategy.SEPARATOR:
        return chunker.chunk_by_separator(text, separator, chunk_size)
    elif strategy == ChunkingStrategy.SENTENCE:
        return chunker.chunk_by_sentences(text, max_chunk_size=chunk_size)
    else:  # recursive (default)
        return chunker.chunk_text(text, chunk_size, chunk_overlap, separator)

class FixedSizeChunker(TextChunker):
    """
    Fixed-size chunker that splits text into equal-sized chunks.
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 0):
        super().__init__(chunk_size, chunk_overlap)
    
    def chunk_text(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separator: str = "\n\n"
    ) -> List[Dict[str, Any]]:
        """
        Fixed-size chunking with minimal overlap.
        """
        if chunk_size is None:
            chunk_size = self.chunk_size
        if chunk_overlap is None:
            chunk_overlap = self.chunk_overlap
        
        # Simple fixed-size splitting
        words = text.split()
        chunks = []
        
        step = chunk_size - chunk_overlap
        if step <= 0:
            step = chunk_size  # Fallback if overlap too large
        
        for i in range(0, len(words), step):
            chunk_words = words[i:i + chunk_size]
            if chunk_words:  # Only add non-empty chunks
                chunk_text = ' '.join(chunk_words)
                chunks.append({
                    'text': chunk_text,
                    'token_count': len(chunk_words),
                    'chunk_index': len(chunks),
                    'is_last': i + chunk_size >= len(words)
                })
        
        return chunks

class SemanticChunker(TextChunker):
    """
    Semantic chunker that splits text based on semantic boundaries.
    Uses embeddings to find natural break points.
    """
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        super().__init__(chunk_size, chunk_overlap)
        self.embedding_model = None
    
    def chunk_text(
        self,
        text: str,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separator: str = "\n\n"
    ) -> List[Dict[str, Any]]:
        """
        Semantic chunking using embeddings (simplified version).
        Falls back to recursive chunking if embeddings not available.
        """
        try:
            # Try to use semantic chunking if sentence-transformers available
            from sentence_transformers import SentenceTransformer
            import numpy as np
            
            if self.embedding_model is None:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Simple implementation: chunk by paragraphs with semantic scoring
            paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
            
            if len(paragraphs) <= 1:
                return super().chunk_text(text, chunk_size, chunk_overlap, separator)
            
            # Get embeddings for paragraphs
            embeddings = self.embedding_model.encode(paragraphs)
            
            # Calculate similarities between consecutive paragraphs
            similarities = []
            for i in range(len(embeddings) - 1):
                sim = np.dot(embeddings[i], embeddings[i + 1]) / (
                    np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i + 1])
                )
                similarities.append(sim)
            
            # Find low similarity points (natural breaks)
            chunks = []
            current_chunk = []
            
            for i, para in enumerate(paragraphs):
                current_chunk.append(para)
                
                # Check if this is a break point
                if i < len(similarities) and similarities[i] < 0.5:
                    if current_chunk:
                        chunk_text = "\n\n".join(current_chunk)
                        if len(chunk_text.split()) > (chunk_size or self.chunk_size):
                            # If too large, split recursively
                            sub_chunks = super().chunk_text(
                                chunk_text, chunk_size, chunk_overlap, separator
                            )
                            chunks.extend(sub_chunks)
                        else:
                            chunks.append({
                                'text': chunk_text,
                                'token_count': len(chunk_text.split()),
                                'chunk_index': len(chunks),
                                'is_last': False,
                                'semantic_boundary': True
                            })
                        current_chunk = []
            
            # Add last chunk
            if current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append({
                    'text': chunk_text,
                    'token_count': len(chunk_text.split()),
                    'chunk_index': len(chunks),
                    'is_last': True,
                    'semantic_boundary': False
                })
            
            # Update is_last flag
            if chunks:
                chunks[-1]['is_last'] = True
            
            return chunks
            
        except ImportError:
            # Fall back to regular chunking if sentence-transformers not available
            logger.warning("sentence-transformers not available, using recursive chunking")
            return super().chunk_text(text, chunk_size, chunk_overlap, separator)


# Update __all__ to include SemanticChunker
__all__ = [
    'chunk_text', 
    'TextChunker', 
    'Chunk', 
    'ChunkingStrategy', 
    'create_chunker',
    'SemanticChunker'
]