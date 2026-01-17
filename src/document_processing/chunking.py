# src/document_processing/chunking.py

"""
Intelligent text chunking module for DocuBot
Implements chunking with 500 tokens and 50 token overlap
"""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class Chunk:
    """Represents a text chunk"""
    text: str
    start_pos: int
    end_pos: int
    token_count: int
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class IntelligentChunker:
    """Intelligent text chunking with overlap"""
    
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Initialize chunker
        
        Args:
            chunk_size: Target tokens per chunk (default: 500)
            chunk_overlap: Token overlap between chunks (default: 50)
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
    
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """
        Split text into intelligent chunks with overlap
        
        Args:
            text: Input text to chunk
            metadata: Optional metadata to attach to chunks
            
        Returns:
            List of Chunk objects
        """
        if not text or not text.strip():
            return []
        
        # Clean text
        cleaned_text = text.strip()
        
        # Estimate tokens (rough approximation: 1 token ≈ 4 chars for English)
        estimated_tokens = self._estimate_tokens(cleaned_text)
        
        # If text is smaller than chunk size, return as single chunk
        if estimated_tokens <= self.chunk_size:
            return [Chunk(
                text=cleaned_text,
                start_pos=0,
                end_pos=len(cleaned_text),
                token_count=estimated_tokens,
                metadata=metadata or {}
            )]
        
        # Split by paragraphs first (most natural boundary)
        paragraphs = cleaned_text.split('\n\n')
        
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue
                
            para_tokens = self._estimate_tokens(paragraph)
            
            # If paragraph itself is larger than chunk size, split further
            if para_tokens > self.chunk_size:
                # Split by sentences
                sentences = self._split_sentences(paragraph)
                for sentence in sentences:
                    sent_tokens = self._estimate_tokens(sentence)
                    if current_tokens + sent_tokens > self.chunk_size and current_chunk:
                        # Save current chunk
                        chunk_text = ' '.join(current_chunk)
                        chunks.append(Chunk(
                            text=chunk_text,
                            start_pos=self._find_position(cleaned_text, chunk_text, chunks),
                            end_pos=self._find_position(cleaned_text, chunk_text, chunks) + len(chunk_text),
                            token_count=current_tokens,
                            metadata=metadata or {}
                        ))
                        
                        # Start new chunk with overlap
                        overlap_text = ' '.join(current_chunk[-2:]) if len(current_chunk) > 1 else current_chunk[-1]
                        current_chunk = [overlap_text, sentence] if overlap_text != sentence else [sentence]
                        current_tokens = self._estimate_tokens(overlap_text) + sent_tokens
                    else:
                        current_chunk.append(sentence)
                        current_tokens += sent_tokens
            else:
                # Add paragraph to current chunk
                if current_tokens + para_tokens > self.chunk_size and current_chunk:
                    # Save current chunk
                    chunk_text = ' '.join(current_chunk)
                    chunks.append(Chunk(
                        text=chunk_text,
                        start_pos=self._find_position(cleaned_text, chunk_text, chunks),
                        end_pos=self._find_position(cleaned_text, chunk_text, chunks) + len(chunk_text),
                        token_count=current_tokens,
                        metadata=metadata or {}
                    ))
                    
                    # Start new chunk with overlap
                    overlap_text = ' '.join(current_chunk[-1:])
                    current_chunk = [overlap_text, paragraph] if overlap_text != paragraph else [paragraph]
                    current_tokens = self._estimate_tokens(overlap_text) + para_tokens
                else:
                    current_chunk.append(paragraph)
                    current_tokens += para_tokens
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                start_pos=self._find_position(cleaned_text, chunk_text, chunks),
                end_pos=self._find_position(cleaned_text, chunk_text, chunks) + len(chunk_text),
                token_count=current_tokens,
                metadata=metadata or {}
            ))
        
        return chunks
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (approximate)"""
        # Simple estimation: 1 token ≈ 4 characters for English
        # More accurate would require actual tokenizer
        words = len(text.split())
        chars = len(text)
        return max(words, chars // 4)
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _find_position(self, source_text: str, chunk_text: str, existing_chunks: List[Chunk]) -> int:
        """Find starting position of chunk in source text"""
        # Try to find exact match
        position = source_text.find(chunk_text)
        
        # If not found, estimate based on previous chunks
        if position == -1 and existing_chunks:
            last_chunk = existing_chunks[-1]
            return last_chunk.end_pos + 1  # Just after previous chunk
        
        # If not found and no previous chunks, return 0
        if position == -1:
            return 0
            
        return position 
    
    def chunk_by_tokens(self, text: str, metadata: Dict[str, Any] = None) -> List[Chunk]:
        """
        Alternative: Simple token-based chunking with overlap
        
        Args:
            text: Input text
            metadata: Optional metadata
            
        Returns:
            List of chunks
        """
        words = text.split()
        chunks = []
        start = 0
        
        while start < len(words):
            # Calculate end position with overlap
            end = min(start + self.chunk_size, len(words))
            
            # Adjust to not cut words in middle (simple approach)
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            
            chunks.append(Chunk(
                text=chunk_text,
                start_pos=self._find_position(text, chunk_text, chunks),
                end_pos=self._find_position(text, chunk_text, chunks) + len(chunk_text),
                token_count=len(chunk_words),
                metadata=metadata or {}
            ))
            
            # Move start with overlap
            start = end - min(self.chunk_overlap, end - start)
        
        return chunks

# Factory function for easy use
def create_chunker(chunk_size: int = 500, chunk_overlap: int = 50) -> IntelligentChunker:
    """Create and return a chunker instance"""
    return IntelligentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

# Test the chunker
if __name__ == "__main__":
    # Test with sample text
    sample_text = """
    This is a test document. It contains multiple paragraphs.
    
    Each paragraph should be chunked intelligently.
    The chunking should respect natural boundaries like paragraphs and sentences.
    
    The goal is to create chunks of approximately 500 tokens each,
    with 50 tokens of overlap between consecutive chunks.
    
    This ensures that context is preserved across chunks
    and the RAG system can retrieve relevant information effectively.
    """
    
    chunker = IntelligentChunker(chunk_size=500, chunk_overlap=50)
    chunks = chunker.chunk_text(sample_text, {"source": "test_doc"})
    
    print(f"Created {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks):
        print(f"\nChunk {i+1}:")
        print(f"  Tokens: {chunk.token_count}")
        print(f"  Text preview: {chunk.text[:100]}...")