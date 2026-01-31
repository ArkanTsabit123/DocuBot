# docubot/src/document_processing/metadata.py

"""
Metadata Extraction Module for DocuBot
Extracts metadata from text documents.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List
import re
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Extract metadata from text content."""
    
    def __init__(self):
        self.title_patterns = [
            r'^#\s+(.+)$',  # Markdown title
            r'^Title[:\s]+(.+)$',
            r'^<title>(.+)</title>',  # HTML title
        ]
        
        self.author_patterns = [
            r'Author[:\s]+([^\n]+)',
            r'By[:\s]+([^\n]+)',
            r'©\s*([^\n]+)',
            r'Written by[:\s]+([^\n]+)',
        ]
        
        self.date_patterns = [
            r'Date[:\s]+([^\n]+)',
            r'Created[:\s]+([^\n]+)',
            r'Published[:\s]+([^\n]+)',
            r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
            r'(\d{2}/\d{2}/\d{4})',  # DD/MM/YYYY
            r'(\d{2}-\d{2}-\d{4})',  # DD-MM-YYYY
        ]
    
    def extract(self, text: str, file_path: Path) -> Dict[str, Any]:
        """Extract basic metadata from text."""
        try:
            metadata = {
                'title': self._extract_title(text, file_path),
                'author': self._extract_author(text),
                'date': self._extract_date(text),
                'word_count': len(text.split()),
                'char_count': len(text),
                'line_count': len(text.splitlines()),
                'sentence_count': self._count_sentences(text),
                'file_name': file_path.name,
                'file_extension': file_path.suffix.lower(),
                'file_size': file_path.stat().st_size if file_path.exists() else 0,
                'language': self._detect_language(text),
                'processing_date': datetime.now().isoformat(),
                'tags': self._generate_tags(text),
            }
            return metadata
        except Exception as e:
            logger.error(f"Metadata extraction failed: {e}")
            # Return minimal metadata if extraction fails
            return self._get_minimal_metadata(text, file_path)
    
    def _extract_title(self, text: str, file_path: Path) -> str:
        """Extract title from text or use filename."""
        # Try patterns first
        for pattern in self.title_patterns:
            matches = re.findall(pattern, text, re.MULTILINE | re.IGNORECASE)
            if matches:
                return matches[0].strip()
        
        # Try first non-empty line
        lines = text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line and len(line) < 200:  # Not too long
                # Remove markdown formatting if present
                clean_line = re.sub(r'^#+\s*', '', line)  # Remove markdown headers
                clean_line = re.sub(r'\*.*?\*|_.*?_|`.*?`', '', clean_line)  # Remove formatting
                return clean_line[:100]  # Limit length
        
        # Default to filename without extension
        return file_path.stem
    
    def _extract_author(self, text: str) -> str:
        """Extract author from text patterns."""
        for pattern in self.author_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                author = match.group(1).strip()
                # Clean up author string
                author = re.sub(r'[<>\(\)\[\]\{\}]', '', author)
                return author[:100]  # Limit length
        
        return "Unknown"
    
    def _extract_date(self, text: str) -> str:
        """Extract date from text patterns."""
        for pattern in self.date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                date_str = matches[0].strip()
                # Try to parse date
                try:
                    # Normalize date format
                    date_formats = [
                        '%Y-%m-%d',
                        '%d/%m/%Y', 
                        '%d-%m-%Y',
                        '%m/%d/%Y',
                    ]
                    
                    for fmt in date_formats:
                        try:
                            dt = datetime.strptime(date_str, fmt)
                            return dt.strftime('%Y-%m-%d')
                        except ValueError:
                            continue
                    
                    # If parsing fails, return raw string
                    return date_str[:50]
                except:
                    return date_str[:50]
        
        return datetime.now().strftime("%Y-%m-%d")
    
    def _count_sentences(self, text: str) -> int:
        """Count sentences in text."""
        # Simple sentence counting
        sentences = re.split(r'[.!?]+', text)
        # Filter out empty strings
        sentences = [s.strip() for s in sentences if s.strip()]
        return len(sentences)
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection based on common words."""
        # Check for Indonesian words
        id_words = ['yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'pada', 'ini', 'itu']
        # Check for English words  
        en_words = ['the', 'and', 'to', 'of', 'in', 'that', 'is', 'for', 'it']
        
        text_lower = text.lower()
        
        id_count = sum(1 for word in id_words if word in text_lower)
        en_count = sum(1 for word in en_words if word in text_lower)
        
        if id_count > en_count and id_count > 2:
            return 'id'  # Indonesian
        elif en_count > id_count and en_count > 2:
            return 'en'  # English
        else:
            return 'unknown'
    
    def _generate_tags(self, text: str, max_tags: int = 5) -> List[str]:
        """Generate simple tags from text."""
        # Extract potential keywords (simple approach)
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        
        # Common stopwords to exclude
        stopwords = {
            'yang', 'dan', 'di', 'ke', 'dari', 'untuk', 'pada', 'ini', 'itu',
            'the', 'and', 'to', 'of', 'in', 'that', 'is', 'for', 'it', 'with',
            'this', 'are', 'was', 'were', 'have', 'has', 'had', 'been', 'being'
        }
        
        # Count word frequency
        word_freq = {}
        for word in words:
            if word not in stopwords:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top words as tags
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        tags = [word for word, freq in sorted_words[:max_tags]]
        
        return tags
    
    def _get_minimal_metadata(self, text: str, file_path: Path) -> Dict[str, Any]:
        """Return minimal metadata when extraction fails."""
        return {
            'title': file_path.stem,
            'author': 'Unknown',
            'date': datetime.now().strftime("%Y-%m-%d"),
            'word_count': len(text.split()),
            'file_name': file_path.name,
            'file_size': file_path.stat().st_size if file_path.exists() else 0,
            'processing_date': datetime.now().isoformat(),
        }


# Factory function for convenience
def create_metadata_extractor() -> MetadataExtractor:
    """Create and return a MetadataExtractor instance."""
    return MetadataExtractor()