# docubot/src/document_processing/extractors/txt_extractor.py
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
        
        # Detect encoding
        encoding = self.detect_encoding(file_path)
        
        try:
            # Try with detected encoding
            with open(file_path, 'r', encoding=encoding, errors='replace') as f:
                text = f.read()
        except (UnicodeDecodeError, LookupError):
            # Fallback to utf-8 with ignore errors
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read()
            except Exception as e:
                logger.error(f"Failed to read file {file_path} even with utf-8: {e}")
                text = ""
        
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
            # Read the file for analysis
            encoding = self.detect_encoding(file_path)
            try:
                with open(file_path, 'r', encoding=encoding, errors='ignore') as f:
                    content = f.read()
            except:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
            
            lines = content.split('\n')
            words = content.split()
            
            metadata.update({
                'line_count': len(lines),
                'word_count': len(words),
                'character_count': len(content),
                'encoding': encoding,
                'has_content': len(content.strip()) > 0
            })
            
            # Add preview (first 5 non-empty lines)
            non_empty_lines = [line.strip() for line in lines if line.strip()]
            preview_lines = non_empty_lines[:5]
            metadata['preview'] = '\n'.join(preview_lines)
            
        except Exception as e:
            logger.warning(f"Could not analyze text file {file_path}: {e}")
        
        return metadata


# Test code
if __name__ == "__main__":
    import sys
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        file_path = Path(sys.argv[1])
        
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            sys.exit(1)
        
        # Create extractor
        extractor = TXTExtractor()
        
        # Check if file is supported
        if not extractor.can_extract(file_path):
            print(f"Error: File extension {file_path.suffix} not supported")
            print(f"Supported extensions: {extractor.supported_extensions}")
            sys.exit(1)
        
        # Extract text and metadata
        result = extractor.extract_with_metadata(file_path)
        
        if result['success']:
            text = result['text']
            metadata = result['metadata']
            
            print("=" * 60)
            print(f"EXTRACTION SUCCESSFUL")
            print("=" * 60)
            print(f"File: {metadata.get('file_name', 'Unknown')}")
            print(f"Size: {metadata.get('file_size', 0)} bytes")
            print(f"Text length: {len(text)} characters")
            print(f"Lines: {metadata.get('line_count', 'N/A')}")
            print(f"Words: {metadata.get('word_count', 'N/A')}")
            print(f"Encoding: {metadata.get('encoding', 'N/A')}")
            print(f"Has content: {metadata.get('has_content', False)}")
            print("\nPreview:")
            print("-" * 40)
            preview = metadata.get('preview', 'No preview available')
            if len(preview) > 500:
                preview = preview[:497] + "..."
            print(preview)
            print("-" * 40)
            
            # Show sample of text
            if text:
                print("\nFirst 300 characters of text:")
                print("-" * 40)
                sample = text[:300] + ("..." if len(text) > 300 else "")
                print(sample)
            
        else:
            print("EXTRACTION FAILED")
            print(f"Error: {result['error']}")
            
    else:
        # Show usage and test with sample
        print("TXT Extractor - DocuBot")
        print("=" * 60)
        print("Usage: python txt_extractor.py <text_file>")
        print(f"\nSupported extensions: {TXTExtractor().supported_extensions}")
        
        # Create a test file
        test_content = """DocuBot TXT Extractor Test File

This is a test document to demonstrate the TXT extractor capabilities.

Features:
✓ Automatic encoding detection
✓ Metadata extraction
✓ Error handling for corrupted files
✓ Support for multiple text formats (.txt, .log, .csv, .tsv)

Date: January 2024
Status: Working
"""

        test_file = Path("test_document.txt")
        try:
            with open(test_file, 'w', encoding='utf-8') as f:
                f.write(test_content)
            
            print(f"\nCreated test file: {test_file}")
            
            # Test extraction
            extractor = TXTExtractor()
            result = extractor.extract_with_metadata(test_file)
            
            if result['success']:
                print("✓ Test extraction successful!")
                print(f"  Text length: {len(result['text'])} characters")
                print(f"  Lines: {result['metadata'].get('line_count')}")
                print(f"  Words: {result['metadata'].get('word_count')}")
            else:
                print(f"✗ Test extraction failed: {result['error']}")
                
        finally:
            # Clean up
            if test_file.exists():
                test_file.unlink()
                print(f"\nCleaned up test file.")