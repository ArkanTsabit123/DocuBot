"""
Microsoft Word DOCX Document Extractor
"""

from pathlib import Path
from typing import Dict, Any
import logging

from .base_extractor import BaseExtractor

try:
    import docx
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False
    logging.warning("python-docx not installed. DOCX extraction disabled.")


logger = logging.getLogger(__name__)


class DOCXExtractor(BaseExtractor):
    """Microsoft Word DOCX document extractor"""
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.docx', '.doc']
        self.name = 'DOCXExtractor'
        
        if not DOCX_SUPPORT:
            raise ImportError("Required package not installed: python-docx")
    
    def extract(self, file_path: Path) -> str:
        """
        Extract text from DOCX file.
        
        Args:
            file_path: Path to DOCX file
            
        Returns:
            Extracted text
        """
        self.validate_file(file_path)
        
        try:
            doc = docx.Document(file_path)
            
            text_parts = []
            for para in doc.paragraphs:
                if para.text.strip():
                    text_parts.append(para.text)
            
            for table in doc.tables:
                table_text = self._extract_table_text(table)
                if table_text:
                    text_parts.append(f"
{table_text}
")
            
            extracted_text = "

".join(text_parts)
            
            if not extracted_text.strip():
                raise ValueError(f"No text found in DOCX file: {file_path}")
            
            logger.info(f"Extracted {len(extracted_text)} characters from DOCX: {file_path.name}")
            return extracted_text
            
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            raise
    
    def _extract_table_text(self, table) -> str:
        """
        Extract text from a table.
        
        Args:
            table: docx table object
            
        Returns:
            Formatted table text
        """
        table_text = []
        
        for row in table.rows:
            row_cells = []
            for cell in row.cells:
                cell_text = cell.text.strip()
                if cell_text:
                    row_cells.append(cell_text)
            
            if row_cells:
                table_text.append(" | ".join(row_cells))
        
        return "
".join(table_text)
    
    def get_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract DOCX-specific metadata.
        
        Args:
            file_path: Path to DOCX file
            
        Returns:
            Dictionary with metadata
        """
        metadata = super().get_metadata(file_path)
        
        if DOCX_SUPPORT:
            try:
                doc = docx.Document(file_path)
                core_props = doc.core_properties
                
                metadata.update({
                    'docx_title': core_props.title or '',
                    'docx_author': core_props.author or '',
                    'docx_subject': core_props.subject or '',
                    'docx_keywords': core_props.keywords or '',
                    'docx_comments': core_props.comments or '',
                    'docx_created': core_props.created.isoformat() if core_props.created else '',
                    'docx_modified': core_props.modified.isoformat() if core_props.modified else '',
                    'docx_category': core_props.category or '',
                    'docx_paragraph_count': len(doc.paragraphs),
                    'docx_table_count': len(doc.tables),
                })
                
                word_count = sum(len(para.text.split()) for para in doc.paragraphs)
                metadata['docx_word_count'] = word_count
                
                preview = ""
                for para in doc.paragraphs[:3]:
                    if para.text.strip():
                        preview += para.text + "
"
                metadata['preview'] = preview.strip()
                
            except Exception as e:
                logger.warning(f"Could not extract DOCX metadata: {e}")
        
        return metadata


if DOCX_SUPPORT:
    from ..extractors import register_extractor
    register_extractor(DOCXExtractor, ['.docx', '.doc'])


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        extractor = DOCXExtractor()
        result = extractor.extract_with_metadata(Path(sys.argv[1]))
        print("Extraction result:", {
            'success': result['success'],
            'text_length': len(result['text']),
            'metadata': result['metadata']
        })
    else:
        print("Usage: python docx_extractor.py <docx_file>")
