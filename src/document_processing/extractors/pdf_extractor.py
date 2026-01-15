"""
PDF Document Extractor
"""

from pathlib import Path
from typing import Dict, Any
import logging

from .base_extractor import BaseExtractor, TextExtractor

try:
    import PyPDF2
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False
    logging.warning("PyPDF2 or pdfplumber not installed. PDF extraction disabled.")


logger = logging.getLogger(__name__)


class PDFExtractor(BaseExtractor):
    """PDF document extractor"""
    
    def __init__(self):
        super().__init__()
        self.supported_extensions = ['.pdf']
        self.name = 'PDFExtractor'
        
        if not PDF_SUPPORT:
            raise ImportError("Required packages not installed: PyPDF2, pdfplumber")
    
    def extract(self, file_path: Path) -> str:
        """
        Extract text from PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        self.validate_file(file_path)
        
        extracted_text = ""
        
        try:
            extracted_text = self._extract_with_pdfplumber(file_path)
            
            if len(extracted_text.strip()) < 100:
                logger.debug(f"pdfplumber extracted only {len(extracted_text)} chars, trying PyPDF2")
                pyPDF_text = self._extract_with_pypdf2(file_path)
                
                if len(pyPDF_text) > len(extracted_text):
                    extracted_text = pyPDF_text
            
        except Exception as e:
            logger.error(f"PDF extraction failed with pdfplumber: {e}")
            extracted_text = self._extract_with_pypdf2(file_path)
        
        if not extracted_text.strip():
            raise ValueError(f"No text could be extracted from PDF: {file_path}")
        
        logger.info(f"Extracted {len(extracted_text)} characters from PDF: {file_path.name}")
        return extracted_text
    
    def _extract_with_pdfplumber(self, file_path: Path) -> str:
        """
        Extract text using pdfplumber.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        text_parts = []
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                try:
                    page_text = page.extract_text()
                    
                    if page_text:
                        text_parts.append(f"--- Page {page_num} ---
{page_text}
")
                    
                    tables = page.extract_tables()
                    for table in tables:
                        if table:
                            table_text = self._format_table(table)
                            text_parts.append(f"
Table on page {page_num}:
{table_text}
")
                            
                except Exception as e:
                    logger.warning(f"Error extracting page {page_num}: {e}")
                    continue
        
        return "

".join(text_parts)
    
    def _extract_with_pypdf2(self, file_path: Path) -> str:
        """
        Extract text using PyPDF2.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Extracted text
        """
        text_parts = []
        
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num in range(len(pdf_reader.pages)):
                try:
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    
                    if page_text:
                        text_parts.append(f"--- Page {page_num + 1} ---
{page_text}
")
                        
                except Exception as e:
                    logger.warning(f"PyPDF2 error on page {page_num + 1}: {e}")
                    continue
        
        return "

".join(text_parts)
    
    def _format_table(self, table_data: list) -> str:
        """
        Format table data as readable text.
        
        Args:
            table_data: 2D list of table cells
            
        Returns:
            Formatted table text
        """
        if not table_data:
            return ""
        
        formatted_lines = []
        
        for row in table_data:
            row_cells = [str(cell) if cell is not None else "" for cell in row]
            formatted_lines.append(" | ".join(row_cells))
        
        return "
".join(formatted_lines)
    
    def get_metadata(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract PDF-specific metadata.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary with metadata
        """
        metadata = super().get_metadata(file_path)
        
        if PDF_SUPPORT:
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    
                    pdf_info = pdf_reader.metadata
                    if pdf_info:
                        metadata.update({
                            'pdf_title': pdf_info.get('/Title', ''),
                            'pdf_author': pdf_info.get('/Author', ''),
                            'pdf_subject': pdf_info.get('/Subject', ''),
                            'pdf_keywords': pdf_info.get('/Keywords', ''),
                            'pdf_creator': pdf_info.get('/Creator', ''),
                            'pdf_producer': pdf_info.get('/Producer', ''),
                            'pdf_creation_date': pdf_info.get('/CreationDate', ''),
                            'pdf_modification_date': pdf_info.get('/ModDate', ''),
                        })
                    
                    metadata['page_count'] = len(pdf_reader.pages)
                    
            except Exception as e:
                logger.warning(f"Could not extract PDF metadata: {e}")
        
        return metadata


if PDF_SUPPORT:
    from ..extractors import register_extractor
    register_extractor(PDFExtractor, ['.pdf'])


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        extractor = PDFExtractor()
        result = extractor.extract_with_metadata(Path(sys.argv[1]))
        print("Extraction result:", {
            'success': result['success'],
            'text_length': len(result['text']),
            'metadata': result['metadata']
        })
    else:
        print("Usage: python pdf_extractor.py <pdf_file>")
