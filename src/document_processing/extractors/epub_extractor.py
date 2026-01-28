"""
EPUB Document Extractor Module for DocuBot

Professional implementation for extracting text and metadata from EPUB files
with multiple extraction strategies, metadata preservation,
and advanced error handling. Part of DocuBot Document Processing Pipeline.
"""

import os
import re
import sys
import json
import html
import logging
import tempfile
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any, Union, Callable
from dataclasses import dataclass, asdict, field
from enum import Enum
from datetime import datetime

# Optional dependencies with graceful degradation
try:
    import ebooklib
    from ebooklib import epub
    from ebooklib.epub import EpubBook, EpubException
    EBOOKLIB_AVAILABLE = True
except ImportError:
    EBOOKLIB_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "ebooklib library not available. Advanced EPUB features will be limited."
    )

try:
    from bs4 import BeautifulSoup
    BEAUTIFULSOUP_AVAILABLE = True
except ImportError:
    BEAUTIFULSOUP_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "BeautifulSoup library not available. HTML parsing will use fallback methods."
    )


class ExtractionMethod(Enum):
    """Enumeration of EPUB extraction methodologies."""
    EBOOKLIB = "ebooklib"
    ZIPFILE = "zipfile"
    AUTO = "auto"


class EPUBError(Exception):
    """Base exception class for EPUB-related errors."""
    pass


class InvalidEPUBError(EPUBError):
    """Raised when EPUB file is invalid, corrupted, or malformed."""
    pass


class EPUBExtractionError(EPUBError):
    """Raised when EPUB extraction process fails."""
    pass


class EPUBValidationError(EPUBError):
    """Raised when EPUB file validation fails."""
    pass


@dataclass
class EPUBMetadata:
    """
    EPUB metadata container following Dublin Core standards
    with extended metadata support.
    """
    
    # Core Dublin Core fields
    title: Optional[str] = None
    subtitle: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    primary_author: Optional[str] = None
    language: Optional[str] = "en"
    publisher: Optional[str] = None
    publication_date: Optional[str] = None
    identifiers: List[str] = field(default_factory=list)
    isbn: Optional[str] = None
    description: Optional[str] = None
    subjects: List[str] = field(default_factory=list)
    rights: Optional[str] = None
    source: Optional[str] = None
    coverage: Optional[str] = None
    contributor: Optional[str] = None
    relation: Optional[str] = None
    
    # Extended metadata
    format: str = "EPUB"
    version: Optional[str] = None
    creator_tool: Optional[str] = None
    date_created: Optional[str] = None
    date_modified: Optional[str] = None
    unique_identifier: Optional[str] = None
    direction: Optional[str] = None  # ltr or rtl
    page_progression: Optional[str] = None
    
    # Processing metadata
    extraction_timestamp: Optional[str] = None
    file_size: Optional[int] = None
    file_path: Optional[str] = None
    
    def __post_init__(self):
        """Initialize post-creation metadata."""
        if not self.extraction_timestamp:
            self.extraction_timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to serializable dictionary."""
        return asdict(self)
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize metadata to JSON format."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EPUBMetadata':
        """Create metadata instance from dictionary."""
        return cls(**data)
    
    def get_dublin_core_fields(self) -> Dict[str, Any]:
        """Extract only Dublin Core standard fields."""
        dc_fields = [
            'title', 'subtitle', 'authors', 'primary_author', 'language',
            'publisher', 'publication_date', 'identifiers', 'isbn',
            'description', 'subjects', 'rights', 'source', 'coverage',
            'contributor', 'relation'
        ]
        return {field: getattr(self, field) for field in dc_fields}
    
    def is_valid(self) -> bool:
        """Check if metadata contains essential information."""
        return bool(self.title or self.authors or self.description)


class EPUBExtractionStatistics:
    """
    Statistics collector for EPUB extraction process with metrics.
    """
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.files_processed: int = 0
        self.total_bytes: int = 0
        self.text_characters: int = 0
        self.text_words: int = 0
        self.metadata_fields_extracted: int = 0
        self.errors_encountered: int = 0
        self.warnings_generated: int = 0
        self.backend_used: Optional[str] = None
        self.epub_version: Optional[str] = None
        
    def start_timing(self):
        """Start extraction timing."""
        import time
        self.start_time = time.time()
    
    def stop_timing(self):
        """Stop extraction timing."""
        import time
        self.end_time = time.time()
    
    @property
    def duration(self) -> Optional[float]:
        """Get extraction duration in seconds."""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None
    
    @property
    def processing_rate(self) -> Optional[float]:
        """Get processing rate in bytes per second."""
        if self.duration and self.duration > 0:
            return self.total_bytes / self.duration
        return None
    
    @property
    def text_extraction_rate(self) -> Optional[float]:
        """Get text extraction rate in characters per second."""
        if self.duration and self.duration > 0:
            return self.text_characters / self.duration
        return None
    
    @property
    def metadata_completeness(self) -> float:
        """Calculate metadata completeness score (0-1)."""
        if self.metadata_fields_extracted == 0:
            return 0.0
        return min(1.0, self.metadata_fields_extracted / 20)  # Assuming ~20 metadata fields
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary."""
        return {
            'duration_seconds': self.duration,
            'files_processed': self.files_processed,
            'total_bytes': self.total_bytes,
            'text_characters': self.text_characters,
            'text_words': self.text_words,
            'metadata_fields_extracted': self.metadata_fields_extracted,
            'errors_encountered': self.errors_encountered,
            'warnings_generated': self.warnings_generated,
            'backend_used': self.backend_used,
            'epub_version': self.epub_version,
            'processing_rate_bps': self.processing_rate,
            'text_extraction_rate_cps': self.text_extraction_rate,
            'metadata_completeness': self.metadata_completeness,
            'text_to_size_ratio': (self.text_characters / max(self.total_bytes, 1)) * 100,
            'error_rate_percent': (self.errors_encountered / max(self.files_processed, 1)) * 100
        }
    
    def get_summary(self) -> str:
        """Get human-readable summary of statistics."""
        stats = self.to_dict()
        summary_lines = [
            "EPUB Extraction Statistics Summary:",
            f"  Duration: {stats.get('duration_seconds', 0):.2f} seconds",
            f"  Files Processed: {stats.get('files_processed', 0)}",
            f"  Total Bytes: {stats.get('total_bytes', 0):,}",
            f"  Text Extracted: {stats.get('text_characters', 0):,} characters",
            f"  Words Extracted: {stats.get('text_words', 0):,}",
            f"  Backend Used: {stats.get('backend_used', 'Unknown')}",
            f"  Processing Rate: {stats.get('processing_rate_bps', 0):,.0f} bytes/sec",
            f"  Text Rate: {stats.get('text_extraction_rate_cps', 0):,.0f} chars/sec",
            f"  Metadata Completeness: {stats.get('metadata_completeness', 0):.1%}",
            f"  Errors: {stats.get('errors_encountered', 0)}",
            f"  Warnings: {stats.get('warnings_generated', 0)}"
        ]
        return '\n'.join(summary_lines)


class EPUBContentStructure:
    """
    Represents the hierarchical structure of EPUB content.
    """
    
    def __init__(self):
        self.chapters: List[Dict[str, Any]] = []
        self.sections: List[Dict[str, Any]] = []
        self.parts: List[Dict[str, Any]] = []
        self.toc_depth: int = 0
        self.total_items: int = 0
        self.max_depth: int = 0
        self.landmarks: List[Dict[str, Any]] = []
        self.page_list: List[Dict[str, Any]] = []
        
    def add_chapter(self, title: str, level: int, href: str, position: int):
        """Add a chapter to the structure."""
        chapter = {
            'title': title,
            'level': level,
            'href': href,
            'position': position,
            'type': 'chapter',
            'children': []
        }
        self.chapters.append(chapter)
        self.total_items += 1
        self.max_depth = max(self.max_depth, level)
        
    def add_section(self, title: str, level: int, href: str, parent_index: Optional[int] = None):
        """Add a section to the structure."""
        section = {
            'title': title,
            'level': level,
            'href': href,
            'type': 'section',
            'parent_index': parent_index
        }
        self.sections.append(section)
        self.total_items += 1
        self.max_depth = max(self.max_depth, level)
        
    def add_landmark(self, title: str, href: str, landmark_type: str):
        """Add a landmark (toc, cover, etc.)."""
        landmark = {
            'title': title,
            'href': href,
            'type': landmark_type
        }
        self.landmarks.append(landmark)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert structure to dictionary."""
        return {
            'chapters': self.chapters,
            'sections': self.sections,
            'parts': self.parts,
            'toc_depth': self.toc_depth,
            'total_items': self.total_items,
            'max_depth': self.max_depth,
            'landmarks': self.landmarks,
            'page_list': self.page_list,
            'has_toc': len(self.chapters) > 0,
            'has_landmarks': len(self.landmarks) > 0,
            'structure_complexity': self._calculate_complexity()
        }
    
    def _calculate_complexity(self) -> float:
        """Calculate structure complexity score (0-1)."""
        if self.total_items == 0:
            return 0.0
        
        depth_score = self.max_depth / 10  # Normalize depth (assume max 10)
        breadth_score = min(1.0, self.total_items / 50)  # Normalize breadth
        landmark_score = min(1.0, len(self.landmarks) / 5)  # Normalize landmarks
        
        return (depth_score + breadth_score + landmark_score) / 3


class EPUBExtractor:
    """
    Professional EPUB extractor with multiple extraction strategies,
    metadata extraction, and advanced text processing.
    
    Features:
    - Dual extraction backends (ebooklib and zipfile)
    - metadata following Dublin Core standards
    - Structural analysis and preservation
    - Advanced text cleaning and normalization
    - Detailed extraction statistics
    - Graceful degradation and fallback mechanisms
    """
    
    def __init__(self,
                 method: ExtractionMethod = ExtractionMethod.AUTO,
                 clean_text: bool = True,
                 extract_metadata: bool = True,
                 preserve_structure: bool = True,
                 max_file_size_mb: int = 500,
                 language: str = 'en',
                 logger: Optional[logging.Logger] = None):
        """
        Initialize EPUB extractor with configuration.
        
        Args:
            method: Preferred extraction methodology
            clean_text: Enable advanced text cleaning and normalization
            extract_metadata: Extract metadata
            preserve_structure: Preserve document structure elements
            max_file_size_mb: Maximum EPUB file size in MB
            language: Default language for text processing
            logger: Custom logger instance for detailed logging
        """
        self.method = method
        self.clean_text = clean_text
        self.extract_metadata = extract_metadata
        self.preserve_structure = preserve_structure
        self.max_file_size_bytes = max_file_size_mb * 1024 * 1024
        self.language = language
        
        self.logger = logger or self._configure_logger()
        self.statistics = EPUBExtractionStatistics()
        self.structure = EPUBContentStructure()
        
        # Supported file formats
        self.supported_extensions = ['.epub']
        
        # Text file extensions to process
        self.text_extensions = ['.xhtml', '.html', '.htm', '.xml', '.xht', '.xhtm']
        
        # Patterns to ignore during extraction
        self.ignore_patterns = [
            r'nav\.x?html?$', r'toc\.x?html?$', r'cover\.x?html?$',
            r'page-map\.xml$', r'stylesheet\.css$', r'.*\.css$',
            r'.*\.js$', r'.*\.svg$', r'.*\.png$', r'.*\.jpg$',
            r'.*\.jpeg$', r'.*\.gif$', r'.*\.woff$', r'.*\.ttf$',
            r'mimetype$', r'Thumbs\.db$', r'\.DS_Store$'
        ]
        
        # Compile regex patterns for efficiency
        self.ignore_regexes = [re.compile(pattern, re.IGNORECASE) 
                              for pattern in self.ignore_patterns]
        
        # HTML cleaning configuration
        self.html_tags_to_remove = [
            'script', 'style', 'nav', 'header', 'footer',
            'aside', 'noscript', 'iframe', 'object', 'embed'
        ]
        
        self.html_tags_to_unwrap = [
            'div', 'span', 'section', 'article', 'main',
            'figure', 'figcaption', 'details', 'summary'
        ]
        
        # Initialize and validate backends
        self._initialize_backends()
        
        self.logger.info(f"EPUB Extractor initialized with method: {self.method.value}")
    
    def _configure_logger(self) -> logging.Logger:
        """Configure logging for the EPUB extractor."""
        logger = logging.getLogger(f"{__name__}.EPUBExtractor")
        
        if not logger.handlers:
            # Create console handler
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            
            # Create file handler for detailed logging
            log_dir = Path('logs')
            log_dir.mkdir(exist_ok=True)
            file_handler = logging.FileHandler(
                log_dir / f'epub_extractor_{datetime.now().strftime("%Y%m%d")}.log'
            )
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)
            logger.setLevel(logging.INFO)
            logger.propagate = False
        
        return logger
    
    def _initialize_backends(self):
        """Initialize and validate extraction backends with detailed status."""
        self.backend_status = {
            'ebooklib': {
                'available': EBOOKLIB_AVAILABLE,
                'version': self._get_ebooklib_version() if EBOOKLIB_AVAILABLE else None,
                'capabilities': ['metadata', 'structure', 'toc', 'advanced_parsing']
            },
            'zipfile': {
                'available': True,
                'version': 'built-in',
                'capabilities': ['basic_extraction', 'fallback']
            },
            'beautifulsoup': {
                'available': BEAUTIFULSOUP_AVAILABLE,
                'version': self._get_beautifulsoup_version() if BEAUTIFULSOUP_AVAILABLE else None,
                'capabilities': ['html_parsing', 'text_cleaning', 'structure_preservation']
            }
        }
        
        # Log backend status
        available_backends = [name for name, status in self.backend_status.items() 
                             if status['available']]
        self.logger.info(
            f"Available extraction backends: {', '.join(available_backends)}"
        )
        
        if not self.backend_status['ebooklib']['available']:
            self.logger.warning(
                "ebooklib not available. Advanced EPUB features (metadata, TOC) will be limited."
            )
        
        if not self.backend_status['beautifulsoup']['available']:
            self.logger.warning(
                "BeautifulSoup not available. Using basic HTML parsing methods."
            )
    
    def _get_ebooklib_version(self) -> Optional[str]:
        """Get ebooklib version if available."""
        try:
            import ebooklib
            return getattr(ebooklib, '__version__', 'unknown')
        except:
            return None
    
    def _get_beautifulsoup_version(self) -> Optional[str]:
        """Get BeautifulSoup version if available."""
        try:
            from bs4 import __version__
            return __version__
        except:
            return None
    
    def can_extract(self, file_path: Union[str, Path]) -> bool:
        """
        Check if the extractor can process the given file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file can be extracted
        """
        path = Path(file_path)
        return path.suffix.lower() in self.supported_extensions
    
    def extract(self, file_path: Union[str, Path]) -> str:
        """
        Extract text content from EPUB file.
        
        Args:
            file_path: Path to EPUB file
            
        Returns:
            Extracted and cleaned text content
            
        Raises:
            FileNotFoundError: If file does not exist
            InvalidEPUBError: If file is not a valid EPUB
            EPUBExtractionError: If extraction process fails
        """
        file_path = Path(file_path).absolute()
        
        self.logger.info(f"Starting EPUB extraction: {file_path.name}")
        self.statistics.start_timing()
        
        try:
            # Validate EPUB file
            validation_result = self._validate_epub_file(file_path)
            if not validation_result['valid']:
                raise InvalidEPUBError(
                    f"EPUB validation failed: {validation_result.get('error', 'Unknown error')}"
                )
            
            # Execute extraction
            if self.method == ExtractionMethod.EBOOKLIB and self.backend_status['ebooklib']['available']:
                self.statistics.backend_used = 'ebooklib'
                extraction_result = self._extract_with_ebooklib(file_path)
            elif self.method == ExtractionMethod.ZIPFILE:
                self.statistics.backend_used = 'zipfile'
                extraction_result = self._extract_with_zipfile(file_path)
            else:
                # Auto selection
                if self.backend_status['ebooklib']['available']:
                    self.statistics.backend_used = 'ebooklib'
                    extraction_result = self._extract_with_ebooklib(file_path)
                else:
                    self.statistics.backend_used = 'zipfile'
                    extraction_result = self._extract_with_zipfile(file_path)
            
            extracted_text, metadata = extraction_result
            
            # Apply text cleaning if requested
            if self.clean_text:
                extracted_text = self._apply_advanced_text_cleaning(extracted_text)
            
            # Update statistics
            self.statistics.text_characters = len(extracted_text)
            self.statistics.text_words = len(re.findall(r'\b\w+\b', extracted_text))
            self.statistics.stop_timing()
            
            # Log extraction summary
            self._log_extraction_summary(file_path, extracted_text, metadata)
            
            return extracted_text
            
        except Exception as error:
            self.statistics.errors_encountered += 1
            self.statistics.stop_timing()
            
            error_msg = f"EPUB extraction failed for {file_path.name}: {str(error)}"
            self.logger.error(error_msg, exc_info=True)
            
            if isinstance(error, (FileNotFoundError, InvalidEPUBError)):
                raise
            else:
                raise EPUBExtractionError(error_msg) from error
    
    def extract_with_metadata(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract text content and metadata from EPUB file.
        
        Args:
            file_path: Path to EPUB file
            
        Returns:
            Dictionary containing extracted content, metadata, and processing information
        """
        file_path = Path(file_path).absolute()
        
        self.logger.info(f"Starting EPUB extraction with metadata: {file_path.name}")
        self.statistics.start_timing()
        
        try:
            # Validate file
            validation_result = self._validate_epub_file(file_path)
            
            if not validation_result['valid']:
                return {
                    'content': '',
                    'metadata': {},
                    'structure': {},
                    'statistics': self.statistics.to_dict(),
                    'file_path': str(file_path),
                    'error': f"Validation failed: {validation_result.get('error', 'Unknown')}",
                    'success': False
                }
            
            # Perform extraction
            if self.backend_status['ebooklib']['available']:
                self.statistics.backend_used = 'ebooklib'
                text_content, metadata = self._extract_with_ebooklib(file_path)
            else:
                self.statistics.backend_used = 'zipfile'
                text_content, metadata = self._extract_with_zipfile(file_path)
            
            # Clean text if requested
            if self.clean_text:
                text_content = self._apply_advanced_text_cleaning(text_content)
            
            # Extract structure if available
            structure_data = {}
            if self.preserve_structure and self.backend_status['ebooklib']['available']:
                try:
                    structure_data = self._extract_structure(file_path)
                except Exception as e:
                    self.logger.warning(f"Structure extraction failed: {e}")
                    structure_data = {'error': str(e)}
            
            # Update statistics
            self.statistics.text_characters = len(text_content)
            self.statistics.text_words = len(re.findall(r'\b\w+\b', text_content))
            if metadata:
                metadata_dict = metadata.to_dict()
                self.statistics.metadata_fields_extracted = len(
                    [v for v in metadata_dict.values() if v]
                )
            self.statistics.stop_timing()
            
            # Build result
            result = {
                'content': text_content,
                'metadata': metadata.to_dict() if metadata else {},
                'structure': structure_data,
                'statistics': self.statistics.to_dict(),
                'file_path': str(file_path),
                'file_size': file_path.stat().st_size,
                'file_format': 'EPUB',
                'validation': validation_result,
                'backend_used': self.statistics.backend_used,
                'extraction_timestamp': datetime.now().isoformat(),
                'success': True
            }
            
            self.logger.info(
                f"EPUB extraction completed: {file_path.name} "
                f"({len(text_content):,} characters, "
                f"{self.statistics.metadata_fields_extracted} metadata fields)"
            )
            
            return result
            
        except Exception as error:
            self.statistics.errors_encountered += 1
            self.statistics.stop_timing()
            
            error_msg = f"EPUB extraction failed for {file_path.name}: {str(error)}"
            self.logger.error(error_msg, exc_info=True)
            
            return {
                'content': '',
                'metadata': {},
                'structure': {},
                'statistics': self.statistics.to_dict(),
                'file_path': str(file_path),
                'error': error_msg,
                'success': False
            }
    
    def _validate_epub_file(self, file_path: Path) -> Dict[str, Any]:
        """
        EPUB file validation with detailed diagnostics.
        
        Args:
            file_path: Path to EPUB file
            
        Returns:
            Dictionary with validation results and diagnostics
        """
        validation_result = {
            'valid': False,
            'file_path': str(file_path),
            'file_size': 0,
            'epub_version': None,
            'errors': [],
            'warnings': [],
            'checks_passed': 0,
            'checks_total': 0
        }
        
        try:
            # Check 1: File exists
            validation_result['checks_total'] += 1
            if not file_path.exists():
                validation_result['errors'].append("File does not exist")
                return validation_result
            validation_result['checks_passed'] += 1
            
            # Check 2: Is a file (not directory)
            validation_result['checks_total'] += 1
            if not file_path.is_file():
                validation_result['errors'].append("Path is not a file")
                return validation_result
            validation_result['checks_passed'] += 1
            
            # Check 3: File extension
            validation_result['checks_total'] += 1
            if file_path.suffix.lower() != '.epub':
                validation_result['errors'].append(f"Invalid file extension: {file_path.suffix}")
                return validation_result
            validation_result['checks_passed'] += 1
            
            # Check 4: File size
            file_size = file_path.stat().st_size
            validation_result['file_size'] = file_size
            self.statistics.total_bytes = file_size
            
            validation_result['checks_total'] += 1
            if file_size == 0:
                validation_result['errors'].append("File is empty (0 bytes)")
                return validation_result
            validation_result['checks_passed'] += 1
            
            validation_result['checks_total'] += 1
            if file_size > self.max_file_size_bytes:
                validation_result['errors'].append(
                    f"File too large: {file_size:,} bytes > {self.max_file_size_bytes:,} bytes limit"
                )
                return validation_result
            validation_result['checks_passed'] += 1
            
            # Check 5: ZIP file structure
            validation_result['checks_total'] += 1
            if not zipfile.is_zipfile(file_path):
                validation_result['errors'].append("Not a valid ZIP file")
                return validation_result
            validation_result['checks_passed'] += 1
            
            # Check 6: EPUB-specific structure
            try:
                with zipfile.ZipFile(file_path, 'r') as epub_zip:
                    # Check mimetype
                    validation_result['checks_total'] += 1
                    if 'mimetype' not in epub_zip.namelist():
                        validation_result['errors'].append("Missing mimetype file")
                        return validation_result
                    
                    with epub_zip.open('mimetype') as f:
                        mimetype_content = f.read(100).decode('ascii', errors='ignore').strip()
                        if mimetype_content != 'application/epub+zip':
                            validation_result['errors'].append(
                                f"Invalid mimetype: {mimetype_content}"
                            )
                            return validation_result
                    validation_result['checks_passed'] += 1
                    
                    # Check container
                    validation_result['checks_total'] += 1
                    container_files = [f for f in epub_zip.namelist() 
                                     if 'container.xml' in f.lower()]
                    if not container_files:
                        validation_result['errors'].append("Missing container.xml")
                        return validation_result
                    validation_result['checks_passed'] += 1
                    
                    # Try to determine EPUB version
                    try:
                        container_data = epub_zip.read(container_files[0])
                        root = ET.fromstring(container_data)
                        namespace = {'ns': 'urn:oasis:names:tc:opendocument:xmlns:container'}
                        rootfile = root.find('.//ns:rootfile', namespace)
                        if rootfile is not None:
                            opf_path = rootfile.get('full-path')
                            if opf_path and opf_path in epub_zip.namelist():
                                opf_data = epub_zip.read(opf_path)
                                opf_root = ET.fromstring(opf_data)
                                version_attr = opf_root.get('version')
                                if version_attr:
                                    validation_result['epub_version'] = version_attr
                                    self.statistics.epub_version = version_attr
                    except:
                        pass  # Version detection is optional
                    
            except zipfile.BadZipFile as e:
                validation_result['errors'].append(f"Corrupted ZIP file: {e}")
                return validation_result
            except Exception as e:
                validation_result['errors'].append(f"EPUB structure check failed: {e}")
                return validation_result
            
            # All checks passed
            validation_result['valid'] = True
            self.logger.debug(f"EPUB validation passed: {file_path.name}")
            
        except Exception as e:
            validation_result['errors'].append(f"Validation process error: {e}")
        
        return validation_result
    
    def _extract_with_ebooklib(self, file_path: Path) -> Tuple[str, Optional[EPUBMetadata]]:
        """
        Extract EPUB content using ebooklib backend (preferred method).
        
        Args:
            file_path: Path to EPUB file
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        self.logger.debug("Using ebooklib backend for EPUB extraction")
        
        try:
            epub_book = epub.read_epub(str(file_path))
            
            # Extract metadata
            metadata = None
            if self.extract_metadata:
                metadata = self._extract_metadata(epub_book, file_path)
                self._log_metadata_extraction(metadata)
            
            # Extract text content
            text_content = self._process_ebooklib_content(epub_book)
            
            # Extract structure if requested
            if self.preserve_structure:
                self._extract_ebooklib_structure(epub_book)
            
            return text_content, metadata
            
        except EpubException as epub_error:
            self.logger.warning(f"ebooklib EPUB parsing error: {epub_error}")
            raise InvalidEPUBError(f"EPUB parsing failed: {epub_error}")
        except Exception as e:
            self.logger.warning(f"ebooklib extraction failed: {e}")
            # Fallback to zipfile extraction
            return self._extract_with_zipfile(file_path)
    
    def _extract_metadata(self, epub_book, 
                                       file_path: Path) -> EPUBMetadata:
        """
        Extract metadata from ebooklib EpubBook object.
        
        Args:
            epub_book: ebooklib EpubBook instance
            file_path: Original file path
            
        Returns:
            EPUBMetadata object
        """
        metadata = EPUBMetadata()
        metadata.file_path = str(file_path)
        metadata.file_size = file_path.stat().st_size
        
        try:
            # Core Dublin Core metadata
            metadata.title = self._get_metadata_value(epub_book, 'DC', 'title')
            metadata.subtitle = self._get_metadata_value(epub_book, 'DC', 'subtitle')
            
            # Authors/creators
            creator_data = epub_book.get_metadata('DC', 'creator')
            if creator_data:
                metadata.authors = [
                    str(creator[0]).strip()
                    for creator in creator_data
                    if creator and creator[0]
                ]
                if metadata.authors:
                    metadata.primary_author = metadata.authors[0]
            
            metadata.language = self._get_metadata_value(epub_book, 'DC', 'language')
            metadata.publisher = self._get_metadata_value(epub_book, 'DC', 'publisher')
            metadata.publication_date = self._get_metadata_value(epub_book, 'DC', 'date')
            metadata.description = self._get_metadata_value(epub_book, 'DC', 'description')
            
            # Identifiers (including ISBN)
            identifier_data = epub_book.get_metadata('DC', 'identifier')
            if identifier_data:
                metadata.identifiers = [
                    str(identifier[0]).strip()
                    for identifier in identifier_data
                    if identifier and identifier[0]
                ]
                metadata.isbn = self._extract_isbn_from_identifiers(metadata.identifiers)
                metadata.unique_identifier = self._get_unique_identifier(metadata.identifiers)
            
            # Subjects/tags
            subject_data = epub_book.get_metadata('DC', 'subject')
            if subject_data:
                metadata.subjects = [
                    str(subject[0]).strip()
                    for subject in subject_data
                    if subject and subject[0]
                ]
            
            # Rights and legal information
            metadata.rights = self._get_metadata_value(epub_book, 'DC', 'rights')
            metadata.source = self._get_metadata_value(epub_book, 'DC', 'source')
            metadata.coverage = self._get_metadata_value(epub_book, 'DC', 'coverage')
            metadata.contributor = self._get_metadata_value(epub_book, 'DC', 'contributor')
            metadata.relation = self._get_metadata_value(epub_book, 'DC', 'relation')
            
            # Extended metadata
            metadata.date_created = self._get_metadata_value(epub_book, 'DC', 'created')
            metadata.date_modified = self._get_metadata_value(epub_book, 'DC', 'modified')
            
            # EPUB-specific metadata
            try:
                # Try to get version from package attribute
                if hasattr(epub_book, 'package'):
                    metadata.version = getattr(epub_book.package, 'version', None)
                
                # Try to get direction
                metadata.direction = self._get_metadata_value(epub_book, None, 'dir')
                
                # Try to get page progression
                metadata.page_progression = self._get_metadata_value(epub_book, None, 'page-progression-direction')
                
            except Exception as e:
                self.logger.debug(f"Extended metadata extraction failed: {e}")
            
            # Set default language if not specified
            if not metadata.language:
                metadata.language = self.language
            
        except Exception as e:
            self.logger.warning(f"Metadata extraction incomplete: {e}")
        
        return metadata
    
    def _get_metadata_value(self, epub_book, namespace: str, field: str) -> Optional[str]:
        """
        Safely extract metadata value from ebooklib metadata.
        
        Args:
            epub_book: ebooklib EpubBook instance
            namespace: Metadata namespace (or None for no namespace)
            field: Metadata field name
            
        Returns:
            Metadata value or None
        """
        try:
            if namespace:
                metadata = epub_book.get_metadata(namespace, field)
            else:
                # Try to get without namespace
                metadata = None
                for ns in ['DC', 'OPF', None]:
                    try:
                        metadata = epub_book.get_metadata(ns, field) if ns else None
                        if metadata:
                            break
                    except:
                        continue
            
            if metadata and metadata[0] and metadata[0][0]:
                value = str(metadata[0][0]).strip()
                return value if value else None
        except Exception:
            pass
        
        return None
    
    def _extract_isbn_from_identifiers(self, identifiers: List[str]) -> Optional[str]:
        """
        Extract ISBN from a list of identifiers.
        
        Args:
            identifiers: List of identifier strings
            
        Returns:
            ISBN string or None
        """
        for identifier in identifiers:
            # Clean identifier for checking
            clean_id = identifier.lower().replace('-', '').replace(' ', '')
            
            # Check for ISBN pattern
            if 'isbn' in identifier.lower():
                return identifier
            
            # Check for ISBN-like numeric patterns (10 or 13 digits)
            if len(clean_id) in [10, 13] and clean_id.isdigit():
                # Additional validation for ISBN checksum
                if self._validate_isbn_checksum(clean_id):
                    return identifier
        
        return None
    
    def _validate_isbn_checksum(self, isbn: str) -> bool:
        """
        Validate ISBN checksum (both ISBN-10 and ISBN-13).
        
        Args:
            isbn: ISBN string without dashes/spaces
            
        Returns:
            True if valid ISBN checksum
        """
        if len(isbn) == 10:
            # ISBN-10 checksum validation
            total = 0
            for i in range(9):
                total += int(isbn[i]) * (10 - i)
            
            check_digit = isbn[9]
            if check_digit == 'X' or check_digit == 'x':
                check_value = 10
            else:
                check_value = int(check_digit)
            
            total += check_value
            return total % 11 == 0
            
        elif len(isbn) == 13:
            # ISBN-13 checksum validation
            total = 0
            for i in range(12):
                multiplier = 1 if i % 2 == 0 else 3
                total += int(isbn[i]) * multiplier
            
            check_digit = int(isbn[12])
            return (10 - (total % 10)) % 10 == check_digit
        
        return False
    
    def _get_unique_identifier(self, identifiers: List[str]) -> Optional[str]:
        """
        Get the unique identifier from a list of identifiers.
        
        Args:
            identifiers: List of identifier strings
            
        Returns:
            Unique identifier or None
        """
        for identifier in identifiers:
            if 'uuid' in identifier.lower() or 'urn:uuid:' in identifier.lower():
                return identifier
            if identifier.startswith('urn:') or '://' in identifier:
                return identifier
        
        return identifiers[0] if identifiers else None
    
    def _process_ebooklib_content(self, epub_book) -> str:
        """
        Process all document content from ebooklib EpubBook.
        
        Args:
            epub_book: ebooklib EpubBook instance
            
        Returns:
            Concatenated text content
        """
        text_segments = []
        
        # Get all document items
        document_items = [
            item for item in epub_book.get_items()
            if item.get_type() == ebooklib.ITEM_DOCUMENT
        ]
        
        total_items = len(document_items)
        self.logger.debug(f"Processing {total_items} document items from EPUB")
        
        for index, item in enumerate(document_items, 1):
            try:
                # Decode content
                content_bytes = item.get_content()
                content = content_bytes.decode('utf-8', errors='ignore')
                
                # Process HTML content
                processed_text = self._process_html_content(content, item.get_name())
                
                if processed_text and processed_text.strip():
                    text_segments.append(processed_text)
                    self.statistics.files_processed += 1
                
                # Progress logging
                if index % 10 == 0 and total_items > 20:
                    self.logger.debug(f"Processed {index}/{total_items} items")
                    
            except Exception as item_error:
                self.logger.warning(
                    f"Failed to process item {item.get_name()}: {item_error}"
                )
                self.statistics.errors_encountered += 1
        
        # Join all text segments
        concatenated_text = '\n\n'.join(text_segments)
        
        return concatenated_text
    
    def _extract_ebooklib_structure(self, epub_book):
        """Extract document structure from ebooklib book."""
        try:
            if hasattr(epub_book, 'toc') and epub_book.toc:
                self._process_ebooklib_toc(epub_book.toc)
            
            # Extract landmarks if available
            if hasattr(epub_book, 'guide') and epub_book.guide:
                self._process_ebooklib_guide(epub_book.guide)
                
        except Exception as e:
            self.logger.debug(f"Structure extraction failed: {e}")
    
    def _process_ebooklib_toc(self, toc, level: int = 0):
        """Process ebooklib table of contents."""
        for item in toc:
            if isinstance(item, tuple):
                # Nested structure
                link, title = item[0], item[1]
                if hasattr(link, 'href'):
                    self.structure.add_chapter(
                        title=str(title) if title else 'Untitled',
                        level=level,
                        href=link.href,
                        position=len(self.structure.chapters)
                    )
                
                # Process children if any
                if len(item) > 2:
                    self._process_ebooklib_toc(item[2], level + 1)
                    
            elif isinstance(item, epub.Link):
                # Simple link
                self.structure.add_chapter(
                    title=str(item.title) if hasattr(item, 'title') and item.title else 'Untitled',
                    level=level,
                    href=item.href,
                    position=len(self.structure.chapters)
                )
        
        self.structure.toc_depth = level
    
    def _process_ebooklib_guide(self, guide):
        """Process ebooklib guide (landmarks)."""
        for item in guide:
            if hasattr(item, 'type') and hasattr(item, 'href'):
                self.structure.add_landmark(
                    title=getattr(item, 'title', ''),
                    href=item.href,
                    landmark_type=item.type
                )
    
    def _extract_with_zipfile(self, file_path: Path) -> Tuple[str, Optional[EPUBMetadata]]:
        """
        Extract EPUB content using zipfile backend (fallback method).
        
        Args:
            file_path: Path to EPUB file
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        self.logger.debug("Using zipfile backend for EPUB extraction")
        
        text_segments = []
        metadata = EPUBMetadata() if self.extract_metadata else None
        
        try:
            with zipfile.ZipFile(file_path, 'r') as epub_zip:
                archive_contents = epub_zip.namelist()
                
                # Extract metadata if requested
                if self.extract_metadata:
                    metadata = self._extract_zipfile_metadata(epub_zip, archive_contents, file_path)
                
                # Identify text files to process
                text_files = self._identify_text_files(archive_contents)
                
                self.logger.debug(f"Found {len(text_files)} text files for processing")
                
                # Process each text file
                for filename in text_files:
                    try:
                        # Read and decode file content
                        content_bytes = epub_zip.read(filename)
                        content = content_bytes.decode('utf-8', errors='ignore')
                        
                        # Process HTML content
                        processed_text = self._process_html_content(content, filename)
                        
                        if processed_text and processed_text.strip():
                            text_segments.append(processed_text)
                            self.statistics.files_processed += 1
                            
                    except Exception as file_error:
                        self.logger.warning(f"Failed to process {filename}: {file_error}")
                        self.statistics.errors_encountered += 1
                
        except zipfile.BadZipFile as zip_error:
            raise InvalidEPUBError(f"Invalid ZIP archive: {zip_error}")
        except Exception as e:
            raise EPUBExtractionError(f"Zipfile extraction failed: {e}")
        
        concatenated_text = '\n\n'.join(text_segments)
        
        return concatenated_text, metadata
    
    def _identify_text_files(self, archive_contents: List[str]) -> List[str]:
        """
        Identify text content files in EPUB archive.
        
        Args:
            archive_contents: List of files in EPUB archive
            
        Returns:
            List of text file paths
        """
        text_files = []
        
        for filename in archive_contents:
            # Skip if matches ignore patterns
            if any(pattern.search(filename) for pattern in self.ignore_regexes):
                continue
            
            # Check if it's a text file by extension
            if any(filename.lower().endswith(ext) for ext in self.text_extensions):
                text_files.append(filename)
        
        # Sort files for consistent processing order
        text_files.sort()
        
        return text_files
    
    def _extract_zipfile_metadata(self, epub_zip: zipfile.ZipFile,
                                 archive_contents: List[str],
                                 file_path: Path) -> EPUBMetadata:
        """
        Extract metadata from EPUB zip archive.
        
        Args:
            epub_zip: Open zipfile archive
            archive_contents: List of files in archive
            file_path: Original file path
            
        Returns:
            EPUBMetadata object
        """
        metadata = EPUBMetadata()
        metadata.file_path = str(file_path)
        metadata.file_size = file_path.stat().st_size
        
        try:
            # Find container file
            container_file = next(
                (f for f in archive_contents if 'container.xml' in f.lower()),
                None
            )
            
            if container_file:
                container_data = epub_zip.read(container_file)
                rootfile_path = self._parse_container_for_rootfile(container_data)
                
                if rootfile_path and rootfile_path in archive_contents:
                    opf_data = epub_zip.read(rootfile_path)
                    metadata = self._parse_opf_metadata_xml(opf_data, metadata)
            
        except Exception as e:
            self.logger.warning(f"ZIP metadata extraction incomplete: {e}")
        
        return metadata
    
    def _parse_container_for_rootfile(self, container_data: bytes) -> Optional[str]:
        """
        Parse container.xml to find OPF rootfile path.
        
        Args:
            container_data: container.xml content
            
        Returns:
            Path to OPF file or None
        """
        try:
            root = ET.fromstring(container_data)
            namespace = {'ns': 'urn:oasis:names:tc:opendocument:xmlns:container'}
            
            rootfile = root.find('.//ns:rootfile', namespace)
            if rootfile is not None:
                return rootfile.get('full-path')
                
        except ET.ParseError as e:
            self.logger.debug(f"Failed to parse container.xml: {e}")
        
        return None
    
    def _parse_opf_metadata_xml(self, opf_data: bytes, 
                               metadata: EPUBMetadata) -> EPUBMetadata:
        """
        Parse OPF file for metadata using XML parsing.
        
        Args:
            opf_data: OPF file content
            metadata: Metadata object to update
            
        Returns:
            Updated EPUBMetadata object
        """
        try:
            root = ET.fromstring(opf_data)
            
            # Define namespaces
            namespaces = {
                'opf': 'http://www.idpf.org/2007/opf',
                'dc': 'http://purl.org/dc/elements/1.1/',
                'dcterms': 'http://purl.org/dc/terms/'
            }
            
            # Extract version
            metadata.version = root.get('version')
            if metadata.version:
                self.statistics.epub_version = metadata.version
            
            # Extract metadata elements
            metadata.title = self._extract_xml_text(root, './/dc:title', namespaces)
            
            # Extract authors
            author_elements = root.findall('.//dc:creator', namespaces)
            if author_elements:
                metadata.authors = [
                    elem.text.strip() for elem in author_elements if elem.text
                ]
                if metadata.authors:
                    metadata.primary_author = metadata.authors[0]
            
            metadata.language = self._extract_xml_text(root, './/dc:language', namespaces)
            metadata.publisher = self._extract_xml_text(root, './/dc:publisher', namespaces)
            metadata.publication_date = self._extract_xml_text(root, './/dc:date', namespaces)
            metadata.description = self._extract_xml_text(root, './/dc:description', namespaces)
            
            # Extract identifiers
            identifier_elements = root.findall('.//dc:identifier', namespaces)
            if identifier_elements:
                metadata.identifiers = [
                    elem.text.strip() for elem in identifier_elements if elem.text
                ]
                metadata.isbn = self._extract_isbn_from_identifiers(metadata.identifiers)
                metadata.unique_identifier = self._get_unique_identifier(metadata.identifiers)
            
            # Extract subjects
            subject_elements = root.findall('.//dc:subject', namespaces)
            if subject_elements:
                metadata.subjects = [
                    elem.text.strip() for elem in subject_elements if elem.text
                ]
            
            # Extract other metadata
            metadata.rights = self._extract_xml_text(root, './/dc:rights', namespaces)
            metadata.source = self._extract_xml_text(root, './/dc:source', namespaces)
            metadata.contributor = self._extract_xml_text(root, './/dc:contributor', namespaces)
            
            # Try to get date created/modified
            created_elem = root.find('.//dcterms:created', namespaces)
            if created_elem is not None and created_elem.text:
                metadata.date_created = created_elem.text.strip()
            
            modified_elem = root.find('.//dcterms:modified', namespaces)
            if modified_elem is not None and modified_elem.text:
                metadata.date_modified = modified_elem.text.strip()
            
        except ET.ParseError as e:
            self.logger.warning(f"OPF XML parsing failed: {e}")
        except Exception as e:
            self.logger.warning(f"OPF metadata parsing error: {e}")
        
        return metadata
    
    def _extract_xml_text(self, element: ET.Element,
                         xpath: str,
                         namespaces: Dict[str, str]) -> Optional[str]:
        """
        Extract text from XML element using XPath.
        
        Args:
            element: XML element to search
            xpath: XPath expression
            namespaces: XML namespace mappings
            
        Returns:
            Extracted text or None
        """
        try:
            target = element.find(xpath, namespaces)
            if target is not None and target.text:
                return target.text.strip()
        except Exception:
            pass
        
        return None
    
    def _process_html_content(self, html_content: str,
                             source_name: str = '') -> str:
        """
        Process HTML/XHTML content to extract clean text.
        
        Args:
            html_content: Raw HTML content
            source_name: Source filename for logging
            
        Returns:
            Clean extracted text
        """
        if not html_content or not html_content.strip():
            return ""
        
        if BEAUTIFULSOUP_AVAILABLE:
            return self._extract_with_beautifulsoup(html_content, source_name)
        else:
            return self._extract_with_basic_parsing(html_content)
    
    def _extract_with_beautifulsoup(self, html_content: str,
                                   source_name: str) -> str:
        """
        Extract text using BeautifulSoup with advanced processing.
        
        Args:
            html_content: Raw HTML content
            source_name: Source filename for logging
            
        Returns:
            Clean extracted text
        """
        try:
            soup = BeautifulSoup(html_content, 'lxml')
            
            # Remove unwanted elements
            for tag_name in self.html_tags_to_remove:
                for element in soup.find_all(tag_name):
                    element.decompose()
            
            # Unwrap structural elements if not preserving structure
            if not self.preserve_structure:
                for tag_name in self.html_tags_to_unwrap:
                    for element in soup.find_all(tag_name):
                        element.unwrap()
            
            # Remove footnote/annotation classes
            footnote_classes = ['note', 'footnote', 'citation', 
                               'reference', 'annotation', 'footnotes']
            for element in soup.find_all(['div', 'p', 'span']):
                if 'class' in element.attrs:
                    class_string = ' '.join(element.attrs['class']).lower()
                    if any(fn_class in class_string for fn_class in footnote_classes):
                        element.decompose()
            
            # Get text with proper spacing
            text = soup.get_text(separator=' ', strip=True)
            
            # Unescape HTML entities
            text = html.unescape(text)
            
            # Normalize whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove extra spaces around punctuation
            text = re.sub(r'\s+([.,;:!?])', r'\1', text)
            text = re.sub(r'([([{])\s+', r'\1', text)
            text = re.sub(r'\s+([)\]}])', r'\1', text)
            
            return text.strip()
            
        except Exception as e:
            self.logger.warning(
                f"BeautifulSoup extraction failed for {source_name}: {e}"
            )
            return self._extract_with_basic_parsing(html_content)
    
    def _extract_with_basic_parsing(self, html_content: str) -> str:
        """
        Basic HTML text extraction using regex (fallback method).
        
        Args:
            html_content: Raw HTML content
            
        Returns:
            Clean extracted text
        """
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', html_content)
        
        # Unescape HTML entities
        text = html.unescape(text)
        
        # Remove HTML/XML comments
        text = re.sub(r'<!--.*?-->', ' ', text, flags=re.DOTALL)
        
        # Remove script/style content if tags weren't removed
        text = re.sub(r'<script[^>]*>.*?</script>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
        
        # Decode XML/HTML entities
        text = re.sub(r'&[a-z]+;', ' ', text, flags=re.IGNORECASE)
        text = re.sub(r'&#\d+;', ' ', text)
        text = re.sub(r'&#x[0-9a-f]+;', ' ', text, flags=re.IGNORECASE)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Clean up paragraph breaks
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line and len(line) > 10]
        
        return ' '.join(lines)
    
    def _apply_advanced_text_cleaning(self, text: str) -> str:
        """
        Apply advanced text cleaning and normalization.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned and normalized text
        """
        if not text:
            return ""
        
        # Split into lines for processing
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            if not line:
                continue
            
            # Identify and format structural elements
            if self._is_structural_element(line):
                if self.preserve_structure:
                    # Add extra newlines around structural elements
                    cleaned_lines.append('\n' + line + '\n')
                else:
                    # Just add the line
                    cleaned_lines.append(line)
            else:
                cleaned_lines.append(line)
        
        # Rejoin lines
        cleaned_text = '\n'.join(cleaned_lines)
        
        # Remove excessive newlines
        cleaned_text = re.sub(r'\n{3,}', '\n\n', cleaned_text)
        
        # Fix sentence boundaries
        cleaned_text = re.sub(r'([.!?])\s+', r'\1\n', cleaned_text)
        
        # Fix spacing around punctuation
        cleaned_text = re.sub(r'\s+([.,;:!?])', r'\1', cleaned_text)
        cleaned_text = re.sub(r'([([{])\s+', r'\1', cleaned_text)
        cleaned_text = re.sub(r'\s+([)\]}])', r'\1', cleaned_text)
        
        # Remove multiple spaces
        cleaned_text = re.sub(r' {2,}', ' ', cleaned_text)
        
        # Trim whitespace
        cleaned_text = cleaned_text.strip()
        
        return cleaned_text
    
    def _is_structural_element(self, line: str) -> bool:
        """
        Check if a line is a structural element (chapter, section, etc.).
        
        Args:
            line: Text line to check
            
        Returns:
            True if line appears to be a structural element
        """
        # Check length
        if len(line) > 200:
            return False
        
        # Check for chapter/section indicators
        chapter_patterns = [
            r'^chapter\s+\d+[:.]?\s+', r'^ch\.?\s+\d+',
            r'^section\s+\d+', r'^sec\.?\s+\d+',
            r'^part\s+\d+', r'^book\s+\d+',
            r'^volume\s+\d+', r'^appendix\s+\w+'
        ]
        
        for pattern in chapter_patterns:
            if re.match(pattern, line.lower()):
                return True
        
        # Check for all caps short lines (likely headers)
        if line.isupper() and len(line) < 100 and len(line.split()) < 10:
            return True
        
        # Check for numbered headers
        if re.match(r'^\d+[\.:]\s+', line):
            return True
        
        return False
    
    def _extract_structure(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract document structure from EPUB file.
        
        Args:
            file_path: Path to EPUB file
            
        Returns:
            Dictionary with structure information
        """
        if not self.backend_status['ebooklib']['available']:
            return {'error': 'ebooklib not available for structure extraction'}
        
        try:
            epub_book = epub.read_epub(str(file_path))
            
            # Reset structure
            self.structure = EPUBContentStructure()
            
            # Extract structure
            self._extract_ebooklib_structure(epub_book)
            
            return self.structure.to_dict()
            
        except Exception as e:
            self.logger.warning(f"Structure extraction failed: {e}")
            return {'error': str(e)}
    
    def _log_metadata_extraction(self, metadata: EPUBMetadata):
        """Log metadata extraction summary."""
        if metadata.title or metadata.primary_author:
            self.logger.info(
                f"Extracted metadata: "
                f"Title: {metadata.title or 'Untitled'}, "
                f"Author: {metadata.primary_author or 'Unknown'}, "
                f"Language: {metadata.language or 'Unknown'}"
            )
    
    def _log_extraction_summary(self, file_path: Path,
                               extracted_text: str,
                               metadata: Optional[EPUBMetadata]):
        """Log extraction summary."""
        stats = self.statistics.to_dict()
        
        summary = [
            f"EPUB Extraction Summary - {file_path.name}",
            f"  Duration: {stats.get('duration_seconds', 0):.2f} seconds",
            f"  Backend: {self.statistics.backend_used}",
            f"  Text: {len(extracted_text):,} characters, {len(extracted_text.split()):,} words",
            f"  Files Processed: {stats.get('files_processed', 0)}",
            f"  Processing Rate: {stats.get('processing_rate_bps', 0):,.0f} B/s",
            f"  Text Rate: {stats.get('text_extraction_rate_cps', 0):,.0f} chars/s",
        ]
        
        if metadata:
            summary.append(f"  Metadata: {self.statistics.metadata_fields_extracted} fields extracted")
            if metadata.title:
                summary.append(f"  Title: {metadata.title}")
            if metadata.primary_author:
                summary.append(f"  Author: {metadata.primary_author}")
        
        summary.append(f"  EPUB Version: {self.statistics.epub_version or 'Unknown'}")
        summary.append(f"  Errors: {self.statistics.errors_encountered}")
        
        self.logger.info('\n'.join(summary))
    
    def analyze_extraction(self, text: str,
                          metadata: Optional[EPUBMetadata] = None) -> Dict[str, Any]:
        """
        Perform analysis of extraction results.
        
        Args:
            text: Extracted text content
            metadata: Extracted metadata
            
        Returns:
            Dictionary with detailed analysis
        """
        # Basic text statistics
        words = re.findall(r'\b\w+\b', text)
        sentences = re.split(r'[.!?]+', text)
        lines = text.split('\n')
        
        analysis = {
            'character_count': len(text),
            'character_count_no_spaces': len(text.replace(' ', '').replace('\n', '')),
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'line_count': len(lines),
            'non_empty_line_count': len([l for l in lines if l.strip()]),
            'average_word_length': (
                sum(len(w) for w in words) / len(words) if words else 0
            ),
            'average_sentence_length_words': (
                len(words) / len(sentences) if sentences else 0
            ),
            'average_sentence_length_chars': (
                len(text) / len(sentences) if sentences else 0
            ),
            'text_density': (
                len(text) / max(len(text.replace(' ', '').replace('\n', '')), 1)
            ),
            'paragraph_count': text.count('\n\n') + 1,
            'unique_word_count': len(set(word.lower() for word in words)) if words else 0,
            'lexical_diversity': (
                len(set(word.lower() for word in words)) / len(words) if words else 0
            ),
            'structural_elements': sum(1 for line in lines if self._is_structural_element(line))
        }
        
        # Metadata analysis
        if metadata:
            analysis['metadata_present'] = True
            metadata_dict = metadata.to_dict()
            analysis['metadata_fields'] = len(
                [v for v in metadata_dict.values() if v]
            )
            analysis['has_isbn'] = metadata.isbn is not None
            analysis['author_count'] = len(metadata.authors)
            analysis['subject_count'] = len(metadata.subjects)
            analysis['identifier_count'] = len(metadata.identifiers)
            analysis['metadata_completeness_score'] = analysis['metadata_fields'] / 20  # Approximate max fields
        else:
            analysis['metadata_present'] = False
            analysis['metadata_completeness_score'] = 0.0
        
        # Combine with extraction statistics
        analysis.update(self.statistics.to_dict())
        
        # Calculate quality scores
        analysis['text_quality_score'] = self._calculate_text_quality_score(text)
        analysis['extraction_quality_score'] = (
            analysis['text_quality_score'] * 0.6 +
            analysis.get('metadata_completeness_score', 0) * 0.3 +
            (1 - min(1.0, analysis.get('errors_encountered', 0) / 10)) * 0.1
        )
        
        # Add recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _calculate_text_quality_score(self, text: str) -> float:
        """
        Calculate text quality score based on various metrics.
        
        Args:
            text: Extracted text
            
        Returns:
            Quality score between 0 and 1
        """
        if not text:
            return 0.0
        
        scores = []
        
        # Length score
        if len(text) > 1000:
            scores.append(1.0)
        elif len(text) > 100:
            scores.append(len(text) / 1000)
        else:
            scores.append(0.1)
        
        # Word diversity score
        words = re.findall(r'\b\w+\b', text)
        if words:
            unique_words = set(word.lower() for word in words)
            diversity = len(unique_words) / len(words)
            scores.append(diversity)
        
        # Sentence structure score
        sentences = re.split(r'[.!?]+', text)
        if len(sentences) > 1:
            avg_sentence_length = len(text) / len(sentences)
            if 20 < avg_sentence_length < 100:
                scores.append(1.0)
            else:
                scores.append(0.5)
        
        # Paragraph structure score
        paragraphs = text.count('\n\n') + 1
        if paragraphs > 1:
            avg_paragraph_length = len(text) / paragraphs
            if 100 < avg_paragraph_length < 1000:
                scores.append(1.0)
            else:
                scores.append(0.5)
        
        # Calculate average score
        if scores:
            return sum(scores) / len(scores)
        return 0.5
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on analysis results.
        
        Args:
            analysis: Analysis dictionary
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Text quality recommendations
        if analysis.get('character_count', 0) < 500:
            recommendations.append("Extracted text is very short. Consider checking if extraction captured all content.")
        
        if analysis.get('lexical_diversity', 0) < 0.3:
            recommendations.append("Text has low lexical diversity. May indicate extraction issues.")
        
        if analysis.get('errors_encountered', 0) > 5:
            recommendations.append(f"High error count ({analysis['errors_encountered']}). Consider reviewing EPUB file integrity.")
        
        # Metadata recommendations
        if not analysis.get('metadata_present', False):
            recommendations.append("No metadata extracted. EPUB may lack proper metadata or extraction method may need adjustment.")
        elif analysis.get('metadata_fields', 0) < 5:
            recommendations.append("Limited metadata extracted. Consider using ebooklib backend if available.")
        
        # Performance recommendations
        if analysis.get('duration_seconds', 0) > 10:
            recommendations.append("Extraction took longer than expected. Consider optimizing or checking file size.")
        
        # Backend recommendations
        if analysis.get('backend_used') == 'zipfile' and EBOOKLIB_AVAILABLE:
            recommendations.append("Using basic zipfile extraction. Install ebooklib for enhanced metadata and structure extraction.")
        
        if not BEAUTIFULSOUP_AVAILABLE:
            recommendations.append("Install BeautifulSoup for improved HTML parsing and text cleaning.")
        
        return recommendations


class EPUBExtractorFactory:
    """
    Factory for creating configured EPUB extractor instances.
    
    Provides pre-configured extractors for different use cases
    and quality/performance trade-offs.
    """
    
    @staticmethod
    def create_extractor(config: Dict[str, Any] = None) -> EPUBExtractor:
        """
        Create EPUB extractor with specified configuration.
        
        Args:
            config: Configuration dictionary with extraction parameters
            
        Returns:
            Configured EPUBExtractor instance
        """
        config = config or {}
        
        # Parse method from config
        method_str = config.get('method', 'auto')
        try:
            method = ExtractionMethod(method_str.lower())
        except ValueError:
            method = ExtractionMethod.AUTO
        
        # Create extractor
        extractor = EPUBExtractor(
            method=method,
            clean_text=config.get('clean_text', True),
            extract_metadata=config.get('extract_metadata', True),
            preserve_structure=config.get('preserve_structure', True),
            max_file_size_mb=config.get('max_file_size_mb', 500),
            language=config.get('language', 'en'),
            logger=config.get('logger')
        )
        
        # Set custom class name if provided
        extractor_class_name = config.get('extractor_class', 'EPUBExtractor')
        extractor.__class__.__name__ = extractor_class_name
        
        return extractor
    
    @staticmethod
    def create_basic_extractor() -> EPUBExtractor:
        """
        Create basic EPUB extractor for minimal resource usage.
        
        Returns:
            Basic EPUBExtractor instance
        """
        return EPUBExtractorFactory.create_extractor({
            'method': 'zipfile',
            'clean_text': True,
            'extract_metadata': False,
            'preserve_structure': False,
            'max_file_size_mb': 100,
            'extractor_class': 'BasicEPUBExtractor'
        })
    
    @staticmethod
    def create_advanced_extractor() -> EPUBExtractor:
        """
        Create advanced EPUB extractor with all features enabled.
        
        Returns:
            Advanced EPUBExtractor instance
        """
        return EPUBExtractorFactory.create_extractor({
            'method': 'ebooklib',
            'clean_text': True,
            'extract_metadata': True,
            'preserve_structure': True,
            'max_file_size_mb': 1000,
            'extractor_class': 'AdvancedEPUBExtractor'
        })
    
    @staticmethod
    def create_fast_extractor() -> EPUBExtractor:
        """
        Create fast EPUB extractor optimized for speed.
        
        Returns:
            Fast EPUBExtractor instance
        """
        return EPUBExtractorFactory.create_extractor({
            'method': 'zipfile',
            'clean_text': False,
            'extract_metadata': False,
            'preserve_structure': False,
            'max_file_size_mb': 50,
            'extractor_class': 'FastEPUBExtractor'
        })
    
    @staticmethod
    def create_quality_extractor() -> EPUBExtractor:
        """
        Create quality-focused EPUB extractor with maximum metadata.
        
        Returns:
            Quality-focused EPUBExtractor instance
        """
        return EPUBExtractorFactory.create_extractor({
            'method': 'ebooklib',
            'clean_text': True,
            'extract_metadata': True,
            'preserve_structure': True,
            'max_file_size_mb': 500,
            'extractor_class': 'QualityEPUBExtractor'
        })


# Convenience functions for common use cases

def extract_epub(file_path: Union[str, Path],
                 config: Dict[str, Any] = None) -> Tuple[str, Optional[EPUBMetadata]]:
    """
    Convenience function for EPUB text extraction.
    
    Args:
        file_path: Path to EPUB file
        config: Extraction configuration
        
    Returns:
        Tuple of extracted text and metadata
    """
    extractor = EPUBExtractorFactory.create_extractor(config)
    return extractor.extract(file_path)


def extract_epub_text(file_path: Union[str, Path]) -> str:
    """
    Extract only text content from EPUB file.
    
    Args:
        file_path: Path to EPUB file
        
    Returns:
        Extracted text content
    """
    extractor = EPUBExtractorFactory.create_basic_extractor()
    text, _ = extractor.extract(file_path)
    return text


def extract_epub_full(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Extract information from EPUB file.
    
    Args:
        file_path: Path to EPUB file
        
    Returns:
        Dictionary with content, metadata, structure, and statistics
    """
    extractor = EPUBExtractorFactory.create_advanced_extractor()
    return extractor.extract_with_metadata(file_path)


def get_epub_metadata(file_path: Union[str, Path]) -> EPUBMetadata:
    """
    Extract only metadata from EPUB file.
    
    Args:
        file_path: Path to EPUB file
        
    Returns:
        EPUBMetadata object
    """
    extractor = EPUBExtractor(
        method=ExtractionMethod.AUTO,
        clean_text=False,
        extract_metadata=True,
        preserve_structure=False
    )
    _, metadata = extractor.extract(file_path)
    return metadata or EPUBMetadata()


def analyze_epub_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Analyze EPUB file without extracting full content.
    
    Args:
        file_path: Path to EPUB file
        
    Returns:
        Analysis results
    """
    extractor = EPUBExtractorFactory.create_extractor()
    text, metadata = extractor.extract(file_path)
    return extractor.analyze_extraction(text, metadata)


# Registry integration for DocuBot extractor system

def register_epub_extractor(extractor_registry: Dict[str, Any]) -> None:
    """
    Register EPUB extractor in the global extractor registry.
    
    Args:
        extractor_registry: Dictionary to register extractor in
    """
    # Register with file extension
    extractor_registry['.epub'] = EPUBExtractor()
    
    # Register with format name
    extractor_registry['epub'] = EPUBExtractor()
    
    # Register factory methods
    extractor_registry['epub_factory'] = EPUBExtractorFactory
    extractor_registry['create_epub_extractor'] = EPUBExtractorFactory.create_extractor
    
    logging.getLogger(__name__).info("EPUB extractor registered in extractor registry")


# Auto-register when module is imported (if in DocuBot context)
if __name__ != '__main__':
    try:
        from ..extractors import EXTRACTOR_REGISTRY
        register_epub_extractor(EXTRACTOR_REGISTRY)
    except ImportError:
        # Not in DocuBot context, skip auto-registration
        pass


# Module initialization
__version__ = '1.0.0'
__author__ = 'DocuBot Team'
__all__ = [
    'EPUBExtractor',
    'EPUBMetadata',
    'EPUBExtractorFactory',
    'ExtractionMethod',
    'EPUBError',
    'InvalidEPUBError',
    'EPUBExtractionError',
    'extract_epub',
    'extract_epub_text',
    'extract_epub_full',
    'get_epub_metadata',
    'analyze_epub_file',
    'register_epub_extractor'
]