"""
Text cleaning and normalization utilities for DocuBot.

Provides comprehensive text cleaning functions to prepare extracted document text
for chunking, embedding, and storage. Handles encoding issues, whitespace 
normalization, special character removal, and language-specific cleanup.
"""

import re
import unicodedata
from typing import Optional, List, Dict, Any, Callable
import html
from pathlib import Path
import logging

# Configure module logger
logger = logging.getLogger(__name__)


def fix_encoding_issues(text: str) -> str:
    """
    Fix common encoding problems in text.
    
    Handles UTF-8 BOM, replacement characters, and common encoding artifacts.
    """
    # Remove UTF-8 BOM if present
    if text.startswith('\ufeff'):
        text = text[1:]
    
    # Replace common encoding artifacts
    replacements = [
        ('\u2018', "'"),  # Left single quotation mark
        ('\u2019', "'"),  # Right single quotation mark
        ('\u201c', '"'),  # Left double quotation mark
        ('\u201d', '"'),  # Right double quotation mark
        ('\u2013', '-'),  # En dash
        ('\u2014', '-'),  # Em dash
        ('\u2026', '...'),  # Horizontal ellipsis
        ('\u00a0', ' '),   # Non-breaking space
        ('\u200b', ''),    # Zero-width space
        ('\ufeff', ''),    # Zero-width no-break space
    ]
    
    for old, new in replacements:
        text = text.replace(old, new)
    
    # Remove other non-printable characters (except newlines and tabs)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    
    return text


def strip_html_tags(text: str) -> str:
    """
    Remove HTML/XML tags from text while preserving content.
    
    Also unescapes HTML entities (e.g., &amp; -> &).
    """
    # First unescape HTML entities
    text = html.unescape(text)
    
    # Remove HTML/XML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove common HTML artifacts
    text = re.sub(r'&[a-z]+;', ' ', text)
    
    return text


def normalize_unicode(text: str, form: str = 'NFKC') -> str:
    """
    Normalize Unicode characters to canonical form.
    
    Args:
        text: Input text with potential Unicode variations.
        form: Normalization form ('NFC', 'NFD', 'NFKC', 'NFKD').
              Default is 'NFKC' which is most aggressive.
    
    Returns:
        Unicode-normalized text.
    """
    try:
        normalized = unicodedata.normalize(form, text)
        # Remove combining characters if using decomposed form
        if form in ['NFD', 'NFKD']:
            normalized = ''.join(
                c for c in normalized if not unicodedata.combining(c)
            )
        return normalized
    except Exception as e:
        logger.warning(f"Unicode normalization failed: {e}")
        return text


def remove_special_characters(text: str, 
                            keep: Optional[str] = None) -> str:
    """
    Remove special characters while keeping essential punctuation.
    
    Args:
        text: Input text.
        keep: String of characters to preserve (e.g., '.,!?-:').
              If None, uses default set.
    
    Returns:
        Text with special characters removed.
    """
    if keep is None:
        keep = r'.,!?\-\:;"\'()[]{}'
    
    # Escape special regex characters in keep string
    keep_escaped = re.escape(keep)
    
    # Remove everything except alphanumeric, whitespace, and kept characters
    pattern = f'[^\\w\\s{keep_escaped}]'
    text = re.sub(pattern, ' ', text)
    
    # Remove multiple consecutive special characters
    if keep:
        keep_pattern = f'[{re.escape(keep)}]'
        text = re.sub(f'{keep_pattern}{{2,}}', keep[0], text)
    
    return text


def normalize_whitespace(text: str) -> str:
    """
    Normalize all whitespace characters to standard spaces.
    
    Replaces tabs, non-breaking spaces, and other whitespace with single spaces.
    """
    # Replace all whitespace characters with space
    text = re.sub(r'\s+', ' ', text)
    
    # Trim leading/trailing whitespace
    text = text.strip()
    
    return text


def remove_excessive_newlines(text: str, max_consecutive: int = 2) -> str:
    """
    Reduce excessive newlines to improve readability.
    
    Args:
        text: Input text.
        max_consecutive: Maximum allowed consecutive newlines.
    
    Returns:
        Text with controlled newline spacing.
    """
    # Replace multiple newlines with specified maximum
    pattern = r'\n{' + str(max_consecutive + 1) + ',}'
    replacement = '\n' * max_consecutive
    text = re.sub(pattern, replacement, text)
    
    return text


def convert_to_lowercase(text: str) -> str:
    """Convert all text to lowercase."""
    return text.lower()


def remove_numbers(text: str, 
                  preserve_dates: bool = True,
                  preserve_versions: bool = False) -> str:
    """
    Remove numeric values from text.
    
    Args:
        text: Input text.
        preserve_dates: If True, preserve common date formats.
        preserve_versions: If True, preserve version numbers.
    
    Returns:
        Text with numbers removed.
    """
    if preserve_dates:
        # Protect common date formats
        date_patterns = [
            r'\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b',  # DD/MM/YYYY
            r'\b\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}\b',    # YYYY/MM/DD
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b',  # Month DD, YYYY
        ]
        
        # Replace dates with placeholders
        placeholders = []
        for i, pattern in enumerate(date_patterns):
            for match in re.finditer(pattern, text, re.IGNORECASE):
                placeholder = f'__DATE_{i}_{len(placeholders)}__'
                placeholders.append((placeholder, match.group(0)))
        
        # Temporarily replace dates
        for placeholder, original in placeholders:
            text = text.replace(original, placeholder)
    
    if preserve_versions:
        # Protect version numbers (e.g., v1.2.3, version 2.0)
        version_pattern = r'\b(?:v|version|ver\.?)\s*\d+(?:\.\d+)*\b'
        placeholders = []
        
        for match in re.finditer(version_pattern, text, re.IGNORECASE):
            placeholder = f'__VERSION_{len(placeholders)}__'
            placeholders.append((placeholder, match.group(0)))
            text = text.replace(match.group(0), placeholder)
    
    # Remove standalone numbers
    text = re.sub(r'\b\d+\b', ' ', text)
    
    # Remove decimal numbers
    text = re.sub(r'\b\d+\.\d+\b', ' ', text)
    
    # Restore preserved items
    if preserve_dates:
        for placeholder, original in placeholders:
            if placeholder.startswith('__DATE_'):
                text = text.replace(placeholder, original)
    
    if preserve_versions:
        for placeholder, original in placeholders:
            if placeholder.startswith('__VERSION_'):
                text = text.replace(placeholder, original)
    
    return text


def remove_punctuation(text: str, 
                      keep: Optional[str] = None) -> str:
    """
    Remove punctuation characters from text.
    
    Args:
        text: Input text.
        keep: String of punctuation characters to preserve.
              If None, removes all punctuation.
    
    Returns:
        Text with punctuation removed.
    """
    if keep is None:
        # Remove all punctuation
        text = re.sub(r'[^\w\s]', ' ', text)
    else:
        # Remove all punctuation except specified characters
        keep_escaped = re.escape(keep)
        text = re.sub(f'[^\\w\\s{keep_escaped}]', ' ', text)
    
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text


def remove_short_words(text: str, min_length: int = 2) -> str:
    """
    Remove words shorter than specified length.
    
    Useful for removing single-character words that are often noise.
    """
    words = text.split()
    filtered_words = [word for word in words if len(word) >= min_length]
    return ' '.join(filtered_words)


def remove_long_words(text: str, max_length: int = 50) -> str:
    """
    Remove words longer than specified length.
    
    Useful for removing URLs, encoded strings, or other overly long tokens.
    """
    words = text.split()
    filtered_words = [word for word in words if len(word) <= max_length]
    return ' '.join(filtered_words)


def normalize_citations(text: str) -> str:
    """
    Normalize academic citations and references.
    
    Converts various citation formats to a standard pattern for better processing.
    """
    # Normalize [1], [2,3], [4-6] patterns
    text = re.sub(r'\[(\d+(?:[,-]\s*\d+)*)\]', r'[REF:\1]', text)
    
    # Normalize (Author, Year) patterns
    text = re.sub(r'\(([A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*),\s*(\d{4})\)', 
                 r'(CITE:\1,\2)', text)
    
    # Remove multiple citation markers in sequence
    text = re.sub(r'(\[REF:[^\]]+\]\s*){2,}', '[MULTI_REF]', text)
    
    return text


def normalize_urls(text: str) -> str:
    """
    Normalize URLs to a standard format.
    
    Shortens very long URLs and standardizes common patterns.
    """
    # Find URLs
    url_pattern = r'https?://[^\s]+'
    urls = re.findall(url_pattern, text)
    
    for url in urls:
        # Shorten very long URLs
        if len(url) > 50:
            short_url = url[:40] + '...' + url[-7:]
            text = text.replace(url, f'[URL:{short_url}]')
        else:
            text = text.replace(url, f'[URL]')
    
    return text


def detect_language(text: str, sample_size: int = 1000) -> str:
    """
    Simple language detection based on character distribution.
    
    Note: For production use, consider integrating with specialized libraries
    like langdetect or fasttext.
    
    Args:
        text: Input text to analyze.
        sample_size: Number of characters to sample for detection.
    
    Returns:
        Detected language code ('english', 'indonesian', 'unknown').
    """
    if not text:
        return 'unknown'
    
    # Sample text if it's too long
    sample = text[:sample_size].lower()
    
    # English character patterns
    english_patterns = [
        ('the', 0.05),  # Common English word
        ('and', 0.04),
        ('ing', 0.03),  # Common suffix
        ('th', 0.02),   # Common digraph
    ]
    
    # Indonesian character patterns
    indonesian_patterns = [
        ('dan', 0.05),  # Common Indonesian word
        ('yang', 0.04),
        ('di', 0.03),
        ('ke', 0.02),
    ]
    
    english_score = 0
    indonesian_score = 0
    
    for pattern, weight in english_patterns:
        count = sample.count(pattern)
        english_score += count * weight
    
    for pattern, weight in indonesian_patterns:
        count = sample.count(pattern)
        indonesian_score += count * weight
    
    # Character-based heuristics
    english_chars = set('abcdefghijklmnopqrstuvwxyz')
    indonesian_special = set('áéíóúàèìòùäëïöü')
    
    english_char_ratio = sum(1 for c in sample if c in english_chars) / len(sample)
    indonesian_char_ratio = sum(1 for c in sample if c in indonesian_special) / len(sample)
    
    english_score += english_char_ratio * 0.5
    indonesian_score += indonesian_char_ratio * 0.5
    
    if english_score > indonesian_score and english_score > 0.1:
        return 'english'
    elif indonesian_score > english_score and indonesian_score > 0.1:
        return 'indonesian'
    else:
        return 'unknown'


def create_language_specific_cleaner(language: str = 'english') -> 'TextCleaner':
    """
    Create a TextCleaner instance with language-specific defaults.
    
    Args:
        language: Target language for cleaning ('english' or 'indonesian').
    
    Returns:
        Configured TextCleaner instance.
    """
    if language.lower() == 'indonesian':
        config = {
            'normalize_whitespace': True,
            'remove_extra_newlines': True,
            'fix_encoding': True,
            'remove_special_chars': True,
            'strip_html': True,
            'normalize_unicode': True,
            'lowercase': False,
            'remove_numbers': False,
            'remove_punctuation': False,
            'min_word_length': 2,
            'max_word_length': 50,
            'preserve_citations': True,
            'preserve_urls': True,
            'language': 'indonesian'
        }
    else:  # Default to English
        config = {
            'normalize_whitespace': True,
            'remove_extra_newlines': True,
            'fix_encoding': True,
            'remove_special_chars': True,
            'strip_html': True,
            'normalize_unicode': True,
            'lowercase': False,
            'remove_numbers': False,
            'remove_punctuation': False,
            'min_word_length': 2,
            'max_word_length': 50,
            'preserve_citations': True,
            'preserve_urls': True,
            'language': 'english'
        }
    
    return TextCleaner(config)


def clean_text_basic(text: str) -> str:
    """
    Basic text cleaning using default configuration.
    
    Suitable for most general-purpose text cleaning needs.
    """
    cleaner = TextCleaner()
    return cleaner.clean_text(text)


def clean_text_advanced(text: str, 
                       aggressive: bool = False,
                       language: str = 'english') -> str:
    """
    Advanced text cleaning with configurable aggressiveness.
    
    Args:
        text: Input text to clean.
        aggressive: If True, applies more aggressive cleaning (lowercase, 
                   remove punctuation, etc.).
        language: Language code for language-specific cleaning.
    
    Returns:
        Cleaned text.
    """
    config = {
        'normalize_whitespace': True,
        'remove_extra_newlines': True,
        'fix_encoding': True,
        'remove_special_chars': True,
        'strip_html': True,
        'normalize_unicode': True,
        'lowercase': aggressive,
        'remove_numbers': aggressive,
        'remove_punctuation': aggressive,
        'min_word_length': 2,
        'max_word_length': 50,
        'preserve_citations': not aggressive,
        'preserve_urls': not aggressive,
        'language': language
    }
    
    cleaner = TextCleaner(config)
    return cleaner.clean_text(text)


class TextCleaner:
    """
    Main text cleaning class providing configurable cleaning pipelines.
    
    Supports multiple cleaning strategies and allows custom cleaning function
    registration for specialized document types.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize TextCleaner with optional configuration.
        
        Args:
            config: Dictionary containing cleaning configuration options.
                   Defaults to standard cleaning pipeline if None.
        """
        self.config = config or self._get_default_config()
        self._cleaning_functions = self._initialize_cleaning_functions()
        logger.debug(f"TextCleaner initialized with config: {self.config}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Return default cleaning configuration."""
        return {
            'normalize_whitespace': True,
            'remove_extra_newlines': True,
            'fix_encoding': True,
            'remove_special_chars': True,
            'strip_html': True,
            'normalize_unicode': True,
            'lowercase': False,
            'remove_numbers': False,
            'remove_punctuation': False,
            'min_word_length': 2,
            'max_word_length': 50,
            'preserve_citations': True,
            'preserve_urls': True,
            'language': 'english'
        }
    
    def _initialize_cleaning_functions(self) -> List[Callable[[str], str]]:
        """Initialize and order cleaning functions based on configuration."""
        functions = []
        
        if self.config.get('fix_encoding', True):
            functions.append(fix_encoding_issues)
        
        if self.config.get('strip_html', True):
            functions.append(strip_html_tags)
        
        if self.config.get('normalize_unicode', True):
            functions.append(normalize_unicode)
        
        if self.config.get('remove_special_chars', True):
            functions.append(remove_special_characters)
        
        if self.config.get('normalize_whitespace', True):
            functions.append(normalize_whitespace)
        
        if self.config.get('remove_extra_newlines', True):
            functions.append(remove_excessive_newlines)
        
        if self.config.get('lowercase', False):
            functions.append(convert_to_lowercase)
        
        if self.config.get('remove_numbers', False):
            functions.append(remove_numbers)
        
        if self.config.get('remove_punctuation', False):
            functions.append(remove_punctuation)
        
        functions.append(remove_short_words)
        functions.append(remove_long_words)
        
        if self.config.get('preserve_citations', True):
            functions.append(normalize_citations)
        
        if self.config.get('preserve_urls', True):
            functions.append(normalize_urls)
        
        return functions
    
    def clean_text(self, text: str, custom_pipeline: Optional[List[Callable[[str], str]]] = None) -> str:
        """
        Apply complete cleaning pipeline to text.
        
        Args:
            text: Raw text input to clean.
            custom_pipeline: Optional list of cleaning functions to use instead 
                            of the configured pipeline.
        
        Returns:
            Cleaned and normalized text.
        
        Raises:
            ValueError: If input text is not a string.
        """
        if not isinstance(text, str):
            raise ValueError(f"Expected string input, got {type(text).__name__}")
        
        if not text.strip():
            logger.warning("Received empty or whitespace-only text")
            return text
        
        original_length = len(text)
        cleaned_text = text
        
        pipeline = custom_pipeline or self._cleaning_functions
        
        for i, clean_func in enumerate(pipeline):
            try:
                before_len = len(cleaned_text)
                cleaned_text = clean_func(cleaned_text)
                after_len = len(cleaned_text)
                
                if before_len != after_len:
                    logger.debug(f"Step {i+1}: {clean_func.__name__} "
                                f"reduced text from {before_len} to {after_len} chars")
            except Exception as e:
                logger.error(f"Error in cleaning step {clean_func.__name__}: {e}")
                continue
        
        reduction = original_length - len(cleaned_text)
        if reduction > 0:
            logger.info(f"Text cleaning reduced length by {reduction} characters "
                       f"({reduction/original_length*100:.1f}%)")
        
        return cleaned_text
    
    def batch_clean(self, texts: List[str], show_progress: bool = False) -> List[str]:
        """
        Clean multiple text documents efficiently.
        
        Args:
            texts: List of text strings to clean.
            show_progress: If True, log progress for large batches.
        
        Returns:
            List of cleaned text strings in the same order as input.
        """
        cleaned_texts = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            if show_progress and total > 10 and i % (total // 10) == 0:
                logger.info(f"Cleaning progress: {i}/{total} ({i/total*100:.0f}%)")
            
            cleaned_texts.append(self.clean_text(text))
        
        if show_progress:
            logger.info(f"Batch cleaning complete: {total} documents processed")
        
        return cleaned_texts
    
    def register_custom_cleaner(self, 
                               cleaner_func: Callable[[str], str], 
                               position: Optional[int] = None) -> None:
        """
        Register a custom cleaning function.
        
        Args:
            cleaner_func: Function that takes a string and returns cleaned string.
            position: Optional position in the cleaning pipeline (0-based).
                     If None, appends to the end of the pipeline.
        """
        if position is None:
            self._cleaning_functions.append(cleaner_func)
        else:
            self._cleaning_functions.insert(position, cleaner_func)
        
        logger.debug(f"Registered custom cleaner '{cleaner_func.__name__}' "
                    f"at position {position if position is not None else 'end'}")
    
    def get_cleaning_pipeline_info(self) -> List[Dict[str, Any]]:
        """Return information about the current cleaning pipeline."""
        pipeline_info = []
        for i, func in enumerate(self._cleaning_functions):
            info = {
                'name': func.__name__,
                'description': func.__doc__.split('\n')[0] if func.__doc__ else 'No description',
                'order': i
            }
            pipeline_info.append(info)
        return pipeline_info


if __name__ == '__main__':
    # Example usage and basic testing
    sample_text = """
    <html>
    <body>
    <h1>Test Document</h1>
    <p>This is a test document with <b>HTML tags</b> and special 
    characters: &amp; &quot; &#39;.</p>
    <p>It also has multiple    spaces and 
    
    
    excessive newlines.</p>
    <p>Reference: [1,2,3] and (Smith, 2020).</p>
    <p>URL: https://example.com/very/long/url/path/that/goes/on/and/on.html</p>
    </body>
    </html>
    """
    
    print("Original text:")
    print(sample_text)
    print("\n" + "="*80 + "\n")
    
    cleaner = TextCleaner()
    cleaned = cleaner.clean_text(sample_text)
    
    print("Cleaned text:")
    print(cleaned)
    print("\n" + "="*80 + "\n")
    
    print("Cleaning pipeline:")
    for info in cleaner.get_cleaning_pipeline_info():
        print(f"  {info['order']+1}. {info['name']}: {info['description']}")
    
    print("\n" + "="*80 + "\n")
    
    # Test language detection
    lang = detect_language(cleaned)
    print(f"Detected language: {lang}")
    
    print("\nCleaning complete.")