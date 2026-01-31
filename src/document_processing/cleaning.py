# docubot/src/document_processing/cleaning.py

"""
Text Cleaning and Normalization Module for DocuBot Document Processing Pipeline

Provides text cleaning utilities for preparing extracted document text
for chunking, embedding, and storage operations.
"""

import re
import unicodedata
import html
import logging
from typing import Optional, List, Dict, Any, Callable, Union
from pathlib import Path

logger = logging.getLogger(__name__)


def fix_encoding_issues(text: str) -> str:
    """Fix common encoding problems and remove non-printable characters."""
    if text.startswith('\ufeff'):
        text = text[1:]

    encoding_replacements = [
        ('\u2018', "'"),
        ('\u2019', "'"),
        ('\u201c', '"'),
        ('\u201d', '"'),
        ('\u2013', '-'),
        ('\u2014', '-'),
        ('\u2026', '...'),
        ('\u00a0', ' '),
        ('\u200b', ''),
        ('\ufeff', ''),
    ]

    for old_char, new_char in encoding_replacements:
        text = text.replace(old_char, new_char)

    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    return text


def strip_html_tags(text: str) -> str:
    """Remove HTML/XML tags and unescape HTML entities."""
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&[a-z]+;', ' ', text)
    return text


def normalize_unicode(text: str, form: str = 'NFKC') -> str:
    """Normalize Unicode characters to specified canonical form."""
    try:
        normalized = unicodedata.normalize(form, text)
        if form in ['NFD', 'NFKD']:
            normalized = ''.join(c for c in normalized if not unicodedata.combining(c))
        return normalized
    except Exception as e:
        logger.warning(f"Unicode normalization failed: {e}")
        return text


def remove_special_characters(text: str, keep: Optional[str] = None) -> str:
    """Remove special characters while preserving specified punctuation."""
    if keep is None:
        keep = r'.,!?\-\:;"\'()[]{}'

    keep_escaped = re.escape(keep)
    pattern = f'[^\\w\\s{keep_escaped}]'
    text = re.sub(pattern, ' ', text)

    if keep:
        keep_pattern = f'[{re.escape(keep)}]'
        text = re.sub(f'{keep_pattern}{{2,}}', keep[0], text)

    return text


def normalize_whitespace(text: str) -> str:
    """Normalize all whitespace characters to standard spaces."""
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def remove_excessive_newlines(text: str, max_consecutive: int = 2) -> str:
    """Reduce excessive newlines to improve readability."""
    pattern = r'\n{' + str(max_consecutive + 1) + ',}'
    replacement = '\n' * max_consecutive
    return re.sub(pattern, replacement, text)


def convert_to_lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def remove_numbers(text: str, preserve_dates: bool = True, preserve_versions: bool = False) -> str:
    """Remove numeric values with optional preservation of dates and versions."""
    date_placeholders = []
    version_placeholders = []

    if preserve_dates:
        date_patterns = [
            r'\b\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4}\b',
            r'\b\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2}\b',
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b',
        ]

        for i, pattern in enumerate(date_patterns):
            for match in re.finditer(pattern, text, re.IGNORECASE):
                placeholder = f'__DATE_{i}_{len(date_placeholders)}__'
                date_placeholders.append((placeholder, match.group(0)))
                text = text.replace(match.group(0), placeholder)

    if preserve_versions:
        version_pattern = r'\b(?:v|version|ver\.?)\s*\d+(?:\.\d+)*\b'
        for match in re.finditer(version_pattern, text, re.IGNORECASE):
            placeholder = f'__VERSION_{len(version_placeholders)}__'
            version_placeholders.append((placeholder, match.group(0)))
            text = text.replace(match.group(0), placeholder)

    text = re.sub(r'\b\d+\b', ' ', text)
    text = re.sub(r'\b\d+\.\d+\b', ' ', text)

    for placeholder, original in date_placeholders:
        text = text.replace(placeholder, original)

    for placeholder, original in version_placeholders:
        text = text.replace(placeholder, original)

    return text


def remove_punctuation(text: str, keep: Optional[str] = None) -> str:
    """Remove punctuation characters with optional preservation."""
    if keep is None:
        text = re.sub(r'[^\w\s]', ' ', text)
    else:
        keep_escaped = re.escape(keep)
        text = re.sub(f'[^\\w\\s{keep_escaped}]', ' ', text)

    return re.sub(r'\s+', ' ', text)


def remove_short_words(text: str, min_length: int = 2) -> str:
    """Remove words shorter than specified length."""
    words = text.split()
    filtered_words = [word for word in words if len(word) >= min_length]
    return ' '.join(filtered_words)


def remove_long_words(text: str, max_length: int = 50) -> str:
    """Remove words longer than specified length."""
    words = text.split()
    filtered_words = [word for word in words if len(word) <= max_length]
    return ' '.join(filtered_words)


def normalize_citations(text: str) -> str:
    """Normalize academic citations and references."""
    text = re.sub(r'\[(\d+(?:[,-]\s*\d+)*)\]', r'[REF:\1]', text)
    text = re.sub(r'\(([A-Z][a-z]+(?:,\s*[A-Z][a-z]+)*),\s*(\d{4})\)', r'(CITE:\1,\2)', text)
    text = re.sub(r'(\[REF:[^\]]+\]\s*){2,}', '[MULTI_REF]', text)
    return text


def normalize_urls(text: str) -> str:
    """Normalize URLs to standard format."""
    url_pattern = r'https?://[^\s]+'
    urls = re.findall(url_pattern, text)

    for url in urls:
        if len(url) > 50:
            short_url = url[:40] + '...' + url[-7:]
            text = text.replace(url, f'[URL:{short_url}]')
        else:
            text = text.replace(url, '[URL]')

    return text


def detect_language(text: str, sample_size: int = 1000) -> str:
    """Simple language detection based on character and word patterns."""
    if not text:
        return 'unknown'

    sample = text[:sample_size].lower()

    english_patterns = [('the', 0.05), ('and', 0.04), ('ing', 0.03), ('th', 0.02)]
    indonesian_patterns = [('dan', 0.05), ('yang', 0.04), ('di', 0.03), ('ke', 0.02)]

    english_score = sum(sample.count(pattern) * weight for pattern, weight in english_patterns)
    indonesian_score = sum(sample.count(pattern) * weight for pattern, weight in indonesian_patterns)

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


def clean_text_pipeline(
    text: str,
    remove_extra_spaces: bool = True,
    normalize_unicode_flag: bool = True,
    remove_control_chars: bool = True,
    preserve_line_breaks: bool = False
) -> str:
    """
    Comprehensive text cleaning pipeline with configurable options.
    
    Args:
        text: Input text to clean
        remove_extra_spaces: Normalize whitespace if True
        normalize_unicode_flag: Apply Unicode normalization if True
        remove_control_chars: Remove control characters if True
        preserve_line_breaks: Preserve newline characters if True
        
    Returns:
        Cleaned text string
    """
    if not text:
        return ""

    config = {
        'normalize_whitespace': remove_extra_spaces,
        'remove_extra_newlines': True,
        'fix_encoding': True,
        'remove_special_chars': True,
        'strip_html': True,
        'normalize_unicode': normalize_unicode_flag,
        'lowercase': False,
        'remove_numbers': False,
        'remove_punctuation': False,
        'min_word_length': 2,
        'max_word_length': 50,
        'preserve_citations': True,
        'preserve_urls': True,
    }

    cleaner = TextCleaner(config)
    cleaned_text = cleaner.clean_text(text)

    if remove_control_chars:
        if preserve_line_breaks:
            cleaned_text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', cleaned_text)
        else:
            cleaned_text = re.sub(r'[\x00-\x1f\x7f]', '', cleaned_text)

    return cleaned_text


def clean_text_basic(text: str) -> str:
    """Basic text cleaning using default configuration."""
    cleaner = TextCleaner()
    return cleaner.clean_text(text)


def clean_text_advanced(text: str, aggressive: bool = False, language: str = 'english') -> str:
    """Advanced text cleaning with configurable aggressiveness."""
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


def create_language_specific_cleaner(language: str = 'english') -> 'TextCleaner':
    """Create TextCleaner instance with language-specific defaults."""
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
    else:
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


class TextCleaner:
    """Configurable text cleaning pipeline with support for custom cleaning functions."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or self._get_default_config()
        self._cleaning_functions = self._initialize_cleaning_functions()
        logger.debug(f"TextCleaner initialized with config: {self.config}")

    def _get_default_config(self) -> Dict[str, Any]:
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
        """Apply complete cleaning pipeline to text."""
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
                    logger.debug(
                        f"Step {i+1}: {clean_func.__name__} "
                        f"reduced text from {before_len} to {after_len} chars"
                    )
            except Exception as e:
                logger.error(f"Error in cleaning step {clean_func.__name__}: {e}")
                continue

        reduction = original_length - len(cleaned_text)
        if reduction > 0:
            logger.info(
                f"Text cleaning reduced length by {reduction} characters "
                f"({reduction/original_length*100:.1f}%)"
            )

        return cleaned_text

    def batch_clean(self, texts: List[str], show_progress: bool = False) -> List[str]:
        """Clean multiple text documents efficiently."""
        cleaned_texts = []
        total = len(texts)

        for i, text in enumerate(texts):
            if show_progress and total > 10 and i % (total // 10) == 0:
                logger.info(f"Cleaning progress: {i}/{total} ({i/total*100:.0f}%)")

            cleaned_texts.append(self.clean_text(text))

        if show_progress:
            logger.info(f"Batch cleaning complete: {total} documents processed")

        return cleaned_texts

    def register_custom_cleaner(
        self,
        cleaner_func: Callable[[str], str],
        position: Optional[int] = None
    ) -> None:
        """Register a custom cleaning function."""
        if position is None:
            self._cleaning_functions.append(cleaner_func)
        else:
            self._cleaning_functions.insert(position, cleaner_func)

        logger.debug(
            f"Registered custom cleaner '{cleaner_func.__name__}' "
            f"at position {position if position is not None else 'end'}"
        )

    def get_cleaning_pipeline_info(self) -> List[Dict[str, Any]]:
        """Return information about the current cleaning pipeline."""
        pipeline_info = []
        for i, func in enumerate(self._cleaning_functions):
            docstring = func.__doc__.split('\n')[0] if func.__doc__ else 'No description'
            info = {
                'name': func.__name__,
                'description': docstring,
                'order': i
            }
            pipeline_info.append(info)
        return pipeline_info


def __test_cleaning_module() -> None:
    """Test function for the cleaning module."""
    test_text = """
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
    print(test_text)
    print("\n" + "="*80 + "\n")

    cleaner = TextCleaner()
    cleaned = cleaner.clean_text(test_text)

    print("Cleaned text:")
    print(cleaned)
    print("\n" + "="*80 + "\n")

    print("Cleaning pipeline:")
    for info in cleaner.get_cleaning_pipeline_info():
        print(f"  {info['order']+1}. {info['name']}: {info['description']}")

    print("\n" + "="*80 + "\n")

    detected_lang = detect_language(cleaned)
    print(f"Detected language: {detected_lang}")

    print("\nCleaning module test complete.")


if __name__ == '__main__':
    __test_cleaning_module()