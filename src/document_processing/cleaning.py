"""
Text Cleaning and Normalization Utilities
"""

import re
import unicodedata
from typing import List, Optional
import html


def clean_text(
    text: str,
    remove_html: bool = True,
    normalize_whitespace: bool = True,
    remove_special_chars: bool = False,
    to_lowercase: bool = False
) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text to clean
        remove_html: Remove HTML tags
        normalize_whitespace: Normalize whitespace characters
        remove_special_chars: Remove special characters
        to_lowercase: Convert to lowercase
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    cleaned = text
    
    if remove_html:
        cleaned = html.unescape(cleaned)
        cleaned = re.sub(r'<[^>]+>', ' ', cleaned)
    
    cleaned = unicodedata.normalize('NFKC', cleaned)
    
    if normalize_whitespace:
        cleaned = re.sub(r'[	
]+', ' ', cleaned)
        cleaned = re.sub(r'\s+', ' ', cleaned)
        cleaned = cleaned.strip()
    
    if remove_special_chars:
        cleaned = re.sub(r'[^\w\s.,!?;:\-'"()\[\]]', ' ', cleaned)
    
    if to_lowercase:
        cleaned = cleaned.lower()
    
    return cleaned


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    sentences = re.split(r'[.!?]+\s*', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def remove_stopwords(text: str, custom_stopwords: Optional[List[str]] = None) -> str:
    """
    Remove common stopwords from text.
    
    Args:
        text: Input text
        custom_stopwords: Custom list of stopwords
        
    Returns:
        Text with stopwords removed
    """
    default_stopwords = {
        'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
        'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
        'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
        'will', 'would', 'shall', 'should', 'may', 'might', 'must',
        'can', 'could', 'i', 'you', 'he', 'she', 'it', 'we', 'they'
    }
    
    stopwords = default_stopwords
    if custom_stopwords:
        stopwords.update(custom_stopwords)
    
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stopwords]
    
    return ' '.join(filtered_words)


def normalize_numbers(text: str, replacement: str = "[NUM]") -> str:
    """
    Normalize numbers in text.
    
    Args:
        text: Input text
        replacement: String to replace numbers with
        
    Returns:
        Text with normalized numbers
    """
    normalized = re.sub(r'\d+', replacement, text)
    normalized = re.sub(r'\d+\.\d+', replacement, normalized)
    return normalized


def remove_extra_punctuation(text: str) -> str:
    """
    Remove excessive punctuation.
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized punctuation
    """
    normalized = re.sub(r'([.!?])+', r'', text)
    normalized = re.sub(r'\s+([.,!?;:])', r'', normalized)
    normalized = re.sub(r'([.,!?;:])(?!\s|$)', r' ', normalized)
    return normalized


def truncate_text(text: str, max_length: int = 1000) -> str:
    """
    Truncate text to maximum length while preserving word boundaries.
    
    Args:
        text: Input text
        max_length: Maximum character length
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    
    truncated = text[:max_length]
    last_space = truncated.rfind(' ')
    
    if last_space > max_length * 0.8:
        truncated = truncated[:last_space]
    
    return truncated + "..."


def calculate_readability_score(text: str) -> float:
    """
    Calculate simple readability score.
    
    Args:
        text: Input text
        
    Returns:
        Readability score (higher = easier to read)
    """
    sentences = split_into_sentences(text)
    words = text.split()
    
    if not sentences or not words:
        return 0.0
    
    words_per_sentence = len(words) / len(sentences)
    syllables_per_word = sum(len(re.findall(r'[aeiouy]+', word.lower())) 
                           for word in words) / len(words)
    
    score = 206.835 - (1.015 * words_per_sentence) - (84.6 * syllables_per_word)
    return max(0.0, min(100.0, score))


def clean_text_pipeline(text: str) -> str:
    """
    Complete text cleaning pipeline.
    
    Args:
        text: Input text
        
    Returns:
        Fully cleaned text
    """
    cleaned = clean_text(text, remove_html=True, normalize_whitespace=True)
    cleaned = remove_extra_punctuation(cleaned)
    cleaned = normalize_numbers(cleaned)
    return cleaned


if __name__ == "__main__":
    test_text = "Hello   World!! This is a test. 123 numbers. <b>HTML</b> tags."
    
    print("Original:", test_text)
    print("Cleaned:", clean_text_pipeline(test_text))
    print("Sentences:", split_into_sentences(test_text))
    print("Readability:", calculate_readability_score(test_text))
