# DocuBot/src/ai_engine/summarizer.py

"""
Summarizer module for DocuBot
Provides text summarization functionality.
"""

import logging
from typing import Dict, List, Any, Optional
import re

logger = logging.getLogger(__name__)


class Summarizer:
    """Text summarization using various strategies."""
    
    def __init__(self, strategy: str = "extractive", config: Optional[Dict[str, Any]] = None):
        """
        Initialize summarizer.
        
        Args:
            strategy: Summarization strategy ('extractive', 'abstractive', 'hybrid')
            config: Configuration dictionary
        """
        self.strategy = strategy
        self.config = config or {}
        self.initialized = False
        
    def initialize(self) -> bool:
        """Initialize summarizer resources."""
        try:
            if self.strategy == "extractive":
                # Simple extractive summarization doesn't need special initialization
                pass
            elif self.strategy == "abstractive":
                # Would load abstractive models here
                logger.info("Abstractive summarization would load models here")
            elif self.strategy == "hybrid":
                # Would load both extractive and abstractive components
                logger.info("Hybrid summarization would load models here")
            
            self.initialized = True
            logger.info(f"Summarizer initialized with {self.strategy} strategy")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize summarizer: {e}")
            return False
    
    def summarize(self, text: str, max_sentences: int = 3, 
                  max_length: int = 150) -> Dict[str, Any]:
        """
        Summarize text using the configured strategy.
        
        Args:
            text: Input text to summarize
            max_sentences: Maximum number of sentences in summary
            max_length: Maximum character length of summary
            
        Returns:
            Dictionary with summary and metadata
        """
        if not self.initialized:
            if not self.initialize():
                return {
                    'success': False,
                    'error': 'Summarizer not initialized',
                    'summary': '',
                    'original_length': len(text)
                }
        
        try:
            if self.strategy == "extractive":
                return self._extractive_summarize(text, max_sentences, max_length)
            elif self.strategy == "abstractive":
                return self._abstractive_summarize(text, max_length)
            elif self.strategy == "hybrid":
                return self._hybrid_summarize(text, max_sentences, max_length)
            else:
                return {
                    'success': False,
                    'error': f'Unknown strategy: {self.strategy}',
                    'summary': '',
                    'original_length': len(text)
                }
                
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'summary': '',
                'original_length': len(text)
            }
    
    def _extractive_summarize(self, text: str, max_sentences: int, 
                             max_length: int) -> Dict[str, Any]:
        """Simple extractive summarization."""
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if len(sentences) <= max_sentences:
            summary = text[:max_length]
            return {
                'success': True,
                'summary': summary,
                'original_length': len(text),
                'summary_length': len(summary),
                'sentences_used': len(sentences),
                'strategy': 'extractive',
                'compression_ratio': len(summary) / len(text) if len(text) > 0 else 0
            }
        
        # Simple heuristic: take first and last sentences
        selected_sentences = []
        if sentences:
            selected_sentences.append(sentences[0])  # First sentence
            if len(sentences) > 1:
                selected_sentences.append(sentences[-1])  # Last sentence
        
        # Add middle sentences if needed
        middle_index = len(sentences) // 2
        if len(selected_sentences) < max_sentences and middle_index < len(sentences):
            selected_sentences.append(sentences[middle_index])
        
        summary = ' '.join(selected_sentences)
        if len(summary) > max_length:
            summary = summary[:max_length] + '...'
        
        return {
            'success': True,
            'summary': summary,
            'original_length': len(text),
            'summary_length': len(summary),
            'sentences_used': len(selected_sentences),
            'strategy': 'extractive',
            'compression_ratio': len(summary) / len(text) if len(text) > 0 else 0
        }
    
    def _abstractive_summarize(self, text: str, max_length: int) -> Dict[str, Any]:
        """Abstractive summarization (placeholder)."""
        # In a real implementation, this would use transformer models
        return {
            'success': True,
            'summary': text[:max_length] + '...' if len(text) > max_length else text,
            'original_length': len(text),
            'summary_length': min(len(text), max_length),
            'strategy': 'abstractive',
            'note': 'Abstractive summarization not yet implemented',
            'compression_ratio': min(len(text), max_length) / len(text) if len(text) > 0 else 0
        }
    
    def _hybrid_summarize(self, text: str, max_sentences: int, 
                         max_length: int) -> Dict[str, Any]:
        """Hybrid summarization (placeholder)."""
        # Combine extractive and abstractive approaches
        extractive_result = self._extractive_summarize(text, max_sentences, max_length)
        
        if extractive_result['success']:
            return {
                'success': True,
                'summary': extractive_result['summary'],
                'original_length': extractive_result['original_length'],
                'summary_length': extractive_result['summary_length'],
                'strategy': 'hybrid',
                'note': 'Currently using extractive fallback',
                'compression_ratio': extractive_result['compression_ratio']
            }
        else:
            return extractive_result
    
    def _split_sentences(self, text: str) -> List[str]:
        """Simple sentence splitting."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Simple sentence splitting on punctuation
        sentences = re.split(r'[.!?]+', text)
        
        # Filter out empty strings
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def batch_summarize(self, texts: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Summarize multiple texts."""
        results = []
        for text in texts:
            result = self.summarize(text, **kwargs)
            results.append(result)
        return results
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return {
            'strategy': self.strategy,
            'config': self.config,
            'initialized': self.initialized
        }
    
    def update_config(self, new_config: Dict[str, Any]) -> bool:
        """Update configuration."""
        try:
            if 'strategy' in new_config:
                self.strategy = new_config['strategy']
                self.initialized = False  # Need reinitialization
            
            if 'config' in new_config:
                self.config.update(new_config['config'])
            
            logger.info("Summarizer configuration updated")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update config: {e}")
            return False


def get_summarizer(strategy: str = "extractive", config: Optional[Dict[str, Any]] = None) -> Summarizer:
    """Factory function to get a Summarizer instance."""
    return Summarizer(strategy, config)


# Test the summarizer
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("Testing Summarizer...")
    print("=" * 50)
    
    # Create summarizer
    summarizer = Summarizer(strategy="extractive")
    
    # Test text
    test_text = """
    Artificial intelligence is transforming the way we work and live. 
    AI systems can now understand natural language, recognize images, 
    and make predictions based on data. These capabilities are being 
    applied across industries from healthcare to finance. However, 
    there are also concerns about job displacement and ethical implications.
    Despite these challenges, AI continues to advance rapidly.
    """
    
    # Summarize
    result = summarizer.summarize(test_text, max_sentences=2, max_length=100)
    
    print(f"Original text ({result['original_length']} chars):")
    print(test_text[:200] + "...")
    print()
    print(f"Summary ({result['summary_length']} chars):")
    print(result['summary'])
    print()
    print(f"Strategy: {result['strategy']}")
    print(f"Compression ratio: {result['compression_ratio']:.2f}")
    print(f"Success: {result['success']}")
    
    print("=" * 50)
    print("Summarizer test complete.")