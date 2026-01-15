"""
Embedding Service for DocuBot
"""

from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any, Optional
import logging
import numpy as np
from pathlib import Path
import pickle
import hashlib

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", cache_dir: Optional[Path] = None):
        """
        Initialize embedding service.
        
        Args:
            model_name: Name of the embedding model
            cache_dir: Directory for embedding cache
        """
        self.model_name = model_name
        
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info(f"Model loaded: {model_name} (dimensions: {self.model.get_sentence_embedding_dimension()})")
        
        if cache_dir is None:
            from ..core.constants import DATA_DIR
            cache_dir = DATA_DIR / "cache" / "embeddings"
        
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.cache_index = self._load_cache_index()
        
        logger.info(f"Embedding service initialized with cache: {self.cache_dir}")
    
    def generate_embeddings(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        cached_embeddings, uncached_texts = self._get_cached_embeddings(texts)
        
        if not uncached_texts:
            logger.debug(f"All {len(texts)} embeddings retrieved from cache")
            return cached_embeddings
        
        logger.debug(f"Generating embeddings for {len(uncached_texts)} uncached texts")
        
        try:
            new_embeddings = self.model.encode(
                uncached_texts,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            
            new_embeddings_list = new_embeddings.tolist()
            
            self._cache_embeddings(uncached_texts, new_embeddings_list)
            
            all_embeddings = []
            cache_idx = 0
            new_idx = 0
            
            for text in texts:
                if text in self.cache_index:
                    all_embeddings.append(cached_embeddings[cache_idx])
                    cache_idx += 1
                else:
                    all_embeddings.append(new_embeddings_list[new_idx])
                    new_idx += 1
            
            logger.info(f"Generated {len(uncached_texts)} new embeddings, {len(cached_embeddings)} from cache")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    def generate_single_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        embeddings = self.generate_embeddings([text])
        return embeddings[0] if embeddings else []
    
    def _get_cache_key(self, text: str) -> str:
        """
        Generate cache key for text.
        
        Args:
            text: Input text
            
        Returns:
            Cache key
        """
        hash_input = f"{self.model_name}:{text}"
        return hashlib.md5(hash_input.encode('utf-8')).hexdigest()
    
    def _get_cached_embeddings(self, texts: List[str]) -> tuple:
        """
        Get cached embeddings for texts.
        
        Args:
            texts: List of texts
            
        Returns:
            Tuple of (cached_embeddings, uncached_texts)
        """
        cached_embeddings = []
        uncached_texts = []
        
        for text in texts:
            cache_key = self._get_cache_key(text)
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            if cache_file.exists() and cache_key in self.cache_index:
                try:
                    with open(cache_file, 'rb') as f:
                        embedding = pickle.load(f)
                    cached_embeddings.append(embedding)
                except Exception as e:
                    logger.warning(f"Error loading cached embedding: {e}")
                    uncached_texts.append(text)
            else:
                uncached_texts.append(text)
        
        return cached_embeddings, uncached_texts
    
    def _cache_embeddings(self, texts: List[str], embeddings: List[List[float]]):
        """
        Cache embeddings.
        
        Args:
            texts: List of texts
            embeddings: List of embedding vectors
        """
        for text, embedding in zip(texts, embeddings):
            cache_key = self._get_cache_key(text)
            cache_file = self.cache_dir / f"{cache_key}.pkl"
            
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(embedding, f)
                
                self.cache_index[cache_key] = {
                    'text_hash': cache_key,
                    'model': self.model_name,
                    'timestamp': __import__('datetime').datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.warning(f"Error caching embedding: {e}")
        
        self._save_cache_index()
    
    def _load_cache_index(self) -> Dict[str, Any]:
        """
        Load cache index.
        
        Returns:
            Cache index dictionary
        """
        index_file = self.cache_dir / "index.pkl"
        
        if index_file.exists():
            try:
                with open(index_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                logger.warning(f"Error loading cache index: {e}")
        
        return {}
    
    def _save_cache_index(self):
        """Save cache index"""
        index_file = self.cache_dir / "index.pkl"
        
        try:
            with open(index_file, 'wb') as f:
                pickle.dump(self.cache_index, f)
        except Exception as e:
            logger.warning(f"Error saving cache index: {e}")
    
    def clear_cache(self) -> bool:
        """
        Clear embedding cache.
        
        Returns:
            True if successful
        """
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            
            self.cache_index = {}
            self._save_cache_index()
            
            logger.info(f"Cleared embedding cache: {self.cache_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Cache statistics
        """
        cache_files = list(self.cache_dir.glob("*.pkl"))
        embedding_files = [f for f in cache_files if f.name != "index.pkl"]
        
        total_size = sum(f.stat().st_size for f in cache_files)
        
        return {
            'cache_dir': str(self.cache_dir),
            'total_entries': len(embedding_files),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'model': self.model_name,
            'embedding_dimensions': self.model.get_sentence_embedding_dimension()
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model.
        
        Returns:
            Model information
        """
        return {
            'model_name': self.model_name,
            'embedding_dimensions': self.model.get_sentence_embedding_dimension(),
            'max_seq_length': self.model.max_seq_length,
            'device': str(self.model.device),
            'cache_enabled': True,
            'cache_entries': len(self.cache_index)
        }
    
    def similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Calculate cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score
        """
        if not embedding1 or not embedding2:
            return 0.0
        
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        return float(similarity)
    
    def find_similar(self, query_embedding: List[float], 
                    candidate_embeddings: List[List[float]], 
                    top_k: int = 5) -> List[tuple]:
        """
        Find most similar embeddings.
        
        Args:
            query_embedding: Query embedding
            candidate_embeddings: List of candidate embeddings
            top_k: Number of results to return
            
        Returns:
            List of (index, similarity) tuples
        """
        if not query_embedding or not candidate_embeddings:
            return []
        
        query_vec = np.array(query_embedding)
        candidate_matrix = np.array(candidate_embeddings)
        
        similarities = np.dot(candidate_matrix, query_vec) / (
            np.linalg.norm(candidate_matrix, axis=1) * np.linalg.norm(query_vec)
        )
        
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(int(idx), float(similarities[idx])) for idx in top_indices]


_embedding_instance = None

def get_embedding_service(model_name: str = "all-MiniLM-L6-v2", 
                         cache_dir: Optional[Path] = None) -> EmbeddingService:
    """
    Get or create EmbeddingService instance.
    
    Args:
        model_name: Name of the embedding model
        cache_dir: Directory for embedding cache
        
    Returns:
        EmbeddingService instance
    """
    global _embedding_instance
    
    if _embedding_instance is None:
        _embedding_instance = EmbeddingService(model_name, cache_dir)
    
    return _embedding_instance


if __name__ == "__main__":
    service = EmbeddingService()
    
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Artificial intelligence is transforming the world.",
        "Python is a popular programming language."
    ]
    
    print("Generating embeddings...")
    embeddings = service.generate_embeddings(test_texts)
    
    print(f"Generated {len(embeddings)} embeddings")
    print(f"Embedding dimensions: {len(embeddings[0])}")
    
    sim1 = service.similarity(embeddings[0], embeddings[1])
    sim2 = service.similarity(embeddings[0], embeddings[2])
    
    print(f"Similarity between text 1 and 2: {sim1:.4f}")
    print(f"Similarity between text 1 and 3: {sim2:.4f}")
    
    similar = service.find_similar(embeddings[0], embeddings, top_k=2)
    
    for idx, sim in similar:
        print(f"  Text {idx + 1}: similarity = {sim:.4f}")
    
    print("
Cache statistics:")
    stats = service.get_cache_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
