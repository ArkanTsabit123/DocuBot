# docubot/src/ai_engine/embedding_service.py

"""
DocuBot Embedding Service
Core module for text embedding generation using Sentence Transformers.
Provides model management, caching, and embedding operations for RAG pipeline.
"""

import os
import sys
import logging
import warnings
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime

import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import psutil

# Configure logging
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


class EmbeddingModelConfig:
    """Configuration for embedding models."""
    
    def __init__(self, 
                 model_name: str = "all-MiniLM-L6-v2",
                 device: str = "auto",
                 batch_size: int = 32,
                 normalize_embeddings: bool = True,
                 show_progress_bar: bool = False,
                 cache_size: int = 1000,
                 cache_dir: Optional[str] = None):
        
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.show_progress_bar = show_progress_bar
        self.cache_size = cache_size
        self.cache_dir = cache_dir or str(Path.home() / ".docubot" / "cache" / "embeddings")
        
        # Model registry
        self.model_registry = {
            "all-MiniLM-L6-v2": {
                "name": "all-MiniLM-L6-v2",
                "display_name": "MiniLM L6 v2",
                "dimensions": 384,
                "context_length": 256,
                "size_mb": 90,
                "speed": "Fast",
                "accuracy": "Good",
                "languages": ["en"],
                "default": True,
                "description": "Fast and efficient embedding model (384 dimensions)"
            },
            "all-mpnet-base-v2": {
                "name": "all-mpnet-base-v2",
                "display_name": "MPNet Base v2",
                "dimensions": 768,
                "context_length": 384,
                "size_mb": 420,
                "speed": "Medium",
                "accuracy": "Excellent",
                "languages": ["en"],
                "default": False,
                "description": "High-quality embedding model (768 dimensions)"
            },
            "paraphrase-multilingual-MiniLM-L12-v2": {
                "name": "paraphrase-multilingual-MiniLM-L12-v2",
                "display_name": "Multilingual MiniLM L12 v2",
                "dimensions": 384,
                "context_length": 128,
                "size_mb": 480,
                "speed": "Medium",
                "accuracy": "Good",
                "languages": ["en", "id", "es", "fr", "de", "zh"],
                "default": False,
                "description": "Multilingual embedding model (384 dimensions)"
            }
        }
    
    def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get information about a specific model."""
        name = model_name or self.model_name
        return self.model_registry.get(name, self.model_registry["all-MiniLM-L6-v2"]).copy()
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of all available models."""
        return [info.copy() for info in self.model_registry.values()]
    
    def validate_model_name(self, model_name: str) -> bool:
        """Validate if model name is supported."""
        return model_name in self.model_registry


class EmbeddingCache:
    """LRU cache for embeddings with disk persistence."""
    
    def __init__(self, cache_dir: str, max_size: int = 1000):
        """
        Initialize embedding cache.
        
        Args:
            cache_dir: Directory for cache persistence
            max_size: Maximum number of cache entries
        """
        self.cache_dir = Path(cache_dir)
        self.max_size = max_size
        self.memory_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self._initialize_cache()
    
    def _initialize_cache(self):
        """Initialize cache directory and load existing cache."""
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Load cache metadata if exists
            metadata_file = self.cache_dir / "cache_metadata.json"
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    self.cache_hits = metadata.get('hits', 0)
                    self.cache_misses = metadata.get('misses', 0)
            
            logger.info(f"Embedding cache initialized at {self.cache_dir}")
        except Exception as e:
            logger.warning(f"Failed to initialize cache directory: {e}")
    
    def _get_cache_key(self, text: str, model_name: str) -> str:
        """Generate cache key from text and model name."""
        content = f"{model_name}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, text: str, model_name: str) -> Optional[np.ndarray]:
        """
        Get embedding from cache.
        
        Args:
            text: Text to get embedding for
            model_name: Name of the model used
            
        Returns:
            Cached embedding or None
        """
        key = self._get_cache_key(text, model_name)
        
        # Check memory cache first
        if key in self.memory_cache:
            self.cache_hits += 1
            return self.memory_cache[key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{key}.npy"
        if cache_file.exists():
            try:
                embedding = np.load(str(cache_file))
                self.memory_cache[key] = embedding
                self.cache_hits += 1
                return embedding
            except Exception as e:
                logger.debug(f"Failed to load cached embedding: {e}")
        
        self.cache_misses += 1
        return None
    
    def set(self, text: str, model_name: str, embedding: np.ndarray):
        """
        Store embedding in cache.
        
        Args:
            text: Text associated with embedding
            model_name: Name of the model used
            embedding: Embedding vector to cache
        """
        try:
            key = self._get_cache_key(text, model_name)
            
            # Update memory cache
            if len(self.memory_cache) >= self.max_size:
                # Remove oldest entry (FIFO)
                oldest_key = next(iter(self.memory_cache))
                del self.memory_cache[oldest_key]
            
            self.memory_cache[key] = embedding
            
            # Save to disk
            cache_file = self.cache_dir / f"{key}.npy"
            np.save(str(cache_file), embedding)
            
            # Save metadata periodically
            if (self.cache_hits + self.cache_misses) % 100 == 0:
                self._save_metadata()
                
        except Exception as e:
            logger.warning(f"Failed to cache embedding: {e}")
    
    def _save_metadata(self):
        """Save cache metadata to disk."""
        try:
            metadata = {
                'hits': self.cache_hits,
                'misses': self.cache_misses,
                'size': len(self.memory_cache),
                'max_size': self.max_size,
                'updated_at': datetime.now().isoformat()
            }
            
            metadata_file = self.cache_dir / "cache_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
                
        except Exception as e:
            logger.debug(f"Failed to save cache metadata: {e}")
    
    def clear(self):
        """Clear all cache entries."""
        try:
            self.memory_cache.clear()
            
            # Remove cache files
            for cache_file in self.cache_dir.glob("*.npy"):
                try:
                    cache_file.unlink()
                except:
                    pass
            
            # Remove metadata
            metadata_file = self.cache_dir / "cache_metadata.json"
            if metadata_file.exists():
                try:
                    metadata_file.unlink()
                except:
                    pass
            
            self.cache_hits = 0
            self.cache_misses = 0
            
            logger.info("Embedding cache cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        hit_rate = 0
        if self.cache_hits + self.cache_misses > 0:
            hit_rate = self.cache_hits / (self.cache_hits + self.cache_misses)
        
        return {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'hit_rate': hit_rate,
            'size': len(self.memory_cache),
            'max_size': self.max_size,
            'cache_dir': str(self.cache_dir)
        }


class EmbeddingService:
    """
    Main embedding service for DocuBot.
    Handles model loading, embedding generation, and cache management.
    """
    
    def __init__(self, config: Optional[EmbeddingModelConfig] = None):
        """
        Initialize embedding service.
        
        Args:
            config: EmbeddingModelConfig instance
        """
        self.config = config or EmbeddingModelConfig()
        self.model = None
        self.device = None
        self.initialized = False
        
        # Initialize cache
        self.cache = EmbeddingCache(
            cache_dir=self.config.cache_dir,
            max_size=self.config.cache_size
        )
        
        # Performance tracking
        self.embedding_count = 0
        self.total_processing_time = 0.0
        
        logger.info(f"Initializing EmbeddingService with model: {self.config.model_name}")
    
    def _detect_device(self) -> str:
        """
        Detect and select the best available device.
        
        Returns:
            Device string ('cpu', 'cuda', 'mps')
        """
        if self.config.device != "auto":
            return self.config.device
        
        # Auto-detection logic
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"CUDA available with {gpu_count} GPU(s)")
            return "cuda"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            logger.info("MPS (Apple Silicon) available")
            return "mps"
        else:
            logger.info("Using CPU for embeddings")
            return "cpu"
    
    def _validate_device(self, device: str) -> str:
        """
        Validate and adjust device selection.
        
        Args:
            device: Requested device
            
        Returns:
            Validated device string
        """
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            return "cpu"
        elif device == "mps" and (not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available()):
            logger.warning("MPS requested but not available. Falling back to CPU.")
            return "cpu"
        
        return device
    
    def _get_model_path(self) -> Path:
        """Get path for model storage."""
        if self.config.cache_dir:
            base_dir = Path(self.config.cache_dir).parent.parent / "models"
        else:
            base_dir = Path.home() / ".docubot" / "models"
        
        model_dir = base_dir / "sentence-transformers" / self.config.model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        return model_dir
    
    def initialize(self) -> bool:
        """
        Initialize the embedding model.
        
        Returns:
            True if initialization successful
        """
        if self.initialized and self.model is not None:
            return True
        
        try:
            start_time = datetime.now()
            
            # Detect and validate device
            self.device = self._detect_device()
            self.device = self._validate_device(self.device)
            
            logger.info(f"Loading embedding model '{self.config.model_name}' on {self.device}")
            
            # Get model storage path
            model_path = self._get_model_path()
            
            # Load model with appropriate settings
            self.model = SentenceTransformer(
                model_name_or_path=self.config.model_name,
                device=self.device,
                cache_folder=str(model_path.parent)
            )
            
            # Configure model settings
            if hasattr(self.model, 'max_seq_length'):
                # Adjust batch size based on sequence length
                if self.model.max_seq_length > 512:
                    self.config.batch_size = min(self.config.batch_size, 16)
            
            # Verify model loaded correctly - FIXED THIS VALIDATION
            test_text = "Test initialization"
            test_embedding = self.model.encode(
                test_text,
                batch_size=1,
                show_progress_bar=False,
                normalize_embeddings=self.config.normalize_embeddings,
                convert_to_numpy=True
            )
            
            # Fix: The model.encode returns 2D array for single text: shape (1, dimensions)
            if test_embedding is None or test_embedding.size == 0:
                raise ValueError("Model failed to produce valid embedding")
            
            # Ensure it's a proper vector
            if len(test_embedding.shape) == 2 and test_embedding.shape[0] == 1:
                # This is normal: we got shape (1, n) for single text
                test_embedding = test_embedding.flatten()
            
            if test_embedding.size != self.config.get_model_info()["dimensions"]:
                logger.warning(f"Embedding dimension mismatch: expected {self.config.get_model_info()['dimensions']}, got {test_embedding.size}")
            
            self.initialized = True
            initialization_time = (datetime.now() - start_time).total_seconds()
            
            logger.info(f"Embedding model initialized successfully in {initialization_time:.2f}s")
            logger.info(f"Model dimensions: {self.get_embedding_dimensions()}")
            if hasattr(self.model, 'max_seq_length'):
                logger.info(f"Max sequence length: {self.model.max_seq_length}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            self.model = None
            self.initialized = False
            return False
    
    def encode(self, 
               texts: Union[str, List[str]], 
               batch_size: Optional[int] = None,
               normalize_embeddings: Optional[bool] = None,
               show_progress_bar: Optional[bool] = None,
               use_cache: bool = True) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Encode text(s) into embeddings.
        
        Args:
            texts: Single text or list of texts
            batch_size: Override default batch size
            normalize_embeddings: Override default normalization
            show_progress_bar: Override progress bar setting
            use_cache: Whether to use embedding cache
            
        Returns:
            For single text: 1D numpy array
            For multiple texts: 2D numpy array (n_texts x dimensions)
        """
        if not self.initialized:
            if not self.initialize():
                raise RuntimeError("Embedding service failed to initialize")
        
        # Convert single text to list
        if isinstance(texts, str):
            texts = [texts]
            return_single = True
        else:
            return_single = False
        
        start_time = datetime.now()
        
        # Use configuration defaults if not overridden
        batch_size = batch_size or self.config.batch_size
        normalize_embeddings = normalize_embeddings if normalize_embeddings is not None else self.config.normalize_embeddings
        show_progress_bar = show_progress_bar if show_progress_bar is not None else self.config.show_progress_bar
        
        # Check cache first
        cached_embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        if use_cache:
            for i, text in enumerate(texts):
                cached = self.cache.get(text, self.config.model_name)
                if cached is not None:
                    cached_embeddings.append(cached)
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
        else:
            uncached_texts = texts
            uncached_indices = list(range(len(texts)))
        
        # Generate embeddings for uncached texts
        generated_embeddings = []
        if uncached_texts:
            try:
                generated = self.model.encode(
                    uncached_texts,
                    batch_size=batch_size,
                    show_progress_bar=show_progress_bar,
                    normalize_embeddings=normalize_embeddings,
                    convert_to_numpy=True
                )
                
                # Cache generated embeddings
                for text, embedding in zip(uncached_texts, generated):
                    self.cache.set(text, self.config.model_name, embedding)
                
                generated_embeddings = list(generated)
                
            except Exception as e:
                logger.error(f"Error encoding texts: {e}")
                raise
        
        # Combine cached and generated embeddings
        all_embeddings = [None] * len(texts)
        
        # Place cached embeddings
        cache_idx = 0
        for i, text in enumerate(texts):
            if i not in uncached_indices:
                all_embeddings[i] = cached_embeddings[cache_idx]
                cache_idx += 1
        
        # Place generated embeddings
        gen_idx = 0
        for i in uncached_indices:
            all_embeddings[i] = generated_embeddings[gen_idx]
            gen_idx += 1
        
        # Convert to numpy array
        embeddings = np.array(all_embeddings)
        
        # Update statistics
        processing_time = (datetime.now() - start_time).total_seconds()
        self.embedding_count += len(texts)
        self.total_processing_time += processing_time
        
        logger.debug(f"Encoded {len(texts)} texts in {processing_time:.3f}s "
                    f"(cached: {len(cached_embeddings)}, generated: {len(generated_embeddings)})")
        
        if return_single:
            # Return 1D array for single text
            return embeddings[0]
        else:
            return embeddings
    
    def get_embeddings(self, *args, **kwargs) -> Union[np.ndarray, List[np.ndarray]]:
        """Alias for encode method."""
        return self.encode(*args, **kwargs)
    
    def get_embedding_dimensions(self) -> int:
        """Get dimensions of embeddings."""
        if not self.initialized:
            if not self.initialize():
                # Return from registry if model not loaded
                model_info = self.config.get_model_info(self.config.model_name)
                return model_info.get("dimensions", 384)
        
        if self.model:
            try:
                return self.model.get_sentence_embedding_dimension()
            except:
                pass
        
        # Return from registry as fallback
        model_info = self.config.get_model_info(self.config.model_name)
        return model_info.get("dimensions", 384)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about current model."""
        model_info = self.config.get_model_info(self.config.model_name)
        
        # Add runtime information
        runtime_info = {
            'initialized': self.initialized,
            'device': self.device,
            'embedding_dimensions': self.get_embedding_dimensions(),
            'embedding_count': self.embedding_count,
            'avg_processing_time_ms': 0,
            'cache_stats': self.cache.get_stats()
        }
        
        if self.embedding_count > 0:
            runtime_info['avg_processing_time_ms'] = (
                self.total_processing_time / self.embedding_count * 1000
            )
        
        if self.initialized and self.model:
            try:
                runtime_info['max_sequence_length'] = getattr(self.model, 'max_seq_length', 256)
                if hasattr(self.model, 'tokenizer'):
                    runtime_info['tokenizer_name'] = getattr(self.model.tokenizer, 'name_or_path', 'unknown')
                runtime_info['model_class'] = self.model.__class__.__name__
            except:
                pass
        
        model_info.update(runtime_info)
        return model_info
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available embedding models with download status."""
        models = self.config.get_available_models()
        
        for model_info in models:
            model_name = model_info.get('name')
            if not model_name:
                continue
            
            # Check if model is downloaded
            model_path = self._get_model_path().parent / model_name
            model_info['downloaded'] = model_path.exists() and any(
                f.suffix in ['.bin', '.pt', '.pth', '.model'] 
                for f in model_path.iterdir() if f.is_file()
            )
            
            # Add validation status
            model_info['validated'] = self._validate_model_files(model_name)
        
        return models
    
    def _validate_model_files(self, model_name: str) -> bool:
        """Validate model files exist and are accessible."""
        try:
            model_path = self._get_model_path().parent / model_name
            
            if not model_path.exists():
                return False
            
            # Check for essential files
            required_files = ['config.json', 'pytorch_model.bin']
            existing_files = [f.name for f in model_path.iterdir() if f.is_file()]
            
            # Check config.json
            if 'config.json' not in existing_files:
                return False
            
            # Check for at least one model file
            model_extensions = ['.bin', '.pt', '.pth', '.model']
            model_files = [f for f in existing_files 
                          if any(f.endswith(ext) for ext in model_extensions)]
            
            return len(model_files) > 0
            
        except Exception:
            return False
    
    def switch_model(self, model_name: str) -> bool:
        """
        Switch to a different embedding model.
        
        Args:
            model_name: Name of model to switch to
            
        Returns:
            True if switch successful
        """
        if not self.config.validate_model_name(model_name):
            logger.error(f"Model not supported: {model_name}")
            return False
        
        if model_name == self.config.model_name and self.initialized:
            logger.info(f"Already using model: {model_name}")
            return True
        
        try:
            logger.info(f"Switching embedding model to: {model_name}")
            
            # Save current configuration
            old_model_name = self.config.model_name
            
            # Clear current model
            if self.model is not None:
                # Clean up model resources
                del self.model
                self.model = None
            
            # Update configuration
            self.config.model_name = model_name
            
            # Clear cache (optional, could keep but with different key)
            self.cache.clear()
            
            # Reinitialize
            self.initialized = False
            success = self.initialize()
            
            if success:
                logger.info(f"Successfully switched from {old_model_name} to {model_name}")
            else:
                # Revert on failure
                self.config.model_name = old_model_name
                logger.error(f"Failed to switch to model: {model_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error switching model: {e}")
            return False
    
    def compute_similarity(self, 
                          embedding1: Union[np.ndarray, List[float]], 
                          embedding2: Union[np.ndarray, List[float]]) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            Cosine similarity score (-1 to 1)
        """
        # Convert to numpy arrays if needed
        if not isinstance(embedding1, np.ndarray):
            embedding1 = np.array(embedding1, dtype=np.float32)
        if not isinstance(embedding2, np.ndarray):
            embedding2 = np.array(embedding2, dtype=np.float32)
        
        # Ensure vectors are 1D
        embedding1 = embedding1.flatten()
        embedding2 = embedding2.flatten()
        
        # Normalize vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        
        # Clamp to valid range (floating point errors)
        similarity = max(-1.0, min(1.0, similarity))
        
        return float(similarity)
    
    def find_similar(self, 
                    query_embedding: np.ndarray, 
                    candidate_embeddings: List[np.ndarray],
                    top_k: int = 5,
                    similarity_threshold: float = 0.0) -> List[Tuple[int, float]]:
        """
        Find most similar embeddings to query.
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embeddings
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of (index, similarity) tuples sorted by similarity
        """
        if not candidate_embeddings:
            return []
        
        similarities = []
        
        for i, candidate in enumerate(candidate_embeddings):
            similarity = self.compute_similarity(query_embedding, candidate)
            if similarity >= similarity_threshold:
                similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def batch_similarity(self, 
                        query_embeddings: np.ndarray, 
                        candidate_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute similarity matrix between query and candidate embeddings.
        
        Args:
            query_embeddings: Query embeddings matrix (n_queries x dim)
            candidate_embeddings: Candidate embeddings matrix (n_candidates x dim)
            
        Returns:
            Similarity matrix (n_queries x n_candidates)
        """
        # Normalize embeddings
        query_norm = np.linalg.norm(query_embeddings, axis=1, keepdims=True)
        candidate_norm = np.linalg.norm(candidate_embeddings, axis=1, keepdims=True)
        
        # Avoid division by zero
        query_norm[query_norm == 0] = 1e-10
        candidate_norm[candidate_norm == 0] = 1e-10
        
        query_normalized = query_embeddings / query_norm
        candidate_normalized = candidate_embeddings / candidate_norm
        
        # Compute cosine similarity matrix
        similarity_matrix = np.dot(query_normalized, candidate_normalized.T)
        
        # Clamp to valid range
        np.clip(similarity_matrix, -1.0, 1.0, out=similarity_matrix)
        
        return similarity_matrix
    
    def clear_cache(self):
        """Clear embedding cache."""
        self.cache.clear()
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = {
            'embedding_count': self.embedding_count,
            'total_processing_time_seconds': self.total_processing_time,
            'cache_stats': self.cache.get_stats()
        }
        
        if self.embedding_count > 0:
            stats.update({
                'avg_processing_time_ms': (self.total_processing_time / self.embedding_count) * 1000,
                'embeddings_per_second': self.embedding_count / self.total_processing_time if self.total_processing_time > 0 else 0
            })
        
        return stats
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of embedding service.
        
        Returns:
            Health check results
        """
        start_time = datetime.now()
        
        try:
            # Check model initialization
            model_ok = self.initialized and self.model is not None
            
            # Check device
            device_ok = self.device is not None
            
            # Test encoding
            test_text = "Health check test"
            test_embedding = None
            encoding_ok = False
            
            if model_ok:
                try:
                    test_embedding = self.encode(test_text, use_cache=False)
                    encoding_ok = test_embedding is not None and len(test_embedding) > 0
                except Exception as e:
                    logger.warning(f"Encoding test failed: {e}")
            
            # Check cache
            cache_stats = self.cache.get_stats()
            cache_ok = cache_stats['size'] < cache_stats['max_size'] * 0.9  # Not too full
            
            # Check resources
            memory = psutil.virtual_memory()
            memory_ok = memory.percent < 90
            
            # Calculate health score (0-100)
            health_score = 0
            if model_ok:
                health_score += 30
            if device_ok:
                health_score += 20
            if encoding_ok:
                health_score += 30
            if cache_ok:
                health_score += 10
            if memory_ok:
                health_score += 10
            
            # Determine status
            if health_score >= 80:
                status = "healthy"
            elif health_score >= 50:
                status = "degraded"
            else:
                status = "unhealthy"
            
            result = {
                'success': True,
                'status': status,
                'health_score': health_score,
                'model_initialized': model_ok,
                'device_available': device_ok,
                'encoding_functional': encoding_ok,
                'cache_healthy': cache_ok,
                'memory_ok': memory_ok,
                'performance_stats': self.get_performance_stats(),
                'model_info': self.get_model_info(),
                'check_duration_ms': (datetime.now() - start_time).total_seconds() * 1000,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add recommendations
            recommendations = []
            if not model_ok:
                recommendations.append("Reinitialize embedding model")
            if not encoding_ok:
                recommendations.append("Check model files and reinstall if necessary")
            if not cache_ok:
                recommendations.append("Clear embedding cache")
            if not memory_ok:
                recommendations.append("Free up system memory")
            
            if recommendations:
                result['recommendations'] = recommendations
            
            return result
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'success': False,
                'status': 'unhealthy',
                'health_score': 0,
                'error': str(e),
                'check_duration_ms': (datetime.now() - start_time).total_seconds() * 1000,
                'timestamp': datetime.now().isoformat()
            }
    
    def __del__(self):
        """Cleanup resources."""
        try:
            if hasattr(self, 'cache'):
                self.cache._save_metadata()
        except:
            pass


# Factory functions for service management
_global_embedding_service = None

def get_embedding_service(config: Optional[EmbeddingModelConfig] = None) -> EmbeddingService:
    """
    Get or create global embedding service instance.
    
    Args:
        config: Configuration for embedding service
        
    Returns:
        EmbeddingService instance
    """
    global _global_embedding_service
    
    if _global_embedding_service is None:
        _global_embedding_service = EmbeddingService(config)
        _global_embedding_service.initialize()
    elif config is not None and config.model_name != _global_embedding_service.config.model_name:
        # Switch model if different
        _global_embedding_service.switch_model(config.model_name)
    
    return _global_embedding_service

def create_embedding_service(config: Optional[EmbeddingModelConfig] = None) -> EmbeddingService:
    """
    Create a new embedding service instance.
    
    Args:
        config: Configuration for embedding service
        
    Returns:
        New EmbeddingService instance
    """
    service = EmbeddingService(config)
    service.initialize()
    return service


def test_embedding_service():
    """Test function for embedding service."""
    print("\n" + "=" * 80)
    print("DOCUBOT EMBEDDING SERVICE - TEST")
    print("=" * 80)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)-8s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    try:
        # Create configuration
        config = EmbeddingModelConfig(
            model_name="all-MiniLM-L6-v2",
            device="auto",
            batch_size=8,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # Create service
        print(f"\n1. Initializing service with model: {config.model_name}")
        service = EmbeddingService(config)
        
        if not service.initialize():
            print("   FAILED: Service initialization failed")
            return False
        
        print("   SUCCESS: Service initialized")
        print(f"   Device: {service.device}")
        print(f"   Dimensions: {service.get_embedding_dimensions()}")
        
        # Test single encoding
        print(f"\n2. Testing single text encoding:")
        test_text = "Test document for embedding"
        print(f"   Text: '{test_text}'")
        
        embedding = service.encode(test_text)
        print(f"   SUCCESS: Encoding completed")
        print(f"   Embedding shape: {embedding.shape}")
        print(f"   Embedding dtype: {embedding.dtype}")
        print(f"   Embedding length: {len(embedding)}")
        
        # Test batch encoding
        print(f"\n3. Testing batch encoding:")
        batch_texts = [
            "First test document for batch processing",
            "Second document with different content",
            "Third document for comprehensive testing"
        ]
        
        batch_embeddings = service.encode(batch_texts)
        print(f"   SUCCESS: Batch encoding completed")
        print(f"   Batch shape: {batch_embeddings.shape}")
        print(f"   Number of embeddings: {len(batch_embeddings)}")
        
        # Test similarity
        print(f"\n4. Testing similarity computation:")
        similarity = service.compute_similarity(
            batch_embeddings[0], 
            batch_embeddings[1]
        )
        print(f"   SUCCESS: Similarity computed")
        print(f"   Similarity score: {similarity:.4f}")
        
        # Test model information
        print(f"\n5. Testing model information:")
        model_info = service.get_model_info()
        print(f"   SUCCESS: Model information retrieved")
        print(f"   Model: {model_info.get('display_name', 'Unknown')}")
        print(f"   Dimensions: {model_info.get('dimensions', 'Unknown')}")
        
        # Test available models
        print(f"\n6. Testing available models:")
        available_models = service.get_available_models()
        print(f"   SUCCESS: Found {len(available_models)} available models")
        
        for model in available_models:
            status = "✓" if model.get('downloaded', False) else "○"
            default = " [DEFAULT]" if model.get('default', False) else ""
            print(f"   {status} {model.get('display_name', 'Unknown')}{default}")
        
        # Test cache statistics
        print(f"\n7. Testing cache system:")
        cache_stats = service.cache.get_stats()
        print(f"   Cache hits: {cache_stats['hits']}")
        print(f"   Cache misses: {cache_stats['misses']}")
        print(f"   Hit rate: {cache_stats['hit_rate']:.2%}")
        
        # Test performance statistics
        print(f"\n8. Testing performance statistics:")
        perf_stats = service.get_performance_stats()
        print(f"   Total embeddings processed: {perf_stats['embedding_count']}")
        
        # Test factory functions
        print(f"\n9. Testing factory functions:")
        global_service = get_embedding_service(config)
        print(f"   Global service retrieved: {global_service is not None}")
        
        new_service = create_embedding_service(config)
        print(f"   New service created: {new_service is not None}")
        
        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\nERROR: Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    """
    Command-line test for embedding service.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Test DocuBot Embedding Service")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2", 
                       help="Model to test")
    parser.add_argument("--text", type=str, default="Test document for embedding",
                       help="Text to encode")
    parser.add_argument("--batch", action="store_true",
                       help="Test batch encoding")
    parser.add_argument("--health", action="store_true",
                       help="Run health check")
    parser.add_argument("--verbose", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Run test
    success = test_embedding_service()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)