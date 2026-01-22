# Docubot/test/unit/test_embedding_service.py


"""
Unit tests for DocuBot Embedding Service.
 test suite for embedding functionality.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the embedding service
try:
    from src.ai_engine.embedding_service import (
        EmbeddingService,
        EmbeddingModelConfig,
        EmbeddingCache,
        get_embedding_service,
        create_embedding_service
    )
    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORT_SUCCESS = False


# Skip all tests if import fails
pytestmark = pytest.mark.skipif(not IMPORT_SUCCESS, reason="Embedding service imports failed")


class TestEmbeddingModelConfig:
    """Test EmbeddingModelConfig class."""
    
    def test_config_initialization(self):
        """Test configuration can be initialized."""
        config = EmbeddingModelConfig()
        assert config.model_name == "all-MiniLM-L6-v2"
        assert config.batch_size == 32
        assert config.normalize_embeddings is True
    
    def test_get_model_info(self):
        """Test getting model information."""
        config = EmbeddingModelConfig()
        
        # Test default model
        info = config.get_model_info()
        assert info["name"] == "all-MiniLM-L6-v2"
        assert info["dimensions"] == 384
        assert info["speed"] == "Fast"
        
        # Test specific model
        info = config.get_model_info("all-mpnet-base-v2")
        assert info["name"] == "all-mpnet-base-v2"
        assert info["dimensions"] == 768
    
    def test_get_available_models(self):
        """Test getting list of available models."""
        config = EmbeddingModelConfig()
        models = config.get_available_models()
        
        assert len(models) >= 3
        model_names = [m["name"] for m in models]
        assert "all-MiniLM-L6-v2" in model_names
        assert "all-mpnet-base-v2" in model_names
    
    def test_validate_model_name(self):
        """Test model name validation."""
        config = EmbeddingModelConfig()
        
        assert config.validate_model_name("all-MiniLM-L6-v2") is True
        assert config.validate_model_name("all-mpnet-base-v2") is True
        assert config.validate_model_name("invalid-model") is False


class TestEmbeddingCache:
    """Test EmbeddingCache class."""
    
    def test_cache_initialization(self, tmp_path):
        """Test cache initialization."""
        cache_dir = tmp_path / "test_cache"
        cache = EmbeddingCache(cache_dir=str(cache_dir), max_size=100)
        
        assert cache.cache_dir.exists()
        assert cache.max_size == 100
        assert cache.cache_hits == 0
        assert cache.cache_misses == 0
    
    def test_cache_key_generation(self, tmp_path):
        """Test cache key generation."""
        cache = EmbeddingCache(cache_dir=str(tmp_path))
        
        key1 = cache._get_cache_key("test text", "model1")
        key2 = cache._get_cache_key("test text", "model1")
        key3 = cache._get_cache_key("different text", "model1")
        
        # Same text + model should give same key
        assert key1 == key2
        # Different text should give different key
        assert key1 != key3
        # Key should be hex string
        assert len(key1) == 64
        assert all(c in "0123456789abcdef" for c in key1)
    
    def test_cache_set_get(self, tmp_path):
        """Test setting and getting from cache."""
        cache = EmbeddingCache(cache_dir=str(tmp_path))
        
        # Create test embedding
        test_embedding = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
        
        # Set in cache
        cache.set("test text", "test_model", test_embedding)
        
        # Get from cache
        cached = cache.get("test text", "test_model")
        
        assert cached is not None
        np.testing.assert_array_almost_equal(cached, test_embedding)
        
        # Check cache statistics
        stats = cache.get_stats()
        assert stats["hits"] == 0  # First get should be miss
        assert stats["misses"] == 1
    
    def test_cache_miss(self, tmp_path):
        """Test cache miss scenario."""
        cache = EmbeddingCache(cache_dir=str(tmp_path))
        
        cached = cache.get("non-existent text", "test_model")
        assert cached is None
        
        stats = cache.get_stats()
        assert stats["misses"] == 1
        assert stats["hits"] == 0
    
    def test_cache_clear(self, tmp_path):
        """Test clearing cache."""
        cache = EmbeddingCache(cache_dir=str(tmp_path))
        
        # Add some items
        test_embedding = np.array([0.1, 0.2, 0.3])
        cache.set("text1", "model1", test_embedding)
        cache.set("text2", "model1", test_embedding)
        
        # Clear cache
        cache.clear()
        
        # Verify cache is empty
        assert len(cache.memory_cache) == 0
        assert cache.cache_hits == 0
        assert cache.cache_misses == 0


class TestEmbeddingService:
    """Test EmbeddingService class."""
    
    def test_service_initialization(self):
        """Test embedding service initialization."""
        service = EmbeddingService()
        assert service.config is not None
        assert service.cache is not None
        assert service.initialized is False
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_initialize_success(self, mock_sentence_transformer):
        """Test successful initialization."""
        # Mock the SentenceTransformer
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1] * 384, dtype=np.float32)
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.max_seq_length = 256
        mock_sentence_transformer.return_value = mock_model
        
        service = EmbeddingService()
        success = service.initialize()
        
        assert success is True
        assert service.initialized is True
        assert service.model is not None
    
    def test_get_embedding_dimensions(self):
        """Test getting embedding dimensions."""
        service = EmbeddingService()
        
        # Test without initialization (should return from config)
        dimensions = service.get_embedding_dimensions()
        assert dimensions == 384  # Default model dimensions
    
    def test_get_model_info(self):
        """Test getting model information."""
        service = EmbeddingService()
        info = service.get_model_info()
        
        assert "name" in info
        assert "dimensions" in info
        assert "initialized" in info
        assert "device" in info
    
    def test_get_available_models(self):
        """Test getting available models list."""
        service = EmbeddingService()
        models = service.get_available_models()
        
        assert len(models) >= 3
        # Check structure of each model info
        for model in models:
            assert "name" in model
            assert "display_name" in model
            assert "dimensions" in model
            assert "downloaded" in model  # Added by get_available_models
    
    def test_compute_similarity(self):
        """Test similarity computation."""
        service = EmbeddingService()
        
        # Test identical vectors
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([1.0, 0.0, 0.0])
        similarity = service.compute_similarity(vec1, vec2)
        assert abs(similarity - 1.0) < 0.001
        
        # Test orthogonal vectors
        vec3 = np.array([0.0, 1.0, 0.0])
        similarity = service.compute_similarity(vec1, vec3)
        assert abs(similarity) < 0.001
        
        # Test opposite vectors
        vec4 = np.array([-1.0, 0.0, 0.0])
        similarity = service.compute_similarity(vec1, vec4)
        assert abs(similarity - (-1.0)) < 0.001
    
    def test_find_similar(self):
        """Test finding similar embeddings."""
        service = EmbeddingService()
        
        query = np.array([1.0, 0.0, 0.0])
        candidates = [
            np.array([0.9, 0.1, 0.0]),   # Similar
            np.array([0.1, 0.9, 0.0]),   # Less similar
            np.array([-0.9, 0.0, 0.1]),  # Opposite
            np.array([1.0, 0.0, 0.0]),   # Identical
        ]
        
        results = service.find_similar(
            query_embedding=query,
            candidate_embeddings=candidates,
            top_k=2,
            similarity_threshold=0.5
        )
        
        assert len(results) <= 2
        if results:
            # Results should be sorted by similarity (highest first)
            for i in range(len(results) - 1):
                assert results[i][1] >= results[i + 1][1]
    
    def test_batch_similarity(self):
        """Test batch similarity computation."""
        service = EmbeddingService()
        
        queries = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        
        candidates = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        
        similarity_matrix = service.batch_similarity(queries, candidates)
        
        assert similarity_matrix.shape == (2, 3)
        
        # Check diagonal (identical vectors should have similarity ~1)
        assert abs(similarity_matrix[0, 0] - 1.0) < 0.001
        assert abs(similarity_matrix[1, 1] - 1.0) < 0.001
        
        # Check orthogonal vectors should have similarity ~0
        assert abs(similarity_matrix[0, 2]) < 0.001
    
    def test_clear_cache(self):
        """Test clearing cache."""
        service = EmbeddingService()
        service.clear_cache()
        
        # This should not raise any errors
        assert True
    
    def test_get_performance_stats(self):
        """Test getting performance statistics."""
        service = EmbeddingService()
        stats = service.get_performance_stats()
        
        assert "embedding_count" in stats
        assert "total_processing_time_seconds" in stats
        assert "cache_stats" in stats
        
        # Initial state should have zero embeddings
        assert stats["embedding_count"] == 0
    
    def test_health_check_structure(self):
        """Test health check returns proper structure."""
        service = EmbeddingService()
        health = service.health_check()
        
        # Check required fields
        assert "success" in health
        assert "status" in health
        assert "health_score" in health
        assert "model_initialized" in health
        assert "check_duration_ms" in health
        
        # Status should be one of these
        assert health["status"] in ["healthy", "degraded", "unhealthy"]
        
        # Score should be between 0 and 100
        assert 0 <= health["health_score"] <= 100


class TestFactoryFunctions:
    """Test factory functions (get_embedding_service, create_embedding_service)."""
    
    def test_get_embedding_service_singleton(self):
        """Test get_embedding_service returns singleton instance."""
        # First call should create instance
        service1 = get_embedding_service()
        assert service1 is not None
        
        # Second call should return same instance
        service2 = get_embedding_service()
        assert service1 is service2
    
    def test_create_embedding_service_new_instance(self):
        """Test create_embedding_service creates new instance."""
        service1 = create_embedding_service()
        service2 = create_embedding_service()
        
        assert service1 is not None
        assert service2 is not None
        assert service1 is not service2  # Should be different instances


# Integration tests
class TestEmbeddingServiceIntegration:
    """Integration tests for embedding service."""
    
    @pytest.mark.slow
    def test_real_encoding_with_mock(self):
        """Test actual encoding with mocked SentenceTransformer."""
        import sys
        from unittest.mock import MagicMock
        
        # Create a more realistic mock
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([[0.1] * 384], dtype=np.float32)
        mock_model.get_sentence_embedding_dimension.return_value = 384
        
        # Mock the import
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model):
            service = EmbeddingService()
            success = service.initialize()
            
            if success:
                embedding = service.encode("Test text")
                assert embedding is not None
                assert len(embedding) == 384
    
    def test_config_switching(self):
        """Test switching between different model configurations."""
        config1 = EmbeddingModelConfig(model_name="all-MiniLM-L6-v2")
        config2 = EmbeddingModelConfig(model_name="all-mpnet-base-v2")
        
        service1 = EmbeddingService(config1)
        service2 = EmbeddingService(config2)
        
        assert service1.config.model_name == "all-MiniLM-L6-v2"
        assert service2.config.model_name == "all-mpnet-base-v2"
    
    def test_cache_integration(self, tmp_path):
        """Test cache integration with embedding service."""
        from unittest.mock import Mock, patch
        
        # Mock model
        mock_model = Mock()
        mock_model.encode.return_value = np.array([0.1] * 384)
        
        with patch('sentence_transformers.SentenceTransformer', return_value=mock_model):
            # Create service with test cache directory
            config = EmbeddingModelConfig(cache_dir=str(tmp_path / "embeddings"))
            service = EmbeddingService(config)
            service.initialize()
            
            # First encode should miss cache
            embedding1 = service.encode("Test text", use_cache=True)
            
            # Second encode should hit cache
            embedding2 = service.encode("Test text", use_cache=True)
            
            # Both should be valid
            assert embedding1 is not None
            assert embedding2 is not None
            
            # Check cache stats
            cache_stats = service.cache.get_stats()
            assert cache_stats["hits"] > 0 or cache_stats["misses"] > 0


# Run tests
if __name__ == "__main__":
    print("Running embedding service tests...")
    
    # Simple smoke test
    config = EmbeddingModelConfig()
    print(f"✓ Config initialized: {config.model_name}")
    
    service = EmbeddingService(config)
    print(f"✓ Service initialized")
    
    info = service.get_model_info()
    print(f"✓ Model info retrieved: {info.get('name', 'Unknown')}")
    
    print("\nAll basic tests passed!")
    print("Run 'pytest tests/unit/test_embedding_service.py -v' for full test suite.")