"""
DocuBot - Model Manager Unit Tests

Comprehensive unit tests for the ModelManager class.
"""

import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, List, Any
import threading

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ai_engine.model_manager import (
    DownloadProgress,
    DownloadStatus,
    DiskSpaceError,
    DownloadCancelledError,
    ModelInfo,
    ModelManager,
    ModelManagerError,
    ModelNotFoundError,
    NetworkError,
)


@pytest.fixture
def temp_models_dir():
    """Create temporary models directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        models_dir = Path(temp_dir) / "data" / "models"
        models_dir.mkdir(parents=True)
        
        config_dir = Path(temp_dir) / "data" / "config"
        config_dir.mkdir(parents=True)
        
        config_file = config_dir / "llm_config.yaml"
        config_file.write_text("models:\n  test_model:\n    name: test")
        
        yield temp_dir, models_dir


@pytest.fixture
def model_manager(temp_models_dir):
    """Create ModelManager instance for testing."""
    temp_dir, models_dir = temp_models_dir
    with patch('src.ai_engine.model_manager.Path') as mock_path:
        mock_path.return_value = models_dir
        manager = ModelManager(Path(temp_dir) / "data" / "config" / "llm_config.yaml")
        yield manager


class TestModelManagerInitialization:
    """Test ModelManager initialization."""
    
    def test_model_manager_init(self, temp_models_dir):
        """Test ModelManager initialization."""
        temp_dir, models_dir = temp_models_dir
        manager = ModelManager(Path(temp_dir) / "data" / "config" / "llm_config.yaml")
        assert manager.models_dir == models_dir
        assert hasattr(manager, 'config')
    
    def test_init_without_config(self):
        """Test initialization without configuration file."""
        with patch('src.ai_engine.model_manager.Path') as mock_path:
            mock_path.return_value = Path("/fake/models")
            manager = ModelManager()
            assert manager.models_dir == Path("/fake/models")
    
    def test_init_creates_directory(self):
        """Test that initialization creates models directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            models_path = Path(temp_dir) / "new_models"
            with patch('src.ai_engine.model_manager.Path') as mock_path:
                mock_path.return_value = models_path
                manager = ModelManager()
                assert models_path.exists()


class TestModelListing:
    """Test model listing functionality."""
    
    @patch('subprocess.run')
    def test_list_available_models(self, mock_run, model_manager):
        """Test successful model listing."""
        mock_output = """NAME                ID              SIZE    MODIFIED       
llama2:7b          abc123          3.8GB   2 weeks ago
mistral:7b         def456          4.1GB   1 week ago"""
        
        mock_run.return_value = Mock(
            returncode=0,
            stdout=mock_output,
            stderr=""
        )
        
        models = model_manager.list_available_models()
        
        assert len(models) == 2
        assert models[0]['name'] == 'llama2:7b'
        assert models[1]['name'] == 'mistral:7b'
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_list_available_models_failure(self, mock_run, model_manager):
        """Test model listing failure."""
        mock_run.return_value = Mock(
            returncode=1,
            stdout="",
            stderr="Command failed"
        )
        
        with pytest.raises(ModelManagerError):
            model_manager.list_available_models()
    
    @patch('subprocess.run')
    def test_list_available_models_timeout(self, mock_run, model_manager):
        """Test model listing timeout."""
        mock_run.side_effect = TimeoutError("Command timed out")
        
        with pytest.raises(NetworkError):
            model_manager.list_available_models()


class TestModelDownload:
    """Test model download functionality."""
    
    @patch('subprocess.run')
    @patch('src.ai_engine.model_manager.ModelManager._check_disk_space')
    @patch('src.ai_engine.model_manager.ModelManager._get_model_size')
    def test_download_model_success(
        self, mock_get_size, mock_check_space, mock_run, model_manager
    ):
        """Test successful model download."""
        mock_get_size.return_value = 4 * 1024**3
        mock_check_space.return_value = True
        
        mock_process = Mock()
        mock_process.stdout.readline.side_effect = [
            "downloading layer 1/10 (100MB)",
            "downloading layer 2/10 (200MB)",
            ""
        ]
        mock_process.wait.return_value = 0
        
        with patch('subprocess.Popen', return_value=mock_process):
            progress = model_manager.download_model("llama2:7b")
            
            assert progress.model_name == "llama2:7b"
            assert progress.status == DownloadStatus.DOWNLOADING
        
        mock_get_size.assert_called_once_with("llama2:7b")
        mock_check_space.assert_called_once()
    
    @patch('subprocess.run')
    @patch('src.ai_engine.model_manager.ModelManager._check_disk_space')
    @patch('src.ai_engine.model_manager.ModelManager._get_model_size')
    def test_download_timeout(
        self, mock_get_size, mock_check_space, mock_run, model_manager
    ):
        """Test download timeout."""
        mock_get_size.return_value = 4 * 1024**3
        mock_check_space.return_value = True
        
        mock_process = Mock()
        mock_process.wait.side_effect = TimeoutError("Download timed out")
        
        with patch('subprocess.Popen', return_value=mock_process):
            progress = model_manager.download_model("llama2:7b")
            
            assert progress.status == DownloadStatus.FAILED
    
    @patch('subprocess.run')
    @patch('src.ai_engine.model_manager.ModelManager._check_disk_space')
    @patch('src.ai_engine.model_manager.ModelManager._get_model_size')
    def test_download_network_error(
        self, mock_get_size, mock_check_space, mock_run, model_manager
    ):
        """Test download network error."""
        mock_get_size.return_value = 4 * 1024**3
        mock_check_space.return_value = True
        
        mock_process = Mock()
        mock_process.wait.side_effect = ConnectionError("Network error")
        
        with patch('subprocess.Popen', return_value=mock_process):
            progress = model_manager.download_model("llama2:7b")
            
            assert progress.status == DownloadStatus.FAILED
    
    @patch('src.ai_engine.model_manager.ModelManager._check_disk_space')
    def test_download_insufficient_space(self, mock_check_space, model_manager):
        """Test download with insufficient disk space."""
        mock_check_space.return_value = False
        
        with pytest.raises(DiskSpaceError):
            model_manager.download_model("llama2:7b")
    
    def test_model_already_exists(self, model_manager):
        """Test download when model already exists."""
        with patch.object(model_manager, 'verify_model', return_value=True):
            with pytest.raises(ModelManagerError):
                model_manager.download_model("llama2:7b")
    
    def test_invalid_model_name(self, model_manager):
        """Test download with invalid model name."""
        with patch.object(model_manager, '_get_model_size', return_value=None):
            with pytest.raises(ModelNotFoundError):
                model_manager.download_model("invalid:model")
    
    @patch('subprocess.run')
    @patch('src.ai_engine.model_manager.ModelManager._check_disk_space')
    @patch('src.ai_engine.model_manager.ModelManager._get_model_size')
    def test_download_with_progress_callback(
        self, mock_get_size, mock_check_space, mock_run, model_manager
    ):
        """Test download with progress callback."""
        mock_get_size.return_value = 4 * 1024**3
        mock_check_space.return_value = True
        
        mock_process = Mock()
        mock_process.stdout.readline.side_effect = [
            "downloading layer 1/10 (100MB)",
            "downloading layer 2/10 (200MB)",
            ""
        ]
        mock_process.wait.return_value = 0
        
        callback_calls = []
        
        def progress_callback(progress):
            callback_calls.append(progress.progress_percentage)
        
        with patch('subprocess.Popen', return_value=mock_process):
            model_manager.download_model("llama2:7b", callback=progress_callback)
        
        assert len(callback_calls) > 0
    
    def test_get_download_progress(self, model_manager):
        """Test getting download progress."""
        progress = DownloadProgress(
            model_name="test_model",
            status=DownloadStatus.DOWNLOADING,
            progress_percentage=50.0,
            downloaded_bytes=2 * 1024**3,
            total_bytes=4 * 1024**3,
            speed_bytes_per_sec=1024**2,
            estimated_seconds_remaining=2000,
            start_time=time.time(),
            last_update=time.time()
        )
        
        model_manager._downloads["test_model"] = progress
        
        retrieved_progress = model_manager.get_download_progress("test_model")
        
        assert retrieved_progress == progress
        assert retrieved_progress.progress_percentage == 50.0


class TestDownloadCancellation:
    """Test download cancellation functionality."""
    
    def test_download_cancel(self, model_manager):
        """Test successful download cancellation."""
        mock_process = Mock()
        model_manager._active_downloads["test_model"] = mock_process
        
        result = model_manager.cancel_download("test_model")
        
        assert result is True
        mock_process.terminate.assert_called_once()
    
    def test_cancel_download_not_found(self, model_manager):
        """Test cancellation when download not found."""
        result = model_manager.cancel_download("nonexistent")
        
        assert result is False


class TestModelVerification:
    """Test model verification functionality."""
    
    @patch('subprocess.run')
    def test_verify_model_integrity(self, mock_run, model_manager):
        """Test model integrity verification."""
        mock_output = """digest: sha256:abc123...
status: verified"""
        
        mock_run.return_value = Mock(
            returncode=0,
            stdout=mock_output,
            stderr=""
        )
        
        result = model_manager.verify_model("llama2:7b")
        
        assert result is True
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_verify_model_success(self, mock_run, model_manager):
        """Test successful model verification."""
        mock_run.return_value = Mock(returncode=0)
        
        result = model_manager.verify_model("llama2:7b")
        
        assert result is True
        mock_run.assert_called_once()
    
    @patch('subprocess.run')
    def test_verify_model_failure(self, mock_run, model_manager):
        """Test model verification failure."""
        mock_run.return_value = Mock(returncode=1)
        
        result = model_manager.verify_model("llama2:7b")
        
        assert result is False


class TestModelInformation:
    """Test model information retrieval."""
    
    @patch('subprocess.run')
    def test_get_model_info(self, mock_run, model_manager):
        """Test successful model info retrieval."""
        mock_output = """display_name: Llama 2 7B
description: Meta's Llama 2 7B model
size: 3.8GB
parameter_size: 7B
context_length: 4096"""
        
        mock_run.return_value = Mock(
            returncode=0,
            stdout=mock_output,
            stderr=""
        )
        
        info = model_manager.get_model_info("llama2:7b")
        
        assert info is not None
        assert info.name == "llama2:7b"
        assert info.display_name == "Llama 2 7B"
        assert info.context_length == 4096
    
    @patch('subprocess.run')
    def test_get_model_info_failure(self, mock_run, model_manager):
        """Test model info retrieval failure."""
        mock_run.return_value = Mock(returncode=1)
        
        info = model_manager.get_model_info("llama2:7b")
        
        assert info is None


class TestModelDeletion:
    """Test model deletion functionality."""
    
    @patch('subprocess.run')
    def test_delete_model(self, mock_run, model_manager):
        """Test successful model deletion."""
        mock_run.return_value = Mock(returncode=0)
        
        result = model_manager.delete_model("llama2:7b")
        
        assert result is True
        mock_run.assert_called_with(
            ['ollama', 'rm', 'llama2:7b'],
            capture_output=True,
            text=True,
            timeout=30
        )
    
    @patch('subprocess.run')
    def test_delete_model_failure(self, mock_run, model_manager):
        """Test model deletion failure."""
        mock_run.return_value = Mock(returncode=1)
        
        result = model_manager.delete_model("llama2:7b")
        
        assert result is False
    
    @patch('subprocess.run')
    def test_delete_model_not_found(self, mock_run, model_manager):
        """Test delete model that doesn't exist."""
        mock_run.return_value = Mock(returncode=1, stderr="model not found")
        
        result = model_manager.delete_model("nonexistent")
        
        assert result is False


class TestCleanupFunctionality:
    """Test cleanup functionality."""
    
    @patch('subprocess.run')
    def test_cleanup_partial_downloads(self, mock_run, model_manager):
        """Test cleanup of partial downloads."""
        mock_output = "NAME                ID              SIZE    MODIFIED\nllama2:7b          abc123          3.8GB   2 weeks ago"
        
        mock_run.return_value = Mock(
            returncode=0,
            stdout=mock_output,
            stderr=""
        )
        
        progress = DownloadProgress(
            model_name="failed_model",
            status=DownloadStatus.FAILED,
            progress_percentage=0.0,
            downloaded_bytes=0,
            total_bytes=4 * 1024**3,
            speed_bytes_per_sec=0.0,
            estimated_seconds_remaining=None,
            start_time=time.time(),
            last_update=time.time()
        )
        
        model_manager._downloads["failed_model"] = progress
        
        with patch.object(model_manager, 'delete_model', return_value=True) as mock_delete:
            cleaned = model_manager.cleanup_partial_downloads()
            
            assert "failed_model" in cleaned
            mock_delete.assert_called_once_with("failed_model")


class TestRetryMechanism:
    """Test retry mechanism functionality."""
    
    def test_retry_mechanism_success(self, model_manager):
        """Test successful retry of failed download."""
        progress = DownloadProgress(
            model_name="failed_model",
            status=DownloadStatus.FAILED,
            progress_percentage=0.0,
            downloaded_bytes=0,
            total_bytes=4 * 1024**3,
            speed_bytes_per_sec=0.0,
            estimated_seconds_remaining=None,
            start_time=time.time(),
            last_update=time.time()
        )
        
        model_manager._downloads["failed_model"] = progress
        
        with patch.object(model_manager, 'download_model') as mock_download:
            model_manager.retry_failed_download("failed_model")
            
            mock_download.assert_called_once_with("failed_model", force=True)
    
    def test_retry_mechanism(self, model_manager):
        """Test retry mechanism for failed downloads."""
        # Create a failed download
        failed_progress = DownloadProgress(
            model_name="model1",
            status=DownloadStatus.FAILED,
            progress_percentage=30.0,
            downloaded_bytes=1.2 * 1024**3,
            total_bytes=4 * 1024**3,
            speed_bytes_per_sec=0.0,
            estimated_seconds_remaining=None,
            start_time=time.time(),
            last_update=time.time()
        )
        
        model_manager._downloads["model1"] = failed_progress
        
        # Mock the download function
        with patch.object(model_manager, 'download_model') as mock_download:
            # Retry the failed download
            model_manager.retry_failed_download("model1")
            
            # Verify retry was called
            mock_download.assert_called_once_with("model1", force=True)
            
            # Verify the old progress was cleared
            assert model_manager._downloads["model1"].status == DownloadStatus.FAILED
    
    def test_retry_completed_download(self, model_manager):
        """Test that completed downloads are not retried."""
        completed_progress = DownloadProgress(
            model_name="model2",
            status=DownloadStatus.COMPLETED,
            progress_percentage=100.0,
            downloaded_bytes=4 * 1024**3,
            total_bytes=4 * 1024**3,
            speed_bytes_per_sec=1024**2,
            estimated_seconds_remaining=0,
            start_time=time.time(),
            last_update=time.time()
        )
        
        model_manager._downloads["model2"] = completed_progress
        
        with patch.object(model_manager, 'download_model') as mock_download:
            result = model_manager.retry_failed_download("model2")
            
            # Should not call download_model for completed downloads
            mock_download.assert_not_called()
            assert result is None
    
    def test_retry_multiple_failed_downloads(self, model_manager):
        """Test retrying multiple failed downloads."""
        # Create multiple failed downloads
        for i in range(3):
            progress = DownloadProgress(
                model_name=f"model{i}",
                status=DownloadStatus.FAILED,
                progress_percentage=i * 20,
                downloaded_bytes=i * 0.8 * 1024**3,
                total_bytes=4 * 1024**3,
                speed_bytes_per_sec=0.0,
                estimated_seconds_remaining=None,
                start_time=time.time(),
                last_update=time.time()
            )
            model_manager._downloads[f"model{i}"] = progress
        
        with patch.object(model_manager, 'download_model') as mock_download:
            # Retry all failed downloads
            for i in range(3):
                model_manager.retry_failed_download(f"model{i}")
            
            # Verify each was retried
            assert mock_download.call_count == 3


class TestConcurrentDownloads:
    """Test concurrent download management."""
    
    def test_concurrent_downloads(self, model_manager):
        """Test managing multiple concurrent downloads."""
        # Simulate multiple active downloads
        mock_process1 = Mock()
        mock_process2 = Mock()
        
        model_manager._active_downloads["model1"] = mock_process1
        model_manager._active_downloads["model2"] = mock_process2
        
        # Test getting active downloads
        active_downloads = model_manager.get_active_downloads()
        assert len(active_downloads) == 2
        assert "model1" in active_downloads
        assert "model2" in active_downloads
        
        # Test cancelling one download
        result = model_manager.cancel_download("model1")
        assert result is True
        mock_process1.terminate.assert_called_once()
        
        # Verify only one download remains active
        active_downloads = model_manager.get_active_downloads()
        assert len(active_downloads) == 1
        assert "model2" in active_downloads
    
    def test_get_all_downloads(self, model_manager):
        """Test getting all tracked downloads."""
        # Create downloads with different statuses
        downloading = DownloadProgress(
            model_name="model1",
            status=DownloadStatus.DOWNLOADING,
            progress_percentage=50.0,
            downloaded_bytes=2 * 1024**3,
            total_bytes=4 * 1024**3,
            speed_bytes_per_sec=1024**2,
            estimated_seconds_remaining=2000,
            start_time=time.time(),
            last_update=time.time()
        )
        
        completed = DownloadProgress(
            model_name="model2",
            status=DownloadStatus.COMPLETED,
            progress_percentage=100.0,
            downloaded_bytes=4 * 1024**3,
            total_bytes=4 * 1024**3,
            speed_bytes_per_sec=1024**2,
            estimated_seconds_remaining=0,
            start_time=time.time(),
            last_update=time.time()
        )
        
        failed = DownloadProgress(
            model_name="model3",
            status=DownloadStatus.FAILED,
            progress_percentage=30.0,
            downloaded_bytes=1.2 * 1024**3,
            total_bytes=4 * 1024**3,
            speed_bytes_per_sec=0.0,
            estimated_seconds_remaining=None,
            start_time=time.time(),
            last_update=time.time()
        )
        
        model_manager._downloads["model1"] = downloading
        model_manager._downloads["model2"] = completed
        model_manager._downloads["model3"] = failed
        
        # Get all downloads
        all_downloads = model_manager.get_all_downloads()
        
        assert len(all_downloads) == 3
        assert all_downloads["model1"].status == DownloadStatus.DOWNLOADING
        assert all_downloads["model2"].status == DownloadStatus.COMPLETED
        assert all_downloads["model3"].status == DownloadStatus.FAILED
    
    def test_concurrent_download_limit(self, model_manager):
        """Test that concurrent downloads don't exceed limit."""
        # Create mock downloads up to the limit
        for i in range(5):  # Assuming limit is 3-5 concurrent downloads
            mock_process = Mock()
            model_manager._active_downloads[f"model{i}"] = mock_process
        
        # Try to start a new download
        with patch.object(model_manager, '_check_disk_space', return_value=True):
            with patch.object(model_manager, '_get_model_size', return_value=4 * 1024**3):
                # This should fail because too many concurrent downloads
                with pytest.raises(ModelManagerError):
                    model_manager.download_model("new_model")


class TestMockingFeatures:
    """Test various mocking features."""
    
    @patch('src.ai_engine.model_manager.shutil.disk_usage')
    def test_mock_disk_space_check(self, mock_disk_usage):
        """Test mocking disk space operations."""
        mock_disk_usage.return_value = Mock(free=100 * 1024**3)
        manager = ModelManager()
        
        result = manager._check_disk_space(4 * 1024**3)
        assert result is True
    
    @patch('src.ai_engine.model_manager.requests.get')
    def test_mock_network_calls(self, mock_get):
        """Test mocking network API calls."""
        mock_response = Mock()
        mock_response.json.return_value = {
            'models': [
                {'name': 'llama2:7b', 'size': 3800000000}
            ]
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        manager = ModelManager()
        with patch.object(manager, '_get_model_size', return_value=4 * 1024**3):
            size = manager._get_model_size("llama2:7b")
            assert size == 4 * 1024**3
    
    def test_mock_progress_callbacks(self):
        """Test mocking progress callbacks."""
        manager = ModelManager()
        
        mock_callback = Mock()
        progress = DownloadProgress(
            model_name="test_model",
            status=DownloadStatus.DOWNLOADING,
            progress_percentage=50.0,
            downloaded_bytes=2 * 1024**3,
            total_bytes=4 * 1024**3,
            speed_bytes_per_sec=1024**2,
            estimated_seconds_remaining=2000,
            start_time=time.time(),
            last_update=time.time()
        )
        
        mock_callback(progress)
        mock_callback.assert_called_once_with(progress)
    
    @patch('subprocess.run')
    def test_mock_ollama_api_responses(self, mock_run):
        """Test mocking Ollama API responses."""
        manager = ModelManager()
        
        mock_run.return_value = Mock(
            returncode=0,
            stdout='{"status": "success"}',
            stderr=''
        )
        
        with patch.object(manager, 'list_available_models') as mock_list:
            mock_list.return_value = [{'name': 'llama2:7b'}]
            models = manager.list_available_models()
            assert len(models) == 1


class TestConfigurationIntegration:
    """Test configuration file integration."""
    
    def test_load_model_configs(self, temp_models_dir):
        """Test loading model configurations from YAML."""
        temp_dir, models_dir = temp_models_dir
        config_path = Path(temp_dir) / "data" / "config" / "llm_config.yaml"
        
        config_content = """
models:
  llama2_7b:
    name: "llama2:7b"
    display_name: "Llama 2 7B"
    size: "3.8GB"
  mistral_7b:
    name: "mistral:7b"
    display_name: "Mistral 7B"
    size: "4.1GB"
"""
        config_path.write_text(config_content)
        
        manager = ModelManager(config_path)
        configs = manager._load_model_configs()
        
        assert "llama2_7b" in configs
        assert "mistral_7b" in configs
        assert configs["llama2_7b"]["name"] == "llama2:7b"
    
    def test_get_recommended_models(self, temp_models_dir):
        """Test getting recommended models from config."""
        temp_dir, models_dir = temp_models_dir
        config_path = Path(temp_dir) / "data" / "config" / "llm_config.yaml"
        
        config_content = """
models:
  model1:
    name: "llama2:7b"
  model2:
    name: "mistral:7b"
"""
        config_path.write_text(config_content)
        
        manager = ModelManager(config_path)
        recommended = manager.get_recommended_models()
        
        assert "model1" in recommended
        assert "model2" in recommended
        assert len(recommended) == 2
    
    def test_validate_model_config(self, temp_models_dir):
        """Test validation of model configuration."""
        temp_dir, models_dir = temp_models_dir
        config_path = Path(temp_dir) / "data" / "config" / "llm_config.yaml"
        
        config_content = """
models:
  invalid_model:
    # Missing required fields
"""
        config_path.write_text(config_content)
        
        manager = ModelManager(config_path)
        configs = manager._load_model_configs()
        
        assert "invalid_model" in configs


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])