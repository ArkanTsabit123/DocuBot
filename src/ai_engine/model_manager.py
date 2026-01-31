"""
DocuBot - Model Manager

Manages downloading, verification, and management of AI models
for the DocuBot application with comprehensive error handling
and progress tracking.
"""

import os
import sys
import json
import time
import shutil
import signal
import threading
import subprocess
import queue
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime
import concurrent.futures
from enum import Enum
import yaml


try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama = None


class DownloadStatus(Enum):
    PENDING = "pending"
    DOWNLOADING = "downloading"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL = "partial"


@dataclass
class DownloadProgress:
    model_name: str
    status: DownloadStatus
    progress_percentage: float
    downloaded_bytes: int
    total_bytes: Optional[int]
    speed_bytes_per_sec: float
    estimated_seconds_remaining: Optional[float]
    start_time: datetime
    last_update: datetime
    error_message: Optional[str] = None


@dataclass
class ModelInfo:
    name: str
    display_name: str
    description: Optional[str]
    size_bytes: int
    format: str
    family: str
    parameter_size: str
    quantization: str
    license: Optional[str]
    downloads: int
    tags: List[str]
    required_ram_gb: float
    required_vram_gb: Optional[float]
    context_length: int
    embedding_dimensions: Optional[int]


class ModelManagerError(Exception):
    pass


class NetworkError(ModelManagerError):
    pass


class DiskSpaceError(ModelManagerError):
    pass


class ModelNotFoundError(ModelManagerError):
    pass


class DownloadCancelledError(ModelManagerError):
    pass


class ModelManager:
    """Singleton manager for AI model operations."""
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, config_path: Optional[Path] = None):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, config_path: Optional[Path] = None):
        if self._initialized:
            return
            
        self.config_path = config_path or Path("data/config/llm_config.yaml")
        self.models_dir = Path("data/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self._downloads: Dict[str, DownloadProgress] = {}
        self._download_lock = threading.Lock()
        self._active_downloads: Dict[str, subprocess.Popen] = {}
        self._progress_callbacks: Dict[str, Callable[[DownloadProgress], None]] = {}
        
        self._download_queue = queue.Queue()
        self._queue_processor_thread = threading.Thread(
            target=self._process_download_queue,
            daemon=True
        )
        self._queue_processor_thread.start()
        
        self.model_configs = self._load_model_configs()
        self._initialized = True
    
    def _load_model_configs(self) -> Dict[str, Any]:
        """Load model configurations from YAML file."""
        default_config = {
            'models': {
                'llama2:7b': {
                    'display_name': 'Llama 2 7B',
                    'description': 'Meta Llama 2 7B model',
                    'size_gb': 3.8,
                    'required_ram_gb': 8,
                    'context_length': 4096
                },
                'mistral:7b': {
                    'display_name': 'Mistral 7B',
                    'description': 'Mistral 7B Instruct model',
                    'size_gb': 4.1,
                    'required_ram_gb': 8,
                    'context_length': 8192
                },
                'neural-chat:7b': {
                    'display_name': 'Neural Chat 7B',
                    'description': 'Intel Neural Chat 7B model',
                    'size_gb': 4.3,
                    'required_ram_gb': 8,
                    'context_length': 4096
                }
            },
            'default_model': 'llama2:7b',
            'timeout_seconds': 300,
            'retry_attempts': 3
        }
        
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                return config.get('models', default_config['models'])
        except (FileNotFoundError, yaml.YAMLError):
            pass
        
        return default_config['models']

    def _check_disk_space(self, required_bytes: int) -> bool:
        """Check if sufficient disk space is available."""
        try:
            total, used, free = shutil.disk_usage(self.models_dir)
            return free >= required_bytes
        except OSError as e:
            raise DiskSpaceError(f"Unable to check disk space: {e}")

    def _get_model_size(self, model_name: str) -> int:
        """Estimate model size in bytes."""
        model_config = self.model_configs.get(model_name)
        if model_config:
            size_gb = model_config.get('size_gb', 4)
            return int(size_gb * 1024**3)
        
        return 4 * 1024**3  # Default 4GB

    def _update_progress(self, model_name: str, **kwargs):
        """Update download progress information."""
        with self._download_lock:
            if model_name in self._downloads:
                progress = self._downloads[model_name]
                
                for key, value in kwargs.items():
                    if hasattr(progress, key):
                        setattr(progress, key, value)
                
                progress.last_update = datetime.now()
                
                if model_name in self._progress_callbacks:
                    try:
                        self._progress_callbacks[model_name](progress)
                    except Exception:
                        pass

    def list_available_models(self, local_only: bool = False) -> List[Dict[str, Any]]:
        """List available models from Ollama or configuration."""
        models = []
        
        if OLLAMA_AVAILABLE and not local_only:
            try:
                if ollama:
                    model_list = ollama.list()
                    for model in model_list.get('models', []):
                        models.append({
                            'name': model.get('name'),
                            'display_name': model.get('name'),
                            'size_gb': round(model.get('size', 0) / 1024**3, 1),
                            'modified': model.get('modified_at'),
                            'downloaded': True
                        })
            except Exception:
                pass
        
        if not models:
            for model_name, config in self.model_configs.items():
                models.append({
                    'name': model_name,
                    'display_name': config.get('display_name', model_name),
                    'size_gb': config.get('size_gb', 4),
                    'modified': '',
                    'downloaded': False
                })
        
        return models

    def download_model(
        self,
        model_name: str,
        callback: Optional[Callable[[DownloadProgress], None]] = None,
        force: bool = False
    ) -> DownloadProgress:
        """Initiate model download."""
        if model_name in self._downloads:
            progress = self._downloads[model_name]
            if progress.status == DownloadStatus.DOWNLOADING:
                raise ModelManagerError(f"Download already in progress for {model_name}")
        
        estimated_size = self._get_model_size(model_name)
        
        if not self._check_disk_space(estimated_size):
            raise DiskSpaceError(
                f"Insufficient disk space. Required: {estimated_size / 1024**3:.1f}GB"
            )
        
        progress = DownloadProgress(
            model_name=model_name,
            status=DownloadStatus.PENDING,
            progress_percentage=0.0,
            downloaded_bytes=0,
            total_bytes=estimated_size,
            speed_bytes_per_sec=0.0,
            estimated_seconds_remaining=None,
            start_time=datetime.now(),
            last_update=datetime.now()
        )
        
        with self._download_lock:
            self._downloads[model_name] = progress
            if callback:
                self._progress_callbacks[model_name] = callback
        
        self._download_queue.put((model_name, force))
        
        return progress

    def _process_download_queue(self):
        """Process download queue in background thread."""
        while True:
            try:
                model_name, force = self._download_queue.get()
                if model_name:
                    self._download_thread(model_name, force)
                self._download_queue.task_done()
            except Exception as e:
                print(f"Error in download queue processor: {e}")
            time.sleep(0.1)

    def _download_thread(self, model_name: str, force: bool):
        """Thread function for downloading models."""
        progress = self._downloads[model_name]
        
        try:
            self._update_progress(model_name, status=DownloadStatus.DOWNLOADING)
            
            if not OLLAMA_AVAILABLE:
                raise ModelManagerError("Ollama is not available")
            
            if ollama:
                result = ollama.pull(model_name, stream=True)
                
                for status in result:
                    if isinstance(status, dict) and 'completed' in status and 'total' in status:
                        completed = status['completed']
                        total = status['total']
                        
                        if total > 0:
                            progress_percentage = (completed / total) * 100
                            self._update_progress(
                                model_name,
                                progress_percentage=progress_percentage,
                                downloaded_bytes=completed,
                                total_bytes=total
                            )
                
                self._update_progress(
                    model_name,
                    status=DownloadStatus.COMPLETED,
                    progress_percentage=100.0,
                    downloaded_bytes=progress.total_bytes or 0
                )
            else:
                raise ModelManagerError("Ollama client not initialized")
        
        except Exception as e:
            self._update_progress(
                model_name,
                status=DownloadStatus.FAILED,
                error_message=str(e)
            )
        
        finally:
            with self._download_lock:
                if model_name in self._active_downloads:
                    del self._active_downloads[model_name]
                if model_name in self._progress_callbacks:
                    del self._progress_callbacks[model_name]

    def get_download_status(self, model_name: str) -> Optional[DownloadProgress]:
        """Get download status for a specific model."""
        with self._download_lock:
            return self._downloads.get(model_name)

    def cancel_download(self, model_name: str) -> bool:
        """Cancel an active download."""
        with self._download_lock:
            if model_name in self._active_downloads:
                process = self._active_downloads[model_name]
                process.terminate()
                
                if model_name in self._downloads:
                    self._downloads[model_name].status = DownloadStatus.CANCELLED
                    self._downloads[model_name].error_message = "Download cancelled by user"
                
                return True
            return False

    def delete_model(self, model_name: str) -> bool:
        """Delete a downloaded model."""
        try:
            if OLLAMA_AVAILABLE and ollama:
                ollama.delete(model_name)
                return True
            return False
        except Exception:
            return False

    def verify_model(self, model_name: str) -> bool:
        """Verify that a model is properly downloaded and accessible."""
        try:
            if OLLAMA_AVAILABLE and ollama:
                result = ollama.show(model_name)
                return result is not None
            return False
        except Exception:
            return False

    def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """Get detailed information about a model."""
        config = self.model_configs.get(model_name)
        if not config:
            return None
        
        return ModelInfo(
            name=model_name,
            display_name=config.get('display_name', model_name),
            description=config.get('description', ''),
            size_bytes=int(config.get('size_gb', 4) * 1024**3),
            format='gguf',
            family='unknown',
            parameter_size='7B',
            quantization='Q4_0',
            license='',
            downloads=0,
            tags=[],
            required_ram_gb=float(config.get('required_ram_gb', 8)),
            required_vram_gb=float(config.get('required_vram_gb', 0)),
            context_length=int(config.get('context_length', 4096)),
            embedding_dimensions=None
        )

    def cleanup_partial_downloads(self) -> List[str]:
        """Clean up incomplete or corrupted downloads."""
        cleaned = []
        to_remove = []
        
        with self._download_lock:
            for model_name, progress in self._downloads.items():
                if progress.status in [DownloadStatus.FAILED, DownloadStatus.PARTIAL]:
                    to_remove.append(model_name)
            
            for model_name in to_remove:
                self.delete_model(model_name)
                cleaned.append(model_name)
                del self._downloads[model_name]
        
        return cleaned

    def retry_failed_download(self, model_name: str) -> Optional[DownloadProgress]:
        """Retry a failed download."""
        with self._download_lock:
            if model_name in self._downloads:
                status = self._downloads[model_name].status
                if status in [DownloadStatus.FAILED, DownloadStatus.CANCELLED]:
                    del self._downloads[model_name]
        
        return self.download_model(model_name, force=True)

    def get_active_downloads(self) -> List[str]:
        """Get list of currently active downloads."""
        with self._download_lock:
            return list(self._active_downloads.keys())

    def get_all_downloads(self) -> Dict[str, DownloadProgress]:
        """Get all download progress information."""
        with self._download_lock:
            return self._downloads.copy()

    def get_recommended_models(self) -> List[str]:
        """Get list of recommended models."""
        return list(self.model_configs.keys())
    
    def get_default_model(self) -> str:
        """Get the default model name."""
        return 'llama2:7b'
    
    def is_model_downloaded(self, model_name: str) -> bool:
        """Check if a model is already downloaded."""
        try:
            if OLLAMA_AVAILABLE and ollama:
                model_list = ollama.list()
                for model in model_list.get('models', []):
                    if model.get('name') == model_name:
                        return True
            return False
        except Exception:
            return False


def get_model_manager(config=None) -> ModelManager:
    """Factory function to get ModelManager singleton instance."""
    return ModelManager(config)