# docubot/src/ai_engine/model_manager.py

"""
DocuBot Model Management System
Centralized management for AI models including LLMs and embedding models.
Handles downloading, validation, system resource checking, and model lifecycle.
"""

import os
import sys
import json
import yaml
import time
import logging
import subprocess
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading
from dataclasses import dataclass, asdict, field

# Import checks with proper structure
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    psutil = None


@dataclass
class ModelRequirements:
    """Hardware and software requirements for AI models."""
    
    ram_gb: float = 8.0
    storage_gb: float = 5.0
    cpu_cores: int = 4
    gpu_vram_gb: Optional[float] = None
    python_version: str = "3.11"
    dependencies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelMetadata:
    """Metadata representation for AI models."""
    
    name: str
    display_name: str
    model_type: str
    provider: str
    description: str
    version: str = "1.0"
    license: str = "Apache 2.0"
    author: str = "DocuBot Team"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    
    parameters_billion: Optional[float] = None
    context_length: Optional[int] = None
    embedding_dimensions: Optional[int] = None
    
    requirements: ModelRequirements = field(default_factory=ModelRequirements)
    
    download_size_mb: float = 0.0
    installed_size_mb: float = 0.0
    download_url: Optional[str] = None
    checksum: Optional[str] = None
    
    is_downloaded: bool = False
    is_default: bool = False
    is_active: bool = False
    download_progress: float = 0.0
    last_used: Optional[str] = None
    usage_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['requirements'] = self.requirements.to_dict()
        return result
    
    def update_usage(self) -> None:
        self.usage_count += 1
        self.last_used = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()


class ModelDownloader:
    """Handles model downloading with progress tracking and validation."""
    
    def __init__(self, download_dir: Optional[str] = None):
        self.download_dir = Path(download_dir) if download_dir else Path.home() / ".docubot" / "downloads"
        self.download_dir.mkdir(parents=True, exist_ok=True)
        
        self.active_downloads: Dict[str, Dict[str, Any]] = {}
        self.download_history: List[Dict[str, Any]] = []
        
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.lock = threading.Lock()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"ModelDownloader initialized at {self.download_dir}")
    
    def download_model(self, metadata: ModelMetadata, force: bool = False, verify: bool = True) -> Dict[str, Any]:
        result = {
            'success': False,
            'model': metadata.name,
            'type': metadata.model_type,
            'message': '',
            'downloaded_bytes': 0,
            'total_bytes': 0,
            'duration_seconds': 0,
            'download_path': '',
            'verified': False,
            'timestamp': datetime.now().isoformat()
        }
        
        model_dir = self.download_dir / metadata.model_type / metadata.name
        if model_dir.exists() and not force:
            result['success'] = True
            result['message'] = f"Model already downloaded at {model_dir}"
            result['download_path'] = str(model_dir)
            return result
        
        if not metadata.download_url:
            result['message'] = "No download URL specified"
            return result
        
        start_time = time.time()
        download_id = None
        
        try:
            model_dir.mkdir(parents=True, exist_ok=True)
            
            download_id = f"{metadata.name}_{int(start_time)}"
            with self.lock:
                self.active_downloads[download_id] = {
                    'model': metadata.name,
                    'start_time': start_time,
                    'progress': 0.0,
                    'status': 'starting'
                }
            
            self.logger.info(f"Starting download: {metadata.name} from {metadata.download_url}")
            
            if metadata.download_url.startswith('http'):
                download_path = self._download_http(metadata, model_dir, download_id)
            else:
                download_path = self._copy_local(metadata, model_dir, download_id)
            
            metadata.download_progress = 100.0
            metadata.is_downloaded = True
            metadata.updated_at = datetime.now().isoformat()
            
            if verify and metadata.checksum:
                verified = self._verify_checksum(download_path, metadata.checksum)
                result['verified'] = verified
                if not verified:
                    self.logger.warning(f"Checksum verification failed for {metadata.name}")
            
            installed_size = self._calculate_directory_size(model_dir)
            metadata.installed_size_mb = installed_size / (1024 * 1024)
            
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)
            
            duration = time.time() - start_time
            
            result.update({
                'success': True,
                'message': f"Successfully downloaded {metadata.name}",
                'downloaded_bytes': installed_size,
                'total_bytes': installed_size,
                'duration_seconds': round(duration, 2),
                'download_path': str(model_dir),
                'download_speed_mbps': (installed_size / duration / (1024 * 1024)) if duration > 0 else 0
            })
            
            with self.lock:
                self.active_downloads.pop(download_id, None)
                self.download_history.append(result.copy())
            
            self.logger.info(f"Download completed: {metadata.name} in {duration:.1f}s")
            return result
            
        except Exception as e:
            duration = time.time() - start_time
            
            result.update({
                'message': f"Download failed: {str(e)}",
                'duration_seconds': round(duration, 2)
            })
            
            try:
                if model_dir.exists():
                    shutil.rmtree(model_dir)
            except Exception:
                pass
            
            if download_id:
                with self.lock:
                    self.active_downloads.pop(download_id, None)
            
            self.logger.error(f"Download failed for {metadata.name}: {e}")
            return result
    
    def _download_http(self, metadata: ModelMetadata, model_dir: Path, download_id: str) -> Path:
        filename = metadata.download_url.split('/')[-1]
        if not filename:
            filename = f"{metadata.name}.model"
        
        download_path = model_dir / filename
        
        try:
            import requests
            from tqdm import tqdm
            
            response = requests.get(metadata.download_url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            chunk_size = 8192
            
            with open(download_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, 
                         desc=f"Downloading {metadata.name}", leave=False) as pbar:
                    
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            pbar.update(len(chunk))
                            
                            progress = (downloaded / total_size * 100) if total_size > 0 else 0
                            with self.lock:
                                if download_id in self.active_downloads:
                                    self.active_downloads[download_id].update({
                                        'progress': progress,
                                        'downloaded_bytes': downloaded,
                                        'total_bytes': total_size,
                                        'status': 'downloading'
                                    })
            
            return download_path
            
        except ImportError:
            raise Exception("Required packages not installed: requests, tqdm")
        except Exception as e:
            raise Exception(f"HTTP download failed: {e}")
    
    def _copy_local(self, metadata: ModelMetadata, model_dir: Path, download_id: str) -> Path:
        source_path = Path(metadata.download_url)
        
        if not source_path.exists():
            raise Exception(f"Source file not found: {source_path}")
        
        with self.lock:
            if download_id in self.active_downloads:
                self.active_downloads[download_id].update({
                    'progress': 50.0,
                    'status': 'copying'
                })
        
        dest_path = model_dir / source_path.name
        shutil.copy2(source_path, dest_path)
        
        with self.lock:
            if download_id in self.active_downloads:
                self.active_downloads[download_id].update({
                    'progress': 100.0,
                    'status': 'completed'
                })
        
        return dest_path
    
    def _verify_checksum(self, file_path: Path, expected_checksum: str) -> bool:
        try:
            sha256_hash = hashlib.sha256()
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    sha256_hash.update(chunk)
            
            actual_checksum = sha256_hash.hexdigest()
            return actual_checksum == expected_checksum
            
        except Exception as e:
            self.logger.warning(f"Checksum verification error: {e}")
            return False
    
    def _calculate_directory_size(self, directory: Path) -> int:
        total_size = 0
        
        for file_path in directory.rglob('*'):
            if file_path.is_file():
                total_size += file_path.stat().st_size
        
        return total_size
    
    def get_active_downloads(self) -> List[Dict[str, Any]]:
        with self.lock:
            return list(self.active_downloads.values())
    
    def get_download_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        with self.lock:
            return self.download_history[-limit:]
    
    def cancel_download(self, model_name: str) -> bool:
        self.logger.warning(f"Cancelling downloads not fully implemented for {model_name}")
        return True
    
    def cleanup_downloads(self, max_age_days: int = 30) -> Dict[str, Any]:
        cutoff_time = time.time() - (max_age_days * 24 * 60 * 60)
        removed = 0
        total_freed = 0
        
        for model_type_dir in self.download_dir.iterdir():
            if not model_type_dir.is_dir():
                continue
            
            for model_dir in model_type_dir.iterdir():
                if not model_dir.is_dir():
                    continue
                
                try:
                    mtime = model_dir.stat().st_mtime
                    if mtime < cutoff_time:
                        size = self._calculate_directory_size(model_dir)
                        shutil.rmtree(model_dir)
                        removed += 1
                        total_freed += size
                except Exception as e:
                    self.logger.warning(f"Failed to cleanup {model_dir}: {e}")
        
        return {
            'removed_count': removed,
            'freed_bytes': total_freed,
            'freed_mb': total_freed / (1024 * 1024)
        }


class ModelRegistry:
    """Registry for all available models."""
    
    def __init__(self):
        self.llm_models: Dict[str, ModelMetadata] = {}
        self.embedding_models: Dict[str, ModelMetadata] = {}
        self.custom_models: Dict[str, ModelMetadata] = {}
        
        self._initialize_default_models()
    
    def _initialize_default_models(self) -> None:
        self.llm_models = {
            "llama2:7b": ModelMetadata(
                name="llama2:7b",
                display_name="Llama 2 7B",
                model_type="llm",
                provider="ollama",
                description="Meta's Llama 2 7B parameter model",
                parameters_billion=7.0,
                context_length=4096,
                requirements=ModelRequirements(
                    ram_gb=8.0,
                    storage_gb=4.2,
                    cpu_cores=4
                ),
                download_size_mb=4200,
                is_default=True
            ),
            "mistral:7b": ModelMetadata(
                name="mistral:7b",
                display_name="Mistral 7B",
                model_type="llm",
                provider="ollama",
                description="Mistral AI's 7B parameter model",
                parameters_billion=7.0,
                context_length=8192,
                requirements=ModelRequirements(
                    ram_gb=8.0,
                    storage_gb=4.1,
                    cpu_cores=4
                ),
                download_size_mb=4100,
                is_default=False
            ),
            "neural-chat:7b": ModelMetadata(
                name="neural-chat:7b",
                display_name="Neural Chat 7B",
                model_type="llm",
                provider="ollama",
                description="Intel's fine-tuned neural chat model",
                parameters_billion=7.0,
                context_length=4096,
                requirements=ModelRequirements(
                    ram_gb=8.0,
                    storage_gb=4.3,
                    cpu_cores=4
                ),
                download_size_mb=4300,
                is_default=False
            )
        }
        
        self.embedding_models = {
            "all-MiniLM-L6-v2": ModelMetadata(
                name="all-MiniLM-L6-v2",
                display_name="MiniLM L6 v2",
                model_type="embedding",
                provider="sentence-transformers",
                description="Fast and efficient embedding model (384 dimensions)",
                embedding_dimensions=384,
                context_length=256,
                requirements=ModelRequirements(
                    ram_gb=2.0,
                    storage_gb=0.09,
                    cpu_cores=2
                ),
                download_size_mb=90,
                is_default=True,
                download_url="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2"
            ),
            "all-mpnet-base-v2": ModelMetadata(
                name="all-mpnet-base-v2",
                display_name="MPNet Base v2",
                model_type="embedding",
                provider="sentence-transformers",
                description="High-quality embedding model (768 dimensions)",
                embedding_dimensions=768,
                context_length=384,
                requirements=ModelRequirements(
                    ram_gb=4.0,
                    storage_gb=0.42,
                    cpu_cores=2
                ),
                download_size_mb=420,
                is_default=False,
                download_url="https://huggingface.co/sentence-transformers/all-mpnet-base-v2"
            ),
            "paraphrase-multilingual-MiniLM-L12-v2": ModelMetadata(
                name="paraphrase-multilingual-MiniLM-L12-v2",
                display_name="Multilingual MiniLM L12 v2",
                model_type="embedding",
                provider="sentence-transformers",
                description="Multilingual embedding model (384 dimensions)",
                embedding_dimensions=384,
                context_length=128,
                requirements=ModelRequirements(
                    ram_gb=4.0,
                    storage_gb=0.48,
                    cpu_cores=2
                ),
                download_size_mb=480,
                is_default=False,
                download_url="https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
        }
    
    def get_model(self, model_name: str, model_type: Optional[str] = None) -> Optional[ModelMetadata]:
        if model_type == "llm":
            return self.llm_models.get(model_name)
        elif model_type == "embedding":
            return self.embedding_models.get(model_name)
        
        for registry in [self.llm_models, self.embedding_models, self.custom_models]:
            if model_name in registry:
                return registry[model_name]
        
        return None
    
    def get_all_models(self, model_type: Optional[str] = None) -> List[ModelMetadata]:
        if model_type == "llm":
            return list(self.llm_models.values())
        elif model_type == "embedding":
            return list(self.embedding_models.values())
        elif model_type == "custom":
            return list(self.custom_models.values())
        else:
            all_models = list(self.llm_models.values())
            all_models.extend(self.embedding_models.values())
            all_models.extend(self.custom_models.values())
            return all_models
    
    def register_model(self, metadata: ModelMetadata) -> bool:
        try:
            if metadata.model_type == "llm":
                self.llm_models[metadata.name] = metadata
            elif metadata.model_type == "embedding":
                self.embedding_models[metadata.name] = metadata
            else:
                self.custom_models[metadata.name] = metadata
            
            logging.info(f"Registered model: {metadata.name} ({metadata.model_type})")
            return True
        except Exception as e:
            logging.error(f"Failed to register model: {e}")
            return False
    
    def unregister_model(self, model_name: str) -> bool:
        for registry in [self.llm_models, self.embedding_models, self.custom_models]:
            if model_name in registry:
                del registry[model_name]
                logging.info(f"Unregistered model: {model_name}")
                return True
        
        return False
    
    def get_default_model(self, model_type: str) -> Optional[ModelMetadata]:
        if model_type == "llm":
            for model in self.llm_models.values():
                if model.is_default:
                    return model
        elif model_type == "embedding":
            for model in self.embedding_models.values():
                if model.is_default:
                    return model
        
        return None
    
    def set_default_model(self, model_name: str, model_type: str) -> bool:
        model = self.get_model(model_name, model_type)
        if not model:
            return False
        
        if model_type == "llm":
            for m in self.llm_models.values():
                m.is_default = False
            self.llm_models[model_name].is_default = True
        elif model_type == "embedding":
            for m in self.embedding_models.values():
                m.is_default = False
            self.embedding_models[model_name].is_default = True
        
        logging.info(f"Set default {model_type} model to: {model_name}")
        return True


class SystemResourceMonitor:
    """Monitors system resources for model requirements."""
    
    def __init__(self):
        self.history: List[Dict[str, Any]] = []
        self.max_history = 1000
        self.logger = logging.getLogger(__name__)
    
    def get_system_resources(self) -> Dict[str, Any]:
        try:
            if not PSUTIL_AVAILABLE:
                return {
                    'error': 'psutil not installed',
                    'timestamp': datetime.now().isoformat()
                }
            
            cpu_cores = os.cpu_count() or 1
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_freq = psutil.cpu_freq()
            
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            disk_info = {}
            for path in [Path.home(), Path.home() / ".docubot", Path("/")]:
                try:
                    usage = psutil.disk_usage(str(path))
                    disk_info[str(path)] = {
                        'total_gb': usage.total / (1024 ** 3),
                        'free_gb': usage.free / (1024 ** 3),
                        'used_gb': usage.used / (1024 ** 3),
                        'percent': usage.percent
                    }
                except Exception:
                    continue
            
            gpu_info = self._get_gpu_info()
            net_io = psutil.net_io_counters()
            
            import platform
            system_info = {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'python_version': platform.python_version(),
                'hostname': platform.node()
            }
            
            resources = {
                'cpu': {
                    'cores': cpu_cores,
                    'percent': cpu_percent,
                    'frequency_mhz': cpu_freq.current if cpu_freq else None,
                    'min_required_cores': 4
                },
                'ram': {
                    'total_gb': memory.total / (1024 ** 3),
                    'available_gb': memory.available / (1024 ** 3),
                    'used_gb': memory.used / (1024 ** 3),
                    'percent': memory.percent,
                    'swap_total_gb': swap.total / (1024 ** 3),
                    'swap_used_gb': swap.used / (1024 ** 3),
                    'min_required_gb': 8
                },
                'disk': disk_info,
                'gpu': gpu_info,
                'network': {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv
                },
                'system': system_info,
                'timestamp': datetime.now().isoformat(),
                'uptime_seconds': time.time() - psutil.boot_time()
            }
            
            self.history.append(resources.copy())
            if len(self.history) > self.max_history:
                self.history = self.history[-self.max_history:]
            
            return resources
            
        except Exception as e:
            self.logger.error(f"Error getting system resources: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        try:
            import torch
            if not torch.cuda.is_available():
                return {
                    'available': False,
                    'count': 0,
                    'devices': []
                }
            
            gpu_count = torch.cuda.device_count()
            devices = []
            
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                device_info = {
                    'index': i,
                    'name': torch.cuda.get_device_name(i),
                    'total_memory_gb': props.total_memory / (1024 ** 3),
                    'multi_processor_count': props.multi_processor_count,
                    'major': props.major,
                    'minor': props.minor,
                    'memory_allocated_gb': torch.cuda.memory_allocated(i) / (1024 ** 3),
                    'memory_reserved_gb': torch.cuda.memory_reserved(i) / (1024 ** 3)
                }
                devices.append(device_info)
            
            return {
                'available': True,
                'count': gpu_count,
                'devices': devices
            }
            
        except ImportError:
            return {'available': False, 'error': 'torch not installed'}
        except Exception as e:
            self.logger.warning(f"Error getting GPU info: {e}")
            return {
                'available': False,
                'error': str(e)
            }
    
    def check_requirements(self, requirements: ModelRequirements) -> Dict[str, Any]:
        resources = self.get_system_resources()
        
        if 'error' in resources:
            return {
                'success': False,
                'error': resources['error'],
                'requirements_met': False,
                'details': {}
            }
        
        checks = {}
        
        ram_total_gb = resources['ram']['total_gb']
        ram_ok = ram_total_gb >= requirements.ram_gb
        checks['ram'] = {
            'required_gb': requirements.ram_gb,
            'available_gb': ram_total_gb,
            'met': ram_ok,
            'message': f"RAM: {ram_total_gb:.1f}GB / {requirements.ram_gb}GB required"
        }
        
        cpu_cores = resources['cpu']['cores']
        cpu_ok = cpu_cores >= requirements.cpu_cores
        checks['cpu'] = {
            'required_cores': requirements.cpu_cores,
            'available_cores': cpu_cores,
            'met': cpu_ok,
            'message': f"CPU Cores: {cpu_cores} / {requirements.cpu_cores} required"
        }
        
        home_disk = resources['disk'].get(str(Path.home()), {})
        disk_free_gb = home_disk.get('free_gb', 0)
        disk_ok = disk_free_gb >= requirements.storage_gb
        checks['disk'] = {
            'required_gb': requirements.storage_gb,
            'available_gb': disk_free_gb,
            'met': disk_ok,
            'message': f"Disk Space: {disk_free_gb:.1f}GB / {requirements.storage_gb}GB required"
        }
        
        if requirements.gpu_vram_gb:
            gpu_available = resources['gpu']['available']
            if gpu_available and resources['gpu']['count'] > 0:
                gpu_memory_gb = resources['gpu']['devices'][0]['total_memory_gb']
                gpu_ok = gpu_memory_gb >= requirements.gpu_vram_gb
                checks['gpu'] = {
                    'required_vram_gb': requirements.gpu_vram_gb,
                    'available_vram_gb': gpu_memory_gb,
                    'met': gpu_ok,
                    'message': f"GPU VRAM: {gpu_memory_gb:.1f}GB / {requirements.gpu_vram_gb}GB required"
                }
            else:
                checks['gpu'] = {
                    'required_vram_gb': requirements.gpu_vram_gb,
                    'available_vram_gb': 0,
                    'met': False,
                    'message': "GPU required but not available"
                }
        
        python_version = resources['system']['python_version']
        
        try:
            import packaging.version
            current_version = packaging.version.parse(python_version)
            required_version = packaging.version.parse(requirements.python_version)
            python_ok = current_version >= required_version
        except ImportError:
            python_ok = python_version >= requirements.python_version
        except Exception:
            python_ok = False
        
        checks['python'] = {
            'required_version': requirements.python_version,
            'current_version': python_version,
            'met': python_ok,
            'message': f"Python: {python_version} / {requirements.python_version} required"
        }
        
        all_met = all(check['met'] for check in checks.values())
        
        return {
            'success': True,
            'requirements_met': all_met,
            'details': checks,
            'timestamp': datetime.now().isoformat()
        }
    
    def get_resource_history(self, metric: str = 'ram.percent', time_window_minutes: int = 60) -> List[Dict[str, Any]]:
        cutoff_time = time.time() - (time_window_minutes * 60)
        
        history = []
        for entry in self.history:
            if 'timestamp' in entry:
                try:
                    entry_time = datetime.fromisoformat(entry['timestamp']).timestamp()
                    if entry_time >= cutoff_time:
                        value = self._extract_nested_value(entry, metric)
                        if value is not None:
                            history.append({
                                'timestamp': entry['timestamp'],
                                'value': value
                            })
                except Exception:
                    continue
        
        return history
    
    def _extract_nested_value(self, data: Dict[str, Any], path: str) -> Any:
        keys = path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return None
        
        return current
    
    def get_resource_summary(self) -> Dict[str, Any]:
        resources = self.get_system_resources()
        
        if 'error' in resources:
            return resources
        
        summary = {
            'cpu': {
                'usage_percent': resources['cpu']['percent'],
                'cores': resources['cpu']['cores']
            },
            'ram': {
                'usage_percent': resources['ram']['percent'],
                'total_gb': resources['ram']['total_gb'],
                'available_gb': resources['ram']['available_gb']
            },
            'disk': {
                'home_free_gb': resources['disk'].get(str(Path.home()), {}).get('free_gb', 0)
            },
            'gpu': {
                'available': resources['gpu']['available'],
                'count': resources['gpu'].get('count', 0)
            },
            'system': resources['system']['system'],
            'timestamp': resources['timestamp']
        }
        
        return summary


class ModelManager:
    """Main model manager for DocuBot."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        self.registry = ModelRegistry()
        self.downloader = ModelDownloader(self.config.get('download_dir'))
        self.resource_monitor = SystemResourceMonitor()
        
        self.active_models: Dict[str, ModelMetadata] = {}
        self.model_usage: Dict[str, Dict[str, Any]] = {}
        
        self.ollama_available = self._check_ollama_available()
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("ModelManager initialized")
        self.logger.info(f"Ollama available: {self.ollama_available}")
    
    def _check_ollama_available(self) -> bool:
        try:
            result = subprocess.run(
                ['ollama', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
        except Exception as e:
            self.logger.warning(f"Error checking Ollama: {e}")
            return False
    
    def get_available_models(self, model_type: Optional[str] = None, include_status: bool = True) -> List[Dict[str, Any]]:
        models = self.registry.get_all_models(model_type)
        result = []
        
        for model in models:
            model_info = model.to_dict()
            
            if include_status:
                requirements_check = self.resource_monitor.check_requirements(model.requirements)
                model_info['requirements_check'] = requirements_check
                
                model_info['is_downloaded'] = self._check_model_downloaded(model)
                
                if model.model_type == "llm":
                    model_info['ollama_available'] = self.ollama_available
                
                if model.name in self.model_usage:
                    model_info['usage_stats'] = self.model_usage[model.name]
            
            result.append(model_info)
        
        return result
    
    def _check_model_downloaded(self, model: ModelMetadata) -> bool:
        if model.model_type == "embedding":
            try:
                from sentence_transformers import SentenceTransformer
                SentenceTransformer(model.name)
                return True
            except Exception:
                return False
        
        elif model.model_type == "llm" and self.ollama_available:
            try:
                result = subprocess.run(
                    ['ollama', 'list'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    for line in lines:
                        if model.name in line:
                            return True
            except Exception:
                pass
        
        return False
    
    def get_model_info(self, model_name: str, model_type: Optional[str] = None) -> Dict[str, Any]:
        model = self.registry.get_model(model_name, model_type)
        
        if not model:
            return {
                'success': False,
                'error': f"Model not found: {model_name}",
                'model_name': model_name
            }
        
        model_info = model.to_dict()
        requirements_check = self.resource_monitor.check_requirements(model.requirements)
        model_info['requirements_check'] = requirements_check
        model_info['is_downloaded'] = self._check_model_downloaded(model)
        
        if model_info['is_downloaded'] and model.model_type == "embedding":
            try:
                import sentence_transformers
                model_info['install_path'] = sentence_transformers.util.get_cache_folder()
            except Exception:
                pass
        
        if model_name in self.model_usage:
            model_info['usage_stats'] = self.model_usage[model_name]
        
        model_info['is_active'] = model_name in self.active_models
        model_info['success'] = True
        
        return model_info
    
    def validate_model_download(self, model_name: str, model_type: Optional[str] = None) -> Dict[str, Any]:
        model = self.registry.get_model(model_name, model_type)
        
        if not model:
            return {
                'success': False,
                'valid': False,
                'message': f"Model not found: {model_name}",
                'model_name': model_name,
                'timestamp': datetime.now().isoformat()
            }
        
        requirements_check = self.resource_monitor.check_requirements(model.requirements)
        disk_ok = self._check_disk_space(model.download_size_mb / 1024)
        
        network_ok = True
        if model.download_url and model.download_url.startswith('http'):
            network_ok = self._check_network_connectivity(model.download_url)
        
        valid = requirements_check['requirements_met'] and disk_ok and network_ok
        
        result = {
            'success': True,
            'valid': valid,
            'model_name': model_name,
            'model_type': model.model_type,
            'requirements_check': requirements_check,
            'disk_space_ok': disk_ok,
            'network_ok': network_ok,
            'download_size_mb': model.download_size_mb,
            'message': f"Model {model_name} can be downloaded" if valid else "System requirements not met for download",
            'timestamp': datetime.now().isoformat()
        }
        
        if not valid:
            issues = []
            if not requirements_check['requirements_met']:
                issues.append("System requirements not met")
            if not disk_ok:
                issues.append("Insufficient disk space")
            if not network_ok:
                issues.append("Network connectivity issue")
            result['issues'] = issues
        
        return result
    
    def _check_disk_space(self, required_gb: float) -> bool:
        try:
            if not PSUTIL_AVAILABLE:
                return False
            
            home_usage = psutil.disk_usage(str(Path.home()))
            free_gb = home_usage.free / (1024 ** 3)
            return free_gb >= required_gb
        except Exception as e:
            self.logger.warning(f"Error checking disk space: {e}")
            return False
    
    def _check_network_connectivity(self, url: str) -> bool:
        try:
            import urllib.request
            import socket
            
            socket.setdefaulttimeout(5)
            hostname = urllib.request.urlparse(url).hostname
            if hostname:
                socket.gethostbyname(hostname)
                return True
        except Exception:
            pass
        
        return False
    
    def download_model(self, model_name: str, model_type: Optional[str] = None, 
                      force: bool = False) -> Dict[str, Any]:
        """Download a model.
        
        Args:
            model_name: Name of the model
            model_type: Optional model type filter
            force: Force re-download even if exists
            
        Returns:
            Download results
        """
        if model_type == "llm" and self.ollama_available:
            try:
                result = subprocess.run(
                    ['ollama', 'pull', model_name],
                    capture_output=True,
                    text=True,
                    timeout=300
                )
                if result.returncode == 0:
                    model = self.registry.get_model(model_name, model_type)
                    if model:
                        model.is_downloaded = True
                        model.updated_at = datetime.now().isoformat()
                    
                    return {
                        'success': True,
                        'message': f"Model {model_name} downloaded successfully via Ollama",
                        'model_name': model_name,
                        'download_method': 'ollama_pull'
                    }
                else:
                    return {
                        'success': False,
                        'message': f"Failed to download {model_name} via Ollama: {result.stderr}",
                        'model_name': model_name,
                        'download_method': 'ollama_pull'
                    }
            except Exception as e:
                return {
                    'success': False,
                    'message': f"Error downloading {model_name} via Ollama: {e}",
                    'model_name': model_name,
                    'download_method': 'ollama_pull'
                }
        else:
            model = self.registry.get_model(model_name, model_type)
            if model:
                return self.downloader.download_model(model, force)
            else:
                return {
                    'success': False,
                    'message': f"Model not found: {model_name}",
                    'model_name': model_name,
                    'download_method': 'standard'
                }
    
    def get_system_resources(self) -> Dict[str, Any]:
        return self.resource_monitor.get_system_resources()
    
    def check_system_requirements(self, min_requirements: Optional[ModelRequirements] = None) -> Dict[str, Any]:
        if not min_requirements:
            min_requirements = ModelRequirements(
                ram_gb=8.0,
                storage_gb=10.0,
                cpu_cores=4
            )
        
        return self.resource_monitor.check_requirements(min_requirements)
    
    def validate_model_files(self, model_name: str, model_type: Optional[str] = None) -> Dict[str, Any]:
        model = self.registry.get_model(model_name, model_type)
        
        if not model:
            return {
                'success': False,
                'valid': False,
                'message': f"Model not found: {model_name}",
                'model_name': model_name
            }
        
        if not self._check_model_downloaded(model):
            return {
                'success': True,
                'valid': False,
                'message': f"Model not downloaded: {model_name}",
                'model_name': model_name,
                'files_found': [],
                'files_missing': ['model_files']
            }
        
        if model.model_type == "embedding":
            return self._validate_embedding_model_files(model)
        elif model.model_type == "llm":
            return self._validate_llm_model_files(model)
        
        return {
            'success': True,
            'valid': True,
            'message': f"Model files validated: {model_name}",
            'model_name': model_name
        }
    
    def _validate_embedding_model_files(self, model: ModelMetadata) -> Dict[str, Any]:
        try:
            from sentence_transformers import SentenceTransformer
            SentenceTransformer(model.name)
            
            return {
                'success': True,
                'valid': True,
                'message': f"Embedding model files validated: {model.name}",
                'model_name': model.name,
                'files_found': ['model_files'],
                'files_missing': []
            }
        except Exception as e:
            return {
                'success': False,
                'valid': False,
                'message': f"Failed to validate embedding model: {e}",
                'model_name': model.name,
                'files_found': [],
                'files_missing': ['model_files']
            }
    
    def _validate_llm_model_files(self, model: ModelMetadata) -> Dict[str, Any]:
        if not self.ollama_available:
            return {
                'success': False,
                'valid': False,
                'message': "Ollama not available",
                'model_name': model.name
            }
        
        try:
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return {
                    'success': False,
                    'valid': False,
                    'message': f"Ollama list failed: {result.stderr}",
                    'model_name': model.name
                }
            
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if model.name in line:
                    return {
                        'success': True,
                        'valid': True,
                        'message': f"LLM model validated via Ollama: {model.name}",
                        'model_name': model.name,
                        'files_found': ['ollama_model'],
                        'files_missing': []
                    }
            
            return {
                'success': True,
                'valid': False,
                'message': f"LLM model not found in Ollama: {model.name}",
                'model_name': model.name,
                'files_found': [],
                'files_missing': ['ollama_model']
            }
            
        except Exception as e:
            return {
                'success': False,
                'valid': False,
                'message': f"Error validating LLM model: {e}",
                'model_name': model.name
            }
    
    def set_active_model(self, model_name: str, model_type: str) -> bool:
        model = self.registry.get_model(model_name, model_type)
        
        if not model:
            self.logger.error(f"Cannot set active model: {model_name} not found")
            return False
        
        if not self._check_model_downloaded(model):
            self.logger.error(f"Cannot set active model: {model_name} not downloaded")
            return False
        
        self.active_models[model_type] = model
        
        if model_name not in self.model_usage:
            self.model_usage[model_name] = {
                'activation_count': 0,
                'last_activated': None,
                'total_activation_time': 0
            }
        
        self.model_usage[model_name]['activation_count'] += 1
        self.model_usage[model_name]['last_activated'] = datetime.now().isoformat()
        
        self.registry.set_default_model(model_name, model_type)
        
        self.logger.info(f"Set active {model_type} model to: {model_name}")
        return True
    
    def get_active_model(self, model_type: str) -> Optional[ModelMetadata]:
        return self.active_models.get(model_type)
    
    def health_check(self) -> Dict[str, Any]:
        start_time = time.time()
        
        try:
            resources = self.get_system_resources()
            resources_ok = 'error' not in resources
            
            sys_req_check = self.check_system_requirements()
            sys_req_ok = sys_req_check['requirements_met']
            
            ollama_ok = self.ollama_available
            
            default_llm = self.registry.get_default_model('llm')
            default_embedding = self.registry.get_default_model('embedding')
            
            default_llm_ok = default_llm is not None
            default_embedding_ok = default_embedding is not None
            
            default_llm_downloaded = default_llm_ok and self._check_model_downloaded(default_llm)
            default_embedding_downloaded = default_embedding_ok and self._check_model_downloaded(default_embedding)
            
            health_score = 0
            
            if resources_ok:
                health_score += 20
            
            if sys_req_ok:
                health_score += 20
            
            if ollama_ok:
                health_score += 10
            
            if default_llm_ok:
                health_score += 10
            
            if default_embedding_ok:
                health_score += 10
            
            if default_llm_downloaded:
                health_score += 10
            
            if default_embedding_downloaded:
                health_score += 20
            
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
                'checks': {
                    'system_resources': resources_ok,
                    'system_requirements': sys_req_ok,
                    'ollama_available': ollama_ok,
                    'default_llm_defined': default_llm_ok,
                    'default_embedding_defined': default_embedding_ok,
                    'default_llm_downloaded': default_llm_downloaded,
                    'default_embedding_downloaded': default_embedding_downloaded
                },
                'details': {
                    'system_resources': resources,
                    'system_requirements_check': sys_req_check,
                    'default_llm': default_llm.to_dict() if default_llm else None,
                    'default_embedding': default_embedding.to_dict() if default_embedding else None,
                    'active_models': {
                        model_type: model.to_dict() 
                        for model_type, model in self.active_models.items()
                    }
                },
                'recommendations': [],
                'check_duration_ms': round((time.time() - start_time) * 1000, 2),
                'timestamp': datetime.now().isoformat()
            }
            
            if not resources_ok:
                result['recommendations'].append("Check system resource monitoring")
            
            if not sys_req_ok:
                result['recommendations'].append("Upgrade system to meet minimum requirements")
            
            if not ollama_ok:
                result['recommendations'].append("Install Ollama from https://ollama.ai/")
            
            if not default_llm_ok:
                result['recommendations'].append("Configure a default LLM model")
            
            if not default_embedding_ok:
                result['recommendations'].append("Configure a default embedding model")
            
            if not default_llm_downloaded and default_llm_ok:
                result['recommendations'].append(f"Download default LLM model: {default_llm.name}")
            
            if not default_embedding_downloaded and default_embedding_ok:
                result['recommendations'].append(f"Download default embedding model: {default_embedding.name}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Health check failed: {e}")
            return {
                'success': False,
                'status': 'unhealthy',
                'health_score': 0,
                'error': str(e),
                'check_duration_ms': round((time.time() - start_time) * 1000, 2),
                'timestamp': datetime.now().isoformat()
            }
    
    def get_download_status(self) -> Dict[str, Any]:
        return {
            'active_downloads': self.downloader.get_active_downloads(),
            'download_history': self.downloader.get_download_history(limit=20),
            'timestamp': datetime.now().isoformat()
        }
    
    def format_file_size(self, size_bytes: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        
        return f"{size_bytes:.2f} PB"
    
    def cleanup(self, max_age_days: int = 30) -> Dict[str, Any]:
        cleanup_result = self.downloader.cleanup_downloads(max_age_days)
        
        self.logger.info(f"Cleanup completed: {cleanup_result['removed_count']} items removed, "
                       f"{cleanup_result['freed_mb']:.1f} MB freed")
        
        return cleanup_result
    
    def compare_models(self, model_names: List[str], model_type: Optional[str] = None) -> Dict[str, Any]:
        """Compare multiple models based on their specifications, requirements, and system compatibility."""
        comparison_result = {
            'success': False,
            'models': [],
            'comparison_metrics': {},
            'recommendations': [],
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            models = []
            
            for model_name in model_names:
                model_info = self.get_model_info(model_name, model_type)
                if not model_info.get('success', False):
                    self.logger.warning(f"Model not found for comparison: {model_name}")
                    continue
                
                models.append(model_info)
            
            if len(models) < 2:
                comparison_result['message'] = "At least two valid models required for comparison"
                return comparison_result
            
            comparison_result['success'] = True
            comparison_result['models'] = models
            
            comparison_metrics = self._generate_comparison_metrics(models)
            comparison_result['comparison_metrics'] = comparison_metrics
            
            recommendations = self._generate_model_recommendations(models, comparison_metrics)
            comparison_result['recommendations'] = recommendations
            
            best_model = self._select_best_model(models, comparison_metrics)
            if best_model:
                comparison_result['best_model'] = best_model
            
            comparison_result['message'] = f"Successfully compared {len(models)} models"
            
        except Exception as e:
            comparison_result['message'] = f"Error comparing models: {str(e)}"
            self.logger.error(f"Comparison failed: {e}")
        
        return comparison_result
    
    def _generate_comparison_metrics(self, models: List[Dict[str, Any]]) -> Dict[str, Any]:
        metrics = {
            'performance': [],
            'resource_requirements': [],
            'system_compatibility': [],
            'model_specifications': []
        }
        
        for model in models:
            model_name = model['name']
            
            performance_score = self._calculate_performance_score(model)
            metrics['performance'].append({
                'model': model_name,
                'score': performance_score,
                'parameters_billion': model.get('parameters_billion', 0),
                'context_length': model.get('context_length', 0),
                'embedding_dimensions': model.get('embedding_dimensions', 0)
            })
            
            resource_score = self._calculate_resource_score(model)
            metrics['resource_requirements'].append({
                'model': model_name,
                'score': resource_score,
                'ram_gb': model.get('requirements', {}).get('ram_gb', 0),
                'storage_gb': model.get('requirements', {}).get('storage_gb', 0),
                'gpu_vram_gb': model.get('requirements', {}).get('gpu_vram_gb', 0),
                'cpu_cores': model.get('requirements', {}).get('cpu_cores', 0)
            })
            
            compatibility_check = model.get('requirements_check', {})
            if compatibility_check.get('success', False):
                compatibility_score = sum(
                    1 for check in compatibility_check.get('details', {}).values() 
                    if check.get('met', False)
                ) / len(compatibility_check.get('details', {})) * 100 if compatibility_check.get('details', {}) else 0
                
                metrics['system_compatibility'].append({
                    'model': model_name,
                    'score': compatibility_score,
                    'requirements_met': compatibility_check.get('requirements_met', False),
                    'details': compatibility_check.get('details', {})
                })
            
            model_type = model.get('model_type', '')
            if model_type == 'embedding':
                spec_score = self._calculate_embedding_spec_score(model)
            elif model_type == 'llm':
                spec_score = self._calculate_llm_spec_score(model)
            else:
                spec_score = 50
            
            metrics['model_specifications'].append({
                'model': model_name,
                'score': spec_score,
                'model_type': model_type,
                'provider': model.get('provider', ''),
                'version': model.get('version', ''),
                'download_size_mb': model.get('download_size_mb', 0)
            })
        
        return metrics
    
    def _calculate_performance_score(self, model: Dict[str, Any]) -> float:
        score = 50.0
        
        model_type = model.get('model_type', '')
        
        if model_type == 'embedding':
            dimensions = model.get('embedding_dimensions', 0)
            context_length = model.get('context_length', 0)
            
            dimension_score = min(dimensions / 768 * 50, 50) if dimensions else 25
            context_score = min(context_length / 512 * 25, 25) if context_length else 12.5
            
            score = dimension_score + context_score
        
        elif model_type == 'llm':
            parameters = model.get('parameters_billion', 0)
            context_length = model.get('context_length', 0)
            
            parameter_score = min(parameters / 13 * 40, 40) if parameters else 20
            context_score = min(context_length / 8192 * 35, 35) if context_length else 17.5
            
            score = parameter_score + context_score
        
        return round(score, 2)
    
    def _calculate_resource_score(self, model: Dict[str, Any]) -> float:
        requirements = model.get('requirements', {})
        
        if not requirements:
            return 50.0
        
        ram_gb = requirements.get('ram_gb', 0)
        storage_gb = requirements.get('storage_gb', 0)
        cpu_cores = requirements.get('cpu_cores', 0)
        
        if ram_gb == 0 or storage_gb == 0 or cpu_cores == 0:
            return 50.0
        
        ram_score = max(0, 100 - (ram_gb / 16 * 50))
        storage_score = max(0, 100 - (storage_gb / 10 * 30))
        cpu_score = max(0, 100 - (cpu_cores / 8 * 20))
        
        total_score = (ram_score * 0.5) + (storage_score * 0.3) + (cpu_score * 0.2)
        
        return round(total_score, 2)
    
    def _calculate_embedding_spec_score(self, model: Dict[str, Any]) -> float:
        score = 50.0
        
        dimensions = model.get('embedding_dimensions', 0)
        context_length = model.get('context_length', 0)
        provider = model.get('provider', '').lower()
        
        if dimensions >= 768:
            score += 20
        elif dimensions >= 384:
            score += 10
        
        if context_length >= 512:
            score += 15
        elif context_length >= 256:
            score += 10
        
        if 'sentence-transformers' in provider:
            score += 15
        
        return min(score, 100)
    
    def _calculate_llm_spec_score(self, model: Dict[str, Any]) -> float:
        score = 50.0
        
        parameters = model.get('parameters_billion', 0)
        context_length = model.get('context_length', 0)
        provider = model.get('provider', '').lower()
        
        if parameters >= 13:
            score += 25
        elif parameters >= 7:
            score += 15
        elif parameters >= 3:
            score += 5
        
        if context_length >= 8192:
            score += 20
        elif context_length >= 4096:
            score += 10
        
        if 'ollama' in provider:
            score += 10
        
        return min(score, 100)
    
    def _generate_model_recommendations(self, models: List[Dict[str, Any]], metrics: Dict[str, Any]) -> List[str]:
        recommendations = []
        
        if not models or not metrics:
            return recommendations
        
        performance_scores = metrics.get('performance', [])
        resource_scores = metrics.get('resource_requirements', [])
        compatibility_scores = metrics.get('system_compatibility', [])
        
        if performance_scores and resource_scores:
            performance_ranking = sorted(performance_scores, key=lambda x: x['score'], reverse=True)
            resource_ranking = sorted(resource_scores, key=lambda x: x['score'], reverse=True)
            
            if performance_ranking and resource_ranking:
                best_perf_model = performance_ranking[0]['model']
                best_resource_model = resource_ranking[0]['model']
                
                if best_perf_model == best_resource_model:
                    recommendations.append(f"Model '{best_perf_model}' offers the best balance of performance and resource efficiency")
                else:
                    recommendations.append(f"For maximum performance: '{best_perf_model}'")
                    recommendations.append(f"For resource efficiency: '{best_resource_model}'")
        
        for comp in compatibility_scores:
            if not comp.get('requirements_met', False):
                model_name = comp['model']
                recommendations.append(f"Model '{model_name}' does not meet system requirements - consider upgrading hardware")
        
        model_types = set(model.get('model_type', '') for model in models)
        if 'embedding' in model_types and 'llm' in model_types:
            embedding_models = [m for m in models if m.get('model_type') == 'embedding']
            llm_models = [m for m in models if m.get('model_type') == 'llm']
            
            if embedding_models and llm_models:
                best_embedding = max(embedding_models, key=lambda x: self._calculate_performance_score(x))
                best_llm = max(llm_models, key=lambda x: self._calculate_performance_score(x))
                
                recommendations.append(f"Recommended embedding model for this system: '{best_embedding['name']}'")
                recommendations.append(f"Recommended LLM model for this system: '{best_llm['name']}'")
        
        return recommendations
    
    def _select_best_model(self, models: List[Dict[str, Any]], metrics: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not models or not metrics:
            return None
        
        performance_scores = {item['model']: item['score'] for item in metrics.get('performance', [])}
        resource_scores = {item['model']: item['score'] for item in metrics.get('resource_requirements', [])}
        
        weighted_scores = {}
        
        for model in models:
            model_name = model['name']
            perf_score = performance_scores.get(model_name, 0)
            resource_score = resource_scores.get(model_name, 0)
            
            weighted_score = (perf_score * 0.6) + (resource_score * 0.4)
            weighted_scores[model_name] = weighted_score
        
        if not weighted_scores:
            return None
        
        best_model_name = max(weighted_scores, key=weighted_scores.get)
        
        for model in models:
            if model['name'] == best_model_name:
                return {
                    'name': model['name'],
                    'display_name': model.get('display_name', model['name']),
                    'model_type': model.get('model_type', ''),
                    'weighted_score': weighted_scores[best_model_name],
                    'performance_score': performance_scores.get(best_model_name, 0),
                    'resource_score': resource_scores.get(best_model_name, 0)
                }
        
        return None
    
    def validate_environment(self) -> Dict[str, Any]:
        """Validate the environment for model operations.
        
        Returns:
            Environment validation results
        """
        checks = {
            'ollama_installed': self._check_ollama_available(),
            'system_resources': self.get_system_resources(),
            'requirements_check': self.check_system_requirements(),
            'models_available': len(self.get_available_models()) > 0,
            'health_status': self.health_check()['status']
        }
        
        all_ok = (
            checks['ollama_installed'] and
            'error' not in checks['system_resources'] and
            checks['requirements_check']['requirements_met'] and
            checks['models_available'] and
            checks['health_status'] in ['healthy', 'degraded']
        )
        
        recommendations = []
        if not checks['ollama_installed']:
            recommendations.append("Install Ollama from https://ollama.ai/")
        if 'error' in checks['system_resources']:
            recommendations.append("Check system resources")
        if not checks['requirements_check']['requirements_met']:
            recommendations.append("Upgrade system to meet requirements")
        if not checks['models_available']:
            recommendations.append("Configure models in registry")
        if checks['health_status'] == 'unhealthy':
            recommendations.append("Run health check for details")
        
        recommendations = [rec for rec in recommendations if rec is not None]
        
        return {
            'success': True,
            'environment_ready': all_ok,
            'checks': checks,
            'timestamp': datetime.now().isoformat(),
            'recommendations': recommendations
        }


def get_model_manager(config: Optional[Dict[str, Any]] = None) -> ModelManager:
    return ModelManager(config)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test DocuBot Model Manager")
    parser.add_argument("--test-all", action="store_true", help="Run all tests")
    parser.add_argument("--test-resources", action="store_true", help="Test system resources")
    parser.add_argument("--test-models", action="store_true", help="Test model listing")
    parser.add_argument("--test-download", type=str, help="Test model download")
    parser.add_argument("--test-validation", type=str, help="Test model validation")
    parser.add_argument("--test-comparison", nargs='+', help="Test model comparison with given model names")
    parser.add_argument("--test-environment", action="store_true", help="Test environment validation")
    parser.add_argument("--health-check", action="store_true", help="Run health check")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)-8s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    print("\n" + "=" * 80)
    print("DOCUBOT MODEL MANAGER - TEST")
    print("=" * 80)
    
    try:
        print("\n1. Initializing ModelManager...")
        manager = ModelManager()
        print("   SUCCESS: ModelManager initialized")
        
        if args.test_resources or args.test_all:
            print("\n2. Testing system resources...")
            resources = manager.get_system_resources()
            
            if 'error' in resources:
                print(f"   FAILED: {resources['error']}")
            else:
                print("   SUCCESS: System resources retrieved")
                print(f"   CPU Cores: {resources['cpu']['cores']}")
                print(f"   RAM: {resources['ram']['total_gb']:.1f} GB")
                print(f"   Disk Free (home): {resources['disk'].get(str(Path.home()), {}).get('free_gb', 0):.1f} GB")
                print(f"   GPU Available: {resources['gpu']['available']}")
        
        if args.test_models or args.test_all:
            print("\n3. Testing model listing...")
            
            llm_models = manager.get_available_models('llm')
            print(f"   LLM Models: {len(llm_models)} found")
            for model in llm_models[:3]:
                print(f"   * {model['display_name']} ({model['name']})")
            
            if len(llm_models) > 3:
                print(f"   ... and {len(llm_models) - 3} more")
            
            embedding_models = manager.get_available_models('embedding')
            print(f"   Embedding Models: {len(embedding_models)} found")
            for model in embedding_models:
                status = "YES" if model.get('is_downloaded', False) else "NO"
                default = " [DEFAULT]" if model.get('is_default', False) else ""
                print(f"   {status} {model['display_name']}{default} ({model['embedding_dimensions']}D)")
        
        if args.test_validation or args.test_all:
            print("\n4. Testing model validation...")
            test_model = "all-MiniLM-L6-v2"
            validation = manager.validate_model_download(test_model, 'embedding')
            
            print(f"   Model: {test_model}")
            print(f"   Can download: {validation['valid']}")
            print(f"   Message: {validation['message']}")
            
            if not validation['valid'] and 'issues' in validation:
                print(f"   Issues: {', '.join(validation['issues'])}")
        
        if args.test_download:
            print(f"\n5. Testing model download for: {args.test_download}")
            print("   Note: This would actually download the model")
            print("   Use --test-download with caution")
            
            validation = manager.validate_model_download(args.test_download)
            print(f"   Validation: {'PASS' if validation['valid'] else 'FAIL'}")
        
        if args.test_comparison:
            print(f"\n6. Testing model comparison for: {args.test_comparison}")
            comparison = manager.compare_models(args.test_comparison)
            
            if comparison.get('success', False):
                print(f"   Comparison completed successfully")
                print(f"   Models compared: {len(comparison['models'])}")
                
                if 'best_model' in comparison:
                    best = comparison['best_model']
                    print(f"   Recommended model: {best['display_name']} (Score: {best['weighted_score']:.1f})")
                
                if comparison['recommendations']:
                    print("   Recommendations:")
                    for rec in comparison['recommendations']:
                        print(f"     * {rec}")
            else:
                print(f"   Comparison failed: {comparison.get('message', 'Unknown error')}")
        
        if args.test_environment or args.test_all:
            print("\n7. Testing environment validation...")
            env_validation = manager.validate_environment()
            
            print(f"   Environment ready: {env_validation['environment_ready']}")
            print(f"   Ollama installed: {env_validation['checks']['ollama_installed']}")
            print(f"   Models available: {env_validation['checks']['models_available']}")
            print(f"   Health status: {env_validation['checks']['health_status']}")
            
            if env_validation['recommendations']:
                print("   Recommendations:")
                for rec in env_validation['recommendations']:
                    print(f"     * {rec}")
        
        if args.test_all:
            print("\n8. Testing file validation...")
            test_model = "all-MiniLM-L6-v2"
            file_validation = manager.validate_model_files(test_model, 'embedding')
            
            print(f"   Model: {test_model}")
            print(f"   Files valid: {file_validation['valid']}")
            print(f"   Message: {file_validation['message']}")
        
        if args.health_check or args.test_all:
            print("\n9. Running health check...")
            health = manager.health_check()
            
            if health['success']:
                print(f"   Health status: {health['status'].upper()}")
                print(f"   Health score: {health['health_score']}/100")
                
                if health['recommendations']:
                    print("   Recommendations:")
                    for rec in health['recommendations']:
                        print(f"     * {rec}")
            else:
                print(f"   Health check failed: {health.get('error', 'Unknown error')}")
        
        if args.test_all:
            print("\n10. Testing utility functions...")
            test_sizes = [500, 1500, 1500000, 1500000000]
            for size in test_sizes:
                formatted = manager.format_file_size(size)
                print(f"   {size:,} bytes = {formatted}")
        
        if args.test_all:
            print("\n11. Testing active model management...")
            
            default_embedding = manager.registry.get_default_model('embedding')
            if default_embedding:
                success = manager.set_active_model(default_embedding.name, 'embedding')
                print(f"   Set active embedding model: {'SUCCESS' if success else 'FAILED'}")
                
                active_model = manager.get_active_model('embedding')
                if active_model:
                    print(f"   Active embedding model: {active_model.display_name}")
        
        print("\n" + "=" * 80)
        print("ALL TESTS COMPLETED")
        print("=" * 80)
        
        sys.exit(0)
        
    except Exception as e:
        print(f"\nERROR: Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)