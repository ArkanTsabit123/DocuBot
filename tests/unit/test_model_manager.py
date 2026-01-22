
# docubot/src/ai_engine/model_manager.py 

"""
Model Manager for DocuBot - Handles model downloading, validation, and management.
"""

import os
import json
import time
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Generator, Union
from datetime import datetime
import psutil
import torch

logger = logging.getLogger(__name__)

# LLMClient import dynamically
def _import_llm_client():
    """Dynamically import LLMClient with multiple fallbacks."""
    try:
        # absolute import from src
        from src.ai_engine.llm_client import LLMClient
        return LLMClient, True
    except ImportError:
        try:
            # relative import
            from .llm_client import LLMClient
            return LLMClient, True
        except ImportError:
            try:
                # direct import
                import sys
                current_dir = os.path.dirname(os.path.abspath(__file__))
                sys.path.insert(0, current_dir)
                from llm_client import LLMClient
                return LLMClient, True
            except ImportError as e:
                logger.warning(f"LLMClient not available: {e}")
                return None, False

def _import_embedding_service():
    """Dynamically import embedding service."""
    try:
        from .embedding_service import get_embedding_service
        return get_embedding_service, True
    except ImportError:
        try:
            from src.ai_engine.embedding_service import get_embedding_service
            return get_embedding_service, True
        except ImportError as e:
            logger.warning(f"EmbeddingService not available: {e}")
            return None, False


class ModelManager:
    """
    Manages AI models for DocuBot including:
    - LLM models (via Ollama)
    - Embedding models (Sentence Transformers)
    - Model validation and downloading
    - System resource checking
    """
    
    # Embedding models database
    EMBEDDING_MODELS = {
        "all-MiniLM-L6-v2": {
            "display_name": "MiniLM L6 v2",
            "description": "Fast and efficient embedding model (384 dimensions)",
            "dimensions": 384,
            "context_length": 256,
            "size_mb": 90,
            "speed": "Fast",
            "accuracy": "Good",
            "languages": ["en"],
            "default": True
        },
        "all-mpnet-base-v2": {
            "display_name": "MPNet Base v2",
            "description": "High-quality embedding model (768 dimensions)",
            "dimensions": 768,
            "context_length": 384,
            "size_mb": 420,
            "speed": "Medium",
            "accuracy": "Excellent",
            "languages": ["en"],
            "default": False
        },
        "paraphrase-multilingual-MiniLM-L12-v2": {
            "display_name": "Multilingual MiniLM L12 v2",
            "description": "Multilingual embedding model (384 dimensions)",
            "dimensions": 384,
            "context_length": 128,
            "size_mb": 480,
            "speed": "Medium",
            "accuracy": "Good",
            "languages": ["en", "id", "es", "fr", "de", "zh"],
            "default": False
        }
    }
    
    # LLM models database 
    LLM_MODELS = {
        "llama2:7b": {
            "display_name": "Llama 2 7B",
            "description": "Meta's Llama 2 7B parameter model",
            "ram_required_gb": 8,
            "storage_required_gb": 4.2,
            "context_window": 4096,
            "default": True
        },
        "mistral:7b": {
            "display_name": "Mistral 7B",
            "description": "Mistral AI's 7B parameter model",
            "ram_required_gb": 8,
            "storage_required_gb": 4.1,
            "context_window": 8192,
            "default": False
        },
        "neural-chat:7b": {
            "display_name": "Neural Chat 7B",
            "description": "Intel's fine-tuned neural chat model",
            "ram_required_gb": 8,
            "storage_required_gb": 4.3,
            "context_window": 4096,
            "default": False
        }
    }
    
    def __init__(self, config=None):
        """
        Initialize ModelManager.
        
        Args:
            config: AppConfig instance
        """
        self.config = config
        self._llm_client = None
        self._embedding_service = None
        self._llm_client_class = None
        self._get_embedding_service_func = None
        
        # Try to import classes
        self._llm_client_class, _ = _import_llm_client()
        self._get_embedding_service_func, _ = _import_embedding_service()
        
        logger.info("ModelManager initialized")
    
    @property
    def llm_client(self):
        """Lazy load LLM client."""
        if self._llm_client is None and self._llm_client_class:
            try:
                self._llm_client = self._llm_client_class()
                logger.info("LLMClient initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize LLMClient: {e}")
                self._llm_client = None
        return self._llm_client
    
    @property
    def embedding_service(self):
        """Lazy load embedding service."""
        if self._embedding_service is None and self._get_embedding_service_func:
            try:
                self._embedding_service = self._get_embedding_service_func(self.config)
                logger.info("EmbeddingService initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize EmbeddingService: {e}")
                self._embedding_service = None
        return self._embedding_service
    
    def get_available_llm_models(self) -> List[Dict[str, Any]]:
        """Get available LLM models from Ollama."""
        if not self.llm_client:
            logger.warning("LLM client not available, returning static list")
            return self._get_static_llm_models()
        
        try:
            return self.llm_client.get_available_models(refresh_cache=True)
        except Exception as e:
            logger.error(f"Error getting LLM models: {e}")
            return self._get_static_llm_models()
    
    def _get_static_llm_models(self) -> List[Dict[str, Any]]:
        """Get static LLM model information."""
        models = []
        for model_id, model_info in self.LLM_MODELS.items():
            models.append({
                'name': model_id,
                'display_name': model_info['display_name'],
                'description': model_info['description'],
                'ram_required_gb': model_info['ram_required_gb'],
                'storage_required_gb': model_info['storage_required_gb'],
                'context_window': model_info['context_window'],
                'is_default': model_info['default'],
                'is_supported': True,
                'is_current': False  # Can't determine without LLM client
            })
        return models
    
    def get_available_embedding_models(self) -> List[Dict[str, Any]]:
        """Get available embedding models."""
        models = []
        
        for model_id, model_info in self.EMBEDDING_MODELS.items():
            # Check if model is downloaded
            is_downloaded = self._is_embedding_model_downloaded(model_id)
            
            models.append({
                'name': model_id,
                'display_name': model_info['display_name'],
                'description': model_info['description'],
                'dimensions': model_info['dimensions'],
                'context_length': model_info['context_length'],
                'size_mb': model_info['size_mb'],
                'speed': model_info['speed'],
                'accuracy': model_info['accuracy'],
                'languages': model_info['languages'],
                'is_default': model_info['default'],
                'downloaded': is_downloaded,
                'can_download': True
            })
        
        return models
    
    def _is_embedding_model_downloaded(self, model_name: str) -> bool:
        """Check if embedding model is downloaded locally."""
        if not self.config or not hasattr(self.config, 'paths'):
            return False
        
        try:
            # Get models directory from config or use default
            if hasattr(self.config.paths, 'models_dir'):
                models_dir = Path(self.config.paths.models_dir)
            else:
                models_dir = Path.home() / ".docubot" / "models"
            
            embedding_dir = models_dir / "sentence-transformers" / model_name
            
            # Check if directory exists and has necessary files
            if not embedding_dir.exists():
                return False
            
            # Check for essential files
            required_files = ['config.json']
            existing_files = [f.name for f in embedding_dir.iterdir() if f.is_file()]
            
            # Need at least config.json and one model file
            if 'config.json' not in existing_files:
                return False
            
            # Check for model files
            model_files = [f for f in existing_files 
                          if any(keyword in f.lower() 
                                for keyword in ['model', 'pytorch', 'tensorflow', 'onnx'])]
            
            return len(model_files) > 0
            
        except Exception as e:
            logger.debug(f"Error checking model {model_name}: {e}")
            return False
    
    def validate_model_download(self, model_name: str, model_type: str = 'llm') -> Dict[str, Any]:
        """
        Validate if a model can be downloaded.
        
        Args:
            model_name: Name of the model
            model_type: 'llm' or 'embedding'
        
        Returns:
            Validation results
        """
        result = {
            'success': False,
            'model': model_name,
            'type': model_type,
            'message': '',
            'can_download': False,
            'requirements_met': False,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            if model_type == 'llm':
                # Check if model is supported
                if model_name not in self.LLM_MODELS:
                    result['message'] = f'LLM model {model_name} is not supported'
                    return result
                
                model_info = self.LLM_MODELS[model_name]
                
                # Check RAM requirements
                ram_ok = self.check_ram_requirements(model_info['ram_required_gb'])
                
                # Check disk space
                disk_ok = self.check_disk_space(model_info['storage_required_gb'])
                
                # Check Ollama installation
                ollama_ok = self.check_ollama_installed()
                
                result['requirements'] = {
                    'ram_gb': model_info['ram_required_gb'],
                    'storage_gb': model_info['storage_required_gb'],
                    'ram_ok': ram_ok,
                    'disk_ok': disk_ok,
                    'ollama_ok': ollama_ok
                }
                
                result['requirements_met'] = ram_ok and disk_ok and ollama_ok
                result['can_download'] = result['requirements_met']
                result['success'] = True
                result['message'] = (
                    f"Model {model_name} can be downloaded" 
                    if result['requirements_met'] 
                    else "System requirements not met"
                )
                
            elif model_type == 'embedding':
                # Check if model is supported
                if model_name not in self.EMBEDDING_MODELS:
                    result['message'] = f'Embedding model {model_name} is not supported'
                    return result
                
                model_info = self.EMBEDDING_MODELS[model_name]
                required_space_gb = model_info['size_mb'] / 1024
                
                # Check disk space
                disk_ok = self.check_disk_space(required_space_gb)
                
                # Check RAM for embeddings (less strict)
                ram_ok = self.check_ram_requirements(4)  # 4GB minimum for embeddings
                
                result['requirements'] = {
                    'ram_gb': 4,
                    'storage_gb': required_space_gb,
                    'ram_ok': ram_ok,
                    'disk_ok': disk_ok
                }
                
                result['requirements_met'] = disk_ok and ram_ok
                result['can_download'] = result['requirements_met']
                result['success'] = True
                result['message'] = (
                    f"Embedding model {model_name} can be downloaded"
                    if result['requirements_met']
                    else "Insufficient disk space or RAM"
                )
                result['model_info'] = {
                    'dimensions': model_info['dimensions'],
                    'languages': model_info['languages'],
                    'speed': model_info['speed']
                }
                
            else:
                result['message'] = f'Unknown model type: {model_type}'
        
        except Exception as e:
            result['message'] = f'Validation error: {str(e)}'
            logger.error(f"Model validation error for {model_name}: {e}")
        
        return result
    
    def check_disk_space(self, min_gb: float = 1.0) -> bool:
        """
        Check if sufficient disk space is available.
        
        Args:
            min_gb: Minimum GB required
        
        Returns:
            True if sufficient space available
        """
        try:
            # Determine path to check
            if self.config and hasattr(self.config, 'paths') and hasattr(self.config.paths, 'models_dir'):
                check_path = Path(self.config.paths.models_dir)
            else:
                check_path = Path.home() / ".docubot"
            
            # Create directory if it doesn't exist
            check_path.mkdir(parents=True, exist_ok=True)
            
            disk_usage = psutil.disk_usage(str(check_path))
            free_gb = disk_usage.free / (1024 ** 3)
            
            logger.debug(f"Disk check: {free_gb:.2f}GB free, need {min_gb}GB")
            return free_gb >= min_gb
        
        except Exception as e:
            logger.error(f"Error checking disk space: {e}")
            return False

    def check_ram_requirements(self, min_gb: float = 7.0) -> bool:  # Ubah dari 8.0 ke 7.0
        """
        Check if system meets RAM requirements.
        
        Args:
            min_gb: Minimum GB required (default 7GB untuk lebih realistis)
        
        Returns:
            True if sufficient RAM available
        """
        try:
            memory = psutil.virtual_memory()
            total_gb = memory.total / (1024 ** 3)        
            effective_ram = total_gb * 0.85 
            
            logger.debug(f"RAM check: {total_gb:.2f}GB total, {effective_ram:.2f}GB effective, need {min_gb}GB")
            return effective_ram >= min_gb
        
        except Exception as e:
            logger.error(f"Error checking RAM: {e}")
            return False
    
    def validate_model_files(self, model_name: str) -> Dict[str, Any]:
        """
        Validate existing model files.
        
        Args:
            model_name: Name of the model to validate
        
        Returns:
            Validation results
        """
        result = {
            'valid': False,
            'model': model_name,
            'message': '',
            'files_found': [],
            'files_missing': [],
            'total_size_bytes': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        # Check if it's an embedding model
        if model_name in self.EMBEDDING_MODELS:
            return self._validate_embedding_model_files(model_name)
        
        # For LLM models, check Ollama
        try:
            if self.llm_client:
                model_info = self.llm_client.get_model_info(model_name)
                if model_info.get('success', False):
                    result['valid'] = True
                    result['message'] = f"Model {model_name} is available via Ollama"
                    result['files_found'] = ['ollama_model']
                    result['total_size_bytes'] = model_info.get('size_bytes', 0)
                else:
                    result['message'] = f"Model {model_name} not found in Ollama"
            else:
                result['message'] = "LLM client not available for validation"
        
        except Exception as e:
            result['message'] = f"Validation error: {str(e)}"
        
        return result
    
    def _validate_embedding_model_files(self, model_name: str) -> Dict[str, Any]:
        """Validate embedding model files."""
        result = {
            'valid': False,
            'model': model_name,
            'message': '',
            'files_found': [],
            'files_missing': [],
            'total_size_bytes': 0,
            'timestamp': datetime.now().isoformat()
        }
        
        if not self.config or not hasattr(self.config, 'paths'):
            result['message'] = 'Configuration not available'
            return result
        
        try:
            # Get models directory
            if hasattr(self.config.paths, 'models_dir'):
                models_dir = Path(self.config.paths.models_dir)
            else:
                models_dir = Path.home() / ".docubot" / "models"
            
            embedding_dir = models_dir / "sentence-transformers" / model_name
            
            if not embedding_dir.exists():
                result['message'] = f'Model directory not found: {embedding_dir}'
                return result
            
            # List all files
            files = [f for f in embedding_dir.iterdir() if f.is_file()]
            result['files_found'] = [f.name for f in files]
            result['total_size_bytes'] = sum(f.stat().st_size for f in files)
            
            # Check for essential files
            essential_files = ['config.json']
            missing_files = []
            
            for essential_file in essential_files:
                if essential_file not in result['files_found']:
                    missing_files.append(essential_file)
            
            if missing_files:
                result['files_missing'] = missing_files
                result['message'] = f'Missing essential files: {", ".join(missing_files)}'
            else:
                # Check for at least one model file
                model_files = [f for f in result['files_found'] 
                             if any(keyword in f.lower() 
                                   for keyword in ['model', 'pytorch', 'tensorflow', 'onnx'])]
                
                if model_files:
                    result['valid'] = True
                    result['message'] = f'Embedding model {model_name} files validated successfully'
                else:
                    result['message'] = 'No model files found (expecting .bin, .pt, etc.)'
        
        except Exception as e:
            result['message'] = f'Validation error: {str(e)}'
        
        return result
    
    def get_system_resources(self) -> Dict[str, Any]:
        """
        Get current system resource information.
        
        Returns:
            Dictionary with resource info
        """
        try:
            # CPU
            cpu_cores = os.cpu_count() or 1
            
            # RAM
            memory = psutil.virtual_memory()
            ram_total_gb = memory.total / (1024 ** 3)
            ram_available_gb = memory.available / (1024 ** 3)
            ram_used_percent = memory.percent
            
            # Disk
            if self.config and hasattr(self.config, 'paths') and hasattr(self.config.paths, 'models_dir'):
                check_path = Path(self.config.paths.models_dir)
            else:
                check_path = Path.home() / ".docubot"
            
            check_path.mkdir(parents=True, exist_ok=True)
            disk_usage = psutil.disk_usage(str(check_path))
            disk_total_gb = disk_usage.total / (1024 ** 3)
            disk_free_gb = disk_usage.free / (1024 ** 3)
            disk_used_percent = disk_usage.percent
            
            # GPU
            gpu_available = torch.cuda.is_available()
            gpu_count = torch.cuda.device_count() if gpu_available else 0
            gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else None
            
            # System info
            import platform
            system_info = {
                'system': platform.system(),
                'release': platform.release(),
                'version': platform.version(),
                'machine': platform.machine(),
                'processor': platform.processor(),
                'python_version': platform.python_version()
            }
            
            return {
                'cpu': {
                    'cores': cpu_cores,
                    'usage_percent': psutil.cpu_percent(interval=0.1)
                },
                'ram': {
                    'total_gb': round(ram_total_gb, 2),
                    'available_gb': round(ram_available_gb, 2),
                    'used_percent': ram_used_percent,
                    'minimum_required_gb': 8
                },
                'disk': {
                    'total_gb': round(disk_total_gb, 2),
                    'free_gb': round(disk_free_gb, 2),
                    'used_percent': disk_used_percent,
                    'path': str(check_path)
                },
                'gpu': {
                    'available': gpu_available,
                    'count': gpu_count,
                    'name': gpu_name
                },
                'system': system_info,
                'ollama_installed': self.check_ollama_installed(),
                'timestamp': datetime.now().isoformat()
            }
        
        except Exception as e:
            logger.error(f"Error getting system resources: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def check_ollama_installed(self) -> bool:
        """
        Check if Ollama is installed and accessible.
        
        Returns:
            True if Ollama is installed
        """
        try:
            # Try to run ollama command
            result = subprocess.run(
                ['ollama', '--version'],
                capture_output=True,
                text=True,
                timeout=5,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            installed = result.returncode == 0
            
            if installed:
                logger.debug(f"Ollama found: {result.stdout.strip()}")
            else:
                logger.debug(f"Ollama not found: {result.stderr}")
            
            return installed
            
        except FileNotFoundError:
            logger.debug("Ollama command not found")
            return False
        except subprocess.TimeoutExpired:
            logger.warning("Ollama command timed out")
            return False
        except Exception as e:
            logger.error(f"Error checking Ollama: {e}")
            return False
    
    def format_file_size(self, size_bytes: int) -> str:
        """
        Format file size in human readable format.
        
        Args:
            size_bytes: Size in bytes
        
        Returns:
            Formatted size string
        """
        if size_bytes == 0:
            return "0 B"
        
        size_names = ['B', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']
        i = 0
        
        while size_bytes >= 1024 and i < len(size_names) - 1:
            size_bytes /= 1024.0
            i += 1
        
        if i == 0:
            return f"{size_bytes} {size_names[i]}"
        else:
            return f"{size_bytes:.2f} {size_names[i]}"
    
    def get_model_info(self, model_name: str, model_type: str = 'llm') -> Dict[str, Any]:
        """
        Get detailed information about a specific model.
        
        Args:
            model_name: Name of the model
            model_type: 'llm' or 'embedding'
        
        Returns:
            Model information
        """
        if model_type == 'llm':
            if model_name in self.LLM_MODELS:
                model_info = self.LLM_MODELS[model_name]
                return {
                    'success': True,
                    'model': model_name,
                    'type': 'llm',
                    'display_name': model_info['display_name'],
                    'description': model_info['description'],
                    'ram_required_gb': model_info['ram_required_gb'],
                    'storage_required_gb': model_info['storage_required_gb'],
                    'context_window': model_info['context_window'],
                    'is_default': model_info['default'],
                    'ollama_required': True
                }
            else:
                return {
                    'success': False,
                    'model': model_name,
                    'error': 'LLM model not found in database'
                }
        
        elif model_type == 'embedding':
            if model_name in self.EMBEDDING_MODELS:
                model_info = self.EMBEDDING_MODELS[model_name]
                downloaded = self._is_embedding_model_downloaded(model_name)
                
                return {
                    'success': True,
                    'model': model_name,
                    'type': 'embedding',
                    'display_name': model_info['display_name'],
                    'description': model_info['description'],
                    'dimensions': model_info['dimensions'],
                    'context_length': model_info['context_length'],
                    'size_mb': model_info['size_mb'],
                    'speed': model_info['speed'],
                    'accuracy': model_info['accuracy'],
                    'languages': model_info['languages'],
                    'is_default': model_info['default'],
                    'downloaded': downloaded,
                    'download_size_mb': model_info['size_mb'],
                    'requirements': {
                        'ram_gb': 4,
                        'storage_gb': model_info['size_mb'] / 1024
                    }
                }
            else:
                return {
                    'success': False,
                    'model': model_name,
                    'error': 'Embedding model not found in database'
                }
        
        else:
            return {
                'success': False,
                'model': model_name,
                'error': f'Unknown model type: {model_type}'
            }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform health check of model management system.
        
        Returns:
            Health check results
        """
        start_time = time.time()
        
        try:
            # Get system resources
            resources = self.get_system_resources()
            
            # Check Ollama
            ollama_installed = self.check_ollama_installed()
            
            # Check disk space for model downloads
            disk_ok = self.check_disk_space(min_gb=10)  # 10GB minimum
            
            # Check RAM
            ram_ok = self.check_ram_requirements(min_gb=8)  # 8GB minimum
            
            # Calculate health score (0-100)
            health_score = 0
            if disk_ok:
                health_score += 30
            if ram_ok:
                health_score += 30
            if ollama_installed:
                health_score += 20
            if 'error' not in resources:
                health_score += 20
            
            # Get embedding models status
            embedding_models = self.get_available_embedding_models()
            downloaded_embeddings = sum(1 for m in embedding_models if m['downloaded'])
            
            # Get LLM models status
            llm_models = self.get_available_llm_models()
            available_llms = len(llm_models)
            
            result = {
                'success': True,
                'status': 'healthy' if health_score >= 80 else 'degraded' if health_score >= 50 else 'unhealthy',
                'health_score': health_score,
                'system_resources': resources,
                'requirements': {
                    'disk_space_ok': disk_ok,
                    'ram_ok': ram_ok,
                    'ollama_installed': ollama_installed
                },
                'models': {
                    'embedding_models_total': len(embedding_models),
                    'embedding_models_downloaded': downloaded_embeddings,
                    'llm_models_available': available_llms,
                    'has_default_embedding': any(m['is_default'] and m['downloaded'] for m in embedding_models)
                },
                'recommendations': [],
                'check_duration_ms': round((time.time() - start_time) * 1000, 2),
                'timestamp': datetime.now().isoformat()
            }
            
            # Generate recommendations
            if not disk_ok:
                result['recommendations'].append("Free up disk space (minimum 10GB required)")
            
            if not ram_ok:
                result['recommendations'].append("Consider upgrading RAM (minimum 8GB required)")
            
            if not ollama_installed:
                result['recommendations'].append("Install Ollama from https://ollama.ai/")
            
            if downloaded_embeddings == 0:
                result['recommendations'].append("Download at least one embedding model")
            
            return result
        
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'success': False,
                'status': 'unhealthy',
                'health_score': 0,
                'error': str(e),
                'check_duration_ms': round((time.time() - start_time) * 1000, 2),
                'timestamp': datetime.now().isoformat()
            }


# Factory function
def get_model_manager(config=None):
    """
    Get or create a ModelManager instance.
    
    Args:
        config: AppConfig instance
    
    Returns:
        ModelManager instance
    """
    return ModelManager(config)


if __name__ == "__main__":
    # Test the ModelManager
    import sys
    
    print("=" * 70)
    print("DOCUBOT MODEL MANAGER - TEST")
    print("=" * 70)
    
    # Create a mock config for testing
    class MockPaths:
        models_dir = os.path.join(os.path.expanduser("~"), ".docubot", "models")
    
    class MockConfig:
        paths = MockPaths()
    
    config = MockConfig()
    
    # Create manager
    manager = ModelManager(config)
    
    print("\n1. SYSTEM RESOURCES:")
    print("-" * 40)
    resources = manager.get_system_resources()
    
    if 'error' in resources:
        print(f"  Error: {resources['error']}")
    else:
        print(f"  CPU Cores: {resources['cpu']['cores']}")
        print(f"  RAM: {resources['ram']['total_gb']:.1f} GB total")
        print(f"  Disk: {resources['disk']['free_gb']:.1f} GB free")
        print(f"  GPU Available: {resources['gpu']['available']}")
        if resources['gpu']['available']:
            print(f"  GPU Name: {resources['gpu']['name']}")
    
    print("\n2. REQUIREMENTS CHECK:")
    print("-" * 40)
    print(f"  Disk Space (10GB): {'✓' if manager.check_disk_space(10) else '✗'}")
    print(f"  RAM (8GB): {'✓' if manager.check_ram_requirements(8) else '✗'}")
    print(f"  Ollama Installed: {'✓' if manager.check_ollama_installed() else '✗'}")
    
    print("\n3. EMBEDDING MODELS:")
    print("-" * 40)
    embedding_models = manager.get_available_embedding_models()
    for model in embedding_models:
        status = "✓" if model['downloaded'] else "○"
        default = " [DEFAULT]" if model['is_default'] else ""
        print(f"  {status} {model['name']}{default} - {model['dimensions']}D ({model['size_mb']}MB)")
    
    print("\n4. LLM MODELS:")
    print("-" * 40)
    llm_models = manager.get_available_llm_models()
    for model in llm_models[:3]:  # Show first 3
        default = " [DEFAULT]" if model.get('is_default', False) else ""
        print(f"  ○ {model['name']}{default} - {model.get('display_name', 'N/A')}")
    
    if len(llm_models) > 3:
        print(f"  ... and {len(llm_models) - 3} more")
    
    print("\n5. MODEL VALIDATION:")
    print("-" * 40)
    
    # Test embedding model validation
    test_model = "all-MiniLM-L6-v2"
    validation = manager.validate_model_download(test_model, 'embedding')
    print(f"  {test_model}:")
    print(f"    Can download: {validation['can_download']}")
    print(f"    Message: {validation['message']}")
    
    # Test LLM model validation
    if manager.check_ollama_installed():
        test_llm = "llama2:7b"
        validation = manager.validate_model_download(test_llm, 'llm')
        print(f"  {test_llm}:")
        print(f"    Can download: {validation['can_download']}")
        print(f"    Requirements met: {validation['requirements_met']}")
    
    print("\n6. HEALTH CHECK:")
    print("-" * 40)
    health = manager.health_check()
    
    if health['success']:
        print(f"  Status: {health['status'].upper()}")
        print(f"  Health Score: {health['health_score']}/100")
        print(f"  Models: {health['models']['embedding_models_downloaded']} embeddings, {health['models']['llm_models_available']} LLMs")
        
        if health['recommendations']:
            print("  Recommendations:")
            for rec in health['recommendations']:
                print(f"    • {rec}")
    else:
        print(f"  Health check failed: {health.get('error', 'Unknown error')}")
    
    print("\n7. UTILITY FUNCTIONS:")
    print("-" * 40)
    test_sizes = [500, 1500, 1500000, 1500000000]
    for size in test_sizes:
        formatted = manager.format_file_size(size)
        print(f"  {size:,} bytes = {formatted}")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    
    # Exit with success
    sys.exit(0)