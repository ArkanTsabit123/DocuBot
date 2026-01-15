"""
AI Model Management System
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import json
from datetime import datetime
import subprocess
import shutil
from .llm_client import LLMClient


class ModelManager:
    """
    Manages downloading, updating, and organizing AI models
    """
    
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.llm_client = LLMClient()
        self.model_registry = self.load_model_registry()
    
    def load_model_registry(self) -> Dict[str, Any]:
        """
        Load model registry from file
        
        Returns:
            Model registry dictionary
        """
        registry_file = self.models_dir / "model_registry.json"
        
        if registry_file.exists():
            try:
                with open(registry_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            'llm_models': {},
            'embedding_models': {},
            'last_updated': datetime.now().isoformat()
        }
    
    def save_model_registry(self):
        """
        Save model registry to file
        """
        registry_file = self.models_dir / "model_registry.json"
        self.model_registry['last_updated'] = datetime.now().isoformat()
        
        with open(registry_file, 'w', encoding='utf-8') as f:
            json.dump(self.model_registry, f, indent=2)
    
    def download_llm_model(self, model_name: str) -> Dict[str, Any]:
        """
        Download LLM model using Ollama
        
        Args:
            model_name: Name of model to download
        
        Returns:
            Download status
        """
        print(f"Downloading LLM model: {model_name}")
        
        result = self.llm_client.pull_model(model_name)
        
        if result['success']:
            # Update registry
            self.model_registry['llm_models'][model_name] = {
                'name': model_name,
                'downloaded_at': datetime.now().isoformat(),
                'status': 'downloaded',
                'size': self.estimate_model_size(model_name)
            }
            self.save_model_registry()
        
        return result
    
    def download_embedding_model(self, model_name: str) -> Dict[str, Any]:
        """
        Download embedding model
        
        Args:
            model_name: Name of embedding model
        
        Returns:
            Download status
        """
        from sentence_transformers import SentenceTransformer
        
        print(f"Downloading embedding model: {model_name}")
        
        try:
            start_time = datetime.now()
            
            # Download model
            model = SentenceTransformer(model_name)
            
            # Save model locally
            model_save_path = self.models_dir / "sentence-transformers" / model_name
            model_save_path.mkdir(parents=True, exist_ok=True)
            model.save(str(model_save_path))
            
            download_time = (datetime.now() - start_time).total_seconds()
            
            # Update registry
            self.model_registry['embedding_models'][model_name] = {
                'name': model_name,
                'downloaded_at': datetime.now().isoformat(),
                'status': 'downloaded',
                'download_time_seconds': download_time,
                'local_path': str(model_save_path)
            }
            self.save_model_registry()
            
            return {
                'success': True,
                'model_name': model_name,
                'download_time': download_time,
                'local_path': str(model_save_path)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'model_name': model_name
            }
    
    def get_available_models(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get list of available models
        
        Returns:
            Dictionary with LLM and embedding models
        """
        # Get Ollama models
        ollama_models = self.llm_client.list_available_models()
        
        # Get downloaded embedding models
        embedding_models = list(self.model_registry.get('embedding_models', {}).values())
        
        return {
            'llm_models': ollama_models,
            'embedding_models': embedding_models
        }
    
    def estimate_model_size(self, model_name: str) -> str:
        """
        Estimate model size
        
        Args:
            model_name: Name of model
        
        Returns:
            Estimated size string
        """
        # Rough estimates for common models
        size_map = {
            'llama2:7b': '3.8 GB',
            'mistral:7b': '4.1 GB',
            'neural-chat:7b': '4.3 GB',
            'all-MiniLM-L6-v2': '90 MB',
            'all-mpnet-base-v2': '420 MB'
        }
        
        return size_map.get(model_name, 'Unknown')
    
    def check_model_status(self, model_name: str, model_type: str = 'llm') -> Dict[str, Any]:
        """
        Check status of a model
        
        Args:
            model_name: Name of model
            model_type: Type of model ('llm' or 'embedding')
        
        Returns:
            Model status information
        """
        if model_type == 'llm':
            # Check if model is available in Ollama
            models = self.llm_client.list_available_models()
            model_available = any(m['name'] == model_name for m in models)
            
            return {
                'name': model_name,
                'type': 'llm',
                'available': model_available,
                'downloaded': model_available,
                'in_registry': model_name in self.model_registry.get('llm_models', {})
            }
        
        elif model_type == 'embedding':
            # Check if embedding model is downloaded
            model_info = self.model_registry.get('embedding_models', {}).get(model_name)
            
            if model_info:
                local_path = Path(model_info.get('local_path', ''))
                exists = local_path.exists() if local_path else False
                
                return {
                    'name': model_name,
                    'type': 'embedding',
                    'available': True,
                    'downloaded': exists,
                    'local_path': str(local_path) if local_path else None,
                    'download_time': model_info.get('download_time_seconds')
                }
            else:
                return {
                    'name': model_name,
                    'type': 'embedding',
                    'available': False,
                    'downloaded': False
                }
        
        return {
            'name': model_name,
            'type': model_type,
            'available': False,
            'downloaded': False,
            'error': 'Unknown model type'
        }
    
    def delete_model(self, model_name: str, model_type: str = 'llm') -> Dict[str, Any]:
        """
        Delete a model
        
        Args:
            model_name: Name of model to delete
            model_type: Type of model ('llm' or 'embedding')
        
        Returns:
            Deletion status
        """
        if model_type == 'llm':
            result = self.llm_client.delete_model(model_name)
            
            if result['success']:
                # Remove from registry
                self.model_registry['llm_models'].pop(model_name, None)
                self.save_model_registry()
            
            return result
        
        elif model_type == 'embedding':
            try:
                model_info = self.model_registry['embedding_models'].get(model_name)
                
                if model_info and 'local_path' in model_info:
                    local_path = Path(model_info['local_path'])
                    if local_path.exists():
                        shutil.rmtree(local_path)
                
                # Remove from registry
                self.model_registry['embedding_models'].pop(model_name, None)
                self.save_model_registry()
                
                return {
                    'success': True,
                    'message': f'Deleted embedding model: {model_name}'
                }
                
            except Exception as e:
                return {
                    'success': False,
                    'error': str(e)
                }
        
        return {
            'success': False,
            'error': f'Unknown model type: {model_type}'
        }
    
    def validate_all_models(self) -> Dict[str, Any]:
        """
        Validate all downloaded models
        
        Returns:
            Validation results
        """
        results = {
            'llm_models': {},
            'embedding_models': {},
            'timestamp': datetime.now().isoformat()
        }
        
        # Validate LLM models
        for model_name in self.model_registry.get('llm_models', {}):
            results['llm_models'][model_name] = self.llm_client.validate_model(model_name)
        
        # Validate embedding models
        for model_name, model_info in self.model_registry.get('embedding_models', {}).items():
            local_path = Path(model_info.get('local_path', ''))
            results['embedding_models'][model_name] = {
                'exists': local_path.exists() if local_path else False,
                'path': str(local_path) if local_path else None
            }
        
        return results
    
    def get_disk_usage(self) -> Dict[str, Any]:
        """
        Get disk usage for models
        
        Returns:
            Disk usage information
        """
        total_size = 0
        model_sizes = {}
        
        # Calculate size of embedding models
        for model_name, model_info in self.model_registry.get('embedding_models', {}).items():
            if 'local_path' in model_info:
                path = Path(model_info['local_path'])
                if path.exists():
                    size = self.get_directory_size(path)
                    model_sizes[model_name] = size
                    total_size += size
        
        return {
            'total_size_bytes': total_size,
            'total_size_human': self.format_bytes(total_size),
            'model_sizes': model_sizes,
            'llm_models_count': len(self.model_registry.get('llm_models', {})),
            'embedding_models_count': len(self.model_registry.get('embedding_models', {}))
        }
    
    def get_directory_size(self, path: Path) -> int:
        """
        Calculate directory size
        
        Args:
            path: Directory path
        
        Returns:
            Size in bytes
        """
        total = 0
        for entry in path.rglob('*'):
            if entry.is_file():
                total += entry.stat().st_size
        return total
    
    def format_bytes(self, bytes_size: int) -> str:
        """
        Format bytes to human readable string
        
        Args:
            bytes_size: Size in bytes
        
        Returns:
            Formatted size string
        """
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f} TB"
