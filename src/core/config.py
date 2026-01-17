# docubot/src/core/config.py

"""
DocuBot Configuration Management
"""

import os
import platform  # <- TAMBAH INI
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from dataclasses import dataclass, field
from .constants import DEFAULT_CONFIG


@dataclass
class AppConfig:
    data_dir: Path = field(default_factory=lambda: Path.home() / ".docubot")
    models_dir: Path = field(init=False)
    documents_dir: Path = field(init=False)
    database_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    
    chunk_size: int = 500
    chunk_overlap: int = 50
    max_file_size_mb: int = 100
    supported_formats: list = field(default_factory=lambda: [".pdf", ".docx", ".txt", ".epub", ".md", ".html"])
    
    llm_model: str = "llama2:7b"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 1024
    
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimensions: int = 384
    
    rag_top_k: int = 5
    rag_similarity_threshold: float = 0.7
    enable_hybrid_search: bool = True
    
    ui_theme: str = "dark"
    ui_language: str = "en"
    ui_font_size: int = 12
    
    def __post_init__(self):
        # Use platform-specific data directory
        self.data_dir = AppConfig.get_data_dir()  # <- PERUBAHAN: gunakan get_data_dir()
        self.data_dir.mkdir(exist_ok=True)
        self.models_dir = self.data_dir / "models"
        self.documents_dir = self.data_dir / "documents"
        self.database_dir = self.data_dir / "database"
        self.logs_dir = self.data_dir / "logs"
        
        for directory in [self.models_dir, self.documents_dir, self.database_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def get_data_dir() -> Path:
        """
        Get platform-specific data directory
        
        Returns:
            Path to data directory
        """
        system = platform.system()
        home = Path.home()
        
        if system == "Windows":
            return home / "AppData" / "Local" / "DocuBot"
        elif system == "Darwin":  # macOS
            return home / "Library" / "Application Support" / "DocuBot"
        else:  # Linux/Unix
            return home / ".docubot"
    
    @staticmethod
    def ensure_directories() -> None:
        """Create all required application directories"""
        data_dir = AppConfig.get_data_dir()  # <- PERUBAHAN: AppConfig bukan Config
        
        directories = [
            data_dir,
            data_dir / "models",
            data_dir / "database",
            data_dir / "documents",
            data_dir / "uploads",
            data_dir / "processed",
            data_dir / "exports",
            data_dir / "logs",
            data_dir / "cache",
        ]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                print(f"Created directory: {directory}")
            except Exception as e:
                print(f"Error creating directory {directory}: {e}")

class ConfigManager:
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path.home() / ".docubot" / "config" / "app_config.yaml"
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        self.config = AppConfig()
    
    def load(self) -> AppConfig:
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    yaml_config = yaml.safe_load(f) or {}
                
                for key, value in yaml_config.items():
                    if hasattr(self.config, key):
                        setattr(self.config, key, value)
            
            return self.config
            
        except Exception as e:
            print(f"Warning: Could not load configuration: {e}")
            return self.config
    
    def save(self) -> bool:
        try:
            config_dict = {
                'chunk_size': self.config.chunk_size,
                'chunk_overlap': self.config.chunk_overlap,
                'llm_model': self.config.llm_model,
                'llm_temperature': self.config.llm_temperature,
                'embedding_model': self.config.embedding_model,
                'ui_theme': self.config.ui_theme,
                'ui_language': self.config.ui_language,
            }
            
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False)
            
            return True
            
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
    
    def validate(self) -> Dict[str, Any]:
        issues = []
        
        required_dirs = [
            self.config.data_dir,
            self.config.models_dir,
            self.config.documents_dir,
            self.config.database_dir,
            self.config.logs_dir
        ]
        
        for directory in required_dirs:
            if not directory.exists():
                issues.append(f"Missing directory: {directory}")
        
        if self.config.chunk_size < 100 or self.config.chunk_size > 2000:
            issues.append(f"Invalid chunk_size: {self.config.chunk_size}")
        
        if self.config.chunk_overlap < 0 or self.config.chunk_overlap >= self.config.chunk_size:
            issues.append(f"Invalid chunk_overlap: {self.config.chunk_overlap}")
        
        if self.config.llm_temperature < 0 or self.config.llm_temperature > 2:
            issues.append(f"Invalid llm_temperature: {self.config.llm_temperature}")
        
        return {
            'valid': len(issues) == 0,
            'issues': issues,
            'config_path': str(self.config_path)
        }


default_config_manager = ConfigManager()


def get_config() -> AppConfig:
    return default_config_manager.load()
