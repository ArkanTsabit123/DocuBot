"""
Test configuration module
"""

import pytest
from src.core.config import AppConfig, ConfigManager
from pathlib import Path


def test_app_config_creation():
    config = AppConfig()
    
    assert config.chunk_size == 500
    assert config.chunk_overlap == 50
    assert config.llm_model == "llama2:7b"
    assert config.embedding_model == "all-MiniLM-L6-v2"
    
    assert config.data_dir.exists()
    assert config.models_dir.exists()
    assert config.documents_dir.exists()
    assert config.database_dir.exists()
    assert config.logs_dir.exists()


def test_config_manager():
    manager = ConfigManager()
    config = manager.load()
    
    assert isinstance(config, AppConfig)
    
    validation = manager.validate()
    assert validation['valid'] == True
    assert len(validation['issues']) == 0


def test_config_save_load(tmp_path):
    config_path = tmp_path / "test_config.yaml"
    manager = ConfigManager(config_path)
    
    config = manager.load()
    config.chunk_size = 600
    config.llm_temperature = 0.5
    
    assert manager.save()
    
    manager2 = ConfigManager(config_path)
    config2 = manager2.load()
    
    assert config2.chunk_size == 600
    assert config2.llm_temperature == 0.5
