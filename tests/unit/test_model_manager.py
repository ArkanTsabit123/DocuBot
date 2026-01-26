# docubot/tests/unit/test_model_manager.py

"""
Unit tests for Model Manager.
Tests model downloading, validation, and management functionality.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Mock ALL dependencies before importing
mock_modules = {
    'psutil': MagicMock(),
    'torch': MagicMock(),
    'requests': MagicMock(),
    'tqdm': MagicMock(),
    'sentence_transformers': MagicMock(),
    'chromadb': MagicMock(),
    'ollama': MagicMock(),
    'packaging': MagicMock(),
    'packaging.version': MagicMock()
}

# Apply mocks
for mod_name, mock_obj in mock_modules.items():
    sys.modules[mod_name] = mock_obj

# Now import the module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ai_engine.model_manager import (
    ModelManager,
    ModelMetadata,
    ModelRequirements,
    ModelDownloader,
    ModelRegistry,
    SystemResourceMonitor,
    get_model_manager
)


class TestModelMetadata(unittest.TestCase):
    def test_model_metadata_creation(self):
        requirements = ModelRequirements(
            ram_gb=8.0,
            storage_gb=5.0,
            cpu_cores=4
        )

        metadata = ModelMetadata(
            name="test-model",
            display_name="Test Model",
            model_type="embedding",
            provider="sentence-transformers",
            description="Test embedding model",
            embedding_dimensions=384,
            requirements=requirements
        )

        self.assertEqual(metadata.name, "test-model")
        self.assertEqual(metadata.model_type, "embedding")
        self.assertEqual(metadata.provider, "sentence-transformers")
        self.assertEqual(metadata.embedding_dimensions, 384)
        self.assertFalse(metadata.is_downloaded)
        self.assertIsInstance(metadata.requirements, ModelRequirements)

    def test_model_metadata_to_dict(self):
        requirements = ModelRequirements(
            ram_gb=8.0,
            storage_gb=5.0
        )

        metadata = ModelMetadata(
            name="test-model",
            display_name="Test Model",
            model_type="embedding",
            provider="test-provider",
            description="Test model",
            requirements=requirements
        )

        result = metadata.to_dict()

        self.assertIn("name", result)
        self.assertIn("display_name", result)
        self.assertIn("model_type", result)
        self.assertIn("requirements", result)
        self.assertEqual(result["name"], "test-model")
        self.assertEqual(result["model_type"], "embedding")

    def test_model_metadata_update_usage(self):
        metadata = ModelMetadata(
            name="test-model",
            display_name="Test Model",
            model_type="embedding",
            provider="test-provider",
            description="Test model"
        )

        initial_count = metadata.usage_count
        metadata.update_usage()

        self.assertEqual(metadata.usage_count, initial_count + 1)
        self.assertIsNotNone(metadata.last_used)
        self.assertIsNotNone(metadata.updated_at)


class TestModelRequirements(unittest.TestCase):
    def test_model_requirements_creation(self):
        requirements = ModelRequirements(
            ram_gb=16.0,
            storage_gb=10.0,
            cpu_cores=8,
            gpu_vram_gb=12.0,
            python_version="3.11",
            dependencies=["torch", "transformers"]
        )

        self.assertEqual(requirements.ram_gb, 16.0)
        self.assertEqual(requirements.storage_gb, 10.0)
        self.assertEqual(requirements.cpu_cores, 8)
        self.assertEqual(requirements.gpu_vram_gb, 12.0)
        self.assertEqual(requirements.python_version, "3.11")
        self.assertEqual(len(requirements.dependencies), 2)

    def test_model_requirements_to_dict(self):
        requirements = ModelRequirements(
            ram_gb=8.0,
            storage_gb=5.0,
            cpu_cores=4
        )

        result = requirements.to_dict()

        self.assertIn("ram_gb", result)
        self.assertIn("storage_gb", result)
        self.assertIn("cpu_cores", result)
        self.assertEqual(result["ram_gb"], 8.0)
        self.assertEqual(result["storage_gb"], 5.0)


class TestModelManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())

        self.config = {
            'download_dir': str(self.temp_dir / "downloads"),
            'models_dir': str(self.temp_dir / "models")
        }

        (self.temp_dir / "downloads").mkdir(parents=True, exist_ok=True)
        (self.temp_dir / "models").mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        manager = ModelManager(self.config)

        self.assertIsNotNone(manager.registry)
        self.assertIsNotNone(manager.downloader)
        self.assertIsNotNone(manager.resource_monitor)
        self.assertIsInstance(manager.active_models, dict)

    @patch('src.ai_engine.model_manager.subprocess.run')
    def test_check_ollama_available(self, mock_subprocess):
        manager = ModelManager(self.config)

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "ollama version 0.1.0"
        mock_subprocess.return_value = mock_result

        self.assertTrue(manager._check_ollama_available())

        mock_subprocess.side_effect = FileNotFoundError()
        self.assertFalse(manager._check_ollama_available())

    def test_get_available_models(self):
        manager = ModelManager(self.config)

        all_models = manager.get_available_models()
        self.assertIsInstance(all_models, list)

        llm_models = manager.get_available_models('llm')
        self.assertGreater(len(llm_models), 0)

        embedding_models = manager.get_available_models('embedding')
        self.assertGreater(len(embedding_models), 0)

    @patch('src.ai_engine.model_manager.subprocess.run')
    def test_check_model_downloaded_llm(self, mock_subprocess):
        manager = ModelManager(self.config)
        manager.ollama_available = True

        model = ModelMetadata(
            name="llama2:7b",
            display_name="Llama 2 7B",
            model_type="llm",
            provider="ollama",
            description="Test LLM"
        )

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "NAME\nllama2:7b\nmistral:7b"
        mock_subprocess.return_value = mock_result

        self.assertTrue(manager._check_model_downloaded(model))

        mock_result.stdout = "NAME\nmistral:7b"
        self.assertFalse(manager._check_model_downloaded(model))

    @patch('src.ai_engine.model_manager.SentenceTransformer')
    def test_check_model_downloaded_embedding(self, mock_sentence_transformer):
        manager = ModelManager(self.config)

        model = ModelMetadata(
            name="all-MiniLM-L6-v2",
            display_name="MiniLM L6 v2",
            model_type="embedding",
            provider="sentence-transformers",
            description="Test embedding"
        )

        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model

        self.assertTrue(manager._check_model_downloaded(model))

        mock_sentence_transformer.side_effect = Exception("Model not found")
        self.assertFalse(manager._check_model_downloaded(model))

    @patch('src.ai_engine.model_manager.psutil.disk_usage')
    def test_check_disk_space(self, mock_disk_usage):
        manager = ModelManager(self.config)

        mock_usage = Mock()
        mock_usage.free = 20 * (1024 ** 3)
        mock_disk_usage.return_value = mock_usage

        self.assertTrue(manager._check_disk_space(10.0))
        self.assertFalse(manager._check_disk_space(30.0))

    def test_get_model_info(self):
        manager = ModelManager(self.config)

        result = manager.get_model_info("all-MiniLM-L6-v2", "embedding")

        self.assertTrue(result['success'])
        self.assertEqual(result['model'], "all-MiniLM-L6-v2")
        self.assertEqual(result['type'], 'embedding')
        self.assertIn('display_name', result)
        self.assertIn('description', result)
        self.assertIn('requirements_check', result)

        result = manager.get_model_info("non-existent-model", "embedding")
        self.assertFalse(result['success'])
        self.assertIn('error', result)

    @patch('src.ai_engine.model_manager.subprocess.run')
    def test_validate_model_files_llm(self, mock_subprocess):
        manager = ModelManager(self.config)
        manager.ollama_available = True

        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "NAME\nllama2:7b\nmistral:7b"
        mock_subprocess.return_value = mock_result

        result = manager.validate_model_files("llama2:7b", "llm")

        self.assertTrue(result['success'])
        self.assertTrue(result['valid'])
        self.assertEqual(result['model_name'], "llama2:7b")

        mock_result.stdout = "NAME\nmistral:7b"
        result = manager.validate_model_files("llama2:7b", "llm")

        self.assertTrue(result['success'])
        self.assertFalse(result['valid'])

    @patch('src.ai_engine.model_manager.SentenceTransformer')
    def test_validate_model_files_embedding(self, mock_sentence_transformer):
        manager = ModelManager(self.config)

        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model

        result = manager.validate_model_files("all-MiniLM-L6-v2", "embedding")

        self.assertTrue(result['success'])
        self.assertTrue(result['valid'])
        self.assertEqual(result['model_name'], "all-MiniLM-L6-v2")

        mock_sentence_transformer.side_effect = Exception("Model not found")
        result = manager.validate_model_files("all-MiniLM-L6-v2", "embedding")

        self.assertFalse(result['success'])
        self.assertFalse(result['valid'])

    def test_validate_model_download(self):
        manager = ModelManager(self.config)

        with patch.object(manager, '_check_disk_space', return_value=True):
            with patch.object(manager.resource_monitor, 'check_requirements') as mock_check:
                mock_check.return_value = {
                    'success': True,
                    'requirements_met': True,
                    'details': {}
                }

                result = manager.validate_model_download("all-MiniLM-L6-v2", "embedding")

                self.assertTrue(result['success'])
                self.assertTrue(result['valid'])
                self.assertEqual(result['model_name'], "all-MiniLM-L6-v2")
                self.assertIn('requirements_check', result)
                self.assertIn('disk_space_ok', result)

    def test_set_active_model(self):
        manager = ModelManager(self.config)

        with patch.object(manager, '_check_model_downloaded', return_value=True):
            success = manager.set_active_model("all-MiniLM-L6-v2", "embedding")

            self.assertTrue(success)
            self.assertIn("embedding", manager.active_models)
            active_model = manager.get_active_model("embedding")
            self.assertEqual(active_model.name, "all-MiniLM-L6-v2")

            self.assertIn("all-MiniLM-L6-v2", manager.model_usage)
            self.assertEqual(manager.model_usage["all-MiniLM-L6-v2"]['activation_count'], 1)

    def test_get_active_model(self):
        manager = ModelManager(self.config)

        result = manager.get_active_model("embedding")
        self.assertIsNone(result)

        with patch.object(manager, '_check_model_downloaded', return_value=True):
            manager.set_active_model("all-MiniLM-L6-v2", "embedding")

            result = manager.get_active_model("embedding")
            self.assertIsNotNone(result)
            self.assertEqual(result.name, "all-MiniLM-L6-v2")

    def test_format_file_size(self):
        manager = ModelManager(self.config)

        test_cases = [
            (0, "0 B"),
            (500, "500.00 B"),
            (1500, "1.46 KB"),
            (1500000, "1.43 MB"),
            (1500000000, "1.40 GB"),
            (1500000000000, "1.36 TB")
        ]

        for size_bytes, expected in test_cases:
            result = manager.format_file_size(size_bytes)
            self.assertEqual(result, expected)

    def test_health_check(self):
        manager = ModelManager(self.config)

        with patch.object(manager, 'get_system_resources') as mock_resources:
            with patch.object(manager, 'check_system_requirements') as mock_reqs:
                with patch.object(manager, '_check_ollama_available', return_value=True):
                    with patch.object(manager, '_check_model_downloaded', return_value=True):

                        mock_resources.return_value = {
                            'cpu': {'cores': 8},
                            'ram': {'total_gb': 16.0},
                            'disk': {'free_gb': 50.0}
                        }

                        mock_reqs.return_value = {
                            'success': True,
                            'requirements_met': True
                        }

                        health = manager.health_check()

                        self.assertTrue(health['success'])
                        self.assertIn('status', health)
                        self.assertIn('health_score', health)
                        self.assertIn('recommendations', health)
                        self.assertIn('check_duration_ms', health)

    def test_get_download_status(self):
        manager = ModelManager(self.config)

        mock_downloads = [{'model': 'test-model', 'progress': 50.0}]
        mock_history = [{'model': 'test-model', 'success': True}]

        manager.downloader.get_active_downloads = Mock(return_value=mock_downloads)
        manager.downloader.get_download_history = Mock(return_value=mock_history)

        status = manager.get_download_status()

        self.assertIn('active_downloads', status)
        self.assertIn('download_history', status)
        self.assertIn('timestamp', status)
        self.assertEqual(len(status['active_downloads']), 1)
        self.assertEqual(len(status['download_history']), 1)

    @patch('src.ai_engine.model_manager.shutil.rmtree')
    def test_cleanup(self, mock_rmtree):
        manager = ModelManager(self.config)

        mock_result = {
            'removed_count': 3,
            'freed_bytes': 1500000000,
            'freed_mb': 1430.51
        }
        manager.downloader.cleanup_downloads = Mock(return_value=mock_result)

        result = manager.cleanup(max_age_days=30)

        self.assertEqual(result['removed_count'], 3)
        self.assertEqual(result['freed_bytes'], 1500000000)
        manager.downloader.cleanup_downloads.assert_called_once_with(30)

    def test_get_model_statistics(self):
        manager = ModelManager(self.config)
        
        manager.model_usage = {
            "all-MiniLM-L6-v2": {
                "download_count": 3,
                "activation_count": 15,
                "last_downloaded": "2024-01-15T10:00:00",
                "last_activated": "2024-01-20T14:30:00"
            },
            "llama2:7b": {
                "download_count": 2,
                "activation_count": 8,
                "last_downloaded": "2024-01-10T09:00:00",
                "last_activated": "2024-01-18T11:00:00"
            }
        }
        
        stats = manager.get_model_statistics()
        
        self.assertIn("all-MiniLM-L6-v2", stats)
        self.assertIn("llama2:7b", stats)
        
        model_stats = stats["all-MiniLM-L6-v2"]
        self.assertEqual(model_stats["download_count"], 3)
        self.assertEqual(model_stats["activation_count"], 15)
        self.assertEqual(model_stats["last_downloaded"], "2024-01-15T10:00:00")
        
        self.assertEqual(len(stats), 2)


class TestModelManagerIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = {
            'download_dir': str(self.temp_dir / "downloads"),
            'models_dir': str(self.temp_dir / "models")
        }

        (self.temp_dir / "downloads").mkdir(parents=True, exist_ok=True)
        (self.temp_dir / "models").mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_complete_model_lifecycle(self):
        manager = ModelManager(self.config)

        models = manager.get_available_models('embedding')
        self.assertGreater(len(models), 0)

        model_name = models[0]['name']
        model_info = manager.get_model_info(model_name, 'embedding')
        self.assertTrue(model_info['success'])

        validation = manager.validate_model_download(model_name, 'embedding')
        self.assertIn('valid', validation)

        with patch.object(manager, '_check_model_downloaded', return_value=True):
            success = manager.set_active_model(model_name, 'embedding')
            self.assertTrue(success)

            active = manager.get_active_model('embedding')
            self.assertIsNotNone(active)
            self.assertEqual(active.name, model_name)

        health = manager.health_check()
        self.assertTrue(health['success'])

        resources = manager.get_system_resources()
        self.assertIn('cpu', resources)
        self.assertIn('ram', resources)
        self.assertIn('disk', resources)

        status = manager.get_download_status()
        self.assertIn('active_downloads', status)
        self.assertIn('download_history', status)

    def test_singleton_pattern(self):
        manager1 = get_model_manager(self.config)
        manager2 = get_model_manager(self.config)

        self.assertIs(manager1, manager2)

        manager3 = get_model_manager({'download_dir': '/tmp/test'})
        self.assertIsNot(manager1, manager3)


class TestModelRegistry(unittest.TestCase):
    def setUp(self):
        from src.ai_engine.model_manager import ModelRegistry
        self.registry = ModelRegistry()

    def test_get_model(self):
        model = self.registry.get_model("llama2:7b", "llm")
        self.assertIsNotNone(model)
        self.assertEqual(model.name, "llama2:7b")
        self.assertEqual(model.model_type, "llm")

        model = self.registry.get_model("all-MiniLM-L6-v2", "embedding")
        self.assertIsNotNone(model)
        self.assertEqual(model.name, "all-MiniLM-L6-v2")
        self.assertEqual(model.model_type, "embedding")

        model = self.registry.get_model("non-existent", "llm")
        self.assertIsNone(model)

    def test_get_all_models(self):
        all_models = self.registry.get_all_models()
        self.assertGreater(len(all_models), 0)

        llm_models = self.registry.get_all_models("llm")
        self.assertGreater(len(llm_models), 0)
        self.assertTrue(all(m.model_type == "llm" for m in llm_models))

        embedding_models = self.registry.get_all_models("embedding")
        self.assertGreater(len(embedding_models), 0)
        self.assertTrue(all(m.model_type == "embedding" for m in embedding_models))

    def test_register_model(self):
        new_model = ModelMetadata(
            name="custom-model",
            display_name="Custom Model",
            model_type="custom",
            provider="custom-provider",
            description="Custom test model"
        )

        success = self.registry.register_model(new_model)

        self.assertTrue(success)
        registered = self.registry.get_model("custom-model", "custom")
        self.assertIsNotNone(registered)
        self.assertEqual(registered.name, "custom-model")

    def test_unregister_model(self):
        new_model = ModelMetadata(
            name="to-remove",
            display_name="To Remove",
            model_type="llm",
            provider="test",
            description="To be removed"
        )

        self.registry.register_model(new_model)
        self.assertIsNotNone(self.registry.get_model("to-remove", "llm"))

        success = self.registry.unregister_model("to-remove")
        self.assertTrue(success)
        self.assertIsNone(self.registry.get_model("to-remove", "llm"))

        success = self.registry.unregister_model("non-existent")
        self.assertFalse(success)

    def test_get_default_model(self):
        default_llm = self.registry.get_default_model("llm")
        self.assertIsNotNone(default_llm)
        self.assertTrue(default_llm.is_default)

        default_embedding = self.registry.get_default_model("embedding")
        self.assertIsNotNone(default_embedding)
        self.assertTrue(default_embedding.is_default)

    def test_set_default_model(self):
        current_default = self.registry.get_default_model("llm")
        self.assertIsNotNone(current_default)

        success = self.registry.set_default_model("mistral:7b", "llm")
        self.assertTrue(success)

        new_default = self.registry.get_default_model("llm")
        self.assertIsNotNone(new_default)
        self.assertEqual(new_default.name, "mistral:7b")
        self.assertTrue(new_default.is_default)

        old_model = self.registry.get_model(current_default.name, "llm")
        self.assertFalse(old_model.is_default)


class TestSystemResourceMonitor(unittest.TestCase):
    def setUp(self):
        from src.ai_engine.model_manager import SystemResourceMonitor
        self.monitor = SystemResourceMonitor()

    @patch('src.ai_engine.model_manager.psutil.virtual_memory')
    @patch('src.ai_engine.model_manager.psutil.cpu_percent')
    @patch('src.ai_engine.model_manager.psutil.cpu_freq')
    @patch('src.ai_engine.model_manager.psutil.disk_usage')
    @patch('src.ai_engine.model_manager.platform')
    def test_get_system_resources(self, mock_platform, mock_disk_usage,
                                 mock_cpu_freq, mock_cpu_percent, mock_virtual_memory):
        mock_platform.system.return_value = "Linux"
        mock_platform.release.return_value = "5.15.0"
        mock_platform.version.return_value = "#1 SMP"
        mock_platform.machine.return_value = "x86_64"
        mock_platform.processor.return_value = "Intel"
        mock_platform.python_version.return_value = "3.11.0"
        mock_platform.node.return_value = "test-host"

        mock_cpu_freq.return_value = Mock(current=3200.0)
        mock_cpu_percent.return_value = 25.0

        mock_memory = Mock()
        mock_memory.total = 16 * (1024 ** 3)
        mock_memory.available = 8 * (1024 ** 3)
        mock_memory.used = 8 * (1024 ** 3)
        mock_memory.percent = 50.0
        mock_virtual_memory.return_value = mock_memory

        mock_usage = Mock()
        mock_usage.total = 500 * (1024 ** 3)
        mock_usage.free = 300 * (1024 ** 3)
        mock_usage.used = 200 * (1024 ** 3)
        mock_usage.percent = 40.0
        mock_disk_usage.return_value = mock_usage

        resources = self.monitor.get_system_resources()

        self.assertIn('cpu', resources)
        self.assertIn('ram', resources)
        self.assertIn('disk', resources)
        self.assertIn('system', resources)
        self.assertIn('timestamp', resources)

        self.assertEqual(resources['cpu']['cores'], 1)
        self.assertEqual(resources['ram']['total_gb'], 16.0)
        self.assertEqual(resources['system']['system'], "Linux")

    def test_check_requirements(self):
        requirements = ModelRequirements(
            ram_gb=8.0,
            storage_gb=10.0,
            cpu_cores=4,
            gpu_vram_gb=8.0,
            python_version="3.11"
        )

        mock_resources = {
            'cpu': {'cores': 8, 'percent': 25.0},
            'ram': {'total_gb': 16.0, 'available_gb': 12.0},
            'disk': {'/home/user': {'free_gb': 50.0}},
            'gpu': {'available': True, 'count': 1, 'devices': [{'total_memory_gb': 12.0}]},
            'system': {'python_version': '3.11.5'}
        }

        self.monitor.get_system_resources = Mock(return_value=mock_resources)

        result = self.monitor.check_requirements(requirements)

        self.assertTrue(result['success'])
        self.assertTrue(result['requirements_met'])
        self.assertIn('details', result)

        details = result['details']
        self.assertIn('ram', details)
        self.assertIn('cpu', details)
        self.assertIn('disk', details)
        self.assertIn('gpu', details)
        self.assertIn('python', details)

        self.assertTrue(all(check['met'] for check in details.values()))

    def test_get_resource_history(self):
        self.monitor.history = [
            {
                'timestamp': '2024-01-01T10:00:00',
                'cpu': {'percent': 25.0},
                'ram': {'percent': 50.0}
            },
            {
                'timestamp': '2024-01-01T10:01:00',
                'cpu': {'percent': 30.0},
                'ram': {'percent': 55.0}
            },
            {
                'timestamp': '2024-01-01T10:02:00',
                'cpu': {'percent': 35.0},
                'ram': {'percent': 60.0}
            }
        ]

        cpu_history = self.monitor.get_resource_history('cpu.percent', time_window_minutes=5)

        self.assertEqual(len(cpu_history), 3)
        self.assertEqual(cpu_history[0]['value'], 25.0)
        self.assertEqual(cpu_history[1]['value'], 30.0)
        self.assertEqual(cpu_history[2]['value'], 35.0)

        ram_history = self.monitor.get_resource_history('ram.percent', time_window_minutes=2)

        self.assertGreaterEqual(len(ram_history), 2)

    def test_get_resource_summary(self):
        mock_resources = {
            'cpu': {'percent': 25.0, 'cores': 8},
            'ram': {'percent': 50.0, 'total_gb': 16.0, 'available_gb': 8.0},
            'disk': {'/home/user': {'free_gb': 50.0}},
            'gpu': {'available': True, 'count': 1},
            'system': {'system': 'Linux'},
            'timestamp': '2024-01-01T10:00:00'
        }

        self.monitor.get_system_resources = Mock(return_value=mock_resources)

        summary = self.monitor.get_resource_summary()

        self.assertIn('cpu', summary)
        self.assertIn('ram', summary)
        self.assertIn('disk', summary)
        self.assertIn('gpu', summary)
        self.assertIn('system', summary)
        self.assertIn('timestamp', summary)

        self.assertEqual(summary['cpu']['usage_percent'], 25.0)
        self.assertEqual(summary['ram']['usage_percent'], 50.0)
        self.assertEqual(summary['gpu']['available'], True)


class TestModelDownloader(unittest.TestCase):
    def setUp(self):
        from src.ai_engine.model_manager import ModelDownloader
        self.temp_dir = Path(tempfile.mkdtemp())
        self.downloader = ModelDownloader(str(self.temp_dir))

    def tearDown(self):
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        self.assertEqual(self.downloader.download_dir, self.temp_dir)
        self.assertIsInstance(self.downloader.active_downloads, dict)
        self.assertIsInstance(self.downloader.download_history, list)
        self.assertIsNotNone(self.downloader.executor)
        self.assertIsNotNone(self.downloader.lock)

    @patch('src.ai_engine.model_manager.requests.get')
    def test_download_http_success(self, mock_requests_get):
        metadata = Mock()
        metadata.name = "test-model"
        metadata.model_type = "embedding"
        metadata.download_url = "http://example.com/model.bin"
        metadata.checksum = None
        metadata.download_progress = 0.0
        metadata.is_downloaded = False
        metadata.updated_at = None
        metadata.installed_size_mb = 0.0
        metadata.to_dict.return_value = {}

        model_dir = self.temp_dir / "embedding" / "test-model"
        model_dir.mkdir(parents=True, exist_ok=True)

        mock_response = Mock()
        mock_response.headers = {'content-length': '1000'}
        mock_response.iter_content.return_value = [b'chunk1', b'chunk2', b'chunk3']
        mock_response.raise_for_status.return_value = None
        mock_requests_get.return_value = mock_response

        with patch('src.ai_engine.model_manager.tqdm'):
            result = self.downloader._download_http(
                metadata,
                model_dir,
                "test-download-id"
            )

        self.assertTrue(result.exists())
        self.assertEqual(result.name, "model.bin")
        mock_requests_get.assert_called_once_with(
            "http://example.com/model.bin",
            stream=True,
            timeout=30
        )

    def test_copy_local(self):
        source_file = self.temp_dir / "source_model.bin"
        source_file.write_text("test model content")

        metadata = Mock()
        metadata.download_url = str(source_file)
        metadata.name = "test-model"

        dest_dir = self.temp_dir / "models" / "test-model"
        dest_dir.mkdir(parents=True, exist_ok=True)

        result = self.downloader._copy_local(metadata, dest_dir, "test-download-id")

        self.assertTrue(result.exists())
        self.assertEqual(result.read_text(), "test model content")

    def test_verify_checksum(self):
        test_file = self.temp_dir / "test.bin"
        test_file.write_text("test content")

        import hashlib
        sha256_hash = hashlib.sha256()
        sha256_hash.update(b"test content")
        expected_checksum = sha256_hash.hexdigest()

        valid = self.downloader._verify_checksum(test_file, expected_checksum)
        self.assertTrue(valid)

        invalid = self.downloader._verify_checksum(test_file, "invalid_checksum")
        self.assertFalse(invalid)

    def test_calculate_directory_size(self):
        test_dir = self.temp_dir / "test_dir"
        test_dir.mkdir()

        (test_dir / "file1.txt").write_text("x" * 1000)
        (test_dir / "file2.txt").write_text("x" * 2000)
        (test_dir / "subdir").mkdir()
        (test_dir / "subdir" / "file3.txt").write_text("x" * 3000)

        total_size = self.downloader._calculate_directory_size(test_dir)
        self.assertEqual(total_size, 6000)

    def test_get_active_downloads(self):
        self.downloader.active_downloads = {
            "test-id": {
                "model": "test-model",
                "progress": 50.0,
                "status": "downloading"
            }
        }

        downloads = self.downloader.get_active_downloads()

        self.assertEqual(len(downloads), 1)
        self.assertEqual(downloads[0]["model"], "test-model")
        self.assertEqual(downloads[0]["progress"], 50.0)

    def test_get_download_history(self):
        self.downloader.download_history = [
            {"model": "model1", "success": True},
            {"model": "model2", "success": True},
            {"model": "model3", "success": False}
        ]

        history = self.downloader.get_download_history()
        self.assertEqual(len(history), 3)

        limited = self.downloader.get_download_history(limit=2)
        self.assertEqual(len(limited), 2)

    @patch('src.ai_engine.model_manager.shutil.rmtree')
    def test_cleanup_downloads(self, mock_rmtree):
        import time
        old_time = time.time() - (60 * 60 * 24 * 31)

        old_dir = self.temp_dir / "embedding" / "old-model"
        old_dir.mkdir(parents=True, exist_ok=True)

        test_file = old_dir / "model.bin"
        test_file.write_text("x" * 1000)

        os.utime(old_dir, (old_time, old_time))

        new_dir = self.temp_dir / "embedding" / "new-model"
        new_dir.mkdir(parents=True, exist_ok=True)

        result = self.downloader.cleanup_downloads(max_age_days=30)

        self.assertEqual(result['removed_count'], 1)
        self.assertGreater(result['freed_bytes'], 0)
        mock_rmtree.assert_called_once_with(old_dir)


if __name__ == '__main__':
    unittest.main()