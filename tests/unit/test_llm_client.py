#docubot/tests/unit/test_llm_client.py

"""
Complete unit tests for LLM Client implementation.
Tests multi-model support, configuration, error handling, and all features.
"""

import unittest
import sys
import json
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from pathlib import Path
from datetime import datetime
import time

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from ai_engine.llm_client import (
    LLMConfig, 
    LLMResponse, 
    LLMClient, 
    LLMModel,
    ModelInfo,
    create_llm_client,
    get_global_llm_client,
    validate_llm_environment,
    OLLAMA_AVAILABLE
)


class TestLLMConfig(unittest.TestCase):
    """Test LLMConfig dataclass with validation."""
    
    def test_basic_creation(self):
        """Test basic LLMConfig creation."""
        config = LLMConfig(
            model="test-model",
            host="http://localhost:11434",
            temperature=0.5,
            top_p=0.9,
            top_k=50,
            max_tokens=512,
            context_window=2048,
            repeat_penalty=1.1,
            seed=123,
            timeout=60,
            max_retries=2,
            retry_delay=1
        )
        
        self.assertEqual(config.model, "test-model")
        self.assertEqual(config.host, "http://localhost:11434")
        self.assertEqual(config.temperature, 0.5)
        self.assertEqual(config.max_tokens, 512)
        self.assertEqual(config.timeout, 60)
    
    def test_default_values(self):
        """Test default values in LLMConfig."""
        config = LLMConfig()
        
        self.assertEqual(config.model, "llama2:7b")
        self.assertEqual(config.host, "http://localhost:11434")
        self.assertEqual(config.temperature, 0.1)
        self.assertEqual(config.max_tokens, 1024)
        self.assertEqual(config.timeout, 300)
        self.assertEqual(config.max_retries, 3)
        self.assertEqual(config.retry_delay, 2)
    
    def test_validation_success(self):
        """Test successful configuration validation."""
        config = LLMConfig(
            model="llama2:7b",
            host="http://localhost:11434",
            temperature=0.5,
            max_tokens=1024,
            timeout=30
        )
        
        is_valid, errors = config.validate()
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
    
    def test_validation_failure(self):
        """Test configuration validation failures."""
        # Invalid temperature
        config = LLMConfig(temperature=2.5)
        is_valid, errors = config.validate()
        self.assertFalse(is_valid)
        self.assertIn("Temperature must be between 0.0 and 2.0", errors[0])
        
        # Invalid host
        config = LLMConfig(host="invalid_host")
        is_valid, errors = config.validate()
        self.assertFalse(is_valid)
        self.assertIn("Host must be a valid URL", errors[0])
        
        # Invalid max_tokens
        config = LLMConfig(max_tokens=-10)
        is_valid, errors = config.validate()
        self.assertFalse(is_valid)
        self.assertIn("max_tokens must be positive", errors[0])
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = LLMConfig(
            model="test-model",
            temperature=0.5,
            max_tokens=512
        )
        
        config_dict = config.to_dict()
        
        self.assertEqual(config_dict['model'], "test-model")
        self.assertEqual(config_dict['temperature'], 0.5)
        self.assertEqual(config_dict['max_tokens'], 512)
        self.assertEqual(config_dict['timeout'], 300)  # Default value
        self.assertEqual(config_dict['max_retries'], 3)  # Default value
    
    def test_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            'model': 'custom-model',
            'temperature': 0.7,
            'max_tokens': 2048,
            'host': 'http://custom-host:8080'
        }
        
        config = LLMConfig.from_dict(config_dict)
        
        self.assertEqual(config.model, 'custom-model')
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.max_tokens, 2048)
        self.assertEqual(config.host, 'http://custom-host:8080')
        self.assertEqual(config.timeout, 300)  # Should use default


class TestLLMResponse(unittest.TestCase):
    """Test LLMResponse dataclass."""
    
    def test_basic_response(self):
        """Test basic LLMResponse creation."""
        response = LLMResponse(
            content="Test response content",
            model="llama2:7b",
            prompt_tokens=10,
            completion_tokens=20,
            total_tokens=30,
            processing_time_ms=1500.5,
            finish_reason="stop"
        )
        
        self.assertEqual(response.content, "Test response content")
        self.assertEqual(response.model, "llama2:7b")
        self.assertEqual(response.prompt_tokens, 10)
        self.assertEqual(response.completion_tokens, 20)
        self.assertEqual(response.total_tokens, 30)
        self.assertEqual(response.processing_time_ms, 1500.5)
        self.assertEqual(response.finish_reason, "stop")
    
    def test_with_metadata(self):
        """Test LLMResponse with metadata."""
        metadata = {
            'source_documents': ['doc1.pdf', 'doc2.pdf'],
            'confidence': 0.85,
            'generation_id': '12345'
        }
        
        response = LLMResponse(
            content="Response with metadata",
            model="mistral:7b",
            metadata=metadata
        )
        
        self.assertEqual(response.content, "Response with metadata")
        self.assertEqual(response.model, "mistral:7b")
        self.assertEqual(response.metadata['confidence'], 0.85)
        self.assertEqual(response.metadata['source_documents'], ['doc1.pdf', 'doc2.pdf'])
    
    def test_to_dict(self):
        """Test conversion to dictionary."""
        response = LLMResponse(
            content="Test response",
            model="llama2:7b",
            prompt_tokens=15,
            completion_tokens=25,
            total_tokens=40,
            processing_time_ms=1200.0,
            finish_reason="length"
        )
        
        response_dict = response.to_dict()
        
        self.assertEqual(response_dict['content'], "Test response")
        self.assertEqual(response_dict['model'], "llama2:7b")
        self.assertEqual(response_dict['prompt_tokens'], 15)
        self.assertEqual(response_dict['completion_tokens'], 25)
        self.assertEqual(response_dict['total_tokens'], 40)
        self.assertEqual(response_dict['processing_time_ms'], 1200.0)
        self.assertEqual(response_dict['finish_reason'], "length")


class TestLLMClient(unittest.TestCase):
    """Complete LLMClient unit tests."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock ollama module
        self.ollama_patcher = patch('ai_engine.llm_client.ollama', create=True)
        self.mock_ollama = self.ollama_patcher.start()
        
        # Mock httpx module
        self.httpx_patcher = patch('ai_engine.llm_client.httpx', create=True)
        self.mock_httpx = self.httpx_patcher.start()
        
        # Mock get_config
        self.config_patcher = patch('ai_engine.llm_client.get_config')
        self.mock_get_config = self.config_patcher.start()
        
        # Setup mock configuration
        self.mock_app_config = Mock()
        self.mock_app_config.ai.get.return_value = {
            'model': 'llama2:7b',
            'host': 'http://localhost:11434',
            'temperature': 0.1,
            'max_tokens': 1024
        }
        self.mock_get_config.return_value = self.mock_app_config
        
        # Create mock Ollama client
        self.mock_client_instance = Mock()
        self.mock_ollama.Client.return_value = self.mock_client_instance
        
        # Mock successful list response
        self.mock_client_instance.list.return_value = {
            'models': [
                {'name': 'llama2:7b', 'size': 4200000000},
                {'name': 'mistral:7b', 'size': 4100000000}
            ]
        }
        
        # Create LLMClient instance
        self.llm_client = LLMClient()
    
    def tearDown(self):
        """Clean up after tests."""
        self.ollama_patcher.stop()
        self.httpx_patcher.stop()
        self.config_patcher.stop()
    
    def test_initialization(self):
        """Test LLMClient initialization."""
        self.assertIsNotNone(self.llm_client)
        self.assertEqual(self.llm_client.config.model, 'llama2:7b')
        self.assertEqual(self.llm_client.config.host, 'http://localhost:11434')
        self.assertTrue(self.llm_client.connected)
        
        # Verify Ollama client was created with correct parameters
        self.mock_ollama.Client.assert_called_once_with(
            host='http://localhost:11434',
            timeout=300
        )
    
    def test_initialization_with_custom_config(self):
        """Test initialization with custom configuration."""
        custom_config = LLMConfig(
            model='mistral:7b',
            host='http://custom-host:8080',
            temperature=0.5,
            timeout=60
        )
        
        client = LLMClient(custom_config)
        
        self.assertEqual(client.config.model, 'mistral:7b')
        self.assertEqual(client.config.host, 'http://custom-host:8080')
        self.assertEqual(client.config.temperature, 0.5)
        self.assertEqual(client.config.timeout, 60)
    
    def test_ensure_connection_success(self):
        """Test successful connection check."""
        self.assertTrue(self.llm_client.ensure_connection())
        self.assertTrue(self.llm_client.connected)
        
        # Verify list was called
        self.mock_client_instance.list.assert_called()
    
    def test_ensure_connection_failure(self):
        """Test connection check failure."""
        # Mock connection failure
        self.mock_client_instance.list.side_effect = Exception("Connection failed")
        self.llm_client.connected = False
        
        # Should return False after retries
        result = self.llm_client.ensure_connection()
        self.assertFalse(result)
        self.assertFalse(self.llm_client.connected)
    
    @patch('ai_engine.llm_client.time.time')
    def test_generate_complete_success(self, mock_time):
        """Test successful non-streaming generation."""
        # Mock time
        mock_time.side_effect = [1000.0, 1001.5]  # 1.5 second difference
        
        # Mock successful response
        mock_response = {
            'response': 'This is a test response.',
            'model': 'llama2:7b',
            'prompt_eval_count': 25,
            'eval_count': 75,
            'done_reason': 'stop',
            'total_duration': 1500000000,
            'load_duration': 50000000,
            'prompt_eval_duration': 1000000000,
            'eval_duration': 450000000
        }
        self.mock_client_instance.generate.return_value = mock_response
        
        # Generate response
        response = self.llm_client.generate("Test prompt", stream=False)
        
        # Verify response
        self.assertIsInstance(response, LLMResponse)
        self.assertEqual(response.content, 'This is a test response.')
        self.assertEqual(response.model, 'llama2:7b')
        self.assertEqual(response.prompt_tokens, 25)
        self.assertEqual(response.completion_tokens, 75)
        self.assertEqual(response.total_tokens, 100)
        self.assertEqual(response.processing_time_ms, 1500.0)  # 1.5 seconds
        self.assertEqual(response.finish_reason, 'stop')
        
        # Verify API call
        self.mock_client_instance.generate.assert_called_once_with(
            model='llama2:7b',
            prompt='Test prompt',
            options={
                'temperature': 0.1,
                'top_p': 0.9,
                'top_k': 40,
                'num_predict': 1024,
                'repeat_penalty': 1.1,
                'seed': 42
            },
            stream=False
        )
    
    def test_generate_stream(self):
        """Test streaming generation."""
        # Mock streaming response
        mock_stream = [
            {'response': 'Hello'},
            {'response': ' world'},
            {'response': '!'}
        ]
        self.mock_client_instance.generate.return_value = mock_stream
        
        # Generate streaming response
        generator = self.llm_client.generate("Say hello", stream=True)
        
        # Collect results
        results = list(generator)
        
        # Verify results
        self.assertEqual(results, ['Hello', ' world', '!'])
        self.mock_client_instance.generate.assert_called_once()
    
    def test_chat_complete_with_system_prompt(self):
        """Test chat completion with system prompt."""
        # Mock response
        mock_response = {
            'message': {'content': 'I am DocuBot, how can I help you?'},
            'model': 'llama2:7b',
            'prompt_eval_count': 30,
            'eval_count': 20,
            'done_reason': 'stop'
        }
        self.mock_client_instance.chat.return_value = mock_response
        
        # Generate with system prompt
        response = self.llm_client.generate(
            "What can you do?",
            system_prompt="You are DocuBot, a helpful AI assistant.",
            stream=False
        )
        
        # Verify response
        self.assertEqual(response.content, 'I am DocuBot, how can I help you?')
        
        # Verify messages structure
        call_args = self.mock_client_instance.chat.call_args
        messages = call_args[1]['messages']
        
        self.assertEqual(len(messages), 2)
        self.assertEqual(messages[0]['role'], 'system')
        self.assertEqual(messages[0]['content'], 'You are DocuBot, a helpful AI assistant.')
        self.assertEqual(messages[1]['role'], 'user')
        self.assertEqual(messages[1]['content'], 'What can you do?')
    
    def test_chat_with_conversation_history(self):
        """Test chat with conversation history."""
        # Mock response
        mock_response = {
            'message': {'content': 'Continuing the conversation.'},
            'model': 'llama2:7b'
        }
        self.mock_client_instance.chat.return_value = mock_response
        
        # Conversation history
        history = [
            {'role': 'user', 'content': 'Hello'},
            {'role': 'assistant', 'content': 'Hi there!'},
            {'role': 'user', 'content': 'How are you?'}
        ]
        
        # Generate with history
        response = self.llm_client.generate(
            "What were we talking about?",
            conversation_history=history,
            stream=False
        )
        
        # Verify messages structure
        call_args = self.mock_client_instance.chat.call_args
        messages = call_args[1]['messages']
        
        self.assertEqual(len(messages), 4)  # history + new prompt
        self.assertEqual(messages[0]['role'], 'user')
        self.assertEqual(messages[0]['content'], 'Hello')
        self.assertEqual(messages[3]['role'], 'user')
        self.assertEqual(messages[3]['content'], 'What were we talking about?')
    
    def test_switch_model_success(self):
        """Test successful model switching."""
        # Mock model available
        self.mock_client_instance.list.return_value = {
            'models': [
                {'name': 'llama2:7b'},
                {'name': 'mistral:7b'},
                {'name': 'neural-chat:7b'}
            ]
        }
        
        # Switch model
        result = self.llm_client.switch_model('mistral:7b')
        
        self.assertTrue(result)
        self.assertEqual(self.llm_client.config.model, 'mistral:7b')
    
    def test_switch_model_not_available(self):
        """Test switching to unavailable model."""
        # Mock model not available
        self.mock_client_instance.list.return_value = {
            'models': [{'name': 'llama2:7b'}]
        }
        
        # Try to switch to unavailable model
        result = self.llm_client.switch_model('unknown-model')
        
        self.assertFalse(result)
        self.assertEqual(self.llm_client.config.model, 'llama2:7b')  # Should not change
    
    def test_get_available_models(self):
        """Test getting available models."""
        # Mock list response
        self.mock_client_instance.list.return_value = {
            'models': [
                {'name': 'llama2:7b', 'size': 4200000000, 'modified_at': '2024-01-01'},
                {'name': 'mistral:7b', 'size': 4100000000, 'modified_at': '2024-01-02'}
            ]
        }
        
        models = self.llm_client.get_available_models()
        
        self.assertEqual(len(models), 2)
        self.assertEqual(models[0]['name'], 'llama2:7b')
        self.assertEqual(models[0]['size_gb'], 4.2)  # Converted from bytes
        self.assertEqual(models[0]['is_supported'], True)
        self.assertEqual(models[0]['is_current'], True)
        
        self.assertEqual(models[1]['name'], 'mistral:7b')
        self.assertEqual(models[1]['size_gb'], 4.1)
        self.assertEqual(models[1]['is_supported'], True)
        self.assertEqual(models[1]['is_current'], False)
    
    def test_is_model_available(self):
        """Test checking model availability."""
        # Mock list response
        self.mock_client_instance.list.return_value = {
            'models': [
                {'name': 'llama2:7b'},
                {'name': 'mistral:7b'}
            ]
        }
        
        # Refresh cache
        self.llm_client._available_models_cache = None
        
        # Test available model
        self.assertTrue(self.llm_client.is_model_available('llama2:7b'))
        
        # Test unavailable model
        self.assertFalse(self.llm_client.is_model_available('unknown-model'))
    
    def test_get_supported_models(self):
        """Test getting supported models information."""
        # Mock available models
        self.mock_client_instance.list.return_value = {
            'models': [
                {'name': 'llama2:7b', 'size': 4200000000},
                {'name': 'mistral:7b', 'size': 4100000000}
            ]
        }
        
        models = self.llm_client.get_supported_models()
        
        # Should return all models in database, with availability info
        self.assertGreater(len(models), 5)  # At least 5 models in database
        
        # Find specific models
        llama_model = next(m for m in models if m.name == 'llama2:7b')
        mistral_model = next(m for m in models if m.name == 'mistral:7b')
        
        self.assertEqual(llama_model.display_name, 'Llama 2 7B')
        self.assertTrue(llama_model.is_available)
        self.assertTrue(llama_model.is_current)
        
        self.assertEqual(mistral_model.display_name, 'Mistral 7B')
        self.assertTrue(mistral_model.is_available)
        self.assertFalse(mistral_model.is_current)
    
    def test_get_model_info(self):
        """Test getting detailed model information."""
        # Mock show response
        mock_info = {
            'license': 'Llama 2 Community License',
            'modelfile': '# Modelfile content',
            'parameters': '{"temperature": 0.1}',
            'template': '{{ .Prompt }}',
            'system': 'You are a helpful assistant',
            'details': {'parameter_size': '7B', 'quantization': 'Q4_0'},
            'size': 4200000000,
            'digest': 'sha256:abc123',
            'modified_at': '2024-01-01T12:00:00Z'
        }
        self.mock_client_instance.show.return_value = mock_info
        
        info = self.llm_client.get_model_info('llama2:7b')
        
        self.assertTrue(info['success'])
        self.assertEqual(info['model'], 'llama2:7b')
        self.assertEqual(info['display_name'], 'Llama 2 7B')
        self.assertEqual(info['license'], 'Llama 2 Community License')
        self.assertEqual(info['context_window'], 4096)
        self.assertEqual(info['is_supported'], True)
        self.assertEqual(info['is_current'], True)
    
    def test_health_check_success(self):
        """Test successful health check."""
        # Mock connection and models
        self.mock_client_instance.list.return_value = {
            'models': [
                {'name': 'llama2:7b'},
                {'name': 'mistral:7b'}
            ]
        }
        
        health = self.llm_client.health_check()
        
        self.assertTrue(health['success'])
        self.assertEqual(health['status'], 'healthy')
        self.assertGreaterEqual(health['health_score'], 80)
        self.assertTrue(health['connected'])
        self.assertEqual(health['available_models'], 2)
        self.assertTrue(health['current_model_available'])
    
    def test_health_check_failure(self):
        """Test health check with connection failure."""
        # Mock connection failure
        self.mock_client_instance.list.side_effect = Exception("Connection failed")
        self.llm_client.connected = False
        
        health = self.llm_client.health_check()
        
        self.assertFalse(health['connected'])
        self.assertEqual(health['health_score'], 0)
        self.assertEqual(health['status'], 'unhealthy')
    
    def test_pull_model_generator(self):
        """Test model pulling with generator."""
        # Mock pull response stream
        mock_stream = [
            {'status': 'downloading', 'completed': 10, 'total': 100},
            {'status': 'downloading', 'completed': 50, 'total': 100},
            {'status': 'success', 'digest': 'sha256:abc123'}
        ]
        self.mock_client_instance.pull.return_value = mock_stream
        
        # Pull model
        generator = self.llm_client.pull_model('llama2:7b', show_progress=False)
        results = list(generator)
        
        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]['status'], 'downloading')
        self.assertEqual(results[2]['status'], 'success')
        self.assertEqual(results[2]['digest'], 'sha256:abc123')
        
        self.mock_client_instance.pull.assert_called_once_with(
            model='llama2:7b',
            stream=True,
            insecure=False
        )
    
    def test_delete_model_success(self):
        """Test successful model deletion."""
        # Mock available models
        self.mock_client_instance.list.return_value = {
            'models': [
                {'name': 'llama2:7b'},
                {'name': 'mistral:7b'}
            ]
        }
        
        # Delete model
        result = self.llm_client.delete_model('mistral:7b')
        
        self.assertTrue(result['success'])
        self.assertEqual(result['model'], 'mistral:7b')
        self.mock_client_instance.delete.assert_called_once_with('mistral:7b')
    
    def test_delete_current_model(self):
        """Test deleting current model (should switch to default)."""
        # Mock available models
        self.mock_client_instance.list.return_value = {
            'models': [
                {'name': 'llama2:7b'},
                {'name': 'mistral:7b'}
            ]
        }
        
        # Switch to mistral
        self.llm_client.switch_model('mistral:7b')
        self.assertEqual(self.llm_client.config.model, 'mistral:7b')
        
        # Delete current model
        result = self.llm_client.delete_model('mistral:7b')
        
        self.assertTrue(result['success'])
        # Should switch back to default
        self.assertEqual(self.llm_client.config.model, 'llama2:7b')
    
    def test_validate_model_compatibility(self):
        """Test model compatibility validation."""
        # Test compatible model
        result = self.llm_client.validate_model_compatibility('llama2:7b', available_ram_gb=16.0)
        
        self.assertTrue(result['compatible'])
        self.assertEqual(result['model'], 'llama2:7b')
        self.assertEqual(result['ram_required_gb'], 8.0)
        self.assertEqual(result['ram_available_gb'], 16.0)
        self.assertLess(result['ram_utilization_percent'], 100)
        
        # Test incompatible model
        result = self.llm_client.validate_model_compatibility('llama2:7b', available_ram_gb=4.0)
        
        self.assertFalse(result['compatible'])
        self.assertEqual(result['ram_required_gb'], 8.0)
        self.assertEqual(result['ram_available_gb'], 4.0)
        self.assertGreater(result['ram_utilization_percent'], 100)
    
    def test_get_system_prompt(self):
        """Test getting system prompts."""
        # Test default prompt
        default_prompt = self.llm_client.get_system_prompt('default')
        self.assertIn('DocuBot', default_prompt)
        self.assertIn('helpful AI assistant', default_prompt)
        
        # Test concise prompt
        concise_prompt = self.llm_client.get_system_prompt('concise')
        self.assertIn('concise', concise_prompt)
        
        # Test code prompt
        code_prompt = self.llm_client.get_system_prompt('code')
        self.assertIn('code', code_prompt)
        self.assertIn('programming', code_prompt)
        
        # Test unknown type (should return default)
        unknown_prompt = self.llm_client.get_system_prompt('unknown')
        self.assertEqual(unknown_prompt, default_prompt)
    
    def test_get_config(self):
        """Test getting current configuration."""
        config = self.llm_client.get_config()
        
        self.assertIn('config', config)
        self.assertIn('state', config)
        
        config_part = config['config']
        self.assertEqual(config_part['model'], 'llama2:7b')
        self.assertEqual(config_part['host'], 'http://localhost:11434')
        self.assertEqual(config_part['temperature'], 0.1)
        
        state_part = config['state']
        self.assertTrue(state_part['connected'])
        self.assertEqual(state_part['request_count'], 0)  # No requests made yet
    
    def test_get_performance_metrics(self):
        """Test getting performance metrics."""
        # Simulate some activity
        self.llm_client.request_count = 10
        self.llm_client.total_tokens_processed = 5000
        self.llm_client.total_processing_time_ms = 15000.0
        self.llm_client.error_count = 1
        
        metrics = self.llm_client.get_performance_metrics()
        
        self.assertEqual(metrics['requests']['total'], 10)
        self.assertEqual(metrics['requests']['successful'], 9)  # total - errors
        self.assertEqual(metrics['requests']['failed'], 1)
        self.assertEqual(metrics['requests']['error_rate'], 0.1)  # 10%
        self.assertEqual(metrics['tokens']['total_processed'], 5000)
        self.assertEqual(metrics['timing']['average_processing_time_ms'], 1500.0)  # 15000/10
        self.assertTrue(metrics['connection']['connected'])
    
    def test_reset_stats(self):
        """Test resetting performance statistics."""
        # Set some stats
        self.llm_client.request_count = 100
        self.llm_client.total_tokens_processed = 50000
        self.llm_client.total_processing_time_ms = 60000.0
        self.llm_client.error_count = 5
        
        # Reset stats
        self.llm_client.reset_stats()
        
        self.assertEqual(self.llm_client.request_count, 0)
        self.assertEqual(self.llm_client.total_tokens_processed, 0)
        self.assertEqual(self.llm_client.total_processing_time_ms, 0.0)
        self.assertEqual(self.llm_client.error_count, 0)
    
    def test_error_handling_generation(self):
        """Test error handling during generation."""
        # Mock generation failure
        self.mock_client_instance.generate.side_effect = Exception("Generation failed")
        
        # Should return error response after retries
        response = self.llm_client.generate("Test prompt", max_retries=1)
        
        self.assertIsInstance(response, LLMResponse)
        self.assertIn("Error:", response.content)
        self.assertEqual(response.finish_reason, "error")
        self.assertIn("error", response.metadata)


class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions."""
    
    @patch('ai_engine.llm_client.LLMClient')
    def test_create_llm_client_with_config(self, mock_client_class):
        """Test creating LLMClient with custom configuration."""
        config_dict = {
            'model': 'custom-model',
            'host': 'http://custom:8080',
            'temperature': 0.5
        }
        
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        client = create_llm_client(config_dict)
        
        self.assertEqual(client, mock_client)
        mock_client_class.assert_called_once()
    
    @patch('ai_engine.llm_client.LLMClient')
    def test_create_llm_client_default(self, mock_client_class):
        """Test creating LLMClient with default configuration."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        client = create_llm_client()
        
        self.assertEqual(client, mock_client)
        mock_client_class.assert_called_once_with(None)
    
    @patch('ai_engine.llm_client.get_llm_client_from_app_config')
    def test_get_global_llm_client(self, mock_get_client):
        """Test getting global singleton instance."""
        mock_client = Mock()
        mock_get_client.return_value = mock_client
        
        # First call should create instance
        client1 = get_global_llm_client()
        self.assertEqual(client1, mock_client)
        
        # Second call should return same instance
        client2 = get_global_llm_client()
        self.assertEqual(client2, mock_client)
        
        # Should only be created once
        mock_get_client.assert_called_once()


class TestEnvironmentValidation(unittest.TestCase):
    """Test environment validation functions."""
    
    @patch('ai_engine.llm_client.OLLAMA_AVAILABLE', True)
    @patch('ai_engine.llm_client.HTTPX_AVAILABLE', True)
    def test_validate_llm_environment_success(self):
        """Test successful environment validation."""
        with patch('ai_engine.llm_client.ollama') as mock_ollama, \
             patch('ai_engine.llm_client.httpx') as mock_httpx, \
             patch('ai_engine.llm_client.get_config') as mock_get_config:
            
            # Mock versions
            mock_ollama.__version__ = '0.1.25'
            mock_httpx.__version__ = '0.25.0'
            
            # Mock config import
            mock_get_config.side_effect = ImportError("Test import error")
            
            # Mock Ollama server check
            mock_client = Mock()
            mock_client.list.return_value = {'models': []}
            mock_ollama.Client.return_value = mock_client
            
            result = validate_llm_environment()
            
            self.assertTrue(result['ollama_available'])
            self.assertTrue(result['httpx_available'])
            self.assertFalse(result['core_config_available'])  # Mocked import error
            self.assertTrue(result['environment_valid'])
            self.assertTrue(result['ollama_server_running'])
            self.assertEqual(result['ollama_version'], '0.1.25')
            self.assertEqual(result['httpx_version'], '0.25.0')
    
    @patch('ai_engine.llm_client.OLLAMA_AVAILABLE', False)
    def test_validate_llm_environment_failure(self):
        """Test environment validation with missing ollama."""
        result = validate_llm_environment()
        
        self.assertFalse(result['ollama_available'])
        self.assertFalse(result['environment_valid'])
        self.assertIn('Ollama not available', result['errors'][0])


class TestIntegration(unittest.TestCase):
    """Integration tests for LLMClient."""
    
    def test_end_to_end_workflow(self):
        """Test complete workflow from configuration to generation."""
        with patch('ai_engine.llm_client.ollama') as mock_ollama, \
             patch('ai_engine.llm_client.get_config') as mock_get_config:
            
            # Mock configuration
            mock_app_config = Mock()
            mock_app_config.ai.get.return_value = {
                'model': 'llama2:7b',
                'host': 'http://localhost:11434',
                'temperature': 0.1,
                'max_tokens': 1024
            }
            mock_get_config.return_value = mock_app_config
            
            # Mock Ollama client
            mock_client = Mock()
            mock_ollama.Client.return_value = mock_client
            
            # Mock responses
            mock_client.list.return_value = {'models': [{'name': 'llama2:7b'}]}
            mock_client.generate.return_value = {
                'response': 'Test response',
                'model': 'llama2:7b',
                'prompt_eval_count': 10,
                'eval_count': 20,
                'done_reason': 'stop'
            }
            
            # Create client
            client = LLMClient()
            
            # Verify initialization
            self.assertEqual(client.config.model, 'llama2:7b')
            self.assertTrue(client.connected)
            
            # Generate response
            response = client.generate("Test prompt")
            
            # Verify response
            self.assertIsInstance(response, LLMResponse)
            self.assertEqual(response.content, 'Test response')
            self.assertEqual(response.model, 'llama2:7b')
            
            # Get available models
            models = client.get_available_models()
            self.assertEqual(len(models), 1)
            self.assertEqual(models[0]['name'], 'llama2:7b')
            
            # Health check
            health = client.health_check()
            self.assertTrue(health['success'])
            self.assertEqual(health['status'], 'healthy')
    
    def test_multiple_model_workflow(self):
        """Test workflow with multiple models."""
        with patch('ai_engine.llm_client.ollama') as mock_ollama:
            
            # Mock Ollama client with multiple models
            mock_client = Mock()
            mock_ollama.Client.return_value = mock_client
            
            # Mock available models
            mock_client.list.return_value = {
                'models': [
                    {'name': 'llama2:7b', 'size': 4200000000},
                    {'name': 'mistral:7b', 'size': 4100000000},
                    {'name': 'neural-chat:7b', 'size': 4300000000}
                ]
            }
            
            # Mock generation responses
            def mock_generate(*args, **kwargs):
                model = kwargs.get('model', 'llama2:7b')
                return {
                    'response': f'Response from {model}',
                    'model': model,
                    'prompt_eval_count': 10,
                    'eval_count': 20
                }
            
            mock_client.generate.side_effect = mock_generate
            
            # Create client
            client = LLMClient(LLMConfig(model='llama2:7b'))
            
            # Get supported models
            models = client.get_supported_models()
            self.assertGreater(len(models), 3)
            
            # Check availability
            self.assertTrue(client.is_model_available('llama2:7b'))
            self.assertTrue(client.is_model_available('mistral:7b'))
            
            # Switch model
            client.switch_model('mistral:7b')
            self.assertEqual(client.config.model, 'mistral:7b')
            
            # Generate with new model
            response = client.generate("Test with mistral")
            self.assertIn('mistral:7b', response.content)


if __name__ == '__main__':
    unittest.main()