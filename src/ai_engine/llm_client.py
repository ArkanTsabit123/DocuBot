#docubot/src/ai_engine/llm_client.py

"""
Ollama LLM Client for DocuBot
"""

import ollama
from typing import Dict, List, Any, Optional, Generator
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class LLMClient:
    """Ollama LLM client with streaming support"""
    
    SUPPORTED_MODELS = {
        "llama2:7b": {
            "name": "Llama 2 7B",
            "context_window": 4096,
            "description": "Meta's Llama 2 7B parameter model"
        },
        "mistral:7b": {
            "name": "Mistral 7B",
            "context_window": 8192,
            "description": "Mistral AI's 7B parameter model"
        },
        "neural-chat:7b": {
            "name": "Neural Chat 7B",
            "context_window": 4096,
            "description": "Intel's fine-tuned neural chat model"
        },
        "codellama:7b": {
            "name": "CodeLlama 7B",
            "context_window": 16384,
            "description": "Code generation model"
        }
    }
    
    def __init__(self, model: str = "llama2:7b", host: str = "http://localhost:11434"):
        """
        Initialize LLM client.
        
        Args:
            model: Default model to use
            host: Ollama server host
        """
        # Validate model
        if model not in self.SUPPORTED_MODELS:
            logger.warning(f"Model {model} not in supported models list. Using anyway.")
        
        self.model = model
        self.host = host
        self.max_tokens = 1024  # Default maximum tokens
        
        # Test connection
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to Ollama server"""
        try:
            import httpx
            response = httpx.get(f"{self.host}/api/tags", timeout=5.0)
            if response.status_code == 200:
                logger.info(f"Ollama connection successful: {self.host}")
                return True
            else:
                logger.warning(f"Ollama responded with status {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"Could not connect to Ollama at {self.host}: {e}")
            return False
    
    def generate(self,
                prompt: str,
                model: Optional[str] = None,
                stream: bool = False,
                options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Generate response from LLM.
        
        Args:
            prompt: Input prompt
            model: Model to use
            stream: Whether to stream the response
            options: Generation options
            
        Returns:
            Response dictionary
        """
        model = model or self.model
        
        default_options = {
            'temperature': 0.1,
            'top_p': 0.9,
            'top_k': 40,
            'num_predict': self.max_tokens,  # Use instance variable
            'stop': []
        }
        
        if options:
            default_options.update(options)
        
        try:
            if stream:
                response_generator = self._generate_streaming(
                    prompt=prompt,
                    model=model,
                    options=default_options
                )
                return {
                    'success': True,
                    'model': model,
                    'stream': True,
                    'generator': response_generator,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                response = ollama.generate(
                    model=model,
                    prompt=prompt,
                    options=default_options
                )
                
                return {
                    'success': True,
                    'model': model,
                    'response': response['response'],
                    'total_duration': response.get('total_duration', 0),
                    'load_duration': response.get('load_duration', 0),
                    'prompt_eval_count': response.get('prompt_eval_count', 0),
                    'prompt_eval_duration': response.get('prompt_eval_duration', 0),
                    'eval_count': response.get('eval_count', 0),
                    'eval_duration': response.get('eval_duration', 0),
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'model': model,
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_streaming(self,
                          prompt: str,
                          model: str,
                          options: Dict[str, Any]) -> Generator[str, None, None]:
        """
        Generate streaming response.
        
        Args:
            prompt: Input prompt
            model: Model to use
            options: Generation options
            
        Yields:
            Response chunks
        """
        try:
            stream = ollama.generate(
                model=model,
                prompt=prompt,
                options=options,
                stream=True
            )
            
            for chunk in stream:
                if 'response' in chunk:
                    yield chunk['response']
                    
        except Exception as e:
            logger.error(f"Streaming generation failed: {e}")
            yield f"Error: {str(e)}"
    
    def chat(self,
            messages: List[Dict[str, str]],
            model: Optional[str] = None,
            stream: bool = False,
            options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Chat completion with conversation history.
        
        Args:
            messages: List of message dictionaries
            model: Model to use
            stream: Whether to stream the response
            options: Generation options
            
        Returns:
            Response dictionary
        """
        model = model or self.model
        
        default_options = {
            'temperature': 0.1,
            'top_p': 0.9,
            'num_predict': self.max_tokens  # Use instance variable
        }
        
        if options:
            default_options.update(options)
        
        try:
            if stream:
                response_generator = self._chat_streaming(
                    messages=messages,
                    model=model,
                    options=default_options
                )
                return {
                    'success': True,
                    'model': model,
                    'stream': True,
                    'generator': response_generator,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                response = ollama.chat(
                    model=model,
                    messages=messages,
                    options=default_options
                )
                
                return {
                    'success': True,
                    'model': model,
                    'response': response['message']['content'],
                    'total_duration': response.get('total_duration', 0),
                    'load_duration': response.get('load_duration', 0),
                    'prompt_eval_count': response.get('prompt_eval_count', 0),
                    'prompt_eval_duration': response.get('prompt_eval_duration', 0),
                    'eval_count': response.get('eval_count', 0),
                    'eval_duration': response.get('eval_duration', 0),
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'model': model,
                'timestamp': datetime.now().isoformat()
            }
    
    def _chat_streaming(self,
                       messages: List[Dict[str, str]],
                       model: str,
                       options: Dict[str, Any]) -> Generator[str, None, None]:
        """
        Streaming chat completion.
        
        Args:
            messages: List of message dictionaries
            model: Model to use
            options: Generation options
            
        Yields:
            Response chunks
        """
        try:
            stream = ollama.chat(
                model=model,
                messages=messages,
                options=options,
                stream=True
            )
            
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
                    
        except Exception as e:
            logger.error(f"Streaming chat failed: {e}")
            yield f"Error: {str(e)}"
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models.
        
        Returns:
            List of model information
        """
        try:
            models = ollama.list()
            
            formatted_models = []
            for model in models.get('models', []):
                formatted_models.append({
                    'name': model.get('name', ''),
                    'modified_at': model.get('modified_at', ''),
                    'size': model.get('size', 0),
                    'digest': model.get('digest', ''),
                    'details': model.get('details', {})
                })
            
            return formatted_models
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def pull_model(self, model_name: str) -> Dict[str, Any]:
        """
        Pull/download a model.
        
        Args:
            model_name: Name of model to pull
            
        Returns:
            Pull status
        """
        try:
            response = ollama.pull(model_name)
            
            return {
                'success': True,
                'model': model_name,
                'status': response.get('status', ''),
                'digest': response.get('digest', ''),
                'total': response.get('total', 0),
                'completed': response.get('completed', 0)
            }
            
        except Exception as e:
            logger.error(f"Failed to pull model {model_name}: {e}")
            return {
                'success': False,
                'model': model_name,
                'error': str(e)
            }
    
    def delete_model(self, model_name: str) -> bool:
        """
        Delete a model.
        
        Args:
            model_name: Name of model to delete
            
        Returns:
            True if successful
        """
        try:
            ollama.delete(model_name)
            logger.info(f"Deleted model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete model {model_name}: {e}")
            return False
    
    def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get information about a model.
        
        Args:
            model_name: Model name
            
        Returns:
            Model information
        """
        model_name = model_name or self.model
        
        try:
            info = ollama.show(model_name)
            
            return {
                'success': True,
                'model': model_name,
                'license': info.get('license', ''),
                'modelfile': info.get('modelfile', ''),
                'parameters': info.get('parameters', ''),
                'template': info.get('template', ''),
                'details': info.get('details', {})
            }
            
        except Exception as e:
            logger.error(f"Failed to get model info for {model_name}: {e}")
            return {
                'success': False,
                'model': model_name,
                'error': str(e)
            }
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check if Ollama is running and accessible.
        
        Returns:
            Health check results
        """
        try:
            models = self.list_models()
            
            return {
                'success': True,
                'status': 'healthy',
                'host': self.host,
                'available_models': len(models),
                'default_model': self.model,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return {
                'success': False,
                'status': 'unhealthy',
                'host': self.host,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def list_available_models(self) -> List[Dict]:
        """List all available/supported models"""
        return [
            {
                "id": model_id,
                **model_info
            }
            for model_id, model_info in self.SUPPORTED_MODELS.items()
        ]
    
    def switch_model(self, model_name: str) -> bool:
        """
        Switch to different LLM model
        
        Args:
            model_name: Name of model to switch to
            
        Returns:
            True if successful, False otherwise
        """
        if model_name not in self.SUPPORTED_MODELS:
            logger.warning(f"Model {model_name} not supported. Available models:")
            for model_id in self.SUPPORTED_MODELS:
                logger.warning(f"  - {model_id}")
            return False
        
        self.model = model_name
        logger.info(f"Switched to model: {model_name}")
        return True

    # NEW METHODS ADDED BELOW

    def set_max_tokens(self, max_tokens: int):
        """
        Set maximum tokens for generation
        
        Args:
            max_tokens: Maximum number of tokens to generate
        """
        if max_tokens <= 0:
            logger.warning(f"Invalid max_tokens value: {max_tokens}. Must be positive.")
            return
        
        self.max_tokens = max_tokens
        logger.info(f"Max tokens set to: {max_tokens}")
    
    def get_model_manager(self):
        """
        Get model manager instance
        
        Returns:
            ModelManager instance (placeholder for now)
        """
        # Return a placeholder ModelManager
        class ModelManager:
            def __init__(self):
                self.client = self
            
            def list_models(self):
                return self.client.list_models()
            
            def get_model_info(self, model_name):
                return self.client.get_model_info(model_name)
            
            def switch_model(self, model_name):
                return self.client.switch_model(model_name)
            
            def health_check(self):
                return self.client.health_check()
        
        return ModelManager()
    
    def list_multiple_models(self) -> Dict[str, Any]:
        """
        List all available models from Ollama and supported models
        
        Returns:
            Dictionary containing ollama models and supported models
        """
        try:
            ollama_models = self.list_models()
            supported_models = self.list_available_models()
            
            return {
                "ollama_models": ollama_models,
                "supported_models": supported_models,
                "total_ollama": len(ollama_models),
                "total_supported": len(supported_models),
                "success": True,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to list multiple models: {e}")
            return {
                "success": False,
                "error": str(e),
                "ollama_models": [],
                "supported_models": self.list_available_models(),
                "timestamp": datetime.now().isoformat()
            }


_llm_instance = None

def get_llm_client(model: str = "llama2:7b", host: str = "http://localhost:11434") -> LLMClient:
    """
    Get or create LLMClient instance.
    
    Args:
        model: Default model to use
        host: Ollama server host
        
    Returns:
        LLMClient instance
    """
    global _llm_instance
    
    if _llm_instance is None:
        _llm_instance = LLMClient(model, host)
    
    return _llm_instance


if __name__ == "__main__":
    # Setup basic logging for testing
    logging.basicConfig(level=logging.INFO)
    
    client = LLMClient()
    
    print("="*60)
    print("LLM CLIENT TEST")
    print("="*60)
    
    print("\n1. Performing health check...")
    health = client.health_check()
    print(f"   Status: {health['status']}")
    print(f"   Available models: {health.get('available_models', 0)}")
    
    if health['success']:
        print("\n2. Testing new methods...")
        
        # Test set_max_tokens
        print("\n   Testing set_max_tokens()...")
        client.set_max_tokens(2048)
        print(f"   ✓ Max tokens set to: {client.max_tokens}")
        
        # Test list_multiple_models
        print("\n   Testing list_multiple_models()...")
        multiple_models = client.list_multiple_models()
        if multiple_models['success']:
            print(f"   ✓ Found {multiple_models['total_ollama']} Ollama models")
            print(f"   ✓ Found {multiple_models['total_supported']} supported models")
        else:
            print(f"   ✗ Failed: {multiple_models.get('error', 'Unknown error')}")
        
        # Test get_model_manager
        print("\n   Testing get_model_manager()...")
        model_manager = client.get_model_manager()
        if model_manager:
            print(f"   ✓ Model manager created successfully")
            print(f"   ✓ Model manager has {len(model_manager.list_models())} models")
        else:
            print(f"   ✗ Failed to create model manager")
        
        print("\n3. Testing supported models...")
        supported_models = client.list_available_models()
        print(f"   Supported models: {len(supported_models)}")
        for model in supported_models:
            print(f"   - {model['id']}: {model['name']}")
        
        print("\n4. Testing model switching...")
        print(f"   Current model: {client.model}")
        
        # Try switching to Mistral
        success = client.switch_model("mistral:7b")
        if success:
            print(f"   ✓ Switched to: {client.model}")
        else:
            print(f"   ✗ Failed to switch to Mistral")
        
        # Switch back
        client.switch_model("llama2:7b")
        print(f"   ✓ Switched back to: {client.model}")
        
        print("\n5. Testing generation (short test)...")
        response = client.generate(
            prompt="Say 'Hello from DocuBot!'",
            stream=False
        )
        
        if response['success']:
            print(f"   ✓ Response: {response['response'][:50]}...")
            print(f"   Duration: {response.get('total_duration', 0) / 1e9:.2f}s")
        else:
            print(f"   ✗ Generation failed: {response.get('error', 'Unknown error')}")
            print("   Note: This may fail if Ollama is not running")
        
        print("\n6. Available Ollama models:")
        models = client.list_models()
        if models:
            for model in models[:5]:  # Show first 5
                size_gb = model['size'] / 1e9 if 'size' in model else 0
                print(f"   - {model['name']} ({size_gb:.1f}GB)")
            if len(models) > 5:
                print(f"   ... and {len(models) - 5} more")
        else:
            print("   No models found (Ollama may not be running)")
    
    else:
        print(f"\n✗ Health check failed: {health.get('error', 'Unknown error')}")
        print("  Make sure Ollama is installed and running:")
        print("  1. Download from https://ollama.com/")
        print("  2. Run 'ollama serve' in terminal")
        print("  3. Run 'ollama pull llama2:7b' to download a model")
    
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)