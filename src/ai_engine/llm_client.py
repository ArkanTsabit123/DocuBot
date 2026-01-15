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
    
    def __init__(self, model: str = "llama2:7b", host: str = "http://localhost:11434"):
        """
        Initialize LLM client.
        
        Args:
            model: Default model to use
            host: Ollama server host
        """
        self.model = model
        self.host = host
        
        ollama._client.Client(host=host)
        
        logger.info(f"LLM client initialized with model: {model}")
    
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
            'num_predict': 1024,
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
            'num_predict': 1024
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
    client = LLMClient()
    
    print("Performing health check...")
    health = client.health_check()
    print(f"Health: {health['status']}")
    print(f"Available models: {health.get('available_models', 0)}")
    
    if health['success']:
        print("
Testing generation...")
        response = client.generate(
            prompt="Explain artificial intelligence in one sentence.",
            stream=False
        )
        
        if response['success']:
            print(f"Response: {response['response']}")
            print(f"Duration: {response.get('total_duration', 0) / 1e9:.2f}s")
        
        print("
Available models:")
        models = client.list_models()
        for model in models[:3]:
            print(f"  - {model['name']} ({model['size'] / 1e9:.1f}GB)")
