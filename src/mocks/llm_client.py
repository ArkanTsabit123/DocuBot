import random
from typing import Dict, Any, List, Optional

class MockLLMClient:
    def __init__(self, config=None):
        self.config = config or {}
        self.models = {
            'llama2:7b': {'context_window': 4096, 'temperature': 0.7},
            'mistral:7b': {'context_window': 8192, 'temperature': 0.7},
            'neural-chat:7b': {'context_window': 4096, 'temperature': 0.5}
        }
        
    def generate(self, query: str, context: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        responses = [
            f"Based on your query '{query}', I found relevant information in the documents.",
            f"The documents suggest that the answer to '{query}' is related to...",
            f"I cannot find specific information about '{query}' in the provided documents.",
            f"According to the documents, {query} involves several key factors...",
            f"This appears to be a complex question. The documents provide insights about {query}."
        ]
        
        return {
            'response': random.choice(responses),
            'model_used': 'mock-llm',
            'tokens_used': random.randint(50, 200),
            'processing_time': random.uniform(0.5, 2.0),
            'sources': [
                {'file': 'document1.pdf', 'relevance': 0.85},
                {'file': 'notes.txt', 'relevance': 0.72}
            ]
        }
    
    def list_models(self) -> List[str]:
        return list(self.models.keys())
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        return self.models.get(model_name, {})