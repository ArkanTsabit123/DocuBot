"""
Hybrid Search Engine for Vector Store
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from .chroma_client import ChromaClient


class HybridSearchEngine:
    """
    Implements hybrid search combining vector similarity and keyword matching
    """
    
    def __init__(self, chroma_client: ChromaClient, embedding_model: SentenceTransformer):
        self.chroma_client = chroma_client
        self.embedding_model = embedding_model
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        similarity_threshold: float = 0.7,
        alpha: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Perform hybrid search combining vector and keyword search
        
        Args:
            query: Search query
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
            alpha: Weight for vector vs keyword search (0-1)
        
        Returns:
            List of search results with scores and metadata
        """
        # Vector search
        vector_results = self.vector_search(query, top_k * 2, similarity_threshold)
        
        # Keyword search (simple implementation)
        keyword_results = self.keyword_search(query, top_k * 2)
        
        # Combine results using hybrid scoring
        combined_results = self.combine_results(
            vector_results, keyword_results, alpha, top_k
        )
        
        return combined_results
    
    def vector_search(
        self,
        query: str,
        top_k: int = 10,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search
        
        Args:
            query: Search query
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
        
        Returns:
            List of vector search results
        """
        query_embedding = self.embedding_model.encode(query)
        
        results = self.chroma_client.search(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        
        processed_results = []
        if results and 'documents' in results:
            for i in range(len(results['documents'][0])):
                score = results['distances'][0][i] if 'distances' in results else 1.0
                
                # Convert distance to similarity score (assuming cosine distance)
                similarity = 1.0 - score if isinstance(score, (int, float)) else 0.0
                
                if similarity >= similarity_threshold:
                    result = {
                        'text': results['documents'][0][i],
                        'similarity': similarity,
                        'metadata': results['metadatas'][0][i] if 'metadatas' in results else {},
                        'id': results['ids'][0][i] if 'ids' in results else f"result_{i}",
                        'search_type': 'vector'
                    }
                    processed_results.append(result)
        
        return processed_results
    
    def keyword_search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Perform keyword-based search
        
        Args:
            query: Search query
            top_k: Number of results to return
        
        Returns:
            List of keyword search results
        """
        # Simple keyword matching implementation
        query_keywords = set(query.lower().split())
        
        # This would typically query a separate index or database
        # For now, return empty list - implementation depends on specific setup
        return []
    
    def combine_results(
        self,
        vector_results: List[Dict[str, Any]],
        keyword_results: List[Dict[str, Any]],
        alpha: float = 0.7,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Combine vector and keyword search results
        
        Args:
            vector_results: Results from vector search
            keyword_results: Results from keyword search
            alpha: Weight for vector vs keyword (0 = all keyword, 1 = all vector)
            top_k: Number of final results
        
        Returns:
            Combined and ranked results
        """
        # Create a combined result dictionary
        combined_dict = {}
        
        # Add vector results
        for result in vector_results:
            result_id = result.get('id')
            if result_id:
                combined_dict[result_id] = {
                    'result': result,
                    'vector_score': result.get('similarity', 0.0),
                    'keyword_score': 0.0,
                    'combined_score': 0.0
                }
        
        # Add keyword results
        for i, result in enumerate(keyword_results):
            result_id = result.get('id', f"keyword_{i}")
            keyword_score = 1.0 / (i + 1)  # Simple ranking
            
            if result_id in combined_dict:
                combined_dict[result_id]['keyword_score'] = keyword_score
            else:
                combined_dict[result_id] = {
                    'result': result,
                    'vector_score': 0.0,
                    'keyword_score': keyword_score,
                    'combined_score': 0.0
                }
        
        # Calculate combined scores
        for result_id, data in combined_dict.items():
            vector_score = data['vector_score']
            keyword_score = data['keyword_score']
            
            combined_score = (alpha * vector_score) + ((1 - alpha) * keyword_score)
            data['combined_score'] = combined_score
        
        # Sort by combined score and return top_k
        sorted_results = sorted(
            combined_dict.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )[:top_k]
        
        # Format final results
        final_results = []
        for data in sorted_results:
            result = data['result'].copy()
            result['combined_score'] = data['combined_score']
            result['search_type'] = 'hybrid'
            final_results.append(result)
        
        return final_results
    
    def semantic_search_with_filters(
        self,
        query: str,
        filters: Dict[str, Any],
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search with metadata filters
        
        Args:
            query: Search query
            filters: Dictionary of metadata filters
            top_k: Number of results to return
        
        Returns:
            Filtered search results
        """
        query_embedding = self.embedding_model.encode(query)
        
        # Convert filters to ChromaDB format
        where_filter = {}
        for key, value in filters.items():
            if isinstance(value, list):
                where_filter[key] = {"$in": value}
            else:
                where_filter[key] = {"$eq": value}
        
        results = self.chroma_client.search(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where_filter if where_filter else None
        )
        
        return self._process_search_results(results)
    
    def _process_search_results(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process raw search results into standardized format
        
        Args:
            results: Raw search results from vector store
        
        Returns:
            Processed search results
        """
        processed = []
        
        if not results or 'documents' not in results:
            return processed
        
        for i in range(len(results['documents'][0])):
            processed_result = {
                'text': results['documents'][0][i],
                'similarity': 1.0 - results['distances'][0][i] if 'distances' in results else 0.0,
                'metadata': results['metadatas'][0][i] if 'metadatas' in results else {},
                'id': results['ids'][0][i] if 'ids' in results else f"result_{i}"
            }
            processed.append(processed_result)
        
        return processed
