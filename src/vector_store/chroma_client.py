# docubot/src/vector_store/chroma_client.py

"""
ChromaDB Vector Store Client for DocuBot
"""

import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
import logging
from datetime import datetime
import uuid

# Fix: Import from correct location
try:
    from ..core.constants import DATABASE_DIR, VECTOR_DB_NAME
except ImportError:
    # Fallback if constants not available
    DATABASE_DIR = Path.home() / ".docubot" / "database"
    VECTOR_DB_NAME = "chroma"

logger = logging.getLogger(__name__)


class ChromaClient:
    """ChromaDB vector store client"""
    
    def __init__(self, persist_directory: Optional[Path] = None, collection_name: str = "documents"):
        """
        Initialize ChromaDB client.
        
        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the collection to use
        """
        if persist_directory is None:
            persist_directory = DATABASE_DIR / VECTOR_DB_NAME
        
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.default_top_k = 5  # Default top_k parameter
        
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.collection = self._get_or_create_collection()
        
        logger.info(f"ChromaDB client initialized: {self.persist_directory}")
    
    def _get_or_create_collection(self) -> chromadb.Collection:
        """
        Get existing collection or create new one.
        
        Returns:
            ChromaDB collection
        """
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=self.collection_name)
            logger.debug(f"Using existing collection: {self.collection_name}")
            return collection
        except ValueError:
            # Create new collection if it doesn't exist
            collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "DocuBot document embeddings"}
            )
            logger.info(f"Created new collection: {self.collection_name}")
            return collection
    
    def add_documents(self,
                     texts: List[str],
                     embeddings: Optional[List[List[float]]] = None,
                     metadatas: Optional[List[Dict[str, Any]]] = None,
                     ids: Optional[List[str]] = None) -> List[str]:
        """
        Add documents to vector store.
        
        Args:
            texts: List of text documents
            embeddings: Optional pre-computed embeddings
            metadatas: Optional metadata for each document
            ids: Optional custom IDs for documents
            
        Returns:
            List of document IDs
        """
        if not texts:
            return []
        
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in range(len(texts))]
        
        if metadatas is None:
            metadatas = [{} for _ in range(len(texts))]
        
        # Ensure metadatas has same length as texts
        if len(metadatas) != len(texts):
            metadatas = [{} for _ in range(len(texts))]
        
        timestamp = datetime.now().isoformat()
        for metadata in metadatas:
            metadata['added_at'] = metadata.get('added_at', timestamp)
        
        try:
            if embeddings:
                # Add with embeddings
                self.collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
            else:
                # Add without embeddings (ChromaDB will compute)
                self.collection.add(
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
            
            logger.info(f"Added {len(texts)} documents to vector store")
            return ids
            
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            raise
    
    def search(self,
              query: str,
              n_results: int = 5,
              where: Optional[Dict[str, Any]] = None,
              where_document: Optional[Dict[str, Any]] = None,
              include: List[str] = None) -> Dict[str, Any]:
        """
        Search for similar documents.
        
        Args:
            query: Search query text
            n_results: Number of results to return
            where: Filter by metadata
            where_document: Filter by document content
            include: What to include in results
            
        Returns:
            Search results
        """
        if include is None:
            include = ["documents", "metadatas", "distances"]
        
        try:
            # Handle empty query
            if not query or not query.strip():
                return {
                    'ids': [],
                    'documents': [],
                    'metadatas': [],
                    'distances': [],
                    'count': 0,
                    'error': 'Empty query'
                }
            
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                where_document=where_document,
                include=include
            )
            
            # Format results safely
            formatted_results = {
                'ids': results.get('ids', [[]])[0] if results.get('ids') else [],
                'documents': results.get('documents', [[]])[0] if results.get('documents') else [],
                'metadatas': results.get('metadatas', [[]])[0] if results.get('metadatas') else [],
                'distances': results.get('distances', [[]])[0] if results.get('distances') else [],
                'count': len(results.get('ids', [[]])[0]) if results.get('ids') else 0
            }
            
            logger.debug(f"Search returned {formatted_results['count']} results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return {
                'ids': [],
                'documents': [],
                'metadatas': [],
                'distances': [],
                'count': 0,
                'error': str(e)
            }
    
    def search_with_embeddings(self,
                              query_embeddings: List[List[float]],
                              n_results: int = 5,
                              where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Search using pre-computed embeddings.
        
        Args:
            query_embeddings: Query embeddings
            n_results: Number of results to return
            where: Filter by metadata
            
        Returns:
            Search results
        """
        try:
            if not query_embeddings:
                return {
                    'ids': [],
                    'documents': [],
                    'metadatas': [],
                    'distances': [],
                    'count': 0,
                    'error': 'No query embeddings provided'
                }
            
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results safely
            formatted_results = {
                'ids': results.get('ids', [[]])[0] if results.get('ids') else [],
                'documents': results.get('documents', [[]])[0] if results.get('documents') else [],
                'metadatas': results.get('metadatas', [[]])[0] if results.get('metadatas') else [],
                'distances': results.get('distances', [[]])[0] if results.get('distances') else [],
                'count': len(results.get('ids', [[]])[0]) if results.get('ids') else 0
            }
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error searching with embeddings: {e}")
            return {
                'ids': [],
                'documents': [],
                'metadatas': [],
                'distances': [],
                'count': 0,
                'error': str(e)
            }
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Get document by ID.
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document data or None
        """
        try:
            results = self.collection.get(
                ids=[doc_id],
                include=["documents", "metadatas"]
            )
            
            if results.get('ids') and len(results['ids']) > 0:
                return {
                    'id': results['ids'][0],
                    'document': results.get('documents', [None])[0],
                    'metadata': results.get('metadatas', [None])[0]
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting document {doc_id}: {e}")
            return None
    
    def update_document(self,
                       doc_id: str,
                       text: Optional[str] = None,
                       metadata: Optional[Dict[str, Any]] = None,
                       embedding: Optional[List[float]] = None) -> bool:
        """
        Update document in vector store.
        
        Args:
            doc_id: Document ID
            text: New text
            metadata: New metadata
            embedding: New embedding
            
        Returns:
            True if successful
        """
        try:
            update_data = {}
            
            if text is not None:
                update_data['documents'] = [text]
            
            if metadata is not None:
                update_data['metadatas'] = [metadata]
            
            if embedding is not None:
                update_data['embeddings'] = [embedding]
            
            if update_data:
                self.collection.update(
                    ids=[doc_id],
                    **update_data
                )
                logger.debug(f"Updated document: {doc_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating document {doc_id}: {e}")
            return False
    
    def delete_documents(self, doc_ids: List[str]) -> bool:
        """
        Delete documents from vector store.
        
        Args:
            doc_ids: List of document IDs to delete
            
        Returns:
            True if successful
        """
        try:
            if not doc_ids:
                return True
            
            self.collection.delete(ids=doc_ids)
            logger.info(f"Deleted {len(doc_ids)} documents from vector store")
            return True
            
        except Exception as e:
            logger.error(f"Error deleting documents: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection.
        
        Returns:
            Collection information
        """
        try:
            count = self.collection.count()
            
            # Get a sample document
            sample = self.collection.get(limit=1)
            
            return {
                'name': self.collection_name,
                'document_count': count,
                'has_documents': count > 0,
                'sample_document': sample.get('documents', [None])[0] if sample.get('documents') else None,
                'persist_directory': str(self.persist_directory),
                'collection_metadata': self.collection.metadata
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {
                'name': self.collection_name,
                'error': str(e),
                'persist_directory': str(self.persist_directory)
            }
    
    def reset_collection(self) -> bool:
        """
        Reset/clear the collection.
        
        Returns:
            True if successful
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            # Recreate collection
            self.collection = self._get_or_create_collection()
            logger.info(f"Collection {self.collection_name} has been reset")
            return True
            
        except Exception as e:
            logger.error(f"Error resetting collection: {e}")
            return False
    
    def list_collections(self) -> List[str]:
        """
        List all collections in the database.
        
        Returns:
            List of collection names
        """
        try:
            collections = self.client.list_collections()
            return [col.name for col in collections]
            
        except Exception as e:
            logger.error(f"Error listing collections: {e}")
            return []
    
    def create_new_collection(self, name: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create a new collection.
        
        Args:
            name: Collection name
            metadata: Collection metadata
            
        Returns:
            True if successful
        """
        try:
            if metadata is None:
                metadata = {}
            
            metadata['created_at'] = datetime.now().isoformat()
            metadata['created_by'] = 'DocuBot'
            
            self.client.create_collection(
                name=name,
                metadata=metadata
            )
            logger.info(f"Created new collection: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating collection {name}: {e}")
            return False
    
    def health_check(self) -> Dict[str, Any]:
        """
        Check if ChromaDB is healthy.
        
        Returns:
            Health check results
        """
        try:
            # Try to get collection info
            info = self.get_collection_info()
            
            # Try a simple query
            test_results = self.search("test", n_results=1)
            
            return {
                'status': 'healthy',
                'collection_name': self.collection_name,
                'document_count': info.get('document_count', 0),
                'has_documents': info.get('has_documents', False),
                'search_working': 'error' not in test_results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    # NEW METHODS ADDED BELOW
    
    def get_embedding_function(self):
        """
        Get embedding function for ChromaDB.
        
        Returns:
            Embedding function instance
        """
        try:
            from sentence_transformers import SentenceTransformer
            
            class EmbeddingFunction:
                def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
                    """
                    Initialize embedding function.
                    
                    Args:
                        model_name: Name of the sentence transformer model
                    """
                    self.model = SentenceTransformer(model_name)
                    self.model_name = model_name
                    logger.info(f"Loaded embedding model: {model_name}")
                
                def __call__(self, texts: List[str]) -> List[List[float]]:
                    """
                    Generate embeddings for texts.
                    
                    Args:
                        texts: List of text strings
                        
                    Returns:
                        List of embedding vectors
                    """
                    if not texts:
                        return []
                    
                    try:
                        # Encode texts to embeddings
                        embeddings = self.model.encode(texts)
                        return embeddings.tolist()
                    except Exception as e:
                        logger.error(f"Error generating embeddings: {e}")
                        return []
                
                def get_model_info(self) -> Dict[str, Any]:
                    """
                    Get information about the embedding model.
                    
                    Returns:
                        Model information dictionary
                    """
                    return {
                        'model_name': self.model_name,
                        'embedding_dimension': self.model.get_sentence_embedding_dimension(),
                        'max_seq_length': self.model.max_seq_length,
                        'device': str(self.model.device)
                    }
            
            return EmbeddingFunction()
            
        except ImportError:
            logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
            raise
        except Exception as e:
            logger.error(f"Failed to create embedding function: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 5, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """
        Perform similarity search with threshold filtering.
        
        Args:
            query: Search query
            k: Number of results to return
            threshold: Minimum similarity threshold (0-1, higher means more similar)
            
        Returns:
            List of (document, similarity_score) tuples
        """
        try:
            # Use default top_k if not specified
            if k is None:
                k = self.default_top_k
            
            # Perform search
            results = self.search(query=query, n_results=k)
            
            if 'error' in results:
                logger.error(f"Search error: {results['error']}")
                return []
            
            # Filter by similarity threshold
            filtered_results = []
            documents = results.get('documents', [])
            distances = results.get('distances', [])
            
            # Convert distances to similarity scores
            # ChromaDB returns cosine distances (0-2), where 0 means identical
            # Convert to similarity score (0-1, where 1 means identical)
            for doc, dist in zip(documents, distances):
                # Convert distance to similarity score
                # Cosine distance = 1 - cosine_similarity
                # So similarity = 1 - distance
                similarity = 1.0 - dist if dist <= 2.0 else 0.0
                
                if similarity >= threshold:
                    filtered_results.append((doc, similarity))
            
            logger.info(f"Similarity search: {len(filtered_results)} results after threshold filtering")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []
    
    def set_top_k(self, k: int):
        """
        Set default top_k parameter for search operations.
        
        Args:
            k: New top_k value
        """
        if k <= 0:
            logger.warning(f"Invalid top_k value: {k}. Must be positive.")
            return
        
        old_value = self.default_top_k
        self.default_top_k = k
        logger.info(f"Top-K changed from {old_value} to: {k}")
    
    def get_similarity_with_embeddings(self, 
                                      query_embedding: List[float], 
                                      doc_embeddings: List[List[float]]) -> List[float]:
        """
        Calculate cosine similarity between query embedding and document embeddings.
        
        Args:
            query_embedding: Query embedding vector
            doc_embeddings: List of document embedding vectors
            
        Returns:
            List of similarity scores
        """
        try:
            import numpy as np
            
            # Convert to numpy arrays
            query_vec = np.array(query_embedding)
            doc_vecs = np.array(doc_embeddings)
            
            # Normalize vectors for cosine similarity
            query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
            doc_norms = doc_vecs / (np.linalg.norm(doc_vecs, axis=1, keepdims=True) + 1e-10)
            
            # Calculate cosine similarities
            similarities = np.dot(doc_norms, query_norm)
            
            return similarities.tolist()
            
        except Exception as e:
            logger.error(f"Error calculating similarities: {e}")
            return []
    
    def batch_similarity_search(self, 
                               queries: List[str], 
                               k: int = 5, 
                               threshold: float = 0.7) -> List[List[Tuple[str, float]]]:
        """
        Perform batch similarity search for multiple queries.
        
        Args:
            queries: List of search queries
            k: Number of results per query
            threshold: Minimum similarity threshold
            
        Returns:
            List of results for each query
        """
        all_results = []
        
        for query in queries:
            results = self.similarity_search(query, k, threshold)
            all_results.append(results)
        
        return all_results


_chroma_instance = None

def get_chroma_client(
    persist_directory: Optional[Path] = None,
    collection_name: str = "documents"
) -> ChromaClient:
    """
    Get or create ChromaClient instance.
    
    Args:
        persist_directory: Directory to persist ChromaDB data
        collection_name: Name of the collection to use
        
    Returns:
        ChromaClient instance
    """
    global _chroma_instance
    
    if _chroma_instance is None:
        _chroma_instance = ChromaClient(persist_directory, collection_name)
    
    return _chroma_instance


if __name__ == "__main__":
    import sys
    import os
    
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("CHROMADB CLIENT TEST")
    print("=" * 60)
    
    try:
        client = ChromaClient()
        
        # Test 1: Health check
        print("\n1. Health check...")
        health = client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Documents: {health.get('document_count', 0)}")
        
        # Test 2: Test new methods
        print("\n2. Testing new methods...")
        
        # Test set_top_k
        print("\n   2.1 Testing set_top_k()...")
        client.set_top_k(10)
        print(f"   ✓ Default top_k set to: {client.default_top_k}")
        
        # Test get_embedding_function (requires sentence-transformers)
        try:
            print("\n   2.2 Testing get_embedding_function()...")
            embedding_func = client.get_embedding_function()
            print(f"   ✓ Embedding function created")
            
            # Test embedding generation
            test_texts = ["Hello world", "Test embedding"]
            embeddings = embedding_func(test_texts)
            print(f"   ✓ Generated embeddings: {len(embeddings)} vectors")
            print(f"   ✓ Each vector dimension: {len(embeddings[0]) if embeddings else 'N/A'}")
            
            # Get model info
            model_info = embedding_func.get_model_info()
            print(f"   ✓ Model: {model_info['model_name']}")
            print(f"   ✓ Dimension: {model_info['embedding_dimension']}")
            
        except ImportError:
            print("   ⚠ Sentence-transformers not installed, skipping embedding test")
            print("   Install with: pip install sentence-transformers")
        except Exception as e:
            print(f"   ✗ Embedding function test failed: {e}")
        
        # Test 3: Add test documents
        test_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Artificial intelligence is transforming the world.",
            "Python is a popular programming language for data science.",
            "Machine learning algorithms learn from data patterns.",
            "Natural language processing enables computers to understand human language."
        ]
        
        test_metadatas = [
            {"source": "test", "type": "example", "length": len(test_texts[0])},
            {"source": "test", "type": "example", "length": len(test_texts[1])},
            {"source": "test", "type": "example", "length": len(test_texts[2])},
            {"source": "test", "type": "example", "length": len(test_texts[3])},
            {"source": "test", "type": "example", "length": len(test_texts[4])}
        ]
        
        print("\n3. Adding test documents...")
        doc_ids = client.add_documents(texts=test_texts, metadatas=test_metadatas)
        print(f"   Added {len(doc_ids)} documents")
        
        # Test 4: Test similarity_search with threshold
        print("\n4. Testing similarity_search with threshold...")
        query = "artificial intelligence"
        threshold_results = client.similarity_search(query, k=3, threshold=0.5)
        print(f"   Query: '{query}'")
        print(f"   Found {len(threshold_results)} results above threshold 0.5")
        
        for i, (doc, similarity) in enumerate(threshold_results):
            print(f"   {i+1}. Similarity: {similarity:.4f}")
            print(f"      Document: {doc[:60]}...")
        
        # Test 5: Test batch similarity search
        print("\n5. Testing batch_similarity_search...")
        queries = ["machine learning", "programming language", "data science"]
        batch_results = client.batch_similarity_search(queries, k=2, threshold=0.4)
        
        for i, (query, results) in enumerate(zip(queries, batch_results)):
            print(f"   Query '{query}': {len(results)} results")
        
        # Test 6: Collection info
        print("\n6. Collection info:")
        info = client.get_collection_info()
        for key, value in info.items():
            if key not in ['sample_document']:  # Skip large values
                print(f"   {key}: {value}")
        
        # Test 7: Clean up
        print(f"\n7. Cleaning up...")
        if doc_ids:
            success = client.delete_documents(doc_ids)
            print(f"   Deleted test documents: {'Success' if success else 'Failed'}")
        
        # Reset top_k to default
        client.set_top_k(5)
        print(f"   Reset top_k to: {client.default_top_k}")
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)