"""
ChromaDB Vector Store Client for DocuBot
"""

import chromadb
from chromadb.config import Settings
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
import uuid

from ..core.constants import DATABASE_DIR, VECTOR_DB_NAME

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
            collection = self.client.get_collection(name=self.collection_name)
            logger.debug(f"Using existing collection: {self.collection_name}")
            return collection
        except ValueError:
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
        
        timestamp = datetime.now().isoformat()
        for metadata in metadatas:
            metadata['added_at'] = metadata.get('added_at', timestamp)
        
        try:
            if embeddings:
                self.collection.add(
                    embeddings=embeddings,
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
            else:
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
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where,
                where_document=where_document,
                include=include
            )
            
            formatted_results = {
                'ids': results['ids'][0] if results['ids'] else [],
                'documents': results['documents'][0] if results['documents'] else [],
                'metadatas': results['metadatas'][0] if results['metadatas'] else [],
                'distances': results['distances'][0] if results['distances'] else [],
                'count': len(results['ids'][0]) if results['ids'] else 0
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
            results = self.collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where,
                include=["documents", "metadatas", "distances"]
            )
            
            formatted_results = {
                'ids': results['ids'][0] if results['ids'] else [],
                'documents': results['documents'][0] if results['documents'] else [],
                'metadatas': results['metadatas'][0] if results['metadatas'] else [],
                'distances': results['distances'][0] if results['distances'] else [],
                'count': len(results['ids'][0]) if results['ids'] else 0
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
            
            if results['ids']:
                return {
                    'id': results['ids'][0],
                    'document': results['documents'][0] if results['documents'] else None,
                    'metadata': results['metadatas'][0] if results['metadatas'] else None
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
                update_data['documents'] = text
            
            if metadata is not None:
                update_data['metadatas'] = metadata
            
            if embedding is not None:
                update_data['embeddings'] = embedding
            
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
            
            sample = self.collection.get(limit=1)
            
            return {
                'name': self.collection_name,
                'document_count': count,
                'has_documents': count > 0,
                'sample_document': sample['documents'][0] if sample['documents'] else None,
                'persist_directory': str(self.persist_directory)
            }
            
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {
                'name': self.collection_name,
                'error': str(e)
            }
    
    def reset_collection(self) -> bool:
        """
        Reset/clear the collection.
        
        Returns:
            True if successful
        """
        try:
            self.client.delete_collection(name=self.collection_name)
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
            self.client.create_collection(
                name=name,
                metadata=metadata or {}
            )
            logger.info(f"Created new collection: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating collection {name}: {e}")
            return False


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
    client = ChromaClient()
    
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
    
    print("Adding test documents...")
    doc_ids = client.add_documents(texts=test_texts, metadatas=test_metadatas)
    print(f"Added {len(doc_ids)} documents")
    
    print("
Searching for 'artificial intelligence'...")
    results = client.search(query="artificial intelligence", n_results=3)
    print(f"Found {results['count']} results")
    
    for i, (doc, dist) in enumerate(zip(results['documents'], results['distances'])):
        print(f"{i+1}. {doc[:50]}... (distance: {dist:.4f})")
    
    print("
Collection info:")
    info = client.get_collection_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
