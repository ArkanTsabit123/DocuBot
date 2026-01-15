"""
Vector Store Index Management
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import json
from .chroma_client import ChromaClient


class IndexManager:
    """
    Manages vector store indices and metadata
    """
    
    def __init__(self, chroma_client: ChromaClient):
        self.chroma_client = chroma_client
        self.index_metadata = self.load_index_metadata()
    
    def load_index_metadata(self) -> Dict[str, Any]:
        """
        Load index metadata
        
        Returns:
            Index metadata dictionary
        """
        metadata_file = Path(self.chroma_client.persist_directory) / "index_metadata.json"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            'collections': {},
            'statistics': {},
            'last_updated': datetime.now().isoformat()
        }
    
    def save_index_metadata(self):
        """
        Save index metadata to file
        """
        metadata_file = Path(self.chroma_client.persist_directory) / "index_metadata.json"
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        
        self.index_metadata['last_updated'] = datetime.now().isoformat()
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.index_metadata, f, indent=2)
    
    def create_collection(self, collection_name: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Create a new collection
        
        Args:
            collection_name: Name of collection
            metadata: Optional collection metadata
        
        Returns:
            Creation status
        """
        try:
            collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata=metadata or {}
            )
            
            # Update metadata
            self.index_metadata['collections'][collection_name] = {
                'name': collection_name,
                'created_at': datetime.now().isoformat(),
                'metadata': metadata or {},
                'document_count': 0,
                'embedding_count': 0
            }
            self.save_index_metadata()
            
            return {
                'success': True,
                'collection_name': collection_name,
                'collection': collection
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'collection_name': collection_name
            }
    
    def delete_collection(self, collection_name: str) -> Dict[str, Any]:
        """
        Delete a collection
        
        Args:
            collection_name: Name of collection to delete
        
        Returns:
            Deletion status
        """
        try:
            self.chroma_client.delete_collection(collection_name)
            
            # Update metadata
            self.index_metadata['collections'].pop(collection_name, None)
            self.save_index_metadata()
            
            return {
                'success': True,
                'message': f'Deleted collection: {collection_name}'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_collection_stats(self, collection_name: str) -> Dict[str, Any]:
        """
        Get statistics for a collection
        
        Args:
            collection_name: Name of collection
        
        Returns:
            Collection statistics
        """
        try:
            collection = self.chroma_client.get_collection(collection_name)
            
            if collection:
                count = collection.count()
                
                stats = {
                    'collection_name': collection_name,
                    'document_count': count,
                    'exists': True,
                    'metadata': collection.metadata
                }
                
                # Update cached metadata
                if collection_name in self.index_metadata['collections']:
                    self.index_metadata['collections'][collection_name].update({
                        'document_count': count,
                        'last_accessed': datetime.now().isoformat()
                    })
                    self.save_index_metadata()
                
                return stats
            else:
                return {
                    'collection_name': collection_name,
                    'exists': False,
                    'document_count': 0
                }
                
        except Exception as e:
            return {
                'collection_name': collection_name,
                'exists': False,
                'error': str(e)
            }
    
    def list_collections(self) -> List[Dict[str, Any]]:
        """
        List all collections
        
        Returns:
            List of collection information
        """
        try:
            collections = self.chroma_client.list_collections()
            
            collection_info = []
            for collection in collections:
                stats = self.get_collection_stats(collection.name)
                collection_info.append(stats)
            
            return collection_info
            
        except Exception as e:
            return [{'error': str(e)}]
    
    def update_collection_metadata(self, collection_name: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update collection metadata
        
        Args:
            collection_name: Name of collection
            metadata: New metadata
        
        Returns:
            Update status
        """
        try:
            collection = self.chroma_client.get_collection(collection_name)
            
            if collection:
                # Update metadata in ChromaDB
                current_metadata = collection.metadata or {}
                current_metadata.update(metadata)
                collection.modify(metadata=current_metadata)
                
                # Update local metadata
                if collection_name in self.index_metadata['collections']:
                    self.index_metadata['collections'][collection_name]['metadata'] = current_metadata
                    self.index_metadata['collections'][collection_name]['updated_at'] = datetime.now().isoformat()
                    self.save_index_metadata()
                
                return {
                    'success': True,
                    'collection_name': collection_name,
                    'metadata': current_metadata
                }
            else:
                return {
                    'success': False,
                    'error': f'Collection not found: {collection_name}'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def rebuild_index(self, collection_name: str) -> Dict[str, Any]:
        """
        Rebuild collection index
        
        Args:
            collection_name: Name of collection
        
        Returns:
            Rebuild status
        """
        # Note: ChromaDB handles indexing automatically
        # This method is a placeholder for future index optimization
        return {
            'success': True,
            'message': f'Index for {collection_name} is maintained automatically by ChromaDB',
            'collection_name': collection_name
        }
    
    def optimize_storage(self) -> Dict[str, Any]:
        """
        Optimize vector store storage
        
        Returns:
            Optimization results
        """
        # ChromaDB handles storage optimization internally
        # This method is a placeholder for future optimizations
        return {
            'success': True,
            'message': 'Storage optimization would be implemented here',
            'timestamp': datetime.now().isoformat()
        }
    
    def get_index_statistics(self) -> Dict[str, Any]:
        """
        Get overall index statistics
        
        Returns:
            Index statistics
        """
        collections = self.list_collections()
        
        total_documents = 0
        total_collections = len(collections)
        
        for collection in collections:
            if isinstance(collection, dict) and 'document_count' in collection:
                total_documents += collection['document_count']
        
        stats = {
            'total_collections': total_collections,
            'total_documents': total_documents,
            'collections': collections,
            'timestamp': datetime.now().isoformat()
        }
        
        # Update metadata
        self.index_metadata['statistics'] = stats
        self.save_index_metadata()
        
        return stats
