"""
DocuBot Cache Management System
Implements caching for embeddings, document processing, and LLM responses
"""

import os
import json
import pickle
import hashlib
import sqlite3
from pathlib import Path
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
import threading


class CacheManager:
    """Unified cache management system for DocuBot"""
    
    def __init__(self, cache_dir: Optional[Path] = None):
        self.cache_dir = cache_dir or Path.home() / ".docubot" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.embedding_cache = {}
        self.document_cache = {}
        self.llm_cache = {}
        self._lock = threading.RLock()
        
        self.cache_db = self.cache_dir / "cache.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite cache database"""
        with sqlite3.connect(self.cache_db) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS embedding_cache ("
                "key TEXT PRIMARY KEY,"
                "embedding BLOB NOT NULL,"
                "model TEXT NOT NULL,"
                "timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,"
                "access_count INTEGER DEFAULT 0"
                ")"
            )
            
            conn.execute(
                "CREATE TABLE IF NOT EXISTS document_cache ("
                "document_hash TEXT PRIMARY KEY,"
                "metadata TEXT NOT NULL,"
                "chunks TEXT NOT NULL,"
                "processing_time REAL,"
                "timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ")"
            )
            
            conn.execute(
                "CREATE TABLE IF NOT EXISTS llm_cache ("
                "query_hash TEXT PRIMARY KEY,"
                "response TEXT NOT NULL,"
                "model TEXT NOT NULL,"
                "parameters TEXT NOT NULL,"
                "tokens_used INTEGER,"
                "timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
                ")"
            )
            
            conn.execute(
                "CREATE TABLE IF NOT EXISTS cache_stats ("
                "cache_type TEXT PRIMARY KEY,"
                "hits INTEGER DEFAULT 0,"
                "misses INTEGER DEFAULT 0,"
                "size_bytes INTEGER DEFAULT 0"
                ")"
            )
    
    def get_embedding(self, text: str, model: str) -> Optional[List[float]]:
        """Get embedding from cache if available"""
        key = self._generate_key(text, model)
        
        with self._lock:
            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.execute(
                    "SELECT embedding FROM embedding_cache WHERE key = ?",
                    (key,)
                )
                result = cursor.fetchone()
                
                if result:
                    conn.execute(
                        "UPDATE embedding_cache SET access_count = access_count + 1 WHERE key = ?",
                        (key,)
                    )
                    self._update_stats('embedding', hit=True)
                    return pickle.loads(result[0])
                
                self._update_stats('embedding', hit=False)
                return None
    
    def set_embedding(self, text: str, model: str, embedding: List[float]):
        """Store embedding in cache"""
        key = self._generate_key(text, model)
        
        with self._lock:
            with sqlite3.connect(self.cache_db) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO embedding_cache "
                    "(key, embedding, model, timestamp, access_count) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (key, pickle.dumps(embedding), model, datetime.now().isoformat(), 0)
                )
    
    def get_document_processing(self, document_hash: str) -> Optional[Dict[str, Any]]:
        """Get document processing results from cache"""
        with self._lock:
            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.execute(
                    "SELECT metadata, chunks FROM document_cache WHERE document_hash = ?",
                    (document_hash,)
                )
                result = cursor.fetchone()
                
                if result:
                    self._update_stats('document', hit=True)
                    return {
                        'metadata': json.loads(result[0]),
                        'chunks': json.loads(result[1])
                    }
                
                self._update_stats('document', hit=False)
                return None
    
    def set_document_processing(self, document_hash: str, metadata: Dict[str, Any], chunks: List[Dict[str, Any]]):
        """Store document processing results in cache"""
        with self._lock:
            with sqlite3.connect(self.cache_db) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO document_cache "
                    "(document_hash, metadata, chunks, timestamp) "
                    "VALUES (?, ?, ?, ?)",
                    (document_hash, json.dumps(metadata), json.dumps(chunks), datetime.now().isoformat())
                )
    
    def get_llm_response(self, query: str, model: str, parameters: Dict[str, Any]) -> Optional[str]:
        """Get LLM response from cache"""
        query_hash = self._generate_query_hash(query, model, parameters)
        
        with self._lock:
            with sqlite3.connect(self.cache_db) as conn:
                cursor = conn.execute(
                    "SELECT response FROM llm_cache WHERE query_hash = ?",
                    (query_hash,)
                )
                result = cursor.fetchone()
                
                if result:
                    self._update_stats('llm', hit=True)
                    return result[0]
                
                self._update_stats('llm', hit=False)
                return None
    
    def set_llm_response(self, query: str, model: str, parameters: Dict[str, Any], response: str, tokens_used: int = 0):
        """Store LLM response in cache"""
        query_hash = self._generate_query_hash(query, model, parameters)
        
        with self._lock:
            with sqlite3.connect(self.cache_db) as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO llm_cache "
                    "(query_hash, response, model, parameters, tokens_used, timestamp) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (query_hash, response, model, json.dumps(parameters), tokens_used, datetime.now().isoformat())
                )
    
    def _generate_key(self, text: str, model: str) -> str:
        """Generate cache key from text and model"""
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _generate_query_hash(self, query: str, model: str, parameters: Dict[str, Any]) -> str:
        """Generate hash for LLM query caching"""
        content = f"{model}:{query}:{json.dumps(parameters, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def _update_stats(self, cache_type: str, hit: bool):
        """Update cache statistics"""
        with sqlite3.connect(self.cache_db) as conn:
            cursor = conn.execute(
                "SELECT hits, misses FROM cache_stats WHERE cache_type = ?",
                (cache_type,)
            )
            result = cursor.fetchone()
            
            if result:
                hits, misses = result
                if hit:
                    hits += 1
                else:
                    misses += 1
                
                conn.execute(
                    "UPDATE cache_stats SET hits = ?, misses = ? WHERE cache_type = ?",
                    (hits, misses, cache_type)
                )
            else:
                hits = 1 if hit else 0
                misses = 0 if hit else 1
                
                conn.execute(
                    "INSERT INTO cache_stats (cache_type, hits, misses) VALUES (?, ?, ?)",
                    (cache_type, hits, misses)
                )
    
    def get_cache_stats(self) -> Dict[str, Dict[str, int]]:
        """Get cache statistics"""
        stats = {}
        
        with sqlite3.connect(self.cache_db) as conn:
            cursor = conn.execute("SELECT cache_type, hits, misses FROM cache_stats")
            
            for row in cursor.fetchall():
                cache_type, hits, misses = row
                total = hits + misses
                hit_rate = (hits / total * 100) if total > 0 else 0
                
                stats[cache_type] = {
                    'hits': hits,
                    'misses': misses,
                    'total': total,
                    'hit_rate_percent': hit_rate
                }
        
        return stats
    
    def cleanup_old_entries(self, max_age_days: int = 30):
        """Remove old cache entries"""
        cutoff_date = (datetime.now() - timedelta(days=max_age_days)).isoformat()
        
        with self._lock:
            with sqlite3.connect(self.cache_db) as conn:
                conn.execute(
                    "DELETE FROM embedding_cache WHERE timestamp < ?",
                    (cutoff_date,)
                )
                conn.execute(
                    "DELETE FROM document_cache WHERE timestamp < ?",
                    (cutoff_date,)
                )
                conn.execute(
                    "DELETE FROM llm_cache WHERE timestamp < ?",
                    (cutoff_date,)
                )
    
    def clear_cache(self, cache_type: Optional[str] = None):
        """Clear cache entries"""
        with self._lock:
            with sqlite3.connect(self.cache_db) as conn:
                if cache_type == 'embedding' or cache_type is None:
                    conn.execute("DELETE FROM embedding_cache")
                if cache_type == 'document' or cache_type is None:
                    conn.execute("DELETE FROM document_cache")
                if cache_type == 'llm' or cache_type is None:
                    conn.execute("DELETE FROM llm_cache")
                if cache_type is None:
                    conn.execute("DELETE FROM cache_stats")


# Global cache instance
_cache_manager = None

def get_cache_manager() -> CacheManager:
    """Get singleton cache manager instance"""
    global _cache_manager
    if _cache_manager is None:
        _cache_manager = CacheManager()
    return _cache_manager
