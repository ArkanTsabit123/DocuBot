#!/usr/bin/env python3
"""
DocuBot - Utility Modules Implementation
Complete implementation of caching, performance, error handling, and helper utilities
"""

import os
import sys
import json
import time
import hashlib
import pickle
import sqlite3
import io
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import threading
import queue
from abc import ABC, abstractmethod


class DocuBotUtilityFixer:
    def __init__(self, project_dir: str = "DocuBot"):
        self.project_dir = Path(project_dir).absolute()
        
    def implement_all_utilities(self):
        print("Implementing DocuBot Utility Modules...")
        
        implementations = [
            self.implement_cache_manager,
            self.implement_encryption_module,
            self.implement_performance_monitor,
            self.implement_task_queue,
            self.implement_retry_mechanism,
            self.implement_graceful_degradation,
            self.implement_helper_utilities,
            self.implement_diagnostic_tools,
            self.implement_backup_utility,
            self.update_remaining_files
        ]
        
        for i, impl_func in enumerate(implementations, 1):
            print(f"[{i}/{len(implementations)}] {impl_func.__name__}")
            try:
                impl_func()
                print("  Success")
            except Exception as e:
                print(f"  Failed: {e}")
        
        print("Utility modules implementation completed")
        
    def implement_cache_manager(self):
        cache_file = self.project_dir / "src" / "storage" / "cache_manager.py"
        
        cache_content = '''"""
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
'''
        
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(cache_content)
        
    def implement_encryption_module(self):
        encryption_file = self.project_dir / "src" / "storage" / "encryption.py"
        
        encryption_content = '''"""
DocuBot Encryption Module
Secure encryption for sensitive data storage
"""

import os
import base64
import json
import hashlib
from pathlib import Path
from typing import Optional, Union
from datetime import datetime
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class EncryptionManager:
    """Manage encryption and decryption of sensitive data"""
    
    def __init__(self, key_file: Optional[Path] = None):
        self.key_file = key_file or Path.home() / ".docubot" / "secret.key"
        self.key_file.parent.mkdir(parents=True, exist_ok=True)
        
        self._fernet = None
        self._load_or_generate_key()
    
    def _load_or_generate_key(self):
        """Load existing key or generate new one"""
        if self.key_file.exists() and self.key_file.stat().st_size > 0:
            with open(self.key_file, 'rb') as f:
                key = f.read()
        else:
            key = Fernet.generate_key()
            with open(self.key_file, 'wb') as f:
                f.write(key)
        
        self._fernet = Fernet(key)
    
    def encrypt(self, data: Union[str, bytes]) -> bytes:
        """Encrypt data"""
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        return self._fernet.encrypt(data)
    
    def decrypt(self, encrypted_data: bytes) -> bytes:
        """Decrypt data"""
        return self._fernet.decrypt(encrypted_data)
    
    def encrypt_string(self, text: str) -> str:
        """Encrypt string and return base64 encoded result"""
        encrypted = self.encrypt(text.encode('utf-8'))
        return base64.b64encode(encrypted).decode('ascii')
    
    def decrypt_string(self, encrypted_text: str) -> str:
        """Decrypt base64 encoded string"""
        encrypted_data = base64.b64decode(encrypted_text.encode('ascii'))
        decrypted = self.decrypt(encrypted_data)
        return decrypted.decode('utf-8')
    
    def encrypt_file(self, input_file: Path, output_file: Optional[Path] = None):
        """Encrypt file contents"""
        if output_file is None:
            output_file = input_file.with_suffix(input_file.suffix + '.enc')
        
        with open(input_file, 'rb') as f:
            data = f.read()
        
        encrypted = self.encrypt(data)
        
        with open(output_file, 'wb') as f:
            f.write(encrypted)
    
    def decrypt_file(self, input_file: Path, output_file: Optional[Path] = None):
        """Decrypt file contents"""
        if output_file is None:
            if input_file.suffix == '.enc':
                output_file = input_file.with_suffix('')
            else:
                output_file = input_file.with_suffix(input_file.suffix + '.dec')
        
        with open(input_file, 'rb') as f:
            encrypted = f.read()
        
        decrypted = self.decrypt(encrypted)
        
        with open(output_file, 'wb') as f:
            f.write(decrypted)
    
    def derive_key_from_password(self, password: str, salt: Optional[bytes] = None) -> bytes:
        """Derive encryption key from password"""
        if salt is None:
            salt = os.urandom(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key
    
    def rotate_key(self, new_key_file: Optional[Path] = None):
        """Rotate to new encryption key"""
        old_key = self._fernet._signing_key + self._fernet._encryption_key
        
        new_key = Fernet.generate_key()
        new_fernet = Fernet(new_key)
        
        self._fernet = new_fernet
        
        key_file = new_key_file or self.key_file
        with open(key_file, 'wb') as f:
            f.write(new_key)
        
        return old_key


class SensitiveDataStore:
    """Store and retrieve sensitive data with encryption"""
    
    def __init__(self, encryption_manager: Optional[EncryptionManager] = None):
        self.encryption = encryption_manager or EncryptionManager()
        self.data_dir = Path.home() / ".docubot" / "secure_data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def store(self, key: str, data: Union[str, dict, list], metadata: Optional[dict] = None):
        """Store sensitive data"""
        if isinstance(data, (dict, list)):
            data_str = json.dumps(data)
        else:
            data_str = str(data)
        
        encrypted = self.encryption.encrypt_string(data_str)
        
        record = {
            'data': encrypted,
            'timestamp': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        
        record_file = self.data_dir / f"{hashlib.sha256(key.encode()).hexdigest()}.json"
        
        with open(record_file, 'w', encoding='utf-8') as f:
            json.dump(record, f, indent=2)
    
    def retrieve(self, key: str) -> Optional[Union[str, dict, list]]:
        """Retrieve sensitive data"""
        record_file = self.data_dir / f"{hashlib.sha256(key.encode()).hexdigest()}.json"
        
        if not record_file.exists():
            return None
        
        with open(record_file, 'r', encoding='utf-8') as f:
            record = json.load(f)
        
        decrypted = self.encryption.decrypt_string(record['data'])
        
        try:
            return json.loads(decrypted)
        except json.JSONDecodeError:
            return decrypted
    
    def delete(self, key: str):
        """Delete sensitive data"""
        record_file = self.data_dir / f"{hashlib.sha256(key.encode()).hexdigest()}.json"
        
        if record_file.exists():
            record_file.unlink()


# Global encryption instance
_encryption_manager = None

def get_encryption_manager() -> EncryptionManager:
    """Get singleton encryption manager instance"""
    global _encryption_manager
    if _encryption_manager is None:
        _encryption_manager = EncryptionManager()
    return _encryption_manager


def get_sensitive_data_store() -> SensitiveDataStore:
    """Get singleton sensitive data store instance"""
    return SensitiveDataStore(get_encryption_manager())
'''
        
        encryption_file.parent.mkdir(parents=True, exist_ok=True)
        encryption_file.write_text(encryption_content)
        
    def implement_performance_monitor(self):
        monitor_file = self.project_dir / "src" / "utilities" / "monitor.py"
        
        monitor_content = '''"""
DocuBot Performance Monitoring System
Monitor memory usage, execution times, and system resources
"""

import os
import time
import psutil
import threading
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import statistics


@dataclass
class PerformanceMetric:
    """Single performance metric"""
    timestamp: datetime
    metric_type: str
    value: float
    unit: str
    tags: Dict[str, str] = field(default_factory=dict)


class PerformanceMonitor:
    """Monitor system and application performance"""
    
    def __init__(self, monitor_dir: Optional[Path] = None):
        self.monitor_dir = monitor_dir or Path.home() / ".docubot" / "monitoring"
        self.monitor_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics: List[PerformanceMetric] = []
        self._max_metrics = 10000
        self._lock = threading.RLock()
        
        self.metric_file = self.monitor_dir / "metrics.json"
        self._load_metrics()
        
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
    
    def _load_metrics(self):
        """Load saved metrics from disk"""
        if self.metric_file.exists():
            try:
                with open(self.metric_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.metrics = [
                    PerformanceMetric(
                        timestamp=datetime.fromisoformat(m['timestamp']),
                        metric_type=m['metric_type'],
                        value=m['value'],
                        unit=m['unit'],
                        tags=m.get('tags', {})
                    )
                    for m in data
                ]
            except Exception as e:
                print(f"Warning: Could not load metrics: {e}")
                self.metrics = []
    
    def _save_metrics(self):
        """Save metrics to disk"""
        with self._lock:
            data = [
                {
                    'timestamp': m.timestamp.isoformat(),
                    'metric_type': m.metric_type,
                    'value': m.value,
                    'unit': m.unit,
                    'tags': m.tags
                }
                for m in self.metrics[-1000:]  # Save only recent metrics
            ]
            
            try:
                with open(self.metric_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                print(f"Warning: Could not save metrics: {e}")
    
    def record_metric(self, metric_type: str, value: float, unit: str = "", tags: Optional[Dict[str, str]] = None):
        """Record a performance metric"""
        with self._lock:
            metric = PerformanceMetric(
                timestamp=datetime.now(),
                metric_type=metric_type,
                value=value,
                unit=unit,
                tags=tags or {}
            )
            
            self.metrics.append(metric)
            
            if len(self.metrics) > self._max_metrics:
                self.metrics = self.metrics[-self._max_metrics:]
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        
        return {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': process.memory_percent(),
            'available_mb': psutil.virtual_memory().available / 1024 / 1024,
            'total_mb': psutil.virtual_memory().total / 1024 / 1024,
            'used_percent': psutil.virtual_memory().percent
        }
    
    def get_cpu_usage(self) -> Dict[str, float]:
        """Get CPU usage statistics"""
        process = psutil.Process(os.getpid())
        
        return {
            'process_percent': process.cpu_percent(interval=0.1),
            'system_percent': psutil.cpu_percent(interval=0.1),
            'system_load': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0
        }
    
    def get_disk_usage(self) -> Dict[str, float]:
        """Get disk usage statistics"""
        data_dir = Path.home() / ".docubot"
        
        if data_dir.exists():
            usage = psutil.disk_usage(str(data_dir))
            return {
                'total_gb': usage.total / 1024 / 1024 / 1024,
                'used_gb': usage.used / 1024 / 1024 / 1024,
                'free_gb': usage.free / 1024 / 1024 / 1024,
                'percent': usage.percent
            }
        
        return {}
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get all system metrics"""
        return {
            'memory': self.get_memory_usage(),
            'cpu': self.get_cpu_usage(),
            'disk': self.get_disk_usage(),
            'timestamp': datetime.now().isoformat()
        }
    
    def start_monitoring(self, interval_seconds: int = 60):
        """Start background monitoring thread"""
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            return
        
        self._stop_monitoring.clear()
        
        def monitor_loop():
            while not self._stop_monitoring.wait(interval_seconds):
                try:
                    system_metrics = self.get_system_metrics()
                    
                    self.record_metric(
                        'memory_usage',
                        system_metrics['memory']['rss_mb'],
                        'MB',
                        {'type': 'process'}
                    )
                    
                    self.record_metric(
                        'cpu_usage',
                        system_metrics['cpu']['process_percent'],
                        'percent',
                        {'type': 'process'}
                    )
                    
                    self.record_metric(
                        'disk_usage',
                        system_metrics['disk'].get('percent', 0),
                        'percent',
                        {'type': 'system'}
                    )
                    
                    self._save_metrics()
                    
                except Exception as e:
                    print(f"Monitoring error: {e}")
        
        self._monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        if self._monitoring_thread:
            self._stop_monitoring.set()
            self._monitoring_thread.join(timeout=5)
            self._monitoring_thread = None
    
    def get_metric_summary(self, metric_type: str, hours: int = 24) -> Dict[str, Any]:
        """Get summary statistics for a metric type"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        with self._lock:
            recent_metrics = [
                m for m in self.metrics
                if m.metric_type == metric_type and m.timestamp > cutoff
            ]
            
            if not recent_metrics:
                return {}
            
            values = [m.value for m in recent_metrics]
            
            return {
                'count': len(values),
                'min': min(values),
                'max': max(values),
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'std_dev': statistics.stdev(values) if len(values) > 1 else 0,
                'latest': values[-1],
                'time_range_hours': hours
            }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'system': self.get_system_metrics(),
            'metrics_summary': {}
        }
        
        metric_types = set(m.metric_type for m in self.metrics)
        
        for metric_type in metric_types:
            if metric_type:
                summary = self.get_metric_summary(metric_type, hours=1)
                if summary:
                    report['metrics_summary'][metric_type] = summary
        
        return report
    
    def clear_metrics(self):
        """Clear all stored metrics"""
        with self._lock:
            self.metrics = []
            
            if self.metric_file.exists():
                self.metric_file.unlink()


class ExecutionTimer:
    """Context manager for timing code execution"""
    
    def __init__(self, name: str, monitor: Optional[PerformanceMonitor] = None, tags: Optional[Dict[str, str]] = None):
        self.name = name
        self.monitor = monitor
        self.tags = tags or {}
        self.start_time = None
        self.elapsed = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = time.time() - self.start_time
        
        if self.monitor:
            self.monitor.record_metric(
                f"execution_time_{self.name}",
                self.elapsed,
                "seconds",
                {**self.tags, 'operation': self.name}
            )
        
        return False


# Global monitor instance
_performance_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get singleton performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
    return _performance_monitor


def time_execution(name: str, tags: Optional[Dict[str, str]] = None):
    """Decorator for timing function execution"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with ExecutionTimer(name, get_performance_monitor(), tags):
                return func(*args, **kwargs)
        return wrapper
    return decorator
'''
        
        monitor_file.parent.mkdir(parents=True, exist_ok=True)
        monitor_file.write_text(monitor_content)
        
    def implement_task_queue(self):
        task_queue_file = self.project_dir / "src" / "utilities" / "task_queue.py"
        
        task_queue_content = '''"""
DocuBot Background Task Queue System
Asynchronous task processing with priority and retry logic
"""

import queue
import threading
import time
import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import traceback


class TaskPriority(Enum):
    """Task priority levels"""
    HIGH = 0
    NORMAL = 1
    LOW = 2


class TaskStatus(Enum):
    """Task status states"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class Task:
    """Background task definition"""
    id: str
    function: Callable
    args: tuple = field(default_factory=tuple)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.NORMAL
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: Optional[float] = None
    
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    
    callback: Optional[Callable] = None
    callback_args: tuple = field(default_factory=tuple)
    callback_kwargs: Dict[str, Any] = field(default_factory=dict)
    
    metadata: Dict[str, Any] = field(default_factory=dict)


class TaskQueue:
    """Background task processing queue"""
    
    def __init__(self, max_workers: int = 4, persist_dir: Optional[Path] = None):
        self.max_workers = max_workers
        self.persist_dir = persist_dir or Path.home() / ".docubot" / "task_queue"
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.pending_queue = queue.PriorityQueue()
        self.running_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, Task] = {}
        self.failed_tasks: Dict[str, Task] = {}
        
        self.workers: List[threading.Thread] = []
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        
        self._load_persisted_tasks()
        self._start_workers()
    
    def _start_workers(self):
        """Start worker threads"""
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"TaskWorker-{i}",
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
    
    def _worker_loop(self):
        """Worker thread main loop"""
        while not self._stop_event.is_set():
            try:
                priority, task = self.pending_queue.get(timeout=1.0)
                
                with self._lock:
                    task.status = TaskStatus.RUNNING
                    task.started_at = datetime.now()
                    self.running_tasks[task.id] = task
                
                try:
                    result = self._execute_task(task)
                    
                    with self._lock:
                        task.status = TaskStatus.COMPLETED
                        task.completed_at = datetime.now()
                        task.result = result
                        
                        self.completed_tasks[task.id] = task
                        if task.id in self.running_tasks:
                            del self.running_tasks[task.id]
                    
                    if task.callback:
                        try:
                            task.callback(
                                task.result,
                                *task.callback_args,
                                **task.callback_kwargs
                            )
                        except Exception as e:
                            print(f"Task callback error: {e}")
                    
                    self._persist_task(task)
                
                except Exception as e:
                    self._handle_task_failure(task, e)
                
                finally:
                    self.pending_queue.task_done()
            
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker error: {e}")
    
    def _execute_task(self, task: Task) -> Any:
        """Execute a task with timeout handling"""
        if task.timeout:
            def task_wrapper():
                return task.function(*task.args, **task.kwargs)
            
            future = threading.Thread(target=task_wrapper)
            future.start()
            future.join(timeout=task.timeout)
            
            if future.is_alive():
                raise TimeoutError(f"Task timed out after {task.timeout} seconds")
            
            return task.result
        else:
            return task.function(*task.args, **task.kwargs)
    
    def _handle_task_failure(self, task: Task, error: Exception):
        """Handle task execution failure"""
        with self._lock:
            task.retry_count += 1
            task.error = str(error)
            
            if task.retry_count <= task.max_retries:
                task.status = TaskStatus.RETRYING
                print(f"Task {task.id} failed, retrying ({task.retry_count}/{task.max_retries}): {error}")
                
                time.sleep(task.retry_delay)
                self.add_task(task)
            else:
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.now()
                
                self.failed_tasks[task.id] = task
                if task.id in self.running_tasks:
                    del self.running_tasks[task.id]
                
                print(f"Task {task.id} failed after {task.max_retries} retries: {error}")
                
                error_details = {
                    'error': str(error),
                    'traceback': traceback.format_exc(),
                    'retry_count': task.retry_count,
                    'task_id': task.id
                }
                task.metadata['error_details'] = error_details
            
            self._persist_task(task)
    
    def add_task(
        self,
        function: Callable,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        task_id: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: Optional[float] = None,
        callback: Optional[Callable] = None,
        callback_args: tuple = (),
        callback_kwargs: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Add a task to the queue"""
        if kwargs is None:
            kwargs = {}
        if callback_kwargs is None:
            callback_kwargs = {}
        if metadata is None:
            metadata = {}
        
        task_id = task_id or f"task_{int(time.time())}_{hash(function)}"
        
        task = Task(
            id=task_id,
            function=function,
            args=args,
            kwargs=kwargs,
            priority=priority,
            max_retries=max_retries,
            retry_delay=retry_delay,
            timeout=timeout,
            callback=callback,
            callback_args=callback_args,
            callback_kwargs=callback_kwargs,
            metadata=metadata
        )
        
        priority_value = priority.value
        
        with self._lock:
            self.pending_queue.put((priority_value, task))
            self._persist_task(task)
        
        return task_id
    
    def wait_for_task(self, task_id: str, timeout: Optional[float] = None) -> Optional[Any]:
        """Wait for a specific task to complete"""
        start_time = time.time()
        
        while True:
            with self._lock:
                if task_id in self.completed_tasks:
                    return self.completed_tasks[task_id].result
                
                if task_id in self.failed_tasks:
                    raise RuntimeError(f"Task {task_id} failed: {self.failed_tasks[task_id].error}")
            
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Timeout waiting for task {task_id}")
            
            time.sleep(0.1)
    
    def wait_for_all(self, timeout: Optional[float] = None):
        """Wait for all pending tasks to complete"""
        self.pending_queue.join()
        
        start_time = time.time()
        while self.running_tasks:
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError("Timeout waiting for all tasks")
            time.sleep(0.1)
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get status of a task"""
        with self._lock:
            if task_id in self.running_tasks:
                return self.running_tasks[task_id].status
            elif task_id in self.completed_tasks:
                return self.completed_tasks[task_id].status
            elif task_id in self.failed_tasks:
                return self.failed_tasks[task_id].status
        
        return None
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        with self._lock:
            return {
                'pending': self.pending_queue.qsize(),
                'running': len(self.running_tasks),
                'completed': len(self.completed_tasks),
                'failed': len(self.failed_tasks),
                'workers': len([w for w in self.workers if w.is_alive()]),
                'timestamp': datetime.now().isoformat()
            }
    
    def _persist_task(self, task: Task):
        """Persist task to disk"""
        try:
            task_file = self.persist_dir / f"{task.id}.json"
            
            task_dict = {
                'id': task.id,
                'status': task.status.value,
                'created_at': task.created_at.isoformat(),
                'started_at': task.started_at.isoformat() if task.started_at else None,
                'completed_at': task.completed_at.isoformat() if task.completed_at else None,
                'priority': task.priority.value,
                'max_retries': task.max_retries,
                'retry_count': task.retry_count,
                'error': task.error,
                'metadata': task.metadata
            }
            
            with open(task_file, 'w', encoding='utf-8') as f:
                json.dump(task_dict, f, indent=2)
        
        except Exception as e:
            print(f"Error persisting task {task.id}: {e}")
    
    def _load_persisted_tasks(self):
        """Load persisted tasks from disk"""
        try:
            for task_file in self.persist_dir.glob("*.json"):
                try:
                    with open(task_file, 'r', encoding='utf-8') as f:
                        task_dict = json.load(f)
                    
                    created_at = datetime.fromisoformat(task_dict['created_at'])
                    
                    if datetime.now() - created_at > timedelta(days=7):
                        task_file.unlink()
                        continue
                
                except Exception as e:
                    print(f"Error loading task file {task_file}: {e}")
                    continue
        
        except Exception as e:
            print(f"Error loading persisted tasks: {e}")
    
    def cleanup_old_tasks(self, max_age_days: int = 7):
        """Clean up old task files"""
        cutoff = datetime.now() - timedelta(days=max_age_days)
        
        for task_file in self.persist_dir.glob("*.json"):
            try:
                with open(task_file, 'r', encoding='utf-8') as f:
                    task_dict = json.load(f)
                
                created_at = datetime.fromisoformat(task_dict['created_at'])
                
                if created_at < cutoff:
                    task_file.unlink()
            
            except Exception:
                try:
                    task_file.unlink()
                except:
                    pass
    
    def shutdown(self, wait: bool = True, timeout: Optional[float] = None):
        """Shutdown the task queue"""
        self._stop_event.set()
        
        if wait:
            start_time = time.time()
            
            for worker in self.workers:
                if worker.is_alive():
                    worker.join(timeout=max(0, timeout - (time.time() - start_time)) if timeout else None)
            
            self.workers.clear()


# Global task queue instance
_task_queue = None

def get_task_queue() -> TaskQueue:
    """Get singleton task queue instance"""
    global _task_queue
    if _task_queue is None:
        _task_queue = TaskQueue()
    return _task_queue


def background_task(
    priority: TaskPriority = TaskPriority.NORMAL,
    task_id: Optional[str] = None,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    timeout: Optional[float] = None
):
    """Decorator for running functions as background tasks"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            queue = get_task_queue()
            return queue.add_task(
                function=func,
                args=args,
                kwargs=kwargs,
                priority=priority,
                task_id=task_id,
                max_retries=max_retries,
                retry_delay=retry_delay,
                timeout=timeout
            )
        return wrapper
    return decorator
'''
        
        task_queue_file.parent.mkdir(parents=True, exist_ok=True)
        task_queue_file.write_text(task_queue_content)
        
    def implement_retry_mechanism(self):
        retry_file = self.project_dir / "src" / "utilities" / "retry.py"
        
        retry_content = '''"""
DocuBot Retry Mechanism
Configurable retry logic for unreliable operations
"""

import time
import random
from typing import Callable, Any, Optional, Dict, List
from dataclasses import dataclass, field
from datetime import datetime
import logging


@dataclass
class RetryConfig:
    """Retry configuration"""
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_backoff: bool = True
    jitter: bool = True
    retry_exceptions: tuple = (Exception,)
    
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for retry attempt"""
        if not self.exponential_backoff:
            delay = self.base_delay
        else:
            delay = self.base_delay * (2 ** (attempt - 1))
        
        delay = min(delay, self.max_delay)
        
        if self.jitter:
            delay = delay * (0.5 + random.random() * 0.5)
        
        return delay


class RetryManager:
    """Manage retry logic for operations"""
    
    def __init__(self, config: Optional[RetryConfig] = None):
        self.config = config or RetryConfig()
        self.logger = logging.getLogger(__name__)
    
    def execute(
        self,
        func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with retry logic"""
        last_exception = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                return func(*args, **kwargs)
            
            except self.config.retry_exceptions as e:
                last_exception = e
                
                if attempt == self.config.max_attempts:
                    self.logger.error(
                        f"Operation failed after {attempt} attempts: {e}",
                        extra={
                            'function': func.__name__,
                            'attempts': attempt,
                            'exception': str(e)
                        }
                    )
                    raise
                
                delay = self.config.calculate_delay(attempt)
                
                self.logger.warning(
                    f"Operation failed, retrying in {delay:.2f}s (attempt {attempt}/{self.config.max_attempts}): {e}",
                    extra={
                        'function': func.__name__,
                        'attempt': attempt,
                        'max_attempts': self.config.max_attempts,
                        'delay_seconds': delay,
                        'exception': str(e)
                    }
                )
                
                time.sleep(delay)
        
        raise last_exception
    
    def execute_with_callback(
        self,
        func: Callable,
        success_callback: Optional[Callable] = None,
        failure_callback: Optional[Callable] = None,
        *args,
        **kwargs
    ) -> Any:
        """Execute function with retry and callbacks"""
        last_exception = None
        last_result = None
        
        for attempt in range(1, self.config.max_attempts + 1):
            try:
                result = func(*args, **kwargs)
                last_result = result
                
                if success_callback:
                    success_callback(result, attempt)
                
                return result
            
            except self.config.retry_exceptions as e:
                last_exception = e
                
                if failure_callback:
                    failure_callback(e, attempt)
                
                if attempt == self.config.max_attempts:
                    self.logger.error(
                        f"Operation failed after {attempt} attempts: {e}",
                        extra={
                            'function': func.__name__,
                            'attempts': attempt,
                            'exception': str(e)
                        }
                    )
                    raise
                
                delay = self.config.calculate_delay(attempt)
                time.sleep(delay)
        
        raise last_exception


def retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_backoff: bool = True,
    jitter: bool = True,
    retry_exceptions: tuple = (Exception,)
):
    """Decorator for retry logic"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            config = RetryConfig(
                max_attempts=max_attempts,
                base_delay=base_delay,
                max_delay=max_delay,
                exponential_backoff=exponential_backoff,
                jitter=jitter,
                retry_exceptions=retry_exceptions
            )
            
            manager = RetryManager(config)
            return manager.execute(func, *args, **kwargs)
        return wrapper
    return decorator


def retry_with_callback(
    success_callback: Optional[Callable] = None,
    failure_callback: Optional[Callable] = None,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_backoff: bool = True,
    jitter: bool = True,
    retry_exceptions: tuple = (Exception,)
):
    """Decorator for retry logic with callbacks"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            config = RetryConfig(
                max_attempts=max_attempts,
                base_delay=base_delay,
                max_delay=max_delay,
                exponential_backoff=exponential_backoff,
                jitter=jitter,
                retry_exceptions=retry_exceptions
            )
            
            manager = RetryManager(config)
            return manager.execute_with_callback(
                func,
                success_callback,
                failure_callback,
                *args,
                **kwargs
            )
        return wrapper
    return decorator


class CircuitBreaker:
    """Circuit breaker pattern for failing operations"""
    
    def __init__(
        self,
        failure_threshold: int = 5,
        reset_timeout: float = 60.0,
        expected_exceptions: tuple = (Exception,)
    ):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.expected_exceptions = expected_exceptions
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        
        self.logger = logging.getLogger(__name__)
    
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker"""
        if self.state == "OPEN":
            if self._should_try_reset():
                self.state = "HALF_OPEN"
                self.logger.info("Circuit breaker transitioning to HALF_OPEN")
            else:
                raise CircuitBreakerOpenError(
                    f"Circuit breaker is OPEN. Last failure: {self.last_failure_time}"
                )
        
        try:
            result = func(*args, **kwargs)
            
            if self.state == "HALF_OPEN":
                self._reset()
                self.logger.info("Circuit breaker reset to CLOSED after successful execution")
            
            return result
        
        except self.expected_exceptions as e:
            self._record_failure()
            raise
    
    def _record_failure(self):
        """Record a failure"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == "HALF_OPEN":
            self.state = "OPEN"
            self.logger.warning("Circuit breaker reopened after failure in HALF_OPEN state")
        
        elif self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.logger.error(
                f"Circuit breaker opened after {self.failure_count} failures",
                extra={
                    'failure_count': self.failure_count,
                    'threshold': self.failure_threshold
                }
            )
    
    def _should_try_reset(self) -> bool:
        """Check if circuit breaker should try to reset"""
        if not self.last_failure_time:
            return True
        
        time_since_failure = (datetime.now() - self.last_failure_time).total_seconds()
        return time_since_failure >= self.reset_timeout
    
    def _reset(self):
        """Reset circuit breaker"""
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"
    
    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
            'failure_threshold': self.failure_threshold,
            'reset_timeout': self.reset_timeout
        }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass


def with_circuit_breaker(
    failure_threshold: int = 5,
    reset_timeout: float = 60.0,
    expected_exceptions: tuple = (Exception,)
):
    """Decorator for circuit breaker pattern"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            breaker = CircuitBreaker(
                failure_threshold=failure_threshold,
                reset_timeout=reset_timeout,
                expected_exceptions=expected_exceptions
            )
            return breaker.execute(func, *args, **kwargs)
        return wrapper
    return decorator
'''
        
        retry_file.parent.mkdir(parents=True, exist_ok=True)
        retry_file.write_text(retry_content)
        
    def implement_graceful_degradation(self):
        exceptions_file = self.project_dir / "src" / "core" / "exceptions.py"
        if exceptions_file.exists():
            content = exceptions_file.read_text()
            
            if "class GracefulDegradation" not in content:
                graceful_content = '''

class GracefulDegradation:
    """Implement graceful degradation for failing components"""
    
    @staticmethod
    def fallback_document_processing(file_path, error):
        """Fallback for document processing failures"""
        from pathlib import Path
        return {
            'success': False,
            'text': f"Document processing failed: {error}",
            'metadata': {
                'file_name': Path(file_path).name,
                'error': str(error),
                'degraded_mode': True
            },
            'chunks': []
        }
    
    @staticmethod
    def fallback_llm_response(query, error):
        """Fallback for LLM failures"""
        return {
            'answer': f"I apologize, but I'm unable to process your query at the moment. Error: {error}",
            'sources': [],
            'degraded_mode': True,
            'error': str(error)
        }
    
    @staticmethod
    def fallback_embedding(text, error):
        """Fallback for embedding failures"""
        # Return a simple zero vector as fallback
        return [0.0] * 384
    
    @staticmethod
    def fallback_database_operation(operation, error):
        """Fallback for database failures"""
        return {
            'success': False,
            'operation': operation,
            'error': str(error),
            'degraded_mode': True,
            'data': None
        }


def get_logger(name):
    """Get logger for module"""
    import logging
    return logging.getLogger(name)


def graceful_execute(func, fallback_func, *args, **kwargs):
    """Execute function with graceful degradation fallback"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger = get_logger(__name__)
        logger.warning(f"Primary function failed, using fallback: {e}")
        return fallback_func(*args, e, **kwargs)
'''
                
                lines = content.split('\n')
                if lines and lines[-1].strip() == '':
                    lines = lines[:-1]
                
                lines.append(graceful_content)
                exceptions_file.write_text('\n'.join(lines))
        else:
            # Create a new exceptions file if it doesn't exist
            exceptions_file.parent.mkdir(parents=True, exist_ok=True)
            exceptions_file.write_text('''"""
DocuBot Exceptions and Error Handling
"""

import logging


class GracefulDegradation:
    """Implement graceful degradation for failing components"""
    
    @staticmethod
    def fallback_document_processing(file_path, error):
        """Fallback for document processing failures"""
        from pathlib import Path
        return {
            'success': False,
            'text': f"Document processing failed: {error}",
            'metadata': {
                'file_name': Path(file_path).name,
                'error': str(error),
                'degraded_mode': True
            },
            'chunks': []
        }
    
    @staticmethod
    def fallback_llm_response(query, error):
        """Fallback for LLM failures"""
        return {
            'answer': f"I apologize, but I'm unable to process your query at the moment. Error: {error}",
            'sources': [],
            'degraded_mode': True,
            'error': str(error)
        }
    
    @staticmethod
    def fallback_embedding(text, error):
        """Fallback for embedding failures"""
        # Return a simple zero vector as fallback
        return [0.0] * 384
    
    @staticmethod
    def fallback_database_operation(operation, error):
        """Fallback for database failures"""
        return {
            'success': False,
            'operation': operation,
            'error': str(error),
            'degraded_mode': True,
            'data': None
        }


def get_logger(name):
    """Get logger for module"""
    return logging.getLogger(name)


def graceful_execute(func, fallback_func, *args, **kwargs):
    """Execute function with graceful degradation fallback"""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger = get_logger(__name__)
        logger.warning(f"Primary function failed, using fallback: {e}")
        return fallback_func(*args, e, **kwargs)
''')
        
    def implement_helper_utilities(self):
        helpers_file = self.project_dir / "src" / "utilities" / "helpers.py"
        
        helpers_content = '''"""
DocuBot Helper Utilities
Miscellaneous utility functions for common operations
"""

import os
import sys
import json
import hashlib
import uuid
import random
import string
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
import inspect
import threading
from contextlib import contextmanager
import time


def generate_id(prefix: str = "", length: int = 16) -> str:
    """Generate a unique ID"""
    random_part = ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    
    if prefix:
        return f"{prefix}_{timestamp}_{random_part}"
    return f"{timestamp}_{random_part}"


def safe_json_loads(json_str: str, default: Any = None) -> Any:
    """Safely parse JSON string"""
    try:
        return json.loads(json_str)
    except (json.JSONDecodeError, TypeError):
        return default


def safe_json_dumps(data: Any, default: Optional[Callable] = None, indent: int = 2) -> str:
    """Safely convert data to JSON string"""
    try:
        return json.dumps(data, default=default, indent=indent, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps({"error": "Could not serialize data"}, ensure_ascii=False)


def calculate_file_hash(file_path: Union[str, Path], algorithm: str = "sha256") -> str:
    """Calculate hash of a file"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()


def calculate_text_hash(text: str, algorithm: str = "sha256") -> str:
    """Calculate hash of text"""
    hash_func = hashlib.new(algorithm)
    hash_func.update(text.encode('utf-8'))
    return hash_func.hexdigest()


def get_file_size(file_path: Union[str, Path], human_readable: bool = False) -> Union[int, str]:
    """Get file size in bytes or human readable format"""
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    size_bytes = file_path.stat().st_size
    
    if not human_readable:
        return size_bytes
    
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.1f} PB"


def format_bytes(bytes_size: int) -> str:
    """Format bytes to human readable string"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"


def ensure_directory(directory: Union[str, Path]) -> Path:
    """Ensure directory exists, create if not"""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def clean_filename(filename: str, max_length: int = 255) -> str:
    """Clean filename by removing invalid characters"""
    invalid_chars = '<>:"/\\\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    
    filename = filename.strip().strip('.')
    
    if len(filename) > max_length:
        name, ext = os.path.splitext(filename)
        name = name[:max_length - len(ext)]
        filename = name + ext
    
    return filename


def split_list(input_list: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks of specified size"""
    return [input_list[i:i + chunk_size] for i in range(0, len(input_list), chunk_size)]


def flatten_list(nested_list: List[Any]) -> List[Any]:
    """Flatten a nested list"""
    result = []
    for item in nested_list:
        if isinstance(item, list):
            result.extend(flatten_list(item))
        else:
            result.append(item)
    return result


def remove_duplicates(input_list: List[Any], key: Optional[Callable] = None) -> List[Any]:
    """Remove duplicates from list while preserving order"""
    seen = set()
    result = []
    
    for item in input_list:
        item_key = key(item) if key else item
        if item_key not in seen:
            seen.add(item_key)
            result.append(item)
    
    return result


@contextmanager
def timing_context(name: str = "operation"):
    """Context manager for timing operations"""
    start_time = datetime.now()
    try:
        yield
    finally:
        elapsed = datetime.now() - start_time
        print(f"{name} took {elapsed.total_seconds():.2f} seconds")


class Singleton(type):
    """Singleton metaclass"""
    _instances = {}
    _lock = threading.Lock()
    
    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]


class LRUCache:
    """Least Recently Used Cache"""
    
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.cache = {}
        self.order = []
    
    def get(self, key: Any) -> Optional[Any]:
        """Get value from cache"""
        if key not in self.cache:
            return None
        
        self.order.remove(key)
        self.order.append(key)
        return self.cache[key]
    
    def set(self, key: Any, value: Any):
        """Set value in cache"""
        if key in self.cache:
            self.order.remove(key)
        elif len(self.cache) >= self.capacity:
            lru_key = self.order.pop(0)
            del self.cache[lru_key]
        
        self.cache[key] = value
        self.order.append(key)
    
    def clear(self):
        """Clear cache"""
        self.cache.clear()
        self.order.clear()
    
    def __len__(self) -> int:
        return len(self.cache)


class RateLimiter:
    """Rate limiter for operations"""
    
    def __init__(self, max_requests: int, time_window: float):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self.lock = threading.Lock()
    
    def acquire(self) -> bool:
        """Acquire permission to perform operation"""
        with self.lock:
            now = time.time()
            
            self.requests = [t for t in self.requests if now - t < self.time_window]
            
            if len(self.requests) >= self.max_requests:
                return False
            
            self.requests.append(now)
            return True
    
    def wait(self):
        """Wait until operation can be performed"""
        while not self.acquire():
            time.sleep(0.1)


def get_caller_info(depth: int = 2) -> Dict[str, Any]:
    """Get information about the calling function"""
    frame = inspect.currentframe()
    for _ in range(depth):
        if frame:
            frame = frame.f_back
    
    if frame:
        info = inspect.getframeinfo(frame)
        return {
            'filename': info.filename,
            'function': info.function,
            'line': info.lineno,
            'code_context': info.code_context
        }
    
    return {}


def validate_email(email: str) -> bool:
    """Validate email address format"""
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_url(url: str) -> bool:
    """Validate URL format"""
    import re
    pattern = r'^https?://(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b[-a-zA-Z0-9()@:%_\\+.~#?&//=]*$'
    return bool(re.match(pattern, url))


def normalize_text(text: str) -> str:
    """Normalize text by removing extra whitespace and normalizing characters"""
    import unicodedata
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = ' '.join(text.split())
    return text.strip()
'''
        
        helpers_file.parent.mkdir(parents=True, exist_ok=True)
        helpers_file.write_text(helpers_content)
        
    def implement_diagnostic_tools(self):
        diagnostic_file = self.project_dir / "scripts" / "diagnostic.py"
        
        diagnostic_content = '''#!/usr/bin/env python3
"""
DocuBot Diagnostic Tools
Comprehensive system diagnostics and health checks
"""

import os
import sys
import json
import platform
import psutil
import sqlite3
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import traceback


class DocuBotDiagnostic:
    """Run comprehensive diagnostics on DocuBot installation"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'system': {},
            'installation': {},
            'dependencies': {},
            'data': {},
            'issues': []
        }
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all diagnostic checks"""
        print("Running DocuBot Diagnostics...")
        print("=" * 60)
        
        checks = [
            self.check_system,
            self.check_python_environment,
            self.check_dependencies,
            self.check_project_structure,
            self.check_data_directories,
            self.check_databases,
            self.check_configuration,
            self.check_permissions
        ]
        
        for check in checks:
            try:
                check()
            except Exception as e:
                self.add_issue(f"Check failed: {check.__name__}", str(e))
        
        self.results['summary'] = self.generate_summary()
        
        return self.results
    
    def check_system(self):
        """Check system requirements"""
        print("Checking system requirements...")
        
        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'processor': platform.processor(),
            'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
            'memory_available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024,
            'disk_total_gb': psutil.disk_usage('/').total / 1024 / 1024 / 1024,
            'disk_free_gb': psutil.disk_usage('/').free / 1024 / 1024 / 1024
        }
        
        self.results['system'] = system_info
        
        if system_info['memory_total_gb'] < 8:
            self.add_issue("Low memory", f"Only {system_info['memory_total_gb']:.1f}GB RAM available (8GB recommended)")
        
        if system_info['disk_free_gb'] < 10:
            self.add_issue("Low disk space", f"Only {system_info['disk_free_gb']:.1f}GB free (10GB recommended)")
    
    def check_python_environment(self):
        """Check Python environment"""
        print("Checking Python environment...")
        
        env_info = {
            'virtual_env': os.getenv('VIRTUAL_ENV') is not None,
            'python_path': sys.executable,
            'path': sys.path[:5]
        }
        
        if not env_info['virtual_env']:
            self.add_issue("No virtual environment", "Running outside virtual environment is not recommended")
    
    def check_dependencies(self):
        """Check required dependencies"""
        print("Checking dependencies...")
        
        dependencies = [
            'chromadb',
            'sentence_transformers',
            'customtkinter',
            'ollama',
            'fastapi',
            'pypdf2',
            'pdfplumber',
            'python-docx',
            'sqlalchemy',
            'pyyaml'
        ]
        
        installed = {}
        
        for dep in dependencies:
            try:
                __import__(dep.replace('-', '_'))
                installed[dep] = True
            except ImportError:
                installed[dep] = False
                self.add_issue("Missing dependency", f"Package not installed: {dep}")
        
        self.results['dependencies'] = installed
    
    def check_project_structure(self):
        """Check project directory structure"""
        print("Checking project structure...")
        
        required_dirs = [
            'src',
            'src/core',
            'src/document_processing',
            'src/ai_engine',
            'src/vector_store',
            'src/database',
            'src/ui',
            'data',
            'data/config',
            'data/documents',
            'tests'
        ]
        
        required_files = [
            'src/core/config.py',
            'src/core/app.py',
            'src/document_processing/processor.py',
            'src/ai_engine/llm_client.py',
            'pyproject.toml',
            'requirements.txt'
        ]
        
        structure = {
            'directories': {},
            'files': {}
        }
        
        for dir_path in required_dirs:
            full_path = Path(dir_path)
            exists = full_path.exists() and full_path.is_dir()
            structure['directories'][dir_path] = exists
            
            if not exists:
                self.add_issue("Missing directory", f"Directory not found: {dir_path}")
        
        for file_path in required_files:
            full_path = Path(file_path)
            exists = full_path.exists() and full_path.is_file()
            structure['files'][file_path] = exists
            
            if not exists:
                self.add_issue("Missing file", f"File not found: {file_path}")
            elif full_path.stat().st_size == 0:
                self.add_issue("Empty file", f"File is empty: {file_path}")
        
        self.results['installation'] = structure
    
    def check_data_directories(self):
        """Check data directories and permissions"""
        print("Checking data directories...")
        
        data_dirs = [
            Path.home() / ".docubot",
            Path.home() / ".docubot" / "models",
            Path.home() / ".docubot" / "documents",
            Path.home() / ".docubot" / "database",
            Path.home() / ".docubot" / "logs"
        ]
        
        data_info = {}
        
        for data_dir in data_dirs:
            info = {
                'exists': data_dir.exists(),
                'is_directory': data_dir.exists() and data_dir.is_dir(),
                'writable': False,
                'size_bytes': 0
            }
            
            if data_dir.exists():
                try:
                    test_file = data_dir / ".test_write"
                    test_file.write_text("test")
                    test_file.unlink()
                    info['writable'] = True
                except:
                    self.add_issue("Permission error", f"Cannot write to: {data_dir}")
                
                if data_dir.is_dir():
                    try:
                        total_size = 0
                        for file in data_dir.rglob("*"):
                            if file.is_file():
                                total_size += file.stat().st_size
                        info['size_bytes'] = total_size
                    except:
                        pass
            
            data_info[str(data_dir)] = info
        
        self.results['data'] = data_info
    
    def check_databases(self):
        """Check database files"""
        print("Checking databases...")
        
        db_files = [
            Path.home() / ".docubot" / "database" / "docubot.db",
            Path.home() / ".docubot" / "cache.db"
        ]
        
        for db_file in db_files:
            if db_file.exists():
                try:
                    conn = sqlite3.connect(db_file)
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = cursor.fetchall()
                    conn.close()
                    
                    if not tables:
                        self.add_issue("Empty database", f"No tables in database: {db_file}")
                except Exception as e:
                    self.add_issue("Database error", f"Cannot access database {db_file}: {e}")
    
    def check_configuration(self):
        """Check configuration files"""
        print("Checking configuration...")
        
        config_file = Path.home() / ".docubot" / "config" / "app_config.yaml"
        
        if config_file.exists():
            try:
                import yaml
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                if not config:
                    self.add_issue("Empty configuration", "Configuration file exists but is empty")
            except Exception as e:
                self.add_issue("Configuration error", f"Cannot read configuration: {e}")
        else:
            self.add_issue("Missing configuration", "Configuration file not found")
    
    def check_permissions(self):
        """Check file permissions"""
        print("Checking permissions...")
        
        critical_files = [
            Path.home() / ".docubot" / "secret.key",
            Path.home() / ".docubot" / "database" / "docubot.db"
        ]
        
        for file in critical_files:
            if file.exists():
                mode = file.stat().st_mode
                if mode & 0o077:  # Check if others have write/read permissions
                    self.add_issue("Insecure permissions", f"File has overly permissive permissions: {file}")
    
    def add_issue(self, title: str, description: str, severity: str = "warning"):
        """Add diagnostic issue"""
        self.results['issues'].append({
            'title': title,
            'description': description,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        })
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate diagnostic summary"""
        issues = self.results['issues']
        
        return {
            'total_checks': 8,
            'issues_found': len(issues),
            'critical_issues': len([i for i in issues if i['severity'] == 'critical']),
            'warning_issues': len([i for i in issues if i['severity'] == 'warning']),
            'info_issues': len([i for i in issues if i['severity'] == 'info']),
            'overall_status': 'healthy' if len(issues) == 0 else 'needs_attention'
        }
    
    def print_report(self):
        """Print formatted diagnostic report"""
        print("\n" + "=" * 60)
        print("DIAGNOSTIC REPORT")
        print("=" * 60)
        
        summary = self.results['summary']
        print(f"Overall Status: {summary['overall_status'].upper()}")
        print(f"Issues Found: {summary['issues_found']}")
        print(f"Critical: {summary['critical_issues']}, Warnings: {summary['warning_issues']}")
        
        if self.results['issues']:
            print("\nISSUES:")
            for issue in self.results['issues']:
                print(f"\n[{issue['severity'].upper()}] {issue['title']}")
                print(f"  {issue['description']}")
        
        print("\nSYSTEM INFORMATION:")
        system = self.results['system']
        print(f"  Platform: {system['platform']}")
        print(f"  Python: {system['python_version']}")
        print(f"  Memory: {system['memory_total_gb']:.1f}GB total, {system['memory_available_gb']:.1f}GB available")
        print(f"  Disk: {system['disk_free_gb']:.1f}GB free")
        
        print("\nRECOMMENDATIONS:")
        if summary['issues_found'] > 0:
            print("1. Address critical issues first")
            print("2. Install missing dependencies")
            print("3. Ensure proper file permissions")
        else:
            print("All systems operational. No issues detected.")
        
        print("\n" + "=" * 60)
    
    def save_report(self, output_file: Path):
        """Save diagnostic report to file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"Report saved to: {output_file}")


def main():
    """Main diagnostic entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run DocuBot diagnostics")
    parser.add_argument("--output", "-o", type=Path, help="Output JSON file for report")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode (no console output)")
    
    args = parser.parse_args()
    
    diagnostic = DocuBotDiagnostic()
    results = diagnostic.run_all_checks()
    
    if not args.quiet:
        diagnostic.print_report()
    
    if args.output:
        diagnostic.save_report(args.output)
    
    sys.exit(0 if results['summary']['overall_status'] == 'healthy' else 1)


if __name__ == "__main__":
    main()
'''
        
        diagnostic_file.parent.mkdir(parents=True, exist_ok=True)
        diagnostic_file.write_text(diagnostic_content)
        
    def implement_backup_utility(self):
        backup_file = self.project_dir / "scripts" / "backup.py"
        
        backup_content = '''#!/usr/bin/env python3
"""
DocuBot Backup Utility
Comprehensive backup and restore functionality
"""

import os
import sys
import json
import shutil
import tarfile
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import hashlib
import argparse
import io


class DocuBotBackup:
    """Manage DocuBot backups"""
    
    def __init__(self, backup_dir: Optional[Path] = None):
        self.backup_dir = backup_dir or Path.home() / ".docubot" / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_dir = Path.home() / ".docubot"
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load backup configuration"""
        config_file = self.data_dir / "config" / "backup_config.json"
        
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            'include_databases': True,
            'include_documents': True,
            'include_models': False,
            'include_logs': False,
            'compression': 'gzip',
            'max_backups': 10,
            'encryption_enabled': False
        }
    
    def create_backup(
        self,
        name: Optional[str] = None,
        description: str = "",
        include_databases: Optional[bool] = None,
        include_documents: Optional[bool] = None,
        include_models: Optional[bool] = None,
        include_logs: Optional[bool] = None
    ) -> Path:
        """Create a new backup"""
        print("Creating DocuBot backup...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = name or f"backup_{timestamp}"
        backup_file = self.backup_dir / f"{backup_name}.tar.gz"
        
        include_databases = include_databases if include_databases is not None else self.config['include_databases']
        include_documents = include_documents if include_documents is not None else self.config['include_documents']
        include_models = include_models if include_models is not None else self.config['include_models']
        include_logs = include_logs if include_logs is not None else self.config['include_logs']
        
        metadata = {
            'name': backup_name,
            'description': description,
            'timestamp': datetime.now().isoformat(),
            'version': '1.0',
            'includes': {
                'databases': include_databases,
                'documents': include_documents,
                'models': include_models,
                'logs': include_logs
            },
            'size_bytes': 0,
            'checksum': None
        }
        
        with tarfile.open(backup_file, 'w:gz') as tar:
            # Add metadata
            metadata_str = json.dumps(metadata, indent=2)
            metadata_bytes = metadata_str.encode('utf-8')
            
            metadata_info = tarfile.TarInfo('METADATA.json')
            metadata_info.size = len(metadata_bytes)
            tar.addfile(metadata_info, io.BytesIO(metadata_bytes))
            
            # Add configuration
            config_dir = self.data_dir / "config"
            if config_dir.exists():
                tar.add(config_dir, arcname="config")
            
            # Add databases
            if include_databases:
                db_dir = self.data_dir / "database"
                if db_dir.exists():
                    tar.add(db_dir, arcname="database")
            
            # Add documents
            if include_documents:
                docs_dir = self.data_dir / "documents"
                if docs_dir.exists():
                    tar.add(docs_dir, arcname="documents")
            
            # Add models (optional - can be large)
            if include_models:
                models_dir = self.data_dir / "models"
                if models_dir.exists():
                    tar.add(models_dir, arcname="models")
            
            # Add logs (optional)
            if include_logs:
                logs_dir = self.data_dir / "logs"
                if logs_dir.exists():
                    tar.add(logs_dir, arcname="logs")
        
        # Calculate checksum
        backup_size = backup_file.stat().st_size
        checksum = self._calculate_checksum(backup_file)
        
        metadata['size_bytes'] = backup_size
        metadata['checksum'] = checksum
        
        # Update metadata in archive
        self._update_backup_metadata(backup_file, metadata)
        
        # Update backup index
        self._update_backup_index(backup_file, metadata)
        
        print(f"Backup created: {backup_file}")
        print(f"Size: {backup_size / 1024 / 1024:.2f} MB")
        print(f"Checksum: {checksum}")
        
        return backup_file
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups"""
        backups = []
        
        for backup_file in self.backup_dir.glob("*.tar.gz"):
            try:
                metadata = self._extract_metadata(backup_file)
                if metadata:
                    metadata['filename'] = backup_file.name
                    metadata['file_size'] = backup_file.stat().st_size
                    backups.append(metadata)
            except:
                continue
        
        return sorted(backups, key=lambda x: x['timestamp'], reverse=True)
    
    def restore_backup(
        self,
        backup_name: str,
        restore_databases: bool = True,
        restore_documents: bool = True,
        restore_config: bool = True,
        restore_models: bool = False,
        restore_logs: bool = False,
        dry_run: bool = False
    ) -> bool:
        """Restore from backup"""
        print(f"Restoring from backup: {backup_name}")
        
        backup_file = self.backup_dir / backup_name
        if not backup_file.exists():
            print(f"Error: Backup file not found: {backup_file}")
            return False
        
        try:
            metadata = self._extract_metadata(backup_file)
            if not metadata:
                print("Error: Could not read backup metadata")
                return False
            
            # Verify checksum
            expected_checksum = metadata.get('checksum')
            if expected_checksum:
                actual_checksum = self._calculate_checksum(backup_file)
                if expected_checksum != actual_checksum:
                    print(f"Error: Checksum mismatch. Expected: {expected_checksum}, Got: {actual_checksum}")
                    return False
            
            print(f"Backup Info: {metadata['name']} - {metadata['timestamp']}")
            
            if dry_run:
                print("Dry run mode - no files will be restored")
                return True
            
            # Extract backup
            with tarfile.open(backup_file, 'r:gz') as tar:
                members = tar.getmembers()
                
                for member in members:
                    if member.name == "METADATA.json":
                        continue
                    
                    # Determine what to restore based on path and user selection
                    if member.name.startswith("config/") and restore_config:
                        tar.extract(member, self.data_dir)
                    elif member.name.startswith("database/") and restore_databases:
                        tar.extract(member, self.data_dir)
                    elif member.name.startswith("documents/") and restore_documents:
                        tar.extract(member, self.data_dir)
                    elif member.name.startswith("models/") and restore_models:
                        tar.extract(member, self.data_dir)
                    elif member.name.startswith("logs/") and restore_logs:
                        tar.extract(member, self.data_dir)
            
            print("Restore completed successfully")
            return True
            
        except Exception as e:
            print(f"Error during restore: {e}")
            return False
    
    def delete_backup(self, backup_name: str) -> bool:
        """Delete a backup"""
        backup_file = self.backup_dir / backup_name
        
        if backup_file.exists():
            try:
                backup_file.unlink()
                
                # Update index
                index_file = self.backup_dir / "backup_index.json"
                if index_file.exists():
                    with open(index_file, 'r', encoding='utf-8') as f:
                        index = json.load(f)
                    
                    index['backups'] = [b for b in index['backups'] if b['filename'] != backup_name]
                    
                    with open(index_file, 'w', encoding='utf-8') as f:
                        json.dump(index, f, indent=2)
                
                print(f"Backup deleted: {backup_name}")
                return True
            except Exception as e:
                print(f"Error deleting backup: {e}")
                return False
        else:
            print(f"Backup not found: {backup_name}")
            return False
    
    def cleanup_old_backups(self, keep_last: int = 10) -> List[str]:
        """Clean up old backups, keeping only specified number"""
        backups = self.list_backups()
        
        if len(backups) <= keep_last:
            print(f"Keeping all {len(backups)} backups")
            return []
        
        to_delete = backups[keep_last:]
        deleted = []
        
        for backup in to_delete:
            if self.delete_backup(backup['filename']):
                deleted.append(backup['filename'])
        
        print(f"Deleted {len(deleted)} old backups")
        return deleted
    
    def verify_backup(self, backup_name: str) -> bool:
        """Verify backup integrity"""
        backup_file = self.backup_dir / backup_name
        
        if not backup_file.exists():
            print(f"Error: Backup file not found: {backup_file}")
            return False
        
        try:
            metadata = self._extract_metadata(backup_file)
            if not metadata:
                print("Error: Could not read backup metadata")
                return False
            
            expected_checksum = metadata.get('checksum')
            if not expected_checksum:
                print("Warning: No checksum in metadata")
                return True
            
            actual_checksum = self._calculate_checksum(backup_file)
            
            if expected_checksum == actual_checksum:
                print(f"Backup verified successfully: {backup_name}")
                return True
            else:
                print(f"Error: Checksum mismatch. Expected: {expected_checksum}, Got: {actual_checksum}")
                return False
            
        except Exception as e:
            print(f"Error verifying backup: {e}")
            return False
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file"""
        sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        
        return sha256.hexdigest()
    
    def _extract_metadata(self, backup_file: Path) -> Optional[Dict[str, Any]]:
        """Extract metadata from backup file"""
        try:
            with tarfile.open(backup_file, 'r:gz') as tar:
                metadata_member = tar.getmember('METADATA.json')
                metadata_file = tar.extractfile(metadata_member)
                
                if metadata_file:
                    metadata_bytes = metadata_file.read()
                    return json.loads(metadata_bytes.decode('utf-8'))
        except:
            return None
    
    def _update_backup_metadata(self, backup_file: Path, metadata: Dict[str, Any]):
        """Update metadata in backup file"""
        # This is complex with tar files, so we'll store metadata separately
        metadata_file = backup_file.with_suffix('.json')
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    
    def _update_backup_index(self, backup_file: Path, metadata: Dict[str, Any]):
        """Update backup index file"""
        index_file = self.backup_dir / "backup_index.json"
        
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
        else:
            index = {'backups': []}
        
        index_entry = {
            'filename': backup_file.name,
            'name': metadata['name'],
            'timestamp': metadata['timestamp'],
            'size_bytes': metadata['size_bytes'],
            'checksum': metadata['checksum'],
            'description': metadata.get('description', '')
        }
        
        # Remove existing entry if present
        index['backups'] = [b for b in index['backups'] if b['filename'] != backup_file.name]
        index['backups'].append(index_entry)
        
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2)
    
    def export_backup_info(self, output_file: Path):
        """Export backup information to file"""
        backups = self.list_backups()
        
        info = {
            'timestamp': datetime.now().isoformat(),
            'backup_dir': str(self.backup_dir),
            'total_backups': len(backups),
            'total_size_gb': sum(b['file_size'] for b in backups) / 1024 / 1024 / 1024,
            'backups': backups
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2)
        
        print(f"Backup info exported to: {output_file}")


def main():
    """Main backup utility entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="DocuBot Backup Utility")
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Create backup command
    create_parser = subparsers.add_parser('create', help='Create new backup')
    create_parser.add_argument('--name', '-n', help='Backup name')
    create_parser.add_argument('--description', '-d', default='', help='Backup description')
    create_parser.add_argument('--no-databases', action='store_true', help='Exclude databases')
    create_parser.add_argument('--no-documents', action='store_true', help='Exclude documents')
    create_parser.add_argument('--include-models', action='store_true', help='Include AI models')
    create_parser.add_argument('--include-logs', action='store_true', help='Include log files')
    
    # List backups command
    list_parser = subparsers.add_parser('list', help='List available backups')
    
    # Restore backup command
    restore_parser = subparsers.add_parser('restore', help='Restore from backup')
    restore_parser.add_argument('backup', help='Backup filename to restore')
    restore_parser.add_argument('--no-databases', action='store_true', help='Do not restore databases')
    restore_parser.add_argument('--no-documents', action='store_true', help='Do not restore documents')
    restore_parser.add_argument('--no-config', action='store_true', help='Do not restore configuration')
    restore_parser.add_argument('--dry-run', action='store_true', help='Dry run mode')
    
    # Delete backup command
    delete_parser = subparsers.add_parser('delete', help='Delete backup')
    delete_parser.add_argument('backup', help='Backup filename to delete')
    
    # Verify backup command
    verify_parser = subparsers.add_parser('verify', help='Verify backup integrity')
    verify_parser.add_argument('backup', help='Backup filename to verify')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old backups')
    cleanup_parser.add_argument('--keep', '-k', type=int, default=10, help='Number of backups to keep')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export backup information')
    export_parser.add_argument('--output', '-o', type=Path, required=True, help='Output JSON file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    backup_manager = DocuBotBackup()
    
    try:
        if args.command == 'create':
            backup_file = backup_manager.create_backup(
                name=args.name,
                description=args.description,
                include_databases=not args.no_databases,
                include_documents=not args.no_documents,
                include_models=args.include_models,
                include_logs=args.include_logs
            )
            
        elif args.command == 'list':
            backups = backup_manager.list_backups()
            
            if not backups:
                print("No backups found")
            else:
                print(f"\nFound {len(backups)} backups:\n")
                for i, backup in enumerate(backups, 1):
                    size_mb = backup['file_size'] / 1024 / 1024
                    date = datetime.fromisoformat(backup['timestamp']).strftime("%Y-%m-%d %H:%M")
                    print(f"{i:2}. {backup['filename']}")
                    print(f"     Name: {backup['name']}")
                    print(f"     Date: {date}")
                    print(f"     Size: {size_mb:.1f} MB")
                    print(f"     Desc: {backup.get('description', '')}")
                    print()
        
        elif args.command == 'restore':
            success = backup_manager.restore_backup(
                backup_name=args.backup,
                restore_databases=not args.no_databases,
                restore_documents=not args.no_documents,
                restore_config=not args.no_config,
                dry_run=args.dry_run
            )
            
            if not success:
                sys.exit(1)
        
        elif args.command == 'delete':
            success = backup_manager.delete_backup(args.backup)
            
            if not success:
                sys.exit(1)
        
        elif args.command == 'verify':
            success = backup_manager.verify_backup(args.backup)
            
            if not success:
                sys.exit(1)
        
        elif args.command == 'cleanup':
            deleted = backup_manager.cleanup_old_backups(keep_last=args.keep)
            
            if deleted:
                print(f"Deleted backups: {', '.join(deleted)}")
        
        elif args.command == 'export':
            backup_manager.export_backup_info(args.output)
        
        print("Operation completed successfully")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
'''
        
        backup_file.parent.mkdir(parents=True, exist_ok=True)
        backup_file.write_text(backup_content)
        
    def update_remaining_files(self):
        """Update remaining utility files"""
        
        # Update cleanup.py
        cleanup_file = self.project_dir / "src" / "utilities" / "cleanup.py"
        if cleanup_file.exists():
            cleanup_content = '''"""
DocuBot Resource Cleanup System
Clean up temporary files and optimize resources
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import sqlite3


class ResourceCleanup:
    """Manage resource cleanup and optimization"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path.home() / ".docubot"
    
    def cleanup_temporary_files(self, max_age_days: int = 7) -> List[Path]:
        """Clean up temporary files older than specified days"""
        temp_dirs = [
            self.data_dir / "tmp",
            self.data_dir / "cache" / "temp",
            self.data_dir / "documents" / "uploads" / "temp"
        ]
        
        cleaned = []
        cutoff = datetime.now() - timedelta(days=max_age_days)
        
        for temp_dir in temp_dirs:
            if temp_dir.exists():
                for file in temp_dir.rglob("*"):
                    if file.is_file():
                        try:
                            mtime = datetime.fromtimestamp(file.stat().st_mtime)
                            if mtime < cutoff:
                                file.unlink()
                                cleaned.append(file)
                        except:
                            pass
        
        return cleaned
    
    def cleanup_old_logs(self, max_age_days: int = 30) -> List[Path]:
        """Clean up old log files"""
        logs_dir = self.data_dir / "logs"
        
        if not logs_dir.exists():
            return []
        
        cleaned = []
        cutoff = datetime.now() - timedelta(days=max_age_days)
        
        for log_file in logs_dir.glob("*.log*"):
            if log_file.is_file():
                try:
                    mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if mtime < cutoff:
                        log_file.unlink()
                        cleaned.append(log_file)
                except:
                    pass
        
        return cleaned
    
    def cleanup_old_backups(self, max_age_days: int = 90, keep_minimum: int = 5) -> List[Path]:
        """Clean up old backup files"""
        backups_dir = self.data_dir / "backups"
        
        if not backups_dir.exists():
            return []
        
        backup_files = list(backups_dir.glob("*.tar.gz"))
        backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        if len(backup_files) <= keep_minimum:
            return []
        
        cleaned = []
        cutoff = datetime.now() - timedelta(days=max_age_days)
        
        for backup_file in backup_files[keep_minimum:]:
            try:
                mtime = datetime.fromtimestamp(backup_file.stat().st_mtime)
                if mtime < cutoff:
                    backup_file.unlink()
                    
                    metadata_file = backup_file.with_suffix('.json')
                    if metadata_file.exists():
                        metadata_file.unlink()
                    
                    cleaned.append(backup_file)
            except:
                pass
        
        return cleaned
    
    def optimize_databases(self) -> Dict[str, Any]:
        """Optimize database files"""
        results = {}
        
        db_files = [
            self.data_dir / "database" / "docubot.db",
            self.data_dir / "cache.db"
        ]
        
        for db_file in db_files:
            if db_file.exists():
                try:
                    before_size = db_file.stat().st_size
                    
                    conn = sqlite3.connect(db_file)
                    conn.execute("VACUUM")
                    conn.close()
                    
                    after_size = db_file.stat().st_size
                    
                    results[str(db_file)] = {
                        'before_size_mb': before_size / 1024 / 1024,
                        'after_size_mb': after_size / 1024 / 1024,
                        'reduction_percent': ((before_size - after_size) / before_size * 100) if before_size > 0 else 0
                    }
                except Exception as e:
                    results[str(db_file)] = {'error': str(e)}
        
        return results
    
    def cleanup_empty_directories(self) -> List[Path]:
        """Remove empty directories"""
        cleaned = []
        
        for root, dirs, files in os.walk(self.data_dir, topdown=False):
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                try:
                    if not any(dir_path.iterdir()):
                        dir_path.rmdir()
                        cleaned.append(dir_path)
                except:
                    pass
        
        return cleaned
    
    def get_storage_usage(self) -> Dict[str, float]:
        """Get storage usage statistics"""
        usage = {}
        
        directories = {
            'total': self.data_dir,
            'documents': self.data_dir / "documents",
            'database': self.data_dir / "database",
            'models': self.data_dir / "models",
            'logs': self.data_dir / "logs",
            'backups': self.data_dir / "backups",
            'cache': self.data_dir / "cache"
        }
        
        for name, directory in directories.items():
            if directory.exists():
                total_size = 0
                for file in directory.rglob("*"):
                    if file.is_file():
                        try:
                            total_size += file.stat().st_size
                        except:
                            pass
                
                usage[name] = total_size / 1024 / 1024  # Convert to MB
        
        return usage
    
    def run_complete_cleanup(self, interactive: bool = False) -> Dict[str, Any]:
        """Run complete cleanup routine"""
        results = {}
        
        if interactive:
            print("Running DocuBot Cleanup...")
            print("=" * 60)
        
        # Cleanup temporary files
        temp_files = self.cleanup_temporary_files()
        results['temp_files_cleaned'] = len(temp_files)
        
        if interactive:
            print(f"Cleaned {len(temp_files)} temporary files")
        
        # Cleanup old logs
        old_logs = self.cleanup_old_logs()
        results['old_logs_cleaned'] = len(old_logs)
        
        if interactive:
            print(f"Cleaned {len(old_logs)} old log files")
        
        # Cleanup old backups
        old_backups = self.cleanup_old_backups()
        results['old_backups_cleaned'] = len(old_backups)
        
        if interactive:
            print(f"Cleaned {len(old_backups)} old backups")
        
        # Optimize databases
        db_results = self.optimize_databases()
        results['database_optimization'] = db_results
        
        if interactive:
            for db, stats in db_results.items():
                if 'error' not in stats:
                    print(f"Optimized {Path(db).name}: {stats['reduction_percent']:.1f}% reduction")
        
        # Cleanup empty directories
        empty_dirs = self.cleanup_empty_directories()
        results['empty_directories_cleaned'] = len(empty_dirs)
        
        if interactive:
            print(f"Removed {len(empty_dirs)} empty directories")
        
        # Get storage usage
        storage_usage = self.get_storage_usage()
        results['storage_usage_mb'] = storage_usage
        
        if interactive:
            print("\nStorage Usage:")
            for name, size_mb in storage_usage.items():
                print(f"  {name}: {size_mb:.1f} MB")
        
        if interactive:
            print("\n" + "=" * 60)
            print("Cleanup completed successfully")
        
        return results


# Global cleanup instance
_resource_cleanup = None

def get_resource_cleanup() -> ResourceCleanup:
    """Get singleton resource cleanup instance"""
    global _resource_cleanup
    if _resource_cleanup is None:
        _resource_cleanup = ResourceCleanup()
    return _resource_cleanup


def cleanup_resources():
    """Convenience function for cleaning up resources"""
    cleaner = get_resource_cleanup()
    return cleaner.run_complete_cleanup()
'''
            
            cleanup_file.parent.mkdir(parents=True, exist_ok=True)
            cleanup_file.write_text(cleanup_content)
        else:
            # Create cleanup.py if it doesn't exist
            cleanup_file.parent.mkdir(parents=True, exist_ok=True)
            cleanup_file.write_text(cleanup_content)
        
        # Create missing __init__.py files
        missing_inits = [
            self.project_dir / "src" / "storage" / "__init__.py",
            self.project_dir / "src" / "utilities" / "__init__.py",
            self.project_dir / "tests" / "unit" / "__init__.py",
            self.project_dir / "tests" / "integration" / "__init__.py",
            self.project_dir / "tests" / "e2e" / "__init__.py"
        ]
        
        for init_file in missing_inits:
            if not init_file.exists():
                init_file.parent.mkdir(parents=True, exist_ok=True)
                init_file.write_text('"""Package initialization"""\n')


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Implement DocuBot utility modules"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="DocuBot",
        help="Project directory path"
    )
    
    args = parser.parse_args()
    
    fixer = DocuBotUtilityFixer(args.dir)
    fixer.implement_all_utilities()


if __name__ == "__main__":
    main()