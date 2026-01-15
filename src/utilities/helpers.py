"""
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
    invalid_chars = '<>:"/\\|?*'
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
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_url(url: str) -> bool:
    """Validate URL format"""
    import re
    pattern = r'^https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b[-a-zA-Z0-9()@:%_\+.~#?&//=]*$'
    return bool(re.match(pattern, url))


def normalize_text(text: str) -> str:
    """Normalize text by removing extra whitespace and normalizing characters"""
    import unicodedata
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = ' '.join(text.split())
    return text.strip()
