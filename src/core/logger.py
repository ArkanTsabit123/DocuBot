"""
DocuBot Logging Configuration
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import json
from datetime import datetime


class StructuredFormatter(logging.Formatter):
    def format(self, record):
        log_record = {
            'timestamp': datetime.now().isoformat(),
            'level': record.levelname,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
        }
        
        if hasattr(record, 'extra'):
            log_record.update(record.extra)
        
        return json.dumps(log_record)


def setup_logger(
    name: str = "docubot",
    log_level: str = "INFO",
    log_file: Optional[Path] = None,
    console_output: bool = True
) -> logging.Logger:
    logger = logging.getLogger(name)
    
    logger.handlers.clear()
    
    logger.setLevel(getattr(logging, log_level.upper()))
    
    json_formatter = StructuredFormatter()
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(json_formatter)
        logger.addHandler(file_handler)
    
    logger.propagate = False
    
    return logger


_default_logger = None


def get_logger(name: str = "docubot") -> logging.Logger:
    global _default_logger
    
    if _default_logger is None:
        try:
            from .config import get_config
            config = get_config()
            log_file = config.logs_dir / "app.log"
        except:
            log_file = Path.home() / ".docubot" / "logs" / "app.log"
        
        _default_logger = setup_logger(
            name=name,
            log_level="INFO",
            log_file=log_file,
            console_output=True
        )
    
    return _default_logger


def log_execution_time(func):
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(f"perf.{func.__module__}")
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            if execution_time > 1.0:
                logger.info(
                    "Function execution time",
                    extra={
                        'function': func.__name__,
                        'execution_time_seconds': execution_time,
                        'module': func.__module__
                    }
                )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Function failed after {execution_time:.2f}s",
                extra={
                    'function': func.__name__,
                    'error': str(e),
                    'execution_time_seconds': execution_time
                }
            )
            raise
    
    return wrapper
