"""
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
