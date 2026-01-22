#docubot/src/utilities/logger.py

"""
logging configuration for DocuBot.
Provides structured logging with rotation and different log levels.
"""

import logging
import logging.handlers
import sys
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import structlog


class DocuBotLogger:
    """
    Advanced logging system for DocuBot with structured logging.
    Supports file rotation, console output, and JSON formatting.
    """
    
    def __init__(self, log_dir: Optional[Path] = None, log_level: str = "INFO"):
        """
        Initialize the logging system.
        
        Args:
            log_dir: Directory for log files. If None, logs to console only.
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.log_dir = log_dir
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        
        # Configure structlog for structured logging
        self._configure_structlog()
        
        # Configure Python logging
        self._configure_python_logging()
        
        # Get logger instance
        self.logger = structlog.get_logger()
    
    def _configure_structlog(self) -> None:
        """Configure structlog for structured logging."""
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    def _configure_python_logging(self) -> None:
        """Configure Python's built-in logging."""
        
        # Clear any existing handlers
        logging.getLogger().handlers.clear()
        
        # Create formatters
        if self.log_dir:
            # File formatter (detailed)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            
            # Console formatter (simple)
            console_formatter = logging.Formatter(
                '%(levelname)s: %(message)s'
            )
        else:
            # Console only formatter
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%H:%M:%S'
            )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        console_handler.setFormatter(console_formatter)
        
        # Add handlers to root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        root_logger.addHandler(console_handler)
        
        # File handlers (if log directory specified)
        if self.log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            
            # Main log file with rotation (10MB per file, keep 5 backups)
            log_file = self.log_dir / 'app.log'
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setLevel(self.log_level)
            file_handler.setFormatter(file_formatter)
            root_logger.addHandler(file_handler)
            
            # Error log file (errors only)
            error_file = self.log_dir / 'error.log'
            error_handler = logging.handlers.RotatingFileHandler(
                error_file,
                maxBytes=5 * 1024 * 1024,  # 5MB
                backupCount=3,
                encoding='utf-8'
            )
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(file_formatter)
            root_logger.addHandler(error_handler)
            
            # Performance log file
            perf_file = self.log_dir / 'performance.log'
            perf_handler = logging.handlers.RotatingFileHandler(
                perf_file,
                maxBytes=5 * 1024 * 1024,  # 5MB
                backupCount=2,
                encoding='utf-8'
            )
            perf_handler.setLevel(logging.INFO)
            perf_handler.setFormatter(file_formatter)
            perf_handler.addFilter(self._performance_filter)
            root_logger.addHandler(perf_handler)
    
    def _performance_filter(self, record: logging.LogRecord) -> bool:
        """Filter for performance-related log messages."""
        return 'performance' in record.getMessage().lower() or 'time' in record.getMessage().lower()
    
    def log_startup(self) -> None:
        """Log application startup information."""
        self.logger.info(
            "application_startup",
            timestamp=datetime.now().isoformat(),
            log_level=self.log_level,
            log_dir=str(self.log_dir) if self.log_dir else "console_only"
        )
    
    def log_shutdown(self) -> None:
        """Log application shutdown."""
        self.logger.info("application_shutdown")
    
    def log_performance(self, operation: str, duration_ms: float, details: Dict[str, Any] = None) -> None:
        """
        Log performance metrics.
        
        Args:
            operation: Name of the operation.
            duration_ms: Duration in milliseconds.
            details: Additional performance details.
        """
        self.logger.info(
            "performance_metric",
            operation=operation,
            duration_ms=duration_ms,
            details=details or {}
        )
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any] = None) -> None:
        """
        Log error with context information.
        
        Args:
            error: Exception object.
            context: Context information about where error occurred.
        """
        self.logger.error(
            "error_occurred",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context or {}
        )
    
    def log_document_processing(self, file_path: str, status: str, details: Dict[str, Any] = None) -> None:
        """
        Log document processing events.
        
        Args:
            file_path: Path to the document.
            status: Processing status.
            details: Processing details.
        """
        self.logger.info(
            "document_processing",
            file_path=file_path,
            status=status,
            details=details or {}
        )
    
    def log_ai_interaction(self, operation: str, model: str, details: Dict[str, Any] = None) -> None:
        """
        Log AI/LLM interactions.
        
        Args:
            operation: AI operation (e.g., 'query', 'embedding', 'generation').
            model: Model used.
            details: Interaction details.
        """
        self.logger.info(
            "ai_interaction",
            operation=operation,
            model=model,
            details=details or {}
        )


def setup_logging(log_dir: Optional[Path] = None, log_level: str = "INFO") -> DocuBotLogger:
    """
    Setup and return a DocuBotLogger instance.
    
    Args:
        log_dir: Directory for log files.
        log_level: Logging level.
    
    Returns:
        Configured DocuBotLogger instance.
    """
    return DocuBotLogger(log_dir, log_level)


def get_logger(name: str = "docubot") -> structlog.BoundLogger:
    """
    Get a structured logger by name.
    
    Args:
        name: Logger name.
    
    Returns:
        Structured logger instance.
    """
    return structlog.get_logger(name)


# Example usage and testing
if __name__ == "__main__":
    # Test the logger
    import tempfile
    import time
    
    with tempfile.TemporaryDirectory() as temp_dir:
        log_dir = Path(temp_dir) / "logs"
        logger = setup_logging(log_dir, "DEBUG")
        
        # Test different log levels
        logger.logger.debug("debug_message", test="debug")
        logger.logger.info("info_message", test="info")
        logger.logger.warning("warning_message", test="warning")
        logger.logger.error("error_message", test="error")
        
        # Test specialized logging methods
        logger.log_startup()
        
        # Test performance logging
        start_time = time.time()
        time.sleep(0.1)
        duration = (time.time() - start_time) * 1000
        logger.log_performance("test_operation", duration, {"iterations": 10})
        
        # Test error logging
        try:
            raise ValueError("Test error for logging")
        except ValueError as e:
            logger.log_error_with_context(e, {"phase": "testing", "user": "test_user"})
        
        # Test document processing logging
        logger.log_document_processing(
            "/path/to/document.pdf",
            "processed",
            {"pages": 10, "size_kb": 1024}
        )
        
        logger.log_shutdown()
        
        print(f"Logs created in: {log_dir}")
        for log_file in log_dir.glob("*.log"):
            print(f"  - {log_file.name}: {log_file.stat().st_size} bytes")