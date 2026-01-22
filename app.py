# app.py
"""
DocuBot - Main Application Entry Point
Local AI Knowledge Assistant for Document Management and Querying
Version: 1.0.0
"""

import sys
import os
import json
import time
import signal
import traceback
import argparse
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
import yaml

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


@dataclass
class ApplicationStatus:
    """Application status information."""
    initialized: bool = False
    startup_time: float = 0.0
    components_loaded: List[str] = None
    error_count: int = 0
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.components_loaded is None:
            self.components_loaded = []
        if self.warnings is None:
            self.warnings = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ConfigurationManager:
    """Simple configuration manager."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration manager."""
        self.config_path = config_path or project_root / "data" / "config" / "app_config.yaml"
        self.config = self.load_default_config()
        self.load_config()
    
    def load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            "app": {
                "name": "DocuBot",
                "version": "1.0.0",
                "debug": False,
                "log_level": "INFO"
            },
            "paths": {
                "data_dir": str(project_root / "data"),
                "models_dir": str(project_root / "data" / "models"),
                "documents_dir": str(project_root / "data" / "documents"),
                "database_dir": str(project_root / "data" / "database"),
                "logs_dir": str(project_root / "data" / "logs"),
                "exports_dir": str(project_root / "data" / "exports")
            },
            "document_processing": {
                "chunk_size": 500,
                "chunk_overlap": 50,
                "max_file_size_mb": 100,
                "supported_formats": [".pdf", ".docx", ".txt", ".epub", ".md", ".html"],
                "ocr_enabled": True
            },
            "ai": {
                "llm": {
                    "provider": "ollama",
                    "model": "llama2:7b",
                    "temperature": 0.1,
                    "max_tokens": 1024
                },
                "embeddings": {
                    "model": "all-MiniLM-L6-v2",
                    "dimensions": 384
                },
                "rag": {
                    "top_k": 5,
                    "similarity_threshold": 0.7
                }
            },
            "ui": {
                "theme": "dark",
                "language": "en",
                "font_size": 12
            },
            "storage": {
                "max_documents": 10000,
                "backup_enabled": True
            }
        }
    
    def load_config(self):
        """Load configuration from file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f)
                
                # Merge configurations
                self.merge_configs(self.config, file_config)
                print(f"Loaded configuration from {self.config_path}")
            else:
                print(f"Config file not found at {self.config_path}, using defaults")
        except Exception as e:
            print(f"Error loading config file: {e}")
    
    def merge_configs(self, base: Dict, overlay: Dict):
        """Recursively merge two dictionaries."""
        for key, value in overlay.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self.merge_configs(base[key], value)
            else:
                base[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def save(self):
        """Save configuration to file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            print(f"Configuration saved to {self.config_path}")
            return True
        except Exception as e:
            print(f"Error saving configuration: {e}")
            return False
    
    def validate(self) -> bool:
        """Validate configuration."""
        required_paths = [
            "paths.data_dir",
            "paths.documents_dir",
            "paths.logs_dir"
        ]
        
        for path_key in required_paths:
            path_str = self.get(path_key)
            if not path_str:
                print(f"Missing required configuration: {path_key}")
                return False
        
        return True


class Logger:
    """Simple logger for application."""
    
    def __init__(self, name: str = "docubot", log_dir: Optional[Path] = None, level: str = "INFO"):
        """Initialize logger."""
        self.name = name
        self.log_dir = log_dir or project_root / "data" / "logs"
        self.level = level
        self.log_file = self.log_dir / "app.log"
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging directory and file."""
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Failed to create log directory: {e}")
    
    def log(self, level: str, message: str, **kwargs):
        """Log a message."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"{timestamp} - {level.upper()} - {message}"
        
        if kwargs:
            log_message += f" - {kwargs}"
        
        # Print to console
        print(log_message)
        
        # Write to file
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_message + "\n")
        except Exception:
            pass
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.log("INFO", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.log("ERROR", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.log("WARNING", message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.log("DEBUG", message, **kwargs)


class DocuBotCore:
    """
    Main application orchestrator.
    
    This class coordinates all core functionality including:
    - Document processing and ingestion
    - Vector storage management
    - LLM integration
    - Query processing
    """
    
    def __init__(self, config: ConfigurationManager, logger: Logger):
        """
        Initialize the DocuBot core application.
        
        Args:
            config: Configuration manager instance
            logger: Logger instance
        """
        self.config = config
        self.logger = logger
        self.status = ApplicationStatus()
        self.components = {}
        self.startup_time = time.time()
    
    def initialize(self) -> bool:
        """
        Initialize all core components.
        
        Returns:
            bool: True if initialization succeeded, False otherwise
        """
        try:
            self.logger.info("Initializing DocuBot Core...")
            
            # Create necessary directories
            self.create_directories()
            
            # Initialize components in order
            components_to_load = [
                ("database", self.initialize_database),
                ("vector_store", self.initialize_vector_store),
                ("document_processor", self.initialize_document_processor),
                ("ai_engine", self.initialize_ai_engine)
            ]
            
            for name, init_func in components_to_load:
                try:
                    success = init_func()
                    if success:
                        self.status.components_loaded.append(name)
                        self.logger.info(f"Component '{name}' initialized successfully")
                    else:
                        self.status.warnings.append(f"Component '{name}' initialization failed")
                except Exception as e:
                    self.status.warnings.append(f"Error initializing '{name}': {str(e)}")
                    self.logger.error(f"Failed to initialize component '{name}': {e}")
            
            # Calculate startup time
            self.status.startup_time = time.time() - self.startup_time
            self.status.initialized = len(self.status.components_loaded) > 0
            
            if self.status.initialized:
                self.logger.info(f"DocuBot Core initialized successfully in {self.status.startup_time:.2f}s")
                self.logger.info(f"Loaded components: {', '.join(self.status.components_loaded)}")
            else:
                self.logger.warning("DocuBot Core partially initialized - some components failed")
            
            return self.status.initialized
            
        except Exception as e:
            self.logger.error(f"Failed to initialize DocuBot Core: {e}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def create_directories(self):
        """Create necessary directories."""
        dirs_to_create = [
            self.config.get("paths.data_dir"),
            self.config.get("paths.models_dir"),
            self.config.get("paths.documents_dir"),
            self.config.get("paths.database_dir"),
            self.config.get("paths.logs_dir"),
            self.config.get("paths.exports_dir"),
        ]
        
        for dir_path in dirs_to_create:
            if dir_path:
                try:
                    Path(dir_path).mkdir(parents=True, exist_ok=True)
                    self.logger.debug(f"Created directory: {dir_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to create directory {dir_path}: {e}")
    
    def initialize_database(self) -> bool:
        """Initialize database connections."""
        self.logger.info("Initializing database...")
        # Check if SQLite database exists or can be created
        db_path = Path(self.config.get("paths.database_dir")) / "sqlite.db"
        try:
            import sqlite3
            conn = sqlite3.connect(db_path)
            conn.execute("SELECT 1")
            conn.close()
            self.components["database"] = {"type": "sqlite", "path": str(db_path)}
            return True
        except Exception as e:
            self.logger.warning(f"Database initialization failed: {e}")
            return False
    
    def initialize_vector_store(self) -> bool:
        """Initialize vector store for embeddings."""
        self.logger.info("Initializing vector store...")
        # Check if vector store directory exists
        vector_store_dir = Path(self.config.get("paths.database_dir")) / "chroma"
        try:
            vector_store_dir.mkdir(parents=True, exist_ok=True)
            self.components["vector_store"] = {"type": "chromadb", "path": str(vector_store_dir)}
            return True
        except Exception as e:
            self.logger.warning(f"Vector store initialization failed: {e}")
            return False
    
    def initialize_document_processor(self) -> bool:
        """Initialize document processing pipeline."""
        self.logger.info("Initializing document processor...")
        # Check for required document processing libraries
        try:
            # Try to import basic text processing
            import re
            import chardet
            self.components["document_processor"] = {"status": "available"}
            return True
        except ImportError as e:
            self.logger.warning(f"Document processor dependencies missing: {e}")
            return False
    
    def initialize_ai_engine(self) -> bool:
        """Initialize AI/ML components."""
        self.logger.info("Initializing AI engine...")
        # Check for AI components
        ai_status = {"llm": "not_available", "embeddings": "not_available"}
        
        # Check for Ollama
        try:
            result = subprocess.run(["ollama", "--version"], capture_output=True, text=True, check=False)
            if result.returncode == 0:
                ai_status["llm"] = "available"
                self.logger.info(f"Ollama found: {result.stdout.strip()}")
        except (FileNotFoundError, subprocess.SubprocessError):
            self.logger.warning("Ollama not found - LLM features will be disabled")
        
        # Check for sentence-transformers
        try:
            import sentence_transformers
            ai_status["embeddings"] = "available"
            self.logger.info("Sentence transformers available")
        except ImportError:
            self.logger.warning("Sentence transformers not found - embedding features will be disabled")
        
        self.components["ai_engine"] = ai_status
        return ai_status["llm"] == "available" or ai_status["embeddings"] == "available"
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process a document file.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Dict with processing results
        """
        if not self.status.initialized:
            return {
                "status": "error",
                "message": "DocuBot Core not initialized",
                "file_path": file_path
            }
        
        self.logger.info(f"Processing document: {file_path}")
        
        try:
            # Basic document validation
            path = Path(file_path)
            if not path.exists():
                return {
                    "status": "error",
                    "message": f"File not found: {file_path}",
                    "file_path": file_path
                }
            
            # Check file extension
            file_ext = path.suffix.lower()
            supported_formats = self.config.get("document_processing.supported_formats", [])
            
            if file_ext not in supported_formats:
                return {
                    "status": "error",
                    "message": f"Unsupported file format: {file_ext}. Supported: {supported_formats}",
                    "file_path": file_path
                }
            
            # For now, just return success with basic info
            # Actual processing will be implemented in document_processing module
            return {
                "status": "success",
                "message": "Document queued for processing",
                "file_path": file_path,
                "file_size": path.stat().st_size,
                "file_type": file_ext
            }
            
        except Exception as e:
            self.logger.error(f"Error processing document {file_path}: {e}")
            return {
                "status": "error",
                "message": f"Processing error: {str(e)}",
                "file_path": file_path
            }
    
    def ask_question(self, question: str, context: Dict = None) -> Dict[str, Any]:
        """
        Process a user question using RAG pipeline.
        
        Args:
            question: User's question
            context: Additional context (conversation history, etc.)
            
        Returns:
            Dict with answer and metadata
        """
        if not self.status.initialized:
            return {
                "status": "error",
                "answer": "DocuBot Core not initialized",
                "sources": [],
                "confidence": 0.0
            }
        
        self.logger.info(f"Processing question: {question}")
        
        # Check if AI components are available
        ai_status = self.components.get("ai_engine", {})
        
        if ai_status.get("llm") != "available":
            return {
                "status": "warning",
                "answer": "LLM features are not available. Please install Ollama to enable AI question answering.",
                "sources": [],
                "confidence": 0.0
            }
        
        # Placeholder for actual RAG pipeline
        # This will be implemented in ai_engine module
        return {
            "status": "success",
            "answer": f"This is a placeholder response to your question: '{question}'\n\nAI question answering is not fully implemented yet. Please check back after completing the AI integration tasks.",
            "sources": ["System placeholder"],
            "confidence": 0.3,
            "processing_time": 0.1
        }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current application status.
        
        Returns:
            Dict with status information
        """
        return {
            **self.status.to_dict(),
            "timestamp": datetime.now().isoformat(),
            "components": self.components,
            "config": {
                "app_name": self.config.get("app.name"),
                "version": self.config.get("app.version"),
                "debug": self.config.get("app.debug")
            }
        }
    
    def shutdown(self):
        """Gracefully shutdown all components."""
        self.logger.info("Shutting down DocuBot Core...")
        self.status.initialized = False
        self.components.clear()
        self.logger.info("DocuBot Core shutdown complete")


class DocuBotApplication:
    """
    Main application orchestrator for DocuBot.
    Handles initialization, lifecycle management, and graceful shutdown.
    """
    
    def __init__(self):
        """Initialize the DocuBot application."""
        self.project_root = Path(__file__).parent
        self.config = None
        self.core = None
        self.logger = None
        self.running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle termination signals gracefully."""
        if self.logger:
            self.logger.info(f"Received signal {signum}, initiating shutdown...")
        self.shutdown()
        sys.exit(0)
    
    def initialize(self) -> bool:
        """
        Initialize the application and all components.
        
        Returns:
            bool: True if initialization succeeded, False otherwise
        """
        try:
            # Initialize logging
            self.logger = Logger(
                name="docubot",
                log_dir=self.project_root / "data" / "logs",
                level="INFO"
            )
            self.logger.info("Initializing DocuBot application...")
            self.logger.info(f"Project root: {self.project_root}")
            
            # Load configuration
            self.logger.info("Loading configuration...")
            self.config = ConfigurationManager()
            
            if not self.config.validate():
                self.logger.error("Configuration validation failed")
                return False
            
            # Initialize core application
            self.logger.info("Initializing core application...")
            self.core = DocuBotCore(config=self.config, logger=self.logger)
            
            if not self.core.initialize():
                self.logger.warning("Core initialization had issues, but continuing...")
            
            # Check system requirements
            self.logger.info("Checking system requirements...")
            system_check = self.check_system_requirements()
            
            if not system_check["success"]:
                self.logger.warning(f"System check warnings: {system_check.get('warnings', [])}")
                if system_check.get("critical_errors"):
                    self.logger.error(f"Critical errors: {system_check['critical_errors']}")
                    return False
            
            self.logger.info("Application initialization completed")
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Initialization error: {e}")
                self.logger.debug(traceback.format_exc())
            else:
                print(f"Initialization Error: {e}", file=sys.stderr)
                print(traceback.format_exc())
            return False
    
    def check_system_requirements(self) -> Dict[str, Any]:
        """
        Check if system meets minimum requirements.
        
        Returns:
            Dict containing success status, warnings, and critical errors
        """
        result = {
            "success": True,
            "warnings": [],
            "critical_errors": []
        }
        
        try:
            import psutil
            
            # Check Python version
            python_version = sys.version_info
            if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 11):
                result["critical_errors"].append(f"Python 3.11+ required, found {python_version.major}.{python_version.minor}")
                result["success"] = False
            
            # Check available memory
            memory = psutil.virtual_memory()
            if memory.total < 8 * 1024**3:  # Less than 8GB
                result["warnings"].append(f"Low memory: {memory.total // 1024**3}GB available, 8GB recommended")
            
            # Check disk space
            disk = psutil.disk_usage(str(self.project_root))
            if disk.free < 5 * 1024**3:  # Less than 5GB free
                result["warnings"].append(f"Low disk space: {disk.free // 1024**3}GB free, 5GB recommended")
            
        except ImportError:
            result["warnings"].append("psutil not available, limited system check")
        
        return result
    
    def run_cli(self) -> int:
        """
        Run the application in CLI mode.
        
        Returns:
            int: Exit code (0 for success, non-zero for failure)
        """
        self.logger.info("Starting DocuBot in CLI mode...")
        
        try:
            self.print_banner()
            
            # Simple CLI interface
            print("\n" + "="*60)
            print("DocuBot Command Line Interface")
            print("="*60)
            
            # Show application status
            status = self.core.get_status()
            print(f"\nApplication Status:")
            print(f"  Initialized: {status['initialized']}")
            print(f"  Components loaded: {len(status['components_loaded'])}")
            print(f"  Version: {status['config']['version']}")
            
            if status['warnings']:
                print(f"\nWarnings:")
                for warning in status['warnings']:
                    print(f"  ⚠ {warning}")
            
            # Show available commands
            print("\nAvailable Commands:")
            print("  1. Process a document")
            print("  2. Ask a question")
            print("  3. Show system status")
            print("  4. Exit")
            
            while True:
                try:
                    choice = input("\nEnter choice (1-4): ").strip()
                    
                    if choice == "1":
                        file_path = input("Enter document path: ").strip()
                        if file_path and os.path.exists(file_path):
                            result = self.core.process_document(file_path)
                            print(f"\nResult: {json.dumps(result, indent=2)}")
                        else:
                            print("Error: File not found")
                    
                    elif choice == "2":
                        question = input("Enter your question: ").strip()
                        if question:
                            result = self.core.ask_question(question)
                            print(f"\nAnswer: {result.get('answer', 'No answer')}")
                            if result.get('sources'):
                                print(f"\nSources: {result['sources']}")
                        else:
                            print("Error: Question cannot be empty")
                    
                    elif choice == "3":
                        status = self.core.get_status()
                        print(f"\nSystem Status:")
                        print(json.dumps(status, indent=2))
                    
                    elif choice == "4":
                        print("\nExiting CLI...")
                        return 0
                    
                    else:
                        print("Invalid choice. Please enter 1-4.")
                        
                except KeyboardInterrupt:
                    print("\n\nInterrupted by user")
                    return 130
                except Exception as e:
                    print(f"Error: {e}")
            
        except Exception as e:
            self.logger.error(f"CLI execution error: {e}")
            self.logger.debug(traceback.format_exc())
            return 1
    
    def run_desktop(self) -> int:
        """
        Run the application in desktop GUI mode.
        
        Returns:
            int: Exit code (0 for success, non-zero for failure)
        """
        self.logger.info("Starting DocuBot in desktop mode...")
        
        try:
            # Try to import GUI modules
            try:
                import customtkinter as ctk
            except ImportError:
                self.logger.error("CustomTkinter not installed. Installing...")
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "customtkinter"])
                    import customtkinter as ctk
                except Exception as e:
                    self.logger.error(f"Failed to install CustomTkinter: {e}")
                    print("CustomTkinter is required for desktop mode.")
                    print("Install it with: pip install customtkinter")
                    return 1
            
            # Setup CustomTkinter
            ctk.set_appearance_mode("dark")
            ctk.set_default_color_theme("blue")
            
            # Create main window
            root = ctk.CTk()
            root.title("DocuBot - Local AI Assistant")
            root.geometry("1200x800")
            
            # Create main frame
            main_frame = ctk.CTkFrame(root)
            main_frame.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Header
            header = ctk.CTkLabel(
                main_frame,
                text="DocuBot - Local AI Assistant",
                font=("Arial", 24, "bold")
            )
            header.pack(pady=20)
            
            # Status frame
            status_frame = ctk.CTkFrame(main_frame)
            status_frame.pack(fill="x", padx=20, pady=10)
            
            status = self.core.get_status()
            status_text = f"Status: {'Ready' if status['initialized'] else 'Initializing'}"
            status_label = ctk.CTkLabel(status_frame, text=status_text, font=("Arial", 14))
            status_label.pack(pady=10)
            
            # Components frame
            components_frame = ctk.CTkFrame(main_frame)
            components_frame.pack(fill="x", padx=20, pady=10)
            
            components_label = ctk.CTkLabel(components_frame, text="Loaded Components:", font=("Arial", 12, "bold"))
            components_label.pack(anchor="w", padx=10, pady=5)
            
            for component in status.get('components_loaded', []):
                comp_label = ctk.CTkLabel(components_frame, text=f"• {component}")
                comp_label.pack(anchor="w", padx=30, pady=2)
            
            # Warning frame
            if status.get('warnings'):
                warning_frame = ctk.CTkFrame(main_frame, fg_color="#FFA500")
                warning_frame.pack(fill="x", padx=20, pady=10)
                
                warning_label = ctk.CTkLabel(warning_frame, text="Warnings:", font=("Arial", 12, "bold"))
                warning_label.pack(anchor="w", padx=10, pady=5)
                
                for warning in status['warnings'][:3]:  # Show first 3 warnings
                    warn_label = ctk.CTkLabel(warning_frame, text=f"⚠ {warning}")
                    warn_label.pack(anchor="w", padx=30, pady=2)
            
            # Button frame
            button_frame = ctk.CTkFrame(main_frame)
            button_frame.pack(pady=30)
            
            # Buttons
            def open_document():
                from tkinter import filedialog
                file_path = filedialog.askopenfilename(
                    title="Select Document",
                    filetypes=[
                        ("All supported files", "*.pdf *.docx *.txt *.epub *.md *.html"),
                        ("PDF files", "*.pdf"),
                        ("Word documents", "*.docx"),
                        ("Text files", "*.txt"),
                        ("eBooks", "*.epub"),
                        ("Markdown", "*.md"),
                        ("HTML", "*.html")
                    ]
                )
                if file_path:
                    result = self.core.process_document(file_path)
                    print(f"Document processing result: {result}")
            
            def open_chat():
                # Create chat window
                chat_window = ctk.CTkToplevel(root)
                chat_window.title("DocuBot Chat")
                chat_window.geometry("800x600")
                
                chat_frame = ctk.CTkFrame(chat_window)
                chat_frame.pack(fill="both", expand=True, padx=10, pady=10)
                
                chat_label = ctk.CTkLabel(chat_frame, text="Chat with your documents", font=("Arial", 16))
                chat_label.pack(pady=10)
                
                chat_display = ctk.CTkTextbox(chat_frame, height=400)
                chat_display.pack(fill="both", expand=True, padx=10, pady=10)
                chat_display.insert("1.0", "Welcome to DocuBot Chat!\n\nAsk questions about your documents here.\n\n")
                
                input_frame = ctk.CTkFrame(chat_frame)
                input_frame.pack(fill="x", padx=10, pady=10)
                
                question_entry = ctk.CTkEntry(input_frame, placeholder_text="Type your question here...")
                question_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))
                
                def send_question():
                    question = question_entry.get()
                    if question:
                        chat_display.insert("end", f"\n\nYou: {question}\n")
                        question_entry.delete(0, "end")
                        
                        # Get answer
                        result = self.core.ask_question(question)
                        answer = result.get('answer', 'No answer available')
                        chat_display.insert("end", f"DocuBot: {answer}\n")
                        chat_display.see("end")
                
                send_button = ctk.CTkButton(input_frame, text="Send", command=send_question)
                send_button.pack(side="right")
            
            # Add document button
            add_doc_btn = ctk.CTkButton(
                button_frame,
                text="Add Document",
                command=open_document,
                width=200,
                height=40
            )
            add_doc_btn.pack(pady=10)
            
            # Chat button
            chat_btn = ctk.CTkButton(
                button_frame,
                text="Chat with Documents",
                command=open_chat,
                width=200,
                height=40
            )
            chat_btn.pack(pady=10)
            
            # Exit button
            def exit_app():
                self.shutdown()
                root.quit()
            
            exit_btn = ctk.CTkButton(
                button_frame,
                text="Exit",
                command=exit_app,
                width=200,
                height=40,
                fg_color="#FF5555"
            )
            exit_btn.pack(pady=10)
            
            # Footer
            footer = ctk.CTkLabel(
                main_frame,
                text=f"Version {status['config']['version']} | Status: {len(status['components_loaded'])}/{4} components loaded",
                font=("Arial", 10)
            )
            footer.pack(side="bottom", pady=10)
            
            # Start main loop
            self.logger.info("Starting main event loop...")
            self.running = True
            root.mainloop()
            
            self.logger.info("Desktop application terminated")
            return 0
            
        except Exception as e:
            self.logger.error(f"Desktop execution error: {e}")
            self.logger.debug(traceback.format_exc())
            return 1
    
    def run(self, mode: str = "desktop") -> int:
        """
        Main application entry point.
        
        Args:
            mode: Execution mode - "desktop", "cli", or "web"
            
        Returns:
            int: Exit code (0 for success, non-zero for failure)
        """
        try:
            # Initialize application
            if not self.initialize():
                return 1
            
            self.logger.info(f"Starting DocuBot in {mode} mode")
            
            # Run in specified mode
            if mode == "cli":
                return self.run_cli()
            elif mode == "web":
                return self.run_web()
            elif mode == "desktop":
                return self.run_desktop()
            else:
                self.logger.error(f"Unknown mode: {mode}")
                print(f"Error: Unknown mode '{mode}'. Use 'desktop', 'cli', or 'web'.", file=sys.stderr)
                return 1
                
        except KeyboardInterrupt:
            if self.logger:
                self.logger.info("Application interrupted by user")
            return 130
        except Exception as e:
            if self.logger:
                self.logger.error(f"Unexpected error: {e}")
                self.logger.debug(traceback.format_exc())
            return 1
        finally:
            self.shutdown()
    
    def run_web(self) -> int:
        """
        Run the application in web mode.
        
        Returns:
            int: Exit code (0 for success, non-zero for failure)
        """
        self.logger.info("Starting DocuBot in web mode...")
        
        try:
            # Check for Streamlit
            try:
                import streamlit
            except ImportError:
                self.logger.error("Streamlit not installed")
                print("Streamlit is required for web mode.")
                print("Install it with: pip install streamlit")
                return 1
            
            # Launch Streamlit app
            web_app_path = self.project_root / "src" / "ui" / "web" / "app.py"
            if web_app_path.exists():
                self.logger.info("Launching Streamlit web interface...")
                subprocess.run(["streamlit", "run", str(web_app_path)])
                return 0
            else:
                self.logger.error(f"Web app not found at {web_app_path}")
                print("Web interface not yet implemented.")
                return 1
                
        except Exception as e:
            self.logger.error(f"Web execution error: {e}")
            self.logger.debug(traceback.format_exc())
            return 1
    
    def shutdown(self):
        """Gracefully shutdown the application."""
        if not self.running:
            return
        
        self.running = False
        if self.logger:
            self.logger.info("Initiating application shutdown...")
        
        try:
            # Shutdown core components
            if self.core:
                self.core.shutdown()
            
            if self.logger:
                self.logger.info("Application shutdown completed")
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Error during shutdown: {e}")
    
    def print_banner(self):
        """Print application banner."""
        banner = f"""
{'='*60}
              DocuBot - Local AI Assistant
                  Version 1.0.0
{'='*60}

Features:
• 100% Local & Private
• Multiple Document Formats
• Intelligent Document Processing
• Local AI (Ollama Integration)
• Vector Search with ChromaDB
• Cross-Platform Support

System: {platform.system()} {platform.release()}
Python: {platform.python_version()}

{'='*60}
        """
        print(banner)


def main():
    """
    Main entry point for DocuBot application.
    
    Command line usage:
        python app.py [mode] [options]
    
    Modes:
        desktop    Launch desktop GUI (default)
        cli        Launch command line interface
        web        Launch web interface
    
    Options:
        --help     Show this help message
        --version  Show version information
        --config   Specify custom config file
    
    Examples:
        python app.py
        python app.py desktop
        python app.py cli
        python app.py web --config /path/to/config.yaml
    """
    
    parser = argparse.ArgumentParser(
        description="DocuBot - Local AI Knowledge Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py                    # Launch desktop GUI
  python app.py desktop            # Launch desktop GUI
  python app.py cli                # Launch command line interface
  python app.py web                # Launch web interface
  python app.py --config custom.yaml  # Use custom configuration
        """
    )
    
    parser.add_argument(
        "mode",
        nargs="?",
        default="desktop",
        choices=["desktop", "cli", "web"],
        help="Execution mode (default: desktop)"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Path to custom configuration file"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="DocuBot 1.0.0"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Create and run application
    app = DocuBotApplication()
    
    # Run application
    exit_code = app.run(args.mode)
    
    # Exit with appropriate code
    sys.exit(exit_code)


if __name__ == "__main__":
    main()