#!/usr/bin/env python3
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
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
import yaml

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


@dataclass
class ApplicationStatus:
    """Application status tracking."""
    initialized: bool = False
    startup_time: float = 0.0
    components_loaded: List[str] = None
    error_count: int = 0
    warnings: List[str] = None
    critical_errors: List[str] = None
    
    def __post_init__(self):
        if self.components_loaded is None:
            self.components_loaded = []
        if self.warnings is None:
            self.warnings = []
        if self.critical_errors is None:
            self.critical_errors = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class ConfigurationManager:
    """Application configuration management."""
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize configuration manager."""
        self.config_path = config_path or project_root / "data" / "config" / "app_config.yaml"
        self.config = self.load_default_config()
        self.load_config()
    
    def load_default_config(self) -> Dict[str, Any]:
        """Load default application configuration."""
        return {
            "application": {
                "name": "DocuBot",
                "version": "1.0.0",
                "debug_mode": False,
                "log_level": "INFO",
                "mode": "desktop"
            },
            "directories": {
                "data": str(project_root / "data"),
                "models": str(project_root / "data" / "models"),
                "documents": str(project_root / "data" / "documents"),
                "database": str(project_root / "data" / "database"),
                "logs": str(project_root / "data" / "logs"),
                "exports": str(project_root / "data" / "exports")
            },
            "document_processing": {
                "chunk_size": 500,
                "chunk_overlap": 50,
                "max_file_size_mb": 100,
                "supported_formats": [".pdf", ".docx", ".txt", ".epub", ".md", ".html"],
                "ocr_enabled": True
            },
            "artificial_intelligence": {
                "language_model": {
                    "provider": "ollama",
                    "model": "llama2:7b",
                    "temperature": 0.1,
                    "max_tokens": 1024
                },
                "embeddings": {
                    "model": "all-MiniLM-L6-v2",
                    "dimensions": 384
                },
                "retrieval_augmented_generation": {
                    "top_k": 5,
                    "similarity_threshold": 0.7
                }
            },
            "user_interface": {
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
        """Load configuration from file system."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    file_config = yaml.safe_load(f)
                
                self.merge_configurations(self.config, file_config)
                print(f"Configuration loaded from {self.config_path}")
            else:
                print(f"Configuration file not found at {self.config_path}, using defaults")
        except Exception as e:
            print(f"Configuration loading error: {e}")
    
    def merge_configurations(self, base: Dict, overlay: Dict):
        """Recursively merge configuration dictionaries."""
        for key, value in overlay.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self.merge_configurations(base[key], value)
            else:
                base[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve configuration value using dot notation."""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def save(self):
        """Persist configuration to file system."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            print(f"Configuration saved to {self.config_path}")
            return True
        except Exception as e:
            print(f"Configuration save error: {e}")
            return False
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate configuration integrity."""
        validation_errors = []
        
        required_paths = [
            "directories.data",
            "directories.documents",
            "directories.logs"
        ]
        
        for path_key in required_paths:
            path_str = self.get(path_key)
            if not path_str:
                validation_errors.append(f"Missing required configuration: {path_key}")
        
        return len(validation_errors) == 0, validation_errors


class ApplicationLogger:
    """Unified application logging system."""
    
    def __init__(self, name: str = "docubot", log_dir: Optional[Path] = None, level: str = "INFO"):
        """Initialize application logger."""
        self.name = name
        self.log_dir = log_dir or project_root / "data" / "logs"
        self.level = level
        self.log_file = self.log_dir / "application.log"
        self.setup_logging_infrastructure()
    
    def setup_logging_infrastructure(self):
        """Establish logging directory and file structure."""
        try:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Log directory creation error: {e}")
    
    def log_message(self, level: str, message: str, **kwargs):
        """Log a message with specified severity level."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"{timestamp} - {level.upper()} - {message}"
        
        if kwargs:
            log_message += f" - {kwargs}"
        
        # Console output
        print(log_message)
        
        # File persistence
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(log_message + "\n")
        except Exception:
            pass
    
    def information(self, message: str, **kwargs):
        """Log informational message."""
        self.log_message("INFO", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.log_message("ERROR", message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.log_message("WARNING", message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.log_message("DEBUG", message, **kwargs)


class DocuBotCore:
    """Main application orchestration engine."""
    
    def __init__(self, config: ConfigurationManager, logger: ApplicationLogger):
        """Initialize core application engine."""
        self.config = config
        self.logger = logger
        self.status = ApplicationStatus()
        self.components = {}
        self.startup_time = time.time()
    
    def initialize(self) -> bool:
        """Initialize all core application components."""
        try:
            self.logger.information("Initializing DocuBot Core engine...")
            
            # Establish directory structure
            self.create_required_directories()
            
            # Component initialization sequence
            component_initialization = [
                ("database", self.initialize_database_system),
                ("vector_store", self.initialize_vector_storage),
                ("document_processor", self.initialize_document_processing),
                ("artificial_intelligence", self.initialize_ai_components)
            ]
            
            for component_name, initialization_method in component_initialization:
                try:
                    success = initialization_method()
                    if success:
                        self.status.components_loaded.append(component_name)
                        self.logger.information(f"Component '{component_name}' initialization successful")
                    else:
                        self.status.warnings.append(f"Component '{component_name}' initialization incomplete")
                except Exception as e:
                    self.status.warnings.append(f"Component '{component_name}' initialization error: {str(e)}")
                    self.logger.error(f"Component '{component_name}' initialization failure: {e}")
            
            # Calculate initialization metrics
            self.status.startup_time = time.time() - self.startup_time
            self.status.initialized = len(self.status.components_loaded) > 0
            
            if self.status.initialized:
                self.logger.information(f"DocuBot Core initialization completed in {self.status.startup_time:.2f} seconds")
                self.logger.information(f"Active components: {', '.join(self.status.components_loaded)}")
            else:
                self.logger.warning("DocuBot Core partial initialization - some components unavailable")
            
            return self.status.initialized
            
        except Exception as e:
            self.logger.error(f"DocuBot Core initialization failure: {e}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def create_required_directories(self):
        """Create necessary directory structure."""
        directories_to_create = [
            self.config.get("directories.data"),
            self.config.get("directories.models"),
            self.config.get("directories.documents"),
            self.config.get("directories.database"),
            self.config.get("directories.logs"),
            self.config.get("directories.exports"),
        ]
        
        for directory_path in directories_to_create:
            if directory_path:
                try:
                    Path(directory_path).mkdir(parents=True, exist_ok=True)
                    self.logger.debug(f"Directory created: {directory_path}")
                except Exception as e:
                    self.logger.warning(f"Directory creation error {directory_path}: {e}")
    
    def initialize_database_system(self) -> bool:
        """Initialize database connectivity."""
        self.logger.information("Initializing database system...")
        database_path = Path(self.config.get("directories.database")) / "sqlite.db"
        try:
            import sqlite3
            connection = sqlite3.connect(database_path)
            connection.execute("SELECT 1")
            connection.close()
            self.components["database"] = {"type": "sqlite", "path": str(database_path)}
            return True
        except Exception as e:
            self.logger.warning(f"Database system initialization failure: {e}")
            return False
    
    def initialize_vector_storage(self) -> bool:
        """Initialize vector storage infrastructure."""
        self.logger.information("Initializing vector storage...")
        vector_store_directory = Path(self.config.get("directories.database")) / "chroma"
        try:
            vector_store_directory.mkdir(parents=True, exist_ok=True)
            self.components["vector_store"] = {"type": "chromadb", "path": str(vector_store_directory)}
            return True
        except Exception as e:
            self.logger.warning(f"Vector storage initialization failure: {e}")
            return False
    
    def initialize_document_processing(self) -> bool:
        """Initialize document processing pipeline."""
        self.logger.information("Initializing document processing system...")
        try:
            import re
            import chardet
            self.components["document_processor"] = {"status": "available"}
            return True
        except ImportError as e:
            self.logger.warning(f"Document processing dependencies unavailable: {e}")
            return False
    
    def initialize_ai_components(self) -> bool:
        """Initialize artificial intelligence components."""
        self.logger.information("Initializing AI components...")
        ai_component_status = {"language_model": "unavailable", "embeddings": "unavailable"}
        
        # Verify Ollama availability
        try:
            result = subprocess.run(["ollama", "--version"], capture_output=True, text=True, check=False)
            if result.returncode == 0:
                ai_component_status["language_model"] = "available"
                self.logger.information(f"Ollama detected: {result.stdout.strip()}")
        except (FileNotFoundError, subprocess.SubprocessError):
            self.logger.warning("Ollama not available - language model features disabled")
        
        # Verify sentence transformers
        try:
            import sentence_transformers
            ai_component_status["embeddings"] = "available"
            self.logger.information("Sentence transformers available")
        except ImportError:
            self.logger.warning("Sentence transformers unavailable - embedding features disabled")
        
        self.components["artificial_intelligence"] = ai_component_status
        return ai_component_status["language_model"] == "available" or ai_component_status["embeddings"] == "available"
    
    def process_document(self, file_path: str) -> Dict[str, Any]:
        """
        Process document file through ingestion pipeline.
        
        Args:
            file_path: Path to document file
            
        Returns:
            Processing results dictionary
        """
        if not self.status.initialized:
            return {
                "status": "error",
                "message": "DocuBot Core not initialized",
                "file_path": file_path
            }
        
        self.logger.information(f"Document processing initiated: {file_path}")
        
        try:
            # Document validation
            path = Path(file_path)
            if not path.exists():
                return {
                    "status": "error",
                    "message": f"File not found: {file_path}",
                    "file_path": file_path
                }
            
            # Format validation
            file_extension = path.suffix.lower()
            supported_formats = self.config.get("document_processing.supported_formats", [])
            
            if file_extension not in supported_formats:
                return {
                    "status": "error",
                    "message": f"Unsupported file format: {file_extension}. Supported formats: {supported_formats}",
                    "file_path": file_path
                }
            
            # Document processing placeholder
            return {
                "status": "success",
                "message": "Document queued for processing",
                "file_path": file_path,
                "file_size": path.stat().st_size,
                "file_type": file_extension
            }
            
        except Exception as e:
            self.logger.error(f"Document processing error {file_path}: {e}")
            return {
                "status": "error",
                "message": f"Processing error: {str(e)}",
                "file_path": file_path
            }
    
    def query_documents(self, question: str, context: Dict = None) -> Dict[str, Any]:
        """
        Process document query using retrieval augmented generation.
        
        Args:
            question: Query text
            context: Additional query context
            
        Returns:
            Query response dictionary
        """
        if not self.status.initialized:
            return {
                "status": "error",
                "answer": "DocuBot Core not initialized",
                "sources": [],
                "confidence": 0.0
            }
        
        self.logger.information(f"Document query received: {question}")
        
        # AI component availability check
        ai_status = self.components.get("artificial_intelligence", {})
        
        if ai_status.get("language_model") != "available":
            return {
                "status": "warning",
                "answer": "Language model features unavailable. Install Ollama to enable AI question answering.",
                "sources": [],
                "confidence": 0.0
            }
        
        # Retrieval augmented generation placeholder
        return {
            "status": "success",
            "answer": f"Query response placeholder for: '{question}'\n\nAI question answering functionality requires completion of AI integration tasks.",
            "sources": ["System placeholder"],
            "confidence": 0.3,
            "processing_time": 0.1
        }
    
    def get_application_status(self) -> Dict[str, Any]:
        """Retrieve comprehensive application status."""
        return {
            **self.status.to_dict(),
            "timestamp": datetime.now().isoformat(),
            "components": self.components,
            "configuration": {
                "application_name": self.config.get("application.name"),
                "version": self.config.get("application.version"),
                "debug_mode": self.config.get("application.debug_mode")
            }
        }
    
    def shutdown(self):
        """Gracefully terminate all core components."""
        self.logger.information("Initiating DocuBot Core shutdown...")
        self.status.initialized = False
        self.components.clear()
        self.logger.information("DocuBot Core shutdown complete")


class DocuBotApplication:
    """Primary application orchestrator and lifecycle manager."""
    
    def __init__(self):
        """Initialize DocuBot application."""
        self.project_root = Path(__file__).parent
        self.configuration = None
        self.core = None
        self.logger = None
        self.application_running = False
        
        # Graceful shutdown signal handlers
        signal.signal(signal.SIGINT, self.handle_termination_signal)
        signal.signal(signal.SIGTERM, self.handle_termination_signal)
    
    def handle_termination_signal(self, signal_number, frame):
        """Handle application termination signals gracefully."""
        if self.logger:
            self.logger.information(f"Termination signal received: {signal_number}")
        self.shutdown()
        sys.exit(0)
    
    def initialize(self) -> bool:
        """
        Initialize complete application infrastructure.
        
        Returns:
            bool: Initialization success status
        """
        try:
            # Initialize logging system
            self.logger = ApplicationLogger(
                name="docubot",
                log_dir=self.project_root / "data" / "logs",
                level="INFO"
            )
            self.logger.information("Initializing DocuBot application...")
            self.logger.information(f"Project root directory: {self.project_root}")
            
            # Load application configuration
            self.logger.information("Loading application configuration...")
            self.configuration = ConfigurationManager()
            
            validation_success, validation_errors = self.configuration.validate()
            if not validation_success:
                self.logger.error("Configuration validation failed")
                for error in validation_errors:
                    self.logger.error(f"Validation error: {error}")
                return False
            
            # Initialize core application engine
            self.logger.information("Initializing application core...")
            self.core = DocuBotCore(config=self.configuration, logger=self.logger)
            
            if not self.core.initialize():
                self.logger.warning("Core initialization issues detected, continuing execution...")
            
            # System requirements verification
            self.logger.information("Verifying system requirements...")
            system_verification = self.verify_system_requirements()
            
            if not system_verification["success"]:
                self.logger.warning(f"System verification warnings: {system_verification.get('warnings', [])}")
                if system_verification.get("critical_errors"):
                    self.logger.error(f"Critical system errors: {system_verification['critical_errors']}")
                    return False
            
            self.logger.information("Application initialization completed")
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Initialization error: {e}")
                self.logger.debug(traceback.format_exc())
            else:
                print(f"Initialization Error: {e}", file=sys.stderr)
                print(traceback.format_exc())
            return False
    
    def verify_system_requirements(self) -> Dict[str, Any]:
        """
        Verify minimum system requirements.
        
        Returns:
            System verification results
        """
        verification_result = {
            "success": True,
            "warnings": [],
            "critical_errors": []
        }
        
        try:
            import psutil
            
            # Python version verification
            python_version = sys.version_info
            if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 11):
                verification_result["critical_errors"].append(f"Python 3.11+ required, current version: {python_version.major}.{python_version.minor}")
                verification_result["success"] = False
            
            # Memory availability check
            system_memory = psutil.virtual_memory()
            if system_memory.total < 8 * 1024**3:  # Minimum 8GB
                verification_result["warnings"].append(f"Insufficient memory: {system_memory.total // 1024**3}GB available, 8GB recommended")
            
            # Disk space verification
            disk_usage = psutil.disk_usage(str(self.project_root))
            if disk_usage.free < 5 * 1024**3:  # Minimum 5GB free
                verification_result["warnings"].append(f"Insufficient disk space: {disk_usage.free // 1024**3}GB free, 5GB recommended")
            
        except ImportError:
            verification_result["warnings"].append("psutil unavailable, limited system verification")
        
        return verification_result
    
    def execute_command_line_interface(self) -> int:
        """
        Execute command line interface mode.
        
        Returns:
            int: Exit status code
        """
        self.logger.information("Starting command line interface...")
        
        try:
            self.display_application_header()
            
            print("\n" + "="*60)
            print("DocuBot Command Line Interface")
            print("="*60)
            
            # Display application status
            application_status = self.core.get_application_status()
            print(f"\nApplication Status:")
            print(f"  Initialization: {application_status['initialized']}")
            print(f"  Active Components: {len(application_status['components_loaded'])}")
            print(f"  Version: {application_status['configuration']['version']}")
            
            if application_status['warnings']:
                print(f"\nSystem Warnings:")
                for warning in application_status['warnings']:
                    print(f"  Warning: {warning}")
            
            # Command interface
            print("\nAvailable Commands:")
            print("  1. Process Document")
            print("  2. Query Documents")
            print("  3. Display System Status")
            print("  4. Terminate Application")
            
            while True:
                try:
                    user_selection = input("\nEnter selection (1-4): ").strip()
                    
                    if user_selection == "1":
                        document_path = input("Document file path: ").strip()
                        if document_path and os.path.exists(document_path):
                            processing_result = self.core.process_document(document_path)
                            print(f"\nProcessing Result: {json.dumps(processing_result, indent=2)}")
                        else:
                            print("Error: File not found")
                    
                    elif user_selection == "2":
                        user_query = input("Enter query: ").strip()
                        if user_query:
                            query_result = self.core.query_documents(user_query)
                            print(f"\nResponse: {query_result.get('answer', 'No response available')}")
                            if query_result.get('sources'):
                                print(f"\nSources: {query_result['sources']}")
                        else:
                            print("Error: Query cannot be empty")
                    
                    elif user_selection == "3":
                        status = self.core.get_application_status()
                        print(f"\nSystem Status:")
                        print(json.dumps(status, indent=2))
                    
                    elif user_selection == "4":
                        print("\nTerminating command line interface...")
                        return 0
                    
                    else:
                        print("Invalid selection. Enter 1-4.")
                        
                except KeyboardInterrupt:
                    print("\nInterface terminated by user")
                    return 130
                except Exception as e:
                    print(f"Error: {e}")
            
        except Exception as e:
            self.logger.error(f"Command line interface execution error: {e}")
            self.logger.debug(traceback.format_exc())
            return 1
    
    def execute_desktop_interface(self) -> int:
        """
        Execute desktop graphical interface mode.
        
        Returns:
            int: Exit status code
        """
        self.logger.information("Initializing desktop interface...")
        
        try:
            # Desktop interface dependencies
            try:
                import customtkinter as ctk
            except ImportError:
                self.logger.error("CustomTkinter unavailable")
                print("CustomTkinter required for desktop interface.")
                print("Installation command: pip install customtkinter")
                return 1
            
            # Interface configuration
            ctk.set_appearance_mode("dark")
            ctk.set_default_color_theme("blue")
            
            # Primary application window
            primary_window = ctk.CTk()
            primary_window.title("DocuBot - Local AI Assistant")
            primary_window.geometry("1200x800")
            
            # Main interface container
            main_container = ctk.CTkFrame(primary_window)
            main_container.pack(fill="both", expand=True, padx=10, pady=10)
            
            # Application header
            application_header = ctk.CTkLabel(
                main_container,
                text="DocuBot - Local AI Assistant",
                font=("Arial", 24, "bold")
            )
            application_header.pack(pady=20)
            
            # Status display
            status_container = ctk.CTkFrame(main_container)
            status_container.pack(fill="x", padx=20, pady=10)
            
            application_status = self.core.get_application_status()
            status_display = f"Status: {'Operational' if application_status['initialized'] else 'Initializing'}"
            status_indicator = ctk.CTkLabel(status_container, text=status_display, font=("Arial", 14))
            status_indicator.pack(pady=10)
            
            # Component information
            component_container = ctk.CTkFrame(main_container)
            component_container.pack(fill="x", padx=20, pady=10)
            
            component_label = ctk.CTkLabel(component_container, text="Active Components:", font=("Arial", 12, "bold"))
            component_label.pack(anchor="w", padx=10, pady=5)
            
            for component in application_status.get('components_loaded', []):
                component_display = ctk.CTkLabel(component_container, text=f"• {component}")
                component_display.pack(anchor="w", padx=30, pady=2)
            
            # Warning display
            if application_status.get('warnings'):
                warning_container = ctk.CTkFrame(main_container, fg_color="#FFA500")
                warning_container.pack(fill="x", padx=20, pady=10)
                
                warning_label = ctk.CTkLabel(warning_container, text="System Warnings:", font=("Arial", 12, "bold"))
                warning_label.pack(anchor="w", padx=10, pady=5)
                
                for warning in application_status['warnings'][:3]:
                    warning_display = ctk.CTkLabel(warning_container, text=f"Warning: {warning}")
                    warning_display.pack(anchor="w", padx=30, pady=2)
            
            # Function controls
            control_container = ctk.CTkFrame(main_container)
            control_container.pack(pady=30)
            
            # Document processing control
            def initiate_document_processing():
                from tkinter import filedialog
                selected_file = filedialog.askopenfilename(
                    title="Select Document",
                    filetypes=[
                        ("All supported formats", "*.pdf *.docx *.txt *.epub *.md *.html"),
                        ("PDF documents", "*.pdf"),
                        ("Word documents", "*.docx"),
                        ("Text documents", "*.txt"),
                        ("eBook files", "*.epub"),
                        ("Markdown documents", "*.md"),
                        ("HTML documents", "*.html")
                    ]
                )
                if selected_file:
                    processing_result = self.core.process_document(selected_file)
                    print(f"Document processing result: {processing_result}")
            
            # Document interface control
            def initiate_document_interface():
                interface_window = ctk.CTkToplevel(primary_window)
                interface_window.title("DocuBot Document Interface")
                interface_window.geometry("800x600")
                
                interface_container = ctk.CTkFrame(interface_window)
                interface_container.pack(fill="both", expand=True, padx=10, pady=10)
                
                interface_label = ctk.CTkLabel(interface_container, text="Document Query Interface", font=("Arial", 16))
                interface_label.pack(pady=10)
                
                text_display = ctk.CTkTextbox(interface_container, height=400)
                text_display.pack(fill="both", expand=True, padx=10, pady=10)
                text_display.insert("1.0", "Document Query Interface\n\nSubmit queries regarding your documents.\n\n")
                
                input_container = ctk.CTkFrame(interface_container)
                input_container.pack(fill="x", padx=10, pady=10)
                
                query_input = ctk.CTkEntry(input_container, placeholder_text="Enter document query...")
                query_input.pack(side="left", fill="x", expand=True, padx=(0, 10))
                
                def submit_query():
                    query_text = query_input.get()
                    if query_text:
                        text_display.insert("end", f"\n\nQuery: {query_text}\n")
                        query_input.delete(0, "end")
                        
                        query_response = self.core.query_documents(query_text)
                        response_text = query_response.get('answer', 'No response available')
                        text_display.insert("end", f"Response: {response_text}\n")
                        text_display.see("end")
                
                submit_button = ctk.CTkButton(input_container, text="Submit", command=submit_query)
                submit_button.pack(side="right")
            
            # Control buttons
            document_button = ctk.CTkButton(
                control_container,
                text="Process Document",
                command=initiate_document_processing,
                width=200,
                height=40
            )
            document_button.pack(pady=10)
            
            query_button = ctk.CTkButton(
                control_container,
                text="Document Query",
                command=initiate_document_interface,
                width=200,
                height=40
            )
            query_button.pack(pady=10)
            
            # Application termination
            def terminate_application():
                self.shutdown()
                primary_window.quit()
            
            termination_button = ctk.CTkButton(
                control_container,
                text="Terminate Application",
                command=terminate_application,
                width=200,
                height=40,
                fg_color="#FF5555"
            )
            termination_button.pack(pady=10)
            
            # Application footer
            footer_text = f"Version {application_status['configuration']['version']} | Active Components: {len(application_status['components_loaded'])}/4"
            application_footer = ctk.CTkLabel(
                main_container,
                text=footer_text,
                font=("Arial", 10)
            )
            application_footer.pack(side="bottom", pady=10)
            
            # Main event loop
            self.logger.information("Starting desktop interface event loop...")
            self.application_running = True
            primary_window.mainloop()
            
            self.logger.information("Desktop interface terminated")
            return 0
            
        except Exception as e:
            self.logger.error(f"Desktop interface execution error: {e}")
            self.logger.debug(traceback.format_exc())
            return 1
    
    def execute_web_interface(self) -> int:
        """
        Execute web interface mode.
        
        Returns:
            int: Exit status code
        """
        self.logger.information("Initializing web interface...")
        
        try:
            # Web interface dependency verification
            try:
                import streamlit
            except ImportError:
                self.logger.error("Streamlit unavailable")
                print("Streamlit required for web interface.")
                print("Installation command: pip install streamlit")
                return 1
            
            # Web application launch
            web_application_path = self.project_root / "src" / "ui" / "web" / "streamlit_app.py"
            if web_application_path.exists():
                self.logger.information("Launching Streamlit web interface...")
                subprocess.run(["streamlit", "run", str(web_application_path)])
                return 0
            else:
                self.logger.error(f"Web application not found: {web_application_path}")
                print("Web interface implementation incomplete.")
                return 1
                
        except Exception as e:
            self.logger.error(f"Web interface execution error: {e}")
            self.logger.debug(traceback.format_exc())
            return 1
    
    def execute(self, execution_mode: str = "desktop") -> int:
        """
        Primary application execution method.
        
        Args:
            execution_mode: Application execution mode
            
        Returns:
            int: Exit status code
        """
        try:
            # Application initialization
            if not self.initialize():
                return 1
            
            self.logger.information(f"Starting DocuBot in {execution_mode} mode")
            
            # Mode-specific execution
            if execution_mode == "cli":
                return self.execute_command_line_interface()
            elif execution_mode == "web":
                return self.execute_web_interface()
            elif execution_mode == "desktop":
                return self.execute_desktop_interface()
            else:
                self.logger.error(f"Invalid execution mode: {execution_mode}")
                print(f"Error: Invalid execution mode '{execution_mode}'. Valid modes: desktop, cli, web", file=sys.stderr)
                return 1
                
        except KeyboardInterrupt:
            if self.logger:
                self.logger.information("Application terminated by user")
            return 130
        except Exception as e:
            if self.logger:
                self.logger.error(f"Execution error: {e}")
                self.logger.debug(traceback.format_exc())
            return 1
        finally:
            self.shutdown()
    
    def shutdown(self):
        """Gracefully terminate application."""
        if not self.application_running:
            return
        
        self.application_running = False
        if self.logger:
            self.logger.information("Initiating application termination...")
        
        try:
            # Core component termination
            if self.core:
                self.core.shutdown()
            
            if self.logger:
                self.logger.information("Application termination complete")
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"Termination error: {e}")
    
    def display_application_header(self):
        """Display application information header."""
        application_header = f"""
{'='*60}
                DocuBot - Local AI Assistant
                     Version 1.0.0
{'='*60}

Features:
• Complete Local Processing
• Multi-format Document Support
• Intelligent Document Processing
• Local AI Integration
• Vector-based Document Search
• Cross-platform Compatibility

System Information:
  Operating System: {platform.system()} {platform.release()}
  Python Version: {platform.python_version()}

{'='*60}
        """
        print(application_header)


def main():
    """
    Primary application entry point.
    
    Command line interface:
        python app.py [mode] [options]
    
    Execution Modes:
        desktop    Desktop graphical interface (default)
        cli        Command line interface
        web        Web browser interface
    
    Command Options:
        --help     Display command help
        --version  Display version information
        --config   Specify custom configuration file
        --verbose  Enable verbose logging
    
    Usage Examples:
        python app.py
        python app.py desktop
        python app.py cli
        python app.py web
        python app.py --config custom_config.yaml
    """
    
    argument_parser = argparse.ArgumentParser(
        description="DocuBot - Local AI Knowledge Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  python app.py                    # Desktop interface
  python app.py desktop            # Desktop interface
  python app.py cli                # Command line interface
  python app.py web                # Web interface
  python app.py --config custom.yaml  # Custom configuration
        """
    )
    
    argument_parser.add_argument(
        "mode",
        nargs="?",
        default="desktop",
        choices=["desktop", "cli", "web"],
        help="Application execution mode (default: desktop)"
    )
    
    argument_parser.add_argument(
        "--config",
        type=str,
        help="Custom configuration file path"
    )
    
    argument_parser.add_argument(
        "--version",
        action="version",
        version="DocuBot 1.0.0"
    )
    
    argument_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging output"
    )
    
    parsed_arguments = argument_parser.parse_args()
    
    # Application instantiation and execution
    application_instance = DocuBotApplication()
    
    # Execute application with specified mode
    exit_status = application_instance.execute(parsed_arguments.mode)
    
    # Application termination
    sys.exit(exit_status)


if __name__ == "__main__":
    main()