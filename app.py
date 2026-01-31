# docubot/app.py

"""
DOCUBOT - Main Application Entry Point
Version: 1.0.0
Description: Local AI document assistant with RAG capabilities
Author: DocuBot Team
License: MIT
"""

import sys
import os
import argparse
import logging
import signal
import platform
import subprocess
import time
import traceback
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict

import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data/logs/docubot_startup.log')
    ]
)
logger = logging.getLogger(__name__)

try:
    sys.path.insert(0, str(Path(__file__).parent))
    
    from src.core.config import AppConfig, ConfigManager
    from src.utilities.logger import setup_logging as setup_project_logging
    from src.core.exceptions import DocuBotError, ConfigurationError
    
    try:
        from src.core.exceptions import DatabaseException
        HAS_DATABASE_EXCEPTION = True
    except ImportError:
        HAS_DATABASE_EXCEPTION = False
        
        class DatabaseException(DocuBotError):
            pass
    
    HAS_SRC_MODULES = True
    logger.info("Successfully imported src modules")
    
except ImportError as e:
    HAS_SRC_MODULES = False
    logger.warning(f"Could not import src modules: {e}")
    logger.info("Running in minimal mode...")
    
    class DocuBotError(Exception):
        pass
    
    class ConfigurationError(DocuBotError):
        pass
    
    class DatabaseException(DocuBotError):
        pass


@dataclass
class SystemInfo:
    python_version: str
    platform: str
    architecture: str
    processor: str
    memory_gb: float
    disk_space_gb: float
    in_virtual_env: bool
    has_gpu: bool


@dataclass
class StartupMetrics:
    start_time: float
    initialization_time: float
    component_load_times: Dict[str, float]
    memory_usage_mb: float
    success: bool
    errors: List[str]


class DocuBotApplication:
    
    def __init__(self):
        self.system_info: Optional[SystemInfo] = None
        self.config: Optional[dict] = None
        self.config_manager: Optional[ConfigManager] = None
        self.metrics = StartupMetrics(
            start_time=0.0,
            initialization_time=0.0,
            component_load_times={},
            memory_usage_mb=0.0,
            success=False,
            errors=[]
        )
        self.logger = logger
        self.mode = "auto"
        self.args = None
        
        self._components_loaded = False
        self._services_running = False
        self._shutdown_requested = False
        self._crash_reports_dir = Path("data/logs/crash_reports")
        
    def setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self.handle_shutdown_signal)
        signal.signal(signal.SIGTERM, self.handle_shutdown_signal)
        
        if platform.system() != "Windows":
            signal.signal(signal.SIGHUP, self.handle_shutdown_signal)
    
    def handle_shutdown_signal(self, signum, frame):
        if self.logger:
            self.logger.info(f"Received shutdown signal {signum}")
        self._shutdown_requested = True
        self.shutdown()
    
    def collect_system_info(self) -> SystemInfo:
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        
        try:
            import psutil
            memory_gb = psutil.virtual_memory().total / (1024**3)
            
            if platform.system() != 'Windows':
                disk_gb = psutil.disk_usage('/').free / (1024**3)
            else:
                disk_gb = psutil.disk_usage('C:\\').free / (1024**3)
        except ImportError:
            memory_gb = 0.0
            disk_gb = 0.0
        
        has_gpu = False
        try:
            import torch
            has_gpu = torch.cuda.is_available()
        except ImportError:
            pass
        
        return SystemInfo(
            python_version=python_version,
            platform=platform.platform(),
            architecture=platform.architecture()[0],
            processor=platform.processor(),
            memory_gb=round(memory_gb, 1),
            disk_space_gb=round(disk_gb, 1),
            in_virtual_env=sys.prefix != sys.base_prefix,
            has_gpu=has_gpu
        )
    
    def validate_system_requirements(self) -> bool:
        self.system_info = self.collect_system_info()
        
        requirements_met = True
        requirements = {
            "Python Version": self.system_info.python_version >= "3.10.0",
            "Memory (4GB min)": self.system_info.memory_gb >= 4.0,
            "Disk Space (5GB min)": self.system_info.disk_space_gb >= 5.0,
        }
        
        for req_name, req_met in requirements.items():
            if not req_met:
                warning_msg = f"Requirement not met: {req_name}"
                self.metrics.errors.append(warning_msg)
                self.logger.warning(warning_msg)
                requirements_met = False
        
        return requirements_met
    
    def validate_python_environment(self) -> bool:
        import importlib.metadata as metadata
        
        required_packages = [
            "torch", "transformers", "chromadb", "customtkinter",
            "sqlalchemy", "pyyaml"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                metadata.version(package)
            except metadata.PackageNotFoundError:
                missing_packages.append(package)
        
        if missing_packages:
            warning_msg = f"Missing packages (can install later): {missing_packages}"
            self.metrics.errors.append(warning_msg)
            self.logger.warning(warning_msg)
        
        return True
    
    def load_configuration(self) -> bool:
        try:
            config_path = Path("data/config/app_config.yaml")
            if not config_path.exists():
                self.logger.info("Creating default configuration...")
                config_path.parent.mkdir(parents=True, exist_ok=True)
                
                default_config = {
                    "app": {
                        "name": "DocuBot",
                        "version": "1.0.0",
                        "environment": "development"
                    },
                    "ui": {
                        "theme": "dark",
                        "font_size": 12
                    },
                    "logging": {
                        "level": "INFO"
                    }
                }
                
                with open(config_path, 'w', encoding='utf-8') as f:
                    yaml.dump(default_config, f, default_flow_style=False)
            
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f) or {}
            
            self.logger.info(f"Configuration loaded from {config_path}")
            return True
            
        except Exception as e:
            error_msg = f"Config loading failed: {str(e)}"
            self.metrics.errors.append(error_msg)
            self.logger.error(error_msg)
            
            self.config = {
                "app": {"name": "DocuBot", "version": "1.0.0"},
                "ui": {"theme": "dark"},
                "logging": {"level": "INFO"}
            }
            return True
    
    def initialize_data_directories(self) -> bool:
        required_dirs = [
            "data/documents",
            "data/logs",
            "data/logs/crash_reports",
            "data/config",
        ]
        
        try:
            for dir_path in required_dirs:
                path = Path(dir_path)
                path.mkdir(parents=True, exist_ok=True)
            
            self.logger.info("Data directories initialized")
            return True
            
        except Exception as e:
            error_msg = f"Directory initialization failed: {str(e)}"
            self.metrics.errors.append(error_msg)
            self.logger.error(error_msg)
            return False
    
    def initialize_logging(self) -> bool:
        try:
            self.logger = logging.getLogger("DocuBot")
            self.logger.setLevel(logging.INFO)
            
            log_dir = Path("data/logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_dir / "docubot.log")
            file_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            
            self.logger.info("Logging system initialized")
            return True
            
        except Exception as e:
            print(f"Failed to initialize logging: {e}")
            self.logger = logging.getLogger("DocuBot")
            self.logger.setLevel(logging.INFO)
            return True
    
    def load_core_components(self) -> bool:
        components_to_load = [
            ("Document Processor", "src.document_processing.processor", "DocumentProcessor"),
            ("Vector Store", "src.vector_store.chroma_client", "ChromaClient"),
            ("Database", "src.database.sqlite_client", "SQLiteClient"),
            ("LLM Engine", "src.ai_engine.llm_client", "LLMClient"),
        ]
        
        loaded_components = {}
        
        for component_name, module_path, class_name in components_to_load:
            try:
                if component_name == "Document Processor" and "base_extractor" in str(sys.exc_info()):
                    from src.mocks import MockDocumentProcessor
                    loaded_components[component_name] = MockDocumentProcessor
                    self.logger.warning(f"Using mock for {component_name}")
                    continue
                    
                if component_name == "LLM Engine" and "get_model_manager" in str(sys.exc_info()):
                    from src.mocks import MockLLMClient
                    loaded_components[component_name] = MockLLMClient
                    self.logger.warning(f"Using mock for {component_name}")
                    continue
                
                module = __import__(module_path, fromlist=[class_name])
                component_class = getattr(module, class_name)
                loaded_components[component_name] = component_class
                self.logger.info(f"Component loaded: {component_name}")
                
            except ImportError as e:
                self.logger.warning(f"Could not load {component_name}: {e}")
                
                if component_name == "Document Processor":
                    from src.mocks import MockDocumentProcessor
                    loaded_components[component_name] = MockDocumentProcessor
                    self.logger.info(f"Using mock for {component_name}")
                elif component_name == "LLM Engine":
                    from src.mocks import MockLLMClient
                    loaded_components[component_name] = MockLLMClient
                    self.logger.info(f"Using mock for {component_name}")
                elif component_name == "Database":
                    from src.mocks import MockSQLiteClient
                    loaded_components[component_name] = MockSQLiteClient
                    self.logger.info(f"Using mock for {component_name}")
                else:
                    loaded_components[component_name] = None
        
        self._components = loaded_components
        self._components_loaded = True
        
        loaded_count = sum(1 for comp in loaded_components.values() if comp is not None)
        self.logger.info(f"Loaded {loaded_count}/{len(components_to_load)} components")
        
        return loaded_count > 0
    
    def save_crash_report(self, exception: Exception) -> Path:
        try:
            crash_data = {
                "timestamp": datetime.now().isoformat(),
                "exception_type": type(exception).__name__,
                "exception_message": str(exception),
                "traceback": traceback.format_exc(),
                "system_info": asdict(self.system_info) if self.system_info else None,
                "application_mode": self.mode,
                "command_line_args": sys.argv
            }
            
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            self._crash_reports_dir.mkdir(parents=True, exist_ok=True)
            crash_path = self._crash_reports_dir / f"crash_{timestamp_str}.json"
            
            with open(crash_path, 'w', encoding='utf-8') as f:
                json.dump(crash_data, f, indent=2, ensure_ascii=False)
            
            return crash_path
            
        except Exception as e:
            self.logger.error(f"Failed to save crash report: {e}")
            return Path("crash_report_failed.txt")
    
    def determine_optimal_mode(self, requested_mode: str) -> str:
        if requested_mode != "auto":
            return requested_mode
        
        if not HAS_SRC_MODULES:
            return "minimal"
        
        if platform.system() == "Windows" or platform.system() == "Darwin":
            return "gui"
        
        if "DISPLAY" in os.environ or "WAYLAND_DISPLAY" in os.environ:
            return "gui"
        
        return "cli"
    
    def run_gui_mode(self) -> int:
        try:
            self.logger.info("Launching GUI mode")
            
            import customtkinter as ctk
            
            try:
                from src.ui.desktop.main_window import MainWindow
            except ImportError as e:
                self.logger.error(f"Cannot import MainWindow: {e}")
                return self.run_fallback_gui()
            
            theme = "dark"
            if self.config and isinstance(self.config, dict):
                theme = self.config.get("ui", {}).get("theme", "dark")
            
            ctk.set_appearance_mode(theme)
            ctk.set_default_color_theme("blue")
            
            root = ctk.CTk()
            root.title("DocuBot - Personal AI Assistant")
            root.geometry("800x600")
            
            label = ctk.CTkLabel(root, text="DocuBot is running!", font=("Arial", 24))
            label.pack(pady=50)
            
            info_text = f"""
            Mode: GUI
            Python: {self.system_info.python_version}
            Platform: {self.system_info.platform}
            Memory: {self.system_info.memory_gb} GB
            """
            
            info_label = ctk.CTkLabel(root, text=info_text, font=("Arial", 12))
            info_label.pack(pady=20)
            
            def on_exit():
                root.destroy()
            
            exit_button = ctk.CTkButton(root, text="Exit", command=on_exit)
            exit_button.pack(pady=20)
            
            root.mainloop()
            
            return 0
            
        except Exception as e:
            crash_path = self.save_crash_report(e)
            self.logger.error(f"GUI mode failed: {e}")
            
            print(f"\nGUI mode failed: {e}")
            print(f"Crash report saved to: {crash_path}")
            
            fallback = input("\nSwitch to CLI mode? (y/n): ")
            if fallback.lower() == 'y':
                return self.run_cli_mode()
            
            return 1
    
    def run_fallback_gui(self) -> int:
        try:
            import customtkinter as ctk
            
            ctk.set_appearance_mode("dark")
            ctk.set_default_color_theme("blue")
            
            root = ctk.CTk()
            root.title("DocuBot - Development Mode")
            root.geometry("600x400")
            
            label = ctk.CTkLabel(
                root, 
                text="DocuBot is in Development Mode\n\nSome features may be unavailable.",
                font=("Arial", 16)
            )
            label.pack(pady=50)
            
            root.mainloop()
            return 0
            
        except Exception as e:
            self.logger.error(f"Fallback GUI also failed: {e}")
            return self.run_cli_mode()
    
    def run_cli_mode(self) -> int:
        try:
            self.logger.info("Launching CLI mode")
            
            print("\n" + "="*60)
            print("DOCUBOT - Command Line Interface")
            print("="*60)
            
            print("\nAvailable Commands:")
            print("1. Setup wizard")
            print("2. System diagnostics")
            print("3. Document processing test")
            print("4. Exit")
            
            try:
                choice = input("\nSelect option (1-4): ").strip()
                
                if choice == "1":
                    return self.run_setup_wizard()
                elif choice == "2":
                    return self.run_diagnostic_mode()
                elif choice == "3":
                    return self.run_document_test()
                elif choice == "4":
                    print("Goodbye!")
                    return 0
                else:
                    print("Invalid choice")
                    return 1
                    
            except KeyboardInterrupt:
                print("\nInterrupted by user")
                return 0
                
        except Exception as e:
            crash_path = self.save_crash_report(e)
            self.logger.error(f"CLI mode failed: {e}")
            print(f"\nCLI mode failed: {e}")
            return 1
    
    def run_document_test(self) -> int:
        print("\nDocument Processing Test")
        print("-" * 30)
        
        test_file = input("Enter path to test file (PDF/TXT): ").strip()
        
        if not test_file or not os.path.exists(test_file):
            print("File not found. Using default test...")
            test_content = "This is a test document for DocuBot.\nIt contains sample text for processing."
            with open("test_document.txt", "w") as f:
                f.write(test_content)
            test_file = "test_document.txt"
        
        print(f"Processing: {test_file}")
        
        if hasattr(self, '_components') and self._components.get("Document Processor"):
            try:
                processor = self._components["Document Processor"]()
                result = processor.process_document(test_file)
                print(f"Success! Processed {result.get('chunks_processed', 0)} chunks")
                return 0
            except Exception as e:
                print(f"Processing failed: {e}")
                return 1
        else:
            print("Document processor not available. Skipping...")
            return 0
    
    def run_web_mode(self) -> int:
        try:
            self.logger.info("Launching Web mode")
            print("Web interface not yet implemented. Use --gui or --cli")
            return 1
            
        except Exception as e:
            crash_path = self.save_crash_report(e)
            self.logger.error(f"Web mode failed: {e}")
            return 1
    
    def run_setup_wizard(self) -> int:
        try:
            self.logger.info("Launching Setup Wizard")
            
            print("\n" + "="*60)
            print("DOCUBOT SETUP WIZARD")
            print("="*60)
            
            print("\n1. Install missing dependencies")
            print("2. Configure paths")
            print("3. Download AI models")
            print("4. Test components")
            print("5. Back to main menu")
            
            choice = input("\nSelect option (1-5): ").strip()
            
            if choice == "1":
                return self.install_dependencies()
            elif choice == "2":
                return self.configure_paths()
            elif choice == "5":
                return 0
            else:
                print("Option not yet implemented")
                return 0
                
        except Exception as e:
            crash_path = self.save_crash_report(e)
            self.logger.error(f"Setup wizard failed: {e}")
            return 1
    
    def install_dependencies(self) -> int:
        print("\nInstalling dependencies...")
        
        try:
            import subprocess
            
            req_file = Path("requirements.txt")
            if req_file.exists():
                print(f"Found requirements.txt ({req_file.stat().st_size} bytes)")
                with open(req_file, 'r') as f:
                    lines = f.readlines()
                    print(f"Found {len(lines)} packages in requirements")
                
                print("Installing packages (this may take a while)...")
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", str(req_file)],
                    capture_output=True,
                    text=True
                )
                
                if result.returncode == 0:
                    print("Dependencies installed successfully!")
                else:
                    print(f"Installation failed: {result.stderr}")
                
                return result.returncode
            else:
                print("requirements.txt not found")
                return 1
                
        except Exception as e:
            print(f"Installation error: {e}")
            return 1
    
    def configure_paths(self) -> int:
        print("\nConfiguring paths...")
        
        default_dir = str(Path.home() / ".docubot")
        data_dir = input(f"Data directory [{default_dir}]: ").strip() or default_dir
        
        try:
            Path(data_dir).mkdir(parents=True, exist_ok=True)
            print(f"Created/verified directory: {data_dir}")
            
            if not self.config:
                self.config = {}
            
            if "paths" not in self.config:
                self.config["paths"] = {}
            
            self.config["paths"]["data_dir"] = data_dir
            
            config_path = Path("data/config/app_config.yaml")
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False)
            
            print(f"Configuration saved to {config_path}")
            return 0
            
        except Exception as e:
            print(f"Configuration failed: {e}")
            return 1
    
    def run_diagnostic_mode(self) -> int:
        try:
            self.logger.info("Running diagnostic mode")
            
            diagnostic_report = {
                "system_info": asdict(self.system_info) if self.system_info else {},
                "startup_metrics": asdict(self.metrics),
                "configuration_loaded": bool(self.config),
                "components_loaded": self._components_loaded,
                "timestamp": datetime.now().isoformat(),
            }
            
            report_path = Path("data/logs/diagnostic_report.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(diagnostic_report, f, indent=2, ensure_ascii=False)
            
            print(f"\nDiagnostic report saved to: {report_path}")
            
            print("\n" + "="*60)
            print("SYSTEM DIAGNOSTICS")
            print("="*60)
            print(f"Python Version: {self.system_info.python_version}")
            print(f"Platform: {self.system_info.platform}")
            print(f"Memory: {self.system_info.memory_gb} GB")
            print(f"Disk Space: {self.system_info.disk_space_gb} GB free")
            print(f"Virtual Environment: {self.system_info.in_virtual_env}")
            print(f"Has GPU: {self.system_info.has_gpu}")
            
            if hasattr(self, '_components'):
                print(f"\nLoaded Components: {sum(1 for c in self._components.values() if c is not None)}/{len(self._components)}")
            
            if self.metrics.errors:
                print(f"\nErrors/Warnings: {len(self.metrics.errors)}")
                for error in self.metrics.errors[:5]:
                    print(f"  - {error}")
            
            input("\nPress Enter to continue...")
            return 0
            
        except Exception as e:
            crash_path = self.save_crash_report(e)
            self.logger.error(f"Diagnostic mode failed: {e}")
            return 1
    
    def run_minimal_mode(self) -> int:
        print("\n" + "="*60)
        print("DOCUBOT - Minimal Mode")
        print("="*60)
        print("Running in minimal mode due to missing dependencies.")
        print("\nAvailable operations:")
        print("1. Setup wizard")
        print("2. Diagnostic check")
        print("3. Exit")
        
        try:
            choice = input("\nSelect option (1-3): ").strip()
            
            if choice == "1":
                return self.run_setup_wizard()
            elif choice == "2":
                return self.run_diagnostic_mode()
            elif choice == "3":
                return 0
            else:
                print("Invalid choice")
                return 1
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
            return 0
        except Exception as e:
            crash_path = self.save_crash_report(e)
            print(f"\nError in minimal mode: {e}")
            return 1
    
    def initialize(self) -> bool:
        self.metrics.start_time = time.time()
        
        try:
            self.setup_signal_handlers()
            
            if not self.initialize_logging():
                print("Warning: Could not initialize logging, using basic logging")
                self.logger = logging.getLogger("DocuBot")
                self.logger.setLevel(logging.INFO)
            
            self.logger.info("=" * 60)
            self.logger.info("DOCUBOT Application Initialization")
            self.logger.info("=" * 60)
            
            if not self.initialize_data_directories():
                self.logger.error("Data directories initialization failed")
                return False
            
            if not self.validate_system_requirements():
                self.logger.warning("System requirements validation had warnings")
            
            if not self.validate_python_environment():
                self.logger.warning("Python environment validation had warnings")
            
            if not self.load_configuration():
                self.logger.warning("Configuration loading had issues")
            
            if HAS_SRC_MODULES:
                if not self.load_core_components():
                    self.logger.warning("Some core components failed to load")
            
            self._services_running = True
            
            init_time = time.time() - self.metrics.start_time
            self.metrics.initialization_time = init_time
            self.metrics.success = True
            
            self.logger.info(f"Initialization completed in {init_time:.2f} seconds")
            self.logger.info(f"Mode: {self.mode}")
            self.logger.info(f"System: {self.system_info.platform if self.system_info else 'Unknown'}")
            
            return True
            
        except Exception as e:
            crash_path = self.save_crash_report(e)
            self.metrics.errors.append(f"Initialization failed: {str(e)}")
            if self.logger:
                self.logger.error(f"Initialization failed: {e}")
            else:
                print(f"Initialization failed: {e}")
            return False
    
    def run(self, mode: str = "auto") -> int:
        self.mode = mode
        
        if not self.initialize():
            self.logger.error("Application initialization had issues, running minimal mode...")
            return self.run_minimal_mode()
        
        optimal_mode = self.determine_optimal_mode(mode)
        self.logger.info(f"Selected mode: {optimal_mode}")
        
        try:
            if optimal_mode == "gui":
                return self.run_gui_mode()
            elif optimal_mode == "cli":
                return self.run_cli_mode()
            elif optimal_mode == "web":
                return self.run_web_mode()
            elif optimal_mode == "setup":
                return self.run_setup_wizard()
            elif optimal_mode == "diagnostic":
                return self.run_diagnostic_mode()
            elif optimal_mode == "minimal":
                return self.run_minimal_mode()
            else:
                self.logger.error(f"Unknown mode: {optimal_mode}")
                return 1
                
        except KeyboardInterrupt:
            self.logger.info("Application interrupted by user")
            return 0
        except Exception as e:
            crash_path = self.save_crash_report(e)
            self.logger.error(f"Application runtime error: {e}")
            traceback.print_exc()
            return 1
    
    def shutdown(self):
        if not self._services_running:
            return
        
        self.logger.info("Initiating graceful shutdown")
        
        self._services_running = False
        self._shutdown_requested = True
        
        self.logger.info("Shutdown completed")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="DocuBot - Local AI Document Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python app.py                  # Launch with auto-detected mode
  python app.py --gui            # Launch desktop GUI
  python app.py --cli            # Launch command-line interface
  python app.py --setup          # Run setup wizard
  python app.py --diagnostic     # Run system diagnostics
        """
    )
    
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--gui", action="store_true", help="Launch desktop GUI")
    mode_group.add_argument("--cli", action="store_true", help="Launch command-line interface")
    mode_group.add_argument("--web", action="store_true", help="Launch web interface")
    mode_group.add_argument("--setup", action="store_true", help="Run setup wizard")
    mode_group.add_argument("--diagnostic", action="store_true", help="Run system diagnostics")
    
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], 
                       default="INFO", help="Set logging level")
    
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    app = DocuBotApplication()
    app.args = args
    
    mode = "auto"
    if args.gui:
        mode = "gui"
    elif args.cli:
        mode = "cli"
    elif args.web:
        mode = "web"
    elif args.setup:
        mode = "setup"
    elif args.diagnostic:
        mode = "diagnostic"
    
    try:
        return app.run(mode=mode)
    except Exception as e:
        print(f"Fatal error: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())