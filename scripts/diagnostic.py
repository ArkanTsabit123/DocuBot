# docubot/scripts/diagnostic.py

"""
DocuBot Diagnostic Tool
Diagnose, report, and automatically fix issues to keep your system healthy.
"""

import os
import sys
import json
import platform
import psutil
import sqlite3
import subprocess
import importlib
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import traceback
import hashlib
import tempfile

# Try to import internal modules
try:
    from src.core.exceptions import (
        DiagnosticError, handle_error, safe_execute, 
        GracefulDegradation, ConfigurationValidator
    )
    from src.utilities.logger import setup_logging, get_logger
    HAS_INTERNAL_IMPORTS = True
except ImportError:
    HAS_INTERNAL_IMPORTS = False
    print("Warning: Could not import internal modules. Running in standalone mode.")


class DocuBotDiagnostic:
    """Run diagnostics on DocuBot installation."""
    
    def __init__(self, debug_mode: bool = False, auto_fix: bool = False):
        """
        Initialize diagnostic system.
        
        Args:
            debug_mode: Enable verbose debug output
            auto_fix: Attempt to automatically fix detected issues
        """
        self.debug_mode = debug_mode
        self.auto_fix = auto_fix
        self.fixes_applied = []
        self.fixes_failed = []
        
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'version': '2.0',
            'diagnostic_id': self._generate_diagnostic_id(),
            'system': {},
            'installation': {},
            'dependencies': {},
            'data': {},
            'configuration': {},
            'performance': {},
            'issues': [],
            'fixes_applied': [],
            'fixes_failed': [],
            'summary': {}
        }
        
        # Setup logging if internal modules are available
        if HAS_INTERNAL_IMPORTS:
            self.logger = setup_logging(Path.home() / ".docubot" / "diagnostic_logs", 
                                       "DEBUG" if debug_mode else "INFO")
        else:
            self.logger = None
    
    def _generate_diagnostic_id(self) -> str:
        """Generate unique diagnostic ID."""
        unique_string = f"{datetime.now().isoformat()}{platform.node()}"
        return hashlib.md5(unique_string.encode()).hexdigest()[:8]
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all diagnostic checks."""
        print("Running DocuBot Diagnostics...")
        print("=" * 70)
        
        # Define check methods in execution order
        checks = [
            ("System Requirements", self.check_system),
            ("Python Environment", self.check_python_environment),
            ("Project Structure", self.check_project_structure),
            ("Dependencies", self.check_dependencies),
            ("Data Directories", self.check_data_directories),
            ("Configuration Files", self.check_configuration),
            ("Database Integrity", self.check_databases),
            ("File Permissions", self.check_permissions),
            ("Network Connectivity", self.check_network_connectivity),
            ("Performance Benchmarks", self.check_performance),
            ("Model Availability", self.check_models),
            ("Integration Tests", self.check_integration)
        ]
        
        total_checks = len(checks)
        
        for i, (check_name, check_func) in enumerate(checks, 1):
            print(f"[{i}/{total_checks}] {check_name}...")
            try:
                check_func()
                if self.debug_mode:
                    print(f"  {check_name} completed successfully")
            except Exception as error:
                self.add_issue(
                    f"Check failed: {check_name}",
                    f"Diagnostic check crashed: {str(error)}",
                    "critical",
                    {"traceback": traceback.format_exc()}
                )
                if self.debug_mode:
                    print(f"  {check_name} failed: {error}")
        
        # Apply auto-fixes if requested
        if self.auto_fix:
            self._apply_automatic_fixes()
        
        # Generate summary
        self.results['summary'] = self.generate_summary()
        
        # Log results if logger available
        if self.logger:
            self.logger.log_diagnostic_run("full_system_check", self.results)
        
        return self.results
    
    def check_system(self):
        """Check system requirements and hardware."""
        print("  Checking system requirements...")
        
        system_info = {
            'platform': platform.platform(),
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'python_implementation': platform.python_implementation(),
        }
        
        # Memory information
        memory = psutil.virtual_memory()
        system_info.update({
            'memory_total_gb': round(memory.total / 1024**3, 2),
            'memory_available_gb': round(memory.available / 1024**3, 2),
            'memory_percent_used': memory.percent,
            'memory_used_gb': round(memory.used / 1024**3, 2),
            'memory_free_gb': round(memory.free / 1024**3, 2),
        })
        
        # Disk information
        disk = psutil.disk_usage('/')
        system_info.update({
            'disk_total_gb': round(disk.total / 1024**3, 2),
            'disk_free_gb': round(disk.free / 1024**3, 2),
            'disk_used_gb': round(disk.used / 1024**3, 2),
            'disk_percent_used': disk.percent,
        })
        
        # CPU information
        system_info.update({
            'cpu_count_physical': psutil.cpu_count(logical=False),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'cpu_percent': psutil.cpu_percent(interval=0.1),
            'cpu_freq_current_mhz': psutil.cpu_freq().current if hasattr(psutil.cpu_freq(), 'current') else None,
        })
        
        self.results['system'] = system_info
        
        # Check requirements
        if system_info['memory_total_gb'] < 8:
            self.add_issue(
                "Insufficient Memory",
                f"System has {system_info['memory_total_gb']:.1f}GB RAM (8GB recommended)",
                "warning",
                {"actual": system_info['memory_total_gb'], "recommended": 8}
            )
        
        if system_info['disk_free_gb'] < 10:
            self.add_issue(
                "Low Disk Space",
                f"Only {system_info['disk_free_gb']:.1f}GB free space (10GB recommended)",
                "warning",
                {"actual": system_info['disk_free_gb'], "recommended": 10}
            )
        
        # Check Python version
        python_version = system_info['python_version']
        if tuple(map(int, python_version.split('.')[:2])) < (3, 11):
            self.add_issue(
                "Outdated Python Version",
                f"Python {python_version} detected (3.11+ required)",
                "critical",
                {"actual": python_version, "required": "3.11+"}
            )
        
        if self.debug_mode:
            print(f"    System check complete: {system_info['platform']}")
    
    def check_python_environment(self):
        """Check Python environment and virtual environment."""
        print("  Checking Python environment...")
        
        env_info = {
            'virtual_env': os.getenv('VIRTUAL_ENV') is not None,
            'conda_env': os.getenv('CONDA_PREFIX') is not None,
            'python_path': sys.executable,
            'python_prefix': sys.prefix,
            'path_separator': os.pathsep,
            'working_directory': str(Path.cwd()),
        }
        
        if env_info['virtual_env']:
            env_info['virtual_env_path'] = os.getenv('VIRTUAL_ENV')
        
        if env_info['conda_env']:
            env_info['conda_env_path'] = os.getenv('CONDA_PREFIX')
        
        self.results['system']['environment'] = env_info
        
        if not env_info['virtual_env'] and not env_info['conda_env']:
            self.add_issue(
                "No Virtual Environment",
                "Running outside virtual environment is not recommended",
                "warning",
                {"recommendation": "Use venv or conda for isolation"}
            )
        
        if self.debug_mode:
            venv_status = "Active" if env_info['virtual_env'] or env_info['conda_env'] else "None"
            print(f"    Virtual Environment: {venv_status}")
    
    def check_project_structure(self):
        """Check project directory structure."""
        print("  Checking project structure...")
        
        # Determine project root
        project_root = self._find_project_root()
        self.results['installation']['project_root'] = str(project_root)
        
        required_dirs = [
            'src',
            'src/core',
            'src/document_processing',
            'src/ai_engine',
            'src/vector_store',
            'src/database',
            'src/ui',
            'src/utilities',
            'data',
            'data/config',
            'data/documents',
            'data/models',
            'data/database',
            'data/logs',
            'tests',
            'tests/unit',
            'tests/integration',
            'scripts',
            'docs',
            'resources'
        ]
        
        required_files = [
            'src/core/config.py',
            'src/core/app.py',
            'src/core/exceptions.py',
            'src/document_processing/processor.py',
            'src/ai_engine/llm_client.py',
            'src/ai_engine/embedding_service.py',
            'src/utilities/logger.py',
            'pyproject.toml',
            'requirements.txt',
            'README.md',
            '.gitignore'
        ]
        
        structure = {
            'directories': {},
            'files': {},
            'project_root_valid': project_root.exists(),
            'project_root_path': str(project_root)
        }
        
        # Check directories
        for dir_path in required_dirs:
            full_path = project_root / dir_path
            exists = full_path.exists() and full_path.is_dir()
            structure['directories'][dir_path] = exists
            
            if not exists:
                self.add_issue(
                    "Missing Directory",
                    f"Required directory not found: {dir_path}",
                    "warning" if dir_path.startswith('tests/') or dir_path.startswith('docs/') else "critical",
                    {"path": str(full_path), "type": "directory"}
                )
        
        # Check files
        for file_path in required_files:
            full_path = project_root / file_path
            exists = full_path.exists() and full_path.is_file()
            structure['files'][file_path] = exists
            
            if not exists:
                severity = "warning" if file_path in ['README.md', '.gitignore'] else "critical"
                self.add_issue(
                    "Missing File",
                    f"Required file not found: {file_path}",
                    severity,
                    {"path": str(full_path), "type": "file"}
                )
            elif full_path.stat().st_size == 0:
                self.add_issue(
                    "Empty File",
                    f"File exists but is empty: {file_path}",
                    "warning",
                    {"path": str(full_path), "size": 0}
                )
        
        self.results['installation']['structure'] = structure
        
        if self.debug_mode:
            dirs_found = sum(1 for v in structure['directories'].values() if v)
            files_found = sum(1 for v in structure['files'].values() if v)
            print(f"    Structure: {dirs_found}/{len(required_dirs)} dirs, {files_found}/{len(required_files)} files")
    
    def check_dependencies(self):
        """Check required dependencies and versions."""
        print("  Checking dependencies...")
        
        dependencies = [
            ('PyPDF2', 'PyPDF2', '3.0.0'),
            ('pdfplumber', 'pdfplumber', '0.10.0'),
            ('python-docx', 'python-docx', '0.8.11'),
            ('sqlalchemy', 'sqlalchemy', '2.0.0'),
            ('chromadb', 'chromadb', '0.4.0'),
            ('sentence-transformers', 'sentence_transformers', '2.2.0'),
            ('transformers', 'transformers', '4.30.0'),
            ('torch', 'torch', '2.0.0'),
            ('ollama', 'ollama', '0.1.0'),
            ('customtkinter', 'customtkinter', '5.2.0'),
            ('fastapi', 'fastapi', '0.104.0'),
            ('streamlit', 'streamlit', '1.28.0'),
            ('pytest', 'pytest', '7.4.0'),
            ('PyYAML', 'yaml', '6.0'),
            ('psutil', 'psutil', '5.9.0'),
            ('structlog', 'structlog', '23.1.0'),
            ('pytesseract', 'pytesseract', '0.3.10'),
            ('Pillow', 'PIL', '10.0.0'),
        ]
        
        installed = {}
        missing_critical = []
        missing_optional = []
        
        for package_name, import_name, min_version in dependencies:
            try:
                module = importlib.import_module(import_name)
                version = getattr(module, '__version__', 'unknown')
                
                installed[package_name] = {
                    'installed': True,
                    'version': version,
                    'min_version': min_version,
                    'meets_requirement': self._compare_versions(version, min_version)
                }
                
                if not installed[package_name]['meets_requirement'] and version != 'unknown':
                    self.add_issue(
                        "Outdated Dependency",
                        f"{package_name} version {version} (minimum {min_version} required)",
                        "warning",
                        {"package": package_name, "actual": version, "required": min_version}
                    )
                    
            except ImportError:
                installed[package_name] = {
                    'installed': False,
                    'version': None,
                    'min_version': min_version,
                    'meets_requirement': False
                }
                
                if package_name in ['chromadb', 'sentence-transformers', 'ollama', 'customtkinter']:
                    missing_critical.append(package_name)
                    self.add_issue(
                        "Missing Critical Dependency",
                        f"Required package not installed: {package_name}",
                        "critical",
                        {"package": package_name, "required_version": min_version}
                    )
                else:
                    missing_optional.append(package_name)
                    self.add_issue(
                        "Missing Optional Dependency",
                        f"Optional package not installed: {package_name}",
                        "info",
                        {"package": package_name, "required_version": min_version}
                    )
        
        self.results['dependencies'] = {
            'packages': installed,
            'missing_critical': missing_critical,
            'missing_optional': missing_optional,
            'total_checked': len(dependencies),
            'total_installed': len([p for p in installed.values() if p['installed']]),
            'total_critical_missing': len(missing_critical)
        }
        
        if self.debug_mode:
            print(f"    Dependencies: {len(installed)} checked, {len(missing_critical)} critical missing")
    
    def check_data_directories(self):
        """Check data directories and permissions."""
        print("  Checking data directories...")
        
        # Primary data directory
        primary_data_dir = Path.home() / ".docubot"
        
        data_dirs = [
            primary_data_dir,
            primary_data_dir / "models",
            primary_data_dir / "documents",
            primary_data_dir / "database",
            primary_data_dir / "logs",
            primary_data_dir / "config",
            primary_data_dir / "uploads",
            primary_data_dir / "processed",
            primary_data_dir / "exports",
            primary_data_dir / "cache",
            primary_data_dir / "backups",
            primary_data_dir / "diagnostic_logs"
        ]
        
        data_info = {}
        
        for data_dir in data_dirs:
            info = {
                'exists': data_dir.exists(),
                'is_directory': data_dir.exists() and data_dir.is_dir(),
                'writable': False,
                'size_bytes': 0,
                'file_count': 0,
                'created': None,
                'modified': None
            }
            
            if data_dir.exists():
                # Check write permissions
                try:
                    test_file = data_dir / ".test_write"
                    test_file.write_text("test")
                    test_file.unlink()
                    info['writable'] = True
                except (PermissionError, OSError) as error:
                    self.add_issue(
                        "Directory Not Writable",
                        f"Cannot write to directory: {data_dir}",
                        "warning" if data_dir.name in ['logs', 'cache'] else "critical",
                        {"path": str(data_dir), "error": str(error)}
                    )
                
                # Get directory info
                if data_dir.is_dir():
                    try:
                        total_size = 0
                        file_count = 0
                        for file in data_dir.rglob("*"):
                            if file.is_file():
                                total_size += file.stat().st_size
                                file_count += 1
                        info['size_bytes'] = total_size
                        info['file_count'] = file_count
                        
                        # Get timestamps
                        stat = data_dir.stat()
                        info['created'] = datetime.fromtimestamp(stat.st_ctime).isoformat()
                        info['modified'] = datetime.fromtimestamp(stat.st_mtime).isoformat()
                    except (PermissionError, OSError):
                        # Can't read some directories, that's okay
                        pass
            else:
                # Directory doesn't exist - determine if it's critical
                critical_dirs = ['models', 'database', 'config']
                is_critical = any(data_dir.name == d for d in critical_dirs)
                
                self.add_issue(
                    "Missing Data Directory",
                    f"Data directory not found: {data_dir}",
                    "critical" if is_critical else "warning",
                    {"path": str(data_dir), "type": "data_directory"}
                )
            
            data_info[str(data_dir)] = info
        
        self.results['data']['directories'] = data_info
        
        if self.debug_mode:
            existing_dirs = sum(1 for info in data_info.values() if info['exists'])
            writable_dirs = sum(1 for info in data_info.values() if info['writable'])
            print(f"    Data directories: {existing_dirs}/{len(data_dirs)} exist, {writable_dirs} writable")
    
    def check_databases(self):
        """Check database files and integrity."""
        print("  Checking databases...")
        
        db_files = [
            Path.home() / ".docubot" / "database" / "docubot.db",
            Path.home() / ".docubot" / "cache.db",
            Path.home() / ".docubot" / "database" / "chroma" / "chroma.sqlite3"
        ]
        
        db_info = {}
        
        for db_file in db_files:
            info = {
                'exists': db_file.exists(),
                'size_bytes': db_file.stat().st_size if db_file.exists() else 0,
                'accessible': False,
                'table_count': 0,
                'tables': [],
                'integrity_check': None
            }
            
            if db_file.exists():
                try:
                    conn = sqlite3.connect(db_file)
                    cursor = conn.cursor()
                    
                    # Get tables
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = [row[0] for row in cursor.fetchall()]
                    info['tables'] = tables
                    info['table_count'] = len(tables)
                    
                    # Run integrity check
                    cursor.execute("PRAGMA integrity_check")
                    integrity_result = cursor.fetchall()
                    info['integrity_check'] = integrity_result[0][0] if integrity_result else "unknown"
                    
                    conn.close()
                    info['accessible'] = True
                    
                    if len(tables) == 0:
                        self.add_issue(
                            "Empty Database",
                            f"Database has no tables: {db_file.name}",
                            "warning",
                            {"database": str(db_file), "table_count": 0}
                        )
                    
                    if info['integrity_check'] != 'ok':
                        self.add_issue(
                            "Database Integrity Issue",
                            f"Database integrity check failed: {db_file.name} - {info['integrity_check']}",
                            "critical",
                            {"database": str(db_file), "integrity_result": info['integrity_check']}
                        )
                        
                except sqlite3.Error as error:
                    self.add_issue(
                        "Database Error",
                        f"Cannot access database {db_file.name}: {error}",
                        "critical",
                        {"database": str(db_file), "error": str(error)}
                    )
            else:
                # Check if this is a critical database
                if db_file.name in ['docubot.db', 'chroma.sqlite3']:
                    self.add_issue(
                        "Missing Database",
                        f"Database file not found: {db_file.name}",
                        "critical" if db_file.name == 'docubot.db' else "warning",
                        {"database": str(db_file), "type": "sqlite"}
                    )
            
            db_info[str(db_file)] = info
        
        self.results['data']['databases'] = db_info
        
        if self.debug_mode:
            accessible_dbs = sum(1 for info in db_info.values() if info['accessible'])
            print(f"    Databases: {accessible_dbs}/{len(db_files)} accessible")
    
    def check_configuration(self):
        """Check configuration files."""
        print("  Checking configuration...")
        
        config_files = [
            Path.home() / ".docubot" / "config" / "app_config.yaml",
            Path.home() / ".docubot" / "config" / "llm_config.yaml",
            Path.home() / ".docubot" / "config" / "ui_config.yaml",
            Path.home() / ".docubot" / "config" / "model_config.yaml"
        ]
        
        config_info = {}
        
        for config_file in config_files:
            info = {
                'exists': config_file.exists(),
                'size_bytes': config_file.stat().st_size if config_file.exists() else 0,
                'readable': False,
                'valid_yaml': False,
                'has_content': False,
                'key_count': 0
            }
            
            if config_file.exists():
                try:
                    # Test readability
                    with open(config_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    info['readable'] = True
                    info['has_content'] = len(content.strip()) > 0
                    
                    # Try to parse as YAML
                    try:
                        import yaml
                        config_data = yaml.safe_load(content)
                        info['valid_yaml'] = True
                        
                        if config_data:
                            # Count top-level keys
                            info['key_count'] = len(config_data)
                            
                            # Validate based on config type
                            if config_file.name == 'app_config.yaml':
                                if HAS_INTERNAL_IMPORTS:
                                    valid, issues = ConfigurationValidator.validate_app_config(config_data)
                                    if not valid:
                                        self.add_issue(
                                            "Invalid App Configuration",
                                            f"Configuration validation failed: {', '.join(issues)}",
                                            "critical",
                                            {"config_file": str(config_file), "issues": issues}
                                        )
                    except yaml.YAMLError as error:
                        self.add_issue(
                            "Invalid YAML",
                            f"Configuration file is not valid YAML: {config_file.name}",
                            "critical",
                            {"config_file": str(config_file), "error": str(error)}
                        )
                    
                except Exception as error:
                    self.add_issue(
                        "Unreadable Configuration",
                        f"Cannot read configuration file: {config_file.name}",
                        "critical",
                        {"config_file": str(config_file), "error": str(error)}
                    )
            else:
                # Missing configuration file
                if config_file.name == 'app_config.yaml':
                    self.add_issue(
                        "Missing Configuration",
                        f"Main configuration file not found: {config_file.name}",
                        "critical",
                        {"config_file": str(config_file), "type": "yaml"}
                    )
                else:
                    self.add_issue(
                        "Missing Optional Configuration",
                        f"Optional configuration file not found: {config_file.name}",
                        "info",
                        {"config_file": str(config_file), "type": "yaml"}
                    )
            
            config_info[str(config_file)] = info
        
        self.results['configuration'] = config_info
        
        if self.debug_mode:
            valid_configs = sum(1 for info in config_info.values() if info['valid_yaml'])
            print(f"    Configuration: {valid_configs}/{len(config_files)} valid")
    
    def check_permissions(self):
        """Check file and directory permissions."""
        print("  Checking permissions...")
        
        critical_files = [
            Path.home() / ".docubot" / "secret.key",
            Path.home() / ".docubot" / "database" / "docubot.db",
            Path.home() / ".docubot" / "config" / "app_config.yaml"
        ]
        
        permission_info = {}
        
        for file_path in critical_files:
            info = {
                'exists': file_path.exists(),
                'mode': None,
                'owner_readable': False,
                'owner_writable': False,
                'group_readable': False,
                'group_writable': False,
                'others_readable': False,
                'others_writable': False,
                'secure': True
            }
            
            if file_path.exists():
                try:
                    mode = file_path.stat().st_mode
                    info['mode'] = oct(mode)[-3:]
                    
                    # Check permissions
                    info['owner_readable'] = bool(mode & 0o400)
                    info['owner_writable'] = bool(mode & 0o200)
                    info['group_readable'] = bool(mode & 0o040)
                    info['group_writable'] = bool(mode & 0o020)
                    info['others_readable'] = bool(mode & 0o004)
                    info['others_writable'] = bool(mode & 0o002)
                    
                    # Check security
                    if mode & 0o022:  # Group or others have write permission
                        info['secure'] = False
                        self.add_issue(
                            "Insecure Permissions",
                            f"File has overly permissive permissions: {file_path.name} (mode: {info['mode']})",
                            "critical" if file_path.name == 'secret.key' else "warning",
                            {"file": str(file_path), "mode": info['mode']}
                        )
                    
                except Exception as error:
                    self.add_issue(
                        "Permission Check Failed",
                        f"Cannot check permissions for: {file_path.name}",
                        "warning",
                        {"file": str(file_path), "error": str(error)}
                    )
            
            permission_info[str(file_path)] = info
        
        self.results['system']['permissions'] = permission_info
        
        if self.debug_mode:
            insecure_files = sum(1 for info in permission_info.values() if not info['secure'])
            print(f"    Permissions: {insecure_files} insecure files found")
    
    def check_network_connectivity(self):
        """Check network connectivity for external services."""
        print("  Checking network connectivity...")
        
        network_info = {
            'internet_accessible': False,
            'ollama_accessible': False,
            'huggingface_accessible': False,
            'ping_tests': {}
        }
        
        # Test internet connectivity
        try:
            import socket
            socket.create_connection(("8.8.8.8", 53), timeout=3)
            network_info['internet_accessible'] = True
        except:
            self.add_issue(
                "No Internet Connectivity",
                "Cannot reach external network. Some features may be limited.",
                "warning",
                {"test_target": "8.8.8.8:53"}
            )
        
        # Test Ollama if internet is available
        if network_info['internet_accessible']:
            try:
                import requests
                response = requests.get("http://localhost:11434/api/tags", timeout=5)
                network_info['ollama_accessible'] = response.status_code == 200
                
                if not network_info['ollama_accessible']:
                    self.add_issue(
                        "Ollama Not Running",
                        "Ollama service not accessible. Local LLM features will not work.",
                        "warning",
                        {"endpoint": "localhost:11434", "status_code": response.status_code}
                    )
            except:
                network_info['ollama_accessible'] = False
                self.add_issue(
                    "Ollama Connection Failed",
                    "Cannot connect to Ollama service.",
                    "info",
                    {"endpoint": "localhost:11434"}
                )
        
        self.results['system']['network'] = network_info
        
        if self.debug_mode:
            print(f"    Network: Internet={network_info['internet_accessible']}, Ollama={network_info['ollama_accessible']}")
    
    def check_performance(self):
        """Run basic performance benchmarks."""
        print("  Running performance benchmarks...")
        
        performance_info = {
            'disk_speed': None,
            'memory_speed': None,
            'cpu_speed': None,
            'benchmarks': {}
        }
        
        # Disk write speed test
        try:
            with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
                start = datetime.now()
                data = b'x' * (10 * 1024 * 1024)  # 10MB
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
                end = datetime.now()
                
                elapsed = (end - start).total_seconds()
                speed_mbps = (10 / elapsed) if elapsed > 0 else 0
                
                performance_info['disk_speed'] = speed_mbps
                performance_info['benchmarks']['disk_write_10mb'] = {
                    'time_seconds': elapsed,
                    'speed_mbps': speed_mbps,
                    'acceptable': speed_mbps > 5  # 5 MB/s minimum
                }
                
                os.unlink(f.name)
                
                if speed_mbps < 5:
                    self.add_issue(
                        "Slow Disk Performance",
                        f"Disk write speed is slow: {speed_mbps:.1f} MB/s",
                        "warning",
                        {"speed_mbps": speed_mbps, "minimum": 5}
                    )
        except Exception as error:
            if self.debug_mode:
                print(f"    Disk benchmark failed: {error}")
        
        # Memory speed test
        try:
            import time
            start = time.perf_counter()
            test_list = [i for i in range(1000000)]
            end = time.perf_counter()
            
            elapsed = end - start
            performance_info['memory_speed'] = elapsed
            performance_info['benchmarks']['memory_1m_ints'] = {
                'time_seconds': elapsed,
                'acceptable': elapsed < 0.5  # Should complete in under 0.5 seconds
            }
            
            if elapsed > 0.5:
                self.add_issue(
                    "Slow Memory Performance",
                    f"Memory operations are slow: {elapsed:.2f} seconds for 1M integers",
                    "info",
                    {"time_seconds": elapsed, "maximum": 0.5}
                )
        except Exception as error:
            if self.debug_mode:
                print(f"    Memory benchmark failed: {error}")
        
        self.results['performance'] = performance_info
        
        if self.debug_mode:
            disk_speed = performance_info.get('disk_speed', 0)
            print(f"    Performance: Disk={disk_speed:.1f} MB/s")
    
    def check_models(self):
        """Check model availability and integrity."""
        print("  Checking model availability...")
        
        models_dir = Path.home() / ".docubot" / "models"
        model_info = {
            'directory_exists': models_dir.exists(),
            'models_found': [],
            'total_size_gb': 0,
            'embedding_models': [],
            'llm_models': []
        }
        
        if models_dir.exists():
            try:
                # Look for common model files
                model_extensions = ['.bin', '.pth', '.pt', '.safetensors', '.gguf', '.onnx']
                
                for model_file in models_dir.rglob("*"):
                    if model_file.suffix in model_extensions:
                        model_info['models_found'].append({
                            'name': model_file.name,
                            'path': str(model_file),
                            'size_mb': model_file.stat().st_size / 1024 / 1024,
                            'type': self._guess_model_type(model_file.name)
                        })
                
                # Calculate total size
                total_size = sum(m['size_mb'] for m in model_info['models_found'])
                model_info['total_size_gb'] = total_size / 1024
                
                # Categorize models
                for model in model_info['models_found']:
                    if 'embedding' in model['name'].lower() or 'mini' in model['name'].lower():
                        model_info['embedding_models'].append(model['name'])
                    elif any(x in model['name'].lower() for x in ['llama', 'mistral', 'neural', 'model']):
                        model_info['llm_models'].append(model['name'])
                
                if len(model_info['models_found']) == 0:
                    self.add_issue(
                        "No Models Found",
                        "No AI models found in models directory. Download models to enable AI features.",
                        "warning",
                        {"models_directory": str(models_dir)}
                    )
                
            except Exception as error:
                self.add_issue(
                    "Model Directory Error",
                    f"Cannot read models directory: {error}",
                    "warning",
                    {"models_directory": str(models_dir), "error": str(error)}
                )
        else:
            self.add_issue(
                "Missing Models Directory",
                "Models directory does not exist. AI features will not work.",
                "warning",
                {"models_directory": str(models_dir)}
            )
        
        self.results['data']['models'] = model_info
        
        if self.debug_mode:
            model_count = len(model_info['models_found'])
            print(f"    Models: {model_count} found, {model_info['total_size_gb']:.1f} GB total")
    
    def check_integration(self):
        """Run basic integration tests."""
        print("  Running integration tests...")
        
        integration_info = {
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'results': []
        }
        
        # Simple import tests
        test_cases = [
            ("Core Config Import", "from src.core.config import Config"),
            ("Document Processor Import", "from src.document_processing.processor import DocumentProcessor"),
            ("LLM Client Import", "from src.ai_engine.llm_client import LLMClient"),
            ("Logger Import", "from src.utilities.logger import setup_logging"),
        ]
        
        for test_name, import_stmt in test_cases:
            try:
                exec(import_stmt, {})
                integration_info['tests_run'] += 1
                integration_info['tests_passed'] += 1
                integration_info['results'].append({
                    'test': test_name,
                    'passed': True,
                    'error': None
                })
            except Exception as error:
                integration_info['tests_run'] += 1
                integration_info['tests_failed'] += 1
                integration_info['results'].append({
                    'test': test_name,
                    'passed': False,
                    'error': str(error)
                })
                
                self.add_issue(
                    "Import Test Failed",
                    f"Failed to import module: {test_name}",
                    "warning",
                    {"test": test_name, "import": import_stmt, "error": str(error)}
                )
        
        self.results['installation']['integration_tests'] = integration_info
        
        if self.debug_mode:
            pass_rate = (integration_info['tests_passed'] / integration_info['tests_run'] * 100) if integration_info['tests_run'] > 0 else 0
            print(f"    Integration: {integration_info['tests_passed']}/{integration_info['tests_run']} passed ({pass_rate:.0f}%)")
    
    def add_issue(
        self, 
        title: str, 
        description: str, 
        severity: str = "warning",
        details: Optional[Dict[str, Any]] = None
    ):
        """Add diagnostic issue with structured information."""
        issue = {
            'id': f"ISSUE_{len(self.results['issues']) + 1:04d}",
            'title': title,
            'description': description,
            'severity': severity,
            'timestamp': datetime.now().isoformat(),
            'details': details or {},
            'auto_fixable': self._is_auto_fixable(title, severity, details)
        }
        
        self.results['issues'].append(issue)
    
    def _is_auto_fixable(self, title: str, severity: str, details: Dict[str, Any]) -> bool:
        """Determine if an issue can be automatically fixed."""
        auto_fixable_titles = [
            "Missing Directory",
            "Missing Optional Configuration",
            "No Virtual Environment",
            "Empty File",
            "Missing Optional Dependency"
        ]
        
        # Don't auto-fix critical issues
        if severity == 'critical':
            return False
        
        return title in auto_fixable_titles
    
    def _apply_automatic_fixes(self):
        """Attempt to automatically fix detected issues."""
        if not self.auto_fix:
            return
        
        print("\nApplying automatic fixes...")
        
        for issue in self.results['issues']:
            if issue.get('auto_fixable', False):
                success, message = self._fix_issue(issue)
                
                if success:
                    self.fixes_applied.append({
                        'issue_id': issue['id'],
                        'title': issue['title'],
                        'message': message,
                        'timestamp': datetime.now().isoformat()
                    })
                else:
                    self.fixes_failed.append({
                        'issue_id': issue['id'],
                        'title': issue['title'],
                        'message': message,
                        'timestamp': datetime.now().isoformat()
                    })
        
        self.results['fixes_applied'] = self.fixes_applied
        self.results['fixes_failed'] = self.fixes_failed
    
    def _fix_issue(self, issue: Dict[str, Any]) -> Tuple[bool, str]:
        """Attempt to fix a specific issue."""
        title = issue['title']
        details = issue.get('details', {})
        
        try:
            if title == "Missing Directory":
                path = Path(details.get('path', ''))
                if path and not path.exists():
                    path.mkdir(parents=True, exist_ok=True)
                    return True, f"Created directory: {path}"
            
            elif title == "Missing Optional Configuration":
                path = Path(details.get('path', ''))
                if path and not path.exists():
                    # Create empty config file
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("# Auto-generated configuration file\n")
                    return True, f"Created empty config file: {path.name}"
            
            elif title == "Empty File":
                path = Path(details.get('path', ''))
                if path and path.exists() and path.stat().st_size == 0:
                    # Add minimal content based on file type
                    if path.suffix == '.py':
                        content = '# Empty file - placeholder\n'
                    elif path.suffix in ['.yaml', '.yml']:
                        content = '# Empty configuration\n'
                    elif path.suffix == '.md':
                        content = '# Empty documentation\n'
                    else:
                        content = ''
                    
                    path.write_text(content)
                    return True, f"Added placeholder content to: {path.name}"
            
            elif title == "No Virtual Environment":
                # Suggest virtual environment creation
                return False, "Virtual environment creation requires manual intervention"
            
            elif title == "Missing Optional Dependency":
                package = details.get('package', '')
                if package:
                    try:
                        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                        return True, f"Installed package: {package}"
                    except:
                        return False, f"Failed to install package: {package}"
        
        except Exception as error:
            return False, f"Fix failed: {error}"
        
        return False, "No fix available for this issue type"
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate diagnostic summary with recommendations."""
        if 'summary' in self.results and self.results['summary']:
            return self.results['summary']
        
        issues = self.results['issues']
        
        summary = {
            'total_checks': 12,
            'issues_found': len(issues),
            'critical_issues': len([i for i in issues if i.get('severity') == 'critical']),
            'warning_issues': len([i for i in issues if i.get('severity') == 'warning']),
            'info_issues': len([i for i in issues if i.get('severity') == 'info']),
            'fixes_applied': len(self.fixes_applied),
            'fixes_failed': len(self.fixes_failed),
            'overall_status': 'healthy',
            'health_score': 100,
            'recommendations': []
        }
        
        # Calculate health score
        if summary['critical_issues'] > 0:
            summary['overall_status'] = 'critical'
            summary['health_score'] = max(0, 100 - (summary['critical_issues'] * 30))
        elif summary['warning_issues'] > 0:
            summary['overall_status'] = 'needs_attention'
            summary['health_score'] = max(0, 100 - (summary['warning_issues'] * 10))
        elif summary['info_issues'] > 0:
            summary['overall_status'] = 'healthy_with_notes'
            summary['health_score'] = 90
        
        # Generate recommendations
        if summary['critical_issues'] > 0:
            summary['recommendations'].append("Address critical issues immediately before using DocuBot")
        
        if summary['warning_issues'] > 0:
            summary['recommendations'].append("Resolve warning issues for optimal performance")
        
        # Check for specific common issues
        missing_deps = [i for i in issues if "Missing Critical Dependency" in i.get('title', '')]
        if missing_deps:
            dep_names = [d.get('details', {}).get('package', '') for d in missing_deps]
            dep_names = [d for d in dep_names if d]
            if dep_names:
                summary['recommendations'].append(f"Install missing dependencies: {', '.join(dep_names)}")
        
        config_issues = [i for i in issues if "Configuration" in i.get('title', '') and i.get('severity') in ['critical', 'warning']]
        if config_issues:
            summary['recommendations'].append("Fix configuration files or run setup wizard")
        
        return summary
    
    def print_report(self):
        """Print formatted diagnostic report."""
        print("\n" + "=" * 70)
        print("DOCUBOT DIAGNOSTIC REPORT")
        print("=" * 70)
        
        # Ensure summary exists
        summary = self.generate_summary()
        
        # Get values with defaults
        status = summary.get('overall_status', 'unknown').upper()
        health_score = summary.get('health_score', 0)
        issues_found = summary.get('issues_found', 0)
        critical_issues = summary.get('critical_issues', 0)
        warning_issues = summary.get('warning_issues', 0)
        info_issues = summary.get('info_issues', 0)
        fixes_applied = summary.get('fixes_applied', 0)
        fixes_failed = summary.get('fixes_failed', 0)
        
        status_color = {
            'HEALTHY': '\033[92m',  # Green
            'HEALTHY_WITH_NOTES': '\033[93m',  # Yellow
            'NEEDS_ATTENTION': '\033[93m',  # Yellow
            'CRITICAL': '\033[91m'  # Red
        }.get(status, '\033[0m')
        
        reset_color = '\033[0m'
        
        print(f"Overall Status: {status_color}{status}{reset_color}")
        print(f"Health Score: {health_score}/100")
        print(f"Issues Found: {issues_found} (Critical: {critical_issues}, "
              f"Warnings: {warning_issues}, Info: {info_issues})")
        print(f"Fixes Applied: {fixes_applied}, Failed: {fixes_failed}")
        
        if self.results['issues']:
            print("\nDETAILED ISSUES:")
            print("-" * 70)
            
            # Group by severity
            for severity in ['critical', 'warning', 'info']:
                severity_issues = [i for i in self.results['issues'] if i.get('severity') == severity]
                
                if severity_issues:
                    severity_display = severity.upper()
                    if severity == 'critical':
                        severity_display = f"\033[91m{severity_display}\033[0m"
                    elif severity == 'warning':
                        severity_display = f"\033[93m{severity_display}\033[0m"
                    
                    print(f"\n{severity_display} ISSUES ({len(severity_issues)}):")
                    
                    for issue in severity_issues[:5]:  # Show first 5 of each severity
                        issue_id = issue.get('id', 'UNKNOWN')
                        issue_title = issue.get('title', 'Unknown')
                        issue_desc = issue.get('description', 'No description')
                        
                        print(f"\n  [{issue_id}] {issue_title}")
                        print(f"      {issue_desc}")
                        
                        if issue.get('auto_fixable'):
                            print(f"      \033[92m✓ Auto-fix available\033[0m")
                    
                    if len(severity_issues) > 5:
                        print(f"      ... and {len(severity_issues) - 5} more {severity} issues")
        
        print("\nSYSTEM INFORMATION:")
        print("-" * 70)
        system = self.results.get('system', {})
        print(f"  Platform: {system.get('platform', 'Unknown')}")
        print(f"  Python: {system.get('python_version', 'Unknown')}")
        
        memory_total = system.get('memory_total_gb', 0)
        memory_available = system.get('memory_available_gb', 0)
        print(f"  Memory: {memory_total:.1f}GB total, {memory_available:.1f}GB available")
        
        disk_free = system.get('disk_free_gb', 0)
        disk_total = system.get('disk_total_gb', 0)
        print(f"  Disk: {disk_free:.1f}GB free of {disk_total:.1f}GB total")
        
        if 'environment' in system:
            env = system['environment']
            venv_status = "Active" if env.get('virtual_env') or env.get('conda_env') else "None"
            print(f"  Virtual Environment: {venv_status}")
        
        print("\nRECOMMENDATIONS:")
        print("-" * 70)
        
        recommendations = summary.get('recommendations', [])
        if recommendations:
            for i, rec in enumerate(recommendations[:5], 1):
                print(f"  {i}. {rec}")
        else:
            print("  No specific recommendations. System appears healthy.")
        
        print("\n" + "=" * 70)
        print(f"Diagnostic ID: {self.results.get('diagnostic_id', 'UNKNOWN')}")
        print(f"Generated: {self.results.get('timestamp', 'Unknown')}")
        print("=" * 70)
    
    def save_report(self, output_file: Path):
        """Save diagnostic report to file."""
        # Ensure summary is generated before saving
        if 'summary' not in self.results or not self.results['summary']:
            self.results['summary'] = self.generate_summary()
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"\nFull diagnostic report saved to: {output_file}")
    
    def validate_installation(self) -> bool:
        """Validate complete installation meets minimum requirements."""
        summary = self.generate_summary()
        
        # Must have no critical issues
        if summary.get('critical_issues', 0) > 0:
            return False
        
        # Must have Python 3.11+
        python_version = self.results.get('system', {}).get('python_version', '0.0')
        try:
            if tuple(map(int, python_version.split('.')[:2])) < (3, 11):
                return False
        except:
            return False
        
        # Must have at least 4GB RAM
        if self.results.get('system', {}).get('memory_total_gb', 0) < 4:
            return False
        
        # Must have at least 5GB free disk
        if self.results.get('system', {}).get('disk_free_gb', 0) < 5:
            return False
        
        return True
    
    def enable_debug_mode(self):
        """Enable verbose debug output."""
        self.debug_mode = True
        print("Debug mode enabled")
    
    # ============================================================================
    # HELPER METHODS
    # ============================================================================
    
    def _find_project_root(self) -> Path:
        """Find the DocuBot project root directory."""
        possible_roots = [
            Path.cwd(),
            Path.cwd() / "DocuBot",
            Path.home() / "DocuBot",
            Path(__file__).parent.parent.parent  # scripts directory parent
        ]
        
        for root in possible_roots:
            if (root / "src" / "core").exists() and (root / "pyproject.toml").exists():
                return root
        
        # Default to current directory
        return Path.cwd()
    
    def _compare_versions(self, version1: str, version2: str) -> bool:
        """Compare version strings."""
        if version1 == 'unknown':
            return False
        
        try:
            v1_parts = list(map(int, version1.split('.')[:3]))
            v2_parts = list(map(int, version2.split('.')[:3]))
            
            # Pad with zeros if needed
            while len(v1_parts) < 3:
                v1_parts.append(0)
            while len(v2_parts) < 3:
                v2_parts.append(0)
            
            return v1_parts >= v2_parts
        except:
            return False
    
    def _guess_model_type(self, filename: str) -> str:
        """Guess model type from filename."""
        filename_lower = filename.lower()
        
        if any(x in filename_lower for x in ['embedding', 'mini', 'mpnet', 'sentence']):
            return 'embedding'
        elif any(x in filename_lower for x in ['llama', 'mistral', 'neural', 'chat', 'instruct']):
            return 'llm'
        elif any(x in filename_lower for x in ['ocr', 'tesseract', 'detect', 'recognize']):
            return 'ocr'
        elif any(x in filename_lower for x in ['summary', 'summarize']):
            return 'summarization'
        else:
            return 'unknown'


def main():
    """Main diagnostic entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="DocuBot Diagnostic Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # Run basic diagnostics
  %(prog)s --debug            # Run with verbose output
  %(prog)s --auto-fix         # Attempt to fix issues automatically
  %(prog)s --output report.json  # Save full report to JSON
  %(prog)s --quiet --validate    # Run validation quietly
        """
    )
    
    parser.add_argument("--debug", "-d", action="store_true", 
                       help="Enable verbose debug output")
    parser.add_argument("--auto-fix", "-a", action="store_true",
                       help="Attempt to automatically fix detected issues")
    parser.add_argument("--output", "-o", type=Path, 
                       help="Output JSON file for detailed report")
    parser.add_argument("--quiet", "-q", action="store_true",
                       help="Quiet mode (minimal console output)")
    parser.add_argument("--validate", "-v", action="store_true",
                       help="Run validation and return exit code (0=valid, 1=invalid)")
    
    args = parser.parse_args()
    
    # Create diagnostic instance
    diagnostic = DocuBotDiagnostic(debug_mode=args.debug, auto_fix=args.auto_fix)
    
    # Run diagnostics
    results = diagnostic.run_all_checks()
    
    # Print report if not in quiet mode
    if not args.quiet:
        diagnostic.print_report()
    
    # Save report if requested
    if args.output:
        diagnostic.save_report(args.output)
    
    # Run validation if requested
    if args.validate:
        is_valid = diagnostic.validate_installation()
        if not args.quiet:
            print(f"\nInstallation validation: {'PASS' if is_valid else 'FAIL'}")
        sys.exit(0 if is_valid else 1)
    else:
        # Return exit code based on health
        health_score = results.get('summary', {}).get('health_score', 0)
        sys.exit(0 if health_score >= 70 else 1)


if __name__ == "__main__":
    main()