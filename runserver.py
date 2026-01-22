#runserver.py
"""
DOCUBOT SYSTEM CHECKER, MONITORING SERVER & SERVICE MANAGER
Version: 3.1.1 (Fixed Status Logic)
License: MIT
"""

import os
import sys
import json
import time
import shutil
import subprocess
import platform
import sqlite3
import threading
import concurrent.futures
import http.server
import socketserver
import signal
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable
import warnings
import argparse

# Suppress warnings
warnings.filterwarnings('ignore')

# Check optional dependencies
DEPENDENCIES = {
    'requests': False,
    'yaml': False,
    'psutil': False,
    'pkg_resources': False,
    'chromadb': False,
    'sentence_transformers': False,
    'tomli': False
}

try:
    import requests
    DEPENDENCIES['requests'] = True
except ImportError:
    pass

try:
    import yaml
    DEPENDENCIES['yaml'] = True
except ImportError:
    pass

try:
    import psutil
    DEPENDENCIES['psutil'] = True
except ImportError:
    pass

try:
    import pkg_resources
    DEPENDENCIES['pkg_resources'] = True
except ImportError:
    pass

try:
    import chromadb
    DEPENDENCIES['chromadb'] = True
except ImportError:
    pass

try:
    import sentence_transformers
    DEPENDENCIES['sentence_transformers'] = True
except ImportError:
    pass

try:
    import tomli
    DEPENDENCIES['tomli'] = True
except ImportError:
    pass


class ColorFormatter:
    """ANSI color formatter."""
    
    COLORS = {
        'HEADER': '\033[95m',
        'BLUE': '\033[94m',
        'CYAN': '\033[96m',
        'GREEN': '\033[92m',
        'YELLOW': '\033[93m',
        'RED': '\033[91m',
        'WHITE': '\033[97m',
        'BOLD': '\033[1m',
        'UNDERLINE': '\033[4m',
        'END': '\033[0m'
    }
    
    @staticmethod
    def colorize(text: str, color: str = 'WHITE', bold: bool = False) -> str:
        color_code = ColorFormatter.COLORS.get(color.upper(), '')
        bold_code = ColorFormatter.COLORS['BOLD'] if bold else ''
        return f"{bold_code}{color_code}{text}{ColorFormatter.COLORS['END']}"
    
    @staticmethod
    def print(text: str, color: str = 'WHITE', bold: bool = False, end: str = '\n'):
        print(ColorFormatter.colorize(text, color, bold), end=end)


class ProgressSpinner:
    """Progress spinner for long operations."""
    
    def __init__(self, message: str = "Processing"):
        self.message = message
        self.running = False
        self.thread = None
        self.frames = ['|', '/', '-', '\\']
    
    def _spin(self):
        frame = 0
        while self.running:
            sys.stdout.write(f"\r{self.message} {self.frames[frame % 4]}")
            sys.stdout.flush()
            time.sleep(0.1)
            frame += 1
    
    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._spin, daemon=True)
        self.thread.start()
    
    def stop(self, message: str = None):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)
        if message:
            sys.stdout.write(f"\r{message}\n")
        else:
            sys.stdout.write("\r" + " " * (len(self.message) + 2) + "\r")
        sys.stdout.flush()


class ServiceChecker:
    """Base class for all service checkers."""
    
    def __init__(self, name: str, required: bool = True, essential: bool = False):
        self.name = name
        self.required = required
        self.essential = essential  # Whether this service is essential for operation
        self.result = {
            'name': name,
            'status': 'unknown',
            'required': required,
            'essential': essential,
            'error': None,
            'details': {},
            'timestamp': datetime.now().isoformat()
        }
    
    def check(self) -> Dict[str, Any]:
        raise NotImplementedError


class SystemChecker(ServiceChecker):
    """System information checker."""
    
    def __init__(self):
        super().__init__("System Information", True, True)
    
    def check(self) -> Dict[str, Any]:
        self.result['status'] = 'complete'
        info = self._collect_system_info()
        self.result['details'] = info
        return self.result
    
    def _collect_system_info(self) -> Dict[str, Any]:
        info = {
            'platform': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'python_implementation': platform.python_implementation(),
            'hostname': platform.node(),
            'cpu_count': os.cpu_count(),
            'dependencies': DEPENDENCIES
        }
        
        if DEPENDENCIES['psutil']:
            try:
                # CPU
                cpu_info = {
                    'physical_cores': psutil.cpu_count(logical=False),
                    'logical_cores': psutil.cpu_count(logical=True),
                    'usage_percent': psutil.cpu_percent(interval=0.1)
                }
                
                # Memory
                mem = psutil.virtual_memory()
                mem_info = {
                    'total_gb': round(mem.total / (1024**3), 2),
                    'available_gb': round(mem.available / (1024**3), 2),
                    'used_gb': round(mem.used / (1024**3), 2),
                    'percent': mem.percent
                }
                
                # Disk
                disk = psutil.disk_usage(str(Path.cwd()))
                disk_info = {
                    'total_gb': round(disk.total / (1024**3), 2),
                    'free_gb': round(disk.free / (1024**3), 2),
                    'used_gb': round(disk.used / (1024**3), 2),
                    'percent': disk.percent
                }
                
                info.update({
                    'cpu': cpu_info,
                    'memory': mem_info,
                    'disk': disk_info,
                    'boot_time': datetime.fromtimestamp(psutil.boot_time()).isoformat()
                })
            except Exception as e:
                info['error'] = str(e)
        
        return info


class OllamaChecker(ServiceChecker):
    """Ollama server checker."""
    
    def __init__(self, required: bool = True):
        super().__init__("Ollama", required, True)  # Essential for AI features
    
    def check(self) -> Dict[str, Any]:
        if not DEPENDENCIES['requests']:
            self.result['status'] = 'error'
            self.result['error'] = 'requests library required'
            return self.result
        
        endpoints = [
            ('http://localhost:11434', 'localhost'),
            ('http://127.0.0.1:11434', 'loopback')
        ]
        
        for url, name in endpoints:
            try:
                start = time.time()
                response = requests.get(f"{url}/api/tags", timeout=5)
                response_time = time.time() - start
                
                if response.status_code == 200:
                    data = response.json()
                    self.result['status'] = 'running'
                    self.result['details'] = {
                        'endpoint': url,
                        'response_time_ms': round(response_time * 1000, 2),
                        'models': [m['name'] for m in data.get('models', [])],
                        'model_count': len(data.get('models', []))
                    }
                    
                    # Get version
                    try:
                        ver_resp = requests.get(f"{url}/api/version", timeout=2)
                        if ver_resp.status_code == 200:
                            self.result['details']['version'] = ver_resp.json().get('version')
                    except:
                        pass
                    
                    self._check_model_coverage()
                    break
                    
            except requests.exceptions.ConnectionError:
                continue
            except Exception as e:
                self.result['error'] = str(e)
        
        if self.result['status'] == 'unknown':
            self.result['status'] = 'not_running'
            self.result['error'] = 'Cannot connect to Ollama server'
        
        return self.result
    
    def _check_model_coverage(self):
        """Check if required models are available."""
        required = ['llama2', 'mistral', 'neural-chat']
        available = self.result['details'].get('models', [])
        
        found = []
        missing = []
        for req in required:
            if any(req in model.lower() for model in available):
                found.append(req)
            else:
                missing.append(req)
        
        self.result['details']['found_models'] = found
        self.result['details']['missing_models'] = missing
        if required:
            self.result['details']['coverage'] = len(found) / len(required)
        else:
            self.result['details']['coverage'] = 0.0


class ChromaDBChecker(ServiceChecker):
    """ChromaDB vector database checker."""
    
    def __init__(self, project_root: Path, required: bool = True):
        super().__init__("ChromaDB", required, True)  # Essential for vector search
        self.project_root = project_root
    
    def check(self) -> Dict[str, Any]:
        chroma_path = self.project_root / "data" / "database" / "chroma"
        
        if not DEPENDENCIES['chromadb']:
            self.result['status'] = 'error'
            self.result['error'] = 'chromadb library required'
            return self.result
        
        if not chroma_path.exists():
            self.result['status'] = 'not_found'
            self.result['error'] = f'Directory not found: {chroma_path}'
            return self.result
        
        try:
            from chromadb.config import Settings
            
            client = chromadb.PersistentClient(
                path=str(chroma_path),
                settings=Settings(anonymized_telemetry=False)
            )
            
            collections = client.list_collections()
            self.result['status'] = 'accessible'
            self.result['details'] = {
                'path': str(chroma_path),
                'collection_count': len(collections),
                'collections': [{'name': c.name, 'count': c.count()} for c in collections],
                'total_documents': sum(c.count() for c in collections)
            }
            
        except Exception as e:
            self.result['status'] = 'error'
            self.result['error'] = str(e)
        
        return self.result


class DatabaseChecker(ServiceChecker):
    """SQLite database checker."""
    
    def __init__(self, project_root: Path, required: bool = True):
        super().__init__("Database", required, True)  # Essential for data storage
        self.project_root = project_root
    
    def check(self) -> Dict[str, Any]:
        db_path = self.project_root / "data" / "database" / "sqlite.db"
        
        if not db_path.exists():
            self.result['status'] = 'not_found'
            self.result['error'] = f'Database not found: {db_path}'
            return self.result
        
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Get tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Get row counts
            row_counts = {}
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table};")
                    row_counts[table] = cursor.fetchone()[0]
                except:
                    pass
            
            conn.close()
            
            self.result['status'] = 'accessible'
            self.result['details'] = {
                'path': str(db_path),
                'size_mb': round(db_path.stat().st_size / (1024**2), 2),
                'tables': tables,
                'table_count': len(tables),
                'row_counts': row_counts,
                'total_rows': sum(row_counts.values())
            }
            
        except Exception as e:
            self.result['status'] = 'error'
            self.result['error'] = str(e)
        
        return self.result


class DependencyChecker(ServiceChecker):
    """Dependency checker."""
    
    def __init__(self, project_root: Path):
        super().__init__("Dependencies", True, True)  # Essential for application
        self.project_root = project_root
    
    def check(self) -> Dict[str, Any]:
        self.result['status'] = 'checking'
        
        # Find requirements
        req_file = self._find_requirements()
        if not req_file:
            self.result['status'] = 'error'
            self.result['error'] = 'No requirements file found'
            return self.result
        
        # Parse requirements
        requirements = self._parse_requirements(req_file)
        
        # Check installed
        installed = self._get_installed_packages()
        missing = []
        installed_count = 0
        
        for req in requirements:
            if req in installed:
                installed_count += 1
            else:
                missing.append(req)
        
        if requirements:
            coverage = (installed_count / len(requirements)) * 100
        else:
            coverage = 0
        
        # Update status based on coverage
        if coverage >= 90:
            self.result['status'] = 'complete'
        elif coverage >= 70:
            self.result['status'] = 'partial'
        elif coverage > 0:
            self.result['status'] = 'incomplete'
        else:
            self.result['status'] = 'error'
        
        self.result['details'] = {
            'requirements_file': str(req_file),
            'requirements_count': len(requirements),
            'installed_count': installed_count,
            'missing_packages': missing[:10],  # First 10 only
            'coverage_percent': round(coverage, 1),
            'critical_packages': self._check_critical_packages()
        }
        
        return self.result
    
    def _find_requirements(self) -> Optional[Path]:
        for name in ["requirements.txt", "pyproject.toml"]:
            path = self.project_root / name
            if path.exists():
                return path
        return None
    
    def _parse_requirements(self, req_file: Path) -> List[str]:
        requirements = []
        
        try:
            if req_file.name == "pyproject.toml":
                if not DEPENDENCIES['tomli']:
                    self.result['error'] = 'tomli required to parse pyproject.toml'
                    return requirements
                
                import tomli
                with open(req_file, 'r', encoding='utf-8') as f:
                    data = tomli.loads(f.read())
                deps = data.get('project', {}).get('dependencies', [])
                for dep in deps:
                    pkg = dep.split('>=')[0].split('<=')[0].split('==')[0]
                    pkg = pkg.split('[')[0].strip()
                    requirements.append(pkg)
            else:
                with open(req_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#'):
                            pkg = line.split('>=')[0].split('<=')[0].split('==')[0]
                            pkg = pkg.split('[')[0].strip()
                            requirements.append(pkg)
        except Exception as e:
            self.result['error'] = f"Parse error: {str(e)}"
        
        return requirements
    
    def _get_installed_packages(self) -> Dict[str, str]:
        installed = {}
        
        if DEPENDENCIES['pkg_resources']:
            try:
                for dist in pkg_resources.working_set:
                    installed[dist.key] = dist.version
            except:
                pass
        
        return installed
    
    def _check_critical_packages(self) -> Dict[str, Dict[str, Any]]:
        critical = ['torch', 'transformers', 'sentence-transformers', 
                   'chromadb', 'sqlalchemy', 'customtkinter',
                   'fastapi', 'uvicorn', 'streamlit', 'ollama']
        
        result = {}
        for pkg in critical:
            info = {'installed': False, 'version': None}
            
            if DEPENDENCIES['pkg_resources']:
                try:
                    version = pkg_resources.get_distribution(pkg).version
                    info['installed'] = True
                    info['version'] = version
                except pkg_resources.DistributionNotFound:
                    # Try import as fallback
                    try:
                        import_name = pkg.replace('-', '_')
                        __import__(import_name)
                        info['installed'] = True
                    except ImportError:
                        pass
                except Exception:
                    pass
            else:
                # Try simple import
                try:
                    import_name = pkg.replace('-', '_')
                    __import__(import_name)
                    info['installed'] = True
                except ImportError:
                    pass
            
            result[pkg] = info
        
        return result


class StructureChecker(ServiceChecker):
    """Project structure checker."""
    
    def __init__(self, project_root: Path):
        super().__init__("Project Structure", True, False)  # Not essential
        self.project_root = project_root
    
    def check(self) -> Dict[str, Any]:
        # Required directories (from app.py blueprint)
        required_dirs = [
            'src',
            'src/core',
            'src/document_processing',
            'src/document_processing/extractors',
            'src/ai_engine',
            'src/vector_store',
            'src/database',
            'src/ui/desktop',
            'src/utilities',
            'src/storage',
            'data',
            'data/models',
            'data/database',
            'data/documents',
            'data/config',
            'data/logs'
        ]
        
        # Required files (prefixed with ~ means optional)
        required_files = [
            'app.py',
            'requirements.txt',
            'pyproject.toml',
            'README.md',
            'src/core/config.py',
            'src/core/app.py',
            'src/document_processing/processor.py',
            'src/ai_engine/llm_client.py',
            'src/ai_engine/embedding_service.py',
            'src/vector_store/chroma_client.py',
            'data/config/app_config.yaml'
        ]
        
        # Check directories
        dir_results = self._check_items(required_dirs, is_dir=True)
        file_results = self._check_items(required_files, is_dir=False)
        
        # Calculate scores
        dir_score = (dir_results['found'] / dir_results['total']) * 100 if dir_results['total'] > 0 else 0
        file_score = (file_results['found'] / file_results['total']) * 100 if file_results['total'] > 0 else 0
        overall_score = (dir_score + file_score) / 2
        
        if overall_score >= 80:
            self.result['status'] = 'complete'
        elif overall_score >= 60:
            self.result['status'] = 'partial'
        else:
            self.result['status'] = 'incomplete'
        
        self.result['details'] = {
            'directories': dir_results,
            'files': file_results,
            'overall_score': round(overall_score, 1),
            'completeness': self._get_completeness_label(overall_score)
        }
        
        return self.result
    
    def _check_items(self, items: List[str], is_dir: bool) -> Dict[str, Any]:
        result = {'total': len(items), 'found': 0, 'missing': [], 'details': {}}
        
        for item in items:
            # Handle optional files (~ prefix)
            optional = False
            if item.startswith('~'):
                item = item[1:]
                optional = True
            
            path = self.project_root / item
            exists = path.exists() and (path.is_dir() if is_dir else path.is_file())
            
            if exists or optional:
                result['found'] += 1
                if exists:
                    result['details'][item] = {
                        'exists': True,
                        'size_bytes': path.stat().st_size if not is_dir else None
                    }
                else:
                    result['details'][item] = {'exists': False, 'optional': True}
            else:
                result['missing'].append(item)
                result['details'][item] = {'exists': False}
        
        return result
    
    def _get_completeness_label(self, score: float) -> str:
        if score >= 90:
            return "EXCELLENT"
        elif score >= 80:
            return "GOOD"
        elif score >= 70:
            return "FAIR"
        elif score >= 60:
            return "MINIMAL"
        else:
            return "POOR"


class WebServiceChecker(ServiceChecker):
    """Web service checker (FastAPI, Streamlit)."""
    
    def __init__(self, name: str, port: int, required: bool = False):
        super().__init__(name, required, False)  # Web services are not essential for core functionality
        self.port = port
        self.expected_status = 'running'
    
    def check(self) -> Dict[str, Any]:
        if not DEPENDENCIES['requests']:
            self.result['status'] = 'error'
            self.result['error'] = 'requests library required'
            return self.result
        
        try:
            response = requests.get(f"http://localhost:{self.port}", timeout=3)
            if response.status_code < 500:
                self.result['status'] = 'running'
                self.result['details'] = {
                    'port': self.port,
                    'status_code': response.status_code,
                    'url': f"http://localhost:{self.port}"
                }
            else:
                self.result['status'] = 'error'
                self.result['error'] = f'HTTP {response.status_code}'
                self.result['warning'] = True  # Mark as warning (not critical)
                
        except requests.exceptions.ConnectionError:
            self.result['status'] = 'not_running'
            self.result['warning'] = True  # Mark as warning (not critical)
        except Exception as e:
            self.result['status'] = 'error'
            self.result['error'] = str(e)
            self.result['warning'] = True
        
        # Add recommendation for starting the service
        if self.result['status'] != 'running':
            if self.name == 'FastAPI':
                self.result['recommendation'] = 'Start with: uvicorn src.ui.web.app:app --host 0.0.0.0 --port 8000 --reload'
            elif self.name == 'Streamlit':
                self.result['recommendation'] = 'Start with: streamlit run src/ui/web/app.py --server.port 8501'
        
        return self.result


class DocuBotSystemChecker:
    """Main system checker."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.results = {}
        self.start_time = time.time()
        self.spinner = ProgressSpinner("Running system checks")
        
        # Initialize checkers with essential flag
        self.checkers = [
            ('System', SystemChecker()),
            ('Ollama', OllamaChecker(required=True)),
            ('ChromaDB', ChromaDBChecker(self.project_root, required=True)),
            ('Database', DatabaseChecker(self.project_root, required=True)),
            ('FastAPI', WebServiceChecker("FastAPI", 8000, required=False)),
            ('Streamlit', WebServiceChecker("Streamlit", 8501, required=False)),
            ('Dependencies', DependencyChecker(self.project_root)),
            ('Structure', StructureChecker(self.project_root))
        ]
    
    def run_checks(self, parallel: bool = True) -> Dict[str, Any]:
        """Run all checks."""
        print("\n" + "=" * 80)
        ColorFormatter.print("DOCUBOT SYSTEM CHECKER", "CYAN", bold=True)
        python_version = platform.python_version()
        ColorFormatter.print(f"Python: {python_version} | Project: {self.project_root.name}", "WHITE")
        ColorFormatter.print("=" * 80, "CYAN")
        
        self.spinner.start()
        
        try:
            if parallel and len(self.checkers) > 1:
                with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                    future_to_name = {executor.submit(checker.check): name 
                                    for name, checker in self.checkers}
                    
                    for future in concurrent.futures.as_completed(future_to_name):
                        name = future_to_name[future]
                        try:
                            self.results[name] = future.result()
                        except Exception as e:
                            self.results[name] = {'error': str(e)}
            else:
                for name, checker in self.checkers:
                    try:
                        self.results[name] = checker.check()
                    except Exception as e:
                        self.results[name] = {'error': str(e)}
        
        finally:
            self.spinner.stop()
        
        return self.results
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate summary report with improved logic."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'project_root': str(self.project_root),
            'total_checks': len(self.results),
            'successful_checks': 0,
            'failed_checks': 0,
            'critical_issues': [],
            'warnings': [],
            'recommendations': [],
            'overall_status': 'UNKNOWN',
            'execution_time': round(time.time() - self.start_time, 2),
            'python_version': platform.python_version()
        }
        
        # Define service expectations - what's CRITICAL vs WARNING
        service_expectations = {
            'Ollama': ('running', 'essential'),      # CRITICAL if failed
            'ChromaDB': ('accessible', 'essential'), # CRITICAL if failed
            'Database': ('accessible', 'essential'), # CRITICAL if failed
            'System': ('complete', 'essential'),     # CRITICAL if failed
            'Dependencies': ('complete', 'warning'), # WARNING if partial/incomplete
            'FastAPI': ('running', 'warning'),       # WARNING if failed
            'Streamlit': ('running', 'warning'),     # WARNING if failed
            'Structure': ('complete', 'info')        # INFO only
        }
        
        # Check each service
        for service_name, (expected_status, importance) in service_expectations.items():
            if service_name not in self.results:
                continue
                
            result = self.results[service_name]
            actual_status = result.get('status', 'unknown')
            
            # Skip if status matches expectation
            if actual_status == expected_status:
                summary['successful_checks'] += 1
                continue
            
            # For Dependencies: only warn if coverage is low, not if just "partial"
            if service_name == 'Dependencies':
                details = result.get('details', {})
                coverage = details.get('coverage_percent', 0)
                if coverage >= 70:  # If coverage is 70% or more, it's not critical
                    summary['successful_checks'] += 1
                    continue
                elif coverage >= 50:
                    summary['warnings'].append(f'Dependencies coverage low: {coverage:.1f}%')
                    summary['recommendations'].append('Run: pip install -r requirements.txt')
                    summary['failed_checks'] += 1
                    continue
            
            # For Structure: just info, not a failure
            if service_name == 'Structure':
                details = result.get('details', {})
                score = details.get('overall_score', 0)
                if score >= 60:  # If structure is at least minimal
                    summary['successful_checks'] += 1
                    continue
                else:
                    summary['warnings'].append(f'Project structure incomplete: {score:.1f}%')
                    summary['failed_checks'] += 1
                    continue
            
            # Determine message based on importance
            message = f"{service_name}: expected '{expected_status}', got '{actual_status}'"
            recommendation = result.get('recommendation')
            
            if importance == 'essential':
                summary['critical_issues'].append(message)
                if recommendation:
                    summary['recommendations'].append(recommendation)
                summary['failed_checks'] += 1
            elif importance == 'warning':
                summary['warnings'].append(message)
                if recommendation:
                    summary['recommendations'].append(recommendation)
                summary['failed_checks'] += 1
            else:
                # For info-only services, count as successful
                summary['successful_checks'] += 1
        
        # Add automatic recommendations for web services
        fastapi = self.results.get('FastAPI', {})
        streamlit = self.results.get('Streamlit', {})
        
        if fastapi.get('status') != 'running' and 'FastAPI' in self.results:
            if 'Start FastAPI' not in [r for r in summary['recommendations'] if 'FastAPI' in r]:
                summary['recommendations'].append('Start FastAPI: uvicorn src.ui.web.app:app --host 0.0.0.0 --port 8000 --reload')
        
        if streamlit.get('status') != 'running' and 'Streamlit' in self.results:
            if 'Start Streamlit' not in [r for r in summary['recommendations'] if 'Streamlit' in r]:
                summary['recommendations'].append('Start Streamlit: streamlit run src/ui/web/app.py --server.port 8501')
        
        # Determine overall status with improved logic
        if summary['critical_issues']:
            summary['overall_status'] = 'CRITICAL'
        elif summary['warnings']:
            summary['overall_status'] = 'WARNING'
        elif summary['successful_checks'] == summary['total_checks']:
            summary['overall_status'] = 'HEALTHY'
        else:
            # If we have failed checks but no warnings/critical issues
            summary['overall_status'] = 'DEGRADED'
        
        return summary
    
    def print_report(self, mode: str = 'detailed'):
        """Print report in specified mode."""
        if mode == 'quick':
            self._print_quick_status()
        else:
            self._print_detailed_report()
    
    def _print_detailed_report(self):
        """Print detailed report."""
        print("\n" + "=" * 80)
        ColorFormatter.print("DETAILED SYSTEM REPORT", "CYAN", bold=True)
        print("=" * 80)
        
        # System Info
        sys_info = self.results.get('System', {})
        if sys_info:
            ColorFormatter.print("\nSYSTEM INFORMATION", "WHITE", bold=True)
            details = sys_info.get('details', {})
            print(f"  Platform: {details.get('platform', 'Unknown')} {details.get('release', '')}")
            print(f"  Python: {details.get('python_version', 'Unknown')}")
            print(f"  CPU Cores: {details.get('cpu_count', 'Unknown')}")
            
            if 'memory' in details:
                mem = details['memory']
                print(f"  Memory: {mem.get('available_gb', 0):.1f}/{mem.get('total_gb', 0):.1f} GB ({mem.get('percent', 0)}% used)")
        
        # Services
        ColorFormatter.print("\nSERVICES", "WHITE", bold=True)
        
        services = [
            ('Ollama', 'OLLAMA'),
            ('ChromaDB', 'CHROMADB'),
            ('Database', 'DATABASE'),
            ('FastAPI', 'FASTAPI'),
            ('Streamlit', 'STREAMLIT')
        ]
        
        for key, label in services:
            result = self.results.get(key, {})
            status = result.get('status', 'unknown')
            
            if key == 'Ollama':
                expected = 'running'
            elif key in ['ChromaDB', 'Database']:
                expected = 'accessible'
            elif key in ['FastAPI', 'Streamlit']:
                expected = 'running'
            else:
                expected = 'unknown'
            
            if status == expected:
                ColorFormatter.print(f"  {label}: OK", "GREEN")
                if key == 'Ollama' and 'details' in result:
                    models = result['details'].get('models', [])
                    if models:
                        print(f"    Models: {len(models)} available")
                        if len(models) <= 3:
                            print(f"    Available: {', '.join(models[:3])}")
                            if len(models) > 3:
                                print(f"    ... and {len(models) - 3} more")
            else:
                # Web services get yellow, essential services get red
                color = "YELLOW" if key in ['FastAPI', 'Streamlit'] else "RED"
                ColorFormatter.print(f"  {label}: {status.upper()}", color)
                if result.get('error'):
                    error_msg = result['error']
                    if len(error_msg) > 60:
                        error_msg = error_msg[:57] + "..."
                    print(f"    Error: {error_msg}")
                if key in ['FastAPI', 'Streamlit'] and status != 'running':
                    print(f"    Note: This service is optional but recommended")
        
        # Dependencies
        deps = self.results.get('Dependencies', {})
        if deps:
            ColorFormatter.print("\nDEPENDENCIES", "WHITE", bold=True)
            details = deps.get('details', {})
            installed = details.get('installed_count', 0)
            total = details.get('requirements_count', 0)
            
            if total > 0:
                coverage = details.get('coverage_percent', 0)
                coverage_color = "GREEN" if coverage >= 90 else "YELLOW" if coverage >= 70 else "RED"
                ColorFormatter.print(f"  Installed: {installed}/{total} ({coverage:.1f}%)", coverage_color)
                
                critical = details.get('critical_packages', {})
                if critical:
                    print("  Critical Packages:")
                    for pkg, info in critical.items():
                        if info.get('installed'):
                            ColorFormatter.print(f"    {pkg}: OK", "GREEN")
                        else:
                            ColorFormatter.print(f"    {pkg}: MISSING", "RED")
        
        # Structure
        structure = self.results.get('Structure', {})
        if structure:
            ColorFormatter.print("\nPROJECT STRUCTURE", "WHITE", bold=True)
            details = structure.get('details', {})
            score = details.get('overall_score', 0)
            completeness = details.get('completeness', 'UNKNOWN')
            
            score_color = "GREEN" if score >= 90 else "YELLOW" if score >= 70 else "RED"
            ColorFormatter.print(f"  Completeness: {completeness}", score_color)
            print(f"  Score: {score:.1f}/100")
            
            dirs = details.get('directories', {})
            files = details.get('files', {})
            print(f"  Directories: {dirs.get('found', 0)}/{dirs.get('total', 0)}")
            print(f"  Files: {files.get('found', 0)}/{files.get('total', 0)}")
            
            missing_files = files.get('missing', [])
            if missing_files:
                print(f"  Missing files: {', '.join(missing_files[:3])}")
                if len(missing_files) > 3:
                    print(f"    ... and {len(missing_files) - 3} more")
        
        # Summary
        summary = self.generate_summary()
        print("\n" + "=" * 80)
        ColorFormatter.print("SUMMARY", "CYAN", bold=True)
        print("=" * 80)
        
        status = summary['overall_status']
        status_color = {
            'HEALTHY': 'GREEN',
            'WARNING': 'YELLOW',
            'DEGRADED': 'YELLOW',
            'CRITICAL': 'RED'
        }.get(status, 'WHITE')
        
        ColorFormatter.print(f"Overall Status: {status}", status_color, bold=True)
        print(f"Python Version: {summary['python_version']}")
        print(f"Execution Time: {summary['execution_time']:.2f} seconds")
        print(f"Checks Passed: {summary['successful_checks']}/{summary['total_checks']}")
        
        if summary['critical_issues']:
            ColorFormatter.print("\nCRITICAL ISSUES:", "RED", bold=True)
            for issue in summary['critical_issues']:
                print(f"  • {issue}")
        
        if summary['warnings']:
            ColorFormatter.print("\nWARNINGS:", "YELLOW", bold=True)
            for warning in summary['warnings']:
                print(f"  • {warning}")
        
        if summary['recommendations']:
            ColorFormatter.print("\nRECOMMENDATIONS:", "CYAN", bold=True)
            for rec in summary['recommendations']:
                print(f"  • {rec}")
        
        print("\n" + "=" * 80)
    
    def _print_quick_status(self):
        """Print quick status."""
        summary = self.generate_summary()
        
        print("\n" + "=" * 60)
        ColorFormatter.print(" QUICK STATUS ", "CYAN", bold=True)
        ColorFormatter.print(f"Python: {summary['python_version']}", "WHITE")
        print("=" * 60)
        
        # Key services
        services = [
            ('System', 'System', True),
            ('Ollama', 'Ollama', True),
            ('ChromaDB', 'Chroma', True),
            ('Database', 'DB', True),
            ('FastAPI', 'API', False),
            ('Streamlit', 'WebUI', False),
            ('Dependencies', 'Deps', True)
        ]
        
        for key, label, essential in services:
            if key not in self.results:
                continue
                
            result = self.results.get(key, {})
            status = result.get('status', 'unknown')
            
            if key == 'Dependencies':
                details = result.get('details', {})
                installed = details.get('installed_count', 0)
                total = details.get('requirements_count', 0)
                if total > 0:
                    coverage = details.get('coverage_percent', 0)
                    if coverage >= 90:
                        ColorFormatter.print(f"{label}: {installed}/{total}", "GREEN")
                    elif coverage >= 70:
                        ColorFormatter.print(f"{label}: {installed}/{total}", "YELLOW")
                    else:
                        ColorFormatter.print(f"{label}: {installed}/{total}", "RED")
                continue
            
            # Determine expected status
            if key in ['Ollama', 'FastAPI', 'Streamlit']:
                expected = 'running'
            elif key in ['ChromaDB', 'Database']:
                expected = 'accessible'
            elif key == 'System':
                expected = 'complete'
            else:
                expected = 'unknown'
            
            if status == expected:
                symbol = '✓'
                color = 'GREEN'
            else:
                symbol = '⚠' if not essential else '✗'
                color = 'YELLOW' if not essential else 'RED'
            
            ColorFormatter.print(f"{label}: {symbol}", color)
        
        print("-" * 60)
        
        status = summary['overall_status']
        status_color = {
            'HEALTHY': 'GREEN',
            'WARNING': 'YELLOW',
            'DEGRADED': 'YELLOW',
            'CRITICAL': 'RED'
        }.get(status, 'WHITE')
        
        ColorFormatter.print(f"Status: {status}", status_color, bold=True)
        print(f"Passed: {summary['successful_checks']}/{summary['total_checks']}")
        print("=" * 60)
    
    def save_report(self, path: Optional[Path] = None) -> Path:
        """Save report to JSON file."""
        if not path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self.project_root / f"system_check_{timestamp}.json"
        
        data = {
            'summary': self.generate_summary(),
            'results': self.results,
            'timestamp': datetime.now().isoformat(),
            'checker_version': '3.1.1'
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        return path


class MonitoringHandler(http.server.BaseHTTPRequestHandler):
    """HTTP handler for monitoring endpoints."""
    
    checker = None  # Will be set by main
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/health':
            self.send_health()
        elif self.path == '/status':
            self.send_status()
        elif self.path == '/metrics':
            self.send_metrics()
        elif self.path == '/check':
            self.send_check()
        else:
            self.send_error(404, "Not Found")
    
    def send_health(self):
        """Send health check."""
        data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "service": "docubot-checker",
            "python_version": platform.python_version()
        }
        self._send_json(200, data)
    
    def send_status(self):
        """Send system status."""
        if self.checker:
            self.checker.run_checks(parallel=False)
            summary = self.checker.generate_summary()
            self._send_json(200, summary)
        else:
            self._send_json(503, {"error": "Checker not initialized"})
    
    def send_metrics(self):
        """Send Prometheus metrics."""
        if DEPENDENCIES['psutil']:
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage(str(Path.cwd()))
            
            metrics = [
                f"system_cpu_percent {cpu}",
                f"system_memory_percent {mem.percent}",
                f"system_memory_available_bytes {mem.available}",
                f"system_disk_percent {disk.percent}",
                f"system_disk_free_bytes {disk.free}"
            ]
            
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write("\n".join(metrics).encode())
        else:
            self._send_json(503, {"error": "psutil not available"})
    
    def send_check(self):
        """Run full check and return results."""
        if self.checker:
            self.checker.run_checks(parallel=False)
            self._send_json(200, self.checker.results)
        else:
            self._send_json(503, {"error": "Checker not initialized"})
    
    def _send_json(self, code: int, data: Dict):
        """Send JSON response."""
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def log_message(self, format, *args):
        """Override to reduce log noise."""
        pass


class ServiceManager:
    """Manages DocuBot services (FastAPI, Streamlit)."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.processes = []
        self.threads = []
        self.running = False
        
    def start_fastapi(self):
        """Start FastAPI service."""
        try:
            ColorFormatter.print("Starting FastAPI service...", "CYAN")
            # Check if the FastAPI app exists
            app_path = self.project_root / "src" / "ui" / "web" / "app.py"
            if not app_path.exists():
                ColorFormatter.print(f"FastAPI app not found at: {app_path}", "RED")
                return None
            
            process = subprocess.Popen(
                ["uvicorn", "src.ui.web.app:app", "--host", "0.0.0.0", "--port", "8000", "--reload"],
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            self.processes.append(process)
            
            # Start a thread to read output
            def read_output():
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        print(f"[FastAPI] {output.strip()}")
            
            thread = threading.Thread(target=read_output, daemon=True)
            thread.start()
            self.threads.append(thread)
            
            ColorFormatter.print("FastAPI service started on http://localhost:8000", "GREEN")
            return process
            
        except Exception as e:
            ColorFormatter.print(f"Failed to start FastAPI: {e}", "RED")
            return None
    
    def start_streamlit(self):
        """Start Streamlit service."""
        try:
            ColorFormatter.print("Starting Streamlit service...", "CYAN")
            # Check if the Streamlit app exists
            app_path = self.project_root / "src" / "ui" / "web" / "app.py"
            if not app_path.exists():
                ColorFormatter.print(f"Streamlit app not found at: {app_path}", "RED")
                return None
            
            process = subprocess.Popen(
                ["streamlit", "run", "src/ui/web/app.py", "--server.port", "8501"],
                cwd=self.project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            
            self.processes.append(process)
            
            # Start a thread to read output
            def read_output():
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        print(f"[Streamlit] {output.strip()}")
            
            thread = threading.Thread(target=read_output, daemon=True)
            thread.start()
            self.threads.append(thread)
            
            ColorFormatter.print("Streamlit service started on http://localhost:8501", "GREEN")
            return process
            
        except Exception as e:
            ColorFormatter.print(f"Failed to start Streamlit: {e}", "RED")
            return None
    
    def start_ollama(self):
        """Start Ollama service if not running."""
        checker = OllamaChecker()
        result = checker.check()
        
        if result.get('status') == 'running':
            ColorFormatter.print("Ollama service is already running", "GREEN")
            models = result.get('details', {}).get('models', [])
            if models:
                print(f"  Available models: {len(models)}")
            return True
        else:
            ColorFormatter.print("Ollama service is not running", "YELLOW")
            ColorFormatter.print("Please start Ollama manually: 'ollama serve'", "YELLOW")
            return False
    
    def start_service(self, service_name: str):
        """Start a specific service."""
        if service_name == "fastapi":
            return self.start_fastapi()
        elif service_name == "streamlit":
            return self.start_streamlit()
        elif service_name == "ollama":
            return self.start_ollama()
        else:
            ColorFormatter.print(f"Unknown service: {service_name}", "RED")
            return None
    
    def start_all_services(self):
        """Start all DocuBot services."""
        ColorFormatter.print("\n" + "=" * 80, "CYAN")
        ColorFormatter.print("STARTING DOCUBOT SERVICES", "CYAN", bold=True)
        ColorFormatter.print("=" * 80, "CYAN")
        
        self.running = True
        
        # First check system
        system_checker = DocuBotSystemChecker(str(self.project_root))
        system_checker.run_checks()
        summary = system_checker.generate_summary()
        
        if summary['overall_status'] == 'CRITICAL':
            ColorFormatter.print("Critical issues found. Services may not start properly.", "RED")
            proceed = input("Continue anyway? (y/N): ").lower().strip()
            if proceed != 'y':
                return False
        
        # Start Ollama
        self.start_ollama()
        
        # Start FastAPI
        fastapi_process = self.start_fastapi()
        
        # Start Streamlit
        streamlit_process = self.start_streamlit()
        
        # Display status
        print("\n" + "-" * 80)
        ColorFormatter.print("SERVICE STATUS", "WHITE", bold=True)
        print("-" * 80)
        
        print("FastAPI: http://localhost:8000")
        print("Streamlit: http://localhost:8501")
        print("Ollama: http://localhost:11434")
        print("\n" + "-" * 80)
        ColorFormatter.print("Press Ctrl+C to stop all services", "YELLOW")
        print("-" * 80)
        
        try:
            # Keep main thread alive
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop_all_services()
        
        return True
    
    def stop_all_services(self):
        """Stop all running services."""
        ColorFormatter.print("\nStopping all services...", "YELLOW")
        self.running = False
        
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except:
                try:
                    process.kill()
                except:
                    pass
        
        self.processes.clear()
        ColorFormatter.print("All services stopped", "GREEN")
    
    def check_services_status(self):
        """Check status of all services."""
        checker = DocuBotSystemChecker(str(self.project_root))
        checker.run_checks()
        checker.print_report('detailed')


class HealthMonitor:
    """Continuous health monitoring."""
    
    def __init__(self, project_root: str = ".", interval: int = 30):
        self.project_root = Path(project_root).resolve()
        self.interval = interval
        self.monitoring = False
        
    def start_monitoring(self):
        """Start continuous health monitoring."""
        self.monitoring = True
        checker = DocuBotSystemChecker(str(self.project_root))
        
        ColorFormatter.print("\n" + "=" * 80, "CYAN")
        ColorFormatter.print("STARTING HEALTH MONITORING", "CYAN", bold=True)
        ColorFormatter.print(f"Check interval: {self.interval} seconds", "WHITE")
        ColorFormatter.print("=" * 80, "CYAN")
        
        check_count = 0
        last_status = None
        
        try:
            while self.monitoring:
                check_count += 1
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                print(f"\n{'='*60}")
                ColorFormatter.print(f"HEALTH CHECK #{check_count} - {timestamp}", "CYAN", bold=True)
                print(f"{'='*60}")
                
                checker.run_checks(parallel=True)
                summary = checker.generate_summary()
                
                current_status = summary['overall_status']
                
                # Display status change
                if last_status and current_status != last_status:
                    if current_status == 'HEALTHY':
                        ColorFormatter.print(f"Status changed: {last_status} → {current_status}", "GREEN", bold=True)
                    elif current_status == 'CRITICAL':
                        ColorFormatter.print(f"Status changed: {last_status} → {current_status}", "RED", bold=True)
                    else:
                        ColorFormatter.print(f"Status changed: {last_status} → {current_status}", "YELLOW", bold=True)
                else:
                    status_color = {
                        'HEALTHY': 'GREEN',
                        'WARNING': 'YELLOW',
                        'DEGRADED': 'YELLOW',
                        'CRITICAL': 'RED'
                    }.get(current_status, 'WHITE')
                    ColorFormatter.print(f"System Status: {current_status}", status_color, bold=True)
                
                last_status = current_status
                
                # Show key metrics
                services = ['Ollama', 'ChromaDB', 'Database', 'FastAPI', 'Streamlit']
                for service in services:
                    result = checker.results.get(service, {})
                    status = result.get('status', 'unknown')
                    
                    if service in ['Ollama', 'FastAPI', 'Streamlit']:
                        expected = 'running'
                    elif service in ['ChromaDB', 'Database']:
                        expected = 'accessible'
                    else:
                        expected = 'unknown'
                    
                    if status == expected:
                        symbol = '✓'
                        color = 'GREEN'
                    else:
                        symbol = '⚠' if service in ['FastAPI', 'Streamlit'] else '✗'
                        color = 'YELLOW' if service in ['FastAPI', 'Streamlit'] else 'RED'
                    
                    ColorFormatter.print(f"  {service}: {symbol}", color)
                
                # Countdown for next check
                for i in range(self.interval, 0, -1):
                    if not self.monitoring:
                        break
                    sys.stdout.write(f"\rNext check in {i} seconds... (Press Ctrl+C to stop) ")
                    sys.stdout.flush()
                    time.sleep(1)
                
                print()
                
        except KeyboardInterrupt:
            print("\n\nStopping health monitoring...")
            self.stop_monitoring()
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring = False


def run_monitoring_server(checker: DocuBotSystemChecker, port: int = 8080):
    """Run HTTP monitoring server."""
    MonitoringHandler.checker = checker
    
    handler = MonitoringHandler
    server = socketserver.TCPServer(("", port), handler)
    
    print(f"\nMonitoring server running on port {port}")
    print(f"Python Version: {platform.python_version()}")
    print(f"Endpoints:")
    print(f"  http://localhost:{port}/health    - Basic health check")
    print(f"  http://localhost:{port}/status    - System status")
    print(f"  http://localhost:{port}/metrics   - Prometheus metrics")
    print(f"  http://localhost:{port}/check     - Run full check")
    print(f"\nPress Ctrl+C to stop server")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down monitoring server...")
        server.shutdown()
        server.server_close()


def main():
    """Main entry point with simplified arguments."""
    parser = argparse.ArgumentParser(
        description="DocuBot System Checker, Service Manager & Monitor (v3.1.1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run system check (default)
  python check_server.py
  
  # Run system check with quick report
  python check_server.py --mode check --quick
  
  # Start all services
  python check_server.py --mode start --service all
  
  # Start only FastAPI
  python check_server.py --mode start --service fastapi
  
  # Start only Streamlit
  python check_server.py --mode start --service streamlit
  
  # Run as HTTP monitoring server
  python check_server.py --mode monitor --port 8080
  
  # Continuous health monitoring
  python check_server.py --mode monitor --interval 30
        """
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        choices=["check", "start", "monitor"],
        default="check",
        help="Operation mode: check system, start services, or monitor"
    )
    
    # Service selection (for start mode)
    parser.add_argument(
        "--service",
        choices=["all", "fastapi", "streamlit", "ollama"],
        default="all",
        help="Service to start (for start mode)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--path", "-p",
        default=".",
        help="Project root path (default: current directory)"
    )
    
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Show quick status only (for check mode)"
    )
    
    parser.add_argument(
        "--save",
        help="Save report to JSON file (for check mode)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for HTTP monitoring server (default: 8080)"
    )
    
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Check interval in seconds for continuous monitoring (default: 30)"
    )
    
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )
    
    args = parser.parse_args()
    
    # Disable colors if requested
    if args.no_color:
        ColorFormatter.print = lambda text, color='WHITE', bold=False, end='\n': print(text, end=end)
    
    # Initialize based on mode
    if args.mode == "check":
        # Run system checker
        checker = DocuBotSystemChecker(args.path)
        checker.run_checks()
        
        if args.quick:
            checker.print_report('quick')
        else:
            checker.print_report('detailed')
        
        # Save report if requested
        if args.save:
            path = checker.save_report(Path(args.save))
            print(f"\nReport saved to: {path}")
        elif not args.quick:
            path = checker.save_report()
            print(f"\nFull report saved to: {path}")
        
        # Exit with appropriate code - FIXED LOGIC
        summary = checker.generate_summary()
        if summary['overall_status'] == 'CRITICAL':
            # Only exit with code 1 for truly critical issues
            ColorFormatter.print("\n⚠️  Critical issues found! System cannot function properly.", "RED", bold=True)
            sys.exit(1)
        elif summary['overall_status'] == 'WARNING':
            # Warnings are not critical, just informational
            ColorFormatter.print("\n⚠️  Warnings found. System is functional but could be improved.", "YELLOW")
            sys.exit(0)  # Exit with 0, not 2
        elif summary['overall_status'] == 'DEGRADED':
            ColorFormatter.print("\n⚠️  System is degraded but functional.", "YELLOW")
            sys.exit(0)  # Exit with 0
        else:
            # HEALTHY status
            ColorFormatter.print("\n✅ System is healthy and ready!", "GREEN", bold=True)
            sys.exit(0)
            
    elif args.mode == "start":
        # Start services
        service_manager = ServiceManager(args.path)
        
        if args.service == "all":
            ColorFormatter.print("Starting all DocuBot services...", "CYAN", bold=True)
            service_manager.start_all_services()
        else:
            ColorFormatter.print(f"Starting {args.service} service...", "CYAN", bold=True)
            service_manager.start_service(args.service)
            
            if args.service in ["fastapi", "streamlit"]:
                print(f"\nService started. Press Ctrl+C to stop.")
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    service_manager.stop_all_services()
    
    elif args.mode == "monitor":
        # Check if it's HTTP monitoring or continuous monitoring
        if args.port == 8080 and args.interval == 30:
            # Default: continuous monitoring
            health_monitor = HealthMonitor(args.path, args.interval)
            health_monitor.start_monitoring()
        else:
            # HTTP monitoring server
            checker = DocuBotSystemChecker(args.path)
            run_monitoring_server(checker, args.port)


if __name__ == "__main__":
    main()