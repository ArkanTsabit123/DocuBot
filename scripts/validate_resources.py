# docubot/scripts/validate_resources.py
"""
DocuBot Resource Validation Script
Validates system resources, dependencies, and configuration.
"""

import sys
import os
import json
import yaml
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import platform
import shutil
import importlib
import sqlite3
from datetime import datetime


class ResourceValidator:
    """Comprehensive resource validation for DocuBot."""
    
    def __init__(self, project_dir: Optional[Path] = None):
        self.project_dir = project_dir or Path.cwd()
        self.data_dir = self.project_dir / "data"
        self.config_dir = self.data_dir / "config"
        self.models_dir = self.data_dir / "models"
        self.database_dir = self.data_dir / "database"
        self.logs_dir = self.data_dir / "logs"
        
        self.validation_results = {
            "timestamp": datetime.now().isoformat(),
            "system": platform.system(),
            "python_version": platform.python_version(),
            "checks": {},
            "overall_status": "unknown"
        }
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation checks."""
        print("Running comprehensive validation...")
        print("=" * 60)
        
        # System requirements
        self.validation_results["checks"]["system"] = self.validate_system_requirements()
        
        # Python environment
        self.validation_results["checks"]["python"] = self.validate_python_environment()
        
        # Dependencies
        self.validation_results["checks"]["dependencies"] = self.validate_dependencies()
        
        # Directory structure
        self.validation_results["checks"]["directories"] = self.validate_directory_structure()
        
        # Configuration files
        self.validation_results["checks"]["configuration"] = self.validate_configuration()
        
        # Database
        self.validation_results["checks"]["database"] = self.validate_database()
        
        # AI models
        self.validation_results["checks"]["models"] = self.validate_ai_models()
        
        # External services
        self.validation_results["checks"]["services"] = self.validate_external_services()
        
        # Performance
        self.validation_results["checks"]["performance"] = self.validate_performance()
        
        # Calculate overall status
        self._calculate_overall_status()
        
        return self.validation_results
    
    def validate_system_requirements(self) -> Dict[str, Any]:
        """Validate system hardware requirements."""
        print("Validating system requirements...")
        
        results = {
            "status": "unknown",
            "checks": {},
            "recommendations": []
        }
        
        try:
            import psutil
            
            # CPU cores
            cpu_cores = psutil.cpu_count(logical=True)
            results["checks"]["cpu_cores"] = {
                "value": cpu_cores,
                "minimum": 4,
                "status": "pass" if cpu_cores >= 4 else "fail"
            }
            
            # RAM
            ram_gb = psutil.virtual_memory().total / (1024**3)
            results["checks"]["ram_gb"] = {
                "value": f"{ram_gb:.1f}",
                "minimum": 8.0,
                "status": "pass" if ram_gb >= 8.0 else "warning" if ram_gb >= 4.0 else "fail"
            }
            
            # Disk space
            disk_usage = shutil.disk_usage(self.project_dir)
            free_gb = disk_usage.free / (1024**3)
            results["checks"]["disk_free_gb"] = {
                "value": f"{free_gb:.1f}",
                "minimum": 10.0,
                "status": "pass" if free_gb >= 10.0 else "warning" if free_gb >= 5.0 else "fail"
            }
            
            # Platform
            system = platform.system()
            results["checks"]["platform"] = {
                "value": system,
                "supported": ["Windows", "Linux", "Darwin"],
                "status": "pass" if system in ["Windows", "Linux", "Darwin"] else "warning"
            }
            
            # Determine overall status
            if any(check["status"] == "fail" for check in results["checks"].values()):
                results["status"] = "fail"
                results["recommendations"].append("System does not meet minimum requirements")
            elif any(check["status"] == "warning" for check in results["checks"].values()):
                results["status"] = "warning"
                results["recommendations"].append("System meets minimum requirements but may experience performance issues")
            else:
                results["status"] = "pass"
            
            print(f"  System validation: {results['status'].upper()}")
            return results
            
        except Exception as e:
            print(f"  System validation failed: {e}")
            results["status"] = "error"
            results["error"] = str(e)
            return results
    
    def validate_python_environment(self) -> Dict[str, Any]:
        """Validate Python environment and version."""
        print("Validating Python environment...")
        
        results = {
            "status": "unknown",
            "checks": {},
            "recommendations": []
        }
        
        try:
            # Python version
            version_info = sys.version_info
            version_str = f"{version_info.major}.{version_info.minor}.{version_info.micro}"
            results["checks"]["python_version"] = {
                "value": version_str,
                "required": "3.11+",
                "status": "pass" if version_info.major == 3 and version_info.minor >= 11 else "fail"
            }
            
            # Virtual environment
            in_venv = sys.prefix != sys.base_prefix
            results["checks"]["virtual_env"] = {
                "value": "Yes" if in_venv else "No",
                "recommended": "Yes",
                "status": "pass" if in_venv else "warning"
            }
            
            # Python executable
            python_exe = sys.executable
            results["checks"]["python_executable"] = {
                "value": python_exe,
                "status": "pass" if python_exe else "fail"
            }
            
            # Determine overall status
            if any(check["status"] == "fail" for check in results["checks"].values()):
                results["status"] = "fail"
                results["recommendations"].append("Python environment does not meet requirements")
            elif any(check["status"] == "warning" for check in results["checks"].values()):
                results["status"] = "warning"
                results["recommendations"].append("Consider using a virtual environment")
            else:
                results["status"] = "pass"
            
            print(f"  Python validation: {results['status'].upper()}")
            return results
            
        except Exception as e:
            print(f"  Python validation failed: {e}")
            results["status"] = "error"
            results["error"] = str(e)
            return results
    
    def validate_dependencies(self) -> Dict[str, Any]:
        """Validate Python package dependencies."""
        print("Validating dependencies...")
        
        results = {
            "status": "unknown",
            "checks": {},
            "missing": [],
            "outdated": [],
            "recommendations": []
        }
        
        # Core dependencies to check
        core_dependencies = [
            ("torch", "2.0.0"),
            ("transformers", "4.30.0"),
            ("sentence_transformers", "2.2.0"),
            ("chromadb", "0.4.0"),
            ("sqlalchemy", "2.0.0"),
            ("customtkinter", "5.0.0"),
            ("pypdf2", "3.0.0"),
            ("ollama", "0.1.0"),
        ]
        
        # Optional dependencies
        optional_dependencies = [
            ("pytesseract", "0.3.0"),
            ("pdfplumber", "0.10.0"),
            ("ebooklib", "0.18"),
            ("beautifulsoup4", "4.12.0"),
            ("streamlit", "1.28.0"),
        ]
        
        all_deps = core_dependencies + optional_dependencies
        
        for package_name, min_version in all_deps:
            try:
                # Try to import the package
                module = importlib.import_module(package_name.replace('-', '_'))
                
                # Get version
                version = getattr(module, '__version__', 'unknown')
                
                # Check if version meets minimum
                status = "pass"
                if version != 'unknown':
                    from packaging import version as pkg_version
                    try:
                        if pkg_version.parse(version) < pkg_version.parse(min_version):
                            status = "warning"
                            results["outdated"].append(f"{package_name} ({version} < {min_version})")
                    except:
                        status = "unknown"
                
                results["checks"][package_name] = {
                    "version": version,
                    "required": min_version,
                    "status": status
                }
                
                print(f"  ✓ {package_name}: {version}")
                
            except ImportError:
                if (package_name, min_version) in core_dependencies:
                    results["checks"][package_name] = {
                        "version": "missing",
                        "required": min_version,
                        "status": "fail"
                    }
                    results["missing"].append(package_name)
                    print(f"  ✗ {package_name}: MISSING (required)")
                else:
                    results["checks"][package_name] = {
                        "version": "missing",
                        "required": min_version,
                        "status": "warning"
                    }
                    print(f"  ⚠ {package_name}: MISSING (optional)")
        
        # Determine overall status
        if results["missing"]:
            results["status"] = "fail"
            results["recommendations"].append(f"Missing core dependencies: {', '.join(results['missing'])}")
        elif results["outdated"]:
            results["status"] = "warning"
            results["recommendations"].append(f"Outdated dependencies: {', '.join(results['outdated'][:3])}")
        else:
            results["status"] = "pass"
        
        return results
    
    def validate_directory_structure(self) -> Dict[str, Any]:
        """Validate project directory structure."""
        print("Validating directory structure...")
        
        results = {
            "status": "unknown",
            "checks": {},
            "missing": [],
            "recommendations": []
        }
        
        # Required directories
        required_dirs = [
            ("src", True, "Source code"),
            ("data", True, "Data storage"),
            ("data/config", True, "Configuration"),
            ("data/models", True, "AI models"),
            ("data/documents", True, "User documents"),
            ("data/database", True, "Database"),
            ("data/logs", True, "Log files"),
            ("tests", False, "Test files"),
            ("docs", False, "Documentation"),
            ("scripts", True, "Utility scripts"),
        ]
        
        for dir_path, required, description in required_dirs:
            full_path = self.project_dir / dir_path
            
            if full_path.exists() and full_path.is_dir():
                results["checks"][dir_path] = {
                    "exists": True,
                    "required": required,
                    "status": "pass"
                }
                print(f"  ✓ {dir_path}")
            else:
                status = "fail" if required else "warning"
                results["checks"][dir_path] = {
                    "exists": False,
                    "required": required,
                    "status": status
                }
                
                if required:
                    results["missing"].append(dir_path)
                    print(f"  ✗ {dir_path}: MISSING (required)")
                else:
                    print(f"  ⚠ {dir_path}: MISSING (optional)")
        
        # Check for important files
        important_files = [
            ("requirements.txt", True),
            ("pyproject.toml", False),
            ("README.md", False),
            ("app.py", True),
        ]
        
        for file_path, required in important_files:
            full_path = self.project_dir / file_path
            
            if full_path.exists() and full_path.is_file():
                size = full_path.stat().st_size
                results["checks"][file_path] = {
                    "exists": True,
                    "size_bytes": size,
                    "required": required,
                    "status": "pass" if size > 0 else "warning"
                }
                print(f"  ✓ {file_path} ({size} bytes)")
            else:
                status = "fail" if required else "warning"
                results["checks"][file_path] = {
                    "exists": False,
                    "required": required,
                    "status": status
                }
                
                if required:
                    results["missing"].append(file_path)
                    print(f"  ✗ {file_path}: MISSING (required)")
                else:
                    print(f"  ⚠ {file_path}: MISSING (optional)")
        
        # Determine overall status
        required_missing = [item for item in results["missing"] 
                          if results["checks"][item]["required"]]
        
        if required_missing:
            results["status"] = "fail"
            results["recommendations"].append(f"Missing required directories/files: {', '.join(required_missing)}")
        elif results["missing"]:
            results["status"] = "warning"
            results["recommendations"].append("Some optional directories/files are missing")
        else:
            results["status"] = "pass"
        
        return results
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate configuration files."""
        print("Validating configuration files...")
        
        results = {
            "status": "unknown",
            "checks": {},
            "errors": [],
            "recommendations": []
        }
        
        # Configuration files to validate
        config_files = [
            ("app_config.yaml", True),
            ("llm_config.yaml", True),
        ]
        
        for file_name, required in config_files:
            file_path = self.config_dir / file_name
            
            if file_path.exists() and file_path.is_file():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Parse YAML
                    config = yaml.safe_load(content)
                    
                    # Basic validation
                    if config and isinstance(config, dict):
                        results["checks"][file_name] = {
                            "exists": True,
                            "valid_yaml": True,
                            "keys": list(config.keys()),
                            "status": "pass"
                        }
                        print(f"  ✓ {file_name}: Valid YAML with {len(config)} top-level keys")
                    else:
                        results["checks"][file_name] = {
                            "exists": True,
                            "valid_yaml": False,
                            "status": "warning"
                        }
                        results["errors"].append(f"{file_name}: Empty or invalid YAML")
                        print(f"  ⚠ {file_name}: Empty or invalid YAML")
                        
                except yaml.YAMLError as e:
                    results["checks"][file_name] = {
                        "exists": True,
                        "valid_yaml": False,
                        "error": str(e),
                        "status": "fail" if required else "warning"
                    }
                    results["errors"].append(f"{file_name}: YAML parsing error")
                    print(f"  ✗ {file_name}: YAML parsing error")
                    
                except Exception as e:
                    results["checks"][file_name] = {
                        "exists": True,
                        "valid_yaml": False,
                        "error": str(e),
                        "status": "fail" if required else "warning"
                    }
                    results["errors"].append(f"{file_name}: {str(e)}")
                    print(f"  ✗ {file_name}: {str(e)}")
                    
            else:
                status = "fail" if required else "warning"
                results["checks"][file_name] = {
                    "exists": False,
                    "required": required,
                    "status": status
                }
                
                if required:
                    results["errors"].append(f"{file_name}: Missing")
                    print(f"  ✗ {file_name}: MISSING (required)")
                else:
                    print(f"  ⚠ {file_name}: MISSING (optional)")
        
        # Determine overall status
        required_errors = [err for err in results["errors"] 
                          if any(f" {file}:" in err for file, req in config_files if req)]
        
        if required_errors:
            results["status"] = "fail"
            results["recommendations"].append("Required configuration files have errors")
        elif results["errors"]:
            results["status"] = "warning"
            results["recommendations"].append("Some configuration files have issues")
        else:
            results["status"] = "pass"
        
        return results
    
    def validate_database(self) -> Dict[str, Any]:
        """Validate database setup and connectivity."""
        print("Validating database...")
        
        results = {
            "status": "unknown",
            "checks": {},
            "errors": [],
            "recommendations": []
        }
        
        # Check database directory
        if not self.database_dir.exists():
            results["checks"]["database_dir"] = {
                "exists": False,
                "status": "fail"
            }
            results["errors"].append("Database directory does not exist")
            print("  ✗ Database directory missing")
            results["status"] = "fail"
            return results
        
        # Check for database file
        db_files = list(self.database_dir.glob("*.db"))
        
        if not db_files:
            results["checks"]["database_files"] = {
                "count": 0,
                "status": "warning"
            }
            results["errors"].append("No database files found")
            print("  ⚠ No database files found")
            results["status"] = "warning"
            return results
        
        # Check each database file
        for db_file in db_files[:3]:  # Check first 3 database files
            try:
                conn = sqlite3.connect(db_file)
                cursor = conn.cursor()
                
                # Check if it's a valid SQLite database
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                tables = cursor.fetchall()
                
                # Check for required tables
                required_tables = ["documents", "chunks", "conversations"]
                found_tables = [table[0] for table in tables]
                missing_tables = [table for table in required_tables if table not in found_tables]
                
                # Get database statistics
                cursor.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table';")
                table_count = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(*) FROM documents;")
                document_count = cursor.fetchone()[0] if ("documents",) in tables else 0
                
                cursor.execute("SELECT COUNT(*) FROM chunks;")
                chunk_count = cursor.fetchone()[0] if ("chunks",) in tables else 0
                
                conn.close()
                
                # Determine status
                if missing_tables:
                    status = "fail"
                    error_msg = f"Missing tables: {', '.join(missing_tables)}"
                    results["errors"].append(f"{db_file.name}: {error_msg}")
                else:
                    status = "pass"
                    error_msg = None
                
                results["checks"][db_file.name] = {
                    "tables": found_tables,
                    "table_count": table_count,
                    "document_count": document_count,
                    "chunk_count": chunk_count,
                    "missing_tables": missing_tables,
                    "status": status
                }
                
                if status == "pass":
                    print(f"  ✓ {db_file.name}: {table_count} tables, {document_count} documents, {chunk_count} chunks")
                else:
                    print(f"  ✗ {db_file.name}: {error_msg}")
                
            except sqlite3.Error as e:
                results["checks"][db_file.name] = {
                    "valid": False,
                    "error": str(e),
                    "status": "fail"
                }
                results["errors"].append(f"{db_file.name}: {str(e)}")
                print(f"  ✗ {db_file.name}: SQLite error - {str(e)}")
            
            except Exception as e:
                results["checks"][db_file.name] = {
                    "valid": False,
                    "error": str(e),
                    "status": "fail"
                }
                results["errors"].append(f"{db_file.name}: {str(e)}")
                print(f"  ✗ {db_file.name}: {str(e)}")
        
        # Determine overall status
        if any(check.get("status") == "fail" for check in results["checks"].values()):
            results["status"] = "fail"
            results["recommendations"].append("Database has critical errors")
        elif any(check.get("status") == "warning" for check in results["checks"].values()):
            results["status"] = "warning"
            results["recommendations"].append("Database has some issues")
        elif results["checks"]:
            results["status"] = "pass"
        else:
            results["status"] = "unknown"
        
        return results
    
    def validate_ai_models(self) -> Dict[str, Any]:
        """Validate AI model availability and integrity."""
        print("Validating AI models...")
        
        results = {
            "status": "unknown",
            "checks": {},
            "missing": [],
            "recommendations": []
        }
        
        # Check models directory
        if not self.models_dir.exists():
            results["checks"]["models_dir"] = {
                "exists": False,
                "status": "fail"
            }
            results["missing"].append("Models directory")
            print("  ✗ Models directory missing")
            results["status"] = "fail"
            return results
        
        # Check for embedding models
        embedding_models = list(self.models_dir.glob("*"))
        
        if not embedding_models:
            results["checks"]["embedding_models"] = {
                "count": 0,
                "status": "warning"
            }
            results["missing"].append("Embedding models")
            print("  ⚠ No embedding models found")
        else:
            valid_models = 0
            for model_dir in embedding_models[:5]:  # Check first 5 models
                if model_dir.is_dir():
                    # Check for required files in Sentence Transformers format
                    required_files = ["config.json", "pytorch_model.bin", "vocab.txt"]
                    existing_files = []
                    
                    for file_name in required_files:
                        if (model_dir / file_name).exists():
                            existing_files.append(file_name)
                    
                    if len(existing_files) >= 2:  # At least 2 of 3 required files
                        valid_models += 1
                        status = "pass"
                    else:
                        status = "warning"
                    
                    results["checks"][model_dir.name] = {
                        "type": "embedding",
                        "files_found": existing_files,
                        "files_required": required_files,
                        "status": status
                    }
                    
                    if status == "pass":
                        print(f"  ✓ {model_dir.name}: Embedding model")
                    else:
                        print(f"  ⚠ {model_dir.name}: Incomplete embedding model")
            
            results["checks"]["embedding_models"] = {
                "total": len(embedding_models),
                "valid": valid_models,
                "status": "pass" if valid_models > 0 else "warning"
            }
        
        # Check for Ollama LLM models
        try:
            result = subprocess.run(['ollama', 'list'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                model_count = len(lines) - 1 if lines else 0  # Subtract header
                
                results["checks"]["ollama_models"] = {
                    "count": model_count,
                    "status": "pass" if model_count > 0 else "warning"
                }
                
                if model_count > 0:
                    print(f"  ✓ Ollama: {model_count} model(s) available")
                    
                    # List first 3 models
                    for line in lines[1:4]:  # Skip header
                        if line.strip():
                            parts = line.split()
                            if parts:
                                print(f"    - {parts[0]}")
                else:
                    print("  ⚠ Ollama: No models found")
                    results["missing"].append("Ollama LLM models")
                    
            else:
                results["checks"]["ollama_models"] = {
                    "available": False,
                    "error": result.stderr,
                    "status": "warning"
                }
                print("  ⚠ Ollama: Not responding")
                results["missing"].append("Ollama service")
                
        except FileNotFoundError:
            results["checks"]["ollama_models"] = {
                "available": False,
                "status": "warning"
            }
            print("  ⚠ Ollama: Not installed")
            results["missing"].append("Ollama installation")
        
        # Determine overall status
        has_embedding = results["checks"].get("embedding_models", {}).get("valid", 0) > 0
        has_llm = results["checks"].get("ollama_models", {}).get("count", 0) > 0
        
        if has_embedding and has_llm:
            results["status"] = "pass"
        elif has_embedding or has_llm:
            results["status"] = "warning"
            results["recommendations"].append("Only one type of AI model available")
        else:
            results["status"] = "fail"
            results["recommendations"].append("No AI models available")
        
        return results
    
    def validate_external_services(self) -> Dict[str, Any]:
        """Validate external services and dependencies."""
        print("Validating external services...")
        
        results = {
            "status": "unknown",
            "checks": {},
            "recommendations": []
        }
        
        # Check Tesseract OCR
        try:
            result = subprocess.run(['tesseract', '--version'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                results["checks"]["tesseract"] = {
                    "available": True,
                    "version": version_line,
                    "status": "pass"
                }
                print(f"  ✓ Tesseract: {version_line}")
            else:
                results["checks"]["tesseract"] = {
                    "available": False,
                    "error": result.stderr,
                    "status": "warning"
                }
                print("  ⚠ Tesseract: Not responding")
                results["recommendations"].append("Tesseract OCR may not work correctly")
                
        except FileNotFoundError:
            results["checks"]["tesseract"] = {
                "available": False,
                "status": "warning"
            }
            print("  ⚠ Tesseract: Not installed")
            results["recommendations"].append("Install Tesseract for image OCR support")
        
        # Check internet connectivity
        try:
            import requests
            response = requests.get("https://huggingface.co", timeout=5)
            
            results["checks"]["internet"] = {
                "available": True,
                "status_code": response.status_code,
                "status": "pass"
            }
            print(f"  ✓ Internet: Connected (status {response.status_code})")
            
        except Exception as e:
            results["checks"]["internet"] = {
                "available": False,
                "error": str(e),
                "status": "warning"
            }
            print("  ⚠ Internet: Not connected")
            results["recommendations"].append("Internet connection required for model downloads")
        
        # Determine overall status
        if all(check.get("status") == "pass" for check in results["checks"].values()):
            results["status"] = "pass"
        elif any(check.get("status") == "fail" for check in results["checks"].values()):
            results["status"] = "fail"
        else:
            results["status"] = "warning"
        
        return results
    
    def validate_performance(self) -> Dict[str, Any]:
        """Validate system performance characteristics."""
        print("Validating performance characteristics...")
        
        results = {
            "status": "unknown",
            "checks": {},
            "recommendations": []
        }
        
        try:
            import psutil
            
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            results["checks"]["cpu_usage"] = {
                "value": f"{cpu_percent:.1f}%",
                "threshold": "80%",
                "status": "pass" if cpu_percent < 80 else "warning"
            }
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            results["checks"]["memory_usage"] = {
                "value": f"{memory_percent:.1f}%",
                "threshold": "85%",
                "status": "pass" if memory_percent < 85 else "warning"
            }
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if disk_io:
                read_mb = disk_io.read_bytes / (1024**2)
                write_mb = disk_io.write_bytes / (1024**2)
                results["checks"]["disk_io"] = {
                    "read_mb": f"{read_mb:.1f}",
                    "write_mb": f"{write_mb:.1f}",
                    "status": "pass"
                }
            
            # Print results
            print(f"  CPU Usage: {cpu_percent:.1f}% {'(OK)' if cpu_percent < 80 else '(High)'}")
            print(f"  Memory Usage: {memory_percent:.1f}% {'(OK)' if memory_percent < 85 else '(High)'}")
            
            # Determine overall status
            if any(check["status"] == "warning" for check in results["checks"].values()):
                results["status"] = "warning"
                results["recommendations"].append("System resources are under high load")
            else:
                results["status"] = "pass"
            
        except Exception as e:
            print(f"  Performance validation failed: {e}")
            results["status"] = "error"
            results["error"] = str(e)
        
        return results
    
    def _calculate_overall_status(self) -> None:
        """Calculate overall validation status."""
        statuses = []
        
        for check_name, check_result in self.validation_results["checks"].items():
            if "status" in check_result:
                status = check_result["status"]
                if status in ["pass", "warning", "fail", "error"]:
                    statuses.append(status)
        
        # Determine overall status
        if not statuses:
            self.validation_results["overall_status"] = "unknown"
        elif "fail" in statuses:
            self.validation_results["overall_status"] = "fail"
        elif "error" in statuses:
            self.validation_results["overall_status"] = "error"
        elif "warning" in statuses:
            self.validation_results["overall_status"] = "warning"
        elif all(status == "pass" for status in statuses):
            self.validation_results["overall_status"] = "pass"
        else:
            self.validation_results["overall_status"] = "unknown"
    
    def print_summary(self) -> None:
        """Print validation summary."""
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        
        status = self.validation_results["overall_status"]
        status_symbol = {
            "pass": "✓",
            "warning": "⚠",
            "fail": "✗",
            "error": "⚡",
            "unknown": "?"
        }.get(status, "?")
        
        print(f"\nOverall Status: {status_symbol} {status.upper()}")
        print(f"System: {self.validation_results['system']}")
        print(f"Python: {self.validation_results['python_version']}")
        print(f"Timestamp: {self.validation_results['timestamp']}")
        
        print("\nDetailed Results:")
        for check_name, check_result in self.validation_results["checks"].items():
            status = check_result.get("status", "unknown")
            symbol = {
                "pass": "✓",
                "warning": "⚠",
                "fail": "✗",
                "error": "⚡",
                "unknown": "?"
            }.get(status, "?")
            
            print(f"  {symbol} {check_name.replace('_', ' ').title()}: {status.upper()}")
        
        # Print recommendations
        all_recommendations = []
        for check_result in self.validation_results["checks"].values():
            if "recommendations" in check_result:
                all_recommendations.extend(check_result["recommendations"])
        
        if all_recommendations:
            print("\nRecommendations:")
            for rec in set(all_recommendations):  # Remove duplicates
                print(f"  • {rec}")
        
        print("\n" + "=" * 60)
    
    def save_report(self, output_file: Optional[Path] = None) -> Path:
        """Save validation report to JSON file."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.logs_dir / f"validation_report_{timestamp}.json"
        
        # Ensure logs directory exists
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Save report
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.validation_results, f, indent=2, ensure_ascii=False)
        
        print(f"\nValidation report saved to: {output_file}")
        return output_file


def main():
    """Main entry point for validation script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DocuBot Resource Validator")
    parser.add_argument("--project-dir", type=str, help="Project directory path")
    parser.add_argument("--check", type=str, choices=["all", "system", "python", "deps", 
                                                      "dirs", "config", "db", "models", 
                                                      "services", "performance"],
                       default="all", help="Specific check to run")
    parser.add_argument("--save-report", action="store_true", help="Save report to file")
    parser.add_argument("--quiet", action="store_true", help="Suppress detailed output")
    
    args = parser.parse_args()
    
    # Determine project directory
    if args.project_dir:
        project_dir = Path(args.project_dir)
    else:
        project_dir = Path.cwd()
    
    # Initialize validator
    validator = ResourceValidator(project_dir)
    
    # Run validation
    if args.check == "all":
        results = validator.run_comprehensive_validation()
    else:
        # Run specific check
        check_methods = {
            "system": validator.validate_system_requirements,
            "python": validator.validate_python_environment,
            "deps": validator.validate_dependencies,
            "dirs": validator.validate_directory_structure,
            "config": validator.validate_configuration,
            "db": validator.validate_database,
            "models": validator.validate_ai_models,
            "services": validator.validate_external_services,
            "performance": validator.validate_performance,
        }
        
        if args.check in check_methods:
            print(f"Running {args.check} validation...")
            print("=" * 60)
            results = {args.check: check_methods[args.check]()}
            validator.validation_results["checks"] = results
            validator._calculate_overall_status()
        else:
            print(f"Unknown check: {args.check}")
            return
    
    # Print summary
    if not args.quiet:
        validator.print_summary()
    
    # Save report if requested
    if args.save_report:
        validator.save_report()
    
    # Exit with appropriate code
    status = validator.validation_results["overall_status"]
    if status == "fail":
        sys.exit(1)
    elif status == "error":
        sys.exit(2)
    elif status == "warning":
        sys.exit(0)  # Warning is OK for now
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()