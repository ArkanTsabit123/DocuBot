#!/usr/bin/env python3
"""
DocuBot Diagnostic Tools
Comprehensive system diagnostics and health checks
"""

import os
import sys
import json
import platform
import psutil
import sqlite3
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime
import traceback


class DocuBotDiagnostic:
    """Run comprehensive diagnostics on DocuBot installation"""
    
    def __init__(self):
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'system': {},
            'installation': {},
            'dependencies': {},
            'data': {},
            'issues': []
        }
    
    def run_all_checks(self) -> Dict[str, Any]:
        """Run all diagnostic checks"""
        print("Running DocuBot Diagnostics...")
        print("=" * 60)
        
        checks = [
            self.check_system,
            self.check_python_environment,
            self.check_dependencies,
            self.check_project_structure,
            self.check_data_directories,
            self.check_databases,
            self.check_configuration,
            self.check_permissions
        ]
        
        for check in checks:
            try:
                check()
            except Exception as e:
                self.add_issue(f"Check failed: {check.__name__}", str(e))
        
        self.results['summary'] = self.generate_summary()
        
        return self.results
    
    def check_system(self):
        """Check system requirements"""
        print("Checking system requirements...")
        
        system_info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'processor': platform.processor(),
            'memory_total_gb': psutil.virtual_memory().total / 1024 / 1024 / 1024,
            'memory_available_gb': psutil.virtual_memory().available / 1024 / 1024 / 1024,
            'disk_total_gb': psutil.disk_usage('/').total / 1024 / 1024 / 1024,
            'disk_free_gb': psutil.disk_usage('/').free / 1024 / 1024 / 1024
        }
        
        self.results['system'] = system_info
        
        if system_info['memory_total_gb'] < 8:
            self.add_issue("Low memory", f"Only {system_info['memory_total_gb']:.1f}GB RAM available (8GB recommended)")
        
        if system_info['disk_free_gb'] < 10:
            self.add_issue("Low disk space", f"Only {system_info['disk_free_gb']:.1f}GB free (10GB recommended)")
    
    def check_python_environment(self):
        """Check Python environment"""
        print("Checking Python environment...")
        
        env_info = {
            'virtual_env': os.getenv('VIRTUAL_ENV') is not None,
            'python_path': sys.executable,
            'path': sys.path[:5]
        }
        
        if not env_info['virtual_env']:
            self.add_issue("No virtual environment", "Running outside virtual environment is not recommended")
    
    def check_dependencies(self):
        """Check required dependencies"""
        print("Checking dependencies...")
        
        dependencies = [
            'chromadb',
            'sentence_transformers',
            'customtkinter',
            'ollama',
            'fastapi',
            'pypdf2',
            'pdfplumber',
            'python-docx',
            'sqlalchemy',
            'pyyaml'
        ]
        
        installed = {}
        
        for dep in dependencies:
            try:
                __import__(dep.replace('-', '_'))
                installed[dep] = True
            except ImportError:
                installed[dep] = False
                self.add_issue("Missing dependency", f"Package not installed: {dep}")
        
        self.results['dependencies'] = installed
    
    def check_project_structure(self):
        """Check project directory structure"""
        print("Checking project structure...")
        
        required_dirs = [
            'src',
            'src/core',
            'src/document_processing',
            'src/ai_engine',
            'src/vector_store',
            'src/database',
            'src/ui',
            'data',
            'data/config',
            'data/documents',
            'tests'
        ]
        
        required_files = [
            'src/core/config.py',
            'src/core/app.py',
            'src/document_processing/processor.py',
            'src/ai_engine/llm_client.py',
            'pyproject.toml',
            'requirements.txt'
        ]
        
        structure = {
            'directories': {},
            'files': {}
        }
        
        for dir_path in required_dirs:
            full_path = Path(dir_path)
            exists = full_path.exists() and full_path.is_dir()
            structure['directories'][dir_path] = exists
            
            if not exists:
                self.add_issue("Missing directory", f"Directory not found: {dir_path}")
        
        for file_path in required_files:
            full_path = Path(file_path)
            exists = full_path.exists() and full_path.is_file()
            structure['files'][file_path] = exists
            
            if not exists:
                self.add_issue("Missing file", f"File not found: {file_path}")
            elif full_path.stat().st_size == 0:
                self.add_issue("Empty file", f"File is empty: {file_path}")
        
        self.results['installation'] = structure
    
    def check_data_directories(self):
        """Check data directories and permissions"""
        print("Checking data directories...")
        
        data_dirs = [
            Path.home() / ".docubot",
            Path.home() / ".docubot" / "models",
            Path.home() / ".docubot" / "documents",
            Path.home() / ".docubot" / "database",
            Path.home() / ".docubot" / "logs"
        ]
        
        data_info = {}
        
        for data_dir in data_dirs:
            info = {
                'exists': data_dir.exists(),
                'is_directory': data_dir.exists() and data_dir.is_dir(),
                'writable': False,
                'size_bytes': 0
            }
            
            if data_dir.exists():
                try:
                    test_file = data_dir / ".test_write"
                    test_file.write_text("test")
                    test_file.unlink()
                    info['writable'] = True
                except:
                    self.add_issue("Permission error", f"Cannot write to: {data_dir}")
                
                if data_dir.is_dir():
                    try:
                        total_size = 0
                        for file in data_dir.rglob("*"):
                            if file.is_file():
                                total_size += file.stat().st_size
                        info['size_bytes'] = total_size
                    except:
                        pass
            
            data_info[str(data_dir)] = info
        
        self.results['data'] = data_info
    
    def check_databases(self):
        """Check database files"""
        print("Checking databases...")
        
        db_files = [
            Path.home() / ".docubot" / "database" / "docubot.db",
            Path.home() / ".docubot" / "cache.db"
        ]
        
        for db_file in db_files:
            if db_file.exists():
                try:
                    conn = sqlite3.connect(db_file)
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = cursor.fetchall()
                    conn.close()
                    
                    if not tables:
                        self.add_issue("Empty database", f"No tables in database: {db_file}")
                except Exception as e:
                    self.add_issue("Database error", f"Cannot access database {db_file}: {e}")
    
    def check_configuration(self):
        """Check configuration files"""
        print("Checking configuration...")
        
        config_file = Path.home() / ".docubot" / "config" / "app_config.yaml"
        
        if config_file.exists():
            try:
                import yaml
                with open(config_file, 'r') as f:
                    config = yaml.safe_load(f)
                
                if not config:
                    self.add_issue("Empty configuration", "Configuration file exists but is empty")
            except Exception as e:
                self.add_issue("Configuration error", f"Cannot read configuration: {e}")
        else:
            self.add_issue("Missing configuration", "Configuration file not found")
    
    def check_permissions(self):
        """Check file permissions"""
        print("Checking permissions...")
        
        critical_files = [
            Path.home() / ".docubot" / "secret.key",
            Path.home() / ".docubot" / "database" / "docubot.db"
        ]
        
        for file in critical_files:
            if file.exists():
                mode = file.stat().st_mode
                if mode & 0o077:  # Check if others have write/read permissions
                    self.add_issue("Insecure permissions", f"File has overly permissive permissions: {file}")
    
    def add_issue(self, title: str, description: str, severity: str = "warning"):
        """Add diagnostic issue"""
        self.results['issues'].append({
            'title': title,
            'description': description,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        })
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate diagnostic summary"""
        issues = self.results['issues']
        
        return {
            'total_checks': 8,
            'issues_found': len(issues),
            'critical_issues': len([i for i in issues if i['severity'] == 'critical']),
            'warning_issues': len([i for i in issues if i['severity'] == 'warning']),
            'info_issues': len([i for i in issues if i['severity'] == 'info']),
            'overall_status': 'healthy' if len(issues) == 0 else 'needs_attention'
        }
    
    def print_report(self):
        """Print formatted diagnostic report"""
        print("
" + "=" * 60)
        print("DIAGNOSTIC REPORT")
        print("=" * 60)
        
        summary = self.results['summary']
        print(f"Overall Status: {summary['overall_status'].upper()}")
        print(f"Issues Found: {summary['issues_found']}")
        print(f"Critical: {summary['critical_issues']}, Warnings: {summary['warning_issues']}")
        
        if self.results['issues']:
            print("
ISSUES:")
            for issue in self.results['issues']:
                print(f"
[{issue['severity'].upper()}] {issue['title']}")
                print(f"  {issue['description']}")
        
        print("
SYSTEM INFORMATION:")
        system = self.results['system']
        print(f"  Platform: {system['platform']}")
        print(f"  Python: {system['python_version']}")
        print(f"  Memory: {system['memory_total_gb']:.1f}GB total, {system['memory_available_gb']:.1f}GB available")
        print(f"  Disk: {system['disk_free_gb']:.1f}GB free")
        
        print("
RECOMMENDATIONS:")
        if summary['issues_found'] > 0:
            print("1. Address critical issues first")
            print("2. Install missing dependencies")
            print("3. Ensure proper file permissions")
        else:
            print("All systems operational. No issues detected.")
        
        print("
" + "=" * 60)
    
    def save_report(self, output_file: Path):
        """Save diagnostic report to file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        print(f"Report saved to: {output_file}")


def main():
    """Main diagnostic entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run DocuBot diagnostics")
    parser.add_argument("--output", "-o", type=Path, help="Output JSON file for report")
    parser.add_argument("--quiet", "-q", action="store_true", help="Quiet mode (no console output)")
    
    args = parser.parse_args()
    
    diagnostic = DocuBotDiagnostic()
    results = diagnostic.run_all_checks()
    
    if not args.quiet:
        diagnostic.print_report()
    
    if args.output:
        diagnostic.save_report(args.output)
    
    sys.exit(0 if results['summary']['overall_status'] == 'healthy' else 1)


if __name__ == "__main__":
    main()
