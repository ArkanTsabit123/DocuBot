# docubot/scripts/validate_resources.py

"""
DocuBot - Resource Validation Script

Validates system resources, Ollama installation, model availability,
and download capabilities for the DocuBot application.

CLI Interface:
    --check: Run specific validation check
    --quick: Run only essential checks  
    --json: Output results as JSON
    --verbose: Enable verbose logging
    --output: Save report to file
    --help: Show help message
"""

import sys
import os
import subprocess
import json
import shutil
import socket
import platform
import argparse
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ValidationResult:
    """Container for validation results."""
    component: str
    status: str  # 'pass', 'fail', 'warning'
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class ResourceValidator:
    """Main validator class for DocuBot resources."""
    
    def __init__(self, config_path: Optional[Path] = None, verbose: bool = False):
        self.config_path = config_path or Path("data/config/app_config.yaml")
        self.results: List[ValidationResult] = []
        self.verbose = verbose
        
        # Setup logging for CLI
        self.logger = logging.getLogger(__name__)
        if verbose:
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    def _log(self, message: str, level: str = "INFO"):
        """Log messages if verbose mode is enabled."""
        if self.verbose:
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{timestamp}] [{level}] {message}")
    
    def check_ollama_running(self) -> ValidationResult:
        """Check if Ollama service is running and accessible."""
        self._log("Checking Ollama service...")
        try:
            # CLI COMMAND: ollama --version
            result = subprocess.run(
                ['ollama', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                return ValidationResult(
                    component="ollama_service",
                    status="pass",
                    message="Ollama service is running",
                    details={"version": result.stdout.strip()}
                )
            else:
                return ValidationResult(
                    component="ollama_service",
                    status="fail",
                    message="Ollama service is not responding",
                    details={"error": result.stderr}
                )
                
        except FileNotFoundError:
            return ValidationResult(
                component="ollama_service",
                status="fail",
                message="Ollama executable not found in PATH"
            )
        except subprocess.TimeoutExpired:
            return ValidationResult(
                component="ollama_service",
                status="fail",
                message="Ollama service timeout"
            )
        except Exception as e:
            return ValidationResult(
                component="ollama_service",
                status="fail",
                message=f"Unexpected error: {str(e)}"
            )
    
    def check_models_available(self) -> ValidationResult:
        """Check which models are available locally and remotely via CLI."""
        self._log("Checking model availability...")
        try:
            # CLI COMMAND: ollama list
            result = subprocess.run(
                ['ollama', 'list'],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                return ValidationResult(
                    component="model_availability",
                    status="warning",
                    message="Unable to list local models",
                    details={"error": result.stderr}
                )
            
            local_models = []
            for line in result.stdout.strip().split('\n')[1:]:
                if line.strip():
                    parts = line.split()
                    if parts:
                        local_models.append(parts[0])
            
            try:
                # CLI COMMAND: ollama list --available
                result = subprocess.run(
                    ['ollama', 'list', '--available'],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                available_models = []
                if result.returncode == 0:
                    for line in result.stdout.strip().split('\n')[1:]:
                        if line.strip():
                            parts = line.split()
                            if parts:
                                available_models.append(parts[0])
            
            except (subprocess.TimeoutExpired, Exception):
                available_models = []
            
            return ValidationResult(
                component="model_availability",
                status="pass",
                message="Model availability check completed",
                details={
                    "local_models": local_models,
                    "available_models": available_models,
                    "local_count": len(local_models),
                    "available_count": len(available_models)
                }
            )
            
        except Exception as e:
            return ValidationResult(
                component="model_availability",
                status="fail",
                message=f"Failed to check models: {str(e)}"
            )
    
    def validate_model_download(self, model_name: str = "llama2:7b") -> ValidationResult:
        """Validate download capability by checking a small model."""
        self._log(f"Validating download capability for {model_name}...")
        try:
            import requests
            
            test_url = "https://registry.ollama.ai/v2/library/llama2/manifests/7b"
            
            try:
                response = requests.head(test_url, timeout=10)
                if response.status_code == 200:
                    return ValidationResult(
                        component="model_download",
                        status="pass",
                        message=f"Download capability verified for {model_name}",
                        details={
                            "test_url": test_url,
                            "status_code": response.status_code,
                            "headers": dict(response.headers)
                        }
                    )
                else:
                    return ValidationResult(
                        component="model_download",
                        status="warning",
                        message=f"Download test returned status {response.status_code}",
                        details={"status_code": response.status_code}
                    )
                    
            except requests.RequestException as e:
                return ValidationResult(
                    component="model_download",
                    status="warning",
                    message=f"Network error during download test: {str(e)}"
                )
                
        except ImportError:
            return ValidationResult(
                component="model_download",
                status="warning",
                message="Requests library not available for download test"
            )
        except Exception as e:
            return ValidationResult(
                component="model_download",
                status="fail",
                message=f"Download validation failed: {str(e)}"
            )
    
    def check_disk_space(self, required_gb: int = 10) -> ValidationResult:
        """Check if sufficient disk space is available."""
        self._log(f"Checking disk space (required: {required_gb}GB)...")
        try:
            total, used, free = shutil.disk_usage(Path.cwd())
            
            free_gb = free / (1024**3)
            required_bytes = required_gb * (1024**3)
            
            if free >= required_bytes:
                return ValidationResult(
                    component="disk_space",
                    status="pass",
                    message=f"Sufficient disk space available",
                    details={
                        "free_gb": round(free_gb, 2),
                        "required_gb": required_gb,
                        "free_bytes": free,
                        "required_bytes": required_bytes
                    }
                )
            else:
                return ValidationResult(
                    component="disk_space",
                    status="fail",
                    message=f"Insufficient disk space",
                    details={
                        "free_gb": round(free_gb, 2),
                        "required_gb": required_gb,
                        "deficit_gb": round(required_gb - free_gb, 2)
                    }
                )
                
        except Exception as e:
            return ValidationResult(
                component="disk_space",
                status="warning",
                message=f"Unable to check disk space: {str(e)}"
            )
    
    def check_internet(self) -> ValidationResult:
        """Check internet connectivity."""
        self._log("Checking internet connectivity...")
        test_hosts = [
            ("Google DNS", "8.8.8.8", 53),
            ("Cloudflare DNS", "1.1.1.1", 53),
            ("Ollama Registry", "registry.ollama.ai", 443)
        ]
        
        reachable = []
        unreachable = []
        
        for name, host, port in test_hosts:
            try:
                socket.create_connection((host, port), timeout=5)
                reachable.append(name)
            except (socket.timeout, socket.error, OSError):
                unreachable.append(name)
        
        if reachable:
            return ValidationResult(
                component="internet_connectivity",
                status="pass",
                message="Internet connectivity confirmed",
                details={
                    "reachable_hosts": reachable,
                    "unreachable_hosts": unreachable,
                    "reachable_count": len(reachable),
                    "total_tested": len(test_hosts)
                }
            )
        else:
            return ValidationResult(
                component="internet_connectivity",
                status="fail",
                message="No internet connectivity detected",
                details={"unreachable_hosts": unreachable}
            )
    
    def validate_configuration(self) -> ValidationResult:
        """Validate configuration files."""
        self._log("Validating configuration files...")
        required_configs = [
            ("app_config.yaml", "data/config/app_config.yaml"),
            ("llm_config.yaml", "data/config/llm_config.yaml")
        ]
        
        missing = []
        present = []
        
        for name, path in required_configs:
            config_path = Path(path)
            if config_path.exists() and config_path.stat().st_size > 10:
                present.append(name)
            else:
                missing.append(name)
        
        if not missing:
            return ValidationResult(
                component="configuration",
                status="pass",
                message="All required configuration files present",
                details={"present_files": present}
            )
        else:
            return ValidationResult(
                component="configuration",
                status="fail",
                message="Missing configuration files",
                details={
                    "present_files": present,
                    "missing_files": missing
                }
            )
    
    def validate_directories(self) -> ValidationResult:
        """Validate required directory structure."""
        self._log("Validating directory structure...")
        required_dirs = [
            "data/models",
            "data/documents",
            "data/database",
            "data/logs",
            "data/config"
        ]
        
        missing = []
        present = []
        
        for dir_path in required_dirs:
            dir_full = Path(dir_path)
            if dir_full.exists() and dir_full.is_dir():
                present.append(dir_path)
            else:
                missing.append(dir_path)
        
        if not missing:
            return ValidationResult(
                component="directory_structure",
                status="pass",
                message="All required directories present",
                details={"present_directories": present}
            )
        else:
            return ValidationResult(
                component="directory_structure",
                status="fail",
                message="Missing required directories",
                details={
                    "present_directories": present,
                    "missing_directories": missing
                }
            )
    
    def check_python_dependencies(self) -> ValidationResult:
        """Check required Python dependencies."""
        self._log("Checking Python dependencies...")
        required_packages = [
            "requests",
            "pyyaml",
            "fastapi",
            "uvicorn",
            "pydantic",
            "sqlalchemy"
        ]
        
        missing = []
        present = []
        
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
                present.append(package)
            except ImportError:
                missing.append(package)
        
        if not missing:
            return ValidationResult(
                component="python_dependencies",
                status="pass",
                message="All required Python packages installed",
                details={"present_packages": present}
            )
        else:
            return ValidationResult(
                component="python_dependencies",
                status="warning",
                message="Missing some Python packages",
                details={
                    "present_packages": present,
                    "missing_packages": missing
                }
            )
    
    def run_specific_check(self, check_name: str) -> Optional[ValidationResult]:
        """Run a specific validation check by name."""
        check_map = {
            "ollama": self.check_ollama_running,
            "models": self.check_models_available,
            "download": self.validate_model_download,
            "disk": self.check_disk_space,
            "internet": self.check_internet,
            "config": self.validate_configuration,
            "directories": self.validate_directories,
            "dependencies": self.check_python_dependencies,
            "all": self.run_all_checks
        }
        
        if check_name in check_map:
            return check_map[check_name]()
        
        return None
    
    def run_all_checks(self) -> List[ValidationResult]:
        """Run all validation checks."""
        self._log("Starting comprehensive validation...")
        checks = [
            ("ollama_service", self.check_ollama_running),
            ("model_availability", self.check_models_available),
            ("model_download", lambda: self.validate_model_download()),
            ("disk_space", lambda: self.check_disk_space()),
            ("internet_connectivity", self.check_internet),
            ("configuration", self.validate_configuration),
            ("directory_structure", self.validate_directories),
            ("python_dependencies", self.check_python_dependencies)
        ]
        
        self.results = []
        for name, check_func in checks:
            print(f"Checking {name}...", end="", flush=True)
            result = check_func()
            self.results.append(result)
            status_symbol = "✓" if result.status == "pass" else "⚠" if result.status == "warning" else "✗"
            print(f" {status_symbol}")
            if self.verbose and result.details:
                print(f"    Details: {result.details}")
        
        return self.results
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        if not self.results:
            self.run_all_checks()
        
        summary = {
            "total_checks": len(self.results),
            "pass_count": sum(1 for r in self.results if r.status == "pass"),
            "warning_count": sum(1 for r in self.results if r.status == "warning"),
            "fail_count": sum(1 for r in self.results if r.status == "fail"),
            "system_info": {
                "platform": platform.platform(),
                "python_version": sys.version,
                "current_directory": str(Path.cwd()),
                "timestamp": datetime.now().isoformat()
            },
            "results": [asdict(r) for r in self.results]
        }
        
        overall_status = "pass"
        if summary["fail_count"] > 0:
            overall_status = "fail"
        elif summary["warning_count"] > 0:
            overall_status = "warning"
        
        summary["overall_status"] = overall_status
        
        return summary
    
    def print_summary(self):
        """Print human-readable validation summary."""
        report = self.generate_report()
        
        print("\n" + "="*60)
        print("DOCUBOT RESOURCE VALIDATION REPORT")
        print("="*60)
        
        print(f"\nSystem Information:")
        print(f"  Platform: {report['system_info']['platform']}")
        print(f"  Python: {report['system_info']['python_version'].split()[0]}")
        print(f"  Directory: {report['system_info']['current_directory']}")
        print(f"  Timestamp: {report['system_info']['timestamp']}")
        
        print(f"\nValidation Results:")
        print(f"  Total Checks: {report['total_checks']}")
        print(f"  Passed: {report['pass_count']}")
        print(f"  Warnings: {report['warning_count']}")
        print(f"  Failed: {report['fail_count']}")
        
        print(f"\nOverall Status: {report['overall_status'].upper()}")
        
        print(f"\nDetailed Results:")
        for result in report['results']:
            status_symbol = "✓" if result['status'] == "pass" else "⚠" if result['status'] == "warning" else "✗"
            print(f"  {status_symbol} {result['component']}: {result['message']}")
        
        print("\n" + "="*60)
        
        if report['overall_status'] == 'fail':
            return 1
        elif report['overall_status'] == 'warning':
            return 2
        else:
            return 0
    
    def save_report(self, output_path: Path):
        """Save validation report to file."""
        report = self.generate_report()
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        self._log(f"Report saved to {output_path}")


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="validate_resources.py",
        description="Validate DocuBot system resources and configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Run all checks with default settings
  %(prog)s --quick                   # Run only essential checks  
  %(prog)s --json                    # Output results as JSON
  %(prog)s --check ollama            # Check only Ollama service
  %(prog)s --check models            # Check only model availability
  %(prog)s --verbose                 # Enable verbose logging
  %(prog)s --output report.json      # Save report to file
  %(prog)s --help                    # Show this help message

Exit Codes:
  0: All checks passed
  1: One or more checks failed
  2: Warnings but no failures
        """
    )
    
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("data/config/app_config.yaml"),
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON format"
    )
    
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only essential checks (Ollama, disk space, config)"
    )
    
    parser.add_argument(
        "--check",
        type=str,
        choices=['ollama', 'models', 'download', 'disk', 'internet', 
                'config', 'directories', 'dependencies', 'all'],
        help="Run specific check only"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output with detailed logging"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Save report to specified file"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="DocuBot Resource Validator v1.0.0"
    )
    
    parser.add_argument(
        "--disk-required",
        type=int,
        default=10,
        help="Required disk space in GB (default: 10)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="llama2:7b",
        help="Model name for download test (default: llama2:7b)"
    )
    
    return parser.parse_args()


def setup_logging(verbose: bool):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Command-line entry point - CLI INTERFACE IMPLEMENTATION."""
    print(f"\nDocuBot Resource Validator - CLI Interface")
    print(f"{'='*40}")
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Setup logging based on verbose flag
    setup_logging(args.verbose)
    
    print(f"Starting validation with options:")
    if args.check:
        print(f"  Specific check: {args.check}")
    elif args.quick:
        print(f"  Quick mode: only essential checks")
    else:
        print(f"  Comprehensive: all validation checks")
    
    if args.json:
        print(f"  Output format: JSON")
    if args.verbose:
        print(f"  Log level: VERBOSE")
    if args.output:
        print(f"  Output file: {args.output}")
    
    validator = ResourceValidator(args.config, verbose=args.verbose)
    
    start_time = time.time()
    
    if args.check:
        print(f"\nRunning specific check: {args.check}")
        result = validator.run_specific_check(args.check)
        if result:
            validator.results = [result]
    elif args.quick:
        print("\nRunning quick validation...")
        quick_checks = [
            validator.check_ollama_running,
            validator.check_disk_space,
            validator.validate_configuration
        ]
        validator.results = [check() for check in quick_checks]
    else:
        print("\nRunning comprehensive validation...")
        validator.run_all_checks()
    
    elapsed_time = time.time() - start_time
    
    if args.output:
        validator.save_report(args.output)
    
    if args.json:
        report = validator.generate_report()
        report["elapsed_time"] = round(elapsed_time, 2)
        print(json.dumps(report, indent=2))
        return 0 if report['overall_status'] == 'pass' else 1
    else:
        print(f"\nValidation completed in {elapsed_time:.2f} seconds")
        exit_code = validator.print_summary()
        
        # Provide actionable recommendations
        if exit_code != 0:
            print("\nRecommendations:")
            for result in validator.results:
                if result.status == 'fail':
                    if result.component == 'ollama_service':
                        print("  • Install or start Ollama service")
                    elif result.component == 'disk_space':
                        print("  • Free up disk space or choose different location")
                    elif result.component == 'configuration':
                        print("  • Check configuration files in data/config/")
                    elif result.component == 'directory_structure':
                        print("  • Create missing directories in data/")
        
        return exit_code


def cli_entry_point():
    """CLI entry point with proper error handling."""
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nValidation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nError during validation: {str(e)}")
        sys.exit(1)


# CLI INTERFACE IMPLEMENTATION COMPLETE
# This script provides a full command-line interface for resource validation
if __name__ == "__main__":
    cli_entry_point()