#docubot/scripts/setup.py
"""
DocuBot Setup Script - Python 3.13 Compatible
"""

import sys
import os
import platform
import subprocess
import importlib.util
from pathlib import Path

def print_header(text):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)

def check_python_version():
    """Check Python version requirement."""
    print("Checking Python version...")
    major, minor, micro = sys.version_info[:3]
    
    if major < 3 or (major == 3 and minor < 11):
        print(f"ERROR: Python 3.11+ required. Found {major}.{minor}.{micro}")
        return False
    else:
        print(f"SUCCESS: Python {major}.{minor}.{micro}")
        return True

def install_with_pip(package_spec, upgrade_pip=True):
    """Install a package with pip."""
    try:
        cmd = [sys.executable, "-m", "pip", "install"]
        
        if upgrade_pip:
            # Try with --break-system-packages for newer Python
            cmd.append("--break-system-packages")
        
        cmd.append(package_spec)
        
        print(f"  Installing: {package_spec}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
            timeout=300
        )
        
        if result.returncode == 0:
            print(f"    ✓ SUCCESS: {package_spec}")
            return True
        else:
            print(f"    ✗ FAILED: {package_spec}")
            if "error" in result.stderr.lower():
                error_lines = [line for line in result.stderr.split('\n') if 'error' in line.lower()]
                for err in error_lines[:3]:
                    print(f"      {err[:100]}...")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"    ⏱ TIMEOUT: {package_spec}")
        return False
    except Exception as e:
        print(f"    ✗ ERROR: {package_spec} - {e}")
        return False

def install_requirements_313():
    """Install packages optimized for Python 3.13."""
    print_header("INSTALLING PACKAGES")
    
    requirements_file = Path(__file__).parent.parent / "requirements.txt"
    
    if not requirements_file.exists():
        print(f"ERROR: requirements.txt not found at {requirements_file}")
        return False
    
    print(f"Using requirements from: {requirements_file}")
    
    # First, upgrade pip
    print("\n1. Upgrading pip...")
    install_with_pip("pip", upgrade_pip=False)
    
    # Read requirements
    with open(requirements_file, 'r', encoding='utf-8') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    print(f"\n2. Found {len(requirements)} packages to install")
    
    # Group by priority
    priority_1 = []  # Core framework
    priority_2 = []  # Document processing
    priority_3 = []  # AI/ML
    priority_4 = []  # Optional
    
    for req in requirements:
        req_lower = req.lower()
        
        if any(keyword in req_lower for keyword in [
            'fastapi', 'uvicorn', 'pydantic', 'pyyaml', 
            'requests', 'python-dotenv', 'click', 'rich'
        ]):
            priority_1.append(req)
        elif any(keyword in req_lower for keyword in [
            'pypdf', 'docx', 'beautifulsoup', 'lxml', 
            'pillow', 'markdown', 'customtkinter'
        ]):
            priority_2.append(req)
        elif any(keyword in req_lower for keyword in [
            'torch', 'transformers', 'sentence', 'numpy',
            'scipy', 'scikit', 'chromadb', 'sqlalchemy'
        ]):
            priority_3.append(req)
        else:
            priority_4.append(req)
    
    # Install in priority order
    results = {
        'priority_1': {'total': len(priority_1), 'success': 0},
        'priority_2': {'total': len(priority_2), 'success': 0},
        'priority_3': {'total': len(priority_3), 'success': 0},
        'priority_4': {'total': len(priority_4), 'success': 0},
    }
    
    print("\n3. Installing PRIORITY 1 - Core Framework")
    for package in priority_1:
        if install_with_pip(package):
            results['priority_1']['success'] += 1
    
    print("\n4. Installing PRIORITY 2 - Document Processing")
    for package in priority_2:
        if install_with_pip(package):
            results['priority_2']['success'] += 1
    
    print("\n5. Installing PRIORITY 3 - AI/ML")
    for package in priority_3:
        if install_with_pip(package):
            results['priority_3']['success'] += 1
    
    print("\n6. Installing PRIORITY 4 - Optional")
    for package in priority_4:
        if install_with_pip(package):
            results['priority_4']['success'] += 1
    
    # Summary
    print_header("INSTALLATION SUMMARY")
    
    total_packages = sum(r['total'] for r in results.values())
    total_success = sum(r['success'] for r in results.values())
    
    print(f"Total packages: {total_packages}")
    print(f"Successfully installed: {total_success}")
    print(f"Failed: {total_packages - total_success}")
    
    print("\nBreakdown:")
    for key, data in results.items():
        success_rate = (data['success'] / data['total']) * 100 if data['total'] > 0 else 100
        print(f"  {key}: {data['success']}/{data['total']} ({success_rate:.1f}%)")
    
    # Consider installation successful if core packages are installed
    core_success_rate = results['priority_1']['success'] / results['priority_1']['total'] if results['priority_1']['total'] > 0 else 0
    
    if core_success_rate >= 0.8:  # 80% of core packages
        print(f"\n✓ SUCCESS: Core packages installed ({core_success_rate*100:.1f}%)")
        return True
    else:
        print(f"\n✗ FAILED: Not enough core packages installed ({core_success_rate*100:.1f}%)")
        return False

def create_directories_simple():
    """Create necessary directories."""
    print_header("CREATING DIRECTORIES")
    
    project_root = Path(__file__).parent.parent
    
    # Define directories
    directories = [
        # Project directories
        project_root / "data",
        project_root / "data" / "config",
        project_root / "data" / "logs",
        project_root / "data" / "database",
        project_root / "data" / "models",
        project_root / "data" / "documents",
        project_root / "data" / "exports",
        project_root / "data" / "cache",
        
        # User directories (platform-specific)
        Path.home() / ".docubot",
        Path.home() / ".docubot" / "config",
        Path.home() / ".docubot" / "documents",
        Path.home() / ".docubot" / "models",
    ]
    
    created = 0
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            if directory.exists():
                print(f"  ✓ {directory}")
                created += 1
            else:
                print(f"  ✗ Failed: {directory}")
        except Exception as e:
            print(f"  ✗ Error: {directory} - {e}")
    
    print(f"\nCreated {created} directories")
    return created > 0

def check_ollama_simple():
    """Simple Ollama check."""
    print_header("CHECKING OLLAMA")
    
    try:
        # Try to get version
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False
        )
        
        if result.returncode == 0:
            print(f"✓ Ollama installed: {result.stdout.strip()}")
            
            # Check if service is running
            try:
                import requests
                response = requests.get("http://localhost:11434/api/tags", timeout=3)
                if response.status_code == 200:
                    print("✓ Ollama service is running")
                    models = response.json().get('models', [])
                    if models:
                        print(f"  Available models: {len(models)}")
                    else:
                        print("  No models installed yet")
                else:
                    print("⚠ Ollama service not responding")
            except:
                print("⚠ Could not check Ollama service status")
            
            return True
        else:
            print("⚠ Ollama found but command failed")
            return False
            
    except FileNotFoundError:
        print("ℹ Ollama not installed (optional)")
        print("\nTo install Ollama for AI features:")
        print("  1. Download from https://ollama.ai")
        print("  2. Install and run: ollama pull llama2:7b")
        print("  3. Start service: ollama serve")
        return False
    except Exception as e:
        print(f"⚠ Ollama check error: {e}")
        return False

def verify_core_installation():
    """Verify that core packages are installed."""
    print_header("VERIFYING INSTALLATION")
    
    # Core packages to verify
    core_packages = [
        ("fastapi", "fastapi"),
        ("uvicorn", "uvicorn"),
        ("pydantic", "pydantic"),
        ("PyYAML", "yaml"),
        ("requests", "requests"),
        ("customtkinter", "customtkinter"),
        ("PyPDF2", "PyPDF2"),
        ("chromadb", "chromadb"),
    ]
    
    installed = []
    missing = []
    
    for pip_name, import_name in core_packages:
        try:
            spec = importlib.util.find_spec(import_name)
            if spec is None:
                raise ImportError
            installed.append(pip_name)
            print(f"  ✓ {pip_name}")
        except ImportError:
            missing.append(pip_name)
            print(f"  ✗ {pip_name}")
    
    print(f"\nVerification: {len(installed)}/{len(core_packages)} packages installed")
    
    if len(missing) == 0:
        print("✓ All core packages installed")
        return True
    elif len(missing) <= 2:
        print(f"⚠ {len(missing)} packages missing: {', '.join(missing)}")
        print("  Some features may be limited")
        return True
    else:
        print(f"✗ {len(missing)} packages missing: {', '.join(missing)}")
        return False

def display_next_steps():
    """Display next steps after setup."""
    print_header("SETUP COMPLETE")
    
    print("\nNEXT STEPS:")
    print("1. Test the installation:")
    print("   python -c \"import fastapi; import chromadb; print('Core packages OK')\"")
    
    print("\n2. Run DocuBot:")
    print("   GUI mode:   python app.py")
    print("   CLI mode:   python app.py cli")
    print("   Web mode:   python app.py web")
    
    print("\n3. For AI features, install Ollama:")
    print("   - Download from https://ollama.ai")
    print("   - Run: ollama pull llama2:7b")
    print("   - Start: ollama serve")
    
    print("\n4. Add your documents:")
    print("   - Place files in: ~/.docubot/documents/")
    print("   - Or use the upload feature in the app")
    
    print("\n5. Start querying your documents!")
    print("=" * 60)

def main():
    """Main setup function."""
    print("=" * 60)
    print(" DOCUBOT - PYTHON 3.13 SETUP")
    print("=" * 60)
    
    print(f"\nPython: {platform.python_version()}")
    print(f"System: {platform.system()} {platform.release()}")
    print(f"Arch: {platform.machine()}")
    
    # Run setup steps
    steps = [
        ("Python Version", check_python_version, True),
        ("Package Installation", install_requirements_313, True),
        ("Directory Setup", create_directories_simple, True),
        ("Installation Verify", verify_core_installation, True),
        ("Ollama Check", check_ollama_simple, False),
    ]
    
    results = []
    critical_failures = 0
    
    for step_name, step_func, is_critical in steps:
        print_header(step_name.upper())
        
        try:
            success = step_func()
            status = "PASS" if success else ("WARNING" if not is_critical else "FAIL")
            results.append((step_name, success, is_critical, status))
            
            if not success and is_critical:
                critical_failures += 1
                
        except Exception as e:
            print(f"ERROR: {e}")
            results.append((step_name, False, is_critical, "ERROR"))
            if is_critical:
                critical_failures += 1
    
    # Final summary
    print_header("SETUP SUMMARY")
    
    for step_name, success, is_critical, status in results:
        critical_marker = " (CRITICAL)" if is_critical else ""
        print(f"{status:8} {step_name}{critical_marker}")
    
    # Final decision
    if critical_failures > 0:
        print(f"\n✗ SETUP FAILED with {critical_failures} critical error(s)")
        print("\nTroubleshooting:")
        print("1. Update pip: python -m pip install --upgrade pip")
        print("2. Try manual install: pip install fastapi uvicorn pydantic")
        print("3. Check Python version: Should be 3.11, 3.12, or 3.13")
        return False
    else:
        display_next_steps()
        return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user")
        sys.exit(130)