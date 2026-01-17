#docubot/install_docubot.py

"""
DocuBot Complete Dependency Installer
Installs all required packages for the DocuBot AI document processing system.
"""

import subprocess
import sys
import os
from pathlib import Path


def execute_command(command: str, description: str) -> bool:
    """
    Execute a shell command and report success or failure.
    
    Args:
        command: The shell command to execute
        description: Human-readable description of what the command does
        
    Returns:
        bool: True if command succeeded, False otherwise
    """
    print(f"Installing: {description}")
    print(f"Command: {command}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True
        )
        print(f"Status: Success")
        if result.stdout:
            print(f"Output: {result.stdout[:200]}")
        return True
    except subprocess.CalledProcessError as error:
        print(f"Status: Failed - {error}")
        if error.stderr:
            print(f"Error details: {error.stderr[:500]}")
        return False


def verify_python_version() -> bool:
    """Verify Python version meets minimum requirements."""
    print(f"Python version detected: {sys.version}")
    
    if sys.version_info < (3, 11):
        print("Error: Python 3.11 or higher is required")
        return False
    return True


def install_dependencies():
    """Install all required dependencies for DocuBot."""
    
    print("=" * 60)
    print("DOCUBOT DEPENDENCY INSTALLATION")
    print("=" * 60)
    
    if not verify_python_version():
        return
    
    execute_command(
        "python -m pip install --upgrade pip", 
        "Upgrading package installer"
    )
    
    installation_groups = [
        {
            "name": "Core packages",
            "packages": [
                "pydantic==2.5.3",
                "numpy==1.24.3",
                "requests==2.31.0",
                "pyyaml==6.0.1",
                "tqdm==4.66.1",
            ]
        },
        {
            "name": "PyTorch framework",
            "packages": [
                "torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu"
            ]
        },
        {
            "name": "AI and machine learning packages",
            "packages": [
                "sentence-transformers==2.2.2",
                "transformers==4.36.0",
                "accelerate==0.25.0",
                "scipy==1.11.4",
                "scikit-learn==1.3.2",
            ]
        },
        {
            "name": "Vector database and language chain",
            "packages": [
                "chromadb==0.4.15",
                "hnswlib==0.7.0",
                "langchain==0.0.350",
                "langchain-community==0.0.10",
            ]
        },
        {
            "name": "Document processing libraries",
            "packages": [
                "pypdf2==3.0.1",
                "pdfplumber==0.10.3",
                "python-docx==0.8.11",
                "ebooklib==0.18",
                "beautifulsoup4==4.12.2",
                "lxml==4.9.3",
                "pandas==2.1.4",
                "pytesseract==0.3.10",
                "Pillow==10.1.0",
                "markdown==3.5.1",
            ]
        },
        {
            "name": "Database management",
            "packages": [
                "sqlalchemy==2.0.23",
                "alembic==1.12.1"
            ]
        },
        {
            "name": "Graphical user interface",
            "packages": [
                "customtkinter==5.1.0",
                "tkinterweb==3.11.0",
            ]
        },
        {
            "name": "Utility packages",
            "packages": [
                "fastapi==0.104.1",
                "uvicorn[standard]==0.24.0",
                "python-dotenv==1.0.0",
                "structlog==23.2.0",
                "chardet==5.2.0",
                "click==8.1.7",
                "rich==13.7.0",
                "aiohttp==3.9.1",
                "watchdog==3.0.0",
                "cachetools==5.3.2",
                "orjson==3.9.10",
            ]
        },
        {
            "name": "Development and testing tools",
            "packages": [
                "pytest==7.4.3",
                "pytest-asyncio==0.21.1",
                "pytest-cov==4.1.0",
                "black==23.11.0",
                "flake8==6.1.0",
                "mypy==1.7.0",
            ]
        }
    ]
    
    for group in installation_groups:
        print(f"\n{'='*40}")
        print(f"Category: {group['name']}")
        print(f"{'='*40}")
        
        for package in group["packages"]:
            execute_command(
                f"pip install {package}", 
                f"Package: {package.split('==')[0]}"
            )
    
    validate_installation()


def validate_installation():
    """Validate that all critical packages imported successfully."""
    
    print("\n" + "=" * 60)
    print("INSTALLATION VALIDATION")
    print("=" * 60)
    
    critical_modules = [
        "pydantic",
        "torch",
        "sentence_transformers",
        "chromadb",
        "customtkinter",
        "sqlalchemy",
        "fastapi",
        "pypdf2",
    ]
    
    failed_imports = []
    
    for module_name in critical_modules:
        try:
            __import__(module_name.replace("-", "_"))
            print(f"Successful import: {module_name}")
        except ImportError as import_error:
            print(f"Failed import: {module_name} - {import_error}")
            failed_imports.append(module_name)
    
    print("\n" + "=" * 60)
    
    if failed_imports:
        print(f"Installation completed with {len(failed_imports)} errors")
        print(f"Failed modules: {', '.join(failed_imports)}")
    else:
        print("All dependencies installed successfully")
    
    print("\nPost-installation steps:")
    print("1. Execute system validation: python smart_validator.py")
    print("2. Test language model integration: python test_llm_fix.py")
    print("3. Launch application: python app.py")


def main():
    """Main execution function."""
    install_dependencies()


if __name__ == "__main__":
    main()