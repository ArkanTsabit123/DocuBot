# docubot/install_docubot.py

"""
DocuBot Complete Dependency Installer
Version: 1.1.0 | Python: 3.12.8
Installs all required packages for the DocuBot AI document processing system.
"""

import subprocess
import sys
import os


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
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True,
            encoding='utf-8'
        )
        print(f"Status: Success")
        return True
    except subprocess.CalledProcessError as error:
        print(f"Status: Failed - {error}")
        if error.stderr:
            print(f"Error details: {error.stderr[:300]}")
        return False


def verify_python_version() -> bool:
    """Verify Python version meets minimum requirements."""
    print(f"Python version: {sys.version}")
    
    if sys.version_info < (3, 12):
        print("Warning: Python 3.12 or higher is recommended")
        print("Continue anyway? (y/n): ", end="")
        response = input().strip().lower()
        return response == 'y'
    return True


def install_from_requirements() -> bool:
    """Install dependencies directly from requirements.txt."""
    print("\n" + "=" * 60)
    print("INSTALLING FROM REQUIREMENTS.TXT")
    print("=" * 60)
    
    req_file = "requirements.txt"
    if not os.path.exists(req_file):
        print(f"Error: {req_file} not found in current directory")
        return False
    
    return execute_command(
        f"pip install -r {req_file}",
        "All packages from requirements.txt"
    )


def install_core_packages():
    """Install packages individually with version control."""
    
    installation_groups = [
        {
            "name": "Core Dependencies",
            "packages": [
                "setuptools>=68.2.0",
                "wheel>=0.41.0",
                "pip>=23.0.0",
            ]
        },
        {
            "name": "AI and Machine Learning",
            "packages": [
                "torch>=2.2.0,<2.3.0",
                "torchvision>=0.17.0,<0.18.0",
                "torchaudio>=2.2.0,<2.3.0",
                "transformers>=4.36.0,<5.0.0",
                "sentence-transformers>=2.2.2,<3.0.0",
                "accelerate>=0.25.0,<0.30.0",
                "tokenizers>=0.15.0,<0.16.0",
                "numpy>=1.24.0,<2.0.0",
                "scipy>=1.11.0,<1.12.0",
                "scikit-learn>=1.3.0,<1.5.0",
            ]
        },
        {
            "name": "LangChain Ecosystem",
            "packages": [
                "langchain>=0.1.20,<0.2.0",
                "langchain-community>=0.0.38,<0.1.0",
                "langchain-core>=0.1.53,<0.2.0",
                "langchain-text-splitters>=0.0.2,<0.1.0",
                "langchain-chroma>=0.1.4,<0.2.0",
            ]
        },
        {
            "name": "Vector Database",
            "packages": [
                "chromadb>=0.4.18,<0.5.0",
                "hnswlib>=0.7.0,<0.8.0",
            ]
        },
        {
            "name": "Document Processing",
            "packages": [
                "PyPDF2>=3.0.1,<4.0.0",
                "pdfplumber>=0.10.2,<0.11.0",
                "pypdf>=3.17.0,<4.0.0",
                "python-docx>=0.8.11,<1.0.0",
                "openpyxl>=3.1.0,<4.0.0",
                "beautifulsoup4>=4.12.0,<5.0.0",
                "lxml>=4.9.0,<5.0.0",
                "pandas>=2.1.0,<2.2.0",
                "Pillow>=10.0.0,<11.0.0",
            ]
        },
        {
            "name": "Desktop Interface",
            "packages": [
                "customtkinter>=5.2.0,<6.0.0",
                "tkinterweb>=3.14.0,<4.0.0",
            ]
        },
        {
            "name": "Database and Storage",
            "packages": [
                "sqlalchemy>=2.0.0,<3.0.0",
                "alembic>=1.13.0,<2.0.0",
            ]
        },
        {
            "name": "Utilities and Configuration",
            "packages": [
                "pyyaml>=6.0.0,<7.0.0",
                "python-dotenv>=1.0.0,<2.0.0",
                "requests>=2.31.0,<3.0.0",
                "aiohttp>=3.9.0,<4.0.0",
                "pydantic>=2.5.0,<3.0.0",
                "pydantic-settings>=2.1.0,<3.0.0",
                "orjson>=3.9.0,<4.0.0",
                "cryptography>=41.0.0,<43.0.0",
            ]
        }
    ]
    
    for group in installation_groups:
        print(f"\nCategory: {group['name']}")
        print("-" * 40)
        
        for package in group["packages"]:
            pkg_name = package.split('>')[0].split('<')[0].split('=')[0].strip()
            execute_command(f"pip install {package}", pkg_name)


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
    
    print("\nInstallation methods:")
    print("1. Install from requirements.txt (recommended)")
    print("2. Install packages individually with version control")
    
    choice = input("\nEnter your choice (1 or 2): ").strip()
    
    if choice == "1":
        success = install_from_requirements()
        if not success:
            print("\nFalling back to individual package installation...")
            install_core_packages()
    else:
        install_core_packages()
    
    validate_installation()


def validate_installation():
    """Validate that all critical packages imported successfully."""
    
    print("\n" + "=" * 60)
    print("INSTALLATION VALIDATION")
    print("=" * 60)
    
    critical_modules = [
        "torch",
        "transformers",
        "langchain",
        "chromadb",
        "sqlalchemy",
        "pydantic",
        "pandas",
        "customtkinter",
    ]
    
    failed_imports = []
    
    for module_name in critical_modules:
        try:
            __import__(module_name.replace("-", "_"))
            print(f"Success: {module_name}")
        except ImportError:
            print(f"Failed: {module_name}")
            failed_imports.append(module_name)
    
    print("\n" + "=" * 60)
    
    if failed_imports:
        print(f"Installation completed with {len(failed_imports)} errors")
        print(f"Failed modules: {', '.join(failed_imports)}")
    else:
        print("All dependencies installed successfully")
    
    print("\nNext steps:")
    print("1. Execute system validation: python smart_validator.py")
    print("2. Test language model integration: python test_llm_fix.py")
    print("3. Launch application: python app.py")


def main():
    """Main execution function."""
    try:
        install_dependencies()
    except KeyboardInterrupt:
        print("\nInstallation cancelled.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()