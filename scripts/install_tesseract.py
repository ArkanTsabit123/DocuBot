#!/usr/bin/env python3
"""
Tesseract OCR Installation Script for DocuBot
Installs Tesseract OCR on Windows, macOS, and Linux
"""

import os
import sys
import platform
import subprocess
from pathlib import Path
import urllib.request
import zipfile
import tarfile

def print_header(text: str):
    """Print formatted header"""
    print("=" * 60)
    print(text)
    print("=" * 60)

def print_step(step_num: int, description: str):
    """Print step information"""
    print(f"\n[{step_num}] {description}")
    print("-" * 40)

def check_tesseract_installed() -> bool:
    """Check if Tesseract is already installed"""
    try:
        result = subprocess.run(
            ['tesseract', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            print(f"✓ Tesseract is already installed: {result.stdout.split('\\n')[0]}")
            return True
    except (subprocess.SubprocessError, FileNotFoundError):
        pass
    return False

def install_tesseract_windows() -> bool:
    """Install Tesseract on Windows"""
    print_step(1, "Windows Tesseract Installation")
    
    installer_url = "https://github.com/UB-Mannheim/tesseract/wiki"
    
    print("For Windows, Tesseract must be installed manually:")
    print(f"1. Download installer from: {installer_url}")
    print("2. Run the installer")
    print("3. Add Tesseract to PATH (usually C:\\Program Files\\Tesseract-OCR)")
    print("")
    print("After installation, verify with: tesseract --version")
    
    choice = input("\nDo you want to open the download page? (y/n): ").lower()
    if choice == 'y':
        import webbrowser
        webbrowser.open(installer_url)
    
    input("Press Enter after installing Tesseract...")
    
    return check_tesseract_installed()

def install_tesseract_linux() -> bool:
    """Install Tesseract on Linux"""
    print_step(1, "Linux Tesseract Installation")
    
    try:
        print("Updating package list...")
        subprocess.run(['sudo', 'apt-get', 'update'], check=True)
        
        print("Installing Tesseract OCR...")
        result = subprocess.run(
            ['sudo', 'apt-get', 'install', '-y', 'tesseract-ocr', 'libtesseract-dev'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✓ Tesseract installed successfully")
            
            print_step(2, "Installing language packs")
            languages = ['eng', 'ind', 'spa', 'fra', 'deu', 'chi-sim']
            for lang in languages:
                try:
                    subprocess.run(['sudo', 'apt-get', 'install', '-y', f'tesseract-ocr-{lang}'], 
                                 capture_output=True)
                    print(f"  ✓ Language pack: {lang}")
                except:
                    print(f"  ✗ Language pack not available: {lang}")
            
            return True
        else:
            print(f"✗ Installation failed: {result.stderr}")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"✗ Package manager error: {e}")
        return False
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def install_tesseract_mac() -> bool:
    """Install Tesseract on macOS"""
    print_step(1, "macOS Tesseract Installation")
    
    try:
        print("Checking for Homebrew...")
        brew_check = subprocess.run(['brew', '--version'], capture_output=True, text=True)
        
        if brew_check.returncode != 0:
            print("Homebrew not found. Installing Homebrew...")
            install_brew = subprocess.run(
                '/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"',
                shell=True,
                capture_output=True,
                text=True
            )
            
            if install_brew.returncode != 0:
                print("✗ Homebrew installation failed")
                return False
        
        print("Installing Tesseract via Homebrew...")
        result = subprocess.run(
            ['brew', 'install', 'tesseract'],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✓ Tesseract installed successfully")
            
            print_step(2, "Installing language packs")
            try:
                subprocess.run(['brew', 'install', 'tesseract-lang'], capture_output=True)
                print("✓ Language packs installed")
            except:
                print("Note: Language packs may need manual installation")
            
            return True
        else:
            print(f"✗ Installation failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def install_python_packages() -> bool:
    """Install required Python packages"""
    print_step(3, "Installing Python packages")
    
    packages = ['pytesseract', 'Pillow']
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                         check=True, capture_output=True)
            print(f"✓ {package} installed")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e.stderr.decode() if e.stderr else str(e)}")
            return False
    
    return True

def update_requirements() -> bool:
    """Update requirements.txt with OCR dependencies"""
    print_step(4, "Updating requirements.txt")
    
    requirements_path = Path("requirements.txt")
    
    if not requirements_path.exists():
        print("✗ requirements.txt not found")
        return False
    
    try:
        with open(requirements_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'pytesseract' not in content.lower():
            with open(requirements_path, 'a', encoding='utf-8') as f:
                f.write('\n# OCR dependencies\npytesseract>=0.3.10\nPillow>=10.0.0\n')
            print("✓ requirements.txt updated with OCR dependencies")
        else:
            print("✓ OCR dependencies already in requirements.txt")
        
        return True
        
    except Exception as e:
        print(f"✗ Error updating requirements.txt: {e}")
        return False

def verify_installation() -> bool:
    """Verify Tesseract installation"""
    print_step(5, "Verifying installation")
    
    checks_passed = 0
    total_checks = 3
    
    try:
        result = subprocess.run(['tesseract', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ Tesseract system installation: {result.stdout.split('\\n')[0]}")
            checks_passed += 1
        else:
            print("✗ Tesseract not found in PATH")
    except:
        print("✗ Tesseract not found in PATH")
    
    try:
        import pytesseract
        print(f"✓ pytesseract Python package: v{pytesseract.__version__}")
        checks_passed += 1
    except ImportError:
        print("✗ pytesseract Python package not found")
    
    try:
        from PIL import Image
        print(f"✓ Pillow Python package: v{Image.__version__}")
        checks_passed += 1
    except ImportError:
        print("✗ Pillow Python package not found")
    
    print(f"\nVerification: {checks_passed}/{total_checks} checks passed")
    return checks_passed >= 2

def main():
    """Main installation function"""
    print_header("TESSERACT OCR INSTALLATION FOR DOCUBOT")
    
    if check_tesseract_installed():
        print("Tesseract is already installed. Skipping system installation.")
    else:
        system = platform.system()
        
        if system == "Windows":
            success = install_tesseract_windows()
        elif system == "Linux":
            success = install_tesseract_linux()
        elif system == "Darwin":
            success = install_tesseract_mac()
        else:
            print(f"✗ Unsupported operating system: {system}")
            success = False
        
        if not success:
            print("\n✗ Tesseract system installation failed")
            print("Please install Tesseract manually and try again.")
            return 1
    
    if not install_python_packages():
        print("\n✗ Python package installation failed")
        return 1
    
    if not update_requirements():
        print("\nWarning: Could not update requirements.txt")
    
    if not verify_installation():
        print("\n✗ Installation verification failed")
        print("Some components may not be properly installed.")
        return 1
    
    print_header("INSTALLATION COMPLETE")
    print("\nTesseract OCR has been successfully installed!")
    print("\nNext steps:")
    print("1. Restart your Python environment")
    print("2. Test OCR with: python -c "import pytesseract; print(pytesseract.get_tesseract_version())"")
    print("3. Add more language packs if needed")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
