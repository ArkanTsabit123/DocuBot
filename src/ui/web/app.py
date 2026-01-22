# src/ui/web/app.py

"""
DocuBot Web Application - Main Entry Point
Compatibility file for backward compatibility
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the FastAPI app
try:
    from .fastapi_app import app
    __all__ = ['app']
except ImportError:
    app = None

def main():
    """Main entry point for backward compatibility"""
    print("DocuBot Web Application")
    print("=" * 50)
    print("\nPlease use one of the following:")
    print("\n1. FastAPI Backend (API):")
    print("   uvicorn src.ui.web.fastapi_app:app --host 0.0.0.0 --port 8000 --reload")
    print("\n2. Streamlit Frontend (Web UI):")
    print("   streamlit run src.ui.web.streamlit_app.py --server.port 8501")
    print("\n3. System Check:")
    print("   python runserver.py --mode check")
    print("\n4. Start all services:")
    print("   python runserver.py --mode start --service all")
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()