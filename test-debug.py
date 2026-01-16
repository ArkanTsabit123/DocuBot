import traceback
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    # Coba import document_processing module
    import src.document_processing.processor
    print("✓ src.document_processing.processor berhasil diimport")
    
    # Coba import extractors
    import src.document_processing.extractors.pdf_extractor
    print("✓ pdf_extractor berhasil diimport")
    
    # Coba import lainnya
    import src.document_processing.chunking
    print("✓ chunking berhasil diimport")
    
except ImportError as e:
    print(f"✗ ImportError: {e}")
    print(traceback.format_exc())
except Exception as e:
    print(f"✗ Error lain: {e}")
    print(traceback.format_exc())