"""
DocuBot Application Constants
"""

import os
from pathlib import Path

APP_NAME = "DocuBot"
APP_VERSION = "1.0.0"
APP_AUTHOR = "Your Name"
APP_LICENSE = "MIT"
APP_DESCRIPTION = "Local AI Personal Knowledge Assistant"

DOCUMENT_EXTENSIONS = {
    '.pdf': 'PDF Document',
    '.docx': 'Word Document',
    '.txt': 'Text File',
    '.epub': 'EPUB Ebook',
    '.md': 'Markdown',
    '.html': 'HTML Document',
    '.htm': 'HTML Document',
    '.csv': 'CSV Spreadsheet',
    '.jpg': 'JPEG Image',
    '.jpeg': 'JPEG Image',
    '.png': 'PNG Image',
    '.gif': 'GIF Image',
    '.bmp': 'Bitmap Image',
    '.tiff': 'TIFF Image',
}

LLM_MODELS = {
    'llama2:7b': {
        'name': 'Llama 2 7B',
        'size_gb': 3.8,
        'context_window': 4096,
        'description': 'Good balance of performance and accuracy'
    },
    'mistral:7b': {
        'name': 'Mistral 7B',
        'size_gb': 4.1,
        'context_window': 8192,
        'description': 'Fast with good reasoning'
    },
    'neural-chat:7b': {
        'name': 'Neural Chat 7B',
        'size_gb': 4.3,
        'context_window': 4096,
        'description': 'Optimized for conversation'
    }
}

EMBEDDING_MODELS = {
    'all-MiniLM-L6-v2': {
        'name': 'all-MiniLM-L6-v2',
        'dimensions': 384,
        'size_mb': 90,
        'description': 'Fast and efficient'
    },
    'all-mpnet-base-v2': {
        'name': 'all-mpnet-base-v2',
        'dimensions': 768,
        'size_mb': 420,
        'description': 'High accuracy'
    }
}

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50
MAX_FILE_SIZE_MB = 100
MAX_CHUNKS_PER_DOCUMENT = 1000

DATABASE_NAME = "docubot.db"
VECTOR_DB_NAME = "chroma"
CACHE_DB_NAME = "cache.db"

THEMES = ['dark', 'light', 'system']
LANGUAGES = ['en', 'id']
DEFAULT_THEME = 'dark'
DEFAULT_LANGUAGE = 'en'


def get_data_dir() -> Path:
    home = Path.home()
    
    if os.name == 'nt':
        base_dir = home / "AppData" / "Local" / "DocuBot"
    elif os.name == 'posix':
        if sys.platform == 'darwin':
            base_dir = home / "Library" / "Application Support" / "DocuBot"
        else:
            base_dir = home / ".local" / "share" / "docubot"
    else:
        base_dir = home / ".docubot"
    
    return base_dir


DATA_DIR = get_data_dir()
MODELS_DIR = DATA_DIR / "models"
DOCUMENTS_DIR = DATA_DIR / "documents"
DATABASE_DIR = DATA_DIR / "database"
LOGS_DIR = DATA_DIR / "logs"
EXPORTS_DIR = DATA_DIR / "exports"

for directory in [DATA_DIR, MODELS_DIR, DOCUMENTS_DIR, DATABASE_DIR, LOGS_DIR, EXPORTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

DEFAULT_CONFIG = {
    'app': {
        'name': APP_NAME,
        'version': APP_VERSION,
        'author': APP_AUTHOR,
    },
    'document_processing': {
        'chunk_size': DEFAULT_CHUNK_SIZE,
        'chunk_overlap': DEFAULT_CHUNK_OVERLAP,
        'max_file_size_mb': MAX_FILE_SIZE_MB,
    },
    'ai': {
        'llm': {
            'model': 'llama2:7b',
            'temperature': 0.1,
            'max_tokens': 1024,
        },
        'embeddings': {
            'model': 'all-MiniLM-L6-v2',
            'dimensions': 384,
        },
        'rag': {
            'top_k': 5,
            'similarity_threshold': 0.7,
        }
    },
    'ui': {
        'theme': DEFAULT_THEME,
        'language': DEFAULT_LANGUAGE,
        'font_size': 12,
    }
}
