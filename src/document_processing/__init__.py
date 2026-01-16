# docubot/src/document_processing/extractors/__init__.py
"""
Extractors package - Document extractor factory
"""

import importlib.util
from pathlib import Path
from typing import Dict, Type, Optional
import logging
from .base_extractor import BaseExtractor

logger = logging.getLogger(__name__)

# Global registry
_EXTRACTORS: Dict[str, Type[BaseExtractor]] = {}
_EXTENSION_MAP: Dict[str, Type[BaseExtractor]] = {}


def register_extractor(extractor_class: Type[BaseExtractor], extensions: list = None):
    """Register an extractor class."""
    extractor_instance = extractor_class()
    
    # Use instance's extensions if not provided
    if extensions is None:
        extensions = getattr(extractor_instance, 'supported_extensions', [])
    
    # Register extractor
    _EXTRACTORS[extractor_instance.name] = extractor_class
    
    # Map extensions
    for ext in extensions:
        _EXTENSION_MAP[ext.lower()] = extractor_class
    
    logger.info(f"Registered extractor: {extractor_instance.name} for {extensions}")


def get_extractor(file_extension: str) -> Optional[BaseExtractor]:
    """Get extractor instance for file extension."""
    ext = file_extension.lower()
    if ext in _EXTENSION_MAP:
        extractor_class = _EXTENSION_MAP[ext]
        return extractor_class()
    return None


def get_supported_extensions() -> list:
    """Get all supported file extensions."""
    return list(_EXTENSION_MAP.keys())


# Auto-discover and register extractors in this directory
def _discover_extractors():
    """Auto-discover extractor modules."""
    current_dir = Path(__file__).parent
    
    for py_file in current_dir.glob("*.py"):
        if py_file.name.startswith("_") or py_file.name == "base_extractor.py":
            continue
        
        module_name = py_file.stem
        try:
            spec = importlib.util.spec_from_file_location(module_name, py_file)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Look for extractor classes
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        issubclass(attr, BaseExtractor) and 
                        attr != BaseExtractor):
                        
                        # Check if it has a register function
                        if hasattr(module, 'register_extractor'):
                            getattr(module, 'register_extractor')(attr)
                        else:
                            # Auto-register
                            register_extractor(attr)
                            
        except Exception as e:
            logger.warning(f"Failed to load extractor module {module_name}: {e}")


# Initialize on import
try:
    _discover_extractors()
    logger.info(f"Loaded {len(_EXTRACTORS)} extractor(s)")
    logger.info(f"Supported extensions: {get_supported_extensions()}")
except Exception as e:
    logger.error(f"Failed to initialize extractors: {e}")


# Export
__all__ = [
    'BaseExtractor',
    'register_extractor',
    'get_extractor',
    'get_supported_extensions',
    '_EXTRACTORS',
    '_EXTENSION_MAP'
]