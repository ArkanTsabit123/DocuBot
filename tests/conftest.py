# docubase/tests/conftest.py

"""
Pytest configuration and fixtures
"""

import pytest
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))


@pytest.fixture
def test_data_dir():
    return Path(__file__).parent / "test_data"


@pytest.fixture
def sample_text():
    return "This is a test document. It contains multiple sentences. " * 10


@pytest.fixture
def sample_config():
    return {
        "chunk_size": 500,
        "chunk_overlap": 50,
        "llm_model": "llama2:7b",
        "embedding_model": "all-MiniLM-L6-v2"
    }
