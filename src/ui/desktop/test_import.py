import sys
import os

print("Current directory:", os.getcwd())
print("Python path (sys.path):")
for i, path in enumerate(sys.path):
    print(f"  {i}: {path}")

# Coba import dengan berbagai cara
print("\nTrying imports...")

try:
    from src.core.config import AppConfig
    print("✓ from src.core.config import AppConfig SUCCESS")
except ImportError as e:
    print(f"✗ from src.core.config import AppConfig FAILED: {e}")

try:
    from core.config import AppConfig
    print("✓ from core.config import AppConfig SUCCESS")
except ImportError as e:
    print(f"✗ from core.config import AppConfig FAILED: {e}")

# Coba dengan path manipulation
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
print(f"\nProject root calculated: {project_root}")

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print("Added project root to sys.path")

try:
    from src.core.config import AppConfig
    print("✓ After path fix: from src.core.config import AppConfig SUCCESS")
except ImportError as e:
    print(f"✗ After path fix: from src.core.config import AppConfig FAILED: {e}")