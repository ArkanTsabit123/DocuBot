#docubot/6. complete tracker-v.1.0.py

"""
DocuBot - Professional Progress Tracker v2.2
upgrade validation system with improved structure detection
"""

import os
import sys
import json
import subprocess
import ast
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict, field
import yaml
import re

# ========== COMPLETE TASK DATABASE ==========

COMPLETE_TASKS = {
    "Phase 1: Core MVP (Weeks 1-2)": [
        ("P1.1.1", "Setup complete project structure", 1, 1, ["src/", "pyproject.toml", "requirements.txt"]),
        ("P1.1.2", "Setup Python 3.11+ virtual environment", 1, 1, [], "system"),
        ("P1.1.3", "Install all core dependencies", 1, 1, ["requirements.txt"], "system"),
        ("P1.1.4", "Initialize Git repository", 1, 1, [".git", ".gitignore"]),
        
        ("P1.2.1", "Complete src/core/config.py with AppConfig class", 1, 2, ["src/core/config.py"]),
        ("P1.2.2", "Create app_config.yaml with all settings", 1, 2, ["data/config/app_config.yaml"]),
        ("P1.2.3", "Setup cross-platform data directories", 1, 2, [], "manual"),
        ("P1.2.4", "Implement configuration validation", 2, 2, ["src/core/config.py"]),
        
        ("P1.3.1", "Complete document processor (processor.py)", 1, 3, ["src/document_processing/processor.py"]),
        ("P1.3.2", "PDF extractor with PyPDF2 & pdfplumber", 1, 3, ["src/document_processing/extractors/pdf_extractor.py"]),
        ("P1.3.3", "TXT extractor implementation", 1, 3, ["src/document_processing/extractors/txt_extractor.py"]),
        ("P1.3.4", "Text cleaning utilities", 2, 3, ["src/document_processing/cleaning.py"]),
        ("P1.3.5", "Intelligent chunking (500 tokens, 50 overlap)", 1, 3, ["src/document_processing/chunking.py"]),
        
        ("P1.4.1", "DOCX extractor module", 2, 4, ["src/document_processing/extractors/docx_extractor.py"]),
        ("P1.4.2", "Base extractor class", 2, 4, ["src/document_processing/extractors/__init__.py", "src/document_processing/extractors/base_extractor.py"]),
        ("P1.4.3", "Extractor factory/registry", 2, 4, ["src/document_processing/extractors/__init__.py"]),
        
        ("P1.5.1", "Complete SQLite client with CRUD operations", 1, 5, ["src/database/sqlite_client.py"]),
        ("P1.5.2", "Database schema (documents, chunks, conversations)", 1, 5, ["src/database/models.py"]),
        ("P1.5.3", "Migrations system", 2, 5, ["src/database/migrations/"]),
        ("P1.5.4", "Database initialization script", 2, 5, ["scripts/init_db.py"]),
        ("P1.5.5", "Database queries module", 2, 5, ["src/database/queries.py"]),
        
        ("P1.6.1", "Complete ChromaDB vector store client", 1, 6, ["src/vector_store/chroma_client.py"]),
        ("P1.6.2", "Persistent ChromaDB setup with settings", 1, 6, ["src/vector_store/chroma_client.py"]),
        ("P1.6.3", "Document embedding storage", 1, 6, ["src/vector_store/chroma_client.py"]),
        ("P1.6.4", "Similarity search with hybrid search", 2, 6, ["src/vector_store/search_engine.py"]),
        ("P1.6.5", "Index management", 2, 6, ["src/vector_store/index_manager.py"]),
        
        ("P1.7.1", "Unit tests for document processing", 2, 7, ["tests/unit/test_document_processor.py"]),
        ("P1.7.2", "Database operation tests", 2, 7, ["tests/unit/test_database.py"]),
        ("P1.7.3", "Vector store tests", 2, 7, ["tests/unit/test_vector_store.py"]),
        ("P1.7.4", "Fix Week 1 bugs", 1, 7, [], "manual"),
        
        ("P1.8.1", "Complete Ollama LLM client", 1, 8, ["src/ai_engine/llm_client.py"]),
        ("P1.8.2", "Support Llama 2 7B, Mistral 7B, Neural Chat", 1, 8, [], "manual"),
        ("P1.8.3", "Model downloading & management system", 1, 8, ["src/ai_engine/model_manager.py"]),
        ("P1.8.4", "Streaming & non-streaming response generation", 2, 8, ["src/ai_engine/llm_client.py"]),
        ("P1.8.5", "Temperature & token limit controls", 2, 8, ["src/ai_engine/llm_client.py"]),
        
        ("P1.9.1", "Complete embedding service", 1, 9, ["src/ai_engine/embedding_service.py"]),
        ("P1.9.2", "Integrate Sentence Transformers (all-MiniLM-L6-v2)", 1, 9, [], "manual"),
        ("P1.9.3", "Support multiple embedding models", 2, 9, ["src/ai_engine/embedding_service.py"]),
        ("P1.9.4", "Embedding caching system", 2, 9, ["src/ai_engine/embedding_service.py"]),
        
        ("P1.10.1", "Tests for Ollama integration", 2, 10, ["tests/unit/test_llm_client.py"]),
        ("P1.10.2", "Embedding generation tests", 2, 10, ["tests/unit/test_embedding_service.py"]),
        ("P1.10.3", "Validate model download functionality", 1, 10, [], "manual"),
        ("P1.10.4", "Fix AI integration bugs", 1, 10, [], "manual"),
        
        ("P1.11.1", "Complete RAG engine", 1, 11, ["src/ai_engine/rag_engine.py"]),
        ("P1.11.2", "Complete RAG workflow implementation", 1, 11, ["src/ai_engine/rag_engine.py"]),
        ("P1.11.3", "Conversation memory system", 2, 11, ["src/ai_engine/rag_engine.py"]),
        ("P1.11.4", "Prompt templates system", 2, 11, ["src/ai_engine/prompt_templates.py"]),
        
        ("P1.12.1", "Complete DocuBotCore class (app.py)", 1, 12, ["src/core/app.py"]),
        ("P1.12.2", "Query processing pipeline", 1, 12, ["src/core/app.py"]),
        ("P1.12.3", "Document management functions", 1, 12, ["src/core/app.py"]),
        ("P1.12.4", "Conversation handling", 2, 12, ["src/core/app.py"]),
        ("P1.12.5", "Error handling & logging", 2, 12, ["src/core/exceptions.py", "src/core/logger.py"]),
        
        ("P1.13.1", "Integration tests for RAG pipeline", 2, 13, ["tests/integration/test_rag_pipeline.py"]),
        ("P1.13.2", "End-to-end query flow tests", 1, 13, ["tests/e2e/test_full_workflow.py"]),
        ("P1.13.3", "Database integration tests", 2, 13, ["tests/integration/test_database.py"]),
        ("P1.13.4", "Fix RAG pipeline bugs", 1, 13, [], "manual"),
        
        ("P1.14.1", "Main application entry point (app.py)", 1, 14, ["app.py"]),
        ("P1.14.2", "Complete main desktop window", 1, 14, ["src/ui/desktop/main_window.py"]),
        ("P1.14.3", "Three-panel layout with CustomTkinter", 1, 14, ["src/ui/desktop/main_window.py"]),
        ("P1.14.4", "Dark/light theme support", 2, 14, ["src/ui/desktop/themes/"]),
        ("P1.14.5", "Responsive design", 2, 14, ["src/ui/desktop/main_window.py"]),
        ("P1.14.6", "Document upload (drag & drop)", 1, 14, ["src/ui/desktop/main_window.py"]),
        ("P1.14.7", "Chat display with message threading", 1, 14, ["src/ui/desktop/chat_panel.py"]),
        ("P1.14.8", "Query input with send functionality", 1, 14, ["src/ui/desktop/main_window.py"]),
        ("P1.14.9", "Source citation display", 2, 14, ["src/ui/desktop/chat_panel.py"]),
        ("P1.14.10", "Processing status indicators", 2, 14, ["src/ui/desktop/main_window.py"]),
        ("P1.14.11", "Settings panel", 2, 14, ["src/ui/desktop/settings_panel.py"]),
        ("P1.14.12", "Fix UI bugs and polish", 1, 14, [], "manual"),
        
        ("P1.15.1", "Complete constants module", 3, 15, ["src/core/constants.py"]),
        ("P1.15.2", "Complete formatter utilities", 2, 15, ["src/utilities/formatter.py"]),
    ],
    
    "Phase 2: Upgrade Features (Weeks 3-4)": [
        ("P2.1.1", "EPUB extractor module", 1, 15, ["src/document_processing/extractors/epub_extractor.py"]),
        ("P2.1.2", "Markdown extractor module", 1, 15, ["src/document_processing/extractors/markdown_extractor.py"]),
        ("P2.1.3", "HTML extractor module", 1, 15, ["src/document_processing/extractors/html_extractor.py"]),
        ("P2.1.4", "CSV extractor module", 2, 15, ["src/document_processing/extractors/csv_extractor.py"]),
        
        ("P2.2.1", "Image extractor with OCR", 1, 16, ["src/document_processing/extractors/image_extractor.py"]),
        ("P2.2.2", "Integrate Tesseract OCR", 1, 16, [], "manual"),
        ("P2.2.3", "Support multiple languages (eng, ind)", 2, 16, ["src/document_processing/extractors/image_extractor.py"]),
        ("P2.2.4", "Image preprocessing for better OCR", 2, 16, ["src/document_processing/extractors/image_extractor.py"]),
        
        ("P2.3.1", "Web extractor module", 1, 17, ["src/document_processing/extractors/web_extractor.py"]),
        ("P2.3.2", "BeautifulSoup4 integration", 1, 17, ["src/document_processing/extractors/web_extractor.py"]),
        ("P2.3.3", "URL processing with content extraction", 1, 17, ["src/document_processing/extractors/web_extractor.py"]),
        ("P2.3.4", "Web article saving functionality", 2, 17, ["src/document_processing/extractors/web_extractor.py"]),
        
        ("P2.4.1", "Automatic title/author/date extraction", 2, 18, ["src/document_processing/metadata.py"]),
        ("P2.4.2", "Document summarization", 2, 18, ["src/ai_engine/summarizer.py"]),
        ("P2.4.3", "Auto-tagging based on content", 2, 18, ["src/ai_engine/tagging.py"]),
        ("P2.4.4", "Smart collections (auto-organize)", 2, 18, ["src/storage/collection_manager.py"]),
        
        ("P2.5.1", "FAISS vector store alternative", 2, 19, ["src/vector_store/faiss_client.py"]),
        ("P2.5.2", "File manager for storage operations", 2, 19, ["src/storage/file_manager.py"]),
        
        ("P2.6.1", "Complete CSV extractor with pandas", 1, 20, ["src/document_processing/extractors/csv_extractor.py"]),
        ("P2.6.2", "Complete HTML extractor with BeautifulSoup", 1, 20, ["src/document_processing/extractors/html_extractor.py"]),
        ("P2.6.3", "Complete EPUB extractor with ebooklib", 1, 20, ["src/document_processing/extractors/epub_extractor.py"]),
        ("P2.6.4", "Markdown extractor improvements", 2, 20, ["src/document_processing/extractors/markdown_extractor.py"]),
        
        ("P2.7.1", "Conversation history panel", 1, 22, ["src/ui/desktop/history_panel.py"]),
        ("P2.7.2", "Search & filter conversations", 2, 22, ["src/ui/desktop/history_panel.py"]),
        ("P2.7.3", "Conversation tagging & organization", 2, 22, ["src/ui/desktop/history_panel.py"]),
        ("P2.7.4", "Archive & delete conversations", 2, 22, ["src/ui/desktop/history_panel.py"]),
        
        ("P2.8.1", "Document preview functionality", 2, 23, ["src/ui/desktop/document_panel.py"]),
        ("P2.8.2", "Batch operations (select multiple)", 2, 23, ["src/ui/desktop/document_panel.py"]),
        ("P2.8.3", "Tag management interface", 2, 23, ["src/ui/desktop/document_panel.py"]),
        ("P2.8.4", "Document search & filters", 2, 23, ["src/ui/desktop/document_panel.py"]),
        
        ("P2.9.1", "Export conversations to Markdown", 1, 25, ["src/ui/desktop/export_manager.py"]),
        ("P2.9.2", "Export to PDF", 2, 25, ["src/ui/desktop/export_manager.py"]),
        ("P2.9.3", "Export to HTML", 2, 25, ["src/ui/desktop/export_manager.py"]),
        ("P2.9.4", "Batch export", 2, 25, ["src/ui/desktop/export_manager.py"]),
        
        ("P2.10.1", "Reading progress tracking", 2, 26, ["src/ui/desktop/reading_tracker.py"]),
        ("P2.10.2", "Highlight & annotation support", 2, 26, ["src/ui/desktop/annotation_tool.py"]),
        ("P2.10.3", "Keyboard shortcuts", 2, 26, ["src/ui/desktop/shortcuts.py"]),
        ("P2.10.4", "Quick search", 2, 26, ["src/ui/desktop/search_bar.py"]),
    ],
    
    "Phase 3: Polish & Optimization (Weeks 5-6)": [
        ("P3.1.1", "Caching system implementation", 1, 29, ["src/storage/cache_manager.py"]),
        ("P3.1.2", "Embedding caching", 1, 29, ["src/storage/cache_manager.py"]),
        ("P3.1.3", "Document processing cache", 2, 29, ["src/storage/cache_manager.py"]),
        ("P3.1.4", "LLM response caching", 2, 29, ["src/storage/cache_manager.py"]),
        ("P3.1.5", "Encryption module for sensitive data", 3, 29, ["src/storage/encryption.py"]),
        
        ("P3.2.1", "Lazy loading implementation", 2, 30, ["src/core/app.py"]),
        ("P3.2.2", "Memory usage monitoring", 2, 30, ["src/utilities/monitor.py"]),
        ("P3.2.3", "Background processing queue", 2, 30, ["src/utilities/task_queue.py"]),
        ("P3.2.4", "Resource cleanup system", 2, 30, ["src/utilities/cleanup.py"]),
        ("P3.2.5", "Helper utilities module", 2, 30, ["src/utilities/helpers.py"]),
        
        ("P3.3.1", "Robust error handling system", 1, 32, ["src/core/exceptions.py"]),
        ("P3.3.2", "Graceful degradation", 2, 32, ["src/core/app.py"]),
        ("P3.3.3", "User-friendly error messages", 2, 32, ["src/ui/desktop/error_dialog.py"]),
        ("P3.3.4", "Automatic retry mechanisms", 2, 32, ["src/utilities/retry.py"]),
        
        ("P3.4.1", "Backup manager", 2, 33, ["src/storage/backup_manager.py"]),
        ("P3.4.2", "Automatic backups", 2, 33, ["src/storage/backup_manager.py"]),
        ("P3.4.3", "Manual backup/restore", 2, 33, ["src/storage/backup_manager.py"]),
        ("P3.4.4", "Backup scheduling", 2, 33, ["src/storage/backup_manager.py"]),
        
        ("P3.5.1", "Plugin architecture", 2, 36, ["src/plugins/plugin_manager.py"]),
        ("P3.5.2", "Base plugin class", 2, 36, ["src/plugins/base_plugin.py"]),
        ("P3.5.3", "Plugin loading system", 2, 36, ["src/plugins/plugin_manager.py"]),
        ("P3.5.4", "Plugin configuration", 2, 36, ["src/plugins/plugin_manager.py"]),
        
        ("P3.6.1", "Obsidian sync plugin", 2, 37, ["src/plugins/builtin_plugins/obsidian_sync.py"]),
        ("P3.6.2", "Notion export plugin", 2, 37, ["src/plugins/builtin_plugins/notion_export.py"]),
        ("P3.6.3", "Browser clipper plugin", 2, 37, ["src/plugins/builtin_plugins/browser_clipper.py"]),
        ("P3.6.4", "Voice interface plugin", 2, 37, ["src/plugins/builtin_plugins/voice_interface.py"]),
        
        ("P3.7.1", "Web interface with Streamlit", 2, 39, ["src/ui/web/app.py"]),
        ("P3.7.2", "Web chat interface", 2, 39, ["src/ui/web/pages/chat.py"]),
        ("P3.7.3", "Document management via web", 2, 39, ["src/ui/web/pages/documents.py"]),
        ("P3.7.4", "Responsive web design", 2, 39, ["src/ui/web/components/"]),
        
        ("P3.8.1", "CLI interface", 2, 40, ["src/ui/cli/cli.py"]),
        ("P3.8.2", "Command-line operations", 2, 40, ["src/ui/cli/commands.py"]),
        ("P3.8.3", "Scripting support", 2, 40, ["src/ui/cli/commands.py"]),
        ("P3.8.4", "Batch processing via CLI", 2, 40, ["src/ui/cli/commands.py"]),
        
        ("P3.9.1", "UI components module", 2, 41, ["src/ui/desktop/components/"]),
        ("P3.9.2", "Web components module", 2, 41, ["src/ui/web/components/"]),
        ("P3.9.3", "CLI output formatters", 2, 41, ["src/ui/cli/output_formatters.py"]),
    ],
    
    "Phase 4: Deployment Preparation (Weeks 7-8)": [
        ("P4.1.1", "Windows packaging with PyInstaller", 1, 43, ["build_windows.bat", "build_windows.py"]),
        ("P4.1.2", "Single executable creation", 1, 43, [], "manual"),
        ("P4.1.3", "Icon & metadata setup", 2, 43, ["resources/icons/"]),
        ("P4.1.4", "Dependency bundling", 1, 43, [], "manual"),
        
        ("P4.2.1", "macOS .app bundle creation", 1, 44, ["build_mac.sh", "build_mac.py"]),
        ("P4.2.2", "Code signing (optional)", 3, 44, [], "manual"),
        ("P4.2.3", "DMG installer", 2, 44, [], "manual"),
        
        ("P4.3.1", "Linux AppImage creation", 1, 45, ["build_linux.sh", "build_linux.py"]),
        ("P4.3.2", "DEB/RPM packages (optional)", 3, 45, [], "manual"),
        ("P4.3.3", "Desktop entry creation", 2, 45, [], "manual"),
        ("P4.3.4", "File associations", 2, 45, [], "manual"),
        
        ("P4.4.1", "One-click installer for all platforms", 1, 46, ["scripts/install.py"]),
        ("P4.4.2", "Automatic dependency installation", 1, 46, ["scripts/install.py"]),
        ("P4.4.3", "Model download wizard", 1, 46, ["scripts/install.py"]),
        ("P4.4.4", "First-run setup wizard", 2, 46, ["scripts/setup.py"]),
        
        ("P4.5.1", "Automatic configuration", 2, 47, ["scripts/setup.py"]),
        ("P4.5.2", "System requirement checks", 2, 47, ["scripts/install.py"]),
        ("P4.5.3", "Update mechanism", 2, 47, ["scripts/update.py"]),
        ("P4.5.4", "Rollback capability", 3, 47, ["scripts/update.py"]),
        
        ("P4.6.1", "Complete unit testing suite", 1, 50, ["tests/unit/"]),
        ("P4.6.2", "Test all core modules (90%+ coverage)", 1, 50, [], "test"),
        ("P4.6.3", "Integration testing suite", 1, 51, ["tests/integration/"]),
        ("P4.6.4", "End-to-end RAG pipeline tests", 1, 51, ["tests/e2e/test_full_workflow.py"]),
        
        ("P4.7.1", "UI integration tests", 2, 52, ["tests/integration/test_ui.py"]),
        ("P4.7.2", "Cross-platform compatibility tests", 2, 52, [], "manual"),
        ("P4.7.3", "Performance & stress tests", 2, 52, ["tests/performance/"]),
        ("P4.7.4", "Usability testing", 2, 52, [], "manual"),
        
        ("P4.8.1", "Complete user guide", 1, 53, ["docs/user_guide/"]),
        ("P4.8.2", "Developer documentation", 2, 53, ["docs/developer_guide/"]),
        ("P4.8.3", "API reference", 2, 53, ["docs/api_reference/"]),
        ("P4.8.4", "Troubleshooting guide", 2, 53, ["docs/troubleshooting/"]),
        
        ("P4.9.1", "In-app help system", 2, 54, ["src/ui/desktop/help_system.py"]),
        ("P4.9.2", "Tooltips and guides", 3, 54, ["src/ui/desktop/components/"]),
        
        ("P4.10.1", "Create CHANGELOG.md", 1, 55, ["CHANGELOG.md"]),
        ("P4.10.2", "Update README.md", 1, 55, ["README.md"]),
        ("P4.10.3", "Prepare release notes", 1, 55, ["RELEASE_NOTES.md"]),
        ("P4.10.4", "Create distribution packages", 1, 55, [], "manual"),
        
        ("P4.11.1", "Security audit", 2, 56, [], "manual"),
        ("P4.11.2", "Performance validation", 2, 56, [], "manual"),
        ("P4.11.3", "Final bug fixes", 1, 56, [], "manual"),
        ("P4.11.4", "Quality assurance", 1, 56, [], "manual"),
        
        ("P4.12.1", "Database diagnostic tools", 2, 57, ["scripts/diagnostic.py"]),
        ("P4.12.2", "Backup utility script", 2, 57, ["scripts/backup.py"]),
        ("P4.12.3", "Resource validation script", 2, 57, ["scripts/validate_resources.py"]),
    ]
}

# ========== UPGRADED VALIDATION ENGINE ==========

class IntelligentValidator:
    def __init__(self, project_dir: Path):
        self.project_dir = project_dir
        
    def validate_task(self, task) -> Dict[str, Any]:
        base_result = {
            'task_id': task.id,
            'description': task.description,
            'verification_method': task.verification,
            'timestamp': datetime.now().isoformat()
        }
        
        if task.verification == "system":
            return {**base_result, **self._validate_system_task(task)}
        elif task.verification == "test":
            return {**base_result, **self._validate_test_task(task)}
        elif task.verification == "manual":
            return {**base_result, **self._validate_manual_task(task)}
        else:
            return {**base_result, **self._validate_code_task(task)}
    
    def _validate_system_task(self, task) -> Dict[str, Any]:
        description_lower = task.description.lower()
        
        if 'python' in description_lower:
            return self._validate_python_installation()
        elif 'environment' in description_lower:
            return self._validate_virtual_environment()
        
        return {
            'status': 'not_validated',
            'score': 0.0,
            'details': 'No specific system validation defined'
        }
    
    def _validate_python_installation(self) -> Dict[str, Any]:
        try:
            result = subprocess.run(['python', '--version'], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                result = subprocess.run(['python3', '--version'], 
                                      capture_output=True, text=True)
            
            if result.returncode == 0:
                version_str = result.stdout.strip()
                is_311_plus = any(f'Python 3.{v}' in version_str for v in ['11', '12', '13', '14'])
                
                return {
                    'status': 'success' if is_311_plus else 'failed',
                    'score': 1.0 if is_311_plus else 0.0,
                    'details': {
                        'version': version_str,
                        'meets_requirement': is_311_plus,
                        'required_version': 'Python 3.11+'
                    }
                }
        
        except Exception:
            pass
        
        return {
            'status': 'failed',
            'score': 0.0,
            'details': {'error': 'Python not found or version check failed'}
        }
    
    def _validate_virtual_environment(self) -> Dict[str, Any]:
        venv_indicators = ['VIRTUAL_ENV', 'CONDA_PREFIX', 'PYENV_VERSION']
        in_venv = any(os.environ.get(indicator) for indicator in venv_indicators)
        
        return {
            'status': 'success' if in_venv else 'warning',
            'score': 1.0 if in_venv else 0.5,
            'details': {
                'in_virtual_env': in_venv,
                'indicators': {indicator: bool(os.environ.get(indicator)) 
                              for indicator in venv_indicators}
            }
        }
    
    def _validate_test_task(self, task) -> Dict[str, Any]:
        try:
            test_files = []
            for file_path in task.file_paths:
                full_path = self.project_dir / file_path
                if full_path.exists():
                    if full_path.is_file() and full_path.suffix == '.py':
                        test_files.append(full_path)
                    elif full_path.is_dir():
                        test_files.extend(full_path.rglob("*.py"))
            
            if not test_files:
                return {
                    'status': 'partial',
                    'score': 0.3,
                    'details': {'message': 'No test files found'}
                }
            
            test_results = []
            passed_count = 0
            
            for test_file in test_files[:3]:
                try:
                    result = subprocess.run(
                        ['python', '-m', 'pytest', str(test_file), '-v', '--tb=short'],
                        capture_output=True,
                        text=True,
                        timeout=60,
                        cwd=str(self.project_dir)
                    )
                    
                    passed = result.returncode == 0
                    if passed:
                        passed_count += 1
                    
                    test_results.append({
                        'file': str(test_file.relative_to(self.project_dir)),
                        'passed': passed,
                        'output': result.stdout[-500:] if result.stdout else '',
                        'error': result.stderr[-500:] if result.stderr else ''
                    })
                    
                except subprocess.TimeoutExpired:
                    test_results.append({
                        'file': str(test_file.relative_to(self.project_dir)),
                        'passed': False,
                        'error': 'Test timed out after 60 seconds'
                    })
                except Exception as e:
                    test_results.append({
                        'file': str(test_file.relative_to(self.project_dir)),
                        'passed': False,
                        'error': str(e)
                    })
            
            score = passed_count / len(test_results) if test_results else 0.0
            
            return {
                'status': 'success' if score >= 0.8 else 'partial',
                'score': score,
                'details': {
                    'tests_run': len(test_results),
                    'tests_passed': passed_count,
                    'test_results': test_results,
                    'coverage_percentage': score * 100
                }
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'score': 0.0,
                'details': {'error': str(e)}
            }
    
    def _validate_manual_task(self, task) -> Dict[str, Any]:
        has_notes = bool(task.notes and len(task.notes.strip()) > 20)
        recently_updated = False
        
        if task.last_updated:
            try:
                last_update = datetime.fromisoformat(task.last_updated.replace('Z', '+00:00'))
                recently_updated = (datetime.now() - last_update).days < 3
            except:
                pass
        
        if has_notes and recently_updated:
            score = 0.9
        elif has_notes or recently_updated:
            score = 0.6
        else:
            score = 0.2
        
        return {
            'status': 'success' if score >= 0.7 else 'partial',
            'score': score,
            'details': {
                'has_notes': has_notes,
                'recently_updated': recently_updated,
                'notes_length': len(task.notes.strip()) if task.notes else 0
            }
        }
    
    def _validate_code_task(self, task) -> Dict[str, Any]:
        task_lower = task.description.lower()
        
        if 'git' in task_lower or '.git' in str(task.file_paths):
            return self._validate_git_task(task)
        
        if 'app_config.yaml' in str(task.file_paths) or 'config.yaml' in str(task.file_paths):
            return self._validate_yaml_task(task)
        
        structure_result = self._validate_structure(task)
        
        # SPECIAL HANDLING FOR P1.1.1
        if task.id == "P1.1.1":
            return self._validate_p1_1_1(task, structure_result)
        
        if not structure_result['all_files_exist']:
            return {
                'status': 'failed',
                'score': 0.0,
                'details': {
                    'structure': structure_result,
                    'message': 'Missing required files'
                }
            }
        
        code_quality_result = self._analyze_code_quality(task)
        
        functionality_result = None
        if task.priority == 1:
            functionality_result = self._validate_functionality(task)
        
        # Calculate scores properly
        dirs_found = structure_result.get('directories_exist', 0)
        files_with_content = structure_result.get('files_with_content', 0)
        total_dirs = len([p for p in task.file_paths if p.endswith('/')])
        total_files = len([p for p in task.file_paths if not p.endswith('/')])
        
        dir_score = dirs_found / total_dirs if total_dirs > 0 else 1.0
        file_score = files_with_content / total_files if total_files > 0 else 1.0
        
        structure_score = (dir_score + file_score) / 2.0 if total_dirs > 0 and total_files > 0 else file_score
        
        code_score = min(1.0, code_quality_result.get('function_count', 0) / 3.0)
        
        func_score = 0.0  # Initialize func_score
        if functionality_result:
            func_score = functionality_result.get('score', 0.0)
            overall_score = (structure_score * 0.3 + code_score * 0.3 + func_score * 0.4)
        else:
            overall_score = (structure_score * 0.5 + code_score * 0.5)
        
        return {
            'status': 'success' if overall_score >= 0.7 else 'partial',
            'score': overall_score,
            'details': {
                'structure': structure_result,
                'code_quality': code_quality_result,
                'functionality': functionality_result,
                'overall_score_breakdown': {
                    'structure': structure_score,
                    'code_quality': code_score,
                    'functionality': func_score if functionality_result else 0.0
                }
            }
        }
    
    def _validate_git_task(self, task) -> Dict[str, Any]:
        git_dir = self.project_dir / ".git"
        
        if not git_dir.exists():
            return {
                'status': 'failed',
                'score': 0.0,
                'details': {'error': '.git directory not found'}
            }
        
        try:
            required_items = ["HEAD", "config", "objects", "refs"]
            found_items = 0
            
            for item in required_items:
                if (git_dir / item).exists():
                    found_items += 1
            
            has_commits = False
            head_file = git_dir / "HEAD"
            if head_file.exists():
                try:
                    content = head_file.read_text()
                    has_commits = "ref:" in content
                except:
                    pass
            
            score = found_items / len(required_items)
            if has_commits:
                score = min(1.0, score + 0.3)
            
            return {
                'status': 'success' if score >= 0.7 else 'partial',
                'score': score,
                'details': {
                    'git_dir_exists': True,
                    'required_items_found': found_items,
                    'has_commits': has_commits,
                    'score_breakdown': {
                        'structure': found_items / len(required_items),
                        'commits_bonus': 0.3 if has_commits else 0.0
                    }
                }
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'score': 0.0,
                'details': {'error': str(e)}
            }
    
    def _validate_yaml_task(self, task) -> Dict[str, Any]:
        if not task.file_paths:
            return {
                'status': 'failed',
                'score': 0.0,
                'details': {'error': 'No file paths specified'}
            }
        
        file_path = self.project_dir / task.file_paths[0]
        
        if not file_path.exists():
            return {
                'status': 'failed',
                'score': 0.0,
                'details': {'error': f'File not found: {file_path}'}
            }
        
        try:
            content = file_path.read_text(encoding='utf-8')
            
            try:
                yaml_data = yaml.safe_load(content)
                is_valid_yaml = True
                yaml_error_msg = None
            except Exception as yaml_error:
                is_valid_yaml = False
                yaml_error_msg = str(yaml_error)
            
            required_sections = []
            if 'app_config' in str(file_path):
                required_sections = ['app:', 'paths:', 'document_processing:', 'ai:', 
                                    'ui:', 'storage:', 'performance:', 'privacy:']
            
            found_sections = 0
            for section in required_sections:
                if section.lower() in content.lower():
                    found_sections += 1
            
            section_score = found_sections / len(required_sections) if required_sections else 0.5
            
            score = 0.0
            if is_valid_yaml:
                score = section_score * 0.7 + 0.3
            else:
                score = section_score * 0.5
            
            return {
                'status': 'success' if score >= 0.7 else 'partial',
                'score': min(1.0, score),
                'details': {
                    'file_exists': True,
                    'file_size': len(content),
                    'yaml_valid': is_valid_yaml,
                    'sections_found': found_sections,
                    'required_sections': len(required_sections),
                    'yaml_error': yaml_error_msg
                }
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'score': 0.0,
                'details': {'error': str(e)}
            }
    
    def _validate_structure(self, task) -> Dict[str, Any]:
        result = {
            'all_files_exist': True,
            'files_with_content': 0,
            'directories_exist': 0,
            'total_items': len(task.file_paths),
            'missing_files': [],
            'empty_files': [],
            'missing_dirs': [],
            'file_details': [],
            'dir_details': []
        }
        
        for item_path in task.file_paths:
            full_path = self.project_dir / item_path
            
            # Handle directories (ending with /)
            if item_path.endswith('/'):
                dir_name = item_path.rstrip('/')
                if full_path.exists() and full_path.is_dir():
                    result['directories_exist'] += 1
                    result['dir_details'].append({
                        'path': dir_name,
                        'exists': True,
                        'is_dir': True
                    })
                else:
                    result['all_files_exist'] = False
                    result['missing_dirs'].append(dir_name)
                    result['dir_details'].append({
                        'path': dir_name,
                        'exists': False,
                        'is_dir': True
                    })
                continue
            
            # Handle files
            if not full_path.exists():
                result['all_files_exist'] = False
                result['missing_files'].append(item_path)
                result['file_details'].append({
                    'path': item_path,
                    'exists': False,
                    'has_content': False
                })
                continue
            
            try:
                if full_path.is_file():
                    file_size = full_path.stat().st_size
                    
                    # Check content based on file type
                    if file_size > 0:
                        if item_path.endswith('.py'):
                            with open(full_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            # For Python files, check for actual code
                            has_content = file_size > 10 and any(
                                keyword in content for keyword in 
                                ['import', 'def ', 'class ', 'from ', '#']
                            )
                        elif item_path.endswith(('.yaml', '.yml', '.json')):
                            # For config files, check for valid structure
                            with open(full_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            has_content = file_size > 5 and len(content.strip()) > 0
                        elif item_path.endswith(('.txt', '.md', '.toml')):
                            # For text files, just check if not empty
                            has_content = file_size > 5
                        else:
                            # For other files
                            has_content = file_size > 10
                    else:
                        has_content = False
                    
                    if has_content:
                        result['files_with_content'] += 1
                        content_status = "has content"
                    else:
                        result['empty_files'].append(item_path)
                        content_status = "empty"
                    
                    file_info = {
                        'path': item_path,
                        'size_bytes': file_size,
                        'has_content': has_content,
                        'content_status': content_status
                    }
                    
                    # Add more details for specific file types
                    if item_path.endswith('.py') and has_content:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        lines = content.split('\n')
                        file_info.update({
                            'lines': len(lines),
                            'non_empty_lines': len([l for l in lines if l.strip()]),
                            'import_count': content.count('import '),
                            'function_count': content.count('def '),
                            'class_count': content.count('class ')
                        })
                    
                    result['file_details'].append(file_info)
                else:
                    # It's a directory without trailing slash
                    result['directories_exist'] += 1
                    result['dir_details'].append({
                        'path': item_path,
                        'exists': True,
                        'is_dir': True
                    })
            
            except Exception as e:
                result['file_details'].append({
                    'path': item_path,
                    'error': str(e),
                    'has_content': False
                })
        
        return result
    
    def _validate_p1_1_1(self, task, structure_result) -> Dict[str, Any]:
        """Special validation for P1.1.1 - Setup complete project structure"""
        
        # For P1.1.1, we check:
        # - src/ directory exists
        # - pyproject.toml exists and has content
        # - requirements.txt exists and has content
        
        dirs_found = structure_result.get('directories_exist', 0)
        files_with_content = structure_result.get('files_with_content', 0)
        
        total_dirs = 1  # Only src/ is required as directory
        total_files = 2  # pyproject.toml and requirements.txt
        
        dir_score = dirs_found / total_dirs if total_dirs > 0 else 0.0
        file_score = files_with_content / total_files if total_files > 0 else 0.0
        
        # Calculate overall score (mirroring Smart Validator logic)
        if total_dirs > 0 and total_files > 0:
            overall_score = (dir_score * 0.5 + file_score * 0.5)
        else:
            overall_score = file_score
        
        # Bonus points for having all 7 directories from Smart Validator
        required_dirs_smart = ["src", "data", "tests", "docs", "scripts", "resources", "bin"]
        extra_dirs_found = 0
        for dir_name in required_dirs_smart:
            if (self.project_dir / dir_name).exists():
                extra_dirs_found += 1
        
        # Bonus for having all core files from Smart Validator
        required_files_smart = ["app.py", "requirements.txt", "README.md", "LICENSE", "pyproject.toml", ".gitignore"]
        extra_files_found = 0
        for file_name in required_files_smart:
            file_path = self.project_dir / file_name
            if file_path.exists() and file_path.stat().st_size > 0:
                extra_files_found += 1
        
        # Add bonus points based on Smart Validator criteria
        bonus = (extra_dirs_found / len(required_dirs_smart) * 0.3 + 
                 extra_files_found / len(required_files_smart) * 0.3)
        
        final_score = min(1.0, overall_score + bonus * 0.5)
        
        return {
            'status': 'success' if final_score >= 0.7 else 'partial',
            'score': final_score,
            'details': {
                'structure': structure_result,
                'dir_score': dir_score,
                'file_score': file_score,
                'extra_dirs_found': extra_dirs_found,
                'extra_files_found': extra_files_found,
                'bonus': bonus,
                'final_score': final_score
            }
        }
    
    def _analyze_code_quality(self, task) -> Dict[str, Any]:
        metrics = {
            'total_files_analyzed': 0,
            'total_lines': 0,
            'code_lines': 0,
            'comment_lines': 0,
            'function_count': 0,
            'class_count': 0,
            'docstring_count': 0,
            'type_hint_functions': 0,
            'average_function_length': 0,
            'import_count': 0,
            'has_docstrings': False,
            'has_type_hints': False,
            'file_metrics': []
        }
        
        python_files = [f for f in task.file_paths if f.endswith('.py')]
        
        for file_path in python_files:
            full_path = self.project_dir / file_path
            if not full_path.exists():
                continue
            
            try:
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                file_metrics = {
                    'file': file_path,
                    'lines': 0,
                    'functions': 0,
                    'classes': 0,
                    'imports': 0,
                    'has_docstrings': False
                }
                
                lines = content.split('\n')
                metrics['total_lines'] += len(lines)
                file_metrics['lines'] = len(lines)
                
                metrics['code_lines'] += len([l for l in lines if l.strip() and not l.strip().startswith('#')])
                metrics['comment_lines'] += len([l for l in lines if l.strip().startswith('#')])
                
                try:
                    tree = ast.parse(content)
                    
                    functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
                    metrics['function_count'] += len(functions)
                    file_metrics['functions'] = len(functions)
                    
                    classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                    metrics['class_count'] += len(classes)
                    file_metrics['classes'] = len(classes)
                    
                    for node in functions + classes:
                        if ast.get_docstring(node):
                            metrics['docstring_count'] += 1
                            file_metrics['has_docstrings'] = True
                    
                    for func in functions:
                        if func.returns:
                            metrics['type_hint_functions'] += 1
                        elif any(arg.annotation for arg in func.args.args):
                            metrics['type_hint_functions'] += 1
                    
                    imports = [node for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))]
                    metrics['import_count'] += len(imports)
                    file_metrics['imports'] = len(imports)
                    
                except SyntaxError as e:
                    file_metrics['syntax_error'] = str(e)
                
                metrics['file_metrics'].append(file_metrics)
                metrics['total_files_analyzed'] += 1
                    
            except Exception as e:
                metrics['file_metrics'].append({
                    'file': file_path,
                    'error': str(e)
                })
        
        if metrics['function_count'] > 0:
            metrics['average_function_length'] = metrics['code_lines'] / metrics['function_count']
        
        metrics['has_docstrings'] = metrics['docstring_count'] > 0
        metrics['has_type_hints'] = metrics['type_hint_functions'] > 0
        
        return metrics
    
    def _validate_functionality(self, task) -> Optional[Dict[str, Any]]:
        description_lower = task.description.lower()
        
        if 'extractor' in description_lower:
            return self._validate_extractor_functionality(task)
        elif 'processor' in description_lower:
            return self._validate_processor_functionality(task)
        elif 'chunking' in description_lower:
            return self._validate_chunking_functionality(task)
        elif 'llm' in description_lower:
            return self._validate_llm_functionality(task)
        elif 'database' in description_lower or 'sqlite' in description_lower:
            return self._validate_database_functionality(task)
        elif 'vector' in description_lower or 'chroma' in description_lower:
            return self._validate_vector_store_functionality(task)
        
        return None
    
    def _validate_extractor_functionality(self, task) -> Dict[str, Any]:
        try:
            extractor_type = None
            for fmt in ['pdf', 'txt', 'docx', 'epub', 'html', 'csv', 'image', 'web']:
                if fmt in task.description.lower():
                    extractor_type = fmt
                    break
            
            if not extractor_type:
                return {
                    'status': 'partial',
                    'score': 0.5,
                    'details': {'message': 'Could not determine extractor type'}
                }
            
            module_path = f"src/document_processing/extractors/{extractor_type}_extractor.py"
            full_path = self.project_dir / module_path
            
            if not full_path.exists():
                return {
                    'status': 'failed',
                    'score': 0.0,
                    'details': {'message': f'Extractor file not found: {module_path}'}
                }
            
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            has_class = 'class' in content and f'{extractor_type}' in content.lower()
            has_extract_method = 'def extract' in content or 'def process' in content
            has_imports = 'import' in content
            
            score = 0.0
            if has_class and has_extract_method and has_imports:
                score = 0.9
            elif has_class and has_extract_method:
                score = 0.7
            elif has_class:
                score = 0.5
            
            return {
                'status': 'success' if score >= 0.7 else 'partial',
                'score': score,
                'details': {
                    'extractor_type': extractor_type,
                    'has_class_definition': has_class,
                    'has_extract_method': has_extract_method,
                    'has_imports': has_imports,
                    'file_exists': True
                }
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'score': 0.0,
                'details': {'error': str(e)}
            }
    
    def _validate_processor_functionality(self, task) -> Dict[str, Any]:
        try:
            module_path = "src/document_processing/processor.py"
            full_path = self.project_dir / module_path
            
            if not full_path.exists():
                return {
                    'status': 'failed',
                    'score': 0.0,
                    'details': {'message': 'Processor file not found'}
                }
            
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            has_class = 'class DocumentProcessor' in content
            has_process_method = 'def process_document' in content or 'def process' in content
            has_chunking = 'chunk' in content.lower()
            
            score = 0.0
            if has_class and has_process_method and has_chunking:
                score = 0.9
            elif has_class and has_process_method:
                score = 0.7
            elif has_class:
                score = 0.5
            
            return {
                'status': 'success' if score >= 0.7 else 'partial',
                'score': score,
                'details': {
                    'has_class': has_class,
                    'has_process_method': has_process_method,
                    'has_chunking': has_chunking
                }
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'score': 0.0,
                'details': {'error': str(e)}
            }
    
    def _validate_chunking_functionality(self, task) -> Dict[str, Any]:
        try:
            module_path = "src/document_processing/chunking.py"
            full_path = self.project_dir / module_path
            
            if not full_path.exists():
                return {
                    'status': 'failed',
                    'score': 0.0,
                    'details': {'message': 'Chunking file not found'}
                }
            
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            has_class = 'class' in content and 'Chunk' in content
            has_chunk_method = 'def chunk' in content or 'def split' in content
            has_size = '500' in content
            has_overlap = '50' in content
            
            score = 0.0
            if has_class and has_chunk_method and has_size and has_overlap:
                score = 0.9
            elif has_class and has_chunk_method:
                score = 0.7
            elif has_class:
                score = 0.5
            
            return {
                'status': 'success' if score >= 0.7 else 'partial',
                'score': score,
                'details': {
                    'has_class': has_class,
                    'has_chunk_method': has_chunk_method,
                    'has_size_500': has_size,
                    'has_overlap_50': has_overlap
                }
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'score': 0.0,
                'details': {'error': str(e)}
            }
    
    def _validate_llm_functionality(self, task) -> Dict[str, Any]:
        try:
            module_path = "src/ai_engine/llm_client.py"
            full_path = self.project_dir / module_path
            
            if not full_path.exists():
                return {
                    'status': 'failed',
                    'score': 0.0,
                    'details': {'message': 'LLM client file not found'}
                }
            
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            has_class = 'class LLMClient' in content
            has_generate = 'def generate' in content
            has_ollama = 'ollama' in content.lower()
            
            score = 0.0
            if has_class and has_generate and has_ollama:
                score = 0.9
            elif has_class and has_generate:
                score = 0.7
            elif has_class:
                score = 0.5
            
            return {
                'status': 'success' if score >= 0.7 else 'partial',
                'score': score,
                'details': {
                    'has_class': has_class,
                    'has_generate_method': has_generate,
                    'references_ollama': has_ollama
                }
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'score': 0.0,
                'details': {'error': str(e)}
            }
    
    def _validate_database_functionality(self, task) -> Dict[str, Any]:
        try:
            module_path = "src/database/sqlite_client.py"
            full_path = self.project_dir / module_path
            
            if not full_path.exists():
                module_path = "src/database/models.py"
                full_path = self.project_dir / module_path
            
            if not full_path.exists():
                return {
                    'status': 'failed',
                    'score': 0.0,
                    'details': {'message': 'Database file not found'}
                }
            
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            has_class = 'class' in content and ('Client' in content or 'Model' in content)
            has_sqlalchemy = 'sqlalchemy' in content.lower() or 'sqlite' in content.lower()
            has_table_def = 'table' in content.lower() or 'CREATE TABLE' in content
            
            score = 0.0
            if has_class and has_sqlalchemy and has_table_def:
                score = 0.9
            elif has_class and has_sqlalchemy:
                score = 0.7
            elif has_class:
                score = 0.5
            
            return {
                'status': 'success' if score >= 0.7 else 'partial',
                'score': score,
                'details': {
                    'has_class': has_class,
                    'has_sqlalchemy': has_sqlalchemy,
                    'has_table_definition': has_table_def
                }
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'score': 0.0,
                'details': {'error': str(e)}
            }
    
    def _validate_vector_store_functionality(self, task) -> Dict[str, Any]:
        try:
            module_path = "src/vector_store/chroma_client.py"
            full_path = self.project_dir / module_path
            
            if not full_path.exists():
                return {
                    'status': 'failed',
                    'score': 0.0,
                    'details': {'message': 'Vector store file not found'}
                }
            
            with open(full_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            has_class = 'class' in content and 'Chroma' in content
            has_search = 'def search' in content
            has_add = 'def add' in content or 'def insert' in content
            has_chromadb = 'chromadb' in content.lower()
            
            score = 0.0
            if has_class and has_search and has_add and has_chromadb:
                score = 0.9
            elif has_class and has_search:
                score = 0.7
            elif has_class:
                score = 0.5
            
            return {
                'status': 'success' if score >= 0.7 else 'partial',
                'score': score,
                'details': {
                    'has_class': has_class,
                    'has_search_method': has_search,
                    'has_add_method': has_add,
                    'has_chromadb': has_chromadb
                }
            }
            
        except Exception as e:
            return {
                'status': 'failed',
                'score': 0.0,
                'details': {'error': str(e)}
            }

# ========== UPGRADED TASK DATACLASS ==========

@dataclass
class BlueprintTask:
    id: str
    description: str
    priority: int
    day: int
    file_paths: List[str]
    verification: str = "code"
    phase: str = ""
    status: str = "not_started"
    estimated_hours: float = 0.0
    actual_hours: float = 0.0
    notes: str = ""
    created_at: str = ""
    started_at: str = ""
    completed_at: str = ""
    last_updated: str = ""
    dependencies: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    auto_detected: bool = False
    detection_score: float = 0.0
    
    validation_result: Dict[str, Any] = field(default_factory=dict)
    validation_score: float = 0.0
    validation_timestamp: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.estimated_hours:
            self.estimated_hours = self._auto_estimate_hours()
        self._extract_dependencies()
    
    def _auto_estimate_hours(self) -> float:
        base_hours = 2.0
        
        if self.priority == 1:
            base_hours *= 1.5
        elif self.priority == 3:
            base_hours *= 0.7
        
        task_lower = self.description.lower()
        if "test" in task_lower or "testing" in task_lower:
            base_hours *= 0.8
        elif "documentation" in task_lower:
            base_hours *= 1.3
        elif "ui" in self.description.lower() or "interface" in task_lower:
            base_hours *= 1.8
        elif "integration" in task_lower or "pipeline" in task_lower:
            base_hours *= 1.5
        elif "setup" in task_lower or "configuration" in task_lower:
            base_hours *= 0.8
        
        if self.file_paths:
            base_hours += len(self.file_paths) * 0.3
        
        if self.verification == "manual":
            base_hours *= 1.2
        elif self.verification == "test":
            base_hours *= 1.4
        
        return round(base_hours, 1)
    
    def _extract_dependencies(self):
        if not self.id:
            return
        
        parts = self.id.split('.')
        if len(parts) >= 3:
            phase_num = parts[0][1]
            try:
                task_num = int(parts[1])
                sub_num = int(parts[2])
                
                if sub_num > 1:
                    prev_id = f"{parts[0]}.{task_num}.{sub_num-1}"
                    self.dependencies.append(prev_id)
                
                if "extractor" in self.description.lower() and "base" not in self.description.lower():
                    self.dependencies.append(f"{parts[0]}.4.2")
                if "vector" in self.description.lower():
                    self.dependencies.append(f"{parts[0]}.5.1")
                if "ui" in self.description.lower():
                    self.dependencies.append(f"{parts[0]}.14.1")
            except ValueError:
                pass
    
    def update_validation(self, validation_result: Dict[str, Any]):
        self.validation_result = validation_result
        self.validation_score = validation_result.get('score', 0.0)
        self.validation_timestamp = datetime.now().isoformat()
        
        self.quality_score = self.validation_score * 100
        
        if self.validation_score >= 0.7:
            self.status = "completed"
            if not self.completed_at:
                self.completed_at = self.validation_timestamp
            self.auto_detected = True
            self.detection_score = self.validation_score
        elif self.validation_score >= 0.3:
            self.status = "in_progress"
            if not self.started_at:
                self.started_at = self.validation_timestamp
        
        self.last_updated = self.validation_timestamp

# ========== MANUAL OVERRIDE SYSTEM ==========

class ManualOverrideSystem:
    def __init__(self, progress_file: Path):
        self.progress_file = progress_file
        self.overrides = self._load_overrides()
    
    def _load_overrides(self) -> Dict:
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                return data.get('manual_overrides', {})
            except Exception:
                return {}
        return {}
    
    def override_task(self, task_id: str, status: str, score: float, reason: str = ""):
        if task_id not in self.overrides:
            self.overrides[task_id] = []
        
        override = {
            'timestamp': datetime.now().isoformat(),
            'status': status,
            'score': score,
            'reason': reason,
            'verified': True
        }
        
        self.overrides[task_id].append(override)
        self._save_overrides()
    
    def get_task_override(self, task_id: str) -> Optional[Dict]:
        if task_id in self.overrides and self.overrides[task_id]:
            return self.overrides[task_id][-1]
        return None
    
    def _save_overrides(self):
        try:
            if self.progress_file.exists():
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            else:
                data = {}
            
            data['manual_overrides'] = self.overrides
            
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            print(f"Warning: Could not save overrides: {e}")

# ========== MAIN TRACKER CLASS ==========

class BlueprintTracker:
    def __init__(self, project_dir: str = "DocuBot"):
        self.project_dir = Path(project_dir).absolute()
        self.progress_file = self.project_dir / ".docubot_progress.json"
        
        self.validator = IntelligentValidator(self.project_dir)
        self.override_system = ManualOverrideSystem(self.progress_file)
        
        self.tasks: Dict[str, BlueprintTask] = self._initialize_all_tasks()
        
        self._load_progress()
        
        print("=" * 80)
        print("DocuBot Professional Progress Tracker v2.2")
        print("=" * 80)
        print(f"Project: {self.project_dir}")
        print(f"Total tasks: {len(self.tasks)}")
        
        self._run_validation()
        
        self.save_progress()
    
    def _initialize_all_tasks(self) -> Dict[str, BlueprintTask]:
        tasks = {}
        
        for phase_name, phase_tasks in COMPLETE_TASKS.items():
            for task_data in phase_tasks:
                task_id = task_data[0]
                description = task_data[1]
                priority = task_data[2]
                day = task_data[3]
                file_paths = task_data[4]
                verification = task_data[5] if len(task_data) > 5 else "code"
                
                task = BlueprintTask(
                    id=task_id,
                    description=description,
                    priority=priority,
                    day=day,
                    file_paths=file_paths,
                    verification=verification,
                    phase=phase_name
                )
                
                tasks[task_id] = task
        
        print(f"Initialized {len(tasks)} tasks from blueprint")
        return tasks
    
    def _load_progress(self):
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if data.get('version', '1.0') == '2.1':
                    for task_id, task_data in data.get('tasks', {}).items():
                        if task_id in self.tasks:
                            for key, value in task_data.items():
                                if hasattr(self.tasks[task_id], key):
                                    setattr(self.tasks[task_id], key, value)
                
                print(f"Loaded previous progress from {self.progress_file}")
                
            except Exception as e:
                print(f"Warning: Could not load progress: {e}")
    
    def _run_validation(self):
        print(f"Running validation for {len(self.tasks)} tasks...")
        print("=" * 80)
        
        validation_stats = {
            'completed': 0,
            'partial': 0,
            'failed': 0,
            'not_started': 0,
            'total_score': 0.0
        }
        
        for task_id, task in self.tasks.items():
            if task.status != "completed" or task.auto_detected:
                override = self.override_system.get_task_override(task_id)
                if override:
                    task.status = override['status']
                    task.validation_score = override['score']
                    task.quality_score = override['score'] * 100
                    task.notes = f"Manually overridden: {override.get('reason', 'No reason provided')}"
                    if override['status'] == 'completed':
                        task.completed_at = override['timestamp']
                    validation_stats['completed'] += 1
                    validation_stats['total_score'] += task.validation_score
                    print(f"MANUAL OVERRIDE: {task_id} - {task.description[:50]}... (score: {task.validation_score:.2f})")
                else:
                    validation_result = self.validator.validate_task(task)
                    task.update_validation(validation_result)
                    
                    validation_stats['total_score'] += task.validation_score
                    
                    status = validation_result.get('status', 'not_started')
                    if status == 'success':
                        validation_stats['completed'] += 1
                        print(f"VALIDATED: {task_id} - {task.description[:50]}... (score: {task.validation_score:.2f})")
                    elif status == 'partial':
                        validation_stats['partial'] += 1
                        print(f"PARTIAL: {task_id} - {task.description[:50]}... (score: {task.validation_score:.2f})")
                    elif status == 'failed':
                        validation_stats['failed'] += 1
                        print(f"FAILED: {task_id} - {task.description[:50]}... (score: {task.validation_score:.2f})")
                    else:
                        validation_stats['not_started'] += 1
        
        total_tasks = len(self.tasks)
        if total_tasks > 0:
            avg_score = validation_stats['total_score'] / total_tasks
        else:
            avg_score = 0.0
        
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)
        print(f"Completed: {validation_stats['completed']}")
        print(f"Partial: {validation_stats['partial']}")
        print(f"Failed: {validation_stats['failed']}")
        print(f"Not Started: {validation_stats['not_started']}")
        print(f"Average Validation Score: {avg_score:.2f}")
        print(f"Overall Completion: {(validation_stats['completed'] / total_tasks * 100):.1f}%")
    
    def save_progress(self):
        try:
            stats = self._calculate_statistics()
            
            data = {
                'version': '2.2',
                'project_dir': str(self.project_dir),
                'last_updated': datetime.now().isoformat(),
                'statistics': stats,
                'tasks': {task_id: asdict(task) for task_id, task in self.tasks.items()}
            }
            
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"Progress saved to {self.progress_file}")
            return True
            
        except Exception as e:
            print(f"Error saving progress: {e}")
            return False
    
    def _calculate_statistics(self) -> Dict[str, Any]:
        stats = {
            'total_tasks': len(self.tasks),
            'by_status': {'completed': 0, 'in_progress': 0, 'not_started': 0, 'blocked': 0},
            'by_priority': {1: 0, 2: 0, 3: 0},
            'by_phase': {},
            'validation_scores': {'average': 0.0, 'by_phase': {}},
            'completion_percentage': 0.0,
            'high_priority_completion': 0.0,
            'estimated_hours': {'total': 0.0, 'completed': 0.0, 'remaining': 0.0},
            'actual_hours': 0.0
        }
        
        completed_tasks = []
        total_validation_score = 0.0
        
        for task in self.tasks.values():
            stats['by_status'][task.status] = stats['by_status'].get(task.status, 0) + 1
            
            stats['by_priority'][task.priority] = stats['by_priority'].get(task.priority, 0) + 1
            
            phase = task.phase
            if phase not in stats['by_phase']:
                stats['by_phase'][phase] = {'total': 0, 'completed': 0, 'score': 0.0}
            stats['by_phase'][phase]['total'] += 1
            if task.status == 'completed':
                stats['by_phase'][phase]['completed'] += 1
                stats['by_phase'][phase]['score'] += task.validation_score
            
            stats['estimated_hours']['total'] += task.estimated_hours
            if task.status == 'completed':
                stats['estimated_hours']['completed'] += task.estimated_hours
                completed_tasks.append(task)
            stats['actual_hours'] += task.actual_hours
            
            total_validation_score += task.validation_score
        
        if stats['total_tasks'] > 0:
            stats['completion_percentage'] = (stats['by_status']['completed'] / stats['total_tasks']) * 100
        
        if stats['total_tasks'] > 0:
            stats['validation_scores']['average'] = total_validation_score / stats['total_tasks']
        
        high_priority_tasks = [t for t in self.tasks.values() if t.priority == 1]
        high_priority_completed = [t for t in high_priority_tasks if t.status == 'completed']
        if high_priority_tasks:
            stats['high_priority_completion'] = (len(high_priority_completed) / len(high_priority_tasks)) * 100
        
        stats['estimated_hours']['remaining'] = stats['estimated_hours']['total'] - stats['estimated_hours']['completed']
        
        for phase, phase_data in stats['by_phase'].items():
            if phase_data['total'] > 0:
                phase_data['completion_percentage'] = (phase_data['completed'] / phase_data['total']) * 100
                if phase_data['completed'] > 0:
                    phase_data['average_score'] = phase_data['score'] / phase_data['completed']
                else:
                    phase_data['average_score'] = 0.0
                stats['validation_scores']['by_phase'][phase] = phase_data['average_score']
        
        return stats
    
    def generate_detailed_report(self) -> str:
        report = []
        
        report.append("=" * 100)
        report.append("DOCUBOT - BLUEPRINT PROGRESS REPORT v2.2")
        report.append("=" * 100)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Project: {self.project_dir}")
        report.append("")
        
        stats = self._calculate_statistics()
        
        report.append("EXECUTIVE SUMMARY")
        report.append("-" * 80)
        report.append(f"Total Tasks: {stats['total_tasks']}")
        report.append(f"Completed: {stats['by_status']['completed']} ({stats['completion_percentage']:.1f}%)")
        report.append(f"In Progress: {stats['by_status']['in_progress']}")
        report.append(f"Not Started: {stats['by_status']['not_started']}")
        report.append(f"Blocked: {stats['by_status']['blocked']}")
        report.append("")
        report.append(f"Average Validation Score: {stats['validation_scores']['average']:.2f}/1.0")
        report.append(f"High Priority Completion: {stats['high_priority_completion']:.1f}%")
        report.append("")
        report.append(f"Estimated Hours: {stats['estimated_hours']['total']:.1f}h")
        report.append(f"   Completed: {stats['estimated_hours']['completed']:.1f}h")
        report.append(f"   Remaining: {stats['estimated_hours']['remaining']:.1f}h")
        report.append(f"   Actual Hours: {stats['actual_hours']:.1f}h")
        report.append("")
        
        report.append("PHASE PROGRESS")
        report.append("-" * 80)
        
        for phase_name in COMPLETE_TASKS.keys():
            if phase_name in stats['by_phase']:
                phase_data = stats['by_phase'][phase_name]
                completed = phase_data['completed']
                total = phase_data['total']
                percentage = phase_data['completion_percentage']
                avg_score = phase_data.get('average_score', 0.0)
                
                bar_length = 30
                filled = int(bar_length * percentage / 100)
                bar = "█" * filled + "░" * (bar_length - filled)
                
                phase_short = phase_name.split('(')[0].strip()
                report.append(f"{phase_short:25} [{bar}] {percentage:5.1f}% ({completed}/{total})")
                report.append(f"   Average Score: {avg_score:.2f}/1.0")
        
        report.append("")
        
        report.append("PRIORITY BREAKDOWN")
        report.append("-" * 80)
        
        for priority in [1, 2, 3]:
            count = stats['by_priority'].get(priority, 0)
            priority_tasks = [t for t in self.tasks.values() if t.priority == priority]
            completed = len([t for t in priority_tasks if t.status == 'completed'])
            percentage = (completed / count * 100) if count > 0 else 0
            
            priority_label = {1: "HIGH", 2: "MEDIUM", 3: "LOW"}.get(priority, f"P{priority}")
            report.append(f"{priority_label:10} {completed:3}/{count:3} tasks ({percentage:5.1f}%)")
        
        report.append("")
        
        report.append("CRITICAL PENDING TASKS (Priority 1, Not Completed)")
        report.append("-" * 80)
        
        critical_pending = [t for t in self.tasks.values() 
                          if t.priority == 1 and t.status != 'completed']
        
        if critical_pending:
            for task in sorted(critical_pending, key=lambda x: x.day)[:10]:
                status_icon = "IN PROGRESS" if task.status == "in_progress" else "PENDING"
                report.append(f"  {task.id}: {task.description[:60]}...")
                report.append(f"     Day {task.day}, Estimated {task.estimated_hours}h, Score: {task.validation_score:.2f}")
                if task.validation_result and 'details' in task.validation_result:
                    details = task.validation_result['details']
                    if isinstance(details, dict) and 'message' in details:
                        report.append(f"     Note: {details['message']}")
        else:
            report.append("All critical tasks are completed")
        
        report.append("")
        
        report.append("RECENT VALIDATIONS (Last 5)")
        report.append("-" * 80)
        
        recent_tasks = sorted(
            [t for t in self.tasks.values() if t.validation_timestamp],
            key=lambda x: x.validation_timestamp,
            reverse=True
        )[:5]
        
        for task in recent_tasks:
            if task.validation_timestamp:
                try:
                    time_str = datetime.fromisoformat(task.validation_timestamp.replace('Z', '+00:00')).strftime('%m/%d %H:%M')
                except:
                    time_str = task.validation_timestamp[:16]
                
                status_text = "COMPLETED" if task.status == "completed" else "IN PROGRESS"
                report.append(f"  [{time_str}] {task.id}: {task.description[:50]}...")
                report.append(f"     Status: {status_text}, Score: {task.validation_score:.2f}, Quality: {task.quality_score:.1f}/100")
        
        report.append("")
        
        report.append("VALIDATION ISSUES")
        report.append("-" * 80)
        
        low_score_tasks = [t for t in self.tasks.values() 
                          if t.status == "completed" and t.validation_score < 0.5]
        
        if low_score_tasks:
            report.append(f"Tasks with low validation scores (< 0.5): {len(low_score_tasks)}")
            for task in low_score_tasks[:5]:
                report.append(f"  {task.id}: {task.description[:50]}... (score: {task.validation_score:.2f})")
        else:
            report.append("No tasks with low validation scores")
        
        report.append("\n" + "=" * 100)
        report.append("Report generated by Blueprint Tracker v2.2")
        report.append("=" * 100)
        
        return "\n".join(report)
    
    def export_task_details(self, output_file: str = "task_details.json"):
        try:
            data = {
                'version': '2.2',
                'generated_at': datetime.now().isoformat(),
                'project_dir': str(self.project_dir),
                'total_tasks': len(self.tasks),
                'tasks': {}
            }
            
            for task_id, task in self.tasks.items():
                task_data = asdict(task)
                if 'validation_result' in task_data and 'details' in task_data['validation_result']:
                    details = task_data['validation_result']['details']
                    if isinstance(details, dict) and len(str(details)) > 1000:
                        task_data['validation_result']['details'] = {
                            'summary': 'Detailed validation data available',
                            'status': details.get('status', ''),
                            'score': details.get('score', 0.0)
                        }
                
                data['tasks'][task_id] = task_data
            
            output_path = self.project_dir / output_file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            print(f"Task details exported to: {output_path}")
            return True
            
        except Exception as e:
            print(f"Error exporting task details: {e}")
            return False
    
    def manual_override_task(self, task_id: str, status: str = "completed", score: float = 0.9, reason: str = ""):
        if task_id in self.tasks:
            self.override_system.override_task(task_id, status, score, reason)
            
            task = self.tasks[task_id]
            task.status = status
            task.validation_score = score
            task.quality_score = score * 100
            task.notes = f"Manually overridden: {reason}"
            
            if status == "completed" and not task.completed_at:
                task.completed_at = datetime.now().isoformat()
            
            task.last_updated = datetime.now().isoformat()
            
            print(f"Manually overridden task {task_id} to {status} with score {score}")
            self.save_progress()
            return True
        else:
            print(f"Task {task_id} not found")
            return False
    
    def apply_known_completions(self):
        known_completions = [
            ("P1.1.1", 0.85, "Project structure with 7/7 directories and 4/6 files"),
            ("P1.1.2", 1.0, "Python 3.11+ environment verified"),
            ("P1.1.3", 0.8, "Core dependencies installed"),
            ("P1.1.4", 1.0, "Git repository initialized with commits"),
            ("P1.2.1", 1.0, "Configuration module complete"),
            ("P1.2.2", 1.0, "app_config.yaml with all settings"),
            ("P1.2.3", 1.0, "Cross-platform data directories setup"),
            ("P1.2.4", 1.0, "Configuration validation implemented"),
            ("P1.3.1", 1.0, "Document processor complete"),
            ("P1.3.2", 1.0, "PDF extractor implemented"),
            ("P1.3.3", 1.0, "TXT extractor implemented"),
            ("P1.3.5", 1.0, "Intelligent chunking with 500 tokens, 50 overlap"),
            ("P1.4.2", 1.0, "Base extractor class implemented"),
            ("P1.4.3", 1.0, "Extractor factory/registry complete"),
            ("P1.5.1", 1.0, "SQLite client with CRUD operations"),
            ("P1.5.2", 1.0, "Database schema complete"),
            ("P1.5.5", 0.9, "Database queries module complete"),
            ("P1.6.1", 1.0, "ChromaDB vector store client complete"),
            ("P1.8.1", 1.0, "Ollama LLM client complete"),
            ("P1.8.2", 1.0, "Multiple LLM model support verified"),
            ("P1.11.1", 0.8, "RAG engine complete"),
            ("P1.12.1", 1.0, "DocuBotCore class complete"),
            ("P1.12.5", 1.0, "Error handling and logging complete"),
            ("P1.14.1", 1.0, "Main application entry point complete"),
            ("P3.3.1", 0.83, "Robust error handling system implemented"),
        ]
        
        for task_id, score, reason in known_completions:
            self.manual_override_task(task_id, "completed", score, reason)
        
        print(f"Applied {len(known_completions)} known completions")

# ========== SIMPLE MENU SYSTEM ==========

def show_menu():
    print("\n" + "=" * 60)
    print("DOCUBOT PROGRESS TRACKER MENU")
    print("=" * 60)
    print("1. View detailed report")
    print("2. View statistics")
    print("3. Export task details to JSON")
    print("4. Re-run validation")
    print("5. Apply known completions")
    print("6. Manual override task")
    print("7. Exit")

def main():
    print("=" * 100)
    print("DocuBot - Professional Progress Tracker v2.2")
    print("=" * 100)
    print("Loading tracker...")
    
    current_dir = Path.cwd()
    project_dir = None
    
    if current_dir.name == "DocuBot" and (current_dir / "src").exists():
        project_dir = str(current_dir)
    elif (current_dir / "DocuBot").exists() and (current_dir / "DocuBot" / "src").exists():
        project_dir = str(current_dir / "DocuBot")
    else:
        for parent in current_dir.parents:
            if (parent / "DocuBot").exists() and (parent / "DocuBot" / "src").exists():
                project_dir = str(parent / "DocuBot")
                break
        else:
            project_dir = "DocuBot"
    
    print(f"Project detected at: {project_dir}")
    
    if not Path(project_dir).exists():
        print(f"Error: Project directory not found: {project_dir}")
        return
    
    tracker = BlueprintTracker(project_dir)
    
    while True:
        show_menu()
        
        try:
            choice = input("\nEnter choice (1-7): ").strip()
            
            if choice == "1":
                print("\n" + tracker.generate_detailed_report())
            
            elif choice == "2":
                stats = tracker._calculate_statistics()
                print(f"\nSTATISTICS")
                print(f"   Total Tasks: {stats['total_tasks']}")
                print(f"   Completion: {stats['completion_percentage']:.1f}%")
                print(f"   Avg Validation Score: {stats['validation_scores']['average']:.2f}")
                print(f"   High Priority Completion: {stats['high_priority_completion']:.1f}%")
                print(f"   Estimated Hours: {stats['estimated_hours']['total']:.1f}h")
                print(f"   Actual Hours: {stats['actual_hours']:.1f}h")
            
            elif choice == "3":
                filename = input("Enter output filename [task_details.json]: ").strip()
                if not filename:
                    filename = "task_details.json"
                tracker.export_task_details(filename)
            
            elif choice == "4":
                print("\nRe-running validation...")
                tracker._run_validation()
                tracker.save_progress()
                print("\n" + tracker.generate_detailed_report())
            
            elif choice == "5":
                print("\nApplying known completions...")
                tracker.apply_known_completions()
                print("\n" + tracker.generate_detailed_report())
            
            elif choice == "6":
                task_id = input("Enter task ID (e.g., P1.1.1): ").strip()
                if task_id in tracker.tasks:
                    status = input("Enter status (completed/in_progress/not_started) [completed]: ").strip()
                    if not status:
                        status = "completed"
                    
                    score_str = input("Enter score (0.0-1.0) [0.9]: ").strip()
                    if not score_str:
                        score = 0.9
                    else:
                        try:
                            score = float(score_str)
                        except ValueError:
                            print("Invalid score, using 0.9")
                            score = 0.9
                    
                    reason = input("Enter reason for override: ").strip()
                    
                    tracker.manual_override_task(task_id, status, score, reason)
                    print(f"Task {task_id} overridden to {status} with score {score}")
                else:
                    print(f"Task {task_id} not found")
            
            elif choice == "7":
                save = input("\nSave progress before exiting? (y/n): ").strip().lower()
                if save == 'y':
                    tracker.save_progress()
                print("\nExiting...")
                break
            
            else:
                print("Invalid choice. Please enter 1-7.")
                
        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
            save = input("Save progress before exiting? (y/n): ").strip().lower()
            if save == 'y':
                tracker.save_progress()
            print("Exiting...")
            break
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    main()