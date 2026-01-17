#docubot/9. structure_validator.py
"""
DocuBot Project Structure Validator v1.0
Validates project structure against blueprint specification.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
import argparse


class ProjectStructureValidator:
    """Validates DocuBot project structure against blueprint."""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root).resolve()
        self.expected_structure = self._load_expected_structure()
        self.validation_results = []
        self.missing_items = []
        self.partial_items = []
        
    def _load_expected_structure(self) -> Dict:
        """Load expected structure from blueprint specification."""
        
        return {
            "directories": {
                "src/": {
                    "required": True,
                    "subdirs": [
                        "core/", "document_processing/", "ai_engine/",
                        "vector_store/", "database/", "ui/", "storage/",
                        "utilities/", "plugins/"
                    ]
                },
                "src/core/": {"required": True},
                "src/document_processing/": {"required": True, "subdirs": ["extractors/"]},
                "src/ai_engine/": {"required": True},
                "src/vector_store/": {"required": True},
                "src/database/": {"required": True, "subdirs": ["migrations/"]},
                "src/ui/": {"required": True, "subdirs": ["desktop/", "web/", "cli/"]},
                "src/storage/": {"required": True},
                "src/utilities/": {"required": True},
                "src/plugins/": {"required": True, "subdirs": ["builtin_plugins/"]},
                "data/": {
                    "required": True,
                    "subdirs": ["models/", "database/", "documents/", "config/", "logs/"]
                },
                "tests/": {
                    "required": True,
                    "subdirs": ["unit/", "integration/", "e2e/", "performance/", "test_data/"]
                },
                "docs/": {
                    "required": True,
                    "subdirs": ["user_guide/", "developer_guide/", "api_reference/", "troubleshooting/"]
                },
                "scripts/": {"required": True},
                "resources/": {
                    "required": True,
                    "subdirs": ["icons/", "sounds/", "templates/", "translations/"]
                },
                "bin/": {"required": True},
            },
            "files": {
                "root": [
                    "app.py", "requirements.txt", "requirements-dev.txt",
                    "pyproject.toml", "README.md", "LICENSE", "CHANGELOG.md",
                    ".gitignore", ".env.example"
                ],
                "src/core/": [
                    "__init__.py", "app.py", "config.py", "constants.py", "exceptions.py"
                ],
                "src/document_processing/": [
                    "__init__.py", "processor.py", "chunking.py", "cleaning.py", "metadata.py"
                ],
                "src/document_processing/extractors/": [
                    "__init__.py", "pdf_extractor.py", "docx_extractor.py", "txt_extractor.py",
                    "epub_extractor.py", "image_extractor.py", "web_extractor.py", "csv_extractor.py",
                    "markdown_extractor.py", "html_extractor.py"
                ],
                "src/ai_engine/": [
                    "__init__.py", "llm_client.py", "rag_engine.py", "embedding_service.py",
                    "prompt_templates.py", "model_manager.py", "summarizer.py", "tagging.py"
                ],
                "src/vector_store/": [
                    "__init__.py", "chroma_client.py", "faiss_client.py",
                    "search_engine.py", "index_manager.py"
                ],
                "src/database/": ["__init__.py", "sqlite_client.py", "models.py", "queries.py"],
                "src/database/migrations/": ["__init__.py"],
                "src/ui/desktop/": [
                    "__init__.py", "main_window.py", "chat_panel.py", "document_panel.py",
                    "settings_panel.py", "history_panel.py", "export_manager.py",
                    "reading_tracker.py", "annotation_tool.py", "shortcuts.py",
                    "search_bar.py", "help_system.py", "error_dialog.py"
                ],
                "src/ui/desktop/components/": [
                    "__init__.py", "file_uploader.py", "chat_message.py",
                    "document_card.py", "status_bar.py"
                ],
                "src/ui/desktop/themes/": [
                    "__init__.py", "dark_theme.py", "light_theme.py", "system_theme.py"
                ],
                "src/ui/web/": ["__init__.py", "app.py"],
                "src/ui/web/pages/": ["__init__.py", "chat.py", "documents.py", "settings.py"],
                "src/ui/web/components/": [
                    "__init__.py", "chat_ui.py", "document_list.py", "upload_widget.py"
                ],
                "src/ui/cli/": [
                    "__init__.py", "cli.py", "commands.py", "output_formatters.py"
                ],
                "src/storage/": [
                    "__init__.py", "file_manager.py", "cache_manager.py",
                    "backup_manager.py", "encryption.py", "collection_manager.py"
                ],
                "src/utilities/": [
                    "__init__.py", "logger.py", "validator.py", "formatter.py",
                    "monitor.py", "helpers.py", "retry.py", "cleanup.py", "task_queue.py"
                ],
                "src/plugins/": ["__init__.py", "plugin_manager.py", "base_plugin.py"],
                "src/plugins/builtin_plugins/": [
                    "__init__.py", "obsidian_sync.py", "notion_export.py",
                    "browser_clipper.py", "voice_interface.py"
                ],
                "scripts/": [
                    "install.py", "setup.py", "backup.py", "update.py", "diagnostic.py",
                    "build_windows.py", "build_mac.py", "build_linux.py",
                    "init_db.py", "validate_resources.py"
                ],
                "tests/": ["__init__.py", "conftest.py"],
                "tests/unit/": [
                    "__init__.py", "test_document_processor.py", "test_llm_client.py",
                    "test_rag_engine.py", "test_vector_store.py", "test_database.py",
                    "test_embedding_service.py"
                ],
                "tests/integration/": [
                    "__init__.py", "test_rag_pipeline.py", "test_ui_integration.py",
                    "test_database_integration.py"
                ],
                "tests/e2e/": [
                    "__init__.py", "test_full_workflow.py", "test_user_scenarios.py"
                ],
                "tests/test_data/": ["sample.pdf", "sample.docx", "sample.txt"],
                "bin/": ["docubot", "docubot.exe", "docubot.bat"]
            }
        }
    
    def scan_project(self) -> Dict:
        """Scan actual project structure."""
        
        actual_structure = {
            "directories": set(),
            "files": set()
        }
        
        for root, dirs, files in os.walk(self.project_root):
            root_path = Path(root)
            
            relative_root = root_path.relative_to(self.project_root)
            if str(relative_root) == ".":
                continue
                
            actual_structure["directories"].add(str(relative_root) + "/")
            
            for file in files:
                file_path = root_path / file
                relative_file = file_path.relative_to(self.project_root)
                actual_structure["files"].add(str(relative_file))
        
        return actual_structure
    
    def validate_structure(self) -> Tuple[List, List]:
        """Validate project structure against blueprint."""
        
        actual = self.scan_project()
        missing_dirs = []
        missing_files = []
        
        expected_dirs = self.expected_structure["directories"]
        expected_files = self.expected_structure["files"]
        
        for dir_path, dir_info in expected_dirs.items():
            if dir_info.get("required", True):
                if dir_path not in actual["directories"]:
                    missing_dirs.append(dir_path)
                    
                subdirs = dir_info.get("subdirs", [])
                for subdir in subdirs:
                    full_subdir = dir_path.rstrip("/") + "/" + subdir.lstrip("/")
                    if full_subdir not in actual["directories"]:
                        missing_dirs.append(full_subdir)
        
        for dir_path, files in expected_files.items():
            for file in files:
                file_path = dir_path.rstrip("/") + "/" + file
                if file_path not in actual["files"]:
                    missing_files.append(file_path)
        
        self.missing_items = missing_dirs + missing_files
        
        for item in self.missing_items:
            if item.endswith("/"):
                self.validation_results.append(("MISSING_DIR", item))
            else:
                self.validation_results.append(("MISSING_FILE", item))
        
        return missing_dirs, missing_files
    
    def check_git_status(self) -> Dict:
        """Check Git repository status."""
        
        git_status = {
            "initialized": False,
            "has_commits": False,
            "uncommitted_changes": False
        }
        
        git_dir = self.project_root / ".git"
        git_status["initialized"] = git_dir.exists()
        
        if git_status["initialized"]:
            try:
                import subprocess
                result = subprocess.run(
                    ["git", "status", "--porcelain"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True
                )
                git_status["uncommitted_changes"] = bool(result.stdout.strip())
                
                result = subprocess.run(
                    ["git", "log", "--oneline", "-1"],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True
                )
                git_status["has_commits"] = bool(result.stdout.strip())
            except:
                pass
        
        return git_status
    
    def check_core_modules(self) -> List[str]:
        """Check if core modules have actual implementation."""
        
        core_modules = []
        
        core_files = [
            "src/core/config.py",
            "src/core/app.py",
            "src/document_processing/processor.py",
            "src/ai_engine/llm_client.py",
            "src/database/sqlite_client.py"
        ]
        
        for file in core_files:
            file_path = self.project_root / file
            if file_path.exists():
                content = file_path.read_text(encoding='utf-8', errors='ignore')
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                
                if len(lines) > 10:
                    core_modules.append((file, "IMPLEMENTED"))
                else:
                    core_modules.append((file, "EMPTY"))
            else:
                core_modules.append((file, "MISSING"))
        
        return core_modules
    
    def generate_report(self) -> Dict:
        """Generate comprehensive validation report."""
        
        missing_dirs, missing_files = self.validate_structure()
        git_status = self.check_git_status()
        core_modules = self.check_core_modules()
        
        actual = self.scan_project()
        
        report = {
            "project_root": str(self.project_root),
            "timestamp": self._get_timestamp(),
            "overall_status": {
                "directories_found": len(actual["directories"]),
                "files_found": len(actual["files"]),
                "missing_directories": len(missing_dirs),
                "missing_files": len(missing_files),
                "completion_percentage": self._calculate_completion(actual)
            },
            "git_status": git_status,
            "structure_validation": {
                "missing_directories": missing_dirs,
                "missing_files": missing_files,
                "validation_results": self.validation_results
            },
            "core_modules": core_modules,
            "priority_tasks": self._generate_priority_tasks(missing_dirs, missing_files, git_status)
        }
        
        return report
    
    def _calculate_completion(self, actual_structure: Dict) -> float:
        """Calculate structure completion percentage."""
        
        expected_dirs = set(self.expected_structure["directories"].keys())
        expected_files = set()
        
        for dir_path, files in self.expected_structure["files"].items():
            for file in files:
                expected_files.add(dir_path.rstrip("/") + "/" + file)
        
        dirs_found = len(expected_dirs.intersection(actual_structure["directories"]))
        files_found = len(expected_files.intersection(actual_structure["files"]))
        
        total_expected = len(expected_dirs) + len(expected_files)
        total_found = dirs_found + files_found
        
        if total_expected == 0:
            return 0.0
        
        return round((total_found / total_expected) * 100, 2)
    
    def _generate_priority_tasks(self, missing_dirs: List, missing_files: List, git_status: Dict) -> List[Dict]:
        """Generate priority tasks based on missing items."""
        
        priority_tasks = []
        
        if not git_status["initialized"]:
            priority_tasks.append({
                "task": "Initialize Git repository",
                "priority": "HIGH",
                "action": "Run: git init && git add . && git commit -m 'Initial project structure'",
                "estimated_time": "15 minutes"
            })
        
        if missing_dirs:
            priority_tasks.append({
                "task": "Create missing directories",
                "priority": "HIGH",
                "action": f"Create {len(missing_dirs)} missing directories",
                "details": missing_dirs[:5],
                "estimated_time": "30 minutes"
            })
        
        core_missing = [f for f in missing_files if "src/core/" in f or "src/document_processing/" in f]
        if core_missing:
            priority_tasks.append({
                "task": "Create core module files",
                "priority": "HIGH",
                "action": f"Create {len(core_missing)} core files",
                "details": core_missing[:5],
                "estimated_time": "1-2 hours"
            })
        
        if "src/ai_engine/llm_client.py" in missing_files:
            priority_tasks.append({
                "task": "Implement LLM client",
                "priority": "MEDIUM",
                "action": "Create AI engine integration",
                "estimated_time": "2-3 hours"
            })
        
        if "src/database/sqlite_client.py" in missing_files:
            priority_tasks.append({
                "task": "Implement database client",
                "priority": "MEDIUM",
                "action": "Create SQLite database operations",
                "estimated_time": "2 hours"
            })
        
        return priority_tasks
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def print_report(self, report: Dict):
        """Print formatted report to console."""
        
        print("=" * 80)
        print("DOCUBOT PROJECT STRUCTURE VALIDATION REPORT")
        print("=" * 80)
        print(f"Project Root: {report['project_root']}")
        print(f"Timestamp: {report['timestamp']}")
        print()
        
        status = report["overall_status"]
        print("OVERALL STATUS:")
        print(f"  Directories Found: {status['directories_found']}")
        print(f"  Files Found: {status['files_found']}")
        print(f"  Missing Directories: {status['missing_directories']}")
        print(f"  Missing Files: {status['missing_files']}")
        print(f"  Completion Percentage: {status['completion_percentage']}%")
        print()
        
        git = report["git_status"]
        print("GIT REPOSITORY STATUS:")
        print(f"  Initialized: {'YES' if git['initialized'] else 'NO'}")
        if git["initialized"]:
            print(f"  Has Commits: {'YES' if git['has_commits'] else 'NO'}")
            print(f"  Uncommitted Changes: {'YES' if git['uncommitted_changes'] else 'NO'}")
        print()
        
        print("CORE MODULES STATUS:")
        for module, state in report["core_modules"]:
            print(f"  {module}: {state}")
        print()
        
        if report["structure_validation"]["missing_directories"]:
            print("MISSING DIRECTORIES:")
            for dir_path in report["structure_validation"]["missing_directories"][:10]:
                print(f"  - {dir_path}")
            if len(report["structure_validation"]["missing_directories"]) > 10:
                print(f"  ... and {len(report["structure_validation"]["missing_directories"]) - 10} more")
            print()
        
        if report["structure_validation"]["missing_files"]:
            print("MISSING FILES (First 10):")
            for file_path in report["structure_validation"]["missing_files"][:10]:
                print(f"  - {file_path}")
            if len(report["structure_validation"]["missing_files"]) > 10:
                print(f"  ... and {len(report["structure_validation"]["missing_files"]) - 10} more")
            print()
        
        if report["priority_tasks"]:
            print("PRIORITY TASKS:")
            for i, task in enumerate(report["priority_tasks"], 1):
                print(f"{i}. {task['task']} [{task['priority']}]")
                print(f"   Action: {task['action']}")
                if 'details' in task:
                    for detail in task['details']:
                        print(f"     - {detail}")
                print(f"   Estimated Time: {task['estimated_time']}")
                print()
        
        print("=" * 80)
        print("RECOMMENDED IMMEDIATE ACTIONS:")
        print("=" * 80)
        
        if report["priority_tasks"]:
            for task in report["priority_tasks"][:3]:
                if task["priority"] == "HIGH":
                    print(f"1. {task['action']}")
        else:
            print("Project structure appears to be complete!")
            print("Next: Implement core functionality in the existing files.")
        
        print("=" * 80)


def main():
    """Main entry point."""
    
    parser = argparse.ArgumentParser(
        description="Validate DocuBot project structure against blueprint"
    )
    
    parser.add_argument(
        "--path",
        type=str,
        default=".",
        help="Path to project root (default: current directory)"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output report as JSON"
    )
    
    parser.add_argument(
        "--save",
        type=str,
        help="Save report to specified file"
    )
    
    args = parser.parse_args()
    
    validator = ProjectStructureValidator(args.path)
    
    if not validator.project_root.exists():
        print(f"Error: Project root '{args.path}' does not exist.")
        return 1
    
    report = validator.generate_report()
    
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        validator.print_report(report)
    
    if args.save:
        try:
            with open(args.save, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2)
            print(f"\nReport saved to: {args.save}")
        except Exception as e:
            print(f"Error saving report: {e}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())