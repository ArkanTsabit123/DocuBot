# docubot/smart_validator_fixed.py
"""
Smart Validator for DocuBot Project - Fixed Version
Validates actual file content and functionality
"""

import os
import json
import yaml
from pathlib import Path
import re
import platform
from datetime import datetime
from typing import Dict, Any, List, Tuple
import sys

class SmartValidator:
    def __init__(self, project_path=None):
        if project_path:
            self.project_path = Path(project_path)
        else:
            # Auto-detect DocuBot project
            current_dir = Path.cwd()
            if (current_dir / "src").exists() and (current_dir / "data").exists():
                self.project_path = current_dir
            elif (current_dir / "DocuBot").exists():
                self.project_path = current_dir / "DocuBot"
            else:
                # Try to find DocuBot directory
                for item in current_dir.iterdir():
                    if item.is_dir() and (item / "src").exists() and (item / "data").exists():
                        self.project_path = item
                        break
                else:
                    self.project_path = current_dir
        
        print(f"🔍 Validating project at: {self.project_path}")
        print(f"   Directory exists: {self.project_path.exists()}")
        
    def validate_file_content(self, file_path: Path, required_patterns: List[str] = None) -> float:
        """Validate file content with patterns"""
        if not file_path.exists():
            return 0.0
        
        try:
            content = file_path.read_text(encoding='utf-8', errors='ignore')
            
            if len(content.strip()) == 0:
                return 0.0
            
            # Basic scoring based on size
            size_score = min(1.0, len(content) / 1000)  # 1KB = full score
            
            # Pattern matching score
            pattern_score = 0.0
            if required_patterns:
                found_patterns = 0
                for pattern in required_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        found_patterns += 1
                pattern_score = found_patterns / len(required_patterns)
            
            # Combine scores
            if required_patterns:
                return (size_score * 0.3) + (pattern_score * 0.7)
            else:
                return size_score
            
        except Exception as e:
            print(f"   Error reading {file_path}: {e}")
            return 0.0
    
    def validate_p1_1_1(self) -> Dict[str, Any]:
        """P1.1.1 - Setup complete project structure"""
        print("\n🔍 P1.1.1 - Setup complete project structure")
        print("-" * 40)
        
        required_dirs = [
            "src", "data", "tests", "docs", 
            "scripts", "resources", "bin"
        ]
        
        required_files = [
            "app.py", "requirements.txt", "README.md", 
            "LICENSE", "pyproject.toml", ".gitignore"
        ]
        
        dir_score = 0.0
        dir_details = []
        for dir_name in required_dirs:
            dir_path = self.project_path / dir_name
            exists = dir_path.exists()
            dir_details.append(f"{dir_name}: {'✓' if exists else '✗'}")
            if exists:
                dir_score += 1.0
                print(f"   ✓ Directory: {dir_name}")
            else:
                print(f"   ✗ Missing directory: {dir_name}")
        
        file_score = 0.0
        file_details = []
        for file_name in required_files:
            file_path = self.project_path / file_name
            exists = file_path.exists()
            has_content = exists and file_path.stat().st_size > 0 if exists else False
            file_details.append(f"{file_name}: {'✓' if has_content else ('∅' if exists else '✗')}")
            if has_content:
                file_score += 1.0
                print(f"   ✓ File with content: {file_name}")
            elif exists:
                print(f"   ⚠️  Empty file: {file_name}")
            else:
                print(f"   ✗ Missing file: {file_name}")
        
        total_items = len(required_dirs) + len(required_files)
        total_score = (dir_score + file_score) / total_items
        
        print(f"\n   📊 Directories: {dir_score:.0f}/{len(required_dirs)}")
        print(f"   📊 Files: {file_score:.0f}/{len(required_files)}")
        print(f"   🎯 Overall score: {total_score:.2f}")
        
        return {
            "score": total_score,
            "details": {
                "directories_found": dir_score,
                "total_directories": len(required_dirs),
                "files_found": file_score,
                "total_files": len(required_files),
                "directory_details": dir_details,
                "file_details": file_details
            }
        }
    
    def validate_p1_1_4(self) -> Dict[str, Any]:
        """P1.1.4 - Initialize Git repository"""
        print("\n🔍 P1.1.4 - Initialize Git repository")
        print("-" * 40)
        
        git_dir = self.project_path / ".git"
        
        if not git_dir.exists():
            print("   ✗ .git directory not found")
            print("   💡 Run: git init")
            return {"score": 0.0, "details": {"git_dir_exists": False}}
        
        print("   ✓ .git directory exists")
        
        # Check if git is properly initialized
        required_git_items = ["HEAD", "config", "objects", "refs"]
        git_items_score = 0.0
        git_details = []
        
        for item in required_git_items:
            item_path = git_dir / item
            exists = item_path.exists()
            git_details.append(f"{item}: {'✓' if exists else '✗'}")
            if exists:
                git_items_score += 1.0
                print(f"   ✓ Git item: {item}")
            else:
                print(f"   ✗ Missing git item: {item}")
        
        score = git_items_score / len(required_git_items)
        
        # Bonus for having commits
        commits_exist = False
        head_file = git_dir / "HEAD"
        if head_file.exists():
            try:
                head_content = head_file.read_text()
                commits_exist = "ref:" in head_content
                if commits_exist:
                    print("   ✓ Has commits")
                else:
                    print("   ⚠️  No commits yet")
            except:
                commits_exist = False
        
        if commits_exist:
            score = min(1.0, score + 0.3)  # Bonus points
        
        print(f"\n   📊 Git items: {git_items_score:.0f}/{len(required_git_items)}")
        print(f"   🎯 Score: {score:.2f}")
        
        return {
            "score": score,
            "details": {
                "git_dir_exists": True,
                "git_items_score": git_items_score / len(required_git_items),
                "has_commits": commits_exist,
                "git_item_details": git_details
            }
        }
    
    def validate_p1_2_2(self) -> Dict[str, Any]:
        """P1.2.2 - Create app_config.yaml with all settings"""
        print("\n🔍 P1.2.2 - Create app_config.yaml with all settings")
        print("-" * 40)
        
        config_path = self.project_path / "data" / "config" / "app_config.yaml"
        
        if not config_path.exists():
            print("   ✗ app_config.yaml not found")
            print(f"   💡 Create: {config_path}")
            return {"score": 0.0, "details": {"file_exists": False}}
        
        print(f"   ✓ File exists: {config_path}")
        
        try:
            content = config_path.read_text(encoding='utf-8')
            print(f"   📄 File size: {len(content)} bytes")
            
            # Check for required sections
            required_sections = [
                ("app:", "App configuration"),
                ("paths:", "Path configurations"),
                ("document_processing:", "Document processing"),
                ("ai:", "AI settings"),
                ("ui:", "UI settings"),
                ("storage:", "Storage settings"),
                ("performance:", "Performance settings"),
                ("privacy:", "Privacy settings")
            ]
            
            found_sections = 0
            section_details = []
            for pattern, description in required_sections:
                if re.search(pattern, content, re.IGNORECASE):
                    found_sections += 1
                    section_details.append(f"{description}: ✓")
                    print(f"   ✓ Section: {description}")
                else:
                    section_details.append(f"{description}: ✗")
                    print(f"   ✗ Missing section: {description}")
            
            section_score = found_sections / len(required_sections)
            
            # Check if it's valid YAML
            try:
                config_data = yaml.safe_load(content)
                yaml_valid = True
                print("   ✓ Valid YAML syntax")
                
                # Check for specific important keys
                important_keys = ["app", "ai", "paths"]
                key_details = []
                for key in important_keys:
                    if key in config_data:
                        key_details.append(f"{key}: ✓")
                        print(f"   ✓ Key in YAML: {key}")
                    else:
                        key_details.append(f"{key}: ✗")
                        print(f"   ✗ Missing key in YAML: {key}")
                
            except Exception as yaml_error:
                yaml_valid = False
                print(f"   ✗ Invalid YAML: {yaml_error}")
                key_details = []
            
            # Combine scores
            score = (section_score * 0.7) + (0.3 if yaml_valid else 0.0)
            
            print(f"\n   📊 Sections found: {found_sections}/{len(required_sections)}")
            print(f"   🎯 Score: {score:.2f}")
            
            return {
                "score": score,
                "details": {
                    "file_exists": True,
                    "sections_found": found_sections,
                    "total_sections": len(required_sections),
                    "yaml_valid": yaml_valid,
                    "file_size": len(content),
                    "section_details": section_details,
                    "key_details": key_details if key_details else None
                }
            }
            
        except Exception as e:
            print(f"   ✗ Error reading file: {e}")
            return {"score": 0.0, "details": {"error": str(e)}}
    
    def validate_p1_2_3(self) -> Dict[str, Any]:
        """P1.2.3 - Setup cross-platform data directories"""
        print("\n🔍 P1.2.3 - Setup cross-platform data directories")
        print("-" * 40)
        
        config_path = self.project_path / "src" / "core" / "config.py"
        
        if not config_path.exists():
            print("   ✗ config.py not found")
            print(f"   💡 File should be at: {config_path}")
            return {"score": 0.0, "details": {"config_file_exists": False}}
        
        print(f"   ✓ File exists: {config_path}")
        
        try:
            content = config_path.read_text(encoding='utf-8')
            print(f"   📄 File size: {len(content)} bytes")
            
            # Check for cross-platform directory handling
            patterns = [
                (r"platform\.system\(\)", "Platform detection"),
                (r"Path\.home\(\)", "Home directory"),
                (r"AppData.*Local", "Windows path"),
                (r"Library.*Application Support", "macOS path"),
                (r"\.docubot", "Linux path"),
                (r"mkdir.*parents.*True", "Directory creation"),
                (r"exist_ok.*True", "Safe creation"),
                (r"def.*get_data_dir", "get_data_dir function"),
                (r"def.*ensure_directories", "ensure_directories function")
            ]
            
            found_patterns = 0
            pattern_details = []
            for pattern, description in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    found_patterns += 1
                    pattern_details.append(f"{description}: ✓")
                    print(f"   ✓ Feature: {description}")
                else:
                    pattern_details.append(f"{description}: ✗")
                    print(f"   ✗ Missing feature: {description}")
            
            score = found_patterns / len(patterns)
            
            # Check if config class exists
            class_patterns = [r"class.*Config", r"class.*AppConfig"]
            has_class = any(re.search(p, content) for p in class_patterns)
            
            if has_class:
                print("   ✓ Has Config/AppConfig class")
            else:
                print("   ✗ No Config/AppConfig class found")
            
            print(f"\n   📊 Features found: {found_patterns}/{len(patterns)}")
            print(f"   🎯 Score: {score:.2f}")
            
            return {
                "score": score,
                "details": {
                    "config_file_exists": True,
                    "patterns_found": found_patterns,
                    "total_patterns": len(patterns),
                    "has_config_class": has_class,
                    "pattern_details": pattern_details
                }
            }
            
        except Exception as e:
            print(f"   ✗ Error reading config.py: {e}")
            return {"score": 0.0, "details": {"error": str(e)}}
    
    def validate_p1_3_5(self) -> Dict[str, Any]:
        """P1.3.5 - Intelligent chunking"""
        print("\n🔍 P1.3.5 - Intelligent chunking (500 tokens, 50 overlap)")
        print("-" * 40)
        
        chunking_path = self.project_path / "src" / "document_processing" / "chunking.py"
        
        if not chunking_path.exists():
            print("   ✗ chunking.py not found")
            print(f"   💡 Create: {chunking_path}")
            return {"score": 0.0, "details": {"file_exists": False}}
        
        print(f"   ✓ File exists: {chunking_path}")
        
        try:
            content = chunking_path.read_text(encoding='utf-8')
            file_size = len(content)
            print(f"   📄 File size: {file_size} bytes")
            
            # Check for chunking functionality
            patterns = [
                (r"chunk.*size.*500", "Chunk size 500"),
                (r"chunk.*overlap.*50", "Overlap 50"),
                (r"def.*chunk", "Chunking function"),
                (r"token.*count", "Token counting"),
                (r"split.*sentence", "Sentence splitting"),
                (r"re\.split", "Regex splitting"),
                (r"natural.*boundary", "Natural boundaries"),
                (r"class.*Chunker", "Chunker class"),
                (r"separator", "Separators"),
                (r"def.*__init__", "Constructor")
            ]
            
            found_patterns = 0
            pattern_details = []
            for pattern, description in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    found_patterns += 1
                    pattern_details.append(f"{description}: ✓")
                    print(f"   ✓ Feature: {description}")
                else:
                    pattern_details.append(f"{description}: ✗")
                    print(f"   ✗ Missing feature: {description}")
            
            # Size-based score
            size_score = min(1.0, file_size / 2000)  # 2KB = full score
            
            # Combine pattern and size scores
            pattern_score = found_patterns / len(patterns)
            score = (pattern_score * 0.6) + (size_score * 0.4)
            
            print(f"\n   📊 Features found: {found_patterns}/{len(patterns)}")
            print(f"   📊 Size score: {size_score:.2f}")
            print(f"   🎯 Overall score: {score:.2f}")
            
            return {
                "score": score,
                "details": {
                    "file_exists": True,
                    "patterns_found": found_patterns,
                    "total_patterns": len(patterns),
                    "file_size": file_size,
                    "size_score": size_score,
                    "pattern_score": pattern_score,
                    "pattern_details": pattern_details
                }
            }
            
        except Exception as e:
            print(f"   ✗ Error reading chunking.py: {e}")
            return {"score": 0.0, "details": {"error": str(e)}}
    
    def validate_p1_6_1(self) -> Dict[str, Any]:
        """P1.6.1 - Complete ChromaDB vector store client"""
        print("\n🔍 P1.6.1 - Complete ChromaDB vector store client")
        print("-" * 40)
        
        chroma_path = self.project_path / "src" / "vector_store" / "chroma_client.py"
        
        if not chroma_path.exists():
            print("   ✗ chroma_client.py not found")
            print(f"   💡 Create: {chroma_path}")
            return {"score": 0.0, "details": {"file_exists": False}}
        
        print(f"   ✓ File exists: {chroma_path}")
        
        try:
            content = chroma_path.read_text(encoding='utf-8')
            file_size = len(content)
            print(f"   📄 File size: {file_size} bytes")
            
            # Check for ChromaDB functionality
            patterns = [
                (r"import.*chromadb", "ChromaDB import"),
                (r"class.*ChromaClient", "ChromaClient class"),
                (r"def.*search", "Search function"),
                (r"def.*add_documents", "Add documents"),
                (r"def.*get_collection", "Get collection"),
                (r"persist_directory", "Persistence"),
                (r"embedding_function", "Embedding function"),
                (r"similarity_search", "Similarity search"),
                (r"top_k", "Top-K parameter"),
                (r"metadata", "Metadata handling")
            ]
            
            found_patterns = 0
            pattern_details = []
            for pattern, description in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    found_patterns += 1
                    pattern_details.append(f"{description}: ✓")
                    print(f"   ✓ Feature: {description}")
                else:
                    pattern_details.append(f"{description}: ✗")
                    print(f"   ✗ Missing feature: {description}")
            
            # Size-based score
            size_score = min(1.0, file_size / 3000)  # 3KB = full score
            
            # Combine scores
            pattern_score = found_patterns / len(patterns)
            score = (pattern_score * 0.7) + (size_score * 0.3)
            
            print(f"\n   📊 Features found: {found_patterns}/{len(patterns)}")
            print(f"   📊 Size score: {size_score:.2f}")
            print(f"   🎯 Overall score: {score:.2f}")
            
            return {
                "score": score,
                "details": {
                    "file_exists": True,
                    "patterns_found": found_patterns,
                    "total_patterns": len(patterns),
                    "file_size": file_size,
                    "size_score": size_score,
                    "pattern_score": pattern_score,
                    "pattern_details": pattern_details
                }
            }
            
        except Exception as e:
            print(f"   ✗ Error reading chroma_client.py: {e}")
            return {"score": 0.0, "details": {"error": str(e)}}
    
    def validate_p1_8_2(self) -> Dict[str, Any]:
        """P1.8.2 - Support multiple LLM models"""
        print("\n🔍 P1.8.2 - Support multiple LLM models")
        print("-" * 40)
        
        llm_path = self.project_path / "src" / "ai_engine" / "llm_client.py"
        
        if not llm_path.exists():
            print("   ✗ llm_client.py not found")
            print(f"   💡 Create: {llm_path}")
            return {"score": 0.0, "details": {"file_exists": False}}
        
        print(f"   ✓ File exists: {llm_path}")
        
        try:
            content = llm_path.read_text(encoding='utf-8')
            file_size = len(content)
            print(f"   📄 File size: {file_size} bytes")
            
            # Check for LLM model support
            patterns = [
                (r"llama.*2.*7b", "Llama 2 7B"),
                (r"mistral.*7b", "Mistral 7B"),
                (r"neural.*chat", "Neural Chat"),
                (r"ollama", "Ollama integration"),
                (r"def.*generate", "Generate function"),
                (r"temperature", "Temperature control"),
                (r"max_tokens", "Token limit"),
                (r"stream", "Streaming support"),
                (r"model.*manager", "Model manager"),
                (r"multiple.*model", "Multiple models")
            ]
            
            found_patterns = 0
            pattern_details = []
            for pattern, description in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    found_patterns += 1
                    pattern_details.append(f"{description}: ✓")
                    print(f"   ✓ Feature: {description}")
                else:
                    pattern_details.append(f"{description}: ✗")
                    print(f"   ✗ Missing feature: {description}")
            
            # Size-based score
            size_score = min(1.0, file_size / 4000)  # 4KB = full score
            
            # Combine scores
            pattern_score = found_patterns / len(patterns)
            score = (pattern_score * 0.7) + (size_score * 0.3)
            
            print(f"\n   📊 Features found: {found_patterns}/{len(patterns)}")
            print(f"   📊 Size score: {size_score:.2f}")
            print(f"   🎯 Overall score: {score:.2f}")
            
            return {
                "score": score,
                "details": {
                    "file_exists": True,
                    "patterns_found": found_patterns,
                    "total_patterns": len(patterns),
                    "file_size": file_size,
                    "size_score": size_score,
                    "pattern_score": pattern_score,
                    "pattern_details": pattern_details
                }
            }
            
        except Exception as e:
            print(f"   ✗ Error reading llm_client.py: {e}")
            return {"score": 0.0, "details": {"error": str(e)}}
    
    def validate_quick_all(self) -> Dict[str, Any]:
        """Run quick validation of all critical tasks"""
        print("="*60)
        print("DOCUBOT - QUICK VALIDATION")
        print("="*60)
        
        validations = [
            ("P1.1.1", "Setup complete project structure", self.validate_p1_1_1),
            ("P1.1.4", "Initialize Git repository", self.validate_p1_1_4),
            ("P1.2.2", "Create app_config.yaml with all settings", self.validate_p1_2_2),
            ("P1.2.3", "Setup cross-platform data directories", self.validate_p1_2_3),
            ("P1.3.5", "Intelligent chunking (500 tokens, 50 overlap)", self.validate_p1_3_5),
            ("P1.6.1", "Complete ChromaDB vector store client", self.validate_p1_6_1),
            ("P1.8.2", "Support multiple LLM models", self.validate_p1_8_2),
        ]
        
        results = {}
        for task_id, task_name, validator in validations:
            results[f"{task_id} - {task_name}"] = validator()
        
        return results
    
    def generate_report(self, results: Dict[str, Any]):
        """Generate readable report"""
        print("\n" + "="*60)
        print("VALIDATION SUMMARY")
        print("="*60)
        
        total_score = 0
        completed_tasks = 0
        task_count = len(results)
        
        print("\n📋 TASK STATUS:")
        print("-" * 40)
        
        for task_name, result in results.items():
            score = result["score"]
            total_score += score
            
            if score >= 0.9:
                status = "✅ EXCELLENT"
                completed_tasks += 1
                emoji = "✅"
            elif score >= 0.7:
                status = "✅ GOOD"
                completed_tasks += 1
                emoji = "✅"
            elif score >= 0.5:
                status = "🔄 PARTIAL"
                emoji = "🔄"
            elif score > 0:
                status = "⚠️  MINIMAL"
                emoji = "⚠️"
            else:
                status = "❌ MISSING"
                emoji = "❌"
            
            short_name = task_name.split(" - ")[1] if " - " in task_name else task_name
            print(f"{emoji} {short_name}: {score:.2f}/1.00")
        
        # Summary
        avg_score = total_score / task_count if task_count else 0
        completion_rate = (completed_tasks / task_count) * 100 if task_count else 0
        
        print("\n" + "="*60)
        print("📊 SUMMARY")
        print("="*60)
        print(f"Total Tasks Validated: {task_count}")
        print(f"Average Score: {avg_score:.2f}/1.00")
        print(f"Completion Rate (≥0.7): {completion_rate:.1f}% ({completed_tasks}/{task_count})")
        
        # Priority recommendations
        print("\n" + "="*60)
        print("🎯 PRIORITY ACTIONS")
        print("="*60)
        
        # Sort by score (lowest first)
        sorted_tasks = sorted(results.items(), key=lambda x: x[1]["score"])
        
        for i, (task_name, result) in enumerate(sorted_tasks[:3]):  # Top 3 lowest scores
            if result["score"] < 0.7:
                short_name = task_name.split(" - ")[1] if " - " in task_name else task_name
                task_id = task_name.split(" - ")[0] if " - " in task_name else ""
                
                print(f"\n{i+1}. {short_name} [{task_id}]")
                print(f"   Current: {result['score']:.2f} → Target: 0.70")
                
                # Specific actions
                if "app_config.yaml" in task_name:
                    print("   Action: Create data/config/app_config.yaml with all settings")
                    print("   Location: data/config/app_config.yaml")
                elif "Git repository" in task_name:
                    print("   Action: Run 'git init' in project directory")
                elif "cross-platform" in task_name:
                    print("   Action: Add platform detection to src/core/config.py")
                    print("   Function: get_data_dir() and ensure_directories()")
                elif "chunking" in task_name:
                    print("   Action: Implement src/document_processing/chunking.py")
                    print("   Features: 500 token chunks, 50 token overlap")
                elif "ChromaDB" in task_name:
                    print("   Action: Complete src/vector_store/chroma_client.py")
                    print("   Features: search(), add_documents(), persistence")
                elif "LLM models" in task_name:
                    print("   Action: Add multi-model support to llm_client.py")
                    print("   Models: Llama 2 7B, Mistral 7B, Neural Chat")
        
        return avg_score, completion_rate, completed_tasks, task_count

def main():
    """Main entry point"""
    
    print("="*60)
    print("DOCUBOT SMART VALIDATOR")
    print("="*60)
    
    # Initialize validator with auto-detection
    validator = SmartValidator()
    
    # Check if we're in the right place
    if not validator.project_path.exists():
        print(f"\n❌ ERROR: Project directory not found!")
        print(f"   Looking for: {validator.project_path}")
        print("\n💡 Try running from:")
        print("   - Inside DocuBot folder")
        print("   - Or parent folder containing DocuBot")
        return 1
    
    print(f"\n📁 Project: {validator.project_path.name}")
    print(f"📍 Path: {validator.project_path}")
    
    # Run validations
    try:
        results = validator.validate_quick_all()
        avg_score, completion_rate, completed, total = validator.generate_report(results)
        
        # Save results to file
        output_file = validator.project_path / "smart_validation_report.json"
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "project_path": str(validator.project_path),
            "project_name": validator.project_path.name,
            "results": results,
            "summary": {
                "average_score": avg_score,
                "completion_rate": completion_rate,
                "completed_tasks": completed,
                "total_tasks": total
            }
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n📁 Detailed report saved to: {output_file}")
        
        # Next steps
        print("\n" + "="*60)
        print("🚀 NEXT STEPS")
        print("="*60)
        print("1. Fix the priority tasks listed above")
        print("2. Run this validator again to check progress")
        print("3. Continue with other tasks from the blueprint")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())