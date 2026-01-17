#docubot/11. check_gitignore.py
"""
Git Repository Validator - Check only, no actions
"""

import os
import sys
from pathlib import Path

def check_git_repository():
    """Validate Git repository structure."""
    
    print("=" * 70)
    print("GIT REPOSITORY VALIDATION REPORT")
    print("=" * 70)
    
    current_dir = Path.cwd()
    print(f"Current directory: {current_dir}")
    print()
    
    # ====================
    # CHECK 1: Git Repository
    # ====================
    print("1. GIT REPOSITORY CHECK")
    print("-" * 40)
    
    git_dir = current_dir / ".git"
    if git_dir.exists():
        print("   ✅ .git directory: FOUND")
        
        # Check essential Git files
        git_items = ['.git/HEAD', '.git/config', '.git/objects', '.git/refs']
        found_items = []
        
        for item in git_items:
            item_path = current_dir / item
            if item_path.exists() or (item.endswith('/') and item_path.is_dir()):
                found_items.append(item.split('/')[-1])
        
        print(f"   ✅ Git items found: {', '.join(found_items)}")
        
        # Check if repository has commits
        import subprocess
        try:
            result = subprocess.run(
                ['git', 'log', '--oneline', '-1'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0 and result.stdout.strip():
                print("   ✅ Repository has commits")
            else:
                print("   ⚠️  Repository has no commits")
        except:
            print("   ⚠️  Could not check commit history")
            
    else:
        print("   ❌ .git directory: NOT FOUND")
        print("   ℹ️  This directory is not a Git repository")
    
    print()
    
    # ====================
    # CHECK 2: .gitignore File
    # ====================
    print("2. .GITIGNORE CHECK")
    print("-" * 40)
    
    gitignore_path = current_dir / ".gitignore"
    
    if gitignore_path.exists():
        print(f"   ✅ .gitignore: FOUND at {gitignore_path}")
        
        # Check file size
        file_size = gitignore_path.stat().st_size
        print(f"   📄 File size: {file_size} bytes")
        
        # Read first few lines
        try:
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                print(f"   📝 Line count: {len(lines)}")
                
                # Check for common patterns
                content = ''.join(lines)
                checks = {
                    'Python patterns': '__pycache__' in content or '*.pyc' in content,
                    'Environment': '.env' in content or 'venv/' in content,
                    'IDE': '.vscode/' in content or '.idea/' in content,
                    'System files': '.DS_Store' in content or 'Thumbs.db' in content
                }
                
                print("   🔍 Contains patterns for:")
                for check_name, has_pattern in checks.items():
                    status = "✅" if has_pattern else "❌"
                    print(f"      {status} {check_name}")
                    
        except Exception as e:
            print(f"   ❌ Could not read .gitignore: {e}")
            
    else:
        print("   ❌ .gitignore: NOT FOUND")
        
        # Check parent directory
        parent_gitignore = current_dir.parent / ".gitignore"
        if parent_gitignore.exists():
            print(f"   ℹ️  Found .gitignore in parent directory: {parent_gitignore}")
    
    print()
    
    # ====================
    # CHECK 3: Repository Status
    # ====================
    print("3. REPOSITORY STATUS CHECK")
    print("-" * 40)
    
    if git_dir.exists():
        try:
            import subprocess
            
            # Branch info
            branch_result = subprocess.run(
                ['git', 'branch', '--show-current'],
                capture_output=True,
                text=True
            )
            if branch_result.returncode == 0:
                print(f"   🌿 Current branch: {branch_result.stdout.strip()}")
            
            # Status summary
            status_result = subprocess.run(
                ['git', 'status', '--porcelain'],
                capture_output=True,
                text=True
            )
            
            if status_result.returncode == 0:
                files = [line for line in status_result.stdout.strip().split('\n') if line]
                
                if files:
                    staged = len([f for f in files if f.startswith(('A ', 'M ', 'R ', 'C '))])
                    unstaged = len([f for f in files if f.startswith(' M', ' D', ' R')])
                    untracked = len([f for f in files if f.startswith('??')])
                    
                    print(f"   📊 File status:")
                    print(f"      • Staged/modified: {staged}")
                    print(f"      • Unstaged changes: {unstaged}")
                    print(f"      • Untracked files: {untracked}")
                    
                    if untracked > 10:
                        print(f"      ⚠️  Many untracked files ({untracked})")
                else:
                    print("   ✅ Working directory clean")
                    
        except Exception as e:
            print(f"   ⚠️  Could not check Git status: {e}")
    else:
        print("   ⚠️  Not a Git repository - skipping status check")
    
    print()
    
    # ====================
    # CHECK 4: Location Analysis
    # ====================
    print("4. LOCATION ANALYSIS")
    print("-" * 40)
    
    # Check if we're in the right place
    is_in_docubot = "DocuBot" in str(current_dir)
    has_parent_docubot = "DocuBot v.1" in str(current_dir.parent)
    
    print(f"   📍 In DocuBot folder: {'✅ YES' if is_in_docubot else '❌ NO'}")
    print(f"   📍 Parent is 'DocuBot v.1': {'✅ YES' if has_parent_docubot else '❌ NO'}")
    
    # Check for common project files
    expected_files = ['README.md', 'requirements.txt', 'src/', 'data/']
    print("   📁 Expected project files:")
    
    for file in expected_files:
        file_path = current_dir / file
        exists = file_path.exists()
        status = "✅" if exists else "❌"
        print(f"      {status} {file}")
    
    print()
    
    # ====================
    # SUMMARY
    # ====================
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    issues = []
    
    if not git_dir.exists():
        issues.append("Not a Git repository (.git folder missing)")
    
    gitignore_path = current_dir / ".gitignore"
    if not gitignore_path.exists():
        issues.append(".gitignore file not found in current directory")
    
    if issues:
        print("❌ ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"   {i}. {issue}")
        print()
        print("RECOMMENDATIONS:")
        print("1. Ensure you're in the correct directory (should be inside 'DocuBot' folder)")
        print("2. If .git is missing: This is not a Git repository")
        print("3. If .gitignore is missing: Consider creating one")
    else:
        print("✅ All basic checks passed")
        print("✅ Git repository structure appears valid")
        print()
        print("NEXT STEPS (optional):")
        print("• Run 'git status' to see detailed changes")
        print("• Run 'git log --oneline' to see commit history")
        print("• Review .gitignore patterns if needed")
    
    print("=" * 70)

if __name__ == "__main__":
    check_git_repository()
    
    print()
    print("NOTE: This script only performs validation checks.")
    print("No files were modified, added, committed, or pushed.")