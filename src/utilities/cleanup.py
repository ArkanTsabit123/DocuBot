"""
DocuBot Resource Cleanup System
Clean up temporary files and optimize resources
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
import sqlite3


class ResourceCleanup:
    """Manage resource cleanup and optimization"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or Path.home() / ".docubot"
    
    def cleanup_temporary_files(self, max_age_days: int = 7) -> List[Path]:
        """Clean up temporary files older than specified days"""
        temp_dirs = [
            self.data_dir / "tmp",
            self.data_dir / "cache" / "temp",
            self.data_dir / "documents" / "uploads" / "temp"
        ]
        
        cleaned = []
        cutoff = datetime.now() - timedelta(days=max_age_days)
        
        for temp_dir in temp_dirs:
            if temp_dir.exists():
                for file in temp_dir.rglob("*"):
                    if file.is_file():
                        try:
                            mtime = datetime.fromtimestamp(file.stat().st_mtime)
                            if mtime < cutoff:
                                file.unlink()
                                cleaned.append(file)
                        except:
                            pass
        
        return cleaned
    
    def cleanup_old_logs(self, max_age_days: int = 30) -> List[Path]:
        """Clean up old log files"""
        logs_dir = self.data_dir / "logs"
        
        if not logs_dir.exists():
            return []
        
        cleaned = []
        cutoff = datetime.now() - timedelta(days=max_age_days)
        
        for log_file in logs_dir.glob("*.log*"):
            if log_file.is_file():
                try:
                    mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if mtime < cutoff:
                        log_file.unlink()
                        cleaned.append(log_file)
                except:
                    pass
        
        return cleaned
    
    def cleanup_old_backups(self, max_age_days: int = 90, keep_minimum: int = 5) -> List[Path]:
        """Clean up old backup files"""
        backups_dir = self.data_dir / "backups"
        
        if not backups_dir.exists():
            return []
        
        backup_files = list(backups_dir.glob("*.tar.gz"))
        backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        if len(backup_files) <= keep_minimum:
            return []
        
        cleaned = []
        cutoff = datetime.now() - timedelta(days=max_age_days)
        
        for backup_file in backup_files[keep_minimum:]:
            try:
                mtime = datetime.fromtimestamp(backup_file.stat().st_mtime)
                if mtime < cutoff:
                    backup_file.unlink()
                    
                    metadata_file = backup_file.with_suffix('.json')
                    if metadata_file.exists():
                        metadata_file.unlink()
                    
                    cleaned.append(backup_file)
            except:
                pass
        
        return cleaned
    
    def optimize_databases(self) -> Dict[str, Any]:
        """Optimize database files"""
        results = {}
        
        db_files = [
            self.data_dir / "database" / "docubot.db",
            self.data_dir / "cache.db"
        ]
        
        for db_file in db_files:
            if db_file.exists():
                try:
                    before_size = db_file.stat().st_size
                    
                    conn = sqlite3.connect(db_file)
                    conn.execute("VACUUM")
                    conn.close()
                    
                    after_size = db_file.stat().st_size
                    
                    results[str(db_file)] = {
                        'before_size_mb': before_size / 1024 / 1024,
                        'after_size_mb': after_size / 1024 / 1024,
                        'reduction_percent': ((before_size - after_size) / before_size * 100) if before_size > 0 else 0
                    }
                except Exception as e:
                    results[str(db_file)] = {'error': str(e)}
        
        return results
    
    def cleanup_empty_directories(self) -> List[Path]:
        """Remove empty directories"""
        cleaned = []
        
        for root, dirs, files in os.walk(self.data_dir, topdown=False):
            for dir_name in dirs:
                dir_path = Path(root) / dir_name
                try:
                    if not any(dir_path.iterdir()):
                        dir_path.rmdir()
                        cleaned.append(dir_path)
                except:
                    pass
        
        return cleaned
    
    def get_storage_usage(self) -> Dict[str, float]:
        """Get storage usage statistics"""
        usage = {}
        
        directories = {
            'total': self.data_dir,
            'documents': self.data_dir / "documents",
            'database': self.data_dir / "database",
            'models': self.data_dir / "models",
            'logs': self.data_dir / "logs",
            'backups': self.data_dir / "backups",
            'cache': self.data_dir / "cache"
        }
        
        for name, directory in directories.items():
            if directory.exists():
                total_size = 0
                for file in directory.rglob("*"):
                    if file.is_file():
                        try:
                            total_size += file.stat().st_size
                        except:
                            pass
                
                usage[name] = total_size / 1024 / 1024  # Convert to MB
        
        return usage
    
    def run_complete_cleanup(self, interactive: bool = False) -> Dict[str, Any]:
        """Run complete cleanup routine"""
        results = {}
        
        if interactive:
            print("Running DocuBot Cleanup...")
            print("=" * 60)
        
        # Cleanup temporary files
        temp_files = self.cleanup_temporary_files()
        results['temp_files_cleaned'] = len(temp_files)
        
        if interactive:
            print(f"Cleaned {len(temp_files)} temporary files")
        
        # Cleanup old logs
        old_logs = self.cleanup_old_logs()
        results['old_logs_cleaned'] = len(old_logs)
        
        if interactive:
            print(f"Cleaned {len(old_logs)} old log files")
        
        # Cleanup old backups
        old_backups = self.cleanup_old_backups()
        results['old_backups_cleaned'] = len(old_backups)
        
        if interactive:
            print(f"Cleaned {len(old_backups)} old backups")
        
        # Optimize databases
        db_results = self.optimize_databases()
        results['database_optimization'] = db_results
        
        if interactive:
            for db, stats in db_results.items():
                if 'error' not in stats:
                    print(f"Optimized {Path(db).name}: {stats['reduction_percent']:.1f}% reduction")
        
        # Cleanup empty directories
        empty_dirs = self.cleanup_empty_directories()
        results['empty_directories_cleaned'] = len(empty_dirs)
        
        if interactive:
            print(f"Removed {len(empty_dirs)} empty directories")
        
        # Get storage usage
        storage_usage = self.get_storage_usage()
        results['storage_usage_mb'] = storage_usage
        
        if interactive:
            print("
Storage Usage:")
            for name, size_mb in storage_usage.items():
                print(f"  {name}: {size_mb:.1f} MB")
        
        if interactive:
            print("
" + "=" * 60)
            print("Cleanup completed successfully")
        
        return results


# Global cleanup instance
_resource_cleanup = None

def get_resource_cleanup() -> ResourceCleanup:
    """Get singleton resource cleanup instance"""
    global _resource_cleanup
    if _resource_cleanup is None:
        _resource_cleanup = ResourceCleanup()
    return _resource_cleanup


def cleanup_resources():
    """Convenience function for cleaning up resources"""
    cleaner = get_resource_cleanup()
    return cleaner.run_complete_cleanup()
