#!/usr/bin/env python3
"""
DocuBot Backup Utility
Comprehensive backup and restore functionality
"""

import os
import sys
import json
import shutil
import tarfile
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime
import hashlib
import argparse
import io


class DocuBotBackup:
    """Manage DocuBot backups"""
    
    def __init__(self, backup_dir: Optional[Path] = None):
        self.backup_dir = backup_dir or Path.home() / ".docubot" / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_dir = Path.home() / ".docubot"
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load backup configuration"""
        config_file = self.data_dir / "config" / "backup_config.json"
        
        if config_file.exists():
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            'include_databases': True,
            'include_documents': True,
            'include_models': False,
            'include_logs': False,
            'compression': 'gzip',
            'max_backups': 10,
            'encryption_enabled': False
        }
    
    def create_backup(
        self,
        name: Optional[str] = None,
        description: str = "",
        include_databases: Optional[bool] = None,
        include_documents: Optional[bool] = None,
        include_models: Optional[bool] = None,
        include_logs: Optional[bool] = None
    ) -> Path:
        """Create a new backup"""
        print("Creating DocuBot backup...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = name or f"backup_{timestamp}"
        backup_file = self.backup_dir / f"{backup_name}.tar.gz"
        
        include_databases = include_databases if include_databases is not None else self.config['include_databases']
        include_documents = include_documents if include_documents is not None else self.config['include_documents']
        include_models = include_models if include_models is not None else self.config['include_models']
        include_logs = include_logs if include_logs is not None else self.config['include_logs']
        
        metadata = {
            'name': backup_name,
            'description': description,
            'timestamp': datetime.now().isoformat(),
            'version': '1.0',
            'includes': {
                'databases': include_databases,
                'documents': include_documents,
                'models': include_models,
                'logs': include_logs
            },
            'size_bytes': 0,
            'checksum': None
        }
        
        with tarfile.open(backup_file, 'w:gz') as tar:
            # Add metadata
            metadata_str = json.dumps(metadata, indent=2)
            metadata_bytes = metadata_str.encode('utf-8')
            
            metadata_info = tarfile.TarInfo('METADATA.json')
            metadata_info.size = len(metadata_bytes)
            tar.addfile(metadata_info, io.BytesIO(metadata_bytes))
            
            # Add configuration
            config_dir = self.data_dir / "config"
            if config_dir.exists():
                tar.add(config_dir, arcname="config")
            
            # Add databases
            if include_databases:
                db_dir = self.data_dir / "database"
                if db_dir.exists():
                    tar.add(db_dir, arcname="database")
            
            # Add documents
            if include_documents:
                docs_dir = self.data_dir / "documents"
                if docs_dir.exists():
                    tar.add(docs_dir, arcname="documents")
            
            # Add models (optional - can be large)
            if include_models:
                models_dir = self.data_dir / "models"
                if models_dir.exists():
                    tar.add(models_dir, arcname="models")
            
            # Add logs (optional)
            if include_logs:
                logs_dir = self.data_dir / "logs"
                if logs_dir.exists():
                    tar.add(logs_dir, arcname="logs")
        
        # Calculate checksum
        backup_size = backup_file.stat().st_size
        checksum = self._calculate_checksum(backup_file)
        
        metadata['size_bytes'] = backup_size
        metadata['checksum'] = checksum
        
        # Update metadata in archive
        self._update_backup_metadata(backup_file, metadata)
        
        # Update backup index
        self._update_backup_index(backup_file, metadata)
        
        print(f"Backup created: {backup_file}")
        print(f"Size: {backup_size / 1024 / 1024:.2f} MB")
        print(f"Checksum: {checksum}")
        
        return backup_file
    
    def list_backups(self) -> List[Dict[str, Any]]:
        """List all available backups"""
        backups = []
        
        for backup_file in self.backup_dir.glob("*.tar.gz"):
            try:
                metadata = self._extract_metadata(backup_file)
                if metadata:
                    metadata['filename'] = backup_file.name
                    metadata['file_size'] = backup_file.stat().st_size
                    backups.append(metadata)
            except:
                continue
        
        return sorted(backups, key=lambda x: x['timestamp'], reverse=True)
    
    def restore_backup(
        self,
        backup_name: str,
        restore_databases: bool = True,
        restore_documents: bool = True,
        restore_config: bool = True,
        restore_models: bool = False,
        restore_logs: bool = False,
        dry_run: bool = False
    ) -> bool:
        """Restore from backup"""
        print(f"Restoring from backup: {backup_name}")
        
        backup_file = self.backup_dir / backup_name
        if not backup_file.exists():
            print(f"Error: Backup file not found: {backup_file}")
            return False
        
        try:
            metadata = self._extract_metadata(backup_file)
            if not metadata:
                print("Error: Could not read backup metadata")
                return False
            
            # Verify checksum
            expected_checksum = metadata.get('checksum')
            if expected_checksum:
                actual_checksum = self._calculate_checksum(backup_file)
                if expected_checksum != actual_checksum:
                    print(f"Error: Checksum mismatch. Expected: {expected_checksum}, Got: {actual_checksum}")
                    return False
            
            print(f"Backup Info: {metadata['name']} - {metadata['timestamp']}")
            
            if dry_run:
                print("Dry run mode - no files will be restored")
                return True
            
            # Extract backup
            with tarfile.open(backup_file, 'r:gz') as tar:
                members = tar.getmembers()
                
                for member in members:
                    if member.name == "METADATA.json":
                        continue
                    
                    # Determine what to restore based on path and user selection
                    if member.name.startswith("config/") and restore_config:
                        tar.extract(member, self.data_dir)
                    elif member.name.startswith("database/") and restore_databases:
                        tar.extract(member, self.data_dir)
                    elif member.name.startswith("documents/") and restore_documents:
                        tar.extract(member, self.data_dir)
                    elif member.name.startswith("models/") and restore_models:
                        tar.extract(member, self.data_dir)
                    elif member.name.startswith("logs/") and restore_logs:
                        tar.extract(member, self.data_dir)
            
            print("Restore completed successfully")
            return True
            
        except Exception as e:
            print(f"Error during restore: {e}")
            return False
    
    def delete_backup(self, backup_name: str) -> bool:
        """Delete a backup"""
        backup_file = self.backup_dir / backup_name
        
        if backup_file.exists():
            try:
                backup_file.unlink()
                
                # Update index
                index_file = self.backup_dir / "backup_index.json"
                if index_file.exists():
                    with open(index_file, 'r', encoding='utf-8') as f:
                        index = json.load(f)
                    
                    index['backups'] = [b for b in index['backups'] if b['filename'] != backup_name]
                    
                    with open(index_file, 'w', encoding='utf-8') as f:
                        json.dump(index, f, indent=2)
                
                print(f"Backup deleted: {backup_name}")
                return True
            except Exception as e:
                print(f"Error deleting backup: {e}")
                return False
        else:
            print(f"Backup not found: {backup_name}")
            return False
    
    def cleanup_old_backups(self, keep_last: int = 10) -> List[str]:
        """Clean up old backups, keeping only specified number"""
        backups = self.list_backups()
        
        if len(backups) <= keep_last:
            print(f"Keeping all {len(backups)} backups")
            return []
        
        to_delete = backups[keep_last:]
        deleted = []
        
        for backup in to_delete:
            if self.delete_backup(backup['filename']):
                deleted.append(backup['filename'])
        
        print(f"Deleted {len(deleted)} old backups")
        return deleted
    
    def verify_backup(self, backup_name: str) -> bool:
        """Verify backup integrity"""
        backup_file = self.backup_dir / backup_name
        
        if not backup_file.exists():
            print(f"Error: Backup file not found: {backup_file}")
            return False
        
        try:
            metadata = self._extract_metadata(backup_file)
            if not metadata:
                print("Error: Could not read backup metadata")
                return False
            
            expected_checksum = metadata.get('checksum')
            if not expected_checksum:
                print("Warning: No checksum in metadata")
                return True
            
            actual_checksum = self._calculate_checksum(backup_file)
            
            if expected_checksum == actual_checksum:
                print(f"Backup verified successfully: {backup_name}")
                return True
            else:
                print(f"Error: Checksum mismatch. Expected: {expected_checksum}, Got: {actual_checksum}")
                return False
            
        except Exception as e:
            print(f"Error verifying backup: {e}")
            return False
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of file"""
        sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        
        return sha256.hexdigest()
    
    def _extract_metadata(self, backup_file: Path) -> Optional[Dict[str, Any]]:
        """Extract metadata from backup file"""
        try:
            with tarfile.open(backup_file, 'r:gz') as tar:
                metadata_member = tar.getmember('METADATA.json')
                metadata_file = tar.extractfile(metadata_member)
                
                if metadata_file:
                    metadata_bytes = metadata_file.read()
                    return json.loads(metadata_bytes.decode('utf-8'))
        except:
            return None
    
    def _update_backup_metadata(self, backup_file: Path, metadata: Dict[str, Any]):
        """Update metadata in backup file"""
        # This is complex with tar files, so we'll store metadata separately
        metadata_file = backup_file.with_suffix('.json')
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    
    def _update_backup_index(self, backup_file: Path, metadata: Dict[str, Any]):
        """Update backup index file"""
        index_file = self.backup_dir / "backup_index.json"
        
        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
        else:
            index = {'backups': []}
        
        index_entry = {
            'filename': backup_file.name,
            'name': metadata['name'],
            'timestamp': metadata['timestamp'],
            'size_bytes': metadata['size_bytes'],
            'checksum': metadata['checksum'],
            'description': metadata.get('description', '')
        }
        
        # Remove existing entry if present
        index['backups'] = [b for b in index['backups'] if b['filename'] != backup_file.name]
        index['backups'].append(index_entry)
        
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2)
    
    def export_backup_info(self, output_file: Path):
        """Export backup information to file"""
        backups = self.list_backups()
        
        info = {
            'timestamp': datetime.now().isoformat(),
            'backup_dir': str(self.backup_dir),
            'total_backups': len(backups),
            'total_size_gb': sum(b['file_size'] for b in backups) / 1024 / 1024 / 1024,
            'backups': backups
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2)
        
        print(f"Backup info exported to: {output_file}")


def main():
    """Main backup utility entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="DocuBot Backup Utility")
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Create backup command
    create_parser = subparsers.add_parser('create', help='Create new backup')
    create_parser.add_argument('--name', '-n', help='Backup name')
    create_parser.add_argument('--description', '-d', default='', help='Backup description')
    create_parser.add_argument('--no-databases', action='store_true', help='Exclude databases')
    create_parser.add_argument('--no-documents', action='store_true', help='Exclude documents')
    create_parser.add_argument('--include-models', action='store_true', help='Include AI models')
    create_parser.add_argument('--include-logs', action='store_true', help='Include log files')
    
    # List backups command
    list_parser = subparsers.add_parser('list', help='List available backups')
    
    # Restore backup command
    restore_parser = subparsers.add_parser('restore', help='Restore from backup')
    restore_parser.add_argument('backup', help='Backup filename to restore')
    restore_parser.add_argument('--no-databases', action='store_true', help='Do not restore databases')
    restore_parser.add_argument('--no-documents', action='store_true', help='Do not restore documents')
    restore_parser.add_argument('--no-config', action='store_true', help='Do not restore configuration')
    restore_parser.add_argument('--dry-run', action='store_true', help='Dry run mode')
    
    # Delete backup command
    delete_parser = subparsers.add_parser('delete', help='Delete backup')
    delete_parser.add_argument('backup', help='Backup filename to delete')
    
    # Verify backup command
    verify_parser = subparsers.add_parser('verify', help='Verify backup integrity')
    verify_parser.add_argument('backup', help='Backup filename to verify')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old backups')
    cleanup_parser.add_argument('--keep', '-k', type=int, default=10, help='Number of backups to keep')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export backup information')
    export_parser.add_argument('--output', '-o', type=Path, required=True, help='Output JSON file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    backup_manager = DocuBotBackup()
    
    try:
        if args.command == 'create':
            backup_file = backup_manager.create_backup(
                name=args.name,
                description=args.description,
                include_databases=not args.no_databases,
                include_documents=not args.no_documents,
                include_models=args.include_models,
                include_logs=args.include_logs
            )
            
        elif args.command == 'list':
            backups = backup_manager.list_backups()
            
            if not backups:
                print("No backups found")
            else:
                print(f"
Found {len(backups)} backups:
")
                for i, backup in enumerate(backups, 1):
                    size_mb = backup['file_size'] / 1024 / 1024
                    date = datetime.fromisoformat(backup['timestamp']).strftime("%Y-%m-%d %H:%M")
                    print(f"{i:2}. {backup['filename']}")
                    print(f"     Name: {backup['name']}")
                    print(f"     Date: {date}")
                    print(f"     Size: {size_mb:.1f} MB")
                    print(f"     Desc: {backup.get('description', '')}")
                    print()
        
        elif args.command == 'restore':
            success = backup_manager.restore_backup(
                backup_name=args.backup,
                restore_databases=not args.no_databases,
                restore_documents=not args.no_documents,
                restore_config=not args.no_config,
                dry_run=args.dry_run
            )
            
            if not success:
                sys.exit(1)
        
        elif args.command == 'delete':
            success = backup_manager.delete_backup(args.backup)
            
            if not success:
                sys.exit(1)
        
        elif args.command == 'verify':
            success = backup_manager.verify_backup(args.backup)
            
            if not success:
                sys.exit(1)
        
        elif args.command == 'cleanup':
            deleted = backup_manager.cleanup_old_backups(keep_last=args.keep)
            
            if deleted:
                print(f"Deleted backups: {', '.join(deleted)}")
        
        elif args.command == 'export':
            backup_manager.export_backup_info(args.output)
        
        print("Operation completed successfully")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
