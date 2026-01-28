# docubot/scripts/init_db.py

"""
Database Initialization and Management Script for DocuBot

This script provides comprehensive database initialization, schema management,
validation, and maintenance operations for DocuBot's SQLite database.
"""

import os
import sys
import sqlite3
import hashlib
import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta
import logging
import argparse


class DatabaseInitializationError(Exception):
    """Custom exception for database initialization failures."""
    pass


class DatabaseValidationError(Exception):
    """Custom exception for database validation failures."""
    pass


class DatabaseInitializer:
    """
    Comprehensive database initialization and management for DocuBot.
    
    This class handles schema creation, validation, migrations, backups,
    and maintenance operations for the SQLite database.
    """
    
    SCHEMA_VERSION = 1
    
    REQUIRED_TABLES = {
        "documents": 19,
        "chunks": 10,
        "conversations": 8,
        "messages": 9,
        "tags": 6,
        "document_tags": 3,
        "settings": 3
    }
    
    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the database manager.
        
        Args:
            db_path: Path to SQLite database file. If None, uses default location.
        
        Raises:
            DatabaseInitializationError: If database path cannot be determined.
        """
        self._setup_logging()
        
        try:
            if db_path is None:
                self.db_path = self._get_default_db_path()
            else:
                self.db_path = Path(db_path)
            
            self.connection = None
            self.cursor = None
            
            self.logger.info(f"Initialized database manager for: {self.db_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize database manager: {e}")
            raise DatabaseInitializationError(f"Database initialization failed: {e}")
    
    def _setup_logging(self):
        """Configure logging for the database module."""
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def _get_default_db_path(self) -> Path:
        """
        Determine the default database path based on platform.
        
        Returns:
            Path to default database location.
        
        Raises:
            DatabaseInitializationError: If default path cannot be determined.
        """
        try:
            project_root = Path(__file__).parent.parent
            data_dir = project_root / "data" / "database"
            data_dir.mkdir(parents=True, exist_ok=True)
            return data_dir / "docubot.db"
        except Exception as e:
            raise DatabaseInitializationError(f"Cannot determine default database path: {e}")
    
    def _establish_connection(self) -> bool:
        """
        Establish database connection with optimized settings.
        
        Returns:
            True if connection successful, False otherwise.
        """
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.connection.execute("PRAGMA foreign_keys = ON")
            self.connection.execute("PRAGMA journal_mode = WAL")
            self.connection.execute("PRAGMA synchronous = NORMAL")
            self.connection.execute("PRAGMA cache_size = -10000")
            self.connection.execute("PRAGMA busy_timeout = 5000")
            self.cursor = self.connection.cursor()
            
            self.logger.info(f"Connected to database: {self.db_path}")
            return True
            
        except sqlite3.Error as e:
            self.logger.error(f"Database connection failed: {e}")
            return False
    
    def _close_connection(self):
        """Safely close database connection."""
        if self.connection:
            try:
                self.connection.commit()
                self.connection.close()
                self.logger.debug("Database connection closed")
            except sqlite3.Error as e:
                self.logger.warning(f"Error closing database connection: {e}")
            finally:
                self.connection = None
                self.cursor = None
    
    def _execute_transaction(self, queries: List[str]) -> bool:
        """
        Execute multiple SQL queries in a single transaction.
        
        Args:
            queries: List of SQL queries to execute.
        
        Returns:
            True if all queries executed successfully, False otherwise.
        """
        try:
            for query in queries:
                self.cursor.execute(query)
            self.connection.commit()
            return True
        except sqlite3.Error as e:
            self.logger.error(f"Transaction failed: {e}")
            self.connection.rollback()
            return False
    
    def get_complete_schema(self) -> Dict[str, List[str]]:
        """
        Return complete database schema definition.
        
        Returns:
            Dictionary with table creation SQL and index creation SQL.
        """
        schema = {
            "tables": [
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    file_name TEXT NOT NULL,
                    file_type TEXT NOT NULL,
                    file_size INTEGER,
                    upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    processing_status TEXT DEFAULT 'pending',
                    processing_error TEXT,
                    metadata_json TEXT,
                    vector_ids_json TEXT,
                    chunk_count INTEGER DEFAULT 0,
                    word_count INTEGER DEFAULT 0,
                    language TEXT,
                    tags_json TEXT,
                    summary TEXT,
                    is_indexed BOOLEAN DEFAULT FALSE,
                    indexed_at TIMESTAMP,
                    last_accessed TIMESTAMP,
                    access_count INTEGER DEFAULT 0
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    id TEXT PRIMARY KEY,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    text_content TEXT NOT NULL,
                    cleaned_text TEXT NOT NULL,
                    token_count INTEGER,
                    embedding_model TEXT,
                    vector_id TEXT NOT NULL,
                    metadata_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    message_count INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    tags_json TEXT,
                    is_archived BOOLEAN DEFAULT FALSE,
                    export_path TEXT
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    tokens INTEGER,
                    model_used TEXT,
                    sources_json TEXT,
                    processing_time_ms INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS tags (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL UNIQUE,
                    color TEXT,
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    usage_count INTEGER DEFAULT 0
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS document_tags (
                    document_id TEXT NOT NULL,
                    tag_id TEXT NOT NULL,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (document_id, tag_id),
                    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE,
                    FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
                )
                """,
                """
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            ],
            "indexes": [
                "CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(processing_status)",
                "CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(file_type)",
                "CREATE INDEX IF NOT EXISTS idx_documents_is_indexed ON documents(is_indexed)",
                "CREATE INDEX IF NOT EXISTS idx_documents_upload_date ON documents(upload_date)",
                "CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id)",
                "CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id)",
                "CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at)",
                "CREATE INDEX IF NOT EXISTS idx_tags_name ON tags(name)",
                "CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at)"
            ]
        }
        
        return schema
    
    def get_default_settings(self) -> List[Tuple[str, str]]:
        """
        Return default application settings.
        
        Returns:
            List of (key, value) tuples for default settings.
        """
        current_time = datetime.now().isoformat()
        
        return [
            ("app_version", "1.0.0"),
            ("database_version", str(self.SCHEMA_VERSION)),
            ("database_created", current_time),
            ("database_last_updated", current_time),
            ("chunk_size", "500"),
            ("chunk_overlap", "50"),
            ("max_file_size_mb", "100"),
            ("default_llm_model", "llama2:7b"),
            ("default_embedding_model", "all-MiniLM-L6-v2"),
            ("rag_top_k", "5"),
            ("rag_similarity_threshold", "0.7"),
            ("ui_theme", "dark"),
            ("ui_language", "en"),
            ("auto_backup_enabled", "true"),
            ("backup_interval_hours", "24"),
            ("max_documents", "10000"),
            ("auto_cleanup_days", "90"),
            ("telemetry_enabled", "false"),
            ("auto_update_check", "false"),
            ("crash_reports_enabled", "false")
        ]
    
    def initialize_database(self, force_recreate: bool = False) -> Dict[str, Any]:
        """
        Initialize or recreate the complete database.
        
        Args:
            force_recreate: If True, recreate database even if it exists.
        
        Returns:
            Dictionary with initialization results and statistics.
        
        Raises:
            DatabaseInitializationError: If initialization fails.
        """
        result = {
            "operation": "database_initialization",
            "timestamp": datetime.now().isoformat(),
            "database_path": str(self.db_path),
            "force_recreate": force_recreate,
            "success": False,
            "statistics": {},
            "errors": [],
            "warnings": []
        }
        
        try:
            database_exists = self.db_path.exists()
            
            if database_exists and not force_recreate:
                result["message"] = "Database already exists. Use --force to recreate."
                result["warnings"].append("Database exists but force flag not set")
                return result
            
            if database_exists and force_recreate:
                backup_result = self.create_backup("pre_recreation")
                if backup_result["success"]:
                    result["backup_created"] = backup_result["backup_path"]
                    self.logger.info(f"Created backup before recreation: {backup_result['backup_path']}")
                else:
                    result["warnings"].append(f"Failed to create backup before recreation: {backup_result.get('errors', ['Unknown error'])}")
            
            if not self._establish_connection():
                raise DatabaseInitializationError("Cannot establish database connection")
            
            schema = self.get_complete_schema()
            
            self.logger.info("Creating database schema...")
            
            tables_created = 0
            for table_sql in schema["tables"]:
                try:
                    self.cursor.execute(table_sql)
                    tables_created += 1
                except sqlite3.Error as e:
                    raise DatabaseInitializationError(f"Failed to create table: {e}")
            
            indexes_created = 0
            for index_sql in schema["indexes"]:
                try:
                    self.cursor.execute(index_sql)
                    indexes_created += 1
                except sqlite3.Error as e:
                    self.logger.warning(f"Failed to create index: {e}")
                    result["warnings"].append(f"Index creation failed: {e}")
            
            self._insert_default_settings()
            
            validation_result = self.validate_database()
            
            if not validation_result["is_valid"]:
                result["errors"].extend(validation_result["errors"])
                result["warnings"].extend(validation_result.get("warnings", []))
                
                if validation_result.get("has_errors", True):
                    raise DatabaseInitializationError(f"Database validation failed: {validation_result['errors']}")
                else:
                    self.logger.warning(f"Database validation has warnings: {validation_result['warnings']}")
            
            result["statistics"].update({
                "tables_created": tables_created,
                "indexes_created": indexes_created,
                "schema_version": self.SCHEMA_VERSION,
                "initialization_time": datetime.now().isoformat()
            })
            
            result["success"] = True
            result["message"] = "Database initialized successfully"
            
            self.logger.info(f"Database initialized: {tables_created} tables, {indexes_created} indexes")
            
            return result
            
        except Exception as e:
            result["errors"].append(str(e))
            self.logger.error(f"Database initialization failed: {e}")
            raise
            
        finally:
            self._close_connection()
    
    def _insert_default_settings(self):
        """Insert default application settings into the database."""
        try:
            settings = self.get_default_settings()
            for key, value in settings:
                self.cursor.execute(
                    "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
                    (key, value, datetime.now().isoformat())
                )
            self.logger.info(f"Inserted {len(settings)} default settings")
        except sqlite3.Error as e:
            self.logger.error(f"Failed to insert default settings: {e}")
            raise
    
    def validate_database(self) -> Dict[str, Any]:
        """
        Perform comprehensive database validation.
        
        Returns:
            Dictionary with validation results including schema integrity,
            table structure, and data consistency checks.
        """
        validation_result = {
            "is_valid": False,
            "timestamp": datetime.now().isoformat(),
            "database_path": str(self.db_path),
            "schema_version": None,
            "table_validation": {},
            "foreign_key_validation": {},
            "index_validation": {},
            "data_integrity": {},
            "errors": [],
            "warnings": [],
            "has_errors": False,
            "has_warnings": False
        }
        
        if not self.db_path.exists():
            validation_result["errors"].append(f"Database file not found: {self.db_path}")
            validation_result["has_errors"] = True
            return validation_result
        
        try:
            if not self._establish_connection():
                validation_result["errors"].append("Cannot connect to database")
                validation_result["has_errors"] = True
                return validation_result
            
            schema_version = self._get_schema_version()
            validation_result["schema_version"] = schema_version
            
            table_results = self._validate_tables()
            validation_result["table_validation"] = table_results
            
            foreign_key_results = self._validate_foreign_keys()
            validation_result["foreign_key_validation"] = foreign_key_results
            
            index_results = self._validate_indexes()
            validation_result["index_validation"] = index_results
            
            data_results = self._validate_data_integrity()
            validation_result["data_integrity"] = data_results
            
            table_has_errors = table_results.get("has_errors", False)
            foreign_key_has_errors = foreign_key_results.get("has_errors", False)
            data_has_errors = data_results.get("has_errors", False)
            
            validation_result["has_errors"] = (
                table_has_errors or 
                foreign_key_has_errors or 
                data_has_errors
            )
            
            table_has_warnings = table_results.get("has_warnings", False)
            foreign_key_has_warnings = foreign_key_results.get("has_warnings", False)
            index_has_warnings = index_results.get("has_warnings", False)
            data_has_warnings = data_results.get("has_warnings", False)
            
            validation_result["has_warnings"] = (
                table_has_warnings or 
                foreign_key_has_warnings or 
                index_has_warnings or 
                data_has_warnings
            )
            
            validation_result["is_valid"] = not validation_result["has_errors"]
            
            if validation_result["has_errors"]:
                validation_result["errors"].append("Database validation failed")
            elif validation_result["has_warnings"]:
                validation_result["warnings"].append("Database has validation warnings")
            
            return validation_result
            
        except Exception as e:
            validation_result["errors"].append(f"Validation error: {e}")
            validation_result["has_errors"] = True
            return validation_result
            
        finally:
            self._close_connection()
    
    def _get_schema_version(self) -> Optional[int]:
        """Retrieve current schema version from database."""
        try:
            self.cursor.execute("SELECT value FROM settings WHERE key = 'database_version'")
            result = self.cursor.fetchone()
            if result:
                return int(result[0])
        except (sqlite3.Error, ValueError):
            pass
        return None
    
    def _validate_tables(self) -> Dict[str, Any]:
        """Validate all required tables exist and have correct structure."""
        result = {
            "required_tables": len(self.REQUIRED_TABLES),
            "tables_found": 0,
            "tables_valid": 0,
            "table_details": {},
            "has_errors": False,
            "has_warnings": False
        }
        
        try:
            self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            existing_tables = {row[0] for row in self.cursor.fetchall()}
            
            for table_name, expected_columns in self.REQUIRED_TABLES.items():
                table_info = {
                    "exists": table_name in existing_tables,
                    "expected_columns": expected_columns,
                    "actual_columns": 0,
                    "columns": [],
                    "is_valid": False,
                    "errors": [],
                    "warnings": []
                }
                
                if table_info["exists"]:
                    result["tables_found"] += 1
                    
                    try:
                        self.cursor.execute(f"PRAGMA table_info({table_name})")
                        columns = self.cursor.fetchall()
                        table_info["actual_columns"] = len(columns)
                        table_info["columns"] = [col[1] for col in columns]
                        
                        column_diff = abs(table_info["actual_columns"] - expected_columns)
                        if column_diff <= 2:
                            table_info["is_valid"] = True
                            result["tables_valid"] += 1
                            
                            if column_diff > 0:
                                table_info["warnings"].append(
                                    f"Column count mismatch: expected {expected_columns}, got {table_info['actual_columns']}"
                                )
                                result["has_warnings"] = True
                        else:
                            table_info["errors"].append(
                                f"Column count mismatch: expected {expected_columns}, got {table_info['actual_columns']}"
                            )
                            result["has_errors"] = True
                            
                    except sqlite3.Error as e:
                        table_info["errors"].append(f"Error checking table structure: {e}")
                        result["has_errors"] = True
                else:
                    table_info["errors"].append(f"Table '{table_name}' not found")
                    result["has_errors"] = True
                
                result["table_details"][table_name] = table_info
            
            return result
            
        except sqlite3.Error as e:
            result["has_errors"] = True
            result["error"] = str(e)
            return result
    
    def _validate_foreign_keys(self) -> Dict[str, Any]:
        """Validate foreign key constraints."""
        result = {
            "foreign_keys_enabled": False,
            "constraints_valid": False,
            "constraint_details": [],
            "has_errors": False,
            "has_warnings": False
        }
        
        try:
            self.cursor.execute("PRAGMA foreign_keys")
            foreign_keys_enabled = self.cursor.fetchone()[0] == 1
            result["foreign_keys_enabled"] = foreign_keys_enabled
            
            if not foreign_keys_enabled:
                result["warnings"].append("Foreign keys are not enabled")
                result["has_warnings"] = True
                return result
            
            self.cursor.execute("PRAGMA foreign_key_check")
            violations = self.cursor.fetchall()
            
            if violations:
                for violation in violations:
                    constraint_info = {
                        "table": violation[0],
                        "rowid": violation[1],
                        "referenced_table": violation[2],
                        "foreign_key_index": violation[3]
                    }
                    result["constraint_details"].append(constraint_info)
                    result["has_errors"] = True
                
                result["constraints_valid"] = False
            else:
                result["constraints_valid"] = True
            
            return result
            
        except sqlite3.Error as e:
            result["has_errors"] = True
            result["error"] = str(e)
            return result
    
    def _validate_indexes(self) -> Dict[str, Any]:
        """Validate database indexes."""
        result = {
            "indexes_found": 0,
            "index_details": {},
            "has_warnings": False
        }
        
        try:
            self.cursor.execute("SELECT name, tbl_name, sql FROM sqlite_master WHERE type='index'")
            indexes = self.cursor.fetchall()
            result["indexes_found"] = len(indexes)
            
            for idx_name, table_name, idx_sql in indexes:
                result["index_details"][idx_name] = {
                    "table": table_name,
                    "sql": idx_sql
                }
            
            return result
            
        except sqlite3.Error as e:
            result["has_warnings"] = True
            result["warning"] = str(e)
            return result
    
    def _validate_data_integrity(self) -> Dict[str, Any]:
        """Validate data integrity and consistency."""
        result = {
            "integrity_check": False,
            "row_counts": {},
            "orphaned_records": {},
            "has_errors": False,
            "has_warnings": False
        }
        
        try:
            self.cursor.execute("PRAGMA integrity_check")
            integrity_result = self.cursor.fetchone()
            result["integrity_check"] = integrity_result[0] == "ok"
            
            if not result["integrity_check"]:
                result["has_errors"] = True
                return result
            
            for table_name in self.REQUIRED_TABLES.keys():
                try:
                    self.cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    row_count = self.cursor.fetchone()[0]
                    result["row_counts"][table_name] = row_count
                except sqlite3.Error:
                    pass
            
            result["orphaned_records"] = self._check_orphaned_records()
            
            if any(result["orphaned_records"].values()):
                result["has_warnings"] = True
            
            return result
            
        except sqlite3.Error as e:
            result["has_errors"] = True
            result["error"] = str(e)
            return result
    
    def _check_orphaned_records(self) -> Dict[str, int]:
        """Check for orphaned records in foreign key relationships."""
        orphaned_counts = {}
        
        checks = [
            ("chunks", "document_id", "documents", "id"),
            ("messages", "conversation_id", "conversations", "id"),
            ("document_tags", "document_id", "documents", "id"),
            ("document_tags", "tag_id", "tags", "id")
        ]
        
        for child_table, child_column, parent_table, parent_column in checks:
            try:
                query = f"""
                SELECT COUNT(*) FROM {child_table} c
                LEFT JOIN {parent_table} p ON c.{child_column} = p.{parent_column}
                WHERE p.{parent_column} IS NULL
                """
                self.cursor.execute(query)
                count = self.cursor.fetchone()[0]
                if count > 0:
                    orphaned_counts[f"{child_table}.{child_column}"] = count
            except sqlite3.Error:
                continue
        
        return orphaned_counts
    
    def create_backup(self, reason: str = "scheduled") -> Dict[str, Any]:
        """
        Create a database backup with metadata.
        
        Args:
            reason: Reason for backup creation.
        
        Returns:
            Dictionary with backup results.
        """
        result = {
            "success": False,
            "timestamp": datetime.now().isoformat(),
            "reason": reason,
            "backup_path": None,
            "original_size": 0,
            "backup_size": 0,
            "checksum": None,
            "errors": []
        }
        
        if not self.db_path.exists():
            result["errors"].append(f"Database file not found: {self.db_path}")
            return result
        
        try:
            backup_dir = self.db_path.parent / "backups"
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"docubot_backup_{timestamp}_{reason}.db"
            backup_path = backup_dir / backup_filename
            
            original_size = self.db_path.stat().st_size
            result["original_size"] = original_size
            
            backup_conn = sqlite3.connect(backup_path)
            
            if not self._establish_connection():
                result["errors"].append("Cannot establish database connection")
                backup_conn.close()
                return result
            
            try:
                self.connection.backup(backup_conn)
            finally:
                self._close_connection()
            
            backup_conn.close()
            
            backup_size = backup_path.stat().st_size
            result["backup_size"] = backup_size
            
            with open(backup_path, 'rb') as f:
                file_hash = hashlib.sha256()
                chunk = f.read(8192)
                while chunk:
                    file_hash.update(chunk)
                    chunk = f.read(8192)
                result["checksum"] = file_hash.hexdigest()
            
            backup_metadata = {
                "backup_timestamp": datetime.now().isoformat(),
                "original_database": str(self.db_path),
                "original_size": original_size,
                "backup_size": backup_size,
                "checksum": result["checksum"],
                "reason": reason,
                "schema_version": self.SCHEMA_VERSION
            }
            
            metadata_path = backup_path.with_suffix('.json')
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(backup_metadata, f, indent=2)
            
            result["backup_path"] = str(backup_path)
            result["success"] = True
            
            self.logger.info(f"Backup created: {backup_path} ({backup_size:,} bytes)")
            
            return result
            
        except Exception as e:
            result["errors"].append(str(e))
            self.logger.error(f"Backup creation failed: {e}")
            return result
    
    def get_database_statistics(self) -> Dict[str, Any]:
        """
        Retrieve comprehensive database statistics.
        
        Returns:
            Dictionary with database statistics and metrics.
        """
        stats = {
            "timestamp": datetime.now().isoformat(),
            "database_path": str(self.db_path),
            "file_info": {},
            "table_statistics": {},
            "performance_metrics": {},
            "schema_info": {}
        }
        
        if not self.db_path.exists():
            return stats
        
        try:
            if not self._establish_connection():
                return stats
            
            stats["file_info"] = {
                "exists": True,
                "size_bytes": self.db_path.stat().st_size,
                "modified_time": datetime.fromtimestamp(self.db_path.stat().st_mtime).isoformat()
            }
            
            self.cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in self.cursor.fetchall()]
            
            for table in tables:
                table_stats = {}
                
                try:
                    self.cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    table_stats["row_count"] = self.cursor.fetchone()[0]
                    
                    self.cursor.execute(f"PRAGMA table_info({table})")
                    columns = self.cursor.fetchall()
                    table_stats["column_count"] = len(columns)
                    
                    if table_stats["row_count"] > 0:
                        self.cursor.execute(f"SELECT MIN(rowid), MAX(rowid) FROM {table}")
                        min_max = self.cursor.fetchone()
                        table_stats["min_rowid"] = min_max[0]
                        table_stats["max_rowid"] = min_max[1]
                    
                    stats["table_statistics"][table] = table_stats
                    
                except sqlite3.Error:
                    continue
            
            self.cursor.execute("PRAGMA page_size")
            stats["performance_metrics"]["page_size"] = self.cursor.fetchone()[0]
            
            self.cursor.execute("PRAGMA page_count")
            stats["performance_metrics"]["page_count"] = self.cursor.fetchone()[0]
            
            self.cursor.execute("PRAGMA freelist_count")
            stats["performance_metrics"]["freelist_count"] = self.cursor.fetchone()[0]
            
            schema_version = self._get_schema_version()
            stats["schema_info"] = {
                "schema_version": schema_version,
                "required_tables": len(self.REQUIRED_TABLES),
                "actual_tables": len(tables)
            }
            
            return stats
            
        except Exception as e:
            stats["error"] = str(e)
            return stats
            
        finally:
            self._close_connection()
    
    def cleanup_old_backups(self, max_backups: int = 10, max_age_days: int = 30) -> Dict[str, Any]:
        """
        Cleanup old database backups.
        
        Args:
            max_backups: Maximum number of backups to keep.
            max_age_days: Maximum age of backups in days.
        
        Returns:
            Dictionary with cleanup results.
        """
        result = {
            "timestamp": datetime.now().isoformat(),
            "backup_dir": None,
            "backups_found": 0,
            "backups_deleted": 0,
            "backups_kept": 0,
            "deleted_files": [],
            "errors": []
        }
        
        try:
            backup_dir = self.db_path.parent / "backups"
            if not backup_dir.exists():
                result["backup_dir"] = str(backup_dir)
                return result
            
            backup_files = list(backup_dir.glob("docubot_backup_*.db"))
            result["backups_found"] = len(backup_files)
            
            backup_files_with_metadata = []
            for backup_file in backup_files:
                metadata_file = backup_file.with_suffix('.json')
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        backup_files_with_metadata.append((backup_file, metadata_file, metadata))
                    except json.JSONDecodeError:
                        continue
            
            backup_files_with_metadata.sort(
                key=lambda x: x[2].get('backup_timestamp', ''),
                reverse=True
            )
            
            cutoff_date = datetime.now() - timedelta(days=max_age_days)
            
            for i, (backup_file, metadata_file, metadata) in enumerate(backup_files_with_metadata):
                backup_time_str = metadata.get('backup_timestamp', '')
                
                try:
                    backup_time = datetime.fromisoformat(backup_time_str.replace('Z', '+00:00'))
                except (ValueError, TypeError):
                    continue
                
                should_delete = False
                
                if backup_time < cutoff_date:
                    should_delete = True
                    reason = f"Older than {max_age_days} days"
                elif i >= max_backups:
                    should_delete = True
                    reason = f"Exceeds maximum of {max_backups} backups"
                
                if should_delete:
                    try:
                        backup_file.unlink()
                        metadata_file.unlink()
                        result["backups_deleted"] += 1
                        result["deleted_files"].append({
                            "backup": str(backup_file),
                            "metadata": str(metadata_file),
                            "reason": reason,
                            "backup_time": backup_time_str
                        })
                        self.logger.info(f"Deleted old backup: {backup_file} ({reason})")
                    except Exception as e:
                        result["errors"].append(f"Failed to delete {backup_file}: {e}")
                else:
                    result["backups_kept"] += 1
            
            return result
            
        except Exception as e:
            result["errors"].append(str(e))
            return result


def main():
    """Command-line interface for database management."""
    
    parser = argparse.ArgumentParser(
        description="DocuBot Database Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --init                    Initialize new database
  %(prog)s --init --force            Reinitialize existing database
  %(prog)s --validate                Validate database schema
  %(prog)s --backup                  Create database backup
  %(prog)s --info                    Show database information
  %(prog)s --cleanup-backups         Cleanup old backups
  %(prog)s --statistics              Show database statistics
        """
    )
    
    parser.add_argument("--db-path", help="Path to SQLite database file")
    parser.add_argument("--init", action="store_true", help="Initialize database")
    parser.add_argument("--force", action="store_true", help="Force reinitialization")
    parser.add_argument("--validate", action="store_true", help="Validate database")
    parser.add_argument("--backup", action="store_true", help="Create backup")
    parser.add_argument("--backup-reason", default="manual", help="Reason for backup")
    parser.add_argument("--info", action="store_true", help="Show database info")
    parser.add_argument("--statistics", action="store_true", help="Show statistics")
    parser.add_argument("--cleanup-backups", action="store_true", help="Cleanup old backups")
    parser.add_argument("--max-backups", type=int, default=10, help="Maximum backups to keep")
    parser.add_argument("--max-age-days", type=int, default=30, help="Maximum backup age in days")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        db_path = Path(args.db_path) if args.db_path else None
        initializer = DatabaseInitializer(db_path)
        
        if args.init:
            result = initializer.initialize_database(force_recreate=args.force)
            print(f"Database initialization: {'SUCCESS' if result['success'] else 'FAILED'}")
            if not result['success'] and result.get('errors'):
                for error in result['errors']:
                    print(f"  Error: {error}")
            if result.get('backup_created'):
                print(f"  Backup created: {result['backup_created']}")
        
        elif args.validate:
            result = initializer.validate_database()
            print(f"Database validation: {'VALID' if result['is_valid'] else 'INVALID'}")
            
            if result.get('table_validation'):
                tv = result['table_validation']
                print(f"  Tables: {tv.get('tables_valid', 0)}/{tv.get('required_tables', 0)} valid")
            
            if result.get('foreign_key_validation'):
                fkv = result['foreign_key_validation']
                if fkv.get('constraints_valid'):
                    print("  Foreign keys: VALID")
                elif fkv.get('constraint_details'):
                    print(f"  Foreign keys: {len(fkv['constraint_details'])} violations")
            
            if result.get('errors'):
                print("\nErrors:")
                for error in result['errors']:
                    print(f"  • {error}")
            
            if result.get('warnings'):
                print("\nWarnings:")
                for warning in result['warnings']:
                    print(f"  • {warning}")
        
        elif args.backup:
            result = initializer.create_backup(args.backup_reason)
            if result['success']:
                print(f"Backup created: {result['backup_path']}")
                print(f"  Size: {result['backup_size']:,} bytes")
                print(f"  Checksum: {result['checksum'][:16]}...")
            else:
                print(f"Backup failed: {', '.join(result['errors'])}")
        
        elif args.info:
            stats = initializer.get_database_statistics()
            print(f"Database: {stats['database_path']}")
            
            if stats['file_info'].get('exists'):
                print(f"  Size: {stats['file_info']['size_bytes']:,} bytes")
                print(f"  Modified: {stats['file_info'].get('modified_time', 'unknown')}")
                
                if stats.get('table_statistics'):
                    total_rows = sum(t['row_count'] for t in stats['table_statistics'].values())
                    print(f"  Total rows: {total_rows:,}")
                    print(f"  Tables: {len(stats['table_statistics'])}")
            
            if stats.get('schema_info'):
                print(f"  Schema version: {stats['schema_info'].get('schema_version', 'unknown')}")
        
        elif args.statistics:
            stats = initializer.get_database_statistics()
            print("Database Statistics:")
            print(json.dumps(stats, indent=2, default=str))
        
        elif args.cleanup_backups:
            result = initializer.cleanup_old_backups(args.max_backups, args.max_age_days)
            print(f"Backup cleanup: {result['backups_deleted']} deleted, {result['backups_kept']} kept")
            
            if result['deleted_files']:
                print("\nDeleted files:")
                for file_info in result['deleted_files'][:5]:
                    print(f"  • {Path(file_info['backup']).name} ({file_info['reason']})")
            
            if result.get('errors'):
                print("\nErrors:")
                for error in result['errors']:
                    print(f"  • {error}")
        
        else:
            parser.print_help()
    
    except DatabaseInitializationError as e:
        print(f"Database error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()