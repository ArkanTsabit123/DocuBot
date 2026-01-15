"""
Database Initialization Script
"""

import sqlite3
from pathlib import Path
import json
from datetime import datetime

def initialize_database(db_path: Path):
    """
    Initialize SQLite database with schema
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    connection = sqlite3.connect(str(db_path))
    cursor = connection.cursor()
    
    # Enable foreign keys
    cursor.execute("PRAGMA foreign_keys = ON")
    
    # Create tables
    create_tables_sql = """
    -- Documents table
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
    );
    
    -- Chunks table
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
    );
    
    -- Conversations table
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
    );
    
    -- Messages table
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
    );
    
    -- Settings table
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    """
    
    cursor.executescript(create_tables_sql)
    
    # Create indexes
    cursor.executescript("""
    CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(processing_status);
    CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(file_type);
    CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
    CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
    CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at);
    CREATE INDEX IF NOT EXISTS idx_documents_upload_date ON documents(upload_date);
    """)
    
    # Insert default settings
    default_settings = {
        'app_version': '1.0.0',
        'default_llm_model': 'llama2:7b',
        'default_embedding_model': 'all-MiniLM-L6-v2',
        'chunk_size': '500',
        'chunk_overlap': '50',
        'rag_top_k': '5'
    }
    
    for key, value in default_settings.items():
        cursor.execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            (key, str(value), datetime.now().isoformat())
        )
    
    connection.commit()
    connection.close()
    
    print(f"Database initialized at: {db_path}")

if __name__ == "__main__":
    # Create mock constants if they don't exist
    from pathlib import Path
    DATABASE_DIR = Path.home() / ".docubot" / "data"
    DATABASE_NAME = "docubot.db"
    
    db_path = DATABASE_DIR / DATABASE_NAME
    initialize_database(db_path)
