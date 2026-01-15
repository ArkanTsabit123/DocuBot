"""
Database Query Definitions
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta


def get_document_by_id(doc_id: str) -> str:
    return f"""
    SELECT * FROM documents 
    WHERE id = '{doc_id}'
    """


def get_documents_by_status(status: str, limit: int = 100) -> str:
    return f"""
    SELECT * FROM documents 
    WHERE processing_status = '{status}'
    ORDER BY upload_date DESC
    LIMIT {limit}
    """


def get_recent_documents(days: int = 7, limit: int = 50) -> str:
    date_cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    return f"""
    SELECT * FROM documents 
    WHERE upload_date >= '{date_cutoff}'
    ORDER BY upload_date DESC
    LIMIT {limit}
    """


def search_documents_by_text(search_text: str, limit: int = 20) -> str:
    return f"""
    SELECT d.* FROM documents d
    LEFT JOIN chunks c ON d.id = c.document_id
    WHERE d.file_name LIKE '%{search_text}%'
       OR d.tags_json LIKE '%{search_text}%'
       OR c.text_content LIKE '%{search_text}%'
    GROUP BY d.id
    ORDER BY d.upload_date DESC
    LIMIT {limit}
    """


def get_documents_by_type(file_type: str, limit: int = 50) -> str:
    return f"""
    SELECT * FROM documents 
    WHERE file_type = '{file_type}'
    ORDER BY upload_date DESC
    LIMIT {limit}
    """


def get_chunks_by_document(doc_id: str) -> str:
    return f"""
    SELECT * FROM chunks 
    WHERE document_id = '{doc_id}'
    ORDER BY chunk_index
    """


def get_chunk_by_vector_id(vector_id: str) -> str:
    return f"""
    SELECT c.*, d.file_name, d.file_type 
    FROM chunks c
    JOIN documents d ON c.document_id = d.id
    WHERE c.vector_id = '{vector_id}'
    """


def get_recent_conversations(limit: int = 20) -> str:
    return f"""
    SELECT * FROM conversations 
    WHERE is_archived = FALSE
    ORDER BY updated_at DESC
    LIMIT {limit}
    """


def get_conversation_messages(conversation_id: str, limit: int = 100) -> str:
    return f"""
    SELECT * FROM messages 
    WHERE conversation_id = '{conversation_id}'
    ORDER BY created_at ASC
    LIMIT {limit}
    """


def search_conversations(search_text: str, limit: int = 20) -> str:
    return f"""
    SELECT DISTINCT c.* FROM conversations c
    JOIN messages m ON c.id = m.conversation_id
    WHERE c.title LIKE '%{search_text}%'
       OR m.content LIKE '%{search_text}%'
    ORDER BY c.updated_at DESC
    LIMIT {limit}
    """


def get_document_statistics() -> str:
    return """
    SELECT 
        COUNT(*) as total_documents,
        SUM(CASE WHEN processing_status = 'completed' THEN 1 ELSE 0 END) as processed_documents,
        SUM(CASE WHEN processing_status = 'pending' THEN 1 ELSE 0 END) as pending_documents,
        SUM(CASE WHEN processing_status = 'failed' THEN 1 ELSE 0 END) as failed_documents,
        SUM(chunk_count) as total_chunks,
        SUM(word_count) as total_words,
        AVG(file_size) as avg_file_size
    FROM documents
    """


def get_document_counts_by_type() -> str:
    return """
    SELECT 
        file_type,
        COUNT(*) as count,
        AVG(file_size) as avg_size,
        SUM(chunk_count) as total_chunks
    FROM documents
    WHERE processing_status = 'completed'
    GROUP BY file_type
    ORDER BY count DESC
    """


def get_processing_times() -> str:
    return """
    SELECT 
        file_type,
        COUNT(*) as count,
        AVG(
            CAST(strftime('%s', indexed_at) AS INTEGER) - 
            CAST(strftime('%s', upload_date) AS INTEGER)
        ) as avg_processing_seconds
    FROM documents
    WHERE processing_status = 'completed' AND indexed_at IS NOT NULL
    GROUP BY file_type
    """


def get_popular_tags(limit: int = 20) -> str:
    return f"""
    SELECT t.*, COUNT(dt.document_id) as usage_count
    FROM tags t
    LEFT JOIN document_tags dt ON t.id = dt.tag_id
    GROUP BY t.id
    ORDER BY usage_count DESC
    LIMIT {limit}
    """


def get_documents_by_tag(tag_id: str, limit: int = 50) -> str:
    return f"""
    SELECT d.* FROM documents d
    JOIN document_tags dt ON d.id = dt.document_id
    WHERE dt.tag_id = '{tag_id}'
    ORDER BY d.upload_date DESC
    LIMIT {limit}
    """


def get_old_documents(days: int = 90) -> str:
    date_cutoff = (datetime.now() - timedelta(days=days)).isoformat()
    return f"""
    SELECT * FROM documents 
    WHERE upload_date < '{date_cutoff}'
    AND last_accessed < '{date_cutoff}'
    ORDER BY upload_date ASC
    """


def get_large_documents(size_mb: int = 100) -> str:
    size_bytes = size_mb * 1024 * 1024
    return f"""
    SELECT * FROM documents 
    WHERE file_size > {size_bytes}
    ORDER BY file_size DESC
    """


def count_table_rows(table_name: str) -> str:
    return f"SELECT COUNT(*) as count FROM {table_name}"


def get_table_info(table_name: str) -> str:
    return f"PRAGMA table_info({table_name})"
