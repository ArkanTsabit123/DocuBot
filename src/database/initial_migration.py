# docubot/src/database/migrations/initial_migration.py

"""
Database migration: Initial schema
Based on blueprint database schema
"""

from alembic import op
import sqlalchemy as sa

# Revision identifiers
revision = 'initial'
down_revision = None
branch_labels = None
depends_on = None

def upgrade():
    """Create initial database schema"""
    
    # documents table
    op.create_table('documents',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('file_path', sa.Text(), nullable=False),
        sa.Column('file_name', sa.Text(), nullable=False),
        sa.Column('file_type', sa.String(), nullable=False),
        sa.Column('file_size', sa.Integer()),
        sa.Column('upload_date', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('processing_status', sa.String(), server_default='pending'),
        sa.Column('processing_error', sa.Text()),
        sa.Column('metadata_json', sa.Text()),
        sa.Column('vector_ids_json', sa.Text()),
        sa.Column('chunk_count', sa.Integer(), server_default='0'),
        sa.Column('word_count', sa.Integer(), server_default='0'),
        sa.Column('language', sa.String()),
        sa.Column('tags_json', sa.Text()),
        sa.Column('summary', sa.Text()),
        sa.Column('is_indexed', sa.Boolean(), server_default='0'),
        sa.Column('indexed_at', sa.DateTime()),
        sa.Column('last_accessed', sa.DateTime()),
        sa.Column('access_count', sa.Integer(), server_default='0'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # chunks table
    op.create_table('chunks',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('document_id', sa.String(), nullable=False),
        sa.Column('chunk_index', sa.Integer(), nullable=False),
        sa.Column('text_content', sa.Text(), nullable=False),
        sa.Column('cleaned_text', sa.Text(), nullable=False),
        sa.Column('token_count', sa.Integer()),
        sa.Column('embedding_model', sa.String()),
        sa.Column('vector_id', sa.String(), nullable=False),
        sa.Column('metadata_json', sa.Text()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE')
    )
    
    # conversations table
    op.create_table('conversations',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('title', sa.Text()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('message_count', sa.Integer(), server_default='0'),
        sa.Column('total_tokens', sa.Integer(), server_default='0'),
        sa.Column('tags_json', sa.Text()),
        sa.Column('is_archived', sa.Boolean(), server_default='0'),
        sa.Column('export_path', sa.Text()),
        sa.PrimaryKeyConstraint('id')
    )
    
    # messages table
    op.create_table('messages',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('conversation_id', sa.String(), nullable=False),
        sa.Column('role', sa.String(), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('tokens', sa.Integer()),
        sa.Column('model_used', sa.String()),
        sa.Column('sources_json', sa.Text()),
        sa.Column('processing_time_ms', sa.Integer()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['conversation_id'], ['conversations.id'], ondelete='CASCADE')
    )
    
    # tags table
    op.create_table('tags',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('name', sa.String(), nullable=False, unique=True),
        sa.Column('color', sa.String()),
        sa.Column('description', sa.Text()),
        sa.Column('created_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.Column('usage_count', sa.Integer(), server_default='0'),
        sa.PrimaryKeyConstraint('id')
    )
    
    # document_tags junction table
    op.create_table('document_tags',
        sa.Column('document_id', sa.String(), nullable=False),
        sa.Column('tag_id', sa.String(), nullable=False),
        sa.Column('added_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('document_id', 'tag_id'),
        sa.ForeignKeyConstraint(['document_id'], ['documents.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['tag_id'], ['tags.id'], ondelete='CASCADE')
    )
    
    # settings table
    op.create_table('settings',
        sa.Column('key', sa.String(), nullable=False),
        sa.Column('value', sa.Text()),
        sa.Column('updated_at', sa.DateTime(), server_default=sa.text('CURRENT_TIMESTAMP')),
        sa.PrimaryKeyConstraint('key')
    )
    
    # Create indexes
    op.create_index('idx_documents_status', 'documents', ['processing_status'])
    op.create_index('idx_documents_type', 'documents', ['file_type'])
    op.create_index('idx_chunks_document', 'chunks', ['document_id'])
    op.create_index('idx_messages_conversation', 'messages', ['conversation_id'])
    op.create_index('idx_messages_created', 'messages', ['created_at'])
    op.create_index('idx_documents_upload_date', 'documents', ['upload_date'])

def downgrade():
    """Drop all tables"""
    op.drop_index('idx_documents_upload_date')
    op.drop_index('idx_messages_created')
    op.drop_index('idx_messages_conversation')
    op.drop_index('idx_chunks_document')
    op.drop_index('idx_documents_type')
    op.drop_index('idx_documents_status')
    op.drop_table('settings')
    op.drop_table('document_tags')
    op.drop_table('tags')
    op.drop_table('messages')
    op.drop_table('conversations')
    op.drop_table('chunks')
    op.drop_table('documents')