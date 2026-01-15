"""
Conversation Memory Management
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import json
from ..database.sqlite_client import SQLiteClient


class ConversationMemory:
    """
    Manages conversation history and context
    """
    
    def __init__(self, db_client: SQLiteClient):
        self.db_client = db_client
    
    def create_conversation(self, title: Optional[str] = None, tags: Optional[List[str]] = None) -> str:
        """
        Create a new conversation
        
        Args:
            title: Optional conversation title
            tags: Optional list of tags
        
        Returns:
            Conversation ID
        """
        from uuid import uuid4
        conversation_id = str(uuid4())
        
        self.db_client.execute(
            """
            INSERT INTO conversations (id, title, tags_json)
            VALUES (?, ?, ?)
            """,
            (conversation_id, title or f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}", 
             json.dumps(tags or []))
        )
        
        return conversation_id
    
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        model_used: Optional[str] = None,
        sources: Optional[List[Dict[str, Any]]] = None,
        tokens: Optional[int] = None,
        processing_time_ms: Optional[int] = None
    ) -> str:
        """
        Add a message to conversation
        
        Args:
            conversation_id: Conversation ID
            role: Message role (user/assistant)
            content: Message content
            model_used: Model used for generation
            sources: Retrieved sources for RAG
            tokens: Token count
            processing_time_ms: Processing time in milliseconds
        
        Returns:
            Message ID
        """
        from uuid import uuid4
        message_id = str(uuid4())
        
        self.db_client.execute(
            """
            INSERT INTO messages 
            (id, conversation_id, role, content, model_used, sources_json, tokens, processing_time_ms)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (message_id, conversation_id, role, content, model_used, 
             json.dumps(sources or []), tokens, processing_time_ms)
        )
        
        # Update conversation timestamp and message count
        self.db_client.execute(
            """
            UPDATE conversations 
            SET updated_at = CURRENT_TIMESTAMP, 
                message_count = message_count + 1,
                total_tokens = total_tokens + ?
            WHERE id = ?
            """,
            (tokens or 0, conversation_id)
        )
        
        return message_id
    
    def get_conversation_messages(self, conversation_id: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get messages from a conversation
        
        Args:
            conversation_id: Conversation ID
            limit: Maximum number of messages to return
        
        Returns:
            List of messages
        """
        query = """
        SELECT * FROM messages 
        WHERE conversation_id = ? 
        ORDER BY created_at
        """
        
        if limit:
            query += f" LIMIT {limit}"
        
        messages = self.db_client.fetch_all(query, (conversation_id,))
        
        # Parse JSON fields
        for message in messages:
            if message.get('sources_json'):
                try:
                    message['sources'] = json.loads(message['sources_json'])
                except:
                    message['sources'] = []
        
        return messages
    
    def get_conversation_summary(self, conversation_id: str) -> Dict[str, Any]:
        """
        Get conversation summary
        
        Args:
            conversation_id: Conversation ID
        
        Returns:
            Conversation summary
        """
        conversation = self.db_client.fetch_one(
            "SELECT * FROM conversations WHERE id = ?",
            (conversation_id,)
        )
        
        if not conversation:
            return {}
        
        # Get recent messages
        messages = self.get_conversation_messages(conversation_id, limit=5)
        
        # Parse tags
        tags = []
        if conversation.get('tags_json'):
            try:
                tags = json.loads(conversation['tags_json'])
            except:
                pass
        
        return {
            'id': conversation_id,
            'title': conversation.get('title'),
            'created_at': conversation.get('created_at'),
            'updated_at': conversation.get('updated_at'),
            'message_count': conversation.get('message_count', 0),
            'total_tokens': conversation.get('total_tokens', 0),
            'tags': tags,
            'recent_messages': messages
        }
    
    def list_conversations(self, limit: int = 20, offset: int = 0) -> List[Dict[str, Any]]:
        """
        List all conversations
        
        Args:
            limit: Maximum number of conversations
            offset: Offset for pagination
        
        Returns:
            List of conversations
        """
        conversations = self.db_client.fetch_all(
            """
            SELECT * FROM conversations 
            WHERE is_archived = FALSE
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset)
        )
        
        for conversation in conversations:
            if conversation.get('tags_json'):
                try:
                    conversation['tags'] = json.loads(conversation['tags_json'])
                except:
                    conversation['tags'] = []
        
        return conversations
    
    def archive_conversation(self, conversation_id: str) -> bool:
        """
        Archive a conversation
        
        Args:
            conversation_id: Conversation ID
        
        Returns:
            True if successful
        """
        try:
            self.db_client.execute(
                "UPDATE conversations SET is_archived = TRUE WHERE id = ?",
                (conversation_id,)
            )
            return True
        except:
            return False
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """
        Delete a conversation and all its messages
        
        Args:
            conversation_id: Conversation ID
        
        Returns:
            True if successful
        """
        try:
            # Messages will be deleted automatically due to foreign key cascade
            self.db_client.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,)
            )
            return True
        except:
            return False
    
    def get_conversation_context(self, conversation_id: str, max_tokens: int = 1000) -> str:
        """
        Get conversation context for LLM
        
        Args:
            conversation_id: Conversation ID
            max_tokens: Maximum tokens for context
        
        Returns:
            Formatted conversation context
        """
        messages = self.get_conversation_messages(conversation_id)
        
        if not messages:
            return ""
        
        # Build context from messages
        context_parts = []
        token_count = 0
        
        for message in reversed(messages):  # Start from most recent
            role = message.get('role', 'user')
            content = message.get('content', '')
            
            # Simple token estimation (rough)
            message_tokens = len(content.split())
            
            if token_count + message_tokens > max_tokens:
                break
            
            context_parts.insert(0, f"{role}: {content}")
            token_count += message_tokens
        
        return "\n".join(context_parts)
