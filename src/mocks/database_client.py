from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid

class MockDatabaseClient:
    def __init__(self, db_path: str = "data/database/docubot.db"):
        self.db_path = db_path
        self.documents = []
        self.conversations = []
        
    def add_document(self, file_path: str, metadata: Dict[str, Any]) -> str:
        doc_id = str(uuid.uuid4())
        doc = {
            'id': doc_id,
            'file_path': file_path,
            'file_name': file_path.split('/')[-1],
            'file_type': '.' + file_path.split('.')[-1] if '.' in file_path else '',
            'file_size': 1024 * 100,  # 100KB mock
            'upload_date': datetime.now().isoformat(),
            'processing_status': 'completed',
            'metadata': metadata,
            'chunk_count': 5,
            'is_indexed': True
        }
        self.documents.append(doc)
        return doc_id
    
    def list_documents(self) -> List[Dict[str, Any]]:
        return self.documents
    
    def get_document_count(self) -> int:
        return len(self.documents)
    
    def create_conversation(self, title: str = "New Conversation") -> str:
        conv_id = str(uuid.uuid4())
        conv = {
            'id': conv_id,
            'title': title,
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
            'message_count': 0,
            'is_archived': False
        }
        self.conversations.append(conv)
        return conv_id
    
    def add_message(self, conversation_id: str, role: str, content: str, **kwargs) -> str:
        msg_id = str(uuid.uuid4())
        # Cari conversation
        for conv in self.conversations:
            if conv['id'] == conversation_id:
                conv['message_count'] += 1
                conv['updated_at'] = datetime.now().isoformat()
                break
        return msg_id