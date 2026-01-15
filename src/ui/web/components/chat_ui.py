"""
Web Chat UI Components for Streamlit
"""

import streamlit as st
from typing import List, Dict, Any, Optional
import json
from datetime import datetime


class ChatUI:
    """Chat interface components for Streamlit"""
    
    def __init__(self):
        pass
    
    def display_chat_message(self, role: str, content: str, timestamp: Optional[str] = None):
        """
        Display a chat message.
        
        Args:
            role: Message role
            content: Message content
            timestamp: Optional timestamp
        """
        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
                if timestamp:
                    st.caption(f"User - {timestamp}")
        else:
            with st.chat_message("assistant"):
                st.markdown(content)
                if timestamp:
                    st.caption(f"Assistant - {timestamp}")
    
    def display_chat_history(self, messages: List[Dict[str, Any]]):
        """
        Display chat history.
        
        Args:
            messages: List of message dictionaries
        """
        for message in messages:
            self.display_chat_message(
                role=message.get('role', 'user'),
                content=message.get('content', ''),
                timestamp=message.get('timestamp')
            )
    
    def chat_input(self, placeholder: str = "Type your message...") -> Optional[str]:
        """
        Display chat input.
        
        Args:
            placeholder: Input placeholder text
            
        Returns:
            User input or None
        """
        return st.chat_input(placeholder)
    
    def display_sources(self, sources: List[Dict[str, Any]]):
        """
        Display source citations.
        
        Args:
            sources: List of source dictionaries
        """
        if sources:
            with st.expander("Sources", expanded=False):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"**Source {i}**")
                    if 'title' in source:
                        st.markdown(f"**Title:** {source['title']}")
                    if 'content' in source:
                        st.markdown(f"**Excerpt:** {source['content'][:200]}...")
                    if 'similarity' in source:
                        st.markdown(f"**Relevance:** {source['similarity']:.2%}")
                    st.divider()
    
    def display_processing_status(self, status: str, message: str = ""):
        """
        Display processing status.
        
        Args:
            status: Status
            message: Status message
        """
        if status == "processing":
            with st.spinner(message or "Processing..."):
                pass
        elif status == "success":
            st.success(message or "Success")
        elif status == "error":
            st.error(message or "Error occurred")
    
    def create_sidebar(self):
        """Create application sidebar"""
        with st.sidebar:
            st.title("Settings")
            
            model = st.selectbox(
                "Model",
                ["llama2:7b", "mistral:7b", "neural-chat:7b"],
                index=0
            )
            
            temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.1,
                step=0.05,
                help="Higher values make output more random"
            )
            
            st.divider()
            st.subheader("Conversations")
            
            if st.button("New Conversation", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
            
            if st.button("Clear History", use_container_width=True):
                st.session_state.conversation_history = []
                st.rerun()
            
            return {
                'model': model,
                'temperature': temperature
            }
    
    def display_document_upload(self):
        """
        Display document upload interface.
        
        Returns:
            Uploaded files or None
        """
        st.subheader("Upload Documents")
        
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'txt', 'docx', 'md', 'html'],
            accept_multiple_files=True,
            help="Upload documents to add to knowledge base"
        )
        
        if uploaded_files:
            st.success(f"Selected {len(uploaded_files)} file(s)")
            
            for file in uploaded_files:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"{file.name} ({file.size / 1024:.1f} KB)")
                with col2:
                    st.button("Process", key=f"process_{file.name}")
        
        return uploaded_files
    
    def display_document_list(self, documents: List[Dict[str, Any]]):
        """
        Display list of documents.
        
        Args:
            documents: List of document dictionaries
        """
        st.subheader("Documents")
        
        if not documents:
            st.info("No documents uploaded yet")
            return
        
        for doc in documents:
            with st.expander(f"{doc.get('file_name', 'Unknown')}", expanded=False):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.markdown(f"**Type:** {doc.get('file_type', 'Unknown')}")
                    st.markdown(f"**Size:** {doc.get('file_size', 0) / 1024:.1f} KB")
                    st.markdown(f"**Status:** {doc.get('processing_status', 'Unknown')}")
                    st.markdown(f"**Chunks:** {doc.get('chunk_count', 0)}")
                
                with col2:
                    if st.button("Delete", key=f"delete_{doc.get('id')}"):
                        st.warning(f"Delete {doc.get('file_name')}?")
    
    def display_statistics(self, stats: Dict[str, Any]):
        """
        Display application statistics.
        
        Args:
            stats: Statistics dictionary
        """
        st.subheader("Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Documents", stats.get('total_documents', 0))
        
        with col2:
            st.metric("Chunks", stats.get('total_chunks', 0))
        
        with col3:
            st.metric("Conversations", stats.get('total_conversations', 0))
        
        if 'database_size_mb' in stats:
            st.metric("Database Size", f"{stats['database_size_mb']:.1f} MB")
    
    def display_error(self, error: Exception):
        """
        Display error message.
        
        Args:
            error: Exception object
        """
        st.error(f"Error: {str(error)}")
        
        with st.expander("Error Details", expanded=False):
            st.code(str(error))
    
    def display_info_message(self, message: str):
        """
        Display informational message.
        
        Args:
            message: Message text
        """
        st.info(message)
    
    def display_warning_message(self, message: str):
        """
        Display warning message.
        
        Args:
            message: Message text
        """
        st.warning(message)
    
    def display_success_message(self, message: str):
        """
        Display success message.
        
        Args:
            message: Message text
        """
        st.success(message)


_chat_ui = None

def get_chat_ui() -> ChatUI:
    """
    Get or create ChatUI instance.
    
    Returns:
        ChatUI instance
    """
    global _chat_ui
    
    if _chat_ui is None:
        _chat_ui = ChatUI()
    
    return _chat_ui
