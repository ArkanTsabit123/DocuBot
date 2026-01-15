"""
Chat Panel Component
Message threading and source citation display
"""

import customtkinter as ctk
from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from ..core.logger import get_logger

logger = get_logger(__name__)


class ChatMessage:
    """Individual chat message with metadata"""
    
    def __init__(self, role: str, content: str, timestamp: Optional[str] = None):
        self.role = role  # "user" or "assistant"
        self.content = content
        self.timestamp = timestamp or datetime.now().isoformat()
        self.sources = []  # List of source documents
        self.tokens = 0
        self.model_used = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary"""
        return {
            'role': self.role,
            'content': self.content,
            'timestamp': self.timestamp,
            'sources': self.sources,
            'tokens': self.tokens,
            'model_used': self.model_used
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChatMessage':
        """Create message from dictionary"""
        message = cls(data['role'], data['content'], data.get('timestamp'))
        message.sources = data.get('sources', [])
        message.tokens = data.get('tokens', 0)
        message.model_used = data.get('model_used')
        return message


class ChatPanel(ctk.CTkFrame):
    """Chat interface panel with message threading"""
    
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.messages: List[ChatMessage] = []
        self.current_conversation = None
        
        self._create_widgets()
        self._setup_layout()
        
        logger.debug("Chat panel initialized")
    
    def _create_widgets(self):
        """Create chat panel widgets"""
        
        # Chat header
        self.header_frame = ctk.CTkFrame(self, height=40)
        self.title_label = ctk.CTkLabel(
            self.header_frame,
            text="Chat",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        
        # Message display area
        self.message_frame = ctk.CTkScrollableFrame(self)
        
        # Input area
        self.input_frame = ctk.CTkFrame(self, height=100)
        
        self.message_input = ctk.CTkTextbox(
            self.input_frame,
            height=80,
            wrap="word",
            font=ctk.CTkFont(size=12)
        )
        
        self.send_button = ctk.CTkButton(
            self.input_frame,
            text="Send",
            width=100,
            height=30,
            font=ctk.CTkFont(size=12, weight="bold")
        )
        
        # Source citation area
        self.sources_frame = ctk.CTkFrame(self, height=150)
        self.sources_label = ctk.CTkLabel(
            self.sources_frame,
            text="Sources",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.sources_text = ctk.CTkTextbox(
            self.sources_frame,
            state="disabled",
            height=120
        )
    
    def _setup_layout(self):
        """Arrange chat panel widgets"""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        # Header
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 5))
        self.title_label.pack(side="left", padx=10)
        
        # Message display
        self.message_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)
        
        # Input area
        self.input_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=5)
        self.input_frame.grid_columnconfigure(0, weight=1)
        
        self.message_input.grid(row=0, column=0, sticky="nsew", padx=(0, 5))
        self.send_button.grid(row=0, column=1, sticky="ns")
        
        # Sources area
        self.sources_frame.grid(row=3, column=0, sticky="ew", padx=10, pady=(5, 10))
        self.sources_label.pack(anchor="w", padx=10, pady=(5, 0))
        self.sources_text.pack(fill="both", expand=True, padx=10, pady=(0, 5))
    
    def add_message(self, message: ChatMessage):
        """Add message to chat display"""
        self.messages.append(message)
        self._display_message(message)
        
        # Update sources if present
        if message.sources:
            self._update_sources_display(message.sources)
        
        logger.debug(f"Added {message.role} message: {message.content[:50]}...")
    
    def _display_message(self, message: ChatMessage):
        """Display message in chat area"""
        
        message_frame = ctk.CTkFrame(self.message_frame)
        
        # Role indicator
        role_color = "#2B5278" if message.role == "user" else "#1E6B5E"
        role_label = ctk.CTkLabel(
            message_frame,
            text=message.role.capitalize(),
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color="white",
            fg_color=role_color,
            corner_radius=3
        )
        role_label.pack(anchor="w", padx=5, pady=(5, 0))
        
        # Message content
        content_text = ctk.CTkTextbox(
            message_frame,
            height=min(max(len(message.content) // 50 * 20, 40), 200),
            wrap="word",
            state="disabled",
            font=ctk.CTkFont(size=11)
        )
        content_text.pack(fill="x", padx=5, pady=(0, 5))
        
        # Set message text
        content_text.configure(state="normal")
        content_text.insert("1.0", message.content)
        content_text.configure(state="disabled")
        
        # Timestamp
        timestamp = datetime.fromisoformat(message.timestamp).strftime("%H:%M")
        time_label = ctk.CTkLabel(
            message_frame,
            text=timestamp,
            font=ctk.CTkFont(size=9),
            text_color="gray"
        )
        time_label.pack(anchor="e", padx=5, pady=(0, 5))
        
        # Add to message frame
        message_frame.pack(fill="x", padx=5, pady=2)
        
        # Scroll to bottom
        self.message_frame._parent_canvas.yview_moveto(1.0)
    
    def _update_sources_display(self, sources: List[Dict[str, Any]]):
        """Update source citation display"""
        self.sources_text.configure(state="normal")
        self.sources_text.delete("1.0", "end")
        
        for i, source in enumerate(sources, 1):
            source_text = f"{i}. {source.get('title', 'Unknown')}"
            if 'page' in source:
                source_text += f" (page {source['page']})"
            if 'score' in source:
                source_text += f" - {source['score']:.2f} relevance"
            
            self.sources_text.insert("end", source_text + "\n")
        
        self.sources_text.configure(state="disabled")
    
    def clear_chat(self):
        """Clear all messages from chat"""
        for widget in self.message_frame.winfo_children():
            widget.destroy()
        
        self.messages.clear()
        self.sources_text.configure(state="normal")
        self.sources_text.delete("1.0", "end")
        self.sources_text.configure(state="disabled")
        
        logger.debug("Chat cleared")
    
    def load_conversation(self, messages: List[ChatMessage]):
        """Load conversation from messages"""
        self.clear_chat()
        
        for message in messages:
            self.add_message(message)
        
        logger.debug(f"Loaded conversation with {len(messages)} messages")
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get conversation history as dictionaries"""
        return [msg.to_dict() for msg in self.messages]
    
    def set_conversation_title(self, title: str):
        """Set conversation title"""
        self.title_label.configure(text=title)
    
    def get_input_text(self) -> str:
        """Get text from input field"""
        return self.message_input.get("1.0", "end-1c").strip()
    
    def clear_input(self):
        """Clear input field"""
        self.message_input.delete("1.0", "end")


class SourceCitation:
    """Source citation display component"""
    
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.sources = []
        self._create_widgets()
    
    def _create_widgets(self):
        """Create source citation widgets"""
        self.label = ctk.CTkLabel(
            self,
            text="Cited Sources",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        
        self.listbox = ctk.CTkTextbox(
            self,
            state="disabled",
            height=100
        )
        
        self.label.pack(anchor="w", padx=10, pady=(10, 5))
        self.listbox.pack(fill="both", expand=True, padx=10, pady=(0, 10))
    
    def update_sources(self, sources: List[Dict[str, Any]]):
        """Update displayed sources"""
        self.sources = sources
        
        self.listbox.configure(state="normal")
        self.listbox.delete("1.0", "end")
        
        for i, source in enumerate(sources, 1):
            display_text = self._format_source(source, i)
            self.listbox.insert("end", display_text + "\n\n")
        
        self.listbox.configure(state="disabled")
    
    def _format_source(self, source: Dict[str, Any], index: int) -> str:
        """Format source for display"""
        lines = [f"{index}. {source.get('title', 'Unknown Document')}"]
        
        if 'file_name' in source:
            lines.append(f"   File: {source['file_name']}")
        
        if 'page' in source:
            lines.append(f"   Page: {source['page']}")
        
        if 'score' in source:
            lines.append(f"   Relevance: {source['score']:.2%}")
        
        if 'excerpt' in source and source['excerpt']:
            excerpt = source['excerpt'][:150] + "..." if len(source['excerpt']) > 150 else source['excerpt']
            lines.append(f"   Excerpt: {excerpt}")
        
        return "\n".join(lines)
    
    def clear(self):
        """Clear all sources"""
        self.sources.clear()
        self.listbox.configure(state="normal")
        self.listbox.delete("1.0", "end")
        self.listbox.configure(state="disabled")
