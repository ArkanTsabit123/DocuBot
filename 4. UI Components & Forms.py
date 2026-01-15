#4. UI Components & Forms.py
"""
4. UI Components & Forms
Desktop UI elements (P1.14.3 layout, P1.14.7 chat, P1.14.8 input)
Settings panels (P1.14.11)
Document management UI (P2.8.1 preview, P2.8.2 batch ops)
Export functionality (P2.9.1, P2.9.2, P2.9.3)
"""

import os
import sys
from pathlib import Path
import json
import yaml


class UIComponentsFixer:
    def __init__(self, project_dir="DocuBot"):
        self.project_dir = Path(project_dir).absolute()
    
    def fix_all_ui_components(self):
        print("=" * 60)
        print("UI Components Implementation")
        print("=" * 60)
        
        fixes = [
            self.fix_main_window_layout,
            self.fix_chat_panel,
            self.fix_document_panel,
            self.fix_settings_panel,
            self.fix_export_manager,
            self.fix_ui_components,
            self.fix_themes,
            self.update_ui_config
        ]
        
        for i, fix_func in enumerate(fixes, 1):
            print(f"\n[{i}/{len(fixes)}] Implementing {fix_func.__name__}...")
            try:
                fix_func()
                print("   Implementation complete")
            except Exception as e:
                print(f"   Implementation issue: {e}")
        
        print("\n" + "=" * 60)
        print("UI components implementation complete")
        print("=" * 60)
    
    def fix_main_window_layout(self):
        main_window_file = self.project_dir / "src" / "ui" / "desktop" / "main_window.py"
        
        main_window_content = '''"""
Main Application Window
Three-panel layout implementation with CustomTkinter
"""

import customtkinter as ctk
from typing import Optional, Callable, Dict, Any
from pathlib import Path
import threading
from ..core.logger import get_logger

logger = get_logger(__name__)


class MainWindow(ctk.CTk):
    """Main application window with three-panel layout"""
    
    def __init__(self):
        super().__init__()
        
        self.title("DocuBot - Local AI Knowledge Assistant")
        self.geometry("1200x800")
        
        # Configure grid
        self.grid_columnconfigure(0, weight=1)  # Left sidebar
        self.grid_columnconfigure(1, weight=3)  # Main content
        self.grid_columnconfigure(2, weight=2)  # Right sidebar
        self.grid_rowconfigure(0, weight=1)
        
        # Application state
        self.current_document = None
        self.current_conversation = None
        self.theme_mode = "dark"
        
        # Initialize UI components
        self._setup_theme()
        self._create_widgets()
        self._setup_layout()
        self._bind_events()
        
        logger.info("Main window initialized")
    
    def _setup_theme(self):
        """Configure application theme"""
        ctk.set_appearance_mode(self.theme_mode)
        ctk.set_default_color_theme("blue")
    
    def _create_widgets(self):
        """Create all UI widgets"""
        
        # Left sidebar - Document management
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        
        # Document list
        self.documents_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Documents",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        
        self.documents_listbox = ctk.CTkTextbox(
            self.sidebar_frame,
            height=300,
            state="disabled"
        )
        
        # Upload button
        self.upload_button = ctk.CTkButton(
            self.sidebar_frame,
            text="Upload Document",
            command=self._upload_document
        )
        
        # Main content area - Chat interface
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        
        # Chat display
        self.chat_display = ctk.CTkTextbox(
            self.main_frame,
            state="disabled",
            wrap="word"
        )
        
        # Input area
        self.input_frame = ctk.CTkFrame(self.main_frame)
        
        self.input_text = ctk.CTkTextbox(
            self.input_frame,
            height=80,
            wrap="word"
        )
        
        self.send_button = ctk.CTkButton(
            self.input_frame,
            text="Send",
            width=80,
            command=self._send_message
        )
        
        # Right sidebar - Document preview and tools
        self.right_sidebar = ctk.CTkFrame(self, width=300, corner_radius=0)
        
        # Document preview
        self.preview_label = ctk.CTkLabel(
            self.right_sidebar,
            text="Document Preview",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        
        self.preview_text = ctk.CTkTextbox(
            self.right_sidebar,
            height=400,
            state="disabled"
        )
        
        # Status bar
        self.status_bar = ctk.CTkFrame(self, height=30)
        self.status_label = ctk.CTkLabel(
            self.status_bar,
            text="Ready",
            anchor="w"
        )
    
    def _setup_layout(self):
        """Arrange widgets in three-panel layout"""
        
        # Left sidebar layout
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 5), pady=5)
        self.documents_label.pack(pady=(10, 5), padx=10, anchor="w")
        self.documents_listbox.pack(padx=10, pady=(0, 10), fill="both", expand=True)
        self.upload_button.pack(pady=10, padx=10, fill="x")
        
        # Main content layout
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=5, pady=5)
        self.chat_display.pack(pady=(10, 5), padx=10, fill="both", expand=True)
        
        self.input_frame.pack(pady=5, padx=10, fill="x")
        self.input_text.pack(side="left", fill="both", expand=True, padx=(0, 5))
        self.send_button.pack(side="right", fill="y")
        
        # Right sidebar layout
        self.right_sidebar.grid(row=0, column=2, sticky="nsew", padx=(5, 0), pady=5)
        self.preview_label.pack(pady=(10, 5), padx=10, anchor="w")
        self.preview_text.pack(padx=10, pady=(0, 10), fill="both", expand=True)
        
        # Status bar
        self.status_bar.grid(row=1, column=0, columnspan=3, sticky="ew")
        self.status_label.pack(side="left", padx=10, fill="x", expand=True)
    
    def _bind_events(self):
        """Bind keyboard and mouse events"""
        self.bind("<Return>", self._on_enter_key)
        self.bind("<Control-o>", lambda e: self._upload_document())
        self.bind("<Control-q>", lambda e: self.quit())
    
    def _upload_document(self):
        """Handle document upload"""
        from tkinter import filedialog
        
        file_path = filedialog.askopenfilename(
            title="Select Document",
            filetypes=[
                ("All supported files", "*.pdf *.docx *.txt *.epub *.md *.html"),
                ("PDF files", "*.pdf"),
                ("Word documents", "*.docx"),
                ("Text files", "*.txt"),
                ("EPUB files", "*.epub"),
                ("Markdown files", "*.md"),
                ("HTML files", "*.html")
            ]
        )
        
        if file_path:
            self._process_document(file_path)
    
    def _process_document(self, file_path: str):
        """Process uploaded document"""
        self._update_status(f"Processing {Path(file_path).name}...")
        
        # Process in background thread
        def process_task():
            try:
                # Simulate processing
                import time
                time.sleep(1)
                
                # Add to document list
                doc_name = Path(file_path).name
                self.after(0, self._add_document_to_list, doc_name)
                self.after(0, lambda: self._update_status(f"Added: {doc_name}"))
                
            except Exception as e:
                self.after(0, lambda: self._update_status(f"Error: {str(e)}"))
                logger.error(f"Document processing failed: {e}")
        
        threading.Thread(target=process_task, daemon=True).start()
    
    def _add_document_to_list(self, doc_name: str):
        """Add document to sidebar list"""
        self.documents_listbox.configure(state="normal")
        self.documents_listbox.insert("end", f"• {doc_name}\\n")
        self.documents_listbox.configure(state="disabled")
    
    def _send_message(self):
        """Send chat message"""
        message = self.input_text.get("1.0", "end-1c").strip()
        
        if not message:
            return
        
        # Display user message
        self._add_chat_message("user", message)
        
        # Clear input
        self.input_text.delete("1.0", "end")
        
        # Process AI response in background
        def get_response():
            try:
                # Simulate AI processing
                import time
                time.sleep(1)
                
                response = f"Processing query: {message[:50]}..."
                self.after(0, self._add_chat_message, "assistant", response)
                
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                self.after(0, self._add_chat_message, "assistant", error_msg)
                logger.error(f"Response generation failed: {e}")
        
        threading.Thread(target=get_response, daemon=True).start()
    
    def _add_chat_message(self, sender: str, message: str):
        """Add message to chat display"""
        self.chat_display.configure(state="normal")
        
        if sender == "user":
            prefix = "You: "
            tag = "user"
        else:
            prefix = "DocuBot: "
            tag = "assistant"
        
        self.chat_display.insert("end", f"{prefix}{message}\\n\\n")
        
        # Apply tags for styling
        start = self.chat_display.search(
            prefix, "end-1c linestart", backwards=True, regexp=True
        )
        if start:
            end = f"{start}+{len(prefix)}c"
            self.chat_display.tag_add(tag, start, end)
        
        self.chat_display.configure(state="disabled")
        self.chat_display.see("end")
    
    def _on_enter_key(self, event=None):
        """Handle Enter key press (with Ctrl for newline)"""
        if event.state & 0x4:  # Ctrl key pressed
            self.input_text.insert("insert", "\\n")
            return "break"
        else:
            self._send_message()
            return "break"
    
    def _update_status(self, message: str):
        """Update status bar message"""
        self.status_label.configure(text=message)
    
    def toggle_theme(self):
        """Toggle between dark and light themes"""
        self.theme_mode = "light" if self.theme_mode == "dark" else "dark"
        ctk.set_appearance_mode(self.theme_mode)
        logger.info(f"Theme changed to {self.theme_mode}")
    
    def run(self):
        """Start the application"""
        self.mainloop()


def main():
    """Main entry point for UI"""
    app = MainWindow()
    app.run()


if __name__ == "__main__":
    main()
'''
        
        main_window_file.write_text(main_window_content)
    
    def fix_chat_panel(self):
        chat_panel_file = self.project_dir / "src" / "ui" / "desktop" / "chat_panel.py"
        
        chat_panel_content = '''"""
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
            
            self.sources_text.insert("end", source_text + "\\n")
        
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
            self.listbox.insert("end", display_text + "\\n\\n")
        
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
        
        return "\\n".join(lines)
    
    def clear(self):
        """Clear all sources"""
        self.sources.clear()
        self.listbox.configure(state="normal")
        self.listbox.delete("1.0", "end")
        self.listbox.configure(state="disabled")
'''
        
        chat_panel_file.write_text(chat_panel_content)
    
    def fix_document_panel(self):
        document_panel_file = self.project_dir / "src" / "ui" / "desktop" / "document_panel.py"
        
        document_panel_content = '''"""
Document Management Panel
Document preview, batch operations, and tag management
"""

import customtkinter as ctk
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from tkinter import filedialog
import threading
from datetime import datetime
from ..core.logger import get_logger

logger = get_logger(__name__)


class DocumentCard(ctk.CTkFrame):
    """Individual document card for display"""
    
    def __init__(self, master, document_data: Dict[str, Any], **kwargs):
        super().__init__(master, **kwargs)
        
        self.document_data = document_data
        self.selected = False
        self._create_widgets()
        
        # Bind click events
        self.bind("<Button-1>", self._on_click)
        for child in self.winfo_children():
            child.bind("<Button-1>", self._on_click)
    
    def _create_widgets(self):
        """Create document card widgets"""
        self.configure(height=100, corner_radius=5)
        
        # File icon based on type
        file_type = self.document_data.get('file_type', '').lower()
        icon_text = self._get_icon_for_type(file_type)
        
        self.icon_label = ctk.CTkLabel(
            self,
            text=icon_text,
            font=ctk.CTkFont(size=20),
            width=40
        )
        
        # Document info
        info_frame = ctk.CTkFrame(self, fg_color="transparent")
        
        self.title_label = ctk.CTkLabel(
            info_frame,
            text=self.document_data.get('file_name', 'Unknown'),
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w"
        )
        
        # Metadata
        meta_text = self._format_metadata()
        self.meta_label = ctk.CTkLabel(
            info_frame,
            text=meta_text,
            font=ctk.CTkFont(size=10),
            text_color="gray",
            anchor="w"
        )
        
        # Tags
        tags = self.document_data.get('tags', [])
        if tags:
            self.tags_label = ctk.CTkLabel(
                info_frame,
                text=f"Tags: {', '.join(tags[:3])}",
                font=ctk.CTkFont(size=9),
                text_color="#2B5278",
                anchor="w"
            )
            self.tags_label.pack(anchor="w", pady=(2, 0))
        
        # Layout
        self.icon_label.pack(side="left", padx=10, pady=10)
        info_frame.pack(side="left", fill="both", expand=True, padx=(0, 10), pady=10)
        self.title_label.pack(anchor="w", pady=(0, 2))
        self.meta_label.pack(anchor="w")
    
    def _get_icon_for_type(self, file_type: str) -> str:
        """Get icon character for file type"""
        icons = {
            '.pdf': '📄',
            '.docx': '📝',
            '.txt': '📄',
            '.epub': '📚',
            '.md': '📝',
            '.html': '🌐',
            '.csv': '📊',
        }
        return icons.get(file_type, '📎')
    
    def _format_metadata(self) -> str:
        """Format document metadata for display"""
        parts = []
        
        if 'file_size' in self.document_data:
            size_mb = self.document_data['file_size'] / (1024 * 1024)
            parts.append(f"{size_mb:.1f}MB")
        
        if 'upload_date' in self.document_data:
            try:
                date = datetime.fromisoformat(self.document_data['upload_date'])
                parts.append(date.strftime("%Y-%m-%d"))
            except:
                pass
        
        if 'chunk_count' in self.document_data:
            parts.append(f"{self.document_data['chunk_count']} chunks")
        
        return " • ".join(parts)
    
    def _on_click(self, event=None):
        """Handle card click"""
        self.selected = not self.selected
        self._update_appearance()
        
        # Notify parent if callback exists
        if hasattr(self.master, 'on_document_select'):
            self.master.on_document_select(self.document_data['id'], self.selected)
    
    def _update_appearance(self):
        """Update card appearance based on selection state"""
        if self.selected:
            self.configure(fg_color="#2B5278")
            self.title_label.configure(text_color="white")
            self.meta_label.configure(text_color="#CCCCCC")
            if hasattr(self, 'tags_label'):
                self.tags_label.configure(text_color="#AAAAAA")
        else:
            self.configure(fg_color=["#F0F0F0", "#2B2B2B"])
            self.title_label.configure(text_color=["#000000", "#FFFFFF"])
            self.meta_label.configure(text_color="gray")
            if hasattr(self, 'tags_label'):
                self.tags_label.configure(text_color="#2B5278")
    
    def get_document_id(self) -> str:
        """Get document ID"""
        return self.document_data.get('id', '')


class DocumentPanel(ctk.CTkFrame):
    """Document management panel with search and batch operations"""
    
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.documents: List[Dict[str, Any]] = []
        self.selected_docs: set = set()
        self.filtered_docs: List[Dict[str, Any]] = []
        
        self._create_widgets()
        self._setup_layout()
        
        logger.debug("Document panel initialized")
    
    def _create_widgets(self):
        """Create document panel widgets"""
        
        # Search and filter bar
        self.search_frame = ctk.CTkFrame(self, height=50)
        
        self.search_entry = ctk.CTkEntry(
            self.search_frame,
            placeholder_text="Search documents...",
            width=300
        )
        
        self.filter_combo = ctk.CTkComboBox(
            self.search_frame,
            values=["All", "PDF", "DOCX", "TXT", "EPUB", "HTML"],
            width=120
        )
        
        self.search_button = ctk.CTkButton(
            self.search_frame,
            text="Search",
            width=80
        )
        
        # Toolbar
        self.toolbar_frame = ctk.CTkFrame(self, height=40)
        
        self.upload_button = ctk.CTkButton(
            self.toolbar_frame,
            text="Upload",
            width=80
        )
        
        self.delete_button = ctk.CTkButton(
            self.toolbar_frame,
            text="Delete",
            width=80,
            fg_color="#8B0000",
            hover_color="#6A0000"
        )
        
        self.tag_button = ctk.CTkButton(
            self.toolbar_frame,
            text="Add Tags",
            width=80
        )
        
        self.select_all_button = ctk.CTkButton(
            self.toolbar_frame,
            text="Select All",
            width=80
        )
        
        # Document grid
        self.documents_frame = ctk.CTkScrollableFrame(self)
        
        # Status bar
        self.status_label = ctk.CTkLabel(
            self,
            text="No documents",
            anchor="w"
        )
        
        # Bind events
        self.search_button.configure(command=self._search_documents)
        self.upload_button.configure(command=self._upload_documents)
        self.delete_button.configure(command=self._delete_selected)
        self.tag_button.configure(command=self._show_tag_dialog)
        self.select_all_button.configure(command=self._select_all)
        
        self.search_entry.bind("<Return>", lambda e: self._search_documents())
    
    def _setup_layout(self):
        """Arrange document panel widgets"""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        
        # Search bar
        self.search_frame.grid(row=0, column=0, sticky="ew", padx=10, pady=10)
        self.search_entry.pack(side="left", padx=(0, 5))
        self.filter_combo.pack(side="left", padx=(0, 5))
        self.search_button.pack(side="left")
        
        # Toolbar
        self.toolbar_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0, 10))
        self.upload_button.pack(side="left", padx=(0, 5))
        self.delete_button.pack(side="left", padx=(0, 5))
        self.tag_button.pack(side="left", padx=(0, 5))
        self.select_all_button.pack(side="left")
        
        # Document grid
        self.documents_frame.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 5))
        
        # Status bar
        self.status_label.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))
    
    def load_documents(self, documents: List[Dict[str, Any]]):
        """Load documents into panel"""
        self.documents = documents
        self.filtered_docs = documents
        
        self._refresh_display()
        self._update_status()
    
    def _refresh_display(self):
        """Refresh document display"""
        # Clear current display
        for widget in self.documents_frame.winfo_children():
            widget.destroy()
        
        # Display documents in grid
        for i, doc in enumerate(self.filtered_docs):
            row = i // 3
            col = i % 3
            
            card = DocumentCard(self.documents_frame, doc)
            card.grid(
                row=row,
                column=col,
                padx=5,
                pady=5,
                sticky="nsew"
            )
            
            # Configure grid columns
            self.documents_frame.grid_columnconfigure(col, weight=1)
        
        logger.debug(f"Displaying {len(self.filtered_docs)} documents")
    
    def _search_documents(self):
        """Search and filter documents"""
        search_term = self.search_entry.get().lower()
        filter_type = self.filter_combo.get()
        
        self.filtered_docs = [
            doc for doc in self.documents
            if self._matches_search(doc, search_term, filter_type)
        ]
        
        self._refresh_display()
        self._update_status()
    
    def _matches_search(self, doc: Dict[str, Any], search_term: str, filter_type: str) -> bool:
        """Check if document matches search criteria"""
        # Filter by type
        if filter_type != "All":
            file_ext = doc.get('file_type', '').lower().replace('.', '')
            if file_ext != filter_type.lower():
                return False
        
        # Search in various fields
        if not search_term:
            return True
        
        search_fields = ['file_name', 'title', 'tags', 'content_preview']
        
        for field in search_fields:
            if field in doc and search_term in str(doc[field]).lower():
                return True
        
        return False
    
    def _upload_documents(self):
        """Upload multiple documents"""
        file_paths = filedialog.askopenfilenames(
            title="Select Documents",
            filetypes=[
                ("All supported files", "*.pdf *.docx *.txt *.epub *.md *.html"),
                ("PDF files", "*.pdf"),
                ("Word documents", "*.docx"),
                ("Text files", "*.txt"),
                ("EPUB files", "*.epub"),
                ("Markdown files", "*.md"),
                ("HTML files", "*.html")
            ]
        )
        
        if file_paths:
            for file_path in file_paths:
                self._add_document(file_path)
    
    def _add_document(self, file_path: str):
        """Add document to panel"""
        doc_data = {
            'id': f"doc_{len(self.documents)}",
            'file_name': Path(file_path).name,
            'file_path': file_path,
            'file_type': Path(file_path).suffix,
            'file_size': Path(file_path).stat().st_size,
            'upload_date': datetime.now().isoformat(),
            'tags': [],
            'chunk_count': 0,
            'processing_status': 'pending'
        }
        
        self.documents.append(doc_data)
        self.filtered_docs.append(doc_data)
        
        self._refresh_display()
        self._update_status()
        
        logger.info(f"Added document: {doc_data['file_name']}")
    
    def on_document_select(self, doc_id: str, selected: bool):
        """Handle document selection"""
        if selected:
            self.selected_docs.add(doc_id)
        else:
            self.selected_docs.discard(doc_id)
        
        self._update_status()
    
    def _delete_selected(self):
        """Delete selected documents"""
        if not self.selected_docs:
            return
        
        # Filter out selected documents
        self.documents = [doc for doc in self.documents if doc['id'] not in self.selected_docs]
        self.filtered_docs = [doc for doc in self.filtered_docs if doc['id'] not in self.selected_docs]
        
        self.selected_docs.clear()
        self._refresh_display()
        self._update_status()
        
        logger.info(f"Deleted {len(self.selected_docs)} documents")
    
    def _show_tag_dialog(self):
        """Show tag management dialog"""
        if not self.selected_docs:
            return
        
        dialog = TagDialog(self, self.selected_docs)
        dialog.grab_set()
    
    def _select_all(self):
        """Select all displayed documents"""
        for widget in self.documents_frame.winfo_children():
            if isinstance(widget, DocumentCard):
                widget.selected = True
                widget._update_appearance()
                self.selected_docs.add(widget.get_document_id())
        
        self._update_status()
    
    def _update_status(self):
        """Update status label"""
        total = len(self.filtered_docs)
        selected = len(self.selected_docs)
        
        if selected > 0:
            self.status_label.configure(text=f"{selected} of {total} documents selected")
        else:
            self.status_label.configure(text=f"{total} documents")
    
    def get_selected_documents(self) -> List[Dict[str, Any]]:
        """Get selected document data"""
        return [
            doc for doc in self.documents
            if doc['id'] in self.selected_docs
        ]


class TagDialog(ctk.CTkToplevel):
    """Tag management dialog"""
    
    def __init__(self, parent, document_ids: set):
        super().__init__(parent)
        
        self.document_ids = document_ids
        self.selected_tags = set()
        
        self.title("Manage Tags")
        self.geometry("400x300")
        
        self._create_widgets()
        self._setup_layout()
    
    def _create_widgets(self):
        """Create tag dialog widgets"""
        self.label = ctk.CTkLabel(
            self,
            text=f"Add tags to {len(self.document_ids)} selected documents",
            font=ctk.CTkFont(size=12)
        )
        
        # Existing tags
        self.tags_frame = ctk.CTkScrollableFrame(self, height=150)
        self._load_existing_tags()
        
        # New tag input
        self.new_tag_frame = ctk.CTkFrame(self)
        
        self.new_tag_entry = ctk.CTkEntry(
            self.new_tag_frame,
            placeholder_text="New tag..."
        )
        
        self.add_tag_button = ctk.CTkButton(
            self.new_tag_frame,
            text="Add",
            width=60
        )
        
        # Action buttons
        self.button_frame = ctk.CTkFrame(self)
        
        self.apply_button = ctk.CTkButton(
            self.button_frame,
            text="Apply",
            width=100
        )
        
        self.cancel_button = ctk.CTkButton(
            self.button_frame,
            text="Cancel",
            width=100,
            fg_color="gray",
            hover_color="#555555"
        )
        
        # Bind events
        self.add_tag_button.configure(command=self._add_new_tag)
        self.apply_button.configure(command=self._apply_tags)
        self.cancel_button.configure(command=self.destroy)
        
        self.new_tag_entry.bind("<Return>", lambda e: self._add_new_tag())
    
    def _setup_layout(self):
        """Arrange tag dialog widgets"""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)
        
        self.label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")
        self.tags_frame.grid(row=1, column=0, padx=20, pady=10, sticky="nsew")
        self.new_tag_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        
        self.new_tag_entry.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.add_tag_button.pack(side="right")
        
        self.button_frame.grid(row=3, column=0, padx=20, pady=(10, 20), sticky="e")
        self.cancel_button.pack(side="right", padx=(5, 0))
        self.apply_button.pack(side="right")
    
    def _load_existing_tags(self):
        """Load existing tags from application"""
        # This would normally load from database
        sample_tags = ["research", "work", "personal", "important", "archive", "todo"]
        
        for tag in sample_tags:
            tag_check = ctk.CTkCheckBox(
                self.tags_frame,
                text=tag,
                command=lambda t=tag: self._toggle_tag(t)
            )
            tag_check.pack(anchor="w", pady=2)
    
    def _toggle_tag(self, tag: str):
        """Toggle tag selection"""
        if tag in self.selected_tags:
            self.selected_tags.remove(tag)
        else:
            self.selected_tags.add(tag)
    
    def _add_new_tag(self):
        """Add new tag from input"""
        new_tag = self.new_tag_entry.get().strip()
        
        if new_tag and new_tag not in self.selected_tags:
            self.selected_tags.add(new_tag)
            
            # Add to display
            tag_check = ctk.CTkCheckBox(
                self.tags_frame,
                text=new_tag,
                command=lambda t=new_tag: self._toggle_tag(t)
            )
            tag_check.pack(anchor="w", pady=2)
            tag_check.select()
            
            self.new_tag_entry.delete(0, "end")
    
    def _apply_tags(self):
        """Apply selected tags to documents"""
        if self.selected_tags:
            # This would normally update database
            print(f"Applying tags {self.selected_tags} to documents {self.document_ids}")
        
        self.destroy()
'''
        
        document_panel_file.write_text(document_panel_content)
    
    def fix_settings_panel(self):
        settings_panel_file = self.project_dir / "src" / "ui" / "desktop" / "settings_panel.py"
        
        settings_panel_content = '''"""
Settings Panel
Application configuration and preferences
"""

import customtkinter as ctk
from typing import Dict, Any, Optional, Callable
import json
from pathlib import Path
from ..core.config import ConfigManager, get_config
from ..core.logger import get_logger

logger = get_logger(__name__)


class SettingsPanel(ctk.CTkFrame):
    """Settings panel for application configuration"""
    
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.config_manager = ConfigManager()
        self.config = self.config_manager.load()
        self.unsaved_changes = False
        
        self._create_widgets()
        self._setup_layout()
        self._load_current_settings()
        
        logger.debug("Settings panel initialized")
    
    def _create_widgets(self):
        """Create settings panel widgets"""
        
        # Notebook-style tabs
        self.tabview = ctk.CTkTabview(self)
        self.general_tab = self.tabview.add("General")
        self.ai_tab = self.tabview.add("AI")
        self.document_tab = self.tabview.add("Documents")
        self.ui_tab = self.tabview.add("UI")
        self.advanced_tab = self.tabview.add("Advanced")
        
        # General settings
        self._create_general_tab()
        
        # AI settings
        self._create_ai_tab()
        
        # Document settings
        self._create_document_tab()
        
        # UI settings
        self._create_ui_tab()
        
        # Advanced settings
        self._create_advanced_tab()
        
        # Action buttons
        self.button_frame = ctk.CTkFrame(self)
        
        self.save_button = ctk.CTkButton(
            self.button_frame,
            text="Save Settings",
            width=120,
            command=self._save_settings
        )
        
        self.reset_button = ctk.CTkButton(
            self.button_frame,
            text="Reset to Defaults",
            width=120,
            fg_color="gray",
            hover_color="#555555",
            command=self._reset_settings
        )
        
        self.cancel_button = ctk.CTkButton(
            self.button_frame,
            text="Cancel",
            width=80,
            fg_color="#8B0000",
            hover_color="#6A0000",
            command=self._cancel_changes
        )
    
    def _create_general_tab(self):
        """Create general settings tab"""
        
        # Application settings
        app_label = ctk.CTkLabel(
            self.general_tab,
            text="Application Settings",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        app_label.pack(anchor="w", padx=20, pady=(20, 10))
        
        # Auto-save
        self.auto_save_var = ctk.BooleanVar(value=True)
        self.auto_save_check = ctk.CTkCheckBox(
            self.general_tab,
            text="Enable auto-save",
            variable=self.auto_save_var,
            command=self._mark_unsaved
        )
        self.auto_save_check.pack(anchor="w", padx=40, pady=5)
        
        # Auto-save interval
        self.auto_save_frame = ctk.CTkFrame(self.general_tab, fg_color="transparent")
        
        auto_save_label = ctk.CTkLabel(
            self.auto_save_frame,
            text="Auto-save interval (minutes):"
        )
        auto_save_label.pack(side="left", padx=(0, 10))
        
        self.auto_save_slider = ctk.CTkSlider(
            self.auto_save_frame,
            from_=1,
            to=60,
            number_of_steps=59,
            command=self._mark_unsaved
        )
        self.auto_save_slider.pack(side="left", fill="x", expand=True)
        
        self.auto_save_value = ctk.CTkLabel(
            self.auto_save_frame,
            text="30",
            width=30
        )
        self.auto_save_value.pack(side="left", padx=(10, 0))
        
        self.auto_save_frame.pack(anchor="w", padx=40, pady=5, fill="x")
        
        # Privacy settings
        privacy_label = ctk.CTkLabel(
            self.general_tab,
            text="Privacy",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        privacy_label.pack(anchor="w", padx=20, pady=(20, 10))
        
        self.telemetry_var = ctk.BooleanVar(value=False)
        self.telemetry_check = ctk.CTkCheckBox(
            self.general_tab,
            text="Send anonymous usage statistics",
            variable=self.telemetry_var,
            command=self._mark_unsaved
        )
        self.telemetry_check.pack(anchor="w", padx=40, pady=5)
        
        self.crash_reports_var = ctk.BooleanVar(value=False)
        self.crash_reports_check = ctk.CTkCheckBox(
            self.general_tab,
            text="Send crash reports",
            variable=self.crash_reports_var,
            command=self._mark_unsaved
        )
        self.crash_reports_check.pack(anchor="w", padx=40, pady=5)
    
    def _create_ai_tab(self):
        """Create AI settings tab"""
        
        # LLM settings
        llm_label = ctk.CTkLabel(
            self.ai_tab,
            text="Language Model",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        llm_label.pack(anchor="w", padx=20, pady=(20, 10))
        
        # Model selection
        model_frame = ctk.CTkFrame(self.ai_tab, fg_color="transparent")
        
        model_label = ctk.CTkLabel(
            model_frame,
            text="Model:"
        )
        model_label.pack(side="left", padx=(0, 10))
        
        self.model_combo = ctk.CTkComboBox(
            model_frame,
            values=["llama2:7b", "mistral:7b", "neural-chat:7b"],
            command=self._mark_unsaved
        )
        self.model_combo.pack(side="left", fill="x", expand=True)
        
        model_frame.pack(anchor="w", padx=40, pady=5, fill="x")
        
        # Temperature
        temp_frame = ctk.CTkFrame(self.ai_tab, fg_color="transparent")
        
        temp_label = ctk.CTkLabel(
            temp_frame,
            text="Temperature:"
        )
        temp_label.pack(side="left", padx=(0, 10))
        
        self.temp_slider = ctk.CTkSlider(
            temp_frame,
            from_=0,
            to=1,
            number_of_steps=100,
            command=self._update_temp_label
        )
        self.temp_slider.pack(side="left", fill="x", expand=True)
        
        self.temp_value = ctk.CTkLabel(
            temp_frame,
            text="0.1",
            width=40
        )
        self.temp_value.pack(side="left", padx=(10, 0))
        
        temp_frame.pack(anchor="w", padx=40, pady=5, fill="x")
        
        # Max tokens
        tokens_frame = ctk.CTkFrame(self.ai_tab, fg_color="transparent")
        
        tokens_label = ctk.CTkLabel(
            tokens_frame,
            text="Max tokens:"
        )
        tokens_label.pack(side="left", padx=(0, 10))
        
        self.tokens_entry = ctk.CTkEntry(
            tokens_frame,
            width=100
        )
        self.tokens_entry.pack(side="left")
        self.tokens_entry.bind("<KeyRelease>", lambda e: self._mark_unsaved())
        
        tokens_frame.pack(anchor="w", padx=40, pady=5)
        
        # Embedding settings
        embed_label = ctk.CTkLabel(
            self.ai_tab,
            text="Embeddings",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        embed_label.pack(anchor="w", padx=20, pady=(20, 10))
        
        # Embedding model
        embed_model_frame = ctk.CTkFrame(self.ai_tab, fg_color="transparent")
        
        embed_model_label = ctk.CTkLabel(
            embed_model_frame,
            text="Model:"
        )
        embed_model_label.pack(side="left", padx=(0, 10))
        
        self.embed_model_combo = ctk.CTkComboBox(
            embed_model_frame,
            values=["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
            command=self._mark_unsaved
        )
        self.embed_model_combo.pack(side="left", fill="x", expand=True)
        
        embed_model_frame.pack(anchor="w", padx=40, pady=5, fill="x")
    
    def _create_document_tab(self):
        """Create document settings tab"""
        
        # Processing settings
        process_label = ctk.CTkLabel(
            self.document_tab,
            text="Document Processing",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        process_label.pack(anchor="w", padx=20, pady=(20, 10))
        
        # Chunk size
        chunk_frame = ctk.CTkFrame(self.document_tab, fg_color="transparent")
        
        chunk_label = ctk.CTkLabel(
            chunk_frame,
            text="Chunk size (tokens):"
        )
        chunk_label.pack(side="left", padx=(0, 10))
        
        self.chunk_size_slider = ctk.CTkSlider(
            chunk_frame,
            from_=100,
            to=1000,
            number_of_steps=18,
            command=self._update_chunk_label
        )
        self.chunk_size_slider.pack(side="left", fill="x", expand=True)
        
        self.chunk_size_value = ctk.CTkLabel(
            chunk_frame,
            text="500",
            width=40
        )
        self.chunk_size_value.pack(side="left", padx=(10, 0))
        
        chunk_frame.pack(anchor="w", padx=40, pady=5, fill="x")
        
        # Chunk overlap
        overlap_frame = ctk.CTkFrame(self.document_tab, fg_color="transparent")
        
        overlap_label = ctk.CTkLabel(
            overlap_frame,
            text="Chunk overlap:"
        )
        overlap_label.pack(side="left", padx=(0, 10))
        
        self.overlap_slider = ctk.CTkSlider(
            overlap_frame,
            from_=0,
            to=200,
            number_of_steps=40,
            command=self._update_overlap_label
        )
        self.overlap_slider.pack(side="left", fill="x", expand=True)
        
        self.overlap_value = ctk.CTkLabel(
            overlap_frame,
            text="50",
            width=40
        )
        self.overlap_value.pack(side="left", padx=(10, 0))
        
        overlap_frame.pack(anchor="w", padx=40, pady=5, fill="x")
        
        # Max file size
        size_frame = ctk.CTkFrame(self.document_tab, fg_color="transparent")
        
        size_label = ctk.CTkLabel(
            size_frame,
            text="Max file size (MB):"
        )
        size_label.pack(side="left", padx=(0, 10))
        
        self.max_size_entry = ctk.CTkEntry(
            size_frame,
            width=100
        )
        self.max_size_entry.pack(side="left")
        self.max_size_entry.bind("<KeyRelease>", lambda e: self._mark_unsaved())
        
        size_frame.pack(anchor="w", padx=40, pady=5)
        
        # OCR settings
        ocr_label = ctk.CTkLabel(
            self.document_tab,
            text="OCR Processing",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        ocr_label.pack(anchor="w", padx=20, pady=(20, 10))
        
        self.ocr_enabled_var = ctk.BooleanVar(value=False)
        self.ocr_check = ctk.CTkCheckBox(
            self.document_tab,
            text="Enable OCR for image processing",
            variable=self.ocr_enabled_var,
            command=self._mark_unsaved
        )
        self.ocr_check.pack(anchor="w", padx=40, pady=5)
    
    def _create_ui_tab(self):
        """Create UI settings tab"""
        
        # Theme settings
        theme_label = ctk.CTkLabel(
            self.ui_tab,
            text="Appearance",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        theme_label.pack(anchor="w", padx=20, pady=(20, 10))
        
        # Theme selection
        theme_frame = ctk.CTkFrame(self.ui_tab, fg_color="transparent")
        
        theme_option_label = ctk.CTkLabel(
            theme_frame,
            text="Theme:"
        )
        theme_option_label.pack(side="left", padx=(0, 10))
        
        self.theme_combo = ctk.CTkComboBox(
            theme_frame,
            values=["dark", "light", "system"],
            command=self._mark_unsaved
        )
        self.theme_combo.pack(side="left", fill="x", expand=True)
        
        theme_frame.pack(anchor="w", padx=40, pady=5, fill="x")
        
        # Language selection
        lang_frame = ctk.CTkFrame(self.ui_tab, fg_color="transparent")
        
        lang_label = ctk.CTkLabel(
            lang_frame,
            text="Language:"
        )
        lang_label.pack(side="left", padx=(0, 10))
        
        self.lang_combo = ctk.CTkComboBox(
            lang_frame,
            values=["en", "id"],
            command=self._mark_unsaved
        )
        self.lang_combo.pack(side="left", fill="x", expand=True)
        
        lang_frame.pack(anchor="w", padx=40, pady=5, fill="x")
        
        # Font size
        font_frame = ctk.CTkFrame(self.ui_tab, fg_color="transparent")
        
        font_label = ctk.CTkLabel(
            font_frame,
            text="Font size:"
        )
        font_label.pack(side="left", padx=(0, 10))
        
        self.font_slider = ctk.CTkSlider(
            font_frame,
            from_=8,
            to=20,
            number_of_steps=12,
            command=self._update_font_label
        )
        self.font_slider.pack(side="left", fill="x", expand=True)
        
        self.font_value = ctk.CTkLabel(
            font_frame,
            text="12",
            width=30
        )
        self.font_value.pack(side="left", padx=(10, 0))
        
        font_frame.pack(anchor="w", padx=40, pady=5, fill="x")
        
        # UI animations
        self.animations_var = ctk.BooleanVar(value=True)
        self.animations_check = ctk.CTkCheckBox(
            self.ui_tab,
            text="Enable UI animations",
            variable=self.animations_var,
            command=self._mark_unsaved
        )
        self.animations_check.pack(anchor="w", padx=40, pady=5)
    
    def _create_advanced_tab(self):
        """Create advanced settings tab"""
        
        # Performance settings
        perf_label = ctk.CTkLabel(
            self.advanced_tab,
            text="Performance",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        perf_label.pack(anchor="w", padx=20, pady=(20, 10))
        
        # Worker threads
        workers_frame = ctk.CTkFrame(self.advanced_tab, fg_color="transparent")
        
        workers_label = ctk.CTkLabel(
            workers_frame,
            text="Max worker threads:"
        )
        workers_label.pack(side="left", padx=(0, 10))
        
        self.workers_slider = ctk.CTkSlider(
            workers_frame,
            from_=1,
            to=8,
            number_of_steps=7,
            command=self._update_workers_label
        )
        self.workers_slider.pack(side="left", fill="x", expand=True)
        
        self.workers_value = ctk.CTkLabel(
            workers_frame,
            text="4",
            width=30
        )
        self.workers_value.pack(side="left", padx=(10, 0))
        
        workers_frame.pack(anchor="w", padx=40, pady=5, fill="x")
        
        # Cache settings
        cache_label = ctk.CTkLabel(
            self.advanced_tab,
            text="Cache",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        cache_label.pack(anchor="w", padx=20, pady=(20, 10))
        
        self.cache_enabled_var = ctk.BooleanVar(value=True)
        self.cache_check = ctk.CTkCheckBox(
            self.advanced_tab,
            text="Enable caching",
            variable=self.cache_enabled_var,
            command=self._mark_unsaved
        )
        self.cache_check.pack(anchor="w", padx=40, pady=5)
        
        # Cache size
        cache_size_frame = ctk.CTkFrame(self.advanced_tab, fg_color="transparent")
        
        cache_size_label = ctk.CTkLabel(
            cache_size_frame,
            text="Cache size (MB):"
        )
        cache_size_label.pack(side="left", padx=(0, 10))
        
        self.cache_size_entry = ctk.CTkEntry(
            cache_size_frame,
            width=100
        )
        self.cache_size_entry.pack(side="left")
        self.cache_size_entry.bind("<KeyRelease>", lambda e: self._mark_unsaved())
        
        cache_size_frame.pack(anchor="w", padx=40, pady=5)
        
        # Storage settings
        storage_label = ctk.CTkLabel(
            self.advanced_tab,
            text="Storage",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        storage_label.pack(anchor="w", padx=20, pady=(20, 10))
        
        self.backup_enabled_var = ctk.BooleanVar(value=True)
        self.backup_check = ctk.CTkCheckBox(
            self.advanced_tab,
            text="Enable automatic backups",
            variable=self.backup_enabled_var,
            command=self._mark_unsaved
        )
        self.backup_check.pack(anchor="w", padx=40, pady=5)
        
        # Auto cleanup
        cleanup_frame = ctk.CTkFrame(self.advanced_tab, fg_color="transparent")
        
        cleanup_label = ctk.CTkLabel(
            cleanup_frame,
            text="Auto-cleanup after (days):"
        )
        cleanup_label.pack(side="left", padx=(0, 10))
        
        self.cleanup_entry = ctk.CTkEntry(
            cleanup_frame,
            width=100
        )
        self.cleanup_entry.pack(side="left")
        self.cleanup_entry.bind("<KeyRelease>", lambda e: self._mark_unsaved())
        
        cleanup_frame.pack(anchor="w", padx=40, pady=5)
    
    def _setup_layout(self):
        """Arrange settings panel widgets"""
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        self.tabview.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        self.button_frame.grid(row=1, column=0, sticky="e", padx=10, pady=(0, 10))
        self.cancel_button.pack(side="right", padx=(5, 0))
        self.reset_button.pack(side="right", padx=(5, 0))
        self.save_button.pack(side="right")
    
    def _load_current_settings(self):
        """Load current configuration into UI"""
        
        # General settings
        self.auto_save_var.set(True)  # Default to true
        self.auto_save_slider.set(30)
        self.telemetry_var.set(False)
        self.crash_reports_var.set(False)
        
        # AI settings
        self.model_combo.set(self.config.llm_model)
        self.temp_slider.set(self.config.llm_temperature)
        self.temp_value.configure(text=f"{self.config.llm_temperature:.1f}")
        self.tokens_entry.insert(0, str(self.config.llm_max_tokens))
        self.embed_model_combo.set(self.config.embedding_model)
        
        # Document settings
        self.chunk_size_slider.set(self.config.chunk_size)
        self.chunk_size_value.configure(text=str(self.config.chunk_size))
        self.overlap_slider.set(self.config.chunk_overlap)
        self.overlap_value.configure(text=str(self.config.chunk_overlap))
        self.max_size_entry.insert(0, str(self.config.max_file_size_mb))
        self.ocr_enabled_var.set(False)  # Default to false
        
        # UI settings
        self.theme_combo.set(self.config.ui_theme)
        self.lang_combo.set(self.config.ui_language)
        self.font_slider.set(self.config.ui_font_size)
        self.font_value.configure(text=str(self.config.ui_font_size))
        self.animations_var.set(True)
        
        # Advanced settings
        self.workers_slider.set(4)
        self.workers_value.configure(text="4")
        self.cache_enabled_var.set(True)
        self.cache_size_entry.insert(0, "500")
        self.backup_enabled_var.set(True)
        self.cleanup_entry.insert(0, "90")
    
    def _mark_unsaved(self, *args):
        """Mark settings as unsaved"""
        self.unsaved_changes = True
    
    def _update_temp_label(self, value):
        """Update temperature label"""
        self.temp_value.configure(text=f"{value:.1f}")
        self._mark_unsaved()
    
    def _update_chunk_label(self, value):
        """Update chunk size label"""
        self.chunk_size_value.configure(text=str(int(value)))
        self._mark_unsaved()
    
    def _update_overlap_label(self, value):
        """Update overlap label"""
        self.overlap_value.configure(text=str(int(value)))
        self._mark_unsaved()
    
    def _update_font_label(self, value):
        """Update font size label"""
        self.font_value.configure(text=str(int(value)))
        self._mark_unsaved()
    
    def _update_workers_label(self, value):
        """Update workers label"""
        self.workers_value.configure(text=str(int(value)))
        self._mark_unsaved()
    
    def _save_settings(self):
        """Save current settings"""
        try:
            # Update config from UI values
            self.config.llm_model = self.model_combo.get()
            self.config.llm_temperature = self.temp_slider.get()
            self.config.llm_max_tokens = int(self.tokens_entry.get())
            self.config.embedding_model = self.embed_model_combo.get()
            
            self.config.chunk_size = int(self.chunk_size_slider.get())
            self.config.chunk_overlap = int(self.overlap_slider.get())
            self.config.max_file_size_mb = int(self.max_size_entry.get())
            
            self.config.ui_theme = self.theme_combo.get()
            self.config.ui_language = self.lang_combo.get()
            self.config.ui_font_size = int(self.font_slider.get())
            
            # Save to file
            if self.config_manager.save():
                self.unsaved_changes = False
                
                # Apply theme change immediately
                import customtkinter as ctk
                ctk.set_appearance_mode(self.config.ui_theme)
                
                logger.info("Settings saved successfully")
                
                # Show success message
                self._show_message("Settings saved successfully", "info")
            else:
                self._show_message("Failed to save settings", "error")
                
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
            self._show_message(f"Error: {str(e)}", "error")
    
    def _reset_settings(self):
        """Reset settings to defaults"""
        # This would reset to hardcoded defaults
        self._load_current_settings()
        self.unsaved_changes = True
        logger.info("Settings reset to defaults")
    
    def _cancel_changes(self):
        """Cancel unsaved changes"""
        if self.unsaved_changes:
            # Show confirmation dialog
            self._load_current_settings()
            self.unsaved_changes = False
            logger.info("Changes cancelled")
    
    def _show_message(self, message: str, message_type: str = "info"):
        """Show status message"""
        # This would show a toast or status message
        print(f"[{message_type.upper()}] {message}")
    
    def apply_theme(self):
        """Apply current theme settings"""
        import customtkinter as ctk
        ctk.set_appearance_mode(self.config.ui_theme)
    
    def get_unsaved_changes(self) -> bool:
        """Check if there are unsaved changes"""
        return self.unsaved_changes


class SettingsDialog(ctk.CTkToplevel):
    """Settings dialog window"""
    
    def __init__(self, parent):
        super().__init__(parent)
        
        self.title("DocuBot Settings")
        self.geometry("800x600")
        
        # Make dialog modal
        self.transient(parent)
        self.grab_set()
        
        self.settings_panel = SettingsPanel(self)
        self.settings_panel.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() // 2) - (self.winfo_width() // 2)
        y = parent.winfo_y() + (parent.winfo_height() // 2) - (self.winfo_height() // 2)
        self.geometry(f"+{x}+{y}")
    
    def close(self):
        """Close settings dialog"""
        if self.settings_panel.get_unsaved_changes():
            # Ask for confirmation
            pass
        self.grab_release()
        self.destroy()
'''
        
        settings_panel_file.write_text(settings_panel_content)
    
    def fix_export_manager(self):
        export_manager_file = self.project_dir / "src" / "ui" / "desktop" / "export_manager.py"
        
        export_manager_content = '''"""
Export Manager
Export conversations to various formats (Markdown, PDF, HTML)
"""

import customtkinter as ctk
from typing import List, Dict, Any, Optional
from pathlib import Path
from tkinter import filedialog
import json
import markdown
from datetime import datetime
from ..core.logger import get_logger

logger = get_logger(__name__)


class ExportFormat:
    """Supported export formats"""
    
    MARKDOWN = "markdown"
    PDF = "pdf"
    HTML = "html"
    JSON = "json"
    
    @staticmethod
    def get_all_formats():
        """Get all supported formats"""
        return [
            ("Markdown", ExportFormat.MARKDOWN, ".md"),
            ("PDF", ExportFormat.PDF, ".pdf"),
            ("HTML", ExportFormat.HTML, ".html"),
            ("JSON", ExportFormat.JSON, ".json")
        ]
    
    @staticmethod
    def get_file_extension(format_type: str) -> str:
        """Get file extension for format"""
        extensions = {
            ExportFormat.MARKDOWN: ".md",
            ExportFormat.PDF: ".pdf",
            ExportFormat.HTML: ".html",
            ExportFormat.JSON: ".json"
        }
        return extensions.get(format_type, ".txt")


class ExportManager(ctk.CTkFrame):
    """Export manager for conversation and document export"""
    
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.conversations: List[Dict[str, Any]] = []
        self.selected_conversations: set = set()
        
        self._create_widgets()
        self._setup_layout()
        
        logger.debug("Export manager initialized")
    
    def _create_widgets(self):
        """Create export manager widgets"""
        
        # Header
        self.header_label = ctk.CTkLabel(
            self,
            text="Export Conversations",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        
        # Conversation selection
        self.selection_frame = ctk.CTkFrame(self)
        
        self.select_all_var = ctk.BooleanVar(value=False)
        self.select_all_check = ctk.CTkCheckBox(
            self.selection_frame,
            text="Select All",
            variable=self.select_all_var,
            command=self._toggle_select_all
        )
        
        self.conversations_listbox = ctk.CTkScrollableFrame(self.selection_frame, height=200)
        
        # Format selection
        self.format_frame = ctk.CTkFrame(self)
        
        format_label = ctk.CTkLabel(
            self.format_frame,
            text="Export Format:",
            font=ctk.CTkFont(size=12, weight="bold")
        )
        
        self.format_var = ctk.StringVar(value=ExportFormat.MARKDOWN)
        self.format_buttons = []
        
        for format_name, format_id, _ in ExportFormat.get_all_formats():
            button = ctk.CTkRadioButton(
                self.format_frame,
                text=format_name,
                variable=self.format_var,
                value=format_id
            )
            self.format_buttons.append(button)
        
        # Options
        self.options_frame = ctk.CTkFrame(self)
        
        self.include_timestamps_var = ctk.BooleanVar(value=True)
        self.include_timestamps_check = ctk.CTkCheckBox(
            self.options_frame,
            text="Include timestamps",
            variable=self.include_timestamps_var
        )
        
        self.include_sources_var = ctk.BooleanVar(value=True)
        self.include_sources_check = ctk.CTkCheckBox(
            self.options_frame,
            text="Include sources",
            variable=self.include_sources_var
        )
        
        self.pretty_format_var = ctk.BooleanVar(value=True)
        self.pretty_format_check = ctk.CTkCheckBox(
            self.options_frame,
            text="Pretty formatting",
            variable=self.pretty_format_var
        )
        
        # Export location
        self.location_frame = ctk.CTkFrame(self)
        
        location_label = ctk.CTkLabel(
            self.location_frame,
            text="Export Location:"
        )
        
        self.location_entry = ctk.CTkEntry(
            self.location_frame,
            placeholder_text="Select export directory..."
        )
        
        self.browse_button = ctk.CTkButton(
            self.location_frame,
            text="Browse",
            width=80,
            command=self._browse_location
        )
        
        # Action buttons
        self.button_frame = ctk.CTkFrame(self)
        
        self.export_button = ctk.CTkButton(
            self.button_frame,
            text="Export Selected",
            width=120,
            command=self._export_selected
        )
        
        self.batch_export_button = ctk.CTkButton(
            self.button_frame,
            text="Batch Export",
            width=120,
            command=self._batch_export
        )
        
        self.cancel_button = ctk.CTkButton(
            self.button_frame,
            text="Cancel",
            width=80,
            fg_color="gray",
            hover_color="#555555",
            command=self._cancel
        )
        
        # Status
        self.status_label = ctk.CTkLabel(
            self,
            text="Ready to export",
            anchor="w"
        )
    
    def _setup_layout(self):
        """Arrange export manager widgets"""
        self.grid_columnconfigure(0, weight=1)
        
        # Header
        self.header_label.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")
        
        # Conversation selection
        self.selection_frame.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        self.selection_frame.grid_columnconfigure(0, weight=1)
        
        self.select_all_check.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")
        self.conversations_listbox.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        
        # Format selection
        self.format_frame.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        format_label.pack(anchor="w", padx=10, pady=(10, 5))
        
        for button in self.format_buttons:
            button.pack(anchor="w", padx=20, pady=2)
        
        # Options
        self.options_frame.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        self.include_timestamps_check.pack(anchor="w", padx=10, pady=5)
        self.include_sources_check.pack(anchor="w", padx=10, pady=5)
        self.pretty_format_check.pack(anchor="w", padx=10, pady=5)
        
        # Location
        self.location_frame.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
        self.location_frame.grid_columnconfigure(1, weight=1)
        
        location_label.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="w")
        self.location_entry.grid(row=0, column=1, padx=5, pady=10, sticky="ew")
        self.browse_button.grid(row=0, column=2, padx=(5, 10), pady=10)
        
        # Buttons
        self.button_frame.grid(row=5, column=0, padx=20, pady=10, sticky="e")
        self.cancel_button.pack(side="right", padx=(5, 0))
        self.batch_export_button.pack(side="right", padx=(5, 0))
        self.export_button.pack(side="right")
        
        # Status
        self.status_label.grid(row=6, column=0, padx=20, pady=(0, 20), sticky="ew")
    
    def load_conversations(self, conversations: List[Dict[str, Any]]):
        """Load conversations for export"""
        self.conversations = conversations
        self._populate_conversations_list()
    
    def _populate_conversations_list(self):
        """Populate conversations listbox"""
        # Clear existing items
        for widget in self.conversations_listbox.winfo_children():
            widget.destroy()
        
        # Add conversations
        for i, conv in enumerate(self.conversations):
            conv_frame = ctk.CTkFrame(self.conversations_listbox)
            
            # Checkbox
            check_var = ctk.BooleanVar(value=False)
            check_button = ctk.CTkCheckBox(
                conv_frame,
                text="",
                variable=check_var,
                width=20,
                command=lambda idx=i, var=check_var: self._on_conversation_select(idx, var)
            )
            
            # Conversation info
            title = conv.get('title', f"Conversation {i+1}")
            date = conv.get('created_at', '')
            if date:
                try:
                    date_str = datetime.fromisoformat(date).strftime("%Y-%m-%d")
                except:
                    date_str = date[:10]
            else:
                date_str = "Unknown date"
            
            message_count = conv.get('message_count', 0)
            
            info_label = ctk.CTkLabel(
                conv_frame,
                text=f"{title} ({message_count} messages) - {date_str}",
                anchor="w"
            )
            
            # Layout
            check_button.pack(side="left", padx=(5, 10), pady=5)
            info_label.pack(side="left", fill="x", expand=True, padx=(0, 10), pady=5)
            
            conv_frame.pack(fill="x", padx=5, pady=2)
            
            # Store reference
            conv['check_var'] = check_var
            conv['check_button'] = check_button
        
        self._update_status()
    
    def _on_conversation_select(self, index: int, check_var):
        """Handle conversation selection"""
        conv = self.conversations[index]
        
        if check_var.get():
            self.selected_conversations.add(conv['id'])
        else:
            self.selected_conversations.discard(conv['id'])
        
        # Update select all checkbox
        all_selected = all(conv.get('check_var', ctk.BooleanVar(value=False)).get() 
                          for conv in self.conversations)
        self.select_all_var.set(all_selected)
        
        self._update_status()
    
    def _toggle_select_all(self):
        """Toggle selection of all conversations"""
        select_all = self.select_all_var.get()
        
        for conv in self.conversations:
            if 'check_var' in conv:
                conv['check_var'].set(select_all)
                
                if select_all:
                    self.selected_conversations.add(conv['id'])
                else:
                    self.selected_conversations.discard(conv['id'])
        
        self._update_status()
    
    def _browse_location(self):
        """Browse for export location"""
        directory = filedialog.askdirectory(
            title="Select Export Directory"
        )
        
        if directory:
            self.location_entry.delete(0, "end")
            self.location_entry.insert(0, directory)
    
    def _export_selected(self):
        """Export selected conversations"""
        if not self.selected_conversations:
            self._show_status("No conversations selected", "warning")
            return
        
        export_dir = self.location_entry.get().strip()
        if not export_dir:
            self._show_status("Please select export directory", "warning")
            return
        
        export_dir_path = Path(export_dir)
        if not export_dir_path.exists():
            export_dir_path.mkdir(parents=True)
        
        format_type = self.format_var.get()
        export_count = 0
        
        for conv in self.conversations:
            if conv['id'] in self.selected_conversations:
                success = self._export_conversation(conv, export_dir_path, format_type)
                if success:
                    export_count += 1
        
        self._show_status(f"Exported {export_count} conversations to {export_dir}", "success")
        logger.info(f"Exported {export_count} conversations to {export_dir}")
    
    def _batch_export(self):
        """Batch export all conversations"""
        if not self.conversations:
            self._show_status("No conversations available", "warning")
            return
        
        export_dir = self.location_entry.get().strip()
        if not export_dir:
            self._show_status("Please select export directory", "warning")
            return
        
        export_dir_path = Path(export_dir)
        if not export_dir_path.exists():
            export_dir_path.mkdir(parents=True)
        
        format_type = self.format_var.get()
        export_count = 0
        
        for conv in self.conversations:
            success = self._export_conversation(conv, export_dir_path, format_type)
            if success:
                export_count += 1
        
        self._show_status(f"Batch exported {export_count} conversations to {export_dir}", "success")
        logger.info(f"Batch exported {export_count} conversations to {export_dir}")
    
    def _export_conversation(self, conversation: Dict[str, Any], 
                            export_dir: Path, format_type: str) -> bool:
        """Export single conversation"""
        try:
            # Get conversation data
            conv_id = conversation['id']
            conv_title = conversation.get('title', f"conversation_{conv_id}")
            
            # Sanitize filename
            safe_title = "".join(c for c in conv_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
            safe_title = safe_title.replace(' ', '_')
            
            # Add timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{safe_title}_{timestamp}"
            
            # Add extension
            extension = ExportFormat.get_file_extension(format_type)
            filepath = export_dir / f"{filename}{extension}"
            
            # Get messages (this would normally come from database)
            messages = conversation.get('messages', [])
            
            # Export based on format
            if format_type == ExportFormat.MARKDOWN:
                content = self._format_markdown(conversation, messages)
            elif format_type == ExportFormat.HTML:
                content = self._format_html(conversation, messages)
            elif format_type == ExportFormat.JSON:
                content = self._format_json(conversation, messages)
            elif format_type == ExportFormat.PDF:
                # PDF requires additional libraries
                content = self._format_markdown(conversation, messages)
                # Convert to PDF would go here
                self._show_status("PDF export requires additional setup", "info")
                return False
            else:
                content = self._format_text(conversation, messages)
            
            # Write file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True
            
        except Exception as e:
            logger.error(f"Export failed for {conversation.get('title')}: {e}")
            return False
    
    def _format_markdown(self, conversation: Dict[str, Any], messages: List[Dict[str, Any]]) -> str:
        """Format conversation as Markdown"""
        lines = []
        
        # Header
        title = conversation.get('title', 'Untitled Conversation')
        lines.append(f"# {title}")
        lines.append("")
        
        # Metadata
        if 'created_at' in conversation:
            date = datetime.fromisoformat(conversation['created_at']).strftime("%Y-%m-%d %H:%M")
            lines.append(f"**Date:** {date}")
        
        if 'message_count' in conversation:
            lines.append(f"**Messages:** {conversation['message_count']}")
        
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Messages
        for msg in messages:
            role = msg.get('role', 'unknown').capitalize()
            content = msg.get('content', '')
            timestamp = msg.get('timestamp', '')
            
            if timestamp and self.include_timestamps_var.get():
                try:
                    time_str = datetime.fromisoformat(timestamp).strftime("%H:%M")
                    lines.append(f"### {role} ({time_str})")
                except:
                    lines.append(f"### {role}")
            else:
                lines.append(f"### {role}")
            
            lines.append("")
            lines.append(content)
            lines.append("")
            
            # Sources
            if self.include_sources_var.get() and 'sources' in msg and msg['sources']:
                lines.append("**Sources:**")
                for source in msg['sources']:
                    lines.append(f"- {source.get('title', 'Unknown')}")
                lines.append("")
        
        return "\\n".join(lines)
    
    def _format_html(self, conversation: Dict[str, Any], messages: List[Dict[str, Any]]) -> str:
        """Format conversation as HTML"""
        # Convert markdown to HTML
        markdown_content = self._format_markdown(conversation, messages)
        html_content = markdown.markdown(markdown_content)
        
        # Wrap in HTML document
        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{conversation.get('title', 'Conversation Export')}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
               line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        h3 {{ color: #7f8c8d; margin-top: 20px; }}
        .message {{ margin: 20px 0; padding: 15px; background: #f8f9fa; border-radius: 5px; }}
        .user {{ border-left: 4px solid #3498db; }}
        .assistant {{ border-left: 4px solid #2ecc71; }}
        .timestamp {{ color: #95a5a6; font-size: 0.9em; }}
        .sources {{ background: #ecf0f1; padding: 10px; border-radius: 3px; margin-top: 10px; }}
    </style>
</head>
<body>
    {html_content}
</body>
</html>"""
        
        return html_template
    
    def _format_json(self, conversation: Dict[str, Any], messages: List[Dict[str, Any]]) -> str:
        """Format conversation as JSON"""
        export_data = {
            'conversation': conversation,
            'messages': messages,
            'export_info': {
                'exported_at': datetime.now().isoformat(),
                'format': 'json',
                'version': '1.0'
            }
        }
        
        if self.pretty_format_var.get():
            return json.dumps(export_data, indent=2, ensure_ascii=False)
        else:
            return json.dumps(export_data, ensure_ascii=False)
    
    def _format_text(self, conversation: Dict[str, Any], messages: List[Dict[str, Any]]) -> str:
        """Format conversation as plain text"""
        lines = []
        
        lines.append(f"Conversation: {conversation.get('title', 'Untitled')}")
        lines.append("=" * 50)
        lines.append("")
        
        for msg in messages:
            role = msg.get('role', 'unknown').upper()
            content = msg.get('content', '')
            
            if self.include_timestamps_var.get() and 'timestamp' in msg:
                try:
                    time_str = datetime.fromisoformat(msg['timestamp']).strftime("%H:%M")
                    lines.append(f"[{time_str}] {role}:")
                except:
                    lines.append(f"{role}:")
            else:
                lines.append(f"{role}:")
            
            lines.append(content)
            lines.append("")
            
            if self.include_sources_var.get() and 'sources' in msg and msg['sources']:
                lines.append("Sources:")
                for source in msg['sources']:
                    lines.append(f"  - {source.get('title', 'Unknown')}")
                lines.append("")
        
        return "\\n".join(lines)
    
    def _update_status(self):
        """Update status label"""
        selected = len(self.selected_conversations)
        total = len(self.conversations)
        
        if selected > 0:
            self.status_label.configure(text=f"{selected} of {total} conversations selected for export")
        else:
            self.status_label.configure(text=f"{total} conversations available")
    
    def _show_status(self, message: str, status_type: str = "info"):
        """Show status message"""
        colors = {
            'info': "#2B5278",
            'success': "#1E6B5E",
            'warning': "#8B4513",
            'error': "#8B0000"
        }
        
        self.status_label.configure(text=message, text_color=colors.get(status_type, "#2B5278"))
    
    def _cancel(self):
        """Cancel export operation"""
        self.master.destroy() if hasattr(self.master, 'destroy') else self.destroy()


class ExportDialog(ctk.CTkToplevel):
    """Export dialog window"""
    
    def __init__(self, parent, conversations: List[Dict[str, Any]]):
        super().__init__(parent)
        
        self.title("Export Conversations")
        self.geometry("700x800")
        
        # Make dialog modal
        self.transient(parent)
        self.grab_set()
        
        self.export_manager = ExportManager(self)
        self.export_manager.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Load conversations
        self.export_manager.load_conversations(conversations)
        
        # Set default location
        default_dir = Path.home() / "Documents" / "DocuBot_Exports"
        self.export_manager.location_entry.insert(0, str(default_dir))
        
        # Center on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() // 2) - (self.winfo_width() // 2)
        y = parent.winfo_y() + (parent.winfo_height() // 2) - (self.winfo_height() // 2)
        self.geometry(f"+{x}+{y}")
    
    def close(self):
        """Close export dialog"""
        self.grab_release()
        self.destroy()
'''
        
        export_manager_file.write_text(export_manager_content)
    
    def fix_ui_components(self):
        components_dir = self.project_dir / "src" / "ui" / "desktop" / "components"
        components_dir.mkdir(exist_ok=True)
        
        # Create __init__.py
        init_file = components_dir / "__init__.py"
        init_content = '''"""
UI Components Package
Reusable UI components for DocuBot
"""

from .file_uploader import FileUploader
from .chat_message import ChatMessageWidget
from .document_card import DocumentCard
from .status_bar import StatusBar

__all__ = [
    'FileUploader',
    'ChatMessageWidget',
    'DocumentCard',
    'StatusBar'
]
'''
        init_file.write_text(init_content)
        
        # Create basic component files
        components = {
            'file_uploader.py': '''"""
File Uploader Component
Drag and drop file upload with progress indication
"""

import customtkinter as ctk
from typing import Callable, List, Optional
from pathlib import Path
from tkinter import filedialog
import threading
from ....core.logger import get_logger

logger = get_logger(__name__)


class FileUploader(ctk.CTkFrame):
    """File uploader with drag and drop support"""
    
    def __init__(self, master, on_upload_complete: Callable = None, **kwargs):
        super().__init__(master, **kwargs)
        
        self.on_upload_complete = on_upload_complete
        self.supported_formats = ['.pdf', '.docx', '.txt', '.epub', '.md', '.html']
        
        self._create_widgets()
        self._setup_drag_drop()
        
        logger.debug("File uploader initialized")
    
    def _create_widgets(self):
        """Create uploader widgets"""
        self.configure(height=200, corner_radius=10, fg_color=["#F0F0F0", "#2B2B2B"])
        
        # Drop zone
        self.drop_zone = ctk.CTkFrame(self, corner_radius=8, fg_color=["#E8E8E8", "#333333"])
        
        self.upload_icon = ctk.CTkLabel(
            self.drop_zone,
            text="📁",
            font=ctk.CTkFont(size=40)
        )
        
        self.upload_label = ctk.CTkLabel(
            self.drop_zone,
            text="Drag & Drop files here",
            font=ctk.CTkFont(size=14, weight="bold")
        )
        
        self.format_label = ctk.CTkLabel(
            self.drop_zone,
            text="Supported: PDF, DOCX, TXT, EPUB, MD, HTML",
            font=ctk.CTkFont(size=11),
            text_color="gray"
        )
        
        self.browse_button = ctk.CTkButton(
            self.drop_zone,
            text="or click to browse",
            width=120,
            command=self._browse_files
        )
        
        # Progress bar
        self.progress_frame = ctk.CTkFrame(self, fg_color="transparent")
        
        self.progress_label = ctk.CTkLabel(
            self.progress_frame,
            text="",
            font=ctk.CTkFont(size=11)
        )
        
        self.progress_bar = ctk.CTkProgressBar(
            self.progress_frame
        )
        self.progress_bar.set(0)
        
        # Layout
        self.drop_zone.pack(fill="both", expand=True, padx=20, pady=20)
        self.upload_icon.pack(pady=(30, 10))
        self.upload_label.pack(pady=(0, 5))
        self.format_label.pack(pady=(0, 15))
        self.browse_button.pack(pady=(0, 30))
        
        self.progress_frame.pack(fill="x", padx=20, pady=(0, 10))
        self.progress_label.pack(anchor="w", pady=(0, 5))
        self.progress_bar.pack(fill="x")
        
        # Initially hide progress
        self.progress_frame.pack_forget()
    
    def _setup_drag_drop(self):
        """Setup drag and drop functionality"""
        # This would require platform-specific implementations
        # For now, we'll use button-based upload
        pass
    
    def _browse_files(self):
        """Browse for files to upload"""
        file_paths = filedialog.askopenfilenames(
            title="Select Documents",
            filetypes=[
                ("All supported files", "*.pdf *.docx *.txt *.epub *.md *.html"),
                ("PDF files", "*.pdf"),
                ("Word documents", "*.docx"),
                ("Text files", "*.txt"),
                ("EPUB files", "*.epub"),
                ("Markdown files", "*.md"),
                ("HTML files", "*.html")
            ]
        )
        
        if file_paths:
            self._process_files(file_paths)
    
    def _process_files(self, file_paths: List[str]):
        """Process uploaded files"""
        self._show_progress("Processing files...")
        
        def process_task():
            total = len(file_paths)
            for i, file_path in enumerate(file_paths, 1):
                self._update_progress(i / total, f"Processing {Path(file_path).name}")
                
                # Simulate processing
                import time
                time.sleep(0.5)
                
                # Call completion callback
                if self.on_upload_complete:
                    self.after(0, self.on_upload_complete, file_path)
            
            self.after(0, self._hide_progress)
        
        threading.Thread(target=process_task, daemon=True).start()
    
    def _show_progress(self, message: str):
        """Show progress bar"""
        self.progress_label.configure(text=message)
        self.progress_frame.pack(fill="x", padx=20, pady=(0, 10))
        self.update()
    
    def _update_progress(self, value: float, message: str):
        """Update progress bar"""
        self.progress_bar.set(value)
        self.progress_label.configure(text=message)
        self.update()
    
    def _hide_progress(self):
        """Hide progress bar"""
        self.progress_frame.pack_forget()
        self.progress_bar.set(0)
        self.progress_label.configure(text="")
    
    def set_supported_formats(self, formats: List[str]):
        """Set supported file formats"""
        self.supported_formats = formats
        
        format_text = "Supported: " + ", ".join([f[1:].upper() for f in formats])
        self.format_label.configure(text=format_text)
''',
            
            'chat_message.py': '''"""
Chat Message Component
Individual chat message display widget
"""

import customtkinter as ctk
from typing import Dict, Any, Optional
from datetime import datetime
from ....core.logger import get_logger

logger = get_logger(__name__)


class ChatMessageWidget(ctk.CTkFrame):
    """Widget for displaying individual chat messages"""
    
    def __init__(self, master, message_data: Dict[str, Any], **kwargs):
        super().__init__(master, **kwargs)
        
        self.message_data = message_data
        self._create_widgets()
        
        logger.debug(f"Chat message widget created for {message_data.get('role', 'unknown')}")
    
    def _create_widgets(self):
        """Create message widgets"""
        role = self.message_data.get('role', 'unknown')
        content = self.message_data.get('content', '')
        timestamp = self.message_data.get('timestamp')
        
        # Configure based on role
        if role == 'user':
            bg_color = ["#E3F2FD", "#0D47A1"]
            text_color = ["#000000", "#FFFFFF"]
            align = "right"
        else:
            bg_color = ["#E8F5E9", "#1B5E20"]
            text_color = ["#000000", "#FFFFFF"]
            align = "left"
        
        self.configure(fg_color=bg_color, corner_radius=10)
        
        # Timestamp
        if timestamp:
            try:
                time_str = datetime.fromisoformat(timestamp).strftime("%H:%M")
                self.time_label = ctk.CTkLabel(
                    self,
                    text=time_str,
                    font=ctk.CTkFont(size=9),
                    text_color=["#666666", "#AAAAAA"]
                )
                
                if align == "right":
                    self.time_label.pack(anchor="e", padx=10, pady=(5, 0))
                else:
                    self.time_label.pack(anchor="w", padx=10, pady=(5, 0))
            except:
                pass
        
        # Content
        self.content_text = ctk.CTkTextbox(
            self,
            height=self._calculate_height(content),
            wrap="word",
            state="disabled",
            fg_color="transparent",
            text_color=text_color,
            font=ctk.CTkFont(size=11)
        )
        
        self.content_text.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Set content
        self.content_text.configure(state="normal")
        self.content_text.insert("1.0", content)
        self.content_text.configure(state="disabled")
    
    def _calculate_height(self, content: str) -> int:
        """Calculate appropriate height for content"""
        lines = len(content) // 60 + 1
        return min(max(lines * 20, 40), 200)
    
    def get_message_data(self) -> Dict[str, Any]:
        """Get message data"""
        return self.message_data
    
    def update_content(self, new_content: str):
        """Update message content"""
        self.message_data['content'] = new_content
        
        self.content_text.configure(state="normal")
        self.content_text.delete("1.0", "end")
        self.content_text.insert("1.0", new_content)
        self.content_text.configure(state="disabled")
        
        # Update height
        new_height = self._calculate_height(new_content)
        self.content_text.configure(height=new_height)
''',
            
            'document_card.py': '''"""
Document Card Component
Card widget for document display in lists
"""

import customtkinter as ctk
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from datetime import datetime
from ....core.logger import get_logger

logger = get_logger(__name__)


class DocumentCard(ctk.CTkFrame):
    """Card widget for displaying document information"""
    
    def __init__(self, master, document_data: Dict[str, Any], 
                 on_select: Callable = None, **kwargs):
        super().__init__(master, **kwargs)
        
        self.document_data = document_data
        self.on_select = on_select
        self.selected = False
        
        self._create_widgets()
        self._bind_events()
        
        logger.debug(f"Document card created for {document_data.get('file_name', 'unknown')}")
    
    def _create_widgets(self):
        """Create document card widgets"""
        self.configure(height=120, corner_radius=8, 
                      fg_color=["#FFFFFF", "#2B2B2B"],
                      border_width=1,
                      border_color=["#E0E0E0", "#404040"])
        
        # File icon based on type
        file_type = self.document_data.get('file_type', '').lower()
        icon_text = self._get_icon_for_type(file_type)
        
        self.icon_label = ctk.CTkLabel(
            self,
            text=icon_text,
            font=ctk.CTkFont(size=24),
            width=60
        )
        
        # Document info
        info_frame = ctk.CTkFrame(self, fg_color="transparent")
        
        self.title_label = ctk.CTkLabel(
            info_frame,
            text=self.document_data.get('file_name', 'Unknown Document'),
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w"
        )
        
        # Metadata
        meta_text = self._format_metadata()
        self.meta_label = ctk.CTkLabel(
            info_frame,
            text=meta_text,
            font=ctk.CTkFont(size=10),
            text_color="gray",
            anchor="w"
        )
        
        # Status
        status = self.document_data.get('processing_status', 'unknown')
        self.status_label = ctk.CTkLabel(
            info_frame,
            text=status.capitalize(),
            font=ctk.CTkFont(size=9, weight="bold"),
            text_color=self._get_status_color(status),
            anchor="w"
        )
        
        # Tags
        tags = self.document_data.get('tags', [])
        if tags:
            tags_text = ", ".join(tags[:3])
            if len(tags) > 3:
                tags_text += "..."
            
            self.tags_label = ctk.CTkLabel(
                info_frame,
                text=tags_text,
                font=ctk.CTkFont(size=9),
                text_color="#2B5278",
                anchor="w"
            )
            self.tags_label.pack(anchor="w", pady=(5, 0))
        
        # Layout
        self.icon_label.pack(side="left", padx=15, pady=15)
        info_frame.pack(side="left", fill="both", expand=True, padx=(0, 15), pady=15)
        self.title_label.pack(anchor="w", pady=(0, 5))
        self.meta_label.pack(anchor="w", pady=(0, 5))
        self.status_label.pack(anchor="w")
    
    def _get_icon_for_type(self, file_type: str) -> str:
        """Get icon character for file type"""
        icons = {
            '.pdf': '📄',
            '.docx': '📝',
            '.txt': '📄',
            '.epub': '📚',
            '.md': '📝',
            '.html': '🌐',
            '.csv': '📊',
        }
        return icons.get(file_type, '📎')
    
    def _format_metadata(self) -> str:
        """Format document metadata for display"""
        parts = []
        
        if 'file_size' in self.document_data:
            size_mb = self.document_data['file_size'] / (1024 * 1024)
            parts.append(f"{size_mb:.1f} MB")
        
        if 'upload_date' in self.document_data:
            try:
                date = datetime.fromisoformat(self.document_data['upload_date'])
                parts.append(date.strftime("%Y-%m-%d"))
            except:
                pass
        
        if 'chunk_count' in self.document_data:
            parts.append(f"{self.document_data['chunk_count']} chunks")
        
        return " • ".join(parts)
    
    def _get_status_color(self, status: str) -> str:
        """Get color for status"""
        colors = {
            'completed': "#1E6B5E",
            'processing': "#8B4513",
            'pending': "#2B5278",
            'failed': "#8B0000",
            'unknown': "gray"
        }
        return colors.get(status, "gray")
    
    def _bind_events(self):
        """Bind mouse events"""
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)
        
        # Bind to all child widgets
        for child in self.winfo_children():
            child.bind("<Enter>", self._on_enter)
            child.bind("<Leave>", self._on_leave)
            child.bind("<Button-1>", self._on_click)
    
    def _on_enter(self, event):
        """Handle mouse enter"""
        self.configure(fg_color=["#F8F8F8", "#333333"])
    
    def _on_leave(self, event):
        """Handle mouse leave"""
        if not self.selected:
            self.configure(fg_color=["#FFFFFF", "#2B2B2B"])
    
    def _on_click(self, event):
        """Handle click"""
        self.selected = not self.selected
        self._update_appearance()
        
        if self.on_select:
            self.on_select(self.document_data['id'], self.selected)
    
    def _update_appearance(self):
        """Update appearance based on selection state"""
        if self.selected:
            self.configure(fg_color=["#E3F2FD", "#0D47A1"])
            self.configure(border_color=["#2196F3", "#1565C0"])
        else:
            self.configure(fg_color=["#FFFFFF", "#2B2B2B"])
            self.configure(border_color=["#E0E0E0", "#404040"])
    
    def get_document_id(self) -> str:
        """Get document ID"""
        return self.document_data.get('id', '')
    
    def set_selected(self, selected: bool):
        """Set selection state"""
        self.selected = selected
        self._update_appearance()
    
    def update_status(self, new_status: str):
        """Update document status"""
        self.document_data['processing_status'] = new_status
        self.status_label.configure(
            text=new_status.capitalize(),
            text_color=self._get_status_color(new_status)
        )
''',
            
            'status_bar.py': '''"""
Status Bar Component
Application status bar with progress indicators
"""

import customtkinter as ctk
from typing import Optional, Dict, Any
from datetime import datetime
from ....core.logger import get_logger

logger = get_logger(__name__)


class StatusBar(ctk.CTkFrame):
    """Status bar for displaying application status and progress"""
    
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        
        self.configure(height=30)
        self.status_messages = []
        self.current_status = "Ready"
        
        self._create_widgets()
        
        logger.debug("Status bar initialized")
    
    def _create_widgets(self):
        """Create status bar widgets"""
        self.grid_columnconfigure(0, weight=1)
        
        # Status label
        self.status_label = ctk.CTkLabel(
            self,
            text=self.current_status,
            anchor="w",
            font=ctk.CTkFont(size=10)
        )
        self.status_label.grid(row=0, column=0, sticky="ew", padx=10)
        
        # Progress indicators
        self.progress_frame = ctk.CTkFrame(self, fg_color="transparent", width=200)
        
        self.progress_label = ctk.CTkLabel(
            self.progress_frame,
            text="",
            font=ctk.CTkFont(size=9),
            text_color="gray"
        )
        
        self.progress_bar = ctk.CTkProgressBar(
            self.progress_frame,
            width=150,
            height=6
        )
        self.progress_bar.set(0)
        
        self.progress_label.pack(side="left", padx=(0, 5))
        self.progress_bar.pack(side="left")
        
        self.progress_frame.grid(row=0, column=1, sticky="e", padx=10)
        self.progress_frame.grid_remove()  # Hide initially
        
        # Memory usage
        self.memory_label = ctk.CTkLabel(
            self,
            text="",
            font=ctk.CTkFont(size=9),
            text_color="gray",
            width=100
        )
        self.memory_label.grid(row=0, column=2, sticky="e", padx=10)
        
        self._update_memory_usage()
    
    def set_status(self, message: str, status_type: str = "info", 
                  duration: Optional[int] = None):
        """Set status message"""
        self.current_status = message
        
        # Set color based on type
        colors = {
            'info': ["#000000", "#FFFFFF"],
            'success': ["#1E6B5E", "#4CAF50"],
            'warning': ["#8B4513", "#FF9800"],
            'error': ["#8B0000", "#F44336"]
        }
        
        color = colors.get(status_type, ["#000000", "#FFFFFF"])
        self.status_label.configure(text=message, text_color=color)
        
        # Log status
        logger.info(f"Status: {message}")
        
        # Store message in history
        self.status_messages.append({
            'message': message,
            'type': status_type,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 100 messages
        if len(self.status_messages) > 100:
            self.status_messages.pop(0)
        
        # Auto-clear if duration specified
        if duration:
            self.after(duration * 1000, lambda: self.clear_status())
    
    def clear_status(self):
        """Clear status message"""
        self.current_status = "Ready"
        self.status_label.configure(text="Ready", 
                                   text_color=["#000000", "#FFFFFF"])
    
    def show_progress(self, message: str, value: float = 0):
        """Show progress indicator"""
        self.progress_label.configure(text=message)
        self.progress_bar.set(value)
        self.progress_frame.grid()
        self.update()
    
    def update_progress(self, value: float, message: Optional[str] = None):
        """Update progress indicator"""
        self.progress_bar.set(value)
        
        if message:
            self.progress_label.configure(text=message)
        
        self.update()
    
    def hide_progress(self):
        """Hide progress indicator"""
        self.progress_frame.grid_remove()
        self.progress_bar.set(0)
        self.progress_label.configure(text="")
    
    def _update_memory_usage(self):
        """Update memory usage display"""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / (1024 * 1024)
            self.memory_label.configure(text=f"Memory: {memory_mb:.1f} MB")
        except ImportError:
            self.memory_label.configure(text="")
        except:
            self.memory_label.configure(text="")
        
        # Update every 5 seconds
        self.after(5000, self._update_memory_usage)
    
    def get_status_history(self) -> list:
        """Get status message history"""
        return self.status_messages
    
    def add_status_listener(self, callback):
        """Add callback for status updates"""
        # This would be implemented for real-time status updates
        pass
    
    def set_theme(self, theme: str):
        """Set status bar theme"""
        if theme == "dark":
            self.configure(fg_color="#1E1E1E")
        else:
            self.configure(fg_color="#F0F0F0")
'''
        }
        
        for filename, content in components.items():
            file_path = components_dir / filename
            file_path.write_text(content)
    
    def fix_themes(self):
        themes_dir = self.project_dir / "src" / "ui" / "desktop" / "themes"
        themes_dir.mkdir(exist_ok=True)
        
        # Create __init__.py
        init_file = themes_dir / "__init__.py"
        init_content = '''"""
UI Themes Package
Color themes for DocuBot application
"""

from .dark_theme import DarkTheme
from .light_theme import LightTheme
from .system_theme import SystemTheme

__all__ = [
    'DarkTheme',
    'LightTheme',
    'SystemTheme'
]
'''
        init_file.write_text(init_content)
        
        # Create theme files
        themes = {
            'dark_theme.py': '''"""
Dark Theme Configuration
"""

DARK_THEME = {
    'name': 'dark',
    'bg_color': '#1E1E1E',
    'fg_color': '#2B2B2B',
    'text_color': '#FFFFFF',
    'accent_color': '#2B5278',
    'secondary_accent': '#1E6B5E',
    'border_color': '#404040',
    'hover_color': '#333333',
    'scrollbar_color': '#555555',
    'success_color': '#1E6B5E',
    'warning_color': '#8B4513',
    'error_color': '#8B0000',
    'info_color': '#2B5278',
    
    'fonts': {
        'default': ('Segoe UI', 11),
        'heading': ('Segoe UI', 14, 'bold'),
        'title': ('Segoe UI', 16, 'bold'),
        'monospace': ('Consolas', 10)
    }
}


class DarkTheme:
    """Dark theme implementation"""
    
    @staticmethod
    def apply(window):
        """Apply dark theme to window"""
        import customtkinter as ctk
        
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Additional theme-specific configurations
        window.configure(fg_color=DARK_THEME['bg_color'])
    
    @staticmethod
    def get_colors():
        """Get theme colors"""
        return DARK_THEME
    
    @staticmethod
    def get_font(font_type='default'):
        """Get theme font"""
        return DARK_THEME['fonts'].get(font_type, DARK_THEME['fonts']['default'])
''',
            
            'light_theme.py': '''"""
Light Theme Configuration
"""

LIGHT_THEME = {
    'name': 'light',
    'bg_color': '#F0F0F0',
    'fg_color': '#FFFFFF',
    'text_color': '#000000',
    'accent_color': '#2196F3',
    'secondary_accent': '#4CAF50',
    'border_color': '#E0E0E0',
    'hover_color': '#F5F5F5',
    'scrollbar_color': '#CCCCCC',
    'success_color': '#4CAF50',
    'warning_color': '#FF9800',
    'error_color': '#F44336',
    'info_color': '#2196F3',
    
    'fonts': {
        'default': ('Segoe UI', 11),
        'heading': ('Segoe UI', 14, 'bold'),
        'title': ('Segoe UI', 16, 'bold'),
        'monospace': ('Consolas', 10)
    }
}


class LightTheme:
    """Light theme implementation"""
    
    @staticmethod
    def apply(window):
        """Apply light theme to window"""
        import customtkinter as ctk
        
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        
        # Additional theme-specific configurations
        window.configure(fg_color=LIGHT_THEME['bg_color'])
    
    @staticmethod
    def get_colors():
        """Get theme colors"""
        return LIGHT_THEME
    
    @staticmethod
    def get_font(font_type='default'):
        """Get theme font"""
        return LIGHT_THEME['fonts'].get(font_type, LIGHT_THEME['fonts']['default'])
''',
            
            'system_theme.py': '''"""
System Theme Configuration
Follows system theme settings
"""

import platform


class SystemTheme:
    """System theme implementation"""
    
    @staticmethod
    def apply(window):
        """Apply system theme to window"""
        import customtkinter as ctk
        
        ctk.set_appearance_mode("system")
        ctk.set_default_color_theme("blue")
    
    @staticmethod
    def get_colors():
        """Get current system colors"""
        # This would detect system theme and return appropriate colors
        import customtkinter as ctk
        
        current_mode = ctk.get_appearance_mode()
        
        if current_mode == "dark":
            from .dark_theme import DARK_THEME
            return DARK_THEME
        else:
            from .light_theme import LIGHT_THEME
            return LIGHT_THEME
    
    @staticmethod
    def get_font(font_type='default'):
        """Get theme font"""
        colors = SystemTheme.get_colors()
        return colors['fonts'].get(font_type, colors['fonts']['default'])
    
    @staticmethod
    def is_dark_mode() -> bool:
        """Check if system is in dark mode"""
        try:
            import customtkinter as ctk
            return ctk.get_appearance_mode() == "dark"
        except:
            return False
    
    @staticmethod
    def get_system_info() -> dict:
        """Get system information"""
        return {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'theme_support': 'native' if platform.system() in ['Windows', 'Darwin'] else 'basic'
        }
'''
        }
        
        for filename, content in themes.items():
            file_path = themes_dir / filename
            file_path.write_text(content)
    
    def update_ui_config(self):
        """Update UI configuration file"""
        config_file = self.project_dir / "data" / "config" / "ui_config.yaml"
        
        if not config_file.exists():
            config_file.parent.mkdir(parents=True, exist_ok=True)
        
        ui_config_content = """# UI Configuration
# This file stores UI preferences and layout settings

# Theme settings
theme:
  current: "dark"
  available: ["dark", "light", "system"]
  auto_detect: true

# Window settings
window:
  width: 1200
  height: 800
  maximized: false
  position:
    x: 100
    y: 100

# Layout settings
layout:
  sidebar_width: 250
  right_sidebar_width: 300
  chat_height: 400
  preview_height: 300

# Chat settings
chat:
  font_size: 12
  line_spacing: 1.2
  show_timestamps: true
  show_sources: true
  max_message_history: 1000

# Document view settings
documents:
  view_mode: "grid"  # grid or list
  items_per_row: 3
  show_thumbnails: true
  show_metadata: true
  sort_by: "date"  # name, date, size, type
  sort_order: "descending"

# Export settings
export:
  default_format: "markdown"
  default_location: "${HOME}/Documents/DocuBot_Exports"
  include_timestamps: true
  include_sources: true
  pretty_formatting: true

# Performance settings
performance:
  enable_animations: true
  animation_speed: "normal"  # fast, normal, slow
  lazy_loading: true
  image_cache_size: 100  # MB

# Accessibility
accessibility:
  high_contrast: false
  large_text: false
  screen_reader_support: false
  keyboard_navigation: true

# Customization
customization:
  accent_color: "#2B5278"
  font_family: "Segoe UI"
  custom_css: ""
  
# Recent files
recent:
  documents: []
  conversations: []
  searches: []
  
# User preferences
preferences:
  auto_save: true
  auto_save_interval: 30  # minutes
  confirm_on_exit: true
  check_for_updates: false
"""
        
        config_file.write_text(ui_config_content)


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Fix UI components implementation")
    parser.add_argument("--dir", default="DocuBot", help="Project directory")
    
    args = parser.parse_args()
    
    fixer = UIComponentsFixer(args.dir)
    fixer.fix_all_ui_components()


if __name__ == "__main__":
    main()