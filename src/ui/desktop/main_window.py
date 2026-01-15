"""
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
        self.documents_listbox.insert("end", f"• {doc_name}\n")
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
        
        self.chat_display.insert("end", f"{prefix}{message}\n\n")
        
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
            self.input_text.insert("insert", "\n")
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
