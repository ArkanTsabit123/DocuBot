"""
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
        
        return "\n".join(lines)
    
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
        
        return "\n".join(lines)
    
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
