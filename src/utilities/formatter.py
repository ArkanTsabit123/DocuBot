# docubot/src/utilities/formatter.py

"""
Complete Formatter Utilities Module for DocuBot

Provides comprehensive formatting utilities for text, documents, chat,
markdown, and various output formats. Supports the complete DocuBot
ecosystem including document processing and UI display.
"""

import os
import re
import json
import textwrap
import html
from datetime import datetime, date, time
from typing import Any, Dict, List, Optional, Union, Tuple, Callable
from pathlib import Path
from enum import Enum
from dataclasses import dataclass, asdict, field
import xml.etree.ElementTree as ET
from html import escape


class FormatType(Enum):
    """Supported format types."""
    PLAIN_TEXT = "plain_text"
    MARKDOWN = "markdown"
    HTML = "html"
    JSON = "json"
    XML = "xml"
    CSV = "csv"
    RICH_TEXT = "rich_text"
    TERMINAL = "terminal"


class LanguageCode(Enum):
    """Supported language codes."""
    ENGLISH = "en"
    INDONESIAN = "id"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    CHINESE = "zh"
    JAPANESE = "ja"


@dataclass
class FormattingOptions:
    """Complete configuration for formatting operations."""
    # Text formatting
    max_line_length: int = 80
    indent_spaces: int = 4
    tab_width: int = 4
    preserve_line_breaks: bool = True
    collapse_multiple_spaces: bool = True
    trim_trailing_whitespace: bool = True
    normalize_quotes: bool = True
    convert_dashes: bool = True
    
    # Output settings
    default_format: FormatType = FormatType.PLAIN_TEXT
    encoding: str = "utf-8"
    line_ending: str = "\n"
    ensure_ascii: bool = False
    
    # Language and localization
    language: LanguageCode = LanguageCode.ENGLISH
    locale: str = "en_US"
    timezone: str = "UTC"
    
    # Date/time formatting
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    datetime_format: str = "%Y-%m-%d %H:%M:%S"
    
    # Number formatting
    decimal_separator: str = "."
    thousands_separator: str = ","
    decimal_places: int = 2
    
    # Document-specific
    page_width: int = 210  # A4 width in mm
    page_height: int = 297  # A4 height in mm
    margin_left: int = 20
    margin_right: int = 20
    margin_top: int = 20
    margin_bottom: int = 20
    
    # UI display
    truncate_length: int = 200
    ellipsis_text: str = "..."
    show_line_numbers: bool = False
    syntax_highlighting: bool = True
    
    # Table formatting
    table_border: bool = True
    table_header: bool = True
    table_alignment: str = "left"  # left, center, right, justify
    
    # Export settings
    include_metadata: bool = True
    include_timestamps: bool = True
    include_source_info: bool = True


class BaseFormatter:
    """Base formatter class with common utilities."""
    
    def __init__(self, options: Optional[FormattingOptions] = None):
        self.options = options or FormattingOptions()
        self._setup_locale()
    
    def _setup_locale(self):
        """Setup locale-specific formatting."""
        # This would typically use locale.setlocale() in production
        pass
    
    def clean_text(self, text: str, preserve_formatting: bool = False) -> str:
        """Clean and normalize text with various options."""
        if not text:
            return ""
        
        cleaned = text.strip()
        
        # Remove BOM if present
        if cleaned.startswith('\ufeff'):
            cleaned = cleaned[1:]
        
        if self.options.collapse_multiple_spaces and not preserve_formatting:
            cleaned = re.sub(r'\s+', ' ', cleaned)
        
        if self.options.normalize_quotes:
            # Normalize various quote types to standard ASCII
            quote_pairs = [
                ('"', '"'),  # Straight double quotes
                ("'", "'"),  # Straight single quotes
                ('"', '"'),  # Curly double quotes
                ("'", "'"),  # Curly single quotes
            ]
            for smart, straight in quote_pairs:
                cleaned = cleaned.replace(smart, straight)
        
        if self.options.convert_dashes:
            # Convert various dash types to hyphen
            dash_types = ['–', '—', '―', '‒']
            for dash in dash_types:
                cleaned = cleaned.replace(dash, '-')
        
        if self.options.trim_trailing_whitespace:
            lines = cleaned.split('\n')
            cleaned = '\n'.join(line.rstrip() for line in lines)
        
        return cleaned
    
    def format_paragraph(self, text: str, max_length: Optional[int] = None,
                        justify: bool = False) -> str:
        """Format a paragraph with optional justification."""
        if max_length is None:
            max_length = self.options.max_line_length
        
        cleaned = self.clean_text(text)
        
        if justify:
            return textwrap.fill(
                cleaned,
                width=max_length,
                expand_tabs=False,
                replace_whitespace=True,
                drop_whitespace=True,
                break_long_words=True,
                break_on_hyphens=True,
                subsequent_indent=''
            )
        else:
            return textwrap.fill(
                cleaned,
                width=max_length,
                expand_tabs=False,
                replace_whitespace=True,
                drop_whitespace=True,
                break_long_words=True,
                break_on_hyphens=True
            )
    
    def truncate_with_ellipsis(self, text: str, max_chars: Optional[int] = None,
                              position: str = "end") -> str:
        """Truncate text with ellipsis at specified position."""
        if max_chars is None:
            max_chars = self.options.truncate_length
        
        if not text or len(text) <= max_chars:
            return text or ""
        
        ellipsis = self.options.ellipsis_text
        
        if max_chars <= len(ellipsis):
            return ellipsis[:max_chars]
        
        if position == "start":
            return ellipsis + text[-(max_chars - len(ellipsis)):]
        elif position == "middle":
            half = (max_chars - len(ellipsis)) // 2
            return text[:half] + ellipsis + text[-(max_chars - len(ellipsis) - half):]
        else:  # end
            return text[:max_chars - len(ellipsis)] + ellipsis
    
    def format_number(self, number: Union[int, float], 
                     as_integer: bool = False) -> str:
        """Format number with proper separators."""
        if as_integer:
            num_str = f"{int(number):,}"
        else:
            num_str = f"{number:,.{self.options.decimal_places}f}"
        
        # Replace separators based on locale
        if self.options.thousands_separator != ",":
            num_str = num_str.replace(",", self.options.thousands_separator)
        if self.options.decimal_separator != ".":
            num_str = num_str.replace(".", self.options.decimal_separator)
        
        return num_str
    
    def format_datetime(self, dt: Union[str, datetime, date, time, float, int],
                       format_str: Optional[str] = None) -> str:
        """Format datetime object to string."""
        if isinstance(dt, datetime):
            dt_obj = dt
        elif isinstance(dt, date):
            dt_obj = datetime.combine(dt, datetime.min.time())
        elif isinstance(dt, time):
            dt_obj = datetime.combine(date.today(), dt)
        elif isinstance(dt, (int, float)):
            dt_obj = datetime.fromtimestamp(dt)
        elif isinstance(dt, str):
            try:
                # Try ISO format first
                dt_obj = datetime.fromisoformat(dt.replace('Z', '+00:00'))
            except ValueError:
                # Try common formats
                for fmt in ["%Y-%m-%d %H:%M:%S", "%Y/%m/%d %H:%M:%S",
                           "%d-%m-%Y %H:%M:%S", "%d/%m/%Y %H:%M:%S"]:
                    try:
                        dt_obj = datetime.strptime(dt, fmt)
                        break
                    except ValueError:
                        continue
                else:
                    return dt  # Return original if can't parse
        else:
            return str(dt)
        
        if format_str is None:
            if isinstance(dt, time):
                format_str = self.options.time_format
            elif isinstance(dt, date) and not isinstance(dt, datetime):
                format_str = self.options.date_format
            else:
                format_str = self.options.datetime_format
        
        return dt_obj.strftime(format_str)


class TextFormatter(BaseFormatter):
    """Advanced text formatting with extensive utilities."""
    
    def format_list(self, items: List[Any], 
                   bullet_type: str = "bullet",  # bullet, number, letter, dash
                   start_index: int = 1,
                   indent_level: int = 0) -> str:
        """Format list with various bullet types."""
        if not items:
            return ""
        
        indent = " " * (indent_level * self.options.indent_spaces)
        
        formatted_items = []
        for i, item in enumerate(items, start=start_index):
            item_text = str(item).strip()
            if not item_text:
                continue
            
            if bullet_type == "number":
                prefix = f"{i}."
            elif bullet_type == "letter":
                prefix = f"{chr(96 + i)}." if i <= 26 else f"{i}."
            elif bullet_type == "dash":
                prefix = "-"
            else:  # bullet
                prefix = "•"
            
            formatted_items.append(f"{indent}{prefix} {item_text}")
        
        return '\n'.join(formatted_items)
    
    def format_table(self, data: List[List[Any]], 
                    headers: Optional[List[str]] = None,
                    alignments: Optional[List[str]] = None) -> str:
        """Format tabular data as ASCII table."""
        if not data:
            return ""
        
        if alignments is None:
            alignments = ['left'] * len(data[0])
        
        if headers:
            table_data = [headers] + data
        else:
            table_data = data
        
        # Calculate column widths
        column_widths = []
        for col_idx in range(len(table_data[0])):
            max_width = 0
            for row in table_data:
                cell_text = str(row[col_idx]) if col_idx < len(row) else ""
                max_width = max(max_width, len(cell_text))
            column_widths.append(max_width + 2)  # Add padding
        
        # Build table
        lines = []
        
        if headers and self.options.table_border:
            # Top border
            top_border = "┌" + "┬".join("─" * w for w in column_widths) + "┐"
            lines.append(top_border)
        
        # Headers
        if headers:
            header_cells = []
            for i, header in enumerate(headers):
                width = column_widths[i]
                alignment = alignments[i] if i < len(alignments) else 'left'
                
                if alignment == 'center':
                    cell = header.center(width - 2)
                elif alignment == 'right':
                    cell = header.rjust(width - 2)
                else:
                    cell = header.ljust(width - 2)
                
                header_cells.append(f" {cell} ")
            
            line = "│" + "│".join(header_cells) + "│"
            lines.append(line)
            
            if self.options.table_border:
                # Separator
                separator = "├" + "┼".join("─" * w for w in column_widths) + "┤"
                lines.append(separator)
        
        # Data rows
        for row in data:
            row_cells = []
            for i, cell in enumerate(row):
                width = column_widths[i]
                cell_text = str(cell)
                alignment = alignments[i] if i < len(alignments) else 'left'
                
                if alignment == 'center':
                    cell_formatted = cell_text.center(width - 2)
                elif alignment == 'right':
                    cell_formatted = cell_text.rjust(width - 2)
                else:
                    cell_formatted = cell_text.ljust(width - 2)
                
                row_cells.append(f" {cell_formatted} ")
            
            line = "│" + "│".join(row_cells) + "│"
            lines.append(line)
        
        if self.options.table_border:
            # Bottom border
            bottom_border = "└" + "┴".join("─" * w for w in column_widths) + "┘"
            lines.append(bottom_border)
        
        return '\n'.join(lines)
    
    def format_code_block(self, code: str, language: str = "",
                         line_numbers: Optional[bool] = None) -> str:
        """Format code block with optional line numbers and language."""
        if line_numbers is None:
            line_numbers = self.options.show_line_numbers
        
        lines = code.rstrip().split('\n')
        
        if line_numbers:
            # Calculate width for line numbers
            max_line_num = len(lines)
            line_num_width = len(str(max_line_num))
            
            formatted_lines = []
            for i, line in enumerate(lines, 1):
                line_num = f"{i:{line_num_width}}"
                formatted_lines.append(f"{line_num} │ {line}")
            
            formatted_code = '\n'.join(formatted_lines)
        else:
            formatted_code = '\n'.join(lines)
        
        if language:
            return f"```{language}\n{formatted_code}\n```"
        else:
            return f"```\n{formatted_code}\n```"
    
    def generate_table_of_contents(self, headings: List[Dict[str, Any]],
                                  max_depth: int = 3) -> str:
        """Generate table of contents from headings."""
        if not headings:
            return ""
        
        toc_lines = ["## Table of Contents", ""]
        
        current_depth = 0
        for heading in headings:
            level = heading.get('level', 1)
            text = heading.get('text', '')
            anchor = heading.get('anchor', '')
            
            if level > max_depth:
                continue
            
            indent = "  " * (level - 1)
            if anchor:
                toc_lines.append(f"{indent}- [{text}](#{anchor})")
            else:
                # Create anchor from text
                anchor_text = re.sub(r'[^\w\s-]', '', text.lower())
                anchor_text = re.sub(r'[-\s]+', '-', anchor_text).strip('-')
                toc_lines.append(f"{indent}- [{text}](#{anchor_text})")
        
        return '\n'.join(toc_lines)


class DocumentFormatter(BaseFormatter):
    """Specialized formatter for document processing and display."""
    
    def format_document_metadata(self, metadata: Dict[str, Any]) -> str:
        """Format document metadata for display."""
        lines = ["## Document Information", ""]
        
        # Basic info
        if 'title' in metadata:
            lines.append(f"**Title:** {metadata['title']}")
        if 'author' in metadata:
            lines.append(f"**Author:** {metadata['author']}")
        if 'date' in metadata:
            lines.append(f"**Date:** {self.format_datetime(metadata['date'])}")
        
        # File info
        if 'file_name' in metadata:
            lines.append(f"**File:** {metadata['file_name']}")
        if 'file_size' in metadata:
            lines.append(f"**Size:** {self.format_file_size(metadata['file_size'])}")
        if 'file_type' in metadata:
            lines.append(f"**Type:** {metadata['file_type'].upper()}")
        
        # Processing info
        if 'chunk_count' in metadata:
            lines.append(f"**Chunks:** {metadata['chunk_count']}")
        if 'word_count' in metadata:
            lines.append(f"**Words:** {self.format_number(metadata['word_count'])}")
        if 'language' in metadata:
            lines.append(f"**Language:** {metadata['language']}")
        
        # Timestamps
        if 'upload_date' in metadata:
            lines.append(f"**Uploaded:** {self.format_datetime(metadata['upload_date'])}")
        if 'processed_at' in metadata:
            lines.append(f"**Processed:** {self.format_datetime(metadata['processed_at'])}")
        
        # Tags
        if 'tags' in metadata:
            tags = metadata['tags']
            if isinstance(tags, list):
                lines.append(f"**Tags:** {', '.join(tags)}")
        
        # Summary
        if 'summary' in metadata and metadata['summary']:
            lines.extend(["", "### Summary", metadata['summary']])
        
        return '\n'.join(lines)
    
    def format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        if size_bytes < 0:
            return "0 B"
        
        units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
        unit_index = 0
        
        size = float(size_bytes)
        while size >= 1024.0 and unit_index < len(units) - 1:
            size /= 1024.0
            unit_index += 1
        
        if unit_index == 0:
            return f"{int(size)} {units[unit_index]}"
        elif size < 10:
            return f"{size:.2f} {units[unit_index]}"
        elif size < 100:
            return f"{size:.1f} {units[unit_index]}"
        else:
            return f"{int(size)} {units[unit_index]}"
    
    def format_chunk_preview(self, chunk: Dict[str, Any], 
                            max_lines: int = 5,
                            show_metadata: bool = True) -> str:
        """Format a document chunk for preview."""
        lines = []
        
        if show_metadata:
            metadata = chunk.get('metadata', {})
            if metadata:
                source = metadata.get('source_file', 'Unknown')
                chunk_idx = metadata.get('chunk_index', 0)
                total = metadata.get('total_chunks', 1)
                lines.append(f"**Chunk {chunk_idx + 1}/{total} from {source}**")
                lines.append("")
        
        text = chunk.get('text', '')
        if text:
            # Show first few lines
            text_lines = text.split('\n')
            preview_lines = text_lines[:max_lines]
            preview_text = '\n'.join(preview_lines)
            
            if len(text_lines) > max_lines:
                preview_text += f"\n{self.options.ellipsis_text}"
            
            lines.append(preview_text)
        
        # Add relevance score if available
        score = chunk.get('score')
        if score is not None:
            lines.append(f"\n*Relevance: {score:.2%}*")
        
        return '\n'.join(lines)
    
    def format_citation(self, source: Dict[str, Any], 
                       format_type: str = "inline") -> str:
        """Format citation for sources."""
        if format_type == "inline":
            # Inline citation like [Author, Year]
            author = source.get('author', 'Unknown')
            year = source.get('year', '')
            if year:
                return f"[{author}, {year}]"
            else:
                return f"[{author}]"
        
        elif format_type == "footnote":
            # Footnote style
            author = source.get('author', 'Unknown')
            title = source.get('title', 'Unknown Title')
            year = source.get('year', '')
            pages = source.get('pages', '')
            
            parts = [author]
            if year:
                parts.append(f"({year})")
            parts.append(f"\"{title}\"")
            if pages:
                parts.append(f"pp. {pages}")
            
            return ' '.join(parts)
        
        else:  # full
            # Full citation
            lines = []
            if 'author' in source:
                lines.append(f"**Author:** {source['author']}")
            if 'title' in source:
                lines.append(f"**Title:** {source['title']}")
            if 'year' in source:
                lines.append(f"**Year:** {source['year']}")
            if 'publisher' in source:
                lines.append(f"**Publisher:** {source['publisher']}")
            if 'url' in source:
                lines.append(f"**URL:** {source['url']}")
            if 'pages' in source:
                lines.append(f"**Pages:** {source['pages']}")
            
            return '\n'.join(lines)


class ChatMessageFormatter(BaseFormatter):
    """Formatter for chat messages and conversations."""
    
    def __init__(self, options: Optional[FormattingOptions] = None):
        super().__init__(options)
        self.role_prefixes = {
            'user': '👤 You: ',
            'assistant': '🤖 DocuBot: ',
            'system': '⚙️ System: ',
            'error': '❌ Error: '
        }
    
    def format_message(self, message: Dict[str, Any], 
                      include_timestamp: Optional[bool] = None) -> str:
        """Format a single chat message."""
        if include_timestamp is None:
            include_timestamp = self.options.include_timestamps
        
        role = message.get('role', 'user')
        content = message.get('content', '')
        timestamp = message.get('timestamp')
        
        prefix = self.role_prefixes.get(role, '')
        
        # Format timestamp if present
        time_str = ""
        if include_timestamp and timestamp:
            time_str = self.format_datetime(timestamp, "%H:%M")
            prefix = f"[{time_str}] {prefix}"
        
        # Format content based on type
        if isinstance(content, dict):
            # Handle structured content
            if 'text' in content:
                formatted_content = content['text']
            elif 'answer' in content:
                formatted_content = content['answer']
                if 'sources' in content and content['sources']:
                    formatted_content += "\n\n**Sources:**\n"
                    for source in content['sources'][:3]:
                        formatted_content += f"- {source}\n"
            else:
                formatted_content = str(content)
        else:
            formatted_content = str(content)
        
        # Clean and format
        formatted_content = self.clean_text(formatted_content)
        
        # Add prefix to first line, indent subsequent lines
        lines = formatted_content.split('\n')
        formatted_lines = []
        
        for i, line in enumerate(lines):
            if i == 0:
                formatted_lines.append(f"{prefix}{line}")
            else:
                # Align with prefix length
                indent = " " * len(prefix)
                formatted_lines.append(f"{indent}{line}")
        
        return '\n'.join(formatted_lines)
    
    def format_conversation(self, messages: List[Dict[str, Any]],
                           include_metadata: Optional[bool] = None) -> str:
        """Format complete conversation."""
        if include_metadata is None:
            include_metadata = self.options.include_metadata
        
        formatted_messages = []
        
        # Add conversation header
        if include_metadata and messages:
            first_msg_time = messages[0].get('timestamp')
            last_msg_time = messages[-1].get('timestamp')
            
            if first_msg_time and last_msg_time:
                header = f"## Conversation ({len(messages)} messages)"
                formatted_messages.append(header)
                formatted_messages.append("")
        
        # Format each message
        for msg in messages:
            formatted_messages.append(self.format_message(msg))
            formatted_messages.append("")  # Add spacing between messages
        
        return '\n'.join(formatted_messages).strip()
    
    def format_source_reference(self, source: Dict[str, Any], 
                               format_type: str = "compact") -> str:
        """Format source reference for chat responses."""
        if format_type == "compact":
            # Compact format: [Document Name, Page X]
            doc_name = source.get('document_name', 'Unknown Document')
            page = source.get('page')
            confidence = source.get('confidence')
            
            if page:
                ref = f"[{doc_name}, Page {page}]"
            else:
                ref = f"[{doc_name}]"
            
            if confidence:
                ref += f" ({confidence:.0%})"
            
            return ref
        
        elif format_type == "detailed":
            # Detailed format with preview
            doc_name = source.get('document_name', 'Unknown Document')
            page = source.get('page')
            text = source.get('text', '')
            confidence = source.get('confidence')
            
            lines = [f"**📄 {doc_name}**"]
            
            if page:
                lines.append(f"*Page {page}*")
            
            if confidence:
                lines.append(f"*Relevance: {confidence:.0%}*")
            
            if text:
                preview = self.truncate_with_ellipsis(text, 150)
                lines.append(f"> {preview}")
            
            return '\n'.join(lines)
        
        else:  # inline
            return source.get('document_name', 'Source')


class ExportFormatter(BaseFormatter):
    """Formatter for exporting content to various formats."""
    
    def to_markdown(self, content: Any, metadata: Optional[Dict] = None) -> str:
        """Convert content to Markdown format."""
        lines = []
        
        # Add metadata as frontmatter
        if metadata:
            lines.append("---")
            for key, value in metadata.items():
                if value is not None:
                    lines.append(f"{key}: {value}")
            lines.append("---")
            lines.append("")
        
        # Convert content based on type
        if isinstance(content, str):
            lines.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    # Handle message dict
                    if 'role' in item and 'content' in item:
                        role = item['role']
                        msg_content = item['content']
                        lines.append(f"### {role.capitalize()}")
                        lines.append("")
                        lines.append(msg_content)
                        lines.append("")
                    else:
                        lines.append(str(item))
                else:
                    lines.append(f"- {item}")
        elif isinstance(content, dict):
            # Convert dict to markdown
            for key, value in content.items():
                lines.append(f"## {key}")
                lines.append("")
                if isinstance(value, (list, dict)):
                    lines.append(self.to_markdown(value))
                else:
                    lines.append(str(value))
                lines.append("")
        else:
            lines.append(str(content))
        
        return '\n'.join(lines).strip()
    
    def to_html(self, content: Any, metadata: Optional[Dict] = None) -> str:
        """Convert content to HTML format."""
        lines = ['<!DOCTYPE html>', '<html>', '<head>']
        
        # Add metadata
        if metadata:
            lines.append('<meta charset="UTF-8">')
            if 'title' in metadata:
                lines.append(f'<title>{html.escape(metadata["title"])}</title>')
            if 'author' in metadata:
                lines.append(f'<meta name="author" content="{html.escape(metadata["author"])}">')
            if 'date' in metadata:
                lines.append(f'<meta name="date" content="{html.escape(str(metadata["date"]))}">')
        
        lines.append('<style>')
        lines.append('body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }')
        lines.append('h1, h2, h3 { color: #333; }')
        lines.append('.message { margin-bottom: 20px; padding: 10px; border-left: 3px solid #007bff; }')
        lines.append('.user { background-color: #f0f8ff; }')
        lines.append('.assistant { background-color: #f9f9f9; }')
        lines.append('.source { font-size: 0.9em; color: #666; margin-top: 5px; }')
        lines.append('</style>')
        lines.append('</head>')
        lines.append('<body>')
        
        # Convert content
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and 'role' in item and 'content' in item:
                    role = item['role']
                    msg_content = html.escape(str(item['content']))
                    lines.append(f'<div class="message {role}">')
                    lines.append(f'<strong>{role.capitalize()}:</strong>')
                    lines.append(f'<div>{msg_content}</div>')
                    
                    # Add sources if present
                    if 'sources' in item and item['sources']:
                        lines.append('<div class="source">')
                        lines.append('<strong>Sources:</strong>')
                        for source in item['sources']:
                            lines.append(f'<div>{html.escape(str(source))}</div>')
                        lines.append('</div>')
                    
                    lines.append('</div>')
                else:
                    lines.append(f'<p>{html.escape(str(item))}</p>')
        elif isinstance(content, str):
            # Convert markdown-like formatting to HTML
            html_content = html.escape(content)
            html_content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html_content)
            html_content = re.sub(r'\*(.+?)\*', r'<em>\1</em>', html_content)
            html_content = re.sub(r'`(.+?)`', r'<code>\1</code>', html_content)
            html_content = re.sub(r'^# (.+)$', r'<h1>\1</h1>', html_content, flags=re.MULTILINE)
            html_content = re.sub(r'^## (.+)$', r'<h2>\1</h2>', html_content, flags=re.MULTILINE)
            html_content = re.sub(r'^### (.+)$', r'<h3>\1</h3>', html_content, flags=re.MULTILINE)
            lines.append(html_content)
        else:
            lines.append(f'<p>{html.escape(str(content))}</p>')
        
        lines.append('</body>')
        lines.append('</html>')
        
        return '\n'.join(lines)
    
    def to_json(self, content: Any, metadata: Optional[Dict] = None) -> str:
        """Convert content to JSON format."""
        export_data = {}
        
        if metadata:
            export_data['metadata'] = metadata
        
        if isinstance(content, (dict, list, str, int, float, bool, type(None))):
            export_data['content'] = content
        else:
            export_data['content'] = str(content)
        
        return json.dumps(export_data, indent=2, ensure_ascii=self.options.ensure_ascii)
    
    def to_plain_text(self, content: Any, metadata: Optional[Dict] = None) -> str:
        """Convert content to plain text format."""
        lines = []
        
        if metadata:
            lines.append("=" * 60)
            for key, value in metadata.items():
                if value is not None:
                    lines.append(f"{key}: {value}")
            lines.append("=" * 60)
            lines.append("")
        
        # Convert based on type
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    # Handle chat messages
                    if 'role' in item and 'content' in item:
                        role = item['role'].upper()
                        msg_content = str(item['content'])
                        lines.append(f"[{role}]")
                        lines.append(msg_content)
                        lines.append("")
                    else:
                        lines.append(str(item))
                else:
                    lines.append(f"- {item}")
        elif isinstance(content, dict):
            for key, value in content.items():
                lines.append(f"{key.upper()}:")
                lines.append(str(value))
                lines.append("")
        else:
            lines.append(str(content))
        
        return '\n'.join(lines)


class FormatterFactory:
    """Factory for creating and managing formatters."""
    
    _formatters = {}
    
    @classmethod
    def get_formatter(cls, formatter_type: str = "text", 
                     options: Optional[FormattingOptions] = None) -> BaseFormatter:
        """Get formatter instance by type."""
        if formatter_type not in cls._formatters:
            cls._create_formatter(formatter_type, options)
        
        return cls._formatters[formatter_type]
    
    @classmethod
    def _create_formatter(cls, formatter_type: str, 
                         options: Optional[FormattingOptions] = None):
        """Create a new formatter instance."""
        formatter_classes = {
            'text': TextFormatter,
            'document': DocumentFormatter,
            'chat': ChatMessageFormatter,
            'export': ExportFormatter,
            'markdown': TextFormatter,  # Reuse TextFormatter with markdown capabilities
        }
        
        formatter_class = formatter_classes.get(formatter_type, TextFormatter)
        cls._formatters[formatter_type] = formatter_class(options)
    
    @classmethod
    def format_content(cls, content: Any, format_type: str = "text",
                      formatter_type: str = "text", **kwargs) -> str:
        """Format content using specified formatter and output format."""
        formatter = cls.get_formatter(formatter_type)
        
        if format_type == "markdown":
            if hasattr(formatter, 'to_markdown'):
                return formatter.to_markdown(content, **kwargs)
        elif format_type == "html":
            if hasattr(formatter, 'to_html'):
                return formatter.to_html(content, **kwargs)
        elif format_type == "json":
            if hasattr(formatter, 'to_json'):
                return formatter.to_json(content, **kwargs)
        elif format_type == "plain_text":
            if hasattr(formatter, 'to_plain_text'):
                return formatter.to_plain_text(content, **kwargs)
        
        # Fallback to string representation
        return str(content)


# Convenience functions for common operations
def format_text(text: str, max_length: Optional[int] = None, **kwargs) -> str:
    """Convenience function to format text."""
    formatter = FormatterFactory.get_formatter('text')
    return formatter.format_paragraph(text, max_length, **kwargs)


def format_document_metadata(metadata: Dict[str, Any]) -> str:
    """Convenience function to format document metadata."""
    formatter = FormatterFactory.get_formatter('document')
    return formatter.format_document_metadata(metadata)


def format_chat_message(message: Dict[str, Any], **kwargs) -> str:
    """Convenience function to format chat message."""
    formatter = FormatterFactory.get_formatter('chat')
    return formatter.format_message(message, **kwargs)


def export_to_markdown(content: Any, metadata: Optional[Dict] = None) -> str:
    """Convenience function to export content to markdown."""
    formatter = FormatterFactory.get_formatter('export')
    return formatter.to_markdown(content, metadata)


# Default instances for common use cases
default_text_formatter = TextFormatter()
default_document_formatter = DocumentFormatter()
default_chat_formatter = ChatMessageFormatter()
default_export_formatter = ExportFormatter()