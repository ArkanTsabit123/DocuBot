"""
CLI Output Formatters for DocuBot
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import json
from textwrap import wrap
import sys


class CLIFormatter:
    """Command Line Interface output formatter"""
    
    def __init__(self, width: int = 80):
        """
        Initialize CLI formatter.
        
        Args:
            width: Terminal width for formatting
        """
        self.width = width
    
    def format_header(self, title: str, char: str = "=") -> str:
        """
        Format section header.
        
        Args:
            title: Header title
            char: Character to use for line
            
        Returns:
            Formatted header
        """
        line = char * self.width
        centered = title.center(self.width)
        return f"{line}
{centered}
{line}"
    
    def format_subheader(self, title: str, char: str = "-") -> str:
        """
        Format subheader.
        
        Args:
            title: Subheader title
            char: Character to use for line
            
        Returns:
            Formatted subheader
        """
        return f"{char * 40} {title} {char * (self.width - 42 - len(title))}"
    
    def format_table(self, headers: List[str], rows: List[List[Any]]) -> str:
        """
        Format data as table.
        
        Args:
            headers: Column headers
            rows: Table rows
            
        Returns:
            Formatted table
        """
        if not headers or not rows:
            return ""
        
        col_widths = []
        for i, header in enumerate(headers):
            max_width = len(str(header))
            for row in rows:
                if i < len(row):
                    max_width = max(max_width, len(str(row[i])))
            col_widths.append(min(max_width + 2, 30))
        
        lines = []
        
        header_line = "| " + " | ".join(
            str(header).ljust(width) for header, width in zip(headers, col_widths)
        ) + " |"
        lines.append(header_line)
        
        separator = "+" + "+".join("-" * (width + 2) for width in col_widths) + "+"
        lines.append(separator)
        
        for row in rows:
            row_line = "| " + " | ".join(
                str(cell)[:width].ljust(width) for cell, width in zip(row, col_widths)
            ) + " |"
            lines.append(row_line)
        
        return "
".join(lines)
    
    def format_list(self, items: List[Any], bullet: str = "*") -> str:
        """
        Format list.
        
        Args:
            items: List items
            bullet: Bullet character
            
        Returns:
            Formatted list
        """
        return "
".join(f"  {bullet} {item}" for item in items)
    
    def format_key_value(self, key: str, value: Any, indent: int = 2) -> str:
        """
        Format key-value pair.
        
        Args:
            key: Key name
            value: Value
            indent: Indentation spaces
            
        Returns:
            Formatted key-value
        """
        indent_str = " " * indent
        value_str = str(value)
        
        if len(value_str) > self.width - indent - len(key) - 4:
            wrapped = wrap(value_str, width=self.width - indent - 4)
            value_str = "
" + "
".join(f"{indent_str}    {line}" for line in wrapped)
        
        return f"{indent_str}{key}: {value_str}"
    
    def format_dict(self, data: Dict[str, Any], indent: int = 2) -> str:
        """
        Format dictionary.
        
        Args:
            data: Dictionary to format
            indent: Indentation spaces
            
        Returns:
            Formatted dictionary
        """
        lines = []
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{' ' * indent}{key}:")
                lines.append(self.format_dict(value, indent + 2))
            elif isinstance(value, list):
                lines.append(f"{' ' * indent}{key}:")
                for item in value:
                    lines.append(f"{' ' * (indent + 2)}* {item}")
            else:
                lines.append(self.format_key_value(key, value, indent))
        
        return "
".join(lines)
    
    def format_json(self, data: Any, indent: int = 2) -> str:
        """
        Format as JSON.
        
        Args:
            data: Data to format
            indent: JSON indentation
            
        Returns:
            Formatted JSON
        """
        return json.dumps(data, indent=indent, ensure_ascii=False)
    
    def format_progress_bar(self, current: int, total: int, width: int = 30) -> str:
        """
        Format progress bar.
        
        Args:
            current: Current progress
            total: Total steps
            width: Progress bar width
            
        Returns:
            Formatted progress bar
        """
        if total == 0:
            return "[--------------------] 0/0 (100%)"
        
        percentage = current / total
        filled = int(width * percentage)
        bar = "#" * filled + "-" * (width - filled)
        
        return f"[{bar}] {current}/{total} ({percentage:.1%})"
    
    def format_duration(self, seconds: float) -> str:
        """
        Format duration.
        
        Args:
            seconds: Duration in seconds
            
        Returns:
            Formatted duration
        """
        if seconds < 1:
            return f"{seconds * 1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def format_file_size(self, bytes_size: int) -> str:
        """
        Format file size.
        
        Args:
            bytes_size: Size in bytes
            
        Returns:
            Formatted file size
        """
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_size < 1024.0:
                return f"{bytes_size:.1f} {unit}"
            bytes_size /= 1024.0
        return f"{bytes_size:.1f} TB"
    
    def format_timestamp(self, timestamp: Optional[str] = None) -> str:
        """
        Format timestamp.
        
        Args:
            timestamp: ISO format timestamp
            
        Returns:
            Formatted timestamp
        """
        if timestamp:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        else:
            dt = datetime.now()
        
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    
    def format_document_info(self, document: Dict[str, Any]) -> str:
        """
        Format document information.
        
        Args:
            document: Document dictionary
            
        Returns:
            Formatted document info
        """
        lines = []
        
        lines.append(self.format_header("Document Information"))
        lines.append(f"Name: {document.get('file_name', 'Unknown')}")
        lines.append(f"Type: {document.get('file_type', 'Unknown')}")
        lines.append(f"Size: {self.format_file_size(document.get('file_size', 0))}")
        lines.append(f"Status: {document.get('processing_status', 'Unknown')}")
        lines.append(f"Chunks: {document.get('chunk_count', 0)}")
        lines.append(f"Uploaded: {document.get('upload_date', '')[:10]}")
        
        if document.get('processing_error'):
            lines.append(f"Error: {document.get('processing_error')}")
        
        return "
".join(lines)
    
    def format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """
        Format search results.
        
        Args:
            results: Search results
            
        Returns:
            Formatted results
        """
        if not results:
            return "No results found"
        
        lines = []
        lines.append(self.format_header(f"Search Results ({len(results)} found)"))
        
        for i, result in enumerate(results, 1):
            lines.append(f"{i}. {result.get('file_name', 'Unknown')}")
            lines.append(f"   Type: {result.get('file_type', 'Unknown')}")
            lines.append(f"   Score: {result.get('similarity', 0):.3f}")
            
            if 'excerpt' in result:
                excerpt = result['excerpt']
                if len(excerpt) > 100:
                    excerpt = excerpt[:100] + "..."
                lines.append(f"   Excerpt: {excerpt}")
            
            lines.append("")
        
        return "
".join(lines)
    
    def format_statistics(self, stats: Dict[str, Any]) -> str:
        """
        Format statistics.
        
        Args:
            stats: Statistics dictionary
            
        Returns:
            Formatted statistics
        """
        lines = []
        lines.append(self.format_header("Statistics"))
        
        lines.append(f"Documents: {stats.get('total_documents', 0)}")
        lines.append(f"Chunks: {stats.get('total_chunks', 0)}")
        lines.append(f"Conversations: {stats.get('total_conversations', 0)}")
        
        if 'database_size_bytes' in stats:
            lines.append(f"Database Size: {self.format_file_size(stats['database_size_bytes'])}")
        
        if 'documents_by_status' in stats:
            lines.append("")
            lines.append(self.format_subheader("Document Status"))
            for status, count in stats['documents_by_status'].items():
                lines.append(f"  {status}: {count}")
        
        return "
".join(lines)
    
    def print_colored(self, text: str, color: str = "default") -> None:
        """
        Print colored text.
        
        Args:
            text: Text to print
            color: Color name
        """
        colors = {
            'red': '[91m',
            'green': '[92m',
            'yellow': '[93m',
            'blue': '[94m',
            'magenta': '[95m',
            'cyan': '[96m',
            'white': '[97m',
            'default': '[0m'
        }
        
        end_color = '[0m'
        
        if color in colors and sys.stdout.isatty():
            print(f"{colors[color]}{text}{end_color}")
        else:
            print(text)


_cli_formatter = None

def get_cli_formatter(width: int = 80) -> CLIFormatter:
    """
    Get or create CLIFormatter instance.
    
    Args:
        width: Terminal width
        
    Returns:
        CLIFormatter instance
    """
    global _cli_formatter
    
    if _cli_formatter is None:
        _cli_formatter = CLIFormatter(width)
    
    return _cli_formatter


if __name__ == "__main__":
    formatter = CLIFormatter()
    
    print(formatter.format_header("DocuBot CLI"))
    print()
    
    headers = ["ID", "Name", "Status", "Size"]
    rows = [
        ["1", "document.pdf", "processed", "1.2 MB"],
        ["2", "notes.txt", "pending", "45 KB"],
        ["3", "report.docx", "failed", "2.5 MB"]
    ]
    
    print(formatter.format_table(headers, rows))
    print()
    
    print(formatter.format_key_value("Version", "1.0.0"))
    print(formatter.format_key_value("Model", "llama2:7b"))
    print()
    
    stats = {
        'total_documents': 15,
        'total_chunks': 125,
        'total_conversations': 8,
        'database_size_bytes': 1024 * 1024 * 50,
        'documents_by_status': {
            'completed': 12,
            'pending': 2,
            'failed': 1
        }
    }
    
    print(formatter.format_statistics(stats))
