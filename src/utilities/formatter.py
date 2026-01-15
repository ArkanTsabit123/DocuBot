"""
Text Formatting Utilities
"""

import textwrap
import re
from typing import List, Dict, Any
from datetime import datetime


def wrap_text(text: str, width: int = 80) -> str:
    return textwrap.fill(text, width=width)


def format_timestamp(timestamp: str = None, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    if timestamp:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    else:
        dt = datetime.now()
    
    return dt.strftime(format_str)


def format_file_size(bytes_size: int) -> str:
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"


def format_duration(seconds: float) -> str:
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


def format_list(items: List[Any], bullet: str = "•", indent: int = 2) -> str:
    indent_str = " " * indent
    return "
".join(f"{indent_str}{bullet} {item}" for item in items)


def format_dict(data: Dict[str, Any], indent: int = 2) -> str:
    indent_str = " " * indent
    lines = []
    
    for key, value in data.items():
        if isinstance(value, dict):
            value_str = format_dict(value, indent + 2)
            lines.append(f"{indent_str}{key}:")
            lines.append(value_str)
        elif isinstance(value, list):
            value_str = format_list(value, indent=indent + 2)
            lines.append(f"{indent_str}{key}:")
            lines.append(value_str)
        else:
            lines.append(f"{indent_str}{key}: {value}")
    
    return "
".join(lines)


def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    text = text.replace('
', '
').replace('', '
')
    text = re.sub(r'
\s*
', '

', text)
    
    return text


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)].rstrip() + suffix


def format_markdown_table(headers: List[str], rows: List[List[Any]]) -> str:
    table = ["| " + " | ".join(headers) + " |"]
    table.append("| " + " | ".join(["---"] * len(headers)) + " |")
    
    for row in rows:
        table.append("| " + " | ".join(str(cell) for cell in row) + " |")
    
    return "
".join(table)


def format_json(data: Any, indent: int = 2) -> str:
    import json
    return json.dumps(data, indent=indent, ensure_ascii=False)


def format_code_block(code: str, language: str = "python") -> str:
    return f"```{language}
{code}
```"
