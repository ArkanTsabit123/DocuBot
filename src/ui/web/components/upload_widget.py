# docubot/src/ui/web/components/upload_widget.py

"""
Document Upload Widget
"""

import streamlit as st
from typing import List, Optional
import time

def upload_widget(
    accepted_types: List[str] = None,
    multiple: bool = True,
    process_ocr_default: bool = False
):
    """
    Create a document upload widget
    
    Args:
        accepted_types: List of file extensions to accept
        multiple: Allow multiple file upload
        process_ocr_default: Default value for OCR checkbox
    
    Returns:
        Tuple of (uploaded_files, process_ocr, should_process)
    """
    if accepted_types is None:
        accepted_types = ['.pdf', '.txt', '.docx', '.md', '.html', '.epub']
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose files",
        type=accepted_types,
        accept_multiple_files=multiple,
        help=f"Supported formats: {', '.join(accepted_types)}"
    )
    
    # Options
    col1, col2 = st.columns(2)
    with col1:
        process_ocr = st.checkbox(
            "Enable OCR",
            value=process_ocr_default,
            help="Extract text from images in documents"
        )
    
    with col2:
        chunk_size = st.selectbox(
            "Chunk Size",
            options=[250, 500, 1000, 2000],
            index=1,
            help="Size of text chunks for processing (tokens)"
        )
    
    # Process button
    should_process = False
    if uploaded_files:
        should_process = st.button(
            "Process Uploads",
            type="primary",
            use_container_width=True
        )
    
    return uploaded_files, process_ocr, chunk_size, should_process

def display_upload_progress(file_count: int):
    """
    Display upload progress
    
    Args:
        file_count: Number of files being uploaded
    
    Returns:
        Progress bar and status text
    """
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    return progress_bar, status_text

def update_progress(
    progress_bar,
    status_text,
    current: int,
    total: int,
    filename: str
):
    """
    Update progress display
    
    Args:
        progress_bar: Streamlit progress bar
        status_text: Streamlit status text
        current: Current file index (1-based)
        total: Total number of files
        filename: Current filename
    """
    progress = current / total
    progress_bar.progress(progress)
    status_text.text(f"Processing {filename} ({current}/{total})")