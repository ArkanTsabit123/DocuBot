"""
Document List Components for Streamlit
"""

import streamlit as st
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime


class DocumentListUI:
    """Document list interface components"""
    
    def __init__(self):
        pass
    
    def display_document_table(self, documents: List[Dict[str, Any]]):
        """
        Display documents in a table.
        
        Args:
            documents: List of document dictionaries
        """
        if not documents:
            st.info("No documents available")
            return
        
        df_data = []
        for doc in documents:
            df_data.append({
                'Name': doc.get('file_name', 'Unknown'),
                'Type': doc.get('file_type', 'Unknown'),
                'Size (KB)': doc.get('file_size', 0) / 1024,
                'Status': doc.get('processing_status', 'Unknown'),
                'Chunks': doc.get('chunk_count', 0),
                'Uploaded': doc.get('upload_date', '')[:10] if doc.get('upload_date') else ''
            })
        
        df = pd.DataFrame(df_data)
        
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Size (KB)': st.column_config.NumberColumn(format="%.1f KB"),
                'Chunks': st.column_config.NumberColumn(format="%d")
            }
        )
    
    def display_document_filters(self):
        """
        Display document filters.
        
        Returns:
            Filter criteria
        """
        col1, col2, col3 = st.columns(3)
        
        with col1:
            file_type = st.multiselect(
                "File Type",
                ['PDF', 'DOCX', 'TXT', 'MD', 'HTML'],
                default=[]
            )
        
        with col2:
            status = st.multiselect(
                "Status",
                ['pending', 'processing', 'completed', 'failed'],
                default=['completed']
            )
        
        with col3:
            sort_by = st.selectbox(
                "Sort By",
                ['upload_date', 'file_name', 'file_size', 'chunk_count'],
                index=0
            )
        
        return {
            'file_type': file_type,
            'status': status,
            'sort_by': sort_by
        }
    
    def display_document_actions(self, document_id: str):
        """
        Display document action buttons.
        
        Args:
            document_id: Document ID
            
        Returns:
            Action performed
        """
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("View", key=f"view_{document_id}", use_container_width=True):
                return "view"
        
        with col2:
            if st.button("Delete", key=f"delete_{document_id}", use_container_width=True):
                return "delete"
        
        with col3:
            if st.button("Reprocess", key=f"reprocess_{document_id}", use_container_width=True):
                return "reprocess"
        
        return None
    
    def display_document_preview(self, document: Dict[str, Any]):
        """
        Display document preview.
        
        Args:
            document: Document dictionary
        """
        with st.expander("Document Details", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**File Name:** {document.get('file_name', 'Unknown')}")
                st.markdown(f"**File Type:** {document.get('file_type', 'Unknown')}")
                st.markdown(f"**File Size:** {document.get('file_size', 0) / 1024:.1f} KB")
            
            with col2:
                st.markdown(f"**Status:** {document.get('processing_status', 'Unknown')}")
                st.markdown(f"**Chunks:** {document.get('chunk_count', 0)}")
                st.markdown(f"**Words:** {document.get('word_count', 0)}")
            
            if document.get('metadata'):
                st.divider()
                st.subheader("Metadata")
                st.json(document.get('metadata', {}))
            
            if document.get('tags'):
                st.divider()
                st.subheader("Tags")
                tags = document.get('tags', [])
                tag_chips = " ".join([f"`{tag}`" for tag in tags])
                st.markdown(tag_chips)
            
            if document.get('summary'):
                st.divider()
                st.subheader("Summary")
                st.markdown(document.get('summary', ''))
    
    def display_batch_operations(self):
        """
        Display batch operation controls.
        
        Returns:
            Batch operation to perform
        """
        st.subheader("Batch Operations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Select All", use_container_width=True):
                return "select_all"
        
        with col2:
            if st.button("Process Selected", use_container_width=True):
                return "process_selected"
        
        with col3:
            if st.button("Delete Selected", use_container_width=True):
                return "delete_selected"
        
        return None
    
    def display_upload_progress(self, current: int, total: int, filename: str):
        """
        Display upload progress.
        
        Args:
            current: Current file number
            total: Total files
            filename: Current filename
        """
        progress = current / total
        st.progress(progress, text=f"Uploading {filename} ({current}/{total})")
    
    def display_processing_progress(self, document_name: str, step: str, progress: float):
        """
        Display document processing progress.
        
        Args:
            document_name: Document name
            step: Current processing step
            progress: Progress
        """
        st.progress(progress, text=f"Processing {document_name}: {step}")
    
    def display_document_stats(self, stats: Dict[str, Any]):
        """
        Display document statistics.
        
        Args:
            stats: Statistics dictionary
        """
        st.subheader("Document Statistics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total", stats.get('total', 0))
        
        with col2:
            st.metric("Processed", stats.get('processed', 0))
        
        with col3:
            st.metric("Pending", stats.get('pending', 0))
        
        with col4:
            st.metric("Failed", stats.get('failed', 0))
        
        if 'by_type' in stats:
            st.divider()
            st.subheader("By File Type")
            
            type_data = stats['by_type']
            df = pd.DataFrame(list(type_data.items()), columns=['Type', 'Count'])
            st.bar_chart(df.set_index('Type'))
    
    def display_search_bar(self):
        """
        Display document search bar.
        
        Returns:
            Search query
        """
        return st.text_input(
            "Search Documents",
            placeholder="Search by filename, content, or tags...",
            key="document_search"
        )
    
    def display_empty_state(self, message: str = "No documents found"):
        """
        Display empty state.
        
        Args:
            message: Message to display
        """
        st.markdown(f"""
        <div style='text-align: center; padding: 40px;'>
            <h3>{message}</h3>
            <p>Upload documents to get started</p>
        </div>
        """, unsafe_allow_html=True)


_doc_list_ui = None

def get_document_list_ui() -> DocumentListUI:
    """
    Get or create DocumentListUI instance.
    
    Returns:
        DocumentListUI instance
    """
    global _doc_list_ui
    
    if _doc_list_ui is None:
        _doc_list_ui = DocumentListUI()
    
    return _doc_list_ui
