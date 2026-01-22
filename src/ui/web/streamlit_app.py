# docubot/src/ui/web/streamlit_app.py

"""
Streamlit Web Interface for DocuBot
Web-based interface for document upload and querying
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
from typing import List, Dict, Any
import os
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Page configuration
st.set_page_config(
    page_title="DocuBot - AI Document Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #374151;
        margin-bottom: 0.5rem;
    }
    .info-box {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #D1FAE5;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FEF3C7;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .chat-message-user {
        background-color: #3B82F6;
        color: white;
        padding: 0.75rem;
        border-radius: 1rem 1rem 0 1rem;
        margin-bottom: 0.5rem;
        max-width: 80%;
        margin-left: auto;
    }
    .chat-message-assistant {
        background-color: #E5E7EB;
        color: #1F2937;
        padding: 0.75rem;
        border-radius: 1rem 1rem 1rem 0;
        margin-bottom: 0.5rem;
        max-width: 80%;
        margin-right: auto;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'api_url' not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"
if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'system_status' not in st.session_state:
    st.session_state.system_status = None
if 'selected_docs' not in st.session_state:
    st.session_state.selected_docs = []

def check_api_health():
    """Check if API is reachable"""
    try:
        response = requests.get(f"{st.session_state.api_url}/api/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_system_status():
    """Get system status from API"""
    try:
        response = requests.get(f"{st.session_state.api_url}/api/v1/system", timeout=10)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None

def process_query(query: str):
    """Send query to API and get response"""
    try:
        payload = {
            "query": query,
            "context_length": st.session_state.get('context_length', 500),
            "temperature": st.session_state.get('temperature', 0.1),
            "include_sources": st.session_state.get('include_sources', True)
        }
        
        response = requests.post(
            f"{st.session_state.api_url}/api/v1/query",
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API error: {response.status_code}"}
    except Exception as e:
        return {"error": f"Connection error: {str(e)}"}

def upload_document(file, process_ocr: bool = False):
    """Upload document to API"""
    try:
        files = {"file": (file.name, file.getvalue(), file.type)}
        
        response = requests.post(
            f"{st.session_state.api_url}/api/v1/upload",
            files=files,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Upload failed: {response.status_code}"}
    except Exception as e:
        return {"error": f"Upload error: {str(e)}"}

def get_documents():
    """Get list of documents from API"""
    try:
        response = requests.get(
            f"{st.session_state.api_url}/api/v1/documents",
            params={"limit": 100},
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return []
    except:
        return []

def get_models():
    """Get available models from API"""
    try:
        response = requests.get(
            f"{st.session_state.api_url}/api/v1/models",
            timeout=5
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"models": []}
    except:
        return {"models": []}

# Sidebar
with st.sidebar:
    st.markdown("<div class='main-header'>📚 DocuBot</div>", unsafe_allow_html=True)
    st.markdown("---")
    
    # API Configuration
    st.markdown("<div class='sub-header'>API Configuration</div>", unsafe_allow_html=True)
    
    api_url = st.text_input(
        "API URL",
        value=st.session_state.api_url,
        help="URL of the FastAPI backend"
    )
    
    if api_url != st.session_state.api_url:
        st.session_state.api_url = api_url
    
    # Check API Health
    if st.button("Check Connection", use_container_width=True):
        if check_api_health():
            st.success("API is healthy")
            st.session_state.system_status = get_system_status()
        else:
            st.error("Cannot connect to API")
    
    # System Status
    if st.session_state.system_status:
        st.markdown("---")
        st.markdown("<div class='sub-header'>System Status</div>", unsafe_allow_html=True)
        
        status = st.session_state.system_status.get('system', {}).get('status', 'UNKNOWN')
        if status == 'HEALTHY':
            st.success("System running")
        else:
            st.warning(f"System status: {status}")
        
        # Show models
        models_data = get_models()
        if models_data.get('models'):
            st.info(f"Models: {len(models_data['models'])} available")
    
    # AI Settings
    st.markdown("---")
    st.markdown("<div class='sub-header'>AI Settings</div>", unsafe_allow_html=True)
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="Higher values make output more creative"
    )
    st.session_state.temperature = temperature
    
    context_length = st.slider(
        "Context Length",
        min_value=100,
        max_value=2000,
        value=500,
        step=100,
        help="Context window size in tokens"
    )
    st.session_state.context_length = context_length
    
    include_sources = st.checkbox(
        "Include Sources",
        value=True,
        help="Include source documents in responses"
    )
    st.session_state.include_sources = include_sources
    
    # Document Selection
    st.markdown("---")
    st.markdown("<div class='sub-header'>Documents</div>", unsafe_allow_html=True)
    
    if st.button("Refresh Documents", use_container_width=True):
        st.session_state.documents = get_documents()
    
    if st.session_state.documents:
        doc_options = [f"{doc['filename']} ({doc['file_type']})" for doc in st.session_state.documents]
        selected_docs = st.multiselect(
            "Select documents to query",
            options=doc_options,
            default=st.session_state.selected_docs,
            help="Leave empty to search all documents"
        )
        st.session_state.selected_docs = selected_docs
    else:
        st.info("No documents available")
    
    # About
    st.markdown("---")
    st.markdown("<div class='sub-header'>About</div>", unsafe_allow_html=True)
    st.markdown("""
    **DocuBot** is a local AI document assistant that:
    
    • Processes PDF, DOCX, TXT, HTML, MD, EPUB files
    • Answers questions using RAG (Retrieval Augmented Generation)
    • Runs 100% locally with Ollama LLMs
    • Stores data in ChromaDB vector database
    
    Version: 1.0.0
    """)

# Main Content
st.markdown("<div class='main-header'>DocuBot - AI Document Assistant</div>", unsafe_allow_html=True)
st.markdown("Upload documents and ask questions in natural language")

# Tabs
tab1, tab2, tab3 = st.tabs(["Chat", "Documents", "System"])

with tab1:
    # Chat Interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("<div class='sub-header'>Ask a Question</div>", unsafe_allow_html=True)
        
        question = st.text_area(
            "Enter your question",
            height=100,
            placeholder="Ask anything about your uploaded documents..."
        )
        
        col_a, col_b = st.columns([1, 1])
        with col_a:
            submit_question = st.button("Get Answer", type="primary", use_container_width=True)
        with col_b:
            clear_chat = st.button("Clear Chat", use_container_width=True)
        
        if clear_chat:
            st.session_state.chat_history = []
            st.rerun()
    
    with col2:
        st.markdown("<div class='sub-header'>Selected Documents</div>", unsafe_allow_html=True)
        if st.session_state.selected_docs:
            for doc in st.session_state.selected_docs[:5]:
                st.markdown(f"• {doc}")
            if len(st.session_state.selected_docs) > 5:
                st.caption(f"+ {len(st.session_state.selected_docs) - 5} more")
        else:
            st.info("All documents")
    
    # Process Question
    if submit_question and question:
        # Add user question to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": question,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        
        # Get answer
        with st.spinner("Thinking..."):
            result = process_query(question)
        
        # Add assistant response to chat history
        if "error" not in result:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result.get("answer", "No answer returned"),
                "sources": result.get("sources", []),
                "processing_time": result.get("processing_time", 0),
                "model_used": result.get("model_used"),
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
        else:
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": f"Error: {result['error']}",
                "timestamp": datetime.now().strftime("%H:%M:%S")
            })
        
        st.rerun()
    
    # Display Chat History
    st.markdown("<div class='sub-header'>Conversation</div>", unsafe_allow_html=True)
    
    if st.session_state.chat_history:
        for message in reversed(st.session_state.chat_history[-10:]):
            if message["role"] == "user":
                st.markdown(f"""
                <div class='chat-message-user'>
                    <strong>You</strong> ({message['timestamp']})<br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
            else:
                # Assistant message
                st.markdown(f"""
                <div class='chat-message-assistant'>
                    <strong>DocuBot</strong> ({message['timestamp']})<br>
                    {message['content']}
                </div>
                """, unsafe_allow_html=True)
                
                # Show sources if available
                if "sources" in message and message["sources"]:
                    with st.expander(f"Sources ({len(message['sources'])})"):
                        for i, source in enumerate(message["sources"], 1):
                            st.markdown(f"**{i}. {source.get('filename', 'Unknown')}**")
                            if source.get('similarity'):
                                st.caption(f"Relevance: {source.get('similarity'):.2%}")
                            if source.get('content'):
                                st.text(source.get('content', '')[:200] + "...")
                
                # Show processing info
                if "processing_time" in message:
                    st.caption(f"Processed in {message['processing_time']:.2f} seconds")
                if "model_used" in message and message["model_used"]:
                    st.caption(f"Model: {message['model_used']}")
    else:
        st.info("Start a conversation by asking a question about your documents.")

with tab2:
    # Document Management
    st.markdown("<div class='sub-header'>Document Management</div>", unsafe_allow_html=True)
    
    # Upload Section
    st.markdown("#### Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'txt', 'docx', 'md', 'html', 'epub'],
        accept_multiple_files=True,
        help="Upload documents for processing"
    )
    
    if uploaded_files and st.button("Process Uploads", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, file in enumerate(uploaded_files):
            status_text.text(f"Processing {file.name}...")
            
            result = upload_document(file)
            
            if "error" in result:
                st.error(f"Failed to process {file.name}: {result['error']}")
            else:
                st.success(f"{file.name} processed successfully")
                if "task_id" in result:
                    st.caption(f"Task ID: {result['task_id']}")
            
            progress_bar.progress((i + 1) / len(uploaded_files))
        
        status_text.text("Processing complete!")
        time.sleep(1)
        st.session_state.documents = get_documents()
        st.rerun()
    
    # Document List
    st.markdown("#### Uploaded Documents")
    
    if st.session_state.documents:
        # Convert to DataFrame for better display
        df_data = []
        for doc in st.session_state.documents:
            df_data.append({
                "Filename": doc['filename'],
                "Type": doc['file_type'].upper(),
                "Size": f"{doc['file_size'] / 1024:.1f} KB",
                "Uploaded": doc['upload_date'][:10],
                "Processed": "Yes" if doc['processed'] else "No",
                "Chunks": doc.get('chunk_count', 'N/A')
            })
        
        df = pd.DataFrame(df_data)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )
        
        # Document Actions
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Refresh List", use_container_width=True):
                st.session_state.documents = get_documents()
                st.rerun()
        
        st.info(f"Total documents: {len(st.session_state.documents)}")
    else:
        st.info("No documents uploaded yet. Use the uploader above to add documents.")

with tab3:
    # System Information
    st.markdown("<div class='sub-header'>System Information</div>", unsafe_allow_html=True)
    
    if st.session_state.system_status:
        status = st.session_state.system_status
        
        # Overall Status
        col1, col2, col3 = st.columns(3)
        with col1:
            system_status = status.get('system', {}).get('status', 'UNKNOWN')
            st.metric("Overall Status", system_status)
        with col2:
            st.metric("Documents", status.get('system', {}).get('document_count', 0))
        with col3:
            checks_passed = status.get('system', {}).get('checks_passed', '0/0')
            st.metric("Checks Passed", checks_passed)
        
        # Components
        st.markdown("#### Components")
        
        components = status.get('components', {})
        for name, value in components.items():
            cols = st.columns([1, 3])
            with cols[0]:
                st.success("✓")
            with cols[1]:
                st.text(f"{name}: {value}")
        
        # Models
        models_data = get_models()
        if models_data.get('models'):
            st.markdown("#### Available Models")
            for model in models_data['models'][:5]:
                st.code(model['name'], language=None)
            if len(models_data['models']) > 5:
                st.caption(f"+ {len(models_data['models']) - 5} more models")
        
        # Raw JSON (for debugging)
        with st.expander("Raw System Status"):
            st.json(status)
    else:
        st.info("System status not available. Check API connection in sidebar.")
        
        if st.button("Load System Status"):
            st.session_state.system_status = get_system_status()
            st.rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #6B7280; font-size: 0.9rem;'>
        DocuBot v1.0.0 | Local AI Document Assistant | Data Privacy: 100% Local
    </div>
    """,
    unsafe_allow_html=True
)