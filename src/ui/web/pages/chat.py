# docubot/src/ui/web/pages/chat.py

"""
Chat Page for Streamlit
"""

import streamlit as st

def render():
    """Render chat page"""
    st.title("Chat with Documents")
    st.markdown("Ask questions about your uploaded documents")
    
    # This would be imported from the main app
    # For now, just show instructions
    st.info("""
    This page would contain the chat interface.
    
    In the actual implementation, this would be loaded from
    the main Streamlit app as a page module.
    """)