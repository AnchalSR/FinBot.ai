"""
Streamlit frontend for FinBot.

Provides a user-friendly interface for the financial chatbot including:
- Chat interface with history
- Document upload
- Source references
- Settings and configuration
"""

import streamlit as st
import requests
import json
import time
from datetime import datetime
from typing import List, Dict
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="FinBot - Financial Advisor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 0rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex
    }
    .chat-message.user {
        background-color: #e1f5ff;
    }
    .chat-message.assistant {
        background-color: #f3e5f5;
    }
    .source-box {
        background-color: #fafafa;
        border-left: 4px solid #2196F3;
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 0.25rem;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== Configuration ====================

API_URL = os.getenv("API_URL", "http://localhost:8000")
API_TIMEOUT = 30

# ==================== Session State ====================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False

if "system_status" not in st.session_state:
    st.session_state.system_status = None


# ==================== Helper Functions ====================

@st.cache_resource
def get_api_session():
    """Get a requests session with timeout."""
    session = requests.Session()
    return session


def check_api_health() -> bool:
    """Check if API is running."""
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=API_TIMEOUT
        )
        return response.status_code == 200
    except:
        return False


def get_system_status() -> Dict:
    """Get system status from API."""
    try:
        response = requests.get(
            f"{API_URL}/status",
            timeout=API_TIMEOUT
        )
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None


def send_query(query: str, top_k: int = 5) -> Dict:
    """
    Send a query to the chatbot API.
    
    Args:
        query: User query
        top_k: Number of documents to retrieve
        
    Returns:
        Response from API
    """
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "query": query,
                "top_k": top_k,
                "include_sources": True
            },
            timeout=API_TIMEOUT
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.text}
            
    except requests.exceptions.Timeout:
        return {"error": "API request timed out. Make sure the backend is running."}
    except requests.exceptions.ConnectionError:
        return {"error": f"Cannot connect to API at {API_URL}. Make sure the backend is running."}
    except Exception as e:
        return {"error": str(e)}


def upload_documents(files: List) -> Dict:
    """
    Upload documents to the API.
    
    Args:
        files: List of uploaded files
        
    Returns:
        Upload status
    """
    try:
        file_objects = []
        for file in files:
            file_objects.append(("files", (file.name, file.getbuffer(), "application/octet-stream")))
        
        response = requests.post(
            f"{API_URL}/upload",
            files=file_objects,
            timeout=API_TIMEOUT
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": response.text}
            
    except Exception as e:
        return {"error": str(e)}


def display_message(message: Dict, role: str):
    """Display a chat message."""
    if role == "user":
        with st.chat_message("user"):
            st.markdown(message.get("content", ""))
    else:
        with st.chat_message("assistant"):
            st.markdown(message.get("content", ""))
            
            # Display sources if available
            if "sources" in message and message["sources"]:
                with st.expander("📚 View Sources", expanded=False):
                    for source in message["sources"]:
                        st.markdown(f"""
                        <div class="source-box">
                        <strong>Source:</strong> {source.get('source', 'Unknown')}<br>
                        <strong>Relevance:</strong> {source.get('similarity', 0):.1%}
                        </div>
                        """, unsafe_allow_html=True)


# ==================== Main App ====================

def main():
    """Main Streamlit app."""
    
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("💰 FinBot - Financial Advisor")
        st.caption("Powered by RAG and LLMs")
    
    with col2:
        if check_api_health():
            st.success("✅ API Connected", icon="✔️")
        else:
            st.error("❌ API Offline", icon="❌")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API Configuration
        st.subheader("API Configuration")
        api_url_input = st.text_input(
            "API URL",
            value=API_URL,
            help="Backend API endpoint"
        )
        if api_url_input != API_URL:
            os.environ["API_URL"] = api_url_input
        
        # System Status
        st.subheader("System Status")
        status = get_system_status()
        
        if status:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Documents", status.get("documents_count", 0))
            with col2:
                st.metric("Device", status.get("device", "unknown").upper())
            
            st.caption(f"Embedding: {status.get('embedding_model', 'unknown')}")
        else:
            st.warning("Unable to fetch system status")
        
        # Chat Settings
        st.subheader("Chat Settings")
        top_k = st.slider(
            "Number of Documents to Retrieve",
            min_value=1,
            max_value=10,
            value=5,
            help="How many relevant documents to use for context"
        )
        
        # Document Management
        st.subheader("📁 Document Management")
        
        with st.expander("Upload Documents", expanded=False):
            uploaded_files = st.file_uploader(
                "Upload PDF or TXT files",
                type=["pdf", "txt"],
                accept_multiple_files=True,
                help="Upload financial documents for the RAG pipeline"
            )
            
            if uploaded_files and st.button("Upload Files", key="upload_btn"):
                with st.spinner("Uploading documents..."):
                    result = upload_documents(uploaded_files)
                    
                    if "error" in result:
                        st.error(f"Upload failed: {result['error']}")
                    else:
                        st.success(f"✅ Uploaded {len(result.get('uploaded_files', []))} files")
                        st.session_state.documents_loaded = True
                        time.sleep(1)
                        st.rerun()
        
        # Clear Chat History
        if st.button("🗑️ Clear Chat History", key="clear_history"):
            st.session_state.messages = []
            st.rerun()
        
        st.divider()
        
        # About
        st.subheader("ℹ️ About FinBot")
        st.info(
            "FinBot is an AI-powered financial advisor chatbot that uses "
            "Retrieval Augmented Generation (RAG) to provide accurate, "
            "document-based financial guidance."
        )
        
        st.markdown("""
        **Features:**
        - 📚 RAG-based document retrieval
        - 🤖 LLM-powered responses
        - 📊 Source references
        - 💾 Document management
        - 🚀 Production-ready
        """)
    
    # Main Chat Interface
    st.subheader("💬 Chat Interface")
    
    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant"):
                st.markdown(message["content"])
                
                # Display sources if available
                if "sources" in message and message["sources"]:
                    with st.expander("📚 View Sources", expanded=False):
                        for source in message["sources"]:
                            st.markdown(f"""
                            **Source:** {source.get('source', 'Unknown')}  
                            **Relevance:** {source.get('similarity', 0):.1%}
                            """)
    
    # Chat input
    query = st.chat_input("Ask a financial question...")
    
    if query:
        # Add user message to history
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        
        # Get response from API
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            sources_placeholder = st.empty()
            
            with st.spinner("Thinking..."):
                response = send_query(query, top_k=top_k)
            
            if "error" in response:
                st.error(f"Error: {response['error']}")
                assistant_message = f"Error: {response['error']}"
            else:
                assistant_message = response.get("answer", "No response")
                response_placeholder.markdown(assistant_message)
                
                # Display sources
                sources = response.get("sources", [])
                if sources:
                    with sources_placeholder.expander("📚 View Sources", expanded=False):
                        for source in sources:
                            st.markdown(f"""
                            **Source:** {source.get('source', 'Unknown')}  
                            **Relevance:** {source.get('similarity', 0):.1%}
                            """)
        
        # Add assistant message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": assistant_message,
            "sources": response.get("sources", [])
        })


if __name__ == "__main__":
    main()
