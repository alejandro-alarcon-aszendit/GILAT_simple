"""
Streamlit UI for LangGraph Document Service
==========================================

A user-friendly interface to interact with the document service API.
Provides functionality for document management, summarization, and Q&A.
"""

import streamlit as st
import requests
import json
import os
from typing import List, Dict, Any
from datetime import datetime
import time

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

# Page configuration
st.set_page_config(
    page_title="Document Service UI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .status-ready {
        color: #28a745;
        font-weight: bold;
    }
    .status-failed {
        color: #dc3545;
        font-weight: bold;
    }
    .doc-card {
        border: 1px solid #ddd;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f8f9fa;
    }
    .context-doc {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
def get_auth_headers() -> Dict[str, str]:
    """Get authentication headers for API requests."""
    headers = {}
    if st.session_state.get("jwt_token"):
        headers["Authorization"] = f"Bearer {st.session_state.jwt_token}"
    return headers

def login_with_api_key(api_key: str) -> bool:
    """Login with API key and get JWT token.
    
    Args:
        api_key: API key to authenticate with
        
    Returns:
        True if login successful, False otherwise
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/auth/login",
            json={"api_key": api_key},
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            data = response.json()
            st.session_state.jwt_token = data["access_token"]
            st.session_state.authenticated = True
            # Calculate expiry time (24 hours from now)
            from datetime import datetime, timedelta
            st.session_state.token_expires = datetime.now() + timedelta(seconds=data.get("expires_in", 86400))
            return True
        else:
            st.session_state.jwt_token = ""
            st.session_state.authenticated = False
            st.session_state.token_expires = None
            return False
    except Exception as e:
        st.error(f"Login failed: {str(e)}")
        return False

def verify_current_token() -> bool:
    """Verify if current JWT token is still valid.
    
    Returns:
        True if token is valid, False otherwise
    """
    if not st.session_state.jwt_token:
        return False
        
    try:
        response = requests.get(
            f"{API_BASE_URL}/auth/verify",
            headers=get_auth_headers()
        )
        return response.status_code == 200
    except:
        return False

def logout():
    """Clear authentication state."""
    st.session_state.jwt_token = ""
    st.session_state.authenticated = False
    st.session_state.token_expires = None

def api_request(method: str, endpoint: str, **kwargs) -> Dict[Any, Any]:
    """Make API request with error handling and authentication"""
    url = f"{API_BASE_URL}{endpoint}"
    
    # Add authentication headers
    headers = kwargs.get("headers", {})
    headers.update(get_auth_headers())
    kwargs["headers"] = headers
    
    try:
        response = requests.request(method, url, **kwargs)
        if response.status_code == 401:
            st.error("🔒 Authentication failed. Please login again.")
            # Clear invalid token
            logout()
            return {}
        elif response.status_code >= 400:
            st.error(f"API Error {response.status_code}: {response.text}")
            return {}
        return response.json() if response.content else {}
    except requests.exceptions.ConnectionError:
        st.error(f"❌ Cannot connect to API at {API_BASE_URL}. Make sure the FastAPI server is running.")
        return {}
    except Exception as e:
        st.error(f"❌ Request failed: {str(e)}")
        return {}

def get_documents() -> List[Dict[str, Any]]:
    """Fetch all documents from API"""
    return api_request("GET", "/documents") or []

def format_datetime(dt_str: str) -> str:
    """Format datetime string for display"""
    try:
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        return dt.strftime("%Y-%m-%d %H:%M")
    except:
        return dt_str

# Initialize session state
if "context_docs" not in st.session_state:
    st.session_state.context_docs = []
if "api_key" not in st.session_state:
    st.session_state.api_key = ""
if "jwt_token" not in st.session_state:
    st.session_state.jwt_token = ""
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "token_expires" not in st.session_state:
    st.session_state.token_expires = None

# Main header
st.markdown('<h1 class="main-header">📚 Document Service</h1>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")

# Authentication Section
st.sidebar.subheader("🔐 Authentication")

# Check if token is expired
token_expired = False
if st.session_state.token_expires:
    from datetime import datetime
    if datetime.now() > st.session_state.token_expires:
        token_expired = True
        logout()

if not st.session_state.authenticated or token_expired:
    # Login form
    st.sidebar.write("**Login Required**")
    api_key_input = st.sidebar.text_input(
        "API Key:", 
        type="password", 
        value=st.session_state.api_key,
        placeholder="Enter your API key (optional)"
    )
    
    # Update stored API key
    if api_key_input != st.session_state.api_key:
        st.session_state.api_key = api_key_input
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔑 Login", key="login_btn"):
            if login_with_api_key(st.session_state.api_key):
                st.sidebar.success("✅ Login successful!")
                st.rerun()
            else:
                st.sidebar.error("❌ Login failed!")
    
    with col2:
        if st.button("🔓 No Auth", key="no_auth_btn"):
            # Try to login without API key (for no-auth mode)
            if login_with_api_key(""):
                st.sidebar.info("ℹ️ No auth required")
                st.rerun()
            else:
                st.sidebar.error("❌ Authentication required")
    
    st.sidebar.info("💡 If no authentication is required, click 'No Auth'")

else:
    # Authenticated state
    st.sidebar.success("🔒 **Authenticated**")
    
    # Show token info
    if st.session_state.token_expires:
        time_left = st.session_state.token_expires - datetime.now()
        hours_left = int(time_left.total_seconds() / 3600)
        st.sidebar.caption(f"Token expires in: {hours_left}h")
    
    # Logout and verify buttons
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🚪 Logout", key="logout_btn"):
            logout()
            st.rerun()
    
    with col2:
        if st.button("✅ Verify", key="verify_btn"):
            if verify_current_token():
                st.sidebar.success("Token valid!")
            else:
                st.sidebar.error("Token invalid!")
                logout()
                st.rerun()

st.sidebar.divider()

page = st.sidebar.selectbox(
    "Choose a page:",
    ["📋 Document Management", "📝 Summarize", "❓ Ask Questions", "🔧 API Status"]
)

# Check authentication for protected pages
def require_auth():
    """Check if authentication is required and valid for protected operations."""
    if not st.session_state.authenticated:
        st.warning("🔒 Please login in the sidebar before accessing this feature.")
        st.info("💡 **Tip**: If no API key is required, click 'No Auth' to proceed.")
        return False
    return True

# Document Management Page
if page == "📋 Document Management":
    st.header("📋 Document Management")
    
    if not require_auth():
        st.stop()
    
    # Document input section with tabs
    st.subheader("📤 Add New Document")
    
    # Create tabs for different input methods
    tab1, tab2 = st.tabs(["📁 Upload File", "🌐 Fetch from URL"])
    
    with tab1:
        uploaded_file = st.file_uploader(
            "Choose a file to upload",
            type=[
                # Text formats
                'txt', 'md', 'adoc',
                # Office documents  
                'pdf', 'docx', 'xlsx', 'pptx',
                # Web formats
                'html', 'xhtml',
                # Data formats
                'csv',
                # Image formats
                'png', 'jpg', 'jpeg', 'tiff', 'tif', 'bmp', 'webp',
                # XML formats
                'xml'
            ],
            help="Supported formats: TXT, MD, AsciiDoc, PDF, DOCX, XLSX, PPTX, HTML, CSV, Images (PNG/JPG/TIFF/BMP/WEBP), XML"
        )
        
        if uploaded_file and st.button("Upload Document", type="primary", key="upload_file"):
            with st.spinner("Uploading and processing document..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                result = api_request("POST", "/documents", files=files)
                
                if result:
                    st.success(f"✅ Document uploaded successfully!")
                    st.info(f"Document ID: {result.get('doc_id', 'N/A')}")
                    st.info(f"Chunks created: {result.get('n_chunks', 'N/A')}")
                    time.sleep(1)
                    st.rerun()
    
    with tab2:
        url_input = st.text_input(
            "Enter URL to fetch content:",
            placeholder="https://example.com/article",
            help="Enter a valid URL to fetch and process web content"
        )
        
        name_input = st.text_input(
            "Document name (optional):",
            placeholder="Custom name for this document",
            help="Optional: Provide a custom name, otherwise the URL will be used"
        )
        
        if url_input and st.button("Fetch from URL", type="primary", key="fetch_url"):
            # Basic URL validation
            if not url_input.startswith(('http://', 'https://')):
                st.error("❌ Please enter a valid URL starting with http:// or https://")
            else:
                with st.spinner("Fetching content from URL and processing..."):
                    payload = {
                        "url": url_input,
                        "name": name_input if name_input.strip() else None
                    }
                    result = api_request("POST", "/documents/url", json=payload)
                    
                    if result:
                        st.success(f"✅ URL content processed successfully!")
                        st.info(f"Document ID: {result.get('doc_id', 'N/A')}")
                        st.info(f"Chunks created: {result.get('n_chunks', 'N/A')}")
                        st.info(f"Source: {result.get('source', 'N/A')}")
                        time.sleep(1)
                        st.rerun()
    
    st.divider()
    
    # Document list and management
    st.subheader("📚 Document Library")
    
    docs = get_documents()
    if not docs:
        st.info("No documents found. Upload some documents to get started!")
    else:
        # Search and filter
        col1, col2 = st.columns([3, 1])
        with col1:
            search_term = st.text_input("🔍 Search documents by name:", placeholder="Enter filename...")
        with col2:
            status_filter = st.selectbox("Filter by status:", ["All", "ready", "failed"])
        
        # Filter documents
        filtered_docs = docs
        if search_term:
            filtered_docs = [d for d in filtered_docs if search_term.lower() in d['name'].lower()]
        if status_filter != "All":
            filtered_docs = [d for d in filtered_docs if d['status'] == status_filter]
        
        if not filtered_docs:
            st.warning("No documents match your filters.")
        else:
            # Context management section
            st.subheader("📎 Document Context")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write(f"**Selected documents for operations:** {len(st.session_state.context_docs)}")
                if st.session_state.context_docs:
                    context_names = [doc['name'] for doc in st.session_state.context_docs]
                    st.write(", ".join(context_names))
            with col2:
                if st.button("Clear Context"):
                    st.session_state.context_docs = []
                    st.rerun()
            
            st.divider()
            
            # Document list
            for doc in filtered_docs:
                is_in_context = any(d['id'] == doc['id'] for d in st.session_state.context_docs)
                
                with st.container():
                    card_class = "doc-card context-doc" if is_in_context else "doc-card"
                    st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
                    
                    col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
                    
                    with col1:
                        st.write(f"**📄 {doc['name']}**")
                        st.caption(f"ID: {doc['id'][:8]}...")
                    
                    with col2:
                        status_class = "status-ready" if doc['status'] == 'ready' else "status-failed"
                        st.markdown(f'<span class="{status_class}">{doc["status"].upper()}</span>', unsafe_allow_html=True)
                        if doc['n_chunks']:
                            st.caption(f"{doc['n_chunks']} chunks")
                        st.caption(f"Created: {format_datetime(doc['created_at'])}")
                    
                    with col3:
                        if doc['status'] == 'ready':
                            if is_in_context:
                                if st.button("Remove", key=f"remove_{doc['id']}"):
                                    st.session_state.context_docs = [d for d in st.session_state.context_docs if d['id'] != doc['id']]
                                    st.rerun()
                            else:
                                if st.button("Add to Context", key=f"add_{doc['id']}"):
                                    st.session_state.context_docs.append(doc)
                                    st.rerun()
                    
                    with col4:
                        if st.button("🗑️ Delete", key=f"delete_{doc['id']}", type="secondary"):
                            if st.session_state.get(f"confirm_delete_{doc['id']}", False):
                                # Perform deletion
                                result = api_request("DELETE", f"/documents/{doc['id']}")
                                if result:
                                    st.success("Document deleted!")
                                    # Remove from context if present
                                    st.session_state.context_docs = [d for d in st.session_state.context_docs if d['id'] != doc['id']]
                                    time.sleep(1)
                                    st.rerun()
                            else:
                                # Show confirmation
                                st.session_state[f"confirm_delete_{doc['id']}"] = True
                                st.rerun()
                        
                        # Show confirmation message
                        if st.session_state.get(f"confirm_delete_{doc['id']}", False):
                            st.warning("Click again to confirm deletion")
                    
                    st.markdown('</div>', unsafe_allow_html=True)

# Summarize Page
elif page == "📝 Summarize":
    st.header("📝 Document Summarization")
    
    if not require_auth():
        st.stop()
    
    # Document scope selection
    st.subheader("📎 Document Scope")
    document_scope = st.radio(
        "Choose documents to summarize:",
        ["🌍 All Documents", "📂 Selected Documents"],
        help="All Documents: Search across your entire document collection. Selected Documents: Use only documents in context."
    )
    
    if document_scope == "📂 Selected Documents":
        if not st.session_state.context_docs:
            st.warning("⚠️ No documents in context. Go to Document Management to add documents, or use 'All Documents' mode.")
            st.stop()
        else:
            st.success(f"✅ Ready to summarize {len(st.session_state.context_docs)} selected document(s)")
            
            # Show context documents
            with st.expander("📎 Documents in Context", expanded=False):
                for doc in st.session_state.context_docs:
                    st.write(f"• **{doc['name']}** ({doc['n_chunks']} chunks)")
    else:
        # All documents mode
        all_docs = get_documents()
        ready_docs = [d for d in all_docs if d['status'] == 'ready']
        if not ready_docs:
            st.error("❌ No ready documents found in the system.")
            st.stop()
        else:
            st.success(f"🌍 Ready to search across all {len(ready_docs)} documents in the system")
            
            # Show available documents
            with st.expander("📚 All Available Documents", expanded=False):
                for doc in ready_docs:
                    st.write(f"• **{doc['name']}** ({doc['n_chunks']} chunks)")
    
    # Summary mode selection
    st.subheader("🎯 Summary Mode")
    
    # Adjust options based on document scope
    if document_scope == "🌍 All Documents":
        summary_mode_options = ["🔍 Query-Focused Summary"]
        default_help = "All Documents mode requires a topic/query to search for relevant content across your entire collection."
        st.info("💡 **All Documents mode**: You must provide a topic/query to search for relevant content across your document collection.")
    else:
        summary_mode_options = ["📄 Full Document Summary", "🔍 Query-Focused Summary"]
        default_help = "Full Document: Summarize entire documents. Query-Focused: Focus on specific topics using vector search."
    
    summary_mode = st.radio(
        "Choose summarization approach:",
        summary_mode_options,
        help=default_help
    )
    
    # Query input for focused summary
    query_text = ""
    top_k = 10
    if summary_mode == "🔍 Query-Focused Summary":
        st.subheader("🔍 Focus Query")
        query_text = st.text_area(
            "What topic(s) would you like the summary to focus on?",
            placeholder="Single topic: 'machine learning algorithms'\nMultiple topics: 'machine learning, financial performance, project timeline'",
            height=80,
            help="Enter topic(s) for focused summarization:\n• Single topic: Gets one summary focused on that topic\n• Multiple topics (comma-separated): Gets separate summaries for each topic processed in parallel"
        )
        
        # Parse topics to show preview
        topics = []
        if query_text.strip():
            topics = [topic.strip() for topic in query_text.split(",") if topic.strip()]
            if len(topics) > 1:
                st.info(f"🎯 **Multiple topics detected ({len(topics)}):** {', '.join(topics[:3])}{' ...' if len(topics) > 3 else ''}")
                st.caption("Each topic will get its own focused summary processed in parallel.")
            else:
                st.info(f"📌 **Single focus topic:** {topics[0]}")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            if len(topics) > 1:
                st.warning(f"⚡ Multi-topic mode: {len(topics)} summaries will be generated")
        with col2:
            top_k = st.number_input(
                "Max chunks per topic:",
                min_value=5,
                max_value=50,
                value=10,
                help="Maximum number of relevant text chunks to include for each topic"
            )
    
    # Summary options
    st.subheader("📏 Summary Options")
    
    # Strategy selection
    col1, col2 = st.columns(2)
    with col1:
        strategy = st.selectbox(
            "📝 Summarization strategy:",
            ["abstractive", "extractive", "hybrid"],
            index=0,
            help="""Choose how summaries are generated:
• **Abstractive**: AI generates new sentences by paraphrasing and synthesizing content (like human-written summaries)
• **Extractive**: Selects and extracts key sentences directly from the original text (like highlighting important parts)
• **Hybrid**: Combines both approaches - first extracts key content, then refines it using AI"""
        )
    
    with col2:
        length_option = st.selectbox(
            "📏 Summary length:",
            ["short", "medium", "long", "custom"],
            index=1,
            help="Choose summary length: short (3 sentences), medium (8 sentences), long (15 sentences), or custom (choose exact number)"
        )
        
        # Map categorical values to numbers and handle custom
        if length_option == "short":
            length = 3
        elif length_option == "medium":
            length = 8
        elif length_option == "long":
            length = 15
        else:  # custom
            length = st.slider(
                "Number of sentences:",
                min_value=1,
                max_value=30,
                value=8,
                help="Select the exact number of sentences for your summary"
            )
    
    # Strategy explanation
    if strategy == "extractive":
        st.info("🔍 **Extractive**: Will select the most important sentences from your documents based on word frequency and relevance scoring.")
    elif strategy == "abstractive":
        st.info("✍️ **Abstractive**: Will generate new, paraphrased sentences that capture the essence of your documents.")
    elif strategy == "hybrid":
        st.info("🔀 **Hybrid**: Will first extract key sentences, then use AI to refine them into a more coherent summary.")
    
    # Generate summary button
    can_generate = True
    if summary_mode == "🔍 Query-Focused Summary" and not query_text.strip():
        can_generate = False
        st.warning("⚠️ Please enter a focus query for query-focused summarization.")
    
    if st.button("Generate Summary", type="primary", disabled=not can_generate):
        # Prepare document IDs based on scope
        if document_scope == "📂 Selected Documents":
            doc_ids = [doc['id'] for doc in st.session_state.context_docs]
            params = {
                "doc_id": doc_ids,
                "length": length,
                "strategy": strategy
            }
        else:
            # All documents mode - don't pass doc_id parameter
            params = {
                "length": length,
                "strategy": strategy
            }
        
        # Add query parameters if in focused mode
        if summary_mode == "🔍 Query-Focused Summary" and query_text.strip():
            params["query"] = query_text.strip()
            params["top_k"] = top_k
        
        with st.spinner("Generating summary..."):
            result = api_request("GET", "/summary", params=params)
            
            if result:
                # Handle multi-topic results
                if result.get('type') == 'multi_topic':
                    summaries = result.get('summaries', [])
                    
                    st.subheader(f"📋 Multi-Topic Summaries ({len(summaries)} topics)")
                    
                    # Show overall stats
                    successful = result.get('successful_topics', 0)
                    total = result.get('total_topics', 0)
                    if successful == total:
                        st.success(f"✅ Successfully generated summaries for all {total} topics")
                    else:
                        st.warning(f"⚠️ Generated summaries for {successful} out of {total} topics")
                    
                    # Display each topic summary
                    for i, summary_data in enumerate(summaries, 1):
                        topic = summary_data.get('topic', f'Topic {i}')
                        summary_text = summary_data.get('summary', '')
                        status = summary_data.get('status', 'unknown')
                        chunks = summary_data.get('chunks_processed', 0)
                        
                        with st.container():
                            # Topic header with status indicator
                            if status == 'success':
                                st.markdown(f"### 🎯 {topic}")
                            elif status == 'no_content':
                                st.markdown(f"### ⚠️ {topic}")
                            else:
                                st.markdown(f"### ❌ {topic}")
                            
                            # Summary content
                            if status == 'success':
                                st.write(summary_text)
                                topic_strategy = summary_data.get('strategy', strategy)
                                st.caption(f"📊 {chunks} chunks processed • Strategy: {topic_strategy}")
                            elif status == 'no_content':
                                st.info(summary_text)
                            else:
                                st.error(summary_text)
                            
                            st.divider()
                    
                    # Overall metadata
                    with st.expander("📊 Overall Summary Details"):
                        st.write(f"**Documents processed:** {len(result.get('documents', []))}")
                        st.write(f"**Summary length:** {length} sentences ({length_option})")
                        st.write(f"**Summarization strategy:** {result.get('strategy', strategy)}")
                        st.write(f"**Total chunks processed:** {result.get('total_chunks_processed', 'N/A')}")
                        st.write(f"**Search method:** {result.get('search_method', 'N/A')}")
                        st.write(f"**Topics:** {', '.join(result.get('topics', []))}")
                        
                        # Strategy-specific info
                        used_strategy = result.get('strategy', strategy)
                        if used_strategy == "extractive":
                            st.info("🔍 **Extractive Strategy**: Summaries contain key sentences selected directly from your documents.")
                        elif used_strategy == "abstractive":
                            st.info("✍️ **Abstractive Strategy**: Summaries contain AI-generated sentences that paraphrase and synthesize the content.")
                        elif used_strategy == "hybrid":
                            st.info("🔀 **Hybrid Strategy**: Summaries combine extracted key sentences refined by AI for better coherence.")
                        
                        st.caption("💡 Each topic was processed in parallel using vector similarity search to find the most relevant content.")
                
                # Handle single summary results  
                elif 'summary' in result:
                    st.subheader("📋 Summary")
                    st.write(result['summary'])
                    
                    # Show metadata
                    with st.expander("📊 Summary Details"):
                        st.write(f"**Documents processed:** {len(result.get('documents', []))}")
                        st.write(f"**Summary length:** {length} sentences ({length_option})")
                        st.write(f"**Summarization strategy:** {result.get('strategy', strategy)}")
                        st.write(f"**Chunks processed:** {result.get('chunks_processed', 'N/A')}")
                        st.write(f"**Search method:** {result.get('search_method', 'N/A')}")
                        
                        if result.get('query'):
                            st.write(f"**Focus query:** {result.get('query')}")
                        
                        # Strategy-specific info
                        used_strategy = result.get('strategy', strategy)
                        if used_strategy == "extractive":
                            st.info("🔍 **Extractive Strategy**: Summary contains key sentences selected directly from your documents.")
                        elif used_strategy == "abstractive":
                            st.info("✍️ **Abstractive Strategy**: Summary contains AI-generated sentences that paraphrase and synthesize the content.")
                        elif used_strategy == "hybrid":
                            st.info("🔀 **Hybrid Strategy**: Summary combines extracted key sentences refined by AI for better coherence.")
                        
                        if result.get('query'):
                            st.caption("💡 This summary was generated using vector similarity search to find the most relevant content for your query.")
                
                else:
                    st.error("❌ Unexpected response format from the API.")

# Ask Questions Page
elif page == "❓ Ask Questions":
    st.header("❓ Ask Questions")
    
    if not require_auth():
        st.stop()
    
    # Document scope selection
    st.subheader("📎 Document Scope")
    document_scope_qa = st.radio(
        "Choose documents to search:",
        ["🌍 All Documents", "📂 Selected Documents"],
        help="All Documents: Search across your entire document collection. Selected Documents: Use only documents in context.",
        key="qa_scope"
    )
    
    if document_scope_qa == "📂 Selected Documents":
        if not st.session_state.context_docs:
            st.warning("⚠️ No documents in context. Go to Document Management to add documents, or use 'All Documents' mode.")
            st.stop()
        else:
            st.success(f"✅ Ready to answer questions from {len(st.session_state.context_docs)} selected document(s)")
            
            # Show context documents
            with st.expander("📎 Documents in Context", expanded=False):
                for doc in st.session_state.context_docs:
                    st.write(f"• **{doc['name']}** ({doc['n_chunks']} chunks)")
    else:
        # All documents mode
        all_docs = get_documents()
        ready_docs = [d for d in all_docs if d['status'] == 'ready']
        if not ready_docs:
            st.error("❌ No ready documents found in the system.")
            st.stop()
        else:
            st.success(f"🌍 Ready to search across all {len(ready_docs)} documents in the system")
            
            # Show available documents
            with st.expander("📚 All Available Documents", expanded=False):
                for doc in ready_docs:
                    st.write(f"• **{doc['name']}** ({doc['n_chunks']} chunks)")
    
    # Question input
    question = st.text_area(
        "What would you like to know?",
        placeholder="Ask any question about the documents...",
        height=100
    )
    
    col1, col2 = st.columns([3, 1])
    with col1:
        top_k = st.slider("Number of relevant passages to consider:", 1, 10, 3)
    
    if st.button("Get Answer", type="primary", disabled=not question.strip()):
        with st.spinner("Searching for answers..."):
            # Prepare parameters based on scope
            if document_scope_qa == "📂 Selected Documents":
                doc_ids = [doc['id'] for doc in st.session_state.context_docs]
                params = {
                    "q": question,
                    "doc_id": doc_ids,
                    "top_k": top_k
                }
            else:
                # All documents mode - don't pass doc_id parameter
                params = {
                    "q": question,
                    "top_k": top_k
                }
            
            result = api_request("GET", "/ask", params=params)
            
            if result and 'answer' in result:
                st.subheader("💡 Answer")
                st.write(result['answer'])
                
                # Show relevant snippets
                if result.get('snippets'):
                    with st.expander(f"📄 Relevant Passages ({len(result['snippets'])})", expanded=False):
                        for i, snippet in enumerate(result['snippets'], 1):
                            st.markdown(f"**Passage {i}:**")
                            st.markdown(f"> {snippet['content']}")
                            st.divider()

# API Status Page
elif page == "🔧 API Status":
    st.header("🔧 API Status")
    
    # Test API connection
    if st.button("Test API Connection"):
        with st.spinner("Testing connection..."):
            result = api_request("GET", "/documents")
            if result is not None:
                st.success("✅ API is accessible")
                st.info(f"Found {len(result)} documents in the system")
            else:
                st.error("❌ Cannot connect to API")
    
    st.divider()
    
    # API Information
    st.subheader("📋 API Information")
    st.write(f"**API Base URL:** {API_BASE_URL}")
    st.write("**Available Endpoints:**")
    st.code("""
    GET    /documents          - List all documents
    POST   /documents          - Upload new document
    GET    /documents/{id}     - Get document details
    DELETE /documents/{id}     - Delete document
    GET    /summary            - Generate summary
    GET    /ask                - Ask questions
    """)
    
    # System status
    st.subheader("🖥️ System Status")
    docs = get_documents()
    if docs:
        ready_docs = [d for d in docs if d['status'] == 'ready']
        failed_docs = [d for d in docs if d['status'] == 'failed']
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Documents", len(docs))
        with col2:
            st.metric("Ready Documents", len(ready_docs))
        with col3:
            st.metric("Failed Documents", len(failed_docs))
        
        if failed_docs:
            st.warning("⚠️ Some documents failed to process:")
            for doc in failed_docs:
                st.write(f"• {doc['name']} (ID: {doc['id'][:8]}...)")

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #666; font-size: 0.8rem;'>
        📚 LangGraph Document Service UI | Built with Streamlit
    </div>
    """,
    unsafe_allow_html=True
) 