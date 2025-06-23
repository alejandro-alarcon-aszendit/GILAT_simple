"""
Streamlit UI for LangGraph Document Service
==========================================

A user-friendly interface to interact with the document service API.
Provides functionality for document management, summarization, and Q&A.
"""

import streamlit as st
import requests
import json
from typing import List, Dict, Any
from datetime import datetime
import time

# Configuration
API_BASE_URL = "http://localhost:8000"

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
def api_request(method: str, endpoint: str, **kwargs) -> Dict[Any, Any]:
    """Make API request with error handling"""
    url = f"{API_BASE_URL}{endpoint}"
    try:
        response = requests.request(method, url, **kwargs)
        if response.status_code >= 400:
            st.error(f"API Error {response.status_code}: {response.text}")
            return {}
        return response.json() if response.content else {}
    except requests.exceptions.ConnectionError:
        st.error("❌ Cannot connect to API. Make sure the FastAPI server is running on http://localhost:8000")
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

# Main header
st.markdown('<h1 class="main-header">📚 Document Service</h1>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["📋 Document Management", "📝 Summarize", "❓ Ask Questions", "🔧 API Status"]
)

# Document Management Page
if page == "📋 Document Management":
    st.header("📋 Document Management")
    
    # Document upload section
    st.subheader("📤 Upload New Document")
    uploaded_file = st.file_uploader(
        "Choose a file to upload",
        type=['txt', 'md', 'pdf'],
        help="Supported formats: TXT, MD, PDF"
    )
    
    if uploaded_file and st.button("Upload Document", type="primary"):
        with st.spinner("Uploading and processing document..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            result = api_request("POST", "/documents", files=files)
            
            if result:
                st.success(f"✅ Document uploaded successfully!")
                st.info(f"Document ID: {result.get('doc_id', 'N/A')}")
                st.info(f"Chunks created: {result.get('n_chunks', 'N/A')}")
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
    
    if not st.session_state.context_docs:
        st.warning("⚠️ No documents in context. Go to Document Management to add documents.")
    else:
        st.success(f"✅ Ready to summarize {len(st.session_state.context_docs)} document(s)")
        
        # Show context documents
        with st.expander("📎 Documents in Context", expanded=False):
            for doc in st.session_state.context_docs:
                st.write(f"• **{doc['name']}** ({doc['n_chunks']} chunks)")
        
        # Summary mode selection
        st.subheader("🎯 Summary Mode")
        summary_mode = st.radio(
            "Choose summarization approach:",
            ["📄 Full Document Summary", "🔍 Query-Focused Summary"],
            help="Full Document: Summarize entire documents. Query-Focused: Focus on specific topics using vector search."
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
        length = st.selectbox(
            "Summary length:",
            ["short", "medium", "long"],
            index=1,
            help="Choose summary length: short (≈3 sentences), medium (≈8 sentences), long (≈15 sentences)"
        )
        
        # Generate summary button
        can_generate = True
        if summary_mode == "🔍 Query-Focused Summary" and not query_text.strip():
            can_generate = False
            st.warning("⚠️ Please enter a focus query for query-focused summarization.")
        
        if st.button("Generate Summary", type="primary", disabled=not can_generate):
            doc_ids = [doc['id'] for doc in st.session_state.context_docs]
            
            # Prepare parameters
            params = {
                "doc_id": doc_ids,
                "length": length
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
                                    st.caption(f"📊 {chunks} chunks processed")
                                elif status == 'no_content':
                                    st.info(summary_text)
                                else:
                                    st.error(summary_text)
                                
                                st.divider()
                        
                        # Overall metadata
                        with st.expander("📊 Overall Summary Details"):
                            st.write(f"**Documents processed:** {len(result.get('documents', []))}")
                            st.write(f"**Summary length:** {length}")
                            st.write(f"**Total chunks processed:** {result.get('total_chunks_processed', 'N/A')}")
                            st.write(f"**Search method:** {result.get('search_method', 'N/A')}")
                            st.write(f"**Topics:** {', '.join(result.get('topics', []))}")
                            st.info("💡 Each topic was processed in parallel using vector similarity search to find the most relevant content.")
                    
                    # Handle single summary results  
                    elif 'summary' in result:
                        st.subheader("📋 Summary")
                        st.write(result['summary'])
                        
                        # Show metadata
                        with st.expander("📊 Summary Details"):
                            st.write(f"**Documents processed:** {len(result.get('documents', []))}")
                            st.write(f"**Summary length:** {length}")
                            st.write(f"**Chunks processed:** {result.get('chunks_processed', 'N/A')}")
                            st.write(f"**Search method:** {result.get('search_method', 'N/A')}")
                            
                            if result.get('query'):
                                st.write(f"**Focus query:** {result.get('query')}")
                                st.info("💡 This summary was generated using vector similarity search to find the most relevant content for your query.")
                    
                    else:
                        st.error("❌ Unexpected response format from the API.")

# Ask Questions Page
elif page == "❓ Ask Questions":
    st.header("❓ Ask Questions")
    
    if not st.session_state.context_docs:
        st.warning("⚠️ No documents in context. Go to Document Management to add documents.")
    else:
        st.success(f"✅ Ready to answer questions from {len(st.session_state.context_docs)} document(s)")
        
        # Show context documents
        with st.expander("📎 Documents in Context", expanded=False):
            for doc in st.session_state.context_docs:
                st.write(f"• **{doc['name']}** ({doc['n_chunks']} chunks)")
        
        # Question input
        question = st.text_area(
            "What would you like to know?",
            placeholder="Ask any question about the documents in context...",
            height=100
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            top_k = st.slider("Number of relevant passages to consider:", 1, 10, 3)
        
        if st.button("Get Answer", type="primary", disabled=not question.strip()):
            doc_ids = [doc['id'] for doc in st.session_state.context_docs]
            
            with st.spinner("Searching for answers..."):
                params = {
                    "q": question,
                    "doc_id": doc_ids,
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