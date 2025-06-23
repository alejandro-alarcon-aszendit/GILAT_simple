# 📚 Document Service Streamlit UI

A beautiful and user-friendly web interface for the LangGraph Document Service API.

## 🌟 Features

### 📋 Document Management
- **Upload Documents**: Support for TXT, MD, and PDF files
- **Document Library**: View all documents with search and filtering
- **Context Management**: Add/remove documents to/from working context
- **Document Deletion**: Safe deletion with confirmation
- **Real-time Status**: See processing status and chunk counts

### 📝 Summarization
- **Multi-Document Summaries**: Generate summaries from multiple documents
- **Length Control**: Choose between short, medium, and long summaries
- **Context Awareness**: Only summarize documents in your current context

### ❓ Question & Answer
- **Natural Language Q&A**: Ask questions about your documents
- **Relevant Passages**: See the source passages used to answer your questions
- **Configurable Retrieval**: Adjust how many passages to consider

### 🔧 System Monitoring
- **API Status**: Check connection and system health
- **Document Statistics**: Overview of ready/failed documents
- **Connection Testing**: Verify API connectivity

## 🚀 Quick Start

### Prerequisites
Make sure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

### Option 1: Use the Launcher Script (Recommended)
```bash
python run_ui.py
```

This will start both the FastAPI backend and Streamlit frontend automatically:
- FastAPI: http://localhost:8000
- Streamlit UI: http://localhost:8501

### Option 2: Manual Start

1. **Start the FastAPI backend:**
```bash
uvicorn agent:app --reload
```

2. **Start the Streamlit UI (in another terminal):**
```bash
streamlit run streamlit_app.py
```

## 📖 Usage Guide

### 1. Document Management
1. Navigate to "📋 Document Management"
2. Upload documents using the file uploader
3. Add documents to your context by clicking "Add to Context"
4. Use search and filters to find specific documents

### 2. Generating Summaries
1. Go to "📝 Summarize"
2. Ensure you have documents in context
3. Choose summary length (short/medium/long)
4. Click "Generate Summary"

### 3. Asking Questions
1. Navigate to "❓ Ask Questions"
2. Make sure documents are in context
3. Type your question in the text area
4. Adjust the number of passages to consider
5. Click "Get Answer"

### 4. Monitoring System
1. Visit "🔧 API Status"
2. Test API connection
3. View system statistics
4. Check for failed documents

## 🎨 UI Features

### Visual Design
- **Clean Layout**: Wide layout with sidebar navigation
- **Color-coded Status**: Green for ready, red for failed documents
- **Context Highlighting**: Documents in context are visually distinct
- **Responsive Design**: Works well on different screen sizes

### User Experience
- **Progress Indicators**: Spinners for long-running operations
- **Error Handling**: Clear error messages and connection status
- **Confirmation Dialogs**: Safe deletion with double-click confirmation
- **Session State**: Context persists across page navigation

### Interactive Elements
- **Search & Filter**: Find documents quickly
- **Expandable Sections**: Collapsible details for better organization
- **Real-time Updates**: UI refreshes after operations
- **Keyboard Shortcuts**: Standard Streamlit shortcuts available

## 🛠️ Configuration

### API Connection
The UI connects to the FastAPI backend at `http://localhost:8000` by default. To change this, modify the `API_BASE_URL` variable in `streamlit_app.py`:

```python
API_BASE_URL = "http://your-api-server:8000"
```

### Customization
- **Styling**: Modify the CSS in the `st.markdown()` section
- **Page Layout**: Adjust the page configuration in `st.set_page_config()`
- **Features**: Add new pages by extending the sidebar navigation

## 🔧 Troubleshooting

### Common Issues

**"Cannot connect to API"**
- Ensure the FastAPI server is running on port 8000
- Check if `OPENAI_API_KEY` environment variable is set
- Verify there are no firewall issues

**"Document processing failed"**
- Check the FastAPI logs for detailed error messages
- Ensure uploaded files are text-readable
- Verify OpenAI API key is valid and has credits

**"No documents in context"**
- Upload documents first in Document Management
- Add documents to context using "Add to Context" button
- Check that documents have "ready" status

### Performance Tips
- **Large Documents**: Documents are automatically chunked for better performance
- **Multiple Documents**: Context operations work with multiple documents simultaneously
- **Memory Usage**: Consider the number of documents in context for summary operations

## 📝 File Structure

```
├── streamlit_app.py     # Main Streamlit application
├── run_ui.py           # Launcher script for both services
├── agent.py            # FastAPI backend
├── requirements.txt    # Python dependencies
└── UI_README.md       # This file
```

## 🤝 Contributing

To extend the UI:

1. **Add New Pages**: Extend the sidebar navigation and add new page handlers
2. **Enhance Styling**: Modify the CSS section for visual improvements
3. **Add Features**: Implement new API endpoints and corresponding UI elements
4. **Improve UX**: Add better error handling, loading states, or user feedback

## 📄 License

This UI is part of the LangGraph Document Service project. 