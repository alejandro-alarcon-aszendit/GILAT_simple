#!/usr/bin/env python3
"""
Run script for Document Service UI
=================================

This script helps you run both the FastAPI backend and Streamlit frontend.
"""

import subprocess
import sys
import time
import threading
from pathlib import Path

def run_fastapi():
    """Run the FastAPI server"""
    print("🚀 Starting FastAPI server...")
    subprocess.run([
        sys.executable, "-m", "uvicorn", 
        "agent:app", 
        "--reload", 
        "--host", "0.0.0.0", 
        "--port", "8000"
    ])

def run_streamlit():
    """Run the Streamlit UI"""
    print("🎨 Starting Streamlit UI...")
    time.sleep(3)  # Give FastAPI time to start
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", 
        "streamlit_app.py", 
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])

def main():
    print("📚 Document Service Launcher")
    print("=" * 50)
    
    # Check if files exist
    if not Path("agent.py").exists():
        print("❌ agent.py not found! Make sure you're in the correct directory.")
        sys.exit(1)
    
    if not Path("streamlit_app.py").exists():
        print("❌ streamlit_app.py not found! Make sure you're in the correct directory.")
        sys.exit(1)
    
    print("✅ Starting both FastAPI backend and Streamlit frontend...")
    print()
    print("FastAPI will run on: http://localhost:8000")
    print("Streamlit UI will run on: http://localhost:8501")
    print()
    print("Press Ctrl+C to stop both services")
    print()
    
    try:
        # Start FastAPI in a separate thread
        fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
        fastapi_thread.start()
        
        # Start Streamlit in main thread
        run_streamlit()
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down services...")
        sys.exit(0)

if __name__ == "__main__":
    main() 