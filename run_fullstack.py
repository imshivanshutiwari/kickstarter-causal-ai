#!/usr/bin/env python
"""
🚀 Full Stack Launcher
=====================
Starts both the Python API backend and React frontend.
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def main():
    print("=" * 60)
    print("🚀 KICKSTARTER CAUSAL AI - Full Stack Launcher")
    print("=" * 60)
    
    base_dir = Path(__file__).parent
    frontend_dir = base_dir.parent / "page design"
    
    # Check if frontend exists
    if not frontend_dir.exists():
        print(f"❌ Frontend not found at: {frontend_dir}")
        print("Please ensure the 'page design' folder exists.")
        return
    
    print("\n📡 Starting Python API (FastAPI) on http://localhost:8000 ...")
    api_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api:app", "--reload", "--host", "0.0.0.0", "--port", "8000"],
        cwd=str(base_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT
    )
    
    # Wait for API to start
    time.sleep(3)
    
    print("🎨 Starting React Frontend (Vite) on http://localhost:5173 ...")
    frontend_process = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=str(frontend_dir),
        shell=True
    )
    
    # Wait for frontend to start
    time.sleep(4)
    
    # Open browser
    print("\n🌐 Opening http://localhost:5173 in your browser...")
    webbrowser.open("http://localhost:5173")
    
    print("\n" + "=" * 60)
    print("✅ BOTH SERVERS RUNNING")
    print("   API:      http://localhost:8000/docs")
    print("   Frontend: http://localhost:5173")
    print("=" * 60)
    print("\n⏹️  Press Ctrl+C to stop both servers.\n")
    
    try:
        # Keep running until Ctrl+C
        api_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        api_process.terminate()
        frontend_process.terminate()
        api_process.wait()
        frontend_process.wait()
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()
