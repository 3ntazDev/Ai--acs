"""
API Entry point for Vercel
"""
import sys
import os
from pathlib import Path

# Add parent directory to path
parent_dir = str(Path(__file__).parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import the app
try:
    from accident_api import app
except ImportError:
    # Fallback: create a simple error app
    from fastapi import FastAPI
    app = FastAPI()
    
    @app.get("/")
    def error():
        return {
            "error": "Failed to import main application",
            "path": sys.path,
            "cwd": os.getcwd(),
            "files": os.listdir(".")
        }

# This is what Vercel calls
app = app