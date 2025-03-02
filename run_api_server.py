#!/usr/bin/env python3
"""
Run script for the RAG System API server with UI.
This script provides a convenient way to start the API server.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_environment():
    """Check if the environment is properly set up."""
    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY environment variable is not set.")
        print("You can set it by running: export OPENAI_API_KEY=your_api_key")
        return False
    
    # Check if required directories exist
    data_dir = Path("data/documents")
    if not data_dir.exists():
        print(f"Warning: Data directory {data_dir} does not exist.")
        print("Creating directory...")
        data_dir.mkdir(parents=True, exist_ok=True)
    
    vector_store_dir = Path("data/vector_store")
    if not vector_store_dir.exists():
        print(f"Warning: Vector store directory {vector_store_dir} does not exist.")
        print("Creating directory...")
        vector_store_dir.mkdir(parents=True, exist_ok=True)
    
    return True

def run_server(host="0.0.0.0", port=8000, reload=True):
    """Run the API server."""
    server_file = Path("examples/api_server_with_ui.py")
    
    if not server_file.exists():
        print(f"Error: Server file {server_file} not found.")
        return False
    
    cmd = [
        "uvicorn",
        "examples.api_server_with_ui:app",
        f"--host={host}",
        f"--port={port}"
    ]
    
    if reload:
        cmd.append("--reload")
    
    print(f"Starting server on http://{host}:{port}")
    print("Press Ctrl+C to stop the server.")
    
    try:
        subprocess.run(cmd)
        return True
    except KeyboardInterrupt:
        print("\nServer stopped.")
        return True
    except Exception as e:
        print(f"Error running server: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Run the RAG System API server with UI.")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind the server to")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    
    args = parser.parse_args()
    
    if not check_environment():
        print("Warning: Environment check failed. Continuing anyway...")
    
    success = run_server(
        host=args.host,
        port=args.port,
        reload=not args.no_reload
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 