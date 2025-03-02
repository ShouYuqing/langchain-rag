#!/usr/bin/env python3
"""
Generate secure authentication tokens for Chroma vector database.
This script creates a chroma_auth.json file with randomly generated tokens.
"""

import secrets
import json
import os
from pathlib import Path

def generate_secure_token():
    """Generate a cryptographically secure random token."""
    return secrets.token_hex(32)

def main():
    print("Generating secure authentication tokens for Chroma...")
    
    # Generate tokens for different roles
    tokens = {
        "admin": generate_secure_token(),
        "reader": generate_secure_token(),
        "writer": generate_secure_token(),
    }
    
    # Create auth config
    auth_config = {"tokens": tokens}
    
    # Save to file
    current_dir = Path(__file__).parent
    auth_file = current_dir / "chroma_auth.json"
    
    with open(auth_file, "w") as f:
        json.dump(auth_config, f, indent=2)
    
    print(f"\n✅ Authentication configuration saved to {auth_file}")
    print("\n⚠️  IMPORTANT: Save these credentials securely. You will need them to connect to your database.")
    print("\nGenerated tokens:")
    print("-" * 50)
    for role, token in tokens.items():
        print(f"{role}: {token}")
    print("-" * 50)
    print("\nTo use these tokens with the Python client:")
    print("""
import chromadb
from chromadb.config import Settings

client = chromadb.HttpClient(
    host="localhost",
    port=8000,
    settings=Settings(
        chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
        chroma_client_auth_credentials="<your-token-here>"
    )
)
    """)

if __name__ == "__main__":
    main() 