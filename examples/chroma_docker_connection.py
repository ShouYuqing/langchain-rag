#!/usr/bin/env python3
"""
Example script for connecting to a Chroma vector database deployed with Docker Compose.
"""

import os
import chromadb
from chromadb.config import Settings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from src.rag_system import RAGSystem
from src.config import create_default_config

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Chroma connection settings
CHROMA_HOST = "localhost"
CHROMA_PORT = 8000
COLLECTION_NAME = "langchain_rag_collection"
CHROMA_TOKEN = "your-admin-token"  # From chroma_auth.json

def connect_to_chroma():
    """Connect to the Chroma vector database."""
    print(f"Connecting to Chroma at {CHROMA_HOST}:{CHROMA_PORT}...")
    
    # Initialize ChromaDB client with authentication
    client = chromadb.HttpClient(
        host=CHROMA_HOST,
        port=CHROMA_PORT,
        settings=Settings(
            chroma_client_auth_provider="chromadb.auth.token.TokenAuthClientProvider",
            chroma_client_auth_credentials=CHROMA_TOKEN
        )
    )
    
    # Check connection
    heartbeat = client.heartbeat()
    print(f"Connected to Chroma! Heartbeat: {heartbeat}")
    
    return client

def setup_langchain_rag(client):
    """Set up LangChain RAG with Chroma."""
    # Initialize OpenAI embeddings
    embeddings = OpenAIEmbeddings()
    
    # Get or create collection
    try:
        client.get_collection(COLLECTION_NAME)
        print(f"Using existing collection: {COLLECTION_NAME}")
    except Exception:
        client.create_collection(COLLECTION_NAME)
        print(f"Created new collection: {COLLECTION_NAME}")
    
    # Initialize Chroma vector store with LangChain
    vector_store = Chroma(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings
    )
    
    # Initialize RAG system
    config = create_default_config()
    rag_system = RAGSystem(config=config, vector_store=vector_store)
    
    return rag_system

def main():
    """Main function to demonstrate Chroma connection."""
    # Connect to Chroma
    client = connect_to_chroma()
    
    # Set up RAG system
    rag_system = setup_langchain_rag(client)
    
    # Example query
    query = "What is retrieval-augmented generation?"
    print(f"\nProcessing query: '{query}'")
    
    result = rag_system.query(query)
    print("\nRAG Response:")
    print("-" * 50)
    print(result.answer)
    print("-" * 50)
    
    # List available collections
    collections = client.list_collections()
    print(f"\nAvailable collections: {[c.name for c in collections]}")

if __name__ == "__main__":
    main() 