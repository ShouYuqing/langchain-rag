"""
FastAPI server for the RAG system with a built-in web UI.
This server provides both API endpoints and serves a static HTML client.
"""

import os
import time
import logging
import asyncio
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from dotenv import load_dotenv

# Import RAG system components
from src.rag_system.config import RAGConfig, create_production_config, create_lightweight_config
from src.rag_system.rag_system import RAGSystem
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="RAG System API with UI")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up static files and templates
current_dir = Path(__file__).parent
static_dir = current_dir / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
templates = Jinja2Templates(directory=str(static_dir))

# Global variables
document_index_status = {
    "status": "idle",  # idle, processing, completed, error
    "progress": 0,
    "message": "Indexing not started"
}

# Pydantic models
class QueryRequest(BaseModel):
    query: str
    stream: bool = False
    config_type: Optional[str] = "production"

class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    execution_time: Dict[str, float]
    query: str

class IndexRequest(BaseModel):
    data_dir: str = "data/documents"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    persist_dir: str = "data/vector_store"

class IndexStatusResponse(BaseModel):
    status: str
    progress: int
    message: str

class ConfigResponse(BaseModel):
    config: Dict[str, Any]

# Helper functions
def load_and_process_documents(data_dir: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Load and process documents from a directory."""
    global document_index_status
    
    try:
        document_index_status["status"] = "processing"
        document_index_status["progress"] = 10
        document_index_status["message"] = "Loading documents..."
        
        # Load documents
        loader = DirectoryLoader(data_dir)
        documents = loader.load()
        
        document_index_status["progress"] = 30
        document_index_status["message"] = f"Loaded {len(documents)} documents. Processing..."
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(documents)
        
        document_index_status["progress"] = 60
        document_index_status["message"] = f"Created {len(chunks)} chunks. Ready for indexing."
        
        return chunks
    except Exception as e:
        document_index_status["status"] = "error"
        document_index_status["message"] = f"Error processing documents: {str(e)}"
        logger.error(f"Error processing documents: {e}", exc_info=True)
        raise

def create_vector_store(documents, persist_dir: str):
    """Create a vector store from documents."""
    global document_index_status
    
    try:
        document_index_status["progress"] = 70
        document_index_status["message"] = "Creating vector store..."
        
        # Create vector store
        embeddings = OpenAIEmbeddings()
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_dir
        )
        
        # Persist vector store
        vector_store.persist()
        
        document_index_status["status"] = "completed"
        document_index_status["progress"] = 100
        document_index_status["message"] = "Indexing completed successfully."
        
        return vector_store
    except Exception as e:
        document_index_status["status"] = "error"
        document_index_status["message"] = f"Error creating vector store: {str(e)}"
        logger.error(f"Error creating vector store: {e}", exc_info=True)
        raise

def setup_rag_system(config_type: str = "production"):
    """Initialize the RAG system based on configuration."""
    # Get configuration
    if config_type == "production":
        config = create_production_config()
    else:  # lightweight
        config = create_lightweight_config()
    
    # Initialize vector store
    embeddings = OpenAIEmbeddings()
    persist_dir = config.vector_store.persist_directory
    
    # Check if vector store exists
    if os.path.exists(persist_dir):
        vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings
        )
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Vector store not found at {persist_dir}. Please index documents first."
        )
    
    # Initialize RAG system
    rag_system = RAGSystem(config=config, vector_store=vector_store)
    
    return rag_system

# Background task for indexing
async def index_documents_task(data_dir: str, chunk_size: int, chunk_overlap: int, persist_dir: str):
    """Background task for indexing documents."""
    global document_index_status
    
    try:
        # Load and process documents
        chunks = load_and_process_documents(data_dir, chunk_size, chunk_overlap)
        
        # Create vector store
        create_vector_store(chunks, persist_dir)
    except Exception as e:
        document_index_status["status"] = "error"
        document_index_status["progress"] = 0
        document_index_status["message"] = f"Error during indexing: {str(e)}"
        logger.error(f"Error during indexing: {e}", exc_info=True)

# API endpoints
@app.get("/")
async def serve_ui(request: Request):
    """Serve the web UI."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api")
async def root():
    """Root endpoint."""
    return {"message": "RAG System API is running"}

@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a query using the RAG system."""
    try:
        # Initialize RAG system
        rag_system = setup_rag_system(request.config_type)
        
        # Process query
        start_time = time.time()
        result = await rag_system.aquery(request.query)
        end_time = time.time()
        
        # Prepare response
        return QueryResponse(
            answer=result.answer,
            sources=result.sources,
            execution_time=result.execution_time,
            query=request.query
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/query/stream")
async def stream_query(query: str, config_type: str = "production"):
    """Stream a query response."""
    try:
        # Initialize RAG system
        rag_system = setup_rag_system(config_type)
        
        # Define streaming response
        async def generate():
            try:
                async for token in rag_system.astream(query):
                    yield f"data: {token}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Error in streaming: {e}", exc_info=True)
                yield f"data: Error: {str(e)}\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
    except Exception as e:
        logger.error(f"Error setting up streaming: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/index", response_model=IndexStatusResponse)
async def index_documents(request: IndexRequest, background_tasks: BackgroundTasks):
    """Index documents for the RAG system."""
    global document_index_status
    
    try:
        # Reset status
        document_index_status = {
            "status": "processing",
            "progress": 0,
            "message": "Starting indexing process..."
        }
        
        # Start background task
        background_tasks.add_task(
            index_documents_task,
            request.data_dir,
            request.chunk_size,
            request.chunk_overlap,
            request.persist_dir
        )
        
        return IndexStatusResponse(**document_index_status)
    except Exception as e:
        logger.error(f"Error starting indexing: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/index/status", response_model=IndexStatusResponse)
async def get_index_status():
    """Get the status of the indexing process."""
    global document_index_status
    return IndexStatusResponse(**document_index_status)

@app.get("/api/config/{config_type}", response_model=ConfigResponse)
async def get_config(config_type: str):
    """Get the configuration for the RAG system."""
    try:
        if config_type == "production":
            config = create_production_config()
        elif config_type == "lightweight":
            config = create_lightweight_config()
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid configuration type: {config_type}"
            )
        
        return ConfigResponse(config=config.to_dict())
    except Exception as e:
        logger.error(f"Error getting configuration: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# Run the server
if __name__ == "__main__":
    # Check if OpenAI API key is set
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY environment variable is not set.")
    
    # Run the server
    uvicorn.run(
        "api_server_with_ui:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # For development
    ) 