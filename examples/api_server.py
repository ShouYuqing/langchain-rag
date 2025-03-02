"""
Simple FastAPI server for the RAG system
"""
import os
import time
import logging
import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag_system import RAGSystem, RAGQueryResult
from src.config import RAGConfig, create_production_config, create_lightweight_config

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="RAG System API",
    description="API for the Retrieval-Augmented Generation System",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
rag_system = None
vector_store = None
document_index_status = {"status": "not_started", "progress": 0, "message": ""}


# Pydantic models for API
class QueryRequest(BaseModel):
    query: str = Field(..., description="The query to process")
    stream: bool = Field(False, description="Whether to stream the response")
    config_type: Optional[str] = Field("production", description="Configuration type (production or lightweight)")


class QueryResponse(BaseModel):
    answer: str = Field(..., description="The generated answer")
    sources: List[Dict[str, Any]] = Field([], description="Sources used for the answer")
    execution_time: Dict[str, float] = Field({}, description="Execution time details")
    query: str = Field(..., description="The original query")


class IndexRequest(BaseModel):
    data_dir: str = Field(..., description="Directory containing documents to index")
    chunk_size: int = Field(1000, description="Size of document chunks")
    chunk_overlap: int = Field(200, description="Overlap between chunks")
    persist_dir: Optional[str] = Field("data/vector_store", description="Directory to persist the vector store")


class IndexStatusResponse(BaseModel):
    status: str = Field(..., description="Status of the indexing process")
    progress: float = Field(..., description="Progress of the indexing process (0-100)")
    message: str = Field(..., description="Status message")


class ConfigResponse(BaseModel):
    config: Dict[str, Any] = Field(..., description="RAG system configuration")


# Helper functions
def load_and_process_documents(data_dir: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """Load and process documents from a directory"""
    global document_index_status
    
    try:
        document_index_status = {"status": "loading", "progress": 10, "message": "Loading documents..."}
        
        # Use appropriate loaders based on file types
        loaders = [
            DirectoryLoader(
                data_dir, 
                glob="**/*.txt",
                loader_cls=TextLoader
            ),
            # Add more loaders for different file types as needed
        ]
        
        documents = []
        for loader in loaders:
            documents.extend(loader.load())
        
        document_index_status = {
            "status": "processing", 
            "progress": 40, 
            "message": f"Processing {len(documents)} documents..."
        }
        
        # Process documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
        )
        
        chunks = text_splitter.split_documents(documents)
        
        document_index_status = {
            "status": "processed", 
            "progress": 70, 
            "message": f"Created {len(chunks)} document chunks"
        }
        
        return chunks
    
    except Exception as e:
        document_index_status = {
            "status": "error", 
            "progress": 0, 
            "message": f"Error processing documents: {str(e)}"
        }
        logger.error(f"Error processing documents: {e}")
        raise


def create_vector_store(documents, persist_directory: str = None):
    """Create a vector store from documents"""
    global document_index_status, vector_store
    
    try:
        document_index_status = {
            "status": "indexing", 
            "progress": 80, 
            "message": "Creating vector store..."
        }
        
        # Create embeddings
        embeddings = OpenAIEmbeddings()
        
        # Set up vector store
        if persist_directory:
            # Ensure directory exists
            Path(persist_directory).parent.mkdir(parents=True, exist_ok=True)
        
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        
        if persist_directory:
            vector_store.persist()
        
        document_index_status = {
            "status": "completed", 
            "progress": 100, 
            "message": f"Indexed {len(documents)} document chunks"
        }
        
        return vector_store
    
    except Exception as e:
        document_index_status = {
            "status": "error", 
            "progress": 0, 
            "message": f"Error creating vector store: {str(e)}"
        }
        logger.error(f"Error creating vector store: {e}")
        raise


def setup_rag_system(config_type: str = "production"):
    """Set up the RAG system"""
    global rag_system, vector_store
    
    if vector_store is None:
        raise ValueError("Vector store not initialized. Please index documents first.")
    
    # Load configuration
    if config_type == "production":
        config = create_production_config()
    elif config_type == "lightweight":
        config = create_lightweight_config()
    else:
        raise ValueError(f"Unknown configuration type: {config_type}")
    
    # Create retriever from vector store
    retriever = vector_store.as_retriever(
        search_type=config.retriever.search_type,
        search_kwargs=config.retriever.search_kwargs
    )
    
    # Create language model
    llm = ChatOpenAI(
        model=config.llm.model_name,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
        streaming=config.llm.streaming,
    )
    
    # Initialize RAG system
    rag_system = RAGSystem(
        retriever=retriever,
        llm=llm,
        use_advanced_retrieval=config.retriever.use_query_expansion or config.retriever.use_hypothetical_document,
        use_context_processing=config.context_processor.enabled,
        use_context_augmentation=config.context_augmentation.enabled,
        use_query_routing=config.query_router.enabled,
        rerank_results=config.reranker.use_reranking,
        system_prompt=config.system_prompt,
        verbose=config.verbose
    )
    
    return rag_system


async def index_documents_task(data_dir: str, chunk_size: int, chunk_overlap: int, persist_dir: str):
    """Background task for indexing documents"""
    try:
        # Load and process documents
        documents = load_and_process_documents(data_dir, chunk_size, chunk_overlap)
        
        # Create vector store
        create_vector_store(documents, persist_dir)
        
    except Exception as e:
        logger.error(f"Error in indexing task: {e}")
        document_index_status["status"] = "error"
        document_index_status["message"] = str(e)


# API endpoints
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {"message": "RAG System API is running"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a query using the RAG system"""
    global rag_system
    
    # Check if RAG system is initialized
    if rag_system is None:
        try:
            rag_system = setup_rag_system(request.config_type)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize RAG system: {str(e)}")
    
    try:
        # Process query
        start_time = time.time()
        
        if request.stream:
            # For streaming, we can't return the streaming response directly
            # Instead, we'll process it normally and just note that streaming was requested
            logger.info("Streaming requested but not supported in this endpoint. Use /query/stream instead.")
        
        result = await rag_system.process_query(request.query)
        
        # Convert RAGQueryResult to QueryResponse
        response = QueryResponse(
            answer=result.answer,
            sources=result.sources,
            execution_time=result.execution_time,
            query=request.query
        )
        
        return response
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/query/stream")
async def stream_query(
    query: str = Query(..., description="The query to process"),
    config_type: str = Query("production", description="Configuration type (production or lightweight)")
):
    """Stream a query response using the RAG system"""
    global rag_system
    
    # Check if RAG system is initialized
    if rag_system is None:
        try:
            rag_system = setup_rag_system(config_type)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to initialize RAG system: {str(e)}")
    
    # Create a streaming RAG system if needed
    if not hasattr(rag_system, 'streaming') or not rag_system.streaming:
        # Get the current configuration
        config = create_production_config() if config_type == "production" else create_lightweight_config()
        
        # Create a streaming version of the language model
        streaming_llm = ChatOpenAI(
            model=config.llm.model_name,
            temperature=config.llm.temperature,
            streaming=True,
        )
        
        # Create a new RAG system with streaming enabled
        streaming_rag = RAGSystem(
            retriever=rag_system.retriever,
            llm=streaming_llm,
            use_advanced_retrieval=rag_system.use_advanced_retrieval,
            use_context_processing=rag_system.use_context_processing,
            use_context_augmentation=rag_system.use_context_augmentation,
            use_query_routing=rag_system.use_query_routing,
            rerank_results=rag_system.rerank_results,
            system_prompt=rag_system.system_prompt,
            streaming=True,
            verbose=rag_system.verbose
        )
    else:
        streaming_rag = rag_system
    
    async def generate():
        try:
            async for chunk in streaming_rag.generate_streaming_response(query):
                yield f"data: {chunk}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"Error in streaming response: {e}")
            yield f"data: Error: {str(e)}\n\n"
            yield "data: [DONE]\n\n"
    
    return generate()


@app.post("/index", response_model=IndexStatusResponse)
async def index_documents(request: IndexRequest, background_tasks: BackgroundTasks):
    """Index documents for the RAG system"""
    global document_index_status
    
    # Check if indexing is already in progress
    if document_index_status["status"] in ["loading", "processing", "indexing"]:
        return IndexStatusResponse(**document_index_status)
    
    # Reset status
    document_index_status = {"status": "starting", "progress": 0, "message": "Starting indexing process..."}
    
    # Start indexing in the background
    background_tasks.add_task(
        index_documents_task, 
        request.data_dir, 
        request.chunk_size, 
        request.chunk_overlap, 
        request.persist_dir
    )
    
    return IndexStatusResponse(**document_index_status)


@app.get("/index/status", response_model=IndexStatusResponse)
async def get_index_status():
    """Get the status of the document indexing process"""
    global document_index_status
    return IndexStatusResponse(**document_index_status)


@app.get("/config/{config_type}", response_model=ConfigResponse)
async def get_config(config_type: str = "production"):
    """Get the RAG system configuration"""
    try:
        if config_type == "production":
            config = create_production_config()
        elif config_type == "lightweight":
            config = create_lightweight_config()
        else:
            raise HTTPException(status_code=400, detail=f"Unknown configuration type: {config_type}")
        
        return ConfigResponse(config=config.to_dict())
    
    except Exception as e:
        logger.error(f"Error getting configuration: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting configuration: {str(e)}")


# Run the server
if __name__ == "__main__":
    # Check for OpenAI API key
    if not os.environ.get("OPENAI_API_KEY"):
        logger.warning(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it with your OpenAI API key."
        )
    
    # Run the server
    uvicorn.run("api_server:app", host="0.0.0.0", port=8000, reload=True) 