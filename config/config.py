import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union

# Load environment variables
load_dotenv()

class ChunkingConfig(BaseModel):
    """Configuration for document chunking strategies"""
    chunk_size: int = Field(default=1000, description="Size of chunks in tokens")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks in tokens")
    semantic_chunking: bool = Field(default=True, description="Whether to use semantic chunking")
    semantic_chunking_method: str = Field(default="sentence", description="Method for semantic chunking")

class EmbeddingConfig(BaseModel):
    """Configuration for embeddings"""
    model_name: str = Field(default="text-embedding-3-large", description="Embedding model to use")
    batch_size: int = Field(default=8, description="Batch size for embedding generation")
    dimensions: int = Field(default=1536, description="Dimensions of embeddings")

class VectorStoreConfig(BaseModel):
    """Configuration for vector store"""
    provider: str = Field(default="chroma", description="Vector store provider")
    persist_directory: str = Field(default="./chroma_db", description="Directory to persist vector store")
    collection_name: str = Field(default="documents", description="Name of collection in vector store")

class IndexingConfig(BaseModel):
    """Configuration for document indexing"""
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    hierarchical_indexing: bool = Field(default=True, description="Whether to use hierarchical indexing (RAPTOR)")

class RetrievalConfig(BaseModel):
    """Configuration for retrieval mechanisms"""
    default_top_k: int = Field(default=5, description="Default number of documents to retrieve")
    rerank_enabled: bool = Field(default=True, description="Whether to use reranking")
    rerank_model: str = Field(default="cohere.rerank-english-v3.0", description="Reranking model to use")
    rerank_top_k: int = Field(default=10, description="Number of documents to rerank")
    rag_fusion_enabled: bool = Field(default=True, description="Whether to use RAG-fusion")
    rag_fusion_top_k: int = Field(default=10, description="Number of documents to retrieve per query in RAG-fusion")
    rag_fusion_num_queries: int = Field(default=3, description="Number of queries to generate for RAG-fusion")
    active_retrieval: bool = Field(default=True, description="Whether to use active retrieval")
    min_relevance_score: float = Field(default=0.7, description="Minimum relevance score for documents")

class ContextProcessingConfig(BaseModel):
    """Configuration for context processing"""
    max_tokens: int = Field(default=16000, description="Maximum context window size")
    prioritize_recent: bool = Field(default=True, description="Whether to prioritize recent documents")
    filter_threshold: float = Field(default=0.5, description="Threshold for filtering irrelevant information")
    metadata_fields: List[str] = Field(default=["source", "title", "date"], description="Metadata fields to include")

class RoutingConfig(BaseModel):
    """Configuration for routing mechanisms"""
    logical_routing_enabled: bool = Field(default=True, description="Whether to use logical routing")
    semantic_routing_enabled: bool = Field(default=True, description="Whether to use semantic routing")
    available_retrievers: List[str] = Field(default=["default", "web", "code", "qa"], description="Available retrievers")

class LLMConfig(BaseModel):
    """Configuration for large language models"""
    provider: str = Field(default="openai", description="LLM provider")
    model_name: str = Field(default="gpt-4o", description="LLM model to use")
    temperature: float = Field(default=0.0, description="Temperature for LLM")
    max_tokens: int = Field(default=4000, description="Maximum number of tokens in response")
    system_prompt: str = Field(
        default="You are a helpful assistant that provides accurate and concise information based on the context provided.",
        description="System prompt for LLM"
    )

class APIConfig(BaseModel):
    """Configuration for API"""
    host: str = Field(default="0.0.0.0", description="Host to run API on")
    port: int = Field(default=8000, description="Port to run API on")
    debug: bool = Field(default=False, description="Whether to run API in debug mode")

class AppConfig(BaseModel):
    """Main application configuration"""
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    context_processing: ContextProcessingConfig = Field(default_factory=ContextProcessingConfig)
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    api: APIConfig = Field(default_factory=APIConfig)

def load_config() -> AppConfig:
    """Load configuration from environment variables"""
    config = AppConfig()
    
    # Update config from environment variables
    if os.getenv("EMBEDDING_MODEL"):
        config.indexing.embedding.model_name = os.getenv("EMBEDDING_MODEL")
    
    if os.getenv("LLM_MODEL"):
        config.llm.model_name = os.getenv("LLM_MODEL")
    
    if os.getenv("RERANK_MODEL"):
        config.retrieval.rerank_model = os.getenv("RERANK_MODEL")
    
    if os.getenv("DEFAULT_TOP_K"):
        config.retrieval.default_top_k = int(os.getenv("DEFAULT_TOP_K"))
    
    if os.getenv("RERANK_TOP_K"):
        config.retrieval.rerank_top_k = int(os.getenv("RERANK_TOP_K"))
    
    if os.getenv("RAG_FUSION_TOP_K"):
        config.retrieval.rag_fusion_top_k = int(os.getenv("RAG_FUSION_TOP_K"))
    
    if os.getenv("CHROMA_PERSIST_DIRECTORY"):
        config.indexing.vector_store.persist_directory = os.getenv("CHROMA_PERSIST_DIRECTORY")
    
    if os.getenv("API_HOST"):
        config.api.host = os.getenv("API_HOST")
    
    if os.getenv("API_PORT"):
        config.api.port = int(os.getenv("API_PORT"))
    
    return config

# Load config
config = load_config() 