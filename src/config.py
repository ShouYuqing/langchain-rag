"""
Configuration module for the RAG system.
Provides configuration classes and utilities for setting up the RAG system.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any


@dataclass
class EmbeddingConfig:
    """Configuration for embeddings"""
    provider: str = "openai"  # openai, huggingface, cohere, etc.
    model_name: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 32
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrieverConfig:
    """Configuration for document retrieval"""
    retriever_type: str = "vector"  # vector, hybrid, keyword, etc.
    search_type: str = "similarity"  # similarity, mmr, etc.
    search_kwargs: Dict[str, Any] = field(default_factory=lambda: {"k": 5})
    
    # Advanced retrieval settings
    use_query_expansion: bool = False
    use_hypothetical_document: bool = False
    num_expanded_queries: int = 3
    use_semantic_embeddings: bool = False


@dataclass
class RerankerConfig:
    """Configuration for reranking retrieved documents"""
    use_reranking: bool = False
    reranker_type: str = "hybrid"  # keyword, embedding, llm, hybrid, custom
    weights: Dict[str, float] = field(default_factory=lambda: {
        "keyword": 0.3, 
        "embedding": 0.3, 
        "llm": 0.4
    })
    top_k: int = 5


@dataclass
class ContextProcessorConfig:
    """Configuration for context processing"""
    enabled: bool = True
    max_context_length: int = 4000
    deduplication_threshold: float = 0.85
    remove_redundant: bool = True
    filter_irrelevant: bool = True
    min_relevance_score: float = 0.7
    preserve_order: bool = False
    hierarchical_ordering: bool = False


@dataclass
class ContextAugmentationConfig:
    """Configuration for context augmentation"""
    enabled: bool = False
    add_summaries: bool = True
    add_entity_definitions: bool = True
    add_cross_references: bool = True
    add_knowledge_graph: bool = False


@dataclass
class QueryRouterConfig:
    """Configuration for query routing"""
    enabled: bool = False
    domain_keyword_map: Dict[str, List[str]] = field(default_factory=dict)
    max_parallel_retrievers: int = 3
    threshold_score: float = 0.8


@dataclass
class LLMConfig:
    """Configuration for language models"""
    provider: str = "openai"  # openai, anthropic, etc.
    model_name: str = "gpt-4-turbo"
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    streaming: bool = False
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VectorStoreConfig:
    """Configuration for vector stores"""
    provider: str = "chroma"  # chroma, pinecone, milvus, etc.
    persist_directory: Optional[str] = None
    collection_name: str = "default_collection"
    additional_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGConfig:
    """Main configuration for the RAG system"""
    system_name: str = "RAG System"
    system_prompt: str = "You are a helpful assistant that provides accurate answers based on the retrieved context."
    verbose: bool = False
    
    # Component configs
    llm: LLMConfig = field(default_factory=LLMConfig)
    embeddings: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = field(default_factory=VectorStoreConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    context_processor: ContextProcessorConfig = field(default_factory=ContextProcessorConfig)
    context_augmentation: ContextAugmentationConfig = field(default_factory=ContextAugmentationConfig)
    query_router: QueryRouterConfig = field(default_factory=QueryRouterConfig)
    
    # Additional configurations
    max_retries: int = 3
    timeout: int = 60
    cache_results: bool = True
    log_level: str = "INFO"  # DEBUG, INFO, WARNING, ERROR, CRITICAL
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        result = {}
        for field_name, field_value in self.__dict__.items():
            if hasattr(field_value, "to_dict"):
                result[field_name] = field_value.to_dict()
            elif isinstance(field_value, dict):
                result[field_name] = field_value.copy()
            else:
                result[field_name] = field_value
        return result
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RAGConfig":
        """Create config from dictionary"""
        # Create a new instance with default values
        config = cls()
        
        # Update instance with values from config_dict
        for key, value in config_dict.items():
            if hasattr(config, key):
                attr = getattr(config, key)
                if hasattr(attr, "from_dict") and isinstance(value, dict):
                    setattr(config, key, attr.__class__.from_dict(value))
                else:
                    setattr(config, key, value)
        
        return config
    
    def save(self, filepath: str) -> None:
        """Save configuration to YAML file"""
        with open(filepath, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    @classmethod
    def load(cls, filepath: str) -> "RAGConfig":
        """Load configuration from YAML file"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Config file not found: {filepath}")
        
        with open(filepath, "r") as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)


def create_default_config() -> RAGConfig:
    """Create a default configuration for the RAG system"""
    return RAGConfig()


def create_production_config() -> RAGConfig:
    """Create a production-optimized configuration for the RAG system"""
    config = RAGConfig(
        system_name="Production RAG System",
        verbose=False,
        llm=LLMConfig(
            provider="openai",
            model_name="gpt-4-turbo",
            temperature=0.0,
        ),
        embeddings=EmbeddingConfig(
            provider="openai",
            model_name="text-embedding-3-small",
        ),
        vector_store=VectorStoreConfig(
            provider="chroma",
            persist_directory="data/production/vector_store",
        ),
        retriever=RetrieverConfig(
            retriever_type="vector",
            search_type="similarity",
            search_kwargs={"k": 8},
            use_query_expansion=True,
            use_hypothetical_document=True,
            num_expanded_queries=3,
        ),
        reranker=RerankerConfig(
            use_reranking=True,
            reranker_type="hybrid",
            weights={"keyword": 0.2, "embedding": 0.3, "llm": 0.5},
            top_k=5,
        ),
        context_processor=ContextProcessorConfig(
            enabled=True,
            max_context_length=6000,
            deduplication_threshold=0.85,
            remove_redundant=True,
            filter_irrelevant=True,
        ),
        context_augmentation=ContextAugmentationConfig(
            enabled=True,
            add_summaries=True,
            add_entity_definitions=True,
            add_cross_references=True,
        ),
        query_router=QueryRouterConfig(
            enabled=True,
            max_parallel_retrievers=3,
        ),
        max_retries=5,
        timeout=120,
        cache_results=True,
        log_level="INFO",
    )
    return config


def create_lightweight_config() -> RAGConfig:
    """Create a lightweight configuration for the RAG system (lower resource usage)"""
    config = RAGConfig(
        system_name="Lightweight RAG System",
        verbose=True,
        llm=LLMConfig(
            provider="openai",
            model_name="gpt-3.5-turbo",
            temperature=0.0,
        ),
        embeddings=EmbeddingConfig(
            provider="openai",
            model_name="text-embedding-3-small",
        ),
        vector_store=VectorStoreConfig(
            provider="chroma",
            persist_directory="data/lightweight/vector_store",
        ),
        retriever=RetrieverConfig(
            retriever_type="vector",
            search_type="similarity",
            search_kwargs={"k": 3},
            use_query_expansion=False,
            use_hypothetical_document=False,
        ),
        reranker=RerankerConfig(
            use_reranking=True,
            reranker_type="keyword",  # Lightweight reranking
            top_k=3,
        ),
        context_processor=ContextProcessorConfig(
            enabled=True,
            max_context_length=2000,
            deduplication_threshold=0.9,
            remove_redundant=True,
            filter_irrelevant=True,
        ),
        context_augmentation=ContextAugmentationConfig(
            enabled=False,  # Disable augmentation to save resources
        ),
        query_router=QueryRouterConfig(
            enabled=False,  # Disable routing to save resources
        ),
        max_retries=2,
        timeout=30,
        cache_results=True,
        log_level="INFO",
    )
    return config 