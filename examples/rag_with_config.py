"""
Example of using the RAG system with configuration files
"""
import os
import asyncio
import logging
from pathlib import Path

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag_system import RAGSystem
from src.config import (
    RAGConfig, 
    create_default_config, 
    create_production_config, 
    create_lightweight_config
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_and_process_documents(data_dir: str, chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    Load and process documents from a directory
    
    Args:
        data_dir: Path to the directory containing documents
        chunk_size: Size of each document chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of processed document chunks
    """
    logger.info(f"Loading documents from {data_dir}")
    
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
    
    logger.info(f"Loaded {len(documents)} documents")
    
    # Process documents
    logger.info(f"Processing documents with chunk size {chunk_size} and overlap {chunk_overlap}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} document chunks")
    
    return chunks


def setup_vector_store(config: RAGConfig, documents=None):
    """
    Set up a vector store based on configuration
    
    Args:
        config: RAG configuration
        documents: Optional documents to index
        
    Returns:
        Initialized vector store
    """
    logger.info(f"Setting up {config.vector_store.provider} vector store")
    
    # Create embeddings based on config
    if config.embeddings.provider == "openai":
        embeddings = OpenAIEmbeddings(
            model=config.embeddings.model_name,
            dimensions=config.embeddings.dimensions,
            **config.embeddings.additional_params
        )
    else:
        raise ValueError(f"Unsupported embeddings provider: {config.embeddings.provider}")
    
    # Set up vector store
    persist_dir = config.vector_store.persist_directory
    
    if persist_dir and os.path.exists(persist_dir) and documents is None:
        # Load existing vector store
        logger.info(f"Loading existing vector store from {persist_dir}")
        vector_store = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name=config.vector_store.collection_name,
            **config.vector_store.additional_params
        )
    else:
        # Create new vector store
        if documents is None:
            raise ValueError("Documents must be provided to create a new vector store")
        
        logger.info("Creating new vector store")
        if persist_dir:
            # Ensure directory exists
            Path(persist_dir).parent.mkdir(parents=True, exist_ok=True)
        
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_dir,
            collection_name=config.vector_store.collection_name,
            **config.vector_store.additional_params
        )
        
        if persist_dir:
            vector_store.persist()
    
    return vector_store


def setup_llm(config: RAGConfig):
    """
    Set up a language model based on configuration
    
    Args:
        config: RAG configuration
        
    Returns:
        Initialized language model
    """
    logger.info(f"Setting up {config.llm.provider} language model: {config.llm.model_name}")
    
    if config.llm.provider == "openai":
        llm = ChatOpenAI(
            model=config.llm.model_name,
            temperature=config.llm.temperature,
            max_tokens=config.llm.max_tokens,
            streaming=config.llm.streaming,
            **config.llm.additional_params
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {config.llm.provider}")
    
    return llm


def setup_rag_system(config: RAGConfig, vector_store, llm):
    """
    Set up the RAG system based on configuration
    
    Args:
        config: RAG configuration
        vector_store: Vector store for document retrieval
        llm: Language model for answer generation
        
    Returns:
        Initialized RAG system
    """
    logger.info("Setting up RAG system")
    
    # Create retriever from vector store
    retriever = vector_store.as_retriever(
        search_type=config.retriever.search_type,
        search_kwargs=config.retriever.search_kwargs
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


async def run_example_with_config(config_type: str = "default"):
    """
    Run the RAG system example with a specified configuration type
    
    Args:
        config_type: Type of configuration to use (default, production, lightweight, or path to config file)
    """
    # Set up OpenAI API
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it with your OpenAI API key."
        )
    
    # Load configuration
    if config_type == "default":
        config = create_default_config()
    elif config_type == "production":
        config = create_production_config()
    elif config_type == "lightweight":
        config = create_lightweight_config()
    elif os.path.exists(config_type):
        config = RAGConfig.load(config_type)
    else:
        raise ValueError(f"Unknown configuration type: {config_type}")
    
    logger.info(f"Using {config.system_name} configuration")
    
    # Set log level from config
    logging.getLogger().setLevel(config.log_level)
    
    # Load and process documents
    data_dir = "data/documents"  # Replace with your data directory
    documents = load_and_process_documents(data_dir)
    
    # Set up vector store
    vector_store = setup_vector_store(config, documents)
    
    # Set up language model
    llm = setup_llm(config)
    
    # Set up RAG system
    rag_system = setup_rag_system(config, vector_store, llm)
    
    # Example queries
    queries = [
        "What is the main concept of artificial intelligence?",
        "How do neural networks learn?",
        "What are the ethical concerns with large language models?",
    ]
    
    # Process queries
    logger.info("Processing queries")
    for query in queries:
        logger.info(f"Query: {query}")
        
        result = await rag_system.process_query(query)
        
        print("\n" + "="*80)
        print(f"Query: {query}")
        print("="*80)
        print(f"Answer: {result.answer}")
        print("-"*80)
        print("Sources:")
        for source in result.sources:
            print(f"- {source.get('title', 'Untitled')} ({source.get('source', 'Unknown')})")
        print("-"*80)
        print(f"Execution time: {result.execution_time.get('total', 0):.2f} seconds")
        print("="*80 + "\n")


def save_example_config():
    """Save example configurations to files"""
    configs_dir = "configs"
    os.makedirs(configs_dir, exist_ok=True)
    
    # Save default config
    default_config = create_default_config()
    default_config.save(os.path.join(configs_dir, "default_config.yaml"))
    
    # Save production config
    production_config = create_production_config()
    production_config.save(os.path.join(configs_dir, "production_config.yaml"))
    
    # Save lightweight config
    lightweight_config = create_lightweight_config()
    lightweight_config.save(os.path.join(configs_dir, "lightweight_config.yaml"))
    
    logger.info(f"Saved example configurations to {configs_dir} directory")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RAG system with configuration")
    parser.add_argument(
        "--config", 
        type=str, 
        default="default",
        help="Configuration type (default, production, lightweight) or path to config file"
    )
    parser.add_argument(
        "--save-configs", 
        action="store_true",
        help="Save example configurations to files"
    )
    
    args = parser.parse_args()
    
    if args.save_configs:
        save_example_config()
    
    asyncio.run(run_example_with_config(args.config)) 