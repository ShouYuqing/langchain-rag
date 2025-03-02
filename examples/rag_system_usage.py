"""
Example usage of the RAG system with various configurations
"""
import os
import asyncio
from typing import List
import logging

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.rag_system import RAGSystem
from src.retrieval.advanced_retriever import AdvancedRetriever

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_documents(data_dir: str) -> List:
    """
    Load documents from a directory
    
    Args:
        data_dir: Path to the directory containing documents
        
    Returns:
        List of loaded documents
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
        # DirectoryLoader(data_dir, glob="**/*.pdf", loader_cls=PyPDFLoader),
        # DirectoryLoader(data_dir, glob="**/*.csv", loader_cls=CSVLoader),
    ]
    
    documents = []
    for loader in loaders:
        documents.extend(loader.load())
    
    logger.info(f"Loaded {len(documents)} documents")
    return documents


def process_documents(documents: List, chunk_size: int = 1000, chunk_overlap: int = 200) -> List:
    """
    Process documents by splitting them into chunks
    
    Args:
        documents: List of documents to process
        chunk_size: Size of each document chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of processed document chunks
    """
    logger.info(f"Processing documents with chunk size {chunk_size} and overlap {chunk_overlap}")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} document chunks")
    
    return chunks


def create_vector_store(documents: List, embeddings, persist_directory: str = None):
    """
    Create a vector store from documents
    
    Args:
        documents: List of documents to index
        embeddings: Embeddings model to use
        persist_directory: Optional directory to persist the vector store
        
    Returns:
        Initialized vector store
    """
    logger.info("Creating vector store")
    
    if persist_directory and os.path.exists(persist_directory):
        # Load existing vector store
        logger.info(f"Loading existing vector store from {persist_directory}")
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
    else:
        # Create new vector store
        logger.info("Creating new vector store")
        vector_store = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        if persist_directory:
            vector_store.persist()
    
    return vector_store


async def run_rag_example():
    """Run the RAG system example"""
    # Set up OpenAI API
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it with your OpenAI API key."
        )
    
    # Load and process documents
    data_dir = "data/documents"  # Replace with your data directory
    documents = load_documents(data_dir)
    processed_docs = process_documents(documents)
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    persist_dir = "data/vector_store"
    vector_store = create_vector_store(processed_docs, embeddings, persist_dir)
    
    # Create a retriever from the vector store
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    # Create language model
    llm = ChatOpenAI(
        model="gpt-4-turbo",  # Use an appropriate model
        temperature=0,
    )
    
    # Initialize the RAG system
    logger.info("Initializing RAG system")
    rag_system = RAGSystem(
        retriever=retriever,
        llm=llm,
        use_advanced_retrieval=True,
        use_context_processing=True,
        use_context_augmentation=True,
        use_query_routing=False,
        rerank_results=True,
        system_prompt="You are a helpful assistant that provides accurate answers based on the retrieved context.",
        verbose=True
    )
    
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
    
    # Example of streaming response
    print("\nStreaming example:\n")
    query = "Explain the concept of retrieval-augmented generation"
    
    # Create a streaming RAG system
    streaming_rag = RAGSystem(
        retriever=retriever,
        llm=llm,
        use_advanced_retrieval=True,
        use_context_processing=True,
        streaming=True,
        verbose=False
    )
    
    print(f"Query: {query}")
    print("-"*80)
    
    # Process streaming response
    async for chunk in streaming_rag.generate_streaming_response(query):
        print(chunk, end="", flush=True)
    
    print("\n" + "="*80)


def advanced_retrieval_example():
    """Example of using the advanced retriever directly"""
    # Set up OpenAI API
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. "
            "Please set it with your OpenAI API key."
        )
    
    # Load and process documents
    data_dir = "data/documents"  # Replace with your data directory
    documents = load_documents(data_dir)
    processed_docs = process_documents(documents)
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    persist_dir = "data/vector_store"
    vector_store = create_vector_store(processed_docs, embeddings, persist_dir)
    
    # Create a basic retriever from the vector store
    base_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    
    # Create language model
    llm = ChatOpenAI(
        model="gpt-4-turbo",  # Use an appropriate model
        temperature=0,
    )
    
    # Create an advanced retriever
    advanced_retriever = AdvancedRetriever(
        base_retriever=base_retriever,
        llm=llm,
        use_query_expansion=True,
        use_hypothetical_document=True,
        use_reranking=True,
        use_semantic_embeddings=True,
        top_k=5,
        num_expanded_queries=3
    )
    
    # Example query
    query = "How does backpropagation work in neural networks?"
    
    # Run the query and get results
    results = asyncio.run(advanced_retriever.aget_relevant_documents(query))
    
    print("\n" + "="*80)
    print(f"Query: {query}")
    print("="*80)
    print(f"Retrieved {len(results)} documents")
    
    for i, doc in enumerate(results):
        print(f"\nDocument {i+1}:")
        print(f"Source: {doc.metadata.get('source', 'Unknown')}")
        if "relevance_score" in doc.metadata:
            print(f"Relevance: {doc.metadata['relevance_score']:.4f}")
        print(f"Content: {doc.page_content[:200]}...")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    # Run the asynchronous example
    asyncio.run(run_rag_example())
    
    # Run the advanced retrieval example
    # advanced_retrieval_example() 