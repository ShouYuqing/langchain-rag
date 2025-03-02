"""
Document indexing with RAPTOR hierarchical indexing support
"""
from typing import Dict, List, Optional, Union, Any, Callable
import os
import pickle
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document

from src.indexing.chunking import SemanticChunker, HierarchicalChunker
from src.indexing.embedding import SpecializedEmbeddingGenerator
from src.utils.document import extract_text_from_file, generate_document_id, clean_text, extract_metadata_from_text
from config.config import config

class DocumentIndexer:
    """
    Document indexing system with support for hierarchical indexing (RAPTOR)
    """
    
    def __init__(
        self,
        vector_store_provider: str = "chroma",
        persist_directory: str = "./chroma_db",
        embedding_model_name: str = "text-embedding-3-large",
        chunking_config: Optional[Dict[str, Any]] = None,
        use_hierarchical_indexing: bool = True
    ):
        self.vector_store_provider = vector_store_provider
        self.persist_directory = persist_directory
        self.embedding_model_name = embedding_model_name
        self.use_hierarchical_indexing = use_hierarchical_indexing
        
        # Initialize embedding generator
        self.embedding_generator = SpecializedEmbeddingGenerator(
            default_model_name=embedding_model_name,
            code_model_name="thenlper/gte-large",  # Good for code
            qa_model_name=embedding_model_name
        )
        
        # Initialize chunkers
        if chunking_config is None:
            chunking_config = {}
        
        self.chunker = SemanticChunker(
            chunk_size=chunking_config.get("chunk_size", 1000),
            chunk_overlap=chunking_config.get("chunk_overlap", 200),
            chunking_method=chunking_config.get("semantic_chunking_method", "sentence"),
            chunk_by_tokens=True
        )
        
        self.hierarchical_chunker = HierarchicalChunker(
            parent_chunk_size=chunking_config.get("parent_chunk_size", 2000),
            child_chunk_size=chunking_config.get("child_chunk_size", 500),
            parent_overlap=chunking_config.get("parent_overlap", 200),
            child_overlap=chunking_config.get("child_overlap", 50)
        )
        
        # Initialize vector stores
        self._init_vector_stores()
    
    def _init_vector_stores(self):
        """Initialize vector stores based on configuration"""
        if self.vector_store_provider == "chroma":
            # Regular vector store for documents or child nodes
            self.vector_store = Chroma(
                persist_directory=os.path.join(self.persist_directory, "documents"),
                embedding_function=self.embedding_generator,
            )
            
            # Parent vector store for hierarchical indexing
            if self.use_hierarchical_indexing:
                self.parent_vector_store = Chroma(
                    persist_directory=os.path.join(self.persist_directory, "parents"),
                    embedding_function=self.embedding_generator,
                )
        elif self.vector_store_provider == "faiss":
            # Check if FAISS index exists
            faiss_index_path = os.path.join(self.persist_directory, "documents.faiss")
            faiss_data_path = os.path.join(self.persist_directory, "documents.pkl")
            
            if os.path.exists(faiss_index_path) and os.path.exists(faiss_data_path):
                self.vector_store = FAISS.load_local(
                    self.persist_directory,
                    self.embedding_generator,
                    "documents"
                )
            else:
                # Create an empty FAISS index
                self.vector_store = FAISS(
                    embedding_function=self.embedding_generator,
                    index_name="documents"
                )
            
            # Parent vector store for hierarchical indexing
            if self.use_hierarchical_indexing:
                parent_faiss_index_path = os.path.join(self.persist_directory, "parents.faiss")
                parent_faiss_data_path = os.path.join(self.persist_directory, "parents.pkl")
                
                if os.path.exists(parent_faiss_index_path) and os.path.exists(parent_faiss_data_path):
                    self.parent_vector_store = FAISS.load_local(
                        self.persist_directory,
                        self.embedding_generator,
                        "parents"
                    )
                else:
                    # Create an empty FAISS index
                    self.parent_vector_store = FAISS(
                        embedding_function=self.embedding_generator,
                        index_name="parents"
                    )
        else:
            raise ValueError(f"Unsupported vector store provider: {self.vector_store_provider}")
    
    def _detect_content_type(self, file_path: str, text_content: str) -> str:
        """Detect the content type of a document"""
        file_extension = os.path.splitext(file_path)[1].lower() if file_path else ""
        
        # Code files
        code_extensions = ['.py', '.js', '.java', '.c', '.cpp', '.h', '.cs', '.ts', '.go']
        if file_extension in code_extensions:
            return "code"
        
        # Markdown files
        if file_extension in ['.md', '.markdown']:
            return "markdown"
        
        # HTML files
        if file_extension in ['.html', '.htm']:
            return "html"
        
        # Try to detect content type from content
        if "```" in text_content or "def " in text_content or "function" in text_content or "class" in text_content:
            return "code"
        
        if text_content.startswith('#') or "## " in text_content:
            return "markdown"
        
        # Default to text
        return "text"
    
    def index_file(self, file_path: str) -> Dict[str, Any]:
        """
        Index a single file
        
        Args:
            file_path: Path to the file to index
            
        Returns:
            Dictionary with indexing results
        """
        # Extract text from file
        text = extract_text_from_file(file_path)
        if not text:
            return {"success": False, "error": "Failed to extract text from file"}
        
        # Clean text
        text = clean_text(text)
        
        # Extract basic metadata
        metadata = extract_metadata_from_text(text)
        metadata["source"] = file_path
        metadata["filename"] = os.path.basename(file_path)
        
        # Detect content type
        content_type = self._detect_content_type(file_path, text)
        metadata["content_type"] = content_type
        
        return self.index_text(text, metadata)
    
    def index_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Index a text document
        
        Args:
            text: Text content to index
            metadata: Optional metadata for the document
            
        Returns:
            Dictionary with indexing results
        """
        if not text:
            return {"success": False, "error": "Empty text"}
        
        metadata = metadata or {}
        content_type = metadata.get("content_type", "text")
        
        # Generate document ID if not provided
        if "id" not in metadata:
            metadata["id"] = generate_document_id(text, metadata)
        
        # Apply appropriate chunking strategy
        if self.use_hierarchical_indexing:
            # Hierarchical chunking for RAPTOR
            chunks = self.hierarchical_chunker.split_text(text, metadata, content_type)
            
            # Index parent chunks
            parent_ids = []
            for parent in chunks["parents"]:
                parent_ids.append(parent.metadata["parent_id"])
                
            if self.vector_store_provider == "chroma":
                self.parent_vector_store.add_documents(chunks["parents"])
            elif self.vector_store_provider == "faiss":
                self.parent_vector_store.add_documents(chunks["parents"])
                # Save FAISS index
                self.parent_vector_store.save_local(self.persist_directory, "parents")
            
            # Index child chunks
            child_ids = []
            for child in chunks["children"]:
                child_ids.append(child.metadata.get("child_id"))
                
            if self.vector_store_provider == "chroma":
                self.vector_store.add_documents(chunks["children"])
            elif self.vector_store_provider == "faiss":
                self.vector_store.add_documents(chunks["children"])
                # Save FAISS index
                self.vector_store.save_local(self.persist_directory, "documents")
            
            return {
                "success": True,
                "document_id": metadata["id"],
                "parent_ids": parent_ids,
                "child_ids": child_ids,
                "parent_count": len(chunks["parents"]),
                "child_count": len(chunks["children"])
            }
        else:
            # Regular chunking
            chunks = self.chunker.split_text(text, metadata, content_type)
            
            # Store chunks in vector store
            chunk_ids = []
            for chunk in chunks:
                chunk_id = chunk.metadata.get("chunk_id", f"chunk_{chunks.index(chunk)}")
                chunk_ids.append(chunk_id)
            
            if self.vector_store_provider == "chroma":
                self.vector_store.add_documents(chunks)
            elif self.vector_store_provider == "faiss":
                self.vector_store.add_documents(chunks)
                # Save FAISS index
                self.vector_store.save_local(self.persist_directory, "documents")
            
            return {
                "success": True,
                "document_id": metadata["id"],
                "chunk_ids": chunk_ids,
                "chunk_count": len(chunks)
            }
    
    def index_directory(self, directory_path: str, glob_pattern: str = "**/*.*") -> Dict[str, Any]:
        """
        Index all files in a directory
        
        Args:
            directory_path: Path to the directory to index
            glob_pattern: Pattern to match files
            
        Returns:
            Dictionary with indexing results
        """
        # Check if directory exists
        if not os.path.isdir(directory_path):
            return {"success": False, "error": f"Directory not found: {directory_path}"}
        
        # Initialize results
        results = {
            "success": True,
            "total_files": 0,
            "indexed_files": 0,
            "failed_files": 0,
            "files": []
        }
        
        # Load files from directory
        for root, _, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                
                # Skip binary and non-text files
                file_extension = os.path.splitext(file)[1].lower()
                skip_extensions = ['.exe', '.bin', '.obj', '.jpg', '.png', '.gif', '.mp3', '.mp4', '.zip', '.tar', '.gz']
                
                if file_extension in skip_extensions:
                    continue
                
                results["total_files"] += 1
                
                try:
                    # Index file
                    file_result = self.index_file(file_path)
                    
                    if file_result.get("success", False):
                        results["indexed_files"] += 1
                    else:
                        results["failed_files"] += 1
                    
                    results["files"].append({
                        "file_path": file_path,
                        "result": file_result
                    })
                except Exception as e:
                    results["failed_files"] += 1
                    results["files"].append({
                        "file_path": file_path,
                        "result": {"success": False, "error": str(e)}
                    })
        
        return results 