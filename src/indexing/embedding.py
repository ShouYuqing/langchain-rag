"""
Embedding models and strategies for vector embeddings
"""
from typing import Dict, List, Optional, Union, Any, Callable
import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings, CohereEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

class SpecializedEmbeddingGenerator:
    """
    Generate embeddings using specialized models for different document types
    """
    
    def __init__(
        self,
        default_model_name: str = "text-embedding-3-large",
        code_model_name: Optional[str] = "thenlper/gte-large",
        qa_model_name: Optional[str] = "text-embedding-3-large",
        combine_strategy: str = "weighted_average"
    ):
        self.default_model_name = default_model_name
        self.code_model_name = code_model_name
        self.qa_model_name = qa_model_name
        self.combine_strategy = combine_strategy
        
        # Initialize default embedding model
        if "text-embedding" in default_model_name:
            self.default_embedder = OpenAIEmbeddings(model=default_model_name)
        elif default_model_name.startswith("cohere"):
            self.default_embedder = CohereEmbeddings(model=default_model_name)
        else:
            self.default_embedder = HuggingFaceEmbeddings(model_name=default_model_name)
        
        # Initialize specialized embedding models if provided
        self.specialized_embedders = {}
        
        if code_model_name:
            self.specialized_embedders["code"] = HuggingFaceEmbeddings(model_name=code_model_name)
        
        if qa_model_name and qa_model_name != default_model_name:
            if "text-embedding" in qa_model_name:
                self.specialized_embedders["qa"] = OpenAIEmbeddings(model=qa_model_name)
            elif qa_model_name.startswith("cohere"):
                self.specialized_embedders["qa"] = CohereEmbeddings(model=qa_model_name)
            else:
                self.specialized_embedders["qa"] = HuggingFaceEmbeddings(model_name=qa_model_name)
    
    def _get_embedder_for_content_type(self, content_type: Optional[str] = None) -> Embeddings:
        """Get the appropriate embedder for the content type"""
        if not content_type:
            return self.default_embedder
        
        return self.specialized_embedders.get(content_type, self.default_embedder)
    
    def _combine_embeddings(self, embeddings_list: List[List[float]], weights: Optional[List[float]] = None) -> List[float]:
        """
        Combine multiple embeddings into a single embedding
        
        Args:
            embeddings_list: List of embedding vectors to combine
            weights: Optional weights for each embedding vector
            
        Returns:
            Combined embedding vector
        """
        if not embeddings_list:
            raise ValueError("No embeddings provided to combine")
        
        if len(embeddings_list) == 1:
            return embeddings_list[0]
        
        # Convert to numpy arrays for easier manipulation
        embeddings_array = np.array(embeddings_list)
        
        # Apply combination strategy
        if self.combine_strategy == "average":
            # Simple average
            combined = np.mean(embeddings_array, axis=0)
        elif self.combine_strategy == "weighted_average":
            # Weighted average (default to equal weights if not provided)
            if weights is None:
                weights = [1.0] * len(embeddings_list)
            
            # Normalize weights
            weights = np.array(weights) / sum(weights)
            combined = np.average(embeddings_array, axis=0, weights=weights)
        elif self.combine_strategy == "concatenate":
            # Concatenate and reduce dimension (with PCA-like approach)
            concatenated = np.concatenate(embeddings_array)
            # Simple dimensionality reduction - take first n components
            n = len(embeddings_list[0])
            if len(concatenated) > n:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=n)
                combined = pca.fit_transform([concatenated])[0]
            else:
                combined = concatenated
        else:
            # Default to average
            combined = np.mean(embeddings_array, axis=0)
        
        # Normalize the combined embedding
        combined_norm = np.linalg.norm(combined)
        if combined_norm > 0:
            combined = combined / combined_norm
        
        return combined.tolist()
    
    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        """
        Generate embeddings for a list of documents
        
        Args:
            documents: List of documents to embed
            
        Returns:
            List of embedding vectors
        """
        # Group documents by content type
        documents_by_type = {}
        for doc in documents:
            content_type = doc.metadata.get("content_type")
            if content_type not in documents_by_type:
                documents_by_type[content_type] = []
            documents_by_type[content_type].append(doc)
        
        # Generate embeddings for each content type
        all_embeddings = []
        for content_type, docs in documents_by_type.items():
            embedder = self._get_embedder_for_content_type(content_type)
            texts = [doc.page_content for doc in docs]
            embeddings = embedder.embed_documents(texts)
            
            # Add embeddings in the same order as original documents
            doc_to_embedding = {doc.page_content: emb for doc, emb in zip(docs, embeddings)}
            for doc in docs:
                all_embeddings.append(doc_to_embedding[doc.page_content])
        
        return all_embeddings
    
    def embed_query(self, query: str, content_type: Optional[str] = None) -> List[float]:
        """
        Generate embedding for a query
        
        Args:
            query: Query string to embed
            content_type: Optional content type hint for specialized embeddings
            
        Returns:
            Embedding vector
        """
        embeddings_to_combine = []
        weights = []
        
        # Get default embedding
        default_embedding = self.default_embedder.embed_query(query)
        embeddings_to_combine.append(default_embedding)
        weights.append(1.0)
        
        # Get specialized embeddings if they would be useful
        if "code" in query.lower() and "code" in self.specialized_embedders:
            code_embedding = self.specialized_embedders["code"].embed_query(query)
            embeddings_to_combine.append(code_embedding)
            weights.append(0.7)  # Lower weight for specialized embeddings
        
        if ("?" in query or "what" in query.lower() or "how" in query.lower()) and "qa" in self.specialized_embedders:
            qa_embedding = self.specialized_embedders["qa"].embed_query(query)
            embeddings_to_combine.append(qa_embedding)
            weights.append(0.8)  # Higher weight for QA embeddings as they're more relevant
        
        # Combine embeddings if multiple exist
        if len(embeddings_to_combine) > 1:
            return self._combine_embeddings(embeddings_to_combine, weights)
        else:
            return default_embedding 