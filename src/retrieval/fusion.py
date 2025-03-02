"""
Retrieval fusion techniques to combine results from multiple retrievers
"""
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import numpy as np
from collections import defaultdict
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

class RetrievalFuser:
    """
    Base class for retrieval fusion techniques
    """
    
    def fuse_results(
        self, 
        query: str, 
        retrieval_results: List[List[Document]], 
        top_k: int = 10
    ) -> List[Document]:
        """
        Fuse multiple retrieval results into a single list
        
        Args:
            query: Query string
            retrieval_results: List of retrieval results, each containing list of Documents
            top_k: Number of results to return
            
        Returns:
            Fused list of documents
        """
        raise NotImplementedError("Subclasses must implement fuse_results method")


class RRFusion(RetrievalFuser):
    """
    Reciprocal Rank Fusion algorithm for combining retrieval results
    """
    
    def __init__(self, k: float = 60.0):
        """
        Initialize RRF fusion
        
        Args:
            k: Constant in RRF formula (default: 60.0)
        """
        self.k = k
    
    def fuse_results(
        self, 
        query: str, 
        retrieval_results: List[List[Document]], 
        top_k: int = 10
    ) -> List[Document]:
        """
        Fuse results using Reciprocal Rank Fusion
        
        Args:
            query: Query string
            retrieval_results: List of retrieval results, each containing list of Documents
            top_k: Number of results to return
            
        Returns:
            Fused list of documents
        """
        # Create a dictionary to store document scores
        doc_scores = defaultdict(float)
        
        # Process each retrieval result list
        for result_list in retrieval_results:
            # Process each document in the result
            for rank, doc in enumerate(result_list):
                # Calculate the RRF score for this document in this result list
                # The formula is 1 / (k + rank)
                score = 1.0 / (self.k + rank)
                
                # Use document content hash as key to identify unique documents
                doc_key = hash(doc.page_content)
                
                # Add this score to the document's total score
                doc_scores[doc_key] = doc_scores[doc_key] + score
                
                # Store the document object mapped to its key
                if not hasattr(self, "doc_map"):
                    self.doc_map = {}
                self.doc_map[doc_key] = doc
        
        # Sort documents by score in descending order
        sorted_doc_keys = sorted(doc_scores.keys(), key=lambda k: doc_scores[k], reverse=True)
        
        # Return the top_k documents
        result_docs = []
        for doc_key in sorted_doc_keys[:top_k]:
            doc = self.doc_map[doc_key]
            # Add the fusion score to the document metadata
            doc.metadata["fusion_score"] = doc_scores[doc_key]
            result_docs.append(doc)
        
        return result_docs


class WeightedFusion(RetrievalFuser):
    """
    Weighted score fusion for combining retrieval results
    """
    
    def __init__(self, weights: Optional[List[float]] = None):
        """
        Initialize weighted fusion
        
        Args:
            weights: List of weights for each retriever (default: equal weights)
        """
        self.weights = weights
    
    def fuse_results(
        self, 
        query: str, 
        retrieval_results: List[List[Document]], 
        top_k: int = 10
    ) -> List[Document]:
        """
        Fuse results using weighted score combination
        
        Args:
            query: Query string
            retrieval_results: List of retrieval results, each containing list of Documents
            top_k: Number of results to return
            
        Returns:
            Fused list of documents
        """
        # If weights are not provided, use equal weights
        if self.weights is None:
            self.weights = [1.0 / len(retrieval_results)] * len(retrieval_results)
        
        # Ensure weights sum to 1.0
        weight_sum = sum(self.weights)
        normalized_weights = [w / weight_sum for w in self.weights]
        
        # Create a dictionary to store document scores
        doc_scores = defaultdict(float)
        doc_map = {}
        
        # Process each retrieval result list with its weight
        for i, (result_list, weight) in enumerate(zip(retrieval_results, normalized_weights)):
            # Calculate max_score for normalization within this result list
            max_score = 1.0
            if result_list and "relevance_score" in result_list[0].metadata:
                scores = [doc.metadata.get("relevance_score", 0.0) for doc in result_list]
                if scores:
                    max_score = max(scores) if max(scores) > 0 else 1.0
            
            # Process each document in the result
            for rank, doc in enumerate(result_list):
                # Get document score from metadata or use rank-based score
                if "relevance_score" in doc.metadata:
                    # Normalize the score
                    score = doc.metadata["relevance_score"] / max_score
                else:
                    # Use reciprocal rank as score
                    score = 1.0 / (rank + 1)
                
                # Weight the score
                weighted_score = weight * score
                
                # Use document content hash as key
                doc_key = hash(doc.page_content)
                
                # Add this score to the document's total score
                doc_scores[doc_key] = doc_scores[doc_key] + weighted_score
                
                # Store the document object
                doc_map[doc_key] = doc
        
        # Sort documents by score in descending order
        sorted_doc_keys = sorted(doc_scores.keys(), key=lambda k: doc_scores[k], reverse=True)
        
        # Return the top_k documents
        result_docs = []
        for doc_key in sorted_doc_keys[:top_k]:
            doc = doc_map[doc_key]
            # Add the fusion score to the document metadata
            doc.metadata["fusion_score"] = doc_scores[doc_key]
            result_docs.append(doc)
        
        return result_docs


class MultiStrategyRetriever(BaseRetriever):
    """
    Retriever that combines multiple retrieval strategies
    """
    
    def __init__(
        self,
        retrievers: List[BaseRetriever],
        fusion_strategy: Optional[RetrievalFuser] = None,
        weights: Optional[List[float]] = None,
        top_k: int = 10
    ):
        """
        Initialize multi-strategy retriever
        
        Args:
            retrievers: List of retrievers to use
            fusion_strategy: Strategy for fusing results (default: WeightedFusion)
            weights: Optional weights for each retriever
            top_k: Number of results to return
        """
        self.retrievers = retrievers
        self.fusion_strategy = fusion_strategy or WeightedFusion(weights=weights)
        self.top_k = top_k
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        Get relevant documents from multiple retrievers and fuse results
        
        Args:
            query: Query string
            run_manager: Callback manager
            
        Returns:
            Fused list of documents
        """
        # Get results from each retriever
        retrieval_results = []
        for retriever in self.retrievers:
            results = retriever.get_relevant_documents(
                query, callbacks=run_manager.get_child_callbacks()
            )
            retrieval_results.append(results)
        
        # Fuse results
        fused_docs = self.fusion_strategy.fuse_results(
            query=query,
            retrieval_results=retrieval_results,
            top_k=self.top_k
        )
        
        return fused_docs
    
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        Asynchronously get relevant documents from multiple retrievers and fuse results
        
        Args:
            query: Query string
            run_manager: Callback manager
            
        Returns:
            Fused list of documents
        """
        # Get results from each retriever asynchronously
        retrieval_results = []
        for retriever in self.retrievers:
            results = await retriever.aget_relevant_documents(
                query, callbacks=run_manager.get_child_callbacks()
            )
            retrieval_results.append(results)
        
        # Fuse results
        fused_docs = self.fusion_strategy.fuse_results(
            query=query,
            retrieval_results=retrieval_results,
            top_k=self.top_k
        )
        
        return fused_docs 