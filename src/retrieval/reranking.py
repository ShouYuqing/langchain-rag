"""
Advanced reranking mechanisms for retrieval results
"""
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import numpy as np
from langchain_community.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_community.retrievers.document_compressors import CohereRerank
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

class RelevanceScorer:
    """
    Custom relevance scoring algorithms for document relevance
    """
    
    @staticmethod
    def bm25_score(query: str, document: str) -> float:
        """
        Calculate BM25 relevance score (simplified implementation)
        
        Args:
            query: User query
            document: Document text
            
        Returns:
            Float score of relevance
        """
        # Tokenize
        query_terms = query.lower().split()
        doc_terms = document.lower().split()
        
        # Document length
        doc_len = len(doc_terms)
        
        # IDF constants (simplified)
        k1 = 1.5
        b = 0.75
        avg_doc_len = 500  # Assumed average document length
        
        # Calculate score
        score = 0.0
        term_freqs = {}
        
        for term in doc_terms:
            if term not in term_freqs:
                term_freqs[term] = 0
            term_freqs[term] += 1
        
        for term in query_terms:
            if term in term_freqs:
                # Term frequency in document
                tf = term_freqs[term]
                
                # Simplified IDF (normally calculated across corpus)
                idf = 1.0
                
                # BM25 scoring formula
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_len / avg_doc_len)
                score += idf * numerator / denominator
        
        return score
    
    @staticmethod
    def semantic_similarity_score(query_embedding: List[float], doc_embedding: List[float]) -> float:
        """
        Calculate cosine similarity between query and document embeddings
        
        Args:
            query_embedding: Embedding vector for query
            doc_embedding: Embedding vector for document
            
        Returns:
            Float score of similarity
        """
        # Convert to numpy arrays
        query_vec = np.array(query_embedding)
        doc_vec = np.array(doc_embedding)
        
        # Calculate cosine similarity
        norm_query = np.linalg.norm(query_vec)
        norm_doc = np.linalg.norm(doc_vec)
        
        if norm_query == 0 or norm_doc == 0:
            return 0.0
        
        return np.dot(query_vec, doc_vec) / (norm_query * norm_doc)
    
    @staticmethod
    def hybrid_score(
        query: str, 
        document: str, 
        query_embedding: Optional[List[float]] = None, 
        doc_embedding: Optional[List[float]] = None,
        lexical_weight: float = 0.3,
        semantic_weight: float = 0.7
    ) -> float:
        """
        Calculate hybrid score combining lexical and semantic similarity
        
        Args:
            query: User query
            document: Document text
            query_embedding: Optional embedding vector for query
            doc_embedding: Optional embedding vector for document
            lexical_weight: Weight for lexical score (BM25)
            semantic_weight: Weight for semantic score (embedding similarity)
            
        Returns:
            Float score of relevance
        """
        # Calculate lexical score
        lexical_score = RelevanceScorer.bm25_score(query, document)
        
        # Calculate semantic score if embeddings are provided
        semantic_score = 0.0
        if query_embedding is not None and doc_embedding is not None:
            semantic_score = RelevanceScorer.semantic_similarity_score(query_embedding, doc_embedding)
        
        # Combine scores
        return (lexical_weight * lexical_score + semantic_weight * semantic_score) / (lexical_weight + semantic_weight)

class CustomReranker:
    """
    Custom reranker for retrieval results
    """
    
    def __init__(
        self,
        scoring_function: Optional[Callable] = None,
        cohere_reranker: Optional[CohereRerank] = None,
        min_relevance_score: float = 0.7
    ):
        self.scoring_function = scoring_function or RelevanceScorer.hybrid_score
        self.cohere_reranker = cohere_reranker
        self.min_relevance_score = min_relevance_score
    
    def rerank_documents(
        self,
        query: str,
        documents: List[Document],
        query_embedding: Optional[List[float]] = None,
        top_k: Optional[int] = None
    ) -> List[Document]:
        """
        Rerank documents based on relevance to query
        
        Args:
            query: User query
            documents: List of documents to rerank
            query_embedding: Optional embedding vector for query
            top_k: Optional number of documents to return
            
        Returns:
            Reranked list of documents
        """
        # Use Cohere reranker if available
        if self.cohere_reranker is not None:
            try:
                reranked_docs = self.cohere_reranker.compress_documents(documents, query)
                
                # Filter out documents below minimum relevance threshold
                filtered_docs = []
                for doc in reranked_docs:
                    relevance_score = doc.metadata.get("relevance_score", 0.0)
                    if relevance_score >= self.min_relevance_score:
                        filtered_docs.append(doc)
                
                # Return top_k documents or all if top_k is None
                if top_k is not None:
                    return filtered_docs[:top_k]
                return filtered_docs
            
            except Exception as e:
                print(f"Error using Cohere reranker: {str(e)}")
                # Fall back to custom scoring function
                pass
        
        # Custom reranking using scoring function
        scored_docs = []
        for doc in documents:
            # Extract document embedding if available
            doc_embedding = doc.metadata.get("embedding")
            
            # Calculate score
            score = self.scoring_function(
                query=query,
                document=doc.page_content,
                query_embedding=query_embedding,
                doc_embedding=doc_embedding
            )
            
            # Add score to metadata
            doc.metadata["relevance_score"] = score
            scored_docs.append((doc, score))
        
        # Sort by score in descending order
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Filter out documents below minimum relevance threshold
        filtered_docs = [doc for doc, score in scored_docs if score >= self.min_relevance_score]
        
        # Return top_k documents or all if top_k is None
        if top_k is not None:
            return filtered_docs[:top_k]
        return filtered_docs

class RerankerRetriever:
    """
    Retriever with integrated reranking
    """
    
    def __init__(
        self,
        base_retriever: Any,
        reranker: Optional[CustomReranker] = None,
        top_k: int = 5
    ):
        self.base_retriever = base_retriever
        self.reranker = reranker or CustomReranker()
        self.top_k = top_k
    
    def get_relevant_documents(
        self, 
        query: str, 
        **kwargs
    ) -> List[Document]:
        """
        Retrieve and rerank documents
        
        Args:
            query: User query
            
        Returns:
            List of reranked documents
        """
        # Get documents from base retriever
        docs = self.base_retriever.get_relevant_documents(query, **kwargs)
        
        # Rerank documents
        reranked_docs = self.reranker.rerank_documents(
            query=query,
            documents=docs,
            top_k=self.top_k
        )
        
        return reranked_docs
    
    async def aget_relevant_documents(
        self,
        query: str,
        **kwargs
    ) -> List[Document]:
        """
        Asynchronously retrieve and rerank documents
        
        Args:
            query: User query
            
        Returns:
            List of reranked documents
        """
        # Get documents from base retriever
        docs = await self.base_retriever.aget_relevant_documents(query, **kwargs)
        
        # Rerank documents
        reranked_docs = self.reranker.rerank_documents(
            query=query,
            documents=docs,
            top_k=self.top_k
        )
        
        return reranked_docs 