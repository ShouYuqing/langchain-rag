"""
Advanced retriever that integrates query expansion, reranking, and fusion for optimal retrieval
"""
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
import logging
from pydantic import Field
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseLanguageModel

from src.retrieval.query_expansion import QueryExpander, HypotheticalDocumentEmbedder
from src.retrieval.reranking import CustomReranker, RerankerRetriever
from src.retrieval.fusion import MultiStrategyRetriever, WeightedFusion, RRFusion
from src.indexing.embedding import SpecializedEmbeddingGenerator

logger = logging.getLogger(__name__)

class AdvancedRetriever(BaseRetriever):
    """
    Advanced retriever that combines query expansion, multiple retrieval strategies, 
    and result reranking for optimal document retrieval
    """
    
    def __init__(
        self,
        base_retriever: BaseRetriever,
        llm: Optional[BaseLanguageModel] = None,
        use_query_expansion: bool = True,
        use_hypothetical_document: bool = True,
        use_reranking: bool = True,
        use_specialized_embedding: bool = False,
        embedding_generator: Optional[SpecializedEmbeddingGenerator] = None,
        top_k: int = 10,
        num_expanded_queries: int = 3
    ):
        """
        Initialize advanced retriever
        
        Args:
            base_retriever: Base retriever to use
            llm: Language model for query expansion
            use_query_expansion: Whether to use query expansion
            use_hypothetical_document: Whether to use hypothetical document embedding
            use_reranking: Whether to use result reranking
            use_specialized_embedding: Whether to use specialized embeddings
            embedding_generator: Embedding generator for specialized embeddings
            top_k: Number of results to return
            num_expanded_queries: Number of expanded queries to generate
        """
        self.base_retriever = base_retriever
        self.llm = llm
        self.use_query_expansion = use_query_expansion
        self.use_hypothetical_document = use_hypothetical_document
        self.use_reranking = use_reranking
        self.use_specialized_embedding = use_specialized_embedding
        self.embedding_generator = embedding_generator
        self.top_k = top_k
        self.num_expanded_queries = num_expanded_queries
        
        # Initialize query expansion if enabled
        if self.use_query_expansion and self.llm:
            self.query_expander = QueryExpander(llm=self.llm)
        else:
            self.query_expander = None
        
        # Initialize hypothetical document embedder if enabled
        if self.use_hypothetical_document and self.llm:
            self.hypothetical_embedder = HypotheticalDocumentEmbedder(llm=self.llm)
        else:
            self.hypothetical_embedder = None
        
        # Initialize reranker if enabled
        if self.use_reranking:
            self.reranker = CustomReranker()
        else:
            self.reranker = None
    
    def _get_retrievers_for_query(
        self, 
        query: str, 
        run_manager: CallbackManagerForRetrieverRun
    ) -> List[BaseRetriever]:
        """
        Generate multiple retrievers based on the query expansion strategies
        
        Args:
            query: Original query
            run_manager: Callback manager
            
        Returns:
            List of retrievers
        """
        retrievers = [self.base_retriever]
        
        # If query expansion is enabled and we have a query expander
        if self.use_query_expansion and self.query_expander:
            try:
                # Expand the query
                expanded_queries = self.query_expander.expand_query_sync(
                    query, self.num_expanded_queries
                )
                
                # Log expanded queries
                logger.info(f"Expanded queries: {expanded_queries.expanded_queries}")
                
                # Create a retriever for each expanded query
                for expanded_query in expanded_queries.expanded_queries:
                    # Create a wrapped retriever that substitutes the expanded query
                    expanded_retriever = ExpandedQueryRetriever(
                        base_retriever=self.base_retriever,
                        expanded_query=expanded_query
                    )
                    retrievers.append(expanded_retriever)
            except Exception as e:
                logger.error(f"Error in query expansion: {str(e)}")
        
        # If hypothetical document embedding is enabled and we have a hypothetical embedder
        if self.use_hypothetical_document and self.hypothetical_embedder and self.embedding_generator:
            try:
                # Generate a hypothetical document
                hypothetical_doc = self.hypothetical_embedder.generate_hypothetical_document_sync(query)
                
                # Log hypothetical document
                logger.info(f"Generated hypothetical document: {hypothetical_doc[:100]}...")
                
                # Create a retriever that uses the hypothetical document embedding
                hypothetical_retriever = HypotheticalDocumentRetriever(
                    base_retriever=self.base_retriever,
                    hypothetical_document=hypothetical_doc,
                    embedding_generator=self.embedding_generator
                )
                retrievers.append(hypothetical_retriever)
            except Exception as e:
                logger.error(f"Error in hypothetical document embedding: {str(e)}")
        
        return retrievers
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        Get relevant documents using multiple retrieval strategies
        
        Args:
            query: Query string
            run_manager: Callback manager
            
        Returns:
            List of relevant documents
        """
        # Get retrievers for the query
        retrievers = self._get_retrievers_for_query(query, run_manager)
        
        # If we have multiple retrievers, use fusion
        if len(retrievers) > 1:
            # Create multi-strategy retriever
            fusion_retriever = MultiStrategyRetriever(
                retrievers=retrievers,
                fusion_strategy=RRFusion(),
                top_k=self.top_k * 2  # Retrieve more docs for reranking
            )
            
            # Get results
            results = fusion_retriever.get_relevant_documents(
                query, callbacks=run_manager.get_child_callbacks()
            )
        else:
            # Use base retriever
            results = self.base_retriever.get_relevant_documents(
                query, callbacks=run_manager.get_child_callbacks()
            )
        
        # Apply reranking if enabled
        if self.use_reranking and self.reranker:
            # Get query embedding if we have an embedding generator
            query_embedding = None
            if self.use_specialized_embedding and self.embedding_generator:
                query_embedding = self.embedding_generator.embed_query(query)
            
            # Rerank results
            results = self.reranker.rerank_documents(
                query=query,
                documents=results,
                query_embedding=query_embedding,
                top_k=self.top_k
            )
        
        return results[:self.top_k]
    
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        Asynchronously get relevant documents using multiple retrieval strategies
        
        Args:
            query: Query string
            run_manager: Callback manager
            
        Returns:
            List of relevant documents
        """
        # Get retrievers for the query
        retrievers = self._get_retrievers_for_query(query, run_manager)
        
        # If we have multiple retrievers, use fusion
        if len(retrievers) > 1:
            # Create multi-strategy retriever
            fusion_retriever = MultiStrategyRetriever(
                retrievers=retrievers,
                fusion_strategy=RRFusion(),
                top_k=self.top_k * 2  # Retrieve more docs for reranking
            )
            
            # Get results
            results = await fusion_retriever.aget_relevant_documents(
                query, callbacks=run_manager.get_child_callbacks()
            )
        else:
            # Use base retriever
            results = await self.base_retriever.aget_relevant_documents(
                query, callbacks=run_manager.get_child_callbacks()
            )
        
        # Apply reranking if enabled
        if self.use_reranking and self.reranker:
            # Get query embedding if we have an embedding generator
            query_embedding = None
            if self.use_specialized_embedding and self.embedding_generator:
                query_embedding = self.embedding_generator.embed_query(query)
            
            # Rerank results
            results = self.reranker.rerank_documents(
                query=query,
                documents=results,
                query_embedding=query_embedding,
                top_k=self.top_k
            )
        
        return results[:self.top_k]


class ExpandedQueryRetriever(BaseRetriever):
    """
    Retriever that uses an expanded query instead of the original query
    """
    
    def __init__(self, base_retriever: BaseRetriever, expanded_query: str):
        """
        Initialize expanded query retriever
        
        Args:
            base_retriever: Base retriever to use
            expanded_query: Expanded query to use
        """
        self.base_retriever = base_retriever
        self.expanded_query = expanded_query
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        Get relevant documents using the expanded query
        
        Args:
            query: Original query (ignored)
            run_manager: Callback manager
            
        Returns:
            List of relevant documents
        """
        return self.base_retriever.get_relevant_documents(
            self.expanded_query, callbacks=run_manager.get_child_callbacks()
        )
    
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        Asynchronously get relevant documents using the expanded query
        
        Args:
            query: Original query (ignored)
            run_manager: Callback manager
            
        Returns:
            List of relevant documents
        """
        return await self.base_retriever.aget_relevant_documents(
            self.expanded_query, callbacks=run_manager.get_child_callbacks()
        )


class HypotheticalDocumentRetriever(BaseRetriever):
    """
    Retriever that uses a hypothetical document embedding to retrieve similar documents
    """
    
    def __init__(
        self,
        base_retriever: BaseRetriever,
        hypothetical_document: str,
        embedding_generator: SpecializedEmbeddingGenerator
    ):
        """
        Initialize hypothetical document retriever
        
        Args:
            base_retriever: Base retriever to use
            hypothetical_document: Hypothetical document text
            embedding_generator: Embedding generator for creating embeddings
        """
        self.base_retriever = base_retriever
        self.hypothetical_document = hypothetical_document
        self.embedding_generator = embedding_generator
        
        # Generate embedding for the hypothetical document
        self.hypothetical_embedding = self.embedding_generator.embed_documents(
            [self.hypothetical_document]
        )[0]
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        Get relevant documents using the hypothetical document embedding
        
        Args:
            query: Original query (used for metadata only)
            run_manager: Callback manager
            
        Returns:
            List of relevant documents
        """
        # Check if base_retriever has a vectorstore
        if hasattr(self.base_retriever, "vectorstore"):
            # Use vectorstore's similarity_search_by_vector
            docs = self.base_retriever.vectorstore.similarity_search_by_vector(
                self.hypothetical_embedding,
                k=self.base_retriever.k if hasattr(self.base_retriever, "k") else 4
            )
            return docs
        else:
            # Fall back to original query if we can't use the embedding directly
            return self.base_retriever.get_relevant_documents(
                query, callbacks=run_manager.get_child_callbacks()
            )
    
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        Asynchronously get relevant documents using the hypothetical document embedding
        
        Args:
            query: Original query (used for metadata only)
            run_manager: Callback manager
            
        Returns:
            List of relevant documents
        """
        # Check if base_retriever has a vectorstore
        if hasattr(self.base_retriever, "vectorstore") and hasattr(self.base_retriever.vectorstore, "asimilarity_search_by_vector"):
            # Use vectorstore's similarity_search_by_vector
            docs = await self.base_retriever.vectorstore.asimilarity_search_by_vector(
                self.hypothetical_embedding,
                k=self.base_retriever.k if hasattr(self.base_retriever, "k") else 4
            )
            return docs
        else:
            # Fall back to original query if we can't use the embedding directly
            return await self.base_retriever.aget_relevant_documents(
                query, callbacks=run_manager.get_child_callbacks()
            ) 