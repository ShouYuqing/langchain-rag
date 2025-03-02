"""
Main RAG system manager that orchestrates all components of the Retrieval-Augmented Generation system
"""
import logging
from typing import Dict, List, Optional, Any, Union, Callable

from langchain_core.language_models import BaseLanguageModel
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from src.retrieval.advanced_retriever import AdvancedRetriever
from src.context_processing.processor import ContextProcessor
from src.context_processing.augmentation import ContextAugmenter
from src.context_processing.reranking import CustomReranker
from src.generation.answer_generator import LLMAnswerGenerator, StreamingAnswerGenerator
from src.routing.router import QueryRouter, QueryClassification

logger = logging.getLogger(__name__)


class RAGQueryResult(BaseModel):
    """Model for a complete RAG query result"""
    query: str = Field(..., description="The original query")
    answer: str = Field(..., description="The generated answer")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source documents used")
    retrieval_metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata about the retrieval process")
    query_classification: Optional[QueryClassification] = Field(default=None, description="Classification of the query")
    execution_time: Dict[str, float] = Field(default_factory=dict, description="Execution time for each component")


class RAGSystem:
    """
    Main RAG system that orchestrates all components of the retrieval-augmented generation pipeline
    """
    
    def __init__(
        self,
        retriever: BaseRetriever,
        llm: BaseLanguageModel,
        use_advanced_retrieval: bool = True,
        use_context_processing: bool = True,
        use_context_augmentation: bool = False,
        use_query_routing: bool = False,
        rerank_results: bool = True,
        specialized_retrievers: Optional[Dict[str, BaseRetriever]] = None,
        streaming: bool = False,
        max_context_length: int = 4000,
        default_top_k: int = 5,
        num_expanded_queries: int = 3,
        system_prompt: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Initialize the RAG system with all its components
        
        Args:
            retriever: Base document retriever
            llm: Language model for generation and other tasks
            use_advanced_retrieval: Whether to use advanced retrieval techniques
            use_context_processing: Whether to use context processing 
            use_context_augmentation: Whether to augment context with additional information
            use_query_routing: Whether to use query routing to specialized retrievers
            rerank_results: Whether to rerank retrieval results
            specialized_retrievers: Dictionary of specialized retrievers for different domains
            streaming: Whether to use streaming generation
            max_context_length: Maximum context length in tokens
            default_top_k: Default number of documents to retrieve
            num_expanded_queries: Number of expanded queries to generate
            system_prompt: Custom system prompt for the answer generator
            verbose: Whether to log detailed information
        """
        self.base_retriever = retriever
        self.llm = llm
        self.use_advanced_retrieval = use_advanced_retrieval
        self.use_context_processing = use_context_processing
        self.use_context_augmentation = use_context_augmentation
        self.use_query_routing = use_query_routing
        self.rerank_results = rerank_results
        self.specialized_retrievers = specialized_retrievers or {}
        self.streaming = streaming
        self.max_context_length = max_context_length
        self.default_top_k = default_top_k
        self.num_expanded_queries = num_expanded_queries
        self.system_prompt = system_prompt
        self.verbose = verbose
        
        # Set up logging
        if verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)
        
        # Initialize components
        self._init_components()
    
    def _init_components(self):
        """Initialize all RAG system components based on configuration"""
        # Initialize advanced retriever if enabled
        if self.use_advanced_retrieval:
            self.retriever = AdvancedRetriever(
                base_retriever=self.base_retriever,
                llm=self.llm,
                use_query_expansion=True,
                use_hypothetical_document=True,
                use_reranking=self.rerank_results,
                use_semantic_embeddings=True,
                top_k=self.default_top_k,
                num_expanded_queries=self.num_expanded_queries
            )
        else:
            self.retriever = self.base_retriever
        
        # Initialize context processor if enabled
        if self.use_context_processing:
            self.context_processor = ContextProcessor(
                max_context_length=self.max_context_length,
                deduplicate=True,
                deduplication_threshold=0.85,
                filter_irrelevant=True,
                relevance_threshold=0.25,
                remove_redundant=True,
                order_by_relevance=True
            )
        else:
            self.context_processor = None
        
        # Initialize context augmenter if enabled
        if self.use_context_augmentation:
            self.context_augmenter = ContextAugmenter(
                llm=self.llm,
                add_summaries=True,
                add_entity_definitions=True,
                add_knowledge_graph=False,
                add_cross_references=True
            )
        else:
            self.context_augmenter = None
        
        # Initialize query router if enabled
        if self.use_query_routing:
            self.router = QueryRouter(
                llm=self.llm,
                specialized_retrievers=self.specialized_retrievers,
                default_retriever=self.retriever
            )
        else:
            self.router = None
        
        # Initialize reranker if enabled and not already part of advanced retriever
        if self.rerank_results and not self.use_advanced_retrieval:
            self.reranker = CustomReranker(
                llm=self.llm, 
                use_keywords=True,
                use_embeddings=True,
                use_llm=True,
                keywords_weight=0.3,
                embeddings_weight=0.3,
                llm_weight=0.4
            )
        else:
            self.reranker = None
        
        # Initialize answer generator
        if self.streaming:
            self.answer_generator = StreamingAnswerGenerator(
                llm=self.llm,
                include_sources=True,
                context_processor=self.context_processor,
                max_context_length=self.max_context_length,
                default_system_prompt=self.system_prompt
            )
        else:
            self.answer_generator = LLMAnswerGenerator(
                llm=self.llm,
                use_structured_output=True,
                include_sources=True,
                context_processor=self.context_processor,
                max_context_length=self.max_context_length,
                default_system_prompt=self.system_prompt
            )
    
    async def _process_query(
        self, 
        query: str, 
        retriever: Optional[BaseRetriever] = None,
        top_k: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Process a query through the RAG pipeline
        
        Args:
            query: User query
            retriever: Optional specific retriever to use
            top_k: Optional number of documents to retrieve
            
        Returns:
            Dictionary with processing results and metadata
        """
        result = {
            "query": query,
            "execution_time": {},
            "retrieval_metadata": {}
        }
        
        # Use router if enabled
        if self.use_query_routing and self.router:
            try:
                import time
                start_time = time.time()
                
                routing_result = await self.router.route_query(query)
                result["query_classification"] = routing_result["classification"]
                selected_retriever = routing_result["retriever"]
                
                result["execution_time"]["routing"] = time.time() - start_time
                result["retrieval_metadata"]["routing"] = routing_result
                
                logger.debug(f"Query routed to {selected_retriever.__class__.__name__}")
            except Exception as e:
                logger.error(f"Error in query routing: {str(e)}")
                selected_retriever = retriever or self.retriever
        else:
            selected_retriever = retriever or self.retriever
        
        # Retrieve documents
        try:
            import time
            start_time = time.time()
            
            docs = await selected_retriever.aget_relevant_documents(
                query, 
                k=top_k or self.default_top_k
            )
            
            result["execution_time"]["retrieval"] = time.time() - start_time
            result["retrieval_metadata"]["num_docs_retrieved"] = len(docs)
            
            logger.debug(f"Retrieved {len(docs)} documents")
        except Exception as e:
            logger.error(f"Error in document retrieval: {str(e)}")
            docs = []
        
        # Apply reranking if enabled and not already handled by the retriever
        if (self.rerank_results and self.reranker and 
            not (self.use_advanced_retrieval and isinstance(selected_retriever, AdvancedRetriever))):
            try:
                import time
                start_time = time.time()
                
                reranked_docs = await self.reranker.rerank(query, docs)
                docs = reranked_docs
                
                result["execution_time"]["reranking"] = time.time() - start_time
                
                logger.debug("Documents reranked")
            except Exception as e:
                logger.error(f"Error in reranking: {str(e)}")
        
        # Apply context processing if enabled
        if self.use_context_processing and self.context_processor:
            try:
                import time
                start_time = time.time()
                
                processed_docs = self.context_processor.process_documents(docs, query=query)
                docs = processed_docs
                
                result["execution_time"]["context_processing"] = time.time() - start_time
                result["retrieval_metadata"]["num_docs_after_processing"] = len(docs)
                
                logger.debug(f"Context processed, {len(docs)} documents after processing")
            except Exception as e:
                logger.error(f"Error in context processing: {str(e)}")
        
        # Apply context augmentation if enabled
        if self.use_context_augmentation and self.context_augmenter and docs:
            try:
                import time
                start_time = time.time()
                
                augmented_docs = await self.context_augmenter.augment_documents(docs)
                docs = augmented_docs
                
                result["execution_time"]["context_augmentation"] = time.time() - start_time
                
                logger.debug("Context augmented with additional information")
            except Exception as e:
                logger.error(f"Error in context augmentation: {str(e)}")
        
        # Generate answer
        try:
            import time
            start_time = time.time()
            
            answer_result = await self.answer_generator.generate_answer(query, docs)
            
            # Handle different return types from answer generators
            if isinstance(answer_result, dict):
                result["answer"] = answer_result.get("answer", "")
                result["sources"] = answer_result.get("sources", [])
            else:
                result["answer"] = answer_result
                result["sources"] = []
            
            result["execution_time"]["answer_generation"] = time.time() - start_time
            
            logger.debug("Answer generated")
        except Exception as e:
            logger.error(f"Error in answer generation: {str(e)}")
            result["answer"] = f"Error generating answer: {str(e)}"
            result["sources"] = []
        
        # Calculate total execution time
        result["execution_time"]["total"] = sum(result["execution_time"].values())
        
        return result
    
    async def process_query(
        self, 
        query: str, 
        retriever: Optional[BaseRetriever] = None,
        top_k: Optional[int] = None
    ) -> RAGQueryResult:
        """
        Process a query and return a structured result
        
        Args:
            query: User query
            retriever: Optional specific retriever to use
            top_k: Optional number of documents to retrieve
            
        Returns:
            RAGQueryResult object with answer and metadata
        """
        result_dict = await self._process_query(query, retriever, top_k)
        return RAGQueryResult(**result_dict)
    
    def process_query_sync(
        self, 
        query: str, 
        retriever: Optional[BaseRetriever] = None,
        top_k: Optional[int] = None
    ) -> RAGQueryResult:
        """
        Process a query synchronously and return a structured result
        
        Args:
            query: User query
            retriever: Optional specific retriever to use
            top_k: Optional number of documents to retrieve
            
        Returns:
            RAGQueryResult object with answer and metadata
        """
        import asyncio
        result_dict = asyncio.run(self._process_query(query, retriever, top_k))
        return RAGQueryResult(**result_dict)
    
    async def process_queries(
        self, 
        queries: List[str],
        callback: Optional[Callable[[str, RAGQueryResult], None]] = None
    ) -> List[RAGQueryResult]:
        """
        Process multiple queries and return structured results
        
        Args:
            queries: List of user queries
            callback: Optional callback function to call with each result
            
        Returns:
            List of RAGQueryResult objects
        """
        results = []
        for query in queries:
            result = await self.process_query(query)
            results.append(result)
            
            if callback:
                callback(query, result)
        
        return results
    
    async def generate_streaming_response(
        self, 
        query: str, 
        retriever: Optional[BaseRetriever] = None,
        top_k: Optional[int] = None
    ):
        """
        Generate a streaming response for a query
        
        Args:
            query: User query
            retriever: Optional specific retriever to use
            top_k: Optional number of documents to retrieve
            
        Yields:
            Text chunks for streaming response
        """
        if not self.streaming or not isinstance(self.answer_generator, StreamingAnswerGenerator):
            raise ValueError("Streaming is not enabled or the answer generator doesn't support streaming")
        
        # Use router if enabled
        if self.use_query_routing and self.router:
            try:
                routing_result = await self.router.route_query(query)
                selected_retriever = routing_result["retriever"]
            except Exception as e:
                logger.error(f"Error in query routing: {str(e)}")
                selected_retriever = retriever or self.retriever
        else:
            selected_retriever = retriever or self.retriever
        
        # Retrieve documents
        try:
            docs = await selected_retriever.aget_relevant_documents(
                query, 
                k=top_k or self.default_top_k
            )
        except Exception as e:
            logger.error(f"Error in document retrieval: {str(e)}")
            yield f"Error retrieving documents: {str(e)}"
            return
        
        # Apply reranking if enabled and not already handled by the retriever
        if (self.rerank_results and self.reranker and 
            not (self.use_advanced_retrieval and isinstance(selected_retriever, AdvancedRetriever))):
            try:
                reranked_docs = await self.reranker.rerank(query, docs)
                docs = reranked_docs
            except Exception as e:
                logger.error(f"Error in reranking: {str(e)}")
        
        # Apply context processing if enabled
        if self.use_context_processing and self.context_processor:
            try:
                processed_docs = self.context_processor.process_documents(docs, query=query)
                docs = processed_docs
            except Exception as e:
                logger.error(f"Error in context processing: {str(e)}")
        
        # Apply context augmentation if enabled
        if self.use_context_augmentation and self.context_augmenter and docs:
            try:
                augmented_docs = await self.context_augmenter.augment_documents(docs)
                docs = augmented_docs
            except Exception as e:
                logger.error(f"Error in context augmentation: {str(e)}")
        
        # Generate streaming response
        try:
            async for chunk in self.answer_generator.generate_answer_stream(query, docs):
                yield chunk
        except Exception as e:
            logger.error(f"Error in answer generation: {str(e)}")
            yield f"Error generating answer: {str(e)}" 