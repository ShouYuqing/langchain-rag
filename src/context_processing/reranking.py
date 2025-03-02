"""
Reranking module for ranking retrieved documents based on relevance to query
"""
from typing import Dict, List, Optional, Union, Any, Callable, Type
import re
import numpy as np
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class RerankedResult(BaseModel):
    """Model for reranking results from LLM"""
    relevance_score: float = Field(..., description="Relevance score between 0 and 1")
    reasoning: str = Field(..., description="Explanation of why this score was assigned")


class BaseReranker:
    """
    Base reranker class for ranking retrieved documents based on relevance to query
    """
    
    def __init__(self):
        """Initialize base reranker"""
        pass
    
    def rerank(self, documents: List[Document], query: str) -> List[Document]:
        """
        Rerank documents based on relevance to query
        
        Args:
            documents: List of documents to rerank
            query: Query to rank against
            
        Returns:
            Reranked list of documents
        """
        raise NotImplementedError("Subclasses must implement rerank method")


class KeywordReranker(BaseReranker):
    """
    Keyword-based reranker using basic term matching
    """
    
    def __init__(
        self,
        weight_title: float = 2.0,
        weight_exact_match: float = 3.0,
        weight_partial_match: float = 1.0
    ):
        """
        Initialize keyword reranker
        
        Args:
            weight_title: Weight for matches in title/metadata
            weight_exact_match: Weight for exact matches
            weight_partial_match: Weight for partial matches
        """
        super().__init__()
        self.weight_title = weight_title
        self.weight_exact_match = weight_exact_match
        self.weight_partial_match = weight_partial_match
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for matching
        
        Args:
            text: Text to preprocess
            
        Returns:
            Preprocessed text
        """
        # Convert to lowercase, remove special chars, and normalize whitespace
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _calculate_score(self, document: Document, query_terms: List[str]) -> float:
        """
        Calculate relevance score for a document
        
        Args:
            document: Document to score
            query_terms: Preprocessed query terms
            
        Returns:
            Relevance score
        """
        # Preprocess document content
        content = self._preprocess_text(document.page_content)
        content_words = set(content.split())
        
        # Get metadata for title matching
        metadata = document.metadata or {}
        title = self._preprocess_text(metadata.get("title", ""))
        source = self._preprocess_text(metadata.get("source", ""))
        title_words = set(title.split()) | set(source.split())
        
        # Calculate score components
        score = 0.0
        
        # Exact phrase match
        query_phrase = " ".join(query_terms)
        if query_phrase in content:
            score += self.weight_exact_match
        
        # Term frequency in content
        for term in query_terms:
            # Skip very short terms
            if len(term) <= 2:
                continue
                
            # Check for exact term match in content
            if term in content_words:
                score += 1.0
            
            # Check for partial match in content
            elif any(term in word for word in content_words):
                score += 0.5 * self.weight_partial_match
            
            # Check for exact term match in title/metadata
            if term in title_words:
                score += self.weight_title
        
        # Normalize by query length
        if query_terms:
            score = score / len(query_terms)
        
        return min(1.0, score)  # Cap at 1.0
    
    def rerank(self, documents: List[Document], query: str) -> List[Document]:
        """
        Rerank documents based on keyword relevance to query
        
        Args:
            documents: List of documents to rerank
            query: Query to rank against
            
        Returns:
            Reranked list of documents
        """
        if not documents:
            return []
        
        # Preprocess query
        processed_query = self._preprocess_text(query)
        query_terms = processed_query.split()
        
        # Score each document
        scored_docs = []
        for doc in documents:
            score = self._calculate_score(doc, query_terms)
            
            # Create a new document with score in metadata
            metadata = doc.metadata.copy() if doc.metadata else {}
            metadata["relevance_score"] = score
            
            scored_doc = Document(
                page_content=doc.page_content,
                metadata=metadata
            )
            scored_docs.append((score, scored_doc))
        
        # Sort by score in descending order
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Return documents only
        return [doc for _, doc in scored_docs]


class EmbeddingReranker(BaseReranker):
    """
    Embedding-based reranker using cosine similarity between query and documents
    """
    
    def __init__(
        self,
        embedding_function: Callable[[str], List[float]],
    ):
        """
        Initialize embedding reranker
        
        Args:
            embedding_function: Function to convert text to embeddings
        """
        super().__init__()
        self.embedding_function = embedding_function
    
    def _calculate_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity (0 to 1)
        """
        # Convert to numpy arrays
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        
        # Ensure the result is between 0 and 1
        return max(0.0, min(1.0, float(similarity)))
    
    def rerank(self, documents: List[Document], query: str) -> List[Document]:
        """
        Rerank documents based on embedding similarity to query
        
        Args:
            documents: List of documents to rerank
            query: Query to rank against
            
        Returns:
            Reranked list of documents
        """
        if not documents:
            return []
        
        try:
            # Get query embedding
            query_embedding = self.embedding_function(query)
            
            # Score each document
            scored_docs = []
            for doc in documents:
                # Check if document already has an embedding
                if doc.metadata and "embedding" in doc.metadata:
                    doc_embedding = doc.metadata["embedding"]
                else:
                    # Generate embedding if not present
                    doc_embedding = self.embedding_function(doc.page_content)
                
                # Calculate similarity score
                score = self._calculate_similarity(query_embedding, doc_embedding)
                
                # Create a new document with score in metadata
                metadata = doc.metadata.copy() if doc.metadata else {}
                metadata["relevance_score"] = score
                
                scored_doc = Document(
                    page_content=doc.page_content,
                    metadata=metadata
                )
                scored_docs.append((score, scored_doc))
            
            # Sort by score in descending order
            scored_docs.sort(key=lambda x: x[0], reverse=True)
            
            # Return documents only
            return [doc for _, doc in scored_docs]
            
        except Exception as e:
            print(f"Error in embedding reranking: {str(e)}")
            # Fall back to original order
            return documents


class LLMReranker(BaseReranker):
    """
    LLM-based reranker that uses a language model to score document relevance
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        query_reformulation: bool = True,
        detailed_feedback: bool = True,
        batch_size: int = 5
    ):
        """
        Initialize LLM reranker
        
        Args:
            llm: Language model for reranking
            query_reformulation: Whether to reformulate the query for better matching
            detailed_feedback: Whether to include detailed feedback for each document
            batch_size: Number of documents to rerank in a single LLM call
        """
        super().__init__()
        self.llm = llm
        self.query_reformulation = query_reformulation
        self.detailed_feedback = detailed_feedback
        self.batch_size = batch_size
        
        # Initialize prompt templates
        self._init_prompts()
        
        # Set up output parser
        self.parser = PydanticOutputParser(pydantic_object=RerankedResult)
    
    def _init_prompts(self):
        """Initialize prompt templates for LLM reranking"""
        # Single document reranking prompt
        self.single_doc_prompt_template = PromptTemplate(
            input_variables=["query", "document"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
            template="""You are an expert at determining the relevance of documents to a given query. 
Evaluate how relevant the following document is to the query.

Query: {query}

Document:
{document}

{format_instructions}

Think step by step about why this document is or isn't relevant to the query before providing a final score.
"""
        )
        
        # Query reformulation prompt
        self.query_reformulation_prompt_template = PromptTemplate(
            input_variables=["query"],
            template="""Rewrite the following search query to improve document retrieval results.
The goal is to expand the query with synonyms and related terms while maintaining the original meaning.

Original query: {query}

Rewritten query:"""
        )
    
    def _rerank_single(self, document: Document, query: str) -> Document:
        """
        Rerank a single document using LLM
        
        Args:
            document: Document to rerank
            query: Query to rank against
            
        Returns:
            Reranked document with score in metadata
        """
        try:
            # Format the prompt
            prompt = self.single_doc_prompt_template.format(
                query=query,
                document=document.page_content[:1000]  # Limit length to avoid token limits
            )
            
            # Get LLM response
            result = self.llm.predict(prompt)
            
            # Parse the result
            try:
                parsed_result = self.parser.parse(result)
                score = parsed_result.relevance_score
                reasoning = parsed_result.reasoning
            except Exception:
                # Fall back to simple parsing if structured parsing fails
                score_match = re.search(r"relevance_score[\"']?\s*:\s*([0-9.]+)", result)
                score = float(score_match.group(1)) if score_match else 0.5
                
                reasoning_match = re.search(r"reasoning[\"']?\s*:\s*[\"']([^\"']+)[\"']", result)
                reasoning = reasoning_match.group(1) if reasoning_match else "No reasoning provided"
            
            # Create a new document with score in metadata
            metadata = document.metadata.copy() if document.metadata else {}
            metadata["relevance_score"] = score
            if self.detailed_feedback:
                metadata["relevance_reasoning"] = reasoning
            
            return Document(
                page_content=document.page_content,
                metadata=metadata
            )
            
        except Exception as e:
            print(f"Error in LLM reranking for single document: {str(e)}")
            # Return original document with default score
            metadata = document.metadata.copy() if document.metadata else {}
            metadata["relevance_score"] = 0.5
            return Document(
                page_content=document.page_content,
                metadata=metadata
            )
    
    def _reformulate_query(self, query: str) -> str:
        """
        Reformulate query to improve retrieval
        
        Args:
            query: Original query
            
        Returns:
            Reformulated query
        """
        try:
            # Format the prompt
            prompt = self.query_reformulation_prompt_template.format(query=query)
            
            # Get LLM response
            result = self.llm.predict(prompt)
            
            # Return the result or fall back to original query
            reformulated = result.strip()
            return reformulated if reformulated else query
            
        except Exception as e:
            print(f"Error in query reformulation: {str(e)}")
            return query
    
    def rerank(self, documents: List[Document], query: str) -> List[Document]:
        """
        Rerank documents using LLM
        
        Args:
            documents: List of documents to rerank
            query: Query to rank against
            
        Returns:
            Reranked list of documents
        """
        if not documents:
            return []
        
        # Reformulate query if enabled
        if self.query_reformulation:
            reformulated_query = self._reformulate_query(query)
        else:
            reformulated_query = query
        
        # Process documents individually (could be optimized to batch process)
        reranked_docs = [self._rerank_single(doc, reformulated_query) for doc in documents]
        
        # Sort by score in descending order
        reranked_docs.sort(
            key=lambda x: x.metadata.get("relevance_score", 0.0) if x.metadata else 0.0,
            reverse=True
        )
        
        return reranked_docs


class HybridReranker(BaseReranker):
    """
    Hybrid reranker that combines multiple reranking strategies
    """
    
    def __init__(
        self,
        rerankers: List[BaseReranker],
        weights: Optional[List[float]] = None
    ):
        """
        Initialize hybrid reranker
        
        Args:
            rerankers: List of reranker instances to combine
            weights: Optional weights for each reranker (defaults to equal weights)
        """
        super().__init__()
        self.rerankers = rerankers
        
        # Set weights (default to equal if not provided)
        if weights and len(weights) == len(rerankers):
            # Normalize weights to sum to 1
            total = sum(weights)
            self.weights = [w / total for w in weights]
        else:
            # Equal weights
            self.weights = [1.0 / len(rerankers)] * len(rerankers)
    
    def rerank(self, documents: List[Document], query: str) -> List[Document]:
        """
        Rerank documents using multiple strategies
        
        Args:
            documents: List of documents to rerank
            query: Query to rank against
            
        Returns:
            Reranked list of documents
        """
        if not documents:
            return []
        
        # Create a copy of documents to avoid modifying originals
        docs_copy = [
            Document(
                page_content=doc.page_content,
                metadata=doc.metadata.copy() if doc.metadata else {}
            )
            for doc in documents
        ]
        
        # Track document scores
        doc_scores = [0.0] * len(docs_copy)
        
        # Apply each reranker
        for i, reranker in enumerate(self.rerankers):
            # Get results from this reranker
            reranked = reranker.rerank(docs_copy, query)
            
            # Extract scores and map back to original document positions
            score_map = {}
            for j, doc in enumerate(reranked):
                # Get identifier for the document
                content_hash = hash(doc.page_content)
                
                # Calculate position-based score (higher rank = higher score)
                # Normalize to 0-1 range
                position_score = 1.0 - (j / len(reranked)) if len(reranked) > 1 else 1.0
                
                # Use explicit relevance score if available
                if doc.metadata and "relevance_score" in doc.metadata:
                    score_map[content_hash] = doc.metadata["relevance_score"]
                else:
                    score_map[content_hash] = position_score
            
            # Add weighted scores to overall scores
            for j, doc in enumerate(docs_copy):
                content_hash = hash(doc.page_content)
                if content_hash in score_map:
                    doc_scores[j] += score_map[content_hash] * self.weights[i]
        
        # Create final documents with combined scores
        final_docs = []
        for i, doc in enumerate(docs_copy):
            metadata = doc.metadata.copy() if doc.metadata else {}
            metadata["relevance_score"] = doc_scores[i]
            
            final_docs.append((
                doc_scores[i],
                Document(
                    page_content=doc.page_content,
                    metadata=metadata
                )
            ))
        
        # Sort by combined score in descending order
        final_docs.sort(key=lambda x: x[0], reverse=True)
        
        # Return documents only
        return [doc for _, doc in final_docs]


class CustomReranker:
    """
    Customizable reranker that combines different reranking strategies
    """
    
    def __init__(
        self,
        llm: Optional[BaseLanguageModel] = None,
        embedding_function: Optional[Callable[[str], List[float]]] = None,
        use_llm_reranker: bool = True,
        use_embedding_reranker: bool = True,
        use_keyword_reranker: bool = True,
        llm_weight: float = 0.7,
        embedding_weight: float = 0.2,
        keyword_weight: float = 0.1,
        query_reformulation: bool = True
    ):
        """
        Initialize custom reranker
        
        Args:
            llm: Language model for LLM reranking
            embedding_function: Function to get embeddings
            use_llm_reranker: Whether to use LLM reranking
            use_embedding_reranker: Whether to use embedding reranking
            use_keyword_reranker: Whether to use keyword reranking
            llm_weight: Weight for LLM reranker
            embedding_weight: Weight for embedding reranker
            keyword_weight: Weight for keyword reranker
            query_reformulation: Whether to reformulate queries
        """
        self.llm = llm
        self.embedding_function = embedding_function
        self.use_llm_reranker = use_llm_reranker and llm is not None
        self.use_embedding_reranker = use_embedding_reranker and embedding_function is not None
        self.use_keyword_reranker = use_keyword_reranker
        self.llm_weight = llm_weight
        self.embedding_weight = embedding_weight
        self.keyword_weight = keyword_weight
        self.query_reformulation = query_reformulation
        
        # Initialize rerankers
        self._init_rerankers()
    
    def _init_rerankers(self):
        """Initialize individual rerankers"""
        rerankers = []
        weights = []
        
        # Add keyword reranker
        if self.use_keyword_reranker:
            rerankers.append(KeywordReranker())
            weights.append(self.keyword_weight)
        
        # Add embedding reranker
        if self.use_embedding_reranker and self.embedding_function:
            rerankers.append(EmbeddingReranker(self.embedding_function))
            weights.append(self.embedding_weight)
        
        # Add LLM reranker
        if self.use_llm_reranker and self.llm:
            rerankers.append(LLMReranker(
                self.llm, 
                query_reformulation=self.query_reformulation,
                detailed_feedback=True
            ))
            weights.append(self.llm_weight)
        
        # Fall back to keyword reranker if no other rerankers
        if not rerankers:
            rerankers.append(KeywordReranker())
            weights.append(1.0)
        
        # Create hybrid reranker
        self.reranker = HybridReranker(rerankers, weights)
    
    def rerank(self, documents: List[Document], query: str) -> List[Document]:
        """
        Rerank documents using the configured strategies
        
        Args:
            documents: List of documents to rerank
            query: Query to rank against
            
        Returns:
            Reranked list of documents
        """
        return self.reranker.rerank(documents, query) 