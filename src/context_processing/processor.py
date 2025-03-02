"""
Advanced context processing for handling retrieved documents
"""
from typing import Dict, List, Optional, Union, Any, Callable, Set
import re
import numpy as np
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field


class ContextProcessor:
    """
    Advanced context processor for filtering, ordering, and deduplicating retrieved document chunks
    """
    
    def __init__(
        self,
        max_context_length: int = 4000,
        deduplication_threshold: float = 0.85,
        remove_redundant: bool = True,
        order_by_relevance: bool = True,
        filter_irrelevant: bool = True
    ):
        """
        Initialize context processor
        
        Args:
            max_context_length: Maximum context length to return
            deduplication_threshold: Similarity threshold for deduplication (0.0 to 1.0)
            remove_redundant: Whether to remove redundant information
            order_by_relevance: Whether to order by relevance
            filter_irrelevant: Whether to filter irrelevant chunks
        """
        self.max_context_length = max_context_length
        self.deduplication_threshold = deduplication_threshold
        self.remove_redundant = remove_redundant
        self.order_by_relevance = order_by_relevance
        self.filter_irrelevant = filter_irrelevant
    
    def _calculate_token_count(self, text: str) -> int:
        """
        Calculate the number of tokens in a text (simple estimation)
        
        Args:
            text: Text to calculate token count for
            
        Returns:
            Estimated token count
        """
        # Simple token count estimation (approximately 4 chars per token)
        return len(text) // 4
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate text similarity using character n-grams (simple implementation)
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Convert to lowercase and strip whitespace
        text1 = re.sub(r'\s+', ' ', text1.lower().strip())
        text2 = re.sub(r'\s+', ' ', text2.lower().strip())
        
        # Use character 3-grams
        def get_ngrams(text, n=3):
            return [text[i:i+n] for i in range(len(text) - n + 1)]
        
        ngrams1 = set(get_ngrams(text1))
        ngrams2 = set(get_ngrams(text2))
        
        # Calculate Jaccard similarity
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        
        return intersection / union if union > 0 else 0.0
    
    def _deduplicate_documents(self, documents: List[Document]) -> List[Document]:
        """
        Remove duplicate or highly similar documents
        
        Args:
            documents: List of documents to deduplicate
            
        Returns:
            Deduplicated list of documents
        """
        if not documents:
            return []
        
        # Sort documents by relevance score if available
        if self.order_by_relevance:
            documents = sorted(
                documents, 
                key=lambda x: x.metadata.get("relevance_score", 0.0) if x.metadata else 0.0,
                reverse=True
            )
        
        # Deduplicate
        deduplicated_docs = []
        for doc in documents:
            # Skip if document is too similar to any existing document
            is_duplicate = False
            for existing_doc in deduplicated_docs:
                similarity = self._calculate_text_similarity(doc.page_content, existing_doc.page_content)
                if similarity >= self.deduplication_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated_docs.append(doc)
        
        return deduplicated_docs
    
    def _filter_irrelevant_documents(
        self, 
        documents: List[Document], 
        query: str,
        min_relevance_score: float = 0.25
    ) -> List[Document]:
        """
        Filter out irrelevant documents
        
        Args:
            documents: List of documents to filter
            query: Query to check relevance against
            min_relevance_score: Minimum relevance score to keep document
            
        Returns:
            Filtered list of documents
        """
        if not documents:
            return []
        
        # Filter by relevance score if available
        filtered_docs = []
        for doc in documents:
            relevance_score = doc.metadata.get("relevance_score", 0.0) if doc.metadata else 0.0
            
            # If no relevance score, perform simple keyword matching
            if relevance_score == 0.0:
                # Simple relevance check: count query terms in document
                query_terms = set(re.sub(r'[^\w\s]', '', query.lower()).split())
                doc_text = re.sub(r'[^\w\s]', '', doc.page_content.lower())
                
                term_matches = sum(1 for term in query_terms if term in doc_text)
                if query_terms:
                    relevance_score = term_matches / len(query_terms)
            
            if relevance_score >= min_relevance_score:
                # Set the relevance score in metadata
                if not doc.metadata:
                    doc.metadata = {}
                doc.metadata["relevance_score"] = relevance_score
                filtered_docs.append(doc)
        
        return filtered_docs
    
    def _order_documents_by_metadata(
        self, 
        documents: List[Document], 
        key: str = "relevance_score",
        reverse: bool = True
    ) -> List[Document]:
        """
        Order documents by metadata key
        
        Args:
            documents: List of documents to order
            key: Metadata key to order by
            reverse: Whether to reverse the order (default: True = descending)
            
        Returns:
            Ordered list of documents
        """
        if not documents:
            return []
        
        return sorted(
            documents, 
            key=lambda x: x.metadata.get(key, 0.0) if x.metadata else 0.0,
            reverse=reverse
        )
    
    def _order_documents_hierarchically(self, documents: List[Document]) -> List[Document]:
        """
        Order documents by hierarchical structure (for RAPTOR hierarchical indexing)
        
        Args:
            documents: List of documents
            
        Returns:
            Ordered list of documents
        """
        if not documents:
            return []
        
        # Group documents by parent chunk ID
        parent_docs = []
        child_docs = []
        orphan_docs = []
        
        for doc in documents:
            metadata = doc.metadata or {}
            if metadata.get("is_parent", False):
                parent_docs.append(doc)
            elif "parent_id" in metadata:
                child_docs.append(doc)
            else:
                orphan_docs.append(doc)
        
        # Sort parents by relevance
        parent_docs = self._order_documents_by_metadata(parent_docs, "relevance_score", True)
        
        # Group children by parent ID
        parent_to_children = {}
        for child in child_docs:
            parent_id = child.metadata.get("parent_id")
            if parent_id not in parent_to_children:
                parent_to_children[parent_id] = []
            parent_to_children[parent_id].append(child)
        
        # Order children by relevance within their parent groups
        for parent_id, children in parent_to_children.items():
            parent_to_children[parent_id] = self._order_documents_by_metadata(children, "relevance_score", True)
        
        # Build the ordered list: each parent followed by its children
        ordered_docs = []
        for parent in parent_docs:
            ordered_docs.append(parent)
            parent_id = parent.metadata.get("chunk_id")
            if parent_id in parent_to_children:
                ordered_docs.extend(parent_to_children[parent_id])
        
        # Add orphan docs at the end
        ordered_docs.extend(self._order_documents_by_metadata(orphan_docs, "relevance_score", True))
        
        return ordered_docs
    
    def process_documents(self, documents: List[Document], query: str) -> List[Document]:
        """
        Process retrieved documents for optimal context
        
        Args:
            documents: List of documents to process
            query: Query for relevance filtering
            
        Returns:
            Processed list of documents
        """
        if not documents:
            return []
        
        # Filter irrelevant documents
        if self.filter_irrelevant:
            documents = self._filter_irrelevant_documents(documents, query)
        
        # Deduplicate documents
        if self.remove_redundant:
            documents = self._deduplicate_documents(documents)
        
        # Order documents
        if self.order_by_relevance:
            # Check if we have hierarchical documents
            has_hierarchy = any(
                doc.metadata and (doc.metadata.get("is_parent", False) or "parent_id" in doc.metadata)
                for doc in documents
            )
            
            if has_hierarchy:
                documents = self._order_documents_hierarchically(documents)
            else:
                documents = self._order_documents_by_metadata(documents, "relevance_score", True)
        
        # Truncate to max context length
        if self.max_context_length > 0:
            result_docs = []
            current_length = 0
            
            for doc in documents:
                doc_length = self._calculate_token_count(doc.page_content)
                if current_length + doc_length <= self.max_context_length:
                    result_docs.append(doc)
                    current_length += doc_length
                else:
                    # If we can't add the full document, we stop
                    break
            
            return result_docs
        
        return documents


class DocumentChainProcessor:
    """
    Process document chain/tree relationships (for multi-hop reasoning)
    """
    
    def __init__(
        self,
        llm: Optional[BaseLanguageModel] = None,
        max_chain_length: int = 3,
        min_similarity_threshold: float = 0.35
    ):
        """
        Initialize document chain processor
        
        Args:
            llm: Optional language model for analyzing connections
            max_chain_length: Maximum chain length to follow
            min_similarity_threshold: Minimum similarity threshold for connections
        """
        self.llm = llm
        self.max_chain_length = max_chain_length
        self.min_similarity_threshold = min_similarity_threshold
        
        # Initialize prompt for connection analysis
        self._init_prompts()
    
    def _init_prompts(self):
        """Initialize prompts for document connection analysis"""
        self.connection_prompt_template = PromptTemplate(
            input_variables=["document1", "document2"],
            template="""Analyze these two document excerpts and determine if there's a meaningful connection between them:

Document 1:
{document1}

Document 2:
{document2}

First, identify key entities, concepts, or topics in both documents.
Then, determine if Document 2:
1) Elaborates on information in Document 1
2) Contradicts information in Document 1 
3) Provides prerequisite information for Document 1
4) Is redundant with Document 1
5) Has no meaningful connection to Document 1

Return your analysis in this format:
Connection Type: [Type 1-5]
Confidence: [Number between 0-1]
Explanation: [Brief explanation of connection]
"""
        )
    
    def _calculate_similarity(self, doc1: Document, doc2: Document) -> float:
        """
        Calculate similarity between two documents
        
        Args:
            doc1: First document
            doc2: Second document
            
        Returns:
            Similarity score (0.0 to 1.0)
        """
        # Simple text similarity using character n-grams
        def get_ngrams(text, n=3):
            text = re.sub(r'\s+', ' ', text.lower().strip())
            return [text[i:i+n] for i in range(len(text) - n + 1)]
        
        ngrams1 = set(get_ngrams(doc1.page_content))
        ngrams2 = set(get_ngrams(doc2.page_content))
        
        if not ngrams1 or not ngrams2:
            return 0.0
        
        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))
        
        return intersection / union if union > 0 else 0.0
    
    def _analyze_connection_with_llm(self, doc1: Document, doc2: Document) -> Dict[str, Any]:
        """
        Analyze connection between two documents using LLM
        
        Args:
            doc1: First document
            doc2: Second document
            
        Returns:
            Connection analysis
        """
        if not self.llm:
            return {"connection_type": 5, "confidence": 0.0}
        
        try:
            prompt = self.connection_prompt_template.format(
                document1=doc1.page_content[:1000],  # Limit length
                document2=doc2.page_content[:1000]   # Limit length
            )
            
            result = self.llm.predict(prompt)
            
            # Parse the result (simple parsing for demonstration)
            connection_type = 5  # Default: no connection
            confidence = 0.0
            
            for line in result.split("\n"):
                if line.startswith("Connection Type:"):
                    try:
                        connection_type = int(line.split(":")[1].strip())
                    except:
                        pass
                elif line.startswith("Confidence:"):
                    try:
                        confidence = float(line.split(":")[1].strip())
                    except:
                        pass
            
            return {
                "connection_type": connection_type,
                "confidence": confidence,
                "raw_result": result
            }
        
        except Exception as e:
            print(f"Error analyzing connection with LLM: {str(e)}")
            return {"connection_type": 5, "confidence": 0.0}
    
    def build_knowledge_graph(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Build a knowledge graph from documents
        
        Args:
            documents: List of documents
            
        Returns:
            Knowledge graph representation
        """
        if not documents:
            return {"nodes": [], "edges": []}
        
        # Create nodes for each document
        nodes = []
        for i, doc in enumerate(documents):
            node_id = doc.metadata.get("chunk_id", f"doc_{i}")
            nodes.append({
                "id": node_id,
                "content": doc.page_content[:100] + "...",  # Short preview
                "metadata": doc.metadata
            })
        
        # Create edges between documents
        edges = []
        visited_pairs = set()
        
        for i, doc1 in enumerate(documents):
            doc1_id = doc1.metadata.get("chunk_id", f"doc_{i}")
            
            # Check for explicit parent-child relationships
            if doc1.metadata and "parent_id" in doc1.metadata:
                parent_id = doc1.metadata["parent_id"]
                edges.append({
                    "source": parent_id,
                    "target": doc1_id,
                    "type": "parent-child",
                    "weight": 1.0
                })
            
            # Check for other connections
            for j, doc2 in enumerate(documents):
                if i == j:
                    continue
                
                # Skip if we've already checked this pair
                pair_key = f"{i}:{j}"
                if pair_key in visited_pairs:
                    continue
                
                visited_pairs.add(pair_key)
                visited_pairs.add(f"{j}:{i}")  # Mark reverse direction too
                
                doc2_id = doc2.metadata.get("chunk_id", f"doc_{j}")
                
                # Calculate similarity
                similarity = self._calculate_similarity(doc1, doc2)
                
                if similarity >= self.min_similarity_threshold:
                    # If we have LLM, get deeper analysis
                    connection_info = {"connection_type": 1, "confidence": similarity}
                    if self.llm:
                        connection_info = self._analyze_connection_with_llm(doc1, doc2)
                    
                    # Add edge if connection is meaningful
                    if connection_info["connection_type"] < 5 and connection_info["confidence"] >= self.min_similarity_threshold:
                        edges.append({
                            "source": doc1_id,
                            "target": doc2_id,
                            "type": f"connection_{connection_info['connection_type']}",
                            "weight": connection_info["confidence"]
                        })
        
        return {"nodes": nodes, "edges": edges}
    
    def find_document_chains(self, documents: List[Document], start_idx: int = 0) -> List[List[Document]]:
        """
        Find chains of connected documents for multi-hop reasoning
        
        Args:
            documents: List of documents
            start_idx: Index of the document to start from
            
        Returns:
            List of document chains
        """
        if not documents or start_idx >= len(documents):
            return []
        
        # Build a graph of document connections
        doc_graph = {}
        for i, doc1 in enumerate(documents):
            doc_graph[i] = []
            
            # Add explicit parent-child connections
            if doc1.metadata:
                if doc1.metadata.get("is_parent", False):
                    parent_id = doc1.metadata.get("chunk_id")
                    for j, doc2 in enumerate(documents):
                        if doc2.metadata and doc2.metadata.get("parent_id") == parent_id:
                            doc_graph[i].append((j, 1.0))
                
                if "parent_id" in doc1.metadata:
                    parent_id = doc1.metadata["parent_id"]
                    for j, doc2 in enumerate(documents):
                        if doc2.metadata and doc2.metadata.get("chunk_id") == parent_id:
                            doc_graph[i].append((j, 1.0))
            
            # Add other connections based on similarity
            for j, doc2 in enumerate(documents):
                if i == j:
                    continue
                
                # Skip if already connected through parent-child
                if any(edge[0] == j for edge in doc_graph[i]):
                    continue
                
                similarity = self._calculate_similarity(doc1, doc2)
                if similarity >= self.min_similarity_threshold:
                    doc_graph[i].append((j, similarity))
        
        # Find chains using DFS
        def dfs(node, path, visited, chains):
            if len(path) >= self.max_chain_length:
                chains.append(path[:])
                return
            
            visited.add(node)
            
            for neighbor, weight in sorted(doc_graph[node], key=lambda x: x[1], reverse=True):
                if neighbor not in visited:
                    path.append(neighbor)
                    dfs(neighbor, path, visited, chains)
                    path.pop()
            
            visited.remove(node)
        
        chains = []
        dfs(start_idx, [start_idx], set(), chains)
        
        # Convert indices to documents
        doc_chains = []
        for chain in chains:
            doc_chains.append([documents[idx] for idx in chain])
        
        return doc_chains 