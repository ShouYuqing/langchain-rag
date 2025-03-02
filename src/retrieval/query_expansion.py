"""
Query expansion and rewriting to improve retrieval quality
"""
from typing import Dict, List, Optional, Union, Any, Callable
import logging
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class ExpandedQueries(BaseModel):
    """Pydantic model for expanded queries"""
    original_query: str = Field(description="The original user query")
    expanded_queries: List[str] = Field(description="List of expanded or rewritten queries")
    reasoning: str = Field(description="Reasoning for the expanded queries")

class QueryExpander:
    """
    Expand and rewrite queries to improve retrieval quality
    """
    
    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        self._init_prompts()
    
    def _init_prompts(self):
        """Initialize prompt templates"""
        # Query expansion prompt
        self.expansion_prompt_template = """You are an AI assistant helping to improve retrieval by generating alternative phrasings of a search query.
        
Original query: {query}

Your task is to generate {num_queries} alternative phrasings or expanded versions of this query. 
These should:
1. Preserve the original meaning and intent
2. Use different words and phrasing to capture the same concept
3. Add potential context or missing information if the query is ambiguous
4. Break down complex queries into more specific aspects
5. Consider synonyms and related terminology

Remember, the goal is to improve document retrieval by increasing the chances of matching relevant documents with varied terminology.

{format_instructions}
"""
        # Parser for the output
        self.parser = PydanticOutputParser(pydantic_object=ExpandedQueries)
        self.expansion_prompt = PromptTemplate(
            template=self.expansion_prompt_template,
            input_variables=["query", "num_queries"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
    
    async def expand_query(self, query: str, num_expanded_queries: int = 3) -> ExpandedQueries:
        """
        Expand a query into multiple alternative phrasings
        
        Args:
            query: Original user query
            num_expanded_queries: Number of expanded queries to generate
            
        Returns:
            ExpandedQueries object with original query and expanded queries
        """
        try:
            # Format the prompt
            prompt = self.expansion_prompt.format(
                query=query,
                num_queries=num_expanded_queries
            )
            
            # Generate expanded queries
            result = await self.llm.ainvoke(prompt)
            
            # Parse the result
            expanded_queries = self.parser.parse(result)
            
            return expanded_queries
        
        except Exception as e:
            logging.error(f"Error expanding query: {str(e)}")
            # Fallback to returning just the original query
            return ExpandedQueries(
                original_query=query,
                expanded_queries=[query],
                reasoning="Failed to expand query due to an error."
            )
    
    def expand_query_sync(self, query: str, num_expanded_queries: int = 3) -> ExpandedQueries:
        """
        Synchronous version of expand_query
        
        Args:
            query: Original user query
            num_expanded_queries: Number of expanded queries to generate
            
        Returns:
            ExpandedQueries object with original query and expanded queries
        """
        try:
            # Format the prompt
            prompt = self.expansion_prompt.format(
                query=query,
                num_queries=num_expanded_queries
            )
            
            # Generate expanded queries
            result = self.llm.invoke(prompt)
            
            # Parse the result
            expanded_queries = self.parser.parse(result)
            
            return expanded_queries
        
        except Exception as e:
            logging.error(f"Error expanding query: {str(e)}")
            # Fallback to returning just the original query
            return ExpandedQueries(
                original_query=query,
                expanded_queries=[query],
                reasoning="Failed to expand query due to an error."
            )

class HypotheticalDocumentEmbedder:
    """
    Improve retrieval by generating a hypothetical document that would answer the query
    """
    
    def __init__(self, llm: BaseLanguageModel):
        self.llm = llm
        self._init_prompts()
    
    def _init_prompts(self):
        """Initialize prompt templates"""
        # Hypothetical document prompt
        self.document_prompt_template = """You are an expert knowledge worker helping to improve search retrieval.
        
For the following query, generate a detailed hypothetical document that would perfectly answer this query.
Focus on being informative and comprehensive, including relevant terminology, concepts, and information 
that would likely appear in a document that answers this query well.

Query: {query}

Hypothetical document (aim for about 300-500 words):
"""
        
        self.document_prompt = PromptTemplate(
            template=self.document_prompt_template,
            input_variables=["query"]
        )
    
    async def generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical document that would answer the query
        
        Args:
            query: User query
            
        Returns:
            Hypothetical document text
        """
        try:
            # Format the prompt
            prompt = self.document_prompt.format(query=query)
            
            # Generate hypothetical document
            result = await self.llm.ainvoke(prompt)
            
            return result
        
        except Exception as e:
            logging.error(f"Error generating hypothetical document: {str(e)}")
            # Fallback to a simple document
            return f"This document contains information about {query}."
    
    def generate_hypothetical_document_sync(self, query: str) -> str:
        """
        Synchronous version of generate_hypothetical_document
        
        Args:
            query: User query
            
        Returns:
            Hypothetical document text
        """
        try:
            # Format the prompt
            prompt = self.document_prompt.format(query=query)
            
            # Generate hypothetical document
            result = self.llm.invoke(prompt)
            
            return result
        
        except Exception as e:
            logging.error(f"Error generating hypothetical document: {str(e)}")
            # Fallback to a simple document
            return f"This document contains information about {query}." 