"""
Router for directing queries to appropriate knowledge sources or processes
"""
import re
import json
import logging
from typing import Dict, List, Optional, Union, Any, Callable, Type, Tuple
from enum import Enum

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.retrievers import BaseRetriever
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class QueryType(str, Enum):
    """Enumeration of query types for routing"""
    FACTUAL = "factual"
    CONCEPTUAL = "conceptual"
    PROCEDURAL = "procedural"
    EXPLORATORY = "exploratory"
    ANALYTICAL = "analytical"
    CLARIFICATION = "clarification"
    CODE = "code"
    UNKNOWN = "unknown"


class QueryDomain(str, Enum):
    """Enumeration of knowledge domains for routing"""
    GENERAL = "general"
    TECHNICAL = "technical"
    SCIENTIFIC = "scientific"
    BUSINESS = "business"
    LEGAL = "legal"
    MEDICAL = "medical"
    EDUCATIONAL = "educational"
    CODE = "code"
    OTHER = "other"


class QueryIntent(str, Enum):
    """Enumeration of user intents for routing"""
    INFORMATION = "information_seeking"
    PROBLEM_SOLVING = "problem_solving"
    COMPARISON = "comparison"
    RECOMMENDATION = "recommendation"
    EXPLANATION = "explanation"
    GUIDANCE = "guidance"
    DEFINITION = "definition"
    VERIFICATION = "verification"
    UNKNOWN = "unknown"


class QueryComplexity(str, Enum):
    """Enumeration of query complexity levels"""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    MULTI_HOP = "multi_hop"
    UNKNOWN = "unknown"


class QueryClassification(BaseModel):
    """
    Classification model for understanding query characteristics and routing appropriately
    """
    query_type: QueryType = Field(..., description="The type of query being asked")
    domain: QueryDomain = Field(..., description="The knowledge domain of the query")
    intent: QueryIntent = Field(..., description="The user's intent in asking the query")
    complexity: QueryComplexity = Field(..., description="The complexity level of the query")
    requires_retrieval: bool = Field(..., description="Whether this query requires document retrieval")
    requires_calculation: bool = Field(..., description="Whether this query requires mathematical calculations")
    requires_current_data: bool = Field(..., description="Whether this query requires up-to-date information")
    reasoning: str = Field(..., description="Reasoning behind the classification decisions")


class QueryRouter:
    """
    Router for directing queries to appropriate knowledge sources or processes
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        retrievers: Dict[str, BaseRetriever] = None,
        default_retriever: Optional[BaseRetriever] = None,
        external_tools: Dict[str, Callable] = None,
        domain_keyword_map: Dict[str, List[str]] = None,
    ):
        """
        Initialize the query router
        
        Args:
            llm: Language model for query classification
            retrievers: Dictionary of specialized retrievers for different domains
            default_retriever: Default retriever to use if no specific one is selected
            external_tools: Dictionary of external tools/APIs for specific needs
            domain_keyword_map: Dictionary mapping keywords to specific domains
        """
        self.llm = llm
        self.retrievers = retrievers or {}
        self.default_retriever = default_retriever
        self.external_tools = external_tools or {}
        self.domain_keyword_map = domain_keyword_map or {}
        
        # Initialize classifier components
        self.parser = PydanticOutputParser(pydantic_object=QueryClassification)
        self._init_prompts()
    
    def _init_prompts(self):
        """Initialize prompt templates for query classification"""
        self.classification_prompt = PromptTemplate(
            input_variables=["query"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()},
            template="""You are an expert system that analyzes and classifies user queries to determine the best way to handle them.
Analyze the following query and classify it according to its characteristics.

Query: {query}

{format_instructions}

Think carefully about each aspect of the classification. Consider what type of query it is, what knowledge domain it belongs to, 
what the user's intent is, and how complex the query is. Also determine if it requires document retrieval, calculations, 
or current/real-time data.

Base your classifications only on the content of the query, not on assumptions about what might be available.
"""
        )
    
    def _rule_based_classification(self, query: str) -> Dict[str, Any]:
        """
        Perform basic rule-based classification to identify obvious patterns
        
        Args:
            query: The user's query
            
        Returns:
            Partial classification data
        """
        query = query.lower()
        classification = {}
        
        # Check for current data indicators
        current_data_patterns = [
            r"current", r"latest", r"recent", r"today", r"now", 
            r"up to date", r"newest", r"this year", r"this month", r"this week"
        ]
        if any(re.search(pattern, query) for pattern in current_data_patterns):
            classification["requires_current_data"] = True
        
        # Check for calculation indicators
        calculation_patterns = [
            r"calculate", r"compute", r"how many", r"how much",
            r"average", r"total", r"sum", r"difference", r"percent", r"%",
            r"divided by", r"\d+[\+\-\*\/]\d+"
        ]
        if any(re.search(pattern, query) for pattern in calculation_patterns):
            classification["requires_calculation"] = True
        
        # Check for code indicators
        code_patterns = [
            r"code", r"function", r"program", r"algorithm", r"implement",
            r"debug", r"error", r"exception", r"compile", r"runtime",
            r"python", r"javascript", r"java", r"c\+\+", r"html", r"css",
            r"sql", r"api", r"git", r"docker", r"kubernetes"
        ]
        if any(re.search(pattern, query) for pattern in code_patterns):
            classification["domain"] = QueryDomain.CODE
            classification["query_type"] = QueryType.CODE
        
        # Identify domain by keywords if specified in domain_keyword_map
        for domain, keywords in self.domain_keyword_map.items():
            if any(keyword in query for keyword in keywords):
                try:
                    classification["domain"] = QueryDomain(domain)
                except ValueError:
                    # If not a valid enum value, don't set
                    pass
        
        return classification
    
    async def classify_query(self, query: str) -> QueryClassification:
        """
        Classify a query using LLM to determine how to handle it
        
        Args:
            query: The user's query
            
        Returns:
            QueryClassification object
        """
        try:
            # First apply rule-based classification for obvious patterns
            partial_classification = self._rule_based_classification(query)
            
            # Use LLM for full classification
            prompt = self.classification_prompt.format(query=query)
            result = await self.llm.apredict(prompt)
            
            try:
                # Parse the result
                classification = self.parser.parse(result)
                
                # Override with rule-based results if they exist
                for key, value in partial_classification.items():
                    setattr(classification, key, value)
                
                return classification
                
            except Exception as e:
                logger.error(f"Error parsing classification result: {str(e)}")
                
                # Create a fallback classification
                fallback = QueryClassification(
                    query_type=partial_classification.get("query_type", QueryType.UNKNOWN),
                    domain=partial_classification.get("domain", QueryDomain.GENERAL),
                    intent=QueryIntent.INFORMATION,
                    complexity=QueryComplexity.MEDIUM,
                    requires_retrieval=True,
                    requires_calculation=partial_classification.get("requires_calculation", False),
                    requires_current_data=partial_classification.get("requires_current_data", False),
                    reasoning="Fallback classification due to parsing error"
                )
                return fallback
                
        except Exception as e:
            logger.error(f"Error in query classification: {str(e)}")
            
            # Return a safe default
            return QueryClassification(
                query_type=QueryType.UNKNOWN,
                domain=QueryDomain.GENERAL,
                intent=QueryIntent.INFORMATION,
                complexity=QueryComplexity.MEDIUM,
                requires_retrieval=True,
                requires_calculation=False,
                requires_current_data=False,
                reasoning="Default classification due to error"
            )
    
    def classify_query_sync(self, query: str) -> QueryClassification:
        """
        Synchronous version of classify_query
        
        Args:
            query: The user's query
            
        Returns:
            QueryClassification object
        """
        try:
            # First apply rule-based classification for obvious patterns
            partial_classification = self._rule_based_classification(query)
            
            # Use LLM for full classification
            prompt = self.classification_prompt.format(query=query)
            result = self.llm.predict(prompt)
            
            try:
                # Parse the result
                classification = self.parser.parse(result)
                
                # Override with rule-based results if they exist
                for key, value in partial_classification.items():
                    setattr(classification, key, value)
                
                return classification
                
            except Exception as e:
                logger.error(f"Error parsing classification result: {str(e)}")
                
                # Create a fallback classification
                fallback = QueryClassification(
                    query_type=partial_classification.get("query_type", QueryType.UNKNOWN),
                    domain=partial_classification.get("domain", QueryDomain.GENERAL),
                    intent=QueryIntent.INFORMATION,
                    complexity=QueryComplexity.MEDIUM,
                    requires_retrieval=True,
                    requires_calculation=partial_classification.get("requires_calculation", False),
                    requires_current_data=partial_classification.get("requires_current_data", False),
                    reasoning="Fallback classification due to parsing error"
                )
                return fallback
                
        except Exception as e:
            logger.error(f"Error in query classification: {str(e)}")
            
            # Return a safe default
            return QueryClassification(
                query_type=QueryType.UNKNOWN,
                domain=QueryDomain.GENERAL,
                intent=QueryIntent.INFORMATION,
                complexity=QueryComplexity.MEDIUM,
                requires_retrieval=True,
                requires_calculation=False,
                requires_current_data=False,
                reasoning="Default classification due to error"
            )
    
    def select_retriever(self, classification: QueryClassification) -> Optional[BaseRetriever]:
        """
        Select the appropriate retriever based on query classification
        
        Args:
            classification: Query classification object
            
        Returns:
            Selected retriever or None if retrieval not required
        """
        # If retrieval is not required, return None
        if not classification.requires_retrieval:
            return None
        
        # Try to find a domain-specific retriever
        domain = classification.domain.value
        if domain in self.retrievers:
            return self.retrievers[domain]
        
        # Fall back to default retriever
        return self.default_retriever
    
    def select_tool(self, classification: QueryClassification) -> Optional[Callable]:
        """
        Select an appropriate external tool/API based on query classification
        
        Args:
            classification: Query classification object
            
        Returns:
            Selected tool function or None if no specific tool needed
        """
        # Check for calculation needs
        if classification.requires_calculation and "calculator" in self.external_tools:
            return self.external_tools["calculator"]
        
        # Check for current data needs
        if classification.requires_current_data and "data_fetcher" in self.external_tools:
            return self.external_tools["data_fetcher"]
        
        # Check for domain-specific tools
        domain = classification.domain.value
        if domain in self.external_tools:
            return self.external_tools[domain]
        
        return None
    
    async def route_query(self, query: str) -> Dict[str, Any]:
        """
        Route a query to the appropriate handling method and resources
        
        Args:
            query: The user's query
            
        Returns:
            Dictionary with routing information
        """
        # Classify the query
        classification = await self.classify_query(query)
        
        # Select appropriate retriever
        retriever = self.select_retriever(classification)
        
        # Select appropriate tool
        tool = self.select_tool(classification)
        
        # Determine handling strategy based on classification
        if classification.complexity == QueryComplexity.MULTI_HOP:
            handling_strategy = "multi_hop_retrieval"
        elif classification.requires_current_data:
            handling_strategy = "external_lookup"
        elif classification.requires_calculation:
            handling_strategy = "calculation"
        elif classification.query_type == QueryType.CODE:
            handling_strategy = "code_generation"
        else:
            handling_strategy = "standard_retrieval"
        
        # Return routing information
        return {
            "classification": classification.dict(),
            "retriever": retriever,
            "tool": tool,
            "handling_strategy": handling_strategy
        }
    
    def route_query_sync(self, query: str) -> Dict[str, Any]:
        """
        Synchronous version of route_query
        
        Args:
            query: The user's query
            
        Returns:
            Dictionary with routing information
        """
        # Classify the query
        classification = self.classify_query_sync(query)
        
        # Select appropriate retriever
        retriever = self.select_retriever(classification)
        
        # Select appropriate tool
        tool = self.select_tool(classification)
        
        # Determine handling strategy based on classification
        if classification.complexity == QueryComplexity.MULTI_HOP:
            handling_strategy = "multi_hop_retrieval"
        elif classification.requires_current_data:
            handling_strategy = "external_lookup"
        elif classification.requires_calculation:
            handling_strategy = "calculation"
        elif classification.query_type == QueryType.CODE:
            handling_strategy = "code_generation"
        else:
            handling_strategy = "standard_retrieval"
        
        # Return routing information
        return {
            "classification": classification.dict(),
            "retriever": retriever,
            "tool": tool,
            "handling_strategy": handling_strategy
        }


class MultiHopPlanner:
    """
    Planner for multi-hop queries that need multiple retrieval steps
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        retriever: BaseRetriever,
        max_hops: int = 3
    ):
        """
        Initialize the multi-hop planner
        
        Args:
            llm: Language model for planning and synthesizing results
            retriever: Retriever for fetching documents
            max_hops: Maximum number of retrieval hops allowed
        """
        self.llm = llm
        self.retriever = retriever
        self.max_hops = max_hops
        
        # Initialize prompts
        self._init_prompts()
    
    def _init_prompts(self):
        """Initialize prompt templates for multi-hop planning"""
        # Prompt for planning retrieval steps
        self.planning_prompt = PromptTemplate(
            input_variables=["query", "existing_info"],
            template="""You are planning a multi-step information retrieval process to answer a complex query.
Based on the query and what we already know, determine what additional information we need to search for.

Original Query: {query}

What we already know:
{existing_info}

Please plan the next retrieval step by specifying:
1. What specific information we need to search for next
2. How this information will help answer the original query
3. A specific search query to use for retrieving this information

Next Information Need:"""
        )
        
        # Prompt for synthesizing final answer
        self.synthesis_prompt = PromptTemplate(
            input_variables=["query", "retrieved_info"],
            template="""You are synthesizing information from multiple retrieval steps to answer a complex query.
Analyze all the retrieved information and provide a comprehensive answer to the original query.

Original Query: {query}

Retrieved Information:
{retrieved_info}

Based on the above information, please provide a complete answer to the original query.
Include citations to the retrieved information where appropriate.

Answer:"""
        )
    
    async def generate_search_plan(self, query: str, existing_info: str = "") -> str:
        """
        Generate a plan for the next retrieval step
        
        Args:
            query: The original user query
            existing_info: Information retrieved so far
            
        Returns:
            Plan for the next retrieval step
        """
        prompt = self.planning_prompt.format(
            query=query,
            existing_info=existing_info or "No information retrieved yet."
        )
        
        plan = await self.llm.apredict(prompt)
        return plan
    
    async def synthesize_answer(self, query: str, retrieved_info: List[Tuple[str, str]]) -> str:
        """
        Synthesize a final answer from multiple retrieval steps
        
        Args:
            query: The original user query
            retrieved_info: List of (search_query, results) tuples
            
        Returns:
            Synthesized answer to the original query
        """
        # Format the retrieved information
        formatted_info = ""
        for i, (search_query, results) in enumerate(retrieved_info):
            formatted_info += f"Step {i+1}:\n"
            formatted_info += f"Search Query: {search_query}\n"
            formatted_info += f"Results: {results}\n\n"
        
        prompt = self.synthesis_prompt.format(
            query=query,
            retrieved_info=formatted_info
        )
        
        answer = await self.llm.apredict(prompt)
        return answer
    
    async def execute_multi_hop_query(self, query: str) -> str:
        """
        Execute a multi-hop query using sequential retrieval steps
        
        Args:
            query: The user's query
            
        Returns:
            Final answer after multiple retrieval steps
        """
        retrieved_info = []
        existing_info = ""
        
        # Perform retrieval hops
        for hop in range(self.max_hops):
            # Generate plan for next search
            plan = await self.generate_search_plan(query, existing_info)
            
            # Extract search query from plan
            search_query_lines = [line for line in plan.split('\n') if line.startswith("Search Query:")]
            if search_query_lines:
                search_query = search_query_lines[0].replace("Search Query:", "").strip()
            else:
                # If no specific search query found, use a relevant part of the plan
                plan_lines = plan.split('\n')
                search_query = plan_lines[0] if plan_lines else plan[:100]
            
            # Retrieve documents using the planned search query
            documents = await self.retriever.aget_relevant_documents(search_query)
            
            # Extract and format the relevant information
            results = "\n".join([doc.page_content for doc in documents])
            
            # Store this retrieval step
            retrieved_info.append((search_query, results))
            
            # Update existing information for next iteration
            existing_info += f"From search '{search_query}':\n{results}\n\n"
            
            # Check if we have enough information for an answer
            if "sufficient information" in plan.lower() or "complete answer" in plan.lower():
                break
        
        # Synthesize final answer from all retrieval steps
        answer = await self.synthesize_answer(query, retrieved_info)
        
        return answer
    
    def generate_search_plan_sync(self, query: str, existing_info: str = "") -> str:
        """Synchronous version of generate_search_plan"""
        prompt = self.planning_prompt.format(
            query=query,
            existing_info=existing_info or "No information retrieved yet."
        )
        
        plan = self.llm.predict(prompt)
        return plan
    
    def synthesize_answer_sync(self, query: str, retrieved_info: List[Tuple[str, str]]) -> str:
        """Synchronous version of synthesize_answer"""
        # Format the retrieved information
        formatted_info = ""
        for i, (search_query, results) in enumerate(retrieved_info):
            formatted_info += f"Step {i+1}:\n"
            formatted_info += f"Search Query: {search_query}\n"
            formatted_info += f"Results: {results}\n\n"
        
        prompt = self.synthesis_prompt.format(
            query=query,
            retrieved_info=formatted_info
        )
        
        answer = self.llm.predict(prompt)
        return answer
    
    def execute_multi_hop_query_sync(self, query: str) -> str:
        """Synchronous version of execute_multi_hop_query"""
        retrieved_info = []
        existing_info = ""
        
        # Perform retrieval hops
        for hop in range(self.max_hops):
            # Generate plan for next search
            plan = self.generate_search_plan_sync(query, existing_info)
            
            # Extract search query from plan
            search_query_lines = [line for line in plan.split('\n') if line.startswith("Search Query:")]
            if search_query_lines:
                search_query = search_query_lines[0].replace("Search Query:", "").strip()
            else:
                # If no specific search query found, use a relevant part of the plan
                plan_lines = plan.split('\n')
                search_query = plan_lines[0] if plan_lines else plan[:100]
            
            # Retrieve documents using the planned search query
            documents = self.retriever.get_relevant_documents(search_query)
            
            # Extract and format the relevant information
            results = "\n".join([doc.page_content for doc in documents])
            
            # Store this retrieval step
            retrieved_info.append((search_query, results))
            
            # Update existing information for next iteration
            existing_info += f"From search '{search_query}':\n{results}\n\n"
            
            # Check if we have enough information for an answer
            if "sufficient information" in plan.lower() or "complete answer" in plan.lower():
                break
        
        # Synthesize final answer from all retrieval steps
        answer = self.synthesize_answer_sync(query, retrieved_info)
        
        return answer 