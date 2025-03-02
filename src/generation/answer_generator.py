"""
Answer generator module for synthesizing answers from retrieved context
"""
import logging
import re
from typing import Dict, List, Optional, Any, Union

from langchain_core.language_models import BaseLanguageModel
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from src.context_processing.processor import ContextProcessor

logger = logging.getLogger(__name__)


class AnswerWithSources(BaseModel):
    """Model representing an answer with source citations"""
    answer: str = Field(..., description="The generated answer to the query")
    sources: List[Dict[str, str]] = Field(
        ..., 
        description="List of sources used to generate the answer, with metadata"
    )
    

class BaseAnswerGenerator:
    """Base class for answer generators"""
    
    def __init__(self):
        """Initialize the base answer generator"""
        pass
    
    async def generate_answer(self, query: str, context: List[Document]) -> str:
        """
        Generate an answer from the provided query and context
        
        Args:
            query: User's query
            context: List of context documents
            
        Returns:
            Generated answer
        """
        raise NotImplementedError("Subclasses must implement generate_answer")
    
    def generate_answer_sync(self, query: str, context: List[Document]) -> str:
        """
        Synchronous version of generate_answer
        
        Args:
            query: User's query
            context: List of context documents
            
        Returns:
            Generated answer
        """
        raise NotImplementedError("Subclasses must implement generate_answer_sync")


class LLMAnswerGenerator(BaseAnswerGenerator):
    """
    Answer generator using a language model to synthesize answers from context
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        use_structured_output: bool = True,
        include_sources: bool = True,
        context_processor: Optional[ContextProcessor] = None,
        max_context_length: int = 4000,
        default_system_prompt: Optional[str] = None
    ):
        """
        Initialize the LLM-based answer generator
        
        Args:
            llm: Language model for generating answers
            use_structured_output: Whether to use structured output with citations
            include_sources: Whether to include source citations in the output
            context_processor: Optional context processor for preparing documents
            max_context_length: Maximum token length for context (approximate)
            default_system_prompt: Optional default system prompt to prepend
        """
        super().__init__()
        self.llm = llm
        self.use_structured_output = use_structured_output
        self.include_sources = include_sources
        self.context_processor = context_processor
        self.max_context_length = max_context_length
        self.default_system_prompt = default_system_prompt or (
            "You are a helpful, accurate assistant that provides informative answers based on the given context."
        )
        
        # Initialize prompts and parsers
        self._init_prompts()
        
        if self.use_structured_output and self.include_sources:
            self.parser = PydanticOutputParser(pydantic_object=AnswerWithSources)
    
    def _init_prompts(self):
        """Initialize prompt templates for answer generation"""
        # Basic answer generation prompt
        self.base_prompt_template = PromptTemplate(
            input_variables=["query", "context"],
            template="""
{system_prompt}

Use the provided context to answer the question. Be accurate, helpful, concise, and clear.
If the context doesn't contain the necessary information, indicate that you don't have enough information
rather than making up an answer.

Context:
{context}

Question: {query}

Answer:
"""
        )
        
        # Answer generation prompt with sources
        self.sources_prompt_template = PromptTemplate(
            input_variables=["query", "context"],
            template="""
{system_prompt}

Use the provided context to answer the question. Be accurate, helpful, concise, and clear.
If the context doesn't contain the necessary information, indicate that you don't have enough information
rather than making up an answer.

For each part of your answer, include a citation to the relevant source document like [doc1], [doc2], etc.
Citations should be included inline at the end of the sentence or paragraph they support.

Context:
{context}

Question: {query}

Answer with citations:
"""
        )
        
        # Structured output prompt with sources
        self.structured_prompt_template = PromptTemplate(
            input_variables=["query", "context", "format_instructions"],
            template="""
{system_prompt}

Use the provided context to answer the question. Be accurate, helpful, concise, and clear.
If the context doesn't contain the necessary information, indicate that you don't have enough information
rather than making up an answer.

For each part of your answer, carefully track which source document(s) support that information.
You'll need to provide these sources in a structured format.

Context:
{context}

Question: {query}

{format_instructions}

Generate your answer with sources:
"""
        )
    
    def _prepare_context(self, documents: List[Document]) -> str:
        """
        Prepare context documents for inclusion in the prompt
        
        Args:
            documents: List of context documents
            
        Returns:
            Formatted context string
        """
        # Process context if processor available
        if self.context_processor:
            processed_docs = self.context_processor.process_documents(documents)
        else:
            processed_docs = documents
        
        # Format documents with document IDs
        context_strings = []
        for i, doc in enumerate(processed_docs):
            # Get source if available in metadata
            metadata = doc.metadata or {}
            source = metadata.get("source", f"doc{i+1}")
            
            # Format the document
            doc_string = f"[doc{i+1}] "
            if "title" in metadata:
                doc_string += f"Title: {metadata['title']}\n"
            
            doc_string += f"Source: {source}\n"
            doc_string += doc.page_content
            
            context_strings.append(doc_string)
        
        # Combine all documents
        context_text = "\n\n".join(context_strings)
        
        # Limit context length (simple approximation)
        if len(context_text) > self.max_context_length * 4:  # rough char to token ratio
            truncated_text = context_text[:self.max_context_length * 4]
            return truncated_text + "\n\n[Context truncated due to length]"
        
        return context_text
    
    def _extract_sources(self, documents: List[Document]) -> List[Dict[str, str]]:
        """
        Extract source information from documents
        
        Args:
            documents: List of context documents
            
        Returns:
            List of source dictionaries
        """
        sources = []
        for i, doc in enumerate(documents):
            metadata = doc.metadata or {}
            source_info = {
                "id": f"doc{i+1}",
                "source": metadata.get("source", "Unknown"),
            }
            
            if "title" in metadata:
                source_info["title"] = metadata["title"]
                
            if "url" in metadata:
                source_info["url"] = metadata["url"]
                
            sources.append(source_info)
            
        return sources
    
    def _extract_citations(self, text: str) -> Dict[str, List[str]]:
        """
        Extract citation markers from generated text
        
        Args:
            text: Text with citation markers like [doc1]
            
        Returns:
            Dictionary mapping source IDs to cited text excerpts
        """
        # Find all citation markers like [doc1], [doc2], etc.
        citation_pattern = r'\[doc(\d+)\]'
        citations = {}
        
        # Split text into sentences/segments
        segments = re.split(r'(?<=[.!?])\s+', text)
        
        for segment in segments:
            # Find citations in this segment
            matches = re.findall(citation_pattern, segment)
            
            for doc_id in matches:
                source_id = f"doc{doc_id}"
                if source_id not in citations:
                    citations[source_id] = []
                
                # Add the segment without the citation markers
                clean_segment = re.sub(citation_pattern, '', segment).strip()
                citations[source_id].append(clean_segment)
        
        return citations
    
    async def generate_answer(self, query: str, context: List[Document]) -> Union[str, Dict[str, Any]]:
        """
        Generate an answer from the provided query and context
        
        Args:
            query: User's query
            context: List of context documents
            
        Returns:
            Generated answer (either string or structured output)
        """
        # Prepare context
        formatted_context = self._prepare_context(context)
        
        try:
            # Choose prompt based on configuration
            if self.use_structured_output and self.include_sources:
                prompt = self.structured_prompt_template.format(
                    system_prompt=self.default_system_prompt,
                    query=query,
                    context=formatted_context,
                    format_instructions=self.parser.get_format_instructions()
                )
            elif self.include_sources:
                prompt = self.sources_prompt_template.format(
                    system_prompt=self.default_system_prompt,
                    query=query,
                    context=formatted_context
                )
            else:
                prompt = self.base_prompt_template.format(
                    system_prompt=self.default_system_prompt,
                    query=query,
                    context=formatted_context
                )
            
            # Generate answer using LLM
            result = await self.llm.apredict(prompt)
            
            # Process the result based on configuration
            if self.use_structured_output and self.include_sources:
                try:
                    # Parse structured output
                    parsed_result = self.parser.parse(result)
                    
                    # Return the structured output
                    return {
                        "answer": parsed_result.answer,
                        "sources": parsed_result.sources
                    }
                except Exception as e:
                    logger.error(f"Error parsing structured output: {str(e)}")
                    
                    # Fall back to extracting citations manually
                    citations = self._extract_citations(result)
                    sources = self._extract_sources(context)
                    
                    # Filter sources to only include cited ones
                    cited_sources = [s for s in sources if s["id"] in citations]
                    
                    # Extract the answer part (before any "SOURCES:" section if present)
                    answer_part = result.split("SOURCES:")[0].strip()
                    
                    return {
                        "answer": answer_part,
                        "sources": cited_sources
                    }
            
            elif self.include_sources:
                # Extract citations manually for non-structured output
                citations = self._extract_citations(result)
                sources = self._extract_sources(context)
                
                # Return the answer with extracted citations
                return {
                    "answer": result,
                    "sources": [s for s in sources if s["id"] in citations]
                }
            
            else:
                # Return simple string answer
                return result
        
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "I encountered an error while generating an answer. Please try again."
    
    def generate_answer_sync(self, query: str, context: List[Document]) -> Union[str, Dict[str, Any]]:
        """
        Synchronous version of generate_answer
        
        Args:
            query: User's query
            context: List of context documents
            
        Returns:
            Generated answer (either string or structured output)
        """
        # Prepare context
        formatted_context = self._prepare_context(context)
        
        try:
            # Choose prompt based on configuration
            if self.use_structured_output and self.include_sources:
                prompt = self.structured_prompt_template.format(
                    system_prompt=self.default_system_prompt,
                    query=query,
                    context=formatted_context,
                    format_instructions=self.parser.get_format_instructions()
                )
            elif self.include_sources:
                prompt = self.sources_prompt_template.format(
                    system_prompt=self.default_system_prompt,
                    query=query,
                    context=formatted_context
                )
            else:
                prompt = self.base_prompt_template.format(
                    system_prompt=self.default_system_prompt,
                    query=query,
                    context=formatted_context
                )
            
            # Generate answer using LLM
            result = self.llm.predict(prompt)
            
            # Process the result based on configuration
            if self.use_structured_output and self.include_sources:
                try:
                    # Parse structured output
                    parsed_result = self.parser.parse(result)
                    
                    # Return the structured output
                    return {
                        "answer": parsed_result.answer,
                        "sources": parsed_result.sources
                    }
                except Exception as e:
                    logger.error(f"Error parsing structured output: {str(e)}")
                    
                    # Fall back to extracting citations manually
                    citations = self._extract_citations(result)
                    sources = self._extract_sources(context)
                    
                    # Filter sources to only include cited ones
                    cited_sources = [s for s in sources if s["id"] in citations]
                    
                    # Extract the answer part (before any "SOURCES:" section if present)
                    answer_part = result.split("SOURCES:")[0].strip()
                    
                    return {
                        "answer": answer_part,
                        "sources": cited_sources
                    }
            
            elif self.include_sources:
                # Extract citations manually for non-structured output
                citations = self._extract_citations(result)
                sources = self._extract_sources(context)
                
                # Return the answer with extracted citations
                return {
                    "answer": result,
                    "sources": [s for s in sources if s["id"] in citations]
                }
            
            else:
                # Return simple string answer
                return result
        
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "I encountered an error while generating an answer. Please try again."


class StreamingAnswerGenerator(BaseAnswerGenerator):
    """
    Answer generator that streams responses for better user experience
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        include_sources: bool = True,
        context_processor: Optional[ContextProcessor] = None,
        max_context_length: int = 4000,
        default_system_prompt: Optional[str] = None
    ):
        """
        Initialize the streaming answer generator
        
        Args:
            llm: Language model for generating answers (must support streaming)
            include_sources: Whether to include source citations in the output
            context_processor: Optional context processor for preparing documents
            max_context_length: Maximum token length for context (approximate)
            default_system_prompt: Optional default system prompt to prepend
        """
        super().__init__()
        self.llm = llm
        self.include_sources = include_sources
        self.context_processor = context_processor
        self.max_context_length = max_context_length
        self.default_system_prompt = default_system_prompt or (
            "You are a helpful, accurate assistant that provides informative answers based on the given context."
        )
        
        # Initialize prompts
        self._init_prompts()
    
    def _init_prompts(self):
        """Initialize prompt templates for streaming answer generation"""
        # Basic streaming prompt
        self.streaming_prompt_template = PromptTemplate(
            input_variables=["query", "context"],
            template="""
{system_prompt}

Use the provided context to answer the question. Be accurate, helpful, concise, and clear.
If the context doesn't contain the necessary information, indicate that you don't have enough information
rather than making up an answer.

Context:
{context}

Question: {query}

Answer:
"""
        )
        
        # Streaming prompt with sources
        self.streaming_sources_prompt_template = PromptTemplate(
            input_variables=["query", "context"],
            template="""
{system_prompt}

Use the provided context to answer the question. Be accurate, helpful, concise, and clear.
If the context doesn't contain the necessary information, indicate that you don't have enough information
rather than making up an answer.

For each part of your answer, include a citation to the relevant source document like [doc1], [doc2], etc.
Citations should be included inline at the end of the sentence or paragraph they support.

After completing your answer, on a new line add "SOURCES:" followed by a numbered list of all the sources you cited,
with their corresponding document IDs, titles, and any available metadata.

Context:
{context}

Question: {query}

Answer with citations:
"""
        )
    
    def _prepare_context(self, documents: List[Document]) -> str:
        """
        Prepare context documents for inclusion in the prompt
        
        Args:
            documents: List of context documents
            
        Returns:
            Formatted context string
        """
        # Process context if processor available
        if self.context_processor:
            processed_docs = self.context_processor.process_documents(documents)
        else:
            processed_docs = documents
        
        # Format documents with document IDs
        context_strings = []
        for i, doc in enumerate(processed_docs):
            # Get source if available in metadata
            metadata = doc.metadata or {}
            source = metadata.get("source", f"doc{i+1}")
            
            # Format the document
            doc_string = f"[doc{i+1}] "
            if "title" in metadata:
                doc_string += f"Title: {metadata['title']}\n"
            
            doc_string += f"Source: {source}\n"
            doc_string += doc.page_content
            
            context_strings.append(doc_string)
        
        # Combine all documents
        context_text = "\n\n".join(context_strings)
        
        # Limit context length (simple approximation)
        if len(context_text) > self.max_context_length * 4:  # rough char to token ratio
            truncated_text = context_text[:self.max_context_length * 4]
            return truncated_text + "\n\n[Context truncated due to length]"
        
        return context_text
    
    def _extract_sources(self, documents: List[Document]) -> List[Dict[str, str]]:
        """
        Extract source information from documents
        
        Args:
            documents: List of context documents
            
        Returns:
            List of source dictionaries
        """
        sources = []
        for i, doc in enumerate(documents):
            metadata = doc.metadata or {}
            source_info = {
                "id": f"doc{i+1}",
                "source": metadata.get("source", "Unknown"),
            }
            
            if "title" in metadata:
                source_info["title"] = metadata["title"]
                
            if "url" in metadata:
                source_info["url"] = metadata["url"]
                
            sources.append(source_info)
            
        return sources
    
    async def generate_answer_stream(self, query: str, context: List[Document]):
        """
        Generate a streaming answer from the provided query and context
        
        Args:
            query: User's query
            context: List of context documents
            
        Returns:
            Async generator yielding answer chunks
        """
        # Prepare context
        formatted_context = self._prepare_context(context)
        
        try:
            # Choose prompt based on configuration
            if self.include_sources:
                prompt = self.streaming_sources_prompt_template.format(
                    system_prompt=self.default_system_prompt,
                    query=query,
                    context=formatted_context
                )
            else:
                prompt = self.streaming_prompt_template.format(
                    system_prompt=self.default_system_prompt,
                    query=query,
                    context=formatted_context
                )
            
            # Generate answer using streaming LLM
            async for chunk in self.llm.astream(prompt):
                yield chunk
        
        except Exception as e:
            logger.error(f"Error generating streaming answer: {str(e)}")
            yield "I encountered an error while generating an answer. Please try again."
    
    async def generate_answer(self, query: str, context: List[Document]) -> str:
        """
        Generate a complete answer (non-streaming) from the provided query and context
        
        Args:
            query: User's query
            context: List of context documents
            
        Returns:
            Complete generated answer
        """
        # For the non-streaming version, collect all chunks
        result = ""
        async for chunk in self.generate_answer_stream(query, context):
            result += chunk
        
        return result
    
    def generate_answer_sync(self, query: str, context: List[Document]) -> str:
        """
        Synchronous version of generate_answer
        
        Args:
            query: User's query
            context: List of context documents
            
        Returns:
            Complete generated answer
        """
        # Prepare context
        formatted_context = self._prepare_context(context)
        
        try:
            # Choose prompt based on configuration
            if self.include_sources:
                prompt = self.streaming_sources_prompt_template.format(
                    system_prompt=self.default_system_prompt,
                    query=query,
                    context=formatted_context
                )
            else:
                prompt = self.streaming_prompt_template.format(
                    system_prompt=self.default_system_prompt,
                    query=query,
                    context=formatted_context
                )
            
            # Generate answer using LLM
            return self.llm.predict(prompt)
        
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "I encountered an error while generating an answer. Please try again."


class TemplatedAnswerGenerator(BaseAnswerGenerator):
    """
    Answer generator that uses customizable templates for different query types
    """
    
    def __init__(
        self,
        llm: BaseLanguageModel,
        context_processor: Optional[ContextProcessor] = None,
        max_context_length: int = 4000,
        default_system_prompt: Optional[str] = None,
        templates: Optional[Dict[str, str]] = None
    ):
        """
        Initialize the templated answer generator
        
        Args:
            llm: Language model for generating answers
            context_processor: Optional context processor for preparing documents
            max_context_length: Maximum token length for context
            default_system_prompt: Optional default system prompt to prepend
            templates: Dictionary mapping query types to prompt templates
        """
        super().__init__()
        self.llm = llm
        self.context_processor = context_processor
        self.max_context_length = max_context_length
        self.default_system_prompt = default_system_prompt or (
            "You are a helpful, accurate assistant that provides informative answers based on the given context."
        )
        
        # Initialize templates
        self.templates = templates or self._get_default_templates()
    
    def _get_default_templates(self) -> Dict[str, str]:
        """
        Get default templates for different query types
        
        Returns:
            Dictionary mapping query types to prompt templates
        """
        return {
            "default": """
{system_prompt}

Use the provided context to answer the question. Be accurate, helpful, concise, and clear.
If the context doesn't contain the necessary information, indicate that you don't have enough information
rather than making up an answer.

Context:
{context}

Question: {query}

Answer:
""",
            "factual": """
{system_prompt}

Use the provided context to answer the factual question accurately and concisely.
Stick strictly to the facts provided in the context without adding any speculation.
If the context doesn't contain the necessary information, clearly state that.

Context:
{context}

Factual Question: {query}

Factual Answer:
""",
            "procedural": """
{system_prompt}

The user is asking about a process or procedure. Answer their question using the provided context,
organizing your response as a clear step-by-step guide. Use numbered steps where appropriate.
If the context doesn't contain complete procedural information, indicate what's missing.

Context:
{context}

Procedural Question: {query}

Step-by-Step Answer:
""",
            "conceptual": """
{system_prompt}

The user is asking about a concept or idea. Use the provided context to give a clear explanation
that helps them understand the concept. Include examples if helpful.
If the context doesn't provide enough information about this concept, indicate that.

Context:
{context}

Conceptual Question: {query}

Explanation:
""",
            "comparative": """
{system_prompt}

The user is asking for a comparison. Based on the provided context, identify the key similarities
and differences between the items being compared. Present your answer in a structured way
that makes the comparison clear. Use tables if appropriate.
If the context doesn't contain sufficient comparative information, indicate that.

Context:
{context}

Comparison Question: {query}

Comparative Analysis:
"""
        }
    
    def _prepare_context(self, documents: List[Document]) -> str:
        """
        Prepare context documents for inclusion in the prompt
        
        Args:
            documents: List of context documents
            
        Returns:
            Formatted context string
        """
        # Process context if processor available
        if self.context_processor:
            processed_docs = self.context_processor.process_documents(documents)
        else:
            processed_docs = documents
        
        # Format documents with document IDs
        context_strings = []
        for i, doc in enumerate(processed_docs):
            # Get source if available in metadata
            metadata = doc.metadata or {}
            source = metadata.get("source", f"doc{i+1}")
            
            # Format the document
            doc_string = f"[doc{i+1}] "
            if "title" in metadata:
                doc_string += f"Title: {metadata['title']}\n"
            
            doc_string += f"Source: {source}\n"
            doc_string += doc.page_content
            
            context_strings.append(doc_string)
        
        # Combine all documents
        context_text = "\n\n".join(context_strings)
        
        # Limit context length (simple approximation)
        if len(context_text) > self.max_context_length * 4:  # rough char to token ratio
            truncated_text = context_text[:self.max_context_length * 4]
            return truncated_text + "\n\n[Context truncated due to length]"
        
        return context_text
    
    def _detect_query_type(self, query: str) -> str:
        """
        Detect the type of query to determine which template to use
        
        Args:
            query: User's query
            
        Returns:
            Detected query type (key in templates dict)
        """
        query = query.lower()
        
        # Simple rule-based detection
        if any(word in query for word in ["compare", "difference", "versus", "vs", "similarities", "different"]):
            return "comparative"
        
        if any(word in query for word in ["how to", "steps", "process", "procedure", "guide", "tutorial"]):
            return "procedural"
        
        if any(word in query for word in ["what is", "define", "explain", "concept", "meaning", "understand"]):
            return "conceptual"
        
        if any(word in query for word in ["who", "when", "where", "which", "was", "did", "fact"]):
            return "factual"
        
        # Default to general template
        return "default"
    
    async def generate_answer(self, query: str, context: List[Document]) -> str:
        """
        Generate an answer from the provided query and context
        
        Args:
            query: User's query
            context: List of context documents
            
        Returns:
            Generated answer
        """
        # Prepare context
        formatted_context = self._prepare_context(context)
        
        # Detect query type
        query_type = self._detect_query_type(query)
        
        # Get appropriate template
        template = self.templates.get(query_type, self.templates["default"])
        
        try:
            # Format the prompt
            prompt = template.format(
                system_prompt=self.default_system_prompt,
                query=query,
                context=formatted_context
            )
            
            # Generate answer using LLM
            result = await self.llm.apredict(prompt)
            return result
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "I encountered an error while generating an answer. Please try again."
    
    def generate_answer_sync(self, query: str, context: List[Document]) -> str:
        """
        Synchronous version of generate_answer
        
        Args:
            query: User's query
            context: List of context documents
            
        Returns:
            Generated answer
        """
        # Prepare context
        formatted_context = self._prepare_context(context)
        
        # Detect query type
        query_type = self._detect_query_type(query)
        
        # Get appropriate template
        template = self.templates.get(query_type, self.templates["default"])
        
        try:
            # Format the prompt
            prompt = template.format(
                system_prompt=self.default_system_prompt,
                query=query,
                context=formatted_context
            )
            
            # Generate answer using LLM
            result = self.llm.predict(prompt)
            return result
            
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            return "I encountered an error while generating an answer. Please try again." 