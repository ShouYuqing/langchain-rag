"""
Context augmentation for enhancing retrieval context
"""
from typing import Dict, List, Optional, Union, Any, Callable
import re
import logging
from langchain_core.documents import Document
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate


class ContextAugmenter:
    """
    Augments retrieved context with additional information
    """
    
    def __init__(
        self,
        llm: Optional[BaseLanguageModel] = None,
        add_summaries: bool = True,
        add_entity_definitions: bool = True,
        add_knowledge_graph: bool = False,
        add_cross_references: bool = True
    ):
        """
        Initialize context augmenter
        
        Args:
            llm: Language model for generating summaries and entity definitions
            add_summaries: Whether to add summaries to context
            add_entity_definitions: Whether to add entity definitions
            add_knowledge_graph: Whether to add knowledge graph connections
            add_cross_references: Whether to add cross-references between documents
        """
        self.llm = llm
        self.add_summaries = add_summaries
        self.add_entity_definitions = add_entity_definitions
        self.add_knowledge_graph = add_knowledge_graph
        self.add_cross_references = add_cross_references
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize prompt templates
        self._init_prompts()
    
    def _init_prompts(self):
        """Initialize prompt templates for context augmentation"""
        # Summary generation prompt
        self.summary_prompt_template = PromptTemplate(
            input_variables=["content"],
            template="""Create a concise summary (2-3 sentences) of the following document excerpt:

{content}

Summary:"""
        )
        
        # Entity extraction prompt
        self.entity_extraction_prompt_template = PromptTemplate(
            input_variables=["content"],
            template="""Extract the key entities (people, organizations, technical terms, concepts) from this document:

{content}

For each entity, provide a brief definition based only on the information in this document.
Format as:
Entity: [Entity name]
Definition: [Brief definition based on the document]
"""
        )
        
        # Cross-reference identification prompt
        self.cross_reference_prompt_template = PromptTemplate(
            input_variables=["documents"],
            template="""Below are several document excerpts. Identify connections between them, specifically looking for:
1. When one document refers to a concept explained in another document
2. When documents contain complementary information
3. When documents present conflicting or contradictory information

{documents}

List any cross-references between documents in this format:
- Document [X] and Document [Y]: [Brief description of the connection]
"""
        )
    
    def _extract_entities(self, document: Document) -> List[Dict[str, str]]:
        """
        Extract entities and their definitions from a document
        
        Args:
            document: Document to extract entities from
            
        Returns:
            List of entities with their definitions
        """
        if not self.llm:
            return []
        
        try:
            # Prepare the content (limit length to avoid token limits)
            content = document.page_content
            if len(content) > 2000:
                content = content[:2000]
            
            # Generate entity extraction
            prompt = self.entity_extraction_prompt_template.format(content=content)
            result = self.llm.predict(prompt)
            
            # Parse the result
            entities = []
            current_entity = None
            current_definition = ""
            
            for line in result.split("\n"):
                line = line.strip()
                if not line:
                    continue
                
                if line.startswith("Entity:"):
                    # Save previous entity if exists
                    if current_entity and current_definition:
                        entities.append({
                            "entity": current_entity,
                            "definition": current_definition
                        })
                    
                    # Start new entity
                    current_entity = line[len("Entity:"):].strip()
                    current_definition = ""
                
                elif line.startswith("Definition:") and current_entity:
                    current_definition = line[len("Definition:"):].strip()
            
            # Add the last entity
            if current_entity and current_definition:
                entities.append({
                    "entity": current_entity,
                    "definition": current_definition
                })
            
            return entities
            
        except Exception as e:
            self.logger.error(f"Error extracting entities: {str(e)}")
            return []
    
    def _generate_summary(self, document: Document) -> str:
        """
        Generate a summary for a document
        
        Args:
            document: Document to summarize
            
        Returns:
            Generated summary
        """
        if not self.llm:
            return ""
        
        try:
            # Prepare the content (limit length to avoid token limits)
            content = document.page_content
            if len(content) > 2000:
                content = content[:2000]
            
            # Generate summary
            prompt = self.summary_prompt_template.format(content=content)
            summary = self.llm.predict(prompt)
            
            return summary.strip()
            
        except Exception as e:
            self.logger.error(f"Error generating summary: {str(e)}")
            return ""
    
    def _identify_cross_references(self, documents: List[Document]) -> List[Dict[str, Any]]:
        """
        Identify cross-references between documents
        
        Args:
            documents: List of documents to analyze
            
        Returns:
            List of cross-reference information
        """
        if not self.llm or len(documents) <= 1:
            return []
        
        try:
            # Prepare document excerpts (limited to save tokens)
            doc_excerpts = []
            for i, doc in enumerate(documents[:5]):  # Limit to first 5 documents
                excerpt = doc.page_content
                if len(excerpt) > 500:
                    excerpt = excerpt[:500] + "..."
                
                doc_excerpts.append(f"Document {i+1}:\n{excerpt}\n")
            
            # Format documents text
            documents_text = "\n".join(doc_excerpts)
            
            # Generate cross-references
            prompt = self.cross_reference_prompt_template.format(documents=documents_text)
            result = self.llm.predict(prompt)
            
            # Parse the result (simple regex parsing)
            cross_refs = []
            pattern = r"Document\s+\[?(\d+)\]?\s+and\s+Document\s+\[?(\d+)\]?:(.*?)(?=$|Document|\n\n)"
            matches = re.finditer(pattern, result, re.DOTALL)
            
            for match in matches:
                doc1 = int(match.group(1)) - 1
                doc2 = int(match.group(2)) - 1
                description = match.group(3).strip()
                
                if 0 <= doc1 < len(documents) and 0 <= doc2 < len(documents):
                    cross_refs.append({
                        "document1_idx": doc1,
                        "document2_idx": doc2,
                        "description": description
                    })
            
            return cross_refs
            
        except Exception as e:
            self.logger.error(f"Error identifying cross-references: {str(e)}")
            return []
    
    def augment_document(self, document: Document) -> Document:
        """
        Augment a single document with additional information
        
        Args:
            document: Document to augment
            
        Returns:
            Augmented document
        """
        # Create a copy to avoid modifying the original
        augmented_doc = Document(
            page_content=document.page_content,
            metadata=document.metadata.copy() if document.metadata else {}
        )
        
        # Add summary if requested
        if self.add_summaries and self.llm:
            summary = self._generate_summary(document)
            if summary:
                # Add to metadata
                if not augmented_doc.metadata:
                    augmented_doc.metadata = {}
                augmented_doc.metadata["summary"] = summary
        
        # Add entity definitions if requested
        if self.add_entity_definitions and self.llm:
            entities = self._extract_entities(document)
            if entities:
                # Add to metadata
                if not augmented_doc.metadata:
                    augmented_doc.metadata = {}
                augmented_doc.metadata["entities"] = entities
        
        return augmented_doc
    
    def augment_documents(self, documents: List[Document]) -> List[Document]:
        """
        Augment a list of documents with additional information
        
        Args:
            documents: List of documents to augment
            
        Returns:
            List of augmented documents
        """
        if not documents:
            return []
        
        # Augment individual documents
        augmented_docs = [self.augment_document(doc) for doc in documents]
        
        # Add cross-references if requested
        if self.add_cross_references and self.llm and len(documents) > 1:
            cross_refs = self._identify_cross_references(documents)
            
            # Add cross-references to document metadata
            if cross_refs:
                for cross_ref in cross_refs:
                    doc1_idx = cross_ref["document1_idx"]
                    doc2_idx = cross_ref["document2_idx"]
                    description = cross_ref["description"]
                    
                    if 0 <= doc1_idx < len(augmented_docs) and 0 <= doc2_idx < len(augmented_docs):
                        # Add reference to document 1
                        if not augmented_docs[doc1_idx].metadata:
                            augmented_docs[doc1_idx].metadata = {}
                        
                        if "cross_references" not in augmented_docs[doc1_idx].metadata:
                            augmented_docs[doc1_idx].metadata["cross_references"] = []
                        
                        augmented_docs[doc1_idx].metadata["cross_references"].append({
                            "referenced_doc_idx": doc2_idx,
                            "description": description
                        })
                        
                        # Add reference to document 2
                        if not augmented_docs[doc2_idx].metadata:
                            augmented_docs[doc2_idx].metadata = {}
                        
                        if "cross_references" not in augmented_docs[doc2_idx].metadata:
                            augmented_docs[doc2_idx].metadata["cross_references"] = []
                        
                        augmented_docs[doc2_idx].metadata["cross_references"].append({
                            "referenced_doc_idx": doc1_idx,
                            "description": description
                        })
        
        # Add knowledge graph if requested
        if self.add_knowledge_graph and len(documents) > 1:
            # We'd need a way to generate a knowledge graph
            # This could be from the DocumentChainProcessor if available
            pass
        
        return augmented_docs


class SemanticContextFormatter:
    """
    Format context documents into a semantic structure for better LLM comprehension
    """
    
    def __init__(
        self,
        llm: Optional[BaseLanguageModel] = None,
        add_metadata_to_context: bool = True,
        group_by_source: bool = True,
        organize_by_relevance: bool = True,
        include_summaries: bool = True,
        include_entities: bool = True,
        include_cross_references: bool = True
    ):
        """
        Initialize the semantic context formatter
        
        Args:
            llm: Language model for generating additional context information
            add_metadata_to_context: Whether to add document metadata to formatted context
            group_by_source: Whether to group documents by source
            organize_by_relevance: Whether to organize documents by relevance
            include_summaries: Whether to include document summaries
            include_entities: Whether to include entity definitions
            include_cross_references: Whether to include cross-references
        """
        self.llm = llm
        self.add_metadata_to_context = add_metadata_to_context
        self.group_by_source = group_by_source
        self.organize_by_relevance = organize_by_relevance
        self.include_summaries = include_summaries
        self.include_entities = include_entities
        self.include_cross_references = include_cross_references
        
        self.logger = logging.getLogger(__name__)
    
    def _format_document(self, doc: Document, index: int) -> str:
        """
        Format a single document with its metadata
        
        Args:
            doc: Document to format
            index: Index of the document in the list
            
        Returns:
            Formatted document string
        """
        parts = []
        
        # Add document title/header
        source = doc.metadata.get("source", "") if doc.metadata else ""
        
        if source:
            parts.append(f"\n## Document {index+1}: {source}")
        else:
            parts.append(f"\n## Document {index+1}")
        
        # Add summary if available and requested
        if self.include_summaries and doc.metadata and "summary" in doc.metadata:
            parts.append(f"\nSummary: {doc.metadata['summary']}")
        
        # Add document content
        parts.append(f"\n{doc.page_content}")
        
        # Add metadata if requested
        if self.add_metadata_to_context and doc.metadata:
            # Add relevant metadata (filter out large or internal fields)
            metadata_to_include = {}
            exclude_keys = {"summary", "entities", "cross_references", "embedding", 
                            "relevance_score", "chunk_id", "parent_id"}
            
            for key, value in doc.metadata.items():
                if key not in exclude_keys and not isinstance(value, (list, dict, bytes)):
                    metadata_to_include[key] = value
            
            if metadata_to_include:
                parts.append("\nMetadata:")
                for key, value in metadata_to_include.items():
                    parts.append(f"- {key}: {value}")
        
        # Add entity definitions if available and requested
        if self.include_entities and doc.metadata and "entities" in doc.metadata:
            entities = doc.metadata["entities"]
            if entities:
                parts.append("\nKey entities:")
                for entity in entities:
                    parts.append(f"- {entity['entity']}: {entity['definition']}")
        
        # Add cross-references if available and requested
        if self.include_cross_references and doc.metadata and "cross_references" in doc.metadata:
            cross_refs = doc.metadata["cross_references"]
            if cross_refs:
                parts.append("\nRelated documents:")
                for ref in cross_refs:
                    parts.append(f"- Document {ref['referenced_doc_idx']+1}: {ref['description']}")
        
        return "\n".join(parts)
    
    def format_documents(self, documents: List[Document], query: str = "") -> str:
        """
        Format a list of documents into a semantic structure
        
        Args:
            documents: List of documents to format
            query: Original query for context
            
        Returns:
            Formatted context string
        """
        if not documents:
            return ""
        
        parts = []
        
        # Add header with query context
        if query:
            parts.append(f"Context information for answering the query: \"{query}\"")
        else:
            parts.append("Context information:")
        
        # Group documents by source if requested
        if self.group_by_source:
            # Create source groups
            source_groups = {}
            for i, doc in enumerate(documents):
                source = doc.metadata.get("source", "Unknown") if doc.metadata else "Unknown"
                if source not in source_groups:
                    source_groups[source] = []
                source_groups[source].append((i, doc))
            
            # Format each source group
            for source, docs in source_groups.items():
                if len(source_groups) > 1:
                    parts.append(f"\n# Source: {source}")
                
                # Format each document in the group
                for i, doc in docs:
                    parts.append(self._format_document(doc, i))
        else:
            # Format each document individually
            for i, doc in enumerate(documents):
                parts.append(self._format_document(doc, i))
        
        return "\n".join(parts)
    
    def format_documents_as_json(self, documents: List[Document]) -> Dict[str, Any]:
        """
        Format documents as a JSON structure for programmatic use
        
        Args:
            documents: List of documents to format
            
        Returns:
            Structured JSON representation of the documents
        """
        result = {
            "documents": []
        }
        
        for i, doc in enumerate(documents):
            document_data = {
                "index": i,
                "content": doc.page_content,
            }
            
            # Add metadata
            if doc.metadata:
                filtered_metadata = {
                    k: v for k, v in doc.metadata.items() 
                    if k not in {"embedding"} and not isinstance(v, bytes)
                }
                document_data["metadata"] = filtered_metadata
            
            result["documents"].append(document_data)
        
        return result 