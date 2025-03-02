"""
Intelligent document chunking strategies
"""
from typing import Dict, List, Optional, Union, Any, Callable
import re
import spacy
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    CharacterTextSplitter,
    Language
)
from langchain_core.documents import Document
import nltk
from nltk.tokenize import sent_tokenize

# Download NLTK data for sentence splitting
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# Load spacy model for semantic chunking
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Downloading spaCy model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

class SemanticChunker:
    """
    Advanced semantic chunking that respects document structure and semantics
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        chunking_method: str = "sentence",
        chunk_by_tokens: bool = True,
        max_single_chunk_size: int = 2000
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunking_method = chunking_method
        self.chunk_by_tokens = chunk_by_tokens
        self.max_single_chunk_size = max_single_chunk_size
        
        # Initialize appropriate splitter based on configuration
        if chunk_by_tokens:
            self.token_splitter = TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:
            self.char_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ".", " ", ""]
            )
    
    def _split_by_sentence(self, text: str) -> List[Document]:
        """Split text by sentences and then group sentences into chunks"""
        # First split by sentences
        sentences = sent_tokenize(text)
        
        # Initialize chunks
        chunks = []
        current_chunk = ""
        
        # Group sentences into chunks
        for sentence in sentences:
            # Check if adding this sentence exceeds the chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                chunks.append(Document(page_content=current_chunk))
                # Keep some overlap by retaining the last sentence
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(Document(page_content=current_chunk))
        
        return chunks
    
    def _split_by_paragraph(self, text: str) -> List[Document]:
        """Split text by paragraphs and then group paragraphs into chunks"""
        # Split by paragraphs (double newlines)
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Initialize chunks
        chunks = []
        current_chunk = ""
        
        # Group paragraphs into chunks
        for paragraph in paragraphs:
            # If paragraph is very long, split it further
            if len(paragraph) > self.max_single_chunk_size:
                if current_chunk:
                    chunks.append(Document(page_content=current_chunk))
                    current_chunk = ""
                
                # Split large paragraph using the token or char splitter
                if self.chunk_by_tokens:
                    sub_chunks = self.token_splitter.split_text(paragraph)
                else:
                    sub_chunks = self.char_splitter.split_text(paragraph)
                
                for sub_chunk in sub_chunks:
                    chunks.append(Document(page_content=sub_chunk))
                
                continue
            
            # Check if adding this paragraph exceeds the chunk size
            if len(current_chunk) + len(paragraph) > self.chunk_size and current_chunk:
                chunks.append(Document(page_content=current_chunk))
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(Document(page_content=current_chunk))
        
        return chunks
    
    def _split_by_semantic_units(self, text: str) -> List[Document]:
        """Split text by semantic units using spaCy"""
        # Process the text with spaCy
        doc = nlp(text)
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        
        # Extract semantic units (sentences or paragraphs)
        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_tokens = len(sent_text.split())
            
            # If sentence is very long, create a separate chunk
            if sent_tokens > self.chunk_size:
                if current_chunk:
                    chunks.append(Document(page_content=current_chunk))
                    current_chunk = ""
                    current_tokens = 0
                
                # Split long sentence using the token or char splitter
                if self.chunk_by_tokens:
                    sub_chunks = self.token_splitter.split_text(sent_text)
                else:
                    sub_chunks = self.char_splitter.split_text(sent_text)
                
                for sub_chunk in sub_chunks:
                    chunks.append(Document(page_content=sub_chunk))
                
                continue
            
            # Check if adding this sentence exceeds the chunk size
            if current_tokens + sent_tokens > self.chunk_size and current_chunk:
                chunks.append(Document(page_content=current_chunk))
                current_chunk = sent_text
                current_tokens = sent_tokens
            else:
                if current_chunk:
                    current_chunk += " " + sent_text
                else:
                    current_chunk = sent_text
                current_tokens += sent_tokens
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(Document(page_content=current_chunk))
        
        return chunks
    
    def _split_by_headings(self, text: str) -> List[Document]:
        """Split text by headings (e.g., Markdown headings)"""
        # Find headings in markdown format (# Heading)
        heading_pattern = r'(^|\n)(#{1,6}\s+[^\n]+)'
        parts = re.split(heading_pattern, text, flags=re.MULTILINE)
        
        # Combine heading with content
        sections = []
        for i in range(1, len(parts), 3):
            if i+1 < len(parts):
                sections.append(parts[i] + parts[i+1])
        
        # If no headings found, fallback to paragraph splitting
        if not sections:
            return self._split_by_paragraph(text)
        
        # Initialize chunks
        chunks = []
        current_chunk = ""
        
        # Group sections into chunks
        for section in sections:
            # If section is very long, split it further
            if len(section) > self.max_single_chunk_size:
                if current_chunk:
                    chunks.append(Document(page_content=current_chunk))
                    current_chunk = ""
                
                # Split large section using the token or char splitter
                if self.chunk_by_tokens:
                    sub_chunks = self.token_splitter.split_text(section)
                else:
                    sub_chunks = self.char_splitter.split_text(section)
                
                for sub_chunk in sub_chunks:
                    chunks.append(Document(page_content=sub_chunk))
                
                continue
            
            # Check if adding this section exceeds the chunk size
            if len(current_chunk) + len(section) > self.chunk_size and current_chunk:
                chunks.append(Document(page_content=current_chunk))
                current_chunk = section
            else:
                if current_chunk:
                    current_chunk += "\n\n" + section
                else:
                    current_chunk = section
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(Document(page_content=current_chunk))
        
        return chunks
    
    def _split_code(self, text: str, language: Optional[str] = None) -> List[Document]:
        """Split code by semantic units (functions, classes, etc.)"""
        if language is None:
            # Try to detect language from file extension or content
            if "def " in text and "class " in text:
                language = "python"
            elif "{" in text and "function" in text:
                language = "javascript"
            elif "{" in text and "public " in text and "class " in text:
                language = "java"
            elif "#include" in text and "{" in text:
                language = "cpp"
            else:
                language = "python"  # Default to Python
        
        # Use language-specific splitter
        try:
            splitter = RecursiveCharacterTextSplitter.from_language(
                language=Language(language),  # type: ignore
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            chunks = splitter.split_text(text)
            return [Document(page_content=chunk) for chunk in chunks]
        except:
            # Fallback to token splitter
            return self._split_by_paragraph(text)
    
    def split_text(self, text: str, metadata: Optional[Dict[str, Any]] = None, 
                  content_type: Optional[str] = None) -> List[Document]:
        """
        Split text using the configured chunking method
        
        Args:
            text: The text to split
            metadata: Metadata to attach to each chunk
            content_type: Type of content (e.g., "text", "code", "markdown")
        
        Returns:
            List of Document objects
        """
        # Choose splitting method based on content type or configured method
        if content_type == "code":
            chunks = self._split_code(text)
        elif content_type == "markdown" or self.chunking_method == "heading":
            chunks = self._split_by_headings(text)
        elif self.chunking_method == "sentence":
            chunks = self._split_by_sentence(text)
        elif self.chunking_method == "paragraph":
            chunks = self._split_by_paragraph(text)
        elif self.chunking_method == "semantic":
            chunks = self._split_by_semantic_units(text)
        else:
            # Default to splitting by tokens
            if self.chunk_by_tokens:
                chunks = [Document(page_content=chunk) for chunk in self.token_splitter.split_text(text)]
            else:
                chunks = [Document(page_content=chunk) for chunk in self.char_splitter.split_text(text)]
        
        # Add metadata to chunks
        if metadata:
            for chunk in chunks:
                chunk.metadata = metadata.copy()
                # Add chunk-specific metadata
                chunk.metadata["chunk_index"] = chunks.index(chunk)
                chunk.metadata["chunk_count"] = len(chunks)
        
        return chunks

# Specialized chunker for RAPTOR hierarchical chunking
class HierarchicalChunker:
    """
    Hierarchical chunking for RAPTOR indexing with parent-child relationships
    """
    
    def __init__(
        self,
        parent_chunk_size: int = 2000,
        child_chunk_size: int = 500,
        parent_overlap: int = 200,
        child_overlap: int = 50,
        chunk_by_tokens: bool = True
    ):
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.parent_overlap = parent_overlap
        self.child_overlap = child_overlap
        self.chunk_by_tokens = chunk_by_tokens
        
        # Create parent and child chunkers
        self.parent_chunker = SemanticChunker(
            chunk_size=parent_chunk_size,
            chunk_overlap=parent_overlap,
            chunking_method="paragraph",
            chunk_by_tokens=chunk_by_tokens
        )
        
        self.child_chunker = SemanticChunker(
            chunk_size=child_chunk_size,
            chunk_overlap=child_overlap,
            chunking_method="sentence",
            chunk_by_tokens=chunk_by_tokens
        )
    
    def split_text(self, text: str, metadata: Optional[Dict[str, Any]] = None,
                  content_type: Optional[str] = None) -> Dict[str, List[Document]]:
        """
        Split text into parent and child chunks with references
        
        Args:
            text: The text to split
            metadata: Metadata to attach to each chunk
            content_type: Type of content (e.g., "text", "code", "markdown")
            
        Returns:
            Dictionary with 'parents' and 'children' lists of documents
        """
        # Get parent chunks
        parent_chunks = self.parent_chunker.split_text(text, metadata, content_type)
        
        # Process each parent to get children
        all_children = []
        
        for i, parent in enumerate(parent_chunks):
            # Add parent-specific metadata
            parent.metadata = parent.metadata or {}
            parent.metadata["is_parent"] = True
            parent.metadata["parent_id"] = f"parent_{i}"
            parent.metadata["child_ids"] = []
            
            # Get child chunks from this parent
            child_chunks = self.child_chunker.split_text(
                parent.page_content,
                metadata=metadata.copy() if metadata else {},
                content_type=content_type
            )
            
            # Add child-specific metadata and references
            for j, child in enumerate(child_chunks):
                child_id = f"child_{i}_{j}"
                child.metadata["is_parent"] = False
                child.metadata["child_id"] = child_id
                child.metadata["parent_id"] = parent.metadata["parent_id"]
                
                # Add child ID to parent's child_ids list
                parent.metadata["child_ids"].append(child_id)
                
                all_children.append(child)
        
        return {
            "parents": parent_chunks,
            "children": all_children
        } 