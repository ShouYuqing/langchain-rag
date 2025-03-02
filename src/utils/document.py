"""
Document processing utilities
"""
import os
import re
import json
import hashlib
from typing import Dict, List, Optional, Union, Any
import datetime
import tiktoken
import docx2txt
import fitz  # PyMuPDF
from bs4 import BeautifulSoup

# Get token encoding for OpenAI models
def get_tokenizer(model_name: str = "gpt-4"):
    """Get tokenizer for OpenAI models"""
    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")  # Default for gpt-4, gpt-3.5-turbo
    return encoding

def count_tokens(text: str, model_name: str = "gpt-4") -> int:
    """Count tokens in a text string"""
    tokenizer = get_tokenizer(model_name)
    tokens = tokenizer.encode(text)
    return len(tokens)

def extract_text_from_file(file_path: str) -> str:
    """Extract text from a file based on its extension"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    # Text-based files
    if file_extension in ['.txt', '.md', '.py', '.js', '.html', '.css', '.json', '.csv']:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            return file.read()
    
    # PDF files
    elif file_extension == '.pdf':
        text = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                text += page.get_text()
        return text
    
    # Word documents
    elif file_extension in ['.docx', '.doc']:
        return docx2txt.process(file_path)
    
    # HTML files (with advanced parsing)
    elif file_extension in ['.html', '.htm']:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
            soup = BeautifulSoup(content, 'lxml')
            # Remove scripts, styles, and other non-content elements
            for element in soup(['script', 'style', 'meta', 'noscript', 'header', 'footer']):
                element.decompose()
            return soup.get_text(separator=' ', strip=True)
    
    # Return empty string for unsupported files
    else:
        print(f"Unsupported file format: {file_extension}")
        return ""

def generate_document_id(content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Generate a unique document ID based on content and metadata"""
    metadata_str = json.dumps(metadata) if metadata else ""
    content_hash = hashlib.md5((content + metadata_str).encode()).hexdigest()
    return content_hash

def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace, newlines, etc."""
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove very common non-informative header/footer patterns
    text = re.sub(r'page \d+ of \d+', '', text, flags=re.IGNORECASE)
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    return text.strip()

def extract_metadata_from_text(text: str) -> Dict[str, Any]:
    """Extract metadata from text content when possible"""
    metadata = {
        "extracted_date": datetime.datetime.now().isoformat(),
    }
    
    # Try to extract title (first non-empty line or heading)
    lines = text.split('\n')
    for line in lines:
        if line.strip():
            metadata["title"] = line.strip()[:100]  # Limit title length
            break
    
    # Extract potential dates
    date_patterns = [
        r'\d{1,2}/\d{1,2}/\d{2,4}',  # MM/DD/YYYY or DD/MM/YYYY
        r'\d{4}-\d{2}-\d{2}',        # YYYY-MM-DD
        r'[A-Z][a-z]+ \d{1,2},? \d{4}'  # Month DD, YYYY
    ]
    
    for pattern in date_patterns:
        dates = re.findall(pattern, text)
        if dates:
            metadata["detected_date"] = dates[0]
            break
    
    return metadata 