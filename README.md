# LangChain RAG System

A comprehensive and modular Retrieval-Augmented Generation (RAG) system built with LangChain. This system enhances AI-generated responses by leveraging external knowledge sources, providing more accurate, up-to-date, and verifiable information.

## Features

- **Advanced Retrieval**: Query expansion, hypothetical document embeddings, and multi-query retrieval
- **Context Processing**: Deduplication, relevance filtering, and context compression
- **Context Augmentation**: Summaries, definitions, and entity information
- **Reranking**: Multiple reranking methods including semantic similarity and relevance
- **Query Routing**: Domain-specific handling based on query content
- **Flexible Configuration**: YAML-based or programmatic configuration
- **Streaming Support**: Real-time token streaming for responsive UIs
- **Asynchronous API**: Non-blocking operations for high-performance applications
- **Web UI**: Interactive interface for querying and managing the RAG system

## Architecture

The system follows a modular architecture with the following components:

```
config/
└── config.py                # Configuration templates and defaults
examples/                    # Example implementations
src/
├── rag_system.py            # Main RAG system implementation
├── config.py                # Configuration system
├── context_processing/      # Context processing modules
├── generation/              # Generation modules
├── retrieval/               # Retrieval modules
├── routing/                 # Query routing modules
├── indexing/                # Document indexing modules
└── utils/                   # Utility functions
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/langchain-rag.git
cd langchain-rag
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

```python
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from src.rag_system import RAGSystem
from src.config import create_default_config

# Set up environment variables
os.environ["OPENAI_API_KEY"] = "your-api-key"

# Create a vector store
embeddings = OpenAIEmbeddings()
vector_store = Chroma(persist_directory="data/vector_store", embedding_function=embeddings)

# Initialize the RAG system
config = create_default_config()
rag_system = RAGSystem(config=config, vector_store=vector_store)

# Process a query
result = rag_system.query("What is retrieval-augmented generation?")
print(result.answer)
```

## Using the Web UI

The system includes a web-based user interface for easy interaction:

1. Start the API server with UI:
```bash
./run_api_server.py
```

2. Open your browser and navigate to:
```
http://localhost:8000
```

3. Use the interface to:
   - Submit queries and view responses
   - Index documents for retrieval
   - View and modify configurations

## Using Configurations

### YAML Configuration

```python
from src.config import RAGConfig

# Load configuration from YAML
config = RAGConfig.from_yaml("configs/production.yaml")

# Initialize RAG system with the configuration
rag_system = RAGSystem(config=config, vector_store=vector_store)
```

### Programmatic Configuration

```python
from src.config import create_production_config, create_lightweight_config

# Use a predefined configuration
config = create_production_config()  # or create_lightweight_config()

# Initialize RAG system with the configuration
rag_system = RAGSystem(config=config, vector_store=vector_store)
```

## Examples

The `examples/` directory contains scripts demonstrating various aspects of the system:

- `examples/rag_system_usage.py`: Simple example of using the RAG system
- `examples/rag_with_config.py`: Using different configurations
- `examples/api_server.py`: REST API for the RAG system
- `examples/api_server_with_ui.py`: REST API with web UI and interactive interface

Each example demonstrates different capabilities of the RAG system, from basic usage to advanced configurations and deployments.

## Deploying Chroma with Docker Compose

For improved scalability and persistence, the repository includes Docker Compose configuration for deploying a Chroma vector database:

```bash
# Navigate to the Docker directory
cd docker/chroma

# Generate secure credentials
./generate_credentials.py

# Start the Chroma database
docker-compose up -d
```

Connect to the deployed database using the example script:

```python
# Update token in examples/chroma_docker_connection.py
python examples/chroma_docker_connection.py
```

For detailed instructions, see [docker/chroma/README.md](docker/chroma/README.md).

## Advanced Usage

### Streaming Responses

```python
async for token in rag_system.astream("What is retrieval-augmented generation?"):
    print(token, end="", flush=True)
```

### Using the Advanced Retriever Directly

```python
from src.retrieval.advanced_retriever import AdvancedRetriever

retriever = AdvancedRetriever(
    base_retriever=vector_store.as_retriever(),
    use_query_expansion=True,
    use_hypothetical_document=True
)

documents = retriever.get_relevant_documents("What is RAG?")
```

### Custom Reranking

```python
from src.context_processing.reranker import Reranker

reranker = Reranker(
    semantic_similarity_weight=0.7,
    relevance_weight=0.3
)

reranked_docs = reranker.rerank_documents(documents, query="What is RAG?")
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [LangChain](https://github.com/langchain-ai/langchain) for the core framework
- [OpenAI](https://openai.com/) for the language models 

## Related Research

This project builds upon foundational research in the field of retrieval-augmented generation and large language models. The following papers provide background and context for the techniques implemented in this system:

- Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., ... & Kiela, D. (2020). [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401). *Advances in Neural Information Processing Systems*.
  - The original RAG paper that introduced the concept of combining retrieval with generation.

- Gao, L., Ma, X., Lin, J., & Callan, J. (2022). [Precise Zero-Shot Dense Retrieval without Relevance Labels](https://arxiv.org/abs/2212.10496). *arXiv preprint*.
  - Research on hypothetical document embeddings used in our advanced retrieval module.

- Pradeep, R., Sundarrajan, A., & Lin, J. (2023). [Prompting Is Programming: A Query Language for Large Language Models](https://arxiv.org/abs/2212.06094). *arXiv preprint*.
  - Techniques for effective prompt engineering implemented in our generation module.

- Khattab, O., Santhanam, K., Li, X., Hall, D., Liang, P., Potts, C., & Zaharia, M. (2022). [Demonstrate-Search-Predict: Composing Retrieval and Language Models for Knowledge-Intensive NLP](https://arxiv.org/abs/2212.14024). *arXiv preprint*.
  - Framework for integrating retrieval with LLMs that influenced our system architecture.

- Wang, S., Liu, Y., Xu, Y., Zhu, C., & Zeng, M. (2023). [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511). *arXiv preprint*.
  - Advanced RAG techniques incorporating self-reflection for improved output quality. 