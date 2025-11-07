# RAG Pipeline

## Overview

Multi-stage RAG (Retrieval-Augmented Generation) pipeline for Seoul traffic information Q&A system.

## Architecture

```
Stage 1: Data Collection → Stage 2: Preprocessing → Stage 3: Embedding
→ Stage 4: Graph Construction → Stage 5: Vector Store → Stage 6: Retrieval
```

## Components

### Core Stages
- **Stage 1**: Web crawling with crawl4ai
- **Stage 2**: Markdown preprocessing and chunking
- **Stage 3**: OpenAI text-embedding-3-small embeddings
- **Stage 4**: Knowledge graph construction (in development)
- **Stage 5**: FAISS vector store management
- **Stage 6**: LangChain-based retrieval and QA

### Evaluation
- **Generation Benchmark**: Compare LLM generation quality with fixed contexts
- **RAGAS Integration**: Automatic evaluation with faithfulness, relevancy, correctness metrics
- **Testset Generation**: Automated question generation for benchmarking

## Recent Updates (2025-11-05)

### Fixed Critical Retriever Bug
- **Issue**: Benchmark was running without actual retrieval (empty contexts)
- **Root Cause**: Incorrect `RetrievalStage` instantiation with wrong parameters
- **Solution**: Implemented proper LangChain retriever pattern
- **Impact**: Benchmark results before this date may be invalid

### Correct Usage Pattern

```python
# ✅ CORRECT: LangChain retriever pattern
from stages.stage_05_vector_store import VectorStoreStage

# 1. Initialize vector store
vector_store = VectorStoreStage(model="text-embedding-3-small")

# 2. Load FAISS index using public API
vector_store.load_vector_store("data/vector_store/seoul_traffic")

# 3. Create LangChain retriever
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# 4. Use retriever.invoke() for document retrieval
documents = retriever.invoke("서울시 대중교통 요금은?")
```

## Directory Structure

```
src/rag_pipeline/
├── stages/                  # Pipeline stages 1-6
│   ├── stage_01_crawl.py
│   ├── stage_02_preprocess.py
│   ├── stage_03_embedding.py
│   ├── stage_04_graph.py
│   ├── stage_05_vector_store.py
│   └── stage_06_retrieval.py
├── generation_benchmark.py  # Fixed benchmark module
├── testset_generator.py     # RAGAS testset generation
├── question_classifier.py   # Question complexity analysis
├── DEPRECATED.md            # Deprecated patterns documentation
└── README.md               # This file

tests/rag_pipeline/         # All tests moved here
├── test_retriever_initialization.py
├── test_benchmark_retrieval.py
└── test_ragas_*.py
```

## Testing

Run tests to verify retriever functionality:

```bash
# Test retriever initialization
uv run python tests/rag_pipeline/test_retriever_initialization.py

# Test benchmark with retrieval
uv run python tests/rag_pipeline/test_benchmark_retrieval.py
```

## Vector Store Info

- **Location**: `data/vector_store/seoul_traffic/`
- **Embeddings**: OpenAI text-embedding-3-small
- **Vectors**: 13,176 document chunks
- **Index Type**: FAISS Flat index

## Dependencies

- LangChain for RAG orchestration
- FAISS for vector search
- RAGAS for evaluation metrics
- LiteLLM for multi-model support
- OpenAI API for embeddings and generation

## Known Issues

See `DEPRECATED.md` for anti-patterns to avoid and historical bugs.