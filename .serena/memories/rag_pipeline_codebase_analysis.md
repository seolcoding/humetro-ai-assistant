# RAG Pipeline Codebase - Comprehensive Analysis

## Executive Summary

The humetro-ai-assistant project implements a complete **6-stage RAG (Retrieval-Augmented Generation) pipeline** specifically optimized for Korean-language Seoul Metro information retrieval. The system integrates RAGAS evaluation framework for systematic benchmarking with support for GPT models and local Ollama-based LLMs.

---

## 1. Directory Structure & Organization

### Core RAG Pipeline
```
src/rag_pipeline/
├── stages/                           # 6-stage implementation
│   ├── stage_01_data_collection.py  # Document loading & cleaning
│   ├── stage_02_chunking.py         # Korean-optimized text splitting
│   ├── stage_03_embedding.py        # FAISS vector store creation
│   ├── stage_05_vector_store.py     # FAISS retrieval wrapper
│   ├── stage_06_retrieval.py        # LangChain RAG chain
│   └── stage_04_*.py                # Missing (likely removed)
│
├── base_rag.py                       # Abstract RAG base classes
├── testset_generator.py              # RAGAS testset with caching
├── generate_benchmark_50q.py         # 50-question benchmark generation
│
├── test_ragas_evaluation.py          # Basic evaluation pipeline
├── test_ragas_evaluation_v2.py       # Enhanced evaluation
├── test_ragas_compare_llms.py        # Multi-LLM comparison
├── test_ragas_compare_llms_litellm.py  # LiteLLM integration v1
├── test_ragas_compare_llms_litellm_v2.py  # LiteLLM integration v2
├── test_ragas_testgen_simple.py      # Simple testset generation
└── test_ragas_simple.py              # Minimal test flight
```

### Data & Evaluation
```
data/
├── crawled/
│   └── seoul_traffic/
│       └── markdown_deduplicated/    # Source markdown documents
├── vector_store/                     # FAISS index storage
├── evaluation/
│   ├── testsets/                     # Generated testsets (JSON, CSV, MD)
│   └── llm_comparison/               # Benchmark results
```

---

## 2. 6-Stage RAG Pipeline Architecture

### Stage 1: Data Collection
- **File**: `stages/stage_01_data_collection.py`
- **Purpose**: Load and clean markdown documents
- **Features**:
  - DirectoryLoader for bulk document loading
  - Markdown content extraction
  - Metadata preservation (source, path)
  - Document validation and filtering

### Stage 2: Chunking
- **File**: `stages/stage_02_chunking.py`
- **Class**: `ChunkingStage`
- **Configuration** (Korean-optimized):
  - **Chunk Size**: 512 tokens (~768 chars)
  - **Overlap**: 64 tokens (12.5%)
  - **Separators**: `["\\n\\n", "\\n", ". ", " ", ""]` (Korean-optimized)
- **Purpose**: Split documents for embedding
- **Features**:
  - TextChunker integration
  - Chunk metadata enrichment
  - Statistics calculation (chunks per doc, avg length)
  - Embedding cost estimation

### Stage 3: Embedding
- **File**: `stages/stage_03_embedding.py`
- **Model**: `text-embedding-3-small` (1536-dim vectors)
- **Purpose**: Create vector representations
- **Framework**: FAISS (Facebook AI Similarity Search)

### Stage 5: Vector Store
- **File**: `stages/stage_05_vector_store.py`
- **Class**: `VectorStoreStage`
- **Features**:
  - Load FAISS indices from disk
  - **Similarity Search**: k-NN search (configurable k)
  - **MMR Search**: Maximum Marginal Relevance for diversity
  - **Score-based Search**: Return (document, score) tuples
  - **Retriever Integration**: LangChain-compatible retriever creation
  - **Cosine Similarity**: Normalized vectors for L2 distance equivalence
- **Methods**:
  - `load_vector_store(store_path)` - Load FAISS index
  - `similarity_search(query, k=4)` - Standard retrieval
  - `similarity_search_with_score(query, k=4)` - With relevance scores
  - `max_marginal_relevance_search(...)` - Diverse retrieval
  - `as_retriever(**kwargs)` - Create LangChain retriever
  - `get_store_info()` - Retrieve metadata

### Stage 6: Retrieval
- **File**: `stages/stage_06_retrieval.py`
- **Class**: `RetrievalStage`
- **LLM Model**: `gpt-4o-mini` (temperature=0.0 for determinism)
- **Purpose**: Generate answers from retrieved context
- **RAG Chain**:
  ```
  Retriever → Document Formatting → Prompt Template → LLM → String Output
  ```
- **Default Prompt**: Korean-optimized system prompt with guidelines
  - "당신은 서울시 교통 정보 전문 상담사입니다"
  - Emphasizes faithfulness to provided documents
  - Handles "cannot answer" scenarios
- **Methods**:
  - `query(question)` - Simple query
  - `query_with_sources(question)` - Query with source documents
  - `batch_query(questions)` - Batch processing
  - `get_chain_info()` - Configuration metadata

### Base RAG Implementation
- **File**: `base_rag.py`
- **Classes**:
  - `RAGQuery`: Input dataclass
  - `RAGResponse`: Output dataclass
  - `BaseRAG`: Abstract base with retrieve/generate/rerank/call
  - `BaselineRAG`: Simple RAG (retrieve + generate)
  - `NaiveRAG`: Basic implementation
  - `AdvancedRAG`: Enhanced with reranking
  - `GraphRAG`: Knowledge graph-based (stub)
- **Execution Flow**:
  1. Time measurement start
  2. Document retrieval
  3. Optional reranking
  4. Answer generation
  5. Latency calculation
  6. Returns structured response with contexts & scores

---

## 3. TestSet Generation System

### CachedTestsetGenerator
- **File**: `testset_generator.py`
- **Purpose**: Generate evaluation datasets with reproducibility
- **Key Classes**:
  - `TestsetConfig`: Dataclass for reproducible configuration
  - `CachedTestsetGenerator`: Main generator with caching

### Configuration & Reproducibility
```python
TestsetConfig:
  - llm_model: str (e.g., "gpt-4o-mini")
  - llm_temperature: float (0.3 default)
  - embedding_model: str (e.g., "text-embedding-3-small")
  - num_documents: int
  - document_source: str
  - testset_size: int
  - language: str ("korean")
  - created_at: ISO timestamp (auto)
  - ragas_version: str (auto)
```

### Cache Key Generation
- SHA256 hash of config (excludes created_at, ragas_version)
- 16-char hex string
- Enables reproducible testset reuse

### Caching Strategy
- **Location**: `data/evaluation/testsets/`
- **Files Generated**:
  - `testset_{cache_key}.json` - Complete metadata + testset
  - `testset_{cache_key}.csv` - Tabular format
  - `testset_{cache_key}.md` - Generation documentation
  - `metadata.json` - Version tracking with flags:
    - `is_latest`: Current active version
    - `is_benchmark`: Official benchmark version
    - `description`: Custom metadata

### Korean Persona System
```python
_create_korean_personas():
  1. "서울 시민" (Seoul Resident)
  2. "외국인 거주자" (Foreign Resident)
  3. "교통 관심 시민" (Transit Enthusiast)
```

### RAGAS Integration
- **LLM Wrapper**: LangchainLLMWrapper for GPT models
- **Embeddings Wrapper**: LangchainEmbeddingsWrapper
- **Generator Method**: `generate_with_langchain_docs()`
- **Output**: pandas DataFrame with columns:
  - `user_input` - Question
  - `reference` - Ground truth answer
  - `reference_contexts` - Supporting context list

### Benchmark Generation
- **Script**: `generate_benchmark_50q.py`
- **Configuration**:
  - 50-question benchmark target
  - 50 source documents
  - Korean-only personas
  - GPT-4o-mini as generator
  - Marked as both `is_latest=True` and `is_benchmark=True`
- **Usage**:
  ```bash
  python src/rag_pipeline/generate_benchmark_50q.py
  python src/rag_pipeline/generate_benchmark_50q.py --force  # Ignore cache
  ```

---

## 4. Evaluation Framework (RAGAS)

### RAGAS Metrics
**Available Metrics**:
1. **Faithfulness**: Answers grounded in retrieved context (≤ environment context facts)
2. **Answer Relevancy**: Relevance to user question
3. **Answer Correctness**: F1 score vs ground truth
4. **Context Precision**: Retrieval quality (proportion relevant)
5. **Context Recall**: Retrieval completeness

### Evaluation Pipeline Pattern
```
load_documents()
  ↓
create_korean_testset()
  ↓
retrieve_contexts(question)
  ↓
generate_answer(question, contexts)
  ↓
run_ragas_metrics(dataset)
```

### Known Issue: GPT-5 Temperature
- **Problem**: GPT-5 only supports `temperature=1` (default)
- **RAGAS Default**: Sets `temperature=1e-8` internally
- **Solution**: Patch `.venv/lib/python3.12/site-packages/ragas/llms/base.py`
  ```python
  def get_temperature(self, n: int) -> float:
      return None  # Force None for GPT-5 compatibility
  ```
- **Workaround**: Use GPT-4o as judge (supports temperature parameter)

### LiteLLM Integration
- **File**: `test_ragas_compare_llms_litellm_v2.py`
- **Purpose**: Unified interface for OpenAI + Ollama models
- **Configuration**:
  ```python
  litellm.set_verbose = False
  litellm.drop_params = True  # Auto-remove unsupported params
  ```
- **Models**:
  - `gpt-4o-mini` (OpenAI)
  - `ollama/exaone3.5:7.8b` (Korean-specialized)
  - `ollama/qwen3:8b` (Complex reasoning)
  - `ollama/gemma3:12b` (General fallback)
  - `ollama/gpt-oss:20b` (Open-source GPT)

### Evaluation Pattern
1. **Connection Testing**: `test_model_connection(model_config)`
2. **Context Retrieval**: Get k=4 documents per question
3. **Answer Generation**: LLM-specific generation via LiteLLM
4. **Metrics Calculation**: RAGAS evaluate() with selected metrics
5. **Results Comparison**: Compare across all available models
6. **Results Saving**: JSON with metrics summary per model

---

## 5. Git History & Recent Changes

### Recent Commits (Latest First)
```
b4f4971 feat: add testset metadata management system with version tracking
a58608e feat: add RAGAS testset generation system with caching and Korean personas
27432b1 feat: implement Stage 5 (Vector Store) & Stage 6 (Retrieval) - complete RAG system
df362a8 feat: implement Stage 3 (Embedding) with FAISS and reorganize scripts
fca7969 feat: implement Stage 2 (Chunking) + markdown cleaning for all files
8340128 feat: implement Stage 1 (Data Collection) for RAG pipeline
```

### Current Branch
- **Feature Branch**: `feature/rag-pipeline-refactor`
- **Main Branch**: `main` (PR target)
- **Unstaged Changes**:
  - `src/rag_pipeline/stages/stage_03_embedding.py` (M)
  - `src/rag_pipeline/stages/stage_05_vector_store.py` (M)
  - `scripts/` (new directory)
  - Thesis documentation files

---

## 6. Module Reusability Patterns

### Stage Pattern
Each stage follows consistent pattern:
```python
class SomeStage:
    def __init__(self, model: str = "...", logger: Optional[RAGLogger] = None):
        self.model = model
        self.logger = logger
    
    def process(self, data):
        # Main processing
        if self.logger:
            self.logger.info(f"Processing...")
        return result
    
    def get_info(self) -> dict:
        return {"config": "value"}

def create_some_stage(...) -> SomeStage:
    """Factory function"""
    return SomeStage(...)
```

### Logger Integration
- **Module**: `src/common.logger` (RAGLogger)
- **Usage**: Optional throughout, graceful degradation
- **Pattern**: `if self.logger: self.logger.info(...)`

### Configuration Management
- **LLM Models**: Constructor params with sensible defaults
- **Paths**: Use `Path` objects, support relative paths
- **Flexibility**: All stages customizable but have solid defaults

### Document Processing
- **Container**: `langchain_core.documents.Document`
- **Metadata**: Preserved through pipeline stages
- **Enrichment**: Each stage adds metadata (chunk_id, embedding_model, etc.)

---

## 7. Key Implementation Details

### Chunking Configuration (Optimal for 8B Models)
- 512 tokens chosen because:
  - Fits within embedding model limits (text-embedding-3-small)
  - Appropriate for 8B parameter models
  - Korean text: ~1.5 chars per token
  - ~768 character chunks

### Embedding Model Choice
- **text-embedding-3-small**:
  - 1536 dimensions
  - Cost-effective
  - Strong performance
  - Multi-lingual support

### Vector Store (FAISS)
- **Index Type**: Inner Product (for cosine similarity)
- **Normalization**: L2 normalization enabled
- **Retrieval**: Supports k-NN, MMR, scored search
- **Integration**: Native LangChain retriever support

### RAG Chain (LangChain)
- **Components**:
  1. `RunnablePassthrough()` for question
  2. Document formatter function
  3. ChatPromptTemplate with context injection
  4. ChatOpenAI LLM
  5. StrOutputParser for output
- **Composability**: Uses LCEL (LangChain Expression Language)

### Korean Persona-Driven Generation
- Purpose: Generate realistic Korean-language questions
- Personas represent different user types
- Each persona generates different question styles
- Improves testset diversity and realism

---

## 8. Data Flow & Integration Points

### Complete End-to-End Flow
```
1. Raw Markdown Documents
   ↓ [Stage 1: Data Collection]
2. Cleaned Documents with Metadata
   ↓ [Stage 2: Chunking]
3. Chunked Documents (512 tokens, 64 overlap)
   ↓ [Stage 3: Embedding]
4. Vector Embeddings (1536-dim)
   ↓ [FAISS Index Creation]
5. Vector Store (FAISS with L2 normalization)
   ↓ [Stage 5: Load & Setup]
6. Retriever (LangChain compatible)
   ↓ [Stage 6: RAG Chain]
7. Question → Context → LLM → Answer
   ↓ [Evaluation]
8. RAGAS Metrics → Benchmark Results
```

### Testset Generation Integration
```
Source Documents (Markdown)
   ↓
CachedTestsetGenerator.generate_or_load()
   ├─ Load from cache if exists
   ├─ Create Korean Personas
   ├─ RAGAS TestsetGenerator
   └─ Generate {user_input, reference, reference_contexts}
   ↓
Evaluation Dataset (pandas DataFrame)
   ↓
RAGAS evaluate() with metrics
   ↓
Benchmark Results (JSON, CSV)
```

---

## 9. Production Considerations

### Error Handling
- Most stages have try/except with logging
- Validation of required inputs
- Graceful degradation (optional logger)

### Performance
- Batch operations supported (batch_query)
- Caching for testsets reduces API calls
- FAISS provides O(log n) retrieval

### Reproducibility
- Config-based cache keys (deterministic)
- Metadata versioning
- Full parameter logging

### Scalability
- Stage architecture allows parallel processing
- Modular design enables custom implementations
- LangChain abstractions support multiple backends

---

## 10. Current Gaps & Experimental Opportunities

### Missing Stage
- **Stage 4**: Likely skipped or integrated into Stage 3/5
- Could be for reranking, quality filtering, or metadata enhancement

### Unexplored Extensions
1. **Multi-hop Question Synthesis**: Current system generates single/multi-hop
2. **Contextual Question Difficulty**: Could grade by reasoning complexity
3. **Domain-Specific Metrics**: Beyond faithfulness/relevancy
4. **A/B Testing Framework**: Systematic model comparison beyond RAGAS
5. **Few-Shot Learning**: Leverage exemplars in prompts
6. **Chain-of-Thought**: Explicit reasoning step decomposition

### Data Augmentation Opportunities
1. Persona expansion (currently 3 personas)
2. Language variation (currently Korean-only)
3. Question paraphrase generation
4. Negative example generation (for factual correctness)

---

## Summary Table

| Component | Technology | Configuration | Purpose |
|-----------|------------|---------------|---------|
| Data Loading | LangChain DirectoryLoader | Markdown files | Stage 1 |
| Chunking | TextChunker | 512 tokens, 64 overlap | Stage 2 |
| Embedding | OpenAI text-embedding-3-small | 1536-dim | Stage 3 |
| Vector Store | FAISS | IP distance, L2 norm | Stage 5 |
| Retrieval | LangChain VectorStoreRetriever | k=4 default | Stage 5 |
| Generation | LangChain RAG Chain | GPT-4o-mini, T=0 | Stage 6 |
| Evaluation | RAGAS Framework | 5+ metrics | Assessment |
| Testset Gen | RAGAS TestsetGenerator | Korean personas | Benchmarking |
| LLM Support | LiteLLM | OpenAI + Ollama | Multi-model |
| Caching | JSON + CSV | SHA256 config key | Reproducibility |