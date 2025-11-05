# Deprecated Code and Patterns

## Deprecated as of 2025-11-05 - Unified Benchmark System

### Deprecated Test Scripts ⚠️
All individual test scripts have been consolidated into `unified_benchmark.py`:

**DEPRECATED FILES:**
- ❌ `generate_100q_benchmark.py` → Use `unified_benchmark.py --questions 100`
- ❌ `generate_100q_benchmark_v2.py` → Use `unified_benchmark.py`
- ❌ `generate_benchmark_50q.py` → Use `unified_benchmark.py --questions 50`
- ❌ `test_gpt4o_mini_only.py` → Use `unified_benchmark.py --models gpt-4o-mini`

**Migration Guide:**
```bash
# Old way (DEPRECATED)
python generate_100q_benchmark.py --models gpt-4o-mini ollama/exaone3.5:7.8b

# New way (UNIFIED)
python unified_benchmark.py --questions 100 --models gpt-4o-mini ollama/exaone3.5:7.8b

# Old way (DEPRECATED)
python test_gpt4o_mini_only.py

# New way (UNIFIED)
python unified_benchmark.py --models gpt-4o-mini --questions 5
```

**Key Features of Unified System:**
- ✅ CLI interface for all parameters
- ✅ Retrieval validation built-in
- ✅ Model groups (all, openai, ollama, korean)
- ✅ Cache management
- ✅ GPT-5 as default judge model
- ✅ Comprehensive error handling

## Previous Deprecations (2025-11-05)

### 1. Direct RetrievalStage Instantiation for Retrieval
**Status**: ❌ DEPRECATED
**Location**: Previously in `generation_benchmark.py`
**Reason**: RetrievalStage is for RAG chain assembly, not direct retrieval

#### Old Pattern (WRONG):
```python
from stages.stage_06_retrieval import RetrievalStage

# ❌ DEPRECATED - This doesn't work
self.retriever = RetrievalStage(
    model="gpt-4o-mini",
    k=self.k_documents  # TypeError: 'k' doesn't exist
)
```

#### New Pattern (CORRECT):
```python
from stages.stage_05_vector_store import VectorStoreStage

# ✅ Use LangChain retriever pattern
vector_store = VectorStoreStage(model="text-embedding-3-small")
vector_store.load_vector_store(path)
self.retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)
```

### 2. Direct FAISS Index Manipulation
**Status**: ❌ DEPRECATED
**Reason**: Violates encapsulation, bypasses public API

#### Old Pattern (WRONG):
```python
import faiss
# ❌ DEPRECATED - Direct internal access
self.vector_store.index = faiss.read_index(path)
```

#### New Pattern (CORRECT):
```python
# ✅ Use public API
vector_store.load_vector_store(path)
```

### 3. Silent Failure Pattern
**Status**: ❌ DEPRECATED
**Reason**: Hides critical errors, leads to invalid benchmarks

#### Old Pattern (WRONG):
```python
try:
    # initialization code
except Exception as e:
    logger.error(f"Failed: {e}")
    # ❌ DEPRECATED - Silent continuation
```

#### New Pattern (CORRECT):
```python
try:
    # initialization code
except Exception as e:
    logger.error(f"Failed: {e}")
    # ✅ Fail fast
    raise RuntimeError(f"Failed to initialize: {e}") from e
```

## Test Files Moved

All test files have been relocated from `src/rag_pipeline/` to `tests/rag_pipeline/`:

- `test_ragas_*.py` → `tests/rag_pipeline/`
- `test_retriever_init.py` → `tests/rag_pipeline/test_retriever_initialization.py`
- `test_benchmark_fix.py` → `tests/rag_pipeline/test_benchmark_retrieval.py`

## Notes

- Always follow LangChain patterns for RAG components
- Use public APIs provided by stage classes
- Implement fail-fast error handling for critical operations
- Keep tests in dedicated `tests/` directory structure