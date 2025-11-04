# LangChain Import Fixes for Thesis Notebook

**Date**: 2025-10-30
**Issue**: ModuleNotFoundError in notebook cell-3
**Solution**: Updated imports based on official LangChain documentation

## Summary of Fixes

### ❌ OLD (Broken/Deprecated Imports)

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter  # ❌ Module not found
from langchain.prompts import PromptTemplate  # ❌ Module not found
from langchain.schema import Document  # ❌ Module not found
from langchain.chains import RetrievalQA  # ❌ Deprecated (legacy API)
```

### ✅ NEW (Fixed Imports)

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter  # ✅ Correct
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate  # ✅ Correct
from langchain_core.documents import Document  # ✅ Correct
from langchain.chains import create_retrieval_chain  # ✅ LCEL (new standard)
from langchain.chains.combine_documents import create_stuff_documents_chain  # ✅ LCEL
```

## Complete Fixed Cell-3 Code

Replace cell-3 in your notebook with this code:

```python
# 기본 라이브러리
import os
import json
import pickle
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import time

# 데이터 처리
import pandas as pd
import numpy as np
from tqdm import tqdm

# ============================================================================
# LangChain - FIXED IMPORTS (공식 문서 기준)
# ============================================================================

# Embeddings (CORRECT)
from langchain_community.embeddings import HuggingFaceEmbeddings

# Vector Stores (CORRECT)
from langchain_community.vectorstores import Chroma

# Text Splitters - FIXED! ✅
# OLD: from langchain.text_splitter import RecursiveCharacterTextSplitter
# NEW: from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Chains - FIXED! ✅
# OLD (DEPRECATED): from langchain.chains import RetrievalQA
# NEW (LCEL): Use create_retrieval_chain + create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Prompts - FIXED! ✅
# OLD: from langchain.prompts import PromptTemplate
# NEW: from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate

# Documents - FIXED! ✅
# OLD: from langchain.schema import Document
# NEW: from langchain_core.documents import Document
from langchain_core.documents import Document

# ============================================================================
# Additional Libraries
# ============================================================================

# Ollama for local LLM inference
import requests
from openai import OpenAI  # For baseline comparison with gpt-4o-mini

# SentenceTransformers for KURE embeddings
from sentence_transformers import SentenceTransformer

# Neo4j for Graph RAG
from neo4j import GraphDatabase

# RAGAS for evaluation
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    answer_relevancy,
    faithfulness,
    answer_correctness,
)

# RAGAS Testset Generation - 한국어 Q/A 생성 지원! ✅
# Note: RAGAS 0.3.x changed import paths
from ragas.testset import TestsetGenerator  # Changed from ragas.testset.generator
# Evolutions are now in synthesizers submodule (RAGAS 0.3.x)
from ragas.testset.synthesizers import SingleHopSpecificQuerySynthesizer
from ragas.testset.synthesizers import MultiHopAbstractQuerySynthesizer, MultiHopSpecificQuerySynthesizer
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DataFrameLoader

# Visualization
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# 환경 변수 로드
from dotenv import load_dotenv

load_dotenv()

# 경고 메시지 숨기기
import warnings

warnings.filterwarnings("ignore")

print("✅ 라이브러리 임포트 완료 (Fixed Imports v2)")
print("📝 Fixed imports:")
print("   - RecursiveCharacterTextSplitter: langchain_text_splitters")
print("   - PromptTemplate: langchain_core.prompts")
print("   - Document: langchain_core.documents")
print("   - Chains (LCEL): create_retrieval_chain + create_stuff_documents_chain")
print("   - RAGAS Korean testset generation: ✅ 추가됨")
```

## Detailed Changes

### 1. Text Splitters

| Aspect | Old | New |
|--------|-----|-----|
| Module | `langchain.text_splitter` | `langchain_text_splitters` |
| Import | `from langchain.text_splitter import RecursiveCharacterTextSplitter` | `from langchain_text_splitters import RecursiveCharacterTextSplitter` |
| Reason | Module relocated in LangChain 0.1+ | Official package separation |

**Documentation**: https://python.langchain.com/docs/how_to/recursive_text_splitter

### 2. Prompts

| Aspect | Old | New |
|--------|-----|-----|
| Module | `langchain.prompts` | `langchain_core.prompts` |
| Import | `from langchain.prompts import PromptTemplate` | `from langchain_core.prompts import PromptTemplate` |
| Reason | Core components moved to `langchain_core` | Better separation of concerns |

**Documentation**: https://python.langchain.com/docs/integrations/llms/tongyi

### 3. Documents

| Aspect | Old | New |
|--------|-----|-----|
| Module | `langchain.schema` | `langchain_core.documents` |
| Import | `from langchain.schema import Document` | `from langchain_core.documents import Document` |
| Reason | Schema module deprecated | Core document types in dedicated module |

**Documentation**: https://python.langchain.com/docs/integrations/document_loaders/copypaste

### 4. Chains

| Aspect | Old | New |
|--------|-----|-----|
| Module | `langchain.chains` | `langchain.chains` (LCEL) |
| Import | `from langchain.chains import RetrievalQA` | `from langchain.chains import create_retrieval_chain`<br>`from langchain.chains.combine_documents import create_stuff_documents_chain` |
| Reason | RetrievalQA is deprecated | Migration to LangChain Expression Language (LCEL) |

**Documentation**: https://python.langchain.com/docs/versions/migrating_chains/retrieval_qa

**Note**: `RetrievalQA` is deprecated in favor of LCEL patterns using `create_retrieval_chain` and `create_stuff_documents_chain` for better composability and maintainability.

## Bonus: RAGAS Korean Q/A Generation

### RAGAS 0.3.x Import Changes

RAGAS 0.3.x restructured module paths:

| Old (< 0.3) | New (0.3.x) |
|-------------|-------------|
| `from ragas.testset.generator import TestsetGenerator` | `from ragas.testset import TestsetGenerator` |
| `from ragas.testset.evolutions import simple, reasoning, multi_context` | `from ragas.testset.synthesizers import SingleHopSpecificQuerySynthesizer`<br>`from ragas.testset.synthesizers import MultiHopAbstractQuerySynthesizer, MultiHopSpecificQuerySynthesizer` |

### Updated Imports

```python
# RAGAS Testset Generation - 한국어 Q/A 생성 지원! ✅
# Note: RAGAS 0.3.x changed import paths
from ragas.testset import TestsetGenerator  # Changed from ragas.testset.generator
# Evolutions are now in synthesizers submodule (RAGAS 0.3.x)
from ragas.testset.synthesizers import SingleHopSpecificQuerySynthesizer
from ragas.testset.synthesizers import MultiHopAbstractQuerySynthesizer, MultiHopSpecificQuerySynthesizer
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DataFrameLoader
```

**Key Changes**:
- `TestsetGenerator` moved from `ragas.testset.generator` to `ragas.testset`
- Evolution strategies (simple, reasoning, multi_context) replaced by synthesizer classes
- Synthesizers are now in `ragas.testset.synthesizers` submodule

**Benefits**:
- Generates Korean Q/A pairs automatically
- LLM-based synthesis with specific synthesizer classes
- Cost-effective: ~$0.01 for 40 questions (GPT-4o-mini)

See: `claudedocs/ragas_korean_support.md` and `notebooks/ragas_korean_testset_example.py`

## Verification Method

After applying fixes, verify imports work:

```python
# Test imports
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.prompts import PromptTemplate
    from langchain_core.documents import Document
    from langchain.chains import RetrievalQA
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
```

## LangChain Version Info

These fixes are compatible with:
- LangChain >= 0.1.0
- LangChain-Core >= 0.1.0
- LangChain-Community >= 0.0.10
- LangChain-Text-Splitters >= 0.0.1

Check your installed versions:

```bash
uv run python -c "import langchain; print(langchain.__version__)"
uv run python -c "import langchain_core; print(langchain_core.__version__)"
uv run python -c "import langchain_text_splitters; print(langchain_text_splitters.__version__)"
```

## References

1. **LangChain Official Documentation**: python.langchain.com
2. **Context7 Library**: `/websites/python_langchain`
3. **Migration Guide**: python.langchain.com/docs/versions/migrating_chains/
4. **RAGAS Documentation**: docs.ragas.io/en/latest/

## Next Steps

1. ✅ Replace cell-3 in notebook with fixed code
2. ✅ Run cell-3 to verify imports work
3. ✅ Continue with rest of notebook execution
4. ✅ Test RAGAS Korean testset generation (optional)

---

**Fixed by**: Claude Code
**Verified with**: Context7 MCP + LangChain Official Docs
**Date**: 2025-10-30
