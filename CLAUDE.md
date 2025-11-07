# CRAWLING FRAMEWORK

- use crawl4ai for every crawling task. do not use any other tool.
- refer crawl4ai doc, skill, examples whenever needed.

# GPT-5 WITH RAGAS EVALUATION

## Issue: GPT-5 Temperature Parameter Incompatibility

GPT-5 only supports `temperature=1` (default value). RAGAS framework internally sets `temperature=1e-8` for deterministic evaluation, causing GPT-5 to reject requests with error:
```
Unsupported value: 'temperature' does not support 1E-8 with this model. Only the default (1) value is supported.
```

## Root Cause

RAGAS evaluation flow:
```
Script → RAGAS evaluate() → LangchainLLMWrapper → ChatOpenAI → OpenAI API
```

- RAGAS uses LangChain's `ChatOpenAI` to call OpenAI API directly
- LiteLLM `drop_params` setting doesn't apply (RAGAS bypasses LiteLLM)
- RAGAS internal code sets `temperature=1e-8` for consistency

## Solution: Patch RAGAS Library

Modify `.venv/lib/python3.12/site-packages/ragas/llms/base.py`:

```python
def get_temperature(self, n: int) -> float:
    """Return the temperature to use for completion based on n."""
    return None  # Force None for GPT-5 compatibility
    # Original: return 0.3 if n > 1 else 1e-8
```

## Test Results

After patching:
- ✅ GPT-5 as judge: Successfully evaluated 5 LLMs across 5 questions
- ✅ Metrics computed: Faithfulness, Answer Relevancy, Answer Correctness
- 🏆 Best performers:
  - Faithfulness: Gemma3-12B (0.900)
  - Answer Relevancy: GPT-4o-mini (0.966)
  - Answer Correctness: EXAONE-3.5-7.8B (0.410)

## Important Notes

- This is a **temporary workaround** requiring library modification
- Future RAGAS updates will overwrite this change
- Consider creating a wrapper or fork for production use
- Alternative: Use GPT-4o as judge (supports temperature parameter)

---

# 🔴 CRITICAL: KG CYPHER RAG FIX (2025-11-06)

## Checkpoint: Experimental Design Ground Truth

**Status**: ✅ FIXED | **Impact**: 🔴 CRITICAL - New baseline for all experiments

### Problem Discovered

KG Cypher Generation had **-47% performance degradation** due to missing vector similarity starting point.

```
OLD Implementation (FAILED):
- Pure LLM Cypher generation
- No vector search → No starting point
- Random graph walk → Empty or irrelevant results
- Result: 0.398 faithfulness (vs 0.746 Naive RAG)

ROOT CAUSE: Missing vector similarity search for initial node selection
```

### Solution Implemented

**Hybrid Vector-Graph Architecture** (`src/kg_agent/kg_cypher_retriever.py`):

```python
1. Vector Similarity Search → Find top-k relevant nodes (STARTING POINT)
2. Graph Expansion → Explore 1-hop neighbors from found nodes
3. Original Text Return → Preserve chunk quality (no LLM summary)
```

### Verified Results

**Quick Test (10Q, 3M):**
```
OLD Cypher:     0.398  (-47% vs Naive)
FIXED Cypher:   0.830  (+11% vs Naive, +6% vs KG Simple)
Improvement:    +108.5% 🚀
```

**Model Performance (Multi-hop):**
```
EXAONE-3.5-7.8B:  0.893 (+179% improvement) 🏆
GPT-4o-mini:      0.802 (+122% improvement)
GPT-OSS-20B:      0.795 (+101% improvement)
```

### New Ground Truth

**RAG Performance Ranking (Corrected):**
```
1. FIXED KG Cypher:  0.830 🥇 (Hybrid Vector+Graph)
2. KG Simple:        0.780 🥈 (Vector+Fixed Cypher)
3. Naive RAG:        0.746 🥉 (Vector Only)
```

### Design Principles Established

**All future KG RAG implementations MUST:**
1. ✅ Start with vector similarity search (non-negotiable)
2. ✅ Expand via graph traversal (from found nodes only)
3. ✅ Preserve original text (no LLM summarization)
4. ✅ Use hybrid approach (Vector + Graph = Optimal)

### Reference Documents

- **Full Analysis**: `docs/CHECKPOINT_kg_cypher_fix.md`
- **Quick Analysis**: `docs/cypher_rag_fix_analysis.md`
- **Implementation**: `src/kg_agent/kg_cypher_retriever.py`
- **Config**: `config/retest_cypher_rag_fixed.json`

### Next Steps

1. **Full Retest Required**: 50 questions, 5 models, FIXED Cypher
2. **Update All Reports**: Use FIXED results as new baseline
3. **Methodology Update**: Reflect Hybrid approach in thesis

**⚠️ IMPORTANT**: All previous KG Cypher results are INVALID. Only use FIXED version going forward.
