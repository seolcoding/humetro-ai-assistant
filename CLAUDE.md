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
