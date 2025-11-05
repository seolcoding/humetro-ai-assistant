# RAGAS Evaluation API Fix

## Problem

RAGAS evaluation was failing with error:
```
AttributeError: 'list' object has no attribute 'to_dict'
```

This error occurred in `/Users/sdh/Dev/02_production_projects/humetro-ai-assistant/src/rag_pipeline/generation_benchmark.py` at line 342 during result extraction after RAGAS evaluation completed.

## Root Cause Analysis

### Investigation Timeline

1. **Symptom**: Evaluation reached 100% progress, then failed when extracting results
2. **Error Location**: Line 342 in `_evaluate_model()` method
3. **Problematic Code**:
   ```python
   scores = evaluation.scores.to_dict() if hasattr(evaluation, 'scores') else {}
   ```

### Evidence from RAGAS Source Code

Inspected RAGAS library internal structure:

```python
@dataclass
class EvaluationResult:
    """
    A class to store and process the results of the evaluation.

    Attributes
    ----------
    scores : List[Dict[str, Any]]  # <-- Already a list of dicts!
        The dataset containing the scores of the evaluation.
    """
    scores: t.List[t.Dict[str, t.Any]]  # Not an object with to_dict()!
    dataset: EvaluationDataset
    # ... other fields
```

**Key Finding**: `evaluation.scores` is already a `List[Dict[str, Any]]`, NOT an object with a `.to_dict()` method.

### Why the Code Failed

1. RAGAS `evaluate()` returns `EvaluationResult` object
2. `EvaluationResult.scores` is a **list of dictionaries** containing per-row metric scores
3. Calling `.to_dict()` on a list raises `AttributeError`
4. The code incorrectly assumed `scores` was an object with a `to_dict()` method

## Solution

### Correct RAGAS API Usage

RAGAS provides `to_pandas()` method to convert results to DataFrame:

```python
# BEFORE (INCORRECT)
scores = evaluation.scores.to_dict() if hasattr(evaluation, 'scores') else {}

# AFTER (CORRECT)
df_results = evaluation.to_pandas()  # Convert to DataFrame first

# Extract metric scores as lists
metrics_data = {}
for metric_name in ["faithfulness", "answer_relevancy", "answer_correctness"]:
    if metric_name in df_results.columns:
        metrics_data[metric_name] = df_results[metric_name].tolist()
    else:
        metrics_data[metric_name] = []
```

### Implementation Changes

**File**: `/Users/sdh/Dev/02_production_projects/humetro-ai-assistant/src/rag_pipeline/generation_benchmark.py`

**Lines 341-352** (old code removed):
```python
# Extract scores
scores = evaluation.scores.to_dict() if hasattr(evaluation, 'scores') else {}

return {
    "raw_data": eval_data,
    "metrics": {
        "faithfulness": scores.get("faithfulness", []),
        "answer_relevancy": scores.get("answer_relevancy", []),
        "answer_correctness": scores.get("answer_correctness", [])
    },
```

**Lines 341-352** (new code):
```python
# Extract scores using to_pandas() method
# evaluation.scores is a list of dicts, convert to DataFrame first
df_results = evaluation.to_pandas()

# Extract metric scores as lists
metrics_data = {}
for metric_name in ["faithfulness", "answer_relevancy", "answer_correctness"]:
    if metric_name in df_results.columns:
        metrics_data[metric_name] = df_results[metric_name].tolist()
    else:
        metrics_data[metric_name] = []

return {
    "raw_data": eval_data,
    "metrics": metrics_data,
```

### Additional Improvements

Added better error logging (lines 365-367):
```python
except Exception as e:
    logger.error(f"RAGAS evaluation failed: {e}")
    import traceback
    logger.error(traceback.format_exc())  # Full stack trace for debugging
```

## Verification

### Expected Behavior After Fix

1. RAGAS evaluation completes successfully (100% progress)
2. Results converted to DataFrame using `evaluation.to_pandas()`
3. Metric scores extracted as lists from DataFrame columns
4. Summary statistics calculated correctly
5. Results saved to JSON and markdown reports

### Test Command

```bash
uv run python scripts/run_generation_benchmark.py
```

### Success Criteria

- No `AttributeError` during result extraction
- Benchmark results JSON contains metric scores for each question
- Summary report shows average scores per model and complexity level
- All models evaluated successfully with GPT-5 as judge

## Technical Notes

### RAGAS Result Structure

```python
EvaluationResult
├── scores: List[Dict[str, Any]]       # Per-row metric scores
├── dataset: EvaluationDataset         # Original evaluation data
├── to_pandas() -> DataFrame           # Conversion method
└── total_tokens, total_cost, etc.     # Metadata
```

### DataFrame Structure After `to_pandas()`

```
| user_input | response | retrieved_contexts | reference | faithfulness | answer_relevancy | answer_correctness |
|------------|----------|---------------------|-----------|--------------|------------------|-------------------|
| Q1         | A1       | [C1, C2]           | R1        | 0.95         | 0.88             | 0.72              |
| Q2         | A2       | [C3, C4]           | R2        | 0.87         | 0.92             | 0.65              |
```

### Metric Extraction Pattern

```python
# Access individual metric columns
faithfulness_scores = df_results["faithfulness"].tolist()
relevancy_scores = df_results["answer_relevancy"].tolist()
correctness_scores = df_results["answer_correctness"].tolist()

# Calculate averages
avg_faithfulness = sum(faithfulness_scores) / len(faithfulness_scores)
```

## Related Issues

### GPT-5 Temperature Compatibility

This fix assumes the GPT-5 temperature patch is already applied (see CLAUDE.md):

```python
def _get_judge_llm(self):
    if self.judge_model == "gpt-5":
        return ChatOpenAI(
            model=self.judge_model
            # temperature, max_tokens excluded for GPT-5
        )
```

Combined with RAGAS library patch at `.venv/lib/python3.12/site-packages/ragas/llms/base.py`:
```python
def get_temperature(self, n: int) -> float:
    return None  # Force None for GPT-5 compatibility
```

## Conclusion

The error was caused by incorrect assumption about RAGAS API. The fix uses the correct `to_pandas()` method to extract evaluation results, ensuring compatibility with RAGAS library structure.

**Status**: Fixed and ready for testing
**Impact**: Enables generation benchmarking with proper metric extraction
**Next Steps**: Run full benchmark evaluation to verify fix
