# RAG Benchmark System - Bug Fixes Summary

## Issues Fixed

### 1. QuestionComplexityClassifier Initialization Error
**Error**: `TypeError: QuestionComplexityClassifier.__init__() got an unexpected keyword argument 'model'`

**Root Cause**: In unified_benchmark_v2.py:256, classifier was initialized with `model="gpt-4o-mini"` but the classifier constructor expects `method` parameter instead.

**Fix**: Changed initialization from:
```python
classifier = QuestionComplexityClassifier(model="gpt-4o-mini")
```
to:
```python
classifier = QuestionComplexityClassifier(method="hybrid")
```

**File**: `src/rag_pipeline/unified_benchmark_v2.py:256`

---

### 2. Classification Input Type Mismatch
**Error**: Classifier expected DataFrame but received list of dicts

**Root Cause**: The `classify_questions` function received questions as a list of dicts but `QuestionComplexityClassifier.classify_batch()` expects a DataFrame.

**Fix**: Added DataFrame conversion:
```python
# Convert list of dicts to DataFrame for classifier
questions_df = pd.DataFrame(questions)

classifier = QuestionComplexityClassifier(method="hybrid")
classified = classifier.classify_batch(questions_df, question_column="user_input")

# Convert classification results back to dict format
classified_dicts = [asdict(c) for c in classified]
```

**File**: `src/rag_pipeline/unified_benchmark_v2.py:244-268`

---

### 3. Missing Imports
**Error**: `NameError: name 'pd' is not defined`, `NameError: name 'asdict' is not defined`

**Root Cause**: Missing pandas and dataclasses imports

**Fix**: Added missing imports:
```python
import pandas as pd
from dataclasses import asdict
```

**File**: `src/rag_pipeline/unified_benchmark_v2.py:17,20`

---

### 4. GenerationBenchmark Missing Arguments
**Error**: `TypeError: GenerationBenchmark.run_benchmark() missing 1 required positional argument: 'output_dir'`

**Root Cause**: `GenerationBenchmark.run_benchmark()` requires both `classified_questions: Dict[str, List]` and `output_dir: Path` but only questions were being passed.

**Fix**: Updated method call to include both required arguments:
```python
# Output directory for this model
model_output_dir = Path(self.args.output_dir) / f"exp_{self.exp_id}_{model_key}"
model_output_dir.mkdir(parents=True, exist_ok=True)

# 벤치마크 실행
results = benchmark.run_benchmark(
    classified_questions=classified_questions,
    output_dir=model_output_dir
)
```

**File**: `src/rag_pipeline/unified_benchmark_v2.py:297-305`

---

### 5. Incorrect Questions Data Structure
**Error**: Questions were passed as flat list instead of organized by type

**Root Cause**: `GenerationBenchmark.run_benchmark()` expects questions organized as `Dict[str, List]` with keys "simple" and "multi_reasoning", but was receiving a flat list.

**Fix**: Added organization logic in `run_benchmark()`:
```python
# Organize questions by classification type
classified_questions = {
    "simple": [],
    "multi_reasoning": []
}

for item in classified_list:
    classification = item.get("classification", "simple")
    # Map classification to expected keys
    if classification in ["single_hop", "simple"]:
        classified_questions["simple"].append({
            "question": item["question"],
            "ground_truth": item.get("ground_truth", ""),
            "question_type": "simple"
        })
    else:  # multi_hop or multi_reasoning
        classified_questions["multi_reasoning"].append({
            "question": item["question"],
            "ground_truth": item.get("ground_truth", ""),
            "question_type": "multi_reasoning"
        })
```

**File**: `src/rag_pipeline/unified_benchmark_v2.py:324-348`

---

## Verification Status

### ✅ Successfully Tested
1. **Experiment Creation**: ID 4 created successfully with 41 questions generated via RAGAS
2. **Checkpoint System**: Questions saved to `data/evaluation/experiments/checkpoints/exp_4_questions_generated.pkl`
3. **Resume Capability**: Successfully resumed experiment 4 with `--resume-id 4`
4. **Question Classification**: 41 questions classified successfully using hybrid method
5. **Classification Checkpoint**: Saved to `data/evaluation/experiments/checkpoints/exp_4_questions_classified.pkl`

### 🔄 Next Steps for Complete Verification
- Run full benchmark with actual model evaluation (remove `--skip-benchmark`)
- Verify all 5 thesis models run successfully
- Check final results are saved correctly
- Validate RAGAS evaluation with GPT-5 as judge

---

## Files Modified
- `src/rag_pipeline/unified_benchmark_v2.py` - Main benchmark orchestrator
  - Fixed classifier initialization
  - Added DataFrame conversion
  - Added missing imports
  - Fixed GenerationBenchmark call signature
  - Added questions organization by type

---

## Related Files
- `src/rag_pipeline/question_classifier.py` - Question classifier (no changes)
- `src/rag_pipeline/generation_benchmark.py` - Model benchmark runner (no changes)
- `tests/rag_pipeline/testset_generator.py` - RAGAS testset generation (no changes)
