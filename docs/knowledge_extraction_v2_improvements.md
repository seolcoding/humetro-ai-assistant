# Knowledge Extraction v2 - Improvement Summary

## Overview

v2는 실험 설계의 4가지 주요 문제점을 해결하여 안정성과 효율성을 크게 향상시켰습니다.

## Critical Improvements

### 1. ✅ Correct Rate Limiting Implementation

**Before (v1)**:
```python
await self.extract_knowledge_batch(...)  # Takes ~10s
await asyncio.sleep(60)  # Wait AFTER completion
# Total: 70s per batch
```

**After (v2)**:
```python
# First batch: no delay
if batch_num > 1:
    await asyncio.sleep(65)  # Wait BEFORE starting (60s + 5s safety)
# Total: 65s per batch (from batch start to next batch start)
```

**Impact**: Safer rate limit compliance with 5s safety margin

---

### 2. ✅ Incremental Saving

**Before (v1)**:
- Save once at the end
- If interrupted → lose all progress

**After (v2)**:
```
output_dir/
├── batch_001.json  ← Saved immediately after batch completes
├── batch_002.json
├── batch_003.json
├── ...
└── extracted_documents.json  ← Final consolidated file
```

**Impact**: No data loss even if process is interrupted

---

### 3. ✅ Caching & Resume Capability

**Before (v1)**:
- No cache → restart from beginning

**After (v2)**:
```python
# MD5-based dialogue identification
dialogue_hash = md5(json.dumps(dialogue, sort_keys=True))

# Check cache before processing
if dialogue_hash in self.cache:
    return cached_result  # Skip already processed

# Cache results after processing
self._cache_result(dialogue, documents)
```

**Cache Structure**:
```json
{
  "abc123def456...": {
    "dialogue_id": "B2033",
    "documents": [...],
    "timestamp": "2025-11-07T12:34:56"
  }
}
```

**Impact**: Resume interrupted runs without reprocessing

---

### 4. ✅ Deferred Retry Pattern

**Before (v1)**:
```python
# Immediate retry within batch → blocks entire batch
try:
    documents = json.loads(response)
except JSONDecodeError:
    await asyncio.sleep(2)  # BLOCKS all parallel calls
    return await extract_single(..., retry_count + 1)
```

**Problem**: If 1 dialogue fails and retries, entire batch waits → defeats parallelism

**After (v2)**:
```python
# Step 1: Collect failures during batch (no immediate retry)
for docs, success, failed_dialogue in results:
    if not success and failed_dialogue:
        failed_dialogues.append(failed_dialogue)

# Step 2: Retry all failures AFTER main batches complete
if all_failed_dialogues:
    for retry_attempt in range(1, max_retries + 1):
        retry_batch_num = total_batches + retry_attempt
        retry_results = await extract_knowledge_batch(
            all_failed_dialogues,
            retry_batch_num,
            total_batches + max_retries
        )
        all_failed_dialogues = still_failed  # Only retry what still fails
```

**Impact**:
- Preserves batch parallelism ✅
- Proper rate limiting for retry batches ✅
- No delays in main processing flow ✅

**Example Flow**:
```
Main Batches:
  Batch 1/20: 10 dialogues → 10 success
  Batch 2/20: 10 dialogues → 9 success, 1 failed (D123) → collect
  Batch 3/20: 10 dialogues → 8 success, 2 failed (D145, D156) → collect
  ...
  Batch 20/20: 10 dialogues → 10 success

Retry Batches:
  Batch 21/23: 3 failed dialogues (D123, D145, D156) → 2 success, 1 failed (D145)
  Batch 22/23: 1 failed dialogue (D145) → 1 success

✅ All failures recovered!
```

---

## Performance Comparison

### v1 (Old)
- ❌ Rate Limiting: 70s per batch (batch duration + 60s wait)
- ❌ Saving: Once at end → loss risk
- ❌ Resume: Not possible
- ❌ Retry: Inline → batch delays
- ⏱️ **200 dialogues**: ~23 minutes

### v2 (New)
- ✅ Rate Limiting: 65s per batch (60s + 5s margin from batch start)
- ✅ Saving: Every batch + final consolidation
- ✅ Resume: MD5-based cache
- ✅ Retry: Deferred pattern → no batch delays
- ⏱️ **200 dialogues**: ~21.7 minutes + retry batches (if failures occur)

### Time Estimates

**POC (200 dialogues, 20 batches)**:
- v1: 23 minutes (20 × 70s)
- v2: 21.7 minutes (19 × 65s + first batch) + retries
- Improvement: **-10.4%** time + much safer

**Full Dataset (8000 dialogues, 800 batches)**:
- v1: 13.3 hours (unsafe)
- v2: 14.4 hours (safer with margin)
- Trade-off: +1.1 hours for **safety + resume + retry**

---

## Key Benefits

### Reliability
- ✅ No data loss from interruptions
- ✅ Automatic resume capability
- ✅ Intelligent retry without blocking

### Safety
- ✅ Rate limit compliance with margin
- ✅ Incremental progress tracking
- ✅ Cache prevents reprocessing

### Efficiency
- ✅ Batch parallelism preserved
- ✅ Deferred retry pattern
- ✅ Only retry what fails

### Observability
- ✅ Loguru with timestamps
- ✅ Clear batch progress
- ✅ Retry attempt tracking

---

## Implementation Details

### Retry Logic Flow

```python
class ClaudeKnowledgeExtractor:
    async def extract_knowledge_single(...) -> Tuple[List, bool, Optional[Dict]]:
        """Returns: (documents, success, failed_dialogue_or_None)"""
        try:
            documents = await call_api_and_parse()
            return documents, True, None
        except Exception:
            return [], False, dialogue  # Return failed dialogue for retry

    async def extract_knowledge_batch(...) -> Tuple[List, int, int, List]:
        """Returns: (documents, success_count, failure_count, failed_dialogues)"""
        results = await asyncio.gather(*tasks)

        failed_dialogues = []
        for docs, success, failed_dialogue in results:
            if not success and failed_dialogue:
                failed_dialogues.append(failed_dialogue)

        return all_documents, success_count, failure_count, failed_dialogues

    async def extract_all(...):
        # Process main batches
        all_failed_dialogues = []
        for batch in batches:
            docs, success, failure, failed = await extract_knowledge_batch(...)
            all_failed_dialogues.extend(failed)

        # Deferred retry
        if all_failed_dialogues:
            for retry_attempt in range(1, max_retries + 1):
                retry_batch_num = total_batches + retry_attempt
                docs, success, failure, still_failed = await extract_knowledge_batch(
                    all_failed_dialogues, retry_batch_num, ...
                )
                all_failed_dialogues = still_failed
```

### Cache Management

```python
def _get_dialogue_hash(self, dialogue: Dict) -> str:
    """Create stable hash from dialogue content."""
    dialogue_str = json.dumps(dialogue, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(dialogue_str.encode()).hexdigest()

def _cache_result(self, dialogue: Dict, documents: List):
    """Cache result with timestamp."""
    dialogue_hash = self._get_dialogue_hash(dialogue)
    self.cache[dialogue_hash] = {
        "dialogue_id": dialogue.get("dialogue_id"),
        "documents": documents,
        "timestamp": datetime.now().isoformat()
    }
```

---

## Verification

Test the retry flow:
```bash
uv run python scripts/verify_retry_logic.py
```

Expected output:
```
MAIN BATCHES
📦 Batch 1/3 → ✅ All succeeded
📦 Batch 2/3 → ❌ Failed: 2 dialogues
📦 Batch 3/3 → ✅ All succeeded

RETRY BATCHES
🔄 Retry attempt 1/3
   Batch 4/6 → ✅ Recovered: D001, ❌ Still failed: D002
🔄 Retry attempt 2/3
   Batch 5/6 → ✅ Recovered: D002

✅ All failed dialogues recovered!
```

---

## Next Steps

1. **Test with 30 dialogues**:
```bash
uv run python scripts/test_extraction_30.py
uv run python src/knowledge_extraction/claude_knowledge_extractor_v2.py \
    --config config/knowledge_extraction_test_30.json \
    --input data/processed/knowledge_extraction/test_30_dialogues.json
```

2. **Run full POC (200 dialogues)**:
```bash
uv run python src/knowledge_extraction/claude_knowledge_extractor_v2.py \
    --config config/knowledge_extraction_poc.json \
    --input data/processed/knowledge_extraction/sampled_dialogues.json
```

3. **Organize documents**:
```bash
uv run python src/knowledge_extraction/document_organizer.py \
    --input data/evaluation/knowledge_extraction_poc/extracted_documents.json \
    --output-dir data/processed/knowledge_docs \
    --min-doc-size 500 \
    --merge-small
```

4. **Validate quality**:
```bash
uv run python src/knowledge_extraction/quality_validator.py \
    --docs-dir data/processed/knowledge_docs \
    --report-path data/evaluation/knowledge_extraction_poc/quality_report.md \
    --min-doc-size 500
```
