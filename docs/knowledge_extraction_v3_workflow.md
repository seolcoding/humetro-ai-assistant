# Knowledge Extraction v3 Workflow

Complete workflow for bottom-up knowledge extraction with single-file state management.

## Architecture Overview

### Key Improvements from v2

1. **Single-file state management** - No more batch file overwriting
2. **ID-based dictionary** - Fast status lookups by dialogue_id
3. **Batch-wise synchronous saves** - Not per-dialogue (more efficient)
4. **401 error detection** - Automatic stop with safe resume
5. **Status-driven processing** - Only process pending/fail dialogues
6. **Idempotent execution** - Stop/resume anywhere without data loss

### State File Format

```json
{
  "B2954": {
    "status": "success",
    "dialogue_id": "B2954",
    "documents": [
      {
        "dialogue_id": "B2954",
        "original_question": "...",
        "original_answer": "...",
        "topic_path": "교통 > 버스 > 노선정보",
        "primary_topic": "버스 노선정보",
        "secondary_topics": ["정류장 위치", "운행시간"],
        "document": "# 버스 노선정보..."
      }
    ],
    "error": null,
    "timestamp": "2025-11-08T10:30:00",
    "retry_count": 0
  }
}
```

## Complete Workflow

### Step 0: Merge Existing Batch Files (One-time)

If you have existing `batch_*.json` files from v2, merge them first:

```bash
# Merge all batch_*.json files into state format
uv run python scripts/merge_batch_files.py \
  --batch-dir data/evaluation/knowledge_extraction_full \
  --output data/evaluation/knowledge_extraction_full/extracted_documents.json
```

**What it does:**
- Reads all `batch_001.json`, `batch_002.json`, etc.
- Groups documents by `dialogue_id`
- Converts to state format with `status: "success"`
- Preserves existing successful extractions

### Step 1: Initialize State File

Pre-populate all dialogue IDs with `status: "pending"`:

```bash
# Initialize state file with all dialogue IDs
uv run python scripts/initialize_extraction_state.py \
  --input data/processed/knowledge_extraction/full_dialogues.json \
  --state data/evaluation/knowledge_extraction_full/extracted_documents.json
```

**What it does:**
- Loads all dialogue IDs from input file
- Creates state entry for each with `status: "pending"`
- Preserves existing successful entries (from merge step)
- Shows statistics: success/pending/fail counts

**Output:**
```
📊 State Statistics:
  ✅ Success: 150 (preserved)
  ⏳ Pending: 7850 (new)
  Total: 8000
```

### Step 2: Run Extraction

Process only pending/fail dialogues:

```bash
# Run extraction with 401 detection
uv run python src/knowledge_extraction/claude_knowledge_extractor_v3.py \
  --config config/knowledge_extraction_full.json \
  --input data/processed/knowledge_extraction/full_dialogues.json \
  --state-file data/evaluation/knowledge_extraction_full/extracted_documents.json \
  --debug  # Optional: see raw API responses
```

**What it does:**
- Loads state file
- Filters: only `status == "pending"` or `status == "fail"`
- Processes in batches (10 dialogues, 65s delay)
- Updates state after each batch (synchronous save)
- Detects 401 errors → stops immediately → exits with code 401

**On 401 Error:**
```
🔐 AUTHENTICATION ERROR DETECTED
Extraction stopped due to 401 error.
Please check your API key and re-authenticate.

To resume:
1. Fix authentication issue
2. Run the same command again
   (Progress is saved in state file)
```

**Resume After 401:**
```bash
# Fix auth, then run same command again
# It will skip already successful dialogues
uv run python src/knowledge_extraction/claude_knowledge_extractor_v3.py \
  --config config/knowledge_extraction_full.json \
  --input data/processed/knowledge_extraction/full_dialogues.json \
  --state-file data/evaluation/knowledge_extraction_full/extracted_documents.json
```

### Step 3: Monitor Progress (Optional)

In a separate terminal, monitor real-time progress:

```bash
# Monitor extraction in real-time
uv run python scripts/monitor_extraction.py
```

**Shows:**
- Batch completion count
- Success/fail rates
- Error analysis (401, 429, JSON parse, etc.)
- Estimated time remaining
- Live updates every 5 seconds

### Step 4: Export Final Results

Extract only successful documents as flat array:

```bash
# Export successful documents only
uv run python scripts/export_success_only.py \
  --state data/evaluation/knowledge_extraction_full/extracted_documents.json \
  --output data/evaluation/knowledge_extraction_full/extracted_documents_final.json \
  --verify  # Verify output integrity
```

**What it does:**
- Filters: only `status == "success"` entries
- Extracts `documents` arrays
- Flattens into single array
- Adds metadata (statistics, timestamps)
- Verifies required fields present

**Output Format:**
```json
{
  "metadata": {
    "description": "Extracted knowledge documents...",
    "exported_at": "2025-11-08T10:30:00",
    "statistics": {
      "total_dialogues": 8000,
      "successful_dialogues": 7950,
      "failed_dialogues": 50,
      "total_documents": 15900,
      "documents_per_dialogue": 2.0
    }
  },
  "documents": [...]
}
```

## Status Transitions

```
Initial: pending
  ↓
Processing...
  ↓
Success: success (with documents)
Failure: fail (retry next run)
Auth Error: auth_error (retry after fix)
```

## File Structure

```
data/evaluation/knowledge_extraction_full/
├── extracted_documents.json          # STATE FILE (main)
├── extracted_documents_final.json    # Final export (flat array)
├── extraction_summary.json           # Statistics
└── logs/
    └── knowledge_extraction_*.log    # Debug logs
```

## Common Scenarios

### Scenario 1: Fresh Start (No Existing Data)

```bash
# 1. Initialize state
uv run python scripts/initialize_extraction_state.py \
  --input data/processed/knowledge_extraction/full_dialogues.json \
  --state data/evaluation/knowledge_extraction_full/extracted_documents.json

# 2. Run extraction
uv run python src/knowledge_extraction/claude_knowledge_extractor_v3.py \
  --config config/knowledge_extraction_full.json \
  --input data/processed/knowledge_extraction/full_dialogues.json

# 3. Export results
uv run python scripts/export_success_only.py \
  --state data/evaluation/knowledge_extraction_full/extracted_documents.json \
  --output data/evaluation/knowledge_extraction_full/extracted_documents_final.json
```

### Scenario 2: Migrate from v2 (Has batch_*.json files)

```bash
# 1. Merge existing batch files
uv run python scripts/merge_batch_files.py \
  --batch-dir data/evaluation/knowledge_extraction_full \
  --output data/evaluation/knowledge_extraction_full/extracted_documents.json

# 2. Initialize remaining IDs
uv run python scripts/initialize_extraction_state.py \
  --input data/processed/knowledge_extraction/full_dialogues.json \
  --state data/evaluation/knowledge_extraction_full/extracted_documents.json

# 3. Continue extraction
uv run python src/knowledge_extraction/claude_knowledge_extractor_v3.py \
  --config config/knowledge_extraction_full.json \
  --input data/processed/knowledge_extraction/full_dialogues.json
```

### Scenario 3: Resume After 401 Error

```bash
# 1. Fix authentication (update API key in llm_config.py or proxy)

# 2. Just re-run extraction (it will skip successful ones)
uv run python src/knowledge_extraction/claude_knowledge_extractor_v3.py \
  --config config/knowledge_extraction_full.json \
  --input data/processed/knowledge_extraction/full_dialogues.json

# State file already has progress saved!
```

### Scenario 4: Retry Failed Dialogues

```bash
# Failed dialogues are marked with status: "fail"
# Just re-run extraction - it will retry them automatically

uv run python src/knowledge_extraction/claude_knowledge_extractor_v3.py \
  --config config/knowledge_extraction_full.json \
  --input data/processed/knowledge_extraction/full_dialogues.json

# Extractor filters: status == "pending" OR status == "fail"
```

## Key Features

### 1. No Data Loss
- State saved after each batch (not per-dialogue)
- Can stop/resume at any time
- Successful extractions never re-processed

### 2. 401 Detection
- Automatic detection of auth errors
- Immediate stop (no wasted API calls)
- Safe resume after fixing auth

### 3. Efficient Processing
- Only processes pending/fail dialogues
- Batch-wise saves (not per-dialogue I/O)
- Parallel API calls within batch (10 concurrent)

### 4. Monitoring
- Real-time progress tracking
- Error classification and analysis
- ETA estimation

## Troubleshooting

### Problem: All requests failing with "Expecting value: line 1 column 1"

**Cause:** API authentication or proxy issue
**Solution:**
1. Enable debug mode: `--debug`
2. Check raw responses in logs
3. Verify API key and proxy settings

### Problem: State file has all "pending" even though some succeeded

**Cause:** Forgot to run merge or initialize
**Solution:**
1. Run merge script if you have batch_*.json files
2. Re-run initialize to preserve existing state

### Problem: Want to reset and start over

**Solution:**
```bash
# Option 1: Delete state file
rm data/evaluation/knowledge_extraction_full/extracted_documents.json

# Option 2: Mark all as pending
uv run python scripts/initialize_extraction_state.py \
  --input data/processed/knowledge_extraction/full_dialogues.json \
  --state data/evaluation/knowledge_extraction_full/extracted_documents.json \
  --force  # Reset all to pending
```

## Performance Estimates

- **8000 dialogues** × **65s/batch** ÷ **10 dialogues/batch** = ~14.5 hours
- Rate: ~550 dialogues/hour
- With 401 resume: Lost time = batches completed × 65s (minimal)

## Next Steps

After extraction completes:
1. Export final results with `export_success_only.py`
2. Verify document count and quality
3. Use final JSON for knowledge base ingestion
4. Build graph from extracted documents
