# Dasan Dataset Reorganization: Complete Analysis

**Analysis Date**: 2025-11-11  
**Status**: Pre-Implementation Planning  
**Scope**: Rename `AI_HUB_DASAN_QA` → `AI_HUB_DASAN_QA` with proper directory structure

---

## 1. CURRENT DATA STRUCTURE

### 1.1 Main Directory: `/data/AI_HUB_DASAN_QA/`

**File Count**: 21,477 files  
**Total Size**: ~300+ MB

```
data/AI_HUB_DASAN_QA/
├── extracted/                              # Raw extracted JSON files
│   ├── training/
│   │   └── labeled/                        # Training data
│   └── validation/
│       └── labeled/                        # Validation data
│
├── raw/                                    # Original raw data
│   ├── aihubshell                          # Download script
│   ├── annot_format.md                     # Format documentation
│   ├── docs.pdf                            # Original documentation
│   └── 022.민원(콜센터) 질의-응답 데이터/ # Original data directory
│
├── organized_markdown/                     # Initial markdown organization (small subset)
│   ├── README.md
│   ├── 생활하수도_관련_문의/
│   ├── 일반행정_문의/
│   ├── 주택_및_부동산/
│   └── 코로나19_관련_상담/
│
├── organized_markdown_full/                # COMPLETE markdown organization (10,757+ docs)
│   ├── README.md
│   ├── 건강_의료/
│   ├── 공공서비스/
│   ├── 공과금_납부/
│   ├── 교통/                               # [Main category with many subcategories]
│   ├── 문화_여가/
│   ├── 보건_복지/
│   ├── 상하수도/
│   ├── 생활환경/
│   ├── 일반행정_문의/
│   ├── 주택_및_부동산/
│   └── ... (80+ categories total)
│
├── organized_markdown_normalized/          # Normalized subset (4 categories)
│   ├── 대중교통_안내/
│   ├── 생활하수도_관련_문의/
│   ├── 일반행정_문의/
│   └── 코로나19_관련_상담/
│
├── knowledge_docs.jsonl                    # Consolidated knowledge documents (basic)
├── knowledge_docs_full.jsonl               # Full consolidated documents (45.6 MB)
├── knowledge_docs_normalized.jsonl         # Normalized subset
├── knowledge_docs_normalized_aggressive.jsonl # Aggressively normalized
├── knowledge_docs_full_metadata.json       # Metadata for full documents
├── knowledge_docs_metadata.json            # Metadata for basic documents
│
└── prediction-model-2025-11-08T14_26_42.348286Z_predictions.jsonl  # Model predictions (180 MB)
```

### 1.2 Vector Store Structure

**Location**: `/data/vector_store/`

```
data/vector_store/
└── dasan_test/                             # Test vector store (small)
    ├── index.faiss                         # FAISS index
    ├── index.pkl                           # Pickle metadata
    └── metadata.json                       # Store metadata
```

**Note**: `AI_HUB_DASAN_QA` vector store mentioned in code but NOT currently built/deployed

### 1.3 Processed Data Structure

**Location**: `/data/processed/`

```
data/processed/
└── dasan_eda/                              # EDA analysis artifacts
    ├── dasan_full_dataset.csv              # Full dataset as CSV (52 MB)
    ├── summary_statistics.json
    ├── category_summary.csv
    ├── annotation_analysis.png
    ├── category_distribution.png
    ├── dialogue_turns_analysis.png
    ├── intent_analysis.png
    ├── qa_structure_analysis.png
    └── text_length_analysis.png
```

---

## 2. DATA PROCESSING PIPELINE

### 2.1 Processing Stages

```
RAW INPUT
    ↓
[Stage 0] Extract
    - Source: data/AI_HUB_DASAN_QA/00_raw/
    - Output: data/AI_HUB_DASAN_QA/01_extracted/
    - Script: src/scripts/data_processing/extract_dasan_data.py
    ↓
[Stage 1] Organize (Markdown)
    - Input: data/AI_HUB_DASAN_QA/01_extracted/
    - Output: data/AI_HUB_DASAN_QA/03_markdown_full/
    - Script: src/knowledge_extraction/organize_markdown.py
    - Creates: 10,757+ category-organized markdown files
    ↓
[Stage 2] Consolidate (JSONL)
    - Input: data/AI_HUB_DASAN_QA/01_extracted/ OR markdown
    - Output: data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs*.jsonl
    - Script: src/knowledge_extraction/consolidate_extractions.py
    - Variants: basic, full, normalized (aggressive/standard)
    ↓
[Stage 3] Normalize (Topics)
    - Input: data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs_full.jsonl
    - Output: data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs_normalized.jsonl
    - Script: src/knowledge_extraction/normalize_topics.py
    ↓
[Stage 4] Vector Embedding
    - Input: data/AI_HUB_DASAN_QA/03_markdown_full/
    - Output: data/vector_store/dasan_*/
    - Script: scripts/build_dasan_vector_store.py
    - Model: text-embedding-3-large
    ↓
[Stage 5] RAG Benchmark
    - Input: data/vector_store/dasan_*/ + organized_markdown_full/
    - Output: experiments/benchmark_results/
    - Scripts: unified_benchmark_v4_dasan.py + evaluators
```

### 2.2 Data Format Specifications

**Markdown Format** (organized_markdown_full/):
```yaml
---
dialogue_id: "dialogue_001"
question_id: "q_001"
category: "교통"
subcategory: "대중교통_안내"
entities: ["버스", "지하철"]
kb_tags: ["transit", "public_transport"]
topics: ["경로_검색"]
---

# Question
사용자 질문 내용

## Answer
시스템 응답 내용
```

**JSONL Format** (knowledge_docs*.jsonl):
```json
{
  "dialogue_id": "string",
  "question": "string",
  "answer": "string",
  "category": "string",
  "subcategory": "string",
  "entities": ["string"],
  "kb_tags": ["string"],
  "topics": ["string"],
  "dialogue_turns": number,
  "validation_score": float (optional),
  "source": "string"
}
```

---

## 3. CODE DEPENDENCIES (Hardcoded Paths)

### 3.1 Python Files with "AI_HUB_DASAN_QA" References

**11 Files Total** with hardcoded paths:

| File | Hardcoded Paths | Purpose |
|------|-----------------|---------|
| `src/batch_processing/create_dasan_batch_input.py` | `data/AI_HUB_DASAN_QA/01_extracted/` | Batch input creation for LLM processing |
| `scripts/prepare_full_dialogues.py` | `data/AI_HUB_DASAN_QA/01_extracted/training/labeled/` | Prepare complete dialogue data |
| `src/knowledge_extraction/consolidate_extractions.py` | `data/AI_HUB_DASAN_QA/` (input/output) | Consolidate extracted data to JSONL |
| `scripts/build_dasan_vector_store.py` | `data/AI_HUB_DASAN_QA/03_markdown_full/`, `data/AI_HUB_DASAN_QA/07_vector_stores/full` | Build vector stores from markdown |
| `src/knowledge_extraction/normalize_topics.py` | `data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs_*.jsonl` | Normalize topics in JSONL files |
| `scripts/extract_transport_data.py` | `data/AI_HUB_DASAN_QA/01_extracted/training/labeled/` | Extract transport-specific data |
| `src/knowledge_extraction/process_predictions.py` | `data/AI_HUB_DASAN_QA/06_predictions/prediction-*.jsonl`, `data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs.jsonl` | Process model predictions |
| `src/knowledge_extraction/organize_markdown.py` | `data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs.jsonl`, `data/AI_HUB_DASAN_QA/02_markdown_basic` | Organize knowledge as markdown |
| `src/rag_pipeline/unified_benchmark_v4_dasan.py` | `data/AI_HUB_DASAN_QA/03_markdown_full/`, `data/AI_HUB_DASAN_QA/07_vector_stores/full` | Main RAG benchmark script |
| `src/scripts/data_processing/extract_dasan_data.py` | `data/AI_HUB_DASAN_QA/00_raw/` (input), `data/AI_HUB_DASAN_QA/01_extracted/` (output) | Extract raw data files |
| `src/rag_pipeline/stages/stage_01_dasan_loader.py` | `data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs.jsonl` | RAG pipeline stage for loading Dasan data |

### 3.2 Additional Dasan-Related Python Files

**7 Additional Scripts** (dasan API/crawling - not directly related to rename):

```
src/scripts/dasan_api/
├── scan_dasan_sequences.py
├── fetch_dasan_with_cache.py
├── fetch_dasan_api_data.py
├── fetch_complete_dasan_data.py
├── fetch_all_dasan_types.py
└── discover_dasan_range.py

src/scripts/deprecated/
└── DEPRECATED_test_scan_sequences.py
```

### 3.3 Path Reference Summary

**Total Hardcoded Path Instances**: 45+ references

**Most Frequent Paths**:
1. `data/AI_HUB_DASAN_QA/` - 18 references
2. `data/AI_HUB_DASAN_QA/03_markdown_full/` - 8 references
3. `data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs*.jsonl` - 12 references
4. `data/AI_HUB_DASAN_QA/01_extracted/` - 6 references
5. `data/AI_HUB_DASAN_QA/07_vector_stores/full` - 3 references
6. `data/vector_store/dasan_test` - 2 references

---

## 4. DOCUMENTATION REFERENCES

### 4.1 Markdown Files Mentioning "AI_HUB_DASAN_QA"

**5 Documentation Files**:

1. **docs/topic_normalization_report.md**
   - References: AI_HUB_DASAN_QA data pipeline
   - Update Required: Yes

2. **docs/knowledge_extraction_workflow.md**
   - References: AI_HUB_DASAN_QA processing stages
   - Update Required: Yes

3. **docs/bottom_up_knowledge_extraction_plan.md**
   - References: AI_HUB_DASAN_QA pipeline architecture
   - Update Required: Yes

4. **docs/dasan_api_usage.md**
   - References: AI_HUB_DASAN_QA data source
   - Update Required: Yes

5. **docs/architecture/data_storage_architecture.md**
   - References: AI_HUB_DASAN_QA directory structure
   - Update Required: Yes

### 4.2 Configuration Files to Update

- `.gitignore` (if mentions AI_HUB_DASAN_QA)
- `README.md` (if mentions AI_HUB_DASAN_QA)
- Configuration YAML files in `config/` directory

---

## 5. PROPOSED NEW STRUCTURE

### 5.1 New Directory Names

```
CURRENT                              →   PROPOSED

data/AI_HUB_DASAN_QA/                     →   data/AI_HUB_DASAN_QA/

├── raw/                             →   ├── 00_raw/
├── extracted/                       →   ├── 01_extracted/
├── organized_markdown/              →   ├── 02_markdown_basic/
├── organized_markdown_full/         →   ├── 03_markdown_full/
├── organized_markdown_normalized/   →   ├── 04_markdown_normalized/
├── knowledge_docs*.jsonl            →   ├── 05_consolidated/
└── prediction-*.jsonl               →   └── 06_predictions/

data/vector_store/dasan_*            →   data/vector_store/AI_HUB_DASAN_QA_*

data/AI_HUB_DASAN_QA/08_processed/eda_artifacts/            →   data/processed/AI_HUB_DASAN_QA_eda/
```

### 5.2 Structured Directory Organization

```
data/AI_HUB_DASAN_QA/
│
├── 00_raw/
│   ├── README.md                        # Data source documentation
│   ├── aihubshell                       # Download script
│   ├── annot_format.md                  # Format specification
│   ├── docs.pdf                         # Original documentation
│   └── 022.민원(콜센터)_질의응답_데이터/
│
├── 01_extracted/
│   ├── training/
│   │   └── labeled/                     # Training JSON files
│   └── validation/
│       └── labeled/                     # Validation JSON files
│
├── 02_markdown_basic/                   # Early stage (smaller subset)
│   ├── README.md
│   ├── 생활하수도_관련_문의/
│   ├── 일반행정_문의/
│   ├── 주택_및_부동산/
│   └── 코로나19_관련_상담/
│
├── 03_markdown_full/                    # Complete organized markdown (PRIMARY)
│   ├── README.md
│   ├── 건강_의료/
│   ├── 공공서비스/
│   ├── 교통/
│   ├── 보건_복지/
│   ├── 상하수도/
│   ├── 생활환경/
│   ├── 일반행정_문의/
│   └── ... (80+ categories)
│
├── 04_markdown_normalized/              # Normalized subset
│   ├── README.md
│   └── [4 normalized categories]
│
├── 05_consolidated/                    # JSONL consolidated files
│   ├── knowledge_docs.jsonl
│   ├── knowledge_docs_full.jsonl
│   ├── knowledge_docs_normalized.jsonl
│   ├── knowledge_docs_normalized_aggressive.jsonl
│   ├── knowledge_docs_full_metadata.json
│   └── knowledge_docs_metadata.json
│
├── 06_predictions/                     # Model prediction results
│   └── prediction-model-*.jsonl
│
└── README.md                           # Main directory documentation
```

---

## 6. IMPLEMENTATION PHASES

### Phase 1: Code Updates (11 Python Files)

**Files to Update**:
1. `src/batch_processing/create_dasan_batch_input.py` - 4 path references
2. `scripts/prepare_full_dialogues.py` - 1 reference
3. `src/knowledge_extraction/consolidate_extractions.py` - 2 references (input/output)
4. `scripts/build_dasan_vector_store.py` - 2 references (input/output)
5. `src/knowledge_extraction/normalize_topics.py` - 2 references
6. `scripts/extract_transport_data.py` - 1 reference
7. `src/knowledge_extraction/process_predictions.py` - 2 references
8. `src/knowledge_extraction/organize_markdown.py` - 2 references
9. `src/rag_pipeline/unified_benchmark_v4_dasan.py` - 2 references (constants at line 61-62)
10. `src/scripts/data_processing/extract_dasan_data.py` - 6 references
11. `src/rag_pipeline/stages/stage_01_dasan_loader.py` - 3 references

**Total Path Updates**: 45+ instances

### Phase 2: Documentation Updates (5 Files)

1. `docs/topic_normalization_report.md`
2. `docs/knowledge_extraction_workflow.md`
3. `docs/bottom_up_knowledge_extraction_plan.md`
4. `docs/dasan_api_usage.md`
5. `docs/architecture/data_storage_architecture.md`

### Phase 3: Data Directory Reorganization

1. Create new directory structure under `data/AI_HUB_DASAN_QA/`
2. Create subdirectories: `00_raw/`, `01_extracted/`, `02_markdown_basic/`, `03_markdown_full/`, etc.
3. Move/rename files systematically
4. Update vector store paths: `data/vector_store/dasan_test` → `data/vector_store/AI_HUB_DASAN_QA_test`
5. Update processed data: `data/AI_HUB_DASAN_QA/08_processed/eda_artifacts/` → `data/processed/AI_HUB_DASAN_QA_eda/`

### Phase 4: Configuration File Updates

- `.gitignore` (if applicable)
- `config/` YAML files (if any reference AI_HUB_DASAN_QA)
- `README.md` (project-level documentation)

---

## 7. IMPACT ANALYSIS

### 7.1 Scope Assessment

| Category | Count | Risk Level |
|----------|-------|------------|
| Python files affected | 11 | MEDIUM |
| Total code references | 45+ | MEDIUM |
| Documentation files | 5 | LOW |
| Data files | 21,477 | MEDIUM (size) |
| Vector stores | 1 (test) | LOW |
| Config files | TBD | LOW |

### 7.2 Risk Factors

**HIGH CONFIDENCE** (low-risk changes):
- Path string references in Python files (straightforward regex replacement)
- Documentation text updates
- Directory renaming (filesystem operation)

**MEDIUM CONFIDENCE** (requires validation):
- String references in JSONL files or JSON metadata
- Hard-coded metadata values referencing "AI_HUB_DASAN_QA"
- Any dynamically constructed paths using string templates

**DEPENDENCIES TO VERIFY**:
- Are any scripts running in background jobs that cache paths?
- Are there any Docker volumes or mounted paths referencing `AI_HUB_DASAN_QA`?
- Are there any database records storing paths?

---

## 8. MIGRATION STRATEGY

### Option A: Safe Sequential Migration (Recommended)

1. ✅ Code changes (all Python files)
2. ✅ Documentation updates
3. ✅ Test with new paths before moving data
4. ✅ Create new directory structure
5. ✅ Copy (not move) data to new location
6. ✅ Run validation checks
7. ✅ Update vector stores
8. ✅ Delete old directories (keep backup)

**Duration**: ~2-3 hours  
**Risk**: LOW (reversible until deletion step)

### Option B: Quick In-Place Rename

1. Git commit current state
2. Rename directories
3. Update all code paths
4. Run tests
5. If fails: git revert

**Duration**: ~30 minutes  
**Risk**: MEDIUM (requires successful test pass)

---

## 9. VALIDATION CHECKLIST

Post-migration verification:

- [ ] All Python scripts run without import/path errors
- [ ] Vector store builds successfully from new paths
- [ ] RAG benchmark runs with new data locations
- [ ] All 21,477 data files present in new location
- [ ] JSONL files parse correctly from new paths
- [ ] Documentation builds/renders correctly
- [ ] Git history preserved (if using version control)
- [ ] No dangling references in code comments
- [ ] Environment variables updated (if any)

---

## 10. FILES TO CREATE/MODIFY

### Summary Table

| Action | Count | Files |
|--------|-------|-------|
| Code Updates | 11 | Python files with path references |
| Documentation Updates | 5 | Markdown documentation files |
| Directory Operations | 1 | Main data reorganization |
| Configuration | TBD | Depends on additional discovery |

---

## Appendix A: Full Path Reference Listing

### Extract-Related Paths
- `data/AI_HUB_DASAN_QA/00_raw/` → Raw downloaded data
- `data/AI_HUB_DASAN_QA/01_extracted/training/labeled/` → Extracted training data
- `data/AI_HUB_DASAN_QA/01_extracted/validation/labeled/` → Extracted validation data

### Markdown Organization Paths
- `data/AI_HUB_DASAN_QA/02_markdown_basic/` → Basic organized markdown
- `data/AI_HUB_DASAN_QA/03_markdown_full/` → Complete organized markdown (PRIMARY)
- `data/AI_HUB_DASAN_QA/04_markdown_normalized/` → Normalized subset

### Consolidated Data Paths
- `data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs.jsonl` → Basic consolidated JSONL
- `data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs_full.jsonl` → Full consolidated JSONL
- `data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs_normalized.jsonl` → Normalized JSONL
- `data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs_normalized_aggressive.jsonl` → Aggressively normalized
- `data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs_*.json` → Metadata files

### Prediction Paths
- `data/AI_HUB_DASAN_QA/06_predictions/prediction-model-*.jsonl` → Model predictions

### Vector Store Paths
- `data/AI_HUB_DASAN_QA/07_vector_stores/full/` → Main vector store (not yet built)
- `data/vector_store/dasan_test/` → Test vector store (currently exists)

### Processed Data Paths
- `data/AI_HUB_DASAN_QA/08_processed/eda_artifacts/` → EDA analysis outputs

---

**END OF ANALYSIS**

Generated: 2025-11-11 | Status: Ready for Implementation Planning
