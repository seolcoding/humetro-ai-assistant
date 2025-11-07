# Scripts Directory

Utility scripts organized by module/functionality.

## Directory Structure

```
scripts/
├── crawler/          # Web crawler and content extraction
├── dasan_api/        # Dasan call center API interaction
├── data_processing/  # Data cleaning and preprocessing
├── rag_pipeline/     # RAG pipeline: embedding, vector store
├── experiments/      # Experiments and analysis
└── deprecated/       # Deprecated scripts (not in use)
```

## Module Descriptions

### 📡 crawler/
Web crawling and content extraction scripts.

- `test_content_extractor.py` - Test content extraction functionality
- `discover_urls.py` - Discover URLs for crawling
- `discover_narrow_range.py` - Narrow range URL discovery

### 🏢 dasan_api/
Scripts for interacting with Dasan call center API.

- `discover_dasan_range.py` - Discover available data ranges
- `fetch_*.py` - Various data fetching strategies
- `scan_dasan_sequences.py` - Scan for valid sequence ranges

### 🔧 data_processing/
Data cleaning, deduplication, and preprocessing.

- `clean_and_analyze_markdown.py` - Clean markdown files and generate statistics
- `deduplicate_markdown.py` - Remove duplicate markdown files
- `extract_dasan_data.py` - Extract Dasan call center archives

### 🤖 rag_pipeline/
RAG pipeline: embedding, vector store management, testing.

**Production Scripts:**
- `embed_full_dataset.py` - Embed full dataset (1,969 docs) to FAISS
- `verify_vector_store.py` - Verify vector store integrity
- `test_vector_store_queries.py` - Test with realistic queries

**Development Scripts:**
- `test_embedding_sample.py` - Test with 5 sample documents

### 🧪 experiments/
Experiment orchestration and analysis.

- `run_experiment.py` - Run experiments
- `analyze_results.py` - Analyze experiment results

### 🗑️ deprecated/
Old scripts no longer in use (kept for reference).

## Usage Examples

### RAG Pipeline

```bash
# 1. Embed full dataset (requires user confirmation)
uv run python src/scripts/rag_pipeline/embed_full_dataset.py

# 2. Verify vector store was saved correctly
uv run python src/scripts/rag_pipeline/verify_vector_store.py

# 3. Test with realistic queries
uv run python src/scripts/rag_pipeline/test_vector_store_queries.py
```

### Data Processing

```bash
# Clean and analyze all markdown files
uv run python src/scripts/data_processing/clean_and_analyze_markdown.py

# Deduplicate markdown files
uv run python src/scripts/data_processing/deduplicate_markdown.py
```

### Crawler

```bash
# Test content extractor
uv run python src/scripts/crawler/test_content_extractor.py
```
