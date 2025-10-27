# Seoul Dasan Call Center API Usage Guide

## Overview

This guide explains how to fetch and cache data from the Seoul Dasan Call Center Open API for use in RAG systems and knowledge bases.

## API Structure

### Endpoints

**Detail API**: `SearchDetailsFAQService` and `SearchDetailsSeoulWorkmanualService`

```
http://openAPI.seoul.go.kr:8088/{KEY}/json/{SERVICE}/1/1/{FAQ_TP}/{SEQ_NO}/
```

### FAQ Types (FAQ_TP)

| Code | Type | Description |
|------|------|-------------|
| `F` | FAQ | 자주묻는질문 (Frequently Asked Questions) |
| `S` | Seoul Manual | 서울시 업무매뉴얼 (Seoul City Work Manuals) |
| `J` | District Manual | 자치구 업무매뉴얼 (District Work Manuals) |

### Data Format

#### FAQ Response
```json
{
  "SearchDetailsFAQService": {
    "list_total_count": 1,
    "RESULT": {
      "CODE": "INFO-000",
      "MESSAGE": "정상 처리되었습니다"
    },
    "row": [{
      "FAQ_TP": "F",
      "FAQ_SEQNO": 289803,
      "QUEST": "광진환경백서는 몇년 주기로 발간됩니까?",
      "ANSWER": "광진 환경백서는 3년 마다 발간됩니다...",
      "UPDATE_YMDHMS": "20181122214218"
    }]
  }
}
```

#### Work Manual Response
```json
{
  "SearchDetailsSeoulWorkmanualService": {
    "list_total_count": 1,
    "RESULT": {
      "CODE": "INFO-000",
      "MESSAGE": "정상 처리되었습니다"
    },
    "row": [{
      "FAQ_TP": "S",
      "FAQ_SEQNO": 289756,
      "QUEST": "시월 정동축제",
      "ANSWER": "업무개요\n...",
      "ANSWER_HTML": "<p>업무개요</p>...",
      "UPDATE_YMDHMS": "20200129102726"
    }]
  }
}
```

## Scripts

### 1. `fetch_dasan_api_data.py`

Basic script for fetching specific FAQ or Work Manual details.

```bash
# Run with default sample sequences
uv run python scripts/fetch_dasan_api_data.py
```

**Features:**
- Fetches sample data from known sequences
- Caches responses locally
- Handles rate limiting
- Retry logic for failed requests

### 2. `fetch_all_dasan_types.py`

Complete script that scans sequence ranges and fetches all three data types.

```bash
# Quick scan (smaller range for testing)
uv run python scripts/fetch_all_dasan_types.py --quick

# Full scan from sequence 1 to 400000
uv run python scripts/fetch_all_dasan_types.py --start 1 --end 400000

# Fetch specific types only
uv run python scripts/fetch_all_dasan_types.py --types faq seoul_manual

# Custom range
uv run python scripts/fetch_all_dasan_types.py --start 280000 --end 300000
```

**Features:**
- Scans ranges of sequence numbers to find valid entries
- Parallel scanning with configurable workers
- Fetches all three FAQ types (F, S, J)
- Caches both sequences and detailed data
- Progress indicators and rate limiting

### 3. `scan_dasan_sequences.py`

Utility for discovering valid sequence numbers.

**Features:**
- Binary search to find valid sequence ranges
- Parallel sequence checking
- Smart scanning strategies

## Data Organization

```
data/dasan_api_cache/
├── faq_detail_{seq_no}.json              # Individual FAQ details
├── seoul_manual_detail_{seq_no}.json     # Individual Seoul manual details
├── district_manual_detail_{seq_no}.json  # Individual district manual details
├── faq_{scan_name}_sequences.json        # List of valid FAQ sequences
├── seoul_manual_{scan_name}_sequences.json
├── district_manual_{scan_name}_sequences.json
├── faq_{scan_name}_all.json              # Consolidated FAQ data
├── seoul_manual_{scan_name}_all.json
└── district_manual_{scan_name}_all.json
```

## Usage Examples

### Fetch Sample Data

```bash
# Fetch sample FAQs and manuals
uv run python scripts/fetch_dasan_api_data.py
```

### Scan and Fetch Complete Dataset

```bash
# Scan a specific range and fetch all data types
uv run python scripts/fetch_all_dasan_types.py --start 1 --end 500000
```

### Programmatic Usage

```python
from scripts.fetch_all_dasan_types import CompleteDasanFetcher

# Initialize
fetcher = CompleteDasanFetcher(api_key="your_api_key")

# Scan for FAQs in a range
faq_sequences = fetcher.scan_range("faq", start=1, end=10000)

# Fetch details
faq_details = fetcher.fetch_all_details("faq", faq_sequences)

# Save data
fetcher.save_data("faq", faq_sequences, faq_details, "my_scan")
```

## Environment Setup

1. Create `.env` file:
```bash
SEOUL_DATA_API_KEY=your_api_key_here
```

2. Install dependencies:
```bash
uv add python-dotenv requests types-requests
```

## API Limitations

- **No List Endpoint**: The API only provides detail endpoints, not list endpoints
- **Sequence-Based**: Must know or discover sequence numbers
- **Rate Limiting**: Implement delays between requests (0.2-1.0 seconds recommended)
- **Max Workers**: Use 5-10 parallel workers for optimal performance
- **Response Codes**:
  - `INFO-000`: Success
  - `INFO-200`: No data found for sequence
  - `ERROR-*`: Various error conditions

## Data Collection Strategy

1. **Sequential Scanning**: Start from sequence 1 and scan upwards
2. **Range-Based**: Focus on ranges with known valid sequences (e.g., 280000-300000)
3. **Type-Specific**: Fetch each FAQ type separately for better organization
4. **Caching**: Always use cache to avoid redundant API calls

## Next Steps

1. Run a comprehensive scan across all sequence ranges
2. Build a RAG system using the cached data
3. Create embeddings for semantic search
4. Implement a query interface for the knowledge base

## Related Documentation

- Seoul Open Data Portal: https://data.seoul.go.kr
- API Documentation: Check `docs/api_documentation/`
- Dataset Analysis: See `notebooks/dasan_call_center_eda.ipynb`
