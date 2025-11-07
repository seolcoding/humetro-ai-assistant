# Seoul Dasan Call Center API Cache

This directory contains cached data from the Seoul Dasan Call Center Open API.

## Directory Structure

```
dasan_api_cache/
├── faq_detail_*.json           # Individual FAQ detail files
├── workmanual_detail_*.json    # Individual work manual detail files
├── sample_faqs.json            # Consolidated sample FAQs
└── sample_workmanuals.json     # Consolidated sample work manuals
```

## Data Sources

### APIs Used

1. **SearchDetailsFAQService** - FAQ detail endpoint
   - Base URL: `http://openAPI.seoul.go.kr:8088/{KEY}/json/SearchDetailsFAQService/1/1/F/{FAQ_SEQNO}/`
   - Returns detailed FAQ information including question and answer

2. **SearchDetailsSeoulWorkmanualService** - Work manual detail endpoint
   - Base URL: `http://openAPI.seoul.go.kr:8088/{KEY}/json/SearchDetailsSeoulWorkmanualService/1/1/S/{FAQ_SEQNO}/`
   - Returns detailed work manual information including procedures and guidelines

## Data Format

### FAQ Data Structure

```json
{
  "FAQ_TP": "F",
  "FAQ_SEQNO": 289803,
  "QUEST": "Question text in Korean",
  "ANSWER": "Answer text in Korean",
  "UPDATE_YMDHMS": "20181122214218"
}
```

### Work Manual Data Structure

```json
{
  "FAQ_TP": "S",
  "FAQ_SEQNO": 289756,
  "QUEST": "Work manual title",
  "ANSWER": "Work manual content (plain text)",
  "ANSWER_HTML": "Work manual content (HTML formatted)",
  "UPDATE_YMDHMS": "20200129102726"
}
```

## Usage

This cached data can be used for:

1. **RAG Systems** - Building retrieval-augmented generation systems
2. **Knowledge Bases** - Creating searchable knowledge bases
3. **Training Data** - Fine-tuning language models
4. **Analysis** - Analyzing common citizen inquiries and government responses

## Fetching More Data

To fetch additional data, use the [fetch_dasan_api_data.py](../../scripts/fetch_dasan_api_data.py) script:

```bash
# Fetch sample data (default)
uv run python scripts/fetch_dasan_api_data.py

# Or use the client programmatically
from scripts.fetch_dasan_api_data import DasanAPIClient

client = DasanAPIClient(api_key="your_api_key")

# Fetch specific FAQ
faq = client.fetch_faq_detail("289803")

# Fetch multiple FAQs
faqs = client.fetch_multiple_faqs(["289803", "289801"])

# Discover valid sequence numbers
valid_seqs = client.discover_sequence_numbers("faq", start_seq=1, end_seq=300000, sample_interval=1000)
```

## Notes

- The API requires specific sequence numbers to fetch data
- Not all sequence numbers contain valid data
- The `discover_sequence_numbers` method can help find valid entries
- Data is cached locally to minimize API calls
- The API provides "detail" endpoints only (no list endpoint available)

## Data Updates

- Last updated: 2025-10-27
- Update frequency: On-demand (run the fetch script when needed)
- Source data last modified: 2020-04-03 (according to API metadata)
