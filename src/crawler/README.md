# Content Extractor

Configuration-driven web content extraction using crawl4ai.

## Features

- **YAML-based configuration** - Add new sites without code changes
- **Deep crawling** - Automatic URL discovery with BFSDeepCrawlStrategy
- **Caching** - Skip already-crawled pages to save time
- **Multi-site support** - One extractor, multiple domains via configs
- **Rich metadata** - Extracts title, content, attachments, entities

## Quick Start

### 1. Basic Usage

```python
from pathlib import Path
from src.crawler.content_extractor import ContentExtractor

# Initialize from domain config (loads src/config/sites/news.seoul.go.kr.yaml)
extractor = ContentExtractor.from_domain(
    domain="news.seoul.go.kr",
    output_dir=Path("./data/crawled"),
    download_dir=Path("./data/downloads")
)

# Run deep crawl
results = await extractor.extract_with_deep_crawl(
    start_url="https://news.seoul.go.kr/traffic/archives/category/all",
    max_pages=200,
    skip_existing=True  # Use cache for already-crawled pages
)

# Generate report
report = extractor.generate_extraction_report(results)
Path("./data/report.md").write_text(report)
```

### 2. Run Test Suite

```bash
uv run python src/scripts/test_content_extractor.py
```

Tests:
- ✅ YAML config loading
- ✅ Single page extraction
- ✅ Caching functionality
- ✅ Deep crawl URL discovery

## Configuration

### Site Config Structure

Each site has a YAML config in `src/config/sites/<domain>.yaml`:

```yaml
site_name: "서울시 교통 뉴스"
domain: "news.seoul.go.kr"
base_url: "https://news.seoul.go.kr/traffic"

# Article page selectors
article:
  title:
    selector: "#sub_centent h3.atitle"
    required: true
  content:
    selector: "#sub_centent .a_content"
    required: true

# URL patterns
url_patterns:
  article_patterns:
    - ".*/traffic/archives/\\d+$"
  list_patterns:
    - ".*/traffic/archives/category/.*"

# Crawl rules
crawl_rules:
  max_depth: 4
  delay_between_requests: 1.5
```

### Adding a New Site

1. Create `src/config/sites/yourdomain.com.yaml`
2. Define selectors and patterns (see `news.seoul.go.kr.yaml` as example)
3. Use:
   ```python
   extractor = ContentExtractor.from_domain(
       domain="yourdomain.com",
       output_dir=Path("./data/crawled"),
       download_dir=Path("./data/downloads")
   )
   ```

## Output Structure

```
data/
├── crawled/
│   ├── raw/           # Original HTML
│   ├── markdown/      # Converted markdown
│   └── metadata/      # Structured JSON metadata
└── downloads/
    └── attachments/   # Downloaded files (PDF, HWP, etc.)
```

## API Reference

### ContentExtractor

#### `from_domain(domain, output_dir, download_dir, **kwargs)`
Factory method to create extractor from domain config.

**Parameters:**
- `domain` (str): Domain name (e.g., "news.seoul.go.kr")
- `output_dir` (Path): Output directory for extracted content
- `download_dir` (Path): Directory for attachments
- `headless` (bool): Run browser in headless mode (default: True)
- `verbose` (bool): Enable verbose logging (default: True)

**Returns:** `ContentExtractor` instance

---

#### `extract_with_deep_crawl(start_url, max_pages=200, batch_size=10, skip_existing=True)`
Extract content using deep crawling.

**Parameters:**
- `start_url` (str): Starting URL (category/list page)
- `max_pages` (int): Maximum pages to crawl
- `batch_size` (int): Parallel processing batch size
- `delay_between_batches` (float): Delay between batches (seconds)
- `skip_existing` (bool): Skip already-crawled pages

**Returns:** `List[PageMetadata]`

---

#### `extract_with_metadata(urls, url_tree, batch_size=5, skip_existing=True)`
Extract content from specific URLs.

**Parameters:**
- `urls` (List[Dict]): List of URL dicts with 'url' and 'depth' keys
- `url_tree` (Dict): URL tree structure
- `batch_size` (int): Parallel processing batch size
- `skip_existing` (bool): Skip already-crawled pages

**Returns:** `List[PageMetadata]`

---

#### `generate_extraction_report(all_metadata)`
Generate markdown summary report.

**Returns:** `str` (markdown-formatted report)

## Advanced Usage

### Custom Extraction Parameters

```python
# More control over crawling
results = await extractor.extract_with_metadata(
    urls=[
        {"url": "https://example.com/page1", "depth": 1},
        {"url": "https://example.com/page2", "depth": 1}
    ],
    url_tree={
        "https://example.com/page1": {"parent_url": None, "depth": 0},
        "https://example.com/page2": {"parent_url": None, "depth": 0}
    },
    batch_size=10,
    delay_between_batches=1.0,
    skip_existing=False  # Force re-crawl
)
```

### Accessing Extracted Data

```python
for metadata in results:
    print(f"Title: {metadata.title}")
    print(f"URL: {metadata.url}")
    print(f"Word count: {metadata.word_count}")
    print(f"Attachments: {metadata.attachment_count}")

    # Access raw files
    filename = extractor._get_filename_from_url(metadata.url)
    html_path = extractor.raw_html_dir / f"{filename}.html"
    md_path = extractor.markdown_dir / f"{filename}.md"
    json_path = extractor.metadata_dir / f"{filename}.json"
```

## Dependencies

- `crawl4ai` - Web crawling framework
- `playwright` - Browser automation
- `beautifulsoup4` - HTML parsing
- `pydantic` - Data validation (PageMetadata schema)

## Related Modules

- `src/config/site_config.py` - Site configuration loader
- `src/config/schemas.py` - Data schemas (PageMetadata, etc.)
- `src/crawler/extractors/` - Config-based extraction logic
- `src/utils/entity_preview.py` - Simple entity extraction

## Troubleshooting

### "Failed to load config"
- Ensure YAML file exists in `src/config/sites/<domain>.yaml`
- Check YAML syntax (use YAML validator)

### "No pages extracted"
- Verify URL patterns in site config match target pages
- Check `crawl_rules.max_depth` is sufficient
- Enable `verbose=True` for detailed logs

### Playwright browser errors
- Install browsers: `playwright install chromium`
- Check headless mode: try `headless=False` for debugging

## Example: Full Workflow

```python
import asyncio
from pathlib import Path
from src.crawler.content_extractor import ContentExtractor

async def main():
    # 1. Initialize extractor
    extractor = ContentExtractor.from_domain(
        domain="news.seoul.go.kr",
        output_dir=Path("./data/crawled"),
        download_dir=Path("./data/downloads"),
        headless=True
    )

    # 2. Run deep crawl
    results = await extractor.extract_with_deep_crawl(
        start_url="https://news.seoul.go.kr/traffic/archives/category/all",
        max_pages=100,
        skip_existing=True
    )

    # 3. Generate report
    report = extractor.generate_extraction_report(results)
    Path("./data/extraction_report.md").write_text(report)

    print(f"✓ Extracted {len(results)} pages")
    print(f"✓ Report: ./data/extraction_report.md")

if __name__ == "__main__":
    asyncio.run(main())
```
