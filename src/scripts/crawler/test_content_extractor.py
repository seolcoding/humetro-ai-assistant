#!/usr/bin/env python3
"""
Test script for config-driven ContentExtractor

Validates YAML configuration integration and extraction functionality.
"""

import asyncio
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.crawler.content_extractor import ContentExtractor

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_config_loading():
    """Test 1: Verify YAML config loads correctly"""
    print("\n" + "="*60)
    print("TEST 1: YAML Configuration Loading")
    print("="*60)

    try:
        extractor = ContentExtractor.from_domain(
            domain="news.seoul.go.kr",
            output_dir=Path("./data/test/crawled"),
            download_dir=Path("./data/test/downloads"),
            headless=True,
            verbose=False
        )

        print(f"✓ Extractor initialized successfully")
        print(f"  Site: {extractor.site_config.site_name}")
        print(f"  Domain: {extractor.site_config.domain}")
        print(f"  Base URL: {extractor.site_config.base_url}")
        print(f"  Max depth: {extractor.site_config.crawl_rules.get('max_depth', 'N/A')}")
        print(f"  Article patterns: {extractor.site_config.url_patterns.get('article_patterns', [])[:2]}")

        return True
    except Exception as e:
        print(f"✗ Failed to load config: {e}")
        return False


async def test_single_page_extraction():
    """Test 2: Extract a single test page"""
    print("\n" + "="*60)
    print("TEST 2: Single Page Extraction")
    print("="*60)

    try:
        extractor = ContentExtractor.from_domain(
            domain="news.seoul.go.kr",
            output_dir=Path("./data/test/crawled"),
            download_dir=Path("./data/test/downloads"),
            headless=True,
            verbose=False
        )

        # Test with a known URL (modify to actual article URL)
        test_url = "https://news.seoul.go.kr/traffic"

        url_tree = {test_url: {"parent_url": None, "depth": 0}}
        urls = [{"url": test_url, "depth": 0}]

        print(f"Extracting: {test_url}")
        results = await extractor.extract_with_metadata(
            urls=urls,
            url_tree=url_tree,
            batch_size=1,
            skip_existing=False
        )

        if results:
            metadata = results[0]
            print(f"✓ Extraction successful")
            print(f"  Title: {metadata.title}")
            print(f"  Page type: {metadata.page_type}")
            print(f"  Word count: {metadata.word_count}")
            print(f"  Attachments: {metadata.attachment_count}")
            print(f"  Output dir: {extractor.output_dir}")

            # Check files were saved
            filename = extractor._get_filename_from_url(test_url)
            html_exists = (extractor.raw_html_dir / f"{filename}.html").exists()
            md_exists = (extractor.markdown_dir / f"{filename}.md").exists()
            json_exists = (extractor.metadata_dir / f"{filename}.json").exists()

            print(f"  Files saved: HTML={html_exists}, MD={md_exists}, JSON={json_exists}")

            return True
        else:
            print(f"✗ No results returned")
            return False

    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_cache_functionality():
    """Test 3: Verify caching works (skip_existing)"""
    print("\n" + "="*60)
    print("TEST 3: Cache Functionality")
    print("="*60)

    try:
        extractor = ContentExtractor.from_domain(
            domain="news.seoul.go.kr",
            output_dir=Path("./data/test/crawled"),
            download_dir=Path("./data/test/downloads"),
            headless=True,
            verbose=False
        )

        test_url = "https://news.seoul.go.kr/traffic"
        url_tree = {test_url: {"parent_url": None, "depth": 0}}
        urls = [{"url": test_url, "depth": 0}]

        # First extraction (should hit server)
        print("First extraction (fresh)...")
        results1 = await extractor.extract_with_metadata(
            urls=urls,
            url_tree=url_tree,
            skip_existing=False
        )

        # Second extraction (should use cache)
        print("Second extraction (should use cache)...")
        results2 = await extractor.extract_with_metadata(
            urls=urls,
            url_tree=url_tree,
            skip_existing=True
        )

        if results1 and results2:
            print(f"✓ Cache test passed")
            print(f"  First extraction: {results1[0].title}")
            print(f"  Second extraction: {results2[0].title}")
            print(f"  Titles match: {results1[0].title == results2[0].title}")
            return True
        else:
            print(f"✗ Cache test failed")
            return False

    except Exception as e:
        print(f"✗ Cache test failed: {e}")
        return False


async def test_deep_crawl_discovery():
    """Test 4: Test deep crawl URL discovery (limited)"""
    print("\n" + "="*60)
    print("TEST 4: Deep Crawl URL Discovery")
    print("="*60)

    try:
        extractor = ContentExtractor.from_domain(
            domain="news.seoul.go.kr",
            output_dir=Path("./data/test/crawled"),
            download_dir=Path("./data/test/downloads"),
            headless=True,
            verbose=False
        )

        # Test with limited max_pages
        start_url = "https://news.seoul.go.kr/traffic"
        print(f"Starting deep crawl from: {start_url}")
        print(f"Max pages: 5 (test limit)")

        results = await extractor.extract_with_deep_crawl(
            start_url=start_url,
            max_pages=5,  # Limit for testing
            batch_size=2,
            skip_existing=True
        )

        print(f"✓ Deep crawl completed")
        print(f"  Pages discovered: {len(results)}")
        if results:
            print(f"  Sample titles:")
            for meta in results[:3]:
                print(f"    - {meta.title}")

        return True

    except Exception as e:
        print(f"✗ Deep crawl failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("Content Extractor Test Suite")
    print("Testing YAML config integration")
    print("="*60)

    results = {}

    # Run tests
    results['config_loading'] = await test_config_loading()
    results['single_page'] = await test_single_page_extraction()
    results['caching'] = await test_cache_functionality()
    results['deep_crawl'] = await test_deep_crawl_discovery()

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:20s}: {status}")

    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")

    return all(results.values())


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
