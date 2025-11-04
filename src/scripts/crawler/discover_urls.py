"""
URL Discovery Script for Seoul Traffic News

Discovers actual article URLs from category/list pages instead of guessing IDs.
"""
import asyncio
from pathlib import Path
from typing import List, Set
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from bs4 import BeautifulSoup
import json


async def discover_article_urls(
    start_url: str,
    max_pages: int = 1000
) -> List[str]:
    """
    Discover article URLs from category/list pages.

    Args:
        start_url: Starting category/list URL
        max_pages: Maximum number of articles to discover

    Returns:
        List of discovered article URLs
    """
    discovered_urls: Set[str] = set()

    browser_config = BrowserConfig(
        headless=True,
        verbose=False
    )

    crawler_config = CrawlerRunConfig(
        word_count_threshold=1,
        exclude_external_links=True,
        process_iframes=False,
        screenshot=False
    )

    print(f"\n🔍 Discovering URLs from: {start_url}")
    print(f"   Target: {max_pages} articles\n")

    async with AsyncWebCrawler(config=browser_config) as crawler:
        # Crawl the category page
        result = await crawler.arun(url=start_url, config=crawler_config)

        if not result.success:
            print(f"❌ Failed to crawl: {result.error_message}")
            return []

        soup = BeautifulSoup(result.html, 'html.parser')

        # Find article links - pattern: /traffic/archives/[숫자]
        for link in soup.find_all('a', href=True):
            href = link['href']

            # Match article URLs
            if '/traffic/archives/' in href and href.split('/')[-1].isdigit():
                # Normalize URL
                if href.startswith('http'):
                    full_url = href
                elif href.startswith('/'):
                    full_url = f"https://news.seoul.go.kr{href}"
                else:
                    full_url = f"https://news.seoul.go.kr/traffic/{href}"

                discovered_urls.add(full_url)

                if len(discovered_urls) >= max_pages:
                    break

        print(f"✓ Found {len(discovered_urls)} article URLs from first page")

        # Find pagination links to discover more
        pagination_links = soup.select('.pagination a, .paging a, .page-numbers a')

        if pagination_links and len(discovered_urls) < max_pages:
            print(f"\n📄 Found {len(pagination_links)} pagination links, crawling...")

            for i, page_link in enumerate(pagination_links[:20], 1):  # Limit to 20 pages
                page_href = page_link.get('href')
                if not page_href or '#' in page_href:
                    continue

                # Normalize pagination URL
                if page_href.startswith('http'):
                    page_url = page_href
                elif page_href.startswith('/'):
                    page_url = f"https://news.seoul.go.kr{page_href}"
                else:
                    page_url = f"https://news.seoul.go.kr/traffic/{page_href}"

                # Crawl pagination page
                page_result = await crawler.arun(url=page_url, config=crawler_config)

                if page_result.success:
                    page_soup = BeautifulSoup(page_result.html, 'html.parser')

                    # Find article links
                    for link in page_soup.find_all('a', href=True):
                        href = link['href']

                        if '/traffic/archives/' in href and href.split('/')[-1].isdigit():
                            if href.startswith('http'):
                                full_url = href
                            elif href.startswith('/'):
                                full_url = f"https://news.seoul.go.kr{href}"
                            else:
                                full_url = f"https://news.seoul.go.kr/traffic/{href}"

                            discovered_urls.add(full_url)

                            if len(discovered_urls) >= max_pages:
                                break

                    print(f"   Page {i}: {len(discovered_urls)} total URLs")

                    if len(discovered_urls) >= max_pages:
                        break

                # Delay between pages
                await asyncio.sleep(1.5)

    return sorted(list(discovered_urls), reverse=True)  # Sort by ID descending


async def main():
    """Main discovery workflow"""

    # Start from main category page
    start_url = "https://news.seoul.go.kr/traffic/archives/category/illegel-news_c1"

    # Discover URLs
    discovered = await discover_article_urls(
        start_url=start_url,
        max_pages=1000
    )

    print(f"\n" + "=" * 60)
    print(f"📊 Discovery Complete")
    print(f"=" * 60)
    print(f"   Total URLs: {len(discovered)}")

    if discovered:
        print(f"\n   Sample URLs (first 5):")
        for url in discovered[:5]:
            print(f"      {url}")

        # Save to file
        output_file = Path("data/discovered_urls.json")
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(discovered, f, indent=2, ensure_ascii=False)

        print(f"\n   ✅ URLs saved to: {output_file}")
        print(f"\n💡 Next: Use these URLs with test_crawl.py")
    else:
        print("\n   ❌ No URLs discovered")


if __name__ == "__main__":
    asyncio.run(main())
