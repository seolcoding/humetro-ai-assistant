"""
Configuration-Driven Content Extractor with crawl4ai

Unified content extraction orchestrator that:
- Uses YAML-based site configuration for flexibility
- Supports deep crawling with BFSDeepCrawlStrategy
- Handles attachments (PDF, HWP, XLSX, etc.) via popup detection
- Provides caching to skip already-crawled pages
- Enables multi-site support through domain-specific configs

Usage:
    # Initialize from domain config
    extractor = ContentExtractor.from_domain(
        domain="news.seoul.go.kr",
        output_dir=Path("./data/crawled"),
        download_dir=Path("./data/downloads")
    )

    # Run deep crawl
    results = await extractor.extract_with_deep_crawl(
        start_url="https://news.seoul.go.kr/traffic/archives/category/all",
        max_pages=200
    )
"""

import asyncio
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from urllib.parse import urlparse
import logging
import aiofiles

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter
from playwright.async_api import Browser, BrowserContext, Page

from src.config.schemas import (
    PageMetadata,
    PageType,
    LinkContext,
    BreadcrumbItem,
    AttachedDocument,
    AttachmentType
)
from src.config.site_config import SiteConfig, load_site_config
from src.crawler.extractors import ConfigBasedExtractor
from src.crawler.extractors.base import ExtractionResult
from src.utils.entity_preview import SimpleEntityExtractor

logger = logging.getLogger(__name__)


class ContentExtractor:
    """
    Configuration-driven content extraction orchestrator.

    Features:
    - YAML-based site configuration (easy to add new sites)
    - Deep crawling with automatic URL discovery (BFS strategy)
    - Rich metadata extraction with tree structure preservation
    - Popup window detection for attached documents
    - Multi-format document parsing (PDF, HWP, XLSX, DOCX)
    - Link context analysis for Knowledge Graph preparation
    - Caching support to skip already-crawled pages
    - Simple entity preview for each page
    """

    def __init__(
        self,
        site_config: SiteConfig,
        output_dir: Path,
        download_dir: Path,
        headless: bool = True,
        verbose: bool = True
    ):
        """
        Initialize content extractor with site configuration.

        Args:
            site_config: Site-specific configuration (loaded from YAML)
            output_dir: Output directory for extracted content
            download_dir: Directory for downloaded attachments
            headless: Run browser in headless mode
            verbose: Enable verbose logging
        """
        self.site_config = site_config
        self.output_dir = output_dir
        self.download_dir = download_dir
        self.headless = headless
        self.verbose = verbose

        # Create necessary directories
        self.raw_html_dir = output_dir / "raw"
        self.markdown_dir = output_dir / "markdown"
        self.metadata_dir = output_dir / "metadata"
        self.attachments_dir = download_dir / "attachments"

        for directory in [
            self.raw_html_dir,
            self.markdown_dir,
            self.metadata_dir,
            self.attachments_dir
        ]:
            directory.mkdir(parents=True, exist_ok=True)

        # Initialize extractors
        self.extractor = ConfigBasedExtractor(site_config)
        self.entity_extractor = SimpleEntityExtractor()

        logger.info(
            f"ContentExtractor initialized for {site_config.site_name} "
            f"with output_dir={output_dir}"
        )

    @classmethod
    def from_domain(
        cls,
        domain: str,
        output_dir: Path,
        download_dir: Path,
        **kwargs
    ) -> "ContentExtractor":
        """
        Create extractor by loading configuration for a domain.

        Args:
            domain: Domain name (e.g., 'news.seoul.go.kr')
            output_dir: Output directory for extracted content
            download_dir: Directory for downloaded attachments
            **kwargs: Additional arguments for __init__

        Returns:
            ContentExtractor instance

        Example:
            extractor = ContentExtractor.from_domain(
                domain="news.seoul.go.kr",
                output_dir=Path("./data/crawled"),
                download_dir=Path("./data/downloads")
            )
        """
        site_config = load_site_config(domain)
        return cls(
            site_config=site_config,
            output_dir=output_dir,
            download_dir=download_dir,
            **kwargs
        )

    def _get_filename_from_url(self, url: str) -> str:
        """Generate filename from URL."""
        parsed = urlparse(url)
        return parsed.path.strip('/').replace('/', '_') or 'index'

    def _is_page_cached(self, url: str) -> bool:
        """
        Check if a page has already been crawled.

        Args:
            url: Page URL to check

        Returns:
            True if metadata file exists, False otherwise
        """
        filename = self._get_filename_from_url(url)
        metadata_path = self.metadata_dir / f"{filename}.json"
        return metadata_path.exists()

    def _load_cached_metadata(self, url: str) -> Optional[PageMetadata]:
        """
        Load cached metadata for a URL.

        Args:
            url: Page URL to load

        Returns:
            PageMetadata if cached, None otherwise
        """
        filename = self._get_filename_from_url(url)
        metadata_path = self.metadata_dir / f"{filename}.json"

        if not metadata_path.exists():
            return None

        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return PageMetadata(**data)
        except Exception as e:
            logger.warning(f"Failed to load cached metadata for {url}: {e}")
            return None

    async def extract_with_deep_crawl(
        self,
        start_url: str,
        max_pages: int = 200,
        batch_size: int = 10,
        delay_between_batches: float = 2.0,
        skip_existing: bool = True
    ) -> List[PageMetadata]:
        """
        Extract content using deep crawling from a start URL.

        Uses BFSDeepCrawlStrategy to automatically discover article URLs,
        then extracts content from discovered pages.

        Args:
            start_url: Starting URL (category/list page)
            max_pages: Maximum number of pages to crawl
            batch_size: Number of URLs to process in parallel
            delay_between_batches: Delay in seconds between batches
            skip_existing: If True, skip URLs that have already been crawled (default: True)

        Returns:
            List of PageMetadata objects with complete information

        Example:
            results = await extractor.extract_with_deep_crawl(
                start_url="https://news.seoul.go.kr/traffic/archives/category/all",
                max_pages=200,
                skip_existing=True
            )
        """
        # URL pattern filter based on site config
        url_patterns = self.site_config.url_patterns.get('article_patterns', [])
        if not url_patterns:
            # Fallback: use domain-based filter
            url_patterns = [f"*{self.site_config.domain}/*"]

        url_filter = URLPatternFilter(patterns=url_patterns)

        # Deep crawl strategy
        max_depth = self.site_config.crawl_rules.get('max_depth', 4)
        deep_crawl_strategy = BFSDeepCrawlStrategy(
            max_depth=max_depth,
            include_external=False,
            max_pages=max_pages,
            filter_chain=FilterChain([url_filter])
        )

        # Build URL tree from start
        url_tree = {start_url: {"parent_url": None, "depth": 0}}

        logger.info(f"🔍 Starting deep crawl from: {start_url}")
        logger.info(f"   Max pages: {max_pages}, Batch size: {batch_size}")

        # Configure crawler with deep crawl strategy
        browser_config = BrowserConfig(
            headless=self.headless,
            verbose=self.verbose,
            extra_args=["--disable-blink-features=AutomationControlled"],
            downloads_path=str(self.download_dir)
        )

        # Build excluded elements from config
        excluded_elements = self._build_excluded_selectors()

        crawler_config = CrawlerRunConfig(
            word_count_threshold=10,
            excluded_tags=['nav', 'footer', 'aside'],
            excluded_selector=', '.join(excluded_elements) if excluded_elements else None,
            exclude_external_links=True,
            process_iframes=True,
            remove_overlay_elements=True,
            screenshot=False,
            deep_crawl_strategy=deep_crawl_strategy,
            cache_mode="enabled"
        )

        # Run deep crawl
        async with AsyncWebCrawler(config=browser_config) as crawler:
            logger.info("🕷️  Running deep crawl...")
            result = await crawler.arun(url=start_url, config=crawler_config)

            # Extract discovered URLs from deep crawl results
            crawl_results = result if isinstance(result, list) else [result]

            discovered_urls = []
            for crawl_result in crawl_results:
                if crawl_result.success:
                    discovered_urls.append(crawl_result.url)
                    url_tree[crawl_result.url] = {
                        "parent_url": start_url,
                        "depth": 1
                    }

            logger.info(f"✓ Discovered {len(discovered_urls)} article URLs")

        # Convert to URL dict format
        url_dicts = [{"url": url, "depth": 1} for url in discovered_urls]

        # Extract content from discovered URLs
        return await self.extract_with_metadata(
            urls=url_dicts,
            url_tree=url_tree,
            batch_size=batch_size,
            delay_between_batches=delay_between_batches,
            skip_existing=skip_existing
        )

    async def extract_with_metadata(
        self,
        urls: List[Dict],
        url_tree: Dict[str, Dict],
        batch_size: int = 5,
        delay_between_batches: float = 2.0,
        skip_existing: bool = True
    ) -> List[PageMetadata]:
        """
        Extract content from URLs with rich metadata.

        Args:
            urls: List of URL dictionaries (each with 'url' and 'depth' keys)
            url_tree: Tree structure metadata
            batch_size: Number of URLs to process in parallel
            delay_between_batches: Delay in seconds between batches
            skip_existing: If True, skip URLs that have already been crawled

        Returns:
            List of PageMetadata objects
        """
        all_metadata: List[PageMetadata] = []

        # Browser configuration
        browser_config = BrowserConfig(
            headless=self.headless,
            verbose=self.verbose,
            extra_args=["--disable-blink-features=AutomationControlled"],
            downloads_path=str(self.download_dir)
        )

        # Build excluded elements from config
        excluded_elements = self._build_excluded_selectors()

        crawler_config = CrawlerRunConfig(
            word_count_threshold=10,
            excluded_tags=['nav', 'footer', 'aside'],
            excluded_selector=', '.join(excluded_elements) if excluded_elements else None,
            exclude_external_links=True,
            process_iframes=True,
            remove_overlay_elements=True,
            screenshot=False,
            cache_mode="enabled"
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            # Process URLs in batches
            for i in range(0, len(urls), batch_size):
                batch = urls[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}: {len(batch)} URLs")

                # Filter batch: check cache if skip_existing is enabled
                urls_to_crawl = []
                cached_count = 0
                for url_info in batch:
                    url = url_info['url']

                    if skip_existing and self._is_page_cached(url):
                        # Load from cache
                        cached_metadata = self._load_cached_metadata(url)
                        if cached_metadata:
                            all_metadata.append(cached_metadata)
                            cached_count += 1
                            logger.info(f"⚡ Cached: {url}")
                            continue

                    urls_to_crawl.append(url_info)

                if cached_count > 0:
                    logger.info(f"Loaded {cached_count} pages from cache")

                # Extract batch in parallel
                if urls_to_crawl:
                    batch_results = await asyncio.gather(
                        *[
                            self._extract_single_page(
                                crawler,
                                url_info,
                                url_tree,
                                crawler_config
                            )
                            for url_info in urls_to_crawl
                        ],
                        return_exceptions=True
                    )

                    # Collect successful results
                    for result in batch_results:
                        if isinstance(result, PageMetadata):
                            all_metadata.append(result)
                            logger.info(f"✓ Extracted: {result.url}")
                        elif isinstance(result, Exception):
                            logger.error(f"✗ Extraction failed: {result}")

                # Delay between batches
                if i + batch_size < len(urls):
                    await asyncio.sleep(delay_between_batches)

        logger.info(f"Extraction complete: {len(all_metadata)}/{len(urls)} pages")
        return all_metadata

    async def _extract_single_page(
        self,
        crawler: AsyncWebCrawler,
        url_info: Dict,
        url_tree: Dict[str, Dict],
        config: CrawlerRunConfig
    ) -> PageMetadata:
        """
        Extract content and metadata from a single page using ConfigBasedExtractor.

        Args:
            crawler: AsyncWebCrawler instance
            url_info: URL information dict
            url_tree: URL tree structure
            config: Crawler configuration

        Returns:
            PageMetadata object
        """
        url = url_info['url']
        logger.debug(f"Extracting: {url}")

        try:
            # Step 1: Crawl the page
            result = await crawler.arun(url=url, config=config)

            if not result.success:
                raise ValueError(f"Crawl failed: {result.error_message}")

            html = result.html
            markdown = result.markdown_v2.raw_markdown  # Use markdown_v2 for better quality
            links = result.links.get('internal', [])

            # Step 2: Extract structured content using ConfigBasedExtractor
            extraction_result = self.extractor.extract(html, url)

            # Step 3: Build metadata
            metadata = await self._build_metadata(
                url=url,
                html=html,
                markdown=markdown,
                extraction_result=extraction_result,
                links=links,
                url_tree=url_tree
            )

            # Step 4: Save files
            await self._save_page_files(
                metadata=metadata,
                html=html,
                markdown=markdown
            )

            return metadata

        except Exception as e:
            logger.error(f"Failed to extract {url}: {e}")
            raise

    async def _build_metadata(
        self,
        url: str,
        html: str,
        markdown: str,
        extraction_result: ExtractionResult,
        links: List[str],
        url_tree: Dict[str, Dict]
    ) -> PageMetadata:
        """
        Build comprehensive PageMetadata object.

        Args:
            url: Page URL
            html: Raw HTML content
            markdown: Markdown content
            extraction_result: Structured extraction result from ConfigBasedExtractor
            links: List of internal links
            url_tree: URL tree structure

        Returns:
            PageMetadata object
        """
        # Use extracted data from ConfigBasedExtractor
        title = extraction_result.title or ""
        page_type = extraction_result.page_type

        # Tree structure info
        parent_url = url_tree.get(url, {}).get('parent_url')
        depth = url_tree.get(url, {}).get('depth', 0)

        # Breadcrumb (simplified - can be enhanced with config-based extraction)
        breadcrumb = extraction_result.breadcrumb or []

        # Link contexts
        link_contexts = extraction_result.link_contexts or []

        # Outgoing links
        outgoing_links = [link for link in links if link.startswith('http')]

        # Entity preview from main content
        main_text = markdown[:5000]  # Limit for performance
        entities_preview = self.entity_extractor.extract_preview(main_text)
        named_entities = self.entity_extractor.extract_named_entities(main_text)

        # Attachment info from extraction result
        attached_documents = extraction_result.attachments or []
        has_attachments = len(attached_documents) > 0
        attachment_types = list(set(doc.attachment_type for doc in attached_documents))
        total_attachment_size = sum(doc.file_size for doc in attached_documents)

        # Combine entities from attachments
        for doc in attached_documents:
            named_entities.setdefault('DOCUMENT_ENTITIES', []).extend(
                doc.entities_preview
            )

        return PageMetadata(
            url=url,
            title=title,
            page_type=page_type,
            crawled_at=datetime.utcnow(),
            parent_url=parent_url,
            depth=depth,
            breadcrumb=breadcrumb,
            siblings=[],  # Can be populated in post-processing
            outgoing_links=outgoing_links,
            incoming_links=[],  # Will be populated in post-processing
            link_contexts=link_contexts,
            word_count=len(markdown.split()),
            entities_preview=entities_preview,
            named_entities=named_entities,
            attached_documents=attached_documents,
            has_attachments=has_attachments,
            attachment_types=attachment_types,
            total_attachment_size=total_attachment_size,
            attachment_count=len(attached_documents)
        )

    async def _save_page_files(
        self,
        metadata: PageMetadata,
        html: str,
        markdown: str
    ) -> None:
        """
        Save HTML, Markdown, and JSON metadata to disk.

        Args:
            metadata: Page metadata
            html: Raw HTML content
            markdown: Markdown content
        """
        filename = self._get_filename_from_url(metadata.url)

        # Save HTML
        html_path = self.raw_html_dir / f"{filename}.html"
        html_path.write_text(html, encoding='utf-8')

        # Save Markdown
        md_path = self.markdown_dir / f"{filename}.md"
        md_path.write_text(markdown, encoding='utf-8')

        # Save metadata JSON
        json_path = self.metadata_dir / f"{filename}.json"
        json_path.write_text(
            metadata.model_dump_json(indent=2),
            encoding='utf-8'
        )

        logger.debug(f"Saved files for: {filename}")

    def _build_excluded_selectors(self) -> List[str]:
        """
        Build list of excluded CSS selectors from site config.

        Returns:
            List of CSS selectors to exclude
        """
        # Common navigation/menu elements to exclude
        default_excluded = [
            '#head', '#gnb_part', '#gnb_sec', '#search', '.top-area', '.gnb',
            '#skipNavi', '#AllBox', '.gnbsingle', '#lnb', '.lnb_tit',
            '.lnb_1depth', '.lnb_2depth', '#sub_h3', '.loc', '.invisible'
        ]

        # Can be extended with config-specific exclusions in the future
        return default_excluded

    def generate_extraction_report(
        self,
        all_metadata: List[PageMetadata]
    ) -> str:
        """
        Generate a summary report of the extraction process.

        Args:
            all_metadata: List of all extracted page metadata

        Returns:
            Markdown-formatted report string
        """
        from collections import Counter

        # Statistics
        total_pages = len(all_metadata)
        if total_pages == 0:
            return "# Extraction Report\n\nNo pages extracted."

        page_type_counts = Counter(meta.page_type for meta in all_metadata)
        total_attachments = sum(meta.attachment_count for meta in all_metadata)
        attachment_type_counts = Counter()

        for meta in all_metadata:
            for att_type in meta.attachment_types:
                attachment_type_counts[att_type] += 1

        pages_with_attachments = sum(1 for meta in all_metadata if meta.has_attachments)
        total_attachment_size = sum(meta.total_attachment_size for meta in all_metadata)

        # Generate report
        report = f"""
# Content Extraction Report

Site: {self.site_config.site_name}
Domain: {self.site_config.domain}
Generated: {datetime.utcnow().isoformat()}

## Overview
- **Total Pages Extracted**: {total_pages}
- **Pages with Attachments**: {pages_with_attachments} ({pages_with_attachments/total_pages*100:.1f}%)
- **Total Attachments**: {total_attachments}
- **Total Attachment Size**: {total_attachment_size / (1024*1024):.2f} MB

## Page Types Distribution
"""
        for page_type, count in page_type_counts.most_common():
            report += f"- {page_type.value}: {count} ({count/total_pages*100:.1f}%)\n"

        if attachment_type_counts:
            report += "\n## Attachment Types Distribution\n"
            for att_type, count in attachment_type_counts.most_common():
                report += f"- {att_type.value}: {count}\n"

        report += "\n## Sample Pages with Attachments\n"
        pages_with_atts = [m for m in all_metadata if m.has_attachments][:10]
        for meta in pages_with_atts:
            report += f"\n### {meta.title}\n"
            report += f"- URL: {meta.url}\n"
            report += f"- Attachments: {meta.attachment_count}\n"
            for att in meta.attached_documents:
                report += f"  - {att.original_filename} ({att.attachment_type.value}, {att.file_size/1024:.1f} KB)\n"

        return report


# Example usage
async def main():
    """Example usage of ContentExtractor with YAML config"""

    # Initialize from domain config (loads news.seoul.go.kr.yaml)
    extractor = ContentExtractor.from_domain(
        domain="news.seoul.go.kr",
        output_dir=Path("./data/crawled"),
        download_dir=Path("./data/downloads"),
        headless=True,
        verbose=True
    )

    # Run deep crawl
    all_metadata = await extractor.extract_with_deep_crawl(
        start_url="https://news.seoul.go.kr/traffic/archives/category/all",
        max_pages=100,
        batch_size=10,
        skip_existing=True
    )

    # Generate report
    report = extractor.generate_extraction_report(all_metadata)

    report_path = Path("./data/extraction_report.md")
    report_path.write_text(report, encoding='utf-8')

    print(f"✓ Extraction complete: {len(all_metadata)} pages")
    print(f"✓ Report saved to: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
