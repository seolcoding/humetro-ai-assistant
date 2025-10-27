"""
Seoul Traffic News Content Extractor v2 (Configuration-Driven)

Refactored version using ConfigBasedExtractor for flexible, domain-specific extraction.
Maintains backward compatibility with original ContentExtractor API.
"""

import asyncio
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import logging

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
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


class SeoulTrafficContentExtractorV2:
    """
    Configuration-driven content extraction orchestrator.

    Improvements over v1:
    - Uses ConfigBasedExtractor for flexible, YAML-based extraction
    - Easier to add new sites without code changes
    - Separates CSS selectors from core logic
    - Maintains same API and storage structure as v1

    Features:
    - Rich metadata extraction with tree structure preservation
    - Popup window detection for attached documents
    - Multi-format document parsing (PDF, HWP, XLSX, DOCX)
    - Link context analysis for Knowledge Graph preparation
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
            f"ContentExtractorV2 initialized for {site_config.site_name} "
            f"with output_dir={output_dir}"
        )

    @classmethod
    def from_domain(
        cls,
        domain: str,
        output_dir: Path,
        download_dir: Path,
        **kwargs
    ) -> "SeoulTrafficContentExtractorV2":
        """
        Create extractor by loading configuration for a domain.

        Args:
            domain: Domain name (e.g., 'news.seoul.go.kr')
            output_dir: Output directory for extracted content
            download_dir: Directory for downloaded attachments
            **kwargs: Additional arguments for __init__

        Returns:
            SeoulTrafficContentExtractorV2 instance
        """
        site_config = load_site_config(domain)
        return cls(
            site_config=site_config,
            output_dir=output_dir,
            download_dir=download_dir,
            **kwargs
        )

    async def extract_with_metadata(
        self,
        urls: List[Dict],
        url_tree: Dict[str, Dict],
        batch_size: int = 5,
        delay_between_batches: float = 2.0
    ) -> List[PageMetadata]:
        """
        Extract content from URLs with rich metadata including attachments.

        Args:
            urls: List of URL dictionaries from URL discovery
            url_tree: Tree structure metadata from TreeStructureAnalyzer
            batch_size: Number of URLs to process in parallel
            delay_between_batches: Delay in seconds between batches

        Returns:
            List of PageMetadata objects with complete information
        """
        all_metadata: List[PageMetadata] = []

        # Browser configuration for attachment handling
        browser_config = BrowserConfig(
            headless=self.headless,
            verbose=self.verbose,
            extra_args=["--disable-blink-features=AutomationControlled"],
            downloads_path=str(self.download_dir)
        )

        # Crawler configuration
        crawler_config = CrawlerRunConfig(
            word_count_threshold=10,
            excluded_tags=['nav', 'footer', 'aside'],
            exclude_external_links=True,
            process_iframes=True,
            remove_overlay_elements=True,
            screenshot=False
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            # Process URLs in batches
            for i in range(0, len(urls), batch_size):
                batch = urls[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}: {len(batch)} URLs")

                # Extract batch in parallel
                batch_results = await asyncio.gather(
                    *[
                        self._extract_single_page(
                            crawler,
                            url_info,
                            url_tree,
                            crawler_config
                        )
                        for url_info in batch
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
        Extract content and metadata from a single page.

        This method orchestrates:
        1. Page crawling with Crawl4AI
        2. Content extraction using ConfigBasedExtractor
        3. Attachment detection and processing
        4. Metadata enrichment
        5. File saving
        """
        url = url_info['url']
        logger.debug(f"Extracting: {url}")

        try:
            # Step 1: Crawl the page
            result = await crawler.arun(url=url, config=config)

            if not result.success:
                raise ValueError(f"Crawl failed: {result.error_message}")

            html = result.html
            markdown = result.markdown
            links = result.links.get('internal', [])

            # Step 2: Extract content using ConfigBasedExtractor (HTML-only mode)
            # Note: Browser access removed due to crawl4ai 0.7.6 API changes
            # Attachment processing will be re-implemented using hooks later
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, 'html.parser')

            extraction_result = ExtractionResult(url=url, html=html)

            # Extract title
            if self.site_config.article and self.site_config.article.title:
                title_elem = soup.select_one(self.site_config.article.title.selector)
                if title_elem:
                    extraction_result.title = title_elem.get_text(strip=True)

            # Extract content
            if self.site_config.article and self.site_config.article.content:
                content_elem = soup.select_one(self.site_config.article.content.selector)
                if content_elem:
                    extraction_result.content = content_elem.get_text(separator='\n', strip=True)

            # Extract breadcrumb
            if self.site_config.navigation and self.site_config.navigation.breadcrumb:
                extraction_result.breadcrumb_links = []
                for bc_elem in soup.select(self.site_config.navigation.breadcrumb.selector):
                    link = bc_elem.get('href', '')
                    text = bc_elem.get_text(strip=True)
                    if link and text:
                        extraction_result.breadcrumb_links.append({'url': link, 'text': text})

            # Extract outgoing links
            extraction_result.outgoing_links = [
                a.get('href') for a in soup.select('a[href]')
                if a.get('href') and not a.get('href').startswith('#')
            ]

            # Temporary: Skip attachment processing (requires browser access)
            attached_documents = []
            logger.debug("Attachment processing temporarily disabled (requires hook implementation)")

            # Step 5: Build metadata
            metadata = await self._build_metadata(
                url=url,
                html=html,
                markdown=markdown,
                links=links,
                url_tree=url_tree,
                extraction_result=extraction_result,
                attached_documents=attached_documents
            )

            # Step 6: Save files
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
        links: List[str],
        url_tree: Dict[str, Dict],
        extraction_result,
        attached_documents: List[AttachedDocument]
    ) -> PageMetadata:
        """
        Build comprehensive PageMetadata object using extraction results.
        """
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        # Use extracted title or fallback to <title> tag
        title = extraction_result.title
        if not title:
            title = soup.title.string if soup.title else ""

        # Tree structure analysis
        page_type = self._classify_page_type(url, html, title)
        breadcrumb = extraction_result.breadcrumb_links

        # Convert to BreadcrumbItem format
        breadcrumb_items = []
        for i, bc in enumerate(breadcrumb):
            breadcrumb_items.append(
                BreadcrumbItem(
                    url=bc.get('url', ''),
                    title=bc.get('text', ''),
                    depth=i,
                    page_type=PageType.OTHER
                )
            )

        parent_url = url_tree.get(url, {}).get('parent_url')
        depth = url_tree.get(url, {}).get('depth', 0)
        siblings = self._find_siblings(url, url_tree)

        # Outgoing links from extraction result
        outgoing_links = extraction_result.outgoing_links or []

        # Entity preview from main content
        main_text = extraction_result.content[:5000] if extraction_result.content else ""
        entities_preview = self.entity_extractor.extract_preview(main_text)
        named_entities = self.entity_extractor.extract_named_entities(main_text)

        # Attachment summary
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
            breadcrumb=breadcrumb_items,
            siblings=siblings,
            outgoing_links=outgoing_links,
            incoming_links=[],  # Will be populated in post-processing
            link_contexts=[],  # Can be enhanced with LinkContextAnalyzer
            word_count=len(markdown.split()) if markdown else 0,
            entities_preview=entities_preview,
            named_entities=named_entities,
            attached_documents=attached_documents,
            has_attachments=has_attachments,
            attachment_types=attachment_types,
            total_attachment_size=total_attachment_size,
            attachment_count=len(attached_documents)
        )

    def _classify_page_type(self, url: str, html: str, title: str) -> PageType:
        """Classify page type using URL patterns from config"""
        if self.extractor.is_article_url(url):
            return PageType.ARTICLE
        elif self.extractor.is_list_url(url):
            return PageType.LIST
        else:
            return PageType.OTHER

    def _find_siblings(self, url: str, url_tree: Dict[str, Dict]) -> List[str]:
        """Find sibling URLs in the tree"""
        parent_url = url_tree.get(url, {}).get('parent_url')
        if not parent_url:
            return []

        siblings = []
        for other_url, metadata in url_tree.items():
            if metadata.get('parent_url') == parent_url and other_url != url:
                siblings.append(other_url)

        return siblings

    async def _save_page_files(
        self,
        metadata: PageMetadata,
        html: str,
        markdown: str
    ) -> None:
        """
        Save HTML, Markdown, and JSON metadata to disk.
        """
        # Generate filename from URL
        from urllib.parse import urlparse
        parsed = urlparse(metadata.url)
        filename = parsed.path.strip('/').replace('/', '_') or 'index'

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

    def build_incoming_links_map(
        self,
        all_metadata: List[PageMetadata]
    ) -> Dict[str, List[str]]:
        """
        Post-processing: Build incoming links map and update metadata.
        """
        incoming_map: Dict[str, List[str]] = {}

        # Build incoming links map
        for metadata in all_metadata:
            for outgoing_link in metadata.outgoing_links:
                if outgoing_link not in incoming_map:
                    incoming_map[outgoing_link] = []
                incoming_map[outgoing_link].append(metadata.url)

        # Update metadata with incoming links
        for metadata in all_metadata:
            metadata.incoming_links = incoming_map.get(metadata.url, [])

            # Re-save updated metadata
            from urllib.parse import urlparse
            parsed = urlparse(metadata.url)
            filename = parsed.path.strip('/').replace('/', '_') or 'index'
            json_path = self.metadata_dir / f"{filename}.json"
            json_path.write_text(
                metadata.model_dump_json(indent=2),
                encoding='utf-8'
            )

        return incoming_map

    def generate_extraction_report(
        self,
        all_metadata: List[PageMetadata]
    ) -> str:
        """
        Generate a summary report of the extraction process.
        """
        from collections import Counter

        # Statistics
        total_pages = len(all_metadata)
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
# {self.site_config.site_name} Extraction Report

Generated: {datetime.utcnow().isoformat()}
Site: {self.site_config.domain}

## Overview
- **Total Pages Extracted**: {total_pages}
- **Pages with Attachments**: {pages_with_attachments} ({pages_with_attachments/total_pages*100:.1f}% if total_pages > 0 else 0)
- **Total Attachments**: {total_attachments}
- **Total Attachment Size**: {total_attachment_size / (1024*1024):.2f} MB

## Page Types Distribution
"""
        for page_type, count in page_type_counts.most_common():
            report += f"- {page_type.value}: {count} ({count/total_pages*100:.1f}%)\n"

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
    """Example usage of ContentExtractorV2"""
    # Create extractor for Seoul Traffic News
    extractor = SeoulTrafficContentExtractorV2.from_domain(
        domain="news.seoul.go.kr",
        output_dir=Path("./output"),
        download_dir=Path("./downloads"),
        headless=True,
        verbose=True
    )

    # For testing, create a simple URL list
    test_urls = [
        {"url": "https://news.seoul.go.kr/traffic/archives/513625", "depth": 1}
    ]
    url_tree = {
        "https://news.seoul.go.kr/traffic/archives/513625": {
            "parent_url": None,
            "depth": 1
        }
    }

    # Extract content
    all_metadata = await extractor.extract_with_metadata(
        urls=test_urls,
        url_tree=url_tree,
        batch_size=1,
        delay_between_batches=2.0
    )

    # Post-processing
    incoming_links = extractor.build_incoming_links_map(all_metadata)

    # Generate report
    report = extractor.generate_extraction_report(all_metadata)

    report_path = Path("./output/extraction_report_v2.md")
    report_path.write_text(report, encoding='utf-8')

    print(f"✓ Extraction complete: {len(all_metadata)} pages")
    print(f"✓ Report saved to: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
