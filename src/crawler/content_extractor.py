"""
Seoul Traffic News Content Extractor with Attachment Handling

This module orchestrates the extraction of content from Seoul traffic news pages,
including detection and processing of attached documents (PDF, HWP, XLSX, etc.)
that may be opened in popup windows.
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
from src.crawler.popup_handler import PopupWindowHandler
from src.crawler.document_parser import DocumentParser
from src.crawler.tree_analyzer import TreeStructureAnalyzer
from src.crawler.link_analyzer import LinkContextAnalyzer
from src.utils.entity_preview import SimpleEntityExtractor

logger = logging.getLogger(__name__)


class SeoulTrafficContentExtractor:
    """
    Main content extraction orchestrator with attachment handling.

    Features:
    - Rich metadata extraction with tree structure preservation
    - Popup window detection for attached documents
    - Multi-format document parsing (PDF, HWP, XLSX, DOCX)
    - Link context analysis for Knowledge Graph preparation
    - Simple entity preview for each page
    """

    def __init__(
        self,
        output_dir: Path,
        download_dir: Path,
        headless: bool = True,
        verbose: bool = True
    ):
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

        # Initialize analyzers
        self.tree_analyzer = TreeStructureAnalyzer()
        self.link_analyzer = LinkContextAnalyzer()
        self.entity_extractor = SimpleEntityExtractor()

        # Initialize handlers (will be created per session)
        self.popup_handler: Optional[PopupWindowHandler] = None
        self.document_parser: Optional[DocumentParser] = None

        logger.info(f"ContentExtractor initialized with output_dir={output_dir}")

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
            screenshot=False  # Can enable for visual archival
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            # Initialize handlers with crawler's browser context
            self.popup_handler = PopupWindowHandler(
                download_dir=self.attachments_dir,
                timeout=10000
            )
            self.document_parser = DocumentParser(
                attachments_dir=self.attachments_dir,
                entity_extractor=self.entity_extractor
            )

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
        2. Popup listener setup
        3. Attachment detection and processing
        4. Document parsing
        5. Metadata enrichment
        6. File saving
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

            # Step 2: Access browser context and page for popup handling
            browser: Browser = crawler.crawler_strategy.browser
            context: BrowserContext = await browser.new_context(
                accept_downloads=True
            )
            page: Page = await context.new_page()

            # Setup popup listener
            await self.popup_handler.setup_popup_listener(context, page)

            # Navigate to page
            await page.goto(url, wait_until="domcontentloaded")
            await page.wait_for_timeout(1000)  # Allow dynamic content to load

            # Step 3: Detect attachment links
            attachment_links = await self.popup_handler.detect_attachment_links(
                page=page,
                html=html
            )

            logger.debug(f"Found {len(attachment_links)} potential attachments")

            # Step 4: Process each attachment
            attached_documents: List[AttachedDocument] = []

            for link_info in attachment_links:
                try:
                    attachment = await self.popup_handler.process_attachment_link(
                        page=page,
                        link_info=link_info
                    )

                    if attachment:
                        # Parse document content
                        attachment = await self.document_parser.parse_document(attachment)
                        attached_documents.append(attachment)
                        logger.debug(
                            f"  ✓ Attachment processed: {attachment.original_filename} "
                            f"({attachment.attachment_type.value})"
                        )
                except Exception as e:
                    logger.warning(f"  ✗ Attachment processing failed: {e}")
                    continue

            # Clean up browser context
            await context.close()

            # Step 5: Build metadata
            metadata = await self._build_metadata(
                url=url,
                html=html,
                markdown=markdown,
                links=links,
                url_tree=url_tree,
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
        attached_documents: List[AttachedDocument]
    ) -> PageMetadata:
        """
        Build comprehensive PageMetadata object.
        """
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        # Basic info
        title = soup.title.string if soup.title else ""

        # Tree structure analysis
        page_type = self.tree_analyzer.classify_page_type(url, html, title)
        breadcrumb = self.tree_analyzer.build_breadcrumb(url, url_tree)
        siblings = self.tree_analyzer.find_siblings(url, url_tree)

        parent_url = url_tree.get(url, {}).get('parent_url')
        depth = url_tree.get(url, {}).get('depth', 0)

        # Link context analysis
        link_contexts = self.link_analyzer.extract_link_contexts(html, url)

        # Outgoing links
        outgoing_links = [link for link in links if link.startswith('http')]

        # Entity preview from main content
        main_text = markdown[:5000]  # Limit for performance
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
            breadcrumb=breadcrumb,
            siblings=siblings,
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

        This should be called after all pages are extracted to populate
        the incoming_links field in each PageMetadata.
        """
        incoming_map = self.link_analyzer.build_incoming_links_map(
            {meta.url: meta for meta in all_metadata}
        )

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
# Seoul Traffic News Extraction Report

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
    """Example usage of ContentExtractor"""
    from src.crawler.url_discovery import SeoulTrafficURLDiscovery

    # Step 1: Discover URLs
    discovery = SeoulTrafficURLDiscovery(
        base_url="https://news.seoul.go.kr/traffic",
        output_dir=Path("./output")
    )

    urls = await discovery.discover_all()
    url_tree = discovery.url_tree

    # Step 2: Extract content with attachments
    extractor = SeoulTrafficContentExtractor(
        output_dir=Path("./output"),
        download_dir=Path("./downloads"),
        headless=True,
        verbose=True
    )

    all_metadata = await extractor.extract_with_metadata(
        urls=urls[:100],  # Limit for testing
        url_tree=url_tree,
        batch_size=5,
        delay_between_batches=2.0
    )

    # Step 3: Post-processing
    incoming_links = extractor.build_incoming_links_map(all_metadata)

    # Step 4: Generate report
    report = extractor.generate_extraction_report(all_metadata)

    report_path = Path("./output/extraction_report.md")
    report_path.write_text(report, encoding='utf-8')

    print(f"✓ Extraction complete: {len(all_metadata)} pages")
    print(f"✓ Report saved to: {report_path}")


if __name__ == "__main__":
    asyncio.run(main())
