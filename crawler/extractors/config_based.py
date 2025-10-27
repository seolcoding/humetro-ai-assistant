"""
Configuration-based extractor implementation.

Uses SiteConfig to extract content without hardcoded selectors.
"""

import re
import logging
import asyncio
from typing import List, Dict, Optional
from datetime import datetime

from playwright.async_api import Page
from bs4 import BeautifulSoup

from config.site_config import SiteConfig
from config.schemas import AttachedDocument, AttachmentType, AttachmentSource
from .base import AbstractExtractor, ExtractionResult

logger = logging.getLogger(__name__)


class ConfigBasedExtractor(AbstractExtractor):
    """
    Configuration-driven content extractor.

    Uses SiteConfig YAML to determine:
    - Which CSS selectors to use
    - How to extract metadata
    - How to detect and process attachments
    - How to handle popup windows
    """

    def __init__(self, site_config: SiteConfig):
        super().__init__(site_config)
        self.logger = logging.getLogger(f"{__name__}.{site_config.site_name}")

    async def extract_from_page(
        self,
        page: Page,
        html: str,
        url: str
    ) -> ExtractionResult:
        """
        Extract content from page using configuration.

        Uses selectors from self.config.article for extraction.
        """
        soup = BeautifulSoup(html, 'html.parser')
        result = ExtractionResult(url=url, html=html)

        try:
            # Extract main container first (if configured)
            main_container = None
            if hasattr(self.config.article, 'main_container') and self.config.article.main_container:
                main_container_elem = self.extract_with_selector(
                    soup,
                    self.config.article.main_container
                )
                if main_container_elem:
                    # Create new soup from main container
                    soup = BeautifulSoup(str(soup.select_one(self.config.article.main_container.selector)), 'html.parser')

            # Extract title
            if self.config.article and self.config.article.title:
                result.title = self.extract_with_selector(
                    soup,
                    self.config.article.title
                )
                self.logger.debug(f"Extracted title: {result.title}")

            # Extract content
            if self.config.article and self.config.article.content:
                content_elem = soup.select_one(self.config.article.content.selector)
                if content_elem:
                    result.content = content_elem.get_text(separator='\n', strip=True)
                    self.logger.debug(f"Extracted content: {len(result.content)} chars")

            # Extract metadata fields
            if self.config.article:
                if self.config.article.date:
                    result.date = self.extract_with_selector(
                        soup,
                        self.config.article.date
                    )

                if self.config.article.author:
                    result.author = self.extract_with_selector(
                        soup,
                        self.config.article.author
                    )

                if self.config.article.category:
                    result.category = self.extract_with_selector(
                        soup,
                        self.config.article.category
                    )

                if self.config.article.tags:
                    tags = self.extract_with_selector(
                        soup,
                        self.config.article.tags
                    )
                    if tags:
                        result.tags = tags if isinstance(tags, list) else [tags]

            # Extract custom metadata (site-specific)
            result.custom_metadata = await self._extract_custom_metadata(soup)

            # Extract breadcrumb
            if self.config.navigation and self.config.navigation.breadcrumb:
                breadcrumb_links = await self._extract_breadcrumb(soup)
                result.breadcrumb_links = breadcrumb_links

            # Extract outgoing links
            result.outgoing_links = await self._extract_outgoing_links(soup)

            # Detect attachments
            result.attachment_links, result.popup_buttons = await self._detect_attachments(soup, html)

            result.success = True

        except Exception as e:
            self.logger.error(f"Extraction failed for {url}: {e}")
            result.success = False
            result.errors.append(str(e))

        return result

    async def extract_attachments(
        self,
        page: Page,
        html: str,
        result: ExtractionResult
    ) -> List[AttachedDocument]:
        """
        Process detected attachments.

        For each attachment link or popup button:
        1. Determine if it's popup or direct download
        2. Process accordingly
        3. Create AttachedDocument
        """
        attachments = []

        # Process direct attachment links
        for link_info in result.attachment_links:
            try:
                attachment = await self._process_attachment_link(
                    page,
                    link_info
                )
                if attachment:
                    attachments.append(attachment)
            except Exception as e:
                self.logger.warning(f"Failed to process attachment link: {e}")

        # Process popup buttons
        for button_info in result.popup_buttons:
            try:
                attachment = await self._process_popup_button(
                    page,
                    button_info
                )
                if attachment:
                    attachments.append(attachment)
            except Exception as e:
                self.logger.warning(f"Failed to process popup button: {e}")

        return attachments

    async def extract_popup_content(
        self,
        popup_page: Page,
        popup_url: str
    ) -> str:
        """
        Extract content from popup window using configured selector.

        Uses: self.config.custom['popup_viewer']['text_selector']
        """
        try:
            # Wait for popup content to load
            await popup_page.wait_for_load_state('networkidle', timeout=10000)

            # Get popup viewer config
            popup_viewer_config = self.config.custom.get('popup_viewer', {})
            text_selector = popup_viewer_config.get('text_selector', '#viewerContainer')

            # Check if popup page has viewer container
            has_viewer = await popup_page.query_selector(text_selector)

            if not has_viewer:
                self.logger.warning(f"Popup viewer not found: {text_selector}")
                # Fallback to body text
                return await popup_page.text_content('body')

            # Extract text from viewer
            extraction_method = popup_viewer_config.get('extraction_method', 'textContent')

            if extraction_method == 'textContent':
                text = await popup_page.text_content(text_selector)
            else:  # innerText
                text = await popup_page.inner_text(text_selector)

            self.logger.info(f"Extracted {len(text)} chars from popup")
            return text or ""

        except Exception as e:
            self.logger.error(f"Failed to extract popup content: {e}")
            return ""

    # Helper methods

    async def _extract_custom_metadata(self, soup: BeautifulSoup) -> Dict:
        """Extract site-specific custom metadata"""
        custom_meta = {}

        # Check for definition list metadata structure
        meta_structure = self.config.custom.get('metadata_structure', {})

        if meta_structure.get('type') == 'definition_list':
            container_selector = meta_structure.get('container')
            if container_selector:
                container = soup.select_one(container_selector)
                if container:
                    dts = container.select(meta_structure.get('term_selector', 'dt'))
                    dds = container.select(meta_structure.get('definition_selector', 'dd'))

                    for dt, dd in zip(dts, dds):
                        key = dt.get_text(strip=True)
                        value = dd.get_text(strip=True)
                        custom_meta[key] = value

        return custom_meta

    async def _extract_breadcrumb(self, soup: BeautifulSoup) -> List[Dict[str, str]]:
        """Extract breadcrumb navigation"""
        breadcrumbs = []

        if not self.config.navigation or not self.config.navigation.breadcrumb:
            return breadcrumbs

        breadcrumb_elem = soup.select_one(self.config.navigation.breadcrumb.selector)
        if not breadcrumb_elem:
            return breadcrumbs

        links = breadcrumb_elem.select('a')
        for link in links:
            breadcrumbs.append({
                'text': link.get_text(strip=True),
                'url': link.get('href', '')
            })

        return breadcrumbs

    async def _extract_outgoing_links(self, soup: BeautifulSoup) -> List[str]:
        """Extract all outgoing links from content"""
        links = []

        # Get links from main content area
        if self.config.article and self.config.article.content:
            content_elem = soup.select_one(self.config.article.content.selector)
            if content_elem:
                for a_tag in content_elem.select('a[href]'):
                    href = a_tag.get('href')
                    if href and not self.should_exclude_url(href):
                        links.append(href)

        return links

    async def _detect_attachments(
        self,
        soup: BeautifulSoup,
        html: str
    ) -> tuple[List[Dict], List[Dict]]:
        """
        Detect attachment links and popup buttons.

        Returns:
            (attachment_links, popup_buttons)
        """
        attachment_links = []
        popup_buttons = []

        if not self.config.attachment:
            return attachment_links, popup_buttons

        # Detect attachment links
        if self.config.attachment.links:
            links = soup.select(self.config.attachment.links.selector)

            for link in links:
                href = link.get('href', '')
                text = link.get_text(strip=True)

                # Check if it's a popup trigger
                is_popup = any(
                    indicator in str(link)
                    for indicator in self.config.attachment.popup_indicators
                )

                if is_popup:
                    popup_buttons.append({
                        'element': str(link),
                        'href': href,
                        'text': text,
                        'type': 'link'
                    })
                else:
                    attachment_links.append({
                        'href': href,
                        'text': text,
                        'element': str(link)
                    })

        # Detect popup buttons (e.g., .preview_button)
        popup_indicators = self.config.attachment.popup_indicators
        for indicator in popup_indicators:
            if 'class=' in indicator:
                # Extract class name from indicator
                class_name = indicator.split('=')[1]
                buttons = soup.select(f'.{class_name}')

                for button in buttons:
                    onclick = button.get('onclick', '')
                    title = button.get('title', '')
                    text = button.get_text(strip=True)

                    # Extract URL from onclick if present
                    popup_url = self._extract_url_from_onclick(onclick)

                    popup_buttons.append({
                        'element': str(button),
                        'onclick': onclick,
                        'url': popup_url,
                        'text': text,
                        'title': title,
                        'type': 'button'
                    })

        return attachment_links, popup_buttons

    def _extract_url_from_onclick(self, onclick: str) -> Optional[str]:
        """
        Extract URL from onclick attribute.

        Example: onclick="previewDocument('https://...', 'filename.hwpx')"
        """
        if not onclick:
            return None

        # Use pattern from config if available
        pattern = self.config.custom.get('hwp_viewer_url_pattern')
        if pattern:
            match = re.search(pattern, onclick)
            if match:
                return match.group(1)

        # Fallback: extract first URL-like string
        url_pattern = r'https?://[^\s\'"]+|//[^\s\']+'
        match = re.search(url_pattern, onclick)
        if match:
            return match.group(0)

        return None

    async def _process_attachment_link(
        self,
        page: Page,
        link_info: Dict
    ) -> Optional[AttachedDocument]:
        """
        Process a direct attachment link.

        For direct file links (e.g., /files/filename.pdf), this method:
        1. Extracts file information
        2. Creates AttachedDocument metadata

        Note: Actual file download and parsing should be done separately
        """
        href = link_info.get('href', '')
        text = link_info.get('text', '')

        if not href:
            return None

        try:
            # Determine attachment type from extension
            extension = href.split('.')[-1].lower()
            attachment_type = self._get_attachment_type(extension)

            # Generate attachment ID from URL
            import hashlib
            attachment_id = hashlib.md5(href.encode()).hexdigest()

            # Extract filename
            filename = href.split('/')[-1] if '/' in href else href

            # Create AttachedDocument (placeholder data for now)
            attachment = AttachedDocument(
                attachment_id=attachment_id,
                attachment_type=attachment_type,
                original_filename=filename,
                file_size=0,  # Will be filled after download
                source_type=AttachmentSource.DIRECT_LINK,
                source_url=href,
                trigger_element=link_info.get('element', ''),
                trigger_context=text,
                extracted_text="",  # Will be filled after parsing
                local_path="",  # Will be filled after download
                extraction_method='manual',
                extraction_success=False,
                word_count=0
            )

            self.logger.info(f"Detected direct attachment: {filename}")
            return attachment

        except Exception as e:
            self.logger.error(f"Failed to process attachment link: {e}")
            return None

    async def _process_popup_button(
        self,
        page: Page,
        button_info: Dict
    ) -> Optional[AttachedDocument]:
        """
        Process a popup button trigger and extract content from popup.

        This method:
        1. Sets up popup listener
        2. Clicks the button to open popup
        3. Waits for popup window
        4. Extracts content using configured selector
        5. Creates AttachedDocument with extracted content
        """
        try:
            # Extract popup URL from button info
            popup_url = button_info.get('url')
            if not popup_url:
                onclick = button_info.get('onclick', '')
                popup_url = self._extract_url_from_onclick(onclick)

            if not popup_url:
                self.logger.warning("No popup URL found in button info")
                return None

            # Extract filename from button info
            title = button_info.get('title', '')
            text = button_info.get('text', '')

            # Try to extract filename from onclick
            filename = None
            onclick = button_info.get('onclick', '')
            if onclick:
                # Pattern: previewDocument('url', 'filename.hwpx')
                import re
                match = re.search(r"'([^']+\.(hwpx?|pdf|xlsx?))'", onclick)
                if match:
                    filename = match.group(1)

            if not filename:
                filename = popup_url.split('/')[-1] if '/' in popup_url else 'unknown_file'

            # Determine attachment type from extension
            extension = filename.split('.')[-1].lower()
            attachment_type = self._get_attachment_type(extension)

            # Generate attachment ID
            import hashlib
            attachment_id = hashlib.md5(popup_url.encode()).hexdigest()

            self.logger.info(f"Processing popup for: {filename}")

            # Setup popup listener
            popup_future = asyncio.Future()

            def handle_popup(popup):
                popup_future.set_result(popup)

            page.context.on('page', handle_popup)

            # Click button to trigger popup
            try:
                # Find and click the button
                button_selector = f"button.preview_button[onclick*='{filename}']"
                await page.click(button_selector, timeout=5000)
                self.logger.debug(f"Clicked popup button: {button_selector}")
            except Exception as e:
                # Fallback: try to open popup URL directly
                self.logger.warning(f"Button click failed, opening popup directly: {e}")
                popup_page = await page.context.new_page()
                await popup_page.goto(popup_url, wait_until='networkidle')
                popup_future.set_result(popup_page)

            # Wait for popup window (with timeout)
            try:
                popup_page = await asyncio.wait_for(popup_future, timeout=10.0)
                self.logger.info("Popup window opened")
            except asyncio.TimeoutError:
                self.logger.error("Popup window did not open within timeout")
                return None

            # Extract content from popup
            extracted_text = await self.extract_popup_content(popup_page, popup_url)

            # Close popup
            await popup_page.close()

            # Create AttachedDocument
            attachment = AttachedDocument(
                attachment_id=attachment_id,
                attachment_type=attachment_type,
                original_filename=filename,
                file_size=len(extracted_text.encode('utf-8')),
                source_type=AttachmentSource.POPUP_WINDOW,
                source_url=popup_url,
                trigger_element=button_info.get('element', ''),
                trigger_context=text,
                popup_window_title=title,
                popup_window_url=popup_url,
                extracted_text=extracted_text,
                local_path="",  # Not saved to disk in this implementation
                extraction_method='playwright',
                extraction_success=True,
                word_count=len(extracted_text.split())
            )

            self.logger.info(
                f"✓ Popup processed: {filename} ({attachment.word_count} words)"
            )
            return attachment

        except Exception as e:
            self.logger.error(f"Failed to process popup button: {e}")
            return None

    def _get_attachment_type(self, extension: str) -> AttachmentType:
        """Map file extension to AttachmentType"""
        extension_map = {
            'pdf': AttachmentType.PDF,
            'hwp': AttachmentType.HWP,
            'hwpx': AttachmentType.HWP,
            'xlsx': AttachmentType.XLSX,
            'xls': AttachmentType.XLS,
            'docx': AttachmentType.DOCX,
            'doc': AttachmentType.DOC,
        }
        return extension_map.get(extension, AttachmentType.OTHER)


__all__ = ['ConfigBasedExtractor']
