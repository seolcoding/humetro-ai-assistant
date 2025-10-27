"""
Abstract base classes for configuration-driven extraction.

Separates CSS selectors and site logic from core crawling code.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from playwright.async_api import Page
from bs4 import BeautifulSoup

from config.site_config import SiteConfig
from config.schemas import PageMetadata, AttachedDocument


@dataclass
class ExtractionResult:
    """
    Result of content extraction from a page.

    Contains extracted data before it's converted to PageMetadata schema.
    """

    url: str
    title: Optional[str] = None
    content: Optional[str] = None
    html: Optional[str] = None

    # Metadata fields
    date: Optional[str] = None
    author: Optional[str] = None
    category: Optional[str] = None
    tags: List[str] = None

    # Custom metadata (site-specific)
    custom_metadata: Dict[str, Any] = None

    # Links and navigation
    breadcrumb_links: List[Dict[str, str]] = None
    outgoing_links: List[str] = None

    # Attachments
    attachment_links: List[Dict[str, str]] = None
    popup_buttons: List[Dict[str, str]] = None

    # Extraction info
    extracted_at: datetime = None
    success: bool = True
    errors: List[str] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.custom_metadata is None:
            self.custom_metadata = {}
        if self.breadcrumb_links is None:
            self.breadcrumb_links = []
        if self.outgoing_links is None:
            self.outgoing_links = []
        if self.attachment_links is None:
            self.attachment_links = []
        if self.popup_buttons is None:
            self.popup_buttons = []
        if self.errors is None:
            self.errors = []
        if self.extracted_at is None:
            self.extracted_at = datetime.utcnow()


class AbstractExtractor(ABC):
    """
    Abstract base class for site-specific extractors.

    Implements configuration-driven extraction using SiteConfig.
    Subclasses can override methods for custom behavior.
    """

    def __init__(self, site_config: SiteConfig):
        """
        Initialize extractor with site configuration.

        Args:
            site_config: SiteConfig object with selectors and rules
        """
        self.config = site_config

    @abstractmethod
    async def extract_from_page(
        self,
        page: Page,
        html: str,
        url: str
    ) -> ExtractionResult:
        """
        Extract content from a page.

        Args:
            page: Playwright Page object
            html: Raw HTML content
            url: Page URL

        Returns:
            ExtractionResult with extracted data
        """
        pass

    @abstractmethod
    async def extract_attachments(
        self,
        page: Page,
        html: str,
        result: ExtractionResult
    ) -> List[AttachedDocument]:
        """
        Extract and process attachments from a page.

        Args:
            page: Playwright Page object
            html: Raw HTML content
            result: ExtractionResult from extract_from_page

        Returns:
            List of AttachedDocument objects
        """
        pass

    @abstractmethod
    async def extract_popup_content(
        self,
        popup_page: Page,
        popup_url: str
    ) -> str:
        """
        Extract content from popup window.

        Args:
            popup_page: Playwright Page object for popup
            popup_url: Popup window URL

        Returns:
            Extracted text content from popup
        """
        pass

    def extract_with_selector(
        self,
        soup: BeautifulSoup,
        selector_config: Any,
        default: Any = None
    ) -> Any:
        """
        Extract content using SelectorConfig.

        Args:
            soup: BeautifulSoup object
            selector_config: SelectorConfig object
            default: Default value if extraction fails

        Returns:
            Extracted value or default
        """
        if selector_config is None:
            return default

        try:
            # Try primary selector
            elements = soup.select(selector_config.selector)

            if not elements and selector_config.fallback_selectors:
                # Try fallback selectors
                for fallback in selector_config.fallback_selectors:
                    elements = soup.select(fallback)
                    if elements:
                        break

            if not elements:
                if selector_config.required:
                    raise ValueError(f"Required selector not found: {selector_config.selector}")
                return default

            # Extract attribute or text
            if selector_config.multiple:
                results = []
                for elem in elements:
                    if selector_config.attribute:
                        value = elem.get(selector_config.attribute)
                    else:
                        value = elem.get_text(strip=True)
                    if value:
                        results.append(value)
                return results
            else:
                elem = elements[0]
                if selector_config.attribute:
                    return elem.get(selector_config.attribute, default)
                else:
                    return elem.get_text(strip=True) or default

        except Exception as e:
            if selector_config.required:
                raise
            return default

    def is_article_url(self, url: str) -> bool:
        """Check if URL matches article patterns"""
        if not self.config.url_patterns:
            return False

        import re
        for pattern in self.config.url_patterns.article_patterns:
            if re.match(pattern, url):
                return True
        return False

    def is_list_url(self, url: str) -> bool:
        """Check if URL matches list patterns"""
        if not self.config.url_patterns:
            return False

        import re
        for pattern in self.config.url_patterns.list_patterns:
            if re.match(pattern, url):
                return True
        return False

    def should_exclude_url(self, url: str) -> bool:
        """Check if URL should be excluded"""
        if not self.config.url_patterns:
            return False

        import re
        for pattern in self.config.url_patterns.exclude_patterns:
            if re.match(pattern, url):
                return True
        return False


__all__ = ['AbstractExtractor', 'ExtractionResult']
