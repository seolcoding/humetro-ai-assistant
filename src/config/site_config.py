"""
Site-specific configuration schema for flexible web scraping.

Separates CSS selectors and site-specific logic from core crawling code.
Allows adding new sites by simply creating a YAML configuration file.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from pathlib import Path
import yaml


class SelectorConfig(BaseModel):
    """CSS selector configuration for a specific element type"""

    selector: str = Field(
        ...,
        description="CSS selector string"
    )
    attribute: Optional[str] = Field(
        None,
        description="Attribute to extract (e.g., 'href', 'src'). None = text content"
    )
    multiple: bool = Field(
        False,
        description="Whether to extract multiple elements"
    )
    required: bool = Field(
        True,
        description="Whether this selector must match"
    )
    fallback_selectors: List[str] = Field(
        default_factory=list,
        description="Alternative selectors to try if primary fails"
    )


class ArticleSelectors(BaseModel):
    """Selectors for article/content page elements"""

    title: SelectorConfig = Field(
        ...,
        description="Article title selector"
    )
    content: SelectorConfig = Field(
        ...,
        description="Main article content selector"
    )
    date: Optional[SelectorConfig] = Field(
        None,
        description="Publication/modification date selector"
    )
    author: Optional[SelectorConfig] = Field(
        None,
        description="Author name selector"
    )
    category: Optional[SelectorConfig] = Field(
        None,
        description="Article category/section selector"
    )
    tags: Optional[SelectorConfig] = Field(
        None,
        description="Article tags selector"
    )


class AttachmentSelectors(BaseModel):
    """Selectors for attachment/document detection"""

    links: SelectorConfig = Field(
        ...,
        description="Attachment link selectors (PDF, HWP, XLSX, etc.)"
    )
    trigger_context: Optional[SelectorConfig] = Field(
        None,
        description="Parent element containing trigger link (for context)"
    )
    popup_indicators: List[str] = Field(
        default_factory=list,
        description="Attributes indicating popup behavior (e.g., 'target=_blank', 'onclick')"
    )
    download_indicators: List[str] = Field(
        default_factory=list,
        description="Attributes indicating direct download"
    )


class NavigationSelectors(BaseModel):
    """Selectors for navigation and site structure"""

    breadcrumb: Optional[SelectorConfig] = Field(
        None,
        description="Breadcrumb navigation selector"
    )
    category_links: Optional[SelectorConfig] = Field(
        None,
        description="Category/section navigation links"
    )
    pagination: Optional[SelectorConfig] = Field(
        None,
        description="Pagination links selector"
    )
    next_page: Optional[SelectorConfig] = Field(
        None,
        description="Next page button/link"
    )


class ListSelectors(BaseModel):
    """Selectors for list/index pages"""

    article_links: SelectorConfig = Field(
        ...,
        description="Links to individual articles"
    )
    article_titles: Optional[SelectorConfig] = Field(
        None,
        description="Article titles in list view"
    )
    article_summaries: Optional[SelectorConfig] = Field(
        None,
        description="Article summaries/excerpts"
    )
    article_dates: Optional[SelectorConfig] = Field(
        None,
        description="Article dates in list view"
    )


class UrlPatterns(BaseModel):
    """URL patterns for page type classification"""

    article_patterns: List[str] = Field(
        default_factory=list,
        description="Regex patterns matching article URLs"
    )
    list_patterns: List[str] = Field(
        default_factory=list,
        description="Regex patterns matching list/index URLs"
    )
    category_patterns: List[str] = Field(
        default_factory=list,
        description="Regex patterns matching category URLs"
    )
    exclude_patterns: List[str] = Field(
        default_factory=list,
        description="Regex patterns for URLs to exclude"
    )


class CrawlRules(BaseModel):
    """Crawling behavior rules"""

    max_depth: int = Field(
        5,
        description="Maximum crawl depth"
    )
    delay_between_requests: float = Field(
        1.0,
        description="Delay in seconds between requests"
    )
    respect_robots_txt: bool = Field(
        True,
        description="Whether to respect robots.txt"
    )
    user_agent: Optional[str] = Field(
        None,
        description="Custom user agent string"
    )
    allowed_domains: List[str] = Field(
        default_factory=list,
        description="Allowed domains for following links"
    )
    excluded_extensions: List[str] = Field(
        default_factory=lambda: ['.jpg', '.png', '.gif', '.css', '.js'],
        description="File extensions to exclude"
    )


class SiteConfig(BaseModel):
    """
    Complete site configuration.

    Defines all selectors, patterns, and rules for crawling a specific website.
    Loaded from YAML files in config/sites/
    """

    # Basic information
    site_name: str = Field(
        ...,
        description="Human-readable site name"
    )
    domain: str = Field(
        ...,
        description="Primary domain (e.g., 'news.seoul.go.kr')"
    )
    base_url: str = Field(
        ...,
        description="Base URL for the site"
    )
    description: Optional[str] = Field(
        None,
        description="Site description"
    )

    # Selectors
    article: Optional[ArticleSelectors] = Field(
        None,
        description="Selectors for article pages"
    )
    attachment: Optional[AttachmentSelectors] = Field(
        None,
        description="Selectors for attachments"
    )
    navigation: Optional[NavigationSelectors] = Field(
        None,
        description="Selectors for navigation elements"
    )
    list: Optional[ListSelectors] = Field(
        None,
        description="Selectors for list/index pages"
    )

    # URL patterns
    url_patterns: Optional[UrlPatterns] = Field(
        None,
        description="URL patterns for classification"
    )

    # Crawl rules
    crawl_rules: Optional[CrawlRules] = Field(
        None,
        description="Crawling behavior rules"
    )

    # Custom settings
    custom: Dict[str, Any] = Field(
        default_factory=dict,
        description="Site-specific custom settings"
    )

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> 'SiteConfig':
        """Load configuration from YAML file"""
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls(**data)

    def to_yaml(self, yaml_path: Path) -> None:
        """Save configuration to YAML file"""
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(
                self.model_dump(exclude_none=True),
                f,
                allow_unicode=True,
                sort_keys=False
            )


class SiteConfigRegistry:
    """
    Registry for managing site configurations.

    Loads and caches site configs from YAML files.
    Provides fallback to default config if site-specific config not found.
    """

    def __init__(self, config_dir: Path):
        self.config_dir = config_dir
        self._configs: Dict[str, SiteConfig] = {}
        self._default_config: Optional[SiteConfig] = None

        # Load default config if exists
        default_path = config_dir / "default.yaml"
        if default_path.exists():
            self._default_config = SiteConfig.from_yaml(default_path)

    def get_config(self, domain: str) -> SiteConfig:
        """
        Get configuration for a domain.

        Tries:
        1. Exact domain match (e.g., 'news.seoul.go.kr.yaml')
        2. Base domain match (e.g., 'seoul.go.kr.yaml')
        3. Default config
        4. Raises error if none found
        """
        # Check cache
        if domain in self._configs:
            return self._configs[domain]

        # Try exact domain match
        config_path = self.config_dir / f"{domain}.yaml"
        if config_path.exists():
            config = SiteConfig.from_yaml(config_path)
            self._configs[domain] = config
            return config

        # Try base domain (e.g., 'seoul.go.kr' from 'news.seoul.go.kr')
        parts = domain.split('.')
        if len(parts) > 2:
            base_domain = '.'.join(parts[-2:])
            base_config_path = self.config_dir / f"{base_domain}.yaml"
            if base_config_path.exists():
                config = SiteConfig.from_yaml(base_config_path)
                self._configs[domain] = config
                return config

        # Fallback to default
        if self._default_config:
            return self._default_config

        raise ValueError(f"No configuration found for domain: {domain}")

    def list_available_configs(self) -> List[str]:
        """List all available site configurations"""
        return [
            f.stem for f in self.config_dir.glob("*.yaml")
            if f.stem != "default"
        ]

    def reload_config(self, domain: str) -> SiteConfig:
        """Reload configuration from disk"""
        if domain in self._configs:
            del self._configs[domain]
        return self.get_config(domain)


# Convenience function
def load_site_config(site_name: str, config_dir: Optional[Path] = None) -> SiteConfig:
    """
    Load a site configuration by name.

    Args:
        site_name: Name of the site (e.g., 'seoul_traffic', 'news.seoul.go.kr')
        config_dir: Directory containing config files (defaults to config/sites/)

    Returns:
        SiteConfig object
    """
    if config_dir is None:
        config_dir = Path(__file__).parent / "sites"

    registry = SiteConfigRegistry(config_dir)
    return registry.get_config(site_name)


__all__ = [
    'SelectorConfig',
    'ArticleSelectors',
    'AttachmentSelectors',
    'NavigationSelectors',
    'ListSelectors',
    'UrlPatterns',
    'CrawlRules',
    'SiteConfig',
    'SiteConfigRegistry',
    'load_site_config',
]
