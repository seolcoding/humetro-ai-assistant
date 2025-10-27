"""
Test that all required modules can be imported correctly.

This is a smoke test to verify the development environment is set up correctly.
"""

import pytest


def test_import_crawl4ai():
    """Test Crawl4AI import"""
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import crawl4ai: {e}")


def test_import_playwright():
    """Test Playwright import"""
    try:
        from playwright.async_api import async_playwright
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import playwright: {e}")


def test_import_beautifulsoup():
    """Test BeautifulSoup import"""
    try:
        from bs4 import BeautifulSoup
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import beautifulsoup4: {e}")


def test_import_pydantic():
    """Test Pydantic import"""
    try:
        from pydantic import BaseModel, Field
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import pydantic: {e}")


def test_import_document_parsers():
    """Test document parsing library imports"""
    try:
        import pypdf
        import openpyxl
        import olefile
        import docx
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import document parser: {e}")


def test_import_aiohttp():
    """Test aiohttp import"""
    try:
        import aiohttp
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import aiohttp: {e}")


def test_import_schemas():
    """Test config.schemas import"""
    try:
        from config.schemas import (
            AttachedDocument,
            AttachmentType,
            AttachmentSource,
            PageMetadata,
            PageType,
            LinkContext,
            BreadcrumbItem
        )
        assert True
    except ImportError as e:
        pytest.fail(f"Failed to import config.schemas: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
