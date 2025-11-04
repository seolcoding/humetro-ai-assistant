"""Tests for ContentExtractor."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from langchain_core.documents import Document

from src.crawler.content_extractor import ContentExtractor
from src.config.schemas import PageMetadata
from src.config.site_config import SiteConfig


class TestContentExtractor:
    """Test ContentExtractor functionality."""

    @pytest.fixture
    def temp_dirs(self, tmp_path):
        """Create temporary directories."""
        return {
            "output_dir": tmp_path / "crawled",
            "download_dir": tmp_path / "downloads"
        }

    @pytest.fixture
    def mock_site_config(self):
        """Mock site configuration."""
        return SiteConfig(
            site_name="테스트 사이트",
            domain="test.example.com",
            base_url="https://test.example.com",
            article={
                "title": {"selector": "h1.title", "required": True},
                "content": {"selector": ".content", "required": True}
            },
            url_patterns={
                "article_patterns": [r".*/article/\d+$"],
                "list_patterns": [r".*/category/.*"]
            },
            crawl_rules={
                "max_depth": 3,
                "delay_between_requests": 1.0
            }
        )

    @patch('src.crawler.content_extractor.load_site_config')
    def test_from_domain_initialization(self, mock_load_config, temp_dirs, mock_site_config):
        """Test initialization from domain."""
        mock_load_config.return_value = mock_site_config

        extractor = ContentExtractor.from_domain(
            domain="test.example.com",
            output_dir=temp_dirs["output_dir"],
            download_dir=temp_dirs["download_dir"]
        )

        assert extractor.site_config == mock_site_config
        assert extractor.output_dir == temp_dirs["output_dir"]
        assert extractor.download_dir == temp_dirs["download_dir"]

        # Verify directories were created
        assert extractor.raw_html_dir.exists()
        assert extractor.markdown_dir.exists()
        assert extractor.metadata_dir.exists()
        assert extractor.attachments_dir.exists()

    @patch('src.crawler.content_extractor.load_site_config')
    def test_directory_structure_creation(self, mock_load_config, temp_dirs, mock_site_config):
        """Test that all required directories are created."""
        mock_load_config.return_value = mock_site_config

        extractor = ContentExtractor.from_domain(
            domain="test.example.com",
            output_dir=temp_dirs["output_dir"],
            download_dir=temp_dirs["download_dir"]
        )

        # Check directory structure
        expected_dirs = [
            extractor.raw_html_dir,
            extractor.markdown_dir,
            extractor.metadata_dir,
            extractor.attachments_dir
        ]

        for directory in expected_dirs:
            assert directory.exists()
            assert directory.is_dir()

    @patch('src.crawler.content_extractor.load_site_config')
    def test_get_filename_from_url(self, mock_load_config, temp_dirs, mock_site_config):
        """Test filename generation from URL."""
        mock_load_config.return_value = mock_site_config

        extractor = ContentExtractor.from_domain(
            domain="test.example.com",
            output_dir=temp_dirs["output_dir"],
            download_dir=temp_dirs["download_dir"]
        )

        # Test URL to filename conversion
        url1 = "https://test.example.com/article/123"
        filename1 = extractor._get_filename_from_url(url1)
        assert len(filename1) > 0
        assert "/" not in filename1  # No path separators

        # Same URL should give same filename
        filename2 = extractor._get_filename_from_url(url1)
        assert filename1 == filename2

        # Different URL should give different filename
        url3 = "https://test.example.com/article/456"
        filename3 = extractor._get_filename_from_url(url3)
        assert filename1 != filename3

    @patch('src.crawler.content_extractor.load_site_config')
    def test_is_page_cached(self, mock_load_config, temp_dirs, mock_site_config):
        """Test cache checking."""
        mock_load_config.return_value = mock_site_config

        extractor = ContentExtractor.from_domain(
            domain="test.example.com",
            output_dir=temp_dirs["output_dir"],
            download_dir=temp_dirs["download_dir"]
        )

        test_url = "https://test.example.com/article/123"

        # Initially not cached
        assert not extractor._is_page_cached(test_url)

        # Create metadata file
        filename = extractor._get_filename_from_url(test_url)
        metadata_path = extractor.metadata_dir / f"{filename}.json"
        metadata_path.write_text('{"url": "' + test_url + '"}')

        # Now should be cached
        assert extractor._is_page_cached(test_url)

    @patch('src.crawler.content_extractor.load_site_config')
    def test_load_cached_metadata(self, mock_load_config, temp_dirs, mock_site_config):
        """Test loading cached metadata."""
        mock_load_config.return_value = mock_site_config

        extractor = ContentExtractor.from_domain(
            domain="test.example.com",
            output_dir=temp_dirs["output_dir"],
            download_dir=temp_dirs["download_dir"]
        )

        test_url = "https://test.example.com/article/123"

        # Create cached metadata
        filename = extractor._get_filename_from_url(test_url)
        metadata_path = extractor.metadata_dir / f"{filename}.json"

        cached_data = {
            "url": test_url,
            "title": "테스트 제목",
            "page_type": "article",
            "word_count": 100,
            "attachment_count": 0,
            "crawled_at": "2025-01-01T00:00:00",
            "depth": 1
        }

        import json
        metadata_path.write_text(json.dumps(cached_data, ensure_ascii=False))

        # Load cached metadata
        loaded = extractor._load_cached_metadata(test_url)

        assert loaded is not None
        assert loaded.url == test_url
        assert loaded.title == "테스트 제목"
        assert loaded.word_count == 100

    @patch('src.crawler.content_extractor.load_site_config')
    def test_load_cached_metadata_returns_none_on_error(self, mock_load_config, temp_dirs, mock_site_config):
        """Test that loading invalid cached metadata returns None."""
        mock_load_config.return_value = mock_site_config

        extractor = ContentExtractor.from_domain(
            domain="test.example.com",
            output_dir=temp_dirs["output_dir"],
            download_dir=temp_dirs["download_dir"]
        )

        test_url = "https://test.example.com/article/123"

        # Create invalid metadata file
        filename = extractor._get_filename_from_url(test_url)
        metadata_path = extractor.metadata_dir / f"{filename}.json"
        metadata_path.write_text("invalid json {{{")

        # Should return None on error
        loaded = extractor._load_cached_metadata(test_url)
        assert loaded is None

    @patch('src.crawler.content_extractor.load_site_config')
    def test_generate_extraction_report(self, mock_load_config, temp_dirs, mock_site_config):
        """Test extraction report generation."""
        mock_load_config.return_value = mock_site_config

        extractor = ContentExtractor.from_domain(
            domain="test.example.com",
            output_dir=temp_dirs["output_dir"],
            download_dir=temp_dirs["download_dir"]
        )

        # Create sample metadata
        metadata_list = [
            PageMetadata(
                url="https://test.example.com/article/1",
                title="제목 1",
                page_type="article",
                word_count=100,
                attachment_count=1,
                crawled_at="2025-01-01T00:00:00",
                depth=1
            ),
            PageMetadata(
                url="https://test.example.com/article/2",
                title="제목 2",
                page_type="article",
                word_count=200,
                attachment_count=0,
                crawled_at="2025-01-01T00:05:00",
                depth=1
            ),
        ]

        report = extractor.generate_extraction_report(metadata_list)

        # Verify report content
        assert "테스트 사이트" in report
        assert "Total Pages Extracted" in report or "총 페이지 수" in report
        assert "article: 2" in report
        # Note: Sample pages section may not include all titles

    @patch('src.crawler.content_extractor.load_site_config')
    def test_headless_mode(self, mock_load_config, temp_dirs, mock_site_config):
        """Test headless mode configuration."""
        mock_load_config.return_value = mock_site_config

        # Test headless=True (default)
        extractor1 = ContentExtractor.from_domain(
            domain="test.example.com",
            output_dir=temp_dirs["output_dir"],
            download_dir=temp_dirs["download_dir"],
            headless=True
        )
        assert extractor1.headless is True

        # Test headless=False
        extractor2 = ContentExtractor.from_domain(
            domain="test.example.com",
            output_dir=temp_dirs["output_dir"],
            download_dir=temp_dirs["download_dir"],
            headless=False
        )
        assert extractor2.headless is False

    @patch('src.crawler.content_extractor.load_site_config')
    def test_verbose_mode(self, mock_load_config, temp_dirs, mock_site_config):
        """Test verbose mode configuration."""
        mock_load_config.return_value = mock_site_config

        # Test verbose=True (default)
        extractor1 = ContentExtractor.from_domain(
            domain="test.example.com",
            output_dir=temp_dirs["output_dir"],
            download_dir=temp_dirs["download_dir"],
            verbose=True
        )
        assert extractor1.verbose is True

        # Test verbose=False
        extractor2 = ContentExtractor.from_domain(
            domain="test.example.com",
            output_dir=temp_dirs["output_dir"],
            download_dir=temp_dirs["download_dir"],
            verbose=False
        )
        assert extractor2.verbose is False


class TestContentExtractorIntegration:
    """Integration tests for ContentExtractor with real config files."""

    @pytest.fixture
    def temp_dirs(self, tmp_path):
        """Create temporary directories."""
        return {
            "output_dir": tmp_path / "crawled",
            "download_dir": tmp_path / "downloads"
        }

    def test_load_real_config_file(self, temp_dirs):
        """Test loading a real YAML config file."""
        try:
            extractor = ContentExtractor.from_domain(
                domain="news.seoul.go.kr",
                output_dir=temp_dirs["output_dir"],
                download_dir=temp_dirs["download_dir"]
            )

            # Verify config was loaded
            assert extractor.site_config.site_name == "서울시 교통 뉴스"
            assert extractor.site_config.domain == "news.seoul.go.kr"
            assert extractor.site_config.article is not None
            assert extractor.site_config.url_patterns is not None
            assert extractor.site_config.crawl_rules is not None

        except FileNotFoundError:
            pytest.skip("Config file not found - skipping integration test")
