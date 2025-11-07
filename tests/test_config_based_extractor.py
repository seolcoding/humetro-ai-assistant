"""
Tests for ConfigBasedExtractor
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from src.crawler.extractors.config_based import ConfigBasedExtractor
from src.config.site_config import SiteConfig


class TestConfigBasedExtractor:
    """Test ConfigBasedExtractor functionality"""

    @pytest.fixture
    def seoul_config(self):
        """Load Seoul Traffic News configuration"""
        config_path = Path(__file__).parent.parent / "src" / "config" / "sites" / "news.seoul.go.kr.yaml"
        return SiteConfig.from_yaml(config_path)

    @pytest.fixture
    def extractor(self, seoul_config):
        """Create extractor instance"""
        return ConfigBasedExtractor(seoul_config)

    def test_extractor_initialization(self, extractor, seoul_config):
        """Test that extractor initializes with config"""
        assert extractor.config == seoul_config
        assert extractor.config.site_name == "서울시 교통 뉴스"

    def test_url_classification(self, extractor):
        """Test URL pattern matching"""
        # Article URL
        article_url = "https://news.seoul.go.kr/traffic/archives/513625"
        assert extractor.is_article_url(article_url) is True
        assert extractor.is_list_url(article_url) is False

        # List URL
        list_url = "https://news.seoul.go.kr/traffic/archives/category/public"
        assert extractor.is_list_url(list_url) is True

        # Excluded URL
        excluded_url = "https://news.seoul.go.kr/traffic/wp-content/themes/style.css"
        assert extractor.should_exclude_url(excluded_url) is True

    def test_extract_url_from_onclick(self, extractor):
        """Test extracting URL from onclick attribute"""
        onclick = "previewDocument('https://news.seoul.go.kr/traffic/files/2025/01/test.hwpx','test.hwpx')"
        url = extractor._extract_url_from_onclick(onclick)
        assert url == "https://news.seoul.go.kr/traffic/files/2025/01/test.hwpx"

    def test_get_attachment_type(self, extractor):
        """Test attachment type detection"""
        from src.config.schemas import AttachmentType

        assert extractor._get_attachment_type('pdf') == AttachmentType.PDF
        assert extractor._get_attachment_type('hwpx') == AttachmentType.HWP
        assert extractor._get_attachment_type('xlsx') == AttachmentType.XLSX
        assert extractor._get_attachment_type('unknown') == AttachmentType.OTHER

    @pytest.mark.asyncio
    async def test_extract_from_page_basic(self, extractor):
        """Test basic extraction from HTML"""
        html = """
        <html>
        <body>
            <div id="sub_centent">
                <h3 class="atitle">Test Title</h3>
                <div class="a_content">
                    <p>Test content here</p>
                </div>
            </div>
        </body>
        </html>
        """

        # Mock Playwright page
        mock_page = AsyncMock()

        result = await extractor.extract_from_page(
            page=mock_page,
            html=html,
            url="https://news.seoul.go.kr/traffic/archives/513625"
        )

        assert result.success is True
        assert result.title == "Test Title"
        assert "Test content" in result.content
        assert result.url == "https://news.seoul.go.kr/traffic/archives/513625"

    @pytest.mark.asyncio
    async def test_detect_attachments(self, extractor):
        """Test attachment detection"""
        html = """
        <html>
        <body>
            <div class="a_content">
                <a href="/files/test.pdf">Download PDF</a>
                <button class="preview_button"
                        onclick="previewDocument('https://example.com/file.hwpx','file.hwpx')"
                        title="새 창열림">바로보기</button>
            </div>
        </body>
        </html>
        """

        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        attachment_links, popup_buttons = await extractor._detect_attachments(soup, html)

        # Should detect 1 direct link and 1 popup button
        assert len(attachment_links) == 1
        assert len(popup_buttons) == 1
        assert attachment_links[0]['href'] == '/files/test.pdf'
        assert 'previewDocument' in popup_buttons[0]['onclick']

    @pytest.mark.asyncio
    async def test_extract_popup_content(self, extractor):
        """Test popup content extraction"""
        # Mock popup page
        mock_popup = AsyncMock()
        mock_popup.wait_for_load_state = AsyncMock()
        mock_popup.query_selector = AsyncMock(return_value=True)
        mock_popup.text_content = AsyncMock(return_value="Extracted popup text content")

        content = await extractor.extract_popup_content(
            popup_page=mock_popup,
            popup_url="https://example.com/popup"
        )

        assert content == "Extracted popup text content"
        mock_popup.wait_for_load_state.assert_called_once()
        mock_popup.text_content.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
