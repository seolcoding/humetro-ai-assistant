"""Tests for Stage 1: Data Collection."""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from langchain_core.documents import Document

from src.rag_pipeline.stages.stage_01_data_collection import (
    DataCollectionStage,
    create_data_collection_stage
)


class TestDataCollectionStage:
    """Test DataCollectionStage functionality."""

    @pytest.fixture
    def temp_dirs(self, tmp_path):
        """Create temporary test directories."""
        cache_dir = tmp_path / "seoul_traffic"
        markdown_dir = cache_dir / "markdown_deduplicated"
        markdown_dir.mkdir(parents=True)

        return {
            "cache_dir": cache_dir,
            "markdown_dir": markdown_dir
        }

    @pytest.fixture
    def sample_dirty_markdown(self):
        """Sample markdown with social sharing links."""
        return """# 서울시 교통 뉴스

서울시가 새로운 교통 정책을 발표했습니다.

  * [인쇄](https://news.seoul.go.kr/traffic/archives/505367#print "새창열림")
[스크랩](https://news.seoul.go.kr/traffic/archives/505367#sns_elem_dropdownmenu_top)

  * [ 카카오톡 ](https://news.seoul.go.kr/traffic/archives/505367#kakaotalk "새창열림")
  * [ 페이스북 ](https://news.seoul.go.kr/traffic/archives/505367#facebook "새창열림")
  * [ X ](https://news.seoul.go.kr/traffic/archives/505367#twitter "새창열림")
  * [ 네이버블로그 ](https://news.seoul.go.kr/traffic/archives/505367#naverblog "새창열림")
  * [ URL 복사 ](https://news.seoul.go.kr/traffic/archives/505367#none)

이번 정책은 대중교통 이용을 활성화하기 위한 것입니다.

  * [**네이버blog**](https://news.seoul.go.kr/traffic/archives/505367#네이버blog "새창열림")
  * [**메일전송**](https://news.seoul.go.kr/traffic/archives/505367#메일전송 "새창열림")
  * [**소스복사**](https://news.seoul.go.kr/traffic/archives/505367#sourceScrap_top "소스복사")

자세한 내용은 서울시 홈페이지를 참조하세요.
"""

    @pytest.fixture
    def sample_clean_markdown(self):
        """Expected cleaned markdown output."""
        return """# 서울시 교통 뉴스

서울시가 새로운 교통 정책을 발표했습니다.

이번 정책은 대중교통 이용을 활성화하기 위한 것입니다.

자세한 내용은 서울시 홈페이지를 참조하세요."""

    def test_stage_initialization(self, temp_dirs):
        """Test DataCollectionStage initialization."""
        stage = DataCollectionStage(
            transportation_cache_dir=temp_dirs["cache_dir"]
        )

        assert stage.transportation_cache_dir == temp_dirs["cache_dir"]
        assert stage.markdown_dir == temp_dirs["markdown_dir"]
        assert len(stage.compiled_patterns) > 0

    def test_check_cache_exists_with_valid_cache(self, temp_dirs):
        """Test cache checking with valid cache directory."""
        # Create a sample markdown file
        (temp_dirs["markdown_dir"] / "test.md").write_text("# Test")

        stage = DataCollectionStage(
            transportation_cache_dir=temp_dirs["cache_dir"]
        )

        assert stage.check_cache_exists() is True

    def test_check_cache_exists_missing_cache_dir(self, tmp_path):
        """Test cache checking with missing cache directory."""
        stage = DataCollectionStage(
            transportation_cache_dir=tmp_path / "nonexistent"
        )

        assert stage.check_cache_exists() is False

    def test_check_cache_exists_missing_markdown_dir(self, temp_dirs):
        """Test cache checking with missing markdown directory."""
        # Remove markdown directory
        temp_dirs["markdown_dir"].rmdir()

        stage = DataCollectionStage(
            transportation_cache_dir=temp_dirs["cache_dir"]
        )

        assert stage.check_cache_exists() is False

    def test_check_cache_exists_no_markdown_files(self, temp_dirs):
        """Test cache checking with no markdown files."""
        stage = DataCollectionStage(
            transportation_cache_dir=temp_dirs["cache_dir"]
        )

        assert stage.check_cache_exists() is False

    def test_clean_markdown_content(self, sample_dirty_markdown, sample_clean_markdown):
        """Test markdown content cleaning."""
        stage = DataCollectionStage()

        cleaned = stage.clean_markdown_content(sample_dirty_markdown)

        # Check that social links are removed
        assert "[인쇄]" not in cleaned
        assert "[스크랩]" not in cleaned
        assert "[ 카카오톡 ]" not in cleaned
        assert "[ 페이스북 ]" not in cleaned
        assert "[ X ]" not in cleaned
        assert "[ 네이버블로그 ]" not in cleaned
        assert "[ URL 복사 ]" not in cleaned
        assert "[**네이버blog**]" not in cleaned
        assert "[**메일전송**]" not in cleaned
        assert "[**소스복사**]" not in cleaned

        # Check that actual content is preserved
        assert "서울시 교통 뉴스" in cleaned
        assert "서울시가 새로운 교통 정책을 발표했습니다" in cleaned
        assert "대중교통 이용을 활성화" in cleaned

    def test_clean_markdown_content_removes_excessive_newlines(self):
        """Test that excessive blank lines are normalized."""
        stage = DataCollectionStage()

        content_with_many_newlines = "# Title\n\n\n\n\nContent\n\n\n\nMore content"
        cleaned = stage.clean_markdown_content(content_with_many_newlines)

        # Should have max 2 consecutive newlines
        assert "\n\n\n" not in cleaned

    def test_load_documents_cache_not_found(self, tmp_path):
        """Test loading documents when cache doesn't exist."""
        stage = DataCollectionStage(
            transportation_cache_dir=tmp_path / "nonexistent"
        )

        with pytest.raises(FileNotFoundError) as exc_info:
            stage.load_documents()

        assert "Cache not found" in str(exc_info.value)

    def test_load_documents_no_files(self, temp_dirs):
        """Test loading documents when no markdown files exist."""
        # Directory exists but no markdown files in it
        # This will fail at check_cache_exists() with FileNotFoundError
        stage = DataCollectionStage(
            transportation_cache_dir=temp_dirs["cache_dir"]
        )

        with pytest.raises(FileNotFoundError) as exc_info:
            stage.load_documents()

        assert "Cache not found" in str(exc_info.value)

    def test_load_documents_success(self, temp_dirs, sample_dirty_markdown):
        """Test successful document loading and cleaning."""
        # Create sample markdown files
        (temp_dirs["markdown_dir"] / "doc1.md").write_text(sample_dirty_markdown, encoding="utf-8")
        (temp_dirs["markdown_dir"] / "doc2.md").write_text("# Simple doc\n\nContent here.", encoding="utf-8")

        stage = DataCollectionStage(
            transportation_cache_dir=temp_dirs["cache_dir"]
        )

        documents = stage.load_documents()

        # Check we got 2 documents
        assert len(documents) == 2

        # Check documents are Document objects
        assert all(isinstance(doc, Document) for doc in documents)

        # Check metadata
        for doc in documents:
            assert doc.metadata.get("cleaned") is True
            assert "original_length" in doc.metadata
            assert "cleaned_length" in doc.metadata

        # Check cleaning was applied
        doc1 = next(d for d in documents if "교통 뉴스" in d.page_content)
        assert "[인쇄]" not in doc1.page_content
        assert "[ 카카오톡 ]" not in doc1.page_content

    def test_get_collection_stats_empty(self):
        """Test statistics for empty document list."""
        stage = DataCollectionStage()
        stats = stage.get_collection_stats([])

        assert stats["total_documents"] == 0
        assert stats["total_characters"] == 0
        assert stats["avg_document_length"] == 0

    def test_get_collection_stats_with_documents(self):
        """Test statistics calculation with documents."""
        stage = DataCollectionStage()

        docs = [
            Document(page_content="A" * 100),
            Document(page_content="B" * 200),
            Document(page_content="C" * 300)
        ]

        stats = stage.get_collection_stats(docs)

        assert stats["total_documents"] == 3
        assert stats["total_characters"] == 600
        assert stats["avg_document_length"] == 200
        assert stats["min_document_length"] == 100
        assert stats["max_document_length"] == 300

    def test_create_data_collection_stage_function(self, temp_dirs):
        """Test factory function."""
        stage = create_data_collection_stage(
            transportation_cache_dir=temp_dirs["cache_dir"]
        )

        assert isinstance(stage, DataCollectionStage)
        assert stage.transportation_cache_dir == temp_dirs["cache_dir"]

    def test_stage_with_logger(self, temp_dirs):
        """Test stage with logger integration."""
        mock_logger = Mock()
        mock_logger.verbose = False

        stage = DataCollectionStage(
            transportation_cache_dir=temp_dirs["cache_dir"],
            logger=mock_logger
        )

        # Logger should be called during initialization
        assert mock_logger.info.called

    def test_pattern_compilation(self):
        """Test that regex patterns are compiled correctly."""
        stage = DataCollectionStage()

        # Check all patterns compiled
        assert len(stage.compiled_patterns) == len(DataCollectionStage.SOCIAL_LINK_PATTERNS)

        # Check they're all compiled regex objects
        import re
        assert all(isinstance(p, re.Pattern) for p in stage.compiled_patterns)

    def test_metadata_preservation(self, temp_dirs):
        """Test that original metadata is preserved during cleaning."""
        # Create markdown with custom metadata via filename
        md_file = temp_dirs["markdown_dir"] / "traffic_archives_123.md"
        md_file.write_text("# Test\n\nContent", encoding="utf-8")

        stage = DataCollectionStage(
            transportation_cache_dir=temp_dirs["cache_dir"]
        )

        documents = stage.load_documents()

        # Check metadata exists
        assert len(documents) == 1
        doc = documents[0]
        assert "source" in doc.metadata  # Added by DirectoryLoader
        assert doc.metadata["cleaned"] is True
