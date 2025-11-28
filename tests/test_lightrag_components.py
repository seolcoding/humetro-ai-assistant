#!/usr/bin/env python3
"""
Unit Tests for LightRAG Components
===================================

Test each component before running full pipeline:
1. Data converter (JSONL → text)
2. LightRAG builder (small sample)
3. Retriever wrapper
"""

import sys
import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lightrag_agent.data_converter import DasanDataConverter


class TestDataConverter:
    """Test DasanDataConverter"""

    def test_converter_initialization(self):
        """Test converter initialization"""
        # Should succeed with valid path
        converter = DasanDataConverter(
            Path("data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs_full.jsonl")
        )
        assert converter.jsonl_path.exists()

    def test_load_jsonl(self):
        """Test JSONL loading"""
        converter = DasanDataConverter(
            Path("data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs_full.jsonl")
        )

        # Load first 10 entries
        entries = list(converter.load_jsonl())[:10]

        assert len(entries) > 0
        assert "dialogue_id" in entries[0]
        assert "original_question" in entries[0]
        assert "original_answer" in entries[0]
        assert "document" in entries[0]

        print(f"✅ Loaded {len(entries)} entries")
        print(f"   Sample dialogue_id: {entries[0]['dialogue_id']}")

    def test_convert_to_text_documents(self):
        """Test document conversion"""
        converter = DasanDataConverter(
            Path("data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs_full.jsonl")
        )

        # Convert first 5 documents
        documents = converter.convert_to_text_documents(max_docs=5)

        assert len(documents) == 5
        assert all(isinstance(doc, str) for doc in documents)
        assert all("Question:" in doc for doc in documents)
        assert all("Answer:" in doc for doc in documents)
        assert all("Context:" in doc for doc in documents)

        print(f"✅ Converted {len(documents)} documents")
        print(f"   First doc length: {len(documents[0])} chars")
        print(f"   Sample:\n{documents[0][:200]}...")

    def test_get_statistics(self):
        """Test statistics collection"""
        converter = DasanDataConverter(
            Path("data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs_full.jsonl")
        )

        stats = converter.get_statistics()

        assert "total_documents" in stats
        assert "categories" in stats
        assert stats["total_documents"] > 0

        print(f"✅ Statistics:")
        print(f"   Total documents: {stats['total_documents']}")
        print(f"   Categories: {stats['categories']}")


class TestLightRAGBuilder:
    """Test LightRAG builder (requires OpenAI API key)"""

    @pytest.mark.skipif(
        not Path("data/AI_HUB_DASAN_QA/09_lightrag/test_index").exists(),
        reason="Test index not built yet"
    )
    def test_builder_initialization(self):
        """Test builder initialization"""
        from src.lightrag_agent.lightrag_builder import LightRAGBuilder

        builder = LightRAGBuilder(
            working_dir=Path("data/AI_HUB_DASAN_QA/09_lightrag/test_index"),
            embedding_model="text-embedding-3-large",
            llm_model="gpt-4o-mini"
        )

        assert builder.working_dir.exists()
        assert builder.rag is not None

        print("✅ Builder initialized successfully")


class TestLightRAGRetriever:
    """Test LightRAG retriever (requires built index)"""

    @pytest.mark.skipif(
        not Path("data/AI_HUB_DASAN_QA/09_lightrag/index").exists(),
        reason="Index not built yet"
    )
    def test_retriever_initialization(self):
        """Test retriever initialization"""
        from src.lightrag_agent.lightrag_retriever import create_lightrag_retriever

        retriever = create_lightrag_retriever(
            working_dir="data/AI_HUB_DASAN_QA/09_lightrag/index",
            query_mode="hybrid",
            k=4
        )

        assert retriever is not None
        assert retriever.k == 4
        assert retriever.query_mode == "hybrid"

        print("✅ Retriever initialized successfully")

    @pytest.mark.skipif(
        not Path("data/AI_HUB_DASAN_QA/09_lightrag/index").exists(),
        reason="Index not built yet"
    )
    def test_retriever_query(self):
        """Test retriever query"""
        from src.lightrag_agent.lightrag_retriever import create_lightrag_retriever

        retriever = create_lightrag_retriever(
            working_dir="data/AI_HUB_DASAN_QA/09_lightrag/index",
            query_mode="hybrid",
            k=4
        )

        # Test query
        query = "서울 지하철 1호선 운행 시간은?"
        docs = retriever.invoke(query)

        assert len(docs) > 0
        assert len(docs) <= 4  # Should return at most k documents
        assert all(hasattr(doc, "page_content") for doc in docs)
        assert all(hasattr(doc, "metadata") for doc in docs)

        print(f"✅ Retrieved {len(docs)} documents")
        for i, doc in enumerate(docs, 1):
            print(f"\n[Doc {i}]")
            print(f"Content length: {len(doc.page_content)} chars")
            print(f"Metadata: {doc.metadata}")


def run_converter_tests():
    """Run data converter tests"""
    print("="*70)
    print("Testing Data Converter")
    print("="*70)

    test = TestDataConverter()

    try:
        test.test_converter_initialization()
        print("✅ Initialization test passed\n")
    except Exception as e:
        print(f"❌ Initialization test failed: {e}\n")
        return False

    try:
        test.test_load_jsonl()
        print("✅ JSONL loading test passed\n")
    except Exception as e:
        print(f"❌ JSONL loading test failed: {e}\n")
        return False

    try:
        test.test_convert_to_text_documents()
        print("✅ Document conversion test passed\n")
    except Exception as e:
        print(f"❌ Document conversion test failed: {e}\n")
        return False

    try:
        test.test_get_statistics()
        print("✅ Statistics test passed\n")
    except Exception as e:
        print(f"❌ Statistics test failed: {e}\n")
        return False

    print("="*70)
    print("✅ All Data Converter tests passed!")
    print("="*70)
    return True


if __name__ == "__main__":
    run_converter_tests()
