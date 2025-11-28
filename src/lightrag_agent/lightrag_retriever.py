#!/usr/bin/env python3
"""
LightRAG Retriever for LangChain
=================================

LangChain-compatible retriever wrapper for LightRAG.

Features:
- BaseRetriever interface for seamless integration
- Query mode selection (local, global, hybrid)
- LightRAG result → LangChain Document conversion
- Async support with sync fallback
"""

import os
import re
import asyncio
import logging
from pathlib import Path
from typing import List, Optional, Any, Dict

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field

from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status

logger = logging.getLogger(__name__)


class LightRAGRetriever(BaseRetriever):
    """
    LangChain Retriever wrapper for LightRAG

    Integrates LightRAG graph-based retrieval with LangChain pipelines.

    Usage:
        retriever = LightRAGRetriever(
            working_dir="data/AI_HUB_DASAN_QA/09_lightrag/index",
            query_mode="hybrid",
            k=4
        )

        docs = retriever.invoke("서울 지하철 1호선 운행 시간은?")
    """

    k: int = Field(default=4, description="Number of documents to retrieve")
    working_dir: str = Field(..., description="LightRAG working directory")
    query_mode: str = Field(default="hybrid", description="Query mode: local, global, or hybrid")
    embedding_model: str = Field(default="text-embedding-3-large", description="Embedding model")
    llm_model: str = Field(default="gpt-4o-mini", description="LLM model")
    lightrag: Optional[Any] = Field(default=None, exclude=True)

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **kwargs):
        """Initialize retriever with LightRAG instance"""
        super().__init__(**kwargs)

        # Initialize LightRAG
        logger.info(f"🔧 Initializing LightRAG Retriever")
        logger.info(f"  Working dir: {self.working_dir}")
        logger.info(f"  Query mode: {self.query_mode}")
        logger.info(f"  k: {self.k}")

        # Check if index exists
        index_path = Path(self.working_dir)
        if not index_path.exists():
            raise FileNotFoundError(
                f"LightRAG index not found at {self.working_dir}\n"
                f"Build index first: python scripts/build_lightrag_index.py"
            )

        # Initialize LightRAG instance
        self.lightrag = LightRAG(
            working_dir=str(self.working_dir),
            llm_model_func=gpt_4o_mini_complete,  # Use built-in function
            embedding_func=openai_embed,  # Use built-in function
            llm_model_name=self.llm_model
        )

        logger.info(f"  ✅ LightRAG initialized")

        # Track initialization state
        self._initialized = False

    async def _ensure_initialized(self):
        """Ensure LightRAG storages are initialized"""
        if not self._initialized:
            await self.lightrag.initialize_storages()
            await initialize_pipeline_status()
            self._initialized = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Retrieve relevant documents using LightRAG

        Args:
            query: User query
            run_manager: Callback manager (optional)

        Returns:
            List of LangChain Document objects
        """
        # Run async query in sync context
        result = asyncio.run(self._aquery(query))

        # Parse result into documents
        documents = self._parse_lightrag_result(result, query)

        logger.info(f"  📄 Retrieved {len(documents)} documents from LightRAG")

        return documents[:self.k]

    async def _aquery(self, query: str) -> str:
        """
        Async query to LightRAG

        Args:
            query: User query

        Returns:
            LightRAG response text
        """
        # Ensure storages are initialized before query
        await self._ensure_initialized()

        result = await self.lightrag.aquery(
            query,
            param=QueryParam(mode=self.query_mode)
        )
        return result

    def _parse_lightrag_result(self, result: str, query: str) -> List[Document]:
        """
        Parse LightRAG text result into LangChain Documents

        Strategy:
        LightRAG returns formatted text response with sources.
        We need to extract individual chunks/contexts.

        Challenge:
        - LightRAG output is a synthesized answer, not raw chunks
        - We need to extract source contexts used to generate the answer

        Solution:
        1. Split result into paragraphs
        2. Each paragraph becomes a Document
        3. Include metadata (source: lightrag, query_mode, original_query)

        Args:
            result: LightRAG response text
            query: Original query

        Returns:
            List of Document objects
        """
        # Strategy 1: Split by paragraphs
        paragraphs = [p.strip() for p in result.split('\n\n') if p.strip()]

        documents = []

        for i, para in enumerate(paragraphs):
            # Skip very short paragraphs (likely formatting)
            if len(para) < 50:
                continue

            doc = Document(
                page_content=para,
                metadata={
                    "source": "lightrag",
                    "query_mode": self.query_mode,
                    "retrieval_type": "lightrag_graph",
                    "original_query": query,
                    "chunk_index": i,
                    "total_chunks": len(paragraphs)
                }
            )
            documents.append(doc)

        # If no documents, return the entire result as one document
        if not documents:
            documents = [Document(
                page_content=result,
                metadata={
                    "source": "lightrag",
                    "query_mode": self.query_mode,
                    "retrieval_type": "lightrag_graph",
                    "original_query": query
                }
            )]

        return documents

    def get_retriever_info(self) -> Dict[str, Any]:
        """Get retriever configuration info"""
        return {
            "retriever_type": "lightrag",
            "working_dir": self.working_dir,
            "query_mode": self.query_mode,
            "k": self.k,
            "embedding_model": self.embedding_model,
            "llm_model": self.llm_model
        }


def create_lightrag_retriever(
    working_dir: str = "data/AI_HUB_DASAN_QA/09_lightrag/index",
    query_mode: str = "hybrid",
    k: int = 4,
    **kwargs
) -> LightRAGRetriever:
    """
    Factory function to create LightRAG retriever

    Args:
        working_dir: LightRAG index directory
        query_mode: Query mode (local, global, hybrid)
        k: Number of documents to retrieve
        **kwargs: Additional arguments for LightRAGRetriever

    Returns:
        Configured LightRAGRetriever instance
    """
    return LightRAGRetriever(
        working_dir=working_dir,
        query_mode=query_mode,
        k=k,
        **kwargs
    )


def test_retriever():
    """Test retriever with sample query"""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))

    logging.basicConfig(level=logging.INFO)

    # Create retriever
    retriever = create_lightrag_retriever(
        working_dir="data/AI_HUB_DASAN_QA/09_lightrag/index",
        query_mode="hybrid",
        k=4
    )

    # Test query
    query = "서울 지하철 1호선 운행 시간은 언제인가요?"
    logger.info(f"🔍 Test query: {query}")

    docs = retriever.invoke(query)

    logger.info(f"\n📊 Retrieved {len(docs)} documents:")
    for i, doc in enumerate(docs, 1):
        logger.info(f"\n[Document {i}]")
        logger.info(f"Content (first 200 chars): {doc.page_content[:200]}...")
        logger.info(f"Metadata: {doc.metadata}")


if __name__ == "__main__":
    test_retriever()
