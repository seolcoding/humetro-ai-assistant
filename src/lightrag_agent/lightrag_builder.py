#!/usr/bin/env python3
"""
LightRAG Index Builder
======================

Build LightRAG graph index from AI_HUB_DASAN_QA documents.

Features:
- Batch processing with progress tracking
- Checkpoint system for resume capability
- Configurable embedding and LLM models
- Error handling and logging
"""

import os
import json
import logging
import asyncio
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

from tqdm.asyncio import tqdm_asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete, gpt_4o_mini_complete, openai_embed
from lightrag.kg.shared_storage import initialize_pipeline_status

from .data_converter import DasanDataConverter

logger = logging.getLogger(__name__)


class LightRAGBuilder:
    """
    LightRAG index builder for Dasan QA dataset

    Builds graph-based knowledge index from JSONL documents.
    """

    def __init__(
        self,
        working_dir: Path,
        embedding_model: str = "text-embedding-3-large",
        llm_model: str = "gpt-4o-mini",
        enable_cache: bool = True
    ):
        """
        Initialize LightRAG builder

        Args:
            working_dir: LightRAG working directory (stores graph, embeddings)
            embedding_model: OpenAI embedding model name
            llm_model: LLM model for entity extraction
            enable_cache: Enable LLM response caching
        """
        self.working_dir = Path(working_dir)
        self.working_dir.mkdir(parents=True, exist_ok=True)

        self.embedding_model = embedding_model
        self.llm_model = llm_model

        # Save build config
        self.config = {
            "embedding_model": embedding_model,
            "llm_model": llm_model,
            "enable_cache": enable_cache,
            "created_at": datetime.now().isoformat()
        }
        self._save_config()

        # Initialize LightRAG
        logger.info(f"🚀 Initializing LightRAG")
        logger.info(f"  Working dir: {working_dir}")
        logger.info(f"  Embedding: {embedding_model}")
        logger.info(f"  LLM: {llm_model}")

        self.rag = LightRAG(
            working_dir=str(self.working_dir),
            llm_model_func=gpt_4o_mini_complete,  # Use built-in function
            embedding_func=openai_embed,  # Use built-in function
            llm_model_name=llm_model,
            enable_llm_cache=enable_cache
        )

        # Initialize storages (required by LightRAG)
        self._initialized = False

    async def _ensure_initialized(self):
        """Ensure LightRAG storages are initialized"""
        if not self._initialized:
            logger.info("  Initializing storages...")
            await self.rag.initialize_storages()
            await initialize_pipeline_status()
            self._initialized = True
            logger.info("  ✅ Storages initialized")

    def _save_config(self):
        """Save build configuration"""
        config_file = self.working_dir / "build_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2)

    def _load_checkpoint(self) -> int:
        """
        Load checkpoint

        Returns:
            Number of documents already processed
        """
        checkpoint_file = self.working_dir / "build_checkpoint.json"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
                return data.get("processed_documents", 0)
        return 0

    def _save_checkpoint(self, processed: int):
        """Save checkpoint"""
        checkpoint_file = self.working_dir / "build_checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            json.dump({
                "processed_documents": processed,
                "last_updated": datetime.now().isoformat()
            }, f, indent=2)

    async def build_from_jsonl(
        self,
        jsonl_path: Path,
        batch_size: int = 10,
        max_docs: Optional[int] = None
    ):
        """
        Build LightRAG index from JSONL file

        Args:
            jsonl_path: Path to knowledge_docs_full.jsonl
            batch_size: Documents per batch
            max_docs: Maximum documents to process (None = all)

        Process:
        1. Load documents from JSONL
        2. Convert to LightRAG text format
        3. Insert in batches with progress tracking
        4. Save checkpoints for resume capability
        """
        # Ensure storages are initialized
        await self._ensure_initialized()

        logger.info("="*70)
        logger.info("Building LightRAG Index")
        logger.info("="*70)

        # Load converter
        converter = DasanDataConverter(jsonl_path)
        stats = converter.get_statistics()
        logger.info(f"📊 Dataset: {stats['total_documents']} documents")

        # Convert documents
        logger.info("📝 Converting documents...")
        documents = converter.convert_to_text_documents(max_docs=max_docs)

        # Load checkpoint
        checkpoint = self._load_checkpoint()
        if checkpoint > 0:
            logger.info(f"🔄 Resuming from checkpoint: {checkpoint} docs already processed")
            documents = documents[checkpoint:]

        logger.info(f"📚 Processing {len(documents)} documents (batch_size={batch_size})")

        # Process in batches
        processed = checkpoint
        total_batches = (len(documents) + batch_size - 1) // batch_size

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            batch_num = i // batch_size + 1

            try:
                # Combine batch into single text (LightRAG handles chunking)
                batch_text = "\n\n".join(batch)

                logger.info(f"  Batch {batch_num}/{total_batches}: {len(batch)} docs")

                # Insert into LightRAG
                await self.rag.ainsert(batch_text)

                # Update checkpoint
                processed += len(batch)
                self._save_checkpoint(processed)

            except Exception as e:
                logger.error(f"❌ Error processing batch {batch_num}: {e}")
                logger.info(f"  Checkpoint saved at {processed} documents")
                raise

        logger.info("="*70)
        logger.info(f"✅ Index build complete!")
        logger.info(f"  Total documents: {processed}")
        logger.info(f"  Working dir: {self.working_dir}")
        logger.info("="*70)

    async def test_query(self, query: str, mode: str = "hybrid") -> str:
        """
        Test query on built index

        Args:
            query: Test question
            mode: Query mode (local, global, hybrid)

        Returns:
            LightRAG response
        """
        await self._ensure_initialized()

        logger.info(f"🔍 Test query: {query}")
        logger.info(f"  Mode: {mode}")

        result = await self.rag.aquery(
            query,
            param=QueryParam(mode=mode)
        )

        logger.info(f"  Result length: {len(result)} chars")
        return result


async def main():
    """CLI for building LightRAG index"""
    import argparse

    parser = argparse.ArgumentParser(description="Build LightRAG index from Dasan QA")
    parser.add_argument(
        "--input",
        default="data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs_full.jsonl",
        help="Input JSONL file"
    )
    parser.add_argument(
        "--working-dir",
        default="data/AI_HUB_DASAN_QA/09_lightrag/index",
        help="LightRAG working directory"
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-large",
        help="OpenAI embedding model"
    )
    parser.add_argument(
        "--llm-model",
        default="gpt-4o-mini",
        help="LLM model for entity extraction"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Documents per batch"
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        help="Maximum documents to process (for testing)"
    )
    parser.add_argument(
        "--test-query",
        help="Test query after building"
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S'
    )

    # Build index
    builder = LightRAGBuilder(
        working_dir=Path(args.working_dir),
        embedding_model=args.embedding_model,
        llm_model=args.llm_model
    )

    await builder.build_from_jsonl(
        jsonl_path=Path(args.input),
        batch_size=args.batch_size,
        max_docs=args.max_docs
    )

    # Test query
    if args.test_query:
        result = await builder.test_query(args.test_query, mode="hybrid")
        print("\n" + "="*70)
        print("Test Query Result:")
        print("="*70)
        print(result)


if __name__ == "__main__":
    asyncio.run(main())
