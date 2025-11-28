#!/usr/bin/env python3
"""
Build LightRAG Index Script
============================

Convenience script to build LightRAG index from command line.

Usage:
    # Full index (45k documents, ~5-8 hours)
    python scripts/build_lightrag_index.py

    # Quick test (100 documents)
    python scripts/build_lightrag_index.py --max-docs 100

    # Custom settings
    python scripts/build_lightrag_index.py \
        --embedding-model text-embedding-3-small \
        --llm-model gpt-3.5-turbo \
        --batch-size 20

    # Test query after building
    python scripts/build_lightrag_index.py \
        --max-docs 100 \
        --test-query "서울 지하철 1호선 운행 시간은?"
"""

import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.lightrag_agent.lightrag_builder import main

if __name__ == "__main__":
    asyncio.run(main())
