"""RAG Pipeline Stages.

This package contains modular pipeline stages for the RAG system.
Each stage is independent and composable.
"""

from src.rag_pipeline.stages.stage_01_data_collection import (
    DataCollectionStage,
    create_data_collection_stage
)

__all__ = [
    "DataCollectionStage",
    "create_data_collection_stage",
]
