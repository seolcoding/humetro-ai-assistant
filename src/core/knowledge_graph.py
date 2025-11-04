"""
Knowledge graph generation module.

This module provides knowledge graph generation functionality
using RAGAS framework for enhanced RAG retrieval.
"""

from typing import List, Optional
from langchain_core.documents import Document
from ragas.testset import KnowledgeGraph
from ragas.testset.graph import Node, NodeType
from ragas.testset.transforms import (
    Transforms,
    HeadlinesExtractor,
    HeadlineSplitter,
    KeyphrasesExtractor
)

from src.common.logger import RAGLogger


class KnowledgeGraphBuilder:
    """
    Builds knowledge graphs from documents using RAGAS.

    Creates structured knowledge representation for improved
    RAG retrieval and evaluation.
    """

    def __init__(
        self,
        llm_model: str = "gpt-4o-mini",
        logger: Optional[RAGLogger] = None
    ):
        """
        Initialize knowledge graph builder.

        Args:
            llm_model: LLM model for knowledge extraction
            logger: Optional RAG logger
        """
        self.llm_model = llm_model
        self.logger = logger

        # Initialize transforms
        self.transforms = self._create_transforms()

    def _create_transforms(self) -> Transforms:
        """
        Create RAGAS transforms for knowledge extraction.

        Returns:
            Transforms instance
        """
        return Transforms(
            HeadlinesExtractor(),
            HeadlineSplitter(),
            KeyphrasesExtractor()
        )

    def build_from_documents(self, documents: List[Document]) -> KnowledgeGraph:
        """
        Build knowledge graph from documents.

        Args:
            documents: List of documents to process

        Returns:
            KnowledgeGraph instance
        """
        if self.logger:
            self.logger.info(f"Building knowledge graph from {len(documents)} documents")

        # Create initial knowledge graph
        kg = KnowledgeGraph()

        # Add documents as nodes
        for doc in documents:
            node = Node(
                type=NodeType.DOCUMENT,
                properties={
                    "content": doc.page_content,
                    "metadata": doc.metadata
                }
            )
            kg.nodes.append(node)

        if self.logger:
            self.logger.info(f"Created initial KG with {len(kg.nodes)} document nodes")

        # Apply transforms
        if self.logger:
            self.logger.info("Applying knowledge extraction transforms...")

        kg = self.transforms.apply(kg)

        if self.logger:
            self.logger.info(f"KG transformation complete: {len(kg.nodes)} total nodes")

        return kg

    def get_kg_stats(self, kg: KnowledgeGraph) -> dict:
        """
        Get knowledge graph statistics.

        Args:
            kg: Knowledge graph instance

        Returns:
            Dictionary with KG statistics
        """
        node_types = {}
        for node in kg.nodes:
            node_type = node.type.value if hasattr(node.type, 'value') else str(node.type)
            node_types[node_type] = node_types.get(node_type, 0) + 1

        return {
            "total_nodes": len(kg.nodes),
            "total_relationships": len(kg.relationships),
            "node_types": node_types
        }

    def print_stats(self, kg: KnowledgeGraph):
        """Print knowledge graph statistics."""
        stats = self.get_kg_stats(kg)

        print(f"\n{'='*60}")
        print(f"KNOWLEDGE GRAPH STATISTICS")
        print(f"{'='*60}")
        print(f"Total nodes: {stats['total_nodes']:,}")
        print(f"Total relationships: {stats['total_relationships']:,}")
        print(f"\nNode types:")
        for node_type, count in stats['node_types'].items():
            print(f"  - {node_type}: {count:,}")
        print(f"{'='*60}\n")


def create_kg_builder(
    llm_model: str = "gpt-4o-mini",
    logger: Optional[RAGLogger] = None
) -> KnowledgeGraphBuilder:
    """
    Create knowledge graph builder.

    Args:
        llm_model: LLM model for knowledge extraction
        logger: Optional RAG logger

    Returns:
        KnowledgeGraphBuilder instance
    """
    return KnowledgeGraphBuilder(llm_model=llm_model, logger=logger)
