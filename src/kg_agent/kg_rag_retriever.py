#!/usr/bin/env python3
"""
Knowledge Graph RAG Retriever
Combines vector search with graph traversal for enhanced context retrieval
"""

from typing import List, Dict, Any, Optional, ClassVar
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from neo4j import GraphDatabase
from neo4j_graphrag.retrievers import VectorCypherRetriever
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from neo4j_graphrag.llm import OpenAILLM
import os
from pathlib import Path
from dotenv import load_dotenv
import logging
from pydantic import Field

logger = logging.getLogger(__name__)


class KGRAGRetriever(BaseRetriever):
    """
    Knowledge Graph RAG Retriever

    Implements LangChain BaseRetriever interface using Neo4j GraphRAG.
    Combines:
    - Vector similarity search (embedding-based)
    - Graph traversal (relationship-based context expansion)
    - Entity-aware context aggregation

    Benefits over naive vector search:
    - Captures entity relationships
    - Provides structured context
    - Reduces semantic drift
    - Better multi-hop reasoning
    """

    # Pydantic fields
    k: int = Field(default=4, description="Number of documents to retrieve")
    embedding_model: str = Field(default="text-embedding-3-large")
    neo4j_uri: Optional[str] = Field(default=None)
    neo4j_user: Optional[str] = Field(default=None)
    neo4j_password: Optional[str] = Field(default=None)
    retrieval_query: Optional[str] = Field(default=None)
    neo4j_retriever: Optional[Any] = Field(default=None)

    # Class attributes for Neo4j connection (shared across instances)
    _driver: ClassVar[Any] = None
    _embedder: ClassVar[Any] = None

    class Config:
        arbitrary_types_allowed = True

    def model_post_init(self, __context: Any) -> None:
        """
        Initialize KG RAG Retriever after Pydantic validation

        This is called after __init__ in Pydantic v2
        """
        # Load Neo4j credentials
        self._load_credentials(self.neo4j_uri, self.neo4j_user, self.neo4j_password)

        # Initialize Neo4j driver (shared)
        if KGRAGRetriever._driver is None:
            KGRAGRetriever._driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password)
            )
            logger.info(f"✅ Connected to Neo4j: {self.neo4j_uri}")

        # Initialize embedder (shared) - matches notebook configuration
        if KGRAGRetriever._embedder is None:
            KGRAGRetriever._embedder = OpenAIEmbeddings(model=self.embedding_model)
            logger.info(f"✅ Embedder initialized: {self.embedding_model}")

        # Build retrieval query
        if not self.retrieval_query:
            self.retrieval_query = self._build_default_retrieval_query()

        # Create Neo4j GraphRAG retriever
        self.neo4j_retriever = VectorCypherRetriever(
            driver=KGRAGRetriever._driver,
            index_name="vector",  # Vector index created by SimpleKGPipeline
            embedder=KGRAGRetriever._embedder,
            retrieval_query=self.retrieval_query
        )

        logger.info(f"✅ KGRAGRetriever initialized (k={self.k})")

    def _load_credentials(self, uri: Optional[str], user: Optional[str], password: Optional[str]):
        """Load Neo4j credentials from .env or parameters"""
        # Load from .env
        config_dir = Path(__file__).parent / "config"
        env_path = config_dir / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=True)

        # Use provided values or fall back to environment
        self.neo4j_uri = uri or os.getenv('NEO4J_URI')
        self.neo4j_user = user or os.getenv('NEO4J_USERNAME')
        self.neo4j_password = password or os.getenv('NEO4J_PASSWORD')

        if not all([self.neo4j_uri, self.neo4j_user, self.neo4j_password]):
            raise ValueError(
                "Neo4j credentials not found. "
                "Provide via parameters or set NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD"
            )

    def _build_default_retrieval_query(self) -> str:
        """
        Build default Cypher query for context retrieval

        Simple strategy: Just return the chunk text
        (Complex graph traversal can be added later)

        Returns:
            Cypher query string
        """
        query = """
        // Return chunk text only (simple version)
        RETURN node.text as text, score
        """
        return query

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Retrieve relevant documents using KG-enhanced search

        Args:
            query: Search query
            run_manager: Optional callback manager

        Returns:
            List of LangChain Documents with KG-enriched context
        """
        try:
            logger.info(f"Starting KG retrieval for query: {query[:100]}...")
            logger.info(f"Using index: vector, top_k: {self.k}")

            # Use Neo4j GraphRAG retriever
            logger.info("Calling neo4j_retriever.search()...")
            results = self.neo4j_retriever.search(
                query_text=query,
                top_k=self.k
            )
            logger.info(f"Retrieved {len(results.items)} results from Neo4j")

            # Convert to LangChain Documents
            documents = []
            for i, item in enumerate(results.items):
                logger.debug(f"Processing result {i+1}: {str(item.content)[:100]}...")

                # Combine chunk text with entity context
                content = self._format_context(item.content, item.metadata)

                # Create Document with metadata (handle None metadata gracefully)
                item_meta = item.metadata if item.metadata is not None else {}
                doc = Document(
                    page_content=content,
                    metadata={
                        "id": item_meta.get("id", ""),
                        "document_id": item_meta.get("document_id", ""),
                        "score": item_meta.get("score", 0.0),
                        "source": "knowledge_graph",
                        "retrieval_type": "kg_rag"
                    }
                )
                documents.append(doc)

            logger.info(f"✅ Successfully converted {len(documents)} documents")
            return documents

        except Exception as e:
            logger.error(f"❌ KG retrieval failed: {e}", exc_info=True)
            # Return empty list instead of failing
            return []

    def _format_context(self, text: str, metadata: Dict[str, Any]) -> str:
        """
        Format context by combining chunk text with entity relationships

        Args:
            text: Original chunk text
            metadata: Metadata containing entity_relationships (can be None)

        Returns:
            Enriched context string
        """
        # Start with original text
        context = text

        # Add entity relationship context if available
        if metadata is None:
            return context

        entity_rels = metadata.get("entity_relationships", [])
        if entity_rels and len(entity_rels) > 0:
            # Filter out empty relationships
            valid_rels = [
                rel for rel in entity_rels
                if rel.get("entity") and rel.get("related")
            ]

            if valid_rels:
                context += "\n\n[관련 엔티티 정보]"
                for rel in valid_rels[:5]:  # Limit to 5 relationships
                    entity = rel.get("entity", "?")
                    entity_type = rel.get("entity_type", "?")
                    relationship = rel.get("relationship", "?")
                    related = rel.get("related", "?")
                    related_type = rel.get("related_type", "?")

                    context += f"\n- [{entity_type}] {entity} --[{relationship}]--> [{related_type}] {related}"

        return context

    @classmethod
    def close_driver(cls):
        """Close Neo4j driver (call when done with all retrievers)"""
        if cls._driver:
            cls._driver.close()
            cls._driver = None
            logger.info("Neo4j driver closed")


def create_kg_retriever(k: int = 4, **kwargs) -> KGRAGRetriever:
    """
    Factory function to create KG RAG retriever

    Args:
        k: Number of documents to retrieve
        **kwargs: Additional arguments for KGRAGRetriever

    Returns:
        Configured KGRAGRetriever instance
    """
    return KGRAGRetriever(k=k, **kwargs)


if __name__ == "__main__":
    # Test retriever
    logging.basicConfig(level=logging.INFO)

    print("Testing KG RAG Retriever...")
    retriever = create_kg_retriever(k=3)

    test_query = "서울 지하철 1호선 종각역 출구 정보"
    print(f"\n쿼리: {test_query}")

    docs = retriever._get_relevant_documents(test_query)
    print(f"\n검색 결과: {len(docs)}개 문서")

    for i, doc in enumerate(docs, 1):
        print(f"\n[Document {i}]")
        print(f"Score: {doc.metadata.get('score', 'N/A')}")
        print(f"Content preview: {doc.page_content[:200]}...")

    KGRAGRetriever.close_driver()
    print("\n✅ Test complete!")
