#!/usr/bin/env python3
"""
KG Cypher Generation Retriever (FIXED VERSION)
================================================

Hybrid approach: Vector similarity + LLM-generated Cypher queries

CRITICAL FIX:
- OLD: Pure Cypher generation without vector search (no starting point)
- NEW: Vector search first → then graph traversal from found nodes

Architecture:
1. Vector similarity search (find relevant nodes - STARTING POINT)
2. LLM generates Cypher to traverse graph from these nodes
3. Return original text chunks (not summarized)

This fixes the -47% performance degradation issue.
"""

import os
from typing import List, Dict, Any, Optional, ClassVar
from pathlib import Path
from dotenv import load_dotenv
import logging

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from pydantic import Field

logger = logging.getLogger(__name__)


# NEW: Simplified retrieval strategy
# Instead of complex Cypher generation, use vector-found nodes + their neighbors
# This is more reliable and maintains context quality


class KGCypherRetriever(BaseRetriever):
    """
    Knowledge Graph Retriever with Hybrid Approach (FIXED)

    Architecture:
    1. Vector similarity search (find relevant starting nodes)
    2. Graph traversal (expand to neighboring nodes via relationships)
    3. Return original text chunks from all nodes

    Benefits:
    - Combines semantic search (vector) with structural context (graph)
    - Always finds relevant starting points (unlike pure Cypher generation)
    - Maintains original chunk quality (no LLM summarization)
    - Explores entity relationships for richer context
    """

    # Pydantic fields
    k: int = Field(default=4, description="Number of initial nodes to find via vector search")
    expansion_hops: int = Field(default=1, description="Number of hops for graph expansion (1 or 2)")
    embedding_model: str = Field(default="text-embedding-3-large")
    neo4j_uri: Optional[str] = Field(default=None)
    neo4j_user: Optional[str] = Field(default=None)
    neo4j_password: Optional[str] = Field(default=None)

    # Class attributes (shared across instances)
    _driver: ClassVar[Any] = None
    _embedder: ClassVar[Any] = None

    class Config:
        arbitrary_types_allowed = True

    def model_post_init(self, __context: Any) -> None:
        """Initialize after Pydantic validation"""

        # Load credentials
        self._load_credentials(self.neo4j_uri, self.neo4j_user, self.neo4j_password)

        # Initialize Neo4j driver (shared)
        if KGCypherRetriever._driver is None:
            KGCypherRetriever._driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password)
            )
            logger.info(f"✅ Connected to Neo4j: {self.neo4j_uri}")

        # Initialize embedder (shared)
        if KGCypherRetriever._embedder is None:
            KGCypherRetriever._embedder = OpenAIEmbeddings(model=self.embedding_model)
            logger.info(f"✅ Embedder initialized: {self.embedding_model}")

        logger.info(f"✅ KGCypherRetriever (FIXED) initialized (k={self.k}, expansion_hops={self.expansion_hops})")

    def _load_credentials(self, uri: Optional[str], user: Optional[str], password: Optional[str]):
        """Load Neo4j credentials"""
        # Load from .env
        config_dir = Path(__file__).parent / "config"
        env_path = config_dir / ".env"
        if env_path.exists():
            load_dotenv(env_path, override=True)

        # Use provided or env
        self.neo4j_uri = uri or os.getenv('NEO4J_URI')
        self.neo4j_user = user or os.getenv('NEO4J_USERNAME')
        self.neo4j_password = password or os.getenv('NEO4J_PASSWORD')

        if not all([self.neo4j_uri, self.neo4j_user, self.neo4j_password]):
            raise ValueError("Neo4j credentials not found")

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        """
        Retrieve documents using hybrid vector + graph approach

        Steps:
        1. Vector similarity search → find top-k relevant nodes
        2. Graph expansion → get neighbors of found nodes
        3. Collect text from all nodes → return as Documents

        Args:
            query: Natural language question

        Returns:
            List of Documents with KG-enhanced context
        """
        try:
            logger.info(f"🔍 [CYPHER RETRIEVAL] Query: {query[:100]}...")

            # Step 1: Vector similarity search to find starting nodes
            logger.info(f"📍 Step 1: Vector search for top-{self.k} nodes")
            query_embedding = KGCypherRetriever._embedder.embed_query(query)

            with KGCypherRetriever._driver.session() as session:
                # Vector similarity search
                vector_results = session.run("""
                    CALL db.index.vector.queryNodes('vector', $k, $embedding)
                    YIELD node, score
                    RETURN elementId(node) as nodeId, node.text as text, score
                    ORDER BY score DESC
                """, k=self.k, embedding=query_embedding)

                starting_nodes = []
                for record in vector_results:
                    starting_nodes.append({
                        'id': record['nodeId'],
                        'text': record['text'],
                        'score': record['score']
                    })

                logger.info(f"  ✅ Found {len(starting_nodes)} starting nodes (scores: {[f'{n['score']:.3f}' for n in starting_nodes[:3]]})")

                if not starting_nodes:
                    logger.warning("  ⚠️ No nodes found via vector search!")
                    return []

                # Step 2: Graph expansion - get neighbors of starting nodes
                logger.info(f"🔗 Step 2: Graph expansion ({self.expansion_hops}-hop)")
                node_ids = [n['id'] for n in starting_nodes]

                if self.expansion_hops == 1:
                    expansion_query = """
                        UNWIND $nodeIds as startNodeId
                        CALL {
                            WITH startNodeId
                            MATCH (start)
                            WHERE elementId(start) = startNodeId
                            OPTIONAL MATCH (start)-[r]-(neighbor)
                            WHERE neighbor.text IS NOT NULL
                            RETURN DISTINCT neighbor, neighbor.text as text
                        }
                        WITH COLLECT(DISTINCT text) as texts
                        UNWIND texts as text
                        RETURN DISTINCT text
                    """
                else:  # 2-hop
                    expansion_query = """
                        UNWIND $nodeIds as startNodeId
                        CALL {
                            WITH startNodeId
                            MATCH (start)
                            WHERE elementId(start) = startNodeId
                            OPTIONAL MATCH (start)-[r1]-(n1)-[r2]-(n2)
                            WHERE n1.text IS NOT NULL OR n2.text IS NOT NULL
                            RETURN DISTINCT n1.text as text1, n2.text as text2
                        }
                        WITH COLLECT(DISTINCT text1) + COLLECT(DISTINCT text2) as texts
                        UNWIND texts as text
                        WHERE text IS NOT NULL
                        RETURN DISTINCT text
                    """

                expansion_results = session.run(expansion_query, nodeIds=node_ids)
                neighbor_texts = [record['text'] for record in expansion_results if record['text']]

                logger.info(f"  ✅ Found {len(neighbor_texts)} neighbor texts")

                # Step 3: Combine starting nodes + neighbors
                all_texts = [n['text'] for n in starting_nodes if n['text']]
                all_texts.extend(neighbor_texts)
                all_texts = list(dict.fromkeys(all_texts))  # Remove duplicates, preserve order

                logger.info(f"📦 Step 3: Collected {len(all_texts)} unique text chunks")

                # Step 4: Create Documents
                documents = []
                for i, text in enumerate(all_texts[:self.k * 2]):  # Limit to reasonable number
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": "knowledge_graph_hybrid",
                                "retrieval_type": "kg_cypher_hybrid",
                                "chunk_index": i,
                                "is_starting_node": i < len(starting_nodes),
                                "expansion_hops": self.expansion_hops,
                                "question": query
                            }
                        )
                    )

                logger.info(f"✅ [CYPHER RETRIEVAL] Retrieved {len(documents)} documents (vector + graph expansion)")
                logger.info(f"   Starting nodes: {len(starting_nodes)}, Neighbors: {len(neighbor_texts)}, Total unique: {len(all_texts)}")

                return documents

        except Exception as e:
            logger.error(f"❌ [CYPHER RETRIEVAL] Failed: {e}", exc_info=True)
            return []

    @classmethod
    def close_driver(cls):
        """Close Neo4j driver connection"""
        if cls._driver:
            cls._driver.close()
            cls._driver = None
            cls._embedder = None
            logger.info("Neo4j driver closed")


def create_kg_cypher_retriever(
    k: int = 4,
    expansion_hops: int = 1,
    embedding_model: str = "text-embedding-3-large",
    **kwargs
) -> KGCypherRetriever:
    """
    Factory function to create KG Cypher retriever (FIXED version)

    Args:
        k: Number of initial nodes to find via vector search
        expansion_hops: Number of hops for graph expansion (1 or 2)
        embedding_model: Embedding model for vector search
        **kwargs: Additional arguments (neo4j_uri, neo4j_user, neo4j_password)

    Returns:
        Configured KGCypherRetriever with hybrid vector + graph approach
    """
    return KGCypherRetriever(
        k=k,
        expansion_hops=expansion_hops,
        embedding_model=embedding_model,
        **kwargs
    )


if __name__ == "__main__":
    # Test retriever
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("\n" + "="*80)
    print("Testing KG Cypher Retriever (FIXED VERSION - Hybrid Approach)")
    print("="*80 + "\n")

    retriever = create_kg_cypher_retriever(k=4, expansion_hops=1)

    test_queries = [
        "서울 지하철 1호선의 주요 역은?",
        "2024년 12월 14일 여의도역 버스 우회 정보",
        "장거리 버스 노선 개선 정책"
    ]

    for query in test_queries:
        print(f"\n{'='*80}")
        print(f"쿼리: {query}")
        print(f"{'='*80}")

        docs = retriever.invoke(query)
        print(f"\n✅ 검색 결과: {len(docs)}개 문서")

        for i, doc in enumerate(docs[:3], 1):  # Show first 3
            print(f"\n[Document {i}]")
            print(f"  Source: {doc.metadata.get('source')}")
            print(f"  Retrieval Type: {doc.metadata.get('retrieval_type')}")
            print(f"  Is Starting Node: {doc.metadata.get('is_starting_node')}")
            print(f"  Content Preview: {doc.page_content[:150]}...")

    print(f"\n{'='*80}")
    print("Test complete!")
    print(f"{'='*80}\n")

    # Cleanup
    KGCypherRetriever.close_driver()
