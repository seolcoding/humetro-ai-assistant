#!/usr/bin/env python3
"""
KG Cypher Generation Retriever
================================

LLM-generated Cypher queries for dynamic graph traversal.
Based on notebook L7-chat_with_kg.ipynb pattern.

Differences from kg_rag_retriever.py:
- kg_rag_retriever: Fixed Cypher (vector search only)
- kg_cypher_retriever: LLM-generated Cypher (dynamic graph traversal)
"""

import os
from typing import List, Dict, Any, Optional, ClassVar
from pathlib import Path
from dotenv import load_dotenv
import logging

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.prompts import PromptTemplate
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from pydantic import Field

# Try both import paths for GraphCypherQAChain (langchain version compatibility)
try:
    from langchain.chains import GraphCypherQAChain
except ImportError:
    from langchain_community.chains.graph_qa.cypher import GraphCypherQAChain

logger = logging.getLogger(__name__)


# Cypher generation prompt template (UPDATED: Fixed schema with backtick labels)
CYPHER_GENERATION_TEMPLATE = """Task: Generate Cypher statement to query a graph database.

Instructions:
1. Use ONLY the provided relationship types and properties in the schema below
2. Do NOT use any other relationship types or properties that are not provided
3. CRITICAL: Node labels with spaces MUST use backtick notation (e.g., `Bus Service`, `Traffic Incident`)
4. Labels without spaces can use colon notation (e.g., :Route, :Line, :Station)
5. Property access uses dot notation (e.g., node.name, node.date)

Schema:
{schema}

IMPORTANT LABEL NAMES (use backticks for labels with spaces):
- `Bus Service` (NOT BusService)
- `Traffic Incident` (NOT TrafficIncident)
- `Transportation Policy` (NOT TransportationPolicy)
- `Traffic Report` (NOT TrafficReport)
- `Public Transport Service` (NOT PublicTransportService)
- `Traffic Data` (NOT TrafficData)
- `Accessibility Features` (NOT AccessibilityFeatures)
- `Traffic Application` (NOT TrafficApplication)
- `Train Service` (NOT TrainService)
- `Commute Route` (NOT CommuteRoute)

Note: Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

Examples for Seoul traffic data (based on ACTUAL data patterns):

# 지하철 노선의 역 정보
MATCH (line:Line {{name: "1호선"}})
MATCH (station:Station)-[:IS_PART_OF]->(line)
RETURN station.name, station.description

# 버스 서비스와 노선 정보 (우회, 통제 등)
MATCH (service:`Bus Service`)-[:SERVES]->(route:Route)
WHERE service.name IS NOT NULL AND route.name IS NOT NULL
RETURN service.name, route.name, route.description

# 교통 사건과 관련 노선
MATCH (incident:`Traffic Incident`)-[:OCCURS_ON]->(route:Route)
MATCH (service:`Bus Service`)-[:SERVES]->(route)
WHERE route.name IS NOT NULL
RETURN incident.description, incident.type, route.name, service.name

# 교통 정책과 영향 받는 서비스
MATCH (policy:`Transportation Policy`)-[:AFFECTS]->(service:`Public Transport Service`)
WHERE policy.name IS NOT NULL
RETURN policy.name, policy.description, service.name

# 교통 보고서와 데이터
MATCH (report:`Traffic Report`)-[:PROVIDES]->(data:`Traffic Data`)
WHERE report.title IS NOT NULL
RETURN report.title, data.description, data.type

IMPORTANT DATA LIMITATION:
Most Traffic Incident nodes with route connections do NOT have dates populated.
If a date-specific query returns empty, try a broader search without date filters.

The question is:
{question}"""


class KGCypherRetriever(BaseRetriever):
    """
    Knowledge Graph Retriever with LLM-generated Cypher queries

    Uses LangChain's GraphCypherQAChain to:
    1. Generate Cypher query from natural language question
    2. Execute query against Neo4j graph
    3. Return results as context

    Benefits:
    - Dynamic graph traversal based on question
    - Leverages entity relationships
    - Adapts to different question types
    """

    # Pydantic fields
    k: int = Field(default=4, description="Number of results to return")
    neo4j_uri: Optional[str] = Field(default=None)
    neo4j_user: Optional[str] = Field(default=None)
    neo4j_password: Optional[str] = Field(default=None)
    llm_model: str = Field(default="gpt-4o-mini", description="LLM for Cypher generation")

    # Class attributes
    _graph: ClassVar[Any] = None
    _cypher_chain: ClassVar[Any] = None

    class Config:
        arbitrary_types_allowed = True

    def model_post_init(self, __context: Any) -> None:
        """Initialize after Pydantic validation"""

        # Load credentials
        self._load_credentials(self.neo4j_uri, self.neo4j_user, self.neo4j_password)

        # Initialize Neo4j Graph (shared)
        if KGCypherRetriever._graph is None:
            KGCypherRetriever._graph = Neo4jGraph(
                url=self.neo4j_uri,
                username=self.neo4j_user,
                password=self.neo4j_password,
                database=os.getenv('NEO4J_DATABASE', 'neo4j')
            )
            logger.info(f"✅ Connected to Neo4j Graph: {self.neo4j_uri}")

        # Initialize Cypher Chain (shared)
        if KGCypherRetriever._cypher_chain is None:
            cypher_prompt = PromptTemplate(
                input_variables=["schema", "question"],
                template=CYPHER_GENERATION_TEMPLATE
            )

            KGCypherRetriever._cypher_chain = GraphCypherQAChain.from_llm(
                ChatOpenAI(model=self.llm_model, temperature=0),
                graph=KGCypherRetriever._graph,
                verbose=True,
                cypher_prompt=cypher_prompt,
                return_intermediate_steps=True,  # To get Cypher query
                allow_dangerous_requests=True  # Required for LangChain security
            )
            logger.info(f"✅ Cypher Chain initialized with {self.llm_model}")

        logger.info(f"✅ KGCypherRetriever initialized (k={self.k})")

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
        Retrieve documents using LLM-generated Cypher

        Args:
            query: Natural language question

        Returns:
            List of Documents with graph context
        """
        try:
            logger.info(f"Generating Cypher for query: {query[:100]}...")

            # Run Cypher chain
            result = KGCypherRetriever._cypher_chain.invoke({"query": query})

            # Extract Cypher query and results
            cypher_query = result.get("intermediate_steps", [{}])[0].get("query", "N/A")
            answer = result.get("result", "")

            logger.info(f"Generated Cypher: {cypher_query[:200]}...")
            logger.info(f"Answer length: {len(answer)} chars")

            # Convert to Document format
            documents = [
                Document(
                    page_content=answer,
                    metadata={
                        "source": "knowledge_graph_cypher",
                        "retrieval_type": "kg_cypher_generation",
                        "cypher_query": cypher_query,
                        "question": query
                    }
                )
            ]

            logger.info(f"✅ Retrieved {len(documents)} documents via Cypher generation")
            return documents[:self.k]  # Limit to k

        except Exception as e:
            logger.error(f"❌ Cypher generation retrieval failed: {e}", exc_info=True)
            return []

    @classmethod
    def close_graph(cls):
        """Close Neo4j graph connection"""
        if cls._graph:
            # Neo4jGraph doesn't have explicit close, but driver does
            cls._graph = None
            cls._cypher_chain = None
            logger.info("Neo4j graph closed")


def create_kg_cypher_retriever(
    k: int = 4,
    llm_model: str = "gpt-4o-mini",
    **kwargs
) -> KGCypherRetriever:
    """
    Factory function to create KG Cypher retriever

    Args:
        k: Number of documents to retrieve
        llm_model: LLM for Cypher generation
        **kwargs: Additional arguments

    Returns:
        Configured KGCypherRetriever
    """
    return KGCypherRetriever(k=k, llm_model=llm_model, **kwargs)


if __name__ == "__main__":
    # Test retriever
    logging.basicConfig(level=logging.INFO)

    print("Testing KG Cypher Generation Retriever...")
    retriever = create_kg_cypher_retriever(k=3, llm_model="gpt-4o-mini")

    test_queries = [
        "서울 지하철 1호선의 주요 역은?",
        "2024년 12월 버스 우회 정보",
        "여의도역 근처 교통 정책"
    ]

    for query in test_queries:
        print(f"\n쿼리: {query}")
        docs = retriever.invoke(query)
        print(f"검색 결과: {len(docs)}개 문서")

        for i, doc in enumerate(docs, 1):
            print(f"\n[Document {i}]")
            print(f"Cypher: {doc.metadata.get('cypher_query', 'N/A')[:100]}...")
            print(f"Content: {doc.page_content[:200]}...")
