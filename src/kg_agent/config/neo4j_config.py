"""
Neo4j Aura Connection Configuration
Based on kg_basics/L2-query_with_cypher.ipynb
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.graphs import Neo4jGraph
import warnings

warnings.filterwarnings("ignore")


class Neo4jConnection:
    """Neo4j Aura connection manager"""

    def __init__(self, env_path: str = None):
        """
        Initialize Neo4j connection

        Args:
            env_path: Path to .env file. If None, uses config/.env
        """
        if env_path is None:
            env_path = Path(__file__).parent / '.env'

        # Load environment variables
        load_dotenv(env_path, override=True)

        self.uri = os.getenv('NEO4J_URI')
        self.username = os.getenv('NEO4J_USERNAME')
        self.password = os.getenv('NEO4J_PASSWORD')
        self.database = os.getenv('NEO4J_DATABASE', 'neo4j')

        self._kg = None

    @property
    def kg(self) -> Neo4jGraph:
        """Get or create Neo4j graph connection"""
        if self._kg is None:
            self._kg = Neo4jGraph(
                url=self.uri,
                username=self.username,
                password=self.password,
                database=self.database
            )
        return self._kg

    def query(self, cypher: str, params: dict = None):
        """
        Execute Cypher query

        Args:
            cypher: Cypher query string
            params: Query parameters

        Returns:
            Query results
        """
        return self.kg.query(cypher, params=params or {})

    def close(self):
        """Close connection"""
        if self._kg is not None:
            # Neo4jGraph doesn't have explicit close, but we can reset it
            self._kg = None


def get_neo4j_connection(env_path: str = None) -> Neo4jGraph:
    """
    Get Neo4j connection (convenience function)

    Args:
        env_path: Path to .env file

    Returns:
        Neo4jGraph instance
    """
    conn = Neo4jConnection(env_path)
    return conn.kg
