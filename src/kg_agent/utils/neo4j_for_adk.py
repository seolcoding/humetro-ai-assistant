"""
Neo4j wrapper for Google ADK
Based on agentic_kg_build notebooks
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class GraphDB:
    """Neo4j database wrapper for ADK agents"""

    def __init__(self, env_path: Optional[str] = None):
        """
        Initialize Neo4j connection

        Args:
            env_path: Path to .env file. If None, uses config/.env
        """
        if env_path is None:
            # Look for .env in config directory
            config_dir = Path(__file__).parent.parent / "config"
            env_path = config_dir / ".env"

        # Load environment variables
        load_dotenv(env_path, override=True)

        self.uri = os.getenv('NEO4J_URI')
        self.username = os.getenv('NEO4J_USERNAME')
        self.password = os.getenv('NEO4J_PASSWORD')
        self.database = os.getenv('NEO4J_DATABASE', 'neo4j')

        self.driver = None
        self._connect()

    def _connect(self):
        """Establish connection to Neo4j"""
        try:
            self.driver = GraphDatabase.driver(
                self.uri,
                auth=(self.username, self.password)
            )
            # Verify connection
            self.driver.verify_connectivity()
            logger.info("✅ Neo4j connection established")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Neo4j: {e}")
            raise

    def send_query(self, cypher: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute Cypher query and return formatted results

        This method formats results for ADK agents:
        - Success: {"status": "success", "query_result": [...]}
        - Error: {"status": "error", "error_message": "..."}

        Args:
            cypher: Cypher query string
            params: Query parameters (optional)

        Returns:
            Dict with status and results/error
        """
        if params is None:
            params = {}

        try:
            with self.driver.session(database=self.database) as session:
                result = session.run(cypher, params)
                records = [record.data() for record in result]

                return {
                    "status": "success",
                    "query_result": records
                }

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Query failed: {error_msg}")
            logger.error(f"Query: {cypher}")
            logger.error(f"Params: {params}")

            return {
                "status": "error",
                "error_message": error_msg
            }

    def close(self):
        """Close database connection"""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")


# Global instance for convenience (like in the notebooks)
graphdb = GraphDB()


if __name__ == "__main__":
    # Test the connection
    print("Testing Neo4j connection...")

    result = graphdb.send_query("RETURN 'Neo4j is Ready!' as message")
    print(result)

    if result["status"] == "success":
        print(f"✅ Message: {result['query_result'][0]['message']}")
    else:
        print(f"❌ Error: {result['error_message']}")

    graphdb.close()
