#!/usr/bin/env python3
"""
Neo4j Graph Initialization Script
Deletes all nodes and relationships, then runs basic tests
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.kg_agent.config.neo4j_config import Neo4jConnection
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def main():
    """Initialize graph and run verification tests"""

    logger.info("=" * 60)
    logger.info("Neo4j 그래프 초기화 및 테스트")
    logger.info("=" * 60)

    # Connect
    conn = Neo4jConnection()

    # Check current state
    result = conn.query("MATCH (n) RETURN count(n) AS count")
    initial_count = result[0]['count']
    logger.info(f"초기 노드 개수: {initial_count}")

    # Initialize (delete all)
    logger.info("\n⚠️  모든 데이터를 삭제합니다...")
    conn.query("MATCH (n) DETACH DELETE n")
    logger.info("✅ 그래프 초기화 완료")

    # Verify empty
    result = conn.query("MATCH (n) RETURN count(n) AS count")
    logger.info(f"✅ 남은 노드 개수: {result[0]['count']}")

    # Test: Create test node
    logger.info("\n--- 테스트: 노드 생성 ---")
    conn.query("""
        CREATE (test:TestNode {
            name: 'Post-Initialization Test',
            created_at: datetime()
        })
        RETURN test
    """)
    logger.info("✅ 테스트 노드 생성 성공")

    # Verify
    result = conn.query("MATCH (n:TestNode) RETURN count(n) AS count")
    logger.info(f"✅ 현재 노드 개수: {result[0]['count']}")

    # Clean up test node
    logger.info("\n--- 테스트 노드 정리 ---")
    conn.query("MATCH (n:TestNode) DELETE n")
    logger.info("✅ 테스트 노드 삭제 완료")

    # Final verification
    result = conn.query("MATCH (n) RETURN count(n) AS count")
    final_count = result[0]['count']

    logger.info("\n" + "=" * 60)
    logger.info("초기화 및 테스트 완료")
    logger.info("=" * 60)
    logger.info(f"초기 노드: {initial_count} → 최종 노드: {final_count}")
    logger.info("✅ 그래프가 깨끗하게 초기화되었습니다!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
