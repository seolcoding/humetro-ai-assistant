#!/usr/bin/env python3
"""
Neo4j Aura Connection Test Script
Based on kg_basics/L2-query_with_cypher.ipynb

Tests:
1. Connection
2. Basic queries (READ)
3. CREATE operations
4. UPDATE operations
5. Relationship operations
6. DELETE operations
7. Graph initialization
8. Constraints creation
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


def test_connection():
    """1. Test Neo4j Aura connection"""
    logger.info("=" * 60)
    logger.info("TEST 1: Neo4j Aura 연결 테스트")
    logger.info("=" * 60)

    try:
        conn = Neo4jConnection()
        kg = conn.kg
        logger.info("✅ Neo4j Aura 연결 성공!")
        logger.info(f"   URI: {conn.uri}")
        logger.info(f"   Database: {conn.database}")
        return conn
    except Exception as e:
        logger.error(f"❌ 연결 실패: {e}")
        return None


def test_basic_queries(conn):
    """2. Test basic READ queries"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 2: 기본 쿼리 테스트")
    logger.info("=" * 60)

    # Count all nodes
    cypher = "MATCH (n) RETURN count(n) AS nodeCount"
    result = conn.query(cypher)
    node_count = result[0]['nodeCount']
    logger.info(f"✅ 현재 노드 개수: {node_count}")

    return node_count


def test_create_operations(conn):
    """3. Test CREATE operations"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 3: CREATE 작업 테스트")
    logger.info("=" * 60)

    # Create test node
    cypher = """
    CREATE (test:TestNode {
        name: 'Seoul Metro KG Test',
        created_at: datetime(),
        type: 'connection_test',
        version: '1.0'
    })
    RETURN test
    """

    result = conn.query(cypher)
    logger.info(f"✅ CREATE 성공!")
    logger.info(f"   노드 속성: {result[0]['test']}")

    return result


def test_read_operations(conn):
    """4. Test READ operations"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 4: READ 작업 테스트")
    logger.info("=" * 60)

    # Read test node
    cypher = """
    MATCH (test:TestNode {type: 'connection_test'})
    RETURN test.name AS name,
           test.created_at AS created,
           test.version AS version
    """

    result = conn.query(cypher)
    if result:
        logger.info(f"✅ READ 성공!")
        logger.info(f"   이름: {result[0]['name']}")
        logger.info(f"   생성시간: {result[0]['created']}")
        logger.info(f"   버전: {result[0]['version']}")
    else:
        logger.warning("⚠️ 테스트 노드를 찾을 수 없음")

    return result


def test_update_operations(conn):
    """5. Test UPDATE operations"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 5: UPDATE 작업 테스트")
    logger.info("=" * 60)

    # Update test node
    cypher = """
    MATCH (test:TestNode {type: 'connection_test'})
    SET test.updated_at = datetime(),
        test.status = 'updated',
        test.test_count = 5
    RETURN test
    """

    result = conn.query(cypher)
    if result:
        logger.info(f"✅ UPDATE 성공!")
        logger.info(f"   업데이트된 속성: {result[0]['test']}")
    else:
        logger.warning("⚠️ 업데이트할 노드 없음")

    return result


def test_relationship_operations(conn):
    """6. Test relationship CREATE operations"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 6: 관계 생성 테스트")
    logger.info("=" * 60)

    # Create second node and relationship
    cypher = """
    MATCH (test1:TestNode {type: 'connection_test'})
    CREATE (station:TestNode {
        name: '강남역 테스트',
        type: 'test_station',
        line: '2호선'
    })
    CREATE (test1)-[r:CONNECTED_TO {
        distance_km: 2.5,
        travel_time_min: 5
    }]->(station)
    RETURN test1, r, station
    """

    result = conn.query(cypher)
    if result:
        logger.info(f"✅ RELATIONSHIP 생성 성공!")
        logger.info(f"   관계: {result[0]['r']}")
    else:
        logger.warning("⚠️ 관계 생성 실패")

    # Verify relationship
    verify_cypher = """
    MATCH (t1:TestNode)-[r:CONNECTED_TO]->(t2:TestNode)
    WHERE t1.type = 'connection_test'
    RETURN t1.name AS from_node,
           t2.name AS to_node,
           r.distance_km AS distance
    """

    verify_result = conn.query(verify_cypher)
    if verify_result:
        logger.info(f"   경로: {verify_result[0]['from_node']} → {verify_result[0]['to_node']}")
        logger.info(f"   거리: {verify_result[0]['distance']} km")

    return result


def test_delete_operations(conn):
    """7. Test DELETE operations"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 7: DELETE 작업 테스트")
    logger.info("=" * 60)

    # Delete all test nodes and relationships
    cypher = """
    MATCH (test:TestNode)
    WHERE test.type IN ['connection_test', 'test_station']
    DETACH DELETE test
    """

    conn.query(cypher)

    # Verify deletion
    verify_cypher = """
    MATCH (test:TestNode)
    WHERE test.type IN ['connection_test', 'test_station']
    RETURN count(test) AS count
    """

    result = conn.query(verify_cypher)

    if result[0]['count'] == 0:
        logger.info("✅ DELETE 성공: 모든 테스트 노드 삭제됨")
    else:
        logger.warning(f"⚠️ {result[0]['count']}개의 테스트 노드가 남아있음")

    return result


def test_constraints(conn):
    """8. Test constraint creation"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 8: 제약조건 생성 테스트")
    logger.info("=" * 60)

    # Create unique constraint for Chunk nodes (for future KG)
    constraint_cypher = """
    CREATE CONSTRAINT unique_chunk IF NOT EXISTS
    FOR (c:Chunk) REQUIRE c.chunkId IS UNIQUE
    """

    try:
        conn.query(constraint_cypher)
        logger.info("✅ 제약조건 'unique_chunk' 생성 성공")
    except Exception as e:
        logger.info(f"ℹ️ 제약조건이 이미 존재하거나 오류: {e}")

    # Show all indexes
    indexes = conn.query("SHOW INDEXES")
    logger.info(f"✅ 현재 인덱스 개수: {len(indexes)}")

    for idx in indexes:
        logger.info(f"   - {idx.get('name', 'N/A')}: {idx.get('type', 'N/A')}")

    return indexes


def initialize_graph(conn, skip_prompt=False):
    """9. Graph initialization (optional - deletes ALL data)"""
    logger.info("\n" + "=" * 60)
    logger.info("TEST 9: 그래프 초기화 (선택적)")
    logger.info("=" * 60)

    if skip_prompt:
        logger.info("ℹ️  초기화 건너뜀 (skip_prompt=True)")
        return False

    try:
        response = input("⚠️  경고: 모든 데이터를 삭제하시겠습니까? (yes/no): ")
    except (EOFError, KeyboardInterrupt):
        logger.info("ℹ️  초기화 건너뜀 (입력 불가)")
        return False

    if response.lower() == 'yes':
        # Delete all nodes and relationships
        conn.query("MATCH (n) DETACH DELETE n")
        logger.info("✅ 그래프 초기화 완료")

        # Verify
        result = conn.query("MATCH (n) RETURN count(n) AS count")
        logger.info(f"   남은 노드 개수: {result[0]['count']}")

        return True
    else:
        logger.info("ℹ️  초기화 취소됨")
        return False


def print_summary(initial_count, final_count):
    """Print test summary"""
    logger.info("\n" + "=" * 60)
    logger.info("테스트 요약")
    logger.info("=" * 60)
    logger.info(f"초기 노드 개수: {initial_count}")
    logger.info(f"최종 노드 개수: {final_count}")
    logger.info("=" * 60)
    logger.info("✅ 모든 테스트 완료!")
    logger.info("=" * 60)


def main():
    """Main test execution"""

    logger.info("\n")
    logger.info("🚀 " * 30)
    logger.info("Neo4j Aura 연결 테스트 시작")
    logger.info("🚀 " * 30)

    # 1. Connection test
    conn = test_connection()
    if not conn:
        logger.error("연결 실패로 테스트 중단")
        return

    # 2. Basic queries
    initial_count = test_basic_queries(conn)

    # 3-6. CRUD tests
    test_create_operations(conn)
    test_read_operations(conn)
    test_update_operations(conn)
    test_relationship_operations(conn)

    # 7. Delete test data
    test_delete_operations(conn)

    # 8. Constraints
    test_constraints(conn)

    # 9. Optional initialization (skip in non-interactive mode)
    initialize_graph(conn, skip_prompt=True)

    # Final count
    final_count = test_basic_queries(conn)

    # Summary
    print_summary(initial_count, final_count)


if __name__ == "__main__":
    main()
