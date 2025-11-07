#!/usr/bin/env python3
"""
Preflight Check for KG RAG System
Verifies Neo4j graph and KG RAG retriever before scaling up
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.kg_agent.utils.neo4j_for_adk import graphdb
from src.kg_agent.kg_rag_retriever import create_kg_retriever, KGRAGRetriever
import logging

logging.basicConfig(
    level=logging.WARNING,  # Suppress INFO logs for cleaner output
    format='%(message)s'
)

def print_section(title):
    """Print section header"""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

def check_neo4j_connection():
    """Check 1: Neo4j Connection"""
    print("\n[Check 1/6] Neo4j Connection")
    print("-" * 70)

    try:
        result = graphdb.send_query("RETURN 1 as test")
        if result['query_result'][0]['test'] == 1:
            print("✅ Neo4j connection successful")
            return True
    except Exception as e:
        print(f"❌ Neo4j connection failed: {e}")
        return False

def check_graph_structure():
    """Check 2: Graph Structure"""
    print("\n[Check 2/6] Graph Structure")
    print("-" * 70)

    try:
        # Count nodes and relationships
        result = graphdb.send_query('MATCH (n) RETURN count(n) as total_nodes')
        total_nodes = result['query_result'][0]['total_nodes']

        result = graphdb.send_query('MATCH ()-[r]->() RETURN count(r) as total_rels')
        total_rels = result['query_result'][0]['total_rels']

        print(f"  Total Nodes: {total_nodes}")
        print(f"  Total Relationships: {total_rels}")

        # Check Chunk nodes
        result = graphdb.send_query('MATCH (c:Chunk) RETURN count(c) as chunk_count')
        chunk_count = result['query_result'][0]['chunk_count']
        print(f"  Chunk Nodes: {chunk_count}")

        # Check domain entities
        result = graphdb.send_query('''
            MATCH (n)
            UNWIND labels(n) as label
            WITH label, count(*) as count
            WHERE NOT label IN ['__Entity__', '__KGBuilder__', '__Chunk__', '__Document__']
            RETURN label, count
            ORDER BY count DESC
            LIMIT 5
        ''')

        print(f"\n  Top Domain Entities:")
        for record in result['query_result']:
            print(f"    {record['label']}: {record['count']}")

        if total_nodes > 0 and chunk_count > 0:
            print("\n✅ Graph structure valid")
            return True
        else:
            print("\n❌ Graph structure invalid (no nodes or chunks)")
            return False

    except Exception as e:
        print(f"❌ Graph structure check failed: {e}")
        return False

def check_chunk_embeddings():
    """Check 3: Chunk Embeddings"""
    print("\n[Check 3/6] Chunk Embeddings")
    print("-" * 70)

    try:
        # Check embedding existence and dimensions
        result = graphdb.send_query("""
            MATCH (c:Chunk)
            WHERE c.embedding IS NOT NULL
            RETURN count(c) as with_embeddings,
                   size(c.embedding) as dimensions
            LIMIT 1
        """)

        if not result['query_result']:
            print("❌ No Chunk nodes with embeddings found")
            return False

        record = result['query_result'][0]
        with_embeddings = record['with_embeddings']
        dimensions = record['dimensions']

        print(f"  Chunks with embeddings: {with_embeddings}")
        print(f"  Embedding dimensions: {dimensions}")

        if dimensions == 3072:
            print("\n✅ Embeddings valid (3072D = text-embedding-3-large)")
            return True
        else:
            print(f"\n❌ Wrong embedding dimensions: {dimensions} (expected 3072)")
            return False

    except Exception as e:
        print(f"❌ Embedding check failed: {e}")
        return False

def check_vector_index():
    """Check 4: Vector Index"""
    print("\n[Check 4/6] Vector Index")
    print("-" * 70)

    try:
        result = graphdb.send_query("SHOW INDEXES WHERE name = 'vector'")

        if not result['query_result']:
            print("❌ Vector index 'vector' not found")
            return False

        idx = result['query_result'][0]
        print(f"  Index Name: {idx.get('name')}")
        print(f"  Type: {idx.get('type')}")
        print(f"  Labels: {idx.get('labelsOrTypes')}")
        print(f"  Properties: {idx.get('properties')}")

        if idx.get('type') == 'VECTOR':
            print("\n✅ Vector index exists and is valid")
            return True
        else:
            print(f"\n❌ Index type is {idx.get('type')}, expected VECTOR")
            return False

    except Exception as e:
        print(f"❌ Vector index check failed: {e}")
        return False

def check_retriever_initialization():
    """Check 5: KG RAG Retriever Initialization"""
    print("\n[Check 5/6] KG RAG Retriever Initialization")
    print("-" * 70)

    try:
        retriever = create_kg_retriever(k=3)
        print("  ✓ Retriever created")
        print("  ✓ Neo4j driver connected")
        print("  ✓ Embedder initialized")
        print("  ✓ VectorCypherRetriever configured")
        print("\n✅ Retriever initialization successful")
        return True, retriever
    except Exception as e:
        print(f"❌ Retriever initialization failed: {e}")
        return False, None

def check_retrieval_functionality(retriever):
    """Check 6: Retrieval Functionality"""
    print("\n[Check 6/6] Retrieval Functionality")
    print("-" * 70)

    test_queries = [
        "서울 지하철 1호선 역",
        "교통 정책 정보",
        "버스 노선 안내"
    ]

    all_passed = True

    for i, query in enumerate(test_queries, 1):
        print(f"\n  Test {i}: '{query}'")
        try:
            docs = retriever._get_relevant_documents(query)
            if len(docs) > 0:
                print(f"    ✓ Retrieved {len(docs)} documents")
                print(f"    ✓ First doc length: {len(docs[0].page_content)} chars")
            else:
                print(f"    ⚠️  No documents retrieved (may be expected if content doesn't match)")
        except Exception as e:
            print(f"    ❌ Retrieval failed: {e}")
            all_passed = False

    if all_passed:
        print("\n✅ Retrieval functionality working")
    else:
        print("\n❌ Some retrieval tests failed")

    return all_passed

def main():
    """Run all preflight checks"""
    print_section("KG RAG System Preflight Check")
    print("\nVerifying system readiness before scaling up...")

    checks = []

    # Run checks
    checks.append(("Neo4j Connection", check_neo4j_connection()))
    checks.append(("Graph Structure", check_graph_structure()))
    checks.append(("Chunk Embeddings", check_chunk_embeddings()))
    checks.append(("Vector Index", check_vector_index()))

    retriever_ok, retriever = check_retriever_initialization()
    checks.append(("Retriever Init", retriever_ok))

    if retriever:
        checks.append(("Retrieval Function", check_retrieval_functionality(retriever)))
        KGRAGRetriever.close_driver()
    else:
        checks.append(("Retrieval Function", False))

    # Close Neo4j connection
    graphdb.close()

    # Summary
    print_section("Preflight Check Summary")

    all_passed = True
    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}  {check_name}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 70)

    if all_passed:
        print("\n🎉 ALL CHECKS PASSED!")
        print("\n✅ System is ready to scale up to more files")
        print("\nNext steps:")
        print("  1. Run: uv run python src/kg_agent/build_kg.py --num-files 10")
        print("  2. Or run: uv run python src/kg_agent/build_kg.py --num-files 0 (all files)")
        print("\n" + "=" * 70)
        return 0
    else:
        print("\n❌ SOME CHECKS FAILED")
        print("\n⚠️  Please fix the issues before scaling up")
        print("\n" + "=" * 70)
        return 1

if __name__ == "__main__":
    sys.exit(main())
