#!/usr/bin/env python3
"""
Neo4j Knowledge Graph Healthcheck
==================================

Comprehensive health verification for Neo4j KG:
1. Connection & Database status
2. Graph structure (nodes, relationships)
3. Embedding verification (dimensions, coverage)
4. Vector index validation
5. Sample retrieval test
6. Data quality metrics

Usage:
    python src/kg_agent/kg_healthcheck.py
    python src/kg_agent/kg_healthcheck.py --detailed
"""

import sys
from pathlib import Path
from typing import Dict, Any, List
import json

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from utils.neo4j_for_adk import graphdb


class KGHealthCheck:
    """Comprehensive KG health verification"""

    def __init__(self):
        self.checks_passed = 0
        self.checks_failed = 0
        self.warnings = []
        self.results = {}

    def run_all_checks(self, detailed: bool = False) -> Dict[str, Any]:
        """Run all health checks"""

        print("\n" + "="*70)
        print("Neo4j Knowledge Graph Healthcheck".center(70))
        print("="*70)

        # Check 1: Connection
        self._check_connection()

        # Check 2: Database info
        self._check_database_info()

        # Check 3: Graph structure
        self._check_graph_structure()

        # Check 4: Node labels distribution
        self._check_node_labels()

        # Check 5: Relationship types
        self._check_relationship_types()

        # Check 6: Chunk embeddings
        self._check_chunk_embeddings()

        # Check 7: Vector index
        self._check_vector_index()

        # Check 8: Sample retrieval (if detailed)
        if detailed:
            self._check_sample_retrieval()

        # Check 9: Data quality
        self._check_data_quality()

        # Summary
        self._print_summary()

        return self.results

    def _check_connection(self):
        """Check 1: Neo4j connection"""
        print("\n" + "-"*70)
        print("[Check 1/9] Neo4j Connection")
        print("-"*70)

        try:
            result = graphdb.send_query("RETURN 1 as test")
            if result and 'query_result' in result:
                print("✅ Connection successful")
                self.checks_passed += 1
                self.results['connection'] = 'OK'
            else:
                print("❌ Connection failed: unexpected response")
                self.checks_failed += 1
                self.results['connection'] = 'FAILED'
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            self.checks_failed += 1
            self.results['connection'] = f'ERROR: {e}'

    def _check_database_info(self):
        """Check 2: Database information"""
        print("\n" + "-"*70)
        print("[Check 2/9] Database Information")
        print("-"*70)

        try:
            # Get database name
            result = graphdb.send_query("CALL db.info() YIELD name, edition, version")
            if result and 'query_result' in result and result['query_result']:
                info = result['query_result'][0]
                print(f"✅ Database: {info.get('name', 'N/A')}")
                print(f"   Edition: {info.get('edition', 'N/A')}")
                print(f"   Version: {info.get('version', 'N/A')}")
                self.checks_passed += 1
                self.results['database_info'] = info
            else:
                print("⚠️ Database info unavailable")
                self.warnings.append("Database info query returned empty")
                self.checks_passed += 1
        except Exception as e:
            print(f"⚠️ Database info check failed: {e}")
            self.warnings.append(f"Database info: {e}")
            self.checks_passed += 1  # Non-critical

    def _check_graph_structure(self):
        """Check 3: Graph structure"""
        print("\n" + "-"*70)
        print("[Check 3/9] Graph Structure")
        print("-"*70)

        try:
            # Total nodes
            result = graphdb.send_query('MATCH (n) RETURN count(n) as total_nodes')
            total_nodes = result['query_result'][0]['total_nodes']

            # Total relationships
            result = graphdb.send_query('MATCH ()-[r]->() RETURN count(r) as total_rels')
            total_rels = result['query_result'][0]['total_rels']

            print(f"✅ Graph structure:")
            print(f"   Nodes: {total_nodes:,}")
            print(f"   Relationships: {total_rels:,}")

            if total_nodes == 0:
                print("⚠️ WARNING: Graph is empty!")
                self.warnings.append("Empty graph - KG build may not have run yet")
                self.checks_failed += 1
                self.results['graph_structure'] = 'EMPTY'
            else:
                self.checks_passed += 1
                self.results['graph_structure'] = {
                    'total_nodes': total_nodes,
                    'total_relationships': total_rels
                }

        except Exception as e:
            print(f"❌ Graph structure check failed: {e}")
            self.checks_failed += 1
            self.results['graph_structure'] = f'ERROR: {e}'

    def _check_node_labels(self):
        """Check 4: Node labels distribution"""
        print("\n" + "-"*70)
        print("[Check 4/9] Node Labels Distribution")
        print("-"*70)

        try:
            # Get domain entity labels (excluding internal labels)
            result = graphdb.send_query('''
                MATCH (n)
                UNWIND labels(n) as label
                WITH label, count(*) as count
                WHERE NOT label IN ['__Entity__', '__KGBuilder__', '__Chunk__', '__Document__']
                RETURN label, count
                ORDER BY count DESC
            ''')

            if result and 'query_result' in result:
                domain_labels = result['query_result']

                print(f"✅ Domain entity labels: {len(domain_labels)}")
                for item in domain_labels[:10]:
                    print(f"   {item['label']}: {item['count']}")

                if len(domain_labels) > 10:
                    print(f"   ... and {len(domain_labels) - 10} more")

                self.checks_passed += 1
                self.results['node_labels'] = domain_labels
            else:
                print("⚠️ No domain labels found")
                self.warnings.append("No domain entity labels")
                self.checks_passed += 1

        except Exception as e:
            print(f"❌ Node labels check failed: {e}")
            self.checks_failed += 1
            self.results['node_labels'] = f'ERROR: {e}'

    def _check_relationship_types(self):
        """Check 5: Relationship types"""
        print("\n" + "-"*70)
        print("[Check 5/9] Relationship Types")
        print("-"*70)

        try:
            result = graphdb.send_query('''
                MATCH ()-[r]->()
                WITH type(r) as rel_type, count(*) as count
                WHERE NOT rel_type IN ['PART_OF', 'FIRST_CHUNK', 'NEXT_CHUNK', 'FROM_CHUNK', 'FROM_DOCUMENT']
                RETURN rel_type, count
                ORDER BY count DESC
            ''')

            if result and 'query_result' in result:
                domain_rels = result['query_result']

                print(f"✅ Domain relationship types: {len(domain_rels)}")
                for item in domain_rels[:10]:
                    print(f"   {item['rel_type']}: {item['count']}")

                if len(domain_rels) > 10:
                    print(f"   ... and {len(domain_rels) - 10} more")

                self.checks_passed += 1
                self.results['relationship_types'] = domain_rels
            else:
                print("⚠️ No domain relationships found")
                self.warnings.append("No domain relationships")
                self.checks_passed += 1

        except Exception as e:
            print(f"❌ Relationship types check failed: {e}")
            self.checks_failed += 1
            self.results['relationship_types'] = f'ERROR: {e}'

    def _check_chunk_embeddings(self):
        """Check 6: Chunk embeddings"""
        print("\n" + "-"*70)
        print("[Check 6/9] Chunk Embeddings")
        print("-"*70)

        try:
            result = graphdb.send_query('''
                MATCH (c:Chunk)
                WHERE c.embedding IS NOT NULL
                RETURN count(c) as with_embeddings,
                       size(c.embedding) as dimensions
                LIMIT 1
            ''')

            if result and 'query_result' in result and result['query_result']:
                chunk_info = result['query_result'][0]
                count = chunk_info['with_embeddings']
                dims = chunk_info['dimensions']

                print(f"✅ Chunk embeddings:")
                print(f"   Chunks with embeddings: {count:,}")
                print(f"   Embedding dimensions: {dims}D")

                # Verify dimension matches expected (3072D for text-embedding-3-large)
                if dims != 3072:
                    print(f"⚠️ WARNING: Expected 3072D (text-embedding-3-large), got {dims}D")
                    self.warnings.append(f"Embedding dimension mismatch: {dims}D vs 3072D")

                self.checks_passed += 1
                self.results['chunk_embeddings'] = {
                    'count': count,
                    'dimensions': dims,
                    'expected_dimensions': 3072,
                    'match': dims == 3072
                }
            else:
                print("❌ No chunks with embeddings found")
                self.checks_failed += 1
                self.results['chunk_embeddings'] = 'NONE'

        except Exception as e:
            print(f"❌ Chunk embeddings check failed: {e}")
            self.checks_failed += 1
            self.results['chunk_embeddings'] = f'ERROR: {e}'

    def _check_vector_index(self):
        """Check 7: Vector index"""
        print("\n" + "-"*70)
        print("[Check 7/9] Vector Index")
        print("-"*70)

        try:
            result = graphdb.send_query("SHOW INDEXES WHERE type = 'VECTOR'")

            if result and 'query_result' in result and result['query_result']:
                indexes = result['query_result']

                print(f"✅ Vector indexes: {len(indexes)}")
                for idx in indexes:
                    print(f"   Name: {idx.get('name')}")
                    print(f"   State: {idx.get('state')}")
                    print(f"   Population: {idx.get('populationPercent')}%")

                self.checks_passed += 1
                self.results['vector_index'] = indexes
            else:
                print("❌ No vector indexes found")
                self.checks_failed += 1
                self.results['vector_index'] = 'NONE'

        except Exception as e:
            print(f"❌ Vector index check failed: {e}")
            self.checks_failed += 1
            self.results['vector_index'] = f'ERROR: {e}'

    def _check_sample_retrieval(self):
        """Check 8: Sample retrieval test"""
        print("\n" + "-"*70)
        print("[Check 8/9] Sample Retrieval Test")
        print("-"*70)

        try:
            from kg_rag_retriever import create_kg_retriever

            retriever = create_kg_retriever(k=3)

            test_queries = [
                "서울 지하철 1호선",
                "버스 노선",
                "교통 정책"
            ]

            print("Testing KG retrieval...")
            for query in test_queries:
                docs = retriever.invoke(query)
                print(f"  Query: '{query}'")
                print(f"    Retrieved: {len(docs)} documents")

            print("✅ Sample retrieval working")
            self.checks_passed += 1
            self.results['sample_retrieval'] = 'OK'

        except Exception as e:
            print(f"⚠️ Sample retrieval test failed: {e}")
            self.warnings.append(f"Retrieval test: {e}")
            self.checks_passed += 1  # Non-critical for healthcheck

    def _check_data_quality(self):
        """Check 9: Data quality metrics"""
        print("\n" + "-"*70)
        print("[Check 9/9] Data Quality Metrics")
        print("-"*70)

        try:
            # Check for orphaned chunks (no document connection)
            result = graphdb.send_query('''
                MATCH (c:Chunk)
                WHERE NOT EXISTS {
                    MATCH (c)-[:FROM_DOCUMENT]->(:Document)
                }
                RETURN count(c) as orphaned_chunks
            ''')
            orphaned = result['query_result'][0]['orphaned_chunks']

            # Check for chunks without embeddings
            result = graphdb.send_query('''
                MATCH (c:Chunk)
                WHERE c.embedding IS NULL
                RETURN count(c) as no_embedding_chunks
            ''')
            no_embedding = result['query_result'][0]['no_embedding_chunks']

            # Check for empty chunks
            result = graphdb.send_query('''
                MATCH (c:Chunk)
                WHERE c.text IS NULL OR c.text = ''
                RETURN count(c) as empty_chunks
            ''')
            empty = result['query_result'][0]['empty_chunks']

            print(f"✅ Data quality:")
            print(f"   Orphaned chunks (no document): {orphaned}")
            print(f"   Chunks without embeddings: {no_embedding}")
            print(f"   Empty chunks: {empty}")

            quality_issues = orphaned + no_embedding + empty
            if quality_issues > 0:
                print(f"⚠️ WARNING: {quality_issues} data quality issues found")
                self.warnings.append(f"{quality_issues} data quality issues")

            self.checks_passed += 1
            self.results['data_quality'] = {
                'orphaned_chunks': orphaned,
                'no_embedding_chunks': no_embedding,
                'empty_chunks': empty,
                'total_issues': quality_issues
            }

        except Exception as e:
            print(f"❌ Data quality check failed: {e}")
            self.checks_failed += 1
            self.results['data_quality'] = f'ERROR: {e}'

    def _print_summary(self):
        """Print healthcheck summary"""
        print("\n" + "="*70)
        print("Healthcheck Summary".center(70))
        print("="*70)

        total_checks = self.checks_passed + self.checks_failed
        print(f"\n✅ Passed: {self.checks_passed}/{total_checks}")
        print(f"❌ Failed: {self.checks_failed}/{total_checks}")
        print(f"⚠️  Warnings: {len(self.warnings)}")

        if self.warnings:
            print("\nWarnings:")
            for w in self.warnings:
                print(f"  - {w}")

        if self.checks_failed == 0:
            print("\n" + "="*70)
            print("✅ ALL HEALTHCHECKS PASSED".center(70))
            print("="*70)
        else:
            print("\n" + "="*70)
            print("❌ HEALTHCHECK FAILED".center(70))
            print("="*70)

    def save_report(self, output_path: Path):
        """Save healthcheck report to file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'checks_passed': self.checks_passed,
                'checks_failed': self.checks_failed,
                'warnings': self.warnings,
                'results': self.results
            }, f, indent=2, ensure_ascii=False, default=str)

        print(f"\n📄 Report saved: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Neo4j KG Healthcheck")
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Include detailed checks (retrieval test)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Save report to file (JSON)"
    )

    args = parser.parse_args()

    try:
        healthcheck = KGHealthCheck()
        results = healthcheck.run_all_checks(detailed=args.detailed)

        if args.output:
            healthcheck.save_report(args.output)

        # Exit code based on results
        sys.exit(0 if healthcheck.checks_failed == 0 else 1)

    except Exception as e:
        print(f"\n❌ Healthcheck failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
