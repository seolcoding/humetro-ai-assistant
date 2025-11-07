#!/usr/bin/env python3
"""
Neo4j Vector Store Dump Script
================================

Exports current Neo4j Knowledge Graph state including:
- Node statistics and labels
- Relationship statistics
- Chunk embeddings metadata
- Vector index configuration
- Sample data for verification
- Full export options (JSON/CSV)

Usage:
    python src/kg_agent/dump_vectorstore.py
    python src/kg_agent/dump_vectorstore.py --full  # Export all data to JSON
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from utils.neo4j_for_adk import graphdb


def get_graph_statistics() -> Dict[str, Any]:
    """Get comprehensive graph statistics"""

    print("\n" + "="*70)
    print("Neo4j Vector Store Statistics".center(70))
    print("="*70)

    stats = {}

    # Total counts
    result = graphdb.send_query('MATCH (n) RETURN count(n) as total_nodes')
    stats['total_nodes'] = result['query_result'][0]['total_nodes']

    result = graphdb.send_query('MATCH ()-[r]->() RETURN count(r) as total_relationships')
    stats['total_relationships'] = result['query_result'][0]['total_relationships']

    print(f"\n📊 Total Nodes: {stats['total_nodes']}")
    print(f"🔗 Total Relationships: {stats['total_relationships']}")

    # Node labels distribution
    print("\n" + "-"*70)
    print("Node Labels Distribution:")
    print("-"*70)

    result = graphdb.send_query('''
        MATCH (n)
        UNWIND labels(n) as label
        WITH label, count(*) as count
        RETURN label, count
        ORDER BY count DESC
    ''')

    stats['node_labels'] = {}
    for row in result['query_result']:
        label = row['label']
        count = row['count']
        stats['node_labels'][label] = count
        print(f"  {label}: {count}")

    # Domain entities (excluding internal labels)
    print("\n" + "-"*70)
    print("Domain Entities:")
    print("-"*70)

    result = graphdb.send_query('''
        MATCH (n)
        UNWIND labels(n) as label
        WITH label, count(*) as count
        WHERE NOT label IN ['__Entity__', '__KGBuilder__', '__Chunk__', '__Document__']
        RETURN label, count
        ORDER BY count DESC
    ''')

    stats['domain_entities'] = {}
    for row in result['query_result']:
        label = row['label']
        count = row['count']
        stats['domain_entities'][label] = count
        print(f"  {label}: {count}")

    # Relationship types
    print("\n" + "-"*70)
    print("Relationship Types:")
    print("-"*70)

    result = graphdb.send_query('''
        MATCH ()-[r]->()
        WITH type(r) as rel_type, count(*) as count
        RETURN rel_type, count
        ORDER BY count DESC
    ''')

    stats['relationship_types'] = {}
    for row in result['query_result']:
        rel_type = row['rel_type']
        count = row['count']
        stats['relationship_types'][rel_type] = count
        print(f"  {rel_type}: {count}")

    # Chunk embeddings info
    print("\n" + "-"*70)
    print("Chunk Embeddings:")
    print("-"*70)

    result = graphdb.send_query('''
        MATCH (c:Chunk)
        WHERE c.embedding IS NOT NULL
        RETURN count(c) as with_embeddings,
               size(c.embedding) as dimensions
        LIMIT 1
    ''')

    if result['query_result']:
        chunk_info = result['query_result'][0]
        stats['chunk_embeddings'] = {
            'count': chunk_info['with_embeddings'],
            'dimensions': chunk_info['dimensions']
        }
        print(f"  Chunks with embeddings: {chunk_info['with_embeddings']}")
        print(f"  Embedding dimensions: {chunk_info['dimensions']}")
    else:
        stats['chunk_embeddings'] = None
        print("  No chunks with embeddings found")

    # Vector index info
    print("\n" + "-"*70)
    print("Vector Indexes:")
    print("-"*70)

    result = graphdb.send_query("SHOW INDEXES WHERE type = 'VECTOR'")

    stats['vector_indexes'] = []
    if result['query_result']:
        for idx in result['query_result']:
            index_info = {
                'name': idx.get('name'),
                'state': idx.get('state'),
                'populationPercent': idx.get('populationPercent'),
                'type': idx.get('type')
            }
            stats['vector_indexes'].append(index_info)
            print(f"  Name: {idx.get('name')}")
            print(f"  State: {idx.get('state')}")
            print(f"  Population: {idx.get('populationPercent')}%")
    else:
        print("  No vector indexes found")

    return stats


def get_sample_data() -> Dict[str, Any]:
    """Get sample nodes and relationships for verification"""

    print("\n" + "="*70)
    print("Sample Data (First 5 items each)".center(70))
    print("="*70)

    samples = {}

    # Sample domain entities
    print("\n" + "-"*70)
    print("Sample Domain Entities:")
    print("-"*70)

    result = graphdb.send_query('''
        MATCH (n)
        WHERE NOT any(label IN labels(n) WHERE label IN ['__Entity__', '__KGBuilder__', '__Chunk__', '__Document__'])
        RETURN labels(n) as labels, properties(n) as props
        LIMIT 5
    ''')

    samples['domain_entities'] = []
    for i, row in enumerate(result['query_result'], 1):
        entity = {
            'labels': row['labels'],
            'properties': row['props']
        }
        samples['domain_entities'].append(entity)
        print(f"\n  Entity {i}:")
        print(f"    Labels: {row['labels']}")
        print(f"    Properties: {list(row['props'].keys())}")
        if 'name' in row['props']:
            print(f"    Name: {row['props']['name']}")

    # Sample relationships
    print("\n" + "-"*70)
    print("Sample Relationships:")
    print("-"*70)

    result = graphdb.send_query('''
        MATCH (a)-[r]->(b)
        WHERE NOT type(r) IN ['PART_OF', 'FIRST_CHUNK', 'NEXT_CHUNK']
        RETURN labels(a) as source_labels, type(r) as rel_type, labels(b) as target_labels
        LIMIT 5
    ''')

    samples['relationships'] = []
    for i, row in enumerate(result['query_result'], 1):
        rel = {
            'source': row['source_labels'],
            'type': row['rel_type'],
            'target': row['target_labels']
        }
        samples['relationships'].append(rel)
        print(f"\n  Relationship {i}:")
        print(f"    {row['source_labels']} -[{row['rel_type']}]-> {row['target_labels']}")

    # Sample chunks
    print("\n" + "-"*70)
    print("Sample Chunks:")
    print("-"*70)

    result = graphdb.send_query('''
        MATCH (c:Chunk)
        RETURN c.text as text, c.index as index
        ORDER BY c.index
        LIMIT 3
    ''')

    samples['chunks'] = []
    for i, row in enumerate(result['query_result'], 1):
        chunk = {
            'index': row['index'],
            'text': row['text'][:100] + '...' if len(row['text']) > 100 else row['text']
        }
        samples['chunks'].append(chunk)
        print(f"\n  Chunk {i} (index: {row['index']}):")
        print(f"    {chunk['text']}")

    return samples


def export_full_data(output_dir: Path) -> Dict[str, Path]:
    """Export complete graph data to JSON files"""

    print("\n" + "="*70)
    print("Exporting Full Data".center(70))
    print("="*70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / f"vectorstore_dump_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    exported_files = {}

    # Export all nodes
    print("\n📦 Exporting all nodes...")
    result = graphdb.send_query('''
        MATCH (n)
        RETURN id(n) as id, labels(n) as labels, properties(n) as properties
    ''')

    nodes_file = output_dir / "nodes.json"
    with open(nodes_file, 'w', encoding='utf-8') as f:
        json.dump(result['query_result'], f, ensure_ascii=False, indent=2, default=str)
    exported_files['nodes'] = nodes_file
    print(f"   ✅ Saved to: {nodes_file}")

    # Export all relationships
    print("\n🔗 Exporting all relationships...")
    result = graphdb.send_query('''
        MATCH (a)-[r]->(b)
        RETURN id(a) as source_id, labels(a) as source_labels,
               type(r) as rel_type, properties(r) as rel_properties,
               id(b) as target_id, labels(b) as target_labels
    ''')

    rels_file = output_dir / "relationships.json"
    with open(rels_file, 'w', encoding='utf-8') as f:
        json.dump(result['query_result'], f, ensure_ascii=False, indent=2, default=str)
    exported_files['relationships'] = rels_file
    print(f"   ✅ Saved to: {rels_file}")

    # Export chunks (without embeddings to save space)
    print("\n📄 Exporting chunks metadata...")
    result = graphdb.send_query('''
        MATCH (c:Chunk)
        RETURN c.index as index, c.text as text,
               size(c.embedding) as embedding_dims,
               c.embedding IS NOT NULL as has_embedding
        ORDER BY c.index
    ''')

    chunks_file = output_dir / "chunks.json"
    with open(chunks_file, 'w', encoding='utf-8') as f:
        json.dump(result['query_result'], f, ensure_ascii=False, indent=2, default=str)
    exported_files['chunks'] = chunks_file
    print(f"   ✅ Saved to: {chunks_file}")

    print(f"\n📁 All data exported to: {output_dir}")

    return exported_files


def main():
    parser = argparse.ArgumentParser(description="Dump Neo4j vector store data")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Export full data to JSON files"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/neo4j_dumps"),
        help="Output directory for full export (default: data/neo4j_dumps)"
    )
    args = parser.parse_args()

    try:
        # Get and display statistics
        stats = get_graph_statistics()

        # Get and display sample data
        samples = get_sample_data()

        # Create summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'statistics': stats,
            'samples': samples
        }

        # Save summary
        summary_dir = Path("data/neo4j_dumps")
        summary_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_file = summary_dir / f"vectorstore_summary_{timestamp}.json"

        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)

        print("\n" + "="*70)
        print("Summary Saved".center(70))
        print("="*70)
        print(f"\n📄 {summary_file}")

        # Full export if requested
        if args.full:
            exported_files = export_full_data(args.output_dir)

            print("\n" + "="*70)
            print("Export Complete".center(70))
            print("="*70)
            print("\nExported files:")
            for name, path in exported_files.items():
                print(f"  {name}: {path}")

        print("\n" + "="*70)
        print("✅ Dump Complete!".center(70))
        print("="*70)

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
