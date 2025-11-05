#!/usr/bin/env python3
"""
Quick script to check the Knowledge Graph in Neo4j
"""

from utils.neo4j_for_adk import graphdb

def main():
    print("=" * 70)
    print("Neo4j Knowledge Graph Inspection")
    print("=" * 70)

    # 1. Total counts
    result = graphdb.send_query('MATCH (n) RETURN count(n) as total_nodes')
    total_nodes = result['query_result'][0]['total_nodes']

    result = graphdb.send_query('MATCH ()-[r]->() RETURN count(r) as total_rels')
    total_rels = result['query_result'][0]['total_rels']

    print(f"\n📊 Graph Statistics:")
    print(f"   Total Nodes: {total_nodes}")
    print(f"   Total Relationships: {total_rels}")

    # 2. All labels
    result = graphdb.send_query('CALL db.labels()')
    print(f"\n🏷️  All Node Labels:")
    for record in result['query_result']:
        print(f"   - {record['label']}")

    # 3. Nodes by label (excluding internal labels)
    result = graphdb.send_query('''
        MATCH (n)
        UNWIND labels(n) as label
        WITH label, count(*) as count
        WHERE NOT label IN ['__Entity__', '__KGBuilder__', '__Chunk__', '__Document__']
        RETURN label, count
        ORDER BY count DESC
    ''')
    print(f"\n📦 Domain Entity Counts:")
    for record in result['query_result']:
        print(f"   {record['label']}: {record['count']}")

    # 4. All relationship types
    result = graphdb.send_query('CALL db.relationshipTypes()')
    print(f"\n🔗 All Relationship Types:")
    for record in result['query_result']:
        print(f"   - {record['relationshipType']}")

    # 5. Domain relationships (excluding internal)
    result = graphdb.send_query('''
        MATCH ()-[r]->()
        WITH type(r) as rel_type, count(*) as count
        WHERE NOT rel_type IN ['FROM_CHUNK', 'FROM_DOCUMENT', 'NEXT_CHUNK', 'HAS_ENTITY']
        RETURN rel_type, count
        ORDER BY count DESC
    ''')
    print(f"\n🔗 Domain Relationship Counts:")
    for record in result['query_result']:
        print(f"   {record['rel_type']}: {record['count']}")

    # 6. Sample domain entities
    result = graphdb.send_query('''
        MATCH (n)
        WHERE NOT any(label IN labels(n) WHERE label IN ['__Entity__', '__KGBuilder__', '__Chunk__', '__Document__'])
        RETURN labels(n) as labels, n.name as name, id(n) as id
        LIMIT 10
    ''')
    print(f"\n🔍 Sample Domain Entities:")
    for i, record in enumerate(result['query_result'], 1):
        labels = ', '.join(record['labels'])
        name = record['name'] if record['name'] else '(no name)'
        print(f"   {i}. [{labels}] {name}")

    # 7. Sample relationships with entities
    result = graphdb.send_query('''
        MATCH (a)-[r]->(b)
        WHERE NOT type(r) IN ['FROM_CHUNK', 'FROM_DOCUMENT', 'NEXT_CHUNK', 'HAS_ENTITY']
        RETURN labels(a)[0] as source_label, a.name as source_name,
               type(r) as rel_type,
               labels(b)[0] as target_label, b.name as target_name
        LIMIT 10
    ''')
    print(f"\n🌐 Sample Relationships:")
    for i, record in enumerate(result['query_result'], 1):
        source = f"[{record['source_label']}] {record['source_name'] or '?'}"
        target = f"[{record['target_label']}] {record['target_name'] or '?'}"
        rel = record['rel_type']
        print(f"   {i}. {source} --[{rel}]--> {target}")

    graphdb.close()
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
