"""
Seoul Traffic News Crawling Test Flight & Simple KG Builder

크롤링 테스트 플라이트를 실행하고 심플한 Knowledge Graph를 구축합니다.

Usage:
    uv run python src/scripts/test_crawl_and_kg.py
"""

import asyncio
import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime

from src.crawler.content_extractor_v2 import SeoulTrafficContentExtractorV2
from src.common.project_paths import get_data_dir
from src.config.schemas import PageMetadata


def extract_graph_data(metadata_list: List[PageMetadata]) -> Dict:
    """
    크롤링 메타데이터에서 그래프 데이터 추출

    Returns:
        {
            'nodes': [{id, type, label, properties}, ...],
            'edges': [{source, target, type, properties}, ...]
        }
    """
    nodes = []
    edges = []
    node_ids = set()

    for metadata in metadata_list:
        url = metadata.url

        # 페이지 노드 추가
        if url not in node_ids:
            nodes.append({
                'id': url,
                'type': 'page',
                'label': metadata.title,
                'properties': {
                    'page_type': metadata.page_type,
                    'word_count': metadata.word_count,
                    'crawled_at': metadata.crawled_at.isoformat(),
                    'depth': metadata.depth
                }
            })
            node_ids.add(url)

        # 부모 링크 (계층 구조)
        if metadata.parent_url:
            edges.append({
                'source': metadata.parent_url,
                'target': url,
                'type': 'parent_of',
                'properties': {'depth_diff': 1}
            })

        # 형제 링크
        for sibling_url in metadata.siblings:
            if sibling_url in node_ids:  # 이미 크롤링된 페이지만
                edges.append({
                    'source': url,
                    'target': sibling_url,
                    'type': 'sibling_of',
                    'properties': {}
                })

        # 외부 링크 (link contexts 포함)
        for link_context in metadata.link_contexts:
            target_url = link_context.target_url

            # 내부 링크만 (같은 도메인)
            if 'news.seoul.go.kr' in target_url:
                edges.append({
                    'source': url,
                    'target': target_url,
                    'type': 'links_to',
                    'properties': {
                        'anchor_text': link_context.anchor_text,
                        'context': link_context.surrounding_text[:100] if link_context.surrounding_text else "",
                        'is_navigation': link_context.is_navigation
                    }
                })

        # 엔티티 노드 추가 (간단한 버전)
        for entity in metadata.entities_preview[:5]:  # 상위 5개만
            entity_id = f"entity:{entity}"
            if entity_id not in node_ids:
                nodes.append({
                    'id': entity_id,
                    'type': 'entity',
                    'label': entity,
                    'properties': {}
                })
                node_ids.add(entity_id)

            # 페이지-엔티티 관계
            edges.append({
                'source': url,
                'target': entity_id,
                'type': 'mentions',
                'properties': {}
            })

    # 타입별 개수 집계
    node_types = {}
    for node in nodes:
        node_type = node['type']
        node_types[node_type] = node_types.get(node_type, 0) + 1

    edge_types = {}
    for edge in edges:
        edge_type = edge['type']
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

    return {
        'nodes': nodes,
        'edges': edges,
        'stats': {
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            'node_types': node_types,
            'edge_types': edge_types,
            'created_at': datetime.now().isoformat()
        }
    }


def save_graph(graph_data: Dict, graph_name: str = "seoul_traffic_simple") -> Path:
    """그래프 데이터 저장"""

    # 저장 경로
    data_dir = get_data_dir()
    kg_dir = data_dir / "knowledge_graphs" / graph_name
    kg_dir.mkdir(parents=True, exist_ok=True)

    # 노드 저장
    nodes_path = kg_dir / "nodes.json"
    with open(nodes_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data['nodes'], f, ensure_ascii=False, indent=2)

    # 엣지 저장
    edges_path = kg_dir / "edges.json"
    with open(edges_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data['edges'], f, ensure_ascii=False, indent=2)

    # 전체 그래프 저장
    graph_path = kg_dir / "graph.json"
    with open(graph_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=2)

    # 통계 저장
    stats_path = kg_dir / "stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(graph_data['stats'], f, ensure_ascii=False, indent=2)

    print(f"\n✅ Graph saved to: {kg_dir}")
    print(f"   nodes.json: {len(graph_data['nodes'])} nodes")
    print(f"   edges.json: {len(graph_data['edges'])} edges")

    return kg_dir


def print_graph_summary(graph_data: Dict):
    """그래프 요약 출력"""

    print("\n" + "=" * 60)
    print("📊 Knowledge Graph Summary")
    print("=" * 60)

    # 노드 요약
    print(f"\n🔵 Nodes ({graph_data['stats']['total_nodes']} total):")
    for node_type, count in graph_data['stats']['node_types'].items():
        print(f"   • {node_type}: {count}")

    # 샘플 노드
    print(f"\n📄 Sample Nodes:")
    for node in graph_data['nodes'][:3]:
        print(f"   [{node['type']:8s}] {node['label'][:50]}")

    # 엣지 요약
    print(f"\n🔗 Edges ({graph_data['stats']['total_edges']} total):")
    for edge_type, count in graph_data['stats']['edge_types'].items():
        print(f"   • {edge_type}: {count}")

    # 샘플 엣지
    print(f"\n🔗 Sample Edges:")
    for edge in graph_data['edges'][:5]:
        source_label = next((n['label'] for n in graph_data['nodes'] if n['id'] == edge['source']), 'Unknown')
        target_label = next((n['label'] for n in graph_data['nodes'] if n['id'] == edge['target']), 'Unknown')
        print(f"   {source_label[:25]:25s} --[{edge['type']:12s}]--> {target_label[:25]:25s}")

    print("\n" + "=" * 60)


def verify_crawl_results(output_dir: Path):
    """크롤링 결과 검증"""

    raw_dir = output_dir / "raw"
    markdown_dir = output_dir / "markdown"
    metadata_dir = output_dir / "metadata"
    attachments_dir = output_dir / "attachments"

    print("\n" + "=" * 60)
    print("📁 Crawl Results Summary")
    print("=" * 60)
    print(f"   HTML files:     {len(list(raw_dir.glob('*.html')))}")
    print(f"   Markdown files: {len(list(markdown_dir.glob('*.md')))}")
    print(f"   Metadata files: {len(list(metadata_dir.glob('*.json')))}")
    print(f"   Attachments:    {len(list(attachments_dir.glob('*.*')))}")

    # 메타데이터 샘플 확인
    metadata_files = list(metadata_dir.glob('*.json'))
    if metadata_files:
        with open(metadata_files[0], 'r', encoding='utf-8') as f:
            sample = json.load(f)
            print(f"\n📋 Sample Metadata:")
            print(f"   Title: {sample['title']}")
            print(f"   URL: {sample['url']}")
            print(f"   Word count: {sample['word_count']}")
            print(f"   Has attachments: {sample['has_attachments']}")

    print("=" * 60)


async def main():
    """크롤링 테스트 플라이트 및 심플 KG 구축"""

    print("\n" + "=" * 60)
    print("🚀 Seoul Traffic News Crawling Test Flight")
    print("=" * 60)

    # 1. 경로 설정
    data_dir = get_data_dir()
    output_dir = data_dir / "crawled" / "seoul_traffic"
    download_dir = data_dir / "crawled" / "seoul_traffic"

    print(f"\n📁 Directories:")
    print(f"   Output:    {output_dir}")
    print(f"   Downloads: {download_dir}")

    # 2. Extractor 초기화
    print(f"\n⚙️  Initializing extractor...")
    extractor = SeoulTrafficContentExtractorV2.from_domain(
        domain="news.seoul.go.kr",
        output_dir=output_dir,
        download_dir=download_dir,
        headless=True,
        verbose=False  # 깔끔한 출력을 위해
    )
    print("   ✅ Extractor ready")

    # 3. 테스트 URL (최신 5개 게시물)
    test_urls = [
        {"url": "https://news.seoul.go.kr/traffic/archives/513625", "depth": 1},
        {"url": "https://news.seoul.go.kr/traffic/archives/513624", "depth": 1},
        {"url": "https://news.seoul.go.kr/traffic/archives/513623", "depth": 1},
        {"url": "https://news.seoul.go.kr/traffic/archives/513622", "depth": 1},
        {"url": "https://news.seoul.go.kr/traffic/archives/513621", "depth": 1},
    ]

    url_tree = {}
    for url_dict in test_urls:
        url_tree[url_dict["url"]] = {
            "parent_url": None,
            "depth": url_dict["depth"]
        }

    print(f"\n🕷️  Crawling {len(test_urls)} pages...")
    print("   (This may take 30-60 seconds...)")

    # 4. 크롤링 실행
    try:
        metadata_list = await extractor.extract_with_metadata(
            urls=test_urls,
            url_tree=url_tree,
            batch_size=2,
            delay_between_batches=3.0
        )

        print(f"\n✅ Crawled {len(metadata_list)} pages successfully")

        # 5. 결과 상세
        print(f"\n📄 Crawled Pages:")
        for i, metadata in enumerate(metadata_list, 1):
            print(f"   {i}. {metadata.title}")
            print(f"      Words: {metadata.word_count:4d} | Attachments: {metadata.attachment_count}")

        # 6. 결과 검증
        verify_crawl_results(output_dir)

        # 7. KG 구축
        print(f"\n🕸️  Building Knowledge Graph...")
        graph_data = extract_graph_data(metadata_list)
        print(f"   ✅ Graph built with {graph_data['stats']['total_nodes']} nodes, {graph_data['stats']['total_edges']} edges")

        # 8. KG 저장
        kg_dir = save_graph(graph_data, "seoul_traffic_simple")

        # 9. 요약 출력
        print_graph_summary(graph_data)

        print(f"\n✅ Test Flight Complete!")
        print(f"   📁 Crawled data:    {output_dir}")
        print(f"   🕸️  Knowledge Graph: {kg_dir}")
        print(f"\n💡 Next Steps:")
        print(f"   • Inspect crawled data: ls {output_dir}/metadata/")
        print(f"   • View KG:             cat {kg_dir}/stats.json")
        print(f"   • Build RAG pipeline:  Use KG for semantic search")
        print()

    except Exception as e:
        print(f"\n❌ Error during crawling: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
