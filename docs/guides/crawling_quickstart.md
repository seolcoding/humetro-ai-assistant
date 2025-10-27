# Crawling Quickstart Guide

**Version**: 1.0.0
**Last Updated**: 2025-10-27
**Purpose**: 서울시 교통 뉴스 크롤링 테스트 플라이트 및 KG 구축 가이드

## 개요

이 가이드는 Seoul Traffic News를 크롤링하고, 결과를 바탕으로 심플한 Knowledge Graph를 구축하는 방법을 설명합니다.

## 크롤링 결과 저장 위치

### 표준 저장 경로

```
data/
└── crawled/
    └── seoul_traffic/           # 서울시 교통 뉴스
        ├── raw/                 # 원본 HTML
        ├── markdown/            # Markdown 변환본
        ├── metadata/            # JSON 메타데이터
        └── attachments/         # 첨부파일 (HWP, PDF 등)
```

### 경로 설정 (project_paths 사용)

```python
from src.common.project_paths import get_data_dir

# 표준 경로
data_dir = get_data_dir()
output_dir = data_dir / "crawled" / "seoul_traffic"
download_dir = data_dir / "crawled" / "seoul_traffic"

# 디렉토리 구조:
# output_dir/raw/          → HTML 파일
# output_dir/markdown/     → Markdown 파일
# output_dir/metadata/     → JSON 메타데이터
# download_dir/attachments/ → 첨부파일
```

## 크롤링 테스트 플라이트

### Step 1: 크롤러 초기화

```python
from pathlib import Path
from src.crawler.content_extractor_v2 import SeoulTrafficContentExtractorV2
from src.common.project_paths import get_data_dir

# 저장 경로 설정
data_dir = get_data_dir()
output_dir = data_dir / "crawled" / "seoul_traffic"
download_dir = data_dir / "crawled" / "seoul_traffic"

# Extractor 생성
extractor = SeoulTrafficContentExtractorV2.from_domain(
    domain="news.seoul.go.kr",
    output_dir=output_dir,
    download_dir=download_dir,
    headless=True,
    verbose=True
)

print(f"✅ Extractor initialized")
print(f"📁 Output: {output_dir}")
print(f"📥 Downloads: {download_dir}")
```

### Step 2: 테스트 URL 크롤링 (소규모)

```python
import asyncio

# 테스트 URL 목록 (5-10개 정도)
test_urls = [
    {"url": "https://news.seoul.go.kr/traffic/archives/513625", "depth": 1},
    {"url": "https://news.seoul.go.kr/traffic/archives/513624", "depth": 1},
    {"url": "https://news.seoul.go.kr/traffic/archives/513623", "depth": 1},
    {"url": "https://news.seoul.go.kr/traffic/archives/513622", "depth": 1},
    {"url": "https://news.seoul.go.kr/traffic/archives/513621", "depth": 1},
]

# URL 트리 구조
url_tree = {}
for url_dict in test_urls:
    url_tree[url_dict["url"]] = {
        "parent_url": None,
        "depth": url_dict["depth"]
    }

# 크롤링 실행
async def run_test_crawl():
    metadata_list = await extractor.extract_with_metadata(
        urls=test_urls,
        url_tree=url_tree,
        batch_size=2,  # 테스트용: 2개씩
        delay_between_batches=3.0  # 서버 부하 방지
    )

    print(f"\n✅ Crawled {len(metadata_list)} pages")

    # 결과 확인
    for metadata in metadata_list:
        print(f"\n📄 {metadata.title}")
        print(f"   URL: {metadata.url}")
        print(f"   Word count: {metadata.word_count}")
        print(f"   Attachments: {metadata.attachment_count}")

    return metadata_list

# 실행
metadata_list = await run_test_crawl()
```

### Step 3: 결과 검증

```python
from pathlib import Path
import json

def verify_crawl_results(output_dir: Path):
    """크롤링 결과 검증"""

    # 디렉토리 확인
    raw_dir = output_dir / "raw"
    markdown_dir = output_dir / "markdown"
    metadata_dir = output_dir / "metadata"
    attachments_dir = output_dir / "attachments"

    print("\n📊 Crawl Results Summary:")
    print(f"  HTML files: {len(list(raw_dir.glob('*.html')))}")
    print(f"  Markdown files: {len(list(markdown_dir.glob('*.md')))}")
    print(f"  Metadata files: {len(list(metadata_dir.glob('*.json')))}")
    print(f"  Attachments: {len(list(attachments_dir.glob('*.*')))}")

    # 메타데이터 샘플 확인
    metadata_files = list(metadata_dir.glob('*.json'))
    if metadata_files:
        with open(metadata_files[0], 'r', encoding='utf-8') as f:
            sample = json.load(f)
            print(f"\n📋 Sample Metadata:")
            print(f"  Title: {sample['title']}")
            print(f"  URL: {sample['url']}")
            print(f"  Word count: {sample['word_count']}")
            print(f"  Has attachments: {sample['has_attachments']}")

# 검증 실행
verify_crawl_results(output_dir)
```

## Knowledge Graph 구축 준비

### KG 저장 경로

```
data/
└── knowledge_graphs/
    └── seoul_traffic_simple/     # 심플 KG
        ├── nodes.json           # 노드 목록
        ├── edges.json           # 관계(엣지) 목록
        ├── graph.json           # 전체 그래프
        └── stats.json           # 통계 정보
```

### Step 4: 메타데이터에서 그래프 데이터 추출

```python
from typing import List, Dict, Set
import json
from pathlib import Path
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
                        'context': link_context.surrounding_text[:100],
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

    return {
        'nodes': nodes,
        'edges': edges,
        'stats': {
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            'node_types': _count_by_type(nodes),
            'edge_types': _count_by_type(edges)
        }
    }

def _count_by_type(items: List[Dict]) -> Dict[str, int]:
    """타입별 개수 집계"""
    counts = {}
    for item in items:
        item_type = item['type']
        counts[item_type] = counts.get(item_type, 0) + 1
    return counts

# 그래프 데이터 추출
graph_data = extract_graph_data(metadata_list)

print(f"\n🕸️ Graph Statistics:")
print(f"  Nodes: {graph_data['stats']['total_nodes']}")
print(f"  Edges: {graph_data['stats']['total_edges']}")
print(f"  Node types: {graph_data['stats']['node_types']}")
print(f"  Edge types: {graph_data['stats']['edge_types']}")
```

### Step 5: 그래프 저장

```python
from src.common.project_paths import get_data_dir

def save_graph(graph_data: Dict, graph_name: str = "seoul_traffic_simple"):
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
    print(f"  nodes.json: {len(graph_data['nodes'])} nodes")
    print(f"  edges.json: {len(graph_data['edges'])} edges")

    return kg_dir

# 그래프 저장
kg_dir = save_graph(graph_data)
```

### Step 6: 그래프 시각화 (간단한 확인)

```python
def print_graph_summary(graph_data: Dict):
    """그래프 요약 출력"""

    print("\n📊 Knowledge Graph Summary")
    print("=" * 60)

    # 노드 요약
    print(f"\n🔵 Nodes ({graph_data['stats']['total_nodes']} total):")
    for node_type, count in graph_data['stats']['node_types'].items():
        print(f"  - {node_type}: {count}")

    # 샘플 노드
    print(f"\n📄 Sample Nodes:")
    for node in graph_data['nodes'][:3]:
        print(f"  [{node['type']}] {node['label']}")

    # 엣지 요약
    print(f"\n🔗 Edges ({graph_data['stats']['total_edges']} total):")
    for edge_type, count in graph_data['stats']['edge_types'].items():
        print(f"  - {edge_type}: {count}")

    # 샘플 엣지
    print(f"\n🔗 Sample Edges:")
    for edge in graph_data['edges'][:3]:
        source_label = next((n['label'] for n in graph_data['nodes'] if n['id'] == edge['source']), 'Unknown')
        target_label = next((n['label'] for n in graph_data['nodes'] if n['id'] == edge['target']), 'Unknown')
        print(f"  {source_label[:30]} --[{edge['type']}]--> {target_label[:30]}")

# 요약 출력
print_graph_summary(graph_data)
```

## 완전한 테스트 스크립트

위의 모든 단계를 하나의 스크립트로:

```python
# src/scripts/test_crawl_and_kg.py

import asyncio
import json
from pathlib import Path
from typing import List, Dict

from src.crawler.content_extractor_v2 import SeoulTrafficContentExtractorV2
from src.common.project_paths import get_data_dir
from src.config.schemas import PageMetadata


async def main():
    """크롤링 테스트 플라이트 및 심플 KG 구축"""

    print("🚀 Seoul Traffic News Crawling Test Flight")
    print("=" * 60)

    # 1. 경로 설정
    data_dir = get_data_dir()
    output_dir = data_dir / "crawled" / "seoul_traffic"
    download_dir = data_dir / "crawled" / "seoul_traffic"

    print(f"\n📁 Directories:")
    print(f"  Output: {output_dir}")
    print(f"  Downloads: {download_dir}")

    # 2. Extractor 초기화
    print(f"\n⚙️  Initializing extractor...")
    extractor = SeoulTrafficContentExtractorV2.from_domain(
        domain="news.seoul.go.kr",
        output_dir=output_dir,
        download_dir=download_dir,
        headless=True,
        verbose=True
    )

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

    # 4. 크롤링 실행
    print(f"\n🕷️  Crawling {len(test_urls)} pages...")
    metadata_list = await extractor.extract_with_metadata(
        urls=test_urls,
        url_tree=url_tree,
        batch_size=2,
        delay_between_batches=3.0
    )

    print(f"\n✅ Crawled {len(metadata_list)} pages successfully")

    # 5. 결과 검증
    print(f"\n📊 Crawl Results:")
    for i, metadata in enumerate(metadata_list, 1):
        print(f"  {i}. {metadata.title}")
        print(f"     Words: {metadata.word_count}, Attachments: {metadata.attachment_count}")

    # 6. KG 구축
    print(f"\n🕸️  Building Knowledge Graph...")
    graph_data = extract_graph_data(metadata_list)

    # 7. KG 저장
    kg_dir = save_graph(graph_data, "seoul_traffic_simple")

    # 8. 요약 출력
    print_graph_summary(graph_data)

    print(f"\n✅ Test Flight Complete!")
    print(f"   Crawled data: {output_dir}")
    print(f"   Knowledge Graph: {kg_dir}")


if __name__ == "__main__":
    asyncio.run(main())
```

## 실행 방법

```bash
# 테스트 스크립트 실행
uv run python src/scripts/test_crawl_and_kg.py
```

## 다음 단계

1. **크롤링 확장**: 테스트 성공 후 더 많은 URL 크롤링
2. **KG 고도화**:
   - Named Entity Recognition (NER) 추가
   - 관계 추론 (co-occurrence, semantic similarity)
   - 시간 정보 추가 (temporal KG)
3. **RAG 통합**: KG를 RAG 파이프라인에 통합

## 참고 자료

- [Data Storage Architecture](../architecture/data_storage_architecture.md)
- [Configuration-Driven Extraction](../architecture/config_driven_extraction.md)
- [PageMetadata Schema](../../src/config/schemas.py)

---

**Last Updated**: 2025-10-27
**Version**: 1.0.0
**Status**: Ready for Test Flight 🚀
