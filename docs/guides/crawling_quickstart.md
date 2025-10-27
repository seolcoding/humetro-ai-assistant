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

## Knowledge Graph 구축

Knowledge Graph 구축은 **fully managed service**를 활용할 예정입니다.
크롤링된 메타데이터를 해당 서비스에 입력으로 제공합니다.

### Step 4: 메타데이터 확인 및 활용

크롤링된 메타데이터는 다음과 같은 정보를 포함합니다:

```python
import json
from pathlib import Path

def inspect_metadata(output_dir: Path):
    """메타데이터 검사 및 통계"""
    metadata_dir = output_dir / "metadata"

    print("\n📊 Metadata Analysis:")

    for metadata_file in metadata_dir.glob("*.json"):
        with open(metadata_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        print(f"\n📄 {data['title']}")
        print(f"   URL: {data['url']}")
        print(f"   Type: {data['page_type']}")
        print(f"   Words: {data['word_count']}")
        print(f"   Links: {len(data.get('outgoing_links', []))}")
        print(f"   Entities: {', '.join(data.get('entities_preview', [])[:5])}")

        # Link contexts for KG
        if data.get('link_contexts'):
            print(f"   Link contexts: {len(data['link_contexts'])} (useful for KG)")

# 메타데이터 검사
inspect_metadata(output_dir)
```

이 메타데이터를 fully managed KG service에 입력으로 제공하면 됩니다.

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
# 크롤링 테스트 스크립트 실행
uv run python src/scripts/test_crawl.py
```

## 다음 단계

1. **크롤링 확장**: 테스트 성공 후 더 많은 URL 크롤링
2. **KG 구축**: 크롤링된 메타데이터를 fully managed service에 입력
   - 메타데이터의 `link_contexts`, `entities_preview`, `outgoing_links` 활용
   - 계층 구조 (`parent_url`, `siblings`) 정보 활용
3. **RAG 통합**: 구축된 KG를 RAG 파이프라인에 통합

## 참고 자료

- [Data Storage Architecture](../architecture/data_storage_architecture.md)
- [Configuration-Driven Extraction](../architecture/config_driven_extraction.md)
- [PageMetadata Schema](../../src/config/schemas.py)

---

**Last Updated**: 2025-10-27
**Version**: 1.0.0
**Status**: Ready for Test Flight 🚀
