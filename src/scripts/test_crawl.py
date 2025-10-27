"""
Seoul Traffic News Crawling Test Flight

크롤링 테스트 플라이트를 실행합니다.
Knowledge Graph 구축은 fully managed service를 사용할 예정입니다.

Usage:
    uv run python src/scripts/test_crawl.py
"""

import asyncio
import json
from pathlib import Path
from typing import List

from src.crawler.content_extractor_v2 import SeoulTrafficContentExtractorV2
from src.common.project_paths import get_data_dir
from src.config.schemas import PageMetadata


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
    """크롤링 테스트 플라이트"""

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

        print(f"\n✅ Test Flight Complete!")
        print(f"   📁 Crawled data: {output_dir}")
        print(f"\n💡 Next Steps:")
        print(f"   • Inspect crawled metadata: ls {output_dir}/metadata/")
        print(f"   • Review HTML files:        ls {output_dir}/raw/")
        print(f"   • Check markdown output:    ls {output_dir}/markdown/")
        print(f"   • Feed metadata to fully managed KG service")
        print()

    except Exception as e:
        print(f"\n❌ Error during crawling: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
