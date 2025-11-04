"""
Seoul Traffic News Crawling Test Flight

크롤링 테스트 플라이트를 실행합니다.
Knowledge Graph 구축은 fully managed service를 사용할 예정입니다.

Usage:
    uv run python src/scripts/test_crawl.py [--pages N]
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import List

from src.crawler.content_extractor_v2 import SeoulTrafficContentExtractorV2
from src.common.project_paths import get_data_dir
from src.config.schemas import PageMetadata


def verify_crawl_results(output_dir: Path, metadata_list: List[PageMetadata]):
    """크롤링 결과 검증 및 통계"""
    raw_dir = output_dir / "raw"
    markdown_dir = output_dir / "markdown"
    metadata_dir = output_dir / "metadata"
    attachments_dir = output_dir / "attachments"

    # 에러 페이지 통계
    error_pages = [m for m in metadata_list if m.is_error_page]
    success_pages = [m for m in metadata_list if not m.is_error_page]

    print("\n" + "=" * 60)
    print("📊 Crawl Results Summary")
    print("=" * 60)
    print(f"   Total pages:    {len(metadata_list)}")
    print(f"   ✅ Success:     {len(success_pages)} ({len(success_pages)/len(metadata_list)*100:.1f}%)")
    print(f"   ❌ Errors:      {len(error_pages)} ({len(error_pages)/len(metadata_list)*100:.1f}%)")
    print()
    print(f"   📄 HTML files:     {len(list(raw_dir.glob('*.html')))}")
    print(f"   📝 Markdown files: {len(list(markdown_dir.glob('*.md')))} (errors filtered out)")
    print(f"   📋 Metadata files: {len(list(metadata_dir.glob('*.json')))}")
    print(f"   📎 Attachments:    {len(list(attachments_dir.glob('*.*')))}")

    # 에러 페이지 샘플
    if error_pages:
        print(f"\n⚠️  Error Page Samples (first 5):")
        for i, error_page in enumerate(error_pages[:5], 1):
            print(f"   {i}. {error_page.title[:50]}")
            print(f"      URL: {error_page.url}")
            print(f"      Reason: {error_page.error_reason}")

    # 성공한 페이지 샘플
    if success_pages:
        print(f"\n✅ Success Page Samples (first 3):")
        for i, page in enumerate(success_pages[:3], 1):
            print(f"   {i}. {page.title[:50]}")
            print(f"      Words: {page.word_count}, URL: {page.url}")

    print("=" * 60)


async def main(num_pages: int = 200):
    """크롤링 테스트 플라이트"""

    print("\n" + "=" * 60)
    print(f"🚀 Seoul Traffic News Crawling - {num_pages} Pages")
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

    # 3. URL 생성 (513625부터 역순으로)
    start_id = 513625
    end_id = start_id - num_pages + 1

    test_urls = []
    for i in range(start_id, end_id - 1, -1):
        test_urls.append({
            "url": f"https://news.seoul.go.kr/traffic/archives/{i}",
            "depth": 1
        })

    url_tree = {}
    for url_dict in test_urls:
        url_tree[url_dict["url"]] = {
            "parent_url": None,
            "depth": url_dict["depth"]
        }

    print(f"\n🕷️  Crawling {len(test_urls)} pages...")
    print(f"   Range: {start_id} → {end_id}")
    print(f"   Estimated time: ~{len(test_urls) * 3 / 60:.1f} minutes")
    print(f"   Error pages will be filtered (no markdown saved)")
    print()

    # 4. 크롤링 실행
    try:
        metadata_list = await extractor.extract_with_metadata(
            urls=test_urls,
            url_tree=url_tree,
            batch_size=5,  # 5개씩 병렬 처리
            delay_between_batches=2.0  # 배치 간 2초 대기
        )

        print(f"\n✅ Crawled {len(metadata_list)}/{len(test_urls)} pages")

        # 5. 결과 검증 및 통계
        verify_crawl_results(output_dir, metadata_list)

        # 6. 상세 통계 저장
        stats = {
            "total_pages": len(metadata_list),
            "success_pages": len([m for m in metadata_list if not m.is_error_page]),
            "error_pages": len([m for m in metadata_list if m.is_error_page]),
            "total_words": sum(m.word_count for m in metadata_list if not m.is_error_page),
            "total_attachments": sum(m.attachment_count for m in metadata_list),
            "crawl_range": f"{start_id} - {end_id}",
            "error_page_urls": [
                {"url": m.url, "title": m.title, "reason": m.error_reason}
                for m in metadata_list if m.is_error_page
            ]
        }

        stats_path = output_dir / "crawl_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        print(f"\n✅ Crawl Complete!")
        print(f"   📁 Crawled data: {output_dir}")
        print(f"   📊 Statistics:   {stats_path}")
        print(f"\n💡 Next Steps:")
        print(f"   • Review crawl stats:       cat {stats_path}")
        print(f"   • Check success pages:      ls {output_dir}/markdown/")
        print(f"   • Inspect metadata:         ls {output_dir}/metadata/")
        print(f"   • Feed to KG service:       Use metadata/*.json")
        print()

    except KeyboardInterrupt:
        print(f"\n\n⚠️  Crawling interrupted by user")
        print(f"   Partial data may be saved in: {output_dir}")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Error during crawling: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Crawl Seoul Traffic News")
    parser.add_argument(
        '--pages',
        type=int,
        default=200,
        help='Number of pages to crawl (default: 200)'
    )
    args = parser.parse_args()

    asyncio.run(main(num_pages=args.pages))
