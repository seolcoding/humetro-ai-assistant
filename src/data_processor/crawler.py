import asyncio
import json
import os
import re
from pathlib import Path

from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    LXMLWebScrapingStrategy,
)
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

# 타겟 URL의 타겟 크롤 셀렉터 div.subContents
# URL 패턴 필터 변경 - 더 많은 페이지 크롤링을 위해
url_filter = URLPatternFilter(patterns=["*humetro.busan.kr*"])

deep_crawl_strategy = BFSDeepCrawlStrategy(
    max_depth=2,
    include_external=False,
    max_pages=10,  # 최대 페이지 수 증가
    filter_chain=(FilterChain([url_filter])),
)


def clean_text(text):
    # '\xa0'는 non-breaking space를 의미합니다
    # 이를 일반 공백으로 변환하고, 여러 공백을 하나로 합치고, 양쪽 공백을 제거합니다
    text = text.replace("\xa0", " ")

    # 줄바꿈을 공백으로 대체
    text = text.replace("\n", " ")

    # 여러 공백을 하나로 합치기
    import re

    text = re.sub(r"\s+", " ", text)

    # 양쪽 공백 제거
    text = text.strip()
    text = text.replace(">", "")

    return text


def sanitize_filename(filename: str) -> str:
    return re.sub(r"[^a-z가-힣A-Z0-9_-]", "", filename)


async def crawl_humetro(target_urls):
    # 결과를 저장할 디렉토리 생성
    output_dir = Path("crawl_result")
    output_dir.mkdir(exist_ok=True)

    # 브라우저 설정
    browser_config = BrowserConfig(
        headless=False,
        user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    )

    # 크롤러 설정 - 스트리밍 모드 비활성화
    config = CrawlerRunConfig(
        prettiify=True,
        remove_forms=False,
        check_robots_txt=False,
        log_console=True,
        deep_crawl_strategy=deep_crawl_strategy,
        scraping_strategy=LXMLWebScrapingStrategy(),
        markdown_generator=DefaultMarkdownGenerator(
            options={
                "ignore_links": True,
                "ignore_images": False,
            }
        ),
        verbose=True,
        screenshot=False,
        target_elements=["div.subContents"],
        exclude_social_media_links=True,
        exclude_internal_links=False,  # 내부 링크 허용
        exclude_external_images=True,
        stream=False,  # 스트리밍 모드 비활성화
    )
    target_urls = [i["url"] for i in target_urls]

    results = []
    async with AsyncWebCrawler() as crawler:
        for url in target_urls:
            try:
                crawl_results = await crawler.arun(
                    url=url,
                    config=config,
                    browser_config=browser_config,
                )
                if crawl_results[0].markdown and crawl_results[0].markdown != "":
                    print(crawl_results[0].markdown[:80])
                    results.extend(crawl_results)
                else:
                    print(f"마크다운 생성 실패: {url}")

            except Exception as e:
                print(f"크롤링 오류: {e}")

    return results


async def main(target_urls):
    results = await crawl_humetro(target_urls)
    for result in results:
        try:
            title = result.metadata.get("title", "")
            title = clean_text(title)
            filepath = Path(f"crawl_result/{title}.md")
            if os.path.exists(filepath):
                title = f"{title}_{result.url.split('/')[-1]}"
                filepath = Path(f"crawl_result/{title}.md")
            with open(filepath, "w") as f:
                f.write(result.markdown)
        except Exception:
            print(f"마크다운 저장 실패: {result.url}")


if __name__ == "__main__":
    with open(
        "/Users/sdh/Dev/projects/humetro-ai-assistant/datasets/Rag_Docs/crawl_targets.json",
        "r",
    ) as f:
        target_urls = json.load(f)
    if not target_urls:
        raise
    asyncio.run(main(target_urls))
