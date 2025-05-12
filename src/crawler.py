import asyncio
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import List

import aiofiles
import litellm
from bs4 import BeautifulSoup
from bson import Timestamp
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    LLMConfig,
    LXMLWebScrapingStrategy,
)
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.deep_crawling.filters import FilterChain, URLPatternFilter
from crawl4ai.extraction_strategy import LLMExtractionStrategy
from crawl4ai.models import CrawlResult
from pydantic import BaseModel

# URL 패턴 필터 변경 - 더 많은 페이지 크롤링을 위해
url_filter = URLPatternFilter(patterns=["*humetro.busan.kr*"])

deep_crawl_strategy = BFSDeepCrawlStrategy(
    max_depth=3,
    include_external=False,
    max_pages=10,  # 최대 페이지 수 증가
    filter_chain=(FilterChain([url_filter])),
)


class MetroWebData(BaseModel):
    title: str
    content: str
    tags: List[str]
    source_url: str


def sanitize_filename(filename: str) -> str:
    return re.sub(r"[^a-z가-힣A-Z0-9_-]", "", filename)


async def process_result(result: CrawlResult):
    """비동기적으로 결과를 처리하는 함수"""
    if not result.success:
        print(f"크롤링 실패: {result.url} - {result.error_message}")
        return

    if not result.markdown or not result.markdown.strip():
        print(f"마크다운 없음: {result.url}")
        return

    content = result.markdown.strip()
    if not content:
        return

    # 제목 추출
    subtop_title = BeautifulSoup(result.html, "lxml").select_one("p.subTop-Title")
    if subtop_title:
        file_title = subtop_title.text
    else:
        file_title = "no_title"

    # 결과 디렉토리 생성
    os.makedirs("crawl4ai_result_css", exist_ok=True)

    # 파일명 생성 및 결과 저장
    filename = (
        f"crawl4ai_result_css/{sanitize_filename(file_title)}_{str(time.time())}.md"
    )

    # 비동기 파일 쓰기
    async with aiofiles.open(filename, "w") as f:
        await f.write(result.markdown)

    print(f"저장 완료: {file_title} - {result.url}")

    # 추출 내용 처리
    if result.extracted_content:
        try:
            article = json.loads(result.extracted_content)
            print("+" * 50)
            print(f"추출 내용: {article}")
            print("+" * 50)
        except Exception as e:
            print(f"추출 내용 파싱 오류: {e}")


async def crawl_humetro():
    # 결과를 저장할 디렉토리 생성
    output_dir = Path("crawl4ai_result_css")
    output_dir.mkdir(exist_ok=True)

    # LLM 추출 전략 설정
    llm_strategy = LLMExtractionStrategy(
        llm_config=LLMConfig(
            provider="openai/gpt-4o-mini",
            api_token=os.getenv("OPENAI_API_KEY"),
        ),
        schema=MetroWebData.model_json_schema(),
        extract_method="schema",
        instruction="""
        You are a helpful assistant that extracts information from a given text.
        you must extract all information in KOREAN.
        특히, 도시철도 이용객의 관점에서 유용할만한 정보를 기준으로 추출해야함.

        title은 현재 페이지의 고유정보를 가장 잘 나타내는 제목
        content는 현재 페이지의 내용을 가장 잘 나타내는 내용, 웹페이지의 링크나 메뉴는 제외하고 고유 정보만 추출
        tags는 content의 정보에 대한 다차원적인 인덱싱하기 용이한것으로 최대 5개까지 추출
        source_url은 현재 페이지의 주소
        """,
        input_format="markdown",
    )

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
        verbose=True,
        screenshot=False,
        target_elements=["#section"],
        exclude_social_media_links=True,
        exclude_internal_links=False,  # 내부 링크 허용
        exclude_external_images=True,
        extraction_strategy=llm_strategy,
        stream=False,  # 스트리밍 모드 비활성화
    )

    # 크롤링할 URL 목록
    start_urls = [
        "https://www.humetro.busan.kr/homepage/default/index.do",
        "https://www.humetro.busan.kr/homepage/default/menu/1000.do",
        "https://www.humetro.busan.kr/homepage/default/menu/2000.do",
        "https://www.humetro.busan.kr/homepage/default/menu/3000.do",
        "https://www.humetro.busan.kr/homepage/default/menu/9000.do",
    ]

    async with AsyncWebCrawler() as crawler:
        try:
            print("크롤링 시작...")
            start_time = time.time()

            # 여러 URL 병렬 크롤링 (stream=False)
            results = await crawler.arun_many(
                urls=start_urls, config=config, browser_config=browser_config
            )

            # 결과를 비동기적으로 처리
            tasks = []
            count = 0

            # 모든 결과 비동기 처리
            for result in results:
                task = asyncio.create_task(process_result(result))
                tasks.append(task)
                count += 1

            # 모든 처리 태스크가 완료될 때까지 대기
            if tasks:
                await asyncio.gather(*tasks)

            end_time = time.time()
            print(
                f"크롤링 완료: 총 {count}개 URL 처리, 소요시간: {end_time - start_time:.2f}초"
            )

        except Exception as e:
            print(f"크롤링 오류: {e}")


if __name__ == "__main__":
    asyncio.run(crawl_humetro())
