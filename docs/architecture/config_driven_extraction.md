# Configuration-Driven Web Scraping Architecture

## 개요

웹 크롤링 시스템을 YAML 기반 설정 파일을 사용하는 유연한 아키텍처로 재설계했습니다. 이를 통해 코드 수정 없이 새로운 사이트를 추가하고 CSS 셀렉터를 관리할 수 있습니다.

### 핵심 원칙
- **관심사의 분리**: CSS 셀렉터는 YAML, 추출 로직은 Python
- **전략 패턴**: AbstractExtractor 기반 사이트별 구현
- **설정 기반**: 새 사이트 추가 시 코드 변경 불필요
- **하위 호환성**: 기존 ContentExtractor API 유지

## 아키텍처 다이어그램

```
┌─────────────────────────────────────────────────────────┐
│                    ContentExtractorV2                    │
│  (Orchestrates extraction with ConfigBasedExtractor)    │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ├─> ConfigBasedExtractor (Strategy)
                  │   │
                  │   ├─> extract_from_page()
                  │   ├─> extract_attachments()
                  │   └─> extract_popup_content()
                  │
                  ├─> SiteConfig (loaded from YAML)
                  │   │
                  │   ├─> ArticleSelectors
                  │   ├─> AttachmentSelectors
                  │   ├─> NavigationSelectors
                  │   ├─> UrlPatterns
                  │   └─> CustomSettings
                  │
                  └─> Storage Layer
                      │
                      ├─> raw/ (HTML)
                      ├─> markdown/ (Converted content)
                      ├─> metadata/ (JSON metadata)
                      └─> attachments/ (Downloaded files)
```

## 파일 구조

```
config/
├── site_config.py              # Pydantic schemas for configuration
└── sites/
    └── news.seoul.go.kr.yaml   # Seoul Traffic News configuration

crawler/
├── extractors/
│   ├── __init__.py
│   ├── base.py                 # AbstractExtractor interface
│   └── config_based.py         # ConfigBasedExtractor implementation
├── content_extractor.py        # Original extractor (v1)
└── content_extractor_v2.py     # Configuration-driven extractor (v2)

tests/
└── test_config_based_extractor.py  # 7 passing tests
```

## 설정 파일 구조

### SiteConfig Schema

```python
class SiteConfig(BaseModel):
    # 기본 정보
    site_name: str                          # 사이트 이름
    domain: str                             # 도메인 (예: news.seoul.go.kr)
    base_url: str                           # 기본 URL

    # 셀렉터 설정
    article: ArticleSelectors               # 본문 추출 셀렉터
    attachment: AttachmentSelectors         # 첨부파일 셀렉터
    navigation: NavigationSelectors         # 네비게이션 셀렉터

    # URL 패턴
    url_patterns: UrlPatterns               # URL 분류 패턴

    # 크롤링 규칙
    crawl_rules: CrawlRules                 # 크롤링 동작 규칙

    # 사이트별 커스텀 설정
    custom: Dict[str, Any]                  # 사이트 특화 설정
```

### YAML 설정 예시 (news.seoul.go.kr.yaml)

```yaml
site_name: "서울시 교통 뉴스"
domain: "news.seoul.go.kr"
base_url: "https://news.seoul.go.kr/traffic"

article:
  # 메인 컨테이너
  main_container:
    selector: "#sub_centent"
    required: true

  # 제목
  title:
    selector: "#sub_centent h3.atitle"
    fallback_selectors:
      - "#sub_centent h3"
      - ".a_content h3"

  # 본문
  content:
    selector: "#sub_centent .a_content"
    required: true

attachment:
  # 파일 링크
  links:
    selector: ".a_content a[href*='/files/']"
    attribute: "href"
    multiple: true

  # 팝업 버튼 감지
  popup_indicators:
    - "class=preview_button"
    - "onclick=previewDocument"
    - "title=새 창열림"

# 팝업 뷰어 설정
custom:
  popup_viewer:
    text_selector: "#viewerContainer"
    extraction_method: "textContent"

  # HWP 뷰어 URL 패턴
  hwp_viewer_url_pattern: "previewDocument\\('([^']+)'\\s*,\\s*'([^']+)'\\)"
```

## 사용법

### 1. 도메인으로 Extractor 생성

```python
from pathlib import Path
from crawler.content_extractor_v2 import SeoulTrafficContentExtractorV2

# 도메인 기반 자동 설정 로드
extractor = SeoulTrafficContentExtractorV2.from_domain(
    domain="news.seoul.go.kr",
    output_dir=Path("./output"),
    download_dir=Path("./downloads"),
    headless=True,
    verbose=True
)
```

### 2. 직접 SiteConfig 전달

```python
from config.site_config import load_site_config

# 설정 로드
site_config = load_site_config("news.seoul.go.kr")

# Extractor 생성
extractor = SeoulTrafficContentExtractorV2(
    site_config=site_config,
    output_dir=Path("./output"),
    download_dir=Path("./downloads")
)
```

### 3. 콘텐츠 추출

```python
# URL 목록과 트리 구조 준비
urls = [
    {"url": "https://news.seoul.go.kr/traffic/archives/513625", "depth": 1}
]

url_tree = {
    "https://news.seoul.go.kr/traffic/archives/513625": {
        "parent_url": None,
        "depth": 1
    }
}

# 추출 실행
metadata_list = await extractor.extract_with_metadata(
    urls=urls,
    url_tree=url_tree,
    batch_size=5,
    delay_between_batches=2.0
)

# 후처리: incoming links 맵 생성
incoming_links = extractor.build_incoming_links_map(metadata_list)

# 리포트 생성
report = extractor.generate_extraction_report(metadata_list)
```

## 저장소 구조

### 디렉토리 레이아웃

```
output/
├── raw/                    # 원본 HTML
│   └── traffic_archives_513625.html
├── markdown/               # Markdown 변환본
│   └── traffic_archives_513625.md
└── metadata/               # JSON metadata
    └── traffic_archives_513625.json

downloads/
└── attachments/            # 첨부파일
    └── [attachment_id].hwpx
```

### Metadata JSON 구조

```json
{
  "url": "https://news.seoul.go.kr/traffic/archives/513625",
  "title": "저상버스 예외승인 결과 안내",
  "page_type": "article",
  "crawled_at": "2025-10-27T11:30:00Z",
  "parent_url": null,
  "depth": 1,
  "word_count": 450,
  "has_attachments": true,
  "attachment_count": 1,
  "attached_documents": [
    {
      "attachment_id": "abc123...",
      "attachment_type": "hwp",
      "original_filename": "저상버스 예외승인 결과.hwpx",
      "file_size": 51200,
      "source_type": "popup_window",
      "source_url": "https://news.seoul.go.kr/.../file.hwpx",
      "popup_window_url": "https://news.seoul.go.kr/popup/...",
      "extracted_text": "... (첨부파일 텍스트 내용) ...",
      "word_count": 850,
      "extraction_method": "playwright"
    }
  ],
  "breadcrumb": [...],
  "outgoing_links": [...],
  "entities_preview": [...]
}
```

## 캐싱 전략

### URL 기반 캐싱

```python
# URL → 파일명 변환
from urllib.parse import urlparse

parsed = urlparse(url)
filename = parsed.path.strip('/').replace('/', '_') or 'index'

# 예시:
# https://news.seoul.go.kr/traffic/archives/513625
# → traffic_archives_513625
```

### 데이터 캐싱 (참고: Dasan API 방식)

```python
# SQLite + JSON 조합
cache_db = sqlite3.connect("data/cache/scan_cache.db")

# 스캔 결과 저장
cache_db.execute("""
    INSERT INTO scan_results
    (url, scanned_at, has_content, metadata)
    VALUES (?, ?, ?, ?)
""", (url, datetime.utcnow(), True, json.dumps(metadata)))
```

## 첨부파일 처리 흐름

### 1. 팝업 버튼 감지

```python
# YAML 설정으로 팝업 버튼 감지
attachment:
  popup_indicators:
    - "class=preview_button"
    - "onclick=previewDocument"
```

### 2. 팝업 윈도우 처리

```python
async def _process_popup_button(self, page, button_info):
    # 1. 팝업 리스너 설정
    popup_future = asyncio.Future()
    page.context.on('page', lambda p: popup_future.set_result(p))

    # 2. 버튼 클릭
    await page.click(button_selector)

    # 3. 팝업 대기 (timeout: 10초)
    popup_page = await asyncio.wait_for(popup_future, timeout=10.0)

    # 4. 콘텐츠 추출
    extracted_text = await self.extract_popup_content(popup_page, popup_url)

    # 5. AttachedDocument 생성
    return AttachedDocument(...)
```

### 3. 뷰어 콘텐츠 추출

```python
async def extract_popup_content(self, popup_page, popup_url):
    # 설정에서 셀렉터 로드
    text_selector = self.config.custom['popup_viewer']['text_selector']

    # 페이지 로드 대기
    await popup_page.wait_for_load_state('networkidle')

    # 텍스트 추출
    text = await popup_page.text_content(text_selector)

    return text
```

## 테스트

### 실행

```bash
uv run pytest tests/test_config_based_extractor.py -v
```

### 테스트 커버리지

- ✅ Configuration loading and validation
- ✅ URL pattern matching (article/list/exclude)
- ✅ onclick attribute parsing
- ✅ Attachment type detection
- ✅ Basic content extraction from HTML
- ✅ Attachment link detection
- ✅ Popup content extraction

**결과**: 7/7 tests passed

## 새 사이트 추가하기

### 1. YAML 설정 파일 생성

```bash
touch config/sites/example.com.yaml
```

### 2. 설정 작성

```yaml
site_name: "Example Site"
domain: "example.com"
base_url: "https://example.com"

article:
  title:
    selector: "article h1"
  content:
    selector: "article .content"

url_patterns:
  article_patterns:
    - ".*/article/\\d+$"
```

### 3. 사용

```python
extractor = SeoulTrafficContentExtractorV2.from_domain(
    domain="example.com",
    output_dir=Path("./output"),
    download_dir=Path("./downloads")
)
```

코드 변경 없이 새 사이트 추가 완료!

## 마이그레이션 가이드 (v1 → v2)

### v1 (기존 방식)

```python
# 하드코딩된 셀렉터
title = soup.select_one("#sub_centent h3.atitle").get_text()
content = soup.select_one("#sub_centent .a_content").get_text()

# 새 사이트마다 코드 수정 필요
```

### v2 (설정 기반)

```python
# YAML에서 셀렉터 로드
title = self.extract_with_selector(soup, self.config.article.title)
content = self.extract_with_selector(soup, self.config.article.content)

# 새 사이트는 YAML만 추가
```

### API 호환성

```python
# v1 API (여전히 사용 가능)
from crawler.content_extractor import SeoulTrafficContentExtractor

# v2 API (권장)
from crawler.content_extractor_v2 import SeoulTrafficContentExtractorV2

# 동일한 인터페이스
metadata = await extractor.extract_with_metadata(urls, url_tree)
```

## 성능 최적화

### Fallback Selector 전략

```yaml
title:
  selector: "#main-title"
  fallback_selectors:
    - "h1.title"
    - "article h1"
    - "h1"
```

추출 실패 시 자동으로 fallback 시도

### 배치 처리

```python
# 병렬 처리로 성능 향상
await extractor.extract_with_metadata(
    urls=urls,
    batch_size=5,              # 동시에 5개 URL 처리
    delay_between_batches=2.0  # 배치 간 2초 대기
)
```

### 셀렉터 캐싱

```python
# SiteConfigRegistry가 설정 캐싱
registry = SiteConfigRegistry(config_dir)
config = registry.get_config("news.seoul.go.kr")  # 첫 로드 시에만 파싱
```

## 확장성

### 커스텀 Extractor 구현

```python
from crawler.extractors.base import AbstractExtractor

class CustomExtractor(AbstractExtractor):
    async def extract_from_page(self, page, html, url):
        # 사이트별 특화 로직
        pass

    async def extract_attachments(self, page, html, result):
        # 커스텀 첨부파일 처리
        pass
```

### 플러그인 아키텍처

```python
# 향후 확장 가능성
extractors = {
    'news.seoul.go.kr': ConfigBasedExtractor,
    'special-site.com': CustomExtractor,
}

extractor_class = extractors.get(domain, ConfigBasedExtractor)
extractor = extractor_class(site_config)
```

## 문제 해결

### Q: 새 사이트 셀렉터가 작동하지 않음

A: 브라우저 개발자 도구로 정확한 셀렉터 확인

```bash
# 테스트 스크립트 실행
uv run python -c "
from bs4 import BeautifulSoup
html = open('test.html').read()
soup = BeautifulSoup(html, 'html.parser')
print(soup.select('#selector'))
"
```

### Q: 팝업 윈도우가 열리지 않음

A: 팝업 감지 패턴 확인

```yaml
# 올바른 패턴
popup_indicators:
  - "class=preview_button"    # 정확한 클래스명
  - "onclick=previewDocument"  # onclick 함수명
```

### Q: 첨부파일 텍스트 추출 실패

A: 뷰어 셀렉터 확인

```yaml
custom:
  popup_viewer:
    text_selector: "#viewerContainer"  # 정확한 뷰어 컨테이너
    extraction_method: "textContent"    # or "innerText"
```

## 향후 개선 사항

### Phase 1: 안정화
- [ ] 더 많은 사이트 설정 추가
- [ ] Edge case 처리 강화
- [ ] 에러 처리 개선

### Phase 2: 기능 확장
- [ ] JavaScript 렌더링 지원 강화
- [ ] 동적 콘텐츠 대기 전략
- [ ] Rate limiting 및 retry 로직

### Phase 3: 고도화
- [ ] 자동 셀렉터 탐지 (ML 기반)
- [ ] 설정 파일 자동 생성 도구
- [ ] 웹 UI 설정 편집기

## 참고 자료

- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Playwright Python](https://playwright.dev/python/)
- [BeautifulSoup Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [YAML Specification](https://yaml.org/spec/1.2/spec.html)

## 기여자

- Design & Implementation: Claude Code + Human Collaboration
- Testing: Automated with pytest
- Documentation: This guide

---

**Last Updated**: 2025-10-27
**Version**: 2.0.0
**Status**: Production Ready ✅
