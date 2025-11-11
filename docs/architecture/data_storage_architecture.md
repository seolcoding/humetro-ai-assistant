# Data Storage Architecture

**Version**: 1.0.0
**Last Updated**: 2025-10-27
**Status**: Production Ready ✅

## 개요

Humetro AI Assistant 프로젝트의 데이터 저장 및 캐싱 아키텍처를 정의합니다. 크롤링된 데이터, API 응답, 메타데이터, 첨부파일의 구조화된 관리 방법을 포함합니다.

## 디렉토리 구조

### 전체 레이아웃

```
data/
├── raw/                          # 원본 데이터 (수정 금지)
│   ├── humetro/                 # Humetro 관련 원본 데이터
│   ├── generated_qa/            # 생성된 QA 데이터
│   └── raw/                     # 기타 원본 데이터
│
├── dasan_api_cache/             # Dasan Call Center API 캐시
│   ├── scan_cache.db           # SQLite 캐시 데이터베이스
│   ├── discovery_faq.json      # FAQ 검색 결과
│   ├── narrow_discovery_faq.json
│   ├── faq_detail_*.json       # FAQ 상세 정보
│   ├── workmanual_detail_*.json # 업무 매뉴얼 상세
│   ├── faq_sequences_*.json    # FAQ 시퀀스 범위
│   ├── workmanual_test_sequences.json
│   └── README.md               # 캐시 사용 가이드
│
├── AI_HUB_DASAN_QA/                  # Dasan Call Center 데이터셋
│   ├── raw/                    # 원본 ZIP/JSON 파일
│   ├── extracted/              # 압축 해제된 파일
│   ├── splitted/               # Train/Valid 분리
│   └── generated/              # 전처리된 데이터
│
├── processed/                   # 전처리 및 분석 결과
│   ├── dasan_eda/             # EDA 분석 결과
│   ├── metadata_analysis/      # 메타데이터 분석
│   ├── evaluation/             # 평가 결과
│   └── knowledge_base/         # 지식베이스 빌드
│
├── knowledge_graphs/            # 지식 그래프 데이터
│   ├── gpt5_generated/        # GPT-5 생성 그래프
│   └── opensource_kg/         # 오픈소스 KG 데이터
│
└── vectorstore/                 # 벡터 저장소
    └── [uuid]/                 # 벡터 인덱스 디렉토리
```

## 크롤러 데이터 저장 구조

### Seoul Traffic News (news.seoul.go.kr)

ContentExtractorV2에서 사용하는 표준 저장 구조:

```
output/                          # 설정 가능한 출력 디렉토리
├── raw/                        # 원본 HTML
│   └── traffic_archives_513625.html
├── markdown/                   # Markdown 변환본
│   └── traffic_archives_513625.md
└── metadata/                   # JSON 메타데이터
    └── traffic_archives_513625.json

downloads/                       # 설정 가능한 다운로드 디렉토리
└── attachments/                # 첨부파일
    └── abc123def456.hwpx      # [attachment_id].[ext]
```

### URL → 파일명 변환 규칙

```python
from urllib.parse import urlparse

def url_to_filename(url: str) -> str:
    """
    URL을 파일 시스템 안전 파일명으로 변환

    Examples:
        https://news.seoul.go.kr/traffic/archives/513625
        → traffic_archives_513625

        https://news.seoul.go.kr/traffic/archives/category/public
        → traffic_archives_category_public

        https://news.seoul.go.kr/traffic/
        → index
    """
    parsed = urlparse(url)
    filename = parsed.path.strip('/').replace('/', '_') or 'index'
    return filename
```

### 파일 저장 경로 생성

```python
from pathlib import Path

# URL 기반 파일명 생성
base_name = url_to_filename(url)

# 각 형식별 저장 경로
raw_html_path = output_dir / "raw" / f"{base_name}.html"
markdown_path = output_dir / "markdown" / f"{base_name}.md"
metadata_path = output_dir / "metadata" / f"{base_name}.json"

# 첨부파일 경로 (UUID 기반)
import hashlib
attachment_id = hashlib.sha256(attachment_url.encode()).hexdigest()[:16]
attachment_path = download_dir / "attachments" / f"{attachment_id}.{ext}"
```

## 메타데이터 스키마

### PageMetadata 구조 (src/config/schemas.py)

```json
{
  "url": "https://news.seoul.go.kr/traffic/archives/513625",
  "title": "저상버스 예외승인 결과 안내",
  "page_type": "article",
  "crawled_at": "2025-10-27T11:30:00Z",

  "parent_url": "https://news.seoul.go.kr/traffic/archives/category/public",
  "depth": 2,

  "breadcrumb": [
    {
      "url": "https://news.seoul.go.kr/traffic",
      "title": "서울시 교통뉴스",
      "depth": 0,
      "page_type": "root"
    },
    {
      "url": "https://news.seoul.go.kr/traffic/archives/category/public",
      "title": "공지사항",
      "depth": 1,
      "page_type": "category"
    }
  ],

  "siblings": [
    "https://news.seoul.go.kr/traffic/archives/513624",
    "https://news.seoul.go.kr/traffic/archives/513626"
  ],

  "outgoing_links": [
    "https://news.seoul.go.kr/traffic/files/2025/01/approval.hwpx",
    "https://news.seoul.go.kr/traffic/archives/513500"
  ],

  "incoming_links": [
    "https://news.seoul.go.kr/traffic/archives/category/public"
  ],

  "link_contexts": [
    {
      "target_url": "https://news.seoul.go.kr/traffic/archives/513500",
      "anchor_text": "저상버스 운행 기준",
      "surrounding_text": "관련 공지사항: 저상버스 운행 기준을 참고하시기 바랍니다.",
      "parent_element": "div.a_content p",
      "link_position": 1,
      "is_navigation": false
    }
  ],

  "word_count": 450,

  "entities_preview": [
    "저상버스",
    "서울시",
    "교통국",
    "2025년 1월"
  ],

  "named_entities": {
    "ORG": ["서울시", "교통국"],
    "DATE": ["2025년 1월"],
    "PRODUCT": ["저상버스"]
  },

  "attached_documents": [
    {
      "attachment_id": "abc123def456",
      "attachment_type": "hwp",
      "original_filename": "저상버스 예외승인 결과.hwpx",
      "file_size": 51200,

      "source_type": "popup_window",
      "source_url": "https://news.seoul.go.kr/traffic/files/2025/01/approval.hwpx",
      "trigger_element": "button.preview_button",
      "trigger_context": "onclick=\"previewDocument('...','approval.hwpx')\"",
      "popup_window_title": "문서 뷰어",
      "popup_window_url": "https://news.seoul.go.kr/popup/viewer?file=...",

      "extracted_text": "저상버스 예외승인 결과 안내...",
      "extracted_tables": [],
      "extracted_images": [],
      "page_count": null,
      "sheet_names": null,

      "local_path": "downloads/attachments/abc123def456.hwpx",
      "markdown_path": "output/markdown/traffic_archives_513625_attachment_0.md",

      "extraction_method": "playwright",
      "extraction_success": true,
      "word_count": 850,

      "entities_preview": ["저상버스", "예외승인", "교통약자"],
      "named_entities": {
        "ORG": ["서울시"],
        "PRODUCT": ["저상버스"]
      }
    }
  ],

  "has_attachments": true,
  "attachment_types": ["hwp"],
  "total_attachment_size": 51200,
  "attachment_count": 1
}
```

### AttachedDocument 스키마

```python
class AttachedDocument(BaseModel):
    """첨부 문서 메타데이터"""

    # 식별 정보
    attachment_id: str                      # SHA256 해시 기반 고유 ID
    attachment_type: AttachmentType         # pdf, hwp, xlsx, docx, etc.
    original_filename: str                  # 원본 파일명
    file_size: int                          # 바이트 단위

    # 소스 정보
    source_type: AttachmentSource          # popup_window, direct_link, iframe, javascript
    source_url: str                        # 원본 URL
    trigger_element: Optional[str]         # CSS 셀렉터 (버튼/링크)
    trigger_context: Optional[str]         # HTML 컨텍스트
    popup_window_title: Optional[str]      # 팝업 제목
    popup_window_url: Optional[str]        # 팝업 URL

    # 추출된 콘텐츠
    extracted_text: str                    # 전체 텍스트
    extracted_tables: List[Dict]           # 테이블 데이터
    extracted_images: List[str]            # 이미지 경로
    page_count: Optional[int]              # PDF 페이지 수
    sheet_names: Optional[List[str]]       # Excel 시트 이름

    # 저장 경로
    local_path: str                        # 로컬 파일 경로
    markdown_path: Optional[str]           # Markdown 변환본 경로

    # 추출 메타데이터
    extraction_method: Literal[...]        # pypdf, playwright, openpyxl, etc.
    extraction_success: bool               # 추출 성공 여부
    word_count: int                        # 단어 수

    # 엔티티 정보
    entities_preview: List[str]            # 주요 엔티티 (최대 10개)
    named_entities: Dict[str, List[str]]   # 카테고리별 엔티티
```

## 캐싱 전략

### 1. Dasan API 캐싱 (SQLite + JSON)

**목적**: API 호출 최소화, 오프라인 개발 지원, 빠른 데이터 접근

#### SQLite 데이터베이스 구조

```sql
-- data/dasan_api_cache/scan_cache.db

CREATE TABLE scan_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    url TEXT UNIQUE NOT NULL,
    scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    has_content BOOLEAN,
    metadata TEXT,  -- JSON 형식
    INDEX idx_url (url),
    INDEX idx_scanned_at (scanned_at)
);

CREATE TABLE api_responses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    endpoint TEXT NOT NULL,
    params TEXT,  -- JSON 형식
    response TEXT,  -- JSON 형식
    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ttl INTEGER DEFAULT 3600,  -- seconds
    INDEX idx_endpoint (endpoint),
    INDEX idx_cached_at (cached_at)
);
```

#### 캐시 사용 패턴

```python
import sqlite3
import json
from datetime import datetime, timedelta

class DasanAPICache:
    """Dasan API 응답 캐싱"""

    def __init__(self, cache_db_path: str):
        self.conn = sqlite3.connect(cache_db_path)
        self._init_db()

    def get_cached_response(self, endpoint: str, params: dict) -> Optional[dict]:
        """캐시에서 응답 조회 (TTL 체크)"""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT response, cached_at, ttl
            FROM api_responses
            WHERE endpoint = ? AND params = ?
        """, (endpoint, json.dumps(params, sort_keys=True)))

        row = cursor.fetchone()
        if not row:
            return None

        response, cached_at, ttl = row
        cached_time = datetime.fromisoformat(cached_at)

        if datetime.now() - cached_time > timedelta(seconds=ttl):
            return None  # Expired

        return json.loads(response)

    def cache_response(
        self,
        endpoint: str,
        params: dict,
        response: dict,
        ttl: int = 3600
    ):
        """응답 캐싱"""
        self.conn.execute("""
            INSERT OR REPLACE INTO api_responses
            (endpoint, params, response, cached_at, ttl)
            VALUES (?, ?, ?, ?, ?)
        """, (
            endpoint,
            json.dumps(params, sort_keys=True),
            json.dumps(response),
            datetime.now().isoformat(),
            ttl
        ))
        self.conn.commit()
```

#### JSON 파일 캐시

개별 API 응답은 JSON 파일로도 저장:

```
data/dasan_api_cache/
├── discovery_faq.json           # FAQ 목록 검색 결과
├── faq_detail_289801.json       # FAQ ID 289801 상세 정보
├── faq_sequences_289700_289800.json  # 범위 스캔 결과
├── workmanual_detail_289756.json
└── ...
```

**장점**:
- 쉬운 디버깅 및 데이터 검증
- Git을 통한 버전 관리 가능
- 직접적인 파일 접근 가능

### 2. URL 기반 파일 캐싱

**목적**: 중복 크롤링 방지, 빠른 재처리

```python
from pathlib import Path
import json

def is_url_cached(url: str, output_dir: Path) -> bool:
    """URL이 이미 캐시되었는지 확인"""
    filename = url_to_filename(url)
    metadata_path = output_dir / "metadata" / f"{filename}.json"

    if not metadata_path.exists():
        return False

    # 메타데이터에서 크롤 시간 확인
    with open(metadata_path, 'r', encoding='utf-8') as f:
        metadata = json.load(f)

    crawled_at = datetime.fromisoformat(metadata['crawled_at'])
    age = datetime.utcnow() - crawled_at

    # 7일 이내면 캐시 유효
    return age.days < 7

def get_cached_metadata(url: str, output_dir: Path) -> Optional[dict]:
    """캐시된 메타데이터 로드"""
    filename = url_to_filename(url)
    metadata_path = output_dir / "metadata" / f"{filename}.json"

    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None
```

### 3. 첨부파일 캐싱

**목적**: 동일 첨부파일 중복 다운로드 방지

```python
import hashlib

def get_attachment_cache_path(
    attachment_url: str,
    download_dir: Path
) -> tuple[str, Path]:
    """첨부파일 캐시 경로 생성"""
    # URL 해싱으로 고유 ID 생성
    attachment_id = hashlib.sha256(attachment_url.encode()).hexdigest()[:16]

    # 확장자 추출
    ext = attachment_url.split('.')[-1].lower()

    # 캐시 경로
    cache_path = download_dir / "attachments" / f"{attachment_id}.{ext}"

    return attachment_id, cache_path

def is_attachment_cached(attachment_url: str, download_dir: Path) -> bool:
    """첨부파일이 캐시되었는지 확인"""
    _, cache_path = get_attachment_cache_path(attachment_url, download_dir)
    return cache_path.exists()
```

## 데이터 접근 패턴

### 1. 크롤링 워크플로우

```python
from src.crawler.content_extractor_v2 import SeoulTrafficContentExtractorV2
from src.common.project_paths import get_project_root

# 프로젝트 루트에서 경로 설정
project_root = get_project_root()
output_dir = project_root / "data" / "crawled" / "seoul_traffic"
download_dir = project_root / "data" / "downloads"

# Extractor 생성
extractor = SeoulTrafficContentExtractorV2.from_domain(
    domain="news.seoul.go.kr",
    output_dir=output_dir,
    download_dir=download_dir,
    headless=True,
    verbose=True
)

# 크롤링 실행
urls = [
    {"url": "https://news.seoul.go.kr/traffic/archives/513625", "depth": 1}
]

url_tree = {
    "https://news.seoul.go.kr/traffic/archives/513625": {
        "parent_url": None,
        "depth": 1
    }
}

metadata_list = await extractor.extract_with_metadata(
    urls=urls,
    url_tree=url_tree,
    batch_size=5,
    delay_between_batches=2.0
)
```

### 2. 메타데이터 읽기

```python
import json
from pathlib import Path
from src.common.project_paths import get_project_root

def load_page_metadata(url: str) -> dict:
    """URL에 해당하는 메타데이터 로드"""
    project_root = get_project_root()
    output_dir = project_root / "data" / "crawled" / "seoul_traffic"

    filename = url_to_filename(url)
    metadata_path = output_dir / "metadata" / f"{filename}.json"

    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_all_metadata(output_dir: Path) -> list[dict]:
    """모든 메타데이터 로드"""
    metadata_dir = output_dir / "metadata"
    metadata_list = []

    for json_file in metadata_dir.glob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            metadata_list.append(json.load(f))

    return metadata_list
```

### 3. 첨부파일 접근

```python
def get_attachment_content(
    attachment_id: str,
    download_dir: Path
) -> Optional[bytes]:
    """첨부파일 바이너리 읽기"""
    for ext in ['hwpx', 'pdf', 'xlsx', 'docx']:
        path = download_dir / "attachments" / f"{attachment_id}.{ext}"
        if path.exists():
            return path.read_bytes()
    return None

def get_attachment_text(
    attachment_id: str,
    output_dir: Path
) -> Optional[str]:
    """첨부파일 추출된 텍스트 읽기 (메타데이터에서)"""
    metadata_list = load_all_metadata(output_dir)

    for metadata in metadata_list:
        for doc in metadata.get('attached_documents', []):
            if doc['attachment_id'] == attachment_id:
                return doc['extracted_text']

    return None
```

## 데이터 정리 및 유지보수

### 캐시 정리

```python
from datetime import datetime, timedelta
from pathlib import Path

def cleanup_old_cache(
    output_dir: Path,
    max_age_days: int = 30
):
    """오래된 캐시 파일 정리"""
    cutoff_date = datetime.now() - timedelta(days=max_age_days)

    metadata_dir = output_dir / "metadata"
    for json_file in metadata_dir.glob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        crawled_at = datetime.fromisoformat(metadata['crawled_at'])

        if crawled_at < cutoff_date:
            # 메타데이터 파일 삭제
            json_file.unlink()

            # 관련 파일들도 삭제
            filename = json_file.stem
            (output_dir / "raw" / f"{filename}.html").unlink(missing_ok=True)
            (output_dir / "markdown" / f"{filename}.md").unlink(missing_ok=True)

def cleanup_orphaned_attachments(
    output_dir: Path,
    download_dir: Path
):
    """메타데이터에 참조되지 않는 첨부파일 정리"""
    # 모든 메타데이터에서 첨부파일 ID 수집
    metadata_list = load_all_metadata(output_dir)
    referenced_ids = set()

    for metadata in metadata_list:
        for doc in metadata.get('attached_documents', []):
            referenced_ids.add(doc['attachment_id'])

    # 첨부파일 디렉토리 스캔
    attachments_dir = download_dir / "attachments"
    for file in attachments_dir.iterdir():
        attachment_id = file.stem  # 파일명에서 ID 추출

        if attachment_id not in referenced_ids:
            file.unlink()  # 고아 파일 삭제
```

### 데이터 무결성 검증

```python
def verify_data_integrity(output_dir: Path, download_dir: Path) -> dict:
    """데이터 무결성 검증 리포트"""
    report = {
        'total_pages': 0,
        'missing_html': [],
        'missing_markdown': [],
        'missing_attachments': [],
        'corrupted_metadata': []
    }

    metadata_dir = output_dir / "metadata"

    for json_file in metadata_dir.glob("*.json"):
        report['total_pages'] += 1
        filename = json_file.stem

        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        except json.JSONDecodeError:
            report['corrupted_metadata'].append(str(json_file))
            continue

        # HTML 파일 확인
        html_path = output_dir / "raw" / f"{filename}.html"
        if not html_path.exists():
            report['missing_html'].append(metadata['url'])

        # Markdown 파일 확인
        md_path = output_dir / "markdown" / f"{filename}.md"
        if not md_path.exists():
            report['missing_markdown'].append(metadata['url'])

        # 첨부파일 확인
        for doc in metadata.get('attached_documents', []):
            attachment_path = Path(doc['local_path'])
            if not attachment_path.exists():
                report['missing_attachments'].append({
                    'page_url': metadata['url'],
                    'attachment_id': doc['attachment_id'],
                    'filename': doc['original_filename']
                })

    return report
```

## 성능 최적화

### 배치 로딩

```python
from concurrent.futures import ThreadPoolExecutor

def load_metadata_batch(
    urls: list[str],
    output_dir: Path,
    max_workers: int = 5
) -> list[dict]:
    """병렬로 메타데이터 로드"""
    def load_single(url: str) -> dict:
        filename = url_to_filename(url)
        metadata_path = output_dir / "metadata" / f"{filename}.json"

        with open(metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(load_single, urls))
```

### 인덱싱

```python
import sqlite3
from pathlib import Path

def build_metadata_index(output_dir: Path) -> str:
    """메타데이터 검색용 SQLite 인덱스 생성"""
    index_path = output_dir / "metadata_index.db"
    conn = sqlite3.connect(index_path)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS pages (
            url TEXT PRIMARY KEY,
            title TEXT,
            page_type TEXT,
            crawled_at TIMESTAMP,
            word_count INTEGER,
            has_attachments BOOLEAN,
            depth INTEGER,
            metadata_path TEXT
        )
    """)

    conn.execute("CREATE INDEX IF NOT EXISTS idx_page_type ON pages(page_type)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_crawled_at ON pages(crawled_at)")

    # 메타데이터 파일 스캔하여 인덱스 구축
    metadata_dir = output_dir / "metadata"
    for json_file in metadata_dir.glob("*.json"):
        with open(json_file, 'r', encoding='utf-8') as f:
            metadata = json.load(f)

        conn.execute("""
            INSERT OR REPLACE INTO pages VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            metadata['url'],
            metadata['title'],
            metadata['page_type'],
            metadata['crawled_at'],
            metadata['word_count'],
            metadata['has_attachments'],
            metadata['depth'],
            str(json_file)
        ))

    conn.commit()
    conn.close()

    return str(index_path)

def search_pages(
    output_dir: Path,
    page_type: Optional[str] = None,
    min_word_count: int = 0,
    has_attachments: Optional[bool] = None
) -> list[dict]:
    """인덱스를 사용한 빠른 검색"""
    index_path = output_dir / "metadata_index.db"
    conn = sqlite3.connect(index_path)

    query = "SELECT * FROM pages WHERE 1=1"
    params = []

    if page_type:
        query += " AND page_type = ?"
        params.append(page_type)

    if min_word_count > 0:
        query += " AND word_count >= ?"
        params.append(min_word_count)

    if has_attachments is not None:
        query += " AND has_attachments = ?"
        params.append(has_attachments)

    cursor = conn.execute(query, params)
    results = []

    for row in cursor.fetchall():
        metadata_path = row[7]
        with open(metadata_path, 'r', encoding='utf-8') as f:
            results.append(json.load(f))

    conn.close()
    return results
```

## 모범 사례

### 경로 관리

✅ **권장**: 항상 `src.common.project_paths` 사용

```python
from src.common.project_paths import get_project_root, get_data_dir

project_root = get_project_root()
data_dir = get_data_dir()
output_dir = data_dir / "crawled" / "seoul_traffic"
```

❌ **비권장**: 하드코딩된 경로

```python
output_dir = Path("/Users/sdh/Dev/humetro-ai-assistant/data/crawled")
```

### 파일 네이밍

✅ **권장**: URL 기반 일관된 네이밍

```python
filename = url_to_filename(url)  # traffic_archives_513625
```

❌ **비권장**: 타임스탬프나 랜덤 ID

```python
filename = f"page_{timestamp}_{random_id}"  # 추적 어려움
```

### 메타데이터 저장

✅ **권장**: 구조화된 JSON with ISO 8601 날짜

```python
metadata = {
    "url": url,
    "crawled_at": datetime.utcnow().isoformat(),
    "title": title,
    ...
}
```

❌ **비권장**: 비표준 형식

```python
metadata = {
    "url": url,
    "time": "2025-10-27 11:30",  # 타임존 없음
    ...
}
```

## 문제 해결

### Q: 메타데이터 파일이 손상되었습니다

A: 무결성 검증 실행

```bash
uv run python -c "
from src.common.project_paths import get_data_dir
from scripts.verify_data import verify_data_integrity

data_dir = get_data_dir()
output_dir = data_dir / 'crawled' / 'seoul_traffic'
download_dir = data_dir / 'downloads'

report = verify_data_integrity(output_dir, download_dir)
print(report)
"
```

### Q: 디스크 공간이 부족합니다

A: 오래된 캐시 정리

```bash
uv run python -c "
from src.common.project_paths import get_data_dir
from scripts.cleanup_cache import cleanup_old_cache

data_dir = get_data_dir()
output_dir = data_dir / 'crawled' / 'seoul_traffic'

cleanup_old_cache(output_dir, max_age_days=30)
"
```

### Q: 첨부파일을 찾을 수 없습니다

A: 메타데이터에서 경로 확인

```python
metadata = load_page_metadata(url)
for doc in metadata['attached_documents']:
    print(f"Attachment: {doc['original_filename']}")
    print(f"Path: {doc['local_path']}")
    print(f"Exists: {Path(doc['local_path']).exists()}")
```

## 향후 개선 사항

### Phase 1: 인덱싱 자동화
- [ ] 크롤링 시 자동 인덱스 업데이트
- [ ] 전문 검색 (Full-text search) 지원
- [ ] 첨부파일 내용 인덱싱

### Phase 2: 캐시 관리
- [ ] 자동 캐시 만료 및 정리
- [ ] LRU 캐시 전략 구현
- [ ] 압축 저장 옵션

### Phase 3: 분산 저장
- [ ] S3/Object Storage 지원
- [ ] 데이터베이스 백엔드 옵션
- [ ] 원격 캐시 동기화

## 참고 자료

- [Pydantic Models](src/config/schemas.py)
- [ContentExtractorV2](src/crawler/content_extractor_v2.py)
- [Project Paths Utility](src/common/project_paths.py)
- [Configuration-Driven Extraction](./config_driven_extraction.md)

---

**Last Updated**: 2025-10-27
**Version**: 1.0.0
**Status**: Production Ready ✅
