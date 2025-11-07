# Bottom-up Knowledge Documentation from Dasan Call Center Data

**Date**: 2025-11-07
**Status**: Planning Phase
**Branch**: `feat/bottom-up-kg-construction`

---

## 🎯 Goal

다산콜센터 대화 데이터를 주제별로 구조화된 지식 문서로 변환

**핵심 원칙**:
1. 멀티턴 대화 → 싱글턴 Q/A 압축 (자연스러운 멀티홉)
2. 업무 매뉴얼 형식의 주제 중심 지식 베이스 구축
3. 청킹/임베딩에 최적화된 문서 크기 유지
4. 이후 KG 빌드 또는 RAG 적용을 위한 기반 마련

**중요**: KG를 직접 추출하는 것이 아니라, 구조화된 지식 문서를 먼저 만들고 별도 프로세스로 KG/RAG 적용

---

## 📊 POC Scope

### 대상 데이터
- **출처**: `data/dasan_call/extracted/training/labeled/`
- **총 규모**: 8,176개 대화 (162,394개 턴)
- **POC 규모**: 각 카테고리에서 50개 대화씩 샘플링 (총 200개 대화)
  - 대중교통 안내: 50개 (총 2,030개 중)
  - 생활하수도 관련 문의: 50개 (총 2,038개 중)
  - 일반행정 문의: 50개 (총 2,502개 중)
  - 코로나19 관련 상담: 50개 (총 1,606개 중)

### 처리 제약
- **API**: 로컬 Claude API (localhost:25292/v1)
- **Rate Limit**: 10 calls/minute
- **Request Delay**: 6.0초
- **배치 크기**: 10-20개 대화/call
- **예상 처리 시간**: 20-30분

### 검증 목표
1. 프롬프트 품질 검증
2. 문서 구조 최적화
3. 주제 분류 체계 검증
4. 문서 크기 적정성 확인

---

## 📁 Dasan Call Center Data Structure

### File Organization
```
data/dasan_call/extracted/training/labeled/
├── 민원(콜센터) 질의응답_다산콜센터_코로나19 관련 상담_Training.json (32,475 records)
├── 민원(콜센터) 질의응답_다산콜센터_대중교통 안내_Training.json (38,966 records)
├── 민원(콜센터) 질의응답_다산콜센터_생활하수도 관련 문의_Training.json (40,617 records)
└── 민원(콜센터) 질의응답_다산콜센터_일반행정 문의_Training.json (50,336 records)
```

### JSON Schema (13 fields per record)

```json
{
  "도메인": "다산콜센터",
  "카테고리": "대중교통 안내",
  "대화셋일련번호": "B2033",
  "화자": "고객",
  "문장번호": "1",
  "고객의도": "버스노선",
  "상담사의도": "",
  "QA": "Q",
  "고객질문(요청)": "서울 가산동에서 남대문시장가는 버스노선을 알고싶습니다",
  "상담사질문(요청)": "",
  "고객답변": "",
  "상담사답변": "",
  "개체명": "서울, 가산동, 남대문시장, 버스, 노선",
  "용어사전": "서울/지명/ 가산동/동네/ 남대문시장/지명/ 버스/교통수단",
  "지식베이스": "가산동,교통수단"
}
```

### Key Metadata Fields

| Field | Description | Usage | Coverage |
|-------|-------------|-------|----------|
| **개체명** | Named entities (comma-separated) | Entity extraction | 80.8% |
| **용어사전** | Terminology with semantic types | Type classification | 76.4% |
| **지식베이스** | Knowledge base tags (2-5 concepts) | Topic categorization | 80.8% |
| **고객의도** / **상담사의도** | Speaker intent per turn | Intent tracking | ~100% |
| **대화셋일련번호** | Dialogue ID | Group turns into conversations | 100% |

**활용 방안**:
- **개체명 + 용어사전**: Front matter로 문서 구조화
- **지식베이스 태그**: 주제 분류 및 문서 병합 기준
- **의도 필드**: 대화 내 주제 전환 감지
- **대화 ID**: 멀티턴 대화 재구성

### Dialogue Structure Patterns

- **평균 대화 길이**: 19.9 턴
- **대화 턴 분포**: 4-50+ 턴
- **구조**: Q-A 쌍이 여러 턴에 걸쳐 반복
- **특징**: 한 대화 내에서 의도 전환 발생 가능 (복수 주제)

---

## 🏗️ System Architecture

### Phase 1: Data Preparation

#### 1.1 대화 샘플링 (`src/knowledge_extraction/sample_dialogues.py`)
```python
# 기능:
# - 각 카테고리 JSON에서 50개 대화 무작위 추출
# - 대화 ID 기준 그룹화 (턴 순서 보존)
# - 출력: data/processed/knowledge_extraction/sampled_dialogues.json

# 출력 형식:
{
  "대중교통_안내": [
    {
      "dialogue_id": "B2033",
      "category": "대중교통 안내",
      "turns": [
        {
          "turn_num": 1,
          "speaker": "고객",
          "qa_type": "Q",
          "text": "서울 가산동에서...",
          "intent": "버스노선",
          "entities": ["서울", "가산동", ...],
          "terminology": {"서울": "지명", "가산동": "동네", ...},
          "kb_tags": ["가산동", "교통수단"]
        },
        ...
      ]
    },
    ...
  ],
  ...
}
```

#### 1.2 대화 전처리 (`src/knowledge_extraction/dialogue_preprocessor.py`)
```python
# 기능:
# - 의도 전환 지점 감지 (optional 분할 준비)
# - 메타데이터 정규화 (공백 제거, 타입 변환)
# - Claude API 입력 형식으로 변환
```

### Phase 2: Knowledge Extraction with Claude

#### 2.1 Claude API 통합 (`src/knowledge_extraction/claude_knowledge_extractor.py`)

**설계 원칙**:
- 기존 `src/kg_agent/config/llm_config.py` 활용
- `ParallelEvaluator` 패턴 참고하여 rate limiting 구현

```python
from src.kg_agent.config.llm_config import get_llm
import asyncio

class ClaudeKnowledgeExtractor:
    def __init__(self, request_delay: float = 6.0):
        self.llm = get_llm("claude-sonnet")
        self.request_delay = request_delay

    async def extract_knowledge(self, dialogues: List[Dict]) -> List[str]:
        """
        Args:
            dialogues: 10-20개 대화 배치
        Returns:
            구조화된 Markdown 문서 리스트
        """
        # Rate limiting
        await asyncio.sleep(self.request_delay)

        # LLM 호출
        response = self.llm.llm_client.completion(...)

        return parsed_documents
```

#### 2.2 프롬프트 엔지니어링 (`src/knowledge_extraction/prompts/knowledge_extraction_prompt.txt`)

**입력**:
```
당신은 다산콜센터 상담 데이터를 구조화된 지식 문서로 변환하는 전문가입니다.

주어진 대화들을 분석하여 다음 작업을 수행하세요:

1. **주제 분류**: 각 대화의 주제를 파악하고 2-3 레벨로 분류
   - 카테고리 (대분류): 대중교통_안내, 생활하수도_관련_문의, 등
   - 세부 주제 (중/소분류): 버스_노선안내, 버스_요금_환승, 등

2. **대화 압축**: 멀티턴 대화를 핵심 Q/A 쌍으로 압축
   - 여러 턴에 걸친 질문-답변을 하나의 완전한 Q/A로 통합
   - 자연스럽게 멀티홉 정보가 포함되도록 구성

3. **주제 전환 감지**: 한 대화 내에서 주제가 바뀌면 분리
   - 의도(intent) 변화 지점 식별
   - 각 주제별로 독립적인 Q/A 생성

4. **문서 구조화**: Markdown 형식으로 출력
   - Front matter에 메타데이터 포함
   - 업무 매뉴얼 형식으로 작성
   - 최소 500 토큰 이상 유지 (청킹/임베딩 효율성)

<메타데이터 활용>
- 개체명: 문서의 주요 엔티티로 활용
- 용어사전: 엔티티 타입 정보로 활용
- 지식베이스 태그: 주제 분류 기준으로 활용
- 의도: 주제 전환 감지에 활용

<입력 대화>
{dialogues}

<출력 형식>
각 주제별로 다음 형식의 Markdown 문서를 생성하세요:

---
category: 대중교통_안내
primary_topic: 버스노선
secondary_topics: [버스요금, 환승]
entities: [가산동, 남대문시장, 505번버스]
kb_tags: [교통수단, 지역정보]
source_dialogues: [B2033, B2045]
turns_compressed: 24→3
---

# 버스노선 안내

## 특정 지역 간 버스 찾기

### Q: 가산동에서 남대문시장 가는 버스는?

**A**: 가산동 주민센터에서 남대문시장으로 가는 버스는 505번입니다.

**상세 정보**:
- 출발: 가산동 주민센터
- 도착: 남대문시장
- 노선: 505번
- 관련 문의: 버스 요금, 환승 방법

...

---
*출처: 다산콜센터 상담 사례 (대화 ID: B2033, B2045)*
```

### Phase 3: Document Organization

#### 3.1 주제별 문서 구조화 (`src/knowledge_extraction/document_organizer.py`)

```python
class DocumentOrganizer:
    def organize_by_topic(self, raw_docs: List[str]) -> Dict[str, str]:
        """
        - 2-3 레벨 taxonomy 구축
        - 같은 주제의 작은 문서들 병합
        - 최소 문서 크기 유지 (500 토큰)
        """
        pass

    def merge_small_docs(self, docs: List[str], min_size: int = 500) -> List[str]:
        """
        청킹/임베딩 효율성을 위해 작은 문서 통합
        """
        pass
```

#### 3.2 출력 디렉토리 구조

```
data/processed/knowledge_docs/
├─ 대중교통_안내/
│   ├─ 버스_노선안내.md
│   ├─ 버스_요금_환승.md
│   ├─ 지하철_이용안내.md
│   └─ 택시_정보.md
├─ 생활하수도_관련_문의/
│   ├─ 상수도_요금_납부.md
│   ├─ 하수도_시설_문의.md
│   └─ 수도_계량기_관리.md
├─ 일반행정_문의/
│   ├─ 지방세_납부.md
│   ├─ 문화행사_정보.md
│   ├─ 시설_이용안내.md
│   └─ 주민센터_서비스.md
└─ 코로나19_관련_상담/
    ├─ 지원금_신청.md
    ├─ 방역_지침.md
    └─ 격리_관련_문의.md
```

### Phase 4: Quality Validation

#### 4.1 문서 품질 검증 (`src/knowledge_extraction/quality_validator.py`)

```python
class QualityValidator:
    def validate_documents(self, docs_dir: Path) -> Dict[str, Any]:
        """
        검증 항목:
        1. 문서 크기 통계 (너무 작은 문서 식별)
        2. 주제 분류 일관성 (중복/누락 체크)
        3. 메타데이터 완성도 (필수 필드 존재 여부)
        4. Front matter 형식 검증

        Returns:
            {
                "total_docs": 45,
                "avg_size": 750,
                "small_docs": ["버스_요금.md"],
                "missing_metadata": [],
                "topic_distribution": {...}
            }
        """
        pass

    def generate_sample_report(self) -> str:
        """
        수동 검토용 샘플 문서 리포트 생성
        """
        pass
```

---

## 📝 Output Format Example

### Markdown 문서 구조

```markdown
---
category: 대중교통_안내
primary_topic: 버스노선
secondary_topics: [버스요금, 환승]
entities: [가산동, 남대문시장, 505번버스, 주민센터, 영등포구청역]
kb_tags: [교통수단, 지역정보, 요금정보]
source_dialogues: [B2033, B2045, B2067]
turns_compressed: 24→3
creation_date: 2025-11-07
---

# 버스노선 안내

## 특정 지역 간 버스 찾기

### Q: 가산동에서 남대문시장 가는 버스는?

**A**: 가산동 주민센터에서 남대문시장으로 가는 버스는 505번입니다.

**상세 정보**:
- 출발 지점: 가산동 주민센터
- 도착 지점: 남대문시장
- 버스 노선: 505번
- 운행 구간: 가산동 → 영등포구청역 → 남대문시장
- 관련 문의: 버스 요금, 환승 방법

**추가 정보**:
- 배차 간격: 평일 10-15분
- 첫차/막차: 05:30 / 23:00

### Q: 버스 요금은 얼마인가요?

**A**: 일반 시내버스 기본요금은 1,500원입니다. 교통카드 사용 시 1,400원입니다.

**상세 정보**:
- 일반 (만 19세~64세): 1,500원 (카드 1,400원)
- 청소년 (만 13세~18세): 1,200원 (카드 1,080원)
- 어린이 (만 6세~12세): 600원 (카드 540원)
- 무임 승차: 만 65세 이상, 장애인

### Q: 환승은 어떻게 하나요?

**A**: 버스-지하철 간 환승은 30분 이내에 교통카드로 태그하면 무료입니다.

**상세 정보**:
- 환승 시간: 30분 이내 (버스 하차 후 다음 교통수단 탑승까지)
- 환승 횟수: 최대 4회까지 가능
- 환승 방법: 교통카드 사용 (현금 불가)
- 주의사항: 같은 노선은 환승 불가

---

## 참고 정보

**관련 링크**:
- 서울시 교통정보: https://topis.seoul.go.kr
- 버스 노선 안내: 120 다산콜센터

**출처**: 다산콜센터 상담 사례 (대화 ID: B2033, B2045, B2067)
**최종 수정**: 2025-11-07
```

---

## ⚙️ Configuration

### POC 실험 설정 (`config/knowledge_extraction_poc.json`)

```json
{
  "experiment": {
    "id": "knowledge_extraction_poc_v1",
    "name": "Bottom-up Knowledge Extraction POC",
    "description": "Extract structured knowledge from 200 Dasan dialogues (50 per category)",
    "output_dir": "data/evaluation/knowledge_extraction_poc",
    "tags": ["poc", "knowledge-extraction", "bottom-up"]
  },

  "data": {
    "source_dir": "data/dasan_call/extracted/training/labeled",
    "categories": [
      {
        "name": "대중교통_안내",
        "file": "민원(콜센터) 질의응답_다산콜센터_대중교통 안내_Training.json",
        "sample_size": 50
      },
      {
        "name": "생활하수도_관련_문의",
        "file": "민원(콜센터) 질의응답_다산콜센터_생활하수도 관련 문의_Training.json",
        "sample_size": 50
      },
      {
        "name": "일반행정_문의",
        "file": "민원(콜센터) 질의응답_다산콜센터_일반행정 문의_Training.json",
        "sample_size": 50
      },
      {
        "name": "코로나19_관련_상담",
        "file": "민원(콜센터) 질의응답_다산콜센터_코로나19 관련 상담_Training.json",
        "sample_size": 50
      }
    ],
    "sampled_output": "data/processed/knowledge_extraction/sampled_dialogues.json",
    "final_output_dir": "data/processed/knowledge_docs"
  },

  "llm": {
    "model_name": "claude-sonnet",
    "model_id": "claude-sonnet-4-5-20250929",
    "api_base": "http://localhost:25292/v1",
    "api_key_env": "CLAUDE_CODE_API_KEY",
    "rate_limit": {
      "calls_per_minute": 10,
      "request_delay": 6.0,
      "comment": "10 calls/minute = 6 seconds between requests"
    },
    "batch_size": 15,
    "comment": "Process 15 dialogues per API call for optimal context usage"
  },

  "processing": {
    "min_doc_size": 500,
    "max_doc_size": 2000,
    "merge_small_docs": true,
    "taxonomy_levels": [2, 3],
    "split_on_intent_change": true,
    "preserve_dialogue_context": true,
    "comment": "Min 500 tokens for efficient chunking/embedding"
  },

  "validation": {
    "enabled": true,
    "check_metadata_completeness": true,
    "check_doc_size": true,
    "check_topic_consistency": true,
    "generate_sample_report": true,
    "sample_report_path": "data/evaluation/knowledge_extraction_poc/quality_report.md"
  }
}
```

---

## 🔧 Implementation Files

### File Structure
```
src/knowledge_extraction/
├── __init__.py
├── sample_dialogues.py           # Phase 1.1
├── dialogue_preprocessor.py      # Phase 1.2
├── claude_knowledge_extractor.py # Phase 2.1
├── document_organizer.py         # Phase 3.1
├── quality_validator.py          # Phase 4.1
└── prompts/
    └── knowledge_extraction_prompt.txt  # Phase 2.2

config/
└── knowledge_extraction_poc.json

data/processed/
├── knowledge_extraction/
│   └── sampled_dialogues.json
└── knowledge_docs/
    └── [주제별 Markdown 문서들]
```

### Dependencies
```python
# 기존 모듈 재사용
from src.kg_agent.config.llm_config import get_llm
from src.evaluation.parallel_evaluator import ParallelEvaluator  # Rate limiting 패턴 참고

# 새로운 의존성
import asyncio
from pathlib import Path
from typing import List, Dict, Any
import json
import yaml  # Front matter 파싱
```

---

## 📈 Success Metrics

### POC 성공 기준

| Metric | Target | Measurement |
|--------|--------|-------------|
| **문서 생성 수** | 40-60개 | 200개 대화 → 주제별 문서 개수 |
| **압축률** | 평균 19.9턴 → 1-3개 Q/A | 턴 수 / Q/A 수 비율 |
| **문서 크기** | 평균 500-1000 토큰 | 토큰 카운트 통계 |
| **주제 커버리지** | 20-30개 세부 주제 | 고유 primary_topic 개수 |
| **처리 시간** | 20-30분 이내 | 시작~종료 시간 측정 |
| **메타데이터 완성도** | 95% 이상 | 필수 front matter 필드 존재율 |

### 품질 검증 체크리스트

- [ ] 모든 문서가 최소 500 토큰 이상
- [ ] Front matter에 필수 필드 모두 존재
- [ ] 주제 분류가 일관성 있게 적용됨
- [ ] 멀티턴 대화가 자연스럽게 압축됨
- [ ] 대화 내 주제 전환이 올바르게 분리됨
- [ ] 청킹/임베딩에 적합한 문서 구조

---

## 🔄 Next Steps (POC 이후)

### 단기 (POC 완료 후)
1. **프롬프트 최적화**: 품질 검증 결과 기반 프롬프트 개선
2. **문서 구조 조정**: 문서 크기, taxonomy 레벨 최적화
3. **샘플 문서 검토**: 수동 품질 검증 및 패턴 분석

### 중기 (Full-scale 처리)
4. **전체 데이터 처리**: 8,176개 전체 대화 처리
   - 예상 시간: 13-16시간
   - 배치 처리: 82 배치 × 100개 대화
   - 예상 문서 수: 1,500-2,000개

### 장기 (Knowledge Base 활용)
5. **KG 구축**: 생성된 문서 → Neo4j Knowledge Graph
   - Entity/Relationship 추출
   - 주제별 계층 구조 반영

6. **RAG 통합**: 구조화된 문서 기반 RAG 파이프라인
   - Vector embedding 생성
   - 주제별 인덱싱

7. **성능 평가**: 기존 방법들과 비교
   - Naive RAG vs KG RAG vs Bottom-up Doc RAG
   - Golden Testset 50 questions 평가
   - Answer Correctness, Faithfulness, Relevancy 측정

---

## 🎓 Key Design Decisions

### 1. 왜 문서 먼저 만드는가?
- **문제**: 기존 KG Cypher RAG가 Naive RAG보다 성능 저하 (-6.3%)
- **원인**: Context format 오염, 도메인 스키마 부재, 그래프 탐색 부족
- **해결**: 고품질 문서 기반 → 점진적 구조화 → KG 구축

### 2. 멀티턴 → 싱글턴 압축의 이유
- **원래 문제**: LOCAL 질문 68% vs GLOBAL 질문 28%
- **Graph RAG 강점**: GLOBAL 질문 + 멀티홉 추론
- **Bottom-up 전략**: 멀티턴 대화 압축 = 자연스러운 멀티홉 정보 포함
- **기대 효과**: 복잡한 질문에 대한 답변 품질 향상

### 3. 메타데이터 활용 전략
- **개체명 (80.8%)**: 문서의 핵심 엔티티
- **용어사전 (76.4%)**: 엔티티 타입 정보
- **지식베이스 태그 (80.8%)**: 주제 분류 기준
- **의도**: 주제 전환 감지 및 문서 분할
- **활용**: Front matter로 구조화 → 추후 KG 추출 시 활용

### 4. 문서 크기 최적화
- **최소 크기**: 500 토큰 (청킹 효율성)
- **최대 크기**: 2000 토큰 (컨텍스트 윈도우)
- **병합 전략**: 작은 문서 통합하여 최적 크기 유지
- **분할 전략**: 의도 전환 시점에 분리

---

## 📚 References

### 기존 시스템 참고
- **LLM Config**: `src/kg_agent/config/llm_config.py` (Claude API 연동)
- **Rate Limiting**: `src/evaluation/parallel_evaluator.py` (10 calls/min 패턴)
- **KG Pipeline**: `src/kg_agent/kg_construction.py` (Entity/Relationship 추출)

### 관련 문서
- **데이터 분석**: `notebooks/dasan_call_center_eda.ipynb`
- **평가 결과**: `data/evaluation/3way_rag_comparison/`
- **KG Cypher Fix**: `docs/CHECKPOINT_kg_cypher_fix.md`

---

**작성자**: Claude (with human guidance)
**검토 필요**: 프롬프트 템플릿, 문서 구조, taxonomy 설계
