# Dasan Call 지식베이스 추출 워크플로우

다산콜센터 상담 데이터에서 RAG 시스템용 지식베이스를 추출하는 전체 워크플로우 문서입니다.

## 📊 처리 결과 요약

**최종 산출물** (2025-11-09 기준):
- **원본 데이터**: 9,187개 Gemini Batch API 예측 결과
- **추출된 문서**: 9,632개 지식 문서
- **검증 통과율**: 93.6% (9,011 / 9,632)
- **평균 문서 길이**: 1,618자, 404 토큰, ~0.8 청크
- **총 파일 크기**: 44MB (JSONL), 144KB (메타데이터)

**카테고리 분포**:
- 일반행정_문의: 2,918 (30.3%)
- 대중교통_안내: 2,369 (24.6%)
- 생활하수도_관련_문의: 2,203 (22.9%)
- 코로나19_관련_상담: 1,898 (19.7%)

---

## 🔄 전체 워크플로우

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. 원본 데이터 (AI Hub)                                         │
│    - 다산콜센터 상담 대화 데이터                                 │
│    - 멀티턴 대화, 메타데이터 포함                                │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. Gemini Batch API 처리                                        │
│    - 멀티턴 대화 → 싱글턴 Q/A 재구성                             │
│    - 주제 분류 및 Markdown 문서 생성                             │
│    - Output: prediction-model-*_predictions.jsonl (172MB)       │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. 지식 문서 추출 (process_predictions.py)                      │
│    - Gemini 응답에서 지식 문서 파싱                              │
│    - RAG 메타데이터 추가 (토큰 수, 청크 수 등)                   │
│    - 문서 검증 (필수 필드, 구조, 길이)                           │
│    - Output: knowledge_docs_full.jsonl (44MB)                   │
└──────────────────┬──────────────────────────────────────────────┘
                   │
                   ├──────────────────────────────────────────────┐
                   │                                              │
                   ▼                                              ▼
┌──────────────────────────────────────┐  ┌──────────────────────────────────┐
│ 4a. Markdown 조직화                  │  │ 4b. RAG 파이프라인 투입          │
│     (organize_markdown.py)           │  │     (DasanKnowledgeLoader)       │
│  - 주제별 디렉터리 구조 생성         │  │  - JSONL → LangChain Documents   │
│  - 1,132개 디렉터리                  │  │  - Stage 2: Chunking 연결        │
│  - 9,632개 .md 파일                  │  │  - Stage 3: Embedding 생성       │
│  - 리뷰 및 콘텐츠 관리용             │  │  - Stage 5: Vector Store 구축    │
└──────────────────────────────────────┘  └──────────────────────────────────┘
```

---

## 📝 단계별 가이드

### Step 1: 원본 Predictions JSONL 준비

Gemini Batch API에서 생성된 predictions 파일이 있는지 확인:

```bash
ls -lh data/AI_HUB_DASAN_QA/06_predictions/prediction-model-*_predictions.jsonl
```

파일 구조:
- 각 라인: JSON 객체 (request, response, status, processed_time)
- response.candidates[0].content.parts[0].text: 생성된 지식 문서 (JSON 배열)

### Step 2: 지식 문서 추출

`process_predictions.py` 스크립트로 전체 predictions 처리:

```bash
uv run python src/knowledge_extraction/process_predictions.py \
    --input data/AI_HUB_DASAN_QA/06_predictions/prediction-model-2025-11-08T14_26_42.348286Z_predictions.jsonl \
    --output data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs_full.jsonl
```

**옵션**:
- `--no-validate`: 문서 검증 생략 (빠른 처리)

**출력 파일**:
- `knowledge_docs_full.jsonl`: 추출된 지식 문서 (JSONL 형식)
- `knowledge_docs_full_metadata.json`: 처리 통계 및 검증 결과

**처리 통계 예시**:
```
Total lines processed: 9,187
Total documents extracted: 9,632
Validation passed: 9,011 (93.6%)
Parse errors: 7
Average document: 1,618 chars, 404 tokens
```

### Step 3: Markdown 조직화 (선택사항)

주제별 디렉터리 구조로 Markdown 파일 생성:

```bash
uv run python src/knowledge_extraction/organize_markdown.py \
    --input data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs_full.jsonl \
    --output-dir data/AI_HUB_DASAN_QA/03_markdown_full
```

**생성되는 구조**:
```
organized_markdown_full/
├── README.md                           # 전체 요약
├── 대중교통_안내/
│   ├── 버스/
│   │   ├── index.md                   # 버스 관련 문서 목록
│   │   ├── 버스_노선안내_B12345.md
│   │   └── ...
│   └── 지하철/
│       └── ...
├── 일반행정_문의/
│   └── ...
└── ...
```

**용도**:
- 사람이 리뷰하기 쉬운 구조
- 문서 내용 검증 및 수정
- Git으로 버전 관리

### Step 4: RAG 파이프라인 투입

`DasanKnowledgeLoader`로 JSONL을 LangChain Documents로 로드:

```python
from pathlib import Path
from src.rag_pipeline.stages.stage_01_dasan_loader import DasanKnowledgeLoader

# JSONL 로드
loader = DasanKnowledgeLoader(
    jsonl_path=Path("data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs_full.jsonl"),
    validated_only=True  # 검증된 문서만 로드
)
documents = loader.load_documents()

print(f"Loaded {len(documents):,} documents")
# Output: Loaded 9,011 documents (validated only)
```

**LangChain Document 메타데이터**:
```python
{
    "source": "AI_HUB_DASAN_QA_B12345",
    "dialogue_id": "B12345",
    "topic_path": "대중교통_안내/버스/노선안내",
    "primary_topic": "버스_노선안내",
    "secondary_topics": ["지역간_버스_검색", "버스_요금"],
    "category": "대중교통_안내",
    "domain": "버스",
    "doc_length": 1500,
    "estimated_chunks": 1,
    "validated": true,
    "original_question": "...",  # 평가용
    "original_answer": "...",    # 평가용
    "extraction_model": "gemini-2.5-pro",
    "extraction_date": "2025-11-09"
}
```

**기존 RAG 파이프라인 연결**:
```python
# Stage 1: Data Collection
documents = loader.load_documents()

# Stage 2: Chunking (기존 코드 그대로 사용)
from src.rag_pipeline.stages.stage_02_chunking import ChunkingStage
chunker = ChunkingStage(chunk_size=512, chunk_overlap=64)
chunks = chunker.process(documents)

# Stage 3: Embedding (기존 코드 그대로 사용)
from src.rag_pipeline.stages.stage_03_embedding import EmbeddingStage
embedder = EmbeddingStage(model_name="text-embedding-3-large")
embedded_chunks = embedder.process(chunks)

# Stage 5: Vector Store (기존 코드 그대로 사용)
from src.rag_pipeline.stages.stage_05_vector_store import VectorStoreStage
vector_store = VectorStoreStage(store_type="faiss")
vector_store.process(embedded_chunks)
```

---

## 📦 데이터 구조

### JSONL 문서 구조

각 라인은 하나의 지식 문서:

```json
{
  "dialogue_id": "B24034",
  "original_question": "저신용자인 개인이 받을 수 있는 대출 지원이 있나요? ...",
  "original_answer": "문의하신 1천만원 긴급 대출은 저신용 소상공인의 경영 안정을 위한...",
  "topic_path": "코로나19_관련_상담/금융지원/소상공인_대출",
  "primary_topic": "소상공인_대출",
  "secondary_topics": ["금융지원", "저신용자_지원", "대출_자격조건"],
  "document": "---\ncategory: 코로나19_관련_상담\n...(full markdown)...",
  "metadata": {
    "doc_length_chars": 1105,
    "doc_length_tokens": 276,
    "estimated_chunks": 1,
    "has_front_matter": true,
    "has_structured_qa": true,
    "entities_count": 7,
    "kb_tags_count": 6,
    "extraction_model": "gemini-2.5-pro",
    "extraction_date": "2025-11-09",
    "category": "코로나19_관련_상담",
    "domain": "금융지원",
    "validated": true,
    "validation_issues": []
  }
}
```

### Markdown 문서 구조

```markdown
<!-- Generated by organize_markdown.py -->
<!-- Dialogue ID: B24034 -->
<!-- Extraction Date: 2025-11-09 -->
<!-- Validation: ✓ Passed -->

---
category: 코로나19_관련_상담
primary_topic: 소상공인_대출
secondary_topics: [금융지원, 저신용자_지원, 대출_자격조건]
entities: [저신용자, 긴급 대출, 소상공인, ...]
kb_tags: [금융지원, 대출, 소상공인, 저신용, ...]
source_dialogues: [B24034]
turns_compressed: 19→2
creation_date: 2025-11-09
---

# 소상공인 저신용자 긴급 대출 안내

코로나19 등으로 경영에 어려움을 겪는 저신용 소상공인을 위한...

## 저신용 소상공인 1천만원 긴급 대출

### Q: 저신용 소상공인을 위한 긴급 대출의 자격 조건은?

**A**: 저신용 소상공인 긴급 대출은 사업체를 운영하는...

**1. 지원 대상**
- 사업체를 운영하는 **소상공인**
- ...

---

*출처: 다산콜센터 상담 사례 (대화 ID: B24034)*
*최종 수정: 2025-11-09*
```

---

## 🔍 문서 검증 기준

`process_predictions.py`의 자동 검증 항목:

### 필수 필드 검증
- `dialogue_id`: 원본 대화 ID
- `original_question`: 재구성된 질문
- `original_answer`: 재구성된 답변
- `topic_path`: 주제 경로 (3단계 계층)
- `primary_topic`: 주요 주제
- `secondary_topics`: 보조 주제 리스트
- `document`: Markdown 문서

### 문서 품질 검증
- **최소 길이**: 500자 이상
- **Front matter**: `---`로 시작하는 YAML 헤더
- **구조화된 Q/A**: `### Q:` 및 `**A**:` 포맷
- **주제 정보**: primary_topic 및 secondary_topics 존재

### 검증 실패 시
- `metadata.validated = false`
- `metadata.validation_issues = ["no_qa_format", ...]`
- 문서는 여전히 JSONL에 포함 (선택적 필터링 가능)

---

## 🛠 스크립트 참조

### 1. process_predictions.py

원본 predictions JSONL → 지식베이스 JSONL 추출

**위치**: `src/knowledge_extraction/process_predictions.py`

**주요 기능**:
- Gemini Batch API 응답 파싱
- RAG 메타데이터 계산 (토큰 수, 청크 수)
- 문서 품질 검증
- 통계 및 검증 결과 저장

**사용 예**:
```bash
uv run python src/knowledge_extraction/process_predictions.py \
    --input data/AI_HUB_DASAN_QA/06_predictions/prediction-model-*_predictions.jsonl \
    --output data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs_full.jsonl
```

### 2. organize_markdown.py

JSONL → 주제별 Markdown 디렉터리 구조

**위치**: `src/knowledge_extraction/organize_markdown.py`

**주요 기능**:
- topic_path 기반 디렉터리 생성
- 개별 .md 파일 생성
- index.md 자동 생성 (각 디렉터리)
- README.md 생성 (전체 요약)

**사용 예**:
```bash
uv run python src/knowledge_extraction/organize_markdown.py \
    --input data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs_full.jsonl \
    --output-dir data/AI_HUB_DASAN_QA/03_markdown_full
```

### 3. DasanKnowledgeLoader

JSONL → LangChain Documents (RAG 파이프라인 투입)

**위치**: `src/rag_pipeline/stages/stage_01_dasan_loader.py`

**주요 기능**:
- JSONL 스트리밍 로드
- LangChain Document 변환
- 메타데이터 보존
- 검증된 문서 필터링

**사용 예**:
```python
from src.rag_pipeline.stages.stage_01_dasan_loader import load_dasan_knowledge

documents = load_dasan_knowledge(
    jsonl_path=Path("data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs_full.jsonl"),
    validated_only=True
)
```

---

## 📊 통계 및 품질 지표

### 문서 길이 분포

| 지표 | 값 |
|------|-----|
| 평균 길이 | 1,618자 (404 토큰) |
| 예상 청크 수 | ~0.8 청크/문서 |
| 최소 길이 | 500자 (검증 기준) |

**해석**:
- 대부분의 문서가 512 토큰 청크 1개 이내
- Chunking 후에도 의미 단위 유지 가능
- 멀티홉 추론에 적합한 길이

### 검증 통과율

| 항목 | 값 |
|------|-----|
| 전체 문서 | 9,632 |
| 검증 통과 | 9,011 (93.6%) |
| 검증 실패 | 621 (6.4%) |
| 파싱 오류 | 7 (0.08%) |

**주요 실패 원인**:
- `no_qa_format`: Q/A 구조 누락
- `too_short`: 500자 미만
- `no_frontmatter`: YAML 헤더 누락

### 카테고리 균형

| 카테고리 | 문서 수 | 비율 |
|----------|---------|------|
| 일반행정_문의 | 2,918 | 30.3% |
| 대중교통_안내 | 2,369 | 24.6% |
| 생활하수도_관련_문의 | 2,203 | 22.9% |
| 코로나19_관련_상담 | 1,898 | 19.7% |
| 기타 | 244 | 2.5% |

---

## 🚀 다음 단계

### 1. RAG 파이프라인 통합

```python
# Stage 1: Dasan Knowledge 로드
from src.rag_pipeline.stages.stage_01_dasan_loader import DasanKnowledgeLoader

loader = DasanKnowledgeLoader(jsonl_path=Path("data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs_full.jsonl"))
documents = loader.load_documents()

# Stage 2-6: 기존 파이프라인 실행
# ... (chunking, embedding, vector store, retrieval, generation)
```

### 2. 품질 개선

**검증 실패 문서 수정**:
```bash
# 실패한 문서 목록 확인
python -c "
import json
with open('data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs_full.jsonl', 'r') as f:
    failed = [json.loads(line) for line in f if not json.loads(line)['metadata']['validated']]
print(f'Failed: {len(failed)}')
for doc in failed[:10]:
    print(f\"  {doc['dialogue_id']}: {doc['metadata']['validation_issues']}\")
"
```

### 3. 평가 준비

문서에 포함된 `original_question`과 `original_answer`를 평가 데이터셋으로 활용:

```python
# 평가 데이터셋 생성
import json

eval_dataset = []
with open('data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs_full.jsonl', 'r') as f:
    for line in f:
        doc = json.loads(line)
        if doc['metadata']['validated']:
            eval_dataset.append({
                "question": doc['original_question'],
                "ground_truth": doc['original_answer'],
                "dialogue_id": doc['dialogue_id'],
                "category": doc['metadata']['category']
            })

print(f"Evaluation dataset: {len(eval_dataset)} Q/A pairs")
```

---

## 📁 최종 파일 구조

```
data/AI_HUB_DASAN_QA/
├── prediction-model-*_predictions.jsonl      # 원본 Gemini 출력 (172MB)
├── knowledge_docs_full.jsonl                 # ⭐ 추출된 지식베이스 (44MB)
├── knowledge_docs_full_metadata.json         # 처리 통계 (144KB)
├── organized_markdown_full/                  # Markdown 조직화 결과
│   ├── README.md
│   ├── 대중교통_안내/
│   │   ├── 버스/
│   │   │   ├── index.md
│   │   │   └── *.md (486개)
│   │   └── 지하철/
│   │       └── *.md
│   ├── 일반행정_문의/
│   └── ...
└── ...

src/knowledge_extraction/
├── process_predictions.py                    # ⭐ Predictions → JSONL
├── organize_markdown.py                      # ⭐ JSONL → Markdown
└── consolidate_extractions.py                # (레거시, 개별 파일용)

src/rag_pipeline/stages/
└── stage_01_dasan_loader.py                  # ⭐ JSONL → LangChain Docs

logs/dasan_knowledge_loader/
└── pipeline_*.log                            # 로더 실행 로그
```

---

## ❓ FAQ

### Q1: 전체 처리 시간은?
A: 약 5-10분 (9,187개 predictions 처리 기준)
- `process_predictions.py`: ~1분
- `organize_markdown.py`: ~2분

### Q2: 메모리 사용량은?
A: 약 1-2GB
- JSONL 스트리밍 처리로 메모리 효율적
- 전체 파일을 메모리에 로드하지 않음

### Q3: 검증 실패 문서는 어떻게 처리하나요?
A: 3가지 옵션:
1. **포함 (기본)**: JSONL에 포함, metadata.validated=false로 표시
2. **필터링**: `DasanKnowledgeLoader(validated_only=True)` 사용
3. **수정 후 재처리**: Markdown 파일 수정 → JSONL 재생성

### Q4: 새로운 predictions 추가 시 처리 방법은?
A: JSONL은 append-only 형식:
```bash
# 기존 JSONL 백업
cp knowledge_docs_full.jsonl knowledge_docs_full.backup.jsonl

# 새 predictions 처리 후 병합
cat knowledge_docs_full.backup.jsonl new_docs.jsonl > knowledge_docs_merged.jsonl
```

### Q5: Markdown 파일만 수정했는데 JSONL 업데이트가 필요한가요?
A: 아니요. RAG 파이프라인은 JSONL만 사용합니다.
Markdown은 사람이 리뷰/수정하기 위한 용도이므로, 수정 후 JSONL로 역변환하는 스크립트가 필요하다면 별도 작성해야 합니다.

---

## 📚 참고 문서

- **Gemini Batch API**: `docs/gemini_batch_api_guide.md`
- **RAG Pipeline**: `docs/rag_pipeline_overview.md`
- **Data Schema**: `docs/data_schema.md`

---

**마지막 업데이트**: 2025-11-09
**작성자**: Knowledge Extraction Pipeline
**버전**: 1.0.0
