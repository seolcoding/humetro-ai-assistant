# Vertex AI Integration for RAGAS Evaluation

## 개요

Google Cloud Vertex AI를 RAGAS 평가 시스템에 통합하여 Gemini 모델을 Judge로 사용할 수 있습니다.

### 주요 기능

- ✅ **LiteLLM 통합**: Vertex AI를 LiteLLM을 통해 사용
- ✅ **RAGAS 평가**: Vertex AI Gemini를 Judge 모델로 활용
- ✅ **비동기 지원**: Async call 완벽 지원
- ✅ **한국어 최적화**: 한국어 평가에 적합한 구조

## 설치

### 1. 필수 패키지 설치

```bash
uv add langchain-google-vertexai litellm
```

### 2. Vertex AI 인증 설정

#### 방법 1: gcloud CLI 인증 (권장)

```bash
# gcloud CLI 설치 후
gcloud auth application-default login
```

#### 방법 2: 서비스 계정 JSON 키 사용

```bash
# 환경 변수 설정
export VERTEXAI_API_KEY="/path/to/service-account.json"
export VERTEXAI_PROJECT="your-project-id"
export VERTEXAI_LOCATION="us-central1"
```

`.env` 파일에 추가:

```bash
VERTEXAI_API_KEY=/path/to/service-account.json
VERTEXAI_PROJECT=your-project-id
VERTEXAI_LOCATION=us-central1
```

## 사용법

### 1. 기본 설정

```python
from src.evaluation.vertex_ai_llm_wrapper import (
    VertexAILLMConfig,
    create_vertex_ai_evaluator_llm,
    create_vertex_ai_embeddings,
)

# Vertex AI 설정
config = VertexAILLMConfig(
    model_name="gemini-2.0-flash-001",
    project_id="your-project-id",  # 또는 환경 변수에서 자동 로드
    location="us-central1",
    embedding_model="text-embedding-004",
    temperature=0.0
)

# Evaluator LLM 생성
evaluator_llm = create_vertex_ai_evaluator_llm(config)
evaluator_embeddings = create_vertex_ai_embeddings(config)
```

### 2. RAGAS 평가에 사용

```python
from ragas import evaluate, EvaluationDataset
from ragas.metrics import Faithfulness, AnswerRelevancy, AnswerCorrectness

# RAGAS 메트릭 설정
metrics = [
    Faithfulness(llm=evaluator_llm),
    AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
    AnswerCorrectness(llm=evaluator_llm),
]

# 평가 실행
result = evaluate(
    dataset=your_dataset,
    metrics=metrics,
    llm=evaluator_llm
)
```

### 3. LiteLLM으로 답변 생성

```python
from src.evaluation.vertex_ai_llm_wrapper import create_vertex_ai_litellm_model
from litellm import completion

# Vertex AI 모델 문자열 생성
model = create_vertex_ai_litellm_model(
    model_name="gemini-2.0-flash-001",
    project_id="your-project-id",
    location="us-central1"
)

# 답변 생성
response = completion(
    model=model,
    messages=[
        {"role": "system", "content": "당신은 서울시 교통 정보 전문가입니다."},
        {"role": "user", "content": "지하철 요금은 얼마인가요?"}
    ],
    max_tokens=200,
    temperature=0.1
)

print(response.choices[0].message.content)
```

## 테스트

### 통합 테스트 실행

4단계 테스트가 포함된 통합 테스트:

```bash
uv run tests/rag_pipeline/test_vertex_ai_ragas_integration.py
```

테스트 단계:
1. **Single Call**: 동기 단일 호출 테스트
2. **Async Call**: 비동기 단일 호출 테스트
3. **RAGAS Call**: RAGAS 평가 호출 테스트
4. **RAGAS Async Call**: RAGAS 비동기 평가 테스트

### LLM 비교 평가 (Vertex AI Judge)

Vertex AI를 Judge로 사용하여 여러 LLM 비교:

```bash
uv run tests/rag_pipeline/test_ragas_vertex_ai_judge.py
```

## 지원 모델

### Chat Models (LLM)

| 모델 ID | 설명 | 비용 |
|---------|------|------|
| `gemini-2.0-flash-001` | 최신 Gemini 2.0 Flash - 빠르고 효율적 | Low |
| `gemini-1.5-pro` | Gemini 1.5 Pro - 최고 품질 | Medium |
| `gemini-1.5-flash-001` | Gemini 1.5 Flash - 균형잡힌 성능 | Low |

### Embedding Models

| 모델 ID | 설명 | 차원 |
|---------|------|------|
| `text-embedding-004` | 최신 텍스트 임베딩 모델 | 768 |
| `textembedding-gecko@001` | 레거시 임베딩 모델 | 768 |

## 아키텍처

### 구성 요소

```
┌─────────────────────────────────────────────────────────┐
│                    RAGAS 평가 시스템                     │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────┐      ┌──────────────────┐        │
│  │  Vertex AI LLM   │      │  Vertex AI       │        │
│  │  (Judge)         │      │  Embeddings      │        │
│  │  - Gemini 2.0    │      │  - text-embed-004│        │
│  └────────┬─────────┘      └────────┬─────────┘        │
│           │                         │                   │
│           ▼                         ▼                   │
│  ┌────────────────────────────────────────┐            │
│  │   LangchainLLMWrapper                  │            │
│  │   - Custom is_finished_parser          │            │
│  │   - Gemini completion signal handling  │            │
│  └────────────────┬───────────────────────┘            │
│                   │                                     │
│                   ▼                                     │
│  ┌────────────────────────────────────────┐            │
│  │   RAGAS Metrics                        │            │
│  │   - Faithfulness                       │            │
│  │   - AnswerRelevancy                    │            │
│  │   - AnswerCorrectness                  │            │
│  └────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   답변 생성 시스템                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────────────────────────────────┐          │
│  │  LiteLLM                                  │          │
│  │  - vertex_ai/gemini-2.0-flash-001        │          │
│  │  - 환경 변수 자동 설정                    │          │
│  └────────────────┬─────────────────────────┘          │
│                   │                                     │
│                   ▼                                     │
│  ┌────────────────────────────────────────┐            │
│  │  Vertex AI API                         │            │
│  │  - Project ID                          │            │
│  │  - Location                            │            │
│  │  - Credentials                         │            │
│  └────────────────────────────────────────┘            │
└─────────────────────────────────────────────────────────┘
```

### 흐름도

```
1. 초기화
   ├─ VertexAILLMConfig 생성
   ├─ 인증 정보 로드 (환경 변수 또는 gcloud)
   └─ LLM/Embeddings 래퍼 생성

2. 답변 생성 (LiteLLM)
   ├─ create_vertex_ai_litellm_model()
   ├─ litellm.completion() 호출
   └─ Vertex AI API → Gemini 모델 → 답변

3. RAGAS 평가 (Judge)
   ├─ Evaluator LLM 초기화
   ├─ Metrics 설정 (Faithfulness, AnswerRelevancy 등)
   ├─ evaluate() 호출
   └─ Vertex AI Gemini가 답변 품질 평가

4. 결과 분석
   ├─ 메트릭별 점수 계산
   ├─ 모델 간 비교
   └─ JSON/CSV 저장
```

## 환경 변수 참조

| 변수 | 설명 | 필수 | 기본값 |
|------|------|------|--------|
| `VERTEXAI_PROJECT` | GCP 프로젝트 ID | ✅ | - |
| `VERTEXAI_LOCATION` | GCP 리전 | ❌ | `us-central1` |
| `VERTEXAI_API_KEY` | 서비스 계정 JSON 경로 | ❌ | gcloud 인증 사용 |
| `GOOGLE_APPLICATION_CREDENTIALS` | Google 인증 정보 | ❌ | `VERTEXAI_API_KEY`와 동일 |

## 모범 사례

### 1. 평가 모델 선택

- **빠른 평가**: `gemini-2.0-flash-001` (비용 효율적)
- **고품질 평가**: `gemini-1.5-pro` (더 정확한 판단)
- **균형**: `gemini-1.5-flash-001`

### 2. Temperature 설정

- **평가 모델 (Judge)**: `temperature=0.0` (일관성을 위해)
- **답변 생성 모델**: `temperature=0.1-0.3` (약간의 창의성)

### 3. 비용 최적화

- Flash 모델 우선 사용
- 배치 평가로 API 호출 최소화
- 캐싱 활용

### 4. 에러 처리

```python
from src.evaluation.vertex_ai_llm_wrapper import test_vertex_ai_connection

# 평가 전 연결 테스트
config = VertexAILLMConfig()
if not test_vertex_ai_connection(config):
    print("Vertex AI 연결 실패. 설정을 확인하세요.")
    exit(1)
```

## 문제 해결

### 1. 인증 오류

```
Error: Could not automatically determine credentials
```

**해결 방법**:
```bash
# gcloud 인증
gcloud auth application-default login

# 또는 환경 변수 설정
export VERTEXAI_API_KEY="/path/to/service-account.json"
```

### 2. 프로젝트 ID 누락

```
ValueError: Vertex AI project ID not found
```

**해결 방법**:
```bash
export VERTEXAI_PROJECT="your-project-id"
```

### 3. Temperature 파라미터 오류

Vertex AI는 일부 모델에서 특정 temperature 값만 지원합니다.

**해결 방법**:
- `temperature=0.0` 사용 (권장)
- LiteLLM의 `drop_params=True` 설정 확인

### 4. 할당량 초과

```
Error: Quota exceeded for quota metric 'GenerateContent requests'
```

**해결 방법**:
- GCP Console에서 할당량 증가 요청
- 요청 속도 제한 추가
- 배치 크기 줄이기

## 참고 자료

### 공식 문서

- [Vertex AI Documentation](https://cloud.google.com/vertex-ai/docs)
- [LiteLLM Vertex AI Provider](https://docs.litellm.ai/docs/providers/vertex)
- [RAGAS with Vertex AI](https://docs.ragas.io/en/stable/howtos/applications/vertexai_x_ragas/)

### 관련 파일

- `src/evaluation/vertex_ai_llm_wrapper.py` - Vertex AI 래퍼
- `tests/rag_pipeline/test_vertex_ai_ragas_integration.py` - 통합 테스트
- `tests/rag_pipeline/test_ragas_vertex_ai_judge.py` - LLM 비교 평가

## 라이선스

이 프로젝트는 기존 humetro-ai-assistant 프로젝트의 라이선스를 따릅니다.
