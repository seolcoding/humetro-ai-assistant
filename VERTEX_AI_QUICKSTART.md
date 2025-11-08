# Vertex AI 빠른 시작 가이드

## 1단계: 환경 변수 설정

`.env` 파일에 다음을 추가하세요:

```bash
# Vertex AI 설정
VERTEXAI_PROJECT=your-project-id
VERTEXAI_LOCATION=us-central1
VERTEXAI_API_KEY=/path/to/service-account.json  # 선택사항
```

**또는** 터미널에서 직접 설정:

```bash
export VERTEXAI_PROJECT=your-project-id
export VERTEXAI_LOCATION=us-central1
```

**또는** gcloud 인증 사용:

```bash
gcloud auth application-default login
```

## 2단계: 필요한 패키지 설치 확인

```bash
uv add langchain-google-vertexai
```

## 3단계: 연결 테스트

### 옵션 A: 통합 테스트 실행 (권장)

```bash
uv run tests/rag_pipeline/test_vertex_ai_ragas_integration.py
```

이 테스트는 4단계로 구성됩니다:
- ✅ Single Call (동기 단일 호출)
- ✅ Async Call (비동기 단일 호출)
- ✅ RAGAS Call (RAGAS 평가)
- ✅ RAGAS Async Call (RAGAS 비동기 평가)

### 옵션 B: 예제 코드 실행

```bash
uv run examples/vertex_ai_ragas_example.py
```

### 옵션 C: LLM 비교 평가

```bash
uv run tests/rag_pipeline/test_ragas_vertex_ai_judge.py
```

## 4단계: 코드에서 사용

### 기본 사용법

```python
from src.evaluation.vertex_ai_llm_wrapper import (
    VertexAILLMConfig,
    create_vertex_ai_evaluator_llm,
    create_vertex_ai_embeddings,
)

# 설정 (환경 변수에서 자동 로드)
config = VertexAILLMConfig()

# RAGAS용 LLM 생성
evaluator_llm = create_vertex_ai_evaluator_llm(config)
evaluator_embeddings = create_vertex_ai_embeddings(config)

# RAGAS 평가
from ragas.metrics import Faithfulness
faithfulness = Faithfulness(llm=evaluator_llm)
```

### LiteLLM으로 답변 생성

```python
from src.evaluation.vertex_ai_llm_wrapper import create_vertex_ai_litellm_model
from litellm import completion

# Vertex AI 모델 문자열 생성
model = create_vertex_ai_litellm_model()

# 답변 생성
response = completion(
    model=model,
    messages=[{"role": "user", "content": "서울 지하철 요금은?"}],
    max_tokens=100
)
```

## 문제 해결

### 오류: "Vertex AI project ID not found"

**원인**: VERTEXAI_PROJECT 환경 변수 미설정

**해결**:
```bash
export VERTEXAI_PROJECT=your-project-id
```

### 오류: "Could not automatically determine credentials"

**원인**: Google Cloud 인증 미설정

**해결**:
```bash
gcloud auth application-default login
```

또는 서비스 계정 JSON 키 사용:
```bash
export VERTEXAI_API_KEY=/path/to/service-account.json
```

### 오류: "No module named 'langchain_google_vertexai'"

**원인**: 필수 패키지 미설치

**해결**:
```bash
uv add langchain-google-vertexai
```

## 다음 단계

상세 문서는 다음을 참고하세요:
- 📖 [전체 문서](docs/VERTEX_AI_INTEGRATION.md)
- 💻 [통합 테스트](tests/rag_pipeline/test_vertex_ai_ragas_integration.py)
- 🎯 [예제 코드](examples/vertex_ai_ragas_example.py)
- 🔍 [LLM 비교 평가](tests/rag_pipeline/test_ragas_vertex_ai_judge.py)

## 지원 모델

### LLM (Chat)
- `gemini-2.0-flash-001` (권장, 빠르고 저렴)
- `gemini-1.5-pro` (최고 품질)
- `gemini-1.5-flash-001` (균형)

### Embeddings
- `text-embedding-004` (권장)
- `textembedding-gecko@001` (레거시)

## 비용 최적화 팁

1. **Flash 모델 사용**: `gemini-2.0-flash-001`이 비용 효율적
2. **Temperature=0**: 평가 시 일관성을 위해 temperature=0 사용
3. **배치 처리**: 여러 샘플을 한 번에 평가
4. **캐싱**: 동일한 컨텍스트 재사용
