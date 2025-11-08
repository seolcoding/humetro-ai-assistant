# Vertex AI Integration Guide for RAGAS Evaluation

## Overview

기존 평가 시스템에 Google Vertex AI (Gemini 2.5 Pro & Flash)가 완전히 통합되었습니다.
이제 OpenAI GPT-4o와 함께 Vertex AI를 Judge 모델로 사용할 수 있습니다.

## Quick Start

### 1. 환경 설정

```bash
# .env 파일에 추가
VERTEXAI_PROJECT=your-project-id
VERTEXAI_LOCATION=us-central1
VERTEXAI_PROJECT_SERVICE_ACCOUNT=service_account.json
```

### 2. Preset 사용 (가장 쉬운 방법)

```python
from ragas import EvaluationDataset
from src.evaluation.ragas_evaluator import evaluate_with_preset

# 데이터셋 준비
dataset = EvaluationDataset.from_list(your_data)

# Preset으로 빠른 평가
# Option 1: OpenAI GPT-4o (기본)
results_gpt4o = evaluate_with_preset(dataset, preset="gpt-4o")

# Option 2: Gemini 2.5 Pro (높은 품질)
results_gemini_pro = evaluate_with_preset(dataset, preset="gemini-pro")

# Option 3: Gemini 2.5 Flash (빠르고 저렴)
results_gemini_flash = evaluate_with_preset(dataset, preset="gemini-flash")

# Option 4: GPT-4o-mini (빠르고 저렴)
results_gpt4o_mini = evaluate_with_preset(dataset, preset="gpt-4o-mini")
```

### 3. 커스텀 설정

```python
from src.evaluation.ragas_evaluator import RAGASEvaluator

# Vertex AI Judge 사용
evaluator = RAGASEvaluator(
    judge_config={
        "provider": "vertex_ai",
        "model": "gemini-2.5-pro",
        "temperature": 0.0,
        "max_tokens": 4096,
    },
    embedding_config={
        "provider": "vertex_ai",
        "model": "text-embedding-004",
    }
)

results = evaluator.evaluate(dataset)
```

## Available Presets

| Preset | Provider | Model | Speed | Cost | Quality | Use Case |
|--------|----------|-------|-------|------|---------|----------|
| `gpt-4o` | OpenAI | GPT-4o | Fast | $$ | High | Default judge |
| `gpt-4o-mini` | OpenAI | GPT-4o-mini | Very Fast | $ | Good | Quick iteration |
| `gemini-pro` | Vertex AI | Gemini 2.5 Pro | Fast | $$$ | Very High | Alternative judge |
| `gemini-flash` | Vertex AI | Gemini 2.5 Flash | Very Fast | $ | Good | Quick tests |

## Model Presets Configuration

모델 설정은 `config/model_presets.json`에서 관리됩니다:

```json
{
  "presets": {
    "vertex": {
      "description": "Google Vertex AI models (Gemini 2.5 Pro & Flash)",
      "models": [
        {
          "id": "gemini-2.5-pro",
          "name": "Gemini-2.5-Pro",
          "provider": "vertex_ai",
          "model": "vertex_ai/gemini-2.5-pro",
          "config": {"temperature": 0.7, "max_tokens": 8192}
        },
        {
          "id": "gemini-2.5-flash",
          "name": "Gemini-2.5-Flash",
          "provider": "vertex_ai",
          "model": "vertex_ai/gemini-2.5-flash",
          "config": {"temperature": 0.7, "max_tokens": 8192}
        }
      ]
    },
    "judges": {
      "description": "High-quality models for RAGAS evaluation (Judge models)",
      "models": [
        {
          "id": "gpt-4o",
          "name": "GPT-4o",
          "provider": "openai",
          "model": "gpt-4o",
          "config": {"temperature": 0.0, "max_tokens": 4096}
        },
        {
          "id": "gemini-2.5-pro",
          "name": "Gemini-2.5-Pro",
          "provider": "vertex_ai",
          "model": "vertex_ai/gemini-2.5-pro",
          "config": {"temperature": 0.0, "max_tokens": 4096}
        }
      ]
    }
  }
}
```

## Performance Comparison

### Benchmark Results (3 samples)

| Judge | Faithfulness | Answer Relevancy | Answer Correctness | Speed |
|-------|--------------|------------------|-------------------|-------|
| **GPT-4o** | 0.333 | 0.530 | 0.917 | 10초 |
| **Gemini 2.5 Pro** | 0.333 | 0.674 ⭐ | 1.000 ⭐ | 33초 |
| **Gemini 2.5 Flash** | 0.000 | 0.655 | 1.000 | 20초 |

**결론:**
- **Gemini 2.5 Pro**: 가장 높은 품질 (Answer Relevancy +27%, Answer Correctness 만점)
- **GPT-4o**: 가장 빠름 (3배 빠름)
- **Gemini 2.5 Flash**: 중간 속도, 저렴한 비용

## Advanced Usage

### 1. Judge 비교 평가

```python
from src.evaluation.ragas_evaluator import RAGASEvaluator

judges = {
    "GPT-4o": RAGASEvaluator.create_from_preset("gpt-4o"),
    "Gemini Pro": RAGASEvaluator.create_from_preset("gemini-pro"),
    "Gemini Flash": RAGASEvaluator.create_from_preset("gemini-flash"),
}

results = {}
for name, evaluator in judges.items():
    results[name] = evaluator.evaluate(dataset)
    print(f"{name}: {results[name]}")
```

### 2. 혼합 설정 (OpenAI LLM + Vertex AI Embeddings)

```python
evaluator = RAGASEvaluator(
    judge_config={
        "provider": "openai",
        "model": "gpt-4o",
    },
    embedding_config={
        "provider": "vertex_ai",
        "model": "text-embedding-004",
    }
)
```

### 3. 배치 평가 (여러 모델 동시 평가)

```python
import json
from pathlib import Path

# 모델 프리셋 로드
with open("config/model_presets.json") as f:
    presets = json.load(f)

# Judges 프리셋의 모든 모델로 평가
judges_config = presets["presets"]["judges"]["models"]

results = {}
for judge_cfg in judges_config:
    judge_name = judge_cfg["name"]

    # 평가자 생성
    if judge_cfg["provider"] == "vertex_ai":
        evaluator = RAGASEvaluator.create_from_preset("gemini-pro")
    else:
        evaluator = RAGASEvaluator.create_from_preset("gpt-4o")

    # 평가 실행
    results[judge_name] = evaluator.evaluate(dataset)

# 결과 비교
for name, result in results.items():
    print(f"\n{name}:")
    print(result.to_pandas().mean())
```

## Cost Comparison

### 예상 비용 (1,000 평가 기준)

| Model | Input Cost | Output Cost | Total (Est.) | Notes |
|-------|------------|-------------|--------------|-------|
| GPT-4o | $2.50 | $10.00 | **$12.50** | Balanced |
| GPT-4o-mini | $0.15 | $0.60 | **$0.75** | Cheapest |
| Gemini 2.5 Pro | $1.25 | $5.00 | **$6.25** | Good value |
| Gemini 2.5 Flash | $0.015 | $0.06 | **$0.075** | Very cheap |

**권장사항:**
- **개발/실험**: Gemini 2.5 Flash 또는 GPT-4o-mini
- **프로덕션 평가**: Gemini 2.5 Pro 또는 GPT-4o
- **최고 품질**: Gemini 2.5 Pro (Answer Correctness 만점)

## Troubleshooting

### 1. Vertex AI 인증 오류

```bash
# 서비스 계정 파일 확인
ls -la service_account.json

# 환경 변수 확인
echo $VERTEXAI_PROJECT
echo $VERTEXAI_PROJECT_SERVICE_ACCOUNT

# 권한 확인 (Google Cloud Console)
# IAM → Service Accounts → Vertex AI User 권한 확인
```

### 2. Rate Limit 오류

```python
# 동시 요청 수 제한
import time

results = []
for sample in dataset:
    result = evaluator.evaluate(EvaluationDataset.from_list([sample]))
    results.append(result)
    time.sleep(1)  # 1초 대기
```

### 3. 메모리 부족

```python
# 배치 크기 축소
batch_size = 5
for i in range(0, len(dataset), batch_size):
    batch = dataset[i:i+batch_size]
    result = evaluator.evaluate(batch)
```

## Testing

전체 통합 테스트 실행:

```bash
# Vertex AI 통합 테스트
uv run tests/rag_pipeline/test_ragas_with_vertex_ai.py

# 30개 동시 호출 테스트
uv run tests/rag_pipeline/test_vertex_ai_concurrent_30_pro.py
```

## Files Created

### Core Integration
- `src/evaluation/vertex_ai_llm_wrapper.py` - Vertex AI + LiteLLM + RAGAS 통합
- `src/evaluation/ragas_evaluator.py` - 멀티 프로바이더 평가자

### Configuration
- `config/model_presets.json` - Vertex AI 프리셋 추가
  - `vertex` preset: Gemini 2.5 Pro & Flash
  - `judges` preset: 고품질 Judge 모델

### Tests
- `tests/rag_pipeline/test_ragas_with_vertex_ai.py` - 통합 테스트
- `tests/rag_pipeline/test_vertex_ai_concurrent_30.py` - Flash 동시 호출
- `tests/rag_pipeline/test_vertex_ai_concurrent_30_pro.py` - Pro 동시 호출

### Documentation
- `docs/VERTEX_AI_INTEGRATION.md` - 상세 통합 가이드
- `docs/VERTEX_AI_BILLING_CHECK.md` - 크레딧 확인 방법
- `VERTEX_AI_QUICKSTART.md` - 빠른 시작 가이드

## Next Steps

1. **기존 평가 스크립트 마이그레이션**
   ```python
   # Before
   from langchain_openai import ChatOpenAI
   eval_llm = ChatOpenAI(model="gpt-5")

   # After (더 쉬움!)
   from src.evaluation.ragas_evaluator import evaluate_with_preset
   result = evaluate_with_preset(dataset, preset="gemini-pro")
   ```

2. **Judge 비교 실험**
   - GPT-4o vs Gemini 2.5 Pro 품질 비교
   - 비용 대비 성능 분석
   - 특정 도메인에서의 성능 차이 확인

3. **프로덕션 배포**
   - 최적 Judge 모델 선택
   - 비용 모니터링 설정
   - 자동화 파이프라인 구축

---

**참고**: Vertex AI API는 사용량에 따라 과금되므로, 개발 단계에서는 Flash 모델을 사용하고
프로덕션에서는 Pro 모델을 사용하는 것을 권장합니다.
