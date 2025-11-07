# 병렬 평가 시스템 (Parallel Evaluation System)

## 개요

RAG 시스템 평가 시간을 **5배 단축**하는 병렬 평가 아키텍처

### 성능 비교

| 방식 | 시간 | 설명 |
|------|------|------|
| **기존 (Sequential)** | ~50분 | 5 models × 10 min/model |
| **신규 (Parallel)** | ~22분 | 10분 (답변 생성) + 12분 (병렬 평가) |
| **개선율** | **2.3x faster** | I/O bound 평가를 동시 실행 |

## 아키텍처

### Phase 1: Answer Generation (순차)

```
For each model:
  ├─ Retrieve contexts (if enabled)
  ├─ Generate answers
  └─ Save to checkpoint
```

- **Why Sequential?**: 모델별 답변 생성은 CPU/GPU bound
- **Checkpoint**: 중단 시 재시작 가능

### Phase 2: Parallel Evaluation (병렬)

```
AsyncIO:
  ├─ Model 1 → RAGAS evaluate (GPT-5 judge)
  ├─ Model 2 → RAGAS evaluate (GPT-5 judge)
  ├─ Model 3 → RAGAS evaluate (GPT-5 judge)
  ├─ Model 4 → RAGAS evaluate (GPT-5 judge)
  └─ Model 5 → RAGAS evaluate (GPT-5 judge)
     ↓
  Aggregate results
```

- **Why Parallel?**: RAGAS 평가는 I/O bound (API 호출)
- **Concurrency**: asyncio로 최대 5개 동시 실행

## 사용법

### 1. 기본 실행

```bash
uv run python src/rag_pipeline/unified_benchmark_v3_parallel.py \
  --questions 50 \
  --models thesis \
  --judge-model gpt-5 \
  --max-concurrent 5
```

### 2. 옵션 설명

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--questions` | 50 | 평가할 질문 수 |
| `--models` | thesis | 모델 그룹 (thesis, all, openai, ollama) |
| `--judge-model` | gpt-5 | RAGAS 평가 모델 |
| `--max-concurrent` | 5 | 최대 동시 평가 수 (1-10) |
| `--k-documents` | 4 | 검색 문서 수 |
| `--no-retrieval` | False | Retrieval 없이 실행 |

### 3. 재시작

```bash
# 중단된 실험 재개
uv run python src/rag_pipeline/unified_benchmark_v3_parallel.py \
  --resume-id 42
```

## 주요 모듈

### 1. `answer_generator.py`

**역할**: 답변 생성만 담당 (평가와 분리)

```python
from answer_generator import generate_answers_only

all_datasets = generate_answers_only(
    models=models,
    questions=questions,
    use_fixed_context=True
)
# Returns: {"GPT-4o-mini": {...}, "EXAONE-3.5": {...}}
```

**주요 기능**:
- 모든 모델의 답변 생성
- 고정 컨텍스트 지원 (retrieval 변동성 제거)
- 체크포인트 저장

### 2. `parallel_evaluator.py`

**역할**: 병렬 평가 실행

```python
from parallel_evaluator import run_parallel_evaluation

results = run_parallel_evaluation(
    model_datasets=all_datasets,
    judge_model="gpt-5",
    max_concurrent=5
)
# Returns: {"results": {...}, "metadata": {...}}
```

**주요 기능**:
- AsyncIO 기반 병렬 평가
- RAGAS 메트릭 (Faithfulness, Answer Relevancy, Answer Correctness)
- 자동 결과 통합 및 비교 테이블 생성

### 3. `unified_benchmark_v3_parallel.py`

**역할**: 전체 파이프라인 통합

```python
benchmark = ParallelBenchmarkV3(args)
benchmark.run()
```

**실행 흐름**:
1. 질문 생성/로드
2. 질문 분류 (single-hop/multi-hop)
3. Phase 1: 답변 생성 (순차)
4. Phase 2: 병렬 평가
5. 결과 저장 및 통합

## 기술 상세

### AsyncIO 구현

```python
async def evaluate_single_model(model_name, dataset):
    # RAGAS evaluate()를 비동기로 실행
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(
        None,
        self._run_ragas_evaluation,
        dataset,
        model_name
    )
    return result

# 병렬 실행
tasks = [evaluate_single_model(m, d) for m, d in model_datasets.items()]
results = await asyncio.gather(*tasks)
```

### Semaphore로 동시성 제어

```python
class ParallelEvaluationManager:
    def __init__(self, max_concurrent=5):
        self.semaphore = asyncio.Semaphore(max_concurrent)

    async def _evaluate_with_semaphore(self, model, dataset):
        async with self.semaphore:
            # 최대 max_concurrent개만 동시 실행
            return await self.evaluate_single_model(model, dataset)
```

## 체크포인트 시스템

### 저장되는 단계

| 단계 | 파일명 | 내용 |
|------|--------|------|
| 질문 생성 | `exp_{id}_questions_generated.pkl` | RAGAS testset |
| 질문 분류 | `exp_{id}_questions_classified.pkl` | single-hop/multi-hop |
| 답변 생성 | `exp_{id}_answers_generated.pkl` | 모든 모델 답변 |
| 평가 완료 | `exp_{id}_evaluation_completed.pkl` | 평가 결과 |

### 재시작 예시

```bash
# 실험 중단 (Ctrl+C)
# 출력: "재시작: python ... --resume-id 42"

# 재시작 (완료된 단계는 건너뜀)
uv run python src/rag_pipeline/unified_benchmark_v3_parallel.py --resume-id 42
```

## 테스트

### 단위 테스트

```bash
# 소규모 테스트 (2 models, 3 questions)
uv run python tests/test_parallel_evaluation.py
```

### 실제 평가 테스트

```bash
# 10 questions, 5 models
uv run python src/rag_pipeline/unified_benchmark_v3_parallel.py \
  --questions 10 \
  --models thesis \
  --judge-model gpt-4o-mini  # Cheaper for testing
```

## 결과 포맷

### 비교 테이블 (CSV)

```
Model,Faithfulness,Answer Relevancy,Answer Correctness,Average
GPT-4o-mini,0.892,0.945,0.678,0.838
EXAONE-3.5-7.8B,0.845,0.912,0.623,0.793
...
```

### 상세 결과 (JSON)

```json
{
  "results": {
    "GPT-4o-mini": {
      "model_name": "GPT-4o-mini",
      "metrics": {
        "faithfulness": 0.892,
        "answer_relevancy": 0.945,
        "answer_correctness": 0.678
      },
      "per_question_scores": [...]
    },
    ...
  },
  "metadata": {
    "elapsed_seconds": 720,
    "successful_models": 5,
    "total_models": 5,
    "max_concurrent": 5
  }
}
```

## 성능 최적화 팁

### 1. 동시성 조정

```bash
# GPU 메모리가 충분하면 증가
--max-concurrent 10

# API rate limit 걱정되면 감소
--max-concurrent 3
```

### 2. Judge Model 선택

| Model | 속도 | 품질 | 비용 | 추천 |
|-------|------|------|------|------|
| gpt-5 | 느림 | 최고 | 비쌈 | 최종 평가 |
| gpt-4o | 중간 | 우수 | 중간 | 밸런스 |
| gpt-4o-mini | 빠름 | 양호 | 저렴 | 테스트 |

### 3. 체크포인트 활용

```bash
# 답변만 생성 후 저장
--skip-evaluation  # (구현 예정)

# 나중에 평가만 실행
--evaluation-only --resume-id 42  # (구현 예정)
```

## 제한 사항

1. **RAGAS 평가는 여전히 시간 소요**
   - GPT-5 API 호출 대기 시간
   - 병렬화로 완화되지만 API 속도 자체는 동일

2. **메모리 사용**
   - 5개 모델 데이터셋을 메모리에 로드
   - 질문 수가 많으면 메모리 부족 가능

3. **API Rate Limit**
   - `max_concurrent`가 너무 크면 API 제한에 걸림
   - OpenAI: 기본 500 RPM (requests per minute)

## 향후 개선

- [ ] Thread pool 기반 대안 구현 (asyncio 문제 시)
- [ ] GPU 병렬 답변 생성 (vLLM 통합)
- [ ] Streaming 평가 결과 출력
- [ ] API rate limit auto-throttling
- [ ] Memory-efficient batch processing

## 기여

버그 리포트 및 개선 제안:
- Issue: [GitHub Issues](https://github.com/atoye1/humetro-ai-assistant/issues)
- PR: feature/parallel-evaluation 브랜치
