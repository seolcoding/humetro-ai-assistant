# 통합 병렬 실행 시스템

## 개요

`run_benchmark.py`에 병렬/순차 실행 모드를 통합하여 설정 파일만으로 실행 방식 제어

### 기능

- ✅ 설정 파일 기반 모드 선택 (`mode: "sequential"` or `"parallel"`)
- ✅ 기존 코드 호환성 유지 (기본값: sequential)
- ✅ 모든 RAG 방식 지원 (Naive, KG Simple, KG Cypher)
- ✅ 동적 max_concurrent 설정

---

## 사용법

### 1. 순차 실행 (기본값)

```json
{
  "execution": {
    "mode": "sequential"
  }
}
```

```bash
python run_benchmark.py --config config/healthcheck.json
```

**특징:**
- 기존 방식과 동일
- 안정적이고 예측 가능
- 디버깅 용이

---

### 2. 병렬 실행 (고속)

```json
{
  "execution": {
    "mode": "parallel",
    "parallel": {
      "enabled": true,
      "max_concurrent": 5
    }
  }
}
```

```bash
python run_benchmark.py --config config/examples/parallel_eval_example.json
```

**특징:**
- 2-3x 속도 향상
- I/O bound 평가 동시 실행
- API 호출 효율 최대화

---

## 설정 파일 구조

### Execution 섹션

```json
{
  "execution": {
    "mode": "sequential",           // "sequential" or "parallel"
    "parallel": {
      "enabled": false,              // true로 설정 시 parallel 모드 강제
      "max_concurrent": 5,           // 최대 동시 실행 수 (1-10)
      "comment": "..."
    },
    "checkpoint_interval": 1,
    "verbose": true,
    "dry_run": false
  }
}
```

### 모드 결정 우선순위

1. `parallel.enabled = true` → **parallel** 모드
2. `mode = "parallel"` → **parallel** 모드
3. 기타 → **sequential** 모드 (안전한 기본값)

---

## 성능 비교

### 예시: 50 질문 × 5 모델 × 3 RAG 방식

| 모드 | 답변 생성 | 평가 시간 | 총 시간 | 개선율 |
|------|----------|----------|--------|--------|
| **Sequential** | 1.5시간 | 8.5시간 | **10시간** | - |
| **Parallel** | 1.5시간 | 3.0시간 | **4.5시간** | **2.2x** |

**계산:**
- Sequential: 3 methods × (5 models × 10 min) = 150 min/method = 450 min total
- Parallel: 3 methods × max(5 models parallel) = 3 × 30 min = 90 min total

---

## 아키텍처

### Sequential Mode (기존)

```
For each RAG method:
  For each model:
    Generate answers
    Evaluate with RAGAS
    Save results
```

### Parallel Mode (신규)

```
For each RAG method:
  Phase 1 (순차):
    For each model:
      Generate answers

  Phase 2 (병렬):
    Evaluate all models concurrently
      Model 1 ──┐
      Model 2 ──┤
      Model 3 ──┼─→ AsyncIO → Aggregate
      Model 4 ──┤
      Model 5 ──┘
```

---

## 코드 구조

### 새로 추가된 파일

```
src/rag_pipeline/
├── parallel_benchmark_runner.py   (190 lines) - 병렬 실행 래퍼
├── answer_generator.py           (280 lines) - 답변 생성 분리
└── generation_benchmark.py       (기존) - 순차 실행

src/evaluation/
└── parallel_evaluator.py         (350 lines) - AsyncIO 평가
```

### 통합 지점 (run_benchmark.py)

```python
# 실행 모드 결정
execution_mode = config["execution"]["mode"]
if config["execution"]["parallel"]["enabled"]:
    execution_mode = "parallel"

# 벤치마크 생성
if execution_mode == "parallel":
    benchmark = create_parallel_benchmark(...)
else:
    benchmark = GenerationBenchmark(...)

# 실행 (인터페이스 동일)
results = benchmark.run_benchmark(questions, output_dir)
```

---

## 예제 설정 파일

### 1. 소규모 테스트 (parallel_eval_example.json)

```json
{
  "experiment": {
    "name": "parallel_eval_demo",
    "description": "10 questions, 5 models, 2 RAG methods"
  },
  "questions": {
    "source": "golden",
    "limit": 10
  },
  "models": {
    "evaluation_targets": "thesis"
  },
  "retrieval": [
    {"name": "naive_rag", "k": 4, "naive_vector_store": "..."},
    {"name": "kg_simple", "k": 4, "kg_config": {...}}
  ],
  "evaluation": {
    "judge_model": "gpt-5"
  },
  "execution": {
    "mode": "parallel",
    "parallel": {
      "enabled": true,
      "max_concurrent": 5
    }
  }
}
```

**실행:**
```bash
python run_benchmark.py --config config/examples/parallel_eval_example.json
```

**예상 시간:**
- Sequential: ~25 min (2 methods × 5 models × 2.5 min)
- Parallel: ~10 min (2 methods × max(5 models))
- **개선: 2.5x faster**

---

### 2. 대규모 평가 (50Q, 3 methods)

```json
{
  "questions": {
    "source": "golden",
    "limit": 50
  },
  "retrieval": [
    {"name": "naive_rag", ...},
    {"name": "kg_simple", ...},
    {"name": "kg_cypher", ...}
  ],
  "execution": {
    "mode": "parallel",
    "parallel": {
      "enabled": true,
      "max_concurrent": 5
    }
  }
}
```

**예상 시간:**
- Sequential: ~10 hours
- Parallel: ~4.5 hours
- **개선: 2.2x faster**

---

## 모드 선택 가이드

### Sequential 모드 사용 시

✅ 디버깅 및 테스트
✅ API rate limit 우려
✅ 메모리 제약
✅ 안정성 최우선

### Parallel 모드 사용 시

✅ 대규모 평가 (50+ questions)
✅ 시간 제약
✅ 충분한 API quota
✅ 프로덕션 벤치마크

---

## 제한 사항

### 1. API Rate Limits

**문제:**
```
OpenAI: 500 RPM (requests per minute)
병렬 5 모델 × 50 질문 × 3 메트릭 = 750 requests
```

**해결:**
```json
{
  "parallel": {
    "max_concurrent": 3  // 낮춤
  }
}
```

### 2. 메모리 사용

**Sequential:** ~500MB (모델 1개 데이터셋)
**Parallel:** ~2GB (모델 5개 데이터셋 동시 로드)

**해결:** 질문 수 제한 또는 max_concurrent 조정

### 3. 에러 처리

**Parallel 모드:**
- 개별 모델 실패 시 다른 모델 평가 계속 진행
- 최종 결과에 실패 모델 표시

---

## 디버깅

### 실행 모드 확인

```bash
python run_benchmark.py --config config/healthcheck.json

# Output:
# Running Benchmark (SEQUENTIAL MODE)
# 또는
# Running Benchmark (PARALLEL MODE)
# Max Concurrent: 5
```

### 로그 분석

**Sequential:**
```
INFO: 평가 중: GPT-4o-mini
INFO: ✅ GPT-4o-mini 평가 완료
INFO: 평가 중: EXAONE-3.5
INFO: ✅ EXAONE-3.5 평가 완료
```

**Parallel:**
```
INFO: 🚀 병렬 평가 시작 (최대 동시 실행: 5)
INFO: 🔄 GPT-4o-mini 평가 시작...
INFO: 🔄 EXAONE-3.5 평가 시작...
INFO: 🔄 Qwen3-8B 평가 시작...
모델 평가 진행:  60%|██████    | 3/5 [00:15<00:10,  5.0s/model]
INFO: ✅ 병렬 평가 완료: 18.7초 (5/5 성공)
```

---

## 마이그레이션 가이드

### 기존 설정 파일 업데이트

**Before (v1):**
```json
{
  "execution": {
    "verbose": true
  }
}
```

**After (v2 - 순차 유지):**
```json
{
  "execution": {
    "mode": "sequential",
    "parallel": {
      "enabled": false,
      "max_concurrent": 5
    },
    "verbose": true
  }
}
```

**After (v2 - 병렬 활성화):**
```json
{
  "execution": {
    "mode": "parallel",
    "parallel": {
      "enabled": true,
      "max_concurrent": 5
    },
    "verbose": true
  }
}
```

**호환성:** v1 설정 파일은 기본값(sequential)으로 작동

---

## FAQ

### Q1: 기존 코드 영향?

**A:** 없음. `mode` 미지정 시 자동으로 sequential 모드 (기본값)

### Q2: 모든 RAG 방식 지원?

**A:** 예. Naive, KG Simple, KG Cypher 모두 지원

### Q3: 중간 중단 시 재시작?

**A:** Parallel 모드는 답변 체크포인트 저장, 평가만 재실행 가능

### Q4: 비용 차이?

**A:** API 호출 횟수 동일, 단 병렬 실행으로 시간 절약

### Q5: 정확도 차이?

**A:** 없음. 동일한 RAGAS 평가, 실행 순서만 다름

---

## 다음 단계

### 개선 계획

- [ ] 답변 캐싱으로 평가만 재실행
- [ ] GPU 병렬 답변 생성 (vLLM)
- [ ] Streaming 평가 결과 표시
- [ ] Auto rate limit throttling

---

## 참고

- 병렬 평가 상세: `docs/PARALLEL_EVALUATION.md`
- 설정 파일 스키마: `config/schema.json` (작성 예정)
- 예제 설정: `config/examples/parallel_eval_example.json`
