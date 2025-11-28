# Evaluation Data Structure

개선된 평가 시스템 - 답변 생성과 평가 완전 분리

## 디렉토리 구조

```
data/evaluation/
├── README.md                          # 이 파일
│
├── dasan_real_qa/                     # 기존 체크포인트 (하위 호환)
│   ├── checkpoints/
│   │   ├── experiments.json
│   │   └── exp_*_*.pkl
│   └── experiment_*_results.json
│
├── golden_testset_50.jsonl            # 🔑 고정된 50개 질문 세트
│
├── generations/                       # ✅ 답변 생성 결과 (재사용 가능)
│   ├── naive_rag/                     # Naive RAG 답변
│   │   ├── gpt-4o-mini.jsonl
│   │   └── exaone-3.5-7.8b.jsonl
│   ├── kg_simple/                     # KG Simple RAG 답변
│   │   ├── gpt-4o-mini.jsonl
│   │   └── exaone-3.5-7.8b.jsonl
│   └── kg_cypher/                     # KG Cypher RAG 답변
│       ├── gpt-4o-mini.jsonl
│       └── exaone-3.5-7.8b.jsonl
│
└── evaluations/                       # ✅ 평가 결과 (독립 저장)
    ├── naive_rag_eval.jsonl           # Naive RAG 평가
    ├── naive_rag_eval.summary.json
    ├── kg_simple_eval.jsonl           # KG Simple 평가
    ├── kg_simple_eval.summary.json
    ├── kg_cypher_eval.jsonl           # KG Cypher 평가
    └── kg_cypher_eval.summary.json
```

## 파일 포맷

### generations/*.jsonl (답변 생성 결과)

```jsonl
{"question_id": "q001", "question": "...", "answer": "...", "ground_truth": "...", "contexts": [...]}
{"question_id": "q002", "question": "...", "answer": "...", "ground_truth": "...", "contexts": [...]}
```

### evaluations/*.jsonl (평가 결과)

```jsonl
{"model": "GPT-4o-mini", "question_id": "q001", "faithfulness": 0.85, "answer_relevancy": 0.92, ...}
{"model": "GPT-4o-mini", "question_id": "q002", "faithfulness": 0.78, "answer_relevancy": 0.88, ...}
```

### evaluations/*.summary.json (평가 요약)

```json
{
  "metadata": {
    "judge_model": "vertex_ai/gemini-2.5-pro",
    "elapsed_seconds": 120.5,
    "successful_models": 2
  },
  "models": {
    "GPT-4o-mini": {
      "metrics": {
        "faithfulness": 0.815,
        "answer_relevancy": 0.900,
        "answer_correctness": 0.725
      }
    }
  }
}
```

## 워크플로우

### Phase 1: 답변 생성 (1회만, 비용 발생)

```bash
# Naive RAG
uv run python src/rag_pipeline/unified_benchmark_v4_real_qa.py \
  --golden-testset data/evaluation/golden_testset_50.jsonl \
  --models gpt-4o-mini ollama/exaone3.5:7.8b \
  --output-generations data/evaluation/generations/naive_rag/ \
  --skip-evaluation

# KG Simple
uv run python src/rag_pipeline/unified_benchmark_v4_real_qa.py \
  --golden-testset data/evaluation/golden_testset_50.jsonl \
  --models gpt-4o-mini ollama/exaone3.5:7.8b \
  --rag-method kg_simple \
  --output-generations data/evaluation/generations/kg_simple/ \
  --skip-evaluation

# KG Cypher
uv run python src/rag_pipeline/unified_benchmark_v4_real_qa.py \
  --golden-testset data/evaluation/golden_testset_50.jsonl \
  --models gpt-4o-mini ollama/exaone3.5:7.8b \
  --rag-method kg_cypher \
  --output-generations data/evaluation/generations/kg_cypher/ \
  --skip-evaluation
```

### Phase 2: 평가 (N회 가능, 무료)

```bash
# Naive RAG 평가 (Gemini Pro)
uv run python src/evaluation/evaluate_saved_generations.py \
  --generations-dir data/evaluation/generations/naive_rag/ \
  --judge-model vertex_ai/gemini-2.5-pro \
  --output data/evaluation/evaluations/naive_rag_eval.jsonl

# Naive RAG 재평가 (Gemini Flash)
uv run python src/evaluation/evaluate_saved_generations.py \
  --generations-dir data/evaluation/generations/naive_rag/ \
  --judge-model vertex_ai/gemini-2.5-flash \
  --output data/evaluation/evaluations/naive_rag_eval_flash.jsonl

# KG Simple 평가
uv run python src/evaluation/evaluate_saved_generations.py \
  --generations-dir data/evaluation/generations/kg_simple/ \
  --judge-model vertex_ai/gemini-2.5-pro \
  --output data/evaluation/evaluations/kg_simple_eval.jsonl

# KG Cypher 평가
uv run python src/evaluation/evaluate_saved_generations.py \
  --generations-dir data/evaluation/generations/kg_cypher/ \
  --judge-model vertex_ai/gemini-2.5-pro \
  --output data/evaluation/evaluations/kg_cypher_eval.jsonl
```

## 장점

### ✅ 비용 절감
- 답변 생성은 1회만 (OpenAI API 비용 발생)
- 평가는 무제한 반복 (Vertex AI 무료 할당량 사용)

### ✅ 재현성 보장
- Golden testset (50개 질문) 고정
- 모든 실험에서 동일한 질문 사용
- JSONL 포맷으로 버전 관리 가능

### ✅ 유연한 평가
- Judge 모델 변경 가능 (Pro ↔ Flash)
- 평가 메트릭 추가/변경 용이
- 모델별/RAG별 비교 분석 간편

### ✅ 논문 작성 용이
- 명확한 데이터 출처 추적
- 실험 재현 가능
- 결과 공유 및 검증 용이

## 기존 시스템과 비교

### 기존 (Before)
```
실험 → 질문 샘플링 → 답변 생성 → 평가 → 결과
  ❌ 질문 매번 다름
  ❌ 답변 재사용 불가
  ❌ 평가만 재실행 불가
  ❌ 비용 중복 발생
```

### 개선 (After)
```
Golden Testset → 답변 생성 (1회) → 평가 (N회) → 결과
  ✅ 질문 항상 동일
  ✅ 답변 JSONL 저장
  ✅ 평가 독립 실행
  ✅ 비용 1회만 발생
```

## 논문 실험 계획

### 실험 1: RAG 방법론 비교 (2 Models × 3 RAG Methods)

**Phase 1: 답변 생성**
```bash
# 1. Naive RAG
uv run python src/rag_pipeline/unified_benchmark_v4_real_qa.py \
  --golden-testset data/evaluation/golden_testset_50.jsonl \
  --models gpt-4o-mini ollama/exaone3.5:7.8b \
  --output-generations data/evaluation/generations/naive_rag/ \
  --skip-evaluation

# 2. KG Simple
# ... (위 참고)

# 3. KG Cypher
# ... (위 참고)
```

**Phase 2: 평가**
```bash
# 모든 RAG 방법론 평가
for rag_method in naive_rag kg_simple kg_cypher; do
  uv run python src/evaluation/evaluate_saved_generations.py \
    --generations-dir data/evaluation/generations/${rag_method}/ \
    --judge-model vertex_ai/gemini-2.5-pro \
    --output data/evaluation/evaluations/${rag_method}_eval.jsonl
done
```

**예상 비용:**
- 답변 생성: 2 모델 × 50 질문 × 3 RAG = 300 API 호출 (~$5)
- 평가: 무제한 무료 (Vertex AI 할당량)

### 실험 2: Judge 모델 민감도 분석

동일한 답변에 대해 다른 Judge 모델 비교:

```bash
# Gemini Pro
uv run python src/evaluation/evaluate_saved_generations.py \
  --generations-dir data/evaluation/generations/naive_rag/ \
  --judge-model vertex_ai/gemini-2.5-pro \
  --output data/evaluation/evaluations/naive_rag_eval_pro.jsonl

# Gemini Flash
uv run python src/evaluation/evaluate_saved_generations.py \
  --generations-dir data/evaluation/generations/naive_rag/ \
  --judge-model vertex_ai/gemini-2.5-flash \
  --output data/evaluation/evaluations/naive_rag_eval_flash.jsonl

# GPT-4o
uv run python src/evaluation/evaluate_saved_generations.py \
  --generations-dir data/evaluation/generations/naive_rag/ \
  --judge-model gpt-4o \
  --output data/evaluation/evaluations/naive_rag_eval_gpt4o.jsonl
```

**추가 비용:** $0 (답변 재사용)

## 문제 해결

### Q: Golden testset이 없어요
```bash
# 첫 실험에서 생성된 질문을 golden testset으로 저장
# (체크포인트에서 추출하는 스크립트 필요)
python scripts/extract_golden_testset.py \
  --checkpoint data/evaluation/dasan_real_qa/checkpoints/exp_1_qa_sampled.pkl \
  --output data/evaluation/golden_testset_50.jsonl
```

### Q: 평가만 다시 하고 싶어요
```bash
# 저장된 답변으로 재평가
uv run python src/evaluation/evaluate_saved_generations.py \
  --generations-dir data/evaluation/generations/naive_rag/ \
  --judge-model vertex_ai/gemini-2.5-pro \
  --output data/evaluation/evaluations/naive_rag_eval_v2.jsonl
```

### Q: 새로운 모델 추가하고 싶어요
```bash
# 새 모델만 추가로 답변 생성
uv run python src/rag_pipeline/unified_benchmark_v4_real_qa.py \
  --golden-testset data/evaluation/golden_testset_50.jsonl \
  --models ollama/qwen3:8b \
  --output-generations data/evaluation/generations/naive_rag/ \
  --skip-evaluation

# 기존 평가 재실행 (새 모델 포함)
uv run python src/evaluation/evaluate_saved_generations.py \
  --generations-dir data/evaluation/generations/naive_rag/ \
  --judge-model vertex_ai/gemini-2.5-pro \
  --output data/evaluation/evaluations/naive_rag_eval_updated.jsonl
```

## 다음 단계

1. ✅ Golden testset 생성 (50개 질문)
2. ✅ 디렉토리 구조 생성
3. ⏳ Phase 1: 답변 생성 실행
4. ⏳ Phase 2: 평가 실행
5. ⏳ 결과 분석 및 논문 작성
