# Benchmark Configuration Guide

설정 파일 기반 벤치마크 실행 시스템

## 설정 파일 구조

### 필수 섹션

#### 1. experiment (실험 메타데이터)
```json
{
  "experiment": {
    "id": null,              // 재시작할 실험 ID, null이면 새 실험
    "name": "experiment_name",
    "description": "실험 설명",
    "tags": ["tag1", "tag2"],
    "output_dir": "data/evaluation/experiments"
  }
}
```

#### 2. questions (질문 소스)

**Option A: 캐싱된 질문 사용 (빠름)**
```json
{
  "questions": {
    "source": "cached",
    "cache_key": "359312cb043d5a69"  // 기존 생성된 질문 재사용
  }
}
```

**Option B: 새로 생성**
```json
{
  "questions": {
    "source": "generate",
    "generation": {
      "model": "gpt-4o-mini",
      "num_documents": 50,
      "num_questions": 50,
      "document_source": "data/crawled/seoul_traffic/markdown_filtered",
      "force_regenerate": false  // 같은 설정이면 캐시 재사용
    }
  }
}
```

#### 3. models (평가 대상 모델)

**Option A: Preset 사용**
```json
{
  "models": {
    "evaluation_targets": "thesis"  // "thesis" | "all" | "fast" | "openai" | "ollama" | "korean"
  }
}
```

**Option B: 커스텀 모델 리스트**
```json
{
  "models": {
    "evaluation_targets": [
      {
        "id": "gpt-4o-mini",
        "name": "GPT-4o-mini",
        "provider": "openai",
        "model": "gpt-4o-mini",
        "config": {"temperature": 0.7, "max_tokens": 1024}
      }
    ]
  }
}
```

#### 4. retrieval (검색 방법)

**None (LLM only)**
```json
{
  "retrieval": {
    "method": "none"
  }
}
```

**Naive RAG (FAISS)**
```json
{
  "retrieval": {
    "method": "naive",
    "k": 4,
    "vector_store_path": "data/vector_store/seoul_traffic"
  }
}
```

**Knowledge Graph RAG (Neo4j)**
```json
{
  "retrieval": {
    "method": "kg",
    "k": 4,
    "kg_config": {
      "neo4j_uri": "env:NEO4J_URI",
      "neo4j_user": "env:NEO4J_USER",
      "neo4j_password": "env:NEO4J_PASSWORD",
      "index_name": "vector",
      "embedding_model": "text-embedding-3-large"
    }
  }
}
```

#### 5. evaluation (평가 설정)
```json
{
  "evaluation": {
    "judge_model": "gpt-5",
    "metrics": ["faithfulness", "answer_relevancy", "answer_correctness"]
  }
}
```

## 사용 예시

### 1. 캐싱된 질문으로 빠른 평가
```bash
python run_benchmark.py --config config/examples/quick_eval_cached.json
```

### 2. 100개 새 질문 생성 + 전체 모델 평가
```bash
python run_benchmark.py --config config/examples/generate_100q_full.json
```

### 3. Naive RAG vs KG RAG 비교
```bash
# Naive RAG 실행
python run_benchmark.py --config config/examples/compare_naive_rag.json

# KG RAG 실행 (같은 질문, 같은 모델, 같은 judge)
python run_benchmark.py --config config/examples/compare_kg_rag.json

# 결과 비교
python compare_results.py \
  data/evaluation/rag_comparison/naive_rag_50q/ \
  data/evaluation/rag_comparison/kg_rag_50q/
```

### 4. LLM-only 베이스라인
```bash
python run_benchmark.py --config config/examples/llm_only_baseline.json
```

## Available Model Presets

설정 파일에 정의된 preset (config/model_presets.json):

- `"thesis"` - 논문 실험용 5개 모델 (GPT-4o-mini + 4 Ollama)
- `"fast"` - 빠른 반복용 (GPT-4o-mini만)
- `"openai"` - OpenAI 모델들 (GPT-4o, GPT-4o-mini)
- `"ollama"` - Ollama 한국어 모델들 (EXAONE, Gemma)
- `"korean"` - 한국어 최적화 모델만 (EXAONE)
- `"all"` - 모든 사용 가능한 모델

## 설정 파일 위치

```
config/
├── benchmark_schema.json           # JSON Schema (validation)
├── model_presets.json              # 모델 preset 정의
├── examples/                       # 예시 설정들
│   ├── quick_eval_cached.json     # 캐시 사용 빠른 평가
│   ├── generate_100q_full.json    # 100Q 생성 + 전체 평가
│   ├── compare_naive_rag.json     # Naive RAG 비교용
│   ├── compare_kg_rag.json        # KG RAG 비교용
│   └── llm_only_baseline.json     # LLM-only 베이스라인
```

## 캐시 키 확인

사용 가능한 캐싱된 질문 확인:
```bash
python -c "from src.rag_pipeline.question_generation import list_cached_questions;
import json;
print(json.dumps(list_cached_questions(), indent=2))"
```

## Tips

1. **비교 실험**: 같은 `cache_key` 사용하면 공정한 비교 가능
2. **재현성**: 설정 파일을 git에 커밋하면 실험 완전 재현 가능
3. **빠른 반복**: `dry_run: true`로 계획만 먼저 확인
4. **체크포인트**: 긴 실험은 `checkpoint_interval` 조정
