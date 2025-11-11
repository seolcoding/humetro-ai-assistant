# Deprecated RAG Pipeline Components

이 디렉토리에는 더 이상 사용하지 않는 RAG 파이프라인 컴포넌트들이 있습니다.

## 📅 Deprecated Date: 2025-11-11

## ⚠️ 사용 중단 사유

### Question Generation (질문 생성 시스템)

**이유**: 실제 콜센터 Q&A 데이터를 사용하므로 RAGAS 질문 생성 불필요

**대체 방법**: `DasanQASampler`를 사용하여 `knowledge_docs_full.jsonl`에서 직접 샘플링

**파일들**:
- `question_generation/DEPRECATED_question_generation.py` - RAGAS 질문 생성 모듈
- `question_generation/DEPRECATED_generation_benchmark.py` - 질문 생성 벤치마크
- `question_generation/DEPRECATED_testset_generator.py` - 테스트셋 생성기
- `question_generation/DEPRECATED_generate_benchmark_50q.py` - 50개 질문 생성
- `question_generation/DEPRECATED_generate_100q_benchmark_v2.py` - 100개 질문 생성

### Old Benchmark Versions (구 버전 벤치마크)

**이유**: `unified_benchmark_v4_real_qa.py`로 통합 (실제 Q&A 기반)

**현재 버전**: `src/rag_pipeline/unified_benchmark_v4_real_qa.py`

**파일들**:
- `DEPRECATED_unified_benchmark_v2.py` - 체크포인트 지원 버전
- `DEPRECATED_unified_benchmark_v3_parallel.py` - 병렬 평가 버전
- `DEPRECATED_unified_benchmark_v4_dasan.py` - Dasan 데이터 + 질문 생성 버전

## 🚀 현재 사용 중인 시스템

### 1. 통합 벤치마크 v4 Real QA
```bash
python src/rag_pipeline/unified_benchmark_v4_real_qa.py \
  --num-questions 50 \
  --models thesis \
  --judge-model vertex_ai/gemini-2.5-pro
```

**특징**:
- ✅ 실제 콜센터 Q&A 사용 (질문 생성 불필요)
- ✅ DasanQASampler를 통한 효율적 샘플링
- ✅ Vertex AI Gemini 2.5 Pro 평가
- ✅ 병렬/순차 평가 지원
- ✅ 체크포인트 및 재시작 지원

### 2. Q&A 데이터 샘플러
```python
from src.data_loader.dasan_qa_sampler import DasanQASampler

sampler = DasanQASampler("data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs_full.jsonl")
qa_samples = sampler.sample(n=50, strategy="random")
```

## 📝 마이그레이션 가이드

### 기존 코드 (Deprecated)
```python
from question_generation import generate_questions

# RAGAS로 질문 생성 (느림, 비용 발생)
config, testset_df = generate_questions(
    model="gpt-4o-mini",
    num_questions=50
)
```

### 새 코드 (Recommended)
```python
from src.data_loader.dasan_qa_sampler import DasanQASampler

# 실제 Q&A에서 직접 샘플링 (빠름, 무료)
sampler = DasanQASampler("data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs_full.jsonl")
qa_samples = sampler.sample(n=50, strategy="random", seed=42)
ragas_dataset = sampler.to_ragas_format(qa_samples)
```

## 🗂️ 데이터 구조

### 현재 사용 중인 데이터
```
data/AI_HUB_DASAN_QA/
├── 05_consolidated/
│   └── knowledge_docs_full.jsonl  # 182,719 Q&A pairs
└── 07_vector_stores/
    └── full/  # 벡터 스토어
```

### Q&A 샘플 형식
```json
{
  "id": "dasan_0001",
  "category": "교통",
  "question": "교통카드는 어떻게 충전하나요?",
  "answer": "교통카드는 편의점이나 지하철역 무인충전기에서 충전할 수 있습니다.",
  "context": "교통카드 충전 방법 안내...",
  "metadata": {
    "entities": ["교통카드", "편의점", "충전기"],
    "topics": ["교통", "카드"],
    "kb_tags": ["payment", "transportation"]
  }
}
```

## 🔍 성능 비교

| 방식 | 시간 | 비용 | 품질 |
|------|------|------|------|
| **RAGAS 질문 생성** (Deprecated) | ~30분 | $5-10 | 가변적 |
| **실제 Q&A 샘플링** (Current) | ~1초 | $0 | 실제 데이터 |

## ⚠️ 주의사항

1. **Deprecated 파일 사용 금지**: 이 디렉토리의 파일들은 참고용으로만 보관
2. **Import 경로 변경**: 기존 import는 모두 에러 발생
3. **새 벤치마크 사용**: `unified_benchmark_v4_real_qa.py` 사용 권장

## 📚 관련 문서

- [DasanQASampler 가이드](../../data_loader/README.md)
- [벤치마크 실행 가이드](../README.md)
- [Vertex AI 평가 설정](../../evaluation/README.md)

---

**Last Updated**: 2025-11-11
**Deprecated By**: Question generation system replaced with real Q&A sampling
