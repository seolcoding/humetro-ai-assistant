# RAG Pipeline

통합 RAG 벤치마크 시스템 - 실제 Q&A 데이터 기반 평가

## 🚀 현재 버전: v4 Real QA

**메인 파일**: `unified_benchmark_v4_real_qa.py`

### 주요 특징

✅ **실제 콜센터 Q&A 사용** - 질문 생성 불필요  
✅ **182,719 Q&A pairs** - AI Hub 다산콜센터 데이터  
✅ **효율적 샘플링** - DasanQASampler로 즉시 샘플링  
✅ **Vertex AI 평가** - Gemini 2.5 Pro/Flash  
✅ **병렬/순차 평가** - 안정성 선택 가능  
✅ **체크포인트 지원** - 중단 후 재시작 가능

## 📖 사용 방법

### 기본 실행

```bash
# 50개 질문, 논문용 5개 모델 평가
python src/rag_pipeline/unified_benchmark_v4_real_qa.py \
  --num-questions 50 \
  --models thesis \
  --judge-model vertex_ai/gemini-2.5-pro

# 빠른 테스트 (10개 질문, 1개 모델)
python src/rag_pipeline/unified_benchmark_v4_real_qa.py \
  --num-questions 10 \
  --models gpt-4o-mini \
  --judge-model vertex_ai/gemini-2.5-flash
```

## ⚠️ Deprecated Components

질문 생성 시스템은 더 이상 사용하지 않습니다.

**Deprecated 디렉토리**: `src/rag_pipeline/deprecated/`

**이유**: 실제 콜센터 Q&A 데이터 사용으로 RAGAS 질문 생성 불필요

**자세한 내용**: [deprecated/README.md](deprecated/README.md)

---

**Last Updated**: 2025-11-11  
**Version**: v4 Real QA  
**Status**: ✅ Production Ready
