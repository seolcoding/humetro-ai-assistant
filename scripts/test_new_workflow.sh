#!/bin/bash
# 새로운 워크플로우 테스트 스크립트
# ======================================
#
# 답변 생성과 평가가 완전히 분리된 새로운 시스템 테스트
#
# 사용법:
#   bash scripts/test_new_workflow.sh

set -e  # Exit on error

echo "========================================================================"
echo "           새로운 워크플로우 테스트 (답변 생성 → 평가 분리)"
echo "========================================================================"
echo

# Check if golden testset exists
if [ ! -f "data/evaluation/golden_testset_50.jsonl" ]; then
    echo "❌ Golden testset이 없습니다!"
    echo "   먼저 golden testset을 생성하세요:"
    echo "   python scripts/create_golden_testset.py"
    exit 1
fi

echo "✅ Golden testset 확인: data/evaluation/golden_testset_50.jsonl"
echo

# ============================================================================
# Phase 1: 답변 생성 (2개 모델만 - 테스트용)
# ============================================================================

echo "========================================================================"
echo "Phase 1: 답변 생성 (Naive RAG)"
echo "========================================================================"
echo "  - Golden testset 사용"
echo "  - 모델: GPT-4o-mini, EXAONE-3.5-7.8B"
echo "  - 평가 건너뛰기 (답변만 생성)"
echo

uv run python src/rag_pipeline/unified_benchmark_v4_real_qa.py \
  --golden-testset data/evaluation/golden_testset_50.jsonl \
  --models gpt-4o-mini ollama/exaone3.5:7.8b \
  --output-generations data/evaluation/generations/naive_rag/ \
  --skip-evaluation \
  --output-dir data/evaluation/dasan_real_qa

echo
echo "✅ Phase 1 완료!"
echo "   답변 저장: data/evaluation/generations/naive_rag/"
echo

# Check generated files
if [ -f "data/evaluation/generations/naive_rag/gpt-4o-mini.jsonl" ]; then
    lines=$(wc -l < data/evaluation/generations/naive_rag/gpt-4o-mini.jsonl)
    echo "   📄 gpt-4o-mini.jsonl: ${lines}개 답변"
fi

if [ -f "data/evaluation/generations/naive_rag/exaone-3.5-7.8b.jsonl" ]; then
    lines=$(wc -l < data/evaluation/generations/naive_rag/exaone-3.5-7.8b.jsonl)
    echo "   📄 exaone-3.5-7.8b.jsonl: ${lines}개 답변"
fi

echo

# ============================================================================
# Phase 2: 평가 (저장된 답변 사용)
# ============================================================================

echo "========================================================================"
echo "Phase 2: 평가 (저장된 답변 사용)"
echo "========================================================================"
echo "  - Judge: Vertex AI Gemini 2.5 Pro"
echo "  - 병렬 평가 (max_concurrent=2)"
echo

uv run python src/evaluation/evaluate_saved_generations.py \
  --generations-dir data/evaluation/generations/naive_rag/ \
  --judge-model vertex_ai/gemini-2.5-pro \
  --output data/evaluation/evaluations/naive_rag_eval.jsonl \
  --max-concurrent 2

echo
echo "✅ Phase 2 완료!"
echo "   평가 결과: data/evaluation/evaluations/naive_rag_eval.jsonl"
echo "   요약 결과: data/evaluation/evaluations/naive_rag_eval.summary.json"
echo

# Check evaluation results
if [ -f "data/evaluation/evaluations/naive_rag_eval.summary.json" ]; then
    echo "📊 평가 요약:"
    cat data/evaluation/evaluations/naive_rag_eval.summary.json | python -m json.tool | grep -A 3 "faithfulness"
fi

echo
echo "========================================================================"
echo "✅ 테스트 완료!"
echo "========================================================================"
echo
echo "다음 단계:"
echo "  1. 결과 확인: cat data/evaluation/evaluations/naive_rag_eval.summary.json"
echo "  2. 다른 Judge 모델로 재평가:"
echo "     uv run python src/evaluation/evaluate_saved_generations.py \\"
echo "       --generations-dir data/evaluation/generations/naive_rag/ \\"
echo "       --judge-model vertex_ai/gemini-2.5-flash \\"
echo "       --output data/evaluation/evaluations/naive_rag_eval_flash.jsonl"
echo
echo "  3. KG RAG 답변 생성 및 평가 (동일한 방식)"
echo
