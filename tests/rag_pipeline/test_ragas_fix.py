#!/usr/bin/env python3
"""
RAGAS 평가 수정 테스트 스크립트
"""

import logging
from pathlib import Path
from generation_benchmark import GenerationBenchmark

logging.basicConfig(level=logging.INFO)

# 작은 테스트 데이터
test_questions = {
    "single_hop": [
        {
            "id": 1,
            "question": "2호선 첫차 시간은?",
            "reference_answer": "서울 지하철 2호선의 첫차 시간은 오전 5시 30분입니다.",
            "reference_contexts": ["서울 지하철 2호선 운영 시간: 첫차 5:30, 막차 24:00"]
        }
    ],
    "multi_hop": [
        {
            "id": 2,
            "question": "2호선이 중단되면 강남에서 홍대까지 어떻게 가나요?",
            "reference_answer": "3호선을 이용하여 종로3가에서 환승하거나 7호선을 이용할 수 있습니다.",
            "reference_contexts": ["대체 경로: 3호선, 7호선 이용 가능", "환승역 정보"]
        }
    ]
}

# 모델 설정
models = [
    {
        "name": "GPT-4o-mini",
        "model": "gpt-4o-mini",
        "api_base": None
    }
]

# 벤치마크 실행
print("="*70)
print("RAGAS 평가 수정 테스트")
print("="*70)

benchmark = GenerationBenchmark(
    models=models,
    use_fixed_context=True,
    k_documents=2,
    judge_model="gpt-5"
)

output_dir = Path("data/evaluation/generation_benchmark/test")
output_dir.mkdir(parents=True, exist_ok=True)

results = benchmark.run_benchmark(test_questions, output_dir)

print("\n결과:")
print(f"Single-hop 결과: {results.get('single_hop', {})}")
print(f"Multi-hop 결과: {results.get('multi_hop', {})}")

if "comparison" in results:
    print(f"\n비교 분석: {results['comparison']}")

print("\n✅ 테스트 완료!")