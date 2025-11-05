#!/usr/bin/env python3
"""
50-Question Benchmark TestSet Generation
=========================================

논문용 50개 질문 벤치마크 생성

설정:
- LLM: GPT-4o-mini (비용 효율적, 품질 우수)
- 문서: 전체 크롤링 데이터 활용
- 목표: 50개 질문 생성
- 캐싱: 재실행시 자동 캐시 사용

Usage:
    python src/rag_pipeline/generate_benchmark_50q.py
    python src/rag_pipeline/generate_benchmark_50q.py --force  # 캐시 무시하고 재생성
"""

import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from question_generation import generate_50q_benchmark


def main():
    parser = argparse.ArgumentParser(description="50개 질문 벤치마크 테스트셋 생성")
    parser.add_argument(
        "--force", action="store_true", help="캐시 무시하고 강제 재생성"
    )
    parser.add_argument(
        "--num-docs", type=int, default=50, help="사용할 문서 개수 (기본: 50)"
    )
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  50-Question Benchmark TestSet Generation".center(70))
    print("=" * 70)

    print("\n[설정]")
    print(f"  LLM: gpt-4o-mini (temperature=0.3)")
    print(f"  Embedding: text-embedding-3-small")
    print(f"  문서 개수: {args.num_docs}")
    print(f"  목표 질문 수: 50")
    print(f"  한국어 Persona: True")
    print(f"  강제 재생성: {args.force}")

    # 생성 또는 로드
    print("\n[실행]")
    test_config, testset_df = generate_50q_benchmark(
        model="gpt-4o-mini",
        num_documents=args.num_docs,
        force=args.force
    )

    # 결과 출력
    print("\n" + "=" * 70)
    print("  결과 요약".center(70))
    print("=" * 70)
    print(f"\n✅ 총 질문 수: {len(testset_df)}")
    print(f"📁 캐시 키: {test_config.get_cache_key()}")

    # 샘플 출력 (처음 5개)
    print("\n" + "-" * 70)
    print("샘플 질문 (처음 5개):")
    print("-" * 70)

    for idx, row in testset_df.head(5).iterrows():
        print(f"\n[질문 {idx + 1}]")
        print(f"Q: {row.get('user_input', 'N/A')}")
        answer = row.get("reference", "N/A")
        print(f"A: {answer[:150]}{'...' if len(answer) > 150 else ''}")

    # 파일 경로 출력
    cache_dir = Path("data/evaluation/testsets")
    cache_key = test_config.get_cache_key()

    print("\n" + "=" * 70)
    print("  생성된 파일".center(70))
    print("=" * 70)
    print(f"\n📄 JSON: {cache_dir}/testset_{cache_key}.json")
    print(f"📊 CSV: {cache_dir}/testset_{cache_key}.csv")
    print(f"📝 문서: {cache_dir}/testset_{cache_key}.md")

    print("\n" + "=" * 70)
    print("  다음 단계".center(70))
    print("=" * 70)
    print("\n이제 이 테스트셋으로 LLM 비교 평가를 실행하세요:")
    print("\n  python src/rag_pipeline/run_benchmark_evaluation.py \\")
    print(f"    --testset {cache_dir}/testset_{cache_key}.csv")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
