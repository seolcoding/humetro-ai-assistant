"""
AutoRAG 평가 통합 테스트

생성된 QA로 간단한 RAG 평가 수행:
1. Naive RAG (Vector Search + GPT-4o-mini)
2. RAGAS 평가 (Faithfulness, Answer Relevancy)

사용법:
    python run_evaluation_test.py --data-dir output/e2e_test_XXXXXX
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd

# AutoRAG 임포트
try:
    import autorag
    from autorag.evaluator import Evaluator
    HAS_AUTORAG = True
except ImportError:
    print("Warning: autorag 패키지가 없습니다. pip install autorag")
    HAS_AUTORAG = False


def create_simple_config(output_dir: Path) -> Path:
    """간단한 RAG 평가 config 생성"""
    config = """
# E2E 테스트용 간단한 RAG 설정
node_lines:
  - node_line_name: retrieve_node_line
    nodes:
      - node_type: semantic_retrieval
        strategy:
          metrics: [retrieval_f1, retrieval_recall]
          speed_threshold: 10
        top_k: 5
        embedding_model: openai
        modules:
          - module_type: vectordb
            embedding_model: openai

  - node_line_name: post_retrieve_node_line
    nodes:
      - node_type: prompt_maker
        strategy:
          metrics: [answer_relevancy]
          speed_threshold: 10
        modules:
          - module_type: fstring
            prompt: |
              다음 문서들을 참고하여 질문에 답변하세요.

              문서:
              {retrieved_contents}

              질문: {query}
              답변:

      - node_type: generator
        strategy:
          metrics: [answer_relevancy, g_eval]
          speed_threshold: 30
        modules:
          - module_type: openai_llm
            llm: gpt-4o-mini
            max_tokens: 512
            temperature: 0.1
"""
    config_path = output_dir / "eval_config.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config)

    return config_path


def run_simple_rag_evaluation(data_dir: str):
    """
    간단한 RAG 평가 실행

    1. VectorDB 생성
    2. Retrieval + Generation
    3. RAGAS 평가
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        print(f"Error: 데이터 디렉토리가 없습니다: {data_dir}")
        return

    print("=" * 60)
    print("AutoRAG 평가 통합 테스트")
    print("=" * 60)

    # 데이터 로드
    qa_path = data_dir / "qa.parquet"
    corpus_path = data_dir / "corpus.parquet"

    if not qa_path.exists() or not corpus_path.exists():
        print(f"Error: qa.parquet 또는 corpus.parquet이 없습니다")
        return

    qa_df = pd.read_parquet(qa_path)
    corpus_df = pd.read_parquet(corpus_path)

    print(f"\n[1/4] 데이터 확인...")
    print(f"  - QA: {len(qa_df)}개")
    print(f"  - Corpus: {len(corpus_df)}개")

    # Multi-hop 질문 확인
    print(f"\n  Multi-hop 질문:")
    for i, row in qa_df.iterrows():
        n_docs = len(row['retrieval_gt'])
        if n_docs > 1:
            print(f"    - {row['qid']}: {n_docs}개 문서 필요")

    # 평가 디렉토리 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = data_dir / f"eval_{timestamp}"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Config 생성
    print(f"\n[2/4] 평가 설정 생성...")
    config_path = create_simple_config(eval_dir)
    print(f"  - Config: {config_path}")

    if not HAS_AUTORAG:
        print("\nWarning: autorag가 없어 평가를 건너뜁니다.")
        print("수동 평가를 위해 데이터를 저장합니다...")

        # 수동 검토용 데이터 저장
        manual_review = []
        for i, row in qa_df.iterrows():
            manual_review.append({
                "qid": row["qid"],
                "query": row["query"],
                "retrieval_gt": list(row["retrieval_gt"]),
                "generation_gt": list(row["generation_gt"]),
                "n_docs_required": len(row["retrieval_gt"])
            })

        with open(eval_dir / "manual_review.json", 'w', encoding='utf-8') as f:
            json.dump(manual_review, f, ensure_ascii=False, indent=2)

        print(f"\n저장 완료: {eval_dir / 'manual_review.json'}")
        return

    # AutoRAG 평가 실행
    print(f"\n[3/4] AutoRAG 평가 실행...")
    try:
        # 데이터 복사
        (eval_dir / "data").mkdir(exist_ok=True)
        qa_df.to_parquet(eval_dir / "data" / "qa.parquet", index=False)
        corpus_df.to_parquet(eval_dir / "data" / "corpus.parquet", index=False)

        # Evaluator 실행
        evaluator = Evaluator(
            qa_data_path=str(eval_dir / "data" / "qa.parquet"),
            corpus_data_path=str(eval_dir / "data" / "corpus.parquet"),
            project_dir=str(eval_dir / "project")
        )

        evaluator.start_trial(str(config_path))

        print(f"\n[4/4] 평가 완료!")
        print(f"  - 결과 디렉토리: {eval_dir / 'project'}")

    except Exception as e:
        print(f"\nError: AutoRAG 평가 실패: {e}")
        import traceback
        traceback.print_exc()

    # 결과 요약
    print(f"\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)
    print(f"\n출력 디렉토리: {eval_dir}")


def check_data_quality(data_dir: str):
    """데이터 품질 검토 (AutoRAG 없이)"""
    data_dir = Path(data_dir)

    print("=" * 60)
    print("생성된 QA 데이터 품질 검토")
    print("=" * 60)

    # QA 로드
    qa_path = data_dir / "qa.parquet"
    if not qa_path.exists():
        print(f"Error: {qa_path}가 없습니다")
        return

    qa_df = pd.read_parquet(qa_path)

    print(f"\n총 QA: {len(qa_df)}개")

    # 질문 유형별 분석
    print(f"\n=== 질문 분석 ===")
    for i, row in qa_df.iterrows():
        print(f"\n[{i+1}] {row['qid']}")
        print(f"  Query: {row['query']}")
        print(f"  Answer: {row['generation_gt']}")
        print(f"  Retrieval GT: {row['retrieval_gt']} ({len(row['retrieval_gt'])}개 문서)")

    # Multi-hop 통계
    multihop_count = sum(1 for _, row in qa_df.iterrows() if len(row['retrieval_gt']) > 1)
    print(f"\n=== 통계 ===")
    print(f"  - Single-hop: {len(qa_df) - multihop_count}개")
    print(f"  - Multi-hop: {multihop_count}개")

    # JSON 로드 (상세 정보)
    json_path = data_dir / "generated_qa.json"
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            qa_json = json.load(f)

        print(f"\n=== 질문 유형별 분포 ===")
        type_counts = {}
        for qa in qa_json.get("qa_pairs", []):
            qt = qa.get("question_type", "unknown")
            type_counts[qt] = type_counts.get(qt, 0) + 1

        for qt, cnt in type_counts.items():
            print(f"  - {qt}: {cnt}개")


def main():
    parser = argparse.ArgumentParser(description="AutoRAG 평가 통합 테스트")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="생성된 QA 데이터 디렉토리"
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="데이터 품질만 검토 (평가 없음)"
    )

    args = parser.parse_args()

    # 데이터 디렉토리 자동 탐색
    if args.data_dir is None:
        output_base = Path(__file__).parent / "output"
        dirs = sorted(output_base.glob("e2e_test_*"), reverse=True)
        if dirs:
            args.data_dir = str(dirs[0])
            print(f"가장 최근 데이터 사용: {args.data_dir}")
        else:
            print("Error: 생성된 데이터가 없습니다. run_e2e_test.py를 먼저 실행하세요.")
            return

    if args.check_only:
        check_data_quality(args.data_dir)
    else:
        run_simple_rag_evaluation(args.data_dir)


if __name__ == "__main__":
    main()
