"""
AutoRAG 파일럿 실행 스크립트

사용법:
    uv run python src/autorag_pilot/run_pilot.py
"""

import os
from pathlib import Path

# 환경 변수 로드
from dotenv import load_dotenv

load_dotenv()

from autorag.evaluator import Evaluator


def run_pilot():
    """AutoRAG 파일럿 최적화 실행"""
    # 경로 설정
    base_dir = Path("src/autorag_pilot")
    output_dir = base_dir / "output"
    project_dir = base_dir / "autorag_project"

    qa_path = output_dir / "qa_train.parquet"
    corpus_path = output_dir / "corpus.parquet"
    config_path = base_dir / "config" / "simple_rag.yaml"

    # 파일 존재 확인
    for path in [qa_path, corpus_path, config_path]:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    print("=" * 60)
    print("AutoRAG Pilot Evaluation")
    print("=" * 60)
    print(f"QA Data: {qa_path}")
    print(f"Corpus: {corpus_path}")
    print(f"Config: {config_path}")
    print(f"Project Dir: {project_dir}")
    print("=" * 60)

    # OpenAI API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    # Evaluator 생성 및 실행
    evaluator = Evaluator(
        qa_data_path=str(qa_path),
        corpus_data_path=str(corpus_path),
        project_dir=str(project_dir),
    )

    print("\nStarting evaluation...")
    evaluator.start_trial(str(config_path))

    print("\n" + "=" * 60)
    print("Evaluation Complete!")
    print("=" * 60)
    print(f"Results saved to: {project_dir}")
    print("\nTo view dashboard, run:")
    print(f"  autorag dashboard --trial_dir {project_dir}/0")


if __name__ == "__main__":
    run_pilot()
