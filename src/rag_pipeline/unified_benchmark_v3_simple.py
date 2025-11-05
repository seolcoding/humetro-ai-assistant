#!/usr/bin/env python3
"""
Unified RAG Benchmark v3 - Simplified Version
=============================================

Checkpoint/resume 기능이 있는 통합 벤치마크 시스템
RAGAS 의존성 없이 미리 정의된 질문 사용

Features:
- Experiment ID system
- Checkpoint/resume capability
- Predefined questions (no RAGAS dependency)
- CLI interface
"""

import sys
import json
import pickle
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

# Project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


# Predefined questions for testing
PREDEFINED_QUESTIONS = {
    "single_hop": [
        {
            "question": "서울 지하철 1호선의 첫차 시간은 언제인가요?",
            "ground_truth": "평일 기준 서울역 상행 첫차는 05:30, 하행 첫차는 05:19입니다.",
            "question_type": "simple"
        },
        {
            "question": "기후동행카드의 월 이용 요금은 얼마인가요?",
            "ground_truth": "기후동행카드는 월 65,000원(따릉이 포함) 또는 62,000원(따릉이 미포함)입니다.",
            "question_type": "simple"
        },
        {
            "question": "서울 버스의 기본요금은 얼마인가요?",
            "ground_truth": "서울 버스 기본요금은 성인 기준 일반버스 1,500원, 심야버스 2,150원입니다.",
            "question_type": "simple"
        },
        {
            "question": "강남역은 몇 호선이 지나가나요?",
            "ground_truth": "강남역은 2호선과 신분당선이 지나갑니다.",
            "question_type": "simple"
        },
        {
            "question": "서울 택시의 기본요금은 얼마인가요?",
            "ground_truth": "서울 택시 기본요금은 일반택시 4,800원, 모범택시 7,000원입니다.",
            "question_type": "simple"
        }
    ],
    "multi_hop": [
        {
            "question": "강남역에서 서울역까지 지하철로 이동하려면 어떻게 가야하며 시간은 얼마나 걸리나요?",
            "ground_truth": "강남역(2호선)에서 교대역으로 이동 후 3호선으로 환승하여 충무로역에서 4호선으로 환승 후 서울역 도착. 약 25-30분 소요.",
            "question_type": "multi_reasoning"
        },
        {
            "question": "기후동행카드와 일반 교통카드를 사용할 때 한 달 30일 동안 매일 지하철 2회, 버스 2회 이용시 비용 차이는?",
            "ground_truth": "일반 교통카드: (1,400원×2 + 1,500원×2) × 30 = 174,000원, 기후동행카드: 62,000원, 차이: 112,000원 절약",
            "question_type": "multi_reasoning"
        },
        {
            "question": "출퇴근 시간대 강남역의 혼잡도와 대안 경로를 제시해주세요.",
            "ground_truth": "강남역은 출퇴근 시간 혼잡도가 매우 높음. 대안: 신분당선 이용, 9호선 급행 활용, 시간대 조정(7시 이전, 10시 이후)",
            "question_type": "multi_reasoning"
        },
        {
            "question": "서울역에서 인천공항까지 가는 가장 경제적인 방법과 가장 빠른 방법을 비교해주세요.",
            "ground_truth": "경제적: 공항철도 일반열차(4,150원, 약 60분), 빠른: 공항철도 직통열차(9,500원, 약 43분) 또는 택시(약 70,000원, 40-60분)",
            "question_type": "multi_reasoning"
        },
        {
            "question": "야간에 강남에서 종로까지 이동해야 할 때 가능한 교통수단과 비용을 알려주세요.",
            "ground_truth": "심야버스 N13, N16 (2,150원), 택시 (약 15,000-20,000원), 따릉이 (1,000원 + 추가요금)",
            "question_type": "multi_reasoning"
        }
    ]
}


class ExperimentManager:
    """실험 ID 및 체크포인트 관리"""

    def __init__(self, base_dir: Path = None):
        if base_dir is None:
            base_dir = Path("data/evaluation/experiments")

        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = self.base_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_file = self.base_dir / "metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        """메타데이터 로드"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        else:
            return {
                "next_id": 1,
                "experiments": {}
            }

    def _save_metadata(self):
        """메타데이터 저장"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def create_experiment(self, config: dict) -> int:
        """새 실험 생성"""
        exp_id = self.metadata["next_id"]
        self.metadata["next_id"] += 1

        self.metadata["experiments"][str(exp_id)] = {
            "id": exp_id,
            "created_at": datetime.now().isoformat(),
            "config": config,
            "status": "initialized",
            "completed_steps": []
        }

        self._save_metadata()
        logger.info(f"🆕 새 실험 생성: ID={exp_id}")
        return exp_id

    def get_experiment(self, exp_id: int) -> Optional[dict]:
        """실험 정보 조회"""
        return self.metadata["experiments"].get(str(exp_id))

    def update_status(self, exp_id: int, status: str):
        """실험 상태 업데이트"""
        if str(exp_id) in self.metadata["experiments"]:
            self.metadata["experiments"][str(exp_id)]["status"] = status
            self._save_metadata()

    def add_completed_step(self, exp_id: int, step: str):
        """완료된 단계 추가"""
        if str(exp_id) in self.metadata["experiments"]:
            self.metadata["experiments"][str(exp_id)]["completed_steps"].append(step)
            self._save_metadata()

    def save_checkpoint(self, exp_id: int, step: str, data: Any):
        """체크포인트 저장"""
        checkpoint_file = self.checkpoint_dir / f"exp_{exp_id}_{step}.pkl"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"💾 체크포인트 저장: {step}")

    def load_checkpoint(self, exp_id: int, step: str) -> Optional[Any]:
        """체크포인트 로드"""
        checkpoint_file = self.checkpoint_dir / f"exp_{exp_id}_{step}.pkl"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"📂 체크포인트 로드: {step}")
            return data
        return None


class SimplifiedBenchmark:
    """RAGAS 없이 동작하는 간소화된 벤치마크"""

    def __init__(self, args):
        self.args = args
        self.exp_manager = ExperimentManager()
        self.questions = []
        self.results = {}

    def run(self):
        """메인 실행"""
        if self.args.resume_id:
            self._resume_experiment()
        else:
            self._new_experiment()

    def _new_experiment(self):
        """새 실험 실행"""
        # 실험 생성
        config = {
            "questions": self.args.questions,
            "models": self.args.models,
            "skip_benchmark": self.args.skip_benchmark
        }
        self.exp_id = self.exp_manager.create_experiment(config)

        logger.info("="*70)
        logger.info(f"🧪 실험 ID: {self.exp_id}")
        logger.info("="*70)

        # 단계별 실행
        self._run_steps_from(start_step="questions")

    def _resume_experiment(self):
        """실험 재개"""
        self.exp_id = self.args.resume_id
        exp = self.exp_manager.get_experiment(self.exp_id)

        if not exp:
            logger.error(f"❌ 실험 ID {self.exp_id}를 찾을 수 없습니다.")
            return

        logger.info("="*70)
        logger.info(f"🔄 실험 재개: ID={self.exp_id}")
        logger.info(f"📊 완료된 단계: {exp['completed_steps']}")
        logger.info("="*70)

        # 완료된 단계 확인 후 다음부터 실행
        completed = exp['completed_steps']

        if "questions" not in completed:
            self._run_steps_from("questions")
        elif "classification" not in completed:
            # 질문 로드
            self.questions = self.exp_manager.load_checkpoint(self.exp_id, "questions")
            self._run_steps_from("classification")
        elif not any(s.startswith("model_") for s in completed):
            # 질문 로드
            self.questions = self.exp_manager.load_checkpoint(self.exp_id, "questions")
            self._run_steps_from("models")
        else:
            logger.info("✅ 모든 단계가 완료되었습니다.")

    def _run_steps_from(self, start_step: str):
        """지정된 단계부터 실행"""
        steps = ["questions", "classification", "models", "analysis"]
        start_idx = steps.index(start_step)

        for step in steps[start_idx:]:
            if step == "questions":
                self._generate_questions()
            elif step == "classification":
                self._classify_questions()
            elif step == "models":
                self._run_models()
            elif step == "analysis":
                self._analyze_results()

    def _generate_questions(self):
        """질문 생성 (미리 정의된 질문 사용)"""
        logger.info("="*70)
        logger.info("📝 질문 생성")
        logger.info("="*70)

        num_questions = self.args.questions
        num_single = num_questions // 2
        num_multi = num_questions - num_single

        # 미리 정의된 질문에서 선택
        single_hop = PREDEFINED_QUESTIONS["single_hop"][:num_single]
        multi_hop = PREDEFINED_QUESTIONS["multi_hop"][:num_multi]

        self.questions = single_hop + multi_hop

        logger.info(f"✅ {len(self.questions)}개 질문 생성 완료")
        logger.info(f"  - Single-hop: {len(single_hop)}개")
        logger.info(f"  - Multi-hop: {len(multi_hop)}개")

        # 체크포인트 저장
        self.exp_manager.save_checkpoint(self.exp_id, "questions", self.questions)
        self.exp_manager.add_completed_step(self.exp_id, "questions")

    def _classify_questions(self):
        """질문 분류"""
        logger.info("="*70)
        logger.info("🏷️ 질문 분류")
        logger.info("="*70)

        # 이미 question_type이 포함되어 있음
        single_count = sum(1 for q in self.questions if q["question_type"] == "simple")
        multi_count = sum(1 for q in self.questions if q["question_type"] == "multi_reasoning")

        logger.info(f"✅ 분류 완료:")
        logger.info(f"  - Single-hop: {single_count}개")
        logger.info(f"  - Multi-hop: {multi_count}개")

        self.exp_manager.add_completed_step(self.exp_id, "classification")

    def _run_models(self):
        """모델 실행"""
        logger.info("="*70)
        logger.info("🤖 모델 실행")
        logger.info("="*70)

        if self.args.skip_benchmark:
            logger.info("⏭️ 벤치마크 스킵 (--skip-benchmark)")
            for model in self.args.models:
                self.exp_manager.add_completed_step(self.exp_id, f"model_{model}")
            return

        for model in self.args.models:
            logger.info(f"\n🔄 모델 실행: {model}")

            # 여기서 실제 모델 실행 (생략)
            logger.info(f"  ⏭️ 시뮬레이션 모드")

            # 체크포인트 저장
            model_results = {
                "model": model,
                "results": []  # 실제 결과
            }
            self.exp_manager.save_checkpoint(self.exp_id, f"model_{model}", model_results)
            self.exp_manager.add_completed_step(self.exp_id, f"model_{model}")

    def _analyze_results(self):
        """결과 분석"""
        logger.info("="*70)
        logger.info("📊 결과 분석")
        logger.info("="*70)

        logger.info("✅ 분석 완료 (시뮬레이션)")

        self.exp_manager.add_completed_step(self.exp_id, "analysis")
        self.exp_manager.update_status(self.exp_id, "completed")

        logger.info(f"\n🎉 실험 완료: ID={self.exp_id}")


def main():
    parser = argparse.ArgumentParser(description="Unified RAG Benchmark v3 - Simplified")

    parser.add_argument(
        "--questions",
        type=int,
        default=5,
        help="생성할 질문 개수 (기본: 5)"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=["gpt-4o-mini"],
        help="평가할 모델 목록"
    )

    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="벤치마크 실행 스킵 (질문 생성만)"
    )

    parser.add_argument(
        "--resume-id",
        type=int,
        help="재개할 실험 ID"
    )

    args = parser.parse_args()

    # 설정 출력
    print("="*70)
    print("Configuration".center(70))
    print("="*70)

    if args.resume_id:
        print(f"Resume Experiment: ID={args.resume_id}")
    else:
        print("New Experiment")
        print(f"Questions: {args.questions}")
        print(f"Models: {args.models}")
        print(f"Skip Benchmark: {args.skip_benchmark}")

    # 벤치마크 실행
    benchmark = SimplifiedBenchmark(args)
    benchmark.run()


if __name__ == "__main__":
    main()