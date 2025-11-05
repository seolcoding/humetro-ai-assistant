#!/usr/bin/env python3
"""
통합 RAG 벤치마크 시스템 v2 - 체크포인트 지원
- 실험마다 고유 ID 부여
- 중단 시 재시작 가능
- 체크포인트 자동 저장
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib
import pickle

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Add tests path for testset_generator
sys.path.insert(0, str(project_root / "tests" / "rag_pipeline"))

from generation_benchmark import GenerationBenchmark
from testset_generator import TestsetGenerator
from question_classifier import QuestionComplexityClassifier

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Model configurations (config/models.yaml 기준)
MODEL_CONFIGS = {
    # OpenAI models
    "gpt-4o": {
        "name": "GPT-4o",
        "model": "gpt-4o",
        "api_base": None,
        "provider": "openai"
    },
    "gpt-4o-mini": {
        "name": "GPT-4o-mini",
        "model": "gpt-4o-mini",
        "api_base": None,
        "provider": "openai"
    },

    # Ollama models (논문용)
    "ollama/exaone3.5:7.8b": {
        "name": "EXAONE-3.5-7.8B",
        "model": "ollama/exaone3.5:7.8b",
        "api_base": "http://100.95.220.92:11434",
        "provider": "ollama"
    },
    "ollama/qwen3:8b": {
        "name": "Qwen3-8B",
        "model": "ollama/qwen3:8b",
        "api_base": "http://100.95.220.92:11434",
        "provider": "ollama"
    },
    "ollama/gemma3:12b": {
        "name": "Gemma3-12B",
        "model": "ollama/gemma3:12b",
        "api_base": "http://100.95.220.92:11434",
        "provider": "ollama"
    },
    "ollama/gpt-oss:20b": {
        "name": "GPT-OSS-20B",
        "model": "ollama/gpt-oss:20b",
        "api_base": "http://100.95.220.92:11434",
        "provider": "ollama"
    }
}

# Model groups
MODEL_GROUPS = {
    "all": list(MODEL_CONFIGS.keys()),
    "openai": ["gpt-4o", "gpt-4o-mini"],
    "ollama": [k for k in MODEL_CONFIGS.keys() if k.startswith("ollama/")],
    "thesis": ["gpt-4o-mini", "ollama/exaone3.5:7.8b", "ollama/qwen3:8b", "ollama/gemma3:12b", "ollama/gpt-oss:20b"],
    "korean": ["ollama/exaone3.5:7.8b"],
    "fast": ["gpt-4o-mini", "ollama/qwen3:8b"]
}


class ExperimentManager:
    """실험 ID 및 체크포인트 관리"""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.checkpoint_dir / "experiments.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        """실험 메타데이터 로드"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"next_id": 1, "experiments": {}}

    def _save_metadata(self):
        """메타데이터 저장"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def create_experiment(self, config: dict) -> int:
        """새 실험 생성 및 ID 반환"""
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
        """실험 정보 가져오기"""
        return self.metadata["experiments"].get(str(exp_id))

    def update_experiment(self, exp_id: int, updates: dict):
        """실험 상태 업데이트"""
        if str(exp_id) in self.metadata["experiments"]:
            self.metadata["experiments"][str(exp_id)].update(updates)
            self._save_metadata()

    def save_checkpoint(self, exp_id: int, step: str, data: Any):
        """체크포인트 저장"""
        checkpoint_file = self.checkpoint_dir / f"exp_{exp_id}_{step}.pkl"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(data, f)

        # 완료된 단계 기록
        exp = self.metadata["experiments"].get(str(exp_id))
        if exp and step not in exp["completed_steps"]:
            exp["completed_steps"].append(step)
            exp["last_checkpoint"] = step
            exp["last_updated"] = datetime.now().isoformat()
            self._save_metadata()

        logger.info(f"💾 체크포인트 저장: {step}")

    def load_checkpoint(self, exp_id: int, step: str) -> Optional[Any]:
        """체크포인트 로드"""
        checkpoint_file = self.checkpoint_dir / f"exp_{exp_id}_{step}.pkl"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'rb') as f:
                return pickle.load(f)
        return None


class UnifiedBenchmarkV2:
    """통합 벤치마크 시스템 v2 - 체크포인트 지원"""

    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 체크포인트 관리
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.exp_manager = ExperimentManager(self.checkpoint_dir)

        # 실험 ID 설정
        if args.resume_id:
            self.exp_id = args.resume_id
            self.experiment = self.exp_manager.get_experiment(self.exp_id)
            if not self.experiment:
                raise ValueError(f"실험 ID {self.exp_id}를 찾을 수 없습니다")
            logger.info(f"🔄 실험 재개: ID={self.exp_id}")
            logger.info(f"   완료된 단계: {self.experiment['completed_steps']}")
        else:
            # 새 실험 생성
            config = {
                "questions": args.questions,
                "models": args.models,
                "judge_model": args.judge_model,
                "no_retrieval": args.no_retrieval,
                "k_documents": args.k_documents
            }
            self.exp_id = self.exp_manager.create_experiment(config)
            self.experiment = self.exp_manager.get_experiment(self.exp_id)

        self.timestamp = datetime.now().isoformat()
        self.results = {
            "experiment_id": self.exp_id,
            "timestamp": self.timestamp,
            "config": vars(args)
        }

    def is_step_completed(self, step: str) -> bool:
        """단계가 이미 완료되었는지 확인"""
        return step in self.experiment.get("completed_steps", [])

    def load_or_generate_questions(self) -> dict:
        """질문 생성 또는 체크포인트에서 로드"""
        step = "questions_generated"

        # 이미 완료된 경우 체크포인트에서 로드
        if self.is_step_completed(step):
            logger.info(f"✅ '{step}' 단계 이미 완료 - 체크포인트에서 로드")
            return self.exp_manager.load_checkpoint(self.exp_id, step)

        logger.info("="*70)
        logger.info("📝 질문 생성")
        logger.info("="*70)

        # TestsetGenerator 사용
        generator = TestsetGenerator(
            num_questions=self.args.questions,
            use_cache=not self.args.force_generate,
            model_name=self.args.generation_model,
            num_documents=self.args.num_documents,
            document_source=self.args.document_source
        )

        questions = generator.generate()

        # 체크포인트 저장
        self.exp_manager.save_checkpoint(self.exp_id, step, questions)

        return questions

    def classify_questions(self, questions: dict) -> dict:
        """질문 분류 또는 체크포인트에서 로드"""
        step = "questions_classified"

        if self.is_step_completed(step):
            logger.info(f"✅ '{step}' 단계 이미 완료 - 체크포인트에서 로드")
            return self.exp_manager.load_checkpoint(self.exp_id, step)

        logger.info("="*70)
        logger.info("🏷️ 질문 분류")
        logger.info("="*70)

        classifier = QuestionComplexityClassifier(model="gpt-4o-mini")
        classified = classifier.classify_batch(questions)

        # 체크포인트 저장
        self.exp_manager.save_checkpoint(self.exp_id, step, classified)

        return classified

    def run_benchmark_for_model(self, model_key: str, questions: List[dict]) -> dict:
        """개별 모델 벤치마크 실행"""
        step = f"benchmark_{model_key}"

        if self.is_step_completed(step):
            logger.info(f"✅ 모델 '{model_key}' 이미 완료 - 건너뜀")
            return self.exp_manager.load_checkpoint(self.exp_id, step)

        logger.info(f"\n🤖 모델 테스트: {model_key}")

        try:
            model_config = MODEL_CONFIGS[model_key]

            # GenerationBenchmark 설정
            benchmark = GenerationBenchmark(
                models=[model_config],
                use_fixed_context=not self.args.no_retrieval,
                k_documents=self.args.k_documents,
                judge_model=self.args.judge_model
            )

            # 벤치마크 실행
            results = benchmark.run_benchmark(questions)

            # 체크포인트 저장
            self.exp_manager.save_checkpoint(self.exp_id, step, results)

            logger.info(f"✅ {model_key} 완료")
            return results

        except Exception as e:
            logger.error(f"❌ {model_key} 실패: {e}")
            # 실패해도 계속 진행
            return {"error": str(e)}

    def run_benchmark(self, questions: List[dict]) -> dict:
        """전체 벤치마크 실행"""
        logger.info("="*70)
        logger.info("🚀 벤치마크 실행")
        logger.info("="*70)

        benchmark_results = {}

        # 모델별로 실행 (체크포인트 지원)
        for model_key in self.parse_models():
            result = self.run_benchmark_for_model(model_key, questions)
            benchmark_results[model_key] = result

        return benchmark_results

    def parse_models(self) -> List[str]:
        """모델 목록 파싱"""
        models = []

        for model_arg in self.args.models:
            if model_arg in MODEL_GROUPS:
                # 그룹 확장
                models.extend(MODEL_GROUPS[model_arg])
            elif model_arg in MODEL_CONFIGS:
                # 개별 모델
                models.append(model_arg)
            else:
                logger.warning(f"⚠️ 알 수 없는 모델: {model_arg}")

        # 중복 제거
        return list(dict.fromkeys(models))

    def save_final_results(self):
        """최종 결과 저장"""
        step = "final_results"

        # 결과 파일 경로 (실험 ID 포함)
        results_file = self.output_dir / f"experiment_{self.exp_id}_results.json"

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        logger.info(f"📊 최종 결과 저장: {results_file}")

        # 실험 상태 업데이트
        self.exp_manager.update_experiment(self.exp_id, {
            "status": "completed",
            "results_file": str(results_file),
            "completed_at": datetime.now().isoformat()
        })

    def run(self):
        """전체 파이프라인 실행"""
        logger.info("="*70)
        logger.info(f"🧪 실험 ID: {self.exp_id}")
        logger.info("="*70)

        try:
            # 1. 질문 생성/로드
            questions = self.load_or_generate_questions()
            self.results["questions"] = questions

            # 2. 질문 분류
            classified = self.classify_questions(questions)
            self.results["classified"] = classified

            # 3. 질문 밸런싱
            all_questions = []
            if "single_hop" in classified:
                all_questions.extend(classified["single_hop"][:self.args.questions//2])
            if "multi_hop" in classified:
                all_questions.extend(classified["multi_hop"][:self.args.questions//2])

            logger.info(f"📊 최종 질문 수: {len(all_questions)}개")

            # 4. 벤치마크 실행 (스킵 옵션 확인)
            if not self.args.skip_benchmark:
                benchmark_results = self.run_benchmark(all_questions)
                self.results["benchmark"] = benchmark_results
            else:
                logger.info("⏭️ 벤치마크 실행 건너뜀 (--skip-benchmark)")

            # 5. 최종 결과 저장
            self.save_final_results()

            logger.info("="*70)
            logger.info(f"✅ 실험 완료! (ID: {self.exp_id})")
            logger.info("="*70)

        except KeyboardInterrupt:
            logger.warning("\n⚠️ 사용자 중단 - 체크포인트 저장됨")
            logger.info(f"재시작: python {__file__} --resume-id {self.exp_id}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"❌ 실험 실패: {e}")
            logger.info(f"재시작: python {__file__} --resume-id {self.exp_id}")
            raise


def main():
    parser = argparse.ArgumentParser(
        description="통합 RAG 벤치마크 시스템 v2 - 체크포인트 지원",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # 실험 관리
    parser.add_argument(
        "--resume-id",
        type=int,
        help="재시작할 실험 ID (지정 안하면 새 실험)"
    )

    # 질문 생성
    parser.add_argument(
        "--questions", "-q",
        type=int,
        default=50,
        help="생성/평가할 질문 수 (default: 50)"
    )

    parser.add_argument(
        "--force-generate",
        action="store_true",
        help="캐시 무시하고 강제로 재생성"
    )

    parser.add_argument(
        "--generation-model",
        default="gpt-4o-mini",
        help="질문 생성 모델 (default: gpt-4o-mini)"
    )

    parser.add_argument(
        "--num-documents",
        type=int,
        default=100,
        help="질문 생성에 사용할 문서 수 (default: 100)"
    )

    parser.add_argument(
        "--document-source",
        default="data/crawled/seoul_traffic/markdown_deduplicated",
        help="문서 소스 디렉토리"
    )

    # 모델 설정
    parser.add_argument(
        "--models",
        nargs="+",
        default=["thesis"],
        help="평가할 모델 목록 또는 그룹 (thesis, all, openai, ollama, korean, fast)"
    )

    # 분류 설정
    parser.add_argument(
        "--classification-method",
        choices=["llm", "rules", "hybrid"],
        default="hybrid",
        help="질문 분류 방법 (default: hybrid)"
    )

    # RAG 설정
    parser.add_argument(
        "--no-retrieval",
        action="store_true",
        help="Retrieval 없이 LLM만 사용"
    )

    parser.add_argument(
        "--k-documents",
        type=int,
        default=4,
        help="검색할 문서 수 (default: 4)"
    )

    # 평가 설정
    parser.add_argument(
        "--judge-model",
        default="gpt-5",
        help="RAGAS 평가 모델 (default: gpt-5)"
    )

    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="벤치마크 실행 건너뛰기 (질문만 생성/분류)"
    )

    # 출력 설정
    parser.add_argument(
        "--output-dir",
        default="data/evaluation/unified_benchmark",
        help="출력 디렉토리 (default: data/evaluation/unified_benchmark)"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="상세 로깅"
    )

    args = parser.parse_args()

    # 로깅 레벨 설정
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 설정 출력
    print("="*70)
    print(" "*20 + "Configuration")
    print("="*70)
    if args.resume_id:
        print(f"Resume ID: {args.resume_id}")
    else:
        print("New Experiment")
    print(f"Questions: {args.questions}")
    print(f"Models: {args.models}")
    print(f"Retrieval: {'Disabled' if args.no_retrieval else 'Enabled'}")
    print(f"Judge: {args.judge_model}")
    print(f"Output: {args.output_dir}")
    print()

    # 실행
    benchmark = UnifiedBenchmarkV2(args)
    benchmark.run()


if __name__ == "__main__":
    main()