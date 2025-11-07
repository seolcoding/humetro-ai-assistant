#!/usr/bin/env python3
"""
통합 RAG 벤치마크 v3 - 병렬 평가 지원
========================================

새로운 기능:
1. Answer Generation과 Evaluation 분리
2. 병렬 평가로 5x 속도 향상
3. 체크포인트 시스템 유지

실행 흐름:
- Phase 1: 답변 생성 (순차, 모델별)
- Phase 2: 병렬 평가 (동시, 5개 모델)

성능 개선:
- 기존: 5 models × 10 min = 50 min
- 신규: 10 min (generation) + 12 min (parallel evaluation) = 22 min
- 개선: ~2.3x faster
"""

import sys
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

# Import modules
from unified_benchmark_v2 import (
    ExperimentManager,
    MODEL_CONFIGS,
    MODEL_GROUPS
)
from question_generation import generate_questions
from answer_generator import AnswerGenerator
from src.evaluation.parallel_evaluator import run_parallel_evaluation
from src.rag_pipeline.stages.stage_05_vector_store import VectorStoreStage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class ParallelBenchmarkV3:
    """병렬 평가 지원 벤치마크 시스템"""

    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Checkpoint management
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.exp_manager = ExperimentManager(self.checkpoint_dir)

        # Experiment ID
        if args.resume_id:
            self.exp_id = args.resume_id
            self.experiment = self.exp_manager.get_experiment(self.exp_id)
            if not self.experiment:
                raise ValueError(f"실험 ID {self.exp_id}를 찾을 수 없습니다")
            logger.info(f"🔄 실험 재개: ID={self.exp_id}")
        else:
            config = {
                "questions": args.questions,
                "models": args.models,
                "judge_model": args.judge_model,
                "k_documents": args.k_documents,
                "parallel": True  # New flag
            }
            self.exp_id = self.exp_manager.create_experiment(config)
            self.experiment = self.exp_manager.get_experiment(self.exp_id)

        # Initialize retriever
        self.retriever = self._initialize_retrieval() if not args.no_retrieval else None

        self.results = {
            "experiment_id": self.exp_id,
            "timestamp": datetime.now().isoformat(),
            "config": vars(args)
        }

    def _initialize_retrieval(self):
        """벡터 스토어 및 리트리버 초기화"""
        vector_store_path = Path("data/vector_store/seoul_traffic")

        if not vector_store_path.exists():
            logger.warning("⚠️ 벡터 스토어를 찾을 수 없음")
            return None

        try:
            logger.info("📚 벡터 스토어 로드 중...")
            vector_store_stage = VectorStoreStage(model="text-embedding-3-large")
            vector_store_stage.load_vector_store(vector_store_path)

            retriever = vector_store_stage.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.args.k_documents}
            )

            logger.info(f"  ✅ Retriever 초기화 완료 (k={self.args.k_documents})")
            return retriever

        except Exception as e:
            logger.error(f"❌ 리트리버 초기화 실패: {e}")
            return None

    def is_step_completed(self, step: str) -> bool:
        """단계 완료 여부 확인"""
        return step in self.experiment.get("completed_steps", [])

    def load_or_generate_questions(self) -> dict:
        """질문 생성 또는 로드"""
        step = "questions_generated"

        if self.is_step_completed(step):
            logger.info(f"✅ '{step}' 단계 이미 완료 - 로드")
            return self.exp_manager.load_checkpoint(self.exp_id, step)

        logger.info("="*70)
        logger.info("📝 질문 생성")
        logger.info("="*70)

        config, testset_df = generate_questions(
            model=self.args.generation_model,
            source=self.args.document_source,
            num_documents=self.args.num_documents,
            num_questions=self.args.questions,
            force_regenerate=self.args.force_generate
        )

        questions = testset_df.to_dict('records')
        self.exp_manager.save_checkpoint(self.exp_id, step, questions)

        return questions

    def classify_questions(self, questions: List[dict]) -> List[dict]:
        """질문 분류 (RAGAS synthesizer 기반)"""
        step = "questions_classified"

        if self.is_step_completed(step):
            logger.info(f"✅ '{step}' 단계 이미 완료 - 로드")
            return self.exp_manager.load_checkpoint(self.exp_id, step)

        logger.info("="*70)
        logger.info("🏷️ 질문 분류")
        logger.info("="*70)

        classified_results = []
        for original_q in questions:
            synthesizer = original_q.get("synthesizer_name", "")

            classification = "single_hop" if "single_hop" in synthesizer else "multi_hop"

            classified_results.append({
                "question": original_q.get("user_input", ""),
                "ground_truth": original_q.get("reference", ""),
                "reference_contexts": original_q.get("reference_contexts", []),
                "classification": classification,
                "synthesizer_name": synthesizer
            })

        self.exp_manager.save_checkpoint(self.exp_id, step, classified_results)
        return classified_results

    def generate_answers(self, questions: List[dict]) -> Dict[str, Dict[str, List[str]]]:
        """
        Phase 1: 모든 모델의 답변 생성 (순차)
        """
        step = "answers_generated"

        if self.is_step_completed(step):
            logger.info(f"✅ '{step}' 단계 이미 완료 - 로드")
            return self.exp_manager.load_checkpoint(self.exp_id, step)

        logger.info("="*70)
        logger.info("Phase 1: 답변 생성")
        logger.info("="*70)

        # Parse models
        model_keys = self.parse_models()
        models = [MODEL_CONFIGS[key] for key in model_keys]

        # Initialize answer generator
        generator = AnswerGenerator(
            retriever=self.retriever,
            k_documents=self.args.k_documents
        )

        # Generate answers for all models
        all_datasets = generator.generate_all_answers(
            models=models,
            questions=questions,
            use_fixed_context=not self.args.no_retrieval
        )

        # Save checkpoint
        self.exp_manager.save_checkpoint(self.exp_id, step, all_datasets)

        return all_datasets

    def evaluate_answers(self, all_datasets: Dict[str, Dict[str, List[str]]]) -> Dict[str, Any]:
        """
        Phase 2: 병렬 평가 (동시 실행)
        """
        step = "evaluation_completed"

        if self.is_step_completed(step):
            logger.info(f"✅ '{step}' 단계 이미 완료 - 로드")
            return self.exp_manager.load_checkpoint(self.exp_id, step)

        logger.info("="*70)
        logger.info("Phase 2: 병렬 평가")
        logger.info("="*70)

        # Run parallel evaluation
        evaluation_results = run_parallel_evaluation(
            model_datasets=all_datasets,
            judge_model=self.args.judge_model,
            max_concurrent=self.args.max_concurrent,
            output_dir=self.output_dir,
            experiment_id=self.exp_id
        )

        # Save checkpoint
        self.exp_manager.save_checkpoint(self.exp_id, step, evaluation_results)

        return evaluation_results

    def parse_models(self) -> List[str]:
        """모델 목록 파싱"""
        models = []

        for model_arg in self.args.models:
            if model_arg in MODEL_GROUPS:
                models.extend(MODEL_GROUPS[model_arg])
            elif model_arg in MODEL_CONFIGS:
                models.append(model_arg)
            else:
                logger.warning(f"⚠️ 알 수 없는 모델: {model_arg}")

        return list(dict.fromkeys(models))

    def run(self):
        """전체 파이프라인 실행"""
        logger.info("="*70)
        logger.info(f"🧪 실험 ID: {self.exp_id} (병렬 평가)")
        logger.info("="*70)

        try:
            # 1. 질문 생성/로드
            questions = self.load_or_generate_questions()
            self.results["questions"] = questions

            # 2. 질문 분류
            classified = self.classify_questions(questions)
            self.results["classified"] = classified

            # 3. 질문 밸런싱
            single_hop = [q for q in classified if q["classification"] == "single_hop"]
            multi_hop = [q for q in classified if q["classification"] == "multi_hop"]

            questions_per_type = self.args.questions // 2
            selected = single_hop[:questions_per_type] + multi_hop[:questions_per_type]

            logger.info(f"📊 최종 질문 수: {len(selected)}개")
            logger.info(f"  - Single-hop: {len(single_hop[:questions_per_type])}개")
            logger.info(f"  - Multi-hop: {len(multi_hop[:questions_per_type])}개")

            # 4. Phase 1: 답변 생성
            all_datasets = self.generate_answers(selected)
            self.results["answers"] = all_datasets

            # 5. Phase 2: 병렬 평가
            evaluation_results = self.evaluate_answers(all_datasets)
            self.results["evaluation"] = evaluation_results

            # 6. 최종 결과 저장
            self._save_final_results()

            logger.info("="*70)
            logger.info(f"✅ 실험 완료! (ID: {self.exp_id})")
            logger.info("="*70)

            # Print performance summary
            metadata = evaluation_results.get("metadata", {})
            logger.info(f"\n병렬 평가 성능:")
            logger.info(f"  - 평가 시간: {metadata.get('elapsed_seconds', 0):.1f}초")
            logger.info(f"  - 동시 실행: {metadata.get('max_concurrent', 0)}개 모델")
            logger.info(f"  - 성공 모델: {metadata.get('successful_models', 0)}/{metadata.get('total_models', 0)}")

        except KeyboardInterrupt:
            logger.warning("\n⚠️ 사용자 중단 - 체크포인트 저장됨")
            logger.info(f"재시작: python {__file__} --resume-id {self.exp_id}")
            sys.exit(1)
        except Exception as e:
            logger.error(f"❌ 실험 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise

    def _save_final_results(self):
        """최종 결과 저장"""
        results_file = self.output_dir / f"experiment_{self.exp_id}_parallel_results.json"

        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)

        logger.info(f"📊 최종 결과 저장: {results_file}")

        self.exp_manager.update_experiment(self.exp_id, {
            "status": "completed",
            "results_file": str(results_file),
            "completed_at": datetime.now().isoformat()
        })


def main():
    parser = argparse.ArgumentParser(
        description="병렬 평가 지원 RAG 벤치마크 시스템 v3",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # 실험 관리
    parser.add_argument("--resume-id", type=int, help="재시작할 실험 ID")

    # 질문 생성
    parser.add_argument("--questions", "-q", type=int, default=50, help="질문 수")
    parser.add_argument("--force-generate", action="store_true", help="강제 재생성")
    parser.add_argument("--generation-model", default="gpt-4o-mini", help="질문 생성 모델")
    parser.add_argument("--num-documents", type=int, default=100, help="문서 수")
    parser.add_argument("--document-source", default="data/crawled/seoul_traffic/markdown_filtered")

    # 모델 설정
    parser.add_argument("--models", nargs="+", default=["thesis"], help="평가 모델")

    # RAG 설정
    parser.add_argument("--no-retrieval", action="store_true", help="Retrieval 없이 실행")
    parser.add_argument("--k-documents", type=int, default=4, help="검색 문서 수")

    # 평가 설정
    parser.add_argument("--judge-model", default="gpt-5", help="평가 모델")
    parser.add_argument("--max-concurrent", type=int, default=5, help="최대 동시 평가 수")

    # 출력 설정
    parser.add_argument("--output-dir", default="data/evaluation/parallel_benchmark")
    parser.add_argument("--verbose", action="store_true", help="상세 로깅")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 설정 출력
    print("="*70)
    print(" "*15 + "병렬 평가 벤치마크 v3")
    print("="*70)
    if args.resume_id:
        print(f"Resume ID: {args.resume_id}")
    else:
        print("New Experiment (병렬 평가)")
    print(f"Questions: {args.questions}")
    print(f"Models: {args.models}")
    print(f"Judge: {args.judge_model}")
    print(f"Max Concurrent: {args.max_concurrent}")
    print(f"Output: {args.output_dir}")
    print()

    # 실행
    benchmark = ParallelBenchmarkV3(args)
    benchmark.run()


if __name__ == "__main__":
    main()
