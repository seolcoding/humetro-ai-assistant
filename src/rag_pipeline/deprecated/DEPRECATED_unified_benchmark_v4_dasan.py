#!/usr/bin/env python3
"""
⚠️ DEPRECATED - DO NOT USE ⚠️
================================

이 파일은 더 이상 사용되지 않습니다.

Deprecated Date: 2025-11-11
Reason: 실제 Q&A 데이터 사용으로 질문 생성 불필요

대체 사용:
  python src/rag_pipeline/unified_benchmark_v4_real_qa.py
  from src.data_loader.dasan_qa_sampler import DasanQASampler

자세한 내용: src/rag_pipeline/deprecated/README.md

================================
"""

import warnings
warnings.warn(
    "⚠️ DEPRECATED: This file is no longer maintained. "
    "Use 'unified_benchmark_v4_real_qa.py' or 'DasanQASampler' instead. "
    "See deprecated/README.md for migration guide.",
    DeprecationWarning,
    stacklevel=2
)

"""
통합 RAG 벤치마크 v4 - Dasan 콜센터 데이터 기반 평가
========================================

v4 새로운 기능:
1. Dasan 콜센터 실제 대화 데이터 기반 벤치마크
2. 10,757개 구조화된 Q&A 마크다운 문서 활용
3. 카테고리별 평가 지원 (교통, 코로나19, 상수도 등)
4. v3의 병렬 평가 시스템 유지

데이터 특성:
- 출처: AI Hub 다산콜센터 상담 데이터
- 형식: YAML frontmatter + 구조화된 Q&A
- 카테고리: 교통, 행정, 복지, 의료, 환경 등
- 메타데이터: entities, kb_tags, topics 포함

실행 예시:
  # 기본 실행 (50 questions, 5 models)
  python unified_benchmark_v4_dasan.py --questions 50 --models thesis

  # 특정 카테고리만 평가
  python unified_benchmark_v4_dasan.py --category 교통 --questions 30

  # 빠른 테스트
  python unified_benchmark_v4_dasan.py --questions 10 --models gpt-4o-mini
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


class DasanBenchmarkV4:
    """Dasan 콜센터 데이터 기반 병렬 평가 벤치마크 시스템"""

    # AI Hub Dasan QA 데이터셋 기본 설정
    DASAN_BASE_PATH = Path("data/AI_HUB_DASAN_QA/03_markdown_full")
    DASAN_VECTOR_STORE = Path("data/AI_HUB_DASAN_QA/07_vector_stores/full")

    # 지원 카테고리
    CATEGORIES = {
        "all": "전체",
        "교통": "교통 (대중교통, 차량, 교통법규)",
        "코로나19": "코로나19_관련_상담",
        "상수도": "상수도/수도 관련",
        "행정": "일반행정/민원",
        "복지": "보건_복지/복지_서비스",
        "환경": "생활환경/환경_위생"
    }

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
                "dataset": "dasan_call_center",
                "category": args.category,
                "questions": args.questions,
                "models": args.models,
                "judge_model": args.judge_model,
                "k_documents": args.k_documents,
                "parallel": True,
                "version": "v4"
            }
            self.exp_id = self.exp_manager.create_experiment(config)
            self.experiment = self.exp_manager.get_experiment(self.exp_id)

        # Document source (카테고리별 선택 지원)
        self.document_source = self._get_document_source()

        # Initialize retriever
        self.retriever = self._initialize_retrieval() if not args.no_retrieval else None

        self.results = {
            "experiment_id": self.exp_id,
            "timestamp": datetime.now().isoformat(),
            "dataset": "dasan_call_center",
            "category": args.category,
            "config": vars(args)
        }

    def _get_document_source(self) -> Path:
        """카테고리에 따른 문서 소스 경로 결정"""
        if self.args.category == "all":
            return self.DASAN_BASE_PATH

        # 카테고리별 경로
        category_map = {
            "교통": self.DASAN_BASE_PATH / "교통",
            "코로나19": self.DASAN_BASE_PATH / "코로나19_관련_상담",
            "상수도": self.DASAN_BASE_PATH / "상수도",
            "행정": self.DASAN_BASE_PATH / "일반행정",
            "복지": self.DASAN_BASE_PATH / "복지",
            "환경": self.DASAN_BASE_PATH / "생활환경"
        }

        category_path = category_map.get(self.args.category, self.DASAN_BASE_PATH)

        if not category_path.exists():
            logger.warning(f"⚠️ 카테고리 경로 없음: {category_path}, 전체 데이터 사용")
            return self.DASAN_BASE_PATH

        logger.info(f"📂 문서 소스: {category_path}")
        return category_path

    def _initialize_retrieval(self):
        """벡터 스토어 및 리트리버 초기화"""
        if not self.DASAN_VECTOR_STORE.exists():
            logger.error("❌ Dasan 벡터 스토어를 찾을 수 없습니다")
            logger.info("💡 벡터 스토어 생성 방법:")
            logger.info("   python src/rag_pipeline/stages/stage_05_vector_store.py \\")
            logger.info(f"     --input-dir {self.DASAN_BASE_PATH} \\")
            logger.info(f"     --output-dir {self.DASAN_VECTOR_STORE} \\")
            logger.info("     --model text-embedding-3-large")
            return None

        try:
            logger.info("📚 Dasan 벡터 스토어 로드 중...")
            vector_store_stage = VectorStoreStage(model="text-embedding-3-large")
            vector_store_stage.load_vector_store(self.DASAN_VECTOR_STORE)

            retriever = vector_store_stage.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.args.k_documents}
            )

            logger.info(f"  ✅ Retriever 초기화 완료 (k={self.args.k_documents})")
            logger.info(f"  📊 데이터: Dasan 콜센터 10,757개 문서")
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
        logger.info("📝 Dasan 데이터 기반 질문 생성")
        logger.info("="*70)

        config, testset_df = generate_questions(
            model=self.args.generation_model,
            source=str(self.document_source),
            num_documents=self.args.num_documents,
            num_questions=self.args.questions,
            force_regenerate=True  # Dasan 데이터는 항상 새로 생성
        )

        questions = testset_df.to_dict('records')
        self.exp_manager.save_checkpoint(self.exp_id, step, questions)

        logger.info(f"  ✅ {len(questions)}개 질문 생성 완료")
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
        logger.info(f"🧪 실험 ID: {self.exp_id} (Dasan v4)")
        logger.info(f"📂 카테고리: {self.args.category}")
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
        results_file = self.output_dir / f"experiment_{self.exp_id}_dasan_v4_results.json"

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
        description="Dasan 콜센터 데이터 기반 RAG 벤치마크 시스템 v4",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # 실험 관리
    parser.add_argument("--resume-id", type=int, help="재시작할 실험 ID")

    # 데이터 설정
    parser.add_argument(
        "--category",
        default="all",
        choices=list(DasanBenchmarkV4.CATEGORIES.keys()),
        help="평가할 카테고리 선택"
    )

    # 질문 생성
    parser.add_argument("--questions", "-q", type=int, default=50, help="질문 수")
    parser.add_argument("--generation-model", default="gpt-4o-mini", help="질문 생성 모델")
    parser.add_argument("--num-documents", type=int, default=200, help="문서 수 (Dasan 기본: 200)")

    # 모델 설정
    parser.add_argument("--models", nargs="+", default=["thesis"], help="평가 모델")

    # RAG 설정
    parser.add_argument("--no-retrieval", action="store_true", help="Retrieval 없이 실행")
    parser.add_argument("--k-documents", type=int, default=4, help="검색 문서 수")

    # 평가 설정
    parser.add_argument("--judge-model", default="gpt-5", help="평가 모델")
    parser.add_argument("--max-concurrent", type=int, default=5, help="최대 동시 평가 수")

    # 출력 설정
    parser.add_argument("--output-dir", default="data/evaluation/dasan_benchmark_v4")
    parser.add_argument("--verbose", action="store_true", help="상세 로깅")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 설정 출력
    print("="*70)
    print(" "*15 + "Dasan 벤치마크 v4")
    print("="*70)
    if args.resume_id:
        print(f"Resume ID: {args.resume_id}")
    else:
        print("New Experiment (Dasan 콜센터 데이터)")
    print(f"Category: {args.category} - {DasanBenchmarkV4.CATEGORIES[args.category]}")
    print(f"Questions: {args.questions}")
    print(f"Models: {args.models}")
    print(f"Judge: {args.judge_model}")
    print(f"Max Concurrent: {args.max_concurrent}")
    print(f"Output: {args.output_dir}")
    print()

    # 실행
    benchmark = DasanBenchmarkV4(args)
    benchmark.run()


if __name__ == "__main__":
    main()
