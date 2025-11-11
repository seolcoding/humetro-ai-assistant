#!/usr/bin/env python3
"""
통합 RAG 벤치마크 v4 - Real QA (실제 콜센터 Q&A 기반)
==========================================================

v4_real_qa 핵심 차이점:
1. RAGAS 질문 생성 단계 제거 (이미 실제 Q&A 데이터 존재)
2. knowledge_docs_full.jsonl에서 직접 샘플링
3. 실제 콜센터 질문으로 RAG 성능 평가

워크플로우:
1. JSONL에서 Q&A 샘플링 (랜덤 또는 균등)
2. 각 모델로 답변 생성 (RAG 활용)
3. RAGAS로 평가 (ground truth와 비교)

장점:
- 실제 사용자 질문 기반 평가
- 질문 생성 비용/시간 절약
- Ground truth 답변 품질 보장
"""

import sys
import os
import json
import argparse
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Import modules
from src.data_loader.dasan_qa_sampler import DasanQASampler
from src.rag_pipeline.answer_generator import AnswerGenerator
from src.evaluation.parallel_evaluator import run_parallel_evaluation
from src.rag_pipeline.stages.stage_05_vector_store import VectorStoreStage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Get Ollama base URL from environment variable
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://100.95.220.92:11434")

# Model configurations
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

    # Ollama models (논문용) - using OLLAMA_BASE_URL from env
    "ollama/exaone3.5:7.8b": {
        "name": "EXAONE-3.5-7.8B",
        "model": "ollama/exaone3.5:7.8b",
        "api_base": OLLAMA_BASE_URL,
        "provider": "ollama"
    },
    "ollama/qwen3:8b": {
        "name": "Qwen3-8B",
        "model": "ollama/qwen3:8b",
        "api_base": OLLAMA_BASE_URL,
        "provider": "ollama"
    },
    "ollama/gemma3:12b": {
        "name": "Gemma3-12B",
        "model": "ollama/gemma3:12b",
        "api_base": OLLAMA_BASE_URL,
        "provider": "ollama"
    },
    "ollama/gpt-oss:20b": {
        "name": "GPT-OSS-20B",
        "model": "ollama/gpt-oss:20b",
        "api_base": OLLAMA_BASE_URL,
        "provider": "ollama"
    },
    "ollama/deepseek-r1:7b": {
        "name": "DeepSeek-V3-7B",
        "model": "ollama/deepseek-r1:7b",
        "api_base": OLLAMA_BASE_URL,
        "provider": "ollama"
    }
}

# Model groups
MODEL_GROUPS = {
    "all": list(MODEL_CONFIGS.keys()),
    "openai": ["gpt-4o", "gpt-4o-mini"],
    "ollama": [k for k in MODEL_CONFIGS.keys() if k.startswith("ollama/")],
    "thesis": ["gpt-4o-mini", "ollama/exaone3.5:7.8b", "ollama/gpt-oss:20b", "ollama/gemma3:12b", "ollama/deepseek-r1:7b"],
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


class RealQABenchmarkV4:
    """실제 Q&A 데이터 기반 벤치마크 시스템"""

    # AI Hub Dasan QA 데이터셋 기본 설정
    DASAN_JSONL = Path("data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs_full.jsonl")
    DASAN_VECTOR_STORE = Path("data/AI_HUB_DASAN_QA/07_vector_stores/full")

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
                "dataset": "AI_HUB_DASAN_QA_real_qa",
                "qa_source": "knowledge_docs_full.jsonl",
                "num_questions": args.num_questions,
                "sampling_strategy": args.sampling_strategy,
                "models": args.models,
                "judge_model": args.judge_model,
                "k_documents": args.k_documents,
                "parallel": True,
                "version": "v4_real_qa"
            }
            self.exp_id = self.exp_manager.create_experiment(config)
            self.experiment = self.exp_manager.get_experiment(self.exp_id)

        # Initialize retriever
        self.retriever = self._initialize_retrieval() if not args.no_retrieval else None

        self.results = {
            "experiment_id": self.exp_id,
            "timestamp": datetime.now().isoformat(),
            "dataset": "AI_HUB_DASAN_QA",
            "qa_source": "real_call_center_data",
            "config": vars(args)
        }

    def _initialize_retrieval(self):
        """벡터 스토어 및 리트리버 초기화"""
        if not self.DASAN_VECTOR_STORE.exists():
            logger.error("❌ Vector store를 찾을 수 없습니다")
            logger.info(f"💡 Expected: {self.DASAN_VECTOR_STORE}")
            logger.info("   Run: python scripts/build_dasan_vector_store.py")
            return None

        try:
            logger.info("📚 Vector store 로드 중...")
            vector_store_stage = VectorStoreStage(model="text-embedding-3-large")
            vector_store_stage.load_vector_store(self.DASAN_VECTOR_STORE)

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

    def load_or_sample_qa(self) -> List[dict]:
        """실제 Q&A 데이터 샘플링 또는 골든 테스트셋 로드"""
        step = "qa_sampled"

        if self.is_step_completed(step):
            logger.info(f"✅ '{step}' 단계 이미 완료 - 로드")
            return self.exp_manager.load_checkpoint(self.exp_id, step)

        logger.info("="*70)

        # Check if golden testset is specified
        if self.args.golden_testset:
            logger.info("📝 골든 테스트셋 로드")
            logger.info("="*70)

            golden_path = Path(self.args.golden_testset)
            if not golden_path.exists():
                raise FileNotFoundError(f"Golden testset not found: {golden_path}")

            # Load golden testset
            import json
            qa_samples = []
            with open(golden_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line)
                    # Map golden testset fields to expected format
                    qa_samples.append({
                        "question": data["original_question"],
                        "answer": data["original_answer"],
                        "ground_truth": data["original_answer"],  # RAGAS needs this
                        "reference_document": data["document"],  # Match DasanQASampler format
                        "id": data.get("dialogue_id", "unknown"),
                        "category": data["metadata"]["category"],
                        "metadata": data["metadata"]
                    })

            logger.info(f"  📊 Golden Testset: {len(qa_samples)} Q&A pairs")
            logger.info(f"  📁 Source: {golden_path.name}")

            # Convert to RAGAS format using sampler
            sampler = DasanQASampler(self.DASAN_JSONL)
            ragas_dataset = sampler.to_ragas_format(qa_samples)

        else:
            logger.info("📝 실제 Q&A 데이터 샘플링")
            logger.info("="*70)

            if not self.DASAN_JSONL.exists():
                raise FileNotFoundError(f"JSONL not found: {self.DASAN_JSONL}")

            # Load and sample
            sampler = DasanQASampler(self.DASAN_JSONL)

            # Get statistics
            stats = sampler.get_statistics()
            logger.info(f"  📊 Dataset: {stats['total_qa_pairs']} Q&A pairs")

            # Sample
            qa_samples = sampler.sample(
                n=self.args.num_questions,
                seed=self.args.seed,
                strategy=self.args.sampling_strategy
            )

            # Convert to RAGAS format
            ragas_dataset = sampler.to_ragas_format(qa_samples)

        # Convert to dict for checkpoint
        questions = ragas_dataset.to_dict()

        # Save checkpoint
        self.exp_manager.save_checkpoint(self.exp_id, step, questions)

        logger.info(f"  ✅ {len(qa_samples)}개 Q&A 로드/샘플링 완료")
        return questions

    def generate_answers(self, questions: Dict[str, List]) -> Dict[str, Dict[str, List[str]]]:
        """
        Phase 1: 모든 모델의 답변 생성
        """
        step = "answers_generated"

        if self.is_step_completed(step):
            logger.info(f"✅ '{step}' 단계 이미 완료 - 로드")
            return self.exp_manager.load_checkpoint(self.exp_id, step)

        logger.info("="*70)
        logger.info("Phase 1: 답변 생성 (Real QA)")
        logger.info("="*70)

        # Parse models
        model_keys = self.parse_models()
        models = [MODEL_CONFIGS[key] for key in model_keys]

        # Initialize answer generator
        generator = AnswerGenerator(
            retriever=self.retriever,
            k_documents=self.args.k_documents
        )

        # Convert questions dict to list format for generator
        questions_list = [
            {
                "user_input": q,
                "reference": r,
                "reference_contexts": rc
            }
            for q, r, rc in zip(
                questions["user_input"],
                questions["reference"],
                questions["reference_contexts"]
            )
        ]

        # Generate answers for all models
        all_datasets = generator.generate_all_answers(
            models=models,
            questions=questions_list,
            use_fixed_context=not self.args.no_retrieval
        )

        # Save checkpoint
        self.exp_manager.save_checkpoint(self.exp_id, step, all_datasets)

        return all_datasets

    def evaluate_answers(self, all_datasets: Dict[str, Dict[str, List[str]]]) -> Dict[str, Any]:
        """
        Phase 2: 순차/병렬 평가
        """
        step = "evaluation_completed"

        if self.is_step_completed(step):
            logger.info(f"✅ '{step}' 단계 이미 완료 - 로드")
            return self.exp_manager.load_checkpoint(self.exp_id, step)

        eval_mode = "순차 평가" if self.args.sequential else "병렬 평가"
        logger.info("="*70)
        logger.info(f"Phase 2: {eval_mode} (Vertex AI Gemini)")
        logger.info("="*70)

        # Run evaluation (순차 또는 병렬)
        evaluation_results = run_parallel_evaluation(
            model_datasets=all_datasets,
            judge_model=self.args.judge_model,
            max_concurrent=self.args.max_concurrent,
            sequential=self.args.sequential,  # NEW: 순차 평가 옵션
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
            # Check if it's a model group
            if model_arg in MODEL_GROUPS:
                models.extend(MODEL_GROUPS[model_arg])
            # Check if it's already a full model key
            elif model_arg in MODEL_CONFIGS:
                models.append(model_arg)
            # Try adding ollama/ prefix if not present
            elif f"ollama/{model_arg}" in MODEL_CONFIGS:
                models.append(f"ollama/{model_arg}")
                logger.info(f"  ℹ️  '{model_arg}' → 'ollama/{model_arg}'")
            else:
                logger.warning(f"⚠️ 알 수 없는 모델: {model_arg}")
                logger.info(f"   사용 가능한 모델: {list(MODEL_CONFIGS.keys())}")

        return list(dict.fromkeys(models))

    def run(self):
        """전체 파이프라인 실행"""
        logger.info("="*70)
        logger.info(f"🧪 실험 ID: {self.exp_id} (Real QA v4)")
        logger.info(f"📂 데이터: AI_HUB_DASAN_QA (실제 콜센터 Q&A)")
        logger.info("="*70)

        try:
            # 1. Q&A 샘플링 (질문 생성 대신)
            questions = self.load_or_sample_qa()
            self.results["questions"] = questions

            # 2. Phase 1: 답변 생성
            all_datasets = self.generate_answers(questions)
            self.results["answers"] = all_datasets

            # 3. Phase 2: 병렬 평가
            evaluation_results = self.evaluate_answers(all_datasets)
            self.results["evaluation"] = evaluation_results

            # 4. 최종 결과 저장
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
        results_file = self.output_dir / f"experiment_{self.exp_id}_real_qa_results.json"

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
        description="실제 Q&A 기반 RAG 벤치마크 v4",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # 실험 관리
    parser.add_argument("--resume-id", type=int, help="재시작할 실험 ID")

    # Q&A 샘플링 설정
    parser.add_argument(
        "--golden-testset",
        type=str,
        help="골든 테스트셋 JSONL 파일 경로 (지정 시 샘플링 대신 사용)"
    )
    parser.add_argument(
        "--num-questions", "-q",
        type=int,
        default=25,
        help="샘플링할 Q&A 개수 (골든 테스트셋 미사용 시)"
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=["random", "balanced", "stratified"],
        default="random",
        help="샘플링 전략 (골든 테스트셋 미사용 시)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="랜덤 시드 (재현성, 골든 테스트셋 미사용 시)"
    )

    # 모델 설정
    parser.add_argument("--models", nargs="+", default=["gpt-4o-mini"], help="평가 모델")

    # RAG 설정
    parser.add_argument("--no-retrieval", action="store_true", help="Retrieval 없이 실행")
    parser.add_argument("--k-documents", type=int, default=4, help="검색 문서 수")

    # 평가 설정
    parser.add_argument(
        "--judge-model",
        default="vertex_ai/gemini-2.5-pro",
        help="평가 모델 (Vertex AI Gemini 2.5 Pro)"
    )
    parser.add_argument("--max-concurrent", type=int, default=2, help="최대 동시 평가 수")
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="순차 평가 모드 (병렬 대신, API 안정성 향상)"
    )

    # 출력 설정
    parser.add_argument("--output-dir", default="data/evaluation/dasan_real_qa")
    parser.add_argument("--verbose", action="store_true", help="상세 로깅")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 설정 출력
    print("="*70)
    print(" "*15 + "Real QA 벤치마크 v4")
    print("="*70)
    if args.resume_id:
        print(f"Resume ID: {args.resume_id}")
    else:
        print("New Experiment (실제 콜센터 Q&A)")
    print(f"Questions: {args.num_questions} (샘플링: {args.sampling_strategy})")
    print(f"Models: {args.models}")
    print(f"Judge: {args.judge_model}")
    print(f"Max Concurrent: {args.max_concurrent}")
    print(f"Output: {args.output_dir}")
    print()

    # 실행
    benchmark = RealQABenchmarkV4(args)
    benchmark.run()


if __name__ == "__main__":
    main()
