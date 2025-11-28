#!/usr/bin/env python3
"""
LightRAG Benchmark Script
=========================

Standalone evaluation script for LightRAG performance assessment.

Features:
- Golden testset support
- Multi-model evaluation
- RAGAS metrics (Faithfulness, Answer Relevancy, Answer Correctness)
- Checkpoint system for resume capability
- Compatible with existing evaluation infrastructure

Usage:
    # Quick test (10 questions)
    python src/rag_pipeline/lightrag_benchmark.py \\
        --num-questions 10 \\
        --models gpt-4o-mini

    # Full evaluation (golden testset)
    python src/rag_pipeline/lightrag_benchmark.py \\
        --golden-testset data/evaluation/testsets/golden_testset_50.csv \\
        --models thesis \\
        --query-mode hybrid

    # Compare query modes
    python src/rag_pipeline/lightrag_benchmark.py \\
        --golden-testset data/evaluation/testsets/golden_testset_50.csv \\
        --query-mode local \\
        --output-dir data/evaluation/lightrag_local
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
from src.lightrag_agent.lightrag_retriever import create_lightrag_retriever
from src.rag_pipeline.answer_generator import AnswerGenerator
from src.evaluation.parallel_evaluator import run_parallel_evaluation
from src.data_loader.dasan_qa_sampler import DasanQASampler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Get Ollama base URL from environment variable
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://100.95.220.92:11434")

# Model configurations (same as unified_benchmark_v4_real_qa.py)
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

    # Ollama models
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
    """실험 ID 및 체크포인트 관리 (from unified_benchmark_v4)"""

    def __init__(self, checkpoint_dir: Path):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.checkpoint_dir / "experiments.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"next_id": 1, "experiments": {}}

    def _save_metadata(self):
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def create_experiment(self, config: dict) -> int:
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
        return self.metadata["experiments"].get(str(exp_id))

    def update_experiment(self, exp_id: int, updates: dict):
        if str(exp_id) in self.metadata["experiments"]:
            self.metadata["experiments"][str(exp_id)].update(updates)
            self._save_metadata()

    def save_checkpoint(self, exp_id: int, step: str, data: Any):
        checkpoint_file = self.checkpoint_dir / f"exp_{exp_id}_{step}.pkl"
        with open(checkpoint_file, 'wb') as f:
            pickle.dump(data, f)

        exp = self.metadata["experiments"].get(str(exp_id))
        if exp and step not in exp["completed_steps"]:
            exp["completed_steps"].append(step)
            exp["last_checkpoint"] = step
            exp["last_updated"] = datetime.now().isoformat()
            self._save_metadata()

        logger.info(f"💾 체크포인트 저장: {step}")

    def load_checkpoint(self, exp_id: int, step: str) -> Optional[Any]:
        checkpoint_file = self.checkpoint_dir / f"exp_{exp_id}_{step}.pkl"
        if checkpoint_file.exists():
            with open(checkpoint_file, 'rb') as f:
                return pickle.load(f)
        return None


class LightRAGBenchmark:
    """LightRAG 벤치마크 시스템"""

    # Data paths
    DASAN_JSONL = Path("data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs_full.jsonl")
    LIGHTRAG_INDEX = Path("data/AI_HUB_DASAN_QA/09_lightrag/test_index")  # Using test index

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
                "dataset": "AI_HUB_DASAN_QA",
                "rag_type": "lightrag",
                "query_mode": args.query_mode,
                "num_questions": args.num_questions,
                "models": args.models,
                "judge_model": args.judge_model,
                "k_documents": args.k_documents,
                "version": "lightrag_v1"
            }
            self.exp_id = self.exp_manager.create_experiment(config)
            self.experiment = self.exp_manager.get_experiment(self.exp_id)

        # Initialize LightRAG retriever
        self.retriever = self._initialize_lightrag_retriever()

        self.results = {
            "experiment_id": self.exp_id,
            "timestamp": datetime.now().isoformat(),
            "rag_type": "lightrag",
            "query_mode": args.query_mode,
            "config": vars(args)
        }

    def _initialize_lightrag_retriever(self):
        """Initialize LightRAG retriever"""
        if not self.LIGHTRAG_INDEX.exists():
            logger.error("❌ LightRAG index를 찾을 수 없습니다")
            logger.info(f"💡 Expected: {self.LIGHTRAG_INDEX}")
            logger.info("   Run: python scripts/build_lightrag_index.py")
            return None

        try:
            logger.info("🔧 LightRAG Retriever 초기화 중...")
            retriever = create_lightrag_retriever(
                working_dir=str(self.LIGHTRAG_INDEX),
                query_mode=self.args.query_mode,
                k=self.args.k_documents
            )

            logger.info(f"  ✅ LightRAG Retriever 초기화 완료")
            logger.info(f"     Mode: {self.args.query_mode}")
            logger.info(f"     k: {self.args.k_documents}")
            return retriever

        except Exception as e:
            logger.error(f"❌ LightRAG Retriever 초기화 실패: {e}")
            return None

    def is_step_completed(self, step: str) -> bool:
        return step in self.experiment.get("completed_steps", [])

    def load_or_sample_qa(self) -> List[dict]:
        """Q&A 데이터 로드 (골든 테스트셋 또는 샘플링)"""
        step = "qa_sampled"

        if self.is_step_completed(step):
            logger.info(f"✅ '{step}' 단계 이미 완료 - 로드")
            return self.exp_manager.load_checkpoint(self.exp_id, step)

        logger.info("="*70)
        logger.info("📝 Q&A 데이터 로드")
        logger.info("="*70)

        # Check if golden testset is specified
        if self.args.golden_testset:
            logger.info(f"  📁 Golden Testset: {self.args.golden_testset}")

            golden_path = Path(self.args.golden_testset)
            if not golden_path.exists():
                raise FileNotFoundError(f"Golden testset not found: {golden_path}")

            # Load golden testset (CSV format)
            import csv
            qa_samples = []
            with open(golden_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for i, row in enumerate(reader):
                    # Parse reference_contexts (stringified list)
                    import ast
                    try:
                        contexts = ast.literal_eval(row["reference_contexts"])
                    except:
                        contexts = [row["reference_contexts"]]

                    qa_samples.append({
                        "question": row["user_input"],
                        "answer": row["reference"],
                        "ground_truth": row["reference"],
                        "reference_contexts": contexts,  # Keep as list
                        "reference_document": "\n\n".join(contexts),  # Join for compatibility
                        "id": f"golden_q{i+1:03d}",
                        "category": "transport",  # Default category
                        "metadata": {
                            "category": "transport",
                            "synthesizer": row.get("synthesizer_name", "unknown")
                        }
                    })

                    # Limit to num_questions if specified
                    if self.args.num_questions and len(qa_samples) >= self.args.num_questions:
                        break

            logger.info(f"  ✅ {len(qa_samples)} Q&A pairs loaded")

        else:
            logger.info("  📁 Sampling from JSONL")

            if not self.DASAN_JSONL.exists():
                raise FileNotFoundError(f"JSONL not found: {self.DASAN_JSONL}")

            sampler = DasanQASampler(self.DASAN_JSONL)
            qa_samples = sampler.sample(
                n=self.args.num_questions,
                seed=self.args.seed,
                strategy=self.args.sampling_strategy
            )

            logger.info(f"  ✅ {len(qa_samples)} Q&A pairs sampled")

        # Convert to RAGAS format
        if self.args.golden_testset:
            # Golden testset is already in proper format
            questions = {
                "user_input": [qa["question"] for qa in qa_samples],
                "reference": [qa["ground_truth"] for qa in qa_samples],
                "reference_contexts": [qa["reference_contexts"] for qa in qa_samples]  # Already a list
            }
        else:
            # Use DasanQASampler for JSONL format conversion
            sampler = DasanQASampler(self.DASAN_JSONL)
            ragas_dataset = sampler.to_ragas_format(qa_samples)
            questions = ragas_dataset.to_dict()

        # Save checkpoint
        self.exp_manager.save_checkpoint(self.exp_id, step, questions)

        return questions

    def generate_answers(self, questions: Dict[str, List]) -> Dict[str, Dict[str, List[str]]]:
        """모든 모델의 답변 생성"""
        step = "answers_generated"

        if self.is_step_completed(step):
            logger.info(f"✅ '{step}' 단계 이미 완료 - 로드")
            return self.exp_manager.load_checkpoint(self.exp_id, step)

        logger.info("="*70)
        logger.info("Phase 1: 답변 생성 (LightRAG)")
        logger.info("="*70)

        # Parse models
        model_keys = self.parse_models()
        models = [MODEL_CONFIGS[key] for key in model_keys]

        # Initialize answer generator with LightRAG retriever
        generator = AnswerGenerator(
            retriever=self.retriever,
            k_documents=self.args.k_documents
        )

        # Convert questions dict to list format
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
            use_fixed_context=True  # Use LightRAG retrieval
        )

        # Save checkpoint
        self.exp_manager.save_checkpoint(self.exp_id, step, all_datasets)

        return all_datasets

    def evaluate_answers(self, all_datasets: Dict[str, Dict[str, List[str]]]) -> Dict[str, Any]:
        """RAGAS 평가"""
        step = "evaluation_completed"

        if self.is_step_completed(step):
            logger.info(f"✅ '{step}' 단계 이미 완료 - 로드")
            return self.exp_manager.load_checkpoint(self.exp_id, step)

        eval_mode = "순차 평가" if self.args.sequential else "병렬 평가"
        logger.info("="*70)
        logger.info(f"Phase 2: {eval_mode} (Vertex AI Gemini)")
        logger.info("="*70)

        # Run evaluation
        evaluation_results = run_parallel_evaluation(
            model_datasets=all_datasets,
            judge_model=self.args.judge_model,
            max_concurrent=self.args.max_concurrent,
            sequential=self.args.sequential,
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
            elif f"ollama/{model_arg}" in MODEL_CONFIGS:
                models.append(f"ollama/{model_arg}")
            else:
                logger.warning(f"⚠️ Unknown model: {model_arg}")

        return list(dict.fromkeys(models))

    def run(self):
        """전체 파이프라인 실행"""
        logger.info("="*70)
        logger.info(f"🧪 실험 ID: {self.exp_id} (LightRAG)")
        logger.info(f"   Query Mode: {self.args.query_mode}")
        logger.info("="*70)

        try:
            # 1. Q&A 로드
            questions = self.load_or_sample_qa()
            self.results["questions"] = questions

            # 2. 답변 생성
            all_datasets = self.generate_answers(questions)
            self.results["answers"] = all_datasets

            # 3. 평가
            evaluation_results = self.evaluate_answers(all_datasets)
            self.results["evaluation"] = evaluation_results

            # 4. 최종 결과 저장
            self._save_final_results()

            logger.info("="*70)
            logger.info(f"✅ 실험 완료! (ID: {self.exp_id})")
            logger.info("="*70)

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
        results_file = self.output_dir / f"experiment_{self.exp_id}_lightrag_results.json"

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
        description="LightRAG 벤치마크 평가",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Experiment management
    parser.add_argument("--resume-id", type=int, help="재시작할 실험 ID")

    # Q&A settings
    parser.add_argument(
        "--golden-testset",
        type=str,
        help="골든 테스트셋 JSONL 파일 경로"
    )
    parser.add_argument(
        "--num-questions", "-q",
        type=int,
        default=25,
        help="샘플링할 Q&A 개수"
    )
    parser.add_argument(
        "--sampling-strategy",
        choices=["random", "balanced", "stratified"],
        default="random"
    )
    parser.add_argument("--seed", type=int, default=42)

    # LightRAG settings
    parser.add_argument(
        "--query-mode",
        choices=["local", "global", "hybrid"],
        default="hybrid",
        help="LightRAG query mode"
    )
    parser.add_argument("--k-documents", type=int, default=4)

    # Model settings
    parser.add_argument("--models", nargs="+", default=["gpt-4o-mini"])

    # Evaluation settings
    parser.add_argument(
        "--judge-model",
        default="vertex_ai/gemini-2.5-pro"
    )
    parser.add_argument("--max-concurrent", type=int, default=2)
    parser.add_argument("--sequential", action="store_true")

    # Output settings
    parser.add_argument("--output-dir", default="data/evaluation/lightrag")

    args = parser.parse_args()

    # Run benchmark
    benchmark = LightRAGBenchmark(args)
    benchmark.run()


if __name__ == "__main__":
    main()
