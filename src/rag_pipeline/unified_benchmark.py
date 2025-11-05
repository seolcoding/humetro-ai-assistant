#!/usr/bin/env python3
"""
Unified RAG Benchmark System
=============================

통합 벤치마크 시스템 - 모든 테스트 기능을 하나로 통합
E2E 파이프라인: 질문 생성 → 분류 → RAG 평가 → 결과 분석

Features:
- CLI 인터페이스로 모든 파라미터 제어
- 캐시 관리 (재사용 가능한 질문 뱅크)
- 다중 모델 지원 (OpenAI, Ollama)
- Retrieval 검증 및 오류 처리
- RAGAS 메트릭 평가

Usage:
    # 기본 실행 (50문제, GPT-4o-mini)
    python unified_benchmark.py

    # 전체 벤치마크 (100문제, 모든 모델)
    python unified_benchmark.py --questions 100 --models all

    # 특정 모델만
    python unified_benchmark.py --models gpt-4o-mini ollama/exaone3.5:7.8b

    # 캐시 강제 재생성
    python unified_benchmark.py --force-generate

    # Retrieval 없이 (LLM knowledge only)
    python unified_benchmark.py --no-retrieval
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
from tqdm import tqdm
import hashlib
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment
from dotenv import load_dotenv
load_dotenv()

# Add tests path for testset_generator
sys.path.insert(0, str(project_root / "tests" / "rag_pipeline"))

# Project imports
from testset_generator import CachedTestsetGenerator, TestsetConfig
from question_classifier import QuestionComplexityClassifier
from generation_benchmark import GenerationBenchmark

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

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

    # Ollama models
    "ollama/exaone3.5:7.8b": {
        "name": "EXAONE-3.5-7.8B",
        "model": "ollama/exaone3.5:7.8b",
        "api_base": "http://100.95.220.92:11434",
        "provider": "ollama"
    },
    "ollama/eeve-korean-10.8b:latest": {
        "name": "EEVE-Korean-10.8B",
        "model": "ollama/eeve-korean-10.8b:latest",
        "api_base": "http://100.95.220.92:11434",
        "provider": "ollama"
    },
    "ollama/llama-3.2-3b-instruct:latest": {
        "name": "Llama-3.2-3B",
        "model": "ollama/llama-3.2-3b-instruct:latest",
        "api_base": "http://100.95.220.92:11434",
        "provider": "ollama"
    },
    "ollama/gemma2-9b-it:latest": {
        "name": "Gemma2-9B",
        "model": "ollama/gemma2-9b-it:latest",
        "api_base": "http://100.95.220.92:11434",
        "provider": "ollama"
    }
}

# Shortcuts for model groups
MODEL_GROUPS = {
    "all": list(MODEL_CONFIGS.keys()),
    "openai": ["gpt-4o", "gpt-4o-mini"],
    "ollama": [k for k in MODEL_CONFIGS.keys() if k.startswith("ollama/")],
    "korean": ["ollama/exaone3.5:7.8b", "ollama/eeve-korean-10.8b:latest"],
    "fast": ["gpt-4o-mini", "ollama/llama-3.2-3b-instruct:latest"]
}


class UnifiedBenchmark:
    """통합 벤치마크 시스템"""

    def __init__(self, args):
        self.args = args
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().isoformat()

        # Initialize components
        self.testset_generator = CachedTestsetGenerator()
        self.classifier = QuestionComplexityClassifier()
        self.benchmark = None  # Will be initialized later

        # Results storage
        self.results = {
            "metadata": {
                "timestamp": self.timestamp,
                "args": vars(args)
            },
            "questions": {},
            "evaluations": {}
        }

    def validate_retrieval(self) -> bool:
        """
        Validate retrieval system is working

        Returns:
            True if retrieval works, False otherwise
        """
        if self.args.no_retrieval:
            logger.info("⏭️  Retrieval 건너뜀 (--no-retrieval)")
            return True

        logger.info("🔍 Retrieval 시스템 검증 중...")

        try:
            # Try to initialize a test benchmark
            test_benchmark = GenerationBenchmark(
                models=[MODEL_CONFIGS["gpt-4o-mini"]],
                use_fixed_context=True,
                k_documents=self.args.k_documents,
                judge_model=self.args.judge_model
            )

            # Check if retriever is initialized
            if test_benchmark.retriever is None:
                logger.error("❌ Retriever 초기화 실패!")
                return False

            # Test retrieval with sample question
            test_question = "서울시 대중교통 요금은?"
            try:
                contexts = test_benchmark._retrieve_context(test_question)
                if not contexts:
                    logger.warning("⚠️ Retrieved contexts가 비어있음")
                    return False

                logger.info(f"✅ Retrieval 검증 성공: {len(contexts)} contexts retrieved")
                logger.info(f"   샘플 context: {contexts[0][:100]}...")
                return True

            except Exception as e:
                logger.error(f"❌ Retrieval 테스트 실패: {e}")
                return False

        except Exception as e:
            logger.error(f"❌ Benchmark 초기화 실패: {e}")
            return False

    def generate_questions(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Generate or load questions from cache

        Returns:
            Tuple of (dataframe, config)
        """
        logger.info("="*70)
        logger.info("📝 질문 생성/로드")
        logger.info("="*70)

        # Calculate target sizes
        if self.args.questions <= 50:
            single_hop = min(25, self.args.questions // 2)
            multi_hop = self.args.questions - single_hop
        else:
            # For larger sets, aim for 40/60 split (more multi-hop)
            single_hop = int(self.args.questions * 0.4)
            multi_hop = self.args.questions - single_hop

        logger.info(f"목표: 총 {self.args.questions}개 (Single: {single_hop}, Multi: {multi_hop})")

        # Generate/load testset
        config, testset_df = self.testset_generator.generate_or_load(
            llm_model=self.args.generation_model,
            llm_temperature=0.3,
            embedding_model="text-embedding-3-small",
            document_source=self.args.document_source,
            num_documents=self.args.num_documents,
            testset_size=self.args.questions,
            language="korean",
            force_regenerate=self.args.force_generate,
            use_korean_personas=True,
            is_benchmark=True,
            description=f"Unified benchmark - {self.args.questions} questions"
        )

        logger.info(f"✅ 질문 준비 완료: {len(testset_df)} questions")

        # Store in results
        self.results["questions"]["raw"] = testset_df.to_dict('records')
        self.results["questions"]["config"] = {
            "cache_key": config.get_cache_key(),
            "total": len(testset_df)
        }

        return testset_df, config

    def classify_questions(self, testset_df: pd.DataFrame) -> Dict:
        """
        Classify questions into single-hop and multi-hop

        Returns:
            Classified questions dictionary
        """
        logger.info("="*70)
        logger.info("🏷️  질문 분류 (Single vs Multi-hop)")
        logger.info("="*70)

        classified = {
            "single_hop": [],
            "multi_hop": [],
            "unclassified": [],
            "metadata": {
                "method": self.args.classification_method,
                "timestamp": datetime.now().isoformat()
            }
        }

        for idx, row in tqdm(testset_df.iterrows(), total=len(testset_df), desc="분류 중"):
            question = row.get('user_input', row.get('question', ''))
            reference = row.get('reference', row.get('reference_answer', ''))

            # Classify
            result = self.classifier.classify(
                question=question,
                reference_answer=reference,
                method=self.args.classification_method
            )

            # Prepare question object
            q_obj = {
                "id": idx,
                "question": question,
                "reference_answer": reference,
                "classification": result.complexity.value,
                "confidence": result.confidence,
                "reasoning": result.reasoning
            }

            # Add to appropriate group
            if result.complexity.value == "single_hop":
                classified["single_hop"].append(q_obj)
            elif result.complexity.value == "multi_hop":
                classified["multi_hop"].append(q_obj)
            else:
                classified["unclassified"].append(q_obj)

        # Summary
        classified["metadata"]["counts"] = {
            "single_hop": len(classified["single_hop"]),
            "multi_hop": len(classified["multi_hop"]),
            "unclassified": len(classified["unclassified"])
        }

        logger.info(f"✅ 분류 완료:")
        logger.info(f"   - Single-hop: {len(classified['single_hop'])}")
        logger.info(f"   - Multi-hop: {len(classified['multi_hop'])}")
        logger.info(f"   - Unclassified: {len(classified['unclassified'])}")

        # Store in results
        self.results["questions"]["classified"] = classified

        return classified

    def balance_questions(self, classified: Dict) -> Dict:
        """
        Balance question groups to target sizes

        Returns:
            Balanced questions dictionary
        """
        logger.info("="*70)
        logger.info("⚖️  질문 균형 조정")
        logger.info("="*70)

        # Calculate targets
        target_single = self.args.questions // 2
        target_multi = self.args.questions - target_single

        balanced = {
            "single_hop": [],
            "multi_hop": [],
            "metadata": classified.get("metadata", {})
        }

        # Balance single-hop
        single_questions = classified.get("single_hop", [])
        if len(single_questions) >= target_single:
            # Sort by confidence and take top N
            single_sorted = sorted(single_questions, key=lambda x: x['confidence'], reverse=True)
            balanced["single_hop"] = single_sorted[:target_single]
        else:
            balanced["single_hop"] = single_questions
            logger.warning(f"⚠️ Single-hop 부족: {len(single_questions)}/{target_single}")

        # Balance multi-hop
        multi_questions = classified.get("multi_hop", [])
        if len(multi_questions) >= target_multi:
            multi_sorted = sorted(multi_questions, key=lambda x: x['confidence'], reverse=True)
            balanced["multi_hop"] = multi_sorted[:target_multi]
        else:
            balanced["multi_hop"] = multi_questions
            logger.warning(f"⚠️ Multi-hop 부족: {len(multi_questions)}/{target_multi}")

        # Update metadata
        balanced["metadata"]["balanced"] = True
        balanced["metadata"]["final_counts"] = {
            "single_hop": len(balanced["single_hop"]),
            "multi_hop": len(balanced["multi_hop"]),
            "total": len(balanced["single_hop"]) + len(balanced["multi_hop"])
        }

        logger.info(f"✅ 균형 조정 완료:")
        logger.info(f"   - Single-hop: {len(balanced['single_hop'])}")
        logger.info(f"   - Multi-hop: {len(balanced['multi_hop'])}")
        logger.info(f"   - Total: {balanced['metadata']['final_counts']['total']}")

        # Store in results
        self.results["questions"]["balanced"] = balanced

        return balanced

    def run_benchmark(self, questions: Dict) -> Dict:
        """
        Run benchmark evaluation

        Returns:
            Evaluation results
        """
        logger.info("="*70)
        logger.info("🚀 벤치마크 실행")
        logger.info("="*70)

        # Parse model list
        models_to_eval = self.parse_model_list(self.args.models)
        model_configs = [MODEL_CONFIGS[m] for m in models_to_eval]

        logger.info(f"평가 모델 ({len(model_configs)}개):")
        for cfg in model_configs:
            logger.info(f"  - {cfg['name']} ({cfg['provider']})")

        # Initialize benchmark
        self.benchmark = GenerationBenchmark(
            models=model_configs,
            use_fixed_context=not self.args.no_retrieval,
            k_documents=self.args.k_documents if not self.args.no_retrieval else 0,
            judge_model=self.args.judge_model
        )

        # Validate retrieval if needed
        if not self.args.no_retrieval:
            if not self.validate_retrieval():
                if self.args.strict:
                    raise RuntimeError("Retrieval validation failed in strict mode")
                else:
                    logger.warning("⚠️ Retrieval 실패, LLM knowledge only로 진행")
                    self.benchmark.use_fixed_context = False

        # Run benchmark
        try:
            results = self.benchmark.run_benchmark(
                questions,
                self.output_dir
            )

            logger.info("✅ 벤치마크 완료")

            # Store results
            self.results["evaluations"] = results

            return results

        except Exception as e:
            logger.error(f"❌ 벤치마크 실행 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())

            if self.args.strict:
                raise
            else:
                return {}

    def parse_model_list(self, model_args: List[str]) -> List[str]:
        """Parse model arguments into actual model IDs"""
        models = []

        for arg in model_args:
            if arg in MODEL_GROUPS:
                # It's a group
                models.extend(MODEL_GROUPS[arg])
            elif arg in MODEL_CONFIGS:
                # It's a specific model
                models.append(arg)
            else:
                logger.warning(f"⚠️ Unknown model: {arg}")

        # Remove duplicates while preserving order
        seen = set()
        unique_models = []
        for m in models:
            if m not in seen:
                seen.add(m)
                unique_models.append(m)

        return unique_models

    def save_results(self):
        """Save all results"""
        logger.info("="*70)
        logger.info("💾 결과 저장")
        logger.info("="*70)

        # Main results file
        results_file = self.output_dir / f"unified_benchmark_{self.timestamp.replace(':', '-')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        logger.info(f"📄 전체 결과: {results_file}")

        # Question bank (for reuse)
        if "balanced" in self.results.get("questions", {}):
            question_bank = {
                "metadata": {
                    "version": "2.0",
                    "created_at": self.timestamp,
                    "total": self.results["questions"]["balanced"]["metadata"]["final_counts"]["total"]
                },
                "questions": self.results["questions"]["balanced"]
            }

            bank_file = self.output_dir / "question_bank_latest.json"
            with open(bank_file, 'w', encoding='utf-8') as f:
                json.dump(question_bank, f, ensure_ascii=False, indent=2)
            logger.info(f"📚 질문 뱅크: {bank_file}")

        # Summary report
        self.generate_summary_report()

    def generate_summary_report(self):
        """Generate human-readable summary report"""
        report_file = self.output_dir / f"summary_{self.timestamp.replace(':', '-')}.md"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# Unified Benchmark Report\n\n")
            f.write(f"**Date**: {self.timestamp}\n\n")

            # Configuration
            f.write("## Configuration\n\n")
            f.write(f"- Questions: {self.args.questions}\n")
            f.write(f"- Models: {', '.join(self.parse_model_list(self.args.models))}\n")
            f.write(f"- Retrieval: {'Enabled' if not self.args.no_retrieval else 'Disabled'}\n")
            f.write(f"- Judge: {self.args.judge_model}\n\n")

            # Question statistics
            if "balanced" in self.results.get("questions", {}):
                counts = self.results["questions"]["balanced"]["metadata"]["final_counts"]
                f.write("## Question Distribution\n\n")
                f.write(f"- Single-hop: {counts['single_hop']}\n")
                f.write(f"- Multi-hop: {counts['multi_hop']}\n")
                f.write(f"- Total: {counts['total']}\n\n")

            # Evaluation results
            if self.results.get("evaluations"):
                f.write("## Evaluation Results\n\n")
                f.write("| Model | Complexity | Faithfulness | Relevancy | Correctness |\n")
                f.write("|-------|------------|--------------|-----------|-------------|\n")

                for complexity in ["single_hop", "multi_hop"]:
                    for model_name, model_results in self.results["evaluations"].get(complexity, {}).items():
                        if "metrics" in model_results:
                            metrics = model_results["metrics"]
                            faith = self._format_metric(metrics.get("faithfulness"))
                            relev = self._format_metric(metrics.get("answer_relevancy"))
                            correct = self._format_metric(metrics.get("answer_correctness"))
                            f.write(f"| {model_name} | {complexity} | {faith} | {relev} | {correct} |\n")

        logger.info(f"📊 요약 리포트: {report_file}")

    def _format_metric(self, value):
        """Format metric value for display"""
        if value is None:
            return "N/A"
        elif isinstance(value, list):
            if len(value) > 0:
                avg = sum(value) / len(value)
                return f"{avg:.3f}"
            return "N/A"
        else:
            return f"{value:.3f}"

    def run(self):
        """Main execution flow"""
        print("\n" + "="*70)
        print("UNIFIED RAG BENCHMARK SYSTEM".center(70))
        print("="*70)

        start_time = time.time()

        try:
            # Step 1: Generate/load questions
            testset_df, config = self.generate_questions()

            # Step 2: Classify questions
            classified = self.classify_questions(testset_df)

            # Step 3: Balance questions
            balanced = self.balance_questions(classified)

            # Step 4: Run benchmark (if not skipped)
            if not self.args.skip_benchmark:
                self.run_benchmark(balanced)
            else:
                logger.info("⏭️  벤치마크 건너뜀 (--skip-benchmark)")

            # Step 5: Save results
            self.save_results()

            # Final summary
            elapsed = time.time() - start_time
            print("\n" + "="*70)
            print("✅ 완료!".center(70))
            print("="*70)
            print(f"\n⏱️  실행 시간: {elapsed/60:.1f}분")
            print(f"📁 결과 위치: {self.output_dir}")

            # Print quick metrics if available
            if self.results.get("evaluations"):
                print("\n📊 Quick Metrics:")
                for model in self.parse_model_list(self.args.models):
                    model_name = MODEL_CONFIGS[model]["name"]
                    print(f"  {model_name}:")

                    for complexity in ["single_hop", "multi_hop"]:
                        if complexity in self.results["evaluations"]:
                            if model_name in self.results["evaluations"][complexity]:
                                metrics = self.results["evaluations"][complexity][model_name].get("metrics", {})
                                faith = self._format_metric(metrics.get("faithfulness"))
                                print(f"    {complexity}: F={faith}")

        except Exception as e:
            logger.error(f"❌ 실행 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())

            if self.args.strict:
                raise
            else:
                sys.exit(1)


def main():
    """Main entry point"""

    parser = argparse.ArgumentParser(
        description="Unified RAG Benchmark System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run (50 questions, GPT-4o-mini)
  %(prog)s

  # Full benchmark (100 questions, all models)
  %(prog)s --questions 100 --models all

  # Specific models only
  %(prog)s --models gpt-4o-mini ollama/exaone3.5:7.8b

  # Korean models only
  %(prog)s --models korean

  # Force regenerate questions
  %(prog)s --force-generate

  # Without retrieval (LLM knowledge only)
  %(prog)s --no-retrieval

Model Groups:
  all     - All available models (6)
  openai  - OpenAI models (gpt-4o, gpt-4o-mini)
  ollama  - All Ollama models (4)
  korean  - Korean-optimized models
  fast    - Fast models for quick testing
        """
    )

    # Question generation
    parser.add_argument(
        "--questions", "-q",
        type=int,
        default=50,
        help="Number of questions to generate/evaluate (default: 50)"
    )
    parser.add_argument(
        "--force-generate",
        action="store_true",
        help="Force regeneration of questions (ignore cache)"
    )
    parser.add_argument(
        "--generation-model",
        default="gpt-4o-mini",
        help="Model to use for question generation (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--num-documents",
        type=int,
        default=100,
        help="Number of source documents for question generation (default: 100)"
    )
    parser.add_argument(
        "--document-source",
        default="data/crawled/seoul_traffic/markdown_deduplicated",
        help="Path to source documents"
    )

    # Model selection
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        default=["gpt-4o-mini"],
        help="Models to evaluate (can use groups: all, openai, ollama, korean, fast)"
    )

    # Classification
    parser.add_argument(
        "--classification-method",
        choices=["llm", "rules", "hybrid"],
        default="hybrid",
        help="Question classification method (default: hybrid)"
    )

    # Retrieval settings
    parser.add_argument(
        "--no-retrieval",
        action="store_true",
        help="Disable retrieval (use LLM knowledge only)"
    )
    parser.add_argument(
        "--k-documents",
        type=int,
        default=4,
        help="Number of documents to retrieve (default: 4)"
    )

    # Evaluation settings
    parser.add_argument(
        "--judge-model",
        default="gpt-5",
        help="Model to use as judge for RAGAS evaluation (default: gpt-5)"
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip benchmark evaluation (only generate and classify questions)"
    )

    # Output settings
    parser.add_argument(
        "--output-dir",
        default="data/evaluation/unified_benchmark",
        help="Output directory for results"
    )

    # Execution settings
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Strict mode - fail on any error"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    # Adjust logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Print configuration
    print("\n" + "="*70)
    print("Configuration".center(70))
    print("="*70)
    print(f"Questions: {args.questions}")
    print(f"Models: {args.models}")
    print(f"Retrieval: {'Enabled' if not args.no_retrieval else 'Disabled'}")
    print(f"Judge: {args.judge_model}")
    print(f"Output: {args.output_dir}")

    # Run benchmark
    benchmark = UnifiedBenchmark(args)
    benchmark.run()


if __name__ == "__main__":
    main()