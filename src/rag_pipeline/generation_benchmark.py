#!/usr/bin/env python3
"""
Generation Benchmark Module
============================

Generation 품질에 초점을 맞춘 벤치마크 평가 모듈
Retrieval 변동성을 제거하고 순수 생성 능력만 평가

Version History:
- 2025-11-05: Fixed retriever initialization bug
  - Changed from incorrect RetrievalStage instantiation to LangChain retriever pattern
  - Fixed: TypeError with 'k' parameter, now uses vector_store.as_retriever()
  - Added fail-fast error handling to prevent silent failures
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
from tqdm import tqdm

# RAGAS imports
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    answer_correctness,
)
from ragas.run_config import RunConfig
from ragas.testset.graph import KnowledgeGraph
from datasets import Dataset

# LiteLLM for multi-model support
import litellm

# Local imports
from src.rag_pipeline.stages.stage_05_vector_store import VectorStoreStage

logger = logging.getLogger(__name__)


class GenerationBenchmark:
    """
    Generation-focused benchmark evaluation

    핵심 특징:
    1. 모든 모델에 동일한 context 제공 (retrieval 변동성 제거)
    2. Generation 품질만 평가 (faithfulness, relevancy, correctness)
    3. Single-hop vs Multi-hop 별도 평가
    4. 투명한 결과 저장 구조
    """

    def __init__(
        self,
        models: List[Dict[str, str]],
        use_fixed_context: bool = True,
        k_documents: int = 4,
        judge_model: str = "gpt-5",  # GPT-5를 기본 judge model로 사용
        custom_retriever: Optional[Any] = None  # For KG RAG or other custom retrievers
    ):
        """
        Args:
            models: List of model configurations
                [{"name": "GPT-4o-mini", "model": "gpt-4o-mini", "api_base": None}, ...]
            use_fixed_context: Use same retrieved context for all models
            k_documents: Number of documents to retrieve per question
            judge_model: Model to use as RAGAS judge (GPT-4o recommended)
            custom_retriever: Optional custom retriever (e.g., KG RAG). If None, uses default FAISS.
        """
        self.models = models
        self.use_fixed_context = use_fixed_context
        self.k_documents = k_documents
        self.judge_model = judge_model

        # Initialize vector store and retrieval
        self.vector_store = None
        self.retriever = custom_retriever  # Use custom if provided

        # Only initialize FAISS if no custom retriever provided
        if self.retriever is None:
            self._initialize_retrieval()

        # Configure LiteLLM
        litellm.drop_params = True  # Drop unsupported parameters
        litellm.set_verbose = False

        # Results storage
        self.results = {
            "metadata": {},
            "single_hop": {},
            "multi_hop": {},
            "comparison": {}
        }

    def _initialize_retrieval(self):
        """
        Initialize vector store and retrieval components.

        Uses LangChain pattern:
        1. Create VectorStoreStage
        2. Load FAISS vector store
        3. Create LangChain retriever with as_retriever()
        4. Use retriever.invoke() for document retrieval
        """
        vector_store_path = Path("data/vector_store/seoul_traffic")

        if not vector_store_path.exists():
            logger.warning(f"⚠️ 벡터 스토어를 찾을 수 없음: {vector_store_path}")
            logger.warning("Benchmark will run without retrieval (LLM knowledge only)")
            return

        try:
            logger.info("📚 벡터 스토어 로드 중...")

            # Step 1: Create VectorStoreStage (use text-embedding-3-large for KG compatibility)
            vector_store_stage = VectorStoreStage(
                model="text-embedding-3-large"
            )

            # Step 2: Load FAISS vector store using public API
            vector_store_stage.load_vector_store(vector_store_path)
            logger.info(f"  ✅ {vector_store_stage.vector_store.index.ntotal:,} vectors loaded")

            # Step 3: Create LangChain retriever (NOT RetrievalStage!)
            self.retriever = vector_store_stage.as_retriever(
                search_type="similarity",
                search_kwargs={"k": self.k_documents}
            )
            logger.info(f"  ✅ Retriever initialized (k={self.k_documents})")

        except Exception as e:
            logger.error(f"❌ 리트리버 초기화 실패: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Re-raise to prevent silent failure
            raise RuntimeError(f"Failed to initialize retrieval: {e}") from e

    def run_benchmark(
        self,
        classified_questions: Dict[str, List],
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        Run generation benchmark on classified questions

        Args:
            classified_questions: {
                "single_hop": [questions...],
                "multi_hop": [questions...]
            }
            output_dir: Directory to save results

        Returns:
            Complete benchmark results
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().isoformat()

        # Store metadata
        self.results["metadata"] = {
            "timestamp": timestamp,
            "total_questions": sum(len(q) for q in classified_questions.values()),
            "single_hop_count": len(classified_questions.get("single_hop", [])),
            "multi_hop_count": len(classified_questions.get("multi_hop", [])),
            "models": [m["name"] for m in self.models],
            "k_documents": self.k_documents,
            "use_fixed_context": self.use_fixed_context,
            "judge_model": self.judge_model
        }

        # Evaluate each complexity group
        for complexity in ["single_hop", "multi_hop"]:
            if complexity not in classified_questions:
                continue

            questions = classified_questions[complexity][:25]  # Limit to 25 each
            logger.info(f"\n{'='*70}")
            logger.info(f"Evaluating {complexity}: {len(questions)} questions")
            logger.info(f"{'='*70}")

            self.results[complexity] = self.evaluate_complexity_group(
                questions=questions,
                complexity=complexity
            )

        # Generate comparison analysis
        self.results["comparison"] = self._generate_comparison()

        # Save results
        self._save_results(output_dir, timestamp)

        return self.results

    def evaluate_complexity_group(
        self,
        questions: List[dict],
        complexity: str
    ) -> Dict[str, Any]:
        """
        Evaluate all models on a complexity group

        Args:
            questions: List of question dictionaries
            complexity: "single_hop" or "multi_hop"

        Returns:
            Evaluation results for this complexity group
        """
        group_results = {}

        # Prepare contexts if using fixed context
        fixed_contexts = {}
        if self.use_fixed_context and self.retriever:
            logger.info("고정 컨텍스트 준비 중...")
            for q in tqdm(questions, desc="Retrieving contexts"):
                question_text = q.get("question", q.get("user_input", ""))
                # Retrieve context once
                contexts = self._retrieve_context(question_text)
                fixed_contexts[question_text] = contexts

        # Evaluate each model
        for model_config in self.models:
            model_name = model_config["name"]
            logger.info(f"\n평가 중: {model_name}")

            try:
                # Test connection first
                if not self._test_model_connection(model_config):
                    logger.error(f"❌ {model_name} 연결 실패, 건너뜀")
                    continue

                # Generate answers and evaluate
                model_results = self._evaluate_model(
                    model_config=model_config,
                    questions=questions,
                    fixed_contexts=fixed_contexts if self.use_fixed_context else None
                )

                group_results[model_name] = model_results
                logger.info(f"✅ {model_name} 평가 완료")

            except Exception as e:
                logger.error(f"❌ {model_name} 평가 실패: {e}")
                group_results[model_name] = {"error": str(e)}

        return group_results

    def _retrieve_context(self, question: str) -> List[str]:
        """
        Retrieve context for a question using LangChain retriever.

        Args:
            question: Question text

        Returns:
            List of retrieved context strings
        """
        if not self.retriever:
            return []

        try:
            # Use LangChain Runnable.invoke() interface
            documents = self.retriever.invoke(question)
            contexts = [doc.page_content for doc in documents]
            return contexts
        except Exception as e:
            logger.warning(f"⚠️ Context retrieval failed for '{question[:50]}...': {e}")
            return []

    def _test_model_connection(self, model_config: Dict[str, str]) -> bool:
        """
        Test if model is accessible

        Args:
            model_config: Model configuration

        Returns:
            True if model is accessible
        """
        try:
            response = litellm.completion(
                model=model_config["model"],
                messages=[{"role": "user", "content": "Hello"}],
                api_base=model_config.get("api_base"),
                max_tokens=10,
                timeout=10
            )
            return True
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False

    def _evaluate_model(
        self,
        model_config: Dict[str, str],
        questions: List[dict],
        fixed_contexts: Optional[Dict[str, List[str]]] = None
    ) -> Dict[str, Any]:
        """
        Evaluate a single model on questions

        Args:
            model_config: Model configuration
            questions: List of questions
            fixed_contexts: Pre-retrieved contexts (optional)

        Returns:
            Model evaluation results
        """
        # Prepare data for evaluation
        eval_data = []

        for q in tqdm(questions, desc=f"Generating with {model_config['name']}"):
            question_text = q.get("question", q.get("user_input", ""))
            reference_answer = q.get("ground_truth", q.get("reference_answer", q.get("reference", "")))

            # Get context
            if fixed_contexts and question_text in fixed_contexts:
                contexts = fixed_contexts[question_text]
            else:
                contexts = self._retrieve_context(question_text)

            # Generate answer
            answer = self._generate_answer(
                model_config=model_config,
                question=question_text,
                contexts=contexts
            )

            eval_data.append({
                "user_input": question_text,
                "response": answer,
                "retrieved_contexts": contexts,
                "reference": reference_answer
            })

        # Create dataset for RAGAS
        dataset = Dataset.from_list(eval_data)

        # Run RAGAS evaluation
        try:
            logger.info(f"RAGAS 평가 시작 ({model_config['name']})...")

            # Configure metrics
            metrics = [
                faithfulness,
                answer_relevancy,
                answer_correctness
            ]

            # Evaluate with RAGAS
            evaluation = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=self._get_judge_llm(),
                run_config=RunConfig(
                    max_workers=4,
                    timeout=180
                )
            )

            # Extract scores using to_pandas() method
            # evaluation.scores is a list of dicts, convert to DataFrame first
            df_results = evaluation.to_pandas()

            # Extract metric scores as lists
            metrics_data = {}
            for metric_name in ["faithfulness", "answer_relevancy", "answer_correctness"]:
                if metric_name in df_results.columns:
                    metrics_data[metric_name] = df_results[metric_name].tolist()
                else:
                    metrics_data[metric_name] = []

            return {
                "raw_data": eval_data,
                "metrics": metrics_data,
                "summary": {
                    "faithfulness": self._safe_mean(metrics_data.get("faithfulness", [])),
                    "answer_relevancy": self._safe_mean(metrics_data.get("answer_relevancy", [])),
                    "answer_correctness": self._safe_mean(metrics_data.get("answer_correctness", [])),
                    "num_questions": len(eval_data)
                }
            }

        except Exception as e:
            logger.error(f"RAGAS evaluation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "error": str(e),
                "raw_data": eval_data,
                "summary": {
                    "num_questions": len(eval_data),
                    "evaluation_failed": True
                }
            }

    def _generate_answer(
        self,
        model_config: Dict[str, str],
        question: str,
        contexts: List[str]
    ) -> str:
        """
        Generate answer using LiteLLM

        Args:
            model_config: Model configuration
            question: Question text
            contexts: Retrieved contexts

        Returns:
            Generated answer
        """
        # Prepare context string
        context_str = "\n\n".join(contexts) if contexts else "관련 정보를 찾을 수 없습니다."

        # Create prompt
        prompt = f"""다음 컨텍스트를 바탕으로 질문에 답변해주세요.

컨텍스트:
{context_str}

질문: {question}

답변:"""

        try:
            # Get max_tokens from model config, default to 2048
            max_tokens = 2048
            if "config" in model_config:
                max_tokens = model_config["config"].get("max_tokens", 2048)

            response = litellm.completion(
                model=model_config["model"],
                messages=[
                    {"role": "system", "content": "당신은 서울 대중교통 정보를 제공하는 도우미입니다."},
                    {"role": "user", "content": prompt}
                ],
                api_base=model_config.get("api_base"),
                temperature=0.7,
                max_tokens=max_tokens,
                timeout=60
            )

            return response.choices[0].message.content

        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return f"Error: {str(e)}"

    def _get_judge_llm(self):
        """Get LLM for RAGAS judge"""
        from langchain_openai import ChatOpenAI
        import os

        # Claude models (local OpenAI-compatible endpoint)
        claude_models = [
            "claude-sonnet-4-5-20250929",
            "claude-haiku-4-5-20251001",
            "claude-opus-4-1-20250805"
        ]

        # Check if using local Claude Code endpoint
        if self.judge_model in claude_models:
            # Use local OpenAI-compatible Claude Code endpoint
            api_base = os.getenv("CLAUDE_CODE_API_BASE", "http://localhost:25292/v1")
            api_key = os.getenv("CLAUDE_CODE_API_KEY", "dummy")

            return ChatOpenAI(
                model=self.judge_model,
                openai_api_base=api_base,
                openai_api_key=api_key,
                temperature=1.0,  # Claude default temperature
                max_tokens=8192
            )

        # GPT-5는 thinking 모델로 temperature와 max_tokens 파라미터를 지원하지 않음
        elif self.judge_model == "gpt-5":
            return ChatOpenAI(
                model=self.judge_model
                # temperature, max_tokens 파라미터 모두 제외
            )
        # GPT-4o-mini는 evaluation judge로 충분한 tokens 필요
        elif self.judge_model == "gpt-4o-mini":
            return ChatOpenAI(
                model=self.judge_model,
                temperature=0.0,
                max_tokens=8192  # Sufficient for detailed evaluation
            )
        else:
            return ChatOpenAI(
                model=self.judge_model,
                temperature=0.0,
                max_tokens=4096
            )

    def _safe_mean(self, values: List[float]) -> float:
        """Calculate mean safely"""
        if not values:
            return 0.0
        return sum(values) / len(values)

    def _generate_comparison(self) -> Dict[str, Any]:
        """Generate comparison analysis across models and complexity levels"""
        comparison = {
            "by_complexity": {},
            "by_model": {},
            "best_performers": {}
        }

        # Compare by complexity
        for complexity in ["single_hop", "multi_hop"]:
            if complexity not in self.results or not self.results[complexity]:
                continue

            complexity_scores = {}
            for model_name, model_results in self.results[complexity].items():
                if "summary" in model_results:
                    complexity_scores[model_name] = model_results["summary"]

            if complexity_scores:
                # Find best model for each metric
                best_faithfulness = max(
                    complexity_scores.items(),
                    key=lambda x: x[1].get("faithfulness", 0)
                )[0]
                best_relevancy = max(
                    complexity_scores.items(),
                    key=lambda x: x[1].get("answer_relevancy", 0)
                )[0]
                best_correctness = max(
                    complexity_scores.items(),
                    key=lambda x: x[1].get("answer_correctness", 0)
                )[0]

                comparison["by_complexity"][complexity] = {
                    "scores": complexity_scores,
                    "best_faithfulness": best_faithfulness,
                    "best_relevancy": best_relevancy,
                    "best_correctness": best_correctness
                }

        # Compare by model
        for model_config in self.models:
            model_name = model_config["name"]
            model_comparison = {}

            for complexity in ["single_hop", "multi_hop"]:
                if (complexity in self.results and
                        model_name in self.results[complexity] and
                        "summary" in self.results[complexity][model_name]):
                    model_comparison[complexity] = self.results[complexity][model_name]["summary"]

            if model_comparison:
                # Calculate performance gap
                if "single_hop" in model_comparison and "multi_hop" in model_comparison:
                    gap = {}
                    for metric in ["faithfulness", "answer_relevancy", "answer_correctness"]:
                        single_score = model_comparison["single_hop"].get(metric, 0)
                        multi_score = model_comparison["multi_hop"].get(metric, 0)
                        gap[metric] = single_score - multi_score

                    model_comparison["performance_gap"] = gap

                comparison["by_model"][model_name] = model_comparison

        return comparison

    def _save_results(self, output_dir: Path, timestamp: str):
        """Save results to files"""
        # Save main results
        results_file = output_dir / f"benchmark_results_{timestamp.replace(':', '-')}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"💾 결과 저장: {results_file}")

        # Save summary report
        self._save_summary_report(output_dir, timestamp)

    def _save_summary_report(self, output_dir: Path, timestamp: str):
        """Generate and save human-readable summary report"""
        report_file = output_dir / f"summary_report_{timestamp.replace(':', '-')}.md"

        report = ["# Generation Benchmark Report", ""]
        report.append(f"**Generated**: {timestamp}")
        report.append(f"**Total Questions**: {self.results['metadata']['total_questions']}")
        report.append(f"**Models**: {', '.join(self.results['metadata']['models'])}")
        report.append("")

        # Results by complexity
        report.append("## Results by Complexity")
        for complexity in ["single_hop", "multi_hop"]:
            if complexity not in self.results["comparison"]["by_complexity"]:
                continue

            report.append(f"\n### {complexity.replace('_', ' ').title()}")
            comp_data = self.results["comparison"]["by_complexity"][complexity]

            # Create table
            report.append("\n| Model | Faithfulness | Relevancy | Correctness |")
            report.append("|-------|-------------|-----------|-------------|")

            for model, scores in comp_data["scores"].items():
                report.append(
                    f"| {model} | "
                    f"{scores.get('faithfulness', 0):.3f} | "
                    f"{scores.get('answer_relevancy', 0):.3f} | "
                    f"{scores.get('answer_correctness', 0):.3f} |"
                )

        # Performance gaps
        report.append("\n## Performance Gap Analysis")
        report.append("\n(Single-hop score - Multi-hop score)")
        report.append("\n| Model | Faithfulness Gap | Relevancy Gap | Correctness Gap |")
        report.append("|-------|-----------------|---------------|-----------------|")

        for model, data in self.results["comparison"]["by_model"].items():
            if "performance_gap" in data:
                gap = data["performance_gap"]
                report.append(
                    f"| {model} | "
                    f"{gap.get('faithfulness', 0):+.3f} | "
                    f"{gap.get('answer_relevancy', 0):+.3f} | "
                    f"{gap.get('answer_correctness', 0):+.3f} |"
                )

        # Best performers
        report.append("\n## Best Performers")
        for complexity in ["single_hop", "multi_hop"]:
            if complexity in self.results["comparison"]["by_complexity"]:
                comp = self.results["comparison"]["by_complexity"][complexity]
                report.append(f"\n**{complexity.replace('_', ' ').title()}:**")
                report.append(f"- Best Faithfulness: {comp.get('best_faithfulness', 'N/A')}")
                report.append(f"- Best Relevancy: {comp.get('best_relevancy', 'N/A')}")
                report.append(f"- Best Correctness: {comp.get('best_correctness', 'N/A')}")

        # Save report
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        logger.info(f"📝 요약 리포트 저장: {report_file}")
