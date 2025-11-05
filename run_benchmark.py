#!/usr/bin/env python3
"""
Unified RAG Benchmark Runner
=============================

Configuration-driven benchmark execution system.
Supports all experiment types through JSON config files.

Usage:
    python run_benchmark.py --config config/healthcheck.json
    python run_benchmark.py --config config/examples/compare_naive_rag.json
    python run_benchmark.py --config config/examples/generate_100q_full.json
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv()

from src.rag_pipeline.question_generation import generate_questions, load_cached_questions
from src.rag_pipeline.generation_benchmark import GenerationBenchmark

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and validate benchmark configuration"""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.config = self._load_config()
        self._validate_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load JSON config file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _validate_config(self):
        """Basic validation of required sections"""
        required = ["experiment", "questions", "models", "evaluation"]
        missing = [r for r in required if r not in self.config]

        if missing:
            raise ValueError(f"Missing required sections: {missing}")

        # Validate questions source
        if self.config["questions"]["source"] not in ["cached", "generate"]:
            raise ValueError(f"Invalid questions.source: {self.config['questions']['source']}")

        # Validate retrieval method if present
        if "retrieval" in self.config:
            method = self.config["retrieval"].get("method", "naive")
            if method not in ["none", "naive", "kg", "hybrid"]:
                raise ValueError(f"Invalid retrieval.method: {method}")

    def get(self, key: str, default=None) -> Any:
        """Get config value by dot notation key"""
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default

        return value


class QuestionLoader:
    """Load or generate questions based on config"""

    @staticmethod
    def load(config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Load questions from cache or generate new ones

        Returns:
            List of question dicts with fields: user_input, reference, reference_contexts, etc.
        """
        questions_config = config["questions"]
        source = questions_config["source"]

        if source == "cached":
            logger.info("Loading cached questions...")
            cache_key = questions_config["cache_key"]
            df = load_cached_questions(cache_key)

            logger.info(f"✅ Loaded {len(df)} cached questions (key: {cache_key})")

        elif source == "generate":
            logger.info("Generating new questions...")
            gen_config = questions_config["generation"]

            _, df = generate_questions(
                model=gen_config.get("model", "gpt-4o-mini"),
                source=gen_config.get("document_source", "data/crawled/seoul_traffic/markdown_filtered"),
                num_documents=gen_config.get("num_documents", 50),
                num_questions=gen_config.get("num_questions", 50),
                force_regenerate=gen_config.get("force_regenerate", False),
                temperature=gen_config.get("temperature", 0.3),
                embedding_model=gen_config.get("embedding_model", "text-embedding-3-small"),
                use_korean_personas=gen_config.get("use_korean_personas", True)
            )

            logger.info(f"✅ Generated {len(df)} new questions")

        else:
            raise ValueError(f"Invalid questions.source: {source}")

        # Convert to list of dicts
        return df.to_dict('records')


class ModelResolver:
    """Resolve model presets and configurations"""

    def __init__(self, preset_file: Path = Path("config/model_presets.json")):
        self.presets = self._load_presets(preset_file)

    def _load_presets(self, preset_file: Path) -> Dict[str, Any]:
        """Load model presets from file"""
        if not preset_file.exists():
            logger.warning(f"Preset file not found: {preset_file}, using empty presets")
            return {"presets": {}}

        with open(preset_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def resolve(self, targets: Any) -> List[Dict[str, Any]]:
        """
        Resolve evaluation targets to model list

        Args:
            targets: String (preset name) or List (custom models)

        Returns:
            List of model configurations
        """
        if isinstance(targets, str):
            # Preset name
            preset = self.presets.get("presets", {}).get(targets)
            if not preset:
                raise ValueError(f"Unknown model preset: {targets}")

            logger.info(f"Using model preset: {targets}")
            return preset["models"]

        elif isinstance(targets, list):
            # Custom model list
            logger.info(f"Using custom model list: {len(targets)} models")
            return targets

        else:
            raise ValueError(f"Invalid evaluation_targets type: {type(targets)}")


def run_benchmark_from_config(config_path: Path):
    """
    Run benchmark from configuration file

    Args:
        config_path: Path to JSON configuration file
    """
    print("="*70)
    print("RAG Benchmark Runner".center(70))
    print("="*70)

    # Load configuration
    logger.info(f"Loading config: {config_path}")
    config_loader = ConfigLoader(config_path)
    config = config_loader.config

    # Print experiment info
    exp_info = config["experiment"]
    print(f"\n📋 Experiment: {exp_info['name']}")
    print(f"📝 Description: {exp_info.get('description', 'N/A')}")
    print(f"🏷️  Tags: {', '.join(exp_info.get('tags', []))}")

    # Load questions
    print("\n" + "-"*70)
    print("Step 1: Loading Questions")
    print("-"*70)
    questions = QuestionLoader.load(config)
    print(f"✅ Questions loaded: {len(questions)}")

    # Show question types
    single_hop = sum(1 for q in questions if "single_hop" in q.get("synthesizer_name", "").lower())
    multi_hop = len(questions) - single_hop
    print(f"   Single-hop: {single_hop}, Multi-hop: {multi_hop}")

    # Resolve models
    print("\n" + "-"*70)
    print("Step 2: Resolving Models")
    print("-"*70)
    resolver = ModelResolver()
    models = resolver.resolve(config["models"]["evaluation_targets"])
    print(f"✅ Models to evaluate: {len(models)}")
    for model in models:
        print(f"   - {model.get('name', model.get('id'))}")

    # Setup retrieval
    print("\n" + "-"*70)
    print("Step 3: Setting up Retrieval")
    print("-"*70)
    retrieval_config = config.get("retrieval", {})
    k = retrieval_config.get("k", 4)

    # Check what retrieval methods to test
    naive_store = retrieval_config.get("naive_vector_store")
    kg_config = retrieval_config.get("kg_config")

    retrieval_methods = []

    if naive_store:
        print(f"✅ Naive RAG (FAISS)")
        print(f"   Vector store: {naive_store}")
        print(f"   k={k} documents")
        retrieval_methods.append("naive")
    else:
        logger.warning("⚠️ No naive_vector_store configured - skipping Naive RAG")

    if kg_config:
        print(f"✅ Knowledge Graph RAG (Neo4j)")
        print(f"   k={k} documents")
        retrieval_methods.append("kg")
    else:
        logger.info("ℹ️ No kg_config provided - skipping KG RAG")

    if not retrieval_methods:
        logger.warning("⚠️ No retrieval configured - using LLM knowledge only")
        retrieval_methods.append("none")

    # Setup evaluation
    print("\n" + "-"*70)
    print("Step 4: Setting up Evaluation")
    print("-"*70)
    eval_config = config["evaluation"]
    judge_model = eval_config["judge_model"]
    metrics = eval_config.get("metrics", ["faithfulness", "answer_relevancy", "answer_correctness"])

    print(f"✅ Judge model: {judge_model}")
    print(f"   Metrics: {', '.join(metrics)}")

    # Run benchmark for each retrieval method
    print("\n" + "="*70)
    print("Running Benchmark".center(70))
    print("="*70)

    # Convert models to GenerationBenchmark format
    benchmark_models = []
    for model in models:
        benchmark_models.append({
            "name": model.get("name", model["id"]),
            "model": model.get("model", model["id"]),
            "api_base": model.get("config", {}).get("api_base")
        })

    # Prepare questions in GenerationBenchmark format
    benchmark_questions = {
        "single_hop": [],
        "multi_hop": []
    }

    for q in questions:
        synthesizer = q.get("synthesizer_name", "")
        question_data = {
            "question": q.get("user_input", ""),
            "ground_truth": q.get("reference", ""),
            "reference_contexts": q.get("reference_contexts", [])
        }

        # Classify by synthesizer
        if "single_hop" in synthesizer.lower():
            benchmark_questions["single_hop"].append(question_data)
        else:
            benchmark_questions["multi_hop"].append(question_data)

    # Output directory
    output_dir = Path(exp_info["output_dir"]) / exp_info["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Run for each retrieval method
    for method in retrieval_methods:
        print(f"\n{'='*70}")
        print(f"Benchmark: {method.upper()} RAG".center(70))
        print(f"{'='*70}")

        try:
            if method == "naive":
                # Naive RAG with FAISS
                logger.info(f"Running Naive RAG benchmark...")
                benchmark = GenerationBenchmark(
                    models=benchmark_models,
                    use_fixed_context=True,
                    k_documents=k,
                    judge_model=judge_model
                )

                method_output = output_dir / "naive_rag"
                method_output.mkdir(parents=True, exist_ok=True)

                results = benchmark.run_benchmark(benchmark_questions, method_output)
                all_results["naive_rag"] = results
                logger.info(f"✅ Naive RAG complete: {method_output}")

            elif method == "kg":
                # KG RAG with Neo4j
                logger.info(f"Running KG RAG benchmark...")
                logger.warning("⚠️ KG RAG not yet implemented - skipping")
                # TODO: Implement KG retriever integration
                # from src.kg_agent.kg_rag_retriever import create_kg_retriever
                # kg_retriever = create_kg_retriever(...)
                # benchmark = GenerationBenchmark(models=..., retriever=kg_retriever, ...)
                all_results["kg_rag"] = {"status": "not_implemented"}

            elif method == "none":
                # LLM only (no retrieval)
                logger.info(f"Running LLM-only baseline...")
                benchmark = GenerationBenchmark(
                    models=benchmark_models,
                    use_fixed_context=False,
                    k_documents=0,
                    judge_model=judge_model
                )

                method_output = output_dir / "llm_only"
                method_output.mkdir(parents=True, exist_ok=True)

                results = benchmark.run_benchmark(benchmark_questions, method_output)
                all_results["llm_only"] = results
                logger.info(f"✅ LLM-only complete: {method_output}")

        except Exception as e:
            logger.error(f"❌ {method.upper()} RAG failed: {e}")
            logger.info(f"   Continuing with remaining methods...")
            all_results[method] = {"status": "failed", "error": str(e)}

    # Save config alongside results
    config_out = output_dir / "config.json"
    with open(config_out, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    # Save combined results summary
    summary_out = output_dir / "benchmark_summary.json"
    with open(summary_out, 'w', encoding='utf-8') as f:
        json.dump({
            "experiment": exp_info,
            "questions_count": len(questions),
            "models_count": len(models),
            "retrieval_methods": retrieval_methods,
            "results": all_results
        }, f, ensure_ascii=False, indent=2)

    print("\n" + "="*70)
    print("✅ Benchmark Complete!".center(70))
    print("="*70)
    print(f"\n📁 Results: {output_dir}")
    print(f"📄 Config: {config_out}")
    print(f"📊 Summary: {summary_out}")

    for method in retrieval_methods:
        if method in all_results and all_results[method].get("status") != "failed":
            print(f"   ✓ {method.upper()} RAG: {output_dir}/{method}_rag/")

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Run RAG benchmark from configuration file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Healthcheck (1 question, 1 model, fast)
  python run_benchmark.py --config config/healthcheck.json

  # Quick evaluation with cached questions
  python run_benchmark.py --config config/examples/quick_eval_cached.json

  # Generate 100 new questions and full evaluation
  python run_benchmark.py --config config/examples/generate_100q_full.json

  # Compare Naive RAG vs KG RAG (use same cache_key!)
  python run_benchmark.py --config config/examples/compare_naive_rag.json
  python run_benchmark.py --config config/examples/compare_kg_rag.json
        """
    )

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to JSON configuration file"
    )

    args = parser.parse_args()

    try:
        run_benchmark_from_config(args.config)

    except Exception as e:
        logger.error(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
