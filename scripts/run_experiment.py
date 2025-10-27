#!/usr/bin/env python
"""
Automated Experiment Runner for Humetro AI Assistant Research
Executes all 16 system combinations (4 LLMs × 4 RAG methods)
"""

import argparse
import json
import yaml
import time
from pathlib import Path
from typing import Dict, List, Any
import logging
from datetime import datetime
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.rag_pipeline.base_rag import (
    BaselineRAG, NaiveRAG, AdvancedRAG, GraphRAG, RAGQuery
)
from src.evaluation.evaluator import (
    RAGASEvaluator, LLMAsJudge, AblationAnalyzer,
    ExperimentTracker, EvaluationSample, EvaluationResults
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Main experiment runner"""

    def __init__(self, config_path: str):
        """Initialize experiment runner with config"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_name = self.config['experiment']['name']

        # Initialize evaluators
        self.ragas_evaluator = RAGASEvaluator()
        self.llm_judge = LLMAsJudge(
            judge_model=self.config['evaluation']['llm_judge']['model']
        )
        self.ablation_analyzer = AblationAnalyzer()

        # Initialize experiment tracker
        results_dir = self.config['output']['results_dir'].format(
            timestamp=self.timestamp,
            experiment_name=self.experiment_name
        )
        self.tracker = ExperimentTracker(results_dir)

        # Results storage
        self.all_results = {}

    def load_model(self, model_config: Dict[str, Any]):
        """Load language model"""
        logger.info(f"Loading model: {model_config['name']}")

        if 'api' in model_config:
            # Load API-based model
            if model_config['api'] == 'openai':
                from langchain_openai import ChatOpenAI
                return ChatOpenAI(
                    model=model_config['model'],
                    temperature=model_config['temperature'],
                    max_tokens=model_config.get('max_tokens', 2048)
                )
            elif model_config['api'] == 'google':
                from langchain_google_genai import ChatGoogleGenerativeAI
                return ChatGoogleGenerativeAI(
                    model=model_config['model'],
                    temperature=model_config['temperature'],
                    max_output_tokens=model_config.get('max_tokens', 2048)
                )
        else:
            # Load local model with quantization
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            model_path = model_config['path']
            quantization = model_config.get('quantization', 'none')

            # Mock implementation - replace with actual model loading
            logger.info(f"Loading {model_path} with {quantization} quantization")

            # In production, implement actual model loading with quantization
            class MockLLM:
                def generate(self, prompt, temperature=0.7):
                    return f"Generated answer for: {prompt[:50]}..."

            return MockLLM()

    def initialize_rag_method(
        self,
        method_name: str,
        method_config: Dict[str, Any],
        model: Any
    ) -> Any:
        """Initialize RAG method"""
        logger.info(f"Initializing RAG method: {method_name}")

        rag_type = method_config['type']

        if rag_type == 'baseline':
            return BaselineRAG(
                retriever=None,
                reranker=None,
                generator=model,
                config=method_config.get('config', {})
            )

        elif rag_type == 'naive':
            # Initialize vector store retriever
            retriever = self._init_vector_retriever(method_config['config'])
            return NaiveRAG(
                retriever=retriever,
                reranker=None,
                generator=model,
                config=method_config['config']
            )

        elif rag_type == 'advanced':
            # Initialize hybrid retriever and reranker
            retriever = self._init_hybrid_retriever(method_config['config'])
            reranker = self._init_reranker(method_config['config'])
            return AdvancedRAG(
                retriever=retriever,
                reranker=reranker,
                generator=model,
                config=method_config['config']
            )

        elif rag_type == 'graph':
            # Initialize graph database and retriever
            graph_db = self._init_graph_db(method_config['config'])
            retriever = self._init_graph_retriever(method_config['config'])
            return GraphRAG(
                retriever=retriever,
                reranker=None,
                generator=model,
                graph_db=graph_db,
                config=method_config['config']
            )

        else:
            raise ValueError(f"Unknown RAG type: {rag_type}")

    def _init_vector_retriever(self, config: Dict[str, Any]):
        """Initialize vector store retriever"""
        # Mock implementation
        class MockRetriever:
            def search(self, query, top_k=5, metadata_filter=None):
                return [
                    {
                        'id': f'doc_{i}',
                        'content': f'Retrieved document {i} for query: {query[:30]}',
                        'score': 0.9 - i * 0.1
                    }
                    for i in range(top_k)
                ]
        return MockRetriever()

    def _init_hybrid_retriever(self, config: Dict[str, Any]):
        """Initialize hybrid retriever"""
        # Mock implementation
        class MockHybridRetriever:
            def bm25_search(self, query, top_k=5):
                return [
                    {
                        'id': f'bm25_doc_{i}',
                        'content': f'BM25 result {i}',
                        'score': 0.8 - i * 0.1
                    }
                    for i in range(top_k)
                ]

            def semantic_search(self, query, top_k=5):
                return [
                    {
                        'id': f'semantic_doc_{i}',
                        'content': f'Semantic result {i}',
                        'score': 0.9 - i * 0.1
                    }
                    for i in range(top_k)
                ]

            def get_document(self, doc_id):
                return {
                    'id': doc_id,
                    'content': f'Full document content for {doc_id}'
                }

        return MockHybridRetriever()

    def _init_reranker(self, config: Dict[str, Any]):
        """Initialize reranker"""
        # Mock implementation
        class MockReranker:
            def rerank(self, query, documents):
                # Simply reverse the order as mock reranking
                return documents[::-1]
        return MockReranker()

    def _init_graph_db(self, config: Dict[str, Any]):
        """Initialize graph database"""
        # Mock implementation
        class MockGraphDB:
            def find_nodes(self, entity, max_hops=2):
                return [
                    {
                        'id': f'node_{entity}_{i}',
                        'content': f'Node content for {entity}',
                        'type': 'entity',
                        'relations': [f'rel_{j}' for j in range(2)]
                    }
                    for i in range(3)
                ]

            def extract_subgraph(self, nodes):
                return {
                    'nodes': nodes,
                    'edges': []
                }
        return MockGraphDB()

    def _init_graph_retriever(self, config: Dict[str, Any]):
        """Initialize graph retriever"""
        # Use mock retriever for now
        return self._init_vector_retriever(config)

    def load_evaluation_data(self) -> List[Dict[str, str]]:
        """Load evaluation dataset"""
        logger.info("Loading evaluation data")

        # Mock implementation - load actual data in production
        evaluation_data = []
        for i in range(100):  # Use smaller set for testing
            evaluation_data.append({
                'question': f'테스트 질문 {i}: 부산 도시철도 관련 문의입니다.',
                'ground_truth': f'정답 {i}: 부산 도시철도 관련 답변입니다.',
                'metadata': {'id': f'qa_{i}', 'category': 'transport'}
            })

        logger.info(f"Loaded {len(evaluation_data)} evaluation samples")
        return evaluation_data

    def run_single_experiment(
        self,
        model_name: str,
        rag_method_name: str,
        model: Any,
        rag_pipeline: Any,
        evaluation_data: List[Dict[str, str]]
    ) -> EvaluationResults:
        """Run single experiment configuration"""
        logger.info(f"Running experiment: {model_name} + {rag_method_name}")

        samples = []
        for data in evaluation_data:
            # Create query
            query = RAGQuery(
                text=data['question'],
                metadata=data.get('metadata'),
                top_k=5,
                temperature=0.7
            )

            # Run RAG pipeline
            start_time = time.time()
            response = rag_pipeline(query)
            total_latency = (time.time() - start_time) * 1000

            # Create evaluation sample
            sample = EvaluationSample(
                question=data['question'],
                ground_truth=data['ground_truth'],
                generated_answer=response.answer,
                contexts=response.contexts,
                retrieval_latency_ms=response.latency_ms * 0.3,  # Mock split
                generation_latency_ms=response.latency_ms * 0.7,
                total_latency_ms=total_latency,
                metadata={
                    'model': model_name,
                    'rag_method': rag_method_name
                }
            )
            samples.append(sample)

        # Evaluate with RAGAS
        ragas_metrics = self.ragas_evaluator.evaluate(samples)

        # Evaluate with LLM Judge (optional, can be expensive)
        llm_judge_results = None
        if self.config['evaluation'].get('use_llm_judge', False):
            llm_judge_results = self.llm_judge.evaluate(samples[:10])  # Sample

        # Create results
        results = EvaluationResults(
            model_name=model_name,
            rag_method=rag_method_name,
            samples=samples,
            metrics=ragas_metrics,
            timestamp=self.timestamp
        )

        # Save results
        experiment_id = f"{model_name}_{rag_method_name}"
        self.tracker.save_results(results, experiment_id)

        return results

    def run_all_experiments(self):
        """Run all experiment combinations"""
        evaluation_data = self.load_evaluation_data()

        # Get all models and RAG methods
        models = self.config['models']['opensource']
        if self.config.get('include_baseline', False):
            models.extend(self.config['models']['baseline'])

        rag_methods = self.config['rag_methods']

        total_experiments = len(models) * len(rag_methods)
        logger.info(f"Starting {total_experiments} experiments")

        # Run each combination
        for model_config in models:
            model = self.load_model(model_config)
            model_name = model_config['name']

            for method_name, method_config in rag_methods.items():
                # Initialize RAG pipeline
                rag_pipeline = self.initialize_rag_method(
                    method_name,
                    method_config,
                    model
                )

                # Run experiment
                results = self.run_single_experiment(
                    model_name,
                    method_name,
                    model,
                    rag_pipeline,
                    evaluation_data
                )

                # Store results
                key = f"{model_name}_{method_name}"
                self.all_results[key] = results

                logger.info(f"Completed: {key}")
                logger.info(f"Metrics: {results.metrics}")

        # Generate comparison table
        self._generate_final_report()

    def _generate_final_report(self):
        """Generate final comparison report"""
        logger.info("Generating final report")

        # Create comparison table
        experiment_ids = list(self.all_results.keys())
        comparison_df = self.tracker.generate_comparison_table(experiment_ids)

        logger.info("\n" + "="*80)
        logger.info("EXPERIMENT RESULTS SUMMARY")
        logger.info("="*80)
        print(comparison_df.to_string())

        # Save as CSV
        csv_path = Path(self.tracker.results_dir) / "final_comparison.csv"
        comparison_df.to_csv(csv_path, index=False)

        logger.info(f"\nResults saved to: {self.tracker.results_dir}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run Humetro RAG experiments"
    )
    parser.add_argument(
        '--config',
        type=str,
        default='experiments/configs/base_config.yaml',
        help='Path to experiment config file'
    )
    parser.add_argument(
        '--models',
        type=str,
        nargs='+',
        help='Specific models to run (default: all)'
    )
    parser.add_argument(
        '--methods',
        type=str,
        nargs='+',
        help='Specific RAG methods to run (default: all)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Run in debug mode with reduced dataset'
    )

    args = parser.parse_args()

    # Run experiments
    runner = ExperimentRunner(args.config)
    runner.run_all_experiments()


if __name__ == "__main__":
    main()