"""
Evaluate RAG pipeline using AutoRAG and generated QA datasets.
This module loads QA and corpus data, runs optimization and 
evaluates RAG pipeline performance.
"""

import os
import sys
from pathlib import Path

# Get root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.get_root import get_project_root

# AutoRAG imports
try:
    # 최신 패키지 구조 시도
    try:
        from autorag.evaluator import Evaluator
        from autorag.validator import Validator
        print("Using standard AutoRAG module structure")
    except ImportError:
        # 대체 구조 시도
        from autorag.legacy.evaluator import Evaluator
        from autorag.legacy.validator import Validator
        print("Using legacy AutoRAG module structure")
except ImportError:
    print("AutoRAG not installed. Installing now...")
    import subprocess
    subprocess.run(["pip", "install", "AutoRAG"])
    try:
        from autorag.evaluator import Evaluator
        from autorag.validator import Validator
        print("Using standard AutoRAG module structure")
    except ImportError:
        from autorag.legacy.evaluator import Evaluator
        from autorag.legacy.validator import Validator
        print("Using legacy AutoRAG module structure")

def evaluate_rag_pipeline(
    config_path: str,
    qa_data_path: str,
    corpus_data_path: str,
    project_dir: str
):
    """
    Evaluate RAG pipeline using AutoRAG.
    
    Args:
        config_path: Path to RAG configuration file
        qa_data_path: Path to QA dataset
        corpus_data_path: Path to corpus dataset
        project_dir: Directory to save evaluation results
    """
    print(f"Evaluating RAG pipeline with config: {config_path}")
    print(f"QA data: {qa_data_path}")
    print(f"Corpus data: {corpus_data_path}")
    print(f"Project directory: {project_dir}")
    
    # Create project directory if it doesn't exist
    os.makedirs(project_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = Evaluator(
        qa_data_path=qa_data_path,
        corpus_data_path=corpus_data_path
    )
    
    # Start evaluation
    evaluator.start_trial(config_path)
    
    print("Evaluation complete. Check results in the project directory.")

def validate_rag_pipeline(
    config_path: str,
    qa_data_path: str,
    corpus_data_path: str
):
    """
    Validate RAG pipeline configuration using AutoRAG.
    
    Args:
        config_path: Path to RAG configuration file
        qa_data_path: Path to QA dataset
        corpus_data_path: Path to corpus dataset
    """
    print(f"Validating RAG pipeline with config: {config_path}")
    print(f"QA data: {qa_data_path}")
    print(f"Corpus data: {corpus_data_path}")
    
    # Initialize validator
    validator = Validator(
        qa_data_path=qa_data_path,
        corpus_data_path=corpus_data_path
    )
    
    # Start validation
    validator.validate(config_path)
    
    print("Validation complete.")

def main():
    """
    Main function to evaluate RAG pipeline.
    """
    # Set paths
    root_dir = get_project_root()
    data_processor_dir = os.path.dirname(os.path.abspath(__file__))
    datasets_dir = os.path.join(root_dir, "datasets")
    final_docs_dir = os.path.join(datasets_dir, "final_docs")
    
    # Set input paths
    config_path = os.path.join(data_processor_dir, "rag_config.yaml")
    qa_data_path = os.path.join(final_docs_dir, "qa_test.parquet")
    corpus_data_path = os.path.join(final_docs_dir, "corpus.parquet")
    
    # Set output path
    project_dir = os.path.join(datasets_dir, "rag_evaluation")
    
    # Check if required files exist
    if not os.path.exists(config_path):
        print(f"Error: RAG configuration file not found at {config_path}")
        return
    
    if not os.path.exists(qa_data_path):
        print(f"Error: QA dataset not found at {qa_data_path}")
        return
    
    if not os.path.exists(corpus_data_path):
        print(f"Error: Corpus dataset not found at {corpus_data_path}")
        return
    
    # Load OpenAI API key from .env file
    from dotenv import load_dotenv
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        return
    
    # Validate pipeline first
    validate_rag_pipeline(
        config_path=config_path,
        qa_data_path=qa_data_path,
        corpus_data_path=corpus_data_path
    )
    
    # Evaluate pipeline
    evaluate_rag_pipeline(
        config_path=config_path,
        qa_data_path=qa_data_path,
        corpus_data_path=corpus_data_path,
        project_dir=project_dir
    )

if __name__ == "__main__":
    main()
