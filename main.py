"""
Main CLI script to run the RAG pipeline.
"""

import os
import sys
import argparse

# Get root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.get_root import get_project_root
from src.data_processor.generate_qa_data import (
    load_markdown_files,
    chunk_documents,
    create_corpus_dataframe,
    generate_qa_dataset,
)


def generate_data_command(args):
    """
    Generate corpus and QA datasets from markdown files.
    """
    # Set paths
    root_dir = get_project_root()
    crawl_dir = os.path.join(root_dir, "crawl_result")
    datasets_dir = os.path.join(root_dir, "datasets")
    final_docs_dir = os.path.join(datasets_dir, "final_docs")

    # Create directories if they don't exist
    os.makedirs(final_docs_dir, exist_ok=True)

    # Set output paths
    corpus_path = os.path.join(final_docs_dir, "corpus.parquet")
    qa_path = os.path.join(final_docs_dir, "qa.parquet")

    # Load OpenAI API key from .env file
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("Error: OPENAI_API_KEY not found in environment variables.")
        return

    # Load and process documents
    documents = load_markdown_files(crawl_dir)
    chunks = chunk_documents(
        documents, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )
    corpus_df = create_corpus_dataframe(chunks, corpus_path)

    # Generate QA dataset
    qa_df = generate_qa_dataset(
        corpus_df=corpus_df,
        output_path=qa_path,
        num_samples=args.num_samples,
        questions_per_content=args.questions_per_content,
        api_key=api_key,
    )

    print("QA dataset generation complete.")
    print(f"Corpus saved to: {corpus_path}")
    print(f"QA dataset saved to: {qa_path}")

    # Split into train and test datasets
    train_size = int(len(qa_df) * args.train_ratio)
    qa_train_df = qa_df.iloc[:train_size]
    qa_test_df = qa_df.iloc[train_size:]

    # Save train and test datasets
    qa_train_path = os.path.join(final_docs_dir, "qa_train.parquet")
    qa_test_path = os.path.join(final_docs_dir, "qa_test.parquet")

    qa_train_df.to_parquet(qa_train_path)
    qa_test_df.to_parquet(qa_test_path)

    print(f"Train dataset saved to: {qa_train_path}")
    print(f"Test dataset saved to: {qa_test_path}")


def evaluate_rag_command(args):
    """
    Evaluate RAG pipeline using AutoRAG.
    """
    # Set paths
    root_dir = get_project_root()
    data_processor_dir = os.path.join(root_dir, "src", "data_processor")
    datasets_dir = os.path.join(root_dir, "datasets")
    final_docs_dir = os.path.join(datasets_dir, "final_docs")

    # Set input paths
    config_path = os.path.join(data_processor_dir, "rag_config.yaml")
    qa_data_path = os.path.join(final_docs_dir, "qa_test.parquet")
    corpus_data_path = os.path.join(final_docs_dir, "corpus.parquet")

    # Set output path
    project_dir = os.path.join(datasets_dir, "rag_evaluation")

    # Import evaluate_rag_pipeline
    try:
        # 다이나믹 임포트 시도
        import importlib

        module = importlib.import_module("src.data_processor.evaluate_rag")
        evaluate_rag_pipeline = module.evaluate_rag_pipeline
    except (ImportError, AttributeError) as e:
        print(f"Error importing evaluate_rag_pipeline: {e}")
        print("Trying alternative import method...")
        import sys

        sys.path.append(os.path.join(root_dir, "src", "data_processor"))
        from evaluate_rag import evaluate_rag_pipeline

    # Evaluate pipeline
    evaluate_rag_pipeline(
        config_path=config_path,
        qa_data_path=qa_data_path,
        corpus_data_path=corpus_data_path,
        project_dir=project_dir,
    )


def validate_rag_command(args):
    """
    Validate RAG pipeline configuration using AutoRAG.
    """
    # Set paths
    root_dir = get_project_root()
    data_processor_dir = os.path.join(root_dir, "src", "data_processor")
    datasets_dir = os.path.join(root_dir, "datasets")
    final_docs_dir = os.path.join(datasets_dir, "final_docs")

    # Set input paths
    config_path = os.path.join(data_processor_dir, "rag_config.yaml")
    qa_data_path = os.path.join(final_docs_dir, "qa_test.parquet")
    corpus_data_path = os.path.join(final_docs_dir, "corpus.parquet")

    # Import validate_rag_pipeline
    try:
        # 다이나믹 임포트 시도
        import importlib

        module = importlib.import_module("src.data_processor.evaluate_rag")
        validate_rag_pipeline = module.validate_rag_pipeline
    except (ImportError, AttributeError) as e:
        print(f"Error importing validate_rag_pipeline: {e}")
        print("Trying alternative import method...")
        import sys

        sys.path.append(os.path.join(root_dir, "src", "data_processor"))
        from evaluate_rag import validate_rag_pipeline

    # Validate pipeline
    validate_rag_pipeline(
        config_path=config_path,
        qa_data_path=qa_data_path,
        corpus_data_path=corpus_data_path,
    )


def main():
    """
    Main function to parse arguments and run commands.
    """
    parser = argparse.ArgumentParser(description="RAG Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Generate data command
    generate_parser = subparsers.add_parser(
        "generate", help="Generate corpus and QA datasets"
    )
    generate_parser.add_argument(
        "--chunk-size", type=int, default=512, help="Chunk size for document splitting"
    )
    generate_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=128,
        help="Chunk overlap for document splitting",
    )
    generate_parser.add_argument(
        "--num-samples", type=int, default=100, help="Number of QA samples to generate"
    )
    generate_parser.add_argument(
        "--questions-per-content",
        type=int,
        default=2,
        help="Number of questions per content",
    )
    generate_parser.add_argument(
        "--train-ratio", type=float, default=0.8, help="Ratio of training data"
    )

    # Evaluate RAG command
    subparsers.add_parser("evaluate", help="Evaluate RAG pipeline")

    # Validate RAG command
    subparsers.add_parser("validate", help="Validate RAG pipeline configuration")

    args = parser.parse_args()

    if args.command == "generate":
        generate_data_command(args)
    elif args.command == "evaluate":
        evaluate_rag_command(args)
    elif args.command == "validate":
        validate_rag_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
