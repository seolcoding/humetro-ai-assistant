"""
Generate QA data from markdown files using AutoRAG and LangChain.
This module loads markdown files from the crawl_result directory,
processes them with LangChain, and generates QA datasets using AutoRAG.
"""

import os
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

# LangChain imports
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LlamaIndex imports for OpenAI
try:
    from llama_index.llms.openai import OpenAI
except ImportError:
    print("LlamaIndex OpenAI not found. Installing llama-index-llms-openai...")
    import subprocess

    subprocess.run(["pip", "install", "llama-index-llms-openai"])
    from llama_index.llms.openai import OpenAI

# AutoRAG imports
try:
    # 최신 패키지 구조 시도
    try:
        from autorag.data.legacy.corpus import langchain_documents_to_parquet
        from autorag.data.legacy.qacreation import (
            generate_qa_llama_index,
            make_single_content_qa,
        )

        print("Using legacy AutoRAG module structure")
    except ImportError:
        # 대체 구조 시도
        from autorag.data.corpus import langchain_documents_to_parquet
        from autorag.data.qacreation import (
            generate_qa_llama_index,
            make_single_content_qa,
        )

        print("Using standard AutoRAG module structure")
except ImportError:
    print("AutoRAG not installed. Installing now...")
    import subprocess

    subprocess.run(["pip", "install", "AutoRAG"])
    try:
        from autorag.data.legacy.corpus import langchain_documents_to_parquet
        from autorag.data.legacy.qacreation import (
            generate_qa_llama_index,
            make_single_content_qa,
        )

        print("Using legacy AutoRAG module structure")
    except ImportError:
        from autorag.data.corpus import langchain_documents_to_parquet
        from autorag.data.qacreation import (
            generate_qa_llama_index,
            make_single_content_qa,
        )

        print("Using standard AutoRAG module structure")

# Get root directory
import sys

sys.path.append(Path.cwd().parent.parent)


def load_markdown_files(docs_dir: str) -> List[Dict[str, Any]]:
    """
    Load all markdown files from the specified directory.

    Args:
        docs_dir: Path to the directory containing markdown files

    Returns:
        List of loaded documents
    """
    print(f"Loading markdown files from {docs_dir}")
    loader = DirectoryLoader(
        docs_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    return documents


def chunk_documents(
    documents: List[Dict[str, Any]], chunk_size: int = 512, chunk_overlap: int = 128
) -> List[Dict[str, Any]]:
    """
    Split documents into chunks using RecursiveCharacterTextSplitter.

    Args:
        documents: List of documents to split
        chunk_size: Size of each chunk in characters
        chunk_overlap: Overlap between chunks in characters

    Returns:
        List of chunked documents
    """
    print(
        f"Chunking documents with chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
    )
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def create_corpus_dataframe(
    chunks: List[Dict[str, Any]], output_path: str
) -> pd.DataFrame:
    """
    Convert document chunks to corpus dataframe and save to parquet.

    Args:
        chunks: List of document chunks
        output_path: Path to save corpus parquet file

    Returns:
        Corpus dataframe
    """
    print(f"Creating corpus dataframe and saving to {output_path}")
    corpus_df = langchain_documents_to_parquet(chunks, output_path, upsert=True)
    print(f"Created corpus with {len(corpus_df)} rows")
    return corpus_df


def generate_qa_dataset(
    corpus_df: pd.DataFrame,
    output_path: str,
    num_samples: int = 50,
    questions_per_content: int = 1,
    api_key: str = None,
) -> pd.DataFrame:
    """
    Generate QA dataset from corpus data using AutoRAG.

    Args:
        corpus_df: Corpus dataframe
        output_path: Path to save QA parquet file
        num_samples: Number of samples to generate
        questions_per_content: Number of questions to generate per content
        api_key: OpenAI API key

    Returns:
        QA dataframe
    """
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key

    print(f"Generating QA dataset with {num_samples} samples")
    # LlamaIndex의 OpenAI 모델 사용
    llm = OpenAI(model="gpt-4o-mini", temperature=1.0)

    # Custom prompt for QA generation
    prompt = """
    Based on the following passage, generate {{num_questions}} question and answer pairs.
    
    Passage: {{text}}
    
    NOTE That this Q/A is between customer and subway station clerk.
    Hence you should generate questions that are relevant to the customer's query.
    Question should be a specific query that can be answered using information from the passage.
    Answer should provide a clear and concise response based solely on information in the passage.

    You should not generate questions about train schedule, travel time, or directions
    because there are no information about train schedule or travel time in the passage.
    
    Format your response as follows:
    [Q]: Question here?
    [A]: Answer here.
    
    Questions should be in Korean and answers should be in Korean.
    """

    # Generate QA pairs
    qa_df = make_single_content_qa(
        corpus_df=corpus_df,
        content_size=num_samples,
        qa_creation_func=generate_qa_llama_index,
        llm=llm,
        prompt=prompt,
        question_num_per_content=questions_per_content,
        output_filepath=output_path,
        cache_batch=10,  # Save progress every 10 samples
        upsert=True,
    )

    print(f"Generated QA dataset with {len(qa_df)} rows")
    return qa_df


def main():
    """
    Main function to generate QA data from markdown files.
    """
    # Set paths
    root_dir = Path.cwd()
    # 수정: 올바른 Markdown 파일 경로
    rag_docs_dir = os.path.join(root_dir, "datasets", "final_docs")
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
    # 수정: rag_docs_dir에서 문서 로드
    documents = load_markdown_files(rag_docs_dir)
    chunks = chunk_documents(documents)
    corpus_df = create_corpus_dataframe(chunks, corpus_path)

    # Generate QA dataset
    qa_df = generate_qa_dataset(
        corpus_df=corpus_df,
        output_path=qa_path,
        num_samples=500,  # 50개로 줄임 (처리 시간 감소)
        questions_per_content=5,  # 1개로 설정 (처리 시간 감소)
        api_key=api_key,
    )

    print("QA dataset generation complete.")
    print(f"Corpus saved to: {corpus_path}")
    print(f"QA dataset saved to: {qa_path}")

    # Split into train and test datasets
    train_size = int(len(qa_df) * 0.8)
    qa_train_df = qa_df.iloc[:train_size]
    qa_test_df = qa_df.iloc[train_size:]

    # Save train and test datasets
    qa_train_path = os.path.join(final_docs_dir, "qa_train.parquet")
    qa_test_path = os.path.join(final_docs_dir, "qa_test.parquet")

    qa_train_df.to_parquet(qa_train_path)
    qa_test_df.to_parquet(qa_test_path)

    print(f"Train dataset saved to: {qa_train_path}")
    print(f"Test dataset saved to: {qa_test_path}")


if __name__ == "__main__":
    main()
