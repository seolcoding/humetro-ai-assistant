#!/usr/bin/env python3
"""
Gemini Batch API Input Generator
=================================

질문 데이터셋을 Gemini Batch API 입력 형식(JSONL)으로 변환

Input Format:
    - CSV/JSON testset with questions
    - RAG context (optional)

Output Format:
    - JSONL file with Gemini Batch API request format
    - Each line: {"request": {"contents": [...], "generationConfig": {...}}}

Usage:
    # Generate from existing testset (50 random questions)
    python create_batch_input.py --testset data/evaluation/testsets/golden_testset_50.csv --output batch_input.jsonl --sample 50

    # Generate with RAG context
    python create_batch_input.py --testset testset.csv --with-rag --k-documents 4
"""

import sys
import json
import random
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.rag_pipeline.stages.stage_05_vector_store import VectorStoreStage

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class GeminiBatchInputGenerator:
    """Gemini Batch API 입력 JSONL 생성기"""

    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        temperature: float = 0.7,
        max_output_tokens: int = 2048,
        retriever: Optional[Any] = None
    ):
        """
        Args:
            model: Gemini model name (e.g., "gemini-2.5-pro", "gemini-2.5-flash")
            temperature: Generation temperature (0.0-1.0)
            max_output_tokens: Maximum tokens per response
            retriever: Optional LangChain retriever for RAG context
        """
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.retriever = retriever

    def create_system_prompt(self) -> str:
        """시스템 프롬프트 생성"""
        return """당신은 서울 대중교통 정보를 제공하는 전문 AI 어시스턴트입니다.

역할:
- 서울시 지하철, 버스, 택시 등 대중교통 관련 질문에 답변
- 정확하고 간결한 한국어 응답 제공
- 제공된 컨텍스트 기반 답변 (컨텍스트 없으면 "정보 없음" 표시)

응답 원칙:
1. 사실 기반 정확한 정보만 제공
2. 불확실한 경우 "확인 필요" 명시
3. 간결하고 이해하기 쉬운 한국어 사용
4. 관련 없는 정보는 생략"""

    def create_user_prompt(self, question: str, contexts: Optional[List[str]] = None) -> str:
        """사용자 프롬프트 생성 (RAG 형식)"""
        if contexts and len(contexts) > 0:
            context_str = "\n\n".join([f"[문서 {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)])
            return f"""다음 컨텍스트를 참고하여 질문에 답변해주세요.

컨텍스트:
{context_str}

질문: {question}

답변:"""
        else:
            return f"""질문: {question}

답변:"""

    def retrieve_contexts(self, question: str, k: int = 4) -> List[str]:
        """RAG 컨텍스트 검색"""
        if not self.retriever:
            return []

        try:
            documents = self.retriever.invoke(question)
            contexts = [doc.page_content for doc in documents[:k]]
            logger.debug(f"Retrieved {len(contexts)} contexts for: {question[:50]}...")
            return contexts
        except Exception as e:
            logger.warning(f"⚠️ Context retrieval failed: {e}")
            return []

    def create_request(
        self,
        question: str,
        use_rag: bool = False,
        k_documents: int = 4
    ) -> Dict[str, Any]:
        """
        단일 질문에 대한 Gemini Batch API 요청 생성

        Args:
            question: 질문 텍스트
            use_rag: RAG 컨텍스트 사용 여부
            k_documents: 검색할 문서 수

        Returns:
            Gemini Batch API request format
        """
        # Get RAG contexts if needed
        contexts = []
        if use_rag and self.retriever:
            contexts = self.retrieve_contexts(question, k=k_documents)

        # Create prompts
        system_prompt = self.create_system_prompt()
        user_prompt = self.create_user_prompt(question, contexts)

        # Gemini Batch API request format
        request = {
            "request": {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]
                    }
                ],
                "generationConfig": {
                    "temperature": self.temperature,
                    "maxOutputTokens": self.max_output_tokens
                }
            }
        }

        return request

    def generate_from_testset(
        self,
        testset_path: Path,
        output_path: Path,
        sample_size: Optional[int] = None,
        use_rag: bool = False,
        k_documents: int = 4,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """
        테스트셋에서 Batch API 입력 JSONL 생성

        Args:
            testset_path: 입력 CSV/JSON testset 경로
            output_path: 출력 JSONL 파일 경로
            sample_size: 샘플링할 질문 수 (None = 전체)
            use_rag: RAG 컨텍스트 사용
            k_documents: 검색 문서 수
            random_seed: 랜덤 시드

        Returns:
            통계 정보
        """
        logger.info("="*70)
        logger.info("Gemini Batch API 입력 생성")
        logger.info("="*70)

        # Load testset
        if testset_path.suffix == '.csv':
            df = pd.read_csv(testset_path)
        elif testset_path.suffix == '.json':
            df = pd.read_json(testset_path)
        else:
            raise ValueError(f"Unsupported format: {testset_path.suffix}")

        logger.info(f"📂 Testset loaded: {len(df)} questions")

        # Sample if needed
        if sample_size and sample_size < len(df):
            random.seed(random_seed)
            df = df.sample(n=sample_size, random_state=random_seed)
            logger.info(f"🎲 Sampled: {sample_size} questions")

        # Extract questions
        question_col = None
        for col in ['question', 'user_input', 'query']:
            if col in df.columns:
                question_col = col
                break

        if not question_col:
            raise ValueError(f"No question column found. Available: {df.columns.tolist()}")

        questions = df[question_col].tolist()

        # Generate requests
        logger.info(f"🔄 Generating {len(questions)} requests...")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for i, question in enumerate(questions, 1):
                try:
                    request = self.create_request(
                        question=question,
                        use_rag=use_rag,
                        k_documents=k_documents
                    )

                    # Write as single-line JSON
                    f.write(json.dumps(request, ensure_ascii=False) + '\n')

                    if i % 10 == 0:
                        logger.info(f"  Progress: {i}/{len(questions)}")

                except Exception as e:
                    logger.error(f"❌ Failed to generate request {i}: {e}")
                    continue

        # Statistics
        stats = {
            "total_questions": len(questions),
            "output_file": str(output_path),
            "model": self.model,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "use_rag": use_rag,
            "k_documents": k_documents if use_rag else 0,
            "sample_size": sample_size,
            "random_seed": random_seed
        }

        logger.info("="*70)
        logger.info("✅ Batch input generation complete")
        logger.info(f"📊 Total requests: {stats['total_questions']}")
        logger.info(f"📁 Output: {stats['output_file']}")
        logger.info(f"🤖 Model: {stats['model']}")
        logger.info(f"🌡️  Temperature: {stats['temperature']}")
        logger.info(f"📄 Max tokens: {stats['max_output_tokens']}")
        logger.info(f"🔍 RAG enabled: {stats['use_rag']}")
        if use_rag:
            logger.info(f"📚 Documents per query: {stats['k_documents']}")
        logger.info("="*70)

        # Save metadata
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info(f"💾 Metadata saved: {metadata_path}")

        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate Gemini Batch API input JSONL from testset",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input/Output
    parser.add_argument(
        "--testset",
        type=Path,
        required=True,
        help="Input testset path (CSV or JSON)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/batch_processing/batch_input.jsonl"),
        help="Output JSONL path (default: data/batch_processing/batch_input.jsonl)"
    )

    # Sampling
    parser.add_argument(
        "--sample",
        type=int,
        help="Sample N questions randomly (default: use all)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling (default: 42)"
    )

    # Model config
    parser.add_argument(
        "--model",
        default="gemini-2.5-pro",
        help="Gemini model name (default: gemini-2.5-pro)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature (default: 0.7)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Max output tokens (default: 2048)"
    )

    # RAG options
    parser.add_argument(
        "--with-rag",
        action="store_true",
        help="Enable RAG context retrieval"
    )
    parser.add_argument(
        "--k-documents",
        type=int,
        default=4,
        help="Number of documents to retrieve (default: 4)"
    )
    parser.add_argument(
        "--vector-store",
        type=Path,
        default=Path("data/vector_store/seoul_traffic"),
        help="Vector store path (default: data/vector_store/seoul_traffic)"
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-large",
        help="Embedding model (default: text-embedding-3-large)"
    )

    args = parser.parse_args()

    # Initialize retriever if RAG enabled
    retriever = None
    if args.with_rag:
        if not args.vector_store.exists():
            logger.error(f"❌ Vector store not found: {args.vector_store}")
            sys.exit(1)

        logger.info("📚 Loading vector store for RAG...")
        vector_store_stage = VectorStoreStage(model=args.embedding_model)
        vector_store_stage.load_vector_store(args.vector_store)
        retriever = vector_store_stage.as_retriever(
            search_type="similarity",
            search_kwargs={"k": args.k_documents}
        )
        logger.info("  ✅ Vector store loaded")

    # Generate batch input
    generator = GeminiBatchInputGenerator(
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_tokens,
        retriever=retriever
    )

    stats = generator.generate_from_testset(
        testset_path=args.testset,
        output_path=args.output,
        sample_size=args.sample,
        use_rag=args.with_rag,
        k_documents=args.k_documents,
        random_seed=args.seed
    )

    print("\n✅ Success!")
    print(f"Generated {stats['total_questions']} requests")
    print(f"Output: {stats['output_file']}")


if __name__ == "__main__":
    main()
