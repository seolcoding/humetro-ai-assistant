#!/usr/bin/env python3
"""
Test script to reproduce retriever initialization bug
"""

import logging
from pathlib import Path

# Configure logging FIRST
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def test_wrong_initialization():
    """Test the WRONG way (current bug)"""
    logger.info("=" * 70)
    logger.info("TEST 1: Wrong Initialization (Current Bug)")
    logger.info("=" * 70)

    try:
        from src.rag_pipeline.stages.stage_05_vector_store import VectorStoreStage
        from src.rag_pipeline.stages.stage_06_retrieval import RetrievalStage

        vector_store_path = "data/vector_store/seoul_traffic"

        if Path(vector_store_path).exists():
            logger.info("기존 벡터 스토어 로드 중...")

            # Wrong way 1: Create VectorStoreStage but don't load
            vector_store = VectorStoreStage(
                model="text-embedding-3-small"
            )

            # Wrong way 2: Try to create RetrievalStage without retriever
            logger.info("RetrievalStage 생성 시도...")
            retriever = RetrievalStage(
                model="gpt-4o-mini",
                k=4  # This parameter doesn't exist!
            )
            logger.info("✅ 성공 (이 메시지는 나오면 안 됨!)")
        else:
            logger.warning(f"벡터 스토어를 찾을 수 없음: {vector_store_path}")

    except Exception as e:
        logger.error(f"❌ 예상된 실패: {type(e).__name__}: {e}")
        return False

    return True


def test_correct_initialization():
    """Test the CORRECT way (LangChain pattern)"""
    logger.info("\n" + "=" * 70)
    logger.info("TEST 2: Correct Initialization (LangChain Pattern)")
    logger.info("=" * 70)

    try:
        from src.rag_pipeline.stages.stage_05_vector_store import VectorStoreStage
        from src.rag_pipeline.stages.stage_06_retrieval import RetrievalStage

        vector_store_path = Path("data/vector_store/seoul_traffic")

        if not vector_store_path.exists():
            logger.error(f"Vector store not found: {vector_store_path}")
            return False

        # Step 1: Create and LOAD vector store
        logger.info("Step 1: Vector Store 생성 및 로드...")
        vector_store = VectorStoreStage(
            model="text-embedding-3-small"
        )
        vector_store.load_vector_store(vector_store_path)
        logger.info(f"  ✅ {vector_store.vector_store.index.ntotal:,} vectors loaded")

        # Step 2: Create LangChain retriever
        logger.info("Step 2: LangChain Retriever 생성...")
        langchain_retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        logger.info("  ✅ Retriever created")

        # Step 3: Create RetrievalStage with retriever
        logger.info("Step 3: RetrievalStage 초기화...")
        retrieval_stage = RetrievalStage(
            retriever=langchain_retriever,
            model="gpt-4o-mini",
            temperature=0.0
        )
        logger.info("  ✅ RetrievalStage initialized")

        # Step 4: Test retrieval
        logger.info("\nStep 4: Retrieval 테스트...")
        test_question = "서울시 대중교통 요금은 얼마인가요?"

        # Get documents with sources
        result = retrieval_stage.query_with_sources(test_question)

        logger.info(f"  Question: {result['question']}")
        logger.info(f"  Answer length: {len(result['answer'])} chars")
        logger.info(f"  Sources: {result['num_sources']} documents")

        logger.info("\n  Retrieved contexts:")
        for i, doc in enumerate(result['sources'][:2], 1):  # Show first 2
            logger.info(f"    [{i}] {doc.page_content[:100]}...")

        logger.info("\n✅ 모든 테스트 통과!")
        return True

    except Exception as e:
        logger.error(f"❌ 실패: {type(e).__name__}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


if __name__ == "__main__":
    # Test wrong way first
    wrong_success = test_wrong_initialization()

    # Test correct way
    correct_success = test_correct_initialization()

    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Wrong way (should fail): {'PASSED' if not wrong_success else 'FAILED (no exception!)'}")
    logger.info(f"Correct way (should succeed): {'PASSED' if correct_success else 'FAILED'}")
