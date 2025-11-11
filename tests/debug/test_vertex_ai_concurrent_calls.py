#!/usr/bin/env python3
"""
Test 2: Vertex AI 병렬 호출 테스트 (문제 재현)
목적: RAGAS 평가 시 발생하는 동시성 문제 재현
"""

import asyncio
import pytest
import logging
from concurrent.futures import ThreadPoolExecutor
from src.evaluation.vertex_ai_llm_wrapper import (
    VertexAILLMConfig,
    create_vertex_ai_evaluator_llm,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def single_call(llm, call_id: int) -> str:
    """단일 호출 수행"""
    from langchain_core.messages import HumanMessage

    chat_model = llm.langchain_llm
    message = HumanMessage(content=f"호출 {call_id}: 간단히 '확인 {call_id}'라고만 답변해주세요.")

    try:
        response = chat_model.invoke([message])
        logger.warning(f"✅ Call {call_id}: {response.content[:50]}")
        return response.content
    except Exception as e:
        logger.error(f"❌ Call {call_id} failed: {e}")
        return f"Error: {e}"


def test_concurrent_vertex_ai_calls():
    """병렬 Vertex AI 호출 - 문제 재현"""
    logger.warning("=" * 60)
    logger.warning("Test 2: 병렬 Vertex AI 호출 (5개 동시)")
    logger.warning("=" * 60)

    # Create config
    config = VertexAILLMConfig(
        model_name="gemini-2.5-flash",
        temperature=0.0,
    )

    # Create LLM
    llm = create_vertex_ai_evaluator_llm(config)

    # Concurrent calls
    num_calls = 5
    logger.warning(f"📤 Sending {num_calls} concurrent calls...")

    with ThreadPoolExecutor(max_workers=num_calls) as executor:
        futures = [
            executor.submit(single_call, llm, i)
            for i in range(num_calls)
        ]

        results = [f.result() for f in futures]

    # Check results
    empty_count = sum(1 for r in results if not r or "Error" in r)
    success_count = num_calls - empty_count

    logger.warning(f"\n📊 Results: {success_count}/{num_calls} successful")
    logger.warning(f"❌ Empty/Error responses: {empty_count}")

    # Log each result
    for i, result in enumerate(results):
        status = "✅" if result and "Error" not in result else "❌"
        logger.warning(f"{status} Call {i}: {result[:50] if result else 'Empty'}")

    assert success_count >= num_calls * 0.8, f"Too many failures: {success_count}/{num_calls}"


async def test_async_concurrent_vertex_ai_calls():
    """비동기 병렬 Vertex AI 호출 - 더 심한 문제 재현"""
    logger.warning("=" * 60)
    logger.warning("Test 2b: 비동기 병렬 Vertex AI 호출 (10개 동시)")
    logger.warning("=" * 60)

    # Create config
    config = VertexAILLMConfig(
        model_name="gemini-2.5-flash",
        temperature=0.0,
    )

    # Create LLM
    llm = create_vertex_ai_evaluator_llm(config)

    async def async_call(call_id: int) -> str:
        """비동기 호출"""
        from langchain_core.messages import HumanMessage

        chat_model = llm.langchain_llm
        message = HumanMessage(content=f"비동기 호출 {call_id}: '확인'만 답변")

        try:
            # Use ainvoke for async
            response = await chat_model.ainvoke([message])
            logger.warning(f"✅ Async call {call_id}: {response.content[:30]}")
            return response.content
        except Exception as e:
            logger.error(f"❌ Async call {call_id} failed: {e}")
            return f"Error: {e}"

    # Concurrent async calls
    num_calls = 10
    logger.warning(f"📤 Sending {num_calls} async concurrent calls...")

    tasks = [async_call(i) for i in range(num_calls)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Check results
    empty_count = sum(1 for r in results if not r or isinstance(r, Exception) or "Error" in str(r))
    success_count = num_calls - empty_count

    logger.warning(f"\n📊 Async Results: {success_count}/{num_calls} successful")
    logger.warning(f"❌ Empty/Error responses: {empty_count}")

    # Log failures
    for i, result in enumerate(results):
        if not result or isinstance(result, Exception) or "Error" in str(result):
            logger.error(f"❌ Async call {i} failed: {result}")

    assert success_count >= num_calls * 0.5, f"Too many async failures: {success_count}/{num_calls}"


if __name__ == "__main__":
    # Test sync concurrent calls
    print("\n=== 동기 병렬 호출 테스트 ===")
    test_concurrent_vertex_ai_calls()

    # Test async concurrent calls
    print("\n=== 비동기 병렬 호출 테스트 ===")
    asyncio.run(test_async_concurrent_vertex_ai_calls())
