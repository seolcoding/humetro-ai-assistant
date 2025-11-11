#!/usr/bin/env python3
"""
Test 1: Vertex AI 단일 호출 테스트 (동시성 없음)
목적: Vertex AI 연결과 기본 호출이 정상 작동하는지 확인
"""

import pytest
import logging
from src.evaluation.vertex_ai_llm_wrapper import (
    VertexAILLMConfig,
    create_vertex_ai_evaluator_llm,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_single_vertex_ai_call():
    """단일 Vertex AI 호출 - 동시성 없음"""
    logger.info("=" * 60)
    logger.info("Test 1: 단일 Vertex AI 호출")
    logger.info("=" * 60)

    # Create config
    config = VertexAILLMConfig(
        model_name="gemini-2.5-flash",
        temperature=0.0,
    )

    # Create LLM
    llm = create_vertex_ai_evaluator_llm(config)

    # Single call using underlying LangChain model
    logger.info("📤 Sending single message...")

    # Access underlying LangChain model
    from langchain_core.messages import HumanMessage

    chat_model = llm.langchain_llm
    message = HumanMessage(content="테스트 메시지입니다. 간단히 '확인'이라고만 답변해주세요.")

    response = chat_model.invoke([message])

    logger.info(f"📥 Response: {response.content[:100]}")

    assert response is not None
    assert len(response.content) > 0

    logger.info("✅ 단일 호출 성공")


if __name__ == "__main__":
    test_single_vertex_ai_call()
