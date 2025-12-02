"""
Question Generation Module

Golden Dataset 질문 생성을 위한 모듈

질문 유형:
- Simple Factoid: 단순 사실 질문 (4개/토픽)
- Constraint: 조건부 질문 (4개/토픽)
- Multi-hop (2-hop): 2개 문서 연결 (6개/토픽)
- Multi-hop (3-hop): 3개 문서 연결 (3개/토픽)
- Reasoning: 추론/인과 질문 (3개/토픽)
"""

from enum import Enum


class QuestionType(str, Enum):
    """질문 유형 (schema.py와 동일)"""
    SIMPLE_FACTOID = "simple_factoid"
    CONSTRAINT = "constraint"
    MULTI_HOP_2 = "multi_hop_2"
    MULTI_HOP_3 = "multi_hop_3"
    REASONING = "reasoning"


# 골든 풀 배분 (토픽당, 1.5배)
GOLDEN_POOL_DISTRIBUTION = {
    QuestionType.SIMPLE_FACTOID: 6,
    QuestionType.CONSTRAINT: 6,
    QuestionType.MULTI_HOP_2: 9,
    QuestionType.MULTI_HOP_3: 5,
    QuestionType.REASONING: 4,
}

# 골든 배분 (토픽당)
GOLDEN_DISTRIBUTION = {
    QuestionType.SIMPLE_FACTOID: 4,
    QuestionType.CONSTRAINT: 4,
    QuestionType.MULTI_HOP_2: 6,
    QuestionType.MULTI_HOP_3: 3,
    QuestionType.REASONING: 3,
}

# 실버 배분 (토픽당)
SILVER_DISTRIBUTION = {
    QuestionType.SIMPLE_FACTOID: 2,
    QuestionType.CONSTRAINT: 2,
    QuestionType.MULTI_HOP_2: 2,
    QuestionType.MULTI_HOP_3: 2,
    QuestionType.REASONING: 2,
}

# 브론즈 배분 (토픽당)
BRONZE_DISTRIBUTION = {
    QuestionType.SIMPLE_FACTOID: 1,
    QuestionType.CONSTRAINT: 1,
    QuestionType.MULTI_HOP_2: 1,
    QuestionType.MULTI_HOP_3: 1,
    QuestionType.REASONING: 1,
}
