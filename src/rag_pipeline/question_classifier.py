#!/usr/bin/env python3
"""
Question Complexity Classifier
==============================

질문을 single-hop과 multi-hop으로 분류하는 모듈

Single-hop: 하나의 정보 조각으로 답변 가능한 질문
Multi-hop: 여러 정보를 조합/추론해야 답변 가능한 질문
"""

import re
import json
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class QuestionClassification:
    """Classification result for a single question"""
    question: str
    classification: str  # "single_hop" | "multi_hop"
    confidence: float
    reasoning_steps: int
    features: Dict[str, Any]
    classifier_reasoning: str


class QuestionComplexityClassifier:
    """
    질문을 single-hop과 multi-hop으로 분류

    분류 기준:
    1. 추론 단계 수 (reasoning steps)
    2. 필요한 컨텍스트 조각 수
    3. 복합/조건절 존재 여부
    4. 시간적/인과적 의존성
    """

    def __init__(self, method: str = "hybrid"):
        """
        Args:
            method: "llm", "rules", or "hybrid"
        """
        self.method = method
        self.llm = None

        if method in ["llm", "hybrid"]:
            try:
                from langchain_openai import ChatOpenAI
                self.llm = ChatOpenAI(
                    model="gpt-4o-mini",
                    temperature=0,
                    max_tokens=500
                )
            except Exception as e:
                logger.warning(f"LLM 초기화 실패, 규칙 기반으로 전환: {e}")
                self.method = "rules"

        # Multi-hop indicators for Korean text
        self.multi_hop_indicators = [
            # Conditional patterns
            r"만약|만일|경우|상황|가정",
            r"~면|~다면|~라면|~한다면",

            # Sequential/procedural patterns
            r"그리고|그 다음|이후|다음으로|순서",
            r"먼저.*다음|첫째.*둘째|단계",

            # Comparative patterns
            r"비교|차이|더 나은|보다|대비",
            r"장단점|vs|대|반면",

            # Causal/reasoning patterns
            r"왜|이유|원인|때문",
            r"결과|영향|따라서|그래서",

            # Alternative/choice patterns
            r"또는|혹은|아니면|대신|대안",
            r"선택|옵션|방법들|경로",

            # Complex query patterns
            r"어떻게.*어떤|무엇.*어떻게",
            r"모든|전체|각각|여러",

            # Specific to transportation domain
            r"환승|경유|우회|대체 경로",
            r"운행 중단.*대안|막차.*첫차",
            r"A에서 B.*거쳐|경유하여"
        ]

        # Single-hop indicators
        self.single_hop_indicators = [
            # Simple factual patterns
            r"^언제|^어디|^누가|^무엇",
            r"시간은|위치는|이름은|번호는",

            # Definition patterns
            r"정의|의미|뜻이 뭐",
            r"~이란|~란 무엇",

            # Direct lookup patterns
            r"몇 시|몇 분|몇 번|몇 개",
            r"있나요\?$|있습니까\?$",

            # Simple yes/no patterns
            r"~인가요\?$|~입니까\?$",
            r"가능한가요\?$|되나요\?$"
        ]

    def classify_batch(
        self,
        questions_df: pd.DataFrame,
        question_column: str = "user_input"
    ) -> List[QuestionClassification]:
        """
        Classify all questions in DataFrame

        Args:
            questions_df: DataFrame containing questions
            question_column: Column name containing questions

        Returns:
            List of QuestionClassification objects
        """
        classifications = []

        for idx, row in questions_df.iterrows():
            question = row[question_column]
            classification = self.classify_single(question)
            classifications.append(classification)

            if (idx + 1) % 10 == 0:
                logger.info(f"분류 진행: {idx + 1}/{len(questions_df)}")

        return classifications

    def classify_single(self, question: str) -> QuestionClassification:
        """
        Classify a single question

        Args:
            question: Question text to classify

        Returns:
            QuestionClassification object
        """
        if self.method == "rules":
            return self._classify_with_rules(question)
        elif self.method == "llm":
            return self._classify_with_llm(question)
        else:  # hybrid
            # First try rules for obvious cases
            rule_result = self._classify_with_rules(question)

            # If confidence is high, use rule result
            if rule_result.confidence >= 0.8:
                return rule_result

            # Otherwise, use LLM for borderline cases
            if self.llm:
                return self._classify_with_llm(question)
            else:
                return rule_result

    def _classify_with_rules(self, question: str) -> QuestionClassification:
        """
        Rule-based classification

        Uses pattern matching and heuristics
        """
        features = {
            "has_conditional": False,
            "has_sequential": False,
            "has_comparison": False,
            "has_causal": False,
            "has_alternative": False,
            "multi_hop_score": 0,
            "single_hop_score": 0,
            "question_length": len(question),
            "num_clauses": len(re.split(r'[,，、]', question))
        }

        # Check multi-hop indicators
        for pattern in self.multi_hop_indicators:
            if re.search(pattern, question, re.IGNORECASE):
                features["multi_hop_score"] += 1

                # Categorize the pattern type
                if "만약" in pattern or "경우" in pattern:
                    features["has_conditional"] = True
                elif "그리고" in pattern or "다음" in pattern:
                    features["has_sequential"] = True
                elif "비교" in pattern or "차이" in pattern:
                    features["has_comparison"] = True
                elif "왜" in pattern or "이유" in pattern:
                    features["has_causal"] = True
                elif "또는" in pattern or "대안" in pattern:
                    features["has_alternative"] = True

        # Check single-hop indicators
        for pattern in self.single_hop_indicators:
            if re.search(pattern, question, re.IGNORECASE):
                features["single_hop_score"] += 1

        # Decision logic
        multi_score = features["multi_hop_score"]
        single_score = features["single_hop_score"]

        # Strong multi-hop signal
        if multi_score >= 3 or (multi_score >= 2 and features["num_clauses"] > 2):
            classification = "multi_hop"
            confidence = min(0.9, 0.6 + multi_score * 0.1)
            reasoning_steps = 2 + (multi_score // 2)
            reasoning = f"규칙 기반: {multi_score}개 multi-hop 지표 감지"

        # Strong single-hop signal
        elif single_score >= 2 or (single_score >= 1 and multi_score == 0):
            classification = "single_hop"
            confidence = min(0.9, 0.6 + single_score * 0.15)
            reasoning_steps = 1
            reasoning = f"규칙 기반: {single_score}개 single-hop 지표 감지"

        # Borderline - use question length and complexity as tiebreaker
        else:
            if features["question_length"] > 100 or features["num_clauses"] > 2:
                classification = "multi_hop"
                confidence = 0.6
                reasoning_steps = 2
                reasoning = "규칙 기반: 질문 복잡도 기준 (길이/절 수)"
            else:
                classification = "single_hop"
                confidence = 0.6
                reasoning_steps = 1
                reasoning = "규칙 기반: 단순 질문 구조"

        return QuestionClassification(
            question=question,
            classification=classification,
            confidence=confidence,
            reasoning_steps=reasoning_steps,
            features=features,
            classifier_reasoning=reasoning
        )

    def _classify_with_llm(self, question: str) -> QuestionClassification:
        """
        LLM-based classification using GPT-4o-mini
        """
        if not self.llm:
            logger.warning("LLM not available, falling back to rules")
            return self._classify_with_rules(question)

        classification_prompt = f"""질문을 single-hop 또는 multi-hop으로 분류하세요.

**Single-hop**: 하나의 정보 조각이나 단순 검색으로 답변 가능한 질문
예시:
- "지하철 2호선 첫차 시간은?"
- "강남역은 몇 호선이 지나가나요?"
- "63빌딩 주소가 어떻게 되나요?"

**Multi-hop**: 여러 정보를 조합하거나 추론이 필요한 질문
예시:
- "강남역에서 홍대입구역까지 가는데, 2호선이 운행 중단이면 어떤 대체 경로가 있나요?"
- "평일 출근 시간에 여의도에서 판교까지 가장 빠른 대중교통 경로는?"
- "서울역에서 인천공항까지 막차를 놓치면 어떤 방법이 있나요?"

질문: {question}

다음 JSON 형식으로 정확히 답변하세요:
{{
    "classification": "single_hop" 또는 "multi_hop",
    "reasoning": "분류 이유를 한국어로 설명",
    "confidence": 0.0~1.0 사이의 신뢰도,
    "reasoning_steps": 필요한 추론 단계 수 (1, 2, 3 등),
    "features": {{
        "has_conditional": true/false,
        "has_comparison": true/false,
        "requires_multiple_lookups": true/false,
        "requires_inference": true/false
    }}
}}"""

        try:
            response = self.llm.invoke(classification_prompt)

            # Parse LLM response
            result_text = response.content if hasattr(response, 'content') else str(response)

            # Extract JSON from response
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in LLM response")

            return QuestionClassification(
                question=question,
                classification=result.get("classification", "single_hop"),
                confidence=float(result.get("confidence", 0.5)),
                reasoning_steps=int(result.get("reasoning_steps", 1)),
                features=result.get("features", {}),
                classifier_reasoning=f"LLM 기반: {result.get('reasoning', '')}"
            )

        except Exception as e:
            logger.error(f"LLM 분류 실패, 규칙 기반으로 전환: {e}")
            return self._classify_with_rules(question)

    def get_statistics(
        self,
        classifications: List[QuestionClassification]
    ) -> Dict[str, Any]:
        """
        Get statistics about classifications

        Args:
            classifications: List of classification results

        Returns:
            Statistics dictionary
        """
        if not classifications:
            return {}

        single_hop = [c for c in classifications if c.classification == "single_hop"]
        multi_hop = [c for c in classifications if c.classification == "multi_hop"]

        stats = {
            "total": len(classifications),
            "single_hop_count": len(single_hop),
            "multi_hop_count": len(multi_hop),
            "single_hop_percentage": (
                len(single_hop) / len(classifications) * 100
                if classifications else 0
            ),
            "multi_hop_percentage": (
                len(multi_hop) / len(classifications) * 100
                if classifications else 0
            ),
            "avg_confidence": (
                sum(c.confidence for c in classifications) / len(classifications)
                if classifications else 0
            ),
            "avg_confidence_single": (
                sum(c.confidence for c in single_hop) / len(single_hop)
                if single_hop else 0
            ),
            "avg_confidence_multi": (
                sum(c.confidence for c in multi_hop) / len(multi_hop)
                if multi_hop else 0
            ),
            "low_confidence_count": len([c for c in classifications if c.confidence < 0.7]),
            "classification_method": self.method
        }

        return stats


def test_classifier():
    """Test the classifier with sample questions"""

    test_questions = [
        # Single-hop examples
        "지하철 2호선 첫차 시간은?",
        "강남역 주소가 뭐야?",
        "63빌딩은 몇 층이야?",

        # Multi-hop examples
        "만약 2호선이 운행 중단이면 강남에서 홍대까지 어떻게 가야 해?",
        "평일 출근시간과 주말을 비교했을 때 여의도에서 판교 가는 시간 차이는?",
        "서울역에서 인천공항 막차를 놓쳤을 때 택시 비용과 공항버스 첫차 시간을 알려줘",
    ]

    classifier = QuestionComplexityClassifier(method="hybrid")

    print("="*70)
    print("Question Classifier Test")
    print("="*70)

    for question in test_questions:
        result = classifier.classify_single(question)
        print(f"\n질문: {question}")
        print(f"분류: {result.classification}")
        print(f"신뢰도: {result.confidence:.2f}")
        print(f"이유: {result.classifier_reasoning}")
        print("-"*50)


if __name__ == "__main__":
    # Run test when executed directly
    logging.basicConfig(level=logging.INFO)
    test_classifier()