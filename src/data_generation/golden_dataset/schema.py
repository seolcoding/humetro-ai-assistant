"""
Golden Dataset JSON Schema

타겟 문서와 관련 문서, QA를 통합 관리하는 자료구조
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from enum import Enum
import json


class QuestionType(str, Enum):
    """질문 유형"""
    SIMPLE_FACTOID = "simple_factoid"           # 단순 사실 질문 (4개/토픽)
    CONSTRAINT = "constraint"                    # 조건부 질문 (4개/토픽)
    MULTI_HOP_2 = "multi_hop_2"                 # 2-hop 질문 (6개/토픽)
    MULTI_HOP_3 = "multi_hop_3"                 # 3-hop 질문 (3개/토픽)
    REASONING = "reasoning"                      # 추론 질문 (3개/토픽)


@dataclass
class RelatedDocument:
    """유사 문서 정보"""
    doc_id: str
    doc_title: str
    context: str
    similarity_score: float  # TF-IDF cosine similarity
    context_length: int


@dataclass
class QAPair:
    """질문-답변 쌍"""
    question: str
    answer: str
    question_type: QuestionType
    retrieval_gt: List[str]  # 정답 문서 ID 리스트
    reasoning_steps: Optional[List[str]] = None  # Multi-hop의 경우 추론 단계


@dataclass
class TargetDocument:
    """타겟 문서 (QA 생성 대상)"""
    # 기본 정보
    doc_id: str
    doc_title: str
    doc_source: str
    topic: str
    context: str
    context_length: int

    # 유사 문서 리스트 (TF-IDF 기반)
    related_documents: List[RelatedDocument] = field(default_factory=list)

    # 생성된 QA 리스트
    qa_pairs: List[QAPair] = field(default_factory=list)

    # 메타데이터
    sample_type: str = "target"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "doc_id": self.doc_id,
            "doc_title": self.doc_title,
            "doc_source": self.doc_source,
            "topic": self.topic,
            "context": self.context,
            "context_length": self.context_length,
            "related_documents": [
                {
                    "doc_id": rd.doc_id,
                    "doc_title": rd.doc_title,
                    "context": rd.context,
                    "similarity_score": rd.similarity_score,
                    "context_length": rd.context_length
                }
                for rd in self.related_documents
            ],
            "qa_pairs": [
                {
                    "question": qa.question,
                    "answer": qa.answer,
                    "question_type": qa.question_type.value,
                    "retrieval_gt": qa.retrieval_gt,
                    "reasoning_steps": qa.reasoning_steps
                }
                for qa in self.qa_pairs
            ],
            "sample_type": self.sample_type
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TargetDocument":
        """Create from dictionary"""
        related_docs = [
            RelatedDocument(**rd) for rd in data.get("related_documents", [])
        ]
        qa_pairs = [
            QAPair(
                question=qa["question"],
                answer=qa["answer"],
                question_type=QuestionType(qa["question_type"]),
                retrieval_gt=qa["retrieval_gt"],
                reasoning_steps=qa.get("reasoning_steps")
            )
            for qa in data.get("qa_pairs", [])
        ]
        return cls(
            doc_id=data["doc_id"],
            doc_title=data["doc_title"],
            doc_source=data["doc_source"],
            topic=data["topic"],
            context=data["context"],
            context_length=data["context_length"],
            related_documents=related_docs,
            qa_pairs=qa_pairs,
            sample_type=data.get("sample_type", "target")
        )


@dataclass
class GoldenDataset:
    """Golden Dataset 전체 구조"""
    # 메타데이터
    version: str = "1.0.0"
    created_at: str = ""
    random_state: int = 42

    # 설정
    topics: List[str] = field(default_factory=list)
    question_distribution: Dict[str, int] = field(default_factory=dict)

    # 타겟 문서 리스트
    targets: List[TargetDocument] = field(default_factory=list)

    # 통계
    total_targets: int = 0
    total_qa_pairs: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "metadata": {
                "version": self.version,
                "created_at": self.created_at,
                "random_state": self.random_state
            },
            "config": {
                "topics": self.topics,
                "question_distribution": self.question_distribution
            },
            "statistics": {
                "total_targets": len(self.targets),
                "total_qa_pairs": sum(len(t.qa_pairs) for t in self.targets),
                "targets_per_topic": {
                    topic: len([t for t in self.targets if t.topic == topic])
                    for topic in self.topics
                }
            },
            "targets": [t.to_dict() for t in self.targets]
        }

    def save(self, path: str):
        """Save to JSON file"""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> "GoldenDataset":
        """Load from JSON file"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        dataset = cls(
            version=data["metadata"]["version"],
            created_at=data["metadata"]["created_at"],
            random_state=data["metadata"]["random_state"],
            topics=data["config"]["topics"],
            question_distribution=data["config"]["question_distribution"],
            targets=[TargetDocument.from_dict(t) for t in data["targets"]]
        )
        return dataset


# 질문 유형별 개수 (토픽당 20개)
QUESTION_DISTRIBUTION = {
    QuestionType.SIMPLE_FACTOID.value: 4,
    QuestionType.CONSTRAINT.value: 4,
    QuestionType.MULTI_HOP_2.value: 6,
    QuestionType.MULTI_HOP_3.value: 3,
    QuestionType.REASONING.value: 3,
}

# 타겟 토픽 리스트
TARGET_TOPICS = ["공공행정", "국토관리", "환경기상", "사회복지", "식품건강", "문화관광"]
