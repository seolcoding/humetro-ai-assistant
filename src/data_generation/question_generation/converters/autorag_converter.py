"""
AutoRAG 포맷 변환기

생성된 QA를 AutoRAG의 qa.parquet 포맷으로 변환

AutoRAG QA 포맷:
- qid: 질문 ID (str)
- query: 질문 텍스트 (str)
- retrieval_gt: 정답 문서 ID 리스트 (list[str])
- generation_gt: 정답 텍스트 리스트 (list[str])
"""

import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import hashlib


@dataclass
class AutoRAGQA:
    """AutoRAG QA 포맷"""
    qid: str
    query: str
    retrieval_gt: List[str]
    generation_gt: List[str]


class AutoRAGConverter:
    """
    생성된 QA를 AutoRAG 포맷으로 변환

    변환 규칙:
    - qid: 질문 내용 기반 해시 생성
    - query: 질문 텍스트
    - retrieval_gt: 정답 문서 ID 리스트 (Multi-hop은 여러 개)
    - generation_gt: 답변 텍스트 리스트
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.converted_count = 0

    def convert_qa_list(
        self,
        qa_list: List[Dict[str, Any]],
        id_prefix: str = "golden"
    ) -> List[AutoRAGQA]:
        """
        QA 리스트를 AutoRAG 포맷으로 변환

        Args:
            qa_list: 생성된 QA 딕셔너리 리스트
            id_prefix: QID 접두사

        Returns:
            AutoRAGQA 리스트
        """
        converted = []

        for i, qa in enumerate(qa_list):
            # QID 생성 (해시 기반)
            question = qa.get("question", "")
            qid = self._generate_qid(question, id_prefix, i)

            # retrieval_gt 추출
            retrieval_gt = qa.get("retrieval_gt", [])
            if not retrieval_gt:
                # 단일 문서 질문의 경우
                if qa.get("target_doc_id"):
                    retrieval_gt = [qa["target_doc_id"]]

            # generation_gt 추출
            answer = qa.get("answer", "")
            generation_gt = [answer] if answer else []

            autorag_qa = AutoRAGQA(
                qid=qid,
                query=question,
                retrieval_gt=retrieval_gt,
                generation_gt=generation_gt
            )

            converted.append(autorag_qa)
            self.converted_count += 1

        return converted

    def _generate_qid(self, question: str, prefix: str, index: int) -> str:
        """
        QID 생성

        방식: prefix_hash[:8]_index
        예: golden_a1b2c3d4_001
        """
        # 질문 해시
        question_hash = hashlib.md5(question.encode()).hexdigest()[:8]
        return f"{prefix}_{question_hash}_{index:03d}"

    def to_dataframe(self, qa_list: List[AutoRAGQA]) -> pd.DataFrame:
        """
        AutoRAGQA 리스트를 DataFrame으로 변환

        Args:
            qa_list: AutoRAGQA 리스트

        Returns:
            AutoRAG 호환 DataFrame
        """
        data = []
        for qa in qa_list:
            data.append({
                "qid": qa.qid,
                "query": qa.query,
                "retrieval_gt": qa.retrieval_gt,
                "generation_gt": qa.generation_gt
            })

        df = pd.DataFrame(data)
        return df

    def save_parquet(
        self,
        qa_list: List[AutoRAGQA],
        output_path: str
    ) -> str:
        """
        Parquet 파일로 저장

        Args:
            qa_list: AutoRAGQA 리스트
            output_path: 출력 경로

        Returns:
            저장된 파일 경로
        """
        df = self.to_dataframe(qa_list)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(output_path, index=False)

        print(f"Saved {len(df)} QAs to {output_path}")
        return output_path

    def save_json(
        self,
        qa_list: List[AutoRAGQA],
        output_path: str,
        include_metadata: bool = True
    ) -> str:
        """
        JSON 파일로 저장 (검토용)

        Args:
            qa_list: AutoRAGQA 리스트
            output_path: 출력 경로
            include_metadata: 메타데이터 포함 여부

        Returns:
            저장된 파일 경로
        """
        data = {
            "metadata": {
                "total_count": len(qa_list),
                "format": "AutoRAG QA",
                "random_state": self.random_state
            },
            "qa_pairs": [
                {
                    "qid": qa.qid,
                    "query": qa.query,
                    "retrieval_gt": qa.retrieval_gt,
                    "generation_gt": qa.generation_gt
                }
                for qa in qa_list
            ]
        }

        if not include_metadata:
            data = data["qa_pairs"]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"Saved {len(qa_list)} QAs to {output_path}")
        return output_path


class GoldenPoolToAutoRAG:
    """
    골든 풀에서 AutoRAG 데이터셋 생성

    전체 파이프라인:
    1. 골든 풀 JSON 로드
    2. AutoRAG 포맷 변환
    3. Parquet + JSON 저장
    """

    def __init__(self, converter: Optional[AutoRAGConverter] = None):
        self.converter = converter or AutoRAGConverter()

    def convert_golden_pool(
        self,
        golden_pool_path: str,
        output_dir: str
    ) -> Dict[str, str]:
        """
        골든 풀을 AutoRAG 포맷으로 변환

        Args:
            golden_pool_path: golden_pool_180.json 경로
            output_dir: 출력 디렉토리

        Returns:
            출력 파일 경로 딕셔너리
        """
        # 골든 풀 로드
        with open(golden_pool_path, 'r', encoding='utf-8') as f:
            pool = json.load(f)

        qa_list = pool.get("qa_pairs", pool)  # 구조에 따라

        # 변환
        autorag_qa = self.converter.convert_qa_list(qa_list, id_prefix="golden")

        # 저장
        output_dir = Path(output_dir)
        parquet_path = str(output_dir / "qa.parquet")
        json_path = str(output_dir / "qa_golden.json")

        self.converter.save_parquet(autorag_qa, parquet_path)
        self.converter.save_json(autorag_qa, json_path)

        return {
            "parquet": parquet_path,
            "json": json_path,
            "count": len(autorag_qa)
        }


if __name__ == "__main__":
    # 테스트
    print("=== AutoRAGConverter 테스트 ===\n")

    # 샘플 QA 데이터
    sample_qa = [
        {
            "question": "2024년 강남구 청소년 예산은 얼마인가?",
            "answer": "52억원",
            "question_type": "simple_factoid",
            "retrieval_gt": ["doc_001"],
            "target_doc_id": "doc_001",
            "topic": "공공행정"
        },
        {
            "question": "강남구청 환경과장이 2024년에 수상한 상은?",
            "answer": "우수공무원상",
            "question_type": "multi_hop_2",
            "retrieval_gt": ["doc_002", "doc_003"],
            "reasoning_steps": ["환경과장 확인", "수상 내역 확인"]
        },
        {
            "question": "2024년 환경정책 담당부서장의 최종학력은?",
            "answer": "환경공학 박사",
            "question_type": "multi_hop_3",
            "retrieval_gt": ["doc_004", "doc_005", "doc_006"]
        }
    ]

    # 변환
    converter = AutoRAGConverter()
    autorag_qa = converter.convert_qa_list(sample_qa, id_prefix="test")

    print("=== 변환 결과 ===")
    for qa in autorag_qa:
        print(f"QID: {qa.qid}")
        print(f"  Query: {qa.query}")
        print(f"  Retrieval GT: {qa.retrieval_gt}")
        print(f"  Generation GT: {qa.generation_gt}")
        print()

    # DataFrame 확인
    df = converter.to_dataframe(autorag_qa)
    print("=== DataFrame ===")
    print(df)
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"retrieval_gt type: {type(df.iloc[0]['retrieval_gt'])}")
