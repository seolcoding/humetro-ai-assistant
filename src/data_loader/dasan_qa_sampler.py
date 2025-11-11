#!/usr/bin/env python3
"""
Dasan QA Sampler - 실제 콜센터 Q&A 데이터 샘플링
====================================================

AI_HUB_DASAN_QA JSONL 파일에서 질문-답변 쌍을 샘플링하여
RAGAS 평가 형식으로 변환합니다.

Features:
- 랜덤 샘플링 with reproducible seed
- 카테고리별 균등 샘플링 지원
- RAGAS Dataset 형식으로 자동 변환
- 재사용 가능한 모듈

Usage:
    from src.data_loader.dasan_qa_sampler import DasanQASampler

    sampler = DasanQASampler("data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs_full.jsonl")
    qa_samples = sampler.sample(n=25, seed=42)
    ragas_dataset = sampler.to_ragas_format(qa_samples)
"""

import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict

from datasets import Dataset

logger = logging.getLogger(__name__)


class DasanQASampler:
    """
    Dasan 콜센터 Q&A 데이터 샘플러

    JSONL에서 실제 Q&A 쌍을 샘플링하고 RAGAS 평가 형식으로 변환
    """

    def __init__(self, jsonl_path: Path):
        """
        Args:
            jsonl_path: JSONL 파일 경로 (knowledge_docs_full.jsonl)
        """
        self.jsonl_path = Path(jsonl_path)
        self.qa_data = []
        self.categories = defaultdict(list)

        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"JSONL file not found: {self.jsonl_path}")

        self._load_data()

    def _load_data(self):
        """JSONL 파일 로드 및 카테고리별 인덱싱"""
        logger.info(f"📂 Loading Q&A data from: {self.jsonl_path}")

        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)

                    # 필수 필드 확인
                    if not all(k in data for k in ["original_question", "original_answer", "document"]):
                        continue

                    # Q&A 데이터 저장
                    qa_item = {
                        "dialogue_id": data.get("dialogue_id", f"unknown_{line_num}"),
                        "question": data["original_question"],
                        "ground_truth": data["original_answer"],
                        "reference_document": data["document"],
                        "category": data.get("metadata", {}).get("category", "기타"),
                        "primary_topic": data.get("primary_topic", ""),
                        "metadata": data.get("metadata", {})
                    }

                    self.qa_data.append(qa_item)

                    # 카테고리별 인덱스
                    category = qa_item["category"]
                    self.categories[category].append(len(self.qa_data) - 1)

                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse line {line_num}")
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")

        logger.info(f"  ✅ Loaded {len(self.qa_data)} Q&A pairs")
        logger.info(f"  📊 Categories: {len(self.categories)}")
        for cat, items in sorted(self.categories.items(), key=lambda x: -len(x[1]))[:5]:
            logger.info(f"     - {cat}: {len(items)} items")

    def sample(
        self,
        n: int = 25,
        seed: int = 42,
        strategy: str = "random",
        min_length: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Q&A 샘플링

        Args:
            n: 샘플 개수
            seed: 랜덤 시드 (재현성)
            strategy: 샘플링 전략 ("random" | "balanced" | "stratified")
            min_length: 최소 답변 길이 (문자 수)

        Returns:
            샘플링된 Q&A 리스트
        """
        random.seed(seed)

        # 필터링: 최소 길이 조건
        filtered_qa = [
            qa for qa in self.qa_data
            if len(qa["ground_truth"]) >= min_length
        ]

        logger.info(f"🎲 Sampling {n} Q&A pairs (strategy={strategy}, seed={seed})")
        logger.info(f"  Filtered: {len(filtered_qa)}/{len(self.qa_data)} items (min_length={min_length})")

        if strategy == "random":
            samples = self._sample_random(filtered_qa, n)
        elif strategy == "balanced":
            samples = self._sample_balanced(filtered_qa, n)
        elif strategy == "stratified":
            samples = self._sample_stratified(filtered_qa, n)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        logger.info(f"  ✅ Sampled {len(samples)} items")

        # 샘플 통계
        sample_categories = defaultdict(int)
        for s in samples:
            sample_categories[s["category"]] += 1

        logger.info(f"  📊 Sample distribution:")
        for cat, count in sorted(sample_categories.items(), key=lambda x: -x[1])[:5]:
            logger.info(f"     - {cat}: {count} items")

        return samples

    def _sample_random(self, data: List[Dict], n: int) -> List[Dict]:
        """완전 랜덤 샘플링"""
        return random.sample(data, min(n, len(data)))

    def _sample_balanced(self, data: List[Dict], n: int) -> List[Dict]:
        """카테고리별 균등 샘플링"""
        # 카테고리별로 그룹화
        by_category = defaultdict(list)
        for item in data:
            by_category[item["category"]].append(item)

        # 카테고리당 할당량 계산
        n_categories = len(by_category)
        per_category = n // n_categories
        remainder = n % n_categories

        samples = []
        for i, (cat, items) in enumerate(by_category.items()):
            quota = per_category + (1 if i < remainder else 0)
            samples.extend(random.sample(items, min(quota, len(items))))

        return samples[:n]

    def _sample_stratified(self, data: List[Dict], n: int) -> List[Dict]:
        """계층적 샘플링 (카테고리 비율 유지)"""
        # 카테고리별 비율 계산
        by_category = defaultdict(list)
        for item in data:
            by_category[item["category"]].append(item)

        total = len(data)
        samples = []

        for cat, items in by_category.items():
            # 원본 비율 유지
            ratio = len(items) / total
            quota = max(1, int(n * ratio))  # 최소 1개
            samples.extend(random.sample(items, min(quota, len(items))))

        # 정확히 n개 맞추기
        if len(samples) > n:
            samples = random.sample(samples, n)
        elif len(samples) < n:
            # 부족하면 나머지 랜덤 추가
            remaining = [item for item in data if item not in samples]
            samples.extend(random.sample(remaining, n - len(samples)))

        return samples

    def to_ragas_format(
        self,
        qa_samples: List[Dict[str, Any]]
    ) -> Dataset:
        """
        RAGAS Dataset 형식으로 변환

        RAGAS 요구 형식:
        {
            "user_input": str,        # 질문
            "reference": str,         # Ground truth 답변
            "reference_contexts": [str],  # 참조 문서 리스트
        }

        Args:
            qa_samples: sample() 메서드로 샘플링된 Q&A 리스트

        Returns:
            RAGAS Dataset
        """
        logger.info(f"🔄 Converting {len(qa_samples)} samples to RAGAS format")

        ragas_data = {
            "user_input": [],
            "reference": [],
            "reference_contexts": []
        }

        for qa in qa_samples:
            ragas_data["user_input"].append(qa["question"])
            ragas_data["reference"].append(qa["ground_truth"])

            # reference_contexts는 리스트여야 함 (여러 문서 지원)
            ragas_data["reference_contexts"].append([qa["reference_document"]])

        dataset = Dataset.from_dict(ragas_data)

        logger.info(f"  ✅ RAGAS Dataset created: {len(dataset)} items")
        logger.info(f"  📋 Columns: {dataset.column_names}")

        return dataset

    def get_statistics(self) -> Dict[str, Any]:
        """데이터셋 통계 반환"""
        avg_q_len = sum(len(qa["question"]) for qa in self.qa_data) / len(self.qa_data)
        avg_a_len = sum(len(qa["ground_truth"]) for qa in self.qa_data) / len(self.qa_data)

        return {
            "total_qa_pairs": len(self.qa_data),
            "total_categories": len(self.categories),
            "avg_question_length": avg_q_len,
            "avg_answer_length": avg_a_len,
            "categories": {
                cat: len(items)
                for cat, items in self.categories.items()
            }
        }


def load_and_sample_qa(
    jsonl_path: Path,
    n: int = 25,
    seed: int = 42,
    strategy: str = "random"
) -> Dataset:
    """
    편의 함수: JSONL 로드 + 샘플링 + RAGAS 변환을 한번에

    Args:
        jsonl_path: JSONL 파일 경로
        n: 샘플 개수
        seed: 랜덤 시드
        strategy: 샘플링 전략

    Returns:
        RAGAS Dataset
    """
    sampler = DasanQASampler(jsonl_path)
    samples = sampler.sample(n=n, seed=seed, strategy=strategy)
    return sampler.to_ragas_format(samples)


if __name__ == "__main__":
    # Test sampling
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )

    jsonl_path = Path("data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs_full.jsonl")

    if not jsonl_path.exists():
        print(f"❌ File not found: {jsonl_path}")
        sys.exit(1)

    print("="*70)
    print(" "*20 + "Dasan QA Sampler Test")
    print("="*70)

    # Create sampler
    sampler = DasanQASampler(jsonl_path)

    # Print statistics
    stats = sampler.get_statistics()
    print(f"\n📊 Dataset Statistics:")
    print(f"  Total Q&A pairs: {stats['total_qa_pairs']}")
    print(f"  Total categories: {stats['total_categories']}")
    print(f"  Avg question length: {stats['avg_question_length']:.0f} chars")
    print(f"  Avg answer length: {stats['avg_answer_length']:.0f} chars")

    # Test random sampling
    print(f"\n🎲 Testing random sampling (n=10, seed=42)...")
    samples = sampler.sample(n=10, seed=42, strategy="random")

    print(f"\n📋 Sample Preview:")
    for i, s in enumerate(samples[:3], 1):
        print(f"\n[Sample {i}]")
        print(f"  ID: {s['dialogue_id']}")
        print(f"  Category: {s['category']}")
        print(f"  Question: {s['question'][:80]}...")
        print(f"  Answer: {s['ground_truth'][:80]}...")

    # Convert to RAGAS format
    print(f"\n🔄 Converting to RAGAS format...")
    ragas_dataset = sampler.to_ragas_format(samples)

    print(f"\n✅ RAGAS Dataset created!")
    print(f"  Shape: {ragas_dataset.shape}")
    print(f"  Columns: {ragas_dataset.column_names}")
    print(f"\n  First item:")
    print(f"    Question: {ragas_dataset[0]['user_input'][:80]}...")
    print(f"    Reference: {ragas_dataset[0]['reference'][:80]}...")
    print(f"    Contexts: {len(ragas_dataset[0]['reference_contexts'])} docs")
