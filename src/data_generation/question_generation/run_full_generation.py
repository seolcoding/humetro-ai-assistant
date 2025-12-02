"""
골든 풀 180개 질문 생성 스크립트 (Idempotent 캐싱 지원)

DESIGN.md 기반 질문 유형 비율:
- Simple (40%): 72개
  - Simple Factoid: 36개 (6토픽 × 6개) - 20%
  - Constraint: 18개 (6토픽 × 3개) - 10%
  - Reasoning: 18개 (6토픽 × 3개) - 10%
- Advanced (60%): 108개
  - Multi-doc 1-hop: 36개 (6토픽 × 6개) - 20%
  - Multi-hop 2-hop: 36개 (6토픽 × 6개) - 20%
  - Multi-hop 3-hop: 18개 (6토픽 × 3개) - 10%
  - Multi-hop 5-hop: 18개 (6토픽 × 3개) - 10%

캐싱 기능:
- golden_pool_cache.jsonl에 생성된 QA를 저장
- 재실행 시 캐시에 있는 항목은 스킵 (idempotent)
- 캐시 키: topic + question_type + sorted(doc_ids)

사용법:
    python run_full_generation.py [--dry-run] [--topic TOPIC] [--scale N]

    --dry-run: API 호출 없이 선정 계획만 출력
    --topic TOPIC: 특정 토픽만 생성
    --scale N: 생성 수 스케일 팩터 (1.0 = 전체)
    --no-cache: 캐시 무시하고 재생성
    --retry-failed: 실패한 항목만 재시도
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

# 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from doc_selectors.multihop_selector import (
    MultihopDocumentSelector,
    DocumentPair,
    DocumentTriple,
    DocumentChain5,
    DocumentGroup
)
from generators.question_generator import QuestionGenerator, GeneratedQA, setup_logger, logger
from converters.autorag_converter import AutoRAGConverter
from prompts.system_prompt import SYSTEM_PROMPT, SYSTEM_PROMPT_MULTIHOP


# 캐시 관련 상수
CACHE_FILE = "golden_pool_cache.jsonl"
FAILED_FILE = "golden_pool_failed.jsonl"


def make_cache_key(topic: str, question_type: str, doc_ids: List[str]) -> str:
    """캐시 키 생성 - topic + question_type + sorted(doc_ids)"""
    sorted_ids = "_".join(sorted(doc_ids))
    return f"{topic}|{question_type}|{sorted_ids}"


class CacheManager:
    """Idempotent 캐시 관리자"""

    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = cache_dir / CACHE_FILE
        self.failed_file = cache_dir / FAILED_FILE
        self.cache: Dict[str, Dict] = {}
        self.failed: Dict[str, Dict] = {}
        self._load_cache()

    def _load_cache(self):
        """캐시 파일 로드"""
        # 성공 캐시 로드
        if self.cache_file.exists():
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            key = item.get("cache_key")
                            if key:
                                self.cache[key] = item
                        except json.JSONDecodeError:
                            continue
            logger.info(f"캐시 로드: {len(self.cache)}개 항목")

        # 실패 캐시 로드
        if self.failed_file.exists():
            with open(self.failed_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            item = json.loads(line)
                            key = item.get("cache_key")
                            if key:
                                self.failed[key] = item
                        except json.JSONDecodeError:
                            continue
            logger.info(f"실패 캐시 로드: {len(self.failed)}개 항목")

    def has(self, key: str) -> bool:
        """캐시에 있는지 확인"""
        return key in self.cache

    def has_failed(self, key: str) -> bool:
        """실패 캐시에 있는지 확인"""
        return key in self.failed

    def get(self, key: str) -> Optional[Dict]:
        """캐시에서 가져오기"""
        return self.cache.get(key)

    def add(self, key: str, qa_data: Dict):
        """캐시에 추가 (즉시 파일에 append)"""
        qa_data["cache_key"] = key
        qa_data["cached_at"] = datetime.now().isoformat()
        self.cache[key] = qa_data

        # 파일에 즉시 append
        with open(self.cache_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(qa_data, ensure_ascii=False) + "\n")

    def add_failed(self, key: str, fail_info: Dict):
        """실패 캐시에 추가"""
        fail_info["cache_key"] = key
        fail_info["failed_at"] = datetime.now().isoformat()
        self.failed[key] = fail_info

        with open(self.failed_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(fail_info, ensure_ascii=False) + "\n")

    def get_all_cached(self) -> List[Dict]:
        """모든 캐시된 항목 반환"""
        return list(self.cache.values())

    def clear_failed(self):
        """실패 캐시 초기화 (재시도용)"""
        self.failed = {}
        if self.failed_file.exists():
            self.failed_file.unlink()
        logger.info("실패 캐시 초기화")

    def get_stats(self) -> Dict[str, int]:
        """캐시 통계"""
        return {
            "cached": len(self.cache),
            "failed": len(self.failed)
        }


# 토픽당 질문 유형별 생성 수 (총 30개/토픽)
QUESTIONS_PER_TOPIC = {
    # Simple (40%) - 12개/토픽
    "simple_factoid": 6,   # 20%
    "constraint": 3,       # 10%
    "reasoning": 3,        # 10%
    # Advanced (60%) - 18개/토픽
    "multi_doc_1": 6,      # 20%
    "multi_hop_2": 6,      # 20%
    "multi_hop_3": 3,      # 10%
    "multi_hop_5": 3,      # 10%
}


class GoldenPoolGenerator:
    """골든 풀 180개 질문 생성기 (캐싱 지원)"""

    def __init__(
        self,
        golden_dataset_path: str,
        model: str = "gpt-5.1",
        reasoning_effort: str = "medium",
        output_dir: Optional[str] = None,
        use_cache: bool = True,
        retry_failed: bool = False
    ):
        """
        Args:
            golden_dataset_path: golden_dataset_v1.json 경로
            model: OpenAI 모델명
            reasoning_effort: GPT-5.1 추론 강도
            output_dir: 출력 디렉토리 (None이면 자동 생성)
            use_cache: 캐시 사용 여부 (True: idempotent)
            retry_failed: 실패한 항목만 재시도
        """
        # 문서 선정기 초기화
        self.selector = MultihopDocumentSelector(golden_dataset_path)

        # 질문 생성기 초기화
        self.generator = QuestionGenerator(
            model=model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=20000
        )

        # 출력 디렉토리 (고정 경로 사용 - 캐시 연속성)
        if output_dir:
            self.output_dir = Path(output_dir)
        else:
            # 캐시를 위해 고정된 경로 사용
            self.output_dir = Path(__file__).parent / "output" / "golden_pool"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 로그 디렉토리 설정
        log_dir = self.output_dir / "logs"
        setup_logger(str(log_dir))

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"=" * 60)
        logger.info(f"골든 풀 생성 시작 - {timestamp}")
        logger.info(f"출력 디렉토리: {self.output_dir}")
        logger.info(f"모델: {model}, 추론 강도: {reasoning_effort}")
        logger.info(f"캐시 사용: {use_cache}, 실패 재시도: {retry_failed}")
        logger.info(f"=" * 60)

        # 캐시 관리자 초기화
        self.use_cache = use_cache
        self.retry_failed = retry_failed
        self.cache = CacheManager(self.output_dir)

        if retry_failed:
            self.cache.clear_failed()
            logger.info("실패 항목 재시도 모드 - 실패 캐시 초기화됨")

        cache_stats = self.cache.get_stats()
        logger.info(f"캐시 상태: 성공 {cache_stats['cached']}개, 실패 {cache_stats['failed']}개")

        # 생성 결과 저장
        self.generated_qa: List[Dict] = []
        self.failed_generations: List[Dict] = []
        self.skipped_count: int = 0

        # 토픽 목록
        self.topics = list(set(t["topic"] for t in self.selector.targets))
        logger.info(f"토픽 목록: {self.topics}")
        logger.info(f"총 타겟 수: {len(self.selector.targets)}")

    def generate_simple_questions(
        self,
        topic: str,
        question_type: str,
        count: int
    ) -> List[GeneratedQA]:
        """
        Simple 질문 생성 (Simple Factoid, Constraint, Reasoning)

        Args:
            topic: 토픽
            question_type: simple_factoid | constraint | reasoning
            count: 생성할 질문 수

        Returns:
            GeneratedQA 리스트
        """
        logger.info(f"[{question_type.upper()}] 생성 중... (토픽: {topic}, 목표: {count}개)")

        # 타겟 문서 필터링
        targets = [t for t in self.selector.targets if t["topic"] == topic]
        if len(targets) < count:
            logger.warning(f"타겟 부족 ({len(targets)} < {count})")

        results = []
        skipped = 0
        generated = 0

        for i, target in enumerate(targets[:count]):
            doc_ids = [target["doc_id"]]
            cache_key = make_cache_key(topic, question_type, doc_ids)

            # 캐시 확인
            if self.use_cache and self.cache.has(cache_key):
                cached = self.cache.get(cache_key)
                logger.debug(f"  [{i+1}/{count}] 캐시 히트: {target['doc_id'][:30]}...")
                # 캐시된 데이터로 GeneratedQA 복원
                qa = GeneratedQA(
                    question=cached["question"],
                    answer=cached["answer"],
                    question_type=question_type,
                    evidence=cached.get("evidence"),
                    topic=topic,
                    target_doc_id=target["doc_id"],
                    retrieval_gt=doc_ids
                )
                results.append(qa)
                skipped += 1
                self.skipped_count += 1
                continue

            logger.info(f"  [{i+1}/{count}] {target['doc_id'][:30]}...")

            qa = self.generator.generate_single_doc_question(
                context=target["context"],
                question_type=question_type,
                doc_id=target["doc_id"],
                topic=topic
            )

            if qa:
                results.append(qa)
                generated += 1
                logger.info(f"    ✓ {qa.question[:50]}...")

                # 캐시에 저장
                if self.use_cache:
                    self.cache.add(cache_key, {
                        "question": qa.question,
                        "answer": qa.answer,
                        "evidence": qa.evidence,
                        "question_type": question_type,
                        "topic": topic,
                        "doc_ids": doc_ids
                    })
            else:
                fail_info = {
                    "type": question_type,
                    "topic": topic,
                    "doc_id": target["doc_id"]
                }
                self.failed_generations.append(fail_info)
                logger.warning(f"    ✗ 생성 실패")

                # 실패 캐시에 저장
                if self.use_cache:
                    self.cache.add_failed(cache_key, fail_info)

            # Rate limiting
            time.sleep(0.5)

        logger.info(f"  완료: {len(results)}/{count}개 (신규: {generated}, 캐시: {skipped})")
        return results

    def generate_multidoc_1_questions(
        self,
        topic: str,
        count: int
    ) -> List[GeneratedQA]:
        """
        Multi-doc 1-hop 질문 생성 (비교/종합)

        Args:
            topic: 토픽
            count: 생성할 질문 수

        Returns:
            GeneratedQA 리스트
        """
        logger.info(f"[MULTI_DOC_1] 생성 중... (토픽: {topic}, 목표: {count}개)")

        # 문서 그룹 선정
        groups = self.selector.select_multidoc_groups(
            topic=topic,
            count_per_target=1,
            group_size=3
        )

        if len(groups) < count:
            logger.warning(f"그룹 부족 ({len(groups)} < {count})")

        results = []
        skipped = 0
        generated = 0

        for i, group in enumerate(groups[:count]):
            doc_ids = group.doc_ids
            cache_key = make_cache_key(topic, "multi_doc_1", doc_ids)

            # 캐시 확인
            if self.use_cache and self.cache.has(cache_key):
                cached = self.cache.get(cache_key)
                logger.debug(f"  [{i+1}/{count}] 캐시 히트: {doc_ids[0][:30]}...")
                qa = GeneratedQA(
                    question=cached["question"],
                    answer=cached["answer"],
                    question_type="multi_doc_1",
                    evidence=cached.get("evidence"),
                    topic=topic,
                    retrieval_gt=doc_ids
                )
                results.append(qa)
                skipped += 1
                self.skipped_count += 1
                continue

            logger.info(f"  [{i+1}/{count}] {group.doc_ids[0][:30]}... ({group.group_type})")

            qa = self.generator.generate_multidoc_1_question(
                contexts=group.doc_contexts,
                doc_ids=group.doc_ids,
                topic=topic
            )

            if qa:
                results.append(qa)
                generated += 1
                logger.info(f"    ✓ {qa.question[:50]}...")

                if self.use_cache:
                    self.cache.add(cache_key, {
                        "question": qa.question,
                        "answer": qa.answer,
                        "evidence": qa.evidence,
                        "question_type": "multi_doc_1",
                        "topic": topic,
                        "doc_ids": doc_ids
                    })
            else:
                fail_info = {
                    "type": "multi_doc_1",
                    "topic": topic,
                    "doc_ids": group.doc_ids
                }
                self.failed_generations.append(fail_info)
                logger.warning(f"    ✗ 생성 실패")

                if self.use_cache:
                    self.cache.add_failed(cache_key, fail_info)

            time.sleep(0.5)

        logger.info(f"  완료: {len(results)}/{count}개 (신규: {generated}, 캐시: {skipped})")
        return results

    def generate_multihop_2_questions(
        self,
        topic: str,
        count: int
    ) -> List[GeneratedQA]:
        """
        Multi-hop 2-hop 질문 생성

        Args:
            topic: 토픽
            count: 생성할 질문 수

        Returns:
            GeneratedQA 리스트
        """
        logger.info(f"[MULTI_HOP_2] 생성 중... (토픽: {topic}, 목표: {count}개)")

        # 2-hop 문서 쌍 선정
        pairs = self.selector.select_2hop_pairs(topic=topic, count_per_target=1)

        if len(pairs) < count:
            logger.warning(f"쌍 부족 ({len(pairs)} < {count})")

        results = []
        skipped = 0
        generated = 0

        for i, pair in enumerate(pairs[:count]):
            doc_ids = [pair.target_doc_id, pair.related_doc_id]
            cache_key = make_cache_key(topic, "multi_hop_2", doc_ids)

            # 캐시 확인
            if self.use_cache and self.cache.has(cache_key):
                cached = self.cache.get(cache_key)
                logger.debug(f"  [{i+1}/{count}] 캐시 히트: {doc_ids[0][:30]}...")
                qa = GeneratedQA(
                    question=cached["question"],
                    answer=cached["answer"],
                    question_type="multi_hop_2",
                    evidence=cached.get("evidence"),
                    topic=topic,
                    retrieval_gt=doc_ids
                )
                results.append(qa)
                skipped += 1
                self.skipped_count += 1
                continue

            logger.info(f"  [{i+1}/{count}] {pair.target_doc_id[:30]}... (sim: {pair.similarity_score:.3f})")

            qa = self.generator.generate_multihop_2_question(
                context_1=pair.target_context,
                context_2=pair.related_context,
                doc_id_1=pair.target_doc_id,
                doc_id_2=pair.related_doc_id,
                topic=topic
            )

            if qa:
                results.append(qa)
                generated += 1
                logger.info(f"    ✓ {qa.question[:50]}...")

                if self.use_cache:
                    self.cache.add(cache_key, {
                        "question": qa.question,
                        "answer": qa.answer,
                        "evidence": qa.evidence,
                        "question_type": "multi_hop_2",
                        "topic": topic,
                        "doc_ids": doc_ids
                    })
            else:
                fail_info = {
                    "type": "multi_hop_2",
                    "topic": topic,
                    "doc_ids": doc_ids
                }
                self.failed_generations.append(fail_info)
                logger.warning(f"    ✗ 생성 실패")

                if self.use_cache:
                    self.cache.add_failed(cache_key, fail_info)

            time.sleep(0.5)

        logger.info(f"  완료: {len(results)}/{count}개 (신규: {generated}, 캐시: {skipped})")
        return results

    def generate_multihop_3_questions(
        self,
        topic: str,
        count: int
    ) -> List[GeneratedQA]:
        """
        Multi-hop 3-hop 질문 생성

        Args:
            topic: 토픽
            count: 생성할 질문 수

        Returns:
            GeneratedQA 리스트
        """
        logger.info(f"[MULTI_HOP_3] 생성 중... (토픽: {topic}, 목표: {count}개)")

        # 3-hop 문서 트리플 선정
        triples = self.selector.select_3hop_triples(topic=topic, count_per_target=1)

        if len(triples) < count:
            logger.warning(f"트리플 부족 ({len(triples)} < {count})")

        results = []
        skipped = 0
        generated = 0

        for i, triple in enumerate(triples[:count]):
            doc_ids = [triple.doc1_id, triple.doc2_id, triple.doc3_id]
            cache_key = make_cache_key(topic, "multi_hop_3", doc_ids)

            # 캐시 확인
            if self.use_cache and self.cache.has(cache_key):
                cached = self.cache.get(cache_key)
                logger.debug(f"  [{i+1}/{count}] 캐시 히트: {doc_ids[0][:30]}...")
                qa = GeneratedQA(
                    question=cached["question"],
                    answer=cached["answer"],
                    question_type="multi_hop_3",
                    evidence=cached.get("evidence"),
                    topic=topic,
                    retrieval_gt=doc_ids
                )
                results.append(qa)
                skipped += 1
                self.skipped_count += 1
                continue

            logger.info(f"  [{i+1}/{count}] {triple.doc1_id[:20]}→{triple.doc2_id[:20]}→{triple.doc3_id[:20]}")

            qa = self.generator.generate_multihop_3_question(
                context_1=triple.doc1_context,
                context_2=triple.doc2_context,
                context_3=triple.doc3_context,
                doc_id_1=triple.doc1_id,
                doc_id_2=triple.doc2_id,
                doc_id_3=triple.doc3_id,
                topic=topic
            )

            if qa:
                results.append(qa)
                generated += 1
                logger.info(f"    ✓ {qa.question[:50]}...")

                if self.use_cache:
                    self.cache.add(cache_key, {
                        "question": qa.question,
                        "answer": qa.answer,
                        "evidence": qa.evidence,
                        "question_type": "multi_hop_3",
                        "topic": topic,
                        "doc_ids": doc_ids
                    })
            else:
                fail_info = {
                    "type": "multi_hop_3",
                    "topic": topic,
                    "doc_ids": doc_ids
                }
                self.failed_generations.append(fail_info)
                logger.warning(f"    ✗ 생성 실패")

                if self.use_cache:
                    self.cache.add_failed(cache_key, fail_info)

            time.sleep(0.5)

        logger.info(f"  완료: {len(results)}/{count}개 (신규: {generated}, 캐시: {skipped})")
        return results

    def generate_multihop_5_questions(
        self,
        topic: str,
        count: int
    ) -> List[GeneratedQA]:
        """
        Multi-hop 5-hop 질문 생성 (선형/비선형)

        Args:
            topic: 토픽
            count: 생성할 질문 수

        Returns:
            GeneratedQA 리스트
        """
        logger.info(f"[MULTI_HOP_5] 생성 중... (토픽: {topic}, 목표: {count}개)")

        # 5-hop 문서 체인 선정
        chains = self.selector.select_5hop_chains(topic=topic, count_per_target=1)

        if len(chains) < count:
            logger.warning(f"체인 부족 ({len(chains)} < {count})")

        results = []
        skipped = 0
        generated = 0

        for i, chain in enumerate(chains[:count]):
            doc_ids = [chain.doc1_id, chain.doc2_id, chain.doc3_id,
                       chain.doc4_id, chain.doc5_id]
            cache_key = make_cache_key(topic, "multi_hop_5", doc_ids)

            # 캐시 확인
            if self.use_cache and self.cache.has(cache_key):
                cached = self.cache.get(cache_key)
                logger.debug(f"  [{i+1}/{count}] 캐시 히트: {doc_ids[0][:30]}...")
                qa = GeneratedQA(
                    question=cached["question"],
                    answer=cached["answer"],
                    question_type="multi_hop_5",
                    evidence=cached.get("evidence"),
                    topic=topic,
                    retrieval_gt=doc_ids
                )
                results.append(qa)
                skipped += 1
                self.skipped_count += 1
                continue

            logger.info(f"  [{i+1}/{count}] 5-hop chain (score: {chain.chain_score:.3f})")

            qa = self.generator.generate_multihop_5_question(
                context_1=chain.doc1_context,
                context_2=chain.doc2_context,
                context_3=chain.doc3_context,
                context_4=chain.doc4_context,
                context_5=chain.doc5_context,
                doc_id_1=chain.doc1_id,
                doc_id_2=chain.doc2_id,
                doc_id_3=chain.doc3_id,
                doc_id_4=chain.doc4_id,
                doc_id_5=chain.doc5_id,
                topic=topic
            )

            if qa:
                results.append(qa)
                generated += 1
                logger.info(f"    ✓ {qa.question[:50]}... (structure: {qa.structure_type})")

                if self.use_cache:
                    self.cache.add(cache_key, {
                        "question": qa.question,
                        "answer": qa.answer,
                        "evidence": qa.evidence,
                        "question_type": "multi_hop_5",
                        "topic": topic,
                        "doc_ids": doc_ids
                    })
            else:
                fail_info = {
                    "type": "multi_hop_5",
                    "topic": topic,
                    "doc_ids": doc_ids
                }
                self.failed_generations.append(fail_info)
                logger.warning(f"    ✗ 생성 실패")

                if self.use_cache:
                    self.cache.add_failed(cache_key, fail_info)

            time.sleep(0.5)

        logger.info(f"  완료: {len(results)}/{count}개 (신규: {generated}, 캐시: {skipped})")
        return results

    def generate_all(
        self,
        topics: Optional[List[str]] = None,
        scale_factor: float = 1.0
    ) -> List[GeneratedQA]:
        """
        전체 골든 풀 생성 (180개)

        Args:
            topics: 생성할 토픽 리스트 (None이면 전체)
            scale_factor: 생성 수 스케일 팩터 (테스트용, 0.5 = 절반)

        Returns:
            생성된 GeneratedQA 리스트
        """
        if topics is None:
            topics = self.topics

        logger.info("=" * 70)
        logger.info("골든 풀 질문 생성 시작")
        logger.info("=" * 70)
        logger.info(f"토픽: {topics}")
        logger.info(f"스케일 팩터: {scale_factor}")

        # 예상 생성 수
        expected = sum(QUESTIONS_PER_TOPIC.values()) * len(topics)
        expected_scaled = int(expected * scale_factor)
        cache_stats = self.cache.get_stats()
        logger.info(f"예상 생성 수: {expected_scaled}개 (원본: {expected}개)")
        logger.info(f"캐시 상태: 성공 {cache_stats['cached']}개, 실패 {cache_stats['failed']}개")
        logger.info("=" * 70)

        all_results = []
        start_time = time.time()

        for topic in topics:
            logger.info(f"\n{'='*30} {topic} {'='*30}")

            # Simple 질문
            for q_type in ["simple_factoid", "constraint", "reasoning"]:
                count = int(QUESTIONS_PER_TOPIC[q_type] * scale_factor)
                if count > 0:
                    results = self.generate_simple_questions(topic, q_type, count)
                    all_results.extend(results)

            # Multi-doc 1-hop
            count = int(QUESTIONS_PER_TOPIC["multi_doc_1"] * scale_factor)
            if count > 0:
                results = self.generate_multidoc_1_questions(topic, count)
                all_results.extend(results)

            # Multi-hop 2-hop
            count = int(QUESTIONS_PER_TOPIC["multi_hop_2"] * scale_factor)
            if count > 0:
                results = self.generate_multihop_2_questions(topic, count)
                all_results.extend(results)

            # Multi-hop 3-hop
            count = int(QUESTIONS_PER_TOPIC["multi_hop_3"] * scale_factor)
            if count > 0:
                results = self.generate_multihop_3_questions(topic, count)
                all_results.extend(results)

            # Multi-hop 5-hop
            count = int(QUESTIONS_PER_TOPIC["multi_hop_5"] * scale_factor)
            if count > 0:
                results = self.generate_multihop_5_questions(topic, count)
                all_results.extend(results)

        elapsed = time.time() - start_time
        api_stats = self.generator.get_stats()
        cache_stats = self.cache.get_stats()

        logger.info(f"\n{'='*70}")
        logger.info(f"생성 완료!")
        logger.info(f"{'='*70}")
        logger.info(f"총 결과: {len(all_results)}개")
        logger.info(f"  - 캐시에서 로드: {self.skipped_count}개")
        logger.info(f"  - 신규 생성: {api_stats['successful']}개")
        logger.info(f"  - 실패: {len(self.failed_generations)}개")
        logger.info(f"소요 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")
        logger.info(f"API 호출: {api_stats['total_calls']}회, 토큰: {api_stats['total_tokens']:,}개")
        logger.info(f"캐시 파일: {self.cache.cache_file}")

        return all_results

    def save_results(self, results: List[GeneratedQA]) -> Dict[str, str]:
        """
        결과 저장

        Args:
            results: GeneratedQA 리스트

        Returns:
            저장된 파일 경로 딕셔너리
        """
        print(f"\n=== 결과 저장: {self.output_dir} ===")

        paths = {}

        # 1. JSON 저장 (원본)
        json_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_questions": len(results),
                "model": self.generator.model,
                "api_stats": self.generator.get_stats(),
                "question_distribution": self._get_distribution(results)
            },
            "questions": [asdict(qa) for qa in results],
            "failed_generations": self.failed_generations
        }

        json_path = self.output_dir / "golden_pool.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        paths["json"] = str(json_path)
        print(f"  - JSON: {json_path}")

        # 2. AutoRAG 포맷 변환
        converter = AutoRAGConverter()
        autorag_qa = converter.convert_qa_list(
            [asdict(qa) for qa in results],
            id_prefix="golden"
        )

        qa_df = converter.to_dataframe(autorag_qa)
        qa_path = self.output_dir / "qa.parquet"
        qa_df.to_parquet(qa_path, index=False)
        paths["qa_parquet"] = str(qa_path)
        print(f"  - QA Parquet: {qa_path} ({len(qa_df)}개)")

        # 3. Corpus 생성 (retrieval_gt에 포함된 모든 문서)
        corpus_data = []
        seen_ids = set()

        # 모든 타겟 문서
        for target in self.selector.targets:
            if target["doc_id"] not in seen_ids:
                corpus_data.append({
                    "doc_id": target["doc_id"],
                    "contents": target["context"],
                    "metadata": {
                        "topic": target["topic"],
                        "doc_title": target.get("doc_title", "")
                    }
                })
                seen_ids.add(target["doc_id"])

            # 관련 문서
            for rd in target.get("related_documents", []):
                if rd["doc_id"] not in seen_ids:
                    corpus_data.append({
                        "doc_id": rd["doc_id"],
                        "contents": rd["context"],
                        "metadata": {
                            "doc_title": rd.get("doc_title", "")
                        }
                    })
                    seen_ids.add(rd["doc_id"])

        corpus_df = pd.DataFrame(corpus_data)
        corpus_path = self.output_dir / "corpus.parquet"
        corpus_df.to_parquet(corpus_path, index=False)
        paths["corpus_parquet"] = str(corpus_path)
        print(f"  - Corpus Parquet: {corpus_path} ({len(corpus_df)}개)")

        # 4. 요약 리포트
        report = self._generate_report(results)
        report_path = self.output_dir / "generation_report.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        paths["report"] = str(report_path)
        print(f"  - Report: {report_path}")

        return paths

    def _get_distribution(self, results: List[GeneratedQA]) -> Dict[str, int]:
        """질문 유형별 분포"""
        dist = {}
        for qa in results:
            q_type = qa.question_type
            dist[q_type] = dist.get(q_type, 0) + 1
        return dist

    def _generate_report(self, results: List[GeneratedQA]) -> str:
        """생성 리포트 마크다운"""
        dist = self._get_distribution(results)
        stats = self.generator.get_stats()

        # 토픽별 분포
        topic_dist = {}
        for qa in results:
            topic = qa.topic or "unknown"
            topic_dist[topic] = topic_dist.get(topic, 0) + 1

        report = f"""# 골든 풀 질문 생성 리포트

**생성 일시**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
**총 질문 수**: {len(results)}개
**실패 건수**: {len(self.failed_generations)}개

## 1. 질문 유형별 분포

| 유형 | 생성 수 | 비율 |
|------|---------|------|
"""
        for q_type, count in sorted(dist.items()):
            pct = count / len(results) * 100 if results else 0
            report += f"| {q_type} | {count}개 | {pct:.1f}% |\n"

        report += f"""
## 2. 토픽별 분포

| 토픽 | 생성 수 |
|------|---------|
"""
        for topic, count in sorted(topic_dist.items()):
            report += f"| {topic} | {count}개 |\n"

        report += f"""
## 3. API 통계

- 총 호출: {stats['total_calls']}회
- 성공: {stats['successful']}회
- 실패: {stats['failed']}회
- 총 토큰: {stats['total_tokens']:,}개

## 4. 질문 예시

### Simple Factoid
"""
        # 예시 추가
        for qa in results[:3]:
            if qa.question_type == "simple_factoid":
                report += f"- Q: {qa.question}\n- A: {qa.answer}\n\n"
                break

        report += "### Multi-hop 2-hop\n"
        for qa in results:
            if qa.question_type == "multi_hop_2":
                report += f"- Q: {qa.question}\n- A: {qa.answer}\n"
                if qa.reasoning_steps:
                    for step in qa.reasoning_steps[:2]:
                        report += f"  - {step}\n"
                report += "\n"
                break

        report += "### Multi-hop 5-hop\n"
        for qa in results:
            if qa.question_type == "multi_hop_5":
                report += f"- Q: {qa.question}\n- A: {qa.answer}\n"
                report += f"- Structure: {qa.structure_type}\n"
                if qa.reasoning_steps:
                    for step in qa.reasoning_steps[:3]:
                        report += f"  - {step}\n"
                report += "\n"
                break

        report += f"""
## 5. 실패 목록

총 {len(self.failed_generations)}건 실패

"""
        for fail in self.failed_generations[:10]:
            report += f"- {fail['type']}: {fail.get('doc_id', fail.get('doc_ids', 'N/A'))}\n"

        if len(self.failed_generations) > 10:
            report += f"\n... 외 {len(self.failed_generations) - 10}건\n"

        return report


def dry_run(golden_dataset_path: str, topics: Optional[List[str]] = None):
    """
    Dry run: API 호출 없이 선정 계획만 출력

    Args:
        golden_dataset_path: Golden Dataset 경로
        topics: 생성할 토픽 리스트
    """
    print("=" * 70)
    print("DRY RUN: 선정 계획 미리보기 (API 호출 없음)")
    print("=" * 70)

    selector = MultihopDocumentSelector(golden_dataset_path)

    if topics is None:
        topics = list(set(t["topic"] for t in selector.targets))

    print(f"토픽: {topics}")
    print(f"총 타겟: {len(selector.targets)}개")

    total_expected = 0
    for topic in topics:
        print(f"\n=== {topic} ===")

        # 타겟 수
        targets = [t for t in selector.targets if t["topic"] == topic]
        print(f"타겟 문서: {len(targets)}개")

        # 각 유형별 선정 가능 수
        pairs = selector.select_2hop_pairs(topic=topic, count_per_target=1)
        triples = selector.select_3hop_triples(topic=topic, count_per_target=1)
        chains = selector.select_5hop_chains(topic=topic, count_per_target=1)
        groups = selector.select_multidoc_groups(topic=topic, count_per_target=1)

        print(f"2-hop 쌍: {len(pairs)}개")
        print(f"3-hop 트리플: {len(triples)}개")
        print(f"5-hop 체인: {len(chains)}개")
        print(f"Multi-doc 그룹: {len(groups)}개")

        # 예상 생성 수
        expected = sum(QUESTIONS_PER_TOPIC.values())
        total_expected += expected
        print(f"예상 생성: {expected}개/토픽")

    print(f"\n{'='*70}")
    print(f"총 예상 생성 수: {total_expected}개")
    print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(
        description="골든 풀 180개 질문 생성 (Idempotent 캐싱 지원)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  # 전체 생성 (캐시 사용)
  uv run python run_full_generation.py --scale 1.0

  # 실패한 항목만 재시도
  uv run python run_full_generation.py --retry-failed

  # 캐시 무시하고 전체 재생성
  uv run python run_full_generation.py --no-cache

  # 특정 토픽만 생성
  uv run python run_full_generation.py --topic 공공행정

캐시 파일:
  - output/golden_pool/golden_pool_cache.jsonl  (성공)
  - output/golden_pool/golden_pool_failed.jsonl (실패)
"""
    )
    parser.add_argument("--dry-run", action="store_true", help="API 호출 없이 선정 계획만 출력")
    parser.add_argument("--topic", type=str, help="특정 토픽만 생성")
    parser.add_argument("--scale", type=float, default=1.0, help="생성 수 스케일 팩터 (0.5 = 절반)")
    parser.add_argument("--output-dir", type=str, help="출력 디렉토리")
    parser.add_argument("--model", type=str, default="gpt-5.1", help="OpenAI 모델명")
    parser.add_argument("--no-cache", action="store_true", help="캐시 무시하고 전체 재생성")
    parser.add_argument("--retry-failed", action="store_true", help="실패한 항목만 재시도")
    args = parser.parse_args()

    # Golden Dataset 경로
    base_path = Path(__file__).parent.parent / "golden_dataset" / "output"
    dataset_path = base_path / "golden_dataset_v1.json"

    if not dataset_path.exists():
        print(f"Error: Golden Dataset not found: {dataset_path}")
        print("먼저 golden_dataset 모듈을 실행하세요.")
        sys.exit(1)

    # 토픽 설정
    topics = [args.topic] if args.topic else None

    # Dry run
    if args.dry_run:
        dry_run(str(dataset_path), topics)
        return

    # 캐시 설정
    use_cache = not args.no_cache
    retry_failed = args.retry_failed

    print("=" * 70)
    print("골든 풀 질문 생성기 (Idempotent)")
    print("=" * 70)
    print(f"캐시 사용: {use_cache}")
    print(f"실패 재시도: {retry_failed}")
    print(f"출력 디렉토리: {args.output_dir or 'output/golden_pool'}")
    print("=" * 70)

    # 실제 생성
    generator = GoldenPoolGenerator(
        golden_dataset_path=str(dataset_path),
        model=args.model,
        output_dir=args.output_dir,
        use_cache=use_cache,
        retry_failed=retry_failed
    )

    # 질문 생성
    results = generator.generate_all(topics=topics, scale_factor=args.scale)

    # 결과 저장
    if results:
        paths = generator.save_results(results)
        print(f"\n=== 생성 완료 ===")
        print(f"출력 디렉토리: {generator.output_dir}")
        for name, path in paths.items():
            print(f"  - {name}: {path}")
        print(f"\n캐시 파일: {generator.cache.cache_file}")
        print(f"재실행하면 이미 생성된 {generator.skipped_count}개는 스킵됩니다.")
    else:
        print("\n생성된 질문이 없습니다.")


if __name__ == "__main__":
    main()
