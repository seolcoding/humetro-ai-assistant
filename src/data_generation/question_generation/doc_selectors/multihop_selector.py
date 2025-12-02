"""
Multi-hop 문서 선정 로직

DESIGN.md 기반:
- 2-hop: 유사도 0.2 ~ 0.5 범위
- 3-hop: 체인 구조 (A → B → C) 형성 가능

근거:
- 유사도 > 0.5: 거의 동일 내용, hop 불필요
- 유사도 < 0.2: 무관한 문서, 부자연스러운 연결
"""

import json
import random
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path


@dataclass
class DocumentPair:
    """2-hop 문서 쌍"""
    target_doc_id: str
    target_context: str
    related_doc_id: str
    related_context: str
    similarity_score: float
    topic: str


@dataclass
class DocumentTriple:
    """3-hop 문서 트리플"""
    doc1_id: str
    doc1_context: str
    doc2_id: str
    doc2_context: str
    doc3_id: str
    doc3_context: str
    similarity_1_2: float
    similarity_2_3: float
    topic: str


@dataclass
class DocumentChain5:
    """5-hop 문서 체인"""
    doc1_id: str
    doc1_context: str
    doc2_id: str
    doc2_context: str
    doc3_id: str
    doc3_context: str
    doc4_id: str
    doc4_context: str
    doc5_id: str
    doc5_context: str
    chain_score: float  # 체인 품질 점수
    topic: str


@dataclass
class DocumentGroup:
    """Multi-doc 1-hop용 문서 그룹 (비교/종합용)"""
    doc_ids: List[str]
    doc_contexts: List[str]
    group_type: str  # "comparison" | "aggregation" | "trend"
    topic: str


class MultihopDocumentSelector:
    """
    Multi-hop 질문 생성을 위한 문서 선정기

    선정 기준:
    - 2-hop: 유사도 0.2 ~ 0.5 (적절한 관련성)
    - 3-hop: 체인 구조 형성 가능 (중간 문서 필수)
    """

    def __init__(
        self,
        golden_dataset_path: str,
        min_similarity_2hop: float = 0.2,
        max_similarity_2hop: float = 0.5,
        min_similarity_3hop: float = 0.15,
        max_similarity_3hop: float = 0.45,
        random_state: int = 42
    ):
        """
        Args:
            golden_dataset_path: golden_dataset_v1.json 경로
            min_similarity_2hop: 2-hop 최소 유사도
            max_similarity_2hop: 2-hop 최대 유사도
            min_similarity_3hop: 3-hop 유사도 범위 (체인용)
            max_similarity_3hop: 3-hop 최대 유사도
            random_state: 재현성을 위한 시드
        """
        self.min_sim_2hop = min_similarity_2hop
        self.max_sim_2hop = max_similarity_2hop
        self.min_sim_3hop = min_similarity_3hop
        self.max_sim_3hop = max_similarity_3hop

        random.seed(random_state)

        # 데이터 로드
        self.dataset = self._load_dataset(golden_dataset_path)
        self.targets = self.dataset.get("targets", [])

        print(f"Loaded {len(self.targets)} targets from {golden_dataset_path}")

    def _load_dataset(self, path: str) -> Dict:
        """Golden Dataset 로드"""
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def select_2hop_pairs(
        self,
        topic: Optional[str] = None,
        count_per_target: int = 1
    ) -> List[DocumentPair]:
        """
        2-hop 문서 쌍 선정

        선정 기준 (DESIGN.md):
        - 유사도 0.2 ~ 0.5 범위
        - 공통 엔티티 가능성 높은 문서

        Args:
            topic: 특정 토픽만 선정 (None이면 전체)
            count_per_target: 타겟당 선정할 쌍 수

        Returns:
            DocumentPair 리스트
        """
        pairs = []

        for target in self.targets:
            # 토픽 필터링
            if topic and target["topic"] != topic:
                continue

            # 유사도 범위 내 문서 필터링
            candidates = [
                rd for rd in target.get("related_documents", [])
                if self.min_sim_2hop <= rd["similarity_score"] <= self.max_sim_2hop
            ]

            if not candidates:
                # 범위 내 문서 없으면 가장 가까운 문서 선택
                all_related = target.get("related_documents", [])
                if all_related:
                    # 0.35 (중간값)에 가장 가까운 문서
                    target_sim = 0.35
                    candidates = sorted(
                        all_related,
                        key=lambda x: abs(x["similarity_score"] - target_sim)
                    )[:count_per_target]

            # 상위 N개 선택
            selected = candidates[:count_per_target]

            for related in selected:
                pair = DocumentPair(
                    target_doc_id=target["doc_id"],
                    target_context=target["context"],
                    related_doc_id=related["doc_id"],
                    related_context=related["context"],
                    similarity_score=related["similarity_score"],
                    topic=target["topic"]
                )
                pairs.append(pair)

        return pairs

    def select_3hop_triples(
        self,
        topic: Optional[str] = None,
        count_per_target: int = 1
    ) -> List[DocumentTriple]:
        """
        3-hop 문서 트리플 선정

        선정 기준:
        - 체인 구조: doc1 → doc2 → doc3
        - doc1-doc2, doc2-doc3 연결 필요
        - doc1-doc3 직접 연결 약해야 함

        Args:
            topic: 특정 토픽만 선정
            count_per_target: 타겟당 선정할 트리플 수

        Returns:
            DocumentTriple 리스트
        """
        triples = []

        for target in self.targets:
            if topic and target["topic"] != topic:
                continue

            related_docs = target.get("related_documents", [])
            if len(related_docs) < 2:
                continue

            # 유사도 순 정렬
            sorted_related = sorted(
                related_docs,
                key=lambda x: x["similarity_score"],
                reverse=True
            )

            # 체인 구조 탐색
            # doc2: 타겟과 높은 유사도 (0.3~0.5)
            # doc3: 타겟과 낮은 유사도 (0.15~0.3)
            chain_candidates = []

            for doc2 in sorted_related:
                sim_1_2 = doc2["similarity_score"]
                if not (0.25 <= sim_1_2 <= 0.5):
                    continue

                for doc3 in sorted_related:
                    if doc3["doc_id"] == doc2["doc_id"]:
                        continue

                    sim_1_3 = doc3["similarity_score"]

                    # doc3는 타겟과 덜 유사해야 함 (중간 문서 필수)
                    if sim_1_3 >= sim_1_2:
                        continue

                    if 0.1 <= sim_1_3 <= 0.35:
                        # 체인 강도 = doc2 유사도 - doc3 유사도 (차이 클수록 좋음)
                        chain_strength = sim_1_2 - sim_1_3
                        if chain_strength >= 0.1:
                            chain_candidates.append({
                                "doc2": doc2,
                                "doc3": doc3,
                                "sim_1_2": sim_1_2,
                                "sim_1_3": sim_1_3,
                                "chain_strength": chain_strength
                            })

            # 체인 강도 기준 정렬
            chain_candidates.sort(key=lambda x: x["chain_strength"], reverse=True)

            # 상위 N개 선택
            for candidate in chain_candidates[:count_per_target]:
                triple = DocumentTriple(
                    doc1_id=target["doc_id"],
                    doc1_context=target["context"],
                    doc2_id=candidate["doc2"]["doc_id"],
                    doc2_context=candidate["doc2"]["context"],
                    doc3_id=candidate["doc3"]["doc_id"],
                    doc3_context=candidate["doc3"]["context"],
                    similarity_1_2=candidate["sim_1_2"],
                    similarity_2_3=candidate["sim_1_3"],  # 근사값
                    topic=target["topic"]
                )
                triples.append(triple)

        return triples

    def select_5hop_chains(
        self,
        topic: Optional[str] = None,
        count_per_target: int = 1
    ) -> List[DocumentChain5]:
        """
        5-hop 문서 체인 선정

        선정 기준:
        - 5개 문서가 순차적으로 연결되는 체인 구조
        - 각 hop에서 유사도가 점차 감소 (타겟에서 멀어짐)
        - 중간 문서 없이는 시작과 끝을 연결할 수 없음

        전략:
        - 타겟의 related_documents에서 4개 추가 문서 선정
        - 유사도 순으로 정렬하여 체인 형성
        - doc1(타겟) → doc2(높은 유사도) → doc3 → doc4 → doc5(낮은 유사도)

        Args:
            topic: 특정 토픽만 선정
            count_per_target: 타겟당 선정할 체인 수

        Returns:
            DocumentChain5 리스트
        """
        chains = []

        for target in self.targets:
            if topic and target["topic"] != topic:
                continue

            related_docs = target.get("related_documents", [])
            if len(related_docs) < 4:
                # 5-hop을 위해 최소 4개의 관련 문서 필요
                continue

            # 유사도 순 정렬 (높은 것부터)
            sorted_related = sorted(
                related_docs,
                key=lambda x: x["similarity_score"],
                reverse=True
            )

            # 체인 후보 생성
            # 전략: 유사도가 다양한 4개 문서 선정
            # - doc2: 상위 25% (높은 유사도)
            # - doc3: 상위 25-50%
            # - doc4: 상위 50-75%
            # - doc5: 하위 25% (낮은 유사도)
            chain_candidates = []

            n = len(sorted_related)
            if n >= 4:
                # 각 구간에서 문서 선택
                quartile_1 = sorted_related[:max(1, n//4)]
                quartile_2 = sorted_related[max(1, n//4):max(2, n//2)]
                quartile_3 = sorted_related[max(2, n//2):max(3, 3*n//4)]
                quartile_4 = sorted_related[max(3, 3*n//4):]

                # 빈 구간 처리
                if not quartile_2:
                    quartile_2 = quartile_1
                if not quartile_3:
                    quartile_3 = quartile_2
                if not quartile_4:
                    quartile_4 = quartile_3

                # 조합 생성
                for doc2 in quartile_1[:2]:
                    for doc3 in quartile_2[:2]:
                        if doc3["doc_id"] == doc2["doc_id"]:
                            continue
                        for doc4 in quartile_3[:2]:
                            if doc4["doc_id"] in [doc2["doc_id"], doc3["doc_id"]]:
                                continue
                            for doc5 in quartile_4[:2]:
                                if doc5["doc_id"] in [doc2["doc_id"], doc3["doc_id"], doc4["doc_id"]]:
                                    continue

                                # 체인 점수 계산
                                # - 유사도 분산이 클수록 좋음 (다양한 hop)
                                # - 유사도가 순차적으로 감소하면 보너스
                                sims = [
                                    doc2["similarity_score"],
                                    doc3["similarity_score"],
                                    doc4["similarity_score"],
                                    doc5["similarity_score"]
                                ]

                                # 순차 감소 체크
                                is_decreasing = all(sims[i] >= sims[i+1] for i in range(len(sims)-1))

                                # 점수 = 유사도 범위 + 순차 감소 보너스
                                sim_range = max(sims) - min(sims)
                                chain_score = sim_range + (0.1 if is_decreasing else 0)

                                chain_candidates.append({
                                    "doc2": doc2,
                                    "doc3": doc3,
                                    "doc4": doc4,
                                    "doc5": doc5,
                                    "chain_score": chain_score,
                                    "sims": sims
                                })

            # 점수 기준 정렬
            chain_candidates.sort(key=lambda x: x["chain_score"], reverse=True)

            # 상위 N개 선택
            for candidate in chain_candidates[:count_per_target]:
                chain = DocumentChain5(
                    doc1_id=target["doc_id"],
                    doc1_context=target["context"],
                    doc2_id=candidate["doc2"]["doc_id"],
                    doc2_context=candidate["doc2"]["context"],
                    doc3_id=candidate["doc3"]["doc_id"],
                    doc3_context=candidate["doc3"]["context"],
                    doc4_id=candidate["doc4"]["doc_id"],
                    doc4_context=candidate["doc4"]["context"],
                    doc5_id=candidate["doc5"]["doc_id"],
                    doc5_context=candidate["doc5"]["context"],
                    chain_score=candidate["chain_score"],
                    topic=target["topic"]
                )
                chains.append(chain)

        return chains

    def select_multidoc_groups(
        self,
        topic: Optional[str] = None,
        count_per_target: int = 1,
        group_size: int = 3
    ) -> List[DocumentGroup]:
        """
        Multi-doc 1-hop용 문서 그룹 선정

        비교/종합/추세 분석을 위해 관련 문서들을 그룹화

        선정 기준:
        - 같은 토픽 내 유사한 주제의 문서들
        - 비교 가능한 속성 (시점, 지역, 정책 등)
        - 종합하면 더 풍부한 답변 가능

        Args:
            topic: 특정 토픽만 선정
            count_per_target: 타겟당 선정할 그룹 수
            group_size: 그룹당 문서 수 (2-4개)

        Returns:
            DocumentGroup 리스트
        """
        groups = []

        for target in self.targets:
            if topic and target["topic"] != topic:
                continue

            related_docs = target.get("related_documents", [])
            if len(related_docs) < group_size - 1:
                continue

            # 유사도가 높은 문서들 선정 (비교하기 좋음)
            # Multi-doc 1-hop: 유사도 0.3 이상 (관련성 높은 문서들)
            similar_docs = [
                rd for rd in related_docs
                if rd["similarity_score"] >= 0.25
            ]

            if len(similar_docs) < group_size - 1:
                # 유사도 필터 완화
                similar_docs = sorted(
                    related_docs,
                    key=lambda x: x["similarity_score"],
                    reverse=True
                )[:group_size - 1]

            # 그룹 생성
            selected = similar_docs[:group_size - 1]

            if len(selected) >= 1:
                doc_ids = [target["doc_id"]] + [d["doc_id"] for d in selected]
                doc_contexts = [target["context"]] + [d["context"] for d in selected]

                # 그룹 타입 결정 (휴리스틱)
                avg_sim = sum(d["similarity_score"] for d in selected) / len(selected)
                if avg_sim >= 0.4:
                    group_type = "comparison"  # 높은 유사도 → 비교
                elif avg_sim >= 0.3:
                    group_type = "aggregation"  # 중간 유사도 → 종합
                else:
                    group_type = "trend"  # 낮은 유사도 → 추세/패턴

                group = DocumentGroup(
                    doc_ids=doc_ids,
                    doc_contexts=doc_contexts,
                    group_type=group_type,
                    topic=target["topic"]
                )
                groups.append(group)

        return groups[:count_per_target * len([t for t in self.targets if not topic or t["topic"] == topic])]

    def get_selection_stats(self) -> Dict[str, Any]:
        """선정 통계"""
        stats = {
            "total_targets": len(self.targets),
            "by_topic": {},
        }

        for target in self.targets:
            topic = target["topic"]
            if topic not in stats["by_topic"]:
                stats["by_topic"][topic] = {
                    "targets": 0,
                    "avg_related_docs": 0,
                    "related_in_range_2hop": 0,
                }
            stats["by_topic"][topic]["targets"] += 1

            related = target.get("related_documents", [])
            stats["by_topic"][topic]["avg_related_docs"] += len(related)

            in_range = sum(
                1 for rd in related
                if self.min_sim_2hop <= rd["similarity_score"] <= self.max_sim_2hop
            )
            stats["by_topic"][topic]["related_in_range_2hop"] += in_range

        # 평균 계산
        for topic in stats["by_topic"]:
            n = stats["by_topic"][topic]["targets"]
            if n > 0:
                stats["by_topic"][topic]["avg_related_docs"] /= n
                stats["by_topic"][topic]["related_in_range_2hop"] /= n

        return stats

    def export_selection_plan(
        self,
        output_path: str,
        pairs_per_target: int = 1,
        triples_per_target: int = 1
    ) -> Dict:
        """
        선정 계획 내보내기 (API 호출 전 검토용)

        Args:
            output_path: 출력 JSON 경로
            pairs_per_target: 타겟당 2-hop 쌍 수
            triples_per_target: 타겟당 3-hop 트리플 수

        Returns:
            선정 계획 딕셔너리
        """
        # 문서 선정
        pairs = self.select_2hop_pairs(count_per_target=pairs_per_target)
        triples = self.select_3hop_triples(count_per_target=triples_per_target)

        plan = {
            "metadata": {
                "total_2hop_pairs": len(pairs),
                "total_3hop_triples": len(triples),
                "selection_criteria": {
                    "2hop_similarity_range": f"{self.min_sim_2hop} - {self.max_sim_2hop}",
                    "3hop_chain_structure": "doc1 → doc2 → doc3"
                }
            },
            "2hop_selections": [asdict(p) for p in pairs],
            "3hop_selections": [asdict(t) for t in triples]
        }

        # 저장
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(plan, f, ensure_ascii=False, indent=2)

        print(f"Selection plan exported to {output_path}")
        print(f"  - 2-hop pairs: {len(pairs)}")
        print(f"  - 3-hop triples: {len(triples)}")

        return plan


if __name__ == "__main__":
    # 테스트
    print("=== MultihopDocumentSelector 테스트 ===\n")

    # Golden Dataset 경로
    base_path = Path(__file__).parent.parent.parent / "golden_dataset" / "output"
    dataset_path = base_path / "golden_dataset_v1.json"

    if not dataset_path.exists():
        print(f"Dataset not found: {dataset_path}")
        exit(1)

    # 선정기 초기화
    selector = MultihopDocumentSelector(str(dataset_path))

    # 통계 출력
    stats = selector.get_selection_stats()
    print("=== 선정 통계 ===")
    print(f"총 타겟: {stats['total_targets']}")
    for topic, data in stats["by_topic"].items():
        print(f"  {topic}: {data['targets']} targets, "
              f"avg {data['avg_related_docs']:.1f} related, "
              f"avg {data['related_in_range_2hop']:.1f} in 2-hop range")

    # 2-hop 선정 예시
    print("\n=== 2-hop 선정 예시 (공공행정, 2개) ===")
    pairs = selector.select_2hop_pairs(topic="공공행정", count_per_target=1)[:2]
    for i, pair in enumerate(pairs, 1):
        print(f"\n{i}. Target: {pair.target_doc_id}")
        print(f"   Related: {pair.related_doc_id}")
        print(f"   Similarity: {pair.similarity_score:.3f}")
        print(f"   Target context (100 chars): {pair.target_context[:100]}...")
        print(f"   Related context (100 chars): {pair.related_context[:100]}...")

    # 3-hop 선정 예시
    print("\n=== 3-hop 선정 예시 (공공행정, 1개) ===")
    triples = selector.select_3hop_triples(topic="공공행정", count_per_target=1)[:1]
    for i, triple in enumerate(triples, 1):
        print(f"\n{i}. Chain: {triple.doc1_id} → {triple.doc2_id} → {triple.doc3_id}")
        print(f"   Similarity 1→2: {triple.similarity_1_2:.3f}")
        print(f"   Similarity 2→3: {triple.similarity_2_3:.3f}")

    # 선정 계획 내보내기
    output_path = Path(__file__).parent.parent / "output" / "selection_plan.json"
    selector.export_selection_plan(
        str(output_path),
        pairs_per_target=1,
        triples_per_target=1
    )
