"""
TF-IDF 기반 유사 문서 그룹화

타겟 문서에 대해 같은 토픽 내에서 유사한 문서를 찾아 그룹화
Multi-hop 질문 생성을 위한 관련 문서 리스트 구성
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple
from pathlib import Path
from datetime import datetime

try:
    from .schema import (
        TargetDocument, RelatedDocument, GoldenDataset,
        TARGET_TOPICS, QUESTION_DISTRIBUTION
    )
except ImportError:
    from schema import (
        TargetDocument, RelatedDocument, GoldenDataset,
        TARGET_TOPICS, QUESTION_DISTRIBUTION
    )


class DocumentSimilarityFinder:
    """TF-IDF 기반 문서 유사도 계산 및 그룹화"""

    def __init__(
        self,
        min_similarity: float = 0.1,
        max_similarity: float = 0.8,
        top_k: int = 10
    ):
        """
        Args:
            min_similarity: 최소 유사도 (너무 낮으면 무관한 문서)
            max_similarity: 최대 유사도 (너무 높으면 중복 문서)
            top_k: 각 타겟당 관련 문서 개수
        """
        self.min_similarity = min_similarity
        self.max_similarity = max_similarity
        self.top_k = top_k
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),  # unigram + bigram
            min_df=2
        )

    def find_related_documents(
        self,
        target_df: pd.DataFrame,
        corpus_df: pd.DataFrame
    ) -> List[TargetDocument]:
        """
        타겟 문서별로 관련 문서를 찾아 TargetDocument 리스트 반환

        Args:
            target_df: 타겟 문서 DataFrame (120개)
            corpus_df: 전체 코퍼스 DataFrame (6000개)

        Returns:
            List of TargetDocument with related_documents populated
        """
        results = []

        for topic in TARGET_TOPICS:
            print(f"\nProcessing topic: {topic}")

            # 해당 토픽의 타겟과 코퍼스 필터링
            topic_targets = target_df[target_df['topic'] == topic]
            topic_corpus = corpus_df[corpus_df['topic'] == topic]

            print(f"  Targets: {len(topic_targets)}, Corpus: {len(topic_corpus)}")

            # TF-IDF 벡터화 (코퍼스 전체)
            all_contexts = topic_corpus['context'].tolist()
            tfidf_matrix = self.vectorizer.fit_transform(all_contexts)

            # 타겟별로 유사 문서 찾기
            for _, target_row in topic_targets.iterrows():
                target_context = target_row['context']

                # 타겟 벡터화
                target_vec = self.vectorizer.transform([target_context])

                # 코사인 유사도 계산
                similarities = cosine_similarity(target_vec, tfidf_matrix)[0]

                # 자기 자신 제외하고 유사도 순 정렬
                target_idx_in_corpus = topic_corpus[
                    topic_corpus['doc_id'] == target_row['doc_id']
                ].index

                # 유사도 필터링 및 정렬
                sim_scores = []
                for idx, (corpus_idx, corpus_row) in enumerate(topic_corpus.iterrows()):
                    if corpus_idx in target_idx_in_corpus.tolist():
                        continue  # 자기 자신 제외

                    sim = similarities[idx]
                    if self.min_similarity <= sim <= self.max_similarity:
                        sim_scores.append((corpus_row, sim))

                # Top-K 선택
                sim_scores.sort(key=lambda x: x[1], reverse=True)
                top_related = sim_scores[:self.top_k]

                # RelatedDocument 리스트 생성
                related_docs = [
                    RelatedDocument(
                        doc_id=row['doc_id'],
                        doc_title=row['doc_title'],
                        context=row['context'],
                        similarity_score=round(sim, 4),
                        context_length=row['context_length']
                    )
                    for row, sim in top_related
                ]

                # TargetDocument 생성
                target_doc = TargetDocument(
                    doc_id=target_row['doc_id'],
                    doc_title=target_row['doc_title'],
                    doc_source=target_row['doc_source'],
                    topic=target_row['topic'],
                    context=target_row['context'],
                    context_length=target_row['context_length'],
                    related_documents=related_docs,
                    qa_pairs=[],  # QA는 이후 단계에서 생성
                    sample_type="target"
                )

                results.append(target_doc)
                print(f"    {target_row['doc_title'][:30]}... → {len(related_docs)} related docs")

        return results


def build_golden_dataset(
    target_path: str,
    corpus_path: str,
    output_path: str,
    min_similarity: float = 0.1,
    max_similarity: float = 0.8,
    top_k: int = 10
) -> GoldenDataset:
    """
    Golden Dataset 구축 메인 함수

    Args:
        target_path: 타겟 CSV 경로
        corpus_path: 코퍼스 CSV 경로
        output_path: 출력 JSON 경로

    Returns:
        GoldenDataset object
    """
    print("=== Building Golden Dataset ===")
    print(f"Target: {target_path}")
    print(f"Corpus: {corpus_path}")

    # 데이터 로드
    target_df = pd.read_csv(target_path)
    corpus_df = pd.read_csv(corpus_path)

    print(f"\nLoaded: {len(target_df)} targets, {len(corpus_df)} corpus docs")

    # 유사 문서 찾기
    finder = DocumentSimilarityFinder(
        min_similarity=min_similarity,
        max_similarity=max_similarity,
        top_k=top_k
    )
    targets = finder.find_related_documents(target_df, corpus_df)

    # GoldenDataset 생성
    dataset = GoldenDataset(
        version="1.0.0",
        created_at=datetime.now().isoformat(),
        random_state=42,
        topics=TARGET_TOPICS,
        question_distribution=QUESTION_DISTRIBUTION,
        targets=targets
    )

    # 저장
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    dataset.save(output_path)
    print(f"\nSaved to {output_path}")

    # 통계 출력
    print("\n=== Statistics ===")
    print(f"Total targets: {len(targets)}")
    for topic in TARGET_TOPICS:
        topic_targets = [t for t in targets if t.topic == topic]
        avg_related = np.mean([len(t.related_documents) for t in topic_targets])
        print(f"  {topic}: {len(topic_targets)} targets, avg {avg_related:.1f} related docs")

    return dataset


if __name__ == "__main__":
    # 실행
    BASE_DIR = Path(__file__).parent.parent / "sampling" / "output"

    build_golden_dataset(
        target_path=str(BASE_DIR / "target_120.csv"),
        corpus_path=str(BASE_DIR / "corpus_sampled_6000.csv"),
        output_path=str(Path(__file__).parent / "output" / "golden_dataset_v1.json"),
        min_similarity=0.1,
        max_similarity=0.8,
        top_k=10
    )
