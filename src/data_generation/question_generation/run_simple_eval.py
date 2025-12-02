"""
간단한 RAG 평가 스크립트

AutoRAG 없이 직접 평가 수행:
1. Vector Search로 문서 검색
2. GPT-4o-mini로 답변 생성
3. Retrieval 및 Generation 평가
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

import pandas as pd
import numpy as np

from openai import OpenAI


def embed_texts(client: OpenAI, texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """텍스트 임베딩"""
    response = client.embeddings.create(input=texts, model=model)
    return np.array([e.embedding for e in response.data])


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """코사인 유사도 계산"""
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.dot(a_norm, b_norm.T)


def retrieve_top_k(
    query_embedding: np.ndarray,
    corpus_embeddings: np.ndarray,
    corpus_ids: List[str],
    k: int = 5
) -> List[str]:
    """Top-K 검색"""
    similarities = cosine_similarity(query_embedding.reshape(1, -1), corpus_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:k]
    return [corpus_ids[i] for i in top_indices]


def generate_answer(
    client: OpenAI,
    query: str,
    contexts: List[str],
    model: str = "gpt-4o-mini"
) -> str:
    """답변 생성"""
    context_str = "\n\n".join([f"[문서 {i+1}]\n{ctx}" for i, ctx in enumerate(contexts)])

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "다음 문서들을 참고하여 질문에 간결하게 답변하세요."},
            {"role": "user", "content": f"문서:\n{context_str}\n\n질문: {query}\n답변:"}
        ],
        max_tokens=256,
        temperature=0.1
    )
    return response.choices[0].message.content.strip()


def calculate_retrieval_metrics(
    retrieved_ids: List[str],
    ground_truth_ids: List[str]
) -> Dict[str, float]:
    """Retrieval 메트릭 계산"""
    retrieved_set = set(retrieved_ids)
    gt_set = set(ground_truth_ids)

    if not gt_set:
        return {"precision": 0, "recall": 0, "f1": 0}

    tp = len(retrieved_set & gt_set)
    precision = tp / len(retrieved_set) if retrieved_set else 0
    recall = tp / len(gt_set) if gt_set else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1": f1}


def main():
    print("=" * 70)
    print("간단한 RAG 평가 (GPT-4o-mini)")
    print("=" * 70)

    # 데이터 로드
    output_base = Path(__file__).parent / "output"
    dirs = sorted(output_base.glob("e2e_test_*"), reverse=True)
    if not dirs:
        print("Error: 생성된 데이터가 없습니다")
        return

    data_dir = dirs[0]
    print(f"\n데이터 디렉토리: {data_dir}")

    qa_df = pd.read_parquet(data_dir / "qa.parquet")
    corpus_df = pd.read_parquet(data_dir / "corpus.parquet")

    print(f"QA: {len(qa_df)}개, Corpus: {len(corpus_df)}개")

    # OpenAI 클라이언트
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # 1. Corpus 임베딩
    print("\n[1/4] Corpus 임베딩 생성...")
    corpus_texts = corpus_df["contents"].tolist()
    corpus_ids = corpus_df["doc_id"].tolist()

    # 배치 처리 (100개씩)
    batch_size = 100
    corpus_embeddings = []
    for i in range(0, len(corpus_texts), batch_size):
        batch = corpus_texts[i:i+batch_size]
        embeddings = embed_texts(client, batch)
        corpus_embeddings.append(embeddings)
        print(f"  {min(i+batch_size, len(corpus_texts))}/{len(corpus_texts)} 완료")

    corpus_embeddings = np.vstack(corpus_embeddings)
    print(f"  Corpus embeddings shape: {corpus_embeddings.shape}")

    # 2. Query 임베딩 및 검색
    print("\n[2/4] Query 검색...")
    queries = qa_df["query"].tolist()
    query_embeddings = embed_texts(client, queries)

    retrieval_results = []
    for i, query_emb in enumerate(query_embeddings):
        retrieved = retrieve_top_k(query_emb, corpus_embeddings, corpus_ids, k=5)
        retrieval_results.append(retrieved)
        print(f"  Query {i+1}: Retrieved {len(retrieved)} docs")

    # 3. 답변 생성
    print("\n[3/4] 답변 생성...")
    generated_answers = []
    for i, (query, retrieved) in enumerate(zip(queries, retrieval_results)):
        # 검색된 문서 컨텍스트 가져오기
        contexts = []
        for doc_id in retrieved[:3]:  # Top 3만 사용
            doc_row = corpus_df[corpus_df["doc_id"] == doc_id]
            if not doc_row.empty:
                contexts.append(doc_row.iloc[0]["contents"])

        answer = generate_answer(client, query, contexts)
        generated_answers.append(answer)
        print(f"  Query {i+1}: {answer[:50]}...")

    # 4. 평가
    print("\n[4/4] 평가 결과...")

    results = []
    for i, row in qa_df.iterrows():
        gt_ids = list(row["retrieval_gt"])
        gt_answer = row["generation_gt"][0] if row["generation_gt"] else ""
        retrieved = retrieval_results[i]
        generated = generated_answers[i]

        # Retrieval 메트릭
        ret_metrics = calculate_retrieval_metrics(retrieved, gt_ids)

        # 답변 포함 여부 (간단한 체크)
        answer_match = gt_answer.lower() in generated.lower() if gt_answer else False

        result = {
            "qid": row["qid"],
            "query": row["query"][:50] + "...",
            "n_gt_docs": len(gt_ids),
            "retrieval_recall": ret_metrics["recall"],
            "retrieval_f1": ret_metrics["f1"],
            "answer_contains_gt": answer_match,
            "gt_answer": gt_answer[:30] if gt_answer else "",
            "generated_answer": generated[:50] + "..."
        }
        results.append(result)

    # 결과 출력
    print("\n" + "=" * 70)
    print("평가 결과")
    print("=" * 70)

    results_df = pd.DataFrame(results)
    print("\n=== 개별 결과 ===")
    for _, r in results_df.iterrows():
        hop_type = "Multi-hop" if r["n_gt_docs"] > 1 else "Single-hop"
        print(f"\n[{r['qid']}] ({hop_type}, {r['n_gt_docs']}개 문서)")
        print(f"  Query: {r['query']}")
        print(f"  GT Answer: {r['gt_answer']}")
        print(f"  Generated: {r['generated_answer']}")
        print(f"  Retrieval Recall: {r['retrieval_recall']:.2f}, F1: {r['retrieval_f1']:.2f}")
        print(f"  Answer Match: {'✓' if r['answer_contains_gt'] else '✗'}")

    # 전체 통계
    print("\n" + "=" * 70)
    print("=== 전체 통계 ===")
    print("=" * 70)

    avg_recall = results_df["retrieval_recall"].mean()
    avg_f1 = results_df["retrieval_f1"].mean()
    answer_match_rate = results_df["answer_contains_gt"].mean()

    # Single-hop vs Multi-hop 분리
    single_hop = results_df[results_df["n_gt_docs"] == 1]
    multi_hop = results_df[results_df["n_gt_docs"] > 1]

    print(f"\n전체 ({len(results_df)}개):")
    print(f"  Retrieval Recall: {avg_recall:.3f}")
    print(f"  Retrieval F1: {avg_f1:.3f}")
    print(f"  Answer Match Rate: {answer_match_rate:.3f}")

    if len(single_hop) > 0:
        print(f"\nSingle-hop ({len(single_hop)}개):")
        print(f"  Retrieval Recall: {single_hop['retrieval_recall'].mean():.3f}")
        print(f"  Answer Match Rate: {single_hop['answer_contains_gt'].mean():.3f}")

    if len(multi_hop) > 0:
        print(f"\nMulti-hop ({len(multi_hop)}개):")
        print(f"  Retrieval Recall: {multi_hop['retrieval_recall'].mean():.3f}")
        print(f"  Answer Match Rate: {multi_hop['answer_contains_gt'].mean():.3f}")

    # 결과 저장
    output_path = data_dir / "evaluation_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "metrics": {
                "overall": {
                    "retrieval_recall": avg_recall,
                    "retrieval_f1": avg_f1,
                    "answer_match_rate": answer_match_rate,
                    "n_questions": len(results_df)
                },
                "single_hop": {
                    "retrieval_recall": single_hop['retrieval_recall'].mean() if len(single_hop) > 0 else None,
                    "answer_match_rate": single_hop['answer_contains_gt'].mean() if len(single_hop) > 0 else None,
                    "n_questions": len(single_hop)
                },
                "multi_hop": {
                    "retrieval_recall": multi_hop['retrieval_recall'].mean() if len(multi_hop) > 0 else None,
                    "answer_match_rate": multi_hop['answer_contains_gt'].mean() if len(multi_hop) > 0 else None,
                    "n_questions": len(multi_hop)
                }
            },
            "details": results
        }, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n결과 저장: {output_path}")


if __name__ == "__main__":
    main()
