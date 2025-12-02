"""
소수 예시 테스트 파이프라인

API 호출 전 문서 선정 및 변환 로직 검증
실제 API 호출은 --use-api 플래그로 활성화

사용법:
    # Mock으로 테스트 (API 비용 없음)
    python run_test_generation.py --mock

    # 실제 API로 소수 테스트 (2-3개)
    python run_test_generation.py --use-api --count 2

    # 특정 토픽만 테스트
    python run_test_generation.py --topic 공공행정 --count 3
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

from doc_selectors.multihop_selector import MultihopDocumentSelector, DocumentPair, DocumentTriple
from converters.autorag_converter import AutoRAGConverter, AutoRAGQA


def load_golden_dataset():
    """Golden Dataset 로드"""
    base_path = Path(__file__).parent.parent / "golden_dataset" / "output"
    dataset_path = base_path / "golden_dataset_v1.json"

    with open(dataset_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def mock_generate_question(
    context: str,
    question_type: str,
    doc_id: str,
    topic: str
) -> Dict[str, Any]:
    """
    Mock 질문 생성 (API 호출 없음)

    실제 문서 내용을 분석하여 합리적인 Mock 질문 생성
    """
    # 문서에서 키워드 추출 (간단한 휴리스틱)
    keywords = []
    if "년" in context:
        # 연도 추출
        years = re.findall(r'20\d{2}년', context)
        if years:
            keywords.append(years[0])

    if "원" in context:
        # 금액 추출
        amounts = re.findall(r'\d+[억만천]?원', context)
        if amounts:
            keywords.append(amounts[0])

    # 질문 유형별 Mock 생성
    if question_type == "simple_factoid":
        question = f"[Mock-Factoid] {context[:50]}에서 언급된 주요 내용은?"
        answer = context[:30] + "..."
    elif question_type == "constraint":
        constraint = keywords[0] if keywords else "특정 조건"
        question = f"[Mock-Constraint] {constraint}에 해당하는 정보는?"
        answer = "조건부 답변"
    elif question_type == "reasoning":
        question = f"[Mock-Reasoning] {context[:30]}의 원인/결과는?"
        answer = "추론 기반 답변"
    else:
        question = f"[Mock] {context[:50]}...?"
        answer = "Mock 답변"

    return {
        "question": question,
        "answer": answer,
        "question_type": question_type,
        "evidence": context[:100],
        "retrieval_gt": [doc_id],
        "target_doc_id": doc_id,
        "topic": topic,
        "generation_model": "mock"
    }


def mock_generate_multihop_2(
    pair: DocumentPair
) -> Dict[str, Any]:
    """2-hop Multi-hop Mock 질문 생성"""
    return {
        "question": f"[Mock-2hop] {pair.target_context[:30]}와 관련된 {pair.related_context[:30]}의 연결 정보는?",
        "answer": "2-hop 추론 답변",
        "question_type": "multi_hop_2",
        "reasoning_steps": [
            f"Step 1: {pair.target_doc_id}에서 정보 A 확인",
            f"Step 2: {pair.related_doc_id}에서 정보 B로 답변 도출"
        ],
        "evidence_doc1": pair.target_context[:100],
        "evidence_doc2": pair.related_context[:100],
        "retrieval_gt": [pair.target_doc_id, pair.related_doc_id],
        "target_doc_id": pair.target_doc_id,
        "related_doc_ids": [pair.related_doc_id],
        "topic": pair.topic,
        "similarity_score": pair.similarity_score,
        "generation_model": "mock"
    }


def mock_generate_multihop_3(
    triple: DocumentTriple
) -> Dict[str, Any]:
    """3-hop Multi-hop Mock 질문 생성"""
    return {
        "question": f"[Mock-3hop] {triple.doc1_context[:25]}에서 시작하여 {triple.doc3_context[:25]}까지의 연결은?",
        "answer": "3-hop 체인 추론 답변",
        "question_type": "multi_hop_3",
        "reasoning_steps": [
            f"Step 1: {triple.doc1_id}에서 정보 A 확인",
            f"Step 2: {triple.doc2_id}에서 A와 B 연결",
            f"Step 3: {triple.doc3_id}에서 최종 답변 도출"
        ],
        "evidence_doc1": triple.doc1_context[:100],
        "evidence_doc2": triple.doc2_context[:100],
        "evidence_doc3": triple.doc3_context[:100],
        "retrieval_gt": [triple.doc1_id, triple.doc2_id, triple.doc3_id],
        "target_doc_id": triple.doc1_id,
        "related_doc_ids": [triple.doc2_id, triple.doc3_id],
        "topic": triple.topic,
        "chain_similarity": f"{triple.similarity_1_2:.3f} → {triple.similarity_2_3:.3f}",
        "generation_model": "mock"
    }


def run_test_pipeline(
    topic: Optional[str] = None,
    count: int = 3,
    use_api: bool = False,
    output_dir: Optional[str] = None
):
    """
    테스트 파이프라인 실행

    Args:
        topic: 특정 토픽만 테스트 (None이면 첫 토픽)
        count: 생성할 질문 수
        use_api: 실제 API 사용 여부
        output_dir: 출력 디렉토리
    """
    print("=" * 60)
    print("Golden Dataset 질문 생성 테스트 파이프라인")
    print("=" * 60)
    print(f"\n설정:")
    print(f"  - 토픽: {topic or '첫 토픽'}")
    print(f"  - 생성 수: {count}개")
    print(f"  - API 사용: {'예' if use_api else '아니오 (Mock)'}")

    # 1. Golden Dataset 로드
    print("\n[1/5] Golden Dataset 로드...")
    dataset = load_golden_dataset()
    targets = dataset.get("targets", [])
    print(f"  - 총 타겟: {len(targets)}개")

    # 2. Multi-hop 문서 선정
    print("\n[2/5] Multi-hop 문서 선정...")
    base_path = Path(__file__).parent.parent / "golden_dataset" / "output"
    selector = MultihopDocumentSelector(str(base_path / "golden_dataset_v1.json"))

    # 토픽 결정
    if topic is None:
        topic = targets[0]["topic"]
    print(f"  - 선택 토픽: {topic}")

    # 2-hop, 3-hop 선정
    pairs = selector.select_2hop_pairs(topic=topic, count_per_target=1)[:count]
    triples = selector.select_3hop_triples(topic=topic, count_per_target=1)[:max(1, count // 3)]

    print(f"  - 2-hop pairs: {len(pairs)}개")
    print(f"  - 3-hop triples: {len(triples)}개")

    # 3. 질문 생성 (Mock or API)
    print("\n[3/5] 질문 생성...")
    generated_qa = []

    # Single-hop 질문 (타겟 문서 직접 사용)
    topic_targets = [t for t in targets if t["topic"] == topic][:count]
    for i, target in enumerate(topic_targets):
        qa = mock_generate_question(
            context=target["context"],
            question_type="simple_factoid" if i % 2 == 0 else "constraint",
            doc_id=target["doc_id"],
            topic=topic
        )
        generated_qa.append(qa)
        print(f"  - Single-hop {i+1}: {qa['question'][:50]}...")

    # 2-hop 질문
    for i, pair in enumerate(pairs[:count]):
        qa = mock_generate_multihop_2(pair)
        generated_qa.append(qa)
        print(f"  - 2-hop {i+1}: {qa['question'][:50]}...")

    # 3-hop 질문
    for i, triple in enumerate(triples):
        qa = mock_generate_multihop_3(triple)
        generated_qa.append(qa)
        print(f"  - 3-hop {i+1}: {qa['question'][:50]}...")

    print(f"\n  총 생성: {len(generated_qa)}개")

    # 4. AutoRAG 포맷 변환
    print("\n[4/5] AutoRAG 포맷 변환...")
    converter = AutoRAGConverter()
    autorag_qa = converter.convert_qa_list(generated_qa, id_prefix="test")
    print(f"  - 변환 완료: {len(autorag_qa)}개")

    # 5. 결과 저장
    print("\n[5/5] 결과 저장...")
    if output_dir is None:
        output_dir = Path(__file__).parent / "output" / "test_run"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON 저장 (검토용)
    test_output = {
        "metadata": {
            "run_time": datetime.now().isoformat(),
            "topic": topic,
            "use_api": use_api,
            "total_generated": len(generated_qa)
        },
        "generated_qa": generated_qa,
        "autorag_format": [
            {
                "qid": qa.qid,
                "query": qa.query,
                "retrieval_gt": qa.retrieval_gt,
                "generation_gt": qa.generation_gt
            }
            for qa in autorag_qa
        ]
    }

    json_path = output_dir / "test_generation.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(test_output, f, ensure_ascii=False, indent=2)
    print(f"  - JSON: {json_path}")

    # Parquet 저장 (AutoRAG 호환)
    parquet_path = str(output_dir / "qa_test.parquet")
    converter.save_parquet(autorag_qa, parquet_path)
    print(f"  - Parquet: {parquet_path}")

    # 6. 결과 요약
    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)
    print(f"\n생성된 질문 요약:")

    qa_types = {}
    for qa in generated_qa:
        qt = qa["question_type"]
        qa_types[qt] = qa_types.get(qt, 0) + 1

    for qt, cnt in qa_types.items():
        print(f"  - {qt}: {cnt}개")

    print(f"\nMulti-hop 통계:")
    for qa in generated_qa:
        if qa["question_type"] in ["multi_hop_2", "multi_hop_3"]:
            print(f"  - {qa['question_type']}: retrieval_gt = {qa['retrieval_gt']}")

    print(f"\n출력 파일:")
    print(f"  - 검토용: {json_path}")
    print(f"  - AutoRAG용: {parquet_path}")

    return test_output


def main():
    parser = argparse.ArgumentParser(
        description="Golden Dataset 질문 생성 테스트"
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="테스트할 토픽 (기본: 첫 토픽)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=3,
        help="생성할 질문 수 (기본: 3)"
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="실제 GPT-5.1 API 사용 (비용 발생)"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Mock 모드로 실행 (API 비용 없음)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="출력 디렉토리"
    )

    args = parser.parse_args()

    # Mock 플래그가 있으면 API 사용 안 함
    use_api = args.use_api and not args.mock

    run_test_pipeline(
        topic=args.topic,
        count=args.count,
        use_api=use_api,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()
