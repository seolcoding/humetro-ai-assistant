"""
AI Hub 행정문서 기계독해 데이터 → AutoRAG 포맷 변환기

원본 데이터 구조:
{
  "doc_id": "D0000042870847",
  "doc_title": "문서 제목",
  "paragraphs": [{
    "context": "문서 본문...",
    "context_id": "396473",
    "qas": [{
      "question_id": "5289867",
      "question": "질문 텍스트",
      "answers": {"text": "정답 텍스트", "answer_start": 210}
    }]
  }]
}

AutoRAG 포맷:
- corpus.parquet: doc_id, contents, metadata
- qa.parquet: qid, query, retrieval_gt, generation_gt
"""

import json
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


def load_span_extraction_data(file_path: str) -> dict:
    """span_extraction JSON 파일 로드"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def convert_to_autorag(
    source_path: str,
    output_dir: str,
    sample_size: int = 10,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    AI Hub 데이터를 AutoRAG 포맷으로 변환

    Args:
        source_path: span_extraction JSON 파일 경로
        output_dir: 출력 디렉토리
        sample_size: 샘플링할 QA 개수
        train_ratio: Train 데이터 비율
        seed: 랜덤 시드

    Returns:
        (corpus_df, qa_df) 튜플
    """
    random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 데이터 로드
    print(f"Loading data from {source_path}...")
    data = load_span_extraction_data(source_path)

    # 2. 모든 QA 쌍 수집
    all_qa_pairs = []
    for doc in data["data"]:
        doc_id = doc["doc_id"]
        doc_title = doc.get("doc_title", "")
        doc_source = doc.get("doc_source", "")

        for para in doc["paragraphs"]:
            context_id = para["context_id"]
            context = para["context"]

            for qa in para["qas"]:
                if qa.get("is_impossible", False):
                    continue  # 답변 불가 질문 제외

                all_qa_pairs.append(
                    {
                        "doc_id": doc_id,
                        "doc_title": doc_title,
                        "doc_source": doc_source,
                        "context_id": context_id,
                        "context": context,
                        "question_id": qa["question_id"],
                        "question": qa["question"],
                        "answer_text": qa["answers"]["text"],
                    }
                )

    print(f"Total QA pairs found: {len(all_qa_pairs):,}")

    # 3. 샘플링
    if sample_size < len(all_qa_pairs):
        sampled = random.sample(all_qa_pairs, sample_size)
    else:
        sampled = all_qa_pairs
    print(f"Sampled {len(sampled)} QA pairs")

    # 4. Corpus 데이터 생성 (중복 제거)
    corpus_dict = {}
    for item in sampled:
        # doc_id와 context_id를 조합하여 고유 ID 생성
        unique_doc_id = f"{item['doc_id']}_{item['context_id']}"
        if unique_doc_id not in corpus_dict:
            corpus_dict[unique_doc_id] = {
                "doc_id": unique_doc_id,
                "contents": item["context"],
                "metadata": {
                    "last_modified_datetime": datetime.now(),
                    "original_doc_id": item["doc_id"],
                    "doc_title": item["doc_title"],
                    "doc_source": item["doc_source"],
                    "context_id": item["context_id"],
                },
            }

    corpus_df = pd.DataFrame(list(corpus_dict.values()))
    print(f"Corpus documents: {len(corpus_df)}")

    # 5. QA 데이터 생성
    qa_data = []
    for item in sampled:
        unique_doc_id = f"{item['doc_id']}_{item['context_id']}"
        qa_data.append(
            {
                "qid": item["question_id"],
                "query": item["question"],
                "retrieval_gt": [[unique_doc_id]],  # 2D list 형식
                "generation_gt": [item["answer_text"]],  # list 형식
            }
        )

    qa_df = pd.DataFrame(qa_data)

    # 6. Train/Test 분할
    train_size = int(len(qa_df) * train_ratio)
    indices = list(range(len(qa_df)))
    random.shuffle(indices)

    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    qa_train_df = qa_df.iloc[train_indices].reset_index(drop=True)
    qa_test_df = qa_df.iloc[test_indices].reset_index(drop=True)

    print(f"Train QA: {len(qa_train_df)}, Test QA: {len(qa_test_df)}")

    # 7. Parquet 저장
    corpus_path = output_dir / "corpus.parquet"
    qa_train_path = output_dir / "qa_train.parquet"
    qa_test_path = output_dir / "qa_test.parquet"

    corpus_df.to_parquet(corpus_path, index=False)
    qa_train_df.to_parquet(qa_train_path, index=False)
    qa_test_df.to_parquet(qa_test_path, index=False)

    print(f"\nSaved files:")
    print(f"  - {corpus_path}")
    print(f"  - {qa_train_path}")
    print(f"  - {qa_test_path}")

    return corpus_df, qa_train_df, qa_test_df


def verify_data(corpus_path: str, qa_path: str) -> bool:
    """생성된 데이터 검증"""
    print("\n=== Data Verification ===")

    corpus_df = pd.read_parquet(corpus_path)
    qa_df = pd.read_parquet(qa_path)

    print(f"\nCorpus columns: {list(corpus_df.columns)}")
    print(f"QA columns: {list(qa_df.columns)}")

    # 필수 컬럼 확인
    corpus_required = {"doc_id", "contents", "metadata"}
    qa_required = {"qid", "query", "retrieval_gt", "generation_gt"}

    corpus_ok = corpus_required.issubset(set(corpus_df.columns))
    qa_ok = qa_required.issubset(set(qa_df.columns))

    print(f"\nCorpus required columns: {'✅' if corpus_ok else '❌'}")
    print(f"QA required columns: {'✅' if qa_ok else '❌'}")

    # retrieval_gt ID가 corpus에 존재하는지 확인
    corpus_ids = set(corpus_df["doc_id"].tolist())
    all_gt_ids = set()
    for gt in qa_df["retrieval_gt"]:
        for id_list in gt:
            all_gt_ids.update(id_list)

    missing_ids = all_gt_ids - corpus_ids
    if missing_ids:
        print(f"❌ Missing corpus IDs: {missing_ids}")
        return False
    else:
        print("✅ All retrieval_gt IDs exist in corpus")

    # 샘플 데이터 출력
    print("\n=== Sample Data ===")
    print("\nCorpus sample:")
    print(corpus_df.head(2).to_string())

    print("\nQA sample:")
    print(qa_df.head(2).to_string())

    return corpus_ok and qa_ok


if __name__ == "__main__":
    # 기본 설정
    SOURCE_PATH = "data/016.행정_문서_대상_기계독해_데이터/01.데이터/1.Training/라벨링데이터/TL_span_extraction.json"
    OUTPUT_DIR = "src/autorag_pilot/output"

    # 변환 실행
    corpus_df, qa_train_df, qa_test_df = convert_to_autorag(
        source_path=SOURCE_PATH,
        output_dir=OUTPUT_DIR,
        sample_size=10,  # 파일럿용 10개
        train_ratio=0.8,
        seed=42,
    )

    # 검증
    verify_data(
        corpus_path=f"{OUTPUT_DIR}/corpus.parquet",
        qa_path=f"{OUTPUT_DIR}/qa_train.parquet",
    )
