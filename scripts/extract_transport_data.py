#!/usr/bin/env python3
"""
다산콜센터 대중교통 데이터 추출 및 청킹 스크립트
- 대화셋 단위로 그룹화
- 5개 대화씩 청크로 분할
- 각 청크를 JSON 파일로 저장
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/dasan_transport_analysis/logs/extraction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_transport_data(file_path: str) -> List[Dict[str, Any]]:
    """교통 데이터 JSON 파일 로드"""
    logger.info(f"Loading data from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} records")
    return data

def group_by_dialogue(data: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """대화셋일련번호로 그룹화"""
    logger.info("Grouping data by dialogue set")
    dialogues = defaultdict(list)

    for record in data:
        dialogue_id = record.get('대화셋일련번호', 'UNKNOWN')
        dialogues[dialogue_id].append(record)

    # 각 대화를 문장번호 순으로 정렬
    for dialogue_id in dialogues:
        dialogues[dialogue_id].sort(key=lambda x: int(x.get('문장번호', 0)))

    logger.info(f"Grouped into {len(dialogues)} dialogue sets")
    return dict(dialogues)

def create_dialogue_summary(dialogue_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """하나의 대화셋에서 핵심 정보 추출"""
    summary = {
        "dialogue_id": dialogue_records[0].get('대화셋일련번호'),
        "category": dialogue_records[0].get('카테고리'),
        "turns": len(dialogue_records),
        "conversation": []
    }

    for record in dialogue_records:
        turn = {
            "speaker": record.get('화자'),
            "turn_number": record.get('문장번호'),
            "qa_type": record.get('QA'),
        }

        # 실제 대화 내용 추가
        if record.get('화자') == '고객':
            if record.get('고객질문(요청)'):
                turn['content'] = record.get('고객질문(요청)')
                turn['intent'] = record.get('고객의도')
            elif record.get('고객답변'):
                turn['content'] = record.get('고객답변')
        else:  # 상담사
            if record.get('상담사질문(요청)'):
                turn['content'] = record.get('상담사질문(요청)')
                turn['intent'] = record.get('상담사의도')
            elif record.get('상담사답변'):
                turn['content'] = record.get('상담사답변')

        # 메타데이터 추가
        if record.get('개체명'):
            turn['entities'] = record.get('개체명')
        if record.get('용어사전'):
            turn['terms'] = record.get('용어사전')
        if record.get('지식베이스'):
            turn['knowledge_base'] = record.get('지식베이스')

        summary['conversation'].append(turn)

    return summary

def create_chunks(dialogues: Dict[str, List[Dict[str, Any]]], chunk_size: int = 5) -> List[List[Dict[str, Any]]]:
    """대화셋을 chunk_size 개씩 묶어서 청크 생성"""
    logger.info(f"Creating chunks with size {chunk_size}")

    dialogue_summaries = []
    for dialogue_id, records in dialogues.items():
        summary = create_dialogue_summary(records)
        dialogue_summaries.append(summary)

    # 청크로 분할
    chunks = []
    for i in range(0, len(dialogue_summaries), chunk_size):
        chunk = dialogue_summaries[i:i + chunk_size]
        chunks.append(chunk)

    logger.info(f"Created {len(chunks)} chunks")
    return chunks

def save_chunks(chunks: List[List[Dict[str, Any]]], output_dir: str):
    """청크를 개별 JSON 파일로 저장"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving chunks to {output_dir}")

    chunk_metadata = []

    for idx, chunk in enumerate(chunks, 1):
        chunk_filename = f"chunk_{idx:04d}.json"
        chunk_path = output_path / chunk_filename

        # 청크 데이터 구조화
        chunk_data = {
            "chunk_id": idx,
            "dialogue_count": len(chunk),
            "dialogues": chunk
        }

        with open(chunk_path, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=2)

        # 메타데이터 수집
        chunk_metadata.append({
            "chunk_id": idx,
            "filename": chunk_filename,
            "dialogue_count": len(chunk),
            "total_turns": sum(d['turns'] for d in chunk)
        })

        if idx % 10 == 0:
            logger.info(f"Saved {idx} chunks...")

    # 메타데이터 저장
    metadata_path = output_path / "chunks_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump({
            "total_chunks": len(chunks),
            "chunk_size": 5,
            "chunks": chunk_metadata
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"Completed saving {len(chunks)} chunks")
    return chunk_metadata

def main():
    """메인 실행 함수"""
    # 경로 설정
    input_file = "data/AI_HUB_DASAN_QA/01_extracted/training/labeled/민원(콜센터) 질의응답_다산콜센터_대중교통 안내_Training.json"
    output_dir = "data/dasan_transport_analysis/chunks"

    try:
        # 데이터 로드
        data = load_transport_data(input_file)

        # 대화셋으로 그룹화
        dialogues = group_by_dialogue(data)

        # 청크 생성
        chunks = create_chunks(dialogues, chunk_size=5)

        # 청크 저장
        metadata = save_chunks(chunks, output_dir)

        # 통계 출력
        logger.info("=" * 50)
        logger.info("Extraction completed successfully!")
        logger.info(f"Total dialogues: {len(dialogues)}")
        logger.info(f"Total chunks: {len(chunks)}")
        logger.info(f"Output directory: {output_dir}")
        logger.info("=" * 50)

        return metadata

    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        raise

if __name__ == "__main__":
    main()