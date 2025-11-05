#!/usr/bin/env python3
"""
Claude Code 비대화형 모드를 사용한 병렬 요약 시스템
- asyncio를 사용한 비동기 처리
- 최대 10개 프로세스 동시 실행
- 각 청크에 대해 구조화된 요약 생성
"""

import asyncio
import json
import os
from pathlib import Path
import subprocess
import logging
from typing import Dict, Any, List
import time
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/dasan_transport_analysis/logs/summarization.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Claude 프롬프트 템플릿
SUMMARIZATION_PROMPT = """다음은 다산콜센터 대중교통 관련 상담 대화 데이터입니다.

이 데이터를 분석하여 다음 관점에서 구조화된 요약을 제공해주세요:

1. **주요 질문 유형 분류**
   - 어떤 유형의 교통 관련 질문들이 있는지
   - 각 유형의 빈도와 중요도

2. **빈번한 교통 문제 식별**
   - 시민들이 자주 겪는 교통 관련 문제
   - 지역별, 시간대별 특징이 있다면 포함

3. **효과적인 답변 패턴**
   - 상담사들이 사용하는 효과적인 답변 전략
   - 정보 제공 방식의 패턴

4. **개선 가능한 영역**
   - 반복되는 문제나 비효율적인 부분
   - 시스템적 개선이 필요한 영역

5. **핵심 인사이트**
   - 데이터에서 발견한 중요한 패턴이나 트렌드
   - 향후 서비스 개선을 위한 제언

아래 JSON 형식으로 구조화된 요약을 제공해주세요:

```json
{
  "question_types": [
    {
      "type": "질문유형명",
      "frequency": "빈도",
      "examples": ["예시1", "예시2"]
    }
  ],
  "common_issues": [
    {
      "issue": "문제설명",
      "frequency": "빈도",
      "context": "맥락정보"
    }
  ],
  "answer_patterns": [
    {
      "pattern": "패턴설명",
      "effectiveness": "효과성",
      "use_cases": ["사용사례1", "사용사례2"]
    }
  ],
  "improvement_areas": [
    {
      "area": "개선영역",
      "reason": "이유",
      "suggestion": "제안사항"
    }
  ],
  "key_insights": [
    "인사이트1",
    "인사이트2"
  ]
}
```

데이터:
"""

async def summarize_chunk(chunk_file: Path, output_dir: Path, semaphore: asyncio.Semaphore, model: str = "haiku") -> Dict[str, Any]:
    """단일 청크 파일을 Claude로 요약
    
    Args:
        chunk_file: 청크 파일 경로
        output_dir: 출력 디렉토리
        semaphore: 동시 실행 제한
        model: Claude 모델 (haiku, sonnet, opus)
    """
    async with semaphore:
        chunk_id = chunk_file.stem
        output_file = output_dir / f"{chunk_id}_summary.json"

        # 이미 요약된 파일이 있으면 스킵
        if output_file.exists():
            logger.info(f"Skipping {chunk_id} - summary already exists")
            with open(output_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        logger.info(f"Processing {chunk_id}")
        start_time = time.time()

        try:
            # 청크 데이터 로드
            with open(chunk_file, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)

            # Claude 입력 준비
            input_text = SUMMARIZATION_PROMPT + json.dumps(chunk_data, ensure_ascii=False, indent=2)

            # Claude 프린트 모드 실행 (비대화형)
            # ANTHROPIC_API_KEY 제거하여 구독(subscription) 사용
            env = os.environ.copy()
            env.pop('ANTHROPIC_API_KEY', None)  # API key 제거
            
            process = await asyncio.create_subprocess_exec(
                'claude', '--print', 
                '--model', model,  # 지정된 모델 사용
                input_text,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env  # 수정된 환경변수 사용
            )

            stdout, stderr = await process.communicate()

            # 디코딩
            stdout_text = stdout.decode('utf-8', errors='replace')
            stderr_text = stderr.decode('utf-8', errors='replace')

            # stderr 출력이 있으면 로깅
            if stderr_text:
                logger.warning(f"{chunk_id} stderr: {stderr_text[:500]}")  # 처음 500자만

            if process.returncode != 0:
                logger.error(f"Error processing {chunk_id}:")
                logger.error(f"  Return code: {process.returncode}")
                logger.error(f"  stderr: {stderr_text}")
                logger.error(f"  stdout (first 500 chars): {stdout_text[:500]}")
                return {
                    "chunk_id": chunk_id,
                    "error": stderr_text,
                    "stdout": stdout_text[:1000],  # 디버깅용
                    "return_code": process.returncode,
                    "status": "failed"
                }

            # Claude 응답 파싱
            response = stdout_text
            
            # 응답 길이 로깅
            logger.info(f"{chunk_id} response length: {len(response)} chars")

            # JSON 부분 추출 (```json ... ``` 사이)
            json_start = response.find('```json')
            json_end = response.rfind('```')

            if json_start != -1 and json_end != -1:
                json_str = response[json_start + 7:json_end].strip()
                try:
                    summary = json.loads(json_str)
                    logger.info(f"{chunk_id} JSON parsed successfully from code block")
                except json.JSONDecodeError as e:
                    logger.error(f"{chunk_id} JSON parse error in code block: {e}")
                    logger.error(f"  JSON string (first 500 chars): {json_str[:500]}")
                    summary = {
                        "raw_response": response[:1000],
                        "parse_error": True,
                        "error_detail": str(e)
                    }
            else:
                # JSON 블록이 없으면 전체 응답을 시도
                try:
                    summary = json.loads(response)
                    logger.info(f"{chunk_id} JSON parsed from full response")
                except json.JSONDecodeError as e:
                    # JSON 파싱 실패 시 원본 텍스트 저장
                    logger.error(f"{chunk_id} JSON parse error: {e}")
                    logger.error(f"  Response (first 500 chars): {response[:500]}")
                    summary = {
                        "raw_response": response[:1000],
                        "parse_error": True,
                        "error_detail": str(e)
                    }

            # 메타데이터 추가
            summary["chunk_id"] = chunk_id
            summary["processing_time"] = time.time() - start_time
            summary["processed_at"] = datetime.now().isoformat()
            summary["dialogue_count"] = chunk_data.get("dialogue_count", 0)

            # 결과 저장
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)

            # Claude 원본 응답도 별도 저장 (디버깅용)
            debug_file = output_dir / f"{chunk_id}_raw_response.txt"
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write("=== STDOUT ===\n")
                f.write(response)
                f.write("\n\n")
                if stderr_text:
                    f.write("=== STDERR ===\n")
                    f.write(stderr_text)
                    f.write("\n")

            logger.info(f"Completed {chunk_id} in {summary['processing_time']:.2f}s")
            return summary

        except Exception as e:
            logger.error(f"Exception processing {chunk_id}: {e}")
            logger.exception(f"Full traceback for {chunk_id}:")  # 전체 스택 트레이스 출력
            return {
                "chunk_id": chunk_id,
                "error": str(e),
                "error_type": type(e).__name__,
                "status": "failed"
            }

async def process_all_chunks(chunks_dir: str, output_dir: str, max_concurrent: int = 10, model: str = "haiku"):
    """모든 청크를 병렬로 처리
    
    Args:
        chunks_dir: 청크 디렉토리 경로
        output_dir: 출력 디렉토리 경로
        max_concurrent: 최대 동시 실행 수
        model: Claude 모델 (haiku, sonnet, opus)
    """
    chunks_path = Path(chunks_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 모든 청크 파일 찾기
    chunk_files = sorted(chunks_path.glob("chunk_*.json"))
    logger.info(f"Found {len(chunk_files)} chunks to process")

    # Semaphore로 동시 실행 제한
    semaphore = asyncio.Semaphore(max_concurrent)

    # 모든 청크 처리 태스크 생성
    tasks = [
        summarize_chunk(chunk_file, output_path, semaphore, model)
        for chunk_file in chunk_files
    ]

    # 병렬 실행
    logger.info(f"Starting parallel processing with {max_concurrent} concurrent processes")
    results = await asyncio.gather(*tasks)

    # 결과 통계
    successful = sum(1 for r in results if "error" not in r)
    failed = len(results) - successful

    logger.info(f"Processing completed: {successful} successful, {failed} failed")

    # 처리 결과 메타데이터 저장
    metadata = {
        "total_chunks": len(chunk_files),
        "successful": successful,
        "failed": failed,
        "max_concurrent": max_concurrent,
        "processed_at": datetime.now().isoformat(),
        "results": [
            {
                "chunk_id": r.get("chunk_id"),
                "status": "failed" if "error" in r else "success",
                "processing_time": r.get("processing_time", 0)
            }
            for r in results
        ]
    }

    metadata_file = output_path / "summarization_metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    return results

def main():
    """메인 실행 함수"""
    chunks_dir = "data/dasan_transport_analysis/chunks"
    output_dir = "data/dasan_transport_analysis/summaries"
    model = "haiku"  # 기본 모델: haiku (빠르고 효율적)

    try:
        # 비동기 처리 실행
        logger.info(f"Using Claude model: {model}")
        results = asyncio.run(process_all_chunks(chunks_dir, output_dir, max_concurrent=10, model=model))

        logger.info("=" * 50)
        logger.info("Summarization completed!")
        logger.info(f"Results saved to {output_dir}")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        raise

if __name__ == "__main__":
    main()