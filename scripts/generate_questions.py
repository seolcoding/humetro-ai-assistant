#!/usr/bin/env python3
"""
집계된 요약을 기반으로 테스트 질문 생성
- 마스터 요약 데이터 활용
- Claude Code로 다양한 난이도의 질문 생성
- 실제 시민들이 자주 묻는 패턴 반영
"""

import json
import os
from pathlib import Path
import subprocess
import logging
from typing import Dict, Any, List
from datetime import datetime

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/dasan_transport_analysis/logs/question_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 질문 생성 프롬프트 템플릿
QUESTION_GENERATION_PROMPT = """다음은 서울시 다산콜센터 대중교통 상담 데이터를 분석한 마스터 요약입니다.

이 분석 결과를 바탕으로 실제 시민들이 자주 묻는 패턴을 반영한 20개의 테스트 질문을 생성해주세요.

## 요구사항:
1. **난이도 분포**:
   - 쉬움(7개): 단순 정보 요청, 기본적인 교통 문의
   - 보통(8개): 복합적 경로 문의, 상황별 대안 요청
   - 어려움(5개): 복잡한 환승, 특수 상황, 다중 제약조건

2. **질문 유형 분포**:
   - 버스 노선 및 정류장 관련 (6개)
   - 지하철 및 환승 관련 (5개)
   - 요금 및 할인 관련 (3개)
   - 교통카드 및 결제 관련 (3개)
   - 특수 상황 (야간, 장애인, 분실물 등) (3개)

3. **질문 특징**:
   - 구체적인 지역명이나 노선 번호 포함
   - 실제 시민이 사용할 법한 자연스러운 표현
   - 맥락과 상황 정보 포함
   - 명확한 답변이 가능한 형태

## 출력 형식:
```json
{
  "generated_questions": [
    {
      "id": 1,
      "question": "질문 내용",
      "difficulty": "easy/medium/hard",
      "category": "카테고리명",
      "expected_info_type": "예상되는 답변 정보 유형",
      "context": "질문 맥락 설명"
    }
  ],
  "generation_metadata": {
    "based_on_insights": ["활용한 주요 인사이트 3개"],
    "common_patterns_reflected": ["반영한 패턴 3개"],
    "timestamp": "생성 시각"
  }
}
```

마스터 요약 데이터:
"""

def load_master_summary(file_path: str) -> Dict[str, Any]:
    """마스터 요약 파일 로드"""
    logger.info(f"Loading master summary from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def prepare_summary_for_prompt(master_summary: Dict[str, Any]) -> str:
    """프롬프트용 요약 데이터 준비"""
    # 핵심 정보만 추출하여 간결하게 정리
    condensed_summary = {
        "top_question_types": master_summary.get("question_types", [])[:5],
        "top_common_issues": master_summary.get("common_issues", [])[:5],
        "effective_patterns": master_summary.get("answer_patterns", [])[:3],
        "key_insights": master_summary.get("key_insights", [])[:10],
        "statistics": master_summary.get("summary_statistics", {})
    }

    return json.dumps(condensed_summary, ensure_ascii=False, indent=2)

def generate_questions_with_claude(master_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Claude를 사용하여 질문 생성"""
    logger.info("Generating questions with Claude")

    # 프롬프트 준비
    summary_text = prepare_summary_for_prompt(master_summary)
    full_prompt = QUESTION_GENERATION_PROMPT + summary_text

    try:
        # Claude 헤드리스 모드 실행
        result = subprocess.run(
            ['claude', '-H', '-p', full_prompt],
            capture_output=True,
            text=True,
            timeout=120  # 2분 타임아웃
        )

        if result.returncode != 0:
            logger.error(f"Claude execution failed: {result.stderr}")
            return None

        response = result.stdout

        # JSON 추출
        json_start = response.find('```json')
        json_end = response.rfind('```')

        if json_start != -1 and json_end != -1:
            json_str = response[json_start + 7:json_end].strip()
            questions_data = json.loads(json_str)
        else:
            # JSON 블록이 없으면 전체 응답을 시도
            try:
                questions_data = json.loads(response)
            except:
                logger.error("Failed to parse Claude response as JSON")
                return {
                    "raw_response": response,
                    "parse_error": True
                }

        # 타임스탬프 추가
        if "generation_metadata" not in questions_data:
            questions_data["generation_metadata"] = {}
        questions_data["generation_metadata"]["timestamp"] = datetime.now().isoformat()

        return questions_data

    except subprocess.TimeoutExpired:
        logger.error("Claude execution timed out")
        return None
    except Exception as e:
        logger.error(f"Error generating questions: {e}")
        return None

def validate_questions(questions_data: Dict[str, Any]) -> Dict[str, Any]:
    """생성된 질문 검증 및 보완"""
    if not questions_data or "generated_questions" not in questions_data:
        return questions_data

    questions = questions_data["generated_questions"]

    # 검증 통계
    stats = {
        "total": len(questions),
        "by_difficulty": {},
        "by_category": {}
    }

    for q in questions:
        # ID 확인 및 수정
        if "id" not in q:
            q["id"] = questions.index(q) + 1

        # 난이도 검증
        if "difficulty" in q:
            difficulty = q["difficulty"].lower()
            stats["by_difficulty"][difficulty] = stats["by_difficulty"].get(difficulty, 0) + 1

        # 카테고리 통계
        if "category" in q:
            category = q["category"]
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1

    questions_data["validation_stats"] = stats

    logger.info(f"Validated {stats['total']} questions")
    logger.info(f"Difficulty distribution: {stats['by_difficulty']}")
    logger.info(f"Category distribution: {stats['by_category']}")

    return questions_data

def save_questions(questions_data: Dict[str, Any], output_dir: str):
    """생성된 질문 저장"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 메인 질문 파일 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_path / f"generated_questions_{timestamp}.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(questions_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Saved questions to {output_file}")

    # 질문만 추출한 간단한 버전도 저장
    if "generated_questions" in questions_data:
        simple_questions = [
            {
                "id": q["id"],
                "question": q["question"],
                "difficulty": q.get("difficulty", "unknown"),
                "category": q.get("category", "unknown")
            }
            for q in questions_data["generated_questions"]
        ]

        simple_file = output_path / f"questions_simple_{timestamp}.json"
        with open(simple_file, 'w', encoding='utf-8') as f:
            json.dump(simple_questions, f, ensure_ascii=False, indent=2)

        # 텍스트 버전도 생성
        text_file = output_path / f"questions_text_{timestamp}.txt"
        with open(text_file, 'w', encoding='utf-8') as f:
            for q in simple_questions:
                f.write(f"[{q['id']}] ({q['difficulty']}) {q['category']}: {q['question']}\n")

        logger.info(f"Saved simple version to {simple_file}")
        logger.info(f"Saved text version to {text_file}")

    return output_file

def main():
    """메인 실행 함수"""
    master_summary_file = "data/dasan_transport_analysis/aggregated/master_summary.json"
    output_dir = "data/dasan_transport_analysis/questions"

    try:
        # 마스터 요약 로드
        if not Path(master_summary_file).exists():
            logger.error(f"Master summary file not found: {master_summary_file}")
            logger.info("Please run aggregate_summaries.py first")
            return

        master_summary = load_master_summary(master_summary_file)

        # 질문 생성
        questions_data = generate_questions_with_claude(master_summary)

        if not questions_data:
            logger.error("Failed to generate questions")
            return

        # 질문 검증
        questions_data = validate_questions(questions_data)

        # 질문 저장
        output_file = save_questions(questions_data, output_dir)

        # 결과 출력
        logger.info("=" * 50)
        logger.info("Question generation completed successfully!")

        if "validation_stats" in questions_data:
            stats = questions_data["validation_stats"]
            logger.info(f"Generated {stats['total']} questions")
            logger.info(f"Difficulty: {stats['by_difficulty']}")
            logger.info(f"Categories: {len(stats['by_category'])} unique")

        logger.info(f"Questions saved to {output_dir}")
        logger.info("=" * 50)

        return questions_data

    except Exception as e:
        logger.error(f"Error during question generation: {e}")
        raise

if __name__ == "__main__":
    main()