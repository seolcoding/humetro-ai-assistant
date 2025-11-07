#!/usr/bin/env python3
"""
요약 결과 집계 스크립트 (MapReduce 스타일)
- 모든 개별 요약 파일 로드
- 패턴별 빈도 분석 및 통합
- 전체 인사이트 생성
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict, Counter
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/dasan_transport_analysis/logs/aggregation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SummaryAggregator:
    """요약 결과를 집계하는 클래스"""

    def __init__(self):
        self.question_types = defaultdict(lambda: {"frequency": 0, "examples": []})
        self.common_issues = defaultdict(lambda: {"frequency": 0, "contexts": []})
        self.answer_patterns = defaultdict(lambda: {"effectiveness": [], "use_cases": []})
        self.improvement_areas = []
        self.key_insights = []
        self.chunk_count = 0

    def load_summaries(self, summaries_dir: str) -> List[Dict[str, Any]]:
        """모든 요약 파일 로드"""
        summaries_path = Path(summaries_dir)
        summary_files = sorted(summaries_path.glob("chunk_*_summary.json"))

        logger.info(f"Loading {len(summary_files)} summary files")

        summaries = []
        for file_path in summary_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    summary = json.load(f)
                    if "error" not in summary:  # 성공한 요약만 포함
                        summaries.append(summary)
            except Exception as e:
                logger.warning(f"Failed to load {file_path}: {e}")

        logger.info(f"Successfully loaded {len(summaries)} summaries")
        return summaries

    def aggregate_question_types(self, summaries: List[Dict[str, Any]]):
        """질문 유형 집계"""
        for summary in summaries:
            if "question_types" in summary:
                for qt in summary["question_types"]:
                    type_name = qt.get("type", "Unknown")
                    self.question_types[type_name]["frequency"] += 1

                    # 예시 수집 (최대 5개씩)
                    examples = qt.get("examples", [])
                    current_examples = self.question_types[type_name]["examples"]
                    for ex in examples:
                        if ex not in current_examples and len(current_examples) < 5:
                            current_examples.append(ex)

    def aggregate_common_issues(self, summaries: List[Dict[str, Any]]):
        """공통 문제 집계"""
        for summary in summaries:
            if "common_issues" in summary:
                for issue in summary["common_issues"]:
                    issue_name = issue.get("issue", "Unknown")
                    self.common_issues[issue_name]["frequency"] += 1

                    # 맥락 정보 수집
                    context = issue.get("context")
                    if context and context not in self.common_issues[issue_name]["contexts"]:
                        self.common_issues[issue_name]["contexts"].append(context)

    def aggregate_answer_patterns(self, summaries: List[Dict[str, Any]]):
        """답변 패턴 집계"""
        for summary in summaries:
            if "answer_patterns" in summary:
                for pattern in summary["answer_patterns"]:
                    pattern_name = pattern.get("pattern", "Unknown")

                    # 효과성 점수 수집
                    effectiveness = pattern.get("effectiveness")
                    if effectiveness:
                        self.answer_patterns[pattern_name]["effectiveness"].append(effectiveness)

                    # 사용 사례 수집
                    use_cases = pattern.get("use_cases", [])
                    for uc in use_cases:
                        if uc not in self.answer_patterns[pattern_name]["use_cases"]:
                            self.answer_patterns[pattern_name]["use_cases"].append(uc)

    def aggregate_improvements(self, summaries: List[Dict[str, Any]]):
        """개선 영역 집계"""
        improvement_counter = Counter()

        for summary in summaries:
            if "improvement_areas" in summary:
                for area in summary["improvement_areas"]:
                    area_name = area.get("area", "Unknown")
                    improvement_counter[area_name] += 1

                    # 상세 정보 저장
                    self.improvement_areas.append({
                        "area": area_name,
                        "reason": area.get("reason"),
                        "suggestion": area.get("suggestion"),
                        "chunk_id": summary.get("chunk_id")
                    })

        # 빈도순 정렬
        self.improvement_areas = sorted(
            self.improvement_areas,
            key=lambda x: improvement_counter[x["area"]],
            reverse=True
        )

    def aggregate_insights(self, summaries: List[Dict[str, Any]]):
        """핵심 인사이트 집계"""
        insight_counter = Counter()

        for summary in summaries:
            if "key_insights" in summary:
                for insight in summary["key_insights"]:
                    insight_counter[insight] += 1

        # 빈도가 높은 인사이트 선택
        self.key_insights = [
            insight for insight, count in insight_counter.most_common(20)
        ]

    def create_master_summary(self) -> Dict[str, Any]:
        """최종 마스터 요약 생성"""
        # 질문 유형 정렬 및 정리
        sorted_question_types = sorted(
            self.question_types.items(),
            key=lambda x: x[1]["frequency"],
            reverse=True
        )

        # 공통 문제 정렬 및 정리
        sorted_issues = sorted(
            self.common_issues.items(),
            key=lambda x: x[1]["frequency"],
            reverse=True
        )

        # 답변 패턴 정리
        answer_patterns_list = []
        for pattern_name, data in self.answer_patterns.items():
            # 평균 효과성 계산
            effectiveness_scores = data["effectiveness"]
            avg_effectiveness = (
                sum(float(e) if e.isdigit() else 0 for e in effectiveness_scores) / len(effectiveness_scores)
                if effectiveness_scores else 0
            )

            answer_patterns_list.append({
                "pattern": pattern_name,
                "avg_effectiveness": avg_effectiveness,
                "use_cases": data["use_cases"][:5]  # 상위 5개만
            })

        # 개선 영역 그룹화
        improvement_groups = defaultdict(list)
        for area in self.improvement_areas[:20]:  # 상위 20개
            improvement_groups[area["area"]].append({
                "reason": area["reason"],
                "suggestion": area["suggestion"]
            })

        master_summary = {
            "total_chunks_analyzed": self.chunk_count,
            "question_types": [
                {
                    "type": qt[0],
                    "total_frequency": qt[1]["frequency"],
                    "examples": qt[1]["examples"]
                }
                for qt in sorted_question_types[:10]  # 상위 10개
            ],
            "common_issues": [
                {
                    "issue": issue[0],
                    "total_frequency": issue[1]["frequency"],
                    "contexts": issue[1]["contexts"][:3]  # 상위 3개 맥락
                }
                for issue in sorted_issues[:10]  # 상위 10개
            ],
            "answer_patterns": sorted(
                answer_patterns_list,
                key=lambda x: x["avg_effectiveness"],
                reverse=True
            )[:10],  # 상위 10개
            "improvement_areas": [
                {
                    "area": area,
                    "occurrences": len(details),
                    "details": details[:3]  # 상위 3개 상세
                }
                for area, details in list(improvement_groups.items())[:10]  # 상위 10개
            ],
            "key_insights": self.key_insights[:15],  # 상위 15개
            "summary_statistics": {
                "unique_question_types": len(self.question_types),
                "unique_issues": len(self.common_issues),
                "unique_patterns": len(self.answer_patterns),
                "improvement_suggestions": len(self.improvement_areas)
            }
        }

        return master_summary

    def aggregate_all(self, summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
        """모든 집계 수행"""
        self.chunk_count = len(summaries)

        logger.info("Aggregating question types...")
        self.aggregate_question_types(summaries)

        logger.info("Aggregating common issues...")
        self.aggregate_common_issues(summaries)

        logger.info("Aggregating answer patterns...")
        self.aggregate_answer_patterns(summaries)

        logger.info("Aggregating improvement areas...")
        self.aggregate_improvements(summaries)

        logger.info("Aggregating key insights...")
        self.aggregate_insights(summaries)

        logger.info("Creating master summary...")
        return self.create_master_summary()

def main():
    """메인 실행 함수"""
    summaries_dir = "data/dasan_transport_analysis/summaries"
    output_dir = "data/dasan_transport_analysis/aggregated"

    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    try:
        # 집계 객체 생성
        aggregator = SummaryAggregator()

        # 요약 파일 로드
        summaries = aggregator.load_summaries(summaries_dir)

        if not summaries:
            logger.error("No summaries found to aggregate")
            return

        # 집계 수행
        master_summary = aggregator.aggregate_all(summaries)

        # 결과 저장
        output_file = output_path / "master_summary.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(master_summary, f, ensure_ascii=False, indent=2)

        logger.info("=" * 50)
        logger.info("Aggregation completed successfully!")
        logger.info(f"Master summary saved to {output_file}")
        logger.info(f"Analyzed {master_summary['total_chunks_analyzed']} chunks")
        logger.info(f"Found {master_summary['summary_statistics']['unique_question_types']} unique question types")
        logger.info(f"Identified {master_summary['summary_statistics']['unique_issues']} unique issues")
        logger.info("=" * 50)

        return master_summary

    except Exception as e:
        logger.error(f"Error during aggregation: {e}")
        raise

if __name__ == "__main__":
    main()