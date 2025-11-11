#!/usr/bin/env python3
"""
Gemini Batch API Output Parser
===============================

Gemini Batch API 결과 JSONL 파싱 및 RAGAS 평가 형식 변환

Input Format (Gemini Batch Output):
    {
        "request": {...},
        "response": {
            "candidates": [{
                "content": {"parts": [{"text": "답변"}]},
                "finishReason": "STOP"
            }],
            "usageMetadata": {...}
        },
        "status": "",
        "processed_time": "2025-11-08T12:33:00.701151+00:00"
    }

Output Format (RAGAS Evaluation):
    {
        "model_name": {
            "user_input": ["Q1", "Q2", ...],
            "response": ["A1", "A2", ...],
            "reference": ["GT1", "GT2", ...],
            "retrieved_contexts": [[C1, C2], ...]
        }
    }

Usage:
    # Parse batch output
    python parse_batch_output.py --input batch_output.jsonl --testset testset.csv --output parsed_results.json

    # Parse and create RAGAS dataset
    python parse_batch_output.py --input output.jsonl --testset testset.csv --model-name "Gemini-2.5-Pro" --output dataset.json
"""

import sys
import json
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class GeminiBatchOutputParser:
    """Gemini Batch API 출력 파서"""

    def __init__(self):
        self.stats = {
            "total": 0,
            "success": 0,
            "failed": 0,
            "max_tokens_exceeded": 0,
            "safety_filtered": 0,
            "errors": []
        }

    def parse_response(self, response_line: Dict[str, Any]) -> Optional[str]:
        """
        단일 응답 라인에서 텍스트 추출

        Args:
            response_line: Gemini Batch API response object

        Returns:
            Generated text or None if failed
        """
        self.stats["total"] += 1

        try:
            # Check status
            if response_line.get("status"):
                error_msg = response_line.get("status", "Unknown error")
                self.stats["failed"] += 1
                self.stats["errors"].append(error_msg)
                logger.warning(f"⚠️ Response failed: {error_msg}")
                return None

            # Extract response
            response = response_line.get("response", {})
            candidates = response.get("candidates", [])

            if not candidates:
                self.stats["failed"] += 1
                self.stats["errors"].append("No candidates in response")
                logger.warning(f"⚠️ No candidates found")
                return None

            # Get first candidate
            candidate = candidates[0]
            finish_reason = candidate.get("finishReason", "")

            # Check finish reason
            if finish_reason == "MAX_TOKENS":
                self.stats["max_tokens_exceeded"] += 1
                logger.warning(f"⚠️ Max tokens exceeded")
            elif finish_reason == "SAFETY":
                self.stats["safety_filtered"] += 1
                logger.warning(f"⚠️ Safety filter triggered")

            # Extract text
            content = candidate.get("content", {})
            parts = content.get("parts", [])

            if not parts:
                self.stats["failed"] += 1
                self.stats["errors"].append("No parts in content")
                return None

            text = parts[0].get("text", "")

            if text:
                self.stats["success"] += 1
                return text
            else:
                self.stats["failed"] += 1
                self.stats["errors"].append("Empty text in response")
                return None

        except Exception as e:
            self.stats["failed"] += 1
            self.stats["errors"].append(str(e))
            logger.error(f"❌ Parse error: {e}")
            return None

    def parse_batch_output(
        self,
        output_path: Path,
        testset_path: Optional[Path] = None
    ) -> Dict[str, List[str]]:
        """
        Batch 출력 JSONL 파일 전체 파싱

        Args:
            output_path: Gemini Batch output JSONL path
            testset_path: Original testset path (for questions/references)

        Returns:
            {
                "questions": [...],
                "answers": [...],
                "references": [...],  # if testset provided
                "contexts": [...]      # if available
            }
        """
        logger.info("="*70)
        logger.info("Gemini Batch Output Parsing")
        logger.info("="*70)
        logger.info(f"📥 Input: {output_path}")

        if not output_path.exists():
            raise FileNotFoundError(f"Output file not found: {output_path}")

        # Load testset if provided
        testset_df = None
        if testset_path and testset_path.exists():
            if testset_path.suffix == '.csv':
                testset_df = pd.read_csv(testset_path)
            elif testset_path.suffix == '.json':
                testset_df = pd.read_json(testset_path)
            logger.info(f"📚 Testset loaded: {len(testset_df)} questions")

        # Parse responses
        questions = []
        answers = []
        contexts_list = []

        with open(output_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    response_data = json.loads(line.strip())

                    # Extract question from request
                    request = response_data.get("request", {})
                    contents = request.get("contents", [])

                    question = None
                    if contents:
                        parts = contents[0].get("parts", [])
                        if parts:
                            # Question is in the user prompt
                            full_text = parts[0].get("text", "")
                            # Extract question (after "질문:" marker)
                            if "질문:" in full_text:
                                question = full_text.split("질문:")[-1].split("답변:")[0].strip()

                    # Extract answer
                    answer = self.parse_response(response_data)

                    if question and answer:
                        questions.append(question)
                        answers.append(answer)
                        contexts_list.append([])  # Contexts not in output

                    if line_num % 10 == 0:
                        logger.info(f"  Processed: {line_num} lines")

                except json.JSONDecodeError as e:
                    logger.error(f"❌ JSON decode error at line {line_num}: {e}")
                    continue
                except Exception as e:
                    logger.error(f"❌ Error at line {line_num}: {e}")
                    continue

        # Get references from testset
        references = []
        if testset_df is not None:
            # Match questions to testset
            reference_col = None
            for col in ['reference', 'reference_answer', 'ground_truth']:
                if col in testset_df.columns:
                    reference_col = col
                    break

            if reference_col:
                testset_questions = testset_df['question'].tolist() if 'question' in testset_df.columns else testset_df['user_input'].tolist()
                testset_references = testset_df[reference_col].tolist()

                # Match by question text
                for q in questions:
                    if q in testset_questions:
                        idx = testset_questions.index(q)
                        references.append(testset_references[idx])
                    else:
                        references.append("")  # No match found
            else:
                logger.warning("⚠️ No reference column found in testset")
                references = [""] * len(questions)
        else:
            references = [""] * len(questions)

        result = {
            "questions": questions,
            "answers": answers,
            "references": references,
            "contexts": contexts_list
        }

        # Print statistics
        logger.info("="*70)
        logger.info("✅ Parsing complete")
        logger.info(f"📊 Statistics:")
        logger.info(f"  Total responses: {self.stats['total']}")
        logger.info(f"  Successful: {self.stats['success']}")
        logger.info(f"  Failed: {self.stats['failed']}")
        logger.info(f"  Max tokens exceeded: {self.stats['max_tokens_exceeded']}")
        logger.info(f"  Safety filtered: {self.stats['safety_filtered']}")
        logger.info(f"\n📈 Results:")
        logger.info(f"  Questions: {len(questions)}")
        logger.info(f"  Answers: {len(answers)}")
        logger.info(f"  References: {len([r for r in references if r])}")
        logger.info("="*70)

        return result

    def create_ragas_dataset(
        self,
        parsed_data: Dict[str, List[str]],
        model_name: str = "Gemini-2.5-Pro"
    ) -> Dict[str, Dict[str, List[str]]]:
        """
        RAGAS 평가 형식으로 변환

        Args:
            parsed_data: parse_batch_output() 결과
            model_name: 모델 이름 (평가 결과에 사용)

        Returns:
            RAGAS evaluation dataset format
        """
        ragas_dataset = {
            model_name: {
                "user_input": parsed_data["questions"],
                "response": parsed_data["answers"],
                "reference": parsed_data["references"],
                "retrieved_contexts": parsed_data["contexts"]
            }
        }

        logger.info(f"\n📦 RAGAS dataset created for: {model_name}")
        logger.info(f"  Questions: {len(ragas_dataset[model_name]['user_input'])}")

        return ragas_dataset

    def save_results(
        self,
        parsed_data: Dict[str, List[str]],
        output_path: Path,
        model_name: Optional[str] = None
    ):
        """결과 저장"""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save parsed data
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(parsed_data, f, ensure_ascii=False, indent=2)

        logger.info(f"💾 Parsed data saved: {output_path}")

        # Save RAGAS format if model_name provided
        if model_name:
            ragas_dataset = self.create_ragas_dataset(parsed_data, model_name)
            ragas_path = output_path.with_name(output_path.stem + "_ragas.json")

            with open(ragas_path, 'w', encoding='utf-8') as f:
                json.dump(ragas_dataset, f, ensure_ascii=False, indent=2)

            logger.info(f"💾 RAGAS dataset saved: {ragas_path}")

        # Save statistics
        stats_path = output_path.with_name(output_path.stem + "_stats.json")
        stats_data = {
            **self.stats,
            "parsed_at": datetime.now().isoformat(),
            "total_questions": len(parsed_data["questions"]),
            "model_name": model_name
        }

        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, ensure_ascii=False, indent=2)

        logger.info(f"💾 Statistics saved: {stats_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Parse Gemini Batch API output JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Gemini Batch output JSONL path"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/batch_processing/parsed_output.json"),
        help="Output JSON path (default: data/batch_processing/parsed_output.json)"
    )

    # Optional
    parser.add_argument(
        "--testset",
        type=Path,
        help="Original testset path (CSV/JSON) for references"
    )
    parser.add_argument(
        "--model-name",
        default="Gemini-2.5-Pro",
        help="Model name for RAGAS dataset (default: Gemini-2.5-Pro)"
    )

    args = parser.parse_args()

    # Parse output
    parser_obj = GeminiBatchOutputParser()

    parsed_data = parser_obj.parse_batch_output(
        output_path=args.input,
        testset_path=args.testset
    )

    # Save results
    parser_obj.save_results(
        parsed_data=parsed_data,
        output_path=args.output,
        model_name=args.model_name
    )

    print(f"\n✅ Parsing complete!")
    print(f"Output: {args.output}")
    print(f"RAGAS dataset: {args.output.with_name(args.output.stem + '_ragas.json')}")


if __name__ == "__main__":
    main()
