#!/usr/bin/env python3
"""
Dasan Call Center Dialogue -> Gemini Batch API Input Generator
===============================================================

다산 콜센터 대화 JSON 파일들을 Gemini Batch API 입력 JSONL로 변환

Input:
    - data/AI_HUB_DASAN_QA/01_extracted/ 의 모든 JSON 파일
    - 각 JSON 파일: [{"대화셋일련번호": "B25060", ...}, ...]

Output:
    - JSONL file: 각 대화(대화셋일련번호)당 1개의 batch request
    - 프롬프트: 대화 내용 → 구조화된 정보 추출

Usage:
    # Test with 50 dialogues
    python create_dasan_batch_input.py --output batch_input_test_50.jsonl --sample 50

    # Full dataset (~8000 dialogues)
    python create_dasan_batch_input.py --output batch_input_full.jsonl
"""

import sys
import json
import random
import argparse
import logging
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class DasanDialogueLoader:
    """다산 콜센터 대화 데이터 로더"""

    def __init__(self, data_dir: Path):
        """
        Args:
            data_dir: data/AI_HUB_DASAN_QA/01_extracted 디렉토리
        """
        self.data_dir = data_dir

    def load_all_dialogues(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        모든 JSON 파일에서 대화 로드

        Returns:
            {
                "B25060": [{"화자": "고객", ...}, {"화자": "상담사", ...}],
                "B25061": [...],
                ...
            }
        """
        logger.info("="*70)
        logger.info("Loading Dasan Call Center Dialogues")
        logger.info("="*70)

        # Find all JSON files
        json_files = list(self.data_dir.rglob("*.json"))
        logger.info(f"📂 Found {len(json_files)} JSON files")

        # Load all dialogues
        all_dialogues = defaultdict(list)

        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                logger.info(f"  Loading: {json_file.name} ({len(data)} records)")

                # Group by 대화셋일련번호
                for record in data:
                    dialogue_id = record.get("대화셋일련번호")
                    if dialogue_id:
                        all_dialogues[dialogue_id].append(record)

            except Exception as e:
                logger.error(f"  ❌ Failed to load {json_file.name}: {e}")
                continue

        logger.info(f"\n✅ Loaded {len(all_dialogues)} unique dialogues")
        logger.info(f"📊 Total turns: {sum(len(turns) for turns in all_dialogues.values())}")

        return dict(all_dialogues)

    def format_dialogue(self, dialogue_turns: List[Dict[str, Any]]) -> str:
        """
        대화 턴들을 텍스트로 포맷팅

        Args:
            dialogue_turns: 대화 턴 리스트 (문장번호 순서대로 정렬됨)

        Returns:
            포맷된 대화 텍스트
        """
        # Sort by 문장번호
        sorted_turns = sorted(dialogue_turns, key=lambda x: int(x.get("문장번호", 0)))

        dialogue_text = []

        for turn in sorted_turns:
            speaker = turn.get("화자", "")

            # Get the actual utterance
            if speaker == "고객":
                utterance = turn.get("고객질문(요청)") or turn.get("고객답변", "")
            elif speaker == "상담사":
                utterance = turn.get("상담사질문(요청)") or turn.get("상담사답변", "")
            else:
                utterance = ""

            if utterance and utterance.strip():
                dialogue_text.append(f"{speaker}: {utterance.strip()}")

        return "\n".join(dialogue_text)

    def get_dialogue_metadata(self, dialogue_turns: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        대화 메타데이터 추출

        Returns:
            {"도메인": "...", "카테고리": "...", "대화셋일련번호": "..."}
        """
        if not dialogue_turns:
            return {}

        first_turn = dialogue_turns[0]
        return {
            "도메인": first_turn.get("도메인", ""),
            "카테고리": first_turn.get("카테고리", ""),
            "대화셋일련번호": first_turn.get("대화셋일련번호", "")
        }


class DasanBatchInputGenerator:
    """Gemini Batch API 입력 생성기"""

    def __init__(
        self,
        model: str = "gemini-2.5-pro",
        temperature: float = 0.1,  # 낮춤: 결정론적 지식 추출
        max_output_tokens: int = 8192  # 증가: 긴 대화 처리 (Gemini 2.5 최대치)
    ):
        self.model = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens

    def get_response_schema(self) -> Dict[str, Any]:
        """
        Structured Output을 위한 JSON Schema 정의

        Returns:
            Gemini API response_schema format
        """
        return {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "dialogue_id": {
                        "type": "STRING",
                        "description": "원본 대화 ID (대화셋일련번호)"
                    },
                    "original_question": {
                        "type": "STRING",
                        "description": "자연스럽게 재구성된 고객 질문"
                    },
                    "original_answer": {
                        "type": "STRING",
                        "description": "완결된 상담사 답변"
                    },
                    "topic_path": {
                        "type": "STRING",
                        "description": "계층적 주제 경로 (예: 대중교통_안내/버스/노선안내)"
                    },
                    "primary_topic": {
                        "type": "STRING",
                        "description": "주요 주제 (예: 버스_노선안내)"
                    },
                    "secondary_topics": {
                        "type": "ARRAY",
                        "items": {"type": "STRING"},
                        "description": "보조 주제 리스트"
                    },
                    "document": {
                        "type": "STRING",
                        "description": "Front matter + Markdown 형식의 완성된 지식 문서"
                    }
                },
                "required": [
                    "dialogue_id",
                    "original_question",
                    "original_answer",
                    "topic_path",
                    "primary_topic",
                    "secondary_topics",
                    "document"
                ]
            }
        }

    def load_extraction_prompt_template(self) -> str:
        """
        지식 추출 프롬프트 템플릿 로드

        Returns:
            Prompt template string
        """
        prompt_path = Path("src/knowledge_extraction/prompts/knowledge_extraction_prompt.txt")

        if not prompt_path.exists():
            logger.warning(f"⚠️ Prompt template not found: {prompt_path}")
            logger.warning("Using fallback extraction prompt")
            return self._get_fallback_prompt()

        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                template = f.read()
            logger.info(f"✅ Loaded prompt template: {prompt_path}")
            return template
        except Exception as e:
            logger.error(f"❌ Failed to load prompt template: {e}")
            return self._get_fallback_prompt()

    def _get_fallback_prompt(self) -> str:
        """Fallback extraction prompt (간단 버전)"""
        return """당신은 서울시 120 다산콜센터 지식베이스 구축 전문가입니다.

주어진 대화를 분석하여 재사용 가능한 지식 문서를 생성하세요.

출력 형식: JSON 배열
[
  {
    "dialogue_id": "B2033",
    "original_question": "자연스러운 질문문",
    "original_answer": "완결된 답변",
    "topic_path": "카테고리/주제/세부주제",
    "primary_topic": "주요_주제",
    "secondary_topics": ["보조주제1", "보조주제2"],
    "document": "---\\nmarkdown 문서 내용\\n---"
  }
]

{dialogues_json}

위 대화를 분석하여 지식 문서를 JSON 배열로 생성하세요."""

    def create_extraction_prompt(self, dialogue_text: str, metadata: Dict[str, str]) -> str:
        """
        구조화된 정보 추출 프롬프트 생성

        Args:
            dialogue_text: 포맷된 대화 텍스트
            metadata: 대화 메타데이터

        Returns:
            Extraction prompt
        """
        # Load template
        if not hasattr(self, '_prompt_template'):
            self._prompt_template = self.load_extraction_prompt_template()

        # Prepare dialogue JSON
        dialogue_json = {
            "dialogue_id": metadata.get('대화셋일련번호', 'N/A'),
            "category": metadata.get('카테고리', 'N/A'),
            "domain": metadata.get('도메인', 'N/A'),
            "turns_text": dialogue_text
        }

        dialogues_json_str = json.dumps([dialogue_json], ensure_ascii=False, indent=2)

        # Fill template
        prompt = self._prompt_template.replace("{dialogues_json}", dialogues_json_str)

        return prompt

    def create_batch_request(
        self,
        dialogue_id: str,
        dialogue_turns: List[Dict[str, Any]],
        loader: DasanDialogueLoader
    ) -> Dict[str, Any]:
        """
        단일 대화에 대한 Batch API 요청 생성

        Returns:
            Gemini Batch API request format
        """
        # Format dialogue
        dialogue_text = loader.format_dialogue(dialogue_turns)
        metadata = loader.get_dialogue_metadata(dialogue_turns)

        # Create prompt
        prompt = self.create_extraction_prompt(dialogue_text, metadata)

        # Batch API request format with structured output
        request = {
            "request": {
                "contents": [
                    {
                        "role": "user",
                        "parts": [{"text": prompt}]
                    }
                ],
                "generationConfig": {
                    "temperature": self.temperature,
                    "maxOutputTokens": self.max_output_tokens,
                    "responseMimeType": "application/json",  # JSON 출력 강제
                    "responseSchema": self.get_response_schema()  # Schema 적용
                }
            }
        }

        return request

    def generate_batch_input(
        self,
        dialogues: Dict[str, List[Dict[str, Any]]],
        output_path: Path,
        sample_size: int = None,
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """
        모든 대화에 대한 Batch 입력 JSONL 생성

        Args:
            dialogues: {dialogue_id: [turns]}
            output_path: 출력 JSONL 경로
            sample_size: 샘플링 크기 (None = 전체)
            random_seed: 랜덤 시드

        Returns:
            통계 정보
        """
        logger.info("\n" + "="*70)
        logger.info("Generating Gemini Batch Input JSONL")
        logger.info("="*70)

        # Sample if needed
        dialogue_ids = list(dialogues.keys())

        if sample_size and sample_size < len(dialogue_ids):
            random.seed(random_seed)
            dialogue_ids = random.sample(dialogue_ids, sample_size)
            logger.info(f"🎲 Sampled {sample_size} dialogues from {len(dialogues)}")
        else:
            logger.info(f"📝 Processing all {len(dialogue_ids)} dialogues")

        # Create loader
        loader = DasanDialogueLoader(Path("data/AI_HUB_DASAN_QA/01_extracted"))

        # Generate requests
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for i, dialogue_id in enumerate(dialogue_ids, 1):
                try:
                    dialogue_turns = dialogues[dialogue_id]

                    # Create batch request
                    request = self.create_batch_request(
                        dialogue_id=dialogue_id,
                        dialogue_turns=dialogue_turns,
                        loader=loader
                    )

                    # Write JSONL
                    f.write(json.dumps(request, ensure_ascii=False) + '\n')

                    if i % 100 == 0:
                        logger.info(f"  Progress: {i}/{len(dialogue_ids)}")

                except Exception as e:
                    logger.error(f"❌ Failed to process {dialogue_id}: {e}")
                    continue

        # Statistics
        stats = {
            "total_dialogues": len(dialogue_ids),
            "sample_size": sample_size,
            "random_seed": random_seed,
            "output_file": str(output_path),
            "model": self.model,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "created_at": datetime.now().isoformat()
        }

        logger.info("\n" + "="*70)
        logger.info("✅ Batch input generation complete")
        logger.info(f"📊 Total requests: {stats['total_dialogues']}")
        logger.info(f"📁 Output: {stats['output_file']}")
        logger.info(f"🤖 Model: {stats['model']}")
        logger.info("="*70)

        # Save metadata
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info(f"💾 Metadata saved: {metadata_path}")

        return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate Gemini Batch API input from Dasan dialogues",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Input/Output
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/AI_HUB_DASAN_QA/01_extracted"),
        help="Dasan data directory (default: data/AI_HUB_DASAN_QA/01_extracted)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path"
    )

    # Sampling
    parser.add_argument(
        "--sample",
        type=int,
        help="Sample N dialogues (default: use all)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    # Model config
    parser.add_argument(
        "--model",
        default="gemini-2.5-pro",
        help="Model name (default: gemini-2.5-pro)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature (default: 0.1, low for deterministic extraction)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Max output tokens (default: 8192, Gemini 2.5 maximum)"
    )

    args = parser.parse_args()

    # Load dialogues
    loader = DasanDialogueLoader(args.data_dir)
    dialogues = loader.load_all_dialogues()

    # Generate batch input
    generator = DasanBatchInputGenerator(
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_tokens
    )

    stats = generator.generate_batch_input(
        dialogues=dialogues,
        output_path=args.output,
        sample_size=args.sample,
        random_seed=args.seed
    )

    print(f"\n✅ Success!")
    print(f"Generated {stats['total_dialogues']} requests")
    print(f"Output: {stats['output_file']}")


if __name__ == "__main__":
    main()
