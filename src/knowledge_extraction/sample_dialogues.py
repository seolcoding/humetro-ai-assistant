#!/usr/bin/env python3
"""
Dialogue Sampling Module
========================

Extract sample dialogues from Dasan Call Center training data.

Usage:
    uv run python src/knowledge_extraction/sample_dialogues.py \
        --config config/knowledge_extraction_poc.json
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DialogueSampler:
    """
    Sample dialogues from Dasan Call Center JSON files.
    """

    def __init__(self, source_dir: Path, categories: List[Dict[str, Any]]):
        """
        Args:
            source_dir: Directory containing training JSON files
            categories: List of category configs with file names and sample sizes
        """
        self.source_dir = Path(source_dir)
        self.categories = categories

    def load_dialogues_from_file(self, file_path: Path) -> Dict[str, List[Dict]]:
        """
        Load dialogues from JSON file and group by dialogue ID.

        Args:
            file_path: Path to JSON file

        Returns:
            Dict mapping dialogue_id -> list of turns (sorted by turn number)
        """
        logger.info(f"Loading dialogues from {file_path.name}...")

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Group by dialogue ID
        dialogues = defaultdict(list)
        for record in data:
            dialogue_id = record['대화셋일련번호']
            dialogues[dialogue_id].append(record)

        # Sort turns by turn number
        for dialogue_id in dialogues:
            dialogues[dialogue_id].sort(key=lambda x: int(x['문장번호']))

        logger.info(f"  Loaded {len(dialogues)} dialogues ({len(data)} total turns)")
        return dict(dialogues)

    def sample_dialogues(
        self,
        dialogues: Dict[str, List[Dict]],
        sample_size: int,
        seed: int = 42
    ) -> Dict[str, List[Dict]]:
        """
        Randomly sample dialogues.

        Args:
            dialogues: All dialogues from file
            sample_size: Number of dialogues to sample
            seed: Random seed for reproducibility

        Returns:
            Sampled dialogues
        """
        random.seed(seed)
        dialogue_ids = list(dialogues.keys())

        if len(dialogue_ids) < sample_size:
            logger.warning(
                f"  Requested {sample_size} dialogues but only {len(dialogue_ids)} available"
            )
            sampled_ids = dialogue_ids
        else:
            sampled_ids = random.sample(dialogue_ids, sample_size)

        sampled = {did: dialogues[did] for did in sampled_ids}
        logger.info(f"  Sampled {len(sampled)} dialogues")
        return sampled

    def structure_dialogue(self, turns: List[Dict], category: str) -> Dict[str, Any]:
        """
        Convert raw turns into structured dialogue format.

        Args:
            turns: List of turn records
            category: Category name

        Returns:
            Structured dialogue dict
        """
        dialogue_id = turns[0]['대화셋일련번호']

        structured_turns = []
        for turn in turns:
            # Determine speaker and text
            speaker = "customer" if turn['화자'] == "고객" else "advisor"
            if turn['QA'] == 'Q':
                text = turn.get('고객질문(요청)', '') or turn.get('상담사질문(요청)', '')
                intent = turn.get('고객의도', '') or turn.get('상담사의도', '')
            else:
                text = turn.get('고객답변', '') or turn.get('상담사답변', '')
                intent = turn.get('고객의도', '') or turn.get('상담사의도', '')

            # Parse entities
            entities_raw = turn.get('개체명', '')
            entities = [e.strip() for e in entities_raw.split(',') if e.strip()]

            # Parse terminology
            terminology_raw = turn.get('용어사전', '')
            terminology = {}
            if terminology_raw:
                for item in terminology_raw.split('/'):
                    item = item.strip()
                    if item:
                        parts = item.split('/')
                        if len(parts) >= 2:
                            entity = parts[0].strip()
                            entity_type = parts[1].strip()
                            if entity and entity_type:
                                terminology[entity] = entity_type

            # Parse KB tags
            kb_tags_raw = turn.get('지식베이스', '')
            kb_tags = [tag.strip() for tag in kb_tags_raw.split(',') if tag.strip()]

            structured_turns.append({
                "turn_num": int(turn['문장번호']),
                "speaker": speaker,
                "qa_type": turn['QA'],
                "text": text,
                "intent": intent,
                "entities": entities,
                "terminology": terminology,
                "kb_tags": kb_tags
            })

        return {
            "dialogue_id": dialogue_id,
            "category": category,
            "total_turns": len(structured_turns),
            "turns": structured_turns
        }

    def run(self, output_path: Path) -> Dict[str, Any]:
        """
        Run sampling for all categories and save results.

        Args:
            output_path: Path to output JSON file

        Returns:
            Sampled dialogues by category
        """
        all_sampled = {}
        stats = {
            "total_dialogues": 0,
            "total_turns": 0,
            "categories": {}
        }

        for category_config in self.categories:
            category_name = category_config['name']
            file_name = category_config['file']
            sample_size = category_config['sample_size']

            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {category_name}")
            logger.info(f"{'='*60}")

            # Load dialogues
            file_path = self.source_dir / file_name
            dialogues = self.load_dialogues_from_file(file_path)

            # Sample
            sampled = self.sample_dialogues(dialogues, sample_size)

            # Structure
            structured = []
            total_turns = 0
            for dialogue_id, turns in sampled.items():
                structured_dialogue = self.structure_dialogue(turns, category_name)
                structured.append(structured_dialogue)
                total_turns += len(turns)

            all_sampled[category_name] = structured

            # Update stats
            stats["total_dialogues"] += len(sampled)
            stats["total_turns"] += total_turns
            stats["categories"][category_name] = {
                "dialogues": len(sampled),
                "turns": total_turns,
                "avg_turns": total_turns / len(sampled) if sampled else 0
            }

            logger.info(f"  Dialogues: {len(sampled)}")
            logger.info(f"  Total turns: {total_turns}")
            logger.info(f"  Avg turns/dialogue: {total_turns / len(sampled):.1f}")

        # Save results
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "metadata": {
                "description": "Sampled dialogues from Dasan Call Center for knowledge extraction POC",
                "source_dir": str(self.source_dir),
                "stats": stats
            },
            "dialogues": all_sampled
        }

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2)

        logger.info(f"\n{'='*60}")
        logger.info("SAMPLING COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total dialogues: {stats['total_dialogues']}")
        logger.info(f"Total turns: {stats['total_turns']}")
        logger.info(f"Output saved to: {output_path}")

        return output


def main():
    parser = argparse.ArgumentParser(description="Sample dialogues from Dasan Call Center data")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (overrides config)"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Extract params
    source_dir = Path(config['data']['source_dir'])
    categories = config['data']['categories']
    output_path = Path(args.output) if args.output else Path(config['data']['sampled_output'])

    # Run sampling
    sampler = DialogueSampler(source_dir, categories)
    sampler.run(output_path)


if __name__ == "__main__":
    main()
