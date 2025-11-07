#!/usr/bin/env python3
"""
Claude Knowledge Extractor
===========================

Extract structured knowledge from Dasan dialogues using Claude API.

Usage:
    uv run python src/knowledge_extraction/claude_knowledge_extractor.py \
        --config config/knowledge_extraction_poc.json \
        --input data/processed/knowledge_extraction/sampled_dialogues.json
"""

import json
import asyncio
import argparse
import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import logging

from src.kg_agent.config.llm_config import get_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClaudeKnowledgeExtractor:
    """
    Extract knowledge from dialogues using Claude API with rate limiting.
    """

    def __init__(
        self,
        model_name: str = "claude-sonnet",
        request_delay: float = 6.0,
        batch_size: int = 15
    ):
        """
        Args:
            model_name: Claude model name (from llm_config)
            request_delay: Delay between API calls in seconds
            batch_size: Number of dialogues per API call
        """
        self.model_name = model_name
        self.request_delay = request_delay
        self.batch_size = batch_size
        self.llm = None
        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        """Load prompt template from file."""
        prompt_path = Path(__file__).parent / "prompts" / "knowledge_extraction_prompt.txt"
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def initialize_llm(self):
        """Initialize Claude LLM client."""
        if self.llm is None:
            logger.info(f"Initializing Claude LLM: {self.model_name}")
            self.llm = get_llm(self.model_name)
            logger.info("✅ LLM initialized")

    async def extract_knowledge_single(
        self,
        dialogue: Dict[str, Any],
        dialogue_idx: int,
        total_dialogues: int
    ) -> List[Dict[str, Any]]:
        """
        Extract knowledge from a single dialogue.

        Args:
            dialogue: Single structured dialogue
            dialogue_idx: Dialogue index (1-based)
            total_dialogues: Total number of dialogues

        Returns:
            List of knowledge documents from this dialogue
        """
        # Prepare single dialogue as list (for consistent format)
        dialogues_json = json.dumps([dialogue], ensure_ascii=False, indent=2)

        # Create prompt
        current_date = datetime.now().strftime("%Y-%m-%d")
        prompt = self.prompt_template.replace("{dialogues_json}", dialogues_json)
        prompt = prompt.replace("{current_date}", current_date)

        try:
            # Call Claude API
            logger.info(f"  🤖 [{dialogue_idx}/{total_dialogues}] Calling API for dialogue {dialogue['dialogue_id']}...")
            response = await asyncio.to_thread(
                self.llm.llm_client.completion,
                model=self.llm.model,
                messages=[{"role": "user", "content": prompt}],
                tools=[]
            )

            # Extract response
            if hasattr(response, 'choices') and len(response.choices) > 0:
                response_text = response.choices[0].message.content
            else:
                response_text = str(response)

            # Parse JSON response
            try:
                # Try to extract JSON from markdown code block if present
                if "```json" in response_text:
                    json_start = response_text.find("```json") + 7
                    json_end = response_text.find("```", json_start)
                    json_text = response_text[json_start:json_end].strip()
                elif "```" in response_text:
                    json_start = response_text.find("```") + 3
                    json_end = response_text.find("```", json_start)
                    json_text = response_text[json_start:json_end].strip()
                else:
                    json_text = response_text.strip()

                documents = json.loads(json_text)
                logger.info(f"  ✅ [{dialogue_idx}/{total_dialogues}] Extracted {len(documents)} documents")
                return documents

            except json.JSONDecodeError as e:
                logger.error(f"  ❌ [{dialogue_idx}/{total_dialogues}] Failed to parse JSON: {e}")
                return []

        except Exception as e:
            logger.error(f"  ❌ [{dialogue_idx}/{total_dialogues}] API call failed: {e}")
            return []

    async def extract_knowledge_batch(
        self,
        dialogues: List[Dict[str, Any]],
        batch_num: int,
        total_batches: int
    ) -> List[Dict[str, Any]]:
        """
        Extract knowledge from a batch of dialogues in parallel.

        Args:
            dialogues: List of structured dialogues (max 10)
            batch_num: Current batch number
            total_batches: Total number of batches

        Returns:
            List of knowledge documents
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing Batch {batch_num}/{total_batches}")
        logger.info(f"{'='*60}")
        logger.info(f"Dialogues in batch: {len(dialogues)}")

        # Rate limiting
        if batch_num > 1:
            logger.info(f"⏳ Waiting {self.request_delay}s for rate limit...")
            await asyncio.sleep(self.request_delay)

        # Process all dialogues in batch concurrently
        logger.info(f"🚀 Starting {len(dialogues)} concurrent API calls...")
        tasks = []
        start_idx = (batch_num - 1) * self.batch_size + 1
        for idx, dialogue in enumerate(dialogues, start=start_idx):
            task = self.extract_knowledge_single(
                dialogue,
                idx,
                len(dialogues) * total_batches
            )
            tasks.append(task)

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks)

        # Flatten results
        all_documents = []
        for docs in results:
            all_documents.extend(docs)

        logger.info(f"✅ Batch complete: {len(all_documents)} total documents extracted")
        return all_documents

    async def extract_all(
        self,
        all_dialogues: Dict[str, List[Dict]],
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        Extract knowledge from all dialogues.

        Args:
            all_dialogues: Dialogues grouped by category
            output_dir: Output directory for documents

        Returns:
            Extraction results and statistics
        """
        self.initialize_llm()

        # Flatten all dialogues
        flat_dialogues = []
        for category, dialogues in all_dialogues.items():
            flat_dialogues.extend(dialogues)

        logger.info(f"\n{'='*60}")
        logger.info("KNOWLEDGE EXTRACTION START")
        logger.info(f"{'='*60}")
        logger.info(f"Total dialogues: {len(flat_dialogues)}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Request delay: {self.request_delay}s")

        # Split into batches
        batches = []
        for i in range(0, len(flat_dialogues), self.batch_size):
            batch = flat_dialogues[i:i + self.batch_size]
            batches.append(batch)

        total_batches = len(batches)
        logger.info(f"Total batches: {total_batches}")
        logger.info(f"Estimated time: {total_batches * self.request_delay / 60:.1f} minutes")

        # Process batches
        all_documents = []
        start_time = datetime.now()

        for batch_num, batch in enumerate(batches, 1):
            batch_docs = await self.extract_knowledge_batch(batch, batch_num, total_batches)
            all_documents.extend(batch_docs)

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Save raw documents
        output_dir.mkdir(parents=True, exist_ok=True)
        raw_output_path = output_dir / "extracted_documents.json"

        with open(raw_output_path, 'w', encoding='utf-8') as f:
            json.dump(all_documents, f, ensure_ascii=False, indent=2)

        logger.info(f"\n{'='*60}")
        logger.info("EXTRACTION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total documents: {len(all_documents)}")
        logger.info(f"Processing time: {duration:.1f}s ({duration/60:.1f} min)")
        logger.info(f"Raw output: {raw_output_path}")

        return {
            "total_documents": len(all_documents),
            "total_batches": total_batches,
            "duration_seconds": duration,
            "documents": all_documents,
            "output_path": str(raw_output_path)
        }


def main():
    parser = argparse.ArgumentParser(description="Extract knowledge from Dasan dialogues using Claude")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config JSON file"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to sampled dialogues JSON"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)"
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Load sampled dialogues
    with open(args.input, 'r', encoding='utf-8') as f:
        sampled_data = json.load(f)

    # Extract params
    model_name = config['llm']['model_name']
    request_delay = config['llm']['rate_limit']['request_delay']
    batch_size = config['llm']['batch_size']
    output_dir = Path(args.output_dir) if args.output_dir else Path(config['experiment']['output_dir'])

    # Initialize extractor
    extractor = ClaudeKnowledgeExtractor(
        model_name=model_name,
        request_delay=request_delay,
        batch_size=batch_size
    )

    # Run extraction
    results = asyncio.run(extractor.extract_all(sampled_data['dialogues'], output_dir))

    # Save results summary
    summary_path = output_dir / "extraction_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            "total_documents": results["total_documents"],
            "total_batches": results["total_batches"],
            "duration_seconds": results["duration_seconds"],
            "output_path": results["output_path"],
            "timestamp": datetime.now().isoformat()
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
