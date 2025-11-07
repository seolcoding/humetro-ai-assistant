#!/usr/bin/env python3
"""
Claude Knowledge Extractor v2
==============================

Improved version with:
- Correct rate limiting (from first call, not completion)
- Incremental saving after each batch
- Local caching and resume capability
- Retry logic for JSON parsing errors

Usage:
    uv run python src/knowledge_extraction/claude_knowledge_extractor_v2.py \
        --config config/knowledge_extraction_poc.json \
        --input data/processed/knowledge_extraction/sampled_dialogues.json
"""

import json
import asyncio
import argparse
import os
import time
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import hashlib

from loguru import logger
from src.kg_agent.config.llm_config import get_llm

# Remove default handler
logger.remove()

# Add custom handler with beautiful formatting (no color tags in message)
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - {message}",
    level="INFO",
    colorize=True
)

# Add file handler for persistent logs
logger.add(
    "logs/knowledge_extraction_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    rotation="00:00",  # Rotate at midnight
    retention="30 days",
    compression="zip"
)


class ClaudeKnowledgeExtractorV2:
    """
    Extract knowledge from dialogues using Claude API with robust error handling.

    Features:
    - Correct rate limiting (60s from first call, not completion)
    - Incremental saving after each batch
    - Local caching and resume capability
    - Retry logic for JSON parsing errors
    """

    def __init__(
        self,
        model_name: str = "claude-sonnet",
        request_delay: float = 65.0,
        batch_size: int = 10,
        max_retries: int = 3,
        cache_dir: Optional[Path] = None
    ):
        """
        Args:
            model_name: Claude model name (from llm_config)
            request_delay: Delay between batch starts in seconds (from first call, +5s safety margin)
            batch_size: Number of dialogues per batch
            max_retries: Maximum retries for failed extractions
            cache_dir: Directory for caching results
        """
        self.model_name = model_name
        self.request_delay = request_delay
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.cache_dir = cache_dir
        self.llm = None
        self.prompt_template = self._load_prompt_template()

        # Initialize cache
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self.cache_file = self.cache_dir / "extraction_cache.json"
            self.cache = self._load_cache()
        else:
            self.cache = {}

    def _load_prompt_template(self) -> str:
        """Load prompt template from file."""
        prompt_path = Path(__file__).parent / "prompts" / "knowledge_extraction_prompt.txt"
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read()

    def _load_cache(self) -> Dict[str, Any]:
        """Load extraction cache from disk."""
        if self.cache_file.exists():
            with open(self.cache_file, 'r', encoding='utf-8') as f:
                cache = json.load(f)
                logger.success(f"📂 Loaded cache: {len(cache)} processed dialogues")
                return cache
        logger.info("📂 No existing cache found, starting fresh")
        return {}

    def _save_cache(self):
        """Save extraction cache to disk."""
        if self.cache_dir:
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def _get_dialogue_hash(self, dialogue: Dict[str, Any]) -> str:
        """Generate unique hash for dialogue."""
        dialogue_str = json.dumps(dialogue, sort_keys=True, ensure_ascii=False)
        return hashlib.md5(dialogue_str.encode()).hexdigest()

    def _is_processed(self, dialogue: Dict[str, Any]) -> bool:
        """Check if dialogue has been processed."""
        dialogue_hash = self._get_dialogue_hash(dialogue)
        return dialogue_hash in self.cache

    def _cache_result(self, dialogue: Dict[str, Any], documents: List[Dict[str, Any]]):
        """Cache extraction result."""
        dialogue_hash = self._get_dialogue_hash(dialogue)
        self.cache[dialogue_hash] = {
            "dialogue_id": dialogue.get("dialogue_id"),
            "documents": documents,
            "timestamp": datetime.now().isoformat()
        }

    def initialize_llm(self):
        """Initialize Claude LLM client."""
        if self.llm is None:
            logger.info(f"🤖 Initializing Claude LLM: {self.model_name}")
            self.llm = get_llm(self.model_name)
            logger.success("✅ LLM initialized successfully")

    async def extract_knowledge_single(
        self,
        dialogue: Dict[str, Any],
        dialogue_idx: int,
        total_dialogues: int,
        retry_count: int = 0
    ) -> Tuple[List[Dict[str, Any]], bool, Optional[Dict[str, Any]]]:
        """
        Extract knowledge from a single dialogue (single attempt, no retry).

        Args:
            dialogue: Single structured dialogue
            dialogue_idx: Dialogue index (1-based)
            total_dialogues: Total number of dialogues
            retry_count: Current retry attempt (for logging only)

        Returns:
            Tuple of (documents list, success flag, failed dialogue or None)
        """
        # Check cache first
        if self._is_processed(dialogue):
            dialogue_hash = self._get_dialogue_hash(dialogue)
            cached_docs = self.cache[dialogue_hash]["documents"]
            logger.debug(f"  💾 [{dialogue_idx}/{total_dialogues}] Using cached result for {dialogue['dialogue_id']}")
            return cached_docs, True, None

        # Prepare single dialogue as list (for consistent format)
        dialogues_json = json.dumps([dialogue], ensure_ascii=False, indent=2)

        # Create prompt
        current_date = datetime.now().strftime("%Y-%m-%d")
        prompt = self.prompt_template.replace("{dialogues_json}", dialogues_json)
        prompt = prompt.replace("{current_date}", current_date)

        try:
            # Call Claude API
            retry_suffix = f" (retry {retry_count})" if retry_count > 0 else ""
            logger.info(f"  🤖 [{dialogue_idx}/{total_dialogues}] Calling API for {dialogue['dialogue_id']}{retry_suffix}")

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

            # Parse JSON response with improved error handling
            try:
                documents = self._parse_json_response(response_text)

                if documents:
                    logger.success(f"  ✅ [{dialogue_idx}/{total_dialogues}] Extracted {len(documents)} documents")
                    # Cache successful result
                    self._cache_result(dialogue, documents)
                    return documents, True, None
                else:
                    raise ValueError("Empty document list returned")

            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"  ❌ [{dialogue_idx}/{total_dialogues}] JSON parsing failed: {e}")
                # Return failed dialogue for retry in next batch
                return [], False, dialogue

        except Exception as e:
            logger.error(f"  ❌ [{dialogue_idx}/{total_dialogues}] API call failed: {e}")
            # Return failed dialogue for retry in next batch
            return [], False, dialogue

    def _parse_json_response(self, response_text: str) -> List[Dict[str, Any]]:
        """
        Parse JSON response with multiple strategies.

        Args:
            response_text: Raw response text

        Returns:
            Parsed document list

        Raises:
            json.JSONDecodeError: If all parsing strategies fail
        """
        # Strategy 1: Try to extract JSON from markdown code block
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

        # Strategy 2: Try to find JSON array brackets
        if not json_text.startswith('['):
            array_start = json_text.find('[')
            if array_start != -1:
                json_text = json_text[array_start:]

        if not json_text.endswith(']'):
            array_end = json_text.rfind(']')
            if array_end != -1:
                json_text = json_text[:array_end + 1]

        # Parse JSON
        documents = json.loads(json_text)

        # Validate structure
        if not isinstance(documents, list):
            raise ValueError(f"Expected list, got {type(documents)}")

        # Validate each document has required fields
        required_fields = ["dialogue_id", "original_question", "original_answer",
                          "topic_path", "primary_topic", "secondary_topics", "document"]
        for idx, doc in enumerate(documents):
            missing_fields = [f for f in required_fields if f not in doc]
            if missing_fields:
                raise ValueError(f"Document {idx} missing fields: {missing_fields}")

        return documents

    async def extract_knowledge_batch(
        self,
        dialogues: List[Dict[str, Any]],
        batch_num: int,
        total_batches: int,
        output_dir: Path,
        skip_rate_limit: bool = False
    ) -> Tuple[List[Dict[str, Any]], int, int, List[Dict[str, Any]]]:
        """
        Extract knowledge from a batch of dialogues in parallel.

        Args:
            dialogues: List of structured dialogues
            batch_num: Current batch number
            total_batches: Total number of batches
            output_dir: Output directory for incremental saving

        Returns:
            Tuple of (documents list, success count, failure count, failed dialogues list)
        """
        logger.info("")
        logger.info(f"{'='*40}")
        logger.info(f"📦 Batch {batch_num}/{total_batches}")
        logger.info(f"{'='*40}")
        logger.info(f"📊 Dialogues in batch: {len(dialogues)}")

        # Rate limiting handled by caller
        # Record batch start time
        batch_start = time.time()

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

        # Process results
        all_documents = []
        success_count = 0
        failure_count = 0
        failed_dialogues = []

        for docs, success, failed_dialogue in results:
            all_documents.extend(docs)
            if success:
                success_count += 1
            else:
                failure_count += 1
                if failed_dialogue:
                    failed_dialogues.append(failed_dialogue)

        batch_duration = time.time() - batch_start

        logger.success(f"✅ Batch {batch_num} complete: {len(all_documents)} documents extracted")
        logger.info(f"   📈 Success: {success_count}/{len(dialogues)}, Failed: {failure_count}/{len(dialogues)}")
        logger.info(f"   ⏱️  Duration: {batch_duration:.1f}s")

        # Incremental save after each batch
        self._save_batch_results(batch_num, all_documents, output_dir)
        self._save_cache()

        return all_documents, success_count, failure_count, failed_dialogues

    def _save_batch_results(
        self,
        batch_num: int,
        documents: List[Dict[str, Any]],
        output_dir: Path
    ):
        """Save batch results incrementally."""
        batch_file = output_dir / f"batch_{batch_num:03d}.json"
        with open(batch_file, 'w', encoding='utf-8') as f:
            json.dump(documents, f, ensure_ascii=False, indent=2)
        logger.success(f"💾 Batch saved: {batch_file.name}")

    async def extract_all(
        self,
        all_dialogues: Dict[str, List[Dict]],
        output_dir: Path
    ) -> Dict[str, Any]:
        """
        Extract knowledge from all dialogues with resume capability.

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

        logger.info("")
        logger.info(f"{'='*40}")
        logger.info(f"🚀 KNOWLEDGE EXTRACTION START")
        logger.info(f"{'='*40}")
        logger.info(f"📊 Total dialogues: {len(flat_dialogues)}")
        logger.info(f"📦 Batch size: {self.batch_size}")
        logger.info(f"⏳ Request delay: {self.request_delay}s (from first call, +5s safety margin)")
        logger.info(f"🔄 Max retries: {self.max_retries}")

        # Split into batches
        batches = []
        for i in range(0, len(flat_dialogues), self.batch_size):
            batch = flat_dialogues[i:i + self.batch_size]
            batches.append(batch)

        total_batches = len(batches)
        logger.info(f"📦 Total batches: {total_batches}")

        # Corrected time estimation: batches run in parallel, delay is between batch STARTS
        estimated_time = (total_batches - 1) * self.request_delay  # -1 because first batch has no delay
        logger.info(f"⏱️  Estimated time: {estimated_time / 60:.1f} minutes (minimum)")
        logger.info(f"{'='*40}")

        # Process batches
        output_dir.mkdir(parents=True, exist_ok=True)
        all_documents = []
        total_success = 0
        total_failure = 0
        all_failed_dialogues = []
        start_time = datetime.now()
        last_batch_start_time = None

        for batch_num, batch in enumerate(batches, 1):
            # Rate limiting: Wait until request_delay has passed since last batch START
            if last_batch_start_time is not None:
                elapsed = time.time() - last_batch_start_time
                remaining_delay = self.request_delay - elapsed
                if remaining_delay > 0:
                    logger.info(f"⏳ Rate limit: waiting {remaining_delay:.1f}s (elapsed: {elapsed:.1f}s)")
                    await asyncio.sleep(remaining_delay)
                else:
                    logger.info(f"⏭️  No wait needed (elapsed: {elapsed:.1f}s >= {self.request_delay}s)")

            # Record this batch's start time
            last_batch_start_time = time.time()

            batch_docs, success, failure, failed = await self.extract_knowledge_batch(
                batch, batch_num, total_batches, output_dir, skip_rate_limit=True
            )
            all_documents.extend(batch_docs)
            total_success += success
            total_failure += failure
            all_failed_dialogues.extend(failed)

        # Retry failed dialogues (up to max_retries attempts)
        if all_failed_dialogues:
            logger.info("")
            logger.info(f"{'='*40}")
            logger.info(f"🔄 RETRYING FAILED DIALOGUES")
            logger.info(f"{'='*40}")
            logger.info(f"Total failed: {len(all_failed_dialogues)}")

            for retry_attempt in range(1, self.max_retries + 1):
                if not all_failed_dialogues:
                    break

                logger.info(f"\n🔄 Retry attempt {retry_attempt}/{self.max_retries} for {len(all_failed_dialogues)} dialogues")

                # Create retry batch
                retry_batch_num = total_batches + retry_attempt
                retry_docs, retry_success, retry_failure, still_failed = await self.extract_knowledge_batch(
                    all_failed_dialogues, retry_batch_num, total_batches + self.max_retries, output_dir
                )

                all_documents.extend(retry_docs)
                total_success += retry_success
                total_failure = total_failure - retry_success + retry_failure  # Update failure count

                # Update failed list for next retry
                all_failed_dialogues = still_failed

                if not all_failed_dialogues:
                    logger.success(f"✅ All failed dialogues recovered!")
                    break

            # Final failed count
            if all_failed_dialogues:
                logger.warning(f"⚠️  {len(all_failed_dialogues)} dialogues permanently failed after {self.max_retries} retries")
                # Cache permanently failed dialogues as empty to skip in future runs
                for dialogue in all_failed_dialogues:
                    self._cache_result(dialogue, [])
                self._save_cache()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()

        # Save consolidated results
        raw_output_path = output_dir / "extracted_documents.json"
        with open(raw_output_path, 'w', encoding='utf-8') as f:
            json.dump(all_documents, f, ensure_ascii=False, indent=2)

        logger.info("")
        logger.info(f"{'='*40}")
        logger.info(f"🎉 EXTRACTION COMPLETE")
        logger.info(f"{'='*40}")
        logger.info(f"📄 Total documents: {len(all_documents)}")
        logger.info(f"📈 Success: {total_success}/{len(flat_dialogues)}, Failed: {total_failure}/{len(flat_dialogues)}")
        success_rate = (total_success / len(flat_dialogues) * 100) if len(flat_dialogues) > 0 else 0
        logger.info(f"✅ Success rate: {success_rate:.1f}%")
        logger.info(f"⏱️  Processing time: {duration:.1f}s ({duration/60:.1f} min)")
        logger.info(f"💾 Output: {raw_output_path}")
        logger.info(f"{'='*40}")

        return {
            "total_documents": len(all_documents),
            "total_batches": total_batches,
            "total_success": total_success,
            "total_failure": total_failure,
            "duration_seconds": duration,
            "documents": all_documents,
            "output_path": str(raw_output_path)
        }


def main():
    parser = argparse.ArgumentParser(description="Extract knowledge from Dasan dialogues using Claude (v2)")
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
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Cache directory for resume capability"
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
    cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / "cache"

    # Initialize extractor
    extractor = ClaudeKnowledgeExtractorV2(
        model_name=model_name,
        request_delay=request_delay,
        batch_size=batch_size,
        max_retries=3,
        cache_dir=cache_dir
    )

    # Run extraction
    results = asyncio.run(extractor.extract_all(sampled_data['dialogues'], output_dir))

    # Save results summary
    summary_path = output_dir / "extraction_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump({
            "total_documents": results["total_documents"],
            "total_batches": results["total_batches"],
            "total_success": results["total_success"],
            "total_failure": results["total_failure"],
            "duration_seconds": results["duration_seconds"],
            "output_path": results["output_path"],
            "timestamp": datetime.now().isoformat()
        }, f, ensure_ascii=False, indent=2)

    logger.info(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
