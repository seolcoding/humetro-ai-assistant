#!/usr/bin/env python3
"""
Consolidate extracted knowledge documents into single JSONL.

This script processes individual *_extracted.json files (output from Gemini Batch API)
and consolidates them into a single JSONL file with enhanced metadata for RAG pipeline.

Usage:
    uv run python src/knowledge_extraction/consolidate_extractions.py \
        --input-dir data/AI_HUB_DASAN_QA \
        --output data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs.jsonl \
        --validate

Output:
    - knowledge_docs.jsonl: Consolidated documents (newline-delimited JSON)
    - knowledge_docs_metadata.json: Statistics and validation summary
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Tuple
from datetime import datetime
import argparse


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExtractionConsolidator:
    """Consolidate extracted knowledge documents into JSONL."""

    # Validation rules
    MIN_DOC_LENGTH = 500  # chars
    REQUIRED_FIELDS = [
        'dialogue_id', 'original_question', 'original_answer',
        'topic_path', 'primary_topic', 'secondary_topics', 'document'
    ]
    KOREAN_CHARS_PER_TOKEN = 4  # Estimation for Korean text
    CHUNK_SIZE_TOKENS = 512  # Standard chunk size for RAG

    def __init__(self, validate: bool = True):
        """
        Initialize consolidator.

        Args:
            validate: Whether to perform document validation
        """
        self.validate = validate
        self.stats = {
            "total_files": 0,
            "total_documents": 0,
            "validation_passed": 0,
            "validation_failed": 0,
            "skipped": 0,
            "total_chars": 0,
            "total_tokens": 0,
            "avg_doc_length_chars": 0,
            "avg_doc_length_tokens": 0,
            "categories": {},
            "primary_topics": {},
            "start_time": datetime.now().isoformat(),
            "end_time": None
        }

    def consolidate(
        self,
        input_dir: Path,
        output_path: Path
    ) -> Dict[str, Any]:
        """
        Consolidate all *_extracted.json into single JSONL.

        Args:
            input_dir: Directory containing *_extracted.json files
            output_path: Output JSONL file path

        Returns:
            Statistics about consolidation
        """
        logger.info(f"Starting consolidation from {input_dir}")

        # Find all extracted files
        extracted_files = sorted(input_dir.glob("*_extracted.json"))
        self.stats["total_files"] = len(extracted_files)

        if not extracted_files:
            logger.warning(f"No *_extracted.json files found in {input_dir}")
            return self.stats

        logger.info(f"Found {len(extracted_files)} extraction files")

        all_docs = []

        # Load and consolidate
        for file_path in extracted_files:
            logger.info(f"Processing {file_path.name}")

            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    docs = json.load(f)

                # Handle both single doc and array
                if not isinstance(docs, list):
                    docs = [docs]

                for doc in docs:
                    # Add enhanced metadata
                    doc['metadata'] = self._compute_metadata(doc)

                    # Validate if requested
                    if self.validate:
                        is_valid, issues = self._validate_document(doc)
                        doc['metadata']['validated'] = is_valid
                        doc['metadata']['validation_issues'] = issues

                        if is_valid:
                            self.stats['validation_passed'] += 1
                        else:
                            self.stats['validation_failed'] += 1
                            logger.warning(
                                f"Validation failed for {doc.get('dialogue_id', 'unknown')}: "
                                f"{', '.join(issues)}"
                            )
                    else:
                        doc['metadata']['validated'] = None
                        doc['metadata']['validation_issues'] = []

                    # Update statistics
                    self._update_stats(doc)

                    all_docs.append(doc)
                    self.stats['total_documents'] += 1

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                self.stats['skipped'] += 1

        # Calculate averages
        if self.stats['total_documents'] > 0:
            self.stats['avg_doc_length_chars'] = (
                self.stats['total_chars'] / self.stats['total_documents']
            )
            self.stats['avg_doc_length_tokens'] = (
                self.stats['total_tokens'] / self.stats['total_documents']
            )

        # Write consolidated JSONL
        logger.info(f"Writing {len(all_docs)} documents to {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for doc in all_docs:
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')

        # Write metadata
        self.stats['end_time'] = datetime.now().isoformat()
        metadata_path = output_path.with_suffix('.json').with_stem(
            f'{output_path.stem}_metadata'
        )

        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.stats, f, ensure_ascii=False, indent=2)

        logger.info(f"Metadata saved to {metadata_path}")

        # Log summary
        self._log_summary()

        return self.stats

    def _compute_metadata(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute RAG-ready metadata for document.

        Args:
            doc: Document dictionary

        Returns:
            Metadata dictionary
        """
        document_text = doc.get('document', '')

        # Extract front matter fields if present
        category = None
        entities_count = 0
        kb_tags_count = 0

        if document_text.startswith('---'):
            # Parse front matter (simple extraction)
            try:
                fm_end = document_text.find('---', 3)
                if fm_end > 0:
                    front_matter = document_text[3:fm_end]

                    # Extract category
                    for line in front_matter.split('\n'):
                        if line.startswith('category:'):
                            category = line.split(':', 1)[1].strip()
                        elif line.startswith('entities:'):
                            # Count entities in list format
                            entities_str = line.split(':', 1)[1].strip()
                            if entities_str.startswith('['):
                                entities_count = entities_str.count(',') + 1
                        elif line.startswith('kb_tags:'):
                            # Count kb_tags in list format
                            tags_str = line.split(':', 1)[1].strip()
                            if tags_str.startswith('['):
                                kb_tags_count = tags_str.count(',') + 1
            except Exception as e:
                logger.debug(f"Failed to parse front matter: {e}")

        doc_length_chars = len(document_text)
        doc_length_tokens = doc_length_chars // self.KOREAN_CHARS_PER_TOKEN
        estimated_chunks = max(1, doc_length_tokens // self.CHUNK_SIZE_TOKENS)

        return {
            # Document statistics
            "doc_length_chars": doc_length_chars,
            "doc_length_tokens": doc_length_tokens,
            "estimated_chunks": estimated_chunks,

            # Quality indicators
            "has_front_matter": document_text.startswith('---'),
            "has_structured_qa": '### Q:' in document_text and '**A**:' in document_text,
            "entities_count": entities_count,
            "kb_tags_count": kb_tags_count,

            # Extraction metadata
            "extraction_model": "gemini-2.5-pro",
            "extraction_date": datetime.now().strftime("%Y-%m-%d"),
            "source_dialogue_turns": doc.get('metadata', {}).get('source_dialogue_turns'),
            "compressed_turns": doc.get('metadata', {}).get('compressed_turns'),

            # Categorization
            "category": category or doc.get('topic_path', '').split('/')[0],
            "domain": doc.get('topic_path', '').split('/')[1] if '/' in doc.get('topic_path', '') else None,

            # Validation (filled in by validate method)
            "validated": False,
            "validation_issues": []
        }

    def _validate_document(self, doc: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate document quality.

        Args:
            doc: Document dictionary

        Returns:
            (is_valid, issues)
        """
        issues = []

        # Required fields
        for field in self.REQUIRED_FIELDS:
            if not doc.get(field):
                issues.append(f"Missing required field: {field}")

        # Document quality checks
        doc_text = doc.get('document', '')

        if len(doc_text) < self.MIN_DOC_LENGTH:
            issues.append(f"Document too short ({len(doc_text)} < {self.MIN_DOC_LENGTH} chars)")

        if not doc_text.startswith('---'):
            issues.append("Missing front matter (should start with '---')")

        if '### Q:' not in doc_text:
            issues.append("Missing structured Q&A format (no '### Q:' found)")

        # Topic validation
        if not doc.get('primary_topic'):
            issues.append("Missing primary_topic")

        if not doc.get('secondary_topics') or len(doc.get('secondary_topics', [])) == 0:
            issues.append("Missing or empty secondary_topics")

        return len(issues) == 0, issues

    def _update_stats(self, doc: Dict[str, Any]):
        """Update statistics with document data."""
        metadata = doc.get('metadata', {})

        # Accumulate lengths
        self.stats['total_chars'] += metadata.get('doc_length_chars', 0)
        self.stats['total_tokens'] += metadata.get('doc_length_tokens', 0)

        # Category distribution
        category = metadata.get('category', 'unknown')
        self.stats['categories'][category] = self.stats['categories'].get(category, 0) + 1

        # Primary topic distribution
        primary_topic = doc.get('primary_topic', 'unknown')
        self.stats['primary_topics'][primary_topic] = (
            self.stats['primary_topics'].get(primary_topic, 0) + 1
        )

    def _log_summary(self):
        """Log consolidation summary."""
        logger.info("=" * 60)
        logger.info("CONSOLIDATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total files processed: {self.stats['total_files']}")
        logger.info(f"Total documents: {self.stats['total_documents']}")

        if self.validate:
            logger.info(f"Validation passed: {self.stats['validation_passed']}")
            logger.info(f"Validation failed: {self.stats['validation_failed']}")
            pass_rate = (
                self.stats['validation_passed'] / self.stats['total_documents'] * 100
                if self.stats['total_documents'] > 0 else 0
            )
            logger.info(f"Pass rate: {pass_rate:.1f}%")

        logger.info(f"Skipped (errors): {self.stats['skipped']}")
        logger.info(f"Avg document length: {self.stats['avg_doc_length_chars']:.0f} chars, "
                   f"{self.stats['avg_doc_length_tokens']:.0f} tokens")

        logger.info("\nCategory distribution:")
        for category, count in sorted(self.stats['categories'].items()):
            logger.info(f"  {category}: {count}")

        logger.info("\nTop primary topics:")
        for topic, count in sorted(
            self.stats['primary_topics'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:5]:
            logger.info(f"  {topic}: {count}")

        logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Consolidate extracted knowledge documents into JSONL'
    )
    parser.add_argument(
        '--input-dir',
        type=Path,
        default=Path('data/AI_HUB_DASAN_QA'),
        help='Directory containing *_extracted.json files'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs.jsonl'),
        help='Output JSONL file path'
    )
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip document validation'
    )

    args = parser.parse_args()

    # Run consolidation
    consolidator = ExtractionConsolidator(validate=not args.no_validate)
    stats = consolidator.consolidate(args.input_dir, args.output)

    # Exit with error if validation failed for all documents
    if args.no_validate is False and stats['validation_passed'] == 0:
        logger.error("All documents failed validation!")
        return 1

    logger.info(f"✓ Consolidation complete: {args.output}")
    return 0


if __name__ == '__main__':
    exit(main())
