#!/usr/bin/env python3
"""
Process Gemini Batch API predictions into knowledge base JSONL.

This script processes the raw predictions JSONL file from Gemini Batch API
and extracts knowledge documents with enhanced metadata for RAG pipeline.

Usage:
    uv run python src/knowledge_extraction/process_predictions.py \
        --input data/AI_HUB_DASAN_QA/06_predictions/prediction-model-*_predictions.jsonl \
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


class PredictionProcessor:
    """Process Gemini Batch API predictions into knowledge documents."""

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
        Initialize processor.

        Args:
            validate: Whether to perform document validation
        """
        self.validate = validate
        self.stats = {
            "total_lines": 0,
            "total_documents": 0,
            "validation_passed": 0,
            "validation_failed": 0,
            "skipped_empty_response": 0,
            "skipped_parse_error": 0,
            "skipped_other": 0,
            "total_chars": 0,
            "total_tokens": 0,
            "avg_doc_length_chars": 0,
            "avg_doc_length_tokens": 0,
            "categories": {},
            "primary_topics": {},
            "start_time": datetime.now().isoformat(),
            "end_time": None
        }

    def process(
        self,
        input_path: Path,
        output_path: Path
    ) -> Dict[str, Any]:
        """
        Process predictions JSONL into knowledge base JSONL.

        Args:
            input_path: Input predictions JSONL file (from Gemini Batch API)
            output_path: Output knowledge base JSONL file

        Returns:
            Statistics about processing
        """
        logger.info(f"Processing predictions from {input_path}")
        logger.info(f"Output: {output_path}")

        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        all_docs = []

        # Process JSONL line by line
        with open(input_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                self.stats['total_lines'] += 1

                if line_num % 1000 == 0:
                    logger.info(f"Processed {line_num:,} lines, "
                              f"extracted {self.stats['total_documents']:,} documents...")

                try:
                    # Parse prediction record
                    record = json.loads(line)

                    # Extract generated knowledge documents from response
                    candidates = record.get('response', {}).get('candidates', [])
                    if not candidates:
                        self.stats['skipped_empty_response'] += 1
                        continue

                    # Get text content (JSON array of documents)
                    parts = candidates[0].get('content', {}).get('parts', [])
                    if not parts:
                        self.stats['skipped_empty_response'] += 1
                        continue

                    text_content = parts[0].get('text', '')
                    if not text_content:
                        self.stats['skipped_empty_response'] += 1
                        continue

                    # Parse JSON array from text
                    docs = json.loads(text_content)

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
                                # Only log first few failures in detail
                                if self.stats['validation_failed'] <= 10:
                                    logger.warning(
                                        f"Line {line_num}, {doc.get('dialogue_id', 'unknown')}: "
                                        f"{', '.join(issues)}"
                                    )
                        else:
                            doc['metadata']['validated'] = None
                            doc['metadata']['validation_issues'] = []

                        # Update statistics
                        self._update_stats(doc)

                        all_docs.append(doc)
                        self.stats['total_documents'] += 1

                except json.JSONDecodeError as e:
                    self.stats['skipped_parse_error'] += 1
                    if self.stats['skipped_parse_error'] <= 5:
                        logger.warning(f"Line {line_num}: JSON parse error - {str(e)[:100]}")
                except Exception as e:
                    self.stats['skipped_other'] += 1
                    if self.stats['skipped_other'] <= 5:
                        logger.warning(f"Line {line_num}: {str(e)[:100]}")

        logger.info(f"Finished processing {self.stats['total_lines']:,} lines")

        # Calculate averages
        if self.stats['total_documents'] > 0:
            self.stats['avg_doc_length_chars'] = (
                self.stats['total_chars'] / self.stats['total_documents']
            )
            self.stats['avg_doc_length_tokens'] = (
                self.stats['total_tokens'] / self.stats['total_documents']
            )

        # Write consolidated JSONL
        logger.info(f"Writing {len(all_docs):,} documents to {output_path}")
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

                    # Extract fields
                    for line in front_matter.split('\n'):
                        if line.startswith('category:'):
                            category = line.split(':', 1)[1].strip()
                        elif line.startswith('entities:'):
                            entities_str = line.split(':', 1)[1].strip()
                            if entities_str.startswith('['):
                                entities_count = entities_str.count(',') + 1
                        elif line.startswith('kb_tags:'):
                            tags_str = line.split(':', 1)[1].strip()
                            if tags_str.startswith('['):
                                kb_tags_count = tags_str.count(',') + 1
            except Exception:
                pass

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
                issues.append(f"missing_{field}")

        # Document quality checks
        doc_text = doc.get('document', '')

        if len(doc_text) < self.MIN_DOC_LENGTH:
            issues.append(f"too_short_{len(doc_text)}chars")

        if not doc_text.startswith('---'):
            issues.append("no_frontmatter")

        if '### Q:' not in doc_text:
            issues.append("no_qa_format")

        # Topic validation
        if not doc.get('primary_topic'):
            issues.append("no_primary_topic")

        if not doc.get('secondary_topics') or len(doc.get('secondary_topics', [])) == 0:
            issues.append("no_secondary_topics")

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
        """Log processing summary."""
        logger.info("=" * 70)
        logger.info("PROCESSING SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total lines processed: {self.stats['total_lines']:,}")
        logger.info(f"Total documents extracted: {self.stats['total_documents']:,}")
        logger.info(f"")
        logger.info(f"Skipped:")
        logger.info(f"  Empty responses: {self.stats['skipped_empty_response']:,}")
        logger.info(f"  Parse errors: {self.stats['skipped_parse_error']:,}")
        logger.info(f"  Other errors: {self.stats['skipped_other']:,}")
        logger.info(f"")

        if self.validate:
            logger.info(f"Validation:")
            logger.info(f"  Passed: {self.stats['validation_passed']:,}")
            logger.info(f"  Failed: {self.stats['validation_failed']:,}")
            pass_rate = (
                self.stats['validation_passed'] / self.stats['total_documents'] * 100
                if self.stats['total_documents'] > 0 else 0
            )
            logger.info(f"  Pass rate: {pass_rate:.1f}%")
            logger.info(f"")

        logger.info(f"Average document:")
        logger.info(f"  {self.stats['avg_doc_length_chars']:.0f} chars")
        logger.info(f"  {self.stats['avg_doc_length_tokens']:.0f} tokens")
        logger.info(f"  ~{self.stats['avg_doc_length_tokens'] / self.CHUNK_SIZE_TOKENS:.1f} chunks")
        logger.info(f"")

        logger.info("Category distribution (top 10):")
        for category, count in sorted(
            self.stats['categories'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]:
            pct = count / self.stats['total_documents'] * 100
            logger.info(f"  {category}: {count:,} ({pct:.1f}%)")

        logger.info("=" * 70)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Process Gemini Batch API predictions into knowledge base'
    )
    parser.add_argument(
        '--input',
        type=Path,
        required=True,
        help='Input predictions JSONL file from Gemini Batch API'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs.jsonl'),
        help='Output knowledge base JSONL file'
    )
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip document validation'
    )

    args = parser.parse_args()

    # Run processing
    processor = PredictionProcessor(validate=not args.no_validate)
    stats = processor.process(args.input, args.output)

    # Exit with warning if validation failed for most documents
    if not args.no_validate:
        fail_rate = stats['validation_failed'] / stats['total_documents'] * 100 if stats['total_documents'] > 0 else 0
        if fail_rate > 50:
            logger.warning(f"High validation failure rate: {fail_rate:.1f}%")
            return 1

    logger.info(f"✓ Processing complete: {args.output}")
    logger.info(f"✓ Extracted {stats['total_documents']:,} documents from {stats['total_lines']:,} predictions")
    return 0


if __name__ == '__main__':
    exit(main())
