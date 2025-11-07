#!/usr/bin/env python3
"""
Quality Validator
=================

Validate quality of extracted knowledge documents.

Usage:
    uv run python src/knowledge_extraction/quality_validator.py \
        --docs-dir data/processed/knowledge_docs \
        --report-path data/evaluation/knowledge_extraction_poc/quality_report.md
"""

import argparse
import yaml
from pathlib import Path
from typing import Dict, Any, List
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class QualityValidator:
    """
    Validate quality of knowledge documents.
    """

    def __init__(self, min_doc_size: int = 500):
        """
        Args:
            min_doc_size: Minimum acceptable document size in tokens
        """
        self.min_doc_size = min_doc_size
        self.stats = {
            "total_docs": 0,
            "avg_size": 0,
            "small_docs": [],
            "large_docs": [],
            "missing_metadata": [],
            "topic_distribution": defaultdict(int),
            "category_distribution": defaultdict(int),
            "size_distribution": [],
            "sample_docs": []
        }

    def estimate_token_count(self, text: str) -> int:
        """Estimate token count."""
        return len(text) // 4

    def parse_front_matter(self, content: str) -> Dict[str, Any]:
        """
        Parse YAML front matter from Markdown.

        Args:
            content: Markdown content

        Returns:
            Front matter dict
        """
        if not content.startswith('---'):
            return {}

        try:
            fm_end = content.find('---', 3)
            if fm_end == -1:
                return {}

            fm_text = content[3:fm_end].strip()
            return yaml.safe_load(fm_text) or {}
        except Exception as e:
            logger.warning(f"Failed to parse front matter: {e}")
            return {}

    def validate_document(self, file_path: Path) -> Dict[str, Any]:
        """
        Validate single document.

        Args:
            file_path: Path to Markdown file

        Returns:
            Validation result
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Parse front matter
        front_matter = self.parse_front_matter(content)

        # Estimate size
        token_count = self.estimate_token_count(content)

        # Check metadata
        required_fields = ['category', 'primary_topic']
        missing_fields = [field for field in required_fields if field not in front_matter]

        return {
            "path": str(file_path),
            "size": token_count,
            "front_matter": front_matter,
            "missing_fields": missing_fields,
            "is_small": token_count < self.min_doc_size,
            "is_large": token_count > 2000
        }

    def validate_all(self, docs_dir: Path) -> Dict[str, Any]:
        """
        Validate all documents in directory.

        Args:
            docs_dir: Directory containing Markdown files

        Returns:
            Validation statistics
        """
        logger.info(f"\n{'='*60}")
        logger.info("VALIDATING DOCUMENTS")
        logger.info(f"{'='*60}")
        logger.info(f"Directory: {docs_dir}")

        # Find all Markdown files
        md_files = list(docs_dir.rglob("*.md"))
        logger.info(f"Found {len(md_files)} Markdown files")

        # Validate each
        sizes = []
        for file_path in md_files:
            result = self.validate_document(file_path)

            self.stats["total_docs"] += 1
            sizes.append(result["size"])

            # Track small/large docs
            if result["is_small"]:
                self.stats["small_docs"].append({
                    "path": str(file_path.relative_to(docs_dir)),
                    "size": result["size"]
                })
            if result["is_large"]:
                self.stats["large_docs"].append({
                    "path": str(file_path.relative_to(docs_dir)),
                    "size": result["size"]
                })

            # Track missing metadata
            if result["missing_fields"]:
                self.stats["missing_metadata"].append({
                    "path": str(file_path.relative_to(docs_dir)),
                    "missing": result["missing_fields"]
                })

            # Track topic distribution
            fm = result["front_matter"]
            if "primary_topic" in fm:
                self.stats["topic_distribution"][fm["primary_topic"]] += 1
            if "category" in fm:
                self.stats["category_distribution"][fm["category"]] += 1

            # Sample docs (first 3)
            if len(self.stats["sample_docs"]) < 3:
                self.stats["sample_docs"].append({
                    "path": str(file_path.relative_to(docs_dir)),
                    "size": result["size"],
                    "primary_topic": fm.get("primary_topic", "N/A"),
                    "category": fm.get("category", "N/A")
                })

        # Calculate statistics
        self.stats["avg_size"] = sum(sizes) / len(sizes) if sizes else 0
        self.stats["min_size"] = min(sizes) if sizes else 0
        self.stats["max_size"] = max(sizes) if sizes else 0
        self.stats["size_distribution"] = self._calculate_size_distribution(sizes)

        logger.info(f"\n{'='*60}")
        logger.info("VALIDATION COMPLETE")
        logger.info(f"{'='*60}")
        logger.info(f"Total documents: {self.stats['total_docs']}")
        logger.info(f"Average size: {self.stats['avg_size']:.0f} tokens")
        logger.info(f"Size range: {self.stats['min_size']} - {self.stats['max_size']} tokens")
        logger.info(f"Small docs (<{self.min_doc_size}): {len(self.stats['small_docs'])}")
        logger.info(f"Large docs (>2000): {len(self.stats['large_docs'])}")
        logger.info(f"Missing metadata: {len(self.stats['missing_metadata'])}")

        return self.stats

    def _calculate_size_distribution(self, sizes: List[int]) -> List[Dict[str, Any]]:
        """Calculate size distribution buckets."""
        buckets = [
            (0, 250),
            (250, 500),
            (500, 1000),
            (1000, 1500),
            (1500, 2000),
            (2000, float('inf'))
        ]

        distribution = []
        for min_size, max_size in buckets:
            count = sum(1 for s in sizes if min_size <= s < max_size)
            label = f"{min_size}-{max_size}" if max_size != float('inf') else f"{min_size}+"
            distribution.append({
                "range": label,
                "count": count,
                "percentage": (count / len(sizes) * 100) if sizes else 0
            })

        return distribution

    def generate_report(self, output_path: Path):
        """
        Generate quality report in Markdown.

        Args:
            output_path: Path to output Markdown file
        """
        logger.info(f"\nGenerating quality report...")

        report = f"""# Knowledge Extraction Quality Report

**Generated**: {Path(__file__).parent}
**Total Documents**: {self.stats['total_docs']}

## Summary Statistics

- **Average Document Size**: {self.stats['avg_size']:.0f} tokens
- **Size Range**: {self.stats['min_size']} - {self.stats['max_size']} tokens
- **Documents < {self.min_doc_size} tokens**: {len(self.stats['small_docs'])} ({len(self.stats['small_docs']) / self.stats['total_docs'] * 100:.1f}%)
- **Documents > 2000 tokens**: {len(self.stats['large_docs'])} ({len(self.stats['large_docs']) / self.stats['total_docs'] * 100:.1f}%)
- **Missing Metadata**: {len(self.stats['missing_metadata'])} documents

## Size Distribution

| Range (tokens) | Count | Percentage |
|----------------|-------|------------|
"""
        for bucket in self.stats["size_distribution"]:
            report += f"| {bucket['range']} | {bucket['count']} | {bucket['percentage']:.1f}% |\n"

        report += f"""
## Category Distribution

| Category | Count |
|----------|-------|
"""
        for category, count in sorted(self.stats["category_distribution"].items()):
            report += f"| {category} | {count} |\n"

        report += f"""
## Topic Distribution (Top 10)

| Primary Topic | Count |
|---------------|-------|
"""
        sorted_topics = sorted(self.stats["topic_distribution"].items(), key=lambda x: x[1], reverse=True)
        for topic, count in sorted_topics[:10]:
            report += f"| {topic} | {count} |\n"

        # Small docs
        if self.stats["small_docs"]:
            report += f"""
## Small Documents (<{self.min_doc_size} tokens)

⚠️ **Warning**: {len(self.stats['small_docs'])} documents are below minimum size threshold.
These may need to be merged with related topics.

| Document | Size |
|----------|------|
"""
            for doc in sorted(self.stats["small_docs"], key=lambda x: x["size"])[:10]:
                report += f"| {doc['path']} | {doc['size']} |\n"

        # Missing metadata
        if self.stats["missing_metadata"]:
            report += f"""
## Missing Metadata

❌ **{len(self.stats['missing_metadata'])} documents** have incomplete metadata.

| Document | Missing Fields |
|----------|----------------|
"""
            for doc in self.stats["missing_metadata"][:10]:
                report += f"| {doc['path']} | {', '.join(doc['missing'])} |\n"

        # Sample docs
        report += """
## Sample Documents

"""
        for i, doc in enumerate(self.stats["sample_docs"], 1):
            report += f"""
### {i}. {doc['path']}
- **Size**: {doc['size']} tokens
- **Category**: {doc['category']}
- **Primary Topic**: {doc['primary_topic']}
"""

        # Save report
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info(f"✅ Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Validate knowledge document quality")
    parser.add_argument(
        "--docs-dir",
        type=str,
        required=True,
        help="Directory containing knowledge documents"
    )
    parser.add_argument(
        "--report-path",
        type=str,
        required=True,
        help="Path to output quality report (Markdown)"
    )
    parser.add_argument(
        "--min-doc-size",
        type=int,
        default=500,
        help="Minimum acceptable document size in tokens"
    )
    args = parser.parse_args()

    # Validate
    validator = QualityValidator(min_doc_size=args.min_doc_size)
    validator.validate_all(Path(args.docs_dir))

    # Generate report
    validator.generate_report(Path(args.report_path))


if __name__ == "__main__":
    main()
