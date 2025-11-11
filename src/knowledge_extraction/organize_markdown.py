#!/usr/bin/env python3
"""
Organize knowledge documents into topic-based Markdown directory structure.

This script reads the consolidated JSONL file and creates a topic-based
directory structure with individual Markdown files for human review and
content management.

Usage:
    uv run python src/knowledge_extraction/organize_markdown.py \
        --input data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs.jsonl \
        --output-dir data/AI_HUB_DASAN_QA/02_markdown_basic

Output:
    Hierarchical directory structure:
    organized_markdown/
    ├── 코로나19_관련_상담/
    │   └── 금융지원/
    │       └── 소상공인_대출_B24034.md
    ├── 일반행정_문의/
    │   └── 동물보호/
    │       └── 길고양이_중성화_사업_B22203.md
    └── ...
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import argparse
import re


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MarkdownOrganizer:
    """Organize knowledge documents into topic-based directory structure."""

    def __init__(self):
        """Initialize organizer."""
        self.stats = {
            "total_documents": 0,
            "total_files_created": 0,
            "directories_created": 0,
            "skipped": 0,
            "topic_distribution": {}
        }

    def organize(
        self,
        jsonl_path: Path,
        output_dir: Path,
        create_index: bool = True
    ) -> Dict[str, Any]:
        """
        Organize JSONL documents into topic-based Markdown files.

        Args:
            jsonl_path: Input JSONL file path
            output_dir: Output directory for organized Markdown
            create_index: Whether to create index.md files in each directory

        Returns:
            Statistics about organization
        """
        logger.info(f"Organizing documents from {jsonl_path}")
        logger.info(f"Output directory: {output_dir}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Track directories and their documents
        dir_documents: Dict[Path, List[Dict[str, Any]]] = {}

        # Process JSONL
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    doc = json.loads(line)
                    self.stats['total_documents'] += 1

                    # Create directory structure from topic_path
                    doc_dir = self._create_directory_structure(
                        doc, output_dir
                    )

                    # Generate filename
                    filename = self._generate_filename(doc)
                    file_path = doc_dir / filename

                    # Write Markdown file
                    self._write_markdown_file(doc, file_path)

                    self.stats['total_files_created'] += 1

                    # Track for index creation
                    if create_index:
                        if doc_dir not in dir_documents:
                            dir_documents[doc_dir] = []
                        dir_documents[doc_dir].append(doc)

                    # Update topic distribution
                    primary_topic = doc.get('primary_topic', 'unknown')
                    self.stats['topic_distribution'][primary_topic] = (
                        self.stats['topic_distribution'].get(primary_topic, 0) + 1
                    )

                except Exception as e:
                    logger.error(f"Failed to process line {line_num}: {e}")
                    self.stats['skipped'] += 1

        # Create index files
        if create_index:
            logger.info("Creating index files...")
            self._create_index_files(dir_documents)

        # Create root README
        self._create_root_readme(output_dir)

        # Log summary
        self._log_summary()

        return self.stats

    def _create_directory_structure(
        self,
        doc: Dict[str, Any],
        output_dir: Path
    ) -> Path:
        """
        Create directory structure from topic_path.

        Args:
            doc: Document dictionary
            output_dir: Base output directory

        Returns:
            Path to document's directory
        """
        topic_path = doc.get('topic_path', '')

        if not topic_path:
            # Fallback to primary_topic
            topic_path = doc.get('primary_topic', 'uncategorized')

        # Split path and create directories
        # Example: "코로나19_관련_상담/금융지원/소상공인_대출"
        parts = topic_path.split('/')

        # Remove last part if it's too specific (use primary_topic instead)
        if len(parts) > 2:
            parts = parts[:2]  # Keep only category and domain

        doc_dir = output_dir
        for part in parts:
            # Sanitize directory name
            sanitized = self._sanitize_filename(part)
            doc_dir = doc_dir / sanitized

        # Create directory if not exists
        if not doc_dir.exists():
            doc_dir.mkdir(parents=True, exist_ok=True)
            self.stats['directories_created'] += 1
            logger.debug(f"Created directory: {doc_dir}")

        return doc_dir

    def _generate_filename(self, doc: Dict[str, Any]) -> str:
        """
        Generate filename for document.

        Args:
            doc: Document dictionary

        Returns:
            Filename (e.g., "소상공인_대출_B24034.md")
        """
        primary_topic = doc.get('primary_topic', 'unknown')
        dialogue_id = doc.get('dialogue_id', 'unknown')

        # Sanitize topic name
        sanitized_topic = self._sanitize_filename(primary_topic)

        return f"{sanitized_topic}_{dialogue_id}.md"

    def _sanitize_filename(self, name: str) -> str:
        """
        Sanitize filename/directory name.

        Args:
            name: Original name

        Returns:
            Sanitized name
        """
        # Remove invalid characters for filesystems
        # Keep Korean characters, alphanumeric, underscore, hyphen
        sanitized = re.sub(r'[<>:"/\\|?*]', '', name)

        # Replace spaces with underscores
        sanitized = sanitized.replace(' ', '_')

        # Trim dots and spaces from ends
        sanitized = sanitized.strip('. ')

        return sanitized or 'unnamed'

    def _write_markdown_file(self, doc: Dict[str, Any], file_path: Path):
        """
        Write document to Markdown file.

        Args:
            doc: Document dictionary
            file_path: Output file path
        """
        # Extract Markdown content
        markdown_content = doc.get('document', '')

        # Add file header with metadata
        header = self._generate_file_header(doc)

        # Combine header and content
        full_content = f"{header}\n\n{markdown_content}"

        # Write to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(full_content)

        logger.debug(f"Created: {file_path}")

    def _generate_file_header(self, doc: Dict[str, Any]) -> str:
        """
        Generate file header with metadata.

        Args:
            doc: Document dictionary

        Returns:
            Header markdown
        """
        dialogue_id = doc.get('dialogue_id', 'unknown')
        metadata = doc.get('metadata', {})

        header_lines = [
            "<!-- Generated by organize_markdown.py -->",
            f"<!-- Dialogue ID: {dialogue_id} -->",
            f"<!-- Extraction Date: {metadata.get('extraction_date', 'unknown')} -->",
            f"<!-- Validation: {'✓ Passed' if metadata.get('validated') else '✗ Failed'} -->",
        ]

        return '\n'.join(header_lines)

    def _create_index_files(self, dir_documents: Dict[Path, List[Dict[str, Any]]]):
        """
        Create index.md files in each directory.

        Args:
            dir_documents: Mapping of directory -> list of documents
        """
        for dir_path, docs in dir_documents.items():
            index_path = dir_path / 'index.md'

            # Get directory name
            dir_name = dir_path.name

            # Build index content
            lines = [
                f"# {dir_name}",
                "",
                f"이 디렉터리는 **{dir_name}** 주제와 관련된 지식 문서를 포함합니다.",
                "",
                f"총 {len(docs)}개 문서",
                "",
                "## 문서 목록",
                ""
            ]

            # Sort documents by primary_topic
            sorted_docs = sorted(docs, key=lambda d: d.get('primary_topic', ''))

            for doc in sorted_docs:
                primary_topic = doc.get('primary_topic', 'unknown')
                dialogue_id = doc.get('dialogue_id', 'unknown')
                original_q = doc.get('original_question', '')

                # Truncate question if too long
                if len(original_q) > 80:
                    original_q = original_q[:77] + '...'

                filename = self._generate_filename(doc)

                lines.append(f"- [{primary_topic}](./{filename})")
                lines.append(f"  - ID: `{dialogue_id}`")
                lines.append(f"  - 질문: {original_q}")
                lines.append("")

            # Write index file
            with open(index_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))

            logger.debug(f"Created index: {index_path}")

    def _create_root_readme(self, output_dir: Path):
        """
        Create root README.md with overview.

        Args:
            output_dir: Output directory
        """
        readme_path = output_dir / 'README.md'

        # Get all subdirectories
        subdirs = sorted([d for d in output_dir.iterdir() if d.is_dir()])

        lines = [
            "# 다산콜센터 지식베이스 (주제별 구조)",
            "",
            "이 디렉터리는 다산콜센터 상담 데이터에서 추출한 지식 문서를 주제별로 구조화한 것입니다.",
            "",
            f"- **총 문서 수**: {self.stats['total_documents']}",
            f"- **카테고리 수**: {len(subdirs)}",
            f"- **생성 일시**: {self._get_current_timestamp()}",
            "",
            "## 디렉터리 구조",
            "",
            "```",
            "organized_markdown/",
        ]

        # Add directory tree
        for subdir in subdirs:
            subdir_name = subdir.name
            doc_count = len(list(subdir.glob('*.md'))) - 1  # Exclude index.md

            lines.append(f"├── {subdir_name}/  ({doc_count} documents)")

            # Add subdirectories if any
            sub_subdirs = sorted([d for d in subdir.iterdir() if d.is_dir()])
            for sub_subdir in sub_subdirs:
                sub_doc_count = len(list(sub_subdir.glob('*.md'))) - 1
                lines.append(f"│   └── {sub_subdir.name}/  ({sub_doc_count} documents)")

        lines.append("```")
        lines.append("")
        lines.append("## 카테고리별 문서 분포")
        lines.append("")

        # Add topic distribution
        for topic, count in sorted(
            self.stats['topic_distribution'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            lines.append(f"- **{topic}**: {count}개")

        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append("자동 생성: `organize_markdown.py`")

        # Write README
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        logger.info(f"Created README: {readme_path}")

    def _get_current_timestamp(self) -> str:
        """Get current timestamp in ISO format."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _log_summary(self):
        """Log organization summary."""
        logger.info("=" * 60)
        logger.info("ORGANIZATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total documents processed: {self.stats['total_documents']}")
        logger.info(f"Files created: {self.stats['total_files_created']}")
        logger.info(f"Directories created: {self.stats['directories_created']}")
        logger.info(f"Skipped (errors): {self.stats['skipped']}")
        logger.info("")
        logger.info("Topic distribution:")
        for topic, count in sorted(
            self.stats['topic_distribution'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]:
            logger.info(f"  {topic}: {count}")
        logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Organize knowledge documents into topic-based Markdown structure'
    )
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs.jsonl'),
        help='Input JSONL file'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/AI_HUB_DASAN_QA/02_markdown_basic'),
        help='Output directory for organized Markdown'
    )
    parser.add_argument(
        '--no-index',
        action='store_true',
        help='Skip creating index.md files'
    )

    args = parser.parse_args()

    # Run organization
    organizer = MarkdownOrganizer()
    stats = organizer.organize(
        args.input,
        args.output_dir,
        create_index=not args.no_index
    )

    logger.info(f"✓ Organization complete: {args.output_dir}")
    return 0


if __name__ == '__main__':
    exit(main())
