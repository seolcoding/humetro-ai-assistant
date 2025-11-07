#!/usr/bin/env python3
"""
Markdown Deduplication Script

Detects and removes repeated patterns from crawled markdown files.
Optimized for web crawling data with template-based repetitions.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set, Tuple
import shutil
from datetime import datetime


class MarkdownDeduplicator:
    """Remove repeated patterns from markdown files"""

    def __init__(
        self,
        source_dir: Path,
        output_dir: Path,
        min_line_occurrences: int = 100,
        min_block_occurrences: int = 100,
        min_line_length: int = 20,
        block_size: int = 5
    ):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.min_line_occurrences = min_line_occurrences
        self.min_block_occurrences = min_block_occurrences
        self.min_line_length = min_line_length
        self.block_size = block_size

        self.stats = {
            "total_files": 0,
            "total_lines": 0,
            "repeated_lines": {},
            "repeated_blocks": {},
            "lines_removed": 0,
            "files_processed": 0
        }

    def find_repeated_lines(self, md_files: List[Path]) -> Dict[str, int]:
        """
        Find lines that appear in many files (likely template/navigation)

        Args:
            md_files: List of markdown file paths

        Returns:
            Dictionary of {line: occurrence_count}
        """
        print(f"\n🔍 Analyzing repeated lines across {len(md_files)} files...")
        line_counter = Counter()

        for file_path in md_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    unique_lines = set()  # Avoid counting duplicates within same file
                    for line in f:
                        cleaned = line.strip()
                        if len(cleaned) >= self.min_line_length:
                            unique_lines.add(cleaned)

                    line_counter.update(unique_lines)
            except Exception as e:
                print(f"⚠️  Error reading {file_path}: {e}")

        # Filter by minimum occurrences
        repeated = {
            line: count
            for line, count in line_counter.items()
            if count >= self.min_line_occurrences
        }

        print(f"✅ Found {len(repeated)} repeated line patterns")
        return repeated

    def find_repeated_blocks(self, md_files: List[Path]) -> Dict[Tuple[str, ...], int]:
        """
        Find consecutive line blocks that repeat across files

        Args:
            md_files: List of markdown file paths

        Returns:
            Dictionary of {(line1, line2, ...): occurrence_count}
        """
        print(f"\n🔍 Analyzing repeated {self.block_size}-line blocks...")
        block_counter = Counter()

        for file_path in md_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = [l.strip() for l in f if l.strip()]

                    # Sliding window to create blocks
                    for i in range(len(lines) - self.block_size + 1):
                        block = tuple(lines[i:i+self.block_size])
                        block_counter[block] += 1
            except Exception as e:
                print(f"⚠️  Error reading {file_path}: {e}")

        repeated = {
            block: count
            for block, count in block_counter.items()
            if count >= self.min_block_occurrences
        }

        print(f"✅ Found {len(repeated)} repeated block patterns")
        return repeated

    def create_removal_set(
        self,
        repeated_lines: Dict[str, int],
        repeated_blocks: Dict[Tuple[str, ...], int]
    ) -> Set[str]:
        """
        Create set of lines to remove

        Priority: blocks > individual lines to avoid partial removal
        """
        to_remove = set()

        # Add all lines from repeated blocks
        for block in repeated_blocks.keys():
            to_remove.update(block)

        # Add individual repeated lines
        to_remove.update(repeated_lines.keys())

        print(f"\n📋 Total unique lines to remove: {len(to_remove)}")
        return to_remove

    def deduplicate_file(
        self,
        source_file: Path,
        output_file: Path,
        lines_to_remove: Set[str]
    ) -> Tuple[int, int]:
        """
        Remove repeated lines from a single file

        Returns:
            (original_line_count, new_line_count)
        """
        original_lines = []
        cleaned_lines = []

        with open(source_file, 'r', encoding='utf-8') as f:
            original_lines = f.readlines()

        for line in original_lines:
            cleaned = line.strip()
            if cleaned and cleaned not in lines_to_remove:
                cleaned_lines.append(line)
            elif not cleaned:  # Keep empty lines for formatting
                cleaned_lines.append(line)

        # Write deduplicated content
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(cleaned_lines)

        return len(original_lines), len(cleaned_lines)

    def process_all_files(self):
        """Main processing pipeline"""
        print("=" * 60)
        print("🚀 Markdown Deduplication Pipeline")
        print("=" * 60)

        # Get all markdown files
        md_files = list(self.source_dir.glob("*.md"))
        self.stats["total_files"] = len(md_files)

        if not md_files:
            print(f"❌ No markdown files found in {self.source_dir}")
            return

        print(f"\n📂 Source: {self.source_dir}")
        print(f"📂 Output: {self.output_dir}")
        print(f"📄 Files: {len(md_files)}")

        # Step 1: Find repeated patterns
        repeated_lines = self.find_repeated_lines(md_files)
        repeated_blocks = self.find_repeated_blocks(md_files)

        self.stats["repeated_lines"] = {
            line: count for line, count in
            sorted(repeated_lines.items(), key=lambda x: x[1], reverse=True)[:20]
        }
        self.stats["repeated_blocks"] = len(repeated_blocks)

        # Step 2: Create removal set
        lines_to_remove = self.create_removal_set(repeated_lines, repeated_blocks)

        # Step 3: Process all files
        print(f"\n🔧 Processing files...")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for i, source_file in enumerate(md_files, 1):
            output_file = self.output_dir / source_file.name

            try:
                orig_count, new_count = self.deduplicate_file(
                    source_file,
                    output_file,
                    lines_to_remove
                )

                self.stats["total_lines"] += orig_count
                self.stats["lines_removed"] += (orig_count - new_count)
                self.stats["files_processed"] += 1

                if i % 100 == 0:
                    print(f"   Processed {i}/{len(md_files)} files...")

            except Exception as e:
                print(f"❌ Error processing {source_file.name}: {e}")

        # Save statistics
        self.save_statistics()

        # Print summary
        self.print_summary()

    def save_statistics(self):
        """Save deduplication statistics to JSON"""
        stats_file = self.output_dir / "deduplication_stats.json"

        stats_output = {
            "timestamp": datetime.now().isoformat(),
            "source_directory": str(self.source_dir),
            "output_directory": str(self.output_dir),
            "configuration": {
                "min_line_occurrences": self.min_line_occurrences,
                "min_block_occurrences": self.min_block_occurrences,
                "min_line_length": self.min_line_length,
                "block_size": self.block_size
            },
            "results": {
                "total_files": self.stats["total_files"],
                "files_processed": self.stats["files_processed"],
                "total_lines": self.stats["total_lines"],
                "lines_removed": self.stats["lines_removed"],
                "reduction_percentage": round(
                    (self.stats["lines_removed"] / self.stats["total_lines"] * 100), 2
                ) if self.stats["total_lines"] > 0 else 0,
                "unique_repeated_patterns": len(self.stats["repeated_lines"]),
                "repeated_blocks_found": self.stats["repeated_blocks"]
            },
            "top_repeated_lines": self.stats["repeated_lines"]
        }

        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_output, f, indent=2, ensure_ascii=False)

        print(f"\n💾 Statistics saved to: {stats_file}")

    def print_summary(self):
        """Print processing summary"""
        print("\n" + "=" * 60)
        print("📊 DEDUPLICATION SUMMARY")
        print("=" * 60)
        print(f"✅ Files processed: {self.stats['files_processed']}/{self.stats['total_files']}")
        print(f"📄 Total lines: {self.stats['total_lines']:,}")
        print(f"🗑️  Lines removed: {self.stats['lines_removed']:,}")

        if self.stats["total_lines"] > 0:
            reduction = (self.stats["lines_removed"] / self.stats["total_lines"]) * 100
            print(f"📉 Reduction: {reduction:.2f}%")

        print(f"🔍 Unique patterns found: {len(self.stats['repeated_lines'])}")
        print(f"📦 Block patterns found: {self.stats['repeated_blocks']}")

        print("\n🔝 Top 10 repeated lines:")
        for i, (line, count) in enumerate(list(self.stats["repeated_lines"].items())[:10], 1):
            preview = line[:60] + "..." if len(line) > 60 else line
            print(f"   {i:2d}. ({count:4d}x) {preview}")

        print("=" * 60)


def main():
    """Main entry point"""
    # Configuration
    source_dir = Path("/Users/sdh/Dev/02_production_projects/humetro-ai-assistant/data/crawled/seoul_traffic/markdown")
    output_dir = Path("/Users/sdh/Dev/02_production_projects/humetro-ai-assistant/data/crawled/seoul_traffic/markdown_deduplicated")

    # Create deduplicator instance
    deduplicator = MarkdownDeduplicator(
        source_dir=source_dir,
        output_dir=output_dir,
        min_line_occurrences=100,      # Lines appearing in 100+ files
        min_block_occurrences=100,      # Blocks appearing in 100+ files
        min_line_length=20,             # Ignore short lines (< 20 chars)
        block_size=5                    # Consecutive 5-line blocks
    )

    # Run deduplication
    deduplicator.process_all_files()


if __name__ == "__main__":
    main()
