#!/usr/bin/env python3
"""
Clean markdown files and generate statistics.

Applies social link removal patterns and generates:
- Text length statistics
- Token count statistics (using tiktoken for GPT-4)
"""

import re
import json
from pathlib import Path
from typing import List, Dict
import tiktoken
import statistics


class MarkdownCleaner:
    """Clean markdown files by removing social sharing links."""

    # Same patterns as Stage 1
    SOCIAL_LINK_PATTERNS = [
        r'\[인쇄\]\([^)]+\)',
        r'\[스크랩\]\([^)]+\)',
        r'\[\s*카카오톡\s*\]\([^)]+\)',
        r'\[\s*페이스북\s*\]\([^)]+\)',
        r'\[\s*X\s*\]\([^)]+\)',
        r'\[\s*네이버블로그\s*\]\([^)]+\)',
        r'\[\s*URL\s*복사\s*\]\([^)]+\)',
        r'\[\*\*네이버blog\*\*\]\([^)]+\)',
        r'\[\*\*메일전송\*\*\]\([^)]+\)',
        r'\[\*\*소스복사\*\*\]\([^)]+\)',
        r'\[\s*\*\*[^\]]+\*\*\s*\]\([^)]+(?:"새창열림"|"소스복사")[^)]*\)',
    ]

    def __init__(self):
        """Initialize cleaner with compiled patterns."""
        self.compiled_patterns = [
            re.compile(pattern, re.MULTILINE | re.UNICODE)
            for pattern in self.SOCIAL_LINK_PATTERNS
        ]

        # Initialize tiktoken encoder for GPT-4
        self.encoder = tiktoken.encoding_for_model("gpt-4")

    def clean_content(self, content: str) -> str:
        """
        Clean markdown content.

        Args:
            content: Raw markdown content

        Returns:
            Cleaned content
        """
        cleaned = content

        # Apply all regex patterns
        for pattern in self.compiled_patterns:
            cleaned = pattern.sub('', cleaned)

        # Remove excessive blank lines
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)

        # Strip leading/trailing whitespace
        cleaned = cleaned.strip()

        return cleaned

    def count_tokens(self, text: str) -> int:
        """
        Count tokens using tiktoken.

        Args:
            text: Text content

        Returns:
            Token count
        """
        return len(self.encoder.encode(text))

    def clean_file(self, file_path: Path) -> Dict:
        """
        Clean a single file and return statistics.

        Args:
            file_path: Path to markdown file

        Returns:
            Statistics dict
        """
        # Read original
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()

        # Clean
        cleaned_content = self.clean_content(original_content)

        # Write back
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)

        # Calculate stats
        original_length = len(original_content)
        cleaned_length = len(cleaned_content)
        cleaned_tokens = self.count_tokens(cleaned_content)

        return {
            "file": file_path.name,
            "original_length": original_length,
            "cleaned_length": cleaned_length,
            "cleaned_tokens": cleaned_tokens,
            "reduction_pct": ((original_length - cleaned_length) / original_length * 100) if original_length > 0 else 0
        }


def calculate_statistics(values: List[float]) -> Dict:
    """
    Calculate statistical measures.

    Args:
        values: List of numeric values

    Returns:
        Dict with max, median, avg, IQR
    """
    if not values:
        return {
            "max": 0,
            "median": 0,
            "avg": 0,
            "q1": 0,
            "q3": 0,
            "iqr": 0
        }

    sorted_values = sorted(values)
    q1 = statistics.quantiles(sorted_values, n=4)[0]
    q3 = statistics.quantiles(sorted_values, n=4)[2]

    return {
        "max": max(values),
        "median": statistics.median(values),
        "avg": statistics.mean(values),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1
    }


def main():
    """Main execution."""
    markdown_dir = Path("./data/crawled/seoul_traffic/markdown_deduplicated")

    if not markdown_dir.exists():
        print(f"❌ Directory not found: {markdown_dir}")
        return

    # Get all markdown files
    md_files = list(markdown_dir.glob("*.md"))

    if not md_files:
        print(f"❌ No markdown files found in {markdown_dir}")
        return

    print("=" * 60)
    print("🧹 Markdown Cleaning & Analysis")
    print("=" * 60)
    print(f"\n📂 Directory: {markdown_dir}")
    print(f"📄 Files: {len(md_files)}")

    # Clean all files
    cleaner = MarkdownCleaner()
    file_stats = []

    print(f"\n🔧 Processing files...")
    for i, file_path in enumerate(md_files, 1):
        try:
            stats = cleaner.clean_file(file_path)
            file_stats.append(stats)

            if i % 100 == 0:
                print(f"   Processed {i}/{len(md_files)} files...")

        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")

    print(f"✅ Processed {len(file_stats)}/{len(md_files)} files")

    # Calculate aggregate statistics
    text_lengths = [s["cleaned_length"] for s in file_stats]
    token_counts = [s["cleaned_tokens"] for s in file_stats]
    reduction_pcts = [s["reduction_pct"] for s in file_stats]

    text_stats = calculate_statistics(text_lengths)
    token_stats = calculate_statistics(token_counts)

    # Print report
    print("\n" + "=" * 60)
    print("📊 STATISTICS REPORT")
    print("=" * 60)

    print("\n📝 Text Length (characters):")
    print(f"   Max:     {text_stats['max']:,}")
    print(f"   Median:  {text_stats['median']:,.0f}")
    print(f"   Average: {text_stats['avg']:,.0f}")
    print(f"   Q1:      {text_stats['q1']:,.0f}")
    print(f"   Q3:      {text_stats['q3']:,.0f}")
    print(f"   IQR:     {text_stats['iqr']:,.0f}")

    print("\n🔢 Token Count (GPT-4 encoding):")
    print(f"   Max:     {token_stats['max']:,}")
    print(f"   Median:  {token_stats['median']:,.0f}")
    print(f"   Average: {token_stats['avg']:,.0f}")
    print(f"   Q1:      {token_stats['q1']:,.0f}")
    print(f"   Q3:      {token_stats['q3']:,.0f}")
    print(f"   IQR:     {token_stats['iqr']:,.0f}")

    print("\n📉 Content Reduction:")
    avg_reduction = statistics.mean(reduction_pcts)
    print(f"   Average: {avg_reduction:.2f}%")

    print("\n💾 Total Statistics:")
    total_chars = sum(text_lengths)
    total_tokens = sum(token_counts)
    print(f"   Total characters: {total_chars:,}")
    print(f"   Total tokens:     {total_tokens:,}")

    # Save detailed report
    report_file = markdown_dir / "cleaning_analysis_report.json"
    report_data = {
        "summary": {
            "total_files": len(file_stats),
            "total_characters": total_chars,
            "total_tokens": total_tokens,
            "avg_reduction_pct": avg_reduction
        },
        "text_length_stats": text_stats,
        "token_count_stats": token_stats,
        "top_10_longest": sorted(file_stats, key=lambda x: x["cleaned_length"], reverse=True)[:10],
        "top_10_shortest": sorted(file_stats, key=lambda x: x["cleaned_length"])[:10]
    }

    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Detailed report saved: {report_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
