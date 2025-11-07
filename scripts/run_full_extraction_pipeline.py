#!/usr/bin/env python3
"""
Full Knowledge Extraction Pipeline
====================================

1. Extract knowledge from dialogues (Claude API)
2. Organize documents (Markdown files)
3. Validate quality

Usage:
    python scripts/run_full_extraction_pipeline.py --config config/knowledge_extraction_test_30.json
"""

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*40}")
    print(f"🚀 {description}")
    print(f"{'='*40}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, capture_output=False)

    if result.returncode == 0:
        print(f"\n✅ {description} - SUCCESS")
        return True
    else:
        print(f"\n❌ {description} - FAILED (exit code: {result.returncode})")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run full knowledge extraction pipeline")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to extraction config file"
    )
    parser.add_argument(
        "--input",
        help="Path to input dialogues (optional, read from config if not specified)"
    )
    parser.add_argument(
        "--skip-extraction",
        action="store_true",
        help="Skip extraction step (use existing extracted_documents.json)"
    )
    parser.add_argument(
        "--skip-organization",
        action="store_true",
        help="Skip organization step (use existing markdown files)"
    )

    args = parser.parse_args()

    # Load config to get paths
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Get paths from config
    output_dir = Path(config['experiment']['output_dir'])
    input_dialogues = args.input

    if not input_dialogues and not args.skip_extraction:
        print("❌ Input dialogues path not specified (use --input or set in config)")
        sys.exit(1)

    extracted_docs_path = output_dir / "extracted_documents.json"
    organized_docs_dir = output_dir / "organized_docs"
    quality_report_path = output_dir / "quality_report.md"

    print(f"\n{'='*40}")
    print(f"📋 PIPELINE CONFIGURATION")
    print(f"{'='*40}")
    print(f"Config: {config_path}")
    print(f"Input: {input_dialogues}")
    print(f"Output dir: {output_dir}")
    print(f"Extracted docs: {extracted_docs_path}")
    print(f"Organized docs: {organized_docs_dir}")
    print(f"Quality report: {quality_report_path}")
    print(f"{'='*40}\n")

    # Step 1: Extract knowledge
    if not args.skip_extraction:
        cmd = [
            "uv", "run", "python",
            "src/knowledge_extraction/claude_knowledge_extractor_v2.py",
            "--config", str(config_path),
            "--input", str(input_dialogues)
        ]

        if not run_command(cmd, "Step 1: Extract Knowledge"):
            print("\n❌ Pipeline failed at Step 1")
            sys.exit(1)

        if not extracted_docs_path.exists():
            print(f"\n❌ Expected output not found: {extracted_docs_path}")
            sys.exit(1)
    else:
        print(f"\n⏭️  Skipping extraction step")
        if not extracted_docs_path.exists():
            print(f"❌ Extracted documents not found: {extracted_docs_path}")
            sys.exit(1)

    # Step 2: Organize documents
    if not args.skip_organization:
        cmd = [
            "uv", "run", "python",
            "src/knowledge_extraction/document_organizer.py",
            "--input", str(extracted_docs_path),
            "--output-dir", str(organized_docs_dir),
            "--min-doc-size", "500",
            "--merge-small"
        ]

        if not run_command(cmd, "Step 2: Organize Documents"):
            print("\n❌ Pipeline failed at Step 2")
            sys.exit(1)

        if not organized_docs_dir.exists():
            print(f"\n❌ Expected output not found: {organized_docs_dir}")
            sys.exit(1)
    else:
        print(f"\n⏭️  Skipping organization step")
        if not organized_docs_dir.exists():
            print(f"❌ Organized documents not found: {organized_docs_dir}")
            sys.exit(1)

    # Step 3: Validate quality
    cmd = [
        "uv", "run", "python",
        "src/knowledge_extraction/quality_validator.py",
        "--docs-dir", str(organized_docs_dir),
        "--report-path", str(quality_report_path),
        "--min-doc-size", "500"
    ]

    if not run_command(cmd, "Step 3: Validate Quality"):
        print("\n❌ Pipeline failed at Step 3")
        sys.exit(1)

    if not quality_report_path.exists():
        print(f"\n❌ Expected output not found: {quality_report_path}")
        sys.exit(1)

    # Success summary
    print(f"\n{'='*40}")
    print(f"🎉 PIPELINE COMPLETE")
    print(f"{'='*40}")
    print(f"✅ Extracted documents: {extracted_docs_path}")
    print(f"✅ Organized documents: {organized_docs_dir}")
    print(f"✅ Quality report: {quality_report_path}")
    print(f"{'='*40}\n")

    # Show quality report summary
    print(f"📊 Quality Report Preview:\n")
    with open(quality_report_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        # Print first 30 lines
        for line in lines[:30]:
            print(line.rstrip())

    if len(lines) > 30:
        print(f"\n... ({len(lines) - 30} more lines)")

    print(f"\n💾 Full report: {quality_report_path}")


if __name__ == "__main__":
    main()
