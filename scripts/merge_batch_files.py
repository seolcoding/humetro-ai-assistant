#!/usr/bin/env python3
"""
Merge Batch Files into Single State File
=========================================

Reads all batch_*.json files and merges them into a single state file
with ID-based dictionary structure.
"""

import json
from pathlib import Path
import argparse
from datetime import datetime
from collections import defaultdict


def merge_batch_files(batch_dir: Path, output_file: Path):
    """Merge all batch files into single state file."""

    print(f"{'='*40}")
    print(f"📦 MERGING BATCH FILES")
    print(f"{'='*40}")
    print(f"Batch dir: {batch_dir}")
    print(f"Output: {output_file}")
    print()

    # Find all batch files
    batch_files = sorted(batch_dir.glob("batch_*.json"))

    if not batch_files:
        print(f"❌ No batch files found in {batch_dir}")
        return 1

    print(f"📂 Found {len(batch_files)} batch files")

    # Group documents by dialogue_id
    grouped_docs = defaultdict(list)
    total_docs = 0

    for batch_file in batch_files:
        print(f"  📄 Processing: {batch_file.name}")

        try:
            with open(batch_file, 'r', encoding='utf-8') as f:
                documents = json.load(f)

            # Batch file is an array of documents
            for doc in documents:
                dialogue_id = doc.get("dialogue_id")
                if dialogue_id:
                    grouped_docs[dialogue_id].append(doc)
                    total_docs += 1
                else:
                    print(f"    ⚠️  Document without dialogue_id: {doc.keys()}")

        except Exception as e:
            print(f"    ❌ Error loading {batch_file.name}: {e}")
            continue

    print(f"\n📊 Statistics:")
    print(f"  Total documents: {total_docs}")
    print(f"  Unique dialogues: {len(grouped_docs)}")
    print(f"  Avg docs per dialogue: {total_docs / max(len(grouped_docs), 1):.2f}")

    # Load existing state if present
    state = {}
    if output_file.exists():
        with open(output_file, 'r', encoding='utf-8') as f:
            state = json.load(f)
        print(f"\n📂 Loaded existing state: {len(state)} entries")

    # Merge into state format
    merged_count = 0
    updated_count = 0

    for dialogue_id, docs in grouped_docs.items():
        if dialogue_id in state and state[dialogue_id].get("status") == "success":
            # Already processed - keep existing
            updated_count += 1
        else:
            # Add new entry
            state[dialogue_id] = {
                "status": "success",
                "dialogue_id": dialogue_id,
                "documents": docs,  # Keep original format
                "error": None,
                "timestamp": datetime.now().isoformat(),
                "retry_count": 0
            }
            merged_count += 1

    print(f"\n✅ Merge complete:")
    print(f"  New entries: {merged_count}")
    print(f"  Existing preserved: {updated_count}")
    print(f"  Total state entries: {len(state)}")

    # Save state file
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Saved to: {output_file}")
    print(f"📊 File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"{'='*40}\n")

    return 0


def main():
    parser = argparse.ArgumentParser(description="Merge batch files into state file")
    parser.add_argument(
        "--batch-dir",
        default="data/evaluation/knowledge_extraction_full",
        help="Directory containing batch_*.json files"
    )
    parser.add_argument(
        "--output",
        default="data/evaluation/knowledge_extraction_full/extracted_documents.json",
        help="Output state file path"
    )

    args = parser.parse_args()

    batch_dir = Path(args.batch_dir)
    output_file = Path(args.output)

    if not batch_dir.exists():
        print(f"❌ Batch directory not found: {batch_dir}")
        return 1

    return merge_batch_files(batch_dir, output_file)


if __name__ == "__main__":
    exit(main())
