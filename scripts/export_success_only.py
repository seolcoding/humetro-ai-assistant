#!/usr/bin/env python3
"""
Export Successful Documents from State File
============================================

Extract only successful documents from state file and save as flat array.
This is the final output format for downstream processing.

State Format (Input):
    {
      "dialogue_id": {
        "status": "success"|"fail"|"pending",
        "documents": [...],
        ...
      }
    }

Output Format:
    [
      {
        "dialogue_id": "...",
        "original_question": "...",
        "original_answer": "...",
        "topic_path": "...",
        "primary_topic": "...",
        "secondary_topics": [...],
        "document": "..."
      }
    ]
"""

import json
from pathlib import Path
import argparse
from datetime import datetime


def export_success_only(state_file: Path, output_file: Path, include_metadata: bool = True):
    """Export only successful documents from state file."""

    print(f"{'='*40}")
    print(f"📤 EXPORTING SUCCESSFUL DOCUMENTS")
    print(f"{'='*40}")
    print(f"State: {state_file}")
    print(f"Output: {output_file}")
    print()

    # Load state
    if not state_file.exists():
        print(f"❌ State file not found: {state_file}")
        return 1

    with open(state_file, 'r', encoding='utf-8') as f:
        state = json.load(f)

    print(f"📂 Loaded state: {len(state)} entries")

    # Count by status
    status_counts = {
        "success": 0,
        "fail": 0,
        "pending": 0,
        "auth_error": 0,
        "other": 0
    }

    all_documents = []
    successful_dialogues = []

    for dialogue_id, entry in state.items():
        status = entry.get("status", "unknown")

        if status == "success":
            status_counts["success"] += 1
            documents = entry.get("documents", [])
            all_documents.extend(documents)
            successful_dialogues.append(dialogue_id)
        elif status == "fail":
            status_counts["fail"] += 1
        elif status == "pending":
            status_counts["pending"] += 1
        elif status == "auth_error":
            status_counts["auth_error"] += 1
        else:
            status_counts["other"] += 1

    print(f"\n📊 State Statistics:")
    print(f"  ✅ Success: {status_counts['success']}")
    print(f"  ❌ Failed: {status_counts['fail']}")
    print(f"  ⏳ Pending: {status_counts['pending']}")
    print(f"  🔐 Auth Error: {status_counts['auth_error']}")
    if status_counts['other'] > 0:
        print(f"  ❓ Other: {status_counts['other']}")

    print(f"\n📝 Exported Documents: {len(all_documents)}")
    print(f"📋 Successful Dialogues: {len(successful_dialogues)}")

    # Prepare output
    if include_metadata:
        output_data = {
            "metadata": {
                "description": "Extracted knowledge documents from Dasan call center dialogues",
                "source": "Bottom-up knowledge extraction (Claude API)",
                "exported_at": datetime.now().isoformat(),
                "state_file": str(state_file),
                "statistics": {
                    "total_dialogues": len(state),
                    "successful_dialogues": status_counts["success"],
                    "failed_dialogues": status_counts["fail"],
                    "pending_dialogues": status_counts["pending"],
                    "total_documents": len(all_documents),
                    "documents_per_dialogue": round(len(all_documents) / max(status_counts["success"], 1), 2)
                }
            },
            "documents": all_documents
        }
    else:
        output_data = all_documents

    # Save output
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Export complete!")
    print(f"💾 Saved to: {output_file}")
    print(f"📊 File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"{'='*40}\n")

    return 0


def verify_export(output_file: Path):
    """Verify exported file integrity."""
    if not output_file.exists():
        print(f"❌ Output file not found: {output_file}")
        return False

    with open(output_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if isinstance(data, dict) and 'documents' in data:
        docs = data['documents']
        metadata = data.get('metadata', {})
        stats = metadata.get('statistics', {})

        print(f"\n✅ Export Verification:")
        print(f"  Total documents: {len(docs)}")
        print(f"  Successful dialogues: {stats.get('successful_dialogues', 'N/A')}")
        print(f"  Avg docs per dialogue: {stats.get('documents_per_dialogue', 'N/A')}")

        # Sample check
        if docs:
            sample = docs[0]
            required_fields = ['dialogue_id', 'original_question', 'original_answer', 'topic_path', 'document']
            missing = [f for f in required_fields if f not in sample]
            if missing:
                print(f"  ⚠️  Sample missing fields: {missing}")
            else:
                print(f"  ✅ All required fields present")

        return True
    elif isinstance(data, list):
        print(f"\n✅ Export Verification:")
        print(f"  Total documents: {len(data)}")

        # Sample check
        if data:
            sample = data[0]
            required_fields = ['dialogue_id', 'original_question', 'original_answer', 'topic_path', 'document']
            missing = [f for f in required_fields if f not in sample]
            if missing:
                print(f"  ⚠️  Sample missing fields: {missing}")
            else:
                print(f"  ✅ All required fields present")

        return True
    else:
        print(f"❌ Unexpected file format")
        return False


def main():
    parser = argparse.ArgumentParser(description="Export successful documents from state file")
    parser.add_argument(
        "--state",
        default="data/evaluation/knowledge_extraction_full/extracted_documents.json",
        help="Path to state file"
    )
    parser.add_argument(
        "--output",
        default="data/evaluation/knowledge_extraction_full/extracted_documents_final.json",
        help="Path to output file"
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Export documents only without metadata"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify output after export"
    )

    args = parser.parse_args()

    state_file = Path(args.state)
    output_file = Path(args.output)

    # Export documents
    result = export_success_only(state_file, output_file, include_metadata=not args.no_metadata)

    # Verify if requested
    if result == 0 and args.verify:
        verify_export(output_file)

    return result


if __name__ == "__main__":
    exit(main())
