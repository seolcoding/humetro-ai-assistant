#!/usr/bin/env python3
"""
Test Knowledge Extraction with 30 Dialogues
============================================

Quick test script to validate extraction pipeline with small dataset.

Usage:
    uv run python scripts/test_extraction_30.py
"""

import json
import random
from pathlib import Path
from datetime import datetime

def create_test_dataset():
    """Create test dataset with 30 dialogues (10 from each of 3 categories)."""

    # Load full sampled dialogues
    input_file = Path("data/processed/knowledge_extraction/sampled_dialogues.json")

    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        print("   Please run sample_dialogues.py first")
        return None

    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Select 10 dialogues from first 3 categories
    test_dialogues = {}
    categories_to_use = list(data['dialogues'].keys())[:3]  # First 3 categories

    for category in categories_to_use:
        category_dialogues = data['dialogues'][category]
        # Take first 10 dialogues (or all if less than 10)
        test_dialogues[category] = category_dialogues[:10]

    # Count total dialogues
    total_count = sum(len(dialogues) for dialogues in test_dialogues.values())

    # Create test dataset
    test_data = {
        "metadata": {
            "created_at": datetime.now().isoformat(),
            "description": "Test dataset with 30 dialogues for extraction validation",
            "total_dialogues": total_count,
            "source": str(input_file)
        },
        "dialogues": test_dialogues
    }

    # Save test dataset
    output_file = Path("data/processed/knowledge_extraction/test_30_dialogues.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print("✅ Test dataset created")
    print(f"   📁 Output: {output_file}")
    print(f"   📊 Total dialogues: {total_count}")
    print(f"\n   Categories:")
    for category, dialogues in test_dialogues.items():
        print(f"      - {category}: {len(dialogues)} dialogues")

    return output_file


def create_test_config():
    """Create test configuration with shorter delays."""

    config = {
        "experiment": {
            "id": "knowledge_extraction_test_30",
            "name": "Test Knowledge Extraction (30 dialogues)",
            "description": "Quick validation test with 30 dialogues",
            "output_dir": "data/evaluation/knowledge_extraction_test_30",
            "tags": ["test", "validation", "30-dialogues"]
        },
        "llm": {
            "model_name": "claude-sonnet",
            "model_id": "claude-sonnet-4-5-20250929",
            "api_base": "http://localhost:25292/v1",
            "api_key_env": "CLAUDE_CODE_API_KEY",
            "rate_limit": {
                "calls_per_minute": 1,
                "request_delay": 65.0,
                "comment": "1 call per minute = 60 seconds + 5s safety margin = 65s"
            },
            "batch_size": 10,
            "comment": "Process 10 dialogues per batch (30 dialogues = 3 batches = ~3.3 minutes)"
        },
        "processing": {
            "min_doc_size": 500,
            "max_doc_size": 2000,
            "merge_small_docs": True,
            "taxonomy_levels": [2, 3],
            "split_on_intent_change": True,
            "preserve_dialogue_context": True
        },
        "validation": {
            "enabled": True,
            "check_metadata_completeness": True,
            "check_doc_size": True,
            "check_topic_consistency": True,
            "generate_sample_report": True,
            "sample_report_path": "data/evaluation/knowledge_extraction_test_30/quality_report.md"
        }
    }

    output_file = Path("config/knowledge_extraction_test_30.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Test config created: {output_file}")

    return output_file


def print_run_instructions(dataset_file: Path, config_file: Path):
    """Print instructions to run the test."""

    print("\n" + "="*80)
    print("🚀 READY TO RUN TEST")
    print("="*80)
    print("\n1️⃣  Run extraction:")
    print(f"\nuv run python src/knowledge_extraction/claude_knowledge_extractor_v2.py \\")
    print(f"    --config {config_file} \\")
    print(f"    --input {dataset_file}")

    print("\n2️⃣  Expected results:")
    print("   - 3 batches (10 dialogues each)")
    print("   - ~3.3 minutes total time")
    print("   - ~30 documents extracted")

    print("\n3️⃣  After completion, organize documents:")
    print("\nuv run python src/knowledge_extraction/document_organizer.py \\")
    print("    --input data/evaluation/knowledge_extraction_test_30/extracted_documents.json \\")
    print("    --output-dir data/processed/knowledge_docs_test \\")
    print("    --min-doc-size 500 \\")
    print("    --merge-small")

    print("\n4️⃣  Validate quality:")
    print("\nuv run python src/knowledge_extraction/quality_validator.py \\")
    print("    --docs-dir data/processed/knowledge_docs_test \\")
    print("    --report-path data/evaluation/knowledge_extraction_test_30/quality_report.md \\")
    print("    --min-doc-size 500")

    print("\n" + "="*80)


def main():
    """Main test setup script."""

    print("="*80)
    print("🧪 SETTING UP KNOWLEDGE EXTRACTION TEST (30 dialogues)")
    print("="*80)

    # Create test dataset
    dataset_file = create_test_dataset()
    if not dataset_file:
        return

    # Create test config
    config_file = create_test_config()

    # Print run instructions
    print_run_instructions(dataset_file, config_file)


if __name__ == "__main__":
    main()
