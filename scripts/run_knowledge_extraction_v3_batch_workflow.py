#!/usr/bin/env python3
"""
Knowledge Extraction v3 Batch Workflow Runner
=============================================

This script stitches together the full v3 workflow described in
`docs/knowledge_extraction_v3_workflow.md` into a single executable file.
It performs the following high-level steps for ~8,000 Dasan dialogues:

1. (Optional) Merge legacy `batch_*.json` artifacts into the v3 state file.
2. Initialize/refresh the single state file so every dialogue_id starts in `pending`.
3. Run the v3 extractor (`ClaudeKnowledgeExtractorV3`) with batch-wise saves.
4. Export only the successful documents into a flattened JSON file.

Every user-editable path or environment variable is marked with a 🔧 or 🔐 comment
so you can quickly see what must be customized before running.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable

# Ensure repository root is on sys.path when running from the scripts/ folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import existing helpers so we do not duplicate logic.
from scripts.merge_batch_files import merge_batch_files  # type: ignore
from scripts.export_success_only import export_success_only, verify_export  # type: ignore
from src.knowledge_extraction.claude_knowledge_extractor_v3 import ClaudeKnowledgeExtractorV3  # type: ignore


# ---------------------------------------------------------------------------
# 🔧 USER-EDITABLE DEFAULTS
# Update these defaults before running if your dataset or paths differ.
# You can also override them via CLI flags.
# ---------------------------------------------------------------------------
DEFAULT_DIALOGUES_PATH = PROJECT_ROOT / "data/processed/knowledge_extraction/full_dialogues.json"  # 🔧 Set to the ~8k dialogue JSON.
DEFAULT_STATE_FILE = PROJECT_ROOT / "data/evaluation/knowledge_extraction_full/extracted_documents.json"  # 🔧 Single-file state output.
DEFAULT_BATCH_ARTIFACT_DIR = DEFAULT_STATE_FILE.parent  # 🔧 Folder containing legacy batch_*.json files (if any).
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config/knowledge_extraction_full.json"  # 🔧 Extraction config tuned for 8k dialogues.
DEFAULT_EXPORT_PATH = DEFAULT_STATE_FILE.parent / "extracted_documents_final.json"  # 🔧 Flat export destination.
DEFAULT_SUMMARY_PATH = DEFAULT_STATE_FILE.parent / "extraction_summary.json"
DEFAULT_API_KEY_ENV = "CLAUDE_CODE_API_KEY"  # 🔐 REQUIRED: export this env var with your Claude key before running.
# ---------------------------------------------------------------------------


def load_dialogues(dialogues_path: Path) -> Dict[str, Iterable[Dict[str, Any]]]:
    """Load dialogues grouped by category."""
    if not dialogues_path.exists():
        raise FileNotFoundError(f"Dialogue file not found: {dialogues_path}")

    with open(dialogues_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    dialogues = payload.get("dialogues")
    if isinstance(dialogues, dict):
        return dialogues
    if isinstance(dialogues, list):
        # Some intermediate files are already flattened; wrap them under a single key.
        return {"_default": dialogues}

    raise ValueError(f"Unsupported dialogue structure in {dialogues_path}")


def flatten_dialogues(dialogue_groups: Dict[str, Iterable[Dict[str, Any]]]) -> Iterable[Dict[str, Any]]:
    """Yield every dialogue from the grouped structure, preserving the category label."""
    for category, dialogues in dialogue_groups.items():
        for dialogue in dialogues:
            dialogue.setdefault("category", category)
            yield dialogue


def ensure_state_file(dialogue_groups: Dict[str, Iterable[Dict[str, Any]]], state_file: Path) -> Dict[str, Any]:
    """
    Initialize or refresh the single state file so that every dialogue ID has an entry.
    Existing `success` entries are preserved so the extractor can resume safely.
    """
    state: Dict[str, Any] = {}
    if state_file.exists():
        with open(state_file, "r", encoding="utf-8") as f:
            state = json.load(f)

    total_dialogues = 0
    new_entries = 0

    for dialogue in flatten_dialogues(dialogue_groups):
        dialogue_id = dialogue.get("dialogue_id")
        if not dialogue_id:
            continue

        total_dialogues += 1
        if dialogue_id not in state:
            state[dialogue_id] = {
                "status": "pending",
                "dialogue_id": dialogue_id,
                "documents": [],
                "error": None,
                "timestamp": None,
                "retry_count": 0,
            }
            new_entries += 1
        else:
            # Guarantee dialogue_id + retry_count exist for older entries.
            state_entry = state[dialogue_id]
            state_entry.setdefault("dialogue_id", dialogue_id)
            state_entry.setdefault("retry_count", 0)

    state_file.parent.mkdir(parents=True, exist_ok=True)
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)

    counts = Counter(entry.get("status", "unknown") for entry in state.values())
    print(f"\n📊 State initialized/refreshed at {state_file}")
    print(f"   ✅ Success: {counts.get('success', 0)}")
    print(f"   ⏳ Pending: {counts.get('pending', 0)}")
    print(f"   ❌ Failed: {counts.get('fail', 0)}")
    print(f"   🔐 Auth Error: {counts.get('auth_error', 0)}")
    print(f"   ➕ New entries added: {new_entries}")
    print(f"   📈 Total monitored dialogues: {total_dialogues}\n")

    return state


def merge_legacy_batches(batch_dir: Path, state_file: Path) -> None:
    """
    Merge legacy batch files only if they exist.
    This is Step 0 from the workflow doc and is idempotent.
    """
    batch_files = sorted(batch_dir.glob("batch_*.json"))
    if not batch_files:
        print(f"ℹ️  No legacy batch_*.json files detected in {batch_dir}, skipping merge.")
        return

    print(f"🔄 Found {len(batch_files)} legacy batch files in {batch_dir}. Merging into {state_file} ...")
    rc = merge_batch_files(batch_dir=batch_dir, output_file=state_file)
    if rc != 0:
        raise RuntimeError("Merging legacy batch files failed – resolve the issue before proceeding.")


def ensure_api_key(env_var: str) -> None:
    """Validate that the Claude API key is available before starting extraction."""
    if not os.environ.get(env_var):
        raise EnvironmentError(
            f"Missing API key: export {env_var}=<your_key> before running this workflow."
        )
    print(f"🔐 Using Claude API key from ${env_var}")


def run_extraction(
    dialogue_groups: Dict[str, Iterable[Dict[str, Any]]],
    config_path: Path,
    state_file: Path,
    summary_path: Path,
) -> Dict[str, Any]:
    """Run the Claude extractor end-to-end and persist the summary."""
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    llm_cfg = config["llm"]
    output_dir = state_file.parent

    extractor = ClaudeKnowledgeExtractorV3(
        model_name=llm_cfg["model_name"],
        request_delay=float(llm_cfg["rate_limit"]["request_delay"]),
        batch_size=int(llm_cfg["batch_size"]),
        max_retries=int(llm_cfg.get("max_retries", 3)),
        state_file=state_file,
    )

    print(f"\n🚀 Starting extraction run with model `{llm_cfg['model_name']}`")
    print(f"   Batch size: {llm_cfg['batch_size']} dialogues")
    print(f"   Delay between batches: {llm_cfg['rate_limit']['request_delay']} seconds")
    print(f"   State file: {state_file}")
    print(f"   Summary will be saved to: {summary_path}\n")

    results = asyncio.run(extractor.extract_all(dialogue_groups, output_dir))

    summary = {
        "total_documents": results["total_documents"],
        "total_batches": results["total_batches"],
        "total_success": results["total_success"],
        "total_failure": results["total_failure"],
        "duration_seconds": results["duration_seconds"],
        "state_file": results["state_file"],
        "timestamp": datetime.now().isoformat(),
    }

    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Extraction summary saved to {summary_path}\n")
    return summary


def export_successes(state_file: Path, export_path: Path) -> None:
    """Export only successful documents into a flat JSON file."""
    print("📤 Exporting successful documents...")
    rc = export_success_only(state_file=state_file, output_file=export_path, include_metadata=True)
    if rc != 0:
        raise RuntimeError("Export script reported an error.")

    verify_export(export_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Knowledge Extraction v3 batch workflow end-to-end.")
    parser.add_argument(
        "--dialogues-path",
        type=Path,
        default=DEFAULT_DIALOGUES_PATH,
        help="🔧 Path to the consolidated dialogues JSON (≈8k records).",
    )
    parser.add_argument(
        "--state-file",
        type=Path,
        default=DEFAULT_STATE_FILE,
        help="🔧 Path to the single state JSON file that tracks status per dialogue.",
    )
    parser.add_argument(
        "--batch-artifact-dir",
        type=Path,
        default=DEFAULT_BATCH_ARTIFACT_DIR,
        help="🔧 Directory containing legacy batch_*.json files to merge (if any).",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="🔧 Extraction config tuned for the 8k run (model, rate limit, etc.).",
    )
    parser.add_argument(
        "--export-path",
        type=Path,
        default=DEFAULT_EXPORT_PATH,
        help="🔧 Destination for the flattened export of successful documents.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=DEFAULT_SUMMARY_PATH,
        help="Where to store the extraction summary JSON (defaults next to the state file).",
    )
    parser.add_argument(
        "--claude-api-key-env",
        type=str,
        default=DEFAULT_API_KEY_ENV,
        help="🔐 Environment variable that holds your Claude API key.",
    )
    parser.add_argument(
        "--skip-merge",
        action="store_true",
        help="Skip Step 0 (legacy batch merge).",
    )
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="Skip exporting the flattened success-only JSON at the end.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    ensure_api_key(args.claude_api_key_env)

    dialogue_groups = load_dialogues(args.dialogues_path)
    total_dialogues = sum(1 for _ in flatten_dialogues(dialogue_groups))
    print(f"\n📚 Loaded {total_dialogues} dialogues from {args.dialogues_path}")

    if not args.skip_merge:
        merge_legacy_batches(args.batch_artifact_dir, args.state_file)
    else:
        print("⏭️  Skipping legacy batch merge (per --skip-merge).")

    ensure_state_file(dialogue_groups, args.state_file)

    run_extraction(
        dialogue_groups=dialogue_groups,
        config_path=args.config_path,
        state_file=args.state_file,
        summary_path=args.summary_path,
    )

    if not args.skip_export:
        export_successes(args.state_file, args.export_path)
    else:
        print("⏭️  Skipping final export (per --skip-export).")

    print("\n🎉 Workflow complete! Review the summary + export artifacts for downstream steps.")


if __name__ == "__main__":
    main()
