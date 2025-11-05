#!/usr/bin/env python3
"""
Test partial experiment execution and resume
"""
import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load metadata
metadata_file = Path("data/evaluation/experiments/metadata.json")
with open(metadata_file, 'r') as f:
    metadata = json.load(f)

# Create a new partial experiment
exp_id = metadata["next_id"]
metadata["next_id"] += 1

# Only complete some steps
metadata["experiments"][str(exp_id)] = {
    "id": exp_id,
    "created_at": "2025-11-05T13:45:00",
    "config": {
        "questions": 4,
        "models": ["gpt-4o-mini", "ollama/exaone3.5:7.8b"],
        "skip_benchmark": False
    },
    "status": "in_progress",
    "completed_steps": ["questions", "classification"]  # Partial completion
}

# Save metadata
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Created partial experiment ID: {exp_id}")
print(f"📊 Completed steps: {metadata['experiments'][str(exp_id)]['completed_steps']}")
print(f"⏸️ Status: {metadata['experiments'][str(exp_id)]['status']}")

# Create checkpoint for questions
import pickle
checkpoint_dir = Path("data/evaluation/experiments/checkpoints")
questions = [
    {"question": "Test question 1", "ground_truth": "Answer 1", "question_type": "simple"},
    {"question": "Test question 2", "ground_truth": "Answer 2", "question_type": "simple"},
    {"question": "Test question 3", "ground_truth": "Answer 3", "question_type": "multi_reasoning"},
    {"question": "Test question 4", "ground_truth": "Answer 4", "question_type": "multi_reasoning"},
]
checkpoint_file = checkpoint_dir / f"exp_{exp_id}_questions.pkl"
with open(checkpoint_file, 'wb') as f:
    pickle.dump(questions, f)

print(f"💾 Saved checkpoint: {checkpoint_file}")
print(f"\n🔄 Ready to resume with: --resume-id {exp_id}")