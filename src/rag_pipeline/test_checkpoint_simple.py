#!/usr/bin/env python3
"""
Simple test for checkpoint functionality without RAGAS dependency
"""
import sys
import json
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Create experiments directory
exp_dir = Path("data/evaluation/experiments")
exp_dir.mkdir(parents=True, exist_ok=True)

# Test experiment metadata
metadata = {
    "next_id": 1,
    "experiments": {}
}

# Create metadata file
metadata_file = exp_dir / "metadata.json"
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✅ Created metadata file: {metadata_file}")

# Create a test experiment
exp_id = metadata["next_id"]
metadata["next_id"] += 1
metadata["experiments"][str(exp_id)] = {
    "id": exp_id,
    "created_at": datetime.now().isoformat(),
    "config": {
        "questions": 3,
        "models": ["gpt-4o-mini"],
        "skip_benchmark": True
    },
    "status": "initialized",
    "completed_steps": []
}

# Save updated metadata
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✅ Created experiment ID: {exp_id}")

# Create checkpoint directory
checkpoint_dir = exp_dir / "checkpoints"
checkpoint_dir.mkdir(parents=True, exist_ok=True)
print(f"✅ Created checkpoint directory: {checkpoint_dir}")

# Simulate saving a checkpoint
import pickle
checkpoint_data = {
    "questions": [
        {"question": "서울 지하철 1호선의 첫차 시간은?", "ground_truth": "05:30"},
        {"question": "기후동행카드 가격은?", "ground_truth": "65,000원"},
        {"question": "강남역에서 서울역까지 소요 시간은?", "ground_truth": "약 20분"}
    ]
}
checkpoint_file = checkpoint_dir / f"exp_{exp_id}_questions.pkl"
with open(checkpoint_file, 'wb') as f:
    pickle.dump(checkpoint_data, f)
print(f"✅ Saved checkpoint: {checkpoint_file}")

# Test loading checkpoint
with open(checkpoint_file, 'rb') as f:
    loaded_data = pickle.load(f)
print(f"✅ Loaded checkpoint: {len(loaded_data['questions'])} questions")

print("\n📊 Experiment structure ready:")
print(f"  - Metadata: {metadata_file}")
print(f"  - Checkpoints: {checkpoint_dir}")
print(f"  - Experiment ID: {exp_id}")