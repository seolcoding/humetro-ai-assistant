#!/usr/bin/env python3
"""
Extract Dasan Call Center data from zip files with proper Korean encoding.
"""

import os
import zipfile
from pathlib import Path
import shutil

# Paths
BASE_DIR = Path("/Users/sdh/Dev/02_production_projects/humetro-ai-assistant")
SOURCE_DIR = BASE_DIR / "data/dasan_call/raw/022.민원(콜센터) 질의-응답 데이터/01.데이터"
EXTRACT_DIR = BASE_DIR / "data/dasan_call/extracted"

# Create extraction directories
(EXTRACT_DIR / "training/labeled").mkdir(parents=True, exist_ok=True)
(EXTRACT_DIR / "training/source").mkdir(parents=True, exist_ok=True)
(EXTRACT_DIR / "validation/labeled").mkdir(parents=True, exist_ok=True)
(EXTRACT_DIR / "validation/source").mkdir(parents=True, exist_ok=True)

def extract_dasan_zips():
    """Extract all Dasan Call Center zip files."""

    # Define extraction mappings
    mappings = [
        {
            'source': SOURCE_DIR / "1.Training/라벨링데이터_220121_add/다산콜센터",
            'dest': EXTRACT_DIR / "training/labeled",
            'type': 'Training Labeled'
        },
        {
            'source': SOURCE_DIR / "1.Training/원천데이터_220325_add/다산콜센터",
            'dest': EXTRACT_DIR / "training/source",
            'type': 'Training Source'
        },
        {
            'source': SOURCE_DIR / "2.Validation/라벨링데이터_220121_add/다산콜센터",
            'dest': EXTRACT_DIR / "validation/labeled",
            'type': 'Validation Labeled'
        },
        {
            'source': SOURCE_DIR / "2.Validation/원천데이터_220325_add/다산콜센터",
            'dest': EXTRACT_DIR / "validation/source",
            'type': 'Validation Source'
        }
    ]

    total_files = 0
    total_extracted = 0

    for mapping in mappings:
        source_path = mapping['source']
        dest_path = mapping['dest']
        data_type = mapping['type']

        if not source_path.exists():
            print(f"⚠️  {data_type}: Source directory not found: {source_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Extracting {data_type}")
        print(f"{'='*60}")

        # Find all zip files
        zip_files = list(source_path.glob("*.zip"))
        print(f"Found {len(zip_files)} zip files")

        for zip_path in zip_files:
            print(f"  📦 {zip_path.name}")

            try:
                with zipfile.ZipFile(zip_path, 'r') as zf:
                    # Get list of files in zip
                    file_list = zf.namelist()
                    total_files += len(file_list)

                    # Extract all files
                    zf.extractall(dest_path)
                    total_extracted += len(file_list)

                    print(f"     ✓ Extracted {len(file_list)} files")

            except Exception as e:
                print(f"     ✗ Error: {e}")

    print(f"\n{'='*60}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total files extracted: {total_extracted}/{total_files}")
    print(f"Destination: {EXTRACT_DIR}")

    return total_extracted

def count_extracted_files():
    """Count extracted files by category."""
    print(f"\n{'='*60}")
    print(f"EXTRACTED FILES COUNT")
    print(f"{'='*60}")

    categories = [
        ("Training - Labeled", EXTRACT_DIR / "training/labeled"),
        ("Training - Source", EXTRACT_DIR / "training/source"),
        ("Validation - Labeled", EXTRACT_DIR / "validation/labeled"),
        ("Validation - Source", EXTRACT_DIR / "validation/source")
    ]

    total = 0
    for name, path in categories:
        if path.exists():
            json_files = list(path.glob("**/*.json"))
            wav_files = list(path.glob("**/*.wav"))
            count = len(json_files) + len(wav_files)
            total += count
            print(f"{name:30s}: {count:6,} files ({len(json_files)} JSON, {len(wav_files)} WAV)")
        else:
            print(f"{name:30s}: Directory not found")

    print(f"{'='*60}")
    print(f"{'Total':30s}: {total:6,} files")

    return total

def cleanup_non_dasan_data():
    """Remove non-Dasan data from raw directory."""
    print(f"\n{'='*60}")
    print(f"CLEANUP: Removing non-Dasan data")
    print(f"{'='*60}")

    # Directories to check
    base_dirs = [
        BASE_DIR / "data/dasan_call/raw/022.민원(콜센터) 질의-응답 데이터/01.데이터/1.Training",
        BASE_DIR / "data/dasan_call/raw/022.민원(콜센터) 질의-응답 데이터/01.데이터/2.Validation",
        BASE_DIR / "data/dasan_call/raw/1.Training",
        BASE_DIR / "data/dasan_call/raw/2.Validation"
    ]

    removed_count = 0

    for base_dir in base_dirs:
        if not base_dir.exists():
            continue

        for item in base_dir.iterdir():
            # Skip if it's the 다산콜센터 directory or file
            if "다산콜센터" in item.name or "다산" in item.name:
                continue

            # Remove non-Dasan data
            if item.is_dir():
                try:
                    shutil.rmtree(item)
                    print(f"  ✓ Removed directory: {item.name}")
                    removed_count += 1
                except Exception as e:
                    print(f"  ✗ Failed to remove {item.name}: {e}")
            elif item.is_file() and item.suffix in ['.zip', '.json', '.wav']:
                try:
                    item.unlink()
                    print(f"  ✓ Removed file: {item.name}")
                    removed_count += 1
                except Exception as e:
                    print(f"  ✗ Failed to remove {item.name}: {e}")

    print(f"\nRemoved {removed_count} non-Dasan items")

if __name__ == "__main__":
    print("="*60)
    print("Dasan Call Center Data Extraction")
    print("="*60)

    # Extract data
    extracted = extract_dasan_zips()

    # Count extracted files
    total_files = count_extracted_files()

    # Cleanup non-Dasan data
    cleanup_non_dasan_data()

    print(f"\n{'='*60}")
    print("✅ EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Extracted: {extracted} files")
    print(f"Location: {EXTRACT_DIR}")
