#!/usr/bin/env python3
"""
Test script for AI Hub metadata EDA notebook.
This script tests the core functionality without running the full notebook.
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Test that all required libraries can be imported."""
    print("Testing imports...")
    try:
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        import seaborn as sns
        from tqdm import tqdm
        print("✓ All imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False

def test_paths():
    """Test that required paths exist."""
    print("\nTesting paths...")
    base_dir = project_root
    data_dir = base_dir / "data" / "raw" / "ai_hub_raw_data"

    if not data_dir.exists():
        print(f"✗ Data directory not found: {data_dir}")
        return False

    print(f"✓ Data directory exists: {data_dir}")
    return True

def test_scan_function():
    """Test the core scanning functionality."""
    print("\nTesting data scan functionality...")

    try:
        import pandas as pd
        from pathlib import Path

        data_dir = project_root / "data" / "raw" / "ai_hub_raw_data"

        # Simple scan function
        metadata_records = []

        for root, dirs, files in os.walk(data_dir):
            root_path = Path(root)

            for file in files:
                if file.startswith('.'):
                    continue

                file_path = root_path / file
                rel_path = file_path.relative_to(data_dir)
                parts = rel_path.parts

                record = {
                    'filename': file,
                    'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                    'extension': file_path.suffix,
                }

                if len(parts) >= 1:
                    record['split'] = parts[0]
                if len(parts) >= 2:
                    record['data_batch'] = parts[1]
                if len(parts) >= 3:
                    record['domain'] = parts[2]

                metadata_records.append(record)

        df = pd.DataFrame(metadata_records)

        print(f"✓ Successfully scanned {len(df)} files")
        print(f"  Total size: {df['file_size_mb'].sum():.2f} MB")

        if 'domain' in df.columns:
            print(f"  Domains found: {df['domain'].nunique()}")
            print(f"  Domain distribution:")
            for domain, count in df['domain'].value_counts().head(5).items():
                print(f"    - {domain}: {count} files")

        return True

    except Exception as e:
        print(f"✗ Scan failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("="*60)
    print("AI Hub Metadata EDA - Test Suite")
    print("="*60)

    results = []

    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("Paths", test_paths()))
    results.append(("Data Scan", test_scan_function()))

    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {test_name}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! The notebook should work correctly.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
