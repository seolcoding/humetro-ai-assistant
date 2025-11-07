#!/usr/bin/env python3
"""
Complete script to fetch all types of Dasan Call Center data.

FAQ_TP types:
- F: FAQ (자주묻는질문)
- S: Seoul Work Manual (서울시 업무매뉴얼)
- J: District Work Manual (자치구 업무매뉴얼)

Usage:
    # Default range (280000-300000)
    uv run python scripts/fetch_all_dasan_types.py

    # Quick scan (smaller range)
    uv run python scripts/fetch_all_dasan_types.py --quick

    # Custom range
    uv run python scripts/fetch_all_dasan_types.py --start 280000 --end 295000
"""

import argparse
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from dotenv import load_dotenv
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


class CompleteDasanFetcher:
    """Fetcher for all Dasan Call Center data types."""

    BASE_URL = "http://openAPI.seoul.go.kr:8088"

    # All three FAQ types
    FAQ_TYPES = {
        "faq": ("F", "SearchDetailsFAQService", "FAQ"),
        "seoul_manual": ("S", "SearchDetailsSeoulWorkmanualService", "서울시 업무매뉴얼"),
        "district_manual": ("J", "SearchDetailsSeoulWorkmanualService", "자치구 업무매뉴얼")
    }

    def __init__(self, api_key: str, cache_dir: str = "data/dasan_api_cache"):
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def check_sequence(self, data_type: str, seq_no: int) -> bool:
        """Check if a sequence exists for given data type."""
        faq_type, service_name, _ = self.FAQ_TYPES[data_type]
        url = f"{self.BASE_URL}/{self.api_key}/json/{service_name}/1/1/{faq_type}/{seq_no}/"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if service_name in data:
                result = data[service_name].get("RESULT", {})
                return result.get("CODE") == "INFO-000"

            return False
        except Exception:
            return False

    def scan_range(
        self,
        data_type: str,
        start: int,
        end: int,
        max_workers: int = 10
    ) -> List[int]:
        """Scan a range for valid sequences."""
        _, _, type_name = self.FAQ_TYPES[data_type]
        valid_sequences = []
        total = end - start + 1

        print(f"\nScanning {type_name} ({start:,} to {end:,})...")

        def check_seq(seq: int) -> Optional[int]:
            return seq if self.check_sequence(data_type, seq) else None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(check_seq, seq): seq
                for seq in range(start, end + 1)
            }

            completed = 0
            for future in as_completed(futures):
                result = future.result()
                if result:
                    valid_sequences.append(result)
                    print(f"  ✓ Found: {result}")

                completed += 1
                if completed % 1000 == 0:
                    print(f"  Progress: {completed:,}/{total:,} ({completed*100//total}%)")

                time.sleep(0.05)  # Rate limiting

        valid_sequences.sort()
        print(f"\n✓ Found {len(valid_sequences)} {type_name} sequences")
        return valid_sequences

    def fetch_detail(self, data_type: str, seq_no: int) -> Optional[Dict]:
        """Fetch detail for a specific sequence."""
        faq_type, service_name, _ = self.FAQ_TYPES[data_type]
        cache_file = self.cache_dir / f"{data_type}_detail_{seq_no}.json"

        # Check cache
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)

        # Fetch from API
        url = f"{self.BASE_URL}/{self.api_key}/json/{service_name}/1/1/{faq_type}/{seq_no}/"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if service_name in data:
                result = data[service_name].get("RESULT", {})
                if result.get("CODE") == "INFO-000":
                    # Save to cache
                    with open(cache_file, "w", encoding="utf-8") as f:
                        json.dump(data, f, ensure_ascii=False, indent=2)
                    return data

            return None
        except Exception:
            return None

    def fetch_all_details(
        self,
        data_type: str,
        sequences: List[int]
    ) -> List[Dict]:
        """Fetch all details for given sequences."""
        _, service_name, type_name = self.FAQ_TYPES[data_type]
        all_details = []

        print(f"\nFetching {len(sequences)} {type_name} details...")

        for i, seq_no in enumerate(sequences, 1):
            print(f"  [{i}/{len(sequences)}] {seq_no}...", end="", flush=True)

            data = self.fetch_detail(data_type, seq_no)

            if data and service_name in data:
                row = data[service_name].get("row")
                if row:
                    if isinstance(row, list):
                        all_details.extend(row)
                    else:
                        all_details.append(row)
                    print(" ✓")
                else:
                    print(" - no data")
            else:
                print(" ✗")

            if i < len(sequences):
                time.sleep(0.2)

        return all_details

    def save_data(self, data_type: str, sequences: List[int], details: List[Dict], scan_name: str) -> None:
        """Save sequences and details to files."""
        # Save sequences
        seq_file = self.cache_dir / f"{data_type}_{scan_name}_sequences.json"
        with open(seq_file, "w", encoding="utf-8") as f:
            json.dump(sequences, f, indent=2)

        # Save details
        detail_file = self.cache_dir / f"{data_type}_{scan_name}_all.json"
        with open(detail_file, "w", encoding="utf-8") as f:
            json.dump(details, f, ensure_ascii=False, indent=2)

        print(f"  ✓ Saved to {detail_file}")


def main():
    parser = argparse.ArgumentParser(description="Fetch all Dasan Call Center data types")
    parser.add_argument("--quick", action="store_true", help="Quick scan (289000-290000)")
    parser.add_argument("--start", type=int, help="Start sequence number")
    parser.add_argument("--end", type=int, help="End sequence number")
    parser.add_argument("--types", nargs="+", choices=["faq", "seoul_manual", "district_manual"],
                       help="Data types to fetch (default: all)")

    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("SEOUL_DATA_API_KEY")
    if not api_key:
        raise ValueError("SEOUL_DATA_API_KEY not found in environment variables")

    # Determine scan range
    if args.quick:
        start_seq, end_seq = 289000, 290000
        scan_name = "quick"
    elif args.start and args.end:
        start_seq, end_seq = args.start, args.end
        scan_name = f"range_{start_seq}_{end_seq}"
    else:
        start_seq, end_seq = 280000, 300000
        scan_name = "default"

    # Determine which types to fetch
    data_types = args.types if args.types else ["faq", "seoul_manual", "district_manual"]

    print("=" * 70)
    print("Seoul Dasan Call Center Complete Data Fetcher")
    print("=" * 70)
    print(f"\nScan range: {start_seq:,} to {end_seq:,}")
    print(f"Data types: {', '.join(data_types)}")
    print(f"Scan name: {scan_name}")
    print("\n" + "=" * 70)

    fetcher = CompleteDasanFetcher(api_key)
    results = {}

    for i, data_type in enumerate(data_types, 1):
        type_name = fetcher.FAQ_TYPES[data_type][2]

        print(f"\n[{i}/{len(data_types)}] Processing {type_name}")
        print("=" * 70)

        # Step 1: Scan for sequences
        print(f"\nStep 1: Scanning for {type_name} sequences")
        sequences = fetcher.scan_range(data_type, start_seq, end_seq, max_workers=10)

        if not sequences:
            print(f"  No sequences found for {type_name}")
            continue

        # Step 2: Fetch details
        print(f"\nStep 2: Fetching {type_name} details")
        details = fetcher.fetch_all_details(data_type, sequences)

        # Step 3: Save data
        print(f"\nStep 3: Saving {type_name} data")
        fetcher.save_data(data_type, sequences, details, scan_name)

        results[data_type] = {
            "sequences": len(sequences),
            "details": len(details)
        }

    # Final summary
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"Scan range: {start_seq:,} to {end_seq:,}\n")
    print("Results:")

    total_sequences = 0
    total_details = 0

    for data_type in data_types:
        if data_type in results:
            type_name = fetcher.FAQ_TYPES[data_type][2]
            seq_count = results[data_type]["sequences"]
            det_count = results[data_type]["details"]

            print(f"  {type_name}:")
            print(f"    - Sequences: {seq_count:,}")
            print(f"    - Details: {det_count:,}")

            total_sequences += seq_count
            total_details += det_count

    print(f"\nTotal:")
    print(f"  - Sequences: {total_sequences:,}")
    print(f"  - Details: {total_details:,}")
    print(f"\nCache directory: {fetcher.cache_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
