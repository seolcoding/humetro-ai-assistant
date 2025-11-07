#!/usr/bin/env python3
"""
Complete data fetcher for Seoul Dasan Call Center API.

This script:
1. Scans for all valid sequence numbers in a given range
2. Fetches detailed information for each sequence
3. Saves all data to cache directory

Usage:
    # Quick scan (smaller range, for testing)
    uv run python scripts/fetch_complete_dasan_data.py --quick

    # Full scan (broader range, takes longer)
    uv run python scripts/fetch_complete_dasan_data.py --full

    # Custom range
    uv run python scripts/fetch_complete_dasan_data.py --start 280000 --end 300000
"""

import argparse
import os
import json
import time
from pathlib import Path
from typing import List
from dotenv import load_dotenv
from scan_dasan_sequences import SequenceScanner
from fetch_dasan_api_data import DasanAPIClient


def fetch_all_details(
    api_key: str,
    service_type: str,
    sequences: List[int],
    cache_dir: str = "data/dasan_api_cache"
) -> List[dict]:
    """
    Fetch detailed information for all sequences.

    Args:
        api_key: API key
        service_type: 'faq' or 'workmanual'
        sequences: List of sequence numbers
        cache_dir: Cache directory

    Returns:
        List of detailed data
    """
    client = DasanAPIClient(api_key, cache_dir)
    all_details = []

    print(f"\nFetching {len(sequences)} {service_type} details...")

    for i, seq_no in enumerate(sequences, 1):
        seq_str = str(seq_no)
        print(f"  [{i}/{len(sequences)}] {service_type} {seq_str}...", end="", flush=True)

        if service_type == "faq":
            data = client.fetch_faq_detail(seq_str, use_cache=True)
            service_key = "SearchDetailsFAQService"
        else:
            data = client.fetch_workmanual_detail(seq_str, use_cache=True)
            service_key = "SearchDetailsSeoulWorkmanualService"

        if data and service_key in data:
            row = data[service_key].get("row")
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

        # Rate limiting
        if i < len(sequences):
            time.sleep(0.2)

    return all_details


def main():
    parser = argparse.ArgumentParser(description="Fetch complete Dasan Call Center data")
    parser.add_argument("--quick", action="store_true", help="Quick scan (289000-290000)")
    parser.add_argument("--full", action="store_true", help="Full scan (1-400000)")
    parser.add_argument("--start", type=int, help="Start sequence number")
    parser.add_argument("--end", type=int, help="End sequence number")
    parser.add_argument("--no-fetch-details", action="store_true", help="Only scan sequences, don't fetch details")

    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("SEOUL_DATA_API_KEY")
    if not api_key:
        raise ValueError("SEOUL_DATA_API_KEY not found in environment variables")

    # Determine scan range
    if args.quick:
        start_seq, end_seq = 289000, 290000
        scan_name = "quick"
    elif args.full:
        start_seq, end_seq = 1, 400000
        scan_name = "full"
    elif args.start and args.end:
        start_seq, end_seq = args.start, args.end
        scan_name = f"custom_{start_seq}_{end_seq}"
    else:
        # Default: medium range based on known samples
        start_seq, end_seq = 280000, 300000
        scan_name = "medium"

    print("=" * 70)
    print("Seoul Dasan Call Center Complete Data Fetcher")
    print("=" * 70)
    print(f"\nScan range: {start_seq:,} to {end_seq:,}")
    print(f"Scan name: {scan_name}")
    print("=" * 70)

    scanner = SequenceScanner(api_key)

    # Step 1: Scan for FAQs
    print("\n" + "=" * 70)
    print("[1/4] Scanning for FAQ sequences")
    print("=" * 70)

    faq_sequences = scanner.scan_range(
        "faq",
        start=start_seq,
        end=end_seq,
        step=1,
        max_workers=10
    )

    print(f"\n✓ Found {len(faq_sequences)} FAQ sequences")
    scanner.save_sequences(f"faq_{scan_name}", faq_sequences)

    # Step 2: Scan for Work Manuals
    print("\n" + "=" * 70)
    print("[2/4] Scanning for Work Manual sequences")
    print("=" * 70)

    manual_sequences = scanner.scan_range(
        "workmanual",
        start=start_seq,
        end=end_seq,
        step=1,
        max_workers=10
    )

    print(f"\n✓ Found {len(manual_sequences)} Work Manual sequences")
    scanner.save_sequences(f"workmanual_{scan_name}", manual_sequences)

    if args.no_fetch_details:
        print("\n" + "=" * 70)
        print("Sequence scan complete (details not fetched)")
        print("=" * 70)
        print(f"FAQs: {len(faq_sequences)} sequences")
        print(f"Work Manuals: {len(manual_sequences)} sequences")
        return

    # Step 3: Fetch FAQ details
    print("\n" + "=" * 70)
    print("[3/4] Fetching FAQ details")
    print("=" * 70)

    faq_details = fetch_all_details(api_key, "faq", faq_sequences)

    # Save FAQ details
    faq_file = Path("data/dasan_api_cache") / f"all_faqs_{scan_name}.json"
    with open(faq_file, "w", encoding="utf-8") as f:
        json.dump(faq_details, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Saved {len(faq_details)} FAQ details to {faq_file}")

    # Step 4: Fetch Work Manual details
    print("\n" + "=" * 70)
    print("[4/4] Fetching Work Manual details")
    print("=" * 70)

    manual_details = fetch_all_details(api_key, "workmanual", manual_sequences)

    # Save Manual details
    manual_file = Path("data/dasan_api_cache") / f"all_workmanuals_{scan_name}.json"
    with open(manual_file, "w", encoding="utf-8") as f:
        json.dump(manual_details, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Saved {len(manual_details)} Work Manual details to {manual_file}")

    # Final summary
    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"Scan range: {start_seq:,} to {end_seq:,}")
    print(f"\nResults:")
    print(f"  FAQs: {len(faq_sequences)} sequences, {len(faq_details)} details")
    print(f"  Work Manuals: {len(manual_sequences)} sequences, {len(manual_details)} details")
    print(f"\nFiles saved:")
    print(f"  - {faq_file}")
    print(f"  - {manual_file}")
    print(f"  - data/dasan_api_cache/faq_{scan_name}_sequences.json")
    print(f"  - data/dasan_api_cache/workmanual_{scan_name}_sequences.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
