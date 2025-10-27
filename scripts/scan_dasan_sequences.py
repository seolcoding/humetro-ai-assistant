#!/usr/bin/env python3
"""
Scan Seoul Dasan Call Center API for valid sequence numbers.

Since the list APIs are not accessible, we'll scan through sequence number ranges
to find valid entries. This uses binary search and sampling to efficiently find
all valid sequence numbers.
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Set
from dotenv import load_dotenv
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


class SequenceScanner:
    """Scanner to find valid sequence numbers in Seoul Dasan API."""

    BASE_URL = "http://openAPI.seoul.go.kr:8088"

    def __init__(self, api_key: str, cache_dir: str = "data/dasan_api_cache"):
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def check_sequence(self, service_type: str, seq_no: int) -> bool:
        """
        Check if a sequence number exists.

        Args:
            service_type: 'faq' or 'workmanual'
            seq_no: Sequence number to check

        Returns:
            True if sequence exists, False otherwise
        """
        if service_type == "faq":
            service_name = "SearchDetailsFAQService"
            faq_type = "F"
        else:
            service_name = "SearchDetailsSeoulWorkmanualService"
            faq_type = "S"

        url = f"{self.BASE_URL}/{self.api_key}/json/{service_name}/1/1/{faq_type}/{seq_no}/"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            # Check for the service key in response
            if service_name in data:
                result = data[service_name].get("RESULT", {})
                result_code = result.get("CODE")
                return result_code == "INFO-000"

            return False

        except Exception as e:
            return False

    def binary_search_range(
        self,
        service_type: str,
        min_seq: int,
        max_seq: int
    ) -> tuple[int, int]:
        """
        Use binary search to find the approximate range of valid sequences.

        Args:
            service_type: 'faq' or 'workmanual'
            min_seq: Minimum sequence number to check
            max_seq: Maximum sequence number to check

        Returns:
            Tuple of (first_valid, last_valid) sequence numbers
        """
        print(f"\nBinary search for {service_type} range ({min_seq}-{max_seq})...")

        # Find first valid sequence
        first_valid = None
        left, right = min_seq, max_seq

        while left <= right:
            mid = (left + right) // 2
            print(f"  Checking {mid}...", end="")

            if self.check_sequence(service_type, mid):
                print(" ✓")
                first_valid = mid
                right = mid - 1
            else:
                print(" ✗")
                left = mid + 1

            time.sleep(0.2)

        if not first_valid:
            print(f"No valid sequences found in range")
            return (0, 0)

        # Find last valid sequence
        last_valid = first_valid
        left, right = first_valid, max_seq

        while left <= right:
            mid = (left + right) // 2
            print(f"  Checking {mid}...", end="")

            if self.check_sequence(service_type, mid):
                print(" ✓")
                last_valid = mid
                left = mid + 1
            else:
                print(" ✗")
                right = mid - 1

            time.sleep(0.2)

        print(f"\n✓ Found range: {first_valid} to {last_valid}")
        return (first_valid, last_valid)

    def scan_range(
        self,
        service_type: str,
        start: int,
        end: int,
        step: int = 1,
        max_workers: int = 5
    ) -> List[int]:
        """
        Scan a range of sequence numbers to find valid ones.

        Args:
            service_type: 'faq' or 'workmanual'
            start: Start sequence number
            end: End sequence number
            step: Step size for scanning
            max_workers: Number of parallel workers

        Returns:
            List of valid sequence numbers
        """
        valid_sequences = []
        total = (end - start) // step + 1

        print(f"\nScanning {service_type} sequences {start} to {end} (step={step})...")
        print(f"Total checks: {total}")

        def check_seq(seq: int) -> Optional[int]:
            if self.check_sequence(service_type, seq):
                return seq
            return None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(check_seq, seq): seq
                for seq in range(start, end + 1, step)
            }

            completed = 0
            for future in as_completed(futures):
                result = future.result()
                if result:
                    valid_sequences.append(result)
                    print(f"  ✓ Found: {result}")

                completed += 1
                if completed % 100 == 0:
                    print(f"  Progress: {completed}/{total} ({completed*100//total}%)")

                time.sleep(0.1)  # Rate limiting

        valid_sequences.sort()
        return valid_sequences

    def smart_scan(
        self,
        service_type: str,
        estimated_range: tuple[int, int] = (1, 400000),
        sample_step: int = 1000
    ) -> List[int]:
        """
        Smart scanning strategy combining binary search and sampling.

        Args:
            service_type: 'faq' or 'workmanual'
            estimated_range: Estimated range to scan
            sample_step: Step size for initial sampling

        Returns:
            List of all valid sequence numbers
        """
        print(f"\n{'='*70}")
        print(f"Smart Scan: {service_type}")
        print(f"{'='*70}")

        # Step 1: Binary search to find approximate range
        first_valid, last_valid = self.binary_search_range(
            service_type,
            estimated_range[0],
            estimated_range[1]
        )

        if first_valid == 0:
            return []

        # Step 2: Do a coarse scan to identify dense regions
        print(f"\nPhase 2: Coarse sampling (step={sample_step})...")
        coarse_valid = self.scan_range(
            service_type,
            first_valid,
            last_valid,
            step=sample_step,
            max_workers=5
        )

        print(f"\nFound {len(coarse_valid)} sequences in coarse scan")

        # Step 3: Fine scan around found sequences
        all_valid = set(coarse_valid)
        print(f"\nPhase 3: Fine scanning around found sequences...")

        for seq in coarse_valid:
            # Scan in a window around each found sequence
            window_start = max(first_valid, seq - sample_step)
            window_end = min(last_valid, seq + sample_step)

            print(f"  Scanning window around {seq} ({window_start}-{window_end})...")
            window_valid = self.scan_range(
                service_type,
                window_start,
                window_end,
                step=1,
                max_workers=10
            )
            all_valid.update(window_valid)

        return sorted(list(all_valid))

    def save_sequences(self, service_type: str, sequences: List[int]) -> None:
        """Save found sequences to file."""
        filename = f"{service_type}_sequences.json"
        filepath = self.cache_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(sequences, f, indent=2)

        print(f"\n✓ Saved {len(sequences)} sequences to {filepath}")


def main():
    """Main execution."""
    load_dotenv()

    api_key = os.getenv("SEOUL_DATA_API_KEY")
    if not api_key:
        raise ValueError("SEOUL_DATA_API_KEY not found in environment variables")

    scanner = SequenceScanner(api_key)

    print("=" * 70)
    print("Seoul Dasan API Sequence Scanner")
    print("=" * 70)
    print("\nThis will scan for valid sequence numbers using a smart strategy:")
    print("1. Binary search to find the valid range")
    print("2. Coarse sampling to identify dense regions")
    print("3. Fine scanning around found sequences")
    print("\nThis may take a while. Press Ctrl+C to stop.")
    print("=" * 70)

    try:
        # Scan FAQs
        print("\n\n[1/2] Scanning FAQs...")
        faq_sequences = scanner.smart_scan(
            "faq",
            estimated_range=(280000, 300000),  # Based on sample data
            sample_step=100
        )
        scanner.save_sequences("faq", faq_sequences)

        # Scan work manuals
        print("\n\n[2/2] Scanning Work Manuals...")
        manual_sequences = scanner.smart_scan(
            "workmanual",
            estimated_range=(280000, 300000),  # Based on sample data
            sample_step=100
        )
        scanner.save_sequences("workmanual", manual_sequences)

        # Summary
        print("\n" + "=" * 70)
        print("Scan Complete!")
        print("=" * 70)
        print(f"FAQs found: {len(faq_sequences)}")
        print(f"Work manuals found: {len(manual_sequences)}")
        print(f"\nSequence files saved to: {scanner.cache_dir}")
        print("\nNext steps:")
        print("1. Use fetch_dasan_api_data.py to download details for these sequences")
        print("2. Review the cached data in data/dasan_api_cache/")

    except KeyboardInterrupt:
        print("\n\nScan interrupted by user")


if __name__ == "__main__":
    main()
