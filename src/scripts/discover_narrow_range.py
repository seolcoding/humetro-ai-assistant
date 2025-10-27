#!/usr/bin/env python3
"""
Narrow range discovery focused on known valid sequences.

Since we know 289801 is valid, this script:
1. Tests ranges around known valid sequences
2. Uses binary search to find boundaries
3. Tests dense sampling in promising ranges
"""

import argparse
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dotenv import load_dotenv
from tqdm import tqdm
import json
import time

import sys
sys.path.append(str(Path(__file__).parent))
from fetch_dasan_with_cache import DasanFetcherWithCache


class NarrowRangeDiscovery:
    """Discover range boundaries around known valid sequences."""

    def __init__(self, fetcher: DasanFetcherWithCache, data_type: str):
        self.fetcher = fetcher
        self.data_type = data_type
        self.results = {
            "known_valid": [],
            "boundaries": [],
            "dense_samples": {},
            "recommended_ranges": []
        }

    def test_range_dense(self, start: int, end: int, step: int = 1) -> List[int]:
        """
        Test a range densely with given step.

        Args:
            start: Start sequence
            end: End sequence
            step: Step size (1 for complete scan, >1 for sampling)

        Returns:
            List of valid sequences found
        """
        valid = []
        sequences = list(range(start, end + 1, step))

        print(f"\nDense testing: {start:,} - {end:,} (step={step}, testing {len(sequences):,} sequences)")

        for seq in tqdm(sequences, desc=f"Testing {start:,}-{end:,}"):
            is_valid = self.fetcher.check_sequence(self.data_type, seq, use_cache=True)

            if is_valid:
                valid.append(seq)
                tqdm.write(f"✓ Found: {seq}")

            time.sleep(0.05)

        hit_rate = (len(valid) / len(sequences) * 100) if sequences else 0
        print(f"Result: {len(valid):,}/{len(sequences):,} valid ({hit_rate:.1f}% hit rate)")

        return sorted(valid)

    def binary_search_boundary(self, known_valid: int, direction: str = "lower",
                               max_gap: int = 10000) -> Optional[int]:
        """
        Use binary search to find range boundary.

        Args:
            known_valid: A known valid sequence number
            direction: "lower" or "upper" boundary
            max_gap: Maximum gap to search

        Returns:
            Approximate boundary sequence number
        """
        print(f"\nBinary search for {direction} boundary from {known_valid:,}")

        if direction == "lower":
            low = max(1, known_valid - max_gap)
            high = known_valid
        else:  # upper
            low = known_valid
            high = min(999999, known_valid + max_gap)

        boundary = None

        while low < high:
            mid = (low + high) // 2

            is_valid = self.fetcher.check_sequence(self.data_type, mid, use_cache=True)
            print(f"  Testing {mid:,}: {'valid' if is_valid else 'invalid'}")

            if is_valid:
                boundary = mid
                if direction == "lower":
                    high = mid - 1
                else:
                    low = mid + 1
            else:
                if direction == "lower":
                    low = mid + 1
                else:
                    high = mid - 1

            time.sleep(0.05)

        if boundary:
            print(f"→ Found {direction} boundary around: {boundary:,}")
        else:
            print(f"→ No {direction} boundary found within {max_gap:,} sequences")

        return boundary

    def find_range_around(self, known_valid: int, sample_radius: int = 5000,
                         binary_search_radius: int = 10000) -> Dict:
        """
        Find complete range around a known valid sequence.

        Args:
            known_valid: Known valid sequence
            sample_radius: Radius for dense sampling
            binary_search_radius: Radius for binary search

        Returns:
            Dictionary with range information
        """
        print(f"\n{'='*70}")
        print(f"Finding range around known valid: {known_valid:,}")
        print(f"{'='*70}")

        # Step 1: Dense sample around known valid
        dense_start = max(1, known_valid - sample_radius)
        dense_end = min(999999, known_valid + sample_radius)

        print(f"\nStep 1: Dense sampling ±{sample_radius:,} around {known_valid:,}")
        valid_in_dense = self.test_range_dense(dense_start, dense_end, step=100)

        if not valid_in_dense:
            print(f"⚠ No valid sequences found in dense sample!")
            return {}

        print(f"\n✓ Found {len(valid_in_dense)} valid sequences in dense sample")
        print(f"  Min: {min(valid_in_dense):,}")
        print(f"  Max: {max(valid_in_dense):,}")

        # Step 2: Binary search for boundaries
        print(f"\nStep 2: Binary search for boundaries")
        lower_boundary = self.binary_search_boundary(
            min(valid_in_dense), "lower", binary_search_radius
        )
        upper_boundary = self.binary_search_boundary(
            max(valid_in_dense), "upper", binary_search_radius
        )

        # Step 3: Determine recommended range
        if lower_boundary and upper_boundary:
            recommended_start = lower_boundary - 1000  # Add buffer
            recommended_end = upper_boundary + 1000
        elif valid_in_dense:
            # Use dense sample min/max with buffer
            recommended_start = min(valid_in_dense) - 2000
            recommended_end = max(valid_in_dense) + 2000
        else:
            recommended_start = dense_start
            recommended_end = dense_end

        # Clamp to valid range
        recommended_start = max(1, recommended_start)
        recommended_end = min(999999, recommended_end)

        result = {
            "known_valid": known_valid,
            "dense_sample_range": (dense_start, dense_end),
            "valid_in_dense": valid_in_dense,
            "lower_boundary": lower_boundary,
            "upper_boundary": upper_boundary,
            "recommended_range": (recommended_start, recommended_end),
            "recommended_size": recommended_end - recommended_start + 1
        }

        self.results["boundaries"].append(result)

        print(f"\n{'='*70}")
        print(f"RECOMMENDED SCAN RANGE")
        print(f"{'='*70}")
        print(f"Start: {recommended_start:,}")
        print(f"End: {recommended_end:,}")
        print(f"Size: {recommended_end - recommended_start + 1:,} sequences")

        return result

    def discover_from_known(self, known_valid_list: List[int]) -> Dict:
        """
        Run discovery from list of known valid sequences.

        Args:
            known_valid_list: List of known valid sequence numbers

        Returns:
            Complete discovery results
        """
        print(f"\n{'#'*70}")
        print(f"# Narrow Range Discovery for {self.data_type}")
        print(f"# Starting from {len(known_valid_list)} known valid sequences")
        print(f"{'#'*70}")

        self.results["known_valid"] = known_valid_list

        # Find ranges around each known valid
        for known_valid in known_valid_list:
            result = self.find_range_around(known_valid)
            if result:
                self.results["recommended_ranges"].append(result["recommended_range"])

        return self.results

    def print_summary(self):
        """Print discovery summary."""
        print(f"\n\n{'='*70}")
        print(f"DISCOVERY SUMMARY")
        print(f"{'='*70}")

        if self.results["recommended_ranges"]:
            print(f"\nRecommended ranges for full scan:")

            for i, (start, end) in enumerate(self.results["recommended_ranges"], 1):
                size = end - start + 1
                print(f"\n{i}. Range: {start:,} - {end:,} ({size:,} sequences)")
                print(f"   Command:")
                print(f"   uv run python scripts/fetch_dasan_with_cache.py \\")
                print(f"     --type {self.data_type} \\")
                print(f"     --start {start} \\")
                print(f"     --end {end} \\")
                print(f"     --workers 10")

    def save_results(self, output_file: Path):
        """Save results to JSON."""
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Discover narrow ranges around known valid sequences"
    )
    parser.add_argument(
        "--type",
        required=True,
        choices=["faq", "seoul_manual", "district_manual"],
        help="Data type"
    )
    parser.add_argument(
        "--known-valid",
        type=int,
        nargs="+",
        default=[289801, 289756, 289803, 289808],
        help="Known valid sequence numbers (default: samples from docs)"
    )
    parser.add_argument(
        "--sample-radius",
        type=int,
        default=5000,
        help="Radius for dense sampling (default: 5000)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file (default: data/dasan_api_cache/narrow_discovery_{type}.json)"
    )

    args = parser.parse_args()

    # Load API key
    load_dotenv()
    api_key = os.getenv("SEOUL_DATA_API_KEY")
    if not api_key:
        raise ValueError("SEOUL_DATA_API_KEY not found")

    # Create fetcher
    fetcher = DasanFetcherWithCache(api_key)

    try:
        # Run discovery
        discovery = NarrowRangeDiscovery(fetcher, args.type)
        results = discovery.discover_from_known(args.known_valid)

        # Print summary
        discovery.print_summary()

        # Save results
        output_file = args.output or f"data/dasan_api_cache/narrow_discovery_{args.type}.json"
        discovery.save_results(Path(output_file))

    finally:
        fetcher.close()


if __name__ == "__main__":
    main()
