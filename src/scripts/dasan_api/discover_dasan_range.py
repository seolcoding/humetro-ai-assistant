#!/usr/bin/env python3
"""
Range discovery script for Dasan API sequences.

Strategy:
1. Sample by digit count (1-6 digits)
2. Within each digit range, sample random positions
3. Identify which ranges contain valid data
4. Narrow down to likely valid ranges for full scan

This avoids scanning millions of sequences blindly.
"""

import argparse
import os
import random
import time
from pathlib import Path
from typing import List, Dict, Tuple
from dotenv import load_dotenv
from tqdm import tqdm
import json

# Import from existing script
import sys
sys.path.append(str(Path(__file__).parent))
from fetch_dasan_with_cache import DasanFetcherWithCache


class RangeDiscovery:
    """Discover valid sequence ranges efficiently."""

    DIGIT_RANGES = {
        1: (1, 9),           # 1-digit: 1-9
        2: (10, 99),         # 2-digit: 10-99
        3: (100, 999),       # 3-digit: 100-999
        4: (1000, 9999),     # 4-digit: 1000-9999
        5: (10000, 99999),   # 5-digit: 10000-99999
        6: (100000, 999999), # 6-digit: 100000-999999
    }

    def __init__(self, fetcher: DasanFetcherWithCache, data_type: str):
        self.fetcher = fetcher
        self.data_type = data_type
        self.results = {
            "digit_analysis": {},
            "valid_ranges": [],
            "samples_tested": 0,
            "valid_found": 0
        }

    def sample_digit_range(self, digits: int, samples: int = 20) -> Dict:
        """
        Sample random sequences within a digit range.

        Args:
            digits: Number of digits (1-6)
            samples: Number of random samples to test

        Returns:
            Dictionary with analysis results
        """
        start, end = self.DIGIT_RANGES[digits]

        # Generate random samples
        sample_seqs = random.sample(range(start, end + 1), min(samples, end - start + 1))

        valid_seqs = []

        print(f"\n{'='*70}")
        print(f"Testing {digits}-digit range: {start:,} - {end:,}")
        print(f"Sampling {len(sample_seqs)} random sequences...")
        print(f"{'='*70}")

        for seq in tqdm(sample_seqs, desc=f"{digits}-digit samples"):
            is_valid = self.fetcher.check_sequence(self.data_type, seq, use_cache=True)
            self.results["samples_tested"] += 1

            if is_valid:
                valid_seqs.append(seq)
                self.results["valid_found"] += 1
                tqdm.write(f"✓ Found valid: {seq}")

            time.sleep(0.05)  # Rate limiting

        hit_rate = (len(valid_seqs) / len(sample_seqs) * 100) if sample_seqs else 0

        result = {
            "range": (start, end),
            "samples_tested": len(sample_seqs),
            "valid_found": len(valid_seqs),
            "valid_sequences": sorted(valid_seqs),
            "hit_rate": hit_rate,
            "likely_contains_data": hit_rate > 0
        }

        print(f"Result: {len(valid_seqs)}/{len(sample_seqs)} valid ({hit_rate:.1f}% hit rate)")

        return result

    def subdivide_range(self, start: int, end: int, subdivisions: int = 10,
                       samples_per_div: int = 5) -> List[Dict]:
        """
        Subdivide a range and sample each subdivision.

        Args:
            start: Start of range
            end: End of range
            subdivisions: Number of subdivisions
            samples_per_div: Samples per subdivision

        Returns:
            List of subdivision results
        """
        range_size = end - start + 1
        div_size = range_size // subdivisions

        results = []

        print(f"\n{'='*70}")
        print(f"Subdividing range {start:,} - {end:,} into {subdivisions} parts")
        print(f"Each subdivision: ~{div_size:,} sequences")
        print(f"{'='*70}")

        for i in range(subdivisions):
            div_start = start + (i * div_size)
            div_end = div_start + div_size - 1 if i < subdivisions - 1 else end

            # Sample within this subdivision
            sample_seqs = []
            for _ in range(samples_per_div):
                sample_seqs.append(random.randint(div_start, div_end))

            valid_seqs = []

            print(f"\nSubdivision {i+1}/{subdivisions}: {div_start:,} - {div_end:,}")

            for seq in sample_seqs:
                is_valid = self.fetcher.check_sequence(self.data_type, seq, use_cache=True)
                self.results["samples_tested"] += 1

                if is_valid:
                    valid_seqs.append(seq)
                    self.results["valid_found"] += 1
                    print(f"  ✓ Found valid: {seq}")

                time.sleep(0.05)

            hit_rate = (len(valid_seqs) / len(sample_seqs) * 100) if sample_seqs else 0

            result = {
                "range": (div_start, div_end),
                "samples_tested": len(sample_seqs),
                "valid_found": len(valid_seqs),
                "valid_sequences": sorted(valid_seqs),
                "hit_rate": hit_rate,
                "likely_contains_data": len(valid_seqs) > 0
            }

            results.append(result)

            if len(valid_seqs) > 0:
                print(f"  → Hit rate: {hit_rate:.1f}% - PROMISING RANGE")

        return results

    def discover_ranges(self, max_digits: int = 6, samples_per_digit: int = 20) -> Dict:
        """
        Run complete range discovery.

        Args:
            max_digits: Maximum digit count to test
            samples_per_digit: Number of samples per digit range

        Returns:
            Complete analysis results
        """
        print(f"\n{'#'*70}")
        print(f"# Range Discovery for {self.data_type}")
        print(f"# Strategy: Sample by digit count, then subdivide promising ranges")
        print(f"{'#'*70}\n")

        # Phase 1: Test each digit range
        for digits in range(1, max_digits + 1):
            result = self.sample_digit_range(digits, samples_per_digit)
            self.results["digit_analysis"][digits] = result

            if result["likely_contains_data"]:
                self.results["valid_ranges"].append(result["range"])

        # Phase 2: Subdivide promising ranges for better precision
        print(f"\n\n{'#'*70}")
        print(f"# Phase 2: Subdividing promising ranges")
        print(f"{'#'*70}")

        detailed_ranges = []

        for digit_count, analysis in self.results["digit_analysis"].items():
            if analysis["likely_contains_data"]:
                start, end = analysis["range"]
                subdivisions = self.subdivide_range(start, end, subdivisions=10, samples_per_div=3)

                # Keep only promising subdivisions
                promising = [s for s in subdivisions if s["likely_contains_data"]]
                detailed_ranges.extend(promising)

        self.results["detailed_ranges"] = detailed_ranges

        return self.results

    def print_summary(self):
        """Print discovery summary."""
        print(f"\n\n{'='*70}")
        print(f"DISCOVERY SUMMARY")
        print(f"{'='*70}")
        print(f"Total samples tested: {self.results['samples_tested']:,}")
        print(f"Valid sequences found: {self.results['valid_found']:,}")
        print(f"Overall hit rate: {(self.results['valid_found']/self.results['samples_tested']*100):.2f}%")

        print(f"\n{'='*70}")
        print(f"DIGIT RANGE ANALYSIS")
        print(f"{'='*70}")

        for digits, analysis in sorted(self.results["digit_analysis"].items()):
            start, end = analysis["range"]
            status = "✓ HAS DATA" if analysis["likely_contains_data"] else "✗ No data"
            print(f"{digits}-digit ({start:>6,} - {end:>6,}): "
                  f"{analysis['valid_found']:>3}/{analysis['samples_tested']:>3} valid "
                  f"({analysis['hit_rate']:>5.1f}%) {status}")

        if "detailed_ranges" in self.results and self.results["detailed_ranges"]:
            print(f"\n{'='*70}")
            print(f"PROMISING RANGES FOR FULL SCAN")
            print(f"{'='*70}")

            for i, range_info in enumerate(self.results["detailed_ranges"], 1):
                start, end = range_info["range"]
                print(f"{i}. {start:>6,} - {end:>6,} "
                      f"(hit rate: {range_info['hit_rate']:.1f}%, "
                      f"found: {range_info['valid_found']})")

            print(f"\n{'='*70}")
            print(f"RECOMMENDED SCAN COMMAND")
            print(f"{'='*70}")

            # Suggest most promising range
            best_range = max(self.results["detailed_ranges"],
                           key=lambda x: x["hit_rate"])
            start, end = best_range["range"]

            print(f"\nMost promising range: {start:,} - {end:,}")
            print(f"Hit rate: {best_range['hit_rate']:.1f}%\n")
            print(f"uv run python scripts/fetch_dasan_with_cache.py \\")
            print(f"  --type {self.data_type} \\")
            print(f"  --start {start} \\")
            print(f"  --end {end} \\")
            print(f"  --workers 10")

    def save_results(self, output_file: Path):
        """Save discovery results to JSON."""
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Discover valid sequence ranges for Dasan API"
    )
    parser.add_argument(
        "--type",
        required=True,
        choices=["faq", "seoul_manual", "district_manual"],
        help="Data type to discover"
    )
    parser.add_argument(
        "--max-digits",
        type=int,
        default=6,
        help="Maximum digit count to test (default: 6)"
    )
    parser.add_argument(
        "--samples-per-digit",
        type=int,
        default=20,
        help="Number of random samples per digit range (default: 20)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (default: data/dasan_api_cache/discovery_{type}.json)"
    )

    args = parser.parse_args()

    # Load API key
    load_dotenv()
    api_key = os.getenv("SEOUL_DATA_API_KEY")
    if not api_key:
        raise ValueError("SEOUL_DATA_API_KEY not found in environment variables")

    # Create fetcher
    fetcher = DasanFetcherWithCache(api_key)

    try:
        # Run discovery
        discovery = RangeDiscovery(fetcher, args.type)
        results = discovery.discover_ranges(
            max_digits=args.max_digits,
            samples_per_digit=args.samples_per_digit
        )

        # Print summary
        discovery.print_summary()

        # Save results
        output_file = args.output or f"data/dasan_api_cache/discovery_{args.type}.json"
        discovery.save_results(Path(output_file))

    finally:
        fetcher.close()


if __name__ == "__main__":
    main()
