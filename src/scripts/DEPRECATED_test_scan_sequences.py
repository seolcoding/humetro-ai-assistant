#!/usr/bin/env python3
"""
Test script to find valid sequence ranges in Dasan API.
Uses the sample sequences we know work to guide the search.
"""

import os
from dotenv import load_dotenv
from scan_dasan_sequences import SequenceScanner


def main():
    load_dotenv()

    api_key = os.getenv("SEOUL_DATA_API_KEY")
    if not api_key:
        raise ValueError("SEOUL_DATA_API_KEY not found in environment variables")

    scanner = SequenceScanner(api_key)

    # Test with known working sequences
    known_faq_seqs = [289801, 289803]
    known_manual_seqs = [289756]

    print("=" * 70)
    print("Testing Known Sequences")
    print("=" * 70)

    print("\nTesting FAQs:")
    for seq in known_faq_seqs:
        exists = scanner.check_sequence("faq", seq)
        print(f"  {seq}: {'✓ EXISTS' if exists else '✗ NOT FOUND'}")

    print("\nTesting Work Manuals:")
    for seq in known_manual_seqs:
        exists = scanner.check_sequence("workmanual", seq)
        print(f"  {seq}: {'✓ EXISTS' if exists else '✗ NOT FOUND'}")

    # Scan around known sequences
    print("\n" + "=" * 70)
    print("Scanning Around Known Sequences")
    print("=" * 70)

    # Scan around FAQ sequences (289801-289803)
    print("\nScanning FAQ range 289700-289900:")
    faq_sequences = scanner.scan_range(
        "faq",
        start=289700,
        end=289900,
        step=1,
        max_workers=5
    )

    print(f"\n✓ Found {len(faq_sequences)} FAQs")
    print(f"Sequences: {faq_sequences}")

    # Scan around manual sequences (289756)
    print("\nScanning Work Manual range 289700-289900:")
    manual_sequences = scanner.scan_range(
        "workmanual",
        start=289700,
        end=289900,
        step=1,
        max_workers=5
    )

    print(f"\n✓ Found {len(manual_sequences)} Work Manuals")
    print(f"Sequences: {manual_sequences}")

    # Save results
    if faq_sequences:
        scanner.save_sequences("faq_test", faq_sequences)

    if manual_sequences:
        scanner.save_sequences("workmanual_test", manual_sequences)


if __name__ == "__main__":
    main()
