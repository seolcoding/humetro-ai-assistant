#!/usr/bin/env python3
"""
Script to fetch and cache data from Seoul Dasan Call Center APIs.

This script fetches FAQ details (자주묻는질문) from SearchDetailsFAQService.

Note: The Seoul Open API only provides detail endpoints that require specific sequence numbers.
To fetch multiple FAQs, you need to know the sequence numbers in advance or iterate through them.

Based on the API documentation:
- SearchDetailsFAQService: FAQ detail endpoint (requires FAQ_SEQNO with FAQ_TP='F')
- SearchDetailsSeoulWorkmanualService: Work manual detail endpoint (requires FAQ_SEQNO with FAQ_TP='S')
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
import requests


class DasanAPIClient:
    """Client for Seoul Dasan Call Center Open API."""

    BASE_URL = "http://openAPI.seoul.go.kr:8088"

    def __init__(self, api_key: str, cache_dir: str = "data/dasan_api_cache"):
        """
        Initialize the API client.

        Args:
            api_key: Seoul Open API authentication key
            cache_dir: Directory to store cached API responses
        """
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # API endpoints
        self.services = {
            "workmanual": "SearchDetailsSeoulWorkmanualService",
            "faq": "SearchDetailsFAQService"
        }

    def _build_url(
        self,
        service: str,
        faq_type: str,
        seq_no: str
    ) -> str:
        """
        Build API request URL.

        According to the API docs:
        http://openAPI.seoul.go.kr:8088/{KEY}/json/{SERVICE}/1/1/{FAQ_TP}/{FAQ_SEQNO}/

        Args:
            service: Service name (workmanual or faq)
            faq_type: FAQ type (S for workmanual, F for faq)
            seq_no: Sequence number for specific item

        Returns:
            Complete API URL
        """
        service_name = self.services[service]
        url = f"{self.BASE_URL}/{self.api_key}/json/{service_name}/1/1/{faq_type}/{seq_no}/"
        return url

    def _fetch_data(self, url: str, max_retries: int = 3) -> Optional[Dict]:
        """
        Fetch data from API with retry logic.

        Args:
            url: API URL to fetch
            max_retries: Maximum number of retry attempts

        Returns:
            Parsed JSON response or None if not found

        Raises:
            requests.RequestException: If request fails after all retries
        """
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=30)
                response.raise_for_status()

                data = response.json()

                # Check for API errors
                if "RESULT" in data:
                    result_code = data["RESULT"].get("CODE")

                    if result_code == "INFO-200":
                        # No data found for this sequence number
                        return None
                    elif result_code != "INFO-000":
                        print(f"API Warning: {data['RESULT'].get('MESSAGE')}")
                        return None

                return data

            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"Request failed after {max_retries} attempts: {e}")
                    return None
                time.sleep(2 ** attempt)  # Exponential backoff

        return None

    def _save_to_cache(self, data: Dict, filename: str) -> None:
        """Save data to cache file."""
        cache_file = self.cache_dir / filename
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_from_cache(self, filename: str) -> Optional[Dict]:
        """Load data from cache file if it exists."""
        cache_file = self.cache_dir / filename
        if cache_file.exists():
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def fetch_workmanual_detail(
        self,
        seq_no: str,
        use_cache: bool = True
    ) -> Optional[Dict]:
        """
        Fetch detailed work manual by sequence number.

        Args:
            seq_no: Sequence number of the work manual
            use_cache: Whether to use cached data if available

        Returns:
            Work manual detail data or None if not found
        """
        cache_filename = f"workmanual_detail_{seq_no}.json"

        if use_cache:
            cached = self._load_from_cache(cache_filename)
            if cached:
                return cached

        url = self._build_url("workmanual", "S", seq_no)
        data = self._fetch_data(url)

        if data:
            self._save_to_cache(data, cache_filename)

        return data

    def fetch_faq_detail(
        self,
        seq_no: str,
        use_cache: bool = True
    ) -> Optional[Dict]:
        """
        Fetch detailed FAQ by sequence number.

        Args:
            seq_no: Sequence number of the FAQ
            use_cache: Whether to use cached data if available

        Returns:
            FAQ detail data or None if not found
        """
        cache_filename = f"faq_detail_{seq_no}.json"

        if use_cache:
            cached = self._load_from_cache(cache_filename)
            if cached:
                return cached

        url = self._build_url("faq", "F", seq_no)
        data = self._fetch_data(url)

        if data:
            self._save_to_cache(data, cache_filename)

        return data

    def fetch_multiple_faqs(
        self,
        seq_numbers: List[str],
        use_cache: bool = True,
        delay: float = 0.5
    ) -> List[Dict]:
        """
        Fetch multiple FAQs by sequence numbers.

        Args:
            seq_numbers: List of sequence numbers to fetch
            use_cache: Whether to use cached data if available
            delay: Delay between API calls in seconds

        Returns:
            List of FAQ items (skips items that are not found)
        """
        faqs = []

        for i, seq_no in enumerate(seq_numbers, 1):
            print(f"Fetching FAQ {i}/{len(seq_numbers)}: {seq_no}", end="")

            data = self.fetch_faq_detail(seq_no, use_cache)

            if data and "SearchDetailsFAQService" in data:
                service_data = data["SearchDetailsFAQService"]
                if "row" in service_data:
                    faqs.append(service_data["row"])
                    print(" ✓")
                else:
                    print(" - no data")
            else:
                print(" - not found")

            if i < len(seq_numbers):
                time.sleep(delay)

        return faqs

    def fetch_multiple_workmanuals(
        self,
        seq_numbers: List[str],
        use_cache: bool = True,
        delay: float = 0.5
    ) -> List[Dict]:
        """
        Fetch multiple work manuals by sequence numbers.

        Args:
            seq_numbers: List of sequence numbers to fetch
            use_cache: Whether to use cached data if available
            delay: Delay between API calls in seconds

        Returns:
            List of work manual items (skips items that are not found)
        """
        manuals = []

        for i, seq_no in enumerate(seq_numbers, 1):
            print(f"Fetching work manual {i}/{len(seq_numbers)}: {seq_no}", end="")

            data = self.fetch_workmanual_detail(seq_no, use_cache)

            if data and "SearchDetailsSeoulWorkmanualService" in data:
                service_data = data["SearchDetailsSeoulWorkmanualService"]
                if "row" in service_data:
                    manuals.append(service_data["row"])
                    print(" ✓")
                else:
                    print(" - no data")
            else:
                print(" - not found")

            if i < len(seq_numbers):
                time.sleep(delay)

        return manuals

    def discover_sequence_numbers(
        self,
        service_type: str,
        start_seq: int = 1,
        end_seq: int = 300000,
        sample_interval: int = 1000,
        delay: float = 0.5
    ) -> List[str]:
        """
        Discover valid sequence numbers by sampling.

        Since we don't have a list API, we need to sample sequence numbers
        to find valid ones. This is a discovery method.

        Args:
            service_type: Type of service ('faq' or 'workmanual')
            start_seq: Starting sequence number
            end_seq: Ending sequence number
            sample_interval: Interval between sampled sequence numbers
            delay: Delay between API calls in seconds

        Returns:
            List of valid sequence numbers found
        """
        valid_seq_numbers = []
        faq_type = "F" if service_type == "faq" else "S"

        print(f"\nDiscovering valid sequence numbers for {service_type}...")
        print(f"Sampling range {start_seq}-{end_seq} with interval {sample_interval}")

        for seq in range(start_seq, end_seq + 1, sample_interval):
            seq_str = str(seq)

            if service_type == "faq":
                data = self.fetch_faq_detail(seq_str, use_cache=True)
            else:
                data = self.fetch_workmanual_detail(seq_str, use_cache=True)

            if data:
                valid_seq_numbers.append(seq_str)
                print(f"✓ Found valid sequence: {seq_str}")

            time.sleep(delay)

        print(f"\nDiscovered {len(valid_seq_numbers)} valid sequence numbers")
        return valid_seq_numbers


def main():
    """Main execution function."""
    # Load environment variables
    load_dotenv()

    api_key = os.getenv("SEOUL_DATA_API_KEY")
    if not api_key:
        raise ValueError("SEOUL_DATA_API_KEY not found in environment variables")

    # Initialize client
    client = DasanAPIClient(api_key)

    print("=" * 70)
    print("Seoul Dasan Call Center API Data Fetcher")
    print("=" * 70)

    # Test with sample sequence numbers from the API documentation
    print("\n[1/2] Fetching Sample Work Manuals...")
    print("-" * 70)

    sample_workmanual_seqs = ["289808", "289756"]  # From API docs examples
    workmanuals = client.fetch_multiple_workmanuals(
        sample_workmanual_seqs,
        use_cache=True,
        delay=1.0
    )

    # Save consolidated work manuals
    if workmanuals:
        workmanual_file = client.cache_dir / "sample_workmanuals.json"
        with open(workmanual_file, "w", encoding="utf-8") as f:
            json.dump(workmanuals, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Saved sample work manuals: {workmanual_file}")

    # Fetch sample FAQs
    print("\n[2/2] Fetching Sample FAQs...")
    print("-" * 70)

    sample_faq_seqs = ["289803", "289801"]  # From API docs examples
    faqs = client.fetch_multiple_faqs(
        sample_faq_seqs,
        use_cache=True,
        delay=1.0
    )

    # Save consolidated FAQs
    if faqs:
        faq_file = client.cache_dir / "sample_faqs.json"
        with open(faq_file, "w", encoding="utf-8") as f:
            json.dump(faqs, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Saved sample FAQs: {faq_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Work Manuals fetched: {len(workmanuals)}")
    print(f"FAQs fetched: {len(faqs)}")
    print(f"Cache directory: {client.cache_dir}")
    print("\n" + "=" * 70)
    print("Note: This API requires specific sequence numbers.")
    print("To fetch more data, you need to:")
    print("1. Use the discovery method to find valid sequence numbers")
    print("2. Provide a list of known sequence numbers")
    print("3. Use the related list APIs to get available sequence numbers")
    print("=" * 70)


if __name__ == "__main__":
    main()
