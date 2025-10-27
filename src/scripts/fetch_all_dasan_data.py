#!/usr/bin/env python3
"""
Complete script to fetch all data from Seoul Dasan Call Center APIs.

This script uses the list APIs to get all available sequence numbers,
then fetches the detailed information for each item.

Related APIs mentioned in the documentation:
1. 서울시 다산콜센터 자주묻는질문 목록 조회 - List API
2. 서울시 다산콜센터 자주묻는질문 FAQ 상세 조회 - Detail API (SearchDetailsFAQService)
3. 서울시 다산콜센터 자주묻는질문 업무매뉴얼 상세 조회 - Detail API (SearchDetailsSeoulWorkmanualService)
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
import requests


class CompleteDasanAPIClient:
    """Complete client for Seoul Dasan Call Center Open API with list and detail endpoints."""

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

    def _fetch_data(self, url: str, max_retries: int = 3) -> Optional[Dict]:
        """
        Fetch data from API with retry logic.

        Args:
            url: API URL to fetch
            max_retries: Maximum number of retry attempts

        Returns:
            Parsed JSON response or None if not found
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
                        return None
                    elif result_code != "INFO-000":
                        print(f"API Warning: {data['RESULT'].get('MESSAGE')}")
                        return None

                return data

            except requests.RequestException as e:
                if attempt == max_retries - 1:
                    print(f"Request failed: {e}")
                    return None
                time.sleep(2 ** attempt)

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

    def fetch_faq_list(
        self,
        start_idx: int = 1,
        end_idx: int = 1000,
        use_cache: bool = True
    ) -> Optional[Dict]:
        """
        Fetch FAQ list using the list API.

        Service name might be: SearchFAQKnowlege, ListFAQService, or similar.
        We'll try common naming patterns.

        Args:
            start_idx: Start index for pagination
            end_idx: End index for pagination
            use_cache: Whether to use cached data if available

        Returns:
            FAQ list data or None
        """
        cache_filename = f"faq_list_{start_idx}_{end_idx}.json"

        if use_cache:
            cached = self._load_from_cache(cache_filename)
            if cached:
                return cached

        # Try different possible service names
        possible_services = [
            "SearchFAQKnowlegeService",
            "ListFAQService",
            "SearchFAQListService",
            "FAQListService"
        ]

        for service_name in possible_services:
            url = f"{self.BASE_URL}/{self.api_key}/json/{service_name}/{start_idx}/{end_idx}/"
            print(f"Trying: {service_name}...")

            data = self._fetch_data(url)
            if data:
                self._save_to_cache(data, cache_filename)
                print(f"✓ Success with {service_name}")
                return data

        print("Could not find FAQ list API")
        return None

    def fetch_workmanual_list(
        self,
        start_idx: int = 1,
        end_idx: int = 1000,
        use_cache: bool = True
    ) -> Optional[Dict]:
        """
        Fetch work manual list using the list API.

        Args:
            start_idx: Start index for pagination
            end_idx: End index for pagination
            use_cache: Whether to use cached data if available

        Returns:
            Work manual list data or None
        """
        cache_filename = f"workmanual_list_{start_idx}_{end_idx}.json"

        if use_cache:
            cached = self._load_from_cache(cache_filename)
            if cached:
                return cached

        # Try different possible service names
        possible_services = [
            "SearchSeoulWorkmanualService",
            "ListWorkmanualService",
            "WorkmanualListService"
        ]

        for service_name in possible_services:
            url = f"{self.BASE_URL}/{self.api_key}/json/{service_name}/{start_idx}/{end_idx}/"
            print(f"Trying: {service_name}...")

            data = self._fetch_data(url)
            if data:
                self._save_to_cache(data, cache_filename)
                print(f"✓ Success with {service_name}")
                return data

        print("Could not find work manual list API")
        return None

    def fetch_all_with_pagination(
        self,
        service_type: str,
        batch_size: int = 1000,
        max_items: Optional[int] = None,
        use_cache: bool = True,
        delay: float = 1.0
    ) -> List[Dict]:
        """
        Fetch all items using pagination.

        Args:
            service_type: Type of service ('faq' or 'workmanual')
            batch_size: Number of items per batch (max 1000)
            max_items: Maximum number of items to fetch (None for all)
            use_cache: Whether to use cached data if available
            delay: Delay between API calls in seconds

        Returns:
            List of all items
        """
        all_items = []
        start_idx = 1

        print(f"\nFetching {service_type} list with pagination...")

        while True:
            end_idx = start_idx + batch_size - 1

            if max_items and end_idx > max_items:
                end_idx = max_items

            print(f"Batch: {start_idx}-{end_idx}")

            # Fetch list data
            if service_type == "faq":
                data = self.fetch_faq_list(start_idx, end_idx, use_cache)
            else:
                data = self.fetch_workmanual_list(start_idx, end_idx, use_cache)

            if not data:
                break

            # Extract items - the exact key might vary
            items = None
            for key in data.keys():
                if key != "RESULT" and "row" in data[key]:
                    items = data[key]["row"]
                    total_count = data[key].get("list_total_count", 0)
                    break

            if items:
                if isinstance(items, dict):
                    items = [items]
                all_items.extend(items)

                print(f"  → Got {len(items)} items (total so far: {len(all_items)})")

                # Check if we've reached the end
                if end_idx >= total_count or (max_items and len(all_items) >= max_items):
                    break
            else:
                break

            start_idx += batch_size
            time.sleep(delay)

        print(f"\nTotal {service_type}s fetched: {len(all_items)}")
        return all_items

    def fetch_detail(
        self,
        service_type: str,
        seq_no: str,
        use_cache: bool = True
    ) -> Optional[Dict]:
        """
        Fetch detailed information for a specific item.

        Args:
            service_type: Type of service ('faq' or 'workmanual')
            seq_no: Sequence number
            use_cache: Whether to use cached data if available

        Returns:
            Detail data or None
        """
        if service_type == "faq":
            service_name = "SearchDetailsFAQService"
            faq_type = "F"
        else:
            service_name = "SearchDetailsSeoulWorkmanualService"
            faq_type = "S"

        cache_filename = f"{service_type}_detail_{seq_no}.json"

        if use_cache:
            cached = self._load_from_cache(cache_filename)
            if cached:
                return cached

        url = f"{self.BASE_URL}/{self.api_key}/json/{service_name}/1/1/{faq_type}/{seq_no}/"
        data = self._fetch_data(url)

        if data:
            self._save_to_cache(data, cache_filename)

        return data


def main():
    """Main execution function."""
    load_dotenv()

    api_key = os.getenv("SEOUL_DATA_API_KEY")
    if not api_key:
        raise ValueError("SEOUL_DATA_API_KEY not found in environment variables")

    client = CompleteDasanAPIClient(api_key)

    print("=" * 70)
    print("Seoul Dasan Call Center Complete Data Fetcher")
    print("=" * 70)

    # Try to fetch FAQ list
    print("\n[1/2] Attempting to fetch FAQ list...")
    print("-" * 70)
    faq_list = client.fetch_all_with_pagination(
        "faq",
        batch_size=1000,
        use_cache=True,
        delay=1.0
    )

    if faq_list:
        # Save FAQ list
        faq_list_file = client.cache_dir / "all_faq_list.json"
        with open(faq_list_file, "w", encoding="utf-8") as f:
            json.dump(faq_list, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Saved FAQ list: {faq_list_file}")

        # Extract sequence numbers and fetch details
        print("\nFetching FAQ details...")
        seq_numbers = [str(int(item.get("FAQ_SEQNO", 0))) for item in faq_list if "FAQ_SEQNO" in item]
        print(f"Found {len(seq_numbers)} FAQ sequence numbers")

        # Fetch first 10 details as sample
        if seq_numbers:
            sample_size = min(10, len(seq_numbers))
            print(f"\nFetching first {sample_size} FAQ details as sample...")
            for i, seq_no in enumerate(seq_numbers[:sample_size], 1):
                print(f"  [{i}/{sample_size}] FAQ {seq_no}...", end="")
                detail = client.fetch_detail("faq", seq_no, use_cache=True)
                print(" ✓" if detail else " ✗")
                time.sleep(0.5)

    # Try to fetch work manual list
    print("\n[2/2] Attempting to fetch work manual list...")
    print("-" * 70)
    workmanual_list = client.fetch_all_with_pagination(
        "workmanual",
        batch_size=1000,
        use_cache=True,
        delay=1.0
    )

    if workmanual_list:
        # Save work manual list
        workmanual_list_file = client.cache_dir / "all_workmanual_list.json"
        with open(workmanual_list_file, "w", encoding="utf-8") as f:
            json.dump(workmanual_list, f, ensure_ascii=False, indent=2)
        print(f"\n✓ Saved work manual list: {workmanual_list_file}")

        # Extract sequence numbers and fetch details
        print("\nFetching work manual details...")
        seq_numbers = [str(int(item.get("FAQ_SEQNO", 0))) for item in workmanual_list if "FAQ_SEQNO" in item]
        print(f"Found {len(seq_numbers)} work manual sequence numbers")

        # Fetch first 10 details as sample
        if seq_numbers:
            sample_size = min(10, len(seq_numbers))
            print(f"\nFetching first {sample_size} work manual details as sample...")
            for i, seq_no in enumerate(seq_numbers[:sample_size], 1):
                print(f"  [{i}/{sample_size}] Manual {seq_no}...", end="")
                detail = client.fetch_detail("workmanual", seq_no, use_cache=True)
                print(" ✓" if detail else " ✗")
                time.sleep(0.5)

    # Print summary
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"FAQ list items: {len(faq_list) if faq_list else 0}")
    print(f"Work manual list items: {len(workmanual_list) if workmanual_list else 0}")
    print(f"Cache directory: {client.cache_dir}")

    if not faq_list and not workmanual_list:
        print("\n" + "=" * 70)
        print("NOTE: Could not find list APIs.")
        print("The APIs may have different names or require additional parameters.")
        print("You may need to:")
        print("1. Check the Seoul Open Data Portal for the correct service names")
        print("2. Contact the API provider for documentation")
        print("3. Use the discovery method to find valid sequence numbers")
        print("=" * 70)


if __name__ == "__main__":
    main()
