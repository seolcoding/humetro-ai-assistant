#!/usr/bin/env python3
"""
Seoul Dasan Call Center API fetcher with persistent cache and resume capability.

Features:
- SQLite database for tracking scanned sequences
- Progress logging with ETA
- Resume capability from last checkpoint
- Detailed statistics and reporting
- Incremental updates
"""

import argparse
import os
import json
import time
import sqlite3
import logging
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dotenv import load_dotenv
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dasan_scan.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DasanCacheDB:
    """SQLite database for caching scan results with thread-safe connections."""

    def __init__(self, db_path: str = "data/dasan_api_cache/scan_cache.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        # Create tables in main thread
        self.create_tables()

    def _get_connection(self):
        """Get a thread-local database connection."""
        if not hasattr(self._local, 'conn') or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path, timeout=30.0)
        return self._local.conn

    def create_tables(self):
        """Create necessary tables if they don't exist."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Table for tracking scanned sequences
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scanned_sequences (
                data_type TEXT NOT NULL,
                seq_no INTEGER NOT NULL,
                is_valid BOOLEAN NOT NULL,
                scanned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (data_type, seq_no)
            )
        """)

        # Table for scan progress tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scan_progress (
                scan_id TEXT PRIMARY KEY,
                data_type TEXT NOT NULL,
                start_seq INTEGER NOT NULL,
                end_seq INTEGER NOT NULL,
                current_seq INTEGER NOT NULL,
                total_valid INTEGER DEFAULT 0,
                total_scanned INTEGER DEFAULT 0,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed BOOLEAN DEFAULT FALSE
            )
        """)

        # Index for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_scanned_sequences_type_valid
            ON scanned_sequences(data_type, is_valid)
        """)

        conn.commit()

    def is_sequence_checked(self, data_type: str, seq_no: int) -> Optional[bool]:
        """
        Check if a sequence has been scanned before.

        Returns:
            True if valid, False if invalid, None if not checked
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT is_valid FROM scanned_sequences WHERE data_type = ? AND seq_no = ?",
            (data_type, seq_no)
        )
        result = cursor.fetchone()
        return result[0] if result else None

    def save_sequence_result(self, data_type: str, seq_no: int, is_valid: bool):
        """Save sequence scan result."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO scanned_sequences (data_type, seq_no, is_valid)
            VALUES (?, ?, ?)
        """, (data_type, seq_no, is_valid))
        conn.commit()

    def get_valid_sequences(self, data_type: str) -> List[int]:
        """Get all valid sequences for a data type."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT seq_no FROM scanned_sequences WHERE data_type = ? AND is_valid = TRUE ORDER BY seq_no",
            (data_type,)
        )
        return [row[0] for row in cursor.fetchall()]

    def get_scan_stats(self, data_type: str) -> Dict:
        """Get scanning statistics for a data type."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                COUNT(*) as total_scanned,
                SUM(CASE WHEN is_valid THEN 1 ELSE 0 END) as total_valid,
                MIN(seq_no) as min_seq,
                MAX(seq_no) as max_seq
            FROM scanned_sequences
            WHERE data_type = ?
        """, (data_type,))

        row = cursor.fetchone()
        return {
            "total_scanned": row[0] or 0,
            "total_valid": row[1] or 0,
            "min_seq": row[2] or 0,
            "max_seq": row[3] or 0
        }

    def save_progress(self, scan_id: str, data_type: str, start_seq: int,
                     end_seq: int, current_seq: int, total_valid: int,
                     total_scanned: int, completed: bool = False):
        """Save scan progress."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO scan_progress
            (scan_id, data_type, start_seq, end_seq, current_seq, total_valid,
             total_scanned, started_at, updated_at, completed)
            VALUES (?, ?, ?, ?, ?, ?, ?,
                    COALESCE((SELECT started_at FROM scan_progress WHERE scan_id = ?), CURRENT_TIMESTAMP),
                    CURRENT_TIMESTAMP, ?)
        """, (scan_id, data_type, start_seq, end_seq, current_seq, total_valid,
              total_scanned, scan_id, completed))
        conn.commit()

    def get_progress(self, scan_id: str) -> Optional[Dict]:
        """Get scan progress."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT * FROM scan_progress WHERE scan_id = ?",
            (scan_id,)
        )
        row = cursor.fetchone()
        if row:
            return {
                "scan_id": row[0],
                "data_type": row[1],
                "start_seq": row[2],
                "end_seq": row[3],
                "current_seq": row[4],
                "total_valid": row[5],
                "total_scanned": row[6],
                "started_at": row[7],
                "updated_at": row[8],
                "completed": row[9]
            }
        return None

    def close(self):
        """Close all database connections."""
        if hasattr(self._local, 'conn') and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None


class DasanFetcherWithCache:
    """Dasan fetcher with caching and progress tracking."""

    BASE_URL = "http://openAPI.seoul.go.kr:8088"

    FAQ_TYPES = {
        "faq": ("F", "SearchDetailsFAQService", "FAQ"),
        "seoul_manual": ("S", "SearchDetailsSeoulWorkmanualService", "서울시 업무매뉴얼"),
        "district_manual": ("J", "SearchDetailsSeoulWorkmanualService", "자치구 업무매뉴얼")
    }

    def __init__(self, api_key: str, cache_dir: str = "data/dasan_api_cache"):
        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db = DasanCacheDB()

        # Statistics
        self.stats = {
            "total_checked": 0,
            "total_valid": 0,
            "cache_hits": 0,
            "api_calls": 0
        }

    def check_sequence(self, data_type: str, seq_no: int, use_cache: bool = True) -> bool:
        """Check if a sequence exists, using cache if available."""
        # Check cache first
        if use_cache:
            cached = self.db.is_sequence_checked(data_type, seq_no)
            if cached is not None:
                self.stats["cache_hits"] += 1
                return cached

        # Make API call
        faq_type, service_name, _ = self.FAQ_TYPES[data_type]
        url = f"{self.BASE_URL}/{self.api_key}/json/{service_name}/1/1/{faq_type}/{seq_no}/"

        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            if service_name in data:
                result = data[service_name].get("RESULT", {})
                is_valid = result.get("CODE") == "INFO-000"

                # Save to cache
                self.db.save_sequence_result(data_type, seq_no, is_valid)
                self.stats["api_calls"] += 1

                return is_valid

            return False
        except Exception as e:
            logger.debug(f"Error checking sequence {seq_no}: {e}")
            return False

    def scan_range_with_progress(
        self,
        data_type: str,
        start: int,
        end: int,
        scan_id: str,
        max_workers: int = 10,
        checkpoint_interval: int = 100,
        resume: bool = True
    ) -> List[int]:
        """
        Scan a range with progress tracking and resume capability.

        Args:
            data_type: Type of data to scan
            start: Start sequence
            end: End sequence
            scan_id: Unique ID for this scan
            max_workers: Number of parallel workers
            checkpoint_interval: Save progress every N sequences
            resume: Whether to resume from last checkpoint
        """
        type_name = self.FAQ_TYPES[data_type][2]

        # Check for existing progress
        if resume:
            progress = self.db.get_progress(scan_id)
            if progress and not progress["completed"]:
                start = progress["current_seq"] + 1
                logger.info(f"Resuming scan from sequence {start}")

        total = end - start + 1
        valid_sequences = []

        logger.info(f"Starting scan: {type_name}")
        logger.info(f"Range: {start:,} to {end:,} ({total:,} sequences)")
        logger.info(f"Workers: {max_workers}")
        logger.info(f"Scan ID: {scan_id}")
        logger.info("=" * 70)

        scan_start_time = time.time()
        last_checkpoint_time = time.time()

        def check_seq(seq: int) -> Tuple[int, bool]:
            is_valid = self.check_sequence(data_type, seq)
            self.stats["total_checked"] += 1
            if is_valid:
                self.stats["total_valid"] += 1
            return (seq, is_valid)

        # Scan in batches for better progress tracking
        batch_size = max_workers * 10
        current = start

        while current <= end:
            batch_end = min(current + batch_size - 1, end)
            batch_sequences = range(current, batch_end + 1)

            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(check_seq, seq): seq
                    for seq in batch_sequences
                }

                for future in as_completed(futures):
                    seq, is_valid = future.result()

                    if is_valid:
                        valid_sequences.append(seq)
                        logger.info(f"✓ Found: {seq}")

                    # Progress reporting
                    processed = self.stats["total_checked"]
                    if processed % checkpoint_interval == 0:
                        elapsed = time.time() - scan_start_time
                        rate = processed / elapsed if elapsed > 0 else 0
                        remaining = total - processed
                        eta_seconds = remaining / rate if rate > 0 else 0
                        eta = timedelta(seconds=int(eta_seconds))

                        percent = (processed / total) * 100

                        logger.info("=" * 70)
                        logger.info(f"Progress: {processed:,}/{total:,} ({percent:.1f}%)")
                        logger.info(f"Valid: {self.stats['total_valid']:,} | Cache hits: {self.stats['cache_hits']:,} | API calls: {self.stats['api_calls']:,}")
                        logger.info(f"Rate: {rate:.1f} seq/s | ETA: {eta}")
                        logger.info("=" * 70)

                        # Save checkpoint
                        self.db.save_progress(
                            scan_id, data_type, start, end, seq,
                            self.stats["total_valid"], processed, False
                        )
                        last_checkpoint_time = time.time()

                    time.sleep(0.05)  # Rate limiting

            current = batch_end + 1

        # Save final progress
        self.db.save_progress(
            scan_id, data_type, start, end, end,
            self.stats["total_valid"], self.stats["total_checked"], True
        )

        total_time = time.time() - scan_start_time
        logger.info("\n" + "=" * 70)
        logger.info("SCAN COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Total time: {timedelta(seconds=int(total_time))}")
        logger.info(f"Total checked: {self.stats['total_checked']:,}")
        logger.info(f"Valid found: {self.stats['total_valid']:,}")
        logger.info(f"Cache hit rate: {(self.stats['cache_hits']/self.stats['total_checked']*100):.1f}%")
        logger.info(f"Average rate: {self.stats['total_checked']/total_time:.1f} seq/s")
        logger.info("=" * 70)

        return sorted(valid_sequences)

    def get_all_valid_sequences(self, data_type: str) -> List[int]:
        """Get all valid sequences from cache."""
        return self.db.get_valid_sequences(data_type)

    def get_stats(self, data_type: str) -> Dict:
        """Get statistics from cache."""
        return self.db.get_scan_stats(data_type)

    def close(self):
        """Close resources."""
        self.db.close()


def main():
    parser = argparse.ArgumentParser(description="Fetch Dasan data with caching and resume")
    parser.add_argument("--type", required=True, choices=["faq", "seoul_manual", "district_manual"],
                       help="Data type to fetch")
    parser.add_argument("--start", type=int, default=1, help="Start sequence")
    parser.add_argument("--end", type=int, default=500000, help="End sequence")
    parser.add_argument("--workers", type=int, default=10, help="Number of parallel workers")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from checkpoint")
    parser.add_argument("--stats-only", action="store_true", help="Only show statistics")

    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("SEOUL_DATA_API_KEY")
    if not api_key:
        raise ValueError("SEOUL_DATA_API_KEY not found in environment variables")

    fetcher = DasanFetcherWithCache(api_key)

    try:
        if args.stats_only:
            # Show statistics only
            stats = fetcher.get_stats(args.type)
            valid_seqs = fetcher.get_all_valid_sequences(args.type)

            print("=" * 70)
            print(f"Statistics for {args.type}")
            print("=" * 70)
            print(f"Total scanned: {stats['total_scanned']:,}")
            print(f"Total valid: {stats['total_valid']:,}")
            print(f"Sequence range: {stats['min_seq']:,} to {stats['max_seq']:,}")
            print(f"Valid sequences: {len(valid_seqs):,}")
            print("=" * 70)
        else:
            # Run scan
            scan_id = f"{args.type}_{args.start}_{args.end}_{int(time.time())}"

            valid_sequences = fetcher.scan_range_with_progress(
                args.type,
                args.start,
                args.end,
                scan_id,
                max_workers=args.workers,
                checkpoint_interval=100,
                resume=not args.no_resume
            )

            # Save results
            output_file = fetcher.cache_dir / f"{args.type}_sequences_{args.start}_{args.end}.json"
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(valid_sequences, f, indent=2)

            logger.info(f"\n✓ Saved {len(valid_sequences)} sequences to {output_file}")

    finally:
        fetcher.close()


if __name__ == "__main__":
    main()
