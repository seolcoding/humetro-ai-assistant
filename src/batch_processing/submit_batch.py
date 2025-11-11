#!/usr/bin/env python3
"""
Gemini Batch API Submission
============================

GCS에 업로드된 JSONL 입력으로 Gemini Batch Job 제출

Requirements:
    pip install google-genai

Usage:
    # Submit batch job
    python submit_batch.py --input gs://rag_bucket_yoon/batch_input.jsonl --output gs://rag_bucket_yoon/batch_output

    # Submit with custom model and config
    python submit_batch.py --input gs://bucket/input.jsonl --output gs://bucket/output --model gemini-2.5-flash --temperature 0.5
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

from google import genai
from google.genai.types import CreateBatchJobConfig

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


class GeminiBatchSubmitter:
    """Gemini Batch API Job 제출 관리자"""

    def __init__(
        self,
        project_id: str = "astute-coda-477522-r1",
        location: str = "us-central1",
        model: str = "gemini-2.5-pro"
    ):
        """
        Args:
            project_id: GCP Project ID
            location: GCP Region (e.g., us-central1, asia-northeast3)
            model: Gemini model name
        """
        self.project_id = project_id
        self.location = location
        self.model = model

        # Initialize client
        self.client = genai.Client(
            vertexai=True,
            project=project_id,
            location=location
        )

        logger.info(f"✅ Gemini Batch Client initialized")
        logger.info(f"  Project: {project_id}")
        logger.info(f"  Location: {location}")
        logger.info(f"  Model: {model}")

    def submit_job(
        self,
        input_uri: str,
        output_uri: str,
        generation_config: Optional[Dict[str, Any]] = None,
        job_name: Optional[str] = None
    ) -> Any:
        """
        Batch Job 제출

        Args:
            input_uri: GCS input path (gs://bucket/path/input.jsonl)
            output_uri: GCS output path (gs://bucket/path/)
            generation_config: Optional generation config override
            job_name: Optional custom job name

        Returns:
            Batch job object
        """
        logger.info("="*70)
        logger.info("Gemini Batch Job Submission")
        logger.info("="*70)

        # Validate URIs
        if not input_uri.startswith("gs://"):
            raise ValueError(f"Input URI must start with gs://: {input_uri}")
        if not output_uri.startswith("gs://"):
            raise ValueError(f"Output URI must start with gs://: {output_uri}")

        logger.info(f"📥 Input:  {input_uri}")
        logger.info(f"📤 Output: {output_uri}")
        logger.info(f"🤖 Model:  {self.model}")

        # Create batch job config
        config = CreateBatchJobConfig(dest=output_uri)

        # Add generation config if provided
        if generation_config:
            config.generation_config = generation_config
            logger.info(f"⚙️  Config: {generation_config}")

        try:
            # Submit batch job
            logger.info("\n🚀 Submitting batch job...")

            job = self.client.batches.create(
                model=self.model,
                src=input_uri,
                config=config
            )

            logger.info("\n✅ Batch job submitted successfully!")
            logger.info("="*70)
            logger.info(f"Job Name: {job.name}")
            logger.info(f"State: {job.state}")
            logger.info(f"Output URI: {output_uri}")
            logger.info("="*70)

            # Save job info
            self._save_job_info(job, input_uri, output_uri)

            logger.info("\n📋 Job Monitoring:")
            logger.info(f"  Check status: gcloud ai batch-prediction-jobs describe {job.name.split('/')[-1]}")
            logger.info(f"  List jobs: gcloud ai batch-prediction-jobs list --region={self.location}")

            return job

        except Exception as e:
            logger.error(f"\n❌ Batch job submission failed: {e}")
            raise

    def _save_job_info(
        self,
        job: Any,
        input_uri: str,
        output_uri: str
    ):
        """Job 정보 저장"""
        job_info = {
            "job_name": job.name,
            "state": str(job.state),
            "model": self.model,
            "project_id": self.project_id,
            "location": self.location,
            "input_uri": input_uri,
            "output_uri": output_uri,
            "submitted_at": datetime.now().isoformat(),
            "job_id": job.name.split('/')[-1]
        }

        # Save to local tracking file
        tracking_dir = Path("data/batch_processing/jobs")
        tracking_dir.mkdir(parents=True, exist_ok=True)

        tracking_file = tracking_dir / f"job_{job_info['job_id']}.json"

        import json
        with open(tracking_file, 'w', encoding='utf-8') as f:
            json.dump(job_info, f, ensure_ascii=False, indent=2)

        logger.info(f"\n💾 Job info saved: {tracking_file}")

    def get_job_status(self, job_name: str) -> Dict[str, Any]:
        """
        Job 상태 조회

        Args:
            job_name: Full job name or job ID

        Returns:
            Job status info
        """
        try:
            job = self.client.batches.get(job_name)

            status = {
                "name": job.name,
                "state": str(job.state),
                "create_time": str(getattr(job, 'create_time', 'N/A')),
                "update_time": str(getattr(job, 'update_time', 'N/A'))
            }

            logger.info(f"Job Status: {status['state']}")
            return status

        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            raise

    def list_jobs(self, limit: int = 10) -> list:
        """
        최근 Batch Job 목록 조회

        Args:
            limit: 조회할 job 수

        Returns:
            Job 목록
        """
        try:
            jobs = list(self.client.batches.list())[:limit]

            logger.info(f"\n📋 Recent {len(jobs)} batch jobs:")
            for job in jobs:
                logger.info(f"  - {job.name}: {job.state}")

            return jobs

        except Exception as e:
            logger.error(f"Failed to list jobs: {e}")
            return []


def main():
    parser = argparse.ArgumentParser(
        description="Submit Gemini Batch API job",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required
    parser.add_argument(
        "--input",
        required=True,
        help="GCS input URI (gs://bucket/path/input.jsonl)"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="GCS output URI prefix (gs://bucket/path/)"
    )

    # Optional
    parser.add_argument(
        "--project-id",
        default="astute-coda-477522-r1",
        help="GCP Project ID (default: astute-coda-477522-r1)"
    )
    parser.add_argument(
        "--location",
        default="us-central1",
        help="GCP Region (default: us-central1)"
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-pro",
        help="Gemini model (default: gemini-2.5-pro)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature override (0.0-1.0)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        help="Max output tokens override"
    )

    # Actions
    parser.add_argument(
        "--status",
        help="Get job status by job name/ID"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List recent batch jobs"
    )

    args = parser.parse_args()

    # Initialize submitter
    submitter = GeminiBatchSubmitter(
        project_id=args.project_id,
        location=args.location,
        model=args.model
    )

    # Handle actions
    if args.status:
        submitter.get_job_status(args.status)
        return

    if args.list:
        submitter.list_jobs()
        return

    # Submit new job
    generation_config = {}
    if args.temperature is not None:
        generation_config["temperature"] = args.temperature
    if args.max_tokens is not None:
        generation_config["max_output_tokens"] = args.max_tokens

    job = submitter.submit_job(
        input_uri=args.input,
        output_uri=args.output,
        generation_config=generation_config if generation_config else None
    )

    print(f"\n✅ Job submitted: {job.name}")
    print(f"Monitor at: https://console.cloud.google.com/vertex-ai/batch-predictions")


if __name__ == "__main__":
    main()
