"""
Gemini Batch API Pipeline - Google Colab Version
=================================================

전체 파이프라인을 Colab에서 실행하기 위한 통합 코드

실행 순서:
1. 환경 설정 및 인증
2. 테스트셋 로드 및 샘플링
3. Batch 입력 JSONL 생성
4. GCS 업로드
5. Batch Job 제출
6. (수동) Job 완료 대기
7. 결과 다운로드 및 파싱

Usage in Colab:
    1. 이 파일 전체를 Colab 셀에 복사
    2. 각 단계별로 실행
    3. 또는 전체 자동 실행
"""

# ============================================================================
# STEP 0: 라이브러리 설치 및 임포트
# ============================================================================

# 설치 (Colab에서 최초 1회 실행)
def install_dependencies():
    """필요한 라이브러리 설치"""
    print("📦 Installing dependencies...")
    !pip install --quiet --upgrade google-genai google-cloud-storage pandas

install_dependencies()


import json
import random
from datetime import datetime
from pathlib import Path
import pandas as pd
from google import genai
from google.genai.types import CreateBatchJobConfig
from google.cloud import storage


# ============================================================================
# STEP 1: GCP 인증
# ============================================================================

def authenticate_gcp():
    """
    GCP 인증

    Colab에서 실행 시:
    1. 구글 계정으로 로그인
    2. 프로젝트 접근 권한 허용
    """
    print("🔐 Authenticating with GCP...")

    from google.colab import auth
    auth.authenticate_user()

    print("✅ Authentication complete!")

# Colab에서 실행
authenticate_gcp()


# ============================================================================
# STEP 2: 설정
# ============================================================================

class Config:
    """파이프라인 설정"""

    # GCP 설정
    PROJECT_ID = "astute-coda-477522-r1"
    LOCATION = "us-central1"
    BUCKET_NAME = "rag_bucket_yoon"

    # 모델 설정
    MODEL = "gemini-2.5-pro"
    TEMPERATURE = 0.7
    MAX_OUTPUT_TOKENS = 2048

    # 데이터 설정
    SAMPLE_SIZE = 50  # 테스트용 질문 수
    RANDOM_SEED = 42

    # 타임스탬프
    TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 파일 경로
    INPUT_JSONL = f"batch_input_{TIMESTAMP}.jsonl"
    OUTPUT_PREFIX = f"batch_output_{TIMESTAMP}"

    @classmethod
    def print_config(cls):
        """설정 출력"""
        print("="*70)
        print("Pipeline Configuration")
        print("="*70)
        print(f"Project ID: {cls.PROJECT_ID}")
        print(f"Location: {cls.LOCATION}")
        print(f"Bucket: {cls.BUCKET_NAME}")
        print(f"Model: {cls.MODEL}")
        print(f"Sample Size: {cls.SAMPLE_SIZE}")
        print(f"Timestamp: {cls.TIMESTAMP}")
        print("="*70)

Config.print_config()


# ============================================================================
# STEP 3: 테스트셋 업로드 및 로드
# ============================================================================

def upload_and_load_testset():
    """
    Colab에서 테스트셋 업로드 및 로드

    방법 1: 파일 업로드 (권장)
    방법 2: Google Drive 마운트
    """
    print("\n📤 Upload your testset CSV file...")

    from google.colab import files
    uploaded = files.upload()

    # 업로드된 파일 찾기
    csv_files = [f for f in uploaded.keys() if f.endswith('.csv')]

    if not csv_files:
        raise ValueError("No CSV file uploaded!")

    testset_file = csv_files[0]
    print(f"✅ Loaded: {testset_file}")

    # CSV 읽기
    df = pd.read_csv(testset_file)
    print(f"📊 Total questions: {len(df)}")
    print(f"📋 Columns: {df.columns.tolist()}")

    return df, testset_file

# 테스트셋 로드
testset_df, testset_filename = upload_and_load_testset()


# ============================================================================
# STEP 4: 질문 샘플링 및 JSONL 생성
# ============================================================================

def create_batch_input_jsonl(df, sample_size=50, random_seed=42):
    """
    Batch API 입력 JSONL 생성

    Args:
        df: 테스트셋 DataFrame
        sample_size: 샘플링할 질문 수
        random_seed: 랜덤 시드

    Returns:
        생성된 JSONL 파일 경로
    """
    print("\n" + "="*70)
    print("Creating Batch Input JSONL")
    print("="*70)

    # 질문 컬럼 찾기
    question_col = None
    for col in ['question', 'user_input', 'query']:
        if col in df.columns:
            question_col = col
            break

    if not question_col:
        raise ValueError(f"No question column found in: {df.columns.tolist()}")

    # 샘플링
    if sample_size and sample_size < len(df):
        df_sample = df.sample(n=sample_size, random_state=random_seed)
        print(f"🎲 Sampled {sample_size} questions from {len(df)}")
    else:
        df_sample = df
        print(f"📝 Using all {len(df)} questions")

    questions = df_sample[question_col].tolist()

    # 시스템 프롬프트
    system_prompt = """당신은 서울 대중교통 정보를 제공하는 전문 AI 어시스턴트입니다.

역할:
- 서울시 지하철, 버스, 택시 등 대중교통 관련 질문에 답변
- 정확하고 간결한 한국어 응답 제공
- 사실 기반 정보만 제공

응답 원칙:
1. 정확한 정보만 제공
2. 불확실한 경우 "확인 필요" 명시
3. 간결하고 이해하기 쉬운 한국어 사용"""

    # JSONL 생성
    output_file = Config.INPUT_JSONL

    with open(output_file, 'w', encoding='utf-8') as f:
        for question in questions:
            user_prompt = f"""질문: {question}

답변:"""

            request = {
                "request": {
                    "contents": [
                        {
                            "role": "user",
                            "parts": [{"text": f"{system_prompt}\n\n{user_prompt}"}]
                        }
                    ],
                    "generationConfig": {
                        "temperature": Config.TEMPERATURE,
                        "maxOutputTokens": Config.MAX_OUTPUT_TOKENS
                    }
                }
            }

            f.write(json.dumps(request, ensure_ascii=False) + '\n')

    print(f"✅ JSONL created: {output_file}")
    print(f"📊 Total requests: {len(questions)}")

    # 메타데이터 저장
    metadata = {
        "total_questions": len(questions),
        "sample_size": sample_size,
        "random_seed": random_seed,
        "model": Config.MODEL,
        "temperature": Config.TEMPERATURE,
        "max_output_tokens": Config.MAX_OUTPUT_TOKENS,
        "created_at": datetime.now().isoformat()
    }

    metadata_file = output_file.replace('.jsonl', '_metadata.json')
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"💾 Metadata saved: {metadata_file}")

    return output_file, metadata

# JSONL 생성
input_jsonl, metadata = create_batch_input_jsonl(
    testset_df,
    sample_size=Config.SAMPLE_SIZE,
    random_seed=Config.RANDOM_SEED
)


# ============================================================================
# STEP 5: GCS 업로드
# ============================================================================

def upload_to_gcs(local_file, bucket_name, blob_name=None):
    """
    GCS에 파일 업로드

    Args:
        local_file: 로컬 파일 경로
        bucket_name: GCS 버킷 이름
        blob_name: GCS 객체 이름 (None이면 local_file 이름 사용)

    Returns:
        GCS URI (gs://bucket/blob)
    """
    print(f"\n📤 Uploading to GCS...")

    storage_client = storage.Client(project=Config.PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)

    blob_name = blob_name or Path(local_file).name
    blob = bucket.blob(blob_name)

    blob.upload_from_filename(local_file)

    gcs_uri = f"gs://{bucket_name}/{blob_name}"
    print(f"✅ Uploaded: {gcs_uri}")

    return gcs_uri

# GCS 업로드
gcs_input_uri = upload_to_gcs(input_jsonl, Config.BUCKET_NAME)


# ============================================================================
# STEP 6: Batch Job 제출
# ============================================================================

def submit_batch_job(input_uri, output_prefix):
    """
    Gemini Batch Job 제출

    Args:
        input_uri: GCS input URI (gs://bucket/input.jsonl)
        output_prefix: GCS output prefix (gs://bucket/output_prefix)

    Returns:
        Batch job object
    """
    print("\n" + "="*70)
    print("Submitting Gemini Batch Job")
    print("="*70)

    # Initialize client
    client = genai.Client(
        vertexai=True,
        project=Config.PROJECT_ID,
        location=Config.LOCATION
    )

    output_uri = f"gs://{Config.BUCKET_NAME}/{output_prefix}"

    print(f"📥 Input:  {input_uri}")
    print(f"📤 Output: {output_uri}")
    print(f"🤖 Model:  {Config.MODEL}")

    # Create batch job
    config = CreateBatchJobConfig(dest=output_uri)

    job = client.batches.create(
        model=Config.MODEL,
        src=input_uri,
        config=config
    )

    print("\n✅ Batch job submitted!")
    print("="*70)
    print(f"Job Name: {job.name}")
    print(f"State: {job.state}")
    print(f"Job ID: {job.name.split('/')[-1]}")
    print("="*70)

    # Save job info
    job_info = {
        "job_name": job.name,
        "job_id": job.name.split('/')[-1],
        "state": str(job.state),
        "model": Config.MODEL,
        "input_uri": input_uri,
        "output_uri": output_uri,
        "submitted_at": datetime.now().isoformat()
    }

    job_file = f"job_info_{Config.TIMESTAMP}.json"
    with open(job_file, 'w', encoding='utf-8') as f:
        json.dump(job_info, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Job info saved: {job_file}")

    return job, job_info

# Batch Job 제출
batch_job, job_info = submit_batch_job(gcs_input_uri, Config.OUTPUT_PREFIX)


# ============================================================================
# STEP 7: Job 상태 확인 (선택적)
# ============================================================================

def check_job_status(job_name):
    """
    Batch Job 상태 확인

    Args:
        job_name: Full job name

    Returns:
        Job status dict
    """
    client = genai.Client(
        vertexai=True,
        project=Config.PROJECT_ID,
        location=Config.LOCATION
    )

    job = client.batches.get(job_name)

    status = {
        "name": job.name,
        "state": str(job.state),
        "create_time": str(getattr(job, 'create_time', 'N/A')),
        "update_time": str(getattr(job, 'update_time', 'N/A'))
    }

    print(f"\n📊 Job Status: {status['state']}")

    return status

# 상태 확인 (필요시 실행)
# status = check_job_status(batch_job.name)


# ============================================================================
# STEP 8: 결과 다운로드 및 파싱 (Job 완료 후 실행)
# ============================================================================

def download_batch_results(bucket_name, output_prefix, local_dir="."):
    """
    GCS에서 Batch 결과 다운로드

    Args:
        bucket_name: GCS 버킷 이름
        output_prefix: 출력 prefix
        local_dir: 로컬 저장 디렉토리

    Returns:
        다운로드된 파일 목록
    """
    print(f"\n📥 Downloading batch results from GCS...")

    storage_client = storage.Client(project=Config.PROJECT_ID)
    bucket = storage_client.bucket(bucket_name)

    # List blobs with prefix
    blobs = bucket.list_blobs(prefix=output_prefix)

    downloaded_files = []
    for blob in blobs:
        if blob.name.endswith('.jsonl'):
            local_file = Path(local_dir) / Path(blob.name).name
            blob.download_to_filename(str(local_file))
            downloaded_files.append(str(local_file))
            print(f"  ✅ {blob.name} -> {local_file}")

    if not downloaded_files:
        print("⚠️ No JSONL files found. Job may not be complete yet.")

    return downloaded_files


def parse_batch_output(output_jsonl, original_testset_df):
    """
    Batch 출력 파싱

    Args:
        output_jsonl: Batch output JSONL 파일
        original_testset_df: 원본 테스트셋 DataFrame

    Returns:
        파싱된 결과 dict
    """
    print(f"\n" + "="*70)
    print("Parsing Batch Output")
    print("="*70)

    questions = []
    answers = []

    stats = {
        "total": 0,
        "success": 0,
        "failed": 0,
        "max_tokens": 0
    }

    with open(output_jsonl, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                stats["total"] += 1

                # Extract question from request
                request = data.get("request", {})
                contents = request.get("contents", [])
                if contents:
                    parts = contents[0].get("parts", [])
                    if parts:
                        full_text = parts[0].get("text", "")
                        if "질문:" in full_text:
                            question = full_text.split("질문:")[-1].split("답변:")[0].strip()
                        else:
                            question = ""

                # Extract answer from response
                response = data.get("response", {})
                candidates = response.get("candidates", [])

                if candidates:
                    candidate = candidates[0]
                    content = candidate.get("content", {})
                    parts = content.get("parts", [])

                    if parts:
                        answer = parts[0].get("text", "")

                        if answer:
                            questions.append(question)
                            answers.append(answer)
                            stats["success"] += 1
                        else:
                            stats["failed"] += 1
                    else:
                        stats["failed"] += 1

                    # Check finish reason
                    if candidate.get("finishReason") == "MAX_TOKENS":
                        stats["max_tokens"] += 1
                else:
                    stats["failed"] += 1

            except Exception as e:
                print(f"⚠️ Error at line {line_num}: {e}")
                stats["failed"] += 1
                continue

    # Get references from testset
    references = []
    question_col = 'question' if 'question' in original_testset_df.columns else 'user_input'
    reference_col = None
    for col in ['reference', 'reference_answer', 'ground_truth']:
        if col in original_testset_df.columns:
            reference_col = col
            break

    if reference_col:
        testset_questions = original_testset_df[question_col].tolist()
        testset_references = original_testset_df[reference_col].tolist()

        for q in questions:
            if q in testset_questions:
                idx = testset_questions.index(q)
                references.append(testset_references[idx])
            else:
                references.append("")
    else:
        references = [""] * len(questions)

    # Print statistics
    print(f"📊 Statistics:")
    print(f"  Total: {stats['total']}")
    print(f"  Success: {stats['success']}")
    print(f"  Failed: {stats['failed']}")
    print(f"  Max tokens: {stats['max_tokens']}")
    print(f"\n📈 Results:")
    print(f"  Questions: {len(questions)}")
    print(f"  Answers: {len(answers)}")
    print(f"  References: {len([r for r in references if r])}")

    result = {
        "questions": questions,
        "answers": answers,
        "references": references,
        "stats": stats
    }

    # Save results
    output_file = f"parsed_results_{Config.TIMESTAMP}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n💾 Results saved: {output_file}")

    # Create RAGAS format
    ragas_dataset = {
        "Gemini-2.5-Pro": {
            "user_input": questions,
            "response": answers,
            "reference": references,
            "retrieved_contexts": [[] for _ in questions]
        }
    }

    ragas_file = f"ragas_dataset_{Config.TIMESTAMP}.json"
    with open(ragas_file, 'w', encoding='utf-8') as f:
        json.dump(ragas_dataset, f, ensure_ascii=False, indent=2)

    print(f"💾 RAGAS dataset saved: {ragas_file}")

    return result, ragas_dataset


# ============================================================================
# STEP 9: 완료 후 실행 - 결과 다운로드 및 파싱
# ============================================================================

# Job이 완료되면 아래 코드를 실행하세요
"""
# 결과 다운로드
downloaded_files = download_batch_results(
    Config.BUCKET_NAME,
    Config.OUTPUT_PREFIX
)

# 결과 파싱 (첫 번째 JSONL 파일 사용)
if downloaded_files:
    parsed_results, ragas_dataset = parse_batch_output(
        downloaded_files[0],
        testset_df
    )

    print("\n✅ Pipeline Complete!")
    print(f"Total questions answered: {len(parsed_results['questions'])}")
    print(f"Success rate: {parsed_results['stats']['success']}/{parsed_results['stats']['total']}")
"""


# ============================================================================
# 요약
# ============================================================================

print("\n" + "="*70)
print("Pipeline Summary")
print("="*70)
print(f"✅ Input JSONL created: {input_jsonl}")
print(f"✅ Uploaded to GCS: {gcs_input_uri}")
print(f"✅ Batch job submitted: {batch_job.name}")
print(f"\n⏳ Waiting for job completion...")
print(f"\n📝 Next Steps:")
print(f"1. Wait for job to complete (10-30 minutes)")
print(f"2. Check status: check_job_status('{batch_job.name}')")
print(f"3. Download results: Uncomment and run STEP 9 code")
print(f"\n🌐 Monitor:")
print(f"  GCS: https://console.cloud.google.com/storage/browser/{Config.BUCKET_NAME}")
print(f"  Jobs: https://console.cloud.google.com/vertex-ai/batch-predictions")
print("="*70)
