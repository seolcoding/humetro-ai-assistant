#!/bin/bash
# Gemini Batch API 전체 파이프라인 실행 스크립트
#
# 사용법:
#   ./scripts/run_gemini_batch_pipeline.sh [OPTIONS]
#
# 옵션:
#   --sample N          샘플 질문 수 (기본: 50)
#   --with-rag          RAG 컨텍스트 포함
#   --testset PATH      테스트셋 경로 (기본: data/evaluation/testsets/golden_testset_50.csv)
#   --model MODEL       모델 이름 (기본: gemini-2.5-pro)
#   --bucket BUCKET     GCS 버킷 이름 (기본: rag_bucket_yoon)

set -e  # Exit on error

# Default values
SAMPLE_SIZE=50
WITH_RAG=""
TESTSET="data/evaluation/testsets/golden_testset_50.csv"
MODEL="gemini-2.5-pro"
BUCKET="rag_bucket_yoon"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --sample)
            SAMPLE_SIZE="$2"
            shift 2
            ;;
        --with-rag)
            WITH_RAG="--with-rag"
            shift
            ;;
        --testset)
            TESTSET="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --bucket)
            BUCKET="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "========================================================================"
echo "Gemini Batch API Pipeline"
echo "========================================================================"
echo "Testset: $TESTSET"
echo "Sample size: $SAMPLE_SIZE"
echo "Model: $MODEL"
echo "GCS Bucket: $BUCKET"
echo "RAG: ${WITH_RAG:-disabled}"
echo "Timestamp: $TIMESTAMP"
echo "========================================================================"

# Step 1: Generate batch input JSONL
echo ""
echo "Step 1/5: Generating batch input JSONL..."
INPUT_FILE="data/batch_processing/batch_input_${TIMESTAMP}.jsonl"

uv run python src/batch_processing/create_batch_input.py \
    --testset "$TESTSET" \
    --output "$INPUT_FILE" \
    --sample "$SAMPLE_SIZE" \
    --model "$MODEL" \
    $WITH_RAG

if [ ! -f "$INPUT_FILE" ]; then
    echo "❌ Error: Input file not created"
    exit 1
fi

echo "✅ Input JSONL created: $INPUT_FILE"

# Step 2: Upload to GCS
echo ""
echo "Step 2/5: Uploading to GCS..."
GCS_INPUT="gs://$BUCKET/batch_input_${TIMESTAMP}.jsonl"
gsutil cp "$INPUT_FILE" "$GCS_INPUT"

echo "✅ Uploaded to: $GCS_INPUT"

# Step 3: Submit batch job
echo ""
echo "Step 3/5: Submitting batch job..."
GCS_OUTPUT="gs://$BUCKET/batch_output_${TIMESTAMP}"

uv run python src/batch_processing/submit_batch.py \
    --input "$GCS_INPUT" \
    --output "$GCS_OUTPUT" \
    --model "$MODEL"

echo "✅ Batch job submitted"
echo ""
echo "📋 Job Info:"
echo "  Input:  $GCS_INPUT"
echo "  Output: $GCS_OUTPUT"
echo ""
echo "⏳ Waiting for batch job to complete..."
echo "   (This may take 10-30 minutes depending on job size)"
echo ""
echo "📝 Next steps:"
echo "  1. Wait for job completion (check GCS or Vertex AI console)"
echo "  2. Download output: gsutil cp ${GCS_OUTPUT}/*.jsonl data/batch_processing/"
echo "  3. Parse output: ./scripts/parse_batch_output.sh ${TIMESTAMP}"
echo ""
echo "🌐 Monitor at:"
echo "  https://console.cloud.google.com/storage/browser/${BUCKET}"
echo "  https://console.cloud.google.com/vertex-ai/batch-predictions"
echo ""
echo "========================================================================"
echo "Pipeline Status: Job Submitted (Waiting for completion)"
echo "========================================================================"
