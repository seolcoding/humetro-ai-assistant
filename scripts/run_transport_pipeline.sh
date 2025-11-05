#!/bin/bash
#
# 다산콜센터 교통 데이터 분석 파이프라인 실행 스크립트
# MapReduce 방식으로 데이터를 처리하고 질문을 생성합니다.
#

set -e  # 에러 발생 시 즉시 중단

# 색상 정의 (출력 미화)
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 타임스탬프 함수
timestamp() {
    date +"%Y-%m-%d %H:%M:%S"
}

# 로그 함수들
log_info() {
    echo -e "${GREEN}[$(timestamp)] INFO:${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[$(timestamp)] WARN:${NC} $1"
}

log_error() {
    echo -e "${RED}[$(timestamp)] ERROR:${NC} $1"
}

log_section() {
    echo -e "\n${BLUE}═══════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}═══════════════════════════════════════════════════════${NC}\n"
}

# 파이프라인 시작
START_TIME=$(date +%s)

log_section "다산콜센터 교통 데이터 분석 파이프라인 시작"

# 작업 디렉토리 확인
WORK_DIR="data/dasan_transport_analysis"
if [ ! -d "$WORK_DIR" ]; then
    log_info "작업 디렉토리 생성 중..."
    mkdir -p "$WORK_DIR"/{chunks,summaries,aggregated,questions,logs}
fi

# Python 환경 확인
log_info "Python 환경 확인 중..."
if ! command -v uv &> /dev/null; then
    log_error "uv가 설치되어 있지 않습니다. uv를 먼저 설치해주세요."
    exit 1
fi

# Claude CLI 확인
log_info "Claude CLI 확인 중..."
if ! command -v claude &> /dev/null; then
    log_error "Claude CLI가 설치되어 있지 않습니다."
    exit 1
fi

# 1. 데이터 추출 및 청킹
log_section "Step 1: 데이터 추출 및 청킹"
log_info "교통 데이터를 5개 대화씩 청크로 분할합니다..."

if [ -f "$WORK_DIR/chunks/chunks_metadata.json" ]; then
    log_warn "청크 파일이 이미 존재합니다. 재생성하려면 기존 파일을 삭제하세요."
    read -p "기존 청크를 사용하시겠습니까? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$WORK_DIR/chunks/"*
        uv run python scripts/extract_transport_data.py
    else
        log_info "기존 청크 파일을 사용합니다."
    fi
else
    uv run python scripts/extract_transport_data.py
fi

# 청크 수 확인
if [ -f "$WORK_DIR/chunks/chunks_metadata.json" ]; then
    CHUNK_COUNT=$(uv run python -c "import json; print(json.load(open('$WORK_DIR/chunks/chunks_metadata.json'))['total_chunks'])")
    log_info "총 ${CHUNK_COUNT}개의 청크가 생성되었습니다."
else
    log_error "청크 메타데이터 파일을 찾을 수 없습니다."
    exit 1
fi

# 2. 병렬 요약 실행
log_section "Step 2: 병렬 요약 실행 (Claude 비대화형 모드)"
log_info "10개의 동시 프로세스로 청크를 요약합니다..."
log_warn "이 작업은 청크 수에 따라 30-40분 정도 소요될 수 있습니다."

# 요약 파일 수 확인
EXISTING_SUMMARIES=$(find "$WORK_DIR/summaries" -name "chunk_*_summary.json" 2>/dev/null | wc -l)

if [ "$EXISTING_SUMMARIES" -gt 0 ]; then
    log_warn "이미 ${EXISTING_SUMMARIES}개의 요약이 존재합니다."
    if [ "$EXISTING_SUMMARIES" -eq "$CHUNK_COUNT" ]; then
        log_info "모든 청크가 이미 요약되었습니다."
        read -p "요약을 다시 실행하시겠습니까? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "기존 요약을 사용합니다."
        else
            rm -rf "$WORK_DIR/summaries/"*
            uv run python scripts/parallel_summarizer.py
        fi
    else
        log_info "남은 청크를 요약합니다..."
        uv run python scripts/parallel_summarizer.py
    fi
else
    uv run python scripts/parallel_summarizer.py
fi

# 요약 결과 확인
SUMMARY_COUNT=$(find "$WORK_DIR/summaries" -name "chunk_*_summary.json" 2>/dev/null | wc -l)
log_info "총 ${SUMMARY_COUNT}개의 요약이 완료되었습니다."

# 3. 요약 집계
log_section "Step 3: MapReduce 스타일 요약 집계"
log_info "개별 요약을 통합하여 마스터 요약을 생성합니다..."

uv run python scripts/aggregate_summaries.py

if [ ! -f "$WORK_DIR/aggregated/master_summary.json" ]; then
    log_error "마스터 요약 생성에 실패했습니다."
    exit 1
fi

# 마스터 요약 통계 출력
log_info "마스터 요약 통계:"
uv run python -c "
import json
data = json.load(open('$WORK_DIR/aggregated/master_summary.json'))
stats = data.get('summary_statistics', {})
print(f'  - 고유 질문 유형: {stats.get(\"unique_question_types\", 0)}개')
print(f'  - 고유 문제: {stats.get(\"unique_issues\", 0)}개')
print(f'  - 답변 패턴: {stats.get(\"unique_patterns\", 0)}개')
print(f'  - 개선 제안: {stats.get(\"improvement_suggestions\", 0)}개')
"

# 4. 질문 생성
log_section "Step 4: 테스트 질문 생성"
log_info "집계된 인사이트를 기반으로 테스트 질문을 생성합니다..."

uv run python scripts/generate_questions.py

# 생성된 질문 확인
LATEST_QUESTIONS=$(ls -t "$WORK_DIR/questions/generated_questions_"*.json 2>/dev/null | head -1)

if [ -z "$LATEST_QUESTIONS" ]; then
    log_error "질문 생성에 실패했습니다."
    exit 1
fi

# 질문 통계 출력
log_info "생성된 질문 통계:"
uv run python -c "
import json
data = json.load(open('$LATEST_QUESTIONS'))
if 'validation_stats' in data:
    stats = data['validation_stats']
    print(f'  - 총 질문 수: {stats.get(\"total\", 0)}개')
    print(f'  - 난이도별: {stats.get(\"by_difficulty\", {})}')
    print(f'  - 카테고리: {len(stats.get(\"by_category\", {}))}개')
"

# 5. 결과 요약
log_section "파이프라인 실행 완료"

# 실행 시간 계산
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))

log_info "총 실행 시간: ${MINUTES}분 ${SECONDS}초"
log_info "결과 파일 위치:"
echo "  - 청크 데이터: $WORK_DIR/chunks/"
echo "  - 개별 요약: $WORK_DIR/summaries/"
echo "  - 마스터 요약: $WORK_DIR/aggregated/master_summary.json"
echo "  - 생성된 질문: $LATEST_QUESTIONS"
echo "  - 실행 로그: $WORK_DIR/logs/"

# 간단한 질문 미리보기
log_section "생성된 질문 샘플 (상위 5개)"
if [ -f "${LATEST_QUESTIONS%.json}_text.txt" ]; then
    head -5 "${LATEST_QUESTIONS%.json}_text.txt"
else
    uv run python -c "
import json
data = json.load(open('$LATEST_QUESTIONS'))
questions = data.get('generated_questions', [])[:5]
for q in questions:
    print(f\"[{q.get('id', '?')}] {q.get('question', 'N/A')}\")
"
fi

log_info "전체 질문을 확인하려면 다음 파일을 참조하세요:"
echo "  JSON: $LATEST_QUESTIONS"
echo "  TEXT: ${LATEST_QUESTIONS%.json}_text.txt"

log_section "파이프라인 종료"