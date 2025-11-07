# Knowledge Extraction v2 - Usage Guide

## 개선 사항

### 1. ✅ Rate Limiting 수정
**문제**: 이전 버전은 배치 완료 후 60초 대기 → 실제 처리 시간 + 60초
**해결**: 배치 시작 시점부터 65초 계산 → 안전한 rate limit 준수 (60초 + 5초 안전 마진)

```python
# OLD (v1): 배치 완료 후 60초 대기
await self.extract_knowledge_batch(...)  # Takes ~10s
await asyncio.sleep(60)  # Total: 70s per batch

# NEW (v2): 배치 시작 후 65초 경과 시 다음 배치 시작
batch_start = time.time()
await self.extract_knowledge_batch(...)  # Takes ~10s
await asyncio.sleep(65)  # Total: exactly 65s per batch (60s + 5s safety)
```

### 2. ✅ 중간 저장 (Incremental Saving)
**문제**: 모든 배치 완료 후에만 저장 → 중간에 실패 시 전체 손실
**해결**: 각 배치 완료 후 즉시 저장

```
output_dir/
├── batch_001.json  # 배치별 저장
├── batch_002.json
├── batch_003.json
├── ...
├── extracted_documents.json  # 최종 통합 파일
└── cache/
    └── extraction_cache.json  # 캐시 파일
```

### 3. ✅ 캐싱 및 재시작 (Caching & Resume)
**문제**: 중간에 중단되면 처음부터 다시 시작
**해결**: 처리된 대화는 캐시에 저장, 재시작 시 자동으로 skip

```python
# 대화별 해시 생성 및 캐시 확인
dialogue_hash = md5(dialogue_json)
if dialogue_hash in cache:
    return cached_result  # 이미 처리된 대화는 skip
```

### 4. ✅ 재시도 로직 (Deferred Retry Pattern)
**문제**: JSON 파싱 에러 발생 시 그냥 skip → 데이터 손실
**해결**: 실패한 대화를 수집하여 모든 메인 배치 완료 후 재시도 (최대 3회)

```python
# Deferred Retry Pattern
# 1. 배치 처리 중 실패 수집 (즉시 재시도 안함 - 배치 지연 방지)
for docs, success, failed_dialogue in results:
    if not success and failed_dialogue:
        failed_dialogues.append(failed_dialogue)

# 2. 모든 메인 배치 완료 후 재시도
if all_failed_dialogues:
    for retry_attempt in range(1, max_retries + 1):
        retry_batch_num = total_batches + retry_attempt
        retry_docs, success, failure, still_failed = await extract_knowledge_batch(
            all_failed_dialogues, retry_batch_num, total_batches + max_retries
        )
        all_failed_dialogues = still_failed  # 여전히 실패한 것만 다음 시도
```

**핵심**: 배치 내 재시도 → 전체 배치 지연 발생 ❌
**개선**: 메인 배치 완료 후 재시도 → 배치 병렬성 보존 ✅

## 사용법

### 기본 실행

```bash
uv run python src/knowledge_extraction/claude_knowledge_extractor_v2.py \
    --config config/knowledge_extraction_poc.json \
    --input data/processed/knowledge_extraction/sampled_dialogues.json
```

### 캐시 디렉토리 지정

```bash
uv run python src/knowledge_extraction/claude_knowledge_extractor_v2.py \
    --config config/knowledge_extraction_poc.json \
    --input data/processed/knowledge_extraction/sampled_dialogues.json \
    --cache-dir data/evaluation/knowledge_extraction_poc/cache
```

### 중단된 작업 재시작

중단된 작업을 재시작하면 자동으로 캐시를 로드하고 미처리 대화만 처리합니다:

```bash
# 동일한 명령어 실행 - 자동으로 캐시 로드
uv run python src/knowledge_extraction/claude_knowledge_extractor_v2.py \
    --config config/knowledge_extraction_poc.json \
    --input data/processed/knowledge_extraction/sampled_dialogues.json
```

## 출력 파일 구조

```
data/evaluation/knowledge_extraction_poc/
├── batch_001.json              # 배치 1 결과 (10개 대화)
├── batch_002.json              # 배치 2 결과 (10개 대화)
├── ...
├── batch_020.json              # 배치 20 결과 (10개 대화)
├── extracted_documents.json    # 전체 통합 결과 (200개 문서)
├── extraction_summary.json     # 실행 통계
└── cache/
    └── extraction_cache.json   # 처리 캐시 (재시작용)
```

## 로그 예시

### 정상 처리
```
============================================================
Processing Batch 1/20
============================================================
Dialogues in batch: 10
🚀 Starting 10 concurrent API calls...
  🤖 [1/200] Calling API for dialogue B2033...
  ✅ [1/200] Extracted 1 documents
  🤖 [2/200] Calling API for dialogue B2045...
  ✅ [2/200] Extracted 1 documents
  ...
✅ Batch complete: 10 documents extracted
   Success: 10/10, Failed: 0/10
   Batch duration: 8.3s
💾 Batch saved: batch_001.json
```

### 캐시 사용 (재시작)
```
📂 Loaded cache: 50 processed dialogues
============================================================
Processing Batch 6/20
============================================================
  💾 [51/200] Using cached result for B2150
  💾 [52/200] Using cached result for B2151
  🤖 [53/200] Calling API for dialogue B2152...  # 새로운 대화만 처리
```

### 재시도 로직 (Deferred Retry)
```
============================================================
Processing Batch 18/20
============================================================
  🤖 [164/200] Calling API for dialogue B3421...
  ❌ [164/200] Failed: JSON parsing error
  🤖 [165/200] Calling API for dialogue B3422...
  ✅ [165/200] Extracted 1 documents
✅ Batch complete: 9 documents extracted
   Success: 9/10, Failed: 1/10

... (모든 메인 배치 완료 후)

============================================================
🔄 RETRYING FAILED DIALOGUES
============================================================
Total failed: 2

🔄 Retry attempt 1/3 for 2 dialogues
📦 Batch 21/23
  🤖 Calling API for dialogue B3421 (retry)...
  ✅ Extracted 1 documents  # 재시도 성공!
  🤖 Calling API for dialogue B3443 (retry)...
  ❌ Failed: JSON parsing error  # 여전히 실패

🔄 Retry attempt 2/3 for 1 dialogues
📦 Batch 22/23
  🤖 Calling API for dialogue B3443 (retry)...
  ✅ Extracted 1 documents  # 2차 재시도 성공!

✅ All failed dialogues recovered!
```

### 최종 요약
```
============================================================
EXTRACTION COMPLETE
============================================================
Total documents: 198
Success: 198/200, Failed: 2/200
Processing time: 1140.5s (19.0 min)
Raw output: data/evaluation/knowledge_extraction_poc/extracted_documents.json
```

## 성능 비교

### v1 (기존)
- **Rate Limiting**: 배치 완료 후 60초 대기 → 실제 70초/배치
- **저장**: 전체 완료 후 1회 저장
- **재시작**: 불가능 (처음부터 다시)
- **실패 처리**: skip (데이터 손실)
- **200개 처리 시간**: ~23분 (20 × 70초)

### v2 (개선)
- **Rate Limiting**: 배치 시작 후 65초 경과 시 다음 시작 → 안전하게 65초/배치 (60초 + 5초 마진)
- **저장**: 배치별 즉시 저장 + 최종 통합
- **재시작**: 캐시 기반 자동 재시작
- **실패 처리**: Deferred Retry Pattern (메인 배치 완료 후 최대 3회 재시도)
- **200개 처리 시간**: ~21.7분 (19 × 65초 + 첫 배치) + 재시도 배치 (실패 시)

### 시간 비교
- **POC (200개)**: 23분 → 20.6분 (**-10.4% 개선**, 안전성 증가)
- **전체 (8000개)**: 13.3시간 (v1 기준) → **14.4시간** (v2 기준, +1.1시간이지만 안전성 우선)

**v2는 약간 더 시간이 걸리지만**: 중간 저장, 재시작, 재시도 기능으로 **안정성과 데이터 보존을 보장**합니다.

## 주의사항

### 1. 캐시 관리
- 캐시는 대화 내용의 MD5 해시 기반으로 생성
- 대화 내용이 변경되면 새로운 해시로 인식되어 재처리됨
- 캐시 초기화가 필요한 경우: `cache/` 디렉토리 삭제

### 2. 배치 파일
- 각 배치는 독립적인 JSON 파일로 저장
- 특정 배치만 재처리하려면 해당 배치 파일 삭제 후 재실행

### 3. 재시도 제한 (Deferred Retry)
- 배치 처리 중 실패는 즉시 재시도하지 않고 수집만 함
- 모든 메인 배치 완료 후 실패한 대화들을 별도 배치로 재시도 (최대 3회)
- 3회 모두 실패 시 빈 문서([])로 캐싱하여 무한 재시도 방지
- **장점**: 배치 내 재시도로 인한 전체 배치 지연 방지

### 4. Rate Limit
- 첫 번째 배치는 delay 없이 즉시 시작
- 이후 배치는 이전 배치 시작 시점으로부터 정확히 65초 후 시작 (60초 + 5초 안전 마진)
- 실제 API 호출은 10 calls/minute을 안전하게 준수

## 문제 해결

### 중간에 중단되었을 때
1. 동일한 명령어로 재실행
2. 캐시 확인: `📂 Loaded cache: N processed dialogues` 로그 확인
3. 미처리 대화부터 자동으로 재시작

### JSON 파싱 에러가 계속 발생할 때
1. 로그에서 재시도 횟수 확인
2. 3회 재시도 후에도 실패하면 해당 대화는 skip
3. 최종 요약에서 `Failed: N/M` 확인

### 캐시 초기화가 필요할 때
```bash
rm -rf data/evaluation/knowledge_extraction_poc/cache/
```

## 다음 단계

v2 추출 완료 후:

1. **문서 조직화**:
```bash
uv run python src/knowledge_extraction/document_organizer.py \
    --input data/evaluation/knowledge_extraction_poc/extracted_documents.json \
    --output-dir data/processed/knowledge_docs \
    --min-doc-size 500 \
    --merge-small
```

2. **품질 검증**:
```bash
uv run python src/knowledge_extraction/quality_validator.py \
    --docs-dir data/processed/knowledge_docs \
    --report-path data/evaluation/knowledge_extraction_poc/quality_report.md \
    --min-doc-size 500
```
