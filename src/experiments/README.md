# Experiment Tracking

실험 추적 및 캐싱 레이어로 실험 상태 관리와 비용 절감을 제공합니다.

## 모듈

### `cache_manager.py` - 벡터 임베딩 캐시

콘텐츠 해시 기반 벡터 임베딩 캐싱으로 중복 OpenAI API 호출을 방지합니다.

**주요 기능:**
- 콘텐츠 해시 기반 캐싱
- OpenAI API 비용 절감
- 캐시 히트/미스 통계
- 배치 처리 지원
- 영구 저장 (pickle)

**사용 예시:**
```python
from src.experiments.cache_manager import get_cache

cache = get_cache()

# 임베딩 조회
embedding = cache.get(text="안녕하세요", model="text-embedding-3-small")

if embedding is None:
    # 캐시 미스 - OpenAI API 호출
    embedding = call_openai_api(text, model)

    # 캐시 저장
    cache.put(
        text="안녕하세요",
        model="text-embedding-3-small",
        embedding=embedding,
        tokens=5,
        cost=0.00001
    )

# 배치 처리
texts = ["텍스트1", "텍스트2", "텍스트3"]
cached_embeddings, uncached_texts = cache.get_batch(texts, model)

# uncached_texts에 대해서만 API 호출
new_embeddings = call_openai_api_batch(uncached_texts)
cache.put_batch(uncached_texts, model, new_embeddings)

# 통계 확인
cache.print_stats()
"""
============================================================
EMBEDDING CACHE STATISTICS
============================================================
Total cached items: 1,234
Cache hits: 890
Cache misses: 344
Hit rate: 72.1%
Total saved cost: $2.3456
Cache size: 15.23 MB
============================================================
"""
```

### `tracker.py` - 실험 상태 추적기

실험 상태를 저장하고 중단된 지점부터 재개할 수 있습니다.

**주요 기능:**
- 7단계 파이프라인 상태 추적
- 스테이지별 중간 결과 저장
- 실험 재개 기능
- 실험 설정 및 메타데이터 관리
- 최종 결과 저장

**사용 예시:**
```python
from src.experiments.tracker import get_tracker

tracker = get_tracker("experiment_v1")

# 실험 설정 저장
tracker.save_config({
    "model": "gpt-4",
    "chunk_size": 1024,
    "embedding_model": "text-embedding-3-small"
})

# 실험 메타데이터
tracker.save_metadata(
    description="Baseline RAG experiment with FAISS",
    tags=["baseline", "faiss", "korean"],
    author="Your Name"
)

# 스테이지 실행
for stage_num in range(1, 8):
    if tracker.is_stage_completed(stage_num):
        print(f"Stage {stage_num} already completed, skipping")
        continue

    tracker.start_stage(stage_num)

    try:
        # 스테이지 실행
        result = run_stage(stage_num)

        # 결과 저장
        tracker.save_stage_result(stage_num, result)
        tracker.complete_stage(stage_num, duration_seconds=10.5)

    except Exception as e:
        tracker.fail_stage(stage_num, error=str(e))
        raise

# 최종 결과 저장
tracker.save_final_results({
    "accuracy": 0.95,
    "f1_score": 0.92,
    "total_time": 120.5
})

# 요약 출력
tracker.print_summary()
```

### `results_store.py` - 결과 저장소 (예정)

여러 실험 결과를 비교하고 분석하는 기능을 제공할 예정입니다.

## 테스트

```bash
# 모든 테스트 실행
pytest tests/test_experiments/ -v

# 개별 모듈 테스트
pytest tests/test_experiments/test_cache_manager.py -v
pytest tests/test_experiments/test_tracker.py -v
```

## 의존성

- `pickle` - 캐시 직렬화
- `json` - 상태 저장
- `hashlib` - 캐시 키 생성
- `pathlib` - 경로 처리

## 디렉토리 구조

실험 추적 시 다음과 같은 구조가 생성됩니다:

```
results/experiments/
└── experiment_v1/
    ├── state.json           # 실험 상태
    ├── config.json          # 실험 설정
    ├── metadata.json        # 메타데이터
    ├── results.json         # 최종 결과
    └── stages/              # 스테이지별 결과
        ├── stage_1_result.json
        ├── stage_2_result.json
        └── ...

data/embeddings_cache/
├── embeddings.pkl           # 임베딩 캐시
├── metadata.json            # 캐시 메타데이터
└── stats.json              # 캐시 통계
```

## 비용 절감 예시

임베딩 캐시를 사용하면 다음과 같은 비용 절감 효과를 얻을 수 있습니다:

```python
# 예시: 1,000개 문서를 3번 임베딩하는 경우

# 캐시 없이:
# - 총 API 호출: 3,000회
# - 총 비용: $0.30 (가정)

# 캐시 사용 시:
# - 첫 번째: 1,000회 API 호출 ($0.10)
# - 두 번째: 0회 (캐시 히트 100%)
# - 세 번째: 0회 (캐시 히트 100%)
# - 총 비용: $0.10
# - 절감 금액: $0.20 (66% 절감)
```

## 재개 기능 예시

실험이 중단되었을 때 재개하는 방법:

```python
# 같은 experiment_name으로 tracker 생성
tracker = get_tracker("experiment_v1")

# 다음 미완료 스테이지 찾기
next_stage = tracker.get_next_pending_stage()

if next_stage:
    print(f"Resuming from stage {next_stage}")
    # 해당 스테이지부터 계속 실행
else:
    print("All stages completed")
```
