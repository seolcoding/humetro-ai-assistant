# AutoRAG 실험 설계: Best Pipeline Scale-Up

**작성일**: 2024-11-28 20:30 KST
**상태**: 계획 완료, 구현 대기
**브랜치**: `autorag_optimization`

---

## 실험 전략: 2단계 접근법

**결정 사항:**
- 데이터 규모: **100 QA × 5000 corpus** (대규모 검증)
- 실험 방식: **Best Pipeline Only** (현재 파일럿 결과 기반)

### 전략 요약

```
[Phase 1] 파일럿 결과 분석 (완료됨)
    ↓
[Phase 2] Best Pipeline 확정
    ↓
[Phase 3] 대규모 데이터 준비 (100 QA × 5000 corpus)
    ↓
[Phase 4] Best Pipeline 스케일업 검증
    ↓
[Phase 5] 결과 분석 및 보고서
```

---

## Phase 1: 현재 파일럿 결과 분석

### 완료된 파일럿 실험 결과 (8 QA × 10 corpus)

**summary.csv에서 도출된 Best 조합:**
```
Retrieval:        HybridRRF (weight=4.0)
Passage Augmenter: PrevNextPassageAugmenter (num_passages=1)
Reranker:         PassReranker (pass-through)
Filter:           PercentileCutoff (percentile=0.6)
Compressor:       PassCompressor (no compression)
Prompt:           Fstring (기본 템플릿)
Generator:        Gemma3:12b 🏆
```

### 핵심 인사이트

1. **KoReranker 탈락**: PassReranker가 선택됨 → 소규모 corpus에서는 reranking 불필요
2. **Gemma3 1등**: GPT-4o-mini보다 Gemma3:12b가 더 좋은 성능
3. **Hybrid Retrieval 선택**: BM25 + Vector 조합이 최적

---

## Phase 2: Best Pipeline 확정

### 확정된 Best Pipeline

```yaml
# best_pipeline.yaml
node_lines:
  - node_line_name: retrieve_node_line
    nodes:
      - node_type: hybrid_retrieval
        top_k: 5
        modules:
          - module_type: hybrid_rrf
            weight: 4.0  # 파일럿에서 도출된 최적 weight

  - node_line_name: post_retrieve_node_line
    nodes:
      - node_type: passage_augmenter
        top_k: 3
        modules:
          - module_type: prev_next_augmenter
            num_passages: 1

      - node_type: passage_filter
        modules:
          - module_type: percentile_cutoff
            percentile: 0.6

      - node_type: prompt_maker
        modules:
          - module_type: fstring
            prompt: "다음은 행정 문서에서 발췌한 내용입니다..."

      - node_type: generator
        modules:
          # Top 3 Generators만 스케일업 검증
          - module_type: llama_index_llm
            llm: openailike
            model: gemma3:12b  # 🏆 파일럿 1등

          - module_type: openai_llm
            llm: gpt-4o-mini   # Baseline 비교용

          - module_type: llama_index_llm
            llm: openailike
            model: exaone3.5:7.8b  # 한국어 특화
```

### Generator 선정 근거

| 순위 | 모델 | 선정 이유 |
|-----|------|----------|
| 1 | Gemma3:12b | 파일럿 최고 성능 |
| 2 | GPT-4o-mini | 상용 API baseline |
| 3 | EXAONE-3.5 | 한국어 특화, 비교 대상 |

**제외된 모델:**
- GPT-OSS, Qwen3, EXAONE-Deep: 스케일업 비용 절약을 위해 Top 3만 검증

---

## Phase 3: 데이터 준비 (100 QA × 5000 Corpus)

### 데이터 소스

```
원본: data/016.행정_문서_대상_기계독해_데이터/01.데이터/1.Training/라벨링데이터/TL_span_extraction.json
- 총 63,932 문서
- 총 106,530 QA pairs
- doc_class 분포: 과학기술(23%), 공공행정(20%), 환경기상(18%), 국토관리(18%) 등
```

### Golden QA 선정 기준 (100개)

```python
def select_golden_qa(n_questions=100):
    """
    층화 샘플링으로 100개 QA 선정

    기준:
    1. doc_class 분포 유지 (비율 유지)
       - 과학기술: 23개
       - 공공행정: 20개
       - 환경기상: 18개
       - 국토관리: 18개
       - 기타: 21개

    2. context 길이 필터: 400-1000자 (중간 난이도)

    3. answer 검증: answer_start가 context 내 존재
    """
```

### Corpus 구성 (5000개)

```python
def build_corpus(n_corpus=5000, qa_docs=100):
    """
    Corpus 5000개 구성

    구성:
    - 100개: Golden QA의 정답 문서 (필수 포함)
    - 4900개: 무작위 선택된 다른 문서 (노이즈)

    목적:
    - 검색 난이도: 2% (100/5000)
    - 실제 RAG 환경 시뮬레이션
    """
```

### 메타데이터 스키마

```python
# QA DataFrame
qa_df = pd.DataFrame({
    "qid": str,                    # 질문 ID (필수)
    "query": str,                  # 질문 텍스트 (필수)
    "retrieval_gt": List[str],     # 정답 문서 ID 리스트 (필수)
    "generation_gt": List[str],    # 정답 텍스트 리스트 (필수)
    # 메타데이터 (분석용)
    "doc_class": str,              # 토픽 분류
    "doc_source": str,             # 출처
    "context_len": int,            # 컨텍스트 길이
})

# Corpus DataFrame
corpus_df = pd.DataFrame({
    "doc_id": str,                 # 문서 ID (필수)
    "contents": str,               # 문서 내용 (필수)
    "metadata": dict,              # 메타데이터 (JSON)
})
```

---

## Phase 4: 실험 실행

### 예상 실행 시간

| 단계 | 계산 | 예상 시간 |
|------|------|----------|
| 데이터 준비 | JSON 파싱 + 샘플링 | 2분 |
| Embedding (5000 docs) | 5000 × $0.00002 = $0.10 | 5-10분 |
| BM25 인덱싱 | 5000 docs | 1분 |
| Generation (100Q × 3 models) | 현재 8Q×6모델=9분 → 100Q×3모델≈60분 | 60-90분 |
| 메트릭 계산 | ROUGE, METEOR | 5분 |
| **총합** | | **약 1.5-2시간** |

### 실행 명령어

```bash
# Step 1: 데이터 준비
uv run python src/autorag_pilot/prepare_golden_qa.py \
  --n_questions 100 \
  --n_corpus 5000 \
  --seed 42 \
  --output src/autorag_pilot/output_large

# Step 2: Best Pipeline 실행
rm -rf src/autorag_pilot/autorag_project_large
uv run python src/autorag_pilot/run_pilot.py \
  --config best_pipeline.yaml \
  --data_dir output_large \
  --project_dir autorag_project_large
```

---

## Phase 5: 결과 분석

### AutoRAG 결과 파일 구조 (확인됨)

AutoRAG의 `best_*.parquet` 파일에는 **개별 질문별로 모든 점수**가 저장됩니다:

```python
# best_3.parquet 컬럼 (37개)
{
    # 질문 식별자
    "qid": str,                              # 질문 ID (메타데이터 조인 키)
    "query": str,                            # 질문 텍스트
    "generation_gt": List[str],              # 정답

    # Retrieval 단계별 점수 (개별 질문별)
    "retrieval_f1": float,                   # 검색 F1
    "retrieval_recall": float,               # 검색 Recall
    "retrieval_precision": float,            # 검색 Precision
    "passage_augmenter_retrieval_f1": float, # 증강 후 F1
    "passage_reranker_retrieval_f1": float,  # 리랭킹 후 F1
    "passage_filter_retrieval_f1": float,    # 필터링 후 F1

    # Generation 점수 (개별 질문별)
    "meteor": float,                         # METEOR 점수
    "rouge": float,                          # ROUGE-L 점수
    "generated_texts": str,                  # 생성된 답변
}
```

### 분석 가능 항목

**모델별 × 질문별 × 메트릭 점수 측정 가능!**

```python
# 1. 각 모델별 결과 파일 로드
gemma_results = pd.read_parquet("generator/0.parquet")
gpt_results = pd.read_parquet("generator/1.parquet")
exaone_results = pd.read_parquet("generator/2.parquet")

# 2. QA 메타데이터 로드 (doc_class, doc_source 포함)
qa_meta_df = pd.read_parquet("qa_with_metadata.parquet")

# 3. qid로 조인
merged_df = gemma_results.merge(qa_meta_df[["qid", "doc_class", "doc_source"]], on="qid")

# 4. doc_class별 평균 점수 계산
class_scores = merged_df.groupby("doc_class")[["meteor", "rouge", "retrieval_f1"]].mean()
```

### 분석 항목

1. **전체 성능 비교**
   - Gemma3 vs GPT-4o-mini vs EXAONE
   - ROUGE, METEOR, Retrieval F1

2. **메타데이터 기반 세분화** (qid 조인으로 구현)
   - doc_class별 성능 (과학기술 vs 공공행정 vs ...)
   - doc_source별 성능 (중앙부처 vs 지자체)
   - context 길이별 성능 (짧은/중간/긴 문서)

3. **개별 질문별 상세 분석**
   - 각 질문의 Retrieval → Generation 파이프라인 성능 추적
   - 실패 케이스 분석 (낮은 점수 질문 식별)

4. **스케일업 검증**
   - 8 QA 파일럿 결과와 100 QA 결과 비교
   - 결과 일관성 검증

### 예상 출력

```
=== Best Pipeline 스케일업 검증 결과 ===

| 모델 | ROUGE-L | METEOR | Retrieval F1 |
|------|---------|--------|--------------|
| Gemma3:12b | 0.45 | 0.32 | 0.85 |
| GPT-4o-mini | 0.43 | 0.30 | 0.85 |
| EXAONE-3.5 | 0.42 | 0.31 | 0.85 |

=== doc_class별 성능 (Gemma3:12b) ===

| 토픽 | ROUGE-L | 샘플 수 |
|------|---------|---------|
| 과학기술 | 0.48 | 23 |
| 공공행정 | 0.44 | 20 |
| 환경기상 | 0.43 | 18 |
| ...
```

---

## 파일 생성/수정 목록

| 파일 | 액션 | 설명 |
|------|------|------|
| `src/autorag_pilot/prepare_golden_qa.py` | 생성 | 대규모 데이터 준비 스크립트 |
| `src/autorag_pilot/config/best_pipeline.yaml` | 생성 | 최적 파이프라인 config |
| `src/autorag_pilot/analyze_results.py` | 생성 | 결과 분석 스크립트 |
| `src/autorag_pilot/run_pilot.py` | 수정 | data_dir, project_dir 인자 추가 |

---

## 리스크 및 대응

| 리스크 | 대응 방안 |
|--------|----------|
| 5000 corpus 임베딩 시간 | OpenAI batch API 활용 |
| Ollama VRAM 부족 | 모델별 순차 실행 |
| 결과 불일치 (파일럿 vs 대규모) | 샘플 분석 후 원인 파악 |

---

## 구현 체크리스트

- [x] 플랜 확정 (2024-11-28)
- [ ] `prepare_golden_qa.py` 구현
- [ ] `best_pipeline.yaml` 작성
- [ ] `run_pilot.py` 수정 (인자 추가)
- [ ] 데이터 준비 실행
- [ ] Best Pipeline 스케일업 실행
- [ ] 결과 분석 및 보고서 작성

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2024-11-28 | 초안 작성, 실험 설계 확정 |
