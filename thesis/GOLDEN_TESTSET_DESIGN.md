# Golden Testset 설계 문서

> 작성일: 2024-11-30
> 목적: RAG 최적화 효과 검증을 위한 어려운 테스트셋 구축

---

## 1. 배경 및 문제 정의

### 1.1 현재 문제점

기존 테스트 결과에서 Naive RAG vs Optimized RAG 성능 차이가 미미함:

| Model | Naive G-Eval | Optimized G-Eval | 개선율 |
|-------|--------------|------------------|--------|
| GPT-4o-mini | 4.800 | 4.927 | +2.7% |
| Gemma3:12b | 4.809 | 4.873 | +1.3% |
| EXAONE-3.5:7.8b | 4.773 | 4.827 | +1.1% |

**원인 분석**: 테스트 질문이 너무 쉬워서 Naive RAG로도 대부분 정답 도출 가능

### 1.2 해결 방향

- **어려운 질문 유형** 비율 증가 (응답불가형, Yes/No 함정 등)
- **Distractor 문서** 추가로 Retrieval 난이도 상승
- **계층적 샘플링**으로 주제/유형 균형 확보

---

## 2. 벤치마크 참조 및 근거

### 2.1 코퍼스 크기 결정 근거

주요 RAG 벤치마크의 코퍼스:QA 비율 분석:

| 벤치마크 | 코퍼스 크기 | QA 수 | 비율 | 출처 |
|---------|-----------|-------|------|------|
| MultiHop-RAG | 609 문서 | 2,556 | 1:4.2 | COLM 2024 |
| LegalBench-RAG | 79M chars | 6,858 | - | arXiv 2408.10343 |
| Open RAG Benchmark | 1,000 PDF | - | - | Vectara 2024 |
| **BenchmarkQED** | **1,397 문서** | **200** | **7:1** | **Microsoft Research** |
| RAGBench (CUAD) | 510 문서 | 21,000 | 1:41 | arXiv 2407.11005 |

**선택: Microsoft BenchmarkQED 스타일 (코퍼스:QA ≈ 5:1)**

### 2.2 참고 문헌

1. **MultiHop-RAG** (COLM 2024)
   - URL: https://github.com/yixuantt/MultiHop-RAG
   - 특징: 2-4개 문서에 걸친 Multi-hop 추론 필요
   - 609 news articles, 2,556 queries

2. **RAGBench** (arXiv 2407.11005)
   - URL: https://arxiv.org/abs/2407.11005
   - 특징: 100K examples, 5개 산업 도메인
   - TRACe 평가 프레임워크 제안

3. **LegalBench-RAG** (arXiv 2408.10343)
   - URL: https://arxiv.org/html/2408.10343v1
   - 특징: 법률 도메인 특화, 정밀 검색 강조
   - 6,858 QA pairs, 전문가 어노테이션

4. **BenchmarkQED** (Microsoft Research 2024)
   - URL: https://www.microsoft.com/en-us/research/blog/benchmarkqed-automated-benchmarking-of-rag-systems/
   - 특징: 1,397 문서 → 200 쿼리 (7:1 비율)
   - AutoQ를 통한 자동 쿼리 생성

5. **Open RAG Benchmark** (Vectara 2024)
   - URL: https://www.vectara.com/blog/open-rag-benchmark-a-new-frontier-for-multimodal-pdf-understanding-in-rag
   - 특징: 1,000 PDF, arXiv 전체 카테고리 균등 분포

6. **RAG Evaluation Survey** (Evidently AI)
   - URL: https://www.evidentlyai.com/llm-guide/rag-evaluation
   - 특징: RAG 평가 메트릭 및 베스트 프랙티스

---

## 3. 데이터셋 설계

### 3.1 원천 데이터

**AI Hub 행정문서 대상 기계독해 데이터**
- 총 411,840건 QA
- 6가지 QA 유형
- 11개 주제 카테고리

### 3.2 QA 유형별 특성 및 난이도

| QA 유형 | 데이터량 | 난이도 | RAG 차별화 | 선택 비율 |
|--------|---------|--------|-----------|----------|
| 정답경계 추출형 | 133,166 | ⭐ | 낮음 | 5% |
| 절차(방법)형 | 60,523 | ⭐⭐ | 높음 | 20% |
| Table 정답 추출형 | 131,068 | ⭐⭐ | 중간 | 15% |
| Yes/No 단문형 | 35,125 | ⭐⭐⭐ | 매우 높음 | 25% |
| 다지선다형 | 31,458 | ⭐⭐ | 중간 | 5% |
| **응답불가형** | 20,500 | ⭐⭐⭐⭐ | **최고** | **30%** |

### 3.3 주제 카테고리 선정

상위 6개 카테고리 선정 (전체의 ~80% 커버):

| 카테고리 | 원본 비율 | 선택 문항 |
|---------|----------|----------|
| 국토관리 | 21% | 20 |
| 공공행정 | 18% | 20 |
| 환경기상 | 16% | 20 |
| 과학기술 | 16% | 20 |
| 사회복지 | 5% | 20 |
| 법률 | 4% | 20 |
| **합계** | - | **120** |

---

## 4. 샘플링 전략

### 4.1 계층적 층화 샘플링 (Stratified Hierarchical)

```
1단계: 주제별 균등 (6개 × 20문항 = 120문항)
2단계: 주제 내 난이도 가중 분배
```

**주제당 20문항 내부 구성:**

| 유형 | 문항수 | 비율 | 목적 |
|------|--------|------|------|
| 응답불가 | 6 | 30% | Hallucination 테스트 |
| Yes/No | 5 | 25% | 함정 질문 |
| 절차형 | 4 | 20% | Multi-hop 추론 |
| Table | 3 | 15% | 구조화 데이터 이해 |
| 정답경계+다지선다 | 2 | 10% | Baseline |

### 4.2 코퍼스 구성 (Microsoft 스타일)

**목표 비율: 코퍼스:QA = 5:1**

| 지문 유형 | 수량 | 비율 | 설명 |
|----------|------|------|------|
| 정답 지문 | 60 | 10% | QA의 Ground Truth |
| 동일 주제 Distractor | 480 | 80% | 검색 혼동 유발 (정답당 8개) |
| 다른 주제 Noise | 60 | 10% | 현실적 노이즈 |
| **총 코퍼스** | **600** | 100% | |

### 4.3 Distractor 선정 기준

우선순위:
1. 동일 발행기관 (doc_source) → 높은 혼동
2. 유사 발행시기 (±1년) → 중간 혼동
3. 유사 키워드 포함 → 높은 혼동
4. 동일 주제 내 랜덤 → 기본 노이즈

---

## 5. 프롬프트 전략

### 5.1 기존 5개 프롬프트 유지

- P1: Basic (Baseline)
- P2: Strict Grounding ← **Few-shot 추가**
- P3: Zero-shot CoT
- P4: Domain Expert
- P5: Long Context Reorder

### 5.2 P2 개선 (Few-shot 추가)

```
아래 문서만을 참고하여 질문에 답변하세요.
문서에 답이 없으면 "답변 불가"라고 답하세요.
추측하거나 외부 지식을 사용하지 마세요.

[예시 1 - 단답형]
Q: 황학정 안에 국궁전시관을 설치한 부서는?
A: 안전행정부

[예시 2 - Yes/No]
Q: 싱가포르는 NEWater 공장에서 다중여과공법을 적용하고 있나요?
A: Yes

[예시 3 - 답변 불가]
Q: 미국 NEWater 공장의 처리 용량은?
A: 답변 불가 (NEWater는 싱가포르 시설)

[참고 문서]
{retrieved_contents}

[질문]
{query}

[답변]
```

---

## 6. 기대 효과

### 6.1 예상 성능 차이

| 조합 | 응답불가 정답률 | Yes/No 정답률 | 전체 G-Eval |
|------|---------------|--------------|-------------|
| Naive RAG + Naive Prompt | ~10% | ~60% | ~4.0 |
| Naive RAG + Opt Prompt | ~40% | ~75% | ~4.3 |
| Opt RAG + Naive Prompt | ~30% | ~70% | ~4.2 |
| **Opt RAG + Opt Prompt** | **~70%** | **~85%** | **~4.7** |

### 6.2 Naive vs Optimized 차이 목표

- 현재: +1~3%
- **목표: +10~15%**

---

## 7. 구현 계획

### 7.1 파일 구조

```
src/autorag_golden_test/
├── config/
│   └── golden_test.yaml
├── data/
│   ├── corpus.parquet      # 600개 지문
│   └── qa.parquet          # 120개 QA
├── prompts/
│   └── p2_fewshot.txt      # 개선된 P2
├── scripts/
│   ├── sample_qa.py        # 층화 샘플링
│   └── build_corpus.py     # 코퍼스 구축
└── run_golden_test.py
```

### 7.2 샘플링 알고리즘

```python
def stratified_sample(data, n_per_topic=20, topic_count=6):
    """
    1단계: 주제별 균등 샘플링
    2단계: 주제 내 난이도 가중 샘플링
    """
    result = []
    for topic in TOP_6_TOPICS:
        topic_data = data[data['category'] == topic]

        # 난이도 가중 비율
        type_weights = {
            7: 0.30,  # 응답불가
            5: 0.25,  # Yes/No
            2: 0.20,  # 절차형
            3: 0.15,  # Table
            1: 0.05,  # 정답경계
            6: 0.05,  # 다지선다
        }

        for qa_type, weight in type_weights.items():
            n_sample = int(n_per_topic * weight)
            type_data = topic_data[topic_data['qa_type'] == qa_type]
            sampled = type_data.sample(n=min(n_sample, len(type_data)))
            result.append(sampled)

    return pd.concat(result)
```

---

## 8. 체크리스트

- [ ] 층화 샘플링 스크립트 구현
- [ ] 코퍼스 구축 (600개 지문)
- [ ] QA 추출 (120개)
- [ ] P2 프롬프트 Few-shot 추가
- [ ] AutoRAG config 작성
- [ ] 실험 실행 및 결과 분석

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2024-11-30 | 초안 작성 |
