# Golden Dataset Module

Multi-hop 질문 생성을 위한 타겟 문서 관리 및 유사 문서 그룹화 모듈

---

## 목적

타겟 문서(120개)에 대해:
1. 같은 토픽 내 유사 문서 찾기 (TF-IDF)
2. Multi-hop 질문 생성을 위한 관련 문서 그룹 구성
3. 최종 QA 데이터셋 통합 관리

---

## 질문 유형별 분포 (토픽당 20개)

| 유형 | 개수 | 필요 문서 | 설명 |
|------|------|----------|------|
| Simple Factoid | 4 | 1개 | 단순 사실 질문 |
| Constraint | 4 | 1개 | 조건부 질문 |
| Multi-hop (2-hop) | 6 | 2개 | 2개 문서 연결 추론 |
| Multi-hop (3-hop) | 3 | 3개 | 3개 문서 연결 추론 |
| Reasoning | 3 | 1개 | 추론/인과 질문 |
| **합계** | **20** | - | |

### 전체 (6개 토픽)

| 유형 | 총계 |
|------|------|
| Simple Factoid | 24 |
| Constraint | 24 |
| Multi-hop (2-hop) | 36 |
| Multi-hop (3-hop) | 18 |
| Reasoning | 18 |
| **합계** | **120** |

---

## JSON 자료구조

### GoldenDataset (최상위)

```json
{
  "metadata": {
    "version": "1.0.0",
    "created_at": "2024-12-02T...",
    "random_state": 42
  },
  "config": {
    "topics": ["공공행정", "국토관리", ...],
    "question_distribution": {
      "simple_factoid": 4,
      "constraint": 4,
      "multi_hop_2": 6,
      "multi_hop_3": 3,
      "reasoning": 3
    }
  },
  "statistics": {...},
  "targets": [...]
}
```

### TargetDocument (타겟 문서)

```json
{
  "doc_id": "D0000042870847",
  "doc_title": "문서 제목",
  "doc_source": "서울특별시청",
  "topic": "공공행정",
  "context": "문서 본문...",
  "context_length": 923,
  "related_documents": [
    {
      "doc_id": "D0000042870848",
      "doc_title": "관련 문서 제목",
      "context": "관련 문서 본문...",
      "similarity_score": 0.45,
      "context_length": 856
    }
  ],
  "qa_pairs": [
    {
      "question": "질문 내용",
      "answer": "답변 내용",
      "question_type": "multi_hop_2",
      "retrieval_gt": ["D0000042870847", "D0000042870848"],
      "reasoning_steps": ["Step 1", "Step 2"]
    }
  ]
}
```

---

## 유사 문서 그룹화 (TF-IDF)

### 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| min_similarity | 0.1 | 최소 유사도 (무관한 문서 제외) |
| max_similarity | 0.8 | 최대 유사도 (중복 문서 제외) |
| top_k | 10 | 타겟당 관련 문서 개수 |

### 근거

- **0.1 ~ 0.8 범위**: Multi-hop 질문에 적합한 "관련 있지만 다른 정보" 문서
- **너무 높은 유사도 (>0.8)**: 거의 동일한 내용, Multi-hop 불가
- **너무 낮은 유사도 (<0.1)**: 무관한 문서, 연결 질문 생성 불가

### 결과 통계

| 토픽 | 타겟 수 | 평균 관련 문서 |
|------|---------|---------------|
| 공공행정 | 20 | 9.2 |
| 국토관리 | 20 | 8.9 |
| 환경기상 | 20 | 7.4 |
| 사회복지 | 20 | 9.5 |
| 식품건강 | 20 | 8.9 |
| 문화관광 | 20 | 9.5 |

---

## 파일 구조

```
src/data_generation/golden_dataset/
├── README.md              # 이 문서
├── __init__.py
├── schema.py              # JSON 자료구조 정의
├── similarity.py          # TF-IDF 유사도 계산
└── output/
    └── golden_dataset_v1.json  # 생성된 데이터셋
```

---

## 사용법

### 1. Golden Dataset 생성

```python
from similarity import build_golden_dataset

build_golden_dataset(
    target_path="sampling/output/target_120.csv",
    corpus_path="sampling/output/corpus_sampled_6000.csv",
    output_path="golden_dataset/output/golden_dataset_v1.json",
    min_similarity=0.1,
    max_similarity=0.8,
    top_k=10
)
```

### 2. 데이터셋 로드 및 사용

```python
from schema import GoldenDataset

dataset = GoldenDataset.load("output/golden_dataset_v1.json")

for target in dataset.targets:
    print(f"{target.doc_title}")
    print(f"  Related docs: {len(target.related_documents)}")
    print(f"  QA pairs: {len(target.qa_pairs)}")
```

---

## 다음 단계

1. [ ] LLM 기반 질문 생성 (Single-hop, Multi-hop)
2. [ ] Human-in-the-Loop 검증
3. [ ] 최종 QA 데이터셋 완성 (qa_pairs 채우기)
