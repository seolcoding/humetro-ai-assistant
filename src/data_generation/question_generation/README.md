# Question Generation Module

Golden Dataset 질문 생성 모듈

---

## 목적

타겟 문서(120개)에 대해 5가지 유형의 질문을 LLM으로 생성하고, Human-in-the-Loop 검증을 통해 최종 Golden Dataset 완성

---

## 질문 유형별 분포

| 유형 | 골든 (토픽당) | 풀 1.5배 | 필요 문서 | 난이도 |
|------|-------------|---------|----------|--------|
| Simple Factoid | 4 | 6 | 1개 | ⭐ |
| Constraint | 4 | 6 | 1개 | ⭐⭐ |
| Multi-hop (2-hop) | 6 | 9 | 2개 | ⭐⭐⭐ |
| Multi-hop (3-hop) | 3 | 5 | 3개 | ⭐⭐⭐⭐ |
| Reasoning | 3 | 4 | 1개 | ⭐⭐⭐ |
| **합계** | **20** | **30** | | |

---

## 핵심 설계 원칙

### 1. Passage Dependency (단락 의존성)

> "LLM 사전지식으로 답변 가능한 질문은 RAG 평가에 부적합"

```
❌ Bad: "대한민국의 수도는?" → LLM이 문서 없이 답변 가능
✅ Good: "2024년 강남구 복지예산 비율은?" → 문서 필수
```

**근거**: 강의자료 2-6, RAGAS Faithfulness 지표

### 2. Multi-hop 연결 조건

| hop 수 | 유사도 범위 | 근거 |
|--------|------------|------|
| 2-hop | 0.2 ~ 0.5 | 적절한 관련성, 다른 정보 |
| 3-hop | 체인 구조 | 중간 문서 필수 (직접 연결 불가) |

**근거**:
- 유사도 > 0.5: 거의 동일 내용, hop 불필요
- 유사도 < 0.2: 무관한 문서, 부자연스러운 연결

### 3. Human-in-the-Loop

```
골든 풀 (180) → 자동 필터 → Human 검증 → 골든 (120)
```

검증 기준:
- 질문 자연스러움
- 답변 추출 가능성
- Multi-hop 필수성 확인

---

## 파일 구조

```
question_generation/
├── README.md               # 이 문서
├── DESIGN.md               # 상세 설계 (프롬프트, 알고리즘)
├── MODEL_SELECTION.md      # GPT-5.1 모델 선택 근거
├── __init__.py
├── prompts/                # 질문 유형별 프롬프트
│   ├── system_prompt.py    # 공통 시스템 프롬프트
│   └── question_prompts.py # 5가지 유형별 프롬프트
├── generators/             # 질문 생성기
│   └── question_generator.py  # GPT-5.1 API 호출
├── doc_selectors/          # Multi-hop 문서 선정
│   └── multihop_selector.py   # 2-hop, 3-hop 문서 선정
├── converters/             # 포맷 변환
│   └── autorag_converter.py   # AutoRAG parquet 변환
├── run_test_generation.py  # 테스트 파이프라인
└── output/
    ├── selection_plan.json    # 문서 선정 계획
    └── test_run/              # 테스트 결과
```

---

## GPT-5.1 모델 선택 근거

**선택 모델**: `gpt-5.1` (Reasoning Effort: `medium`, Verbosity: `medium`)

### 왜 GPT-5.1인가?

| 항목 | GPT-5.1 | 비고 |
|------|---------|------|
| SWE-bench Verified | 76.3% | GPT-5 대비 +3.5% |
| 토큰 효율성 | -23% | 복잡 작업 시 |
| 처리 속도 | 2-3x faster | GPT-5 대비 |
| Intelligence Index | 68 (high) | 최고 수준 |

**출처**: [OpenAI GPT-5.1 for Developers](https://openai.com/index/gpt-5-1-for-developers/)

### 왜 Reasoning Effort Medium인가?

| Effort | 적합 작업 | 판단 |
|--------|----------|------|
| none | 단순 분류, 추출 | X |
| low | 간단한 QA | X |
| **medium** | **복잡한 생성 작업** | **선택** |
| high | 수학, 코딩 | 과도 |

- 질문 생성은 "적절한 복잡도"의 작업
- Multi-hop 추론 경로 설계에 적합
- 비용-성능 균형점

상세: [MODEL_SELECTION.md](./MODEL_SELECTION.md)

---

## 사용법

### 1. Mock 테스트 (API 비용 없음)

```bash
# 5개 질문 테스트 (공공행정 토픽)
python run_test_generation.py --mock --count 5 --topic 공공행정
```

### 2. 실제 API 테스트 (소수만)

```bash
# 2개만 생성 (비용 확인)
python run_test_generation.py --use-api --count 2
```

### 3. 문서 선정 계획 확인

```python
from doc_selectors import MultihopDocumentSelector

selector = MultihopDocumentSelector("golden_dataset/output/golden_dataset_v1.json")

# 2-hop 문서 쌍 확인
pairs = selector.select_2hop_pairs(topic="공공행정", count_per_target=1)
for pair in pairs[:3]:
    print(f"Target: {pair.target_doc_id}")
    print(f"Related: {pair.related_doc_id} (sim: {pair.similarity_score:.3f})")

# 선정 계획 내보내기
selector.export_selection_plan("output/selection_plan.json")
```

### 4. AutoRAG 포맷 변환

```python
from converters import AutoRAGConverter

converter = AutoRAGConverter()
qa_list = [...]  # 생성된 QA 리스트

autorag_qa = converter.convert_qa_list(qa_list, id_prefix="golden")
converter.save_parquet(autorag_qa, "output/qa.parquet")
```

---

## 계층 샘플링 (Tier System)

```
Golden Pool (180) → Golden (120) → Silver (60) → Bronze (30)
                        ↓
                  Human-in-the-Loop
```

| 등급 | 유형당 개수 | 총계 | 용도 |
|------|------------|------|------|
| Bronze | 1 | 30 | 빠른 파일럿 |
| Silver | 2 | 60 | 중간 검증 |
| Golden | 4/4/6/3/3 | 120 | 최종 평가 |
| Pool | 6/6/9/5/4 | 180 | 선별 전 후보 |

---

## 참고

- [DESIGN.md](./DESIGN.md): 상세 프롬프트, 알고리즘, 이론적 배경
- [golden_dataset/README.md](../golden_dataset/README.md): 타겟 문서 및 유사 문서 그룹
- [thesis/README.md](../../../thesis/README.md): 연구 가설 및 평가 지표

---

**작성일**: 2025-12-02
