# RAG 평가용 데이터셋 합성 전략 분석

> 본 문서는 다산콜센터 대화 데이터로부터 RAG 평가용 데이터셋을 생성하는 전략을 분석합니다.
> 다른 모델/전문가의 비판적 검토를 위해 작성되었습니다.

---

## 1. 프로젝트 배경

### 1.1 목표
- **연구 주제**: 온프레미스 오픈소스 Graph RAG 시스템의 공공부문 적용
- **평가 목표**: 다양한 RAG 방법론(Naive RAG, KG Simple, KG Cypher, LightRAG) 성능 비교
- **핵심 과제**: 신뢰할 수 있는 평가용 데이터셋 구축

### 1.2 현재 보유 데이터
- **원본**: AI Hub 다산콜센터 민원 상담 대화 데이터 (~45,000 대화)
- **형식**: 턴 단위 JSON (화자, 발화, 의도, 개체명 등 라벨링 완료)
- **문제**: 원본 매뉴얼/KB 문서 없음 (상담사가 참조한 원본 자료 부재)

---

## 2. 원본 데이터 구조

### 2.1 JSON 스키마

```json
{
  "도메인": "다산콜센터",
  "카테고리": "대중교통 안내",
  "대화셋일련번호": "B2033",
  "화자": "고객" | "상담사",
  "문장번호": "1",
  "QA": "Q" | "A",
  "고객의도": "버스노선",
  "상담사의도": "",
  "고객질문(요청)": "서울 가산동에서 남대문시장가는 버스노선을 알고싶습니다",
  "상담사질문(요청)": "",
  "고객답변": "",
  "상담사답변": "",
  "개체명": "서울, 가산동, 남대문시장, 버스, 노선",
  "용어사전": "서울/지명/ 가산동/동네/ 남대문시장/지명/ 버스/교통수단",
  "지식베이스": "가산동,교통수단"
}
```

### 2.2 대화 예시 (B2033)

| 턴 | 화자 | QA | 발화 | 의도 |
|----|------|----|----|------|
| 1 | 고객 | Q | 서울 가산동에서 남대문시장가는 버스노선을 알고싶습니다 | 버스노선 |
| 2 | 상담사 | Q | 가산동 어디에서 출발하십니까? | 버스노선 |
| 3 | 고객 | A | 가산동 주민센터입니다. | 버스노선 |
| 4 | 상담사 | A | 가산동 주민센터에서 남대문시장으로 가는 버스노선은 505번 버스입니다. | 버스노선 |
| 5 | 고객 | Q | 어느정류장에서 타야합니까? | 버스정류장 |
| 6 | 상담사 | A | 문성초등학교 정류장에서 탑승하시면 됩니다. | 버스정류장 |
| 7 | 고객 | Q | 버스요금은 얼마입니까? | 버스요금 |
| 8 | 상담사 | A | 1200원 입니다. | 버스요금 |

### 2.3 데이터 특성

| 항목 | 설명 |
|------|------|
| 대화 수 | ~45,000개 |
| 평균 턴 수 | ~20턴/대화 |
| 카테고리 | 코로나19, 대중교통, 생활/복지, 행정 등 |
| 라벨링 | 의도, 개체명, 용어사전, 지식베이스 완료 |

---

## 3. RAG 평가를 위한 필수 구성요소

### 3.1 표준 RAG 평가 데이터셋 구조

```
┌─────────────────────────────────────────────────────────┐
│  1. Corpus (Retrieval 대상)                              │
│     - 검색 가능한 문서/청크 집합                           │
│     - 예: 매뉴얼, FAQ, 정책문서                           │
└─────────────────────────────────────────────────────────┘
                          ↓ Retrieval
┌─────────────────────────────────────────────────────────┐
│  2. QA Testset (평가용)                                  │
│     - Question: 사용자 질문                               │
│     - Ground Truth: 정답 답변                            │
│     - Retrieval GT: 정답 문서/청크 ID (선택)              │
└─────────────────────────────────────────────────────────┘
```

### 3.2 현재 상황의 문제

```
[보유 데이터]
✅ 대화 로그 (QA 추출 가능)
❌ 매뉴얼/KB 문서 (Corpus 없음)

[문제]
대화 로그 ≠ 검색 가능한 지식 문서
- 대화는 맥락 의존적, 단편적
- 매뉴얼은 체계적, 자기 완결적
```

---

## 4. 제안된 데이터셋 합성 전략

### 4.1 전체 파이프라인

```
[원본 대화 데이터]
        │
        ├─────────────────────────────────┐
        │                                 │
        ▼                                 ▼
┌─────────────────┐            ┌─────────────────────┐
│ QA Testset 추출  │            │ Fact 추출 → 매뉴얼   │
│ (합성 최소화)    │            │ (합성 필요)          │
└─────────────────┘            └─────────────────────┘
        │                                 │
        │                                 ▼
        │                      ┌─────────────────────┐
        │                      │ Corpus (매뉴얼/KB)   │
        │                      │ = Retrieval 대상     │
        │                      └─────────────────────┘
        │                                 │
        ▼                                 ▼
┌───────────────────────────────────────────────────────┐
│                    RAG 평가                            │
│ Query(원본) → Retrieval(매뉴얼) → Generation → GT(원본) │
└───────────────────────────────────────────────────────┘
```

### 4.2 Phase 1: QA Testset 추출

**목표**: 원본 대화에서 Question과 Ground Truth를 추출 (합성 최소화)

**방법**:
```python
def extract_qa_from_dialogue(dialogue):
    """
    대화에서 QA 쌍 추출
    - Question: 고객질문(요청) 필드 그대로 사용
    - Ground Truth: 상담사답변 필드 그대로 사용
    """
    qa_pairs = []
    turns = sorted(dialogue["turns"], key=lambda x: x["turn_no"])

    for i, turn in enumerate(turns):
        if turn["qa_type"] == "Q" and turn["고객질문(요청)"]:
            question = turn["고객질문(요청)"]

            # 대응하는 답변 찾기
            answers = []
            for j in range(i+1, len(turns)):
                if turns[j]["qa_type"] == "A" and turns[j]["상담사답변"]:
                    answers.append(turns[j]["상담사답변"])
                elif turns[j]["qa_type"] == "Q":
                    break

            if answers:
                qa_pairs.append({
                    "question": question,
                    "ground_truth": " ".join(answers),
                    "source": "extracted",  # 합성 아님
                    "dialogue_id": dialogue["id"]
                })

    return qa_pairs
```

**출력 예시**:
```json
{
  "question_id": "Q001",
  "question": "서울 가산동에서 남대문시장가는 버스노선을 알고싶습니다",
  "ground_truth": "가산동 주민센터에서 남대문시장으로 가는 버스노선은 505번 버스입니다.",
  "source": "extracted",
  "dialogue_id": "B2033",
  "category": "대중교통 안내"
}
```

### 4.3 Phase 2: Fact 추출

**목표**: 대화에서 개별 사실(Fact) 정보 추출

**방법 A - 규칙 기반** (합성 최소화):
```python
def extract_facts_rule_based(dialogue):
    """개체명, 용어사전, 지식베이스 필드 활용"""
    facts = []
    for turn in dialogue["turns"]:
        if turn["상담사답변"]:
            facts.append({
                "content": turn["상담사답변"],
                "entities": turn["개체명"],
                "category": turn["상담사의도"],
                "source_dialogue": dialogue["id"]
            })
    return facts
```

**방법 B - LLM 보조** (구조화 목적):
```python
def extract_facts_llm(dialogue):
    """LLM으로 사실 추출 및 정규화"""
    prompt = """
    다음 대화에서 사실(Fact) 정보만 추출하세요.
    - 새로운 정보를 생성하지 마세요
    - 대화에 명시된 내용만 추출하세요

    출력 형식:
    - [카테고리] 사실 내용
    """
    return llm.generate(prompt + dialogue_text)
```

**출력 예시**:
```json
[
  {
    "fact_id": "F001",
    "category": "버스노선",
    "content": "505번 버스: 가산동 주민센터 → 남대문시장",
    "source_dialogue": "B2033"
  },
  {
    "fact_id": "F002",
    "category": "버스정류장",
    "content": "505번 버스 탑승: 문성초등학교 정류장",
    "source_dialogue": "B2033"
  },
  {
    "fact_id": "F003",
    "category": "버스요금",
    "content": "버스 요금: 1,200원",
    "source_dialogue": "B2033"
  }
]
```

### 4.4 Phase 3: 매뉴얼/KB 생성 (Corpus)

**목표**: 추출된 Fact들을 검색 가능한 매뉴얼 문서로 구조화

**방법**:
```python
def generate_manual_from_facts(facts_by_category):
    """
    Fact들을 매뉴얼 형태로 구조화
    - 카테고리별 그룹핑
    - 중복 제거 및 통합
    - 문서 형태로 포맷팅
    """
    manual_sections = []

    for category, facts in facts_by_category.items():
        section = {
            "doc_id": f"manual_{category}",
            "title": f"{category} 안내",
            "content": format_as_manual(facts),
            "metadata": {
                "category": category,
                "fact_count": len(facts),
                "source_dialogues": list(set(f["source_dialogue"] for f in facts))
            }
        }
        manual_sections.append(section)

    return manual_sections
```

**출력 예시**:
```markdown
# 대중교통 안내 매뉴얼

## 1. 버스 노선

### 1.1 가산동 출발
- **505번 버스**: 가산동 주민센터 → 남대문시장
  - 탑승 정류장: 문성초등학교

### 1.2 서울역 출발
- **750A, 750B번 버스**: 서울역 → 서울대학교

## 2. 요금 안내

| 구분 | 요금 |
|------|------|
| 일반 | 1,200원 |
```

### 4.5 Phase 4: Retrieval GT 매핑

**목표**: 각 Question에 대해 정답 문서/청크 지정

**방법**:
```python
def map_retrieval_gt(qa_pair, corpus):
    """
    QA와 Corpus 간 Retrieval GT 매핑
    - 동일 대화에서 추출된 Fact가 포함된 문서 찾기
    """
    relevant_docs = []
    source_dialogue = qa_pair["dialogue_id"]

    for doc in corpus:
        if source_dialogue in doc["metadata"]["source_dialogues"]:
            relevant_docs.append(doc["doc_id"])

    return relevant_docs
```

---

## 5. 비판적 검토

### 5.1 "Fact 추출"도 합성이다

| 구분 | 원본 | 추출 결과 |
|------|------|----------|
| 상담사 발화 | "1200원입니다" | "버스 요금: 1,200원" |
| 문제점 | - | 주어 추론, 형식 변환 발생 |

**결론**: "추출"이라 해도 형식 변환(transformation)이 발생하며, 이는 암묵적 합성임

### 5.2 Retrieval GT 매핑의 모호성

```
Q: "가산동에서 남대문 가는 버스요?"
GT: "505번, 문성초등학교 정류장, 1200원"

필요한 매뉴얼 섹션:
- 섹션 A: 버스 노선 (505번)
- 섹션 B: 정류장 안내 (문성초등학교)
- 섹션 C: 요금 안내 (1200원)

문제: 1:N 매핑으로 복잡, 자동화 어려움
```

### 5.3 매뉴얼 품질 = RAG 성능 상한

```
매뉴얼에서 "문성초등학교 정류장" 누락 시
→ RAG가 아무리 잘해도 정답 불가능
→ 평가 결과가 RAG가 아닌 매뉴얼 품질 반영
```

### 5.4 대화 맥락의 불가피한 손실

```
원본:
  Q: "어느 정류장에서 타야합니까?"  (맥락: 505번 버스)

매뉴얼화 후:
  "505번 버스 탑승: 문성초등학교 정류장"

문제: 조건부 맥락이 flatten됨
```

### 5.5 "원본 QA"의 한계

```
후속 질문들:
- "어느 정류장에서 타야합니까?" → 단독으로 불완전
- "시간은 얼마정도 걸립니까?" → 맥락 없이 해석 불가

해결책:
- 첫 질문만 사용 (샘플 수 감소)
- 또는 맥락 정보 추가 (합성 증가)
```

---

## 6. 합성 수준별 신뢰도 분석

| 구성요소 | 합성 수준 | 신뢰도 | 비고 |
|----------|----------|--------|------|
| Question (첫 질문) | 없음 (원본) | 🟢 높음 | 고객 실제 발화 |
| Question (후속 질문) | 맥락 추가 필요 | 🟠 중간 | 단독 사용 시 불완전 |
| Ground Truth | 없음 (원본) | 🟢 높음 | 상담사 실제 답변 |
| Fact | 형식 변환 | 🟠 중간 | 추출이지만 변환 발생 |
| 매뉴얼 구조 | 합성 필요 | 🟡 낮음-중간 | Fact 기반이나 구조화 |
| Retrieval GT | 규칙/수동 | 🟠 중간 | 1:N 매핑 복잡 |

---

## 7. 대안 전략 비교

### 7.1 전략 A: 현재 제안 (Fact 추출 → 매뉴얼)

```
장점:
- QA의 Q/GT가 원본이라 평가 신뢰도 높음
- 매뉴얼이 실제 RAG 시스템과 유사한 구조

단점:
- Fact 추출/매뉴얼 생성에 합성 필요
- Retrieval GT 매핑 복잡
```

### 7.2 전략 B: 대화 자체를 Corpus로 사용

```
장점:
- 합성 최소화
- Retrieval GT 매핑 단순 (같은 대화 = 정답)

단점:
- 대화가 검색에 부적합 (맥락 의존적)
- 실제 RAG 시스템과 괴리
```

### 7.3 전략 C: AutoRAG 프레임워크 활용

```
장점:
- 표준화된 파이프라인
- Retrieval GT 자동 생성

단점:
- QA 모두 LLM 합성 (원본 활용 안 됨)
- 순환 의존성 문제
```

### 7.4 전략 D: Human Annotation

```
장점:
- 가장 높은 신뢰도
- 논문 심사에서 강점

단점:
- 시간/비용 소요
- 50개 한정 시 현실적
```

---

## 8. 권장 하이브리드 전략

```
[50개 Golden Testset]

├── 40개: 전략 A (Fact 추출 → 매뉴얼)
│   └── Q/GT: 원본 추출
│   └── Corpus: Fact 기반 매뉴얼
│   └── Retrieval GT: 규칙 기반 매핑
│
└── 10개: 전략 D (Human Annotation)
    └── Q/GT: 원본 추출 + 검증
    └── Corpus: 동일
    └── Retrieval GT: 수동 검증
    └── 용도: "Human-verified subset"으로 신뢰도 앵커
```

---

## 9. 최종 데이터셋 스키마

### 9.1 QA Testset (qa.parquet)

```python
{
    "qid": str,                    # 질문 ID
    "question": str,               # 원본 고객 질문
    "ground_truth": str,           # 원본 상담사 답변
    "retrieval_gt": List[str],     # 정답 문서 ID 리스트
    "dialogue_id": str,            # 원본 대화 ID
    "category": str,               # 카테고리
    "question_type": str,          # single-hop / multi-hop
    "turn_position": str,          # first / follow-up
    "source": str,                 # extracted / annotated
    "annotator": Optional[str]     # Human annotation인 경우
}
```

### 9.2 Corpus (corpus.parquet)

```python
{
    "doc_id": str,                 # 문서 ID
    "title": str,                  # 문서 제목
    "contents": str,               # 문서 내용
    "metadata": {
        "category": str,
        "fact_count": int,
        "source_dialogues": List[str]
    }
}
```

### 9.3 Facts (facts.jsonl) - 중간 산출물

```python
{
    "fact_id": str,
    "category": str,
    "content": str,
    "entities": List[str],
    "source_dialogue": str,
    "source_turns": List[int]
}
```

---

## 10. 평가 시 고려사항

### 10.1 신뢰 가능한 평가

```
[신뢰 가능]
- Answer Correctness: GT가 원본이므로 신뢰 가능
- Answer Relevancy: Q가 원본이므로 신뢰 가능

[주의 필요]
- Retrieval 성능: 매뉴얼 품질에 의존
- Faithfulness: 합성 Corpus 대비 측정
```

### 10.2 논문 기술 시 명시사항

```markdown
## Limitations

1. Corpus(매뉴얼)는 대화에서 추출한 Fact를 기반으로 합성됨
2. Retrieval GT는 규칙 기반 매핑으로, 완벽하지 않을 수 있음
3. 후속 질문의 경우 맥락 정보가 일부 손실됨
4. 10개 샘플은 Human Annotation으로 검증됨
```

---

## 11. 검토 요청 사항

본 문서를 검토하는 분께 다음 사항에 대한 의견을 요청드립니다:

1. **Fact 추출 방식**: 규칙 기반 vs LLM 보조, 어느 쪽이 더 적절한가?
2. **Retrieval GT 매핑**: 자동화 가능한 더 나은 방법이 있는가?
3. **매뉴얼 구조**: 청킹 단위를 어떻게 설정해야 하는가?
4. **평가 메트릭**: 합성된 Corpus에서 Faithfulness 측정이 의미 있는가?
5. **대안 전략**: 더 나은 접근법이 있는가?

---

## 부록: 참고 자료

- [RAGAS Documentation](https://docs.ragas.io/)
- [AutoRAG GitHub](https://github.com/Marker-Inc-Korea/AutoRAG)
- [AI Hub 다산콜센터 데이터셋](https://aihub.or.kr/)

---

*문서 작성일: 2025-01-28*
*프로젝트: Humetro AI Assistant - Graph RAG Research*
