# Question Generation Module Design

Golden Dataset 질문 생성을 위한 설계 문서

---

## 1. 개요

### 1.1 목표
- 골든 풀 180개 질문 생성 (Human-in-the-Loop 후 120개 선정)
- 7가지 질문 유형별 차별화된 프롬프트 설계
- Multi-hop 질문을 위한 관련 문서 선정 기준 정의
- GPT-5.1 API를 활용한 고품질 질문 생성

### 1.2 질문 유형별 배분 (토픽당 30개)

| 유형 | 풀 | 비율 | 특성 | 난이도 |
|------|-----|------|------|--------|
| **Simple (단일 문서)** | **12개** | **40%** | | |
| └ Simple Factoid | 6개 | 20% | 단일 문서, 단일 사실 | ⭐ |
| └ Constraint | 3개 | 10% | 단일 문서, 조건부 | ⭐⭐ |
| └ Reasoning | 3개 | 10% | 단일 문서, 추론/인과 | ⭐⭐⭐ |
| **Advanced (다중 문서)** | **18개** | **60%** | | |
| └ Multi-doc (1-hop) | 6개 | 20% | 여러 문서, 비교/종합 | ⭐⭐ |
| └ Multi-hop (2-hop) | 6개 | 20% | 2개 문서 순차 연결 | ⭐⭐⭐ |
| └ Multi-hop (3-hop) | 3개 | 10% | 3개 문서 순차 연결 | ⭐⭐⭐⭐ |
| └ Multi-hop (5-hop) | 3개 | 10% | 5개 문서 연결 (선형/비선형) | ⭐⭐⭐⭐⭐ |
| **합계** | **30개** | 100% | | |

### 1.2.1 전체 규모 (6개 토픽)

| 유형 | 토픽당 | 총계 | 비율 |
|------|--------|------|------|
| Simple Factoid | 6개 | **36개** | 20% |
| Constraint | 3개 | **18개** | 10% |
| Reasoning | 3개 | **18개** | 10% |
| Multi-doc 1-hop | 6개 | **36개** | 20% |
| Multi-hop 2-hop | 6개 | **36개** | 20% |
| Multi-hop 3-hop | 3개 | **18개** | 10% |
| Multi-hop 5-hop | 3개 | **18개** | 10% |
| **합계** | 30개 | **180개** | 100% |

### 1.2.2 비율 설계 근거

```
Simple 40% : Advanced 60% 비율 선택 이유:

1. RAG 분별력 확보
   - Simple만으로는 Naive RAG와 Advanced RAG 차이 미미
   - Multi-hop/Multi-doc에서 Reranker, GraphRAG 효과 극대화

2. 현실적 난이도 분포
   - 실제 민원 QA도 단순 질문(40%)과 복잡 질문(60%) 혼재
   - 베이스라인(Simple)과 분별력(Advanced) 균형

3. 논문 기여도
   - 기존 한국어 RAG 벤치마크: Multi-hop 비중 낮음
   - 60% Multi-doc/Multi-hop으로 차별화
```

### 1.2.3 Multi-doc (1-hop) vs Multi-hop 구분

```
┌─────────────────────────────────────────────────────────────┐
│ Multi-doc (1-hop): 병렬적 정보 통합                          │
│   - 여러 문서가 필요하지만 추론 단계는 1개                    │
│   - 비교, 종합, 공통점/차이점, 추세 분석                      │
│   - 예: "A 정책과 B 정책의 공통점은?" → 두 문서 동시 참조     │
│                                                              │
│   문서A ─┐                                                   │
│   문서B ─┼→ 비교/종합 → 답변                                  │
│   문서C ─┘                                                   │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Multi-hop (N-hop): 순차적 추론 체인                          │
│   - N개의 순차적 추론 단계 필요                              │
│   - 엔티티 연결, 정보 전파                                   │
│   - 예: "A의 담당자가 수상한 상은?"                          │
│         → A→담당자(hop1)→상(hop2)                           │
│                                                              │
│   문서A → 문서B → 문서C → 답변                               │
└─────────────────────────────────────────────────────────────┘
```

### 1.2.4 5-hop 구조 유형

5-hop 질문은 선형/비선형 구조 모두 허용:

```
┌─────────────────────────────────────────────────────────────┐
│ 선형 구조 (Linear):                                          │
│                                                              │
│   A → B → C → D → E → 답변                                   │
│                                                              │
│   각 hop에서 순차적으로 새로운 정보 획득                      │
│   예: 부서→담당자→출신학교→인증→혜택                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ 비선형 구조 (Branching):                                     │
│                                                              │
│   A ─┬→ B → D ─┬→ E → 답변                                   │
│      └→ C ─────┘                                             │
│                                                              │
│   병렬 경로 탐색 후 병합                                      │
│   예: 정책→(예산+일정)→통합결과→최종답변                      │
└─────────────────────────────────────────────────────────────┘
```

### 1.3 전체 규모

| 등급 | 토픽당 | 총계 (6토픽) |
|------|--------|-------------|
| 골든 풀 | 30 | **180** |
| 골든 | 20 | 120 |
| 실버 | 10 | 60 |
| 브론즈 | 5 | 30 |

---

## 2. 이론적 배경 및 근거

### 2.1 좋은 질문의 5가지 특성 (강의자료)

| 특성 | 설명 | 우리 적용 |
|------|------|----------|
| **현실성** | 실제 사용자가 할 것 같은 질문 | 민원 QA 시나리오 기반 |
| **다양성** | 다양한 질문 패턴/스타일 | 5가지 유형으로 분류 |
| **도메인 연관성** | 행정문서와 밀접한 질문 | 6개 공공 토픽 선정 |
| **명확성** | 질문 의도가 명확함 | 프롬프트에 명시적 지침 |
| **분별력** | RAG 성능 차이 드러냄 | Multi-hop으로 난이도 조절 |

**근거**:
> "분별력이 핵심: 기울기를 만들어야 RAG 최적화 효과가 드러남" (강의자료 2-6)

### 2.2 Multi-hop vs Multi-passage (강의자료)

```
Multi-hop: 두 개 이상의 연속된 정보를 이용해서 대답해야 하는 질문
Multi-passage: 정답 단락이 두 개 이상인 질문

관계: Multi-passage이면 보통 Multi-hop입니다
```

**예시**:
```
질문: "나성범 선수가 주장인 팀은 한국시리즈 우승을 몇 번 하였는가?"

단락 1: "기아 타이거즈의 주장은 나성범이다."
단락 2: "기아 타이거즈는 KBO 한국시리즈 11회 우승했다."

추론 경로:
  Hop 1: 나성범 → 기아 타이거즈 (단락 1)
  Hop 2: 기아 타이거즈 → 11회 우승 (단락 2)
```

### 2.3 Passage Dependency (강의자료)

**핵심 원칙**: Advanced RAG 효과를 측정하려면 **반드시 단락 의존적 질문**이어야 함

```
❌ Bad: "대한민국의 수도는 어디인가?" → LLM이 단락 없이도 답변 가능
✅ Good: "2024년 강남구 예산 중 복지 항목 비율은?" → 단락 필수
```

**Passage Dependency Filter**:
- LLM이 컨텍스트 없이 답변 가능한 질문 제거
- Multi-hop 질문에서 단락 하나라도 빠지면 Unanswerable

### 2.4 RAGAS 평가 관점 (Es et al., 2023)

**Faithfulness (충실성)**:
> "답변이 주어진 컨텍스트에 근거해야 한다. 이는 환각(hallucination)을 방지하고 검색된 컨텍스트가 생성된 답변에 대한 정당화 역할을 할 수 있도록 보장하는 데 중요하다."

**질문 생성 시 적용**:
- 답변이 반드시 문서에서 추출 가능해야 함
- 모호하거나 해석이 필요한 질문 지양
- 명확한 정답이 존재하는 질문 설계

### 2.5 RAG vs GraphRAG 인사이트 (Han et al., 2025)

| 질문 유형 | 적합 시스템 | 우리 적용 |
|----------|------------|----------|
| 단일 홉, 세부 정보 | RAG | Simple Factoid |
| 멀티홉 추론 | GraphRAG | Multi-hop 2/3 |
| 비교/시간 순서 | GraphRAG | Constraint |
| 인과관계 | GraphRAG | Reasoning |

**인용**:
> "RAG and GraphRAG are complementary, each excelling in different aspects... Integration strategy improves the best method by 6.4%."

---

## 3. 질문 유형별 설계

### 3.1 Simple Factoid (토픽당 6개 → 4개)

**정의**: 단일 문서에서 단일 사실을 추출하는 질문

**특성**:
- Retrieval GT: 1개 문서
- 난이도: ⭐ (최하)
- RAG 분별력: 낮음 (베이스라인)

**생성 전략**:
```
1. 타겟 문서에서 핵심 사실 추출 (날짜, 이름, 숫자, 장소)
2. 해당 사실에 대한 직접적 질문 생성
3. LLM 사전지식으로 답변 불가능한 것만 선정
```

**프롬프트 설계 근거**:
- GPH Scheme의 Concept Completion, Quantification 유형 참조
- "~은 무엇인가?", "~은 얼마인가?" 형태

**예시**:
```
문서: "2024년 강남구 청소년 예산은 52억원으로 전년 대비 15% 증가했다."
질문: "2024년 강남구 청소년 예산은 얼마인가?"
답변: "52억원"
```

---

### 3.2 Constraint (토픽당 6개 → 4개)

**정의**: 특정 조건/제약이 포함된 질문

**특성**:
- Retrieval GT: 1개 문서
- 난이도: ⭐⭐
- RAG 분별력: 중간 (조건 인식 필요)

**생성 전략**:
```
1. 문서에서 여러 항목/시점/조건 포함된 정보 탐색
2. 특정 조건을 명시한 질문 생성
3. 조건 없이는 답변이 모호해지는 구조
```

**프롬프트 설계 근거**:
- 강의자료: "특정 제약이나 조건이 들어간 질문"
- Numerical, Constraint 유형 참조

**예시**:
```
문서: "2023년 1분기 민원 접수: 교통 1,234건, 환경 987건, 복지 2,345건
       2023년 2분기 민원 접수: 교통 1,456건, 환경 1,102건, 복지 2,123건"
질문: "2023년 1분기에 가장 많이 접수된 민원 유형은 무엇인가?"
답변: "복지 (2,345건)"
```

---

### 3.3 Multi-doc 1-hop (토픽당 6개) - NEW

**정의**: 여러 문서의 정보를 병렬적으로 통합하는 질문 (추론 단계는 1개)

**특성**:
- Retrieval GT: 2-4개 문서
- 난이도: ⭐⭐
- RAG 분별력: 중간 (다중 문서 검색 능력 검증)

**Multi-hop과의 차이**:
```
Multi-doc 1-hop: 문서A + 문서B + 문서C → 종합 → 답변 (병렬)
Multi-hop:       문서A → 문서B → 문서C → 답변 (순차)
```

**문서 선정 기준**:
```python
# 유사도가 높은 문서들 그룹화 (비교/종합하기 좋음)
# 유사도 0.25 이상의 관련 문서들
similar_docs = [d for d in related_docs if d.similarity_score >= 0.25]
```

**생성 전략**:
```
1. 타겟 문서와 유사한 주제의 문서 2-3개 선정
2. 비교, 종합, 공통점/차이점, 추세 분석 질문 생성
3. 모든 문서가 답변에 기여해야 함
```

**적합한 질문 유형**:
- "A와 B의 공통점/차이점은?"
- "여러 정책 중 가장 ~한 것은?"
- "A, B, C를 종합하면?"
- "전체적으로 어떤 추세인가?"

**예시**:
```
문서 1: "2023년 강남구 교통민원 1,234건"
문서 2: "2023년 서초구 교통민원 987건"
문서 3: "2023년 송파구 교통민원 1,456건"

질문: "2023년 강남구, 서초구, 송파구 중 교통민원이 가장 많은 구는?"
답변: "송파구 (1,456건)"
retrieval_gt: [문서1_id, 문서2_id, 문서3_id]
question_type: multi_doc_1
```

---

### 3.4 Multi-hop 2-hop (토픽당 6개)

**정의**: 2개 문서의 정보를 연결해야 답변 가능한 질문

**특성**:
- Retrieval GT: 2개 문서
- 난이도: ⭐⭐⭐
- RAG 분별력: **높음 (Reranker 효과 검증)**

**문서 선정 기준**:
```python
# TF-IDF 유사도 기반 선정
for target in targets:
    related_docs = target.related_documents

    # 2-hop 후보: 유사도 0.2 ~ 0.5 범위
    # - 너무 높으면 (>0.5): 거의 동일 내용, hop 불필요
    # - 너무 낮으면 (<0.2): 연결 고리 없음, 부자연스러운 질문
    candidates = [d for d in related_docs if 0.2 <= d.similarity_score <= 0.5]
```

**생성 전략**:
```
1. 타겟 문서에서 엔티티 A 추출
2. 관련 문서에서 A와 연결되는 엔티티 B 확인
3. A → B → 답변 형태의 추론 경로 설계
4. 두 문서 모두 필요한 질문 생성
```

**프롬프트 설계 근거**:
- 강의자료: "꼬리에 꼬리를 무는 사실 관계가 필요한 질문"
- Incremental Question Generation 방식 참조

**예시**:
```
문서 1: "강남구청 민원실장은 김철수이다."
문서 2: "김철수는 2024년 우수공무원상을 수상했다."

질문: "강남구청 민원실장이 2024년에 수상한 상은 무엇인가?"
추론: 민원실장 → 김철수 (문서1) → 우수공무원상 (문서2)
답변: "우수공무원상"
retrieval_gt: [문서1_id, 문서2_id]
```

---

### 3.4 Multi-hop 3-hop (토픽당 5개 → 3개)

**정의**: 3개 문서의 정보를 연결해야 답변 가능한 질문

**특성**:
- Retrieval GT: 3개 문서
- 난이도: ⭐⭐⭐⭐ (최고)
- RAG 분별력: **매우 높음 (Advanced RAG 효과 검증)**

**문서 선정 기준**:
```python
# 체인 구조 형성 가능한 문서 3개 선정
# A → B → C → 답변 형태

# 방법 1: 타겟 + 관련 문서 2개 (체인)
target → related_1 (sim 0.3~0.5) → related_2 (sim 0.2~0.4)

# 방법 2: 타겟 + 허브 문서 + 관련 문서
target → hub_doc (높은 연결성) → related_doc
```

**생성 전략**:
```
1. 타겟 문서에서 시작 엔티티 추출
2. 중간 문서에서 브릿지 엔티티 확인
3. 최종 문서에서 답변 정보 확인
4. 3단계 추론 경로 설계
```

**프롬프트 설계 근거**:
- 강의자료: Multi-hop Filtering - "정답 단락 중 하나만 사라져도 Unanswerable"
- 각 hop이 필수적인 구조 설계

**예시**:
```
문서 1: "2024년 강남구 환경정책 담당부서는 환경과이다."
문서 2: "환경과장은 이영희이다."
문서 3: "이영희는 환경공학 박사학위를 보유하고 있다."

질문: "2024년 강남구 환경정책 담당부서장의 최종학력은?"
추론: 환경정책 담당부서 → 환경과 (문서1) → 이영희 (문서2) → 박사 (문서3)
답변: "환경공학 박사"
retrieval_gt: [문서1_id, 문서2_id, 문서3_id]
```

---

### 3.5 Multi-hop 5-hop (토픽당 2개)

**정의**: 5개 문서의 정보를 연결해야 답변 가능한 질문

**특성**:
- Retrieval GT: 5개 문서
- 난이도: ⭐⭐⭐⭐⭐ (최고)
- RAG 분별력: **극도로 높음 (GraphRAG 필수 검증)**

**문서 선정 기준**:
```python
# 5단계 체인 구조 형성
# A → B → C → D → E → 답변

def select_5hop_documents(target, corpus):
    """
    5-hop 질문을 위한 문서 5개 선정

    조건:
    1. 각 hop에서 연결 엔티티 존재
    2. 직접 연결 불가 (중간 문서 필수)
    3. 마지막 문서에만 최종 답변 존재
    """
    chain = [target]

    for i in range(4):  # 4번 확장
        current = chain[-1]
        next_doc = find_best_bridge(current, corpus, exclude=chain)
        if next_doc:
            chain.append(next_doc)

    return chain if len(chain) == 5 else None
```

**생성 전략**:
```
1. 타겟 문서에서 시작 엔티티 추출
2. 4개의 브릿지 문서 순차 탐색
3. 각 hop에서 새로운 정보 추가
4. 5단계 추론 경로 설계
```

**예시**:
```
문서 1: "강남구 환경정책 담당부서는 환경과이다"
문서 2: "환경과장은 김철수이다"
문서 3: "김철수는 서울대 환경공학과 출신이다"
문서 4: "서울대 환경공학과는 2020년 환경부 인증을 받았다"
문서 5: "환경부 인증 학과 출신은 정부 연구과제 우선 배정 대상이다"

질문: "강남구 환경정책 담당부서장의 출신 학과가 받은 인증으로 인해
       어떤 혜택을 받을 수 있는가?"

추론:
  Hop 1: 환경정책 담당부서 → 환경과 (문서1)
  Hop 2: 환경과장 → 김철수 (문서2)
  Hop 3: 김철수 → 서울대 환경공학과 (문서3)
  Hop 4: 서울대 환경공학과 → 환경부 인증 (문서4)
  Hop 5: 환경부 인증 → 연구과제 우선 배정 (문서5)

답변: "정부 연구과제 우선 배정"
retrieval_gt: [문서1_id, 문서2_id, 문서3_id, 문서4_id, 문서5_id]
```

**중요 설계 원칙**:
- 어느 한 문서라도 빠지면 답변 불가능
- 각 hop이 논리적으로 연결되어야 함
- 직접 점프 불가 (A→E 직접 연결 금지)

---

### 3.6 Reasoning (토픽당 5개 → 3개)

**정의**: 인과관계, 비교, 추론이 필요한 질문

**특성**:
- Retrieval GT: 1~2개 문서
- 난이도: ⭐⭐⭐
- RAG 분별력: 높음 (LLM 추론 능력 검증)

**생성 전략**:
```
1. 문서에서 원인-결과 관계 탐색
2. 비교 가능한 항목 식별
3. "왜?", "어떻게?", "무엇 때문에?" 형태 질문 생성
```

**프롬프트 설계 근거**:
- GPH Scheme: Causal Antecedent, Causal Consequence, Comparison
- 강의자료: reasoning_evolve_ragas 함수 참조

**예시**:
```
문서: "2024년 강남구 교통민원이 30% 증가한 것은 신규 지하철 공사로 인한
       교통 혼잡 때문이다. 특히 삼성역 일대에서 민원이 집중되었다."

질문: "2024년 강남구 교통민원이 증가한 원인은 무엇인가?"
답변: "신규 지하철 공사로 인한 교통 혼잡"
question_type: reasoning
```

---

## 4. Multi-hop 문서 선정 알고리즘

### 4.1 2-hop 문서 선정

```python
def select_2hop_documents(target: TargetDocument) -> List[Tuple[str, str]]:
    """
    2-hop 질문을 위한 문서 쌍 선정

    선정 기준:
    1. 유사도 0.2 ~ 0.5 범위 (적절한 관련성)
    2. 공통 엔티티 존재 (연결 고리)
    3. 정보 보완성 (중복 아닌 추가 정보)
    """
    candidates = []

    for related in target.related_documents:
        sim = related.similarity_score

        # 유사도 필터
        if not (0.2 <= sim <= 0.5):
            continue

        # 공통 엔티티 확인 (간단한 휴리스틱)
        common_entities = extract_common_entities(
            target.context,
            related.context
        )

        if len(common_entities) >= 1:
            candidates.append({
                'doc': related,
                'similarity': sim,
                'common_entities': common_entities
            })

    # 유사도 기준 정렬, 상위 선택
    candidates.sort(key=lambda x: x['similarity'], reverse=True)
    return candidates[:6]  # 토픽당 6개 2-hop 질문용
```

### 4.2 3-hop 문서 선정

```python
def select_3hop_documents(
    target: TargetDocument,
    corpus: List[Document]
) -> List[Tuple[str, str, str]]:
    """
    3-hop 질문을 위한 문서 트리플 선정

    선정 기준:
    1. 체인 구조 형성 가능 (A → B → C)
    2. 각 hop에서 새로운 정보 추가
    3. 최종 답변이 마지막 문서에만 존재
    """
    triples = []

    for mid_doc in target.related_documents[:5]:
        # 중간 문서의 관련 문서 탐색
        mid_related = find_related_in_corpus(mid_doc, corpus)

        for end_doc in mid_related[:3]:
            # 체인 검증
            if is_valid_chain(target, mid_doc, end_doc):
                triples.append({
                    'start': target,
                    'middle': mid_doc,
                    'end': end_doc,
                    'chain_strength': calculate_chain_strength(...)
                })

    # 체인 강도 기준 정렬
    triples.sort(key=lambda x: x['chain_strength'], reverse=True)
    return triples[:3]  # 토픽당 3개 3-hop 질문용
```

### 4.3 체인 유효성 검증

```python
def is_valid_chain(doc1, doc2, doc3) -> bool:
    """
    3-hop 체인 유효성 검증

    조건:
    1. doc1 → doc2 연결 엔티티 존재
    2. doc2 → doc3 연결 엔티티 존재
    3. doc1과 doc3은 직접 연결 약함 (중간 문서 필수)
    """
    # 연결 엔티티 확인
    link_1_2 = extract_common_entities(doc1.context, doc2.context)
    link_2_3 = extract_common_entities(doc2.context, doc3.context)
    link_1_3 = extract_common_entities(doc1.context, doc3.context)

    # 조건 검증
    has_bridge = len(link_1_2) > 0 and len(link_2_3) > 0
    needs_middle = len(link_1_3) < 2  # 직접 연결 약함

    return has_bridge and needs_middle
```

---

## 5. 프롬프트 설계

### 5.1 공통 시스템 프롬프트

```
당신은 한국 공공행정 문서에 대한 질문을 생성하는 전문가입니다.

다음 원칙을 준수하세요:
1. 질문은 반드시 제공된 문서의 내용에 기반해야 합니다.
2. 일반 상식이나 LLM 사전지식으로 답변 불가능한 질문만 생성하세요.
3. 질문은 명확하고 구체적이어야 합니다.
4. 답변은 문서에서 직접 추출 가능해야 합니다.
5. 한국어로 자연스럽게 작성하세요.
```

**근거**:
- RAGAS Faithfulness: "답변이 주어진 컨텍스트에 근거해야 한다"
- 강의자료: Passage Dependency Filter 원칙

### 5.2 Simple Factoid 프롬프트

```
[시스템 프롬프트]

## 작업
다음 행정문서에서 **단순 사실 질문**을 생성하세요.

## 문서
{context}

## 질문 유형: Simple Factoid
- 날짜, 이름, 숫자, 장소 등 구체적 사실을 묻는 질문
- 단일 문장으로 답변 가능
- "~은 무엇인가?", "~은 언제인가?", "~은 얼마인가?" 형태

## 출력 형식 (JSON)
{
  "question": "질문 내용",
  "answer": "답변 (문서에서 추출)",
  "evidence": "답변의 근거가 되는 문서 내 문장"
}

## 예시
문서: "2024년 강남구 청소년 예산은 52억원으로 전년 대비 15% 증가했다."
출력: {
  "question": "2024년 강남구 청소년 예산은 얼마인가?",
  "answer": "52억원",
  "evidence": "2024년 강남구 청소년 예산은 52억원으로"
}
```

### 5.3 Constraint 프롬프트

```
[시스템 프롬프트]

## 작업
다음 행정문서에서 **조건부 질문**을 생성하세요.

## 문서
{context}

## 질문 유형: Constraint
- 특정 조건이나 제약이 포함된 질문
- 시점, 범위, 대상 등을 한정하는 조건 필수
- 조건 없이는 답변이 모호해지는 구조

## 출력 형식 (JSON)
{
  "question": "질문 내용 (조건 포함)",
  "answer": "답변",
  "constraint": "적용된 조건",
  "evidence": "답변의 근거"
}

## 예시
문서: "2023년 1분기 교통민원 1,234건, 2분기 1,456건 접수"
출력: {
  "question": "2023년 1분기에 접수된 교통민원 건수는?",
  "answer": "1,234건",
  "constraint": "2023년 1분기",
  "evidence": "2023년 1분기 교통민원 1,234건"
}
```

### 5.4 Multi-hop 2-hop 프롬프트

```
[시스템 프롬프트]

## 작업
다음 두 문서의 정보를 연결하는 **2-hop 질문**을 생성하세요.

## 문서 1 (시작)
{context_1}

## 문서 2 (연결)
{context_2}

## 질문 유형: Multi-hop (2-hop)
- 두 문서의 정보를 모두 사용해야 답변 가능
- 문서 1에서 엔티티 A 확인 → 문서 2에서 A 관련 정보로 답변
- 한 문서만으로는 답변 불가능해야 함

## 출력 형식 (JSON)
{
  "question": "질문 내용",
  "answer": "답변",
  "reasoning_steps": [
    "Step 1: 문서 1에서 [정보 A] 확인",
    "Step 2: 문서 2에서 [정보 A]를 통해 [답변] 도출"
  ],
  "evidence_doc1": "문서 1의 근거 문장",
  "evidence_doc2": "문서 2의 근거 문장"
}

## 예시
문서 1: "강남구청 환경과장은 김철수이다."
문서 2: "김철수 과장은 2024년 우수공무원상을 수상했다."
출력: {
  "question": "강남구청 환경과장이 2024년에 수상한 상은?",
  "answer": "우수공무원상",
  "reasoning_steps": [
    "Step 1: 문서 1에서 환경과장이 김철수임을 확인",
    "Step 2: 문서 2에서 김철수가 우수공무원상 수상 확인"
  ],
  "evidence_doc1": "강남구청 환경과장은 김철수이다",
  "evidence_doc2": "김철수 과장은 2024년 우수공무원상을 수상했다"
}
```

### 5.5 Multi-hop 3-hop 프롬프트

```
[시스템 프롬프트]

## 작업
다음 세 문서의 정보를 연결하는 **3-hop 질문**을 생성하세요.

## 문서 1 (시작)
{context_1}

## 문서 2 (중간 브릿지)
{context_2}

## 문서 3 (최종 답변)
{context_3}

## 질문 유형: Multi-hop (3-hop)
- 세 문서의 정보를 순차적으로 연결해야 답변 가능
- 문서 1 → 문서 2 → 문서 3 체인 구조
- 중간 문서 없이는 시작과 끝을 연결할 수 없어야 함

## 출력 형식 (JSON)
{
  "question": "질문 내용",
  "answer": "답변",
  "reasoning_steps": [
    "Step 1: 문서 1에서 [정보 A] 확인",
    "Step 2: 문서 2에서 [정보 A]와 [정보 B] 연결",
    "Step 3: 문서 3에서 [정보 B]를 통해 [답변] 도출"
  ],
  "evidence_doc1": "문서 1의 근거",
  "evidence_doc2": "문서 2의 근거",
  "evidence_doc3": "문서 3의 근거"
}

## 중요
- 어느 한 문서라도 빠지면 답변이 불가능해야 합니다.
- 각 hop에서 새로운 정보가 추가되어야 합니다.
```

### 5.6 Multi-hop 5-hop 프롬프트

```
[시스템 프롬프트]

## 작업
다음 다섯 문서의 정보를 연결하는 **5-hop 질문**을 생성하세요.

## 문서 1 (시작)
{context_1}

## 문서 2 (브릿지 1)
{context_2}

## 문서 3 (브릿지 2)
{context_3}

## 문서 4 (브릿지 3)
{context_4}

## 문서 5 (최종 답변)
{context_5}

## 질문 유형: Multi-hop (5-hop)
- 다섯 문서의 정보를 순차적으로 연결해야 답변 가능
- 문서 1 → 문서 2 → 문서 3 → 문서 4 → 문서 5 체인 구조
- 어느 한 문서라도 빠지면 답변 불가능
- 직접 점프 불가 (시작에서 끝으로 바로 연결 금지)

## 출력 형식 (JSON)
{
  "question": "질문 내용",
  "answer": "답변",
  "reasoning_steps": [
    "Step 1: 문서 1에서 [정보 A] 확인",
    "Step 2: 문서 2에서 [정보 A]와 [정보 B] 연결",
    "Step 3: 문서 3에서 [정보 B]와 [정보 C] 연결",
    "Step 4: 문서 4에서 [정보 C]와 [정보 D] 연결",
    "Step 5: 문서 5에서 [정보 D]를 통해 [답변] 도출"
  ],
  "evidence_doc1": "문서 1의 근거",
  "evidence_doc2": "문서 2의 근거",
  "evidence_doc3": "문서 3의 근거",
  "evidence_doc4": "문서 4의 근거",
  "evidence_doc5": "문서 5의 근거"
}

## 중요
- 5개 문서 모두 필수적이어야 합니다.
- 각 hop에서 새로운 정보가 추가되어야 합니다.
- 논리적 연결 고리가 명확해야 합니다.
```

---

### 5.7 Reasoning 프롬프트

```
[시스템 프롬프트]

## 작업
다음 행정문서에서 **추론/인과 질문**을 생성하세요.

## 문서
{context}

## 질문 유형: Reasoning
- 원인-결과 관계를 묻는 질문
- "왜?", "어떻게?", "무엇 때문에?" 형태
- 단순 사실 나열이 아닌 관계 파악 필요

## 출력 형식 (JSON)
{
  "question": "질문 내용",
  "answer": "답변",
  "reasoning_type": "causal_antecedent | causal_consequence | comparison",
  "evidence": "답변의 근거"
}

## 예시
문서: "2024년 교통민원 30% 증가는 지하철 공사로 인한 혼잡 때문이다."
출력: {
  "question": "2024년 교통민원이 증가한 원인은 무엇인가?",
  "answer": "지하철 공사로 인한 교통 혼잡",
  "reasoning_type": "causal_antecedent",
  "evidence": "지하철 공사로 인한 혼잡 때문이다"
}
```

---

## 6. 품질 검증 (Filtering)

### 6.1 자동 필터링

```python
def auto_filter_questions(questions: List[QA]) -> List[QA]:
    """
    자동 품질 필터링

    필터 기준:
    1. Passage Dependency: LLM이 문서 없이 답변 가능한지 확인
    2. Answer Extractability: 답변이 문서에서 추출 가능한지
    3. Question Clarity: 질문이 명확한지
    """
    filtered = []

    for qa in questions:
        # 1. Passage Dependency Check
        if can_answer_without_context(qa.question):
            continue  # 제외

        # 2. Answer in Context
        if qa.answer.lower() not in qa.context.lower():
            continue  # 제외 (정확 매칭은 아니지만 기본 체크)

        # 3. Question Length
        if len(qa.question) < 10 or len(qa.question) > 200:
            continue  # 너무 짧거나 긴 질문 제외

        filtered.append(qa)

    return filtered
```

### 6.2 Human-in-the-Loop 체크리스트

```markdown
## Human 검증 항목

### 필수 체크
- [ ] 질문이 자연스러운 한국어인가?
- [ ] 답변이 문서에서 명확히 추출되는가?
- [ ] LLM 사전지식으로 답변 불가능한가?
- [ ] 질문 의도가 명확한가?

### Multi-hop 추가 체크
- [ ] 모든 연결 문서가 답변에 필수적인가?
- [ ] 추론 경로가 논리적인가?
- [ ] 한 문서라도 빠지면 답변 불가능한가?

### 등급 (1-5)
- 5: 완벽 (골든 선정)
- 4: 좋음 (골든 후보)
- 3: 보통 (실버/브론즈 후보)
- 2: 수정 필요
- 1: 제외
```

---

## 7. 구현 계획

### 7.1 모듈 구조

```
src/data_generation/question_generation/
├── DESIGN.md                    # 이 문서
├── README.md                    # 사용법
├── __init__.py
├── prompts/
│   ├── system_prompt.py         # 공통 시스템 프롬프트
│   ├── simple_factoid.py        # Simple Factoid 프롬프트
│   ├── constraint.py            # Constraint 프롬프트
│   ├── multi_hop_2.py           # 2-hop 프롬프트
│   ├── multi_hop_3.py           # 3-hop 프롬프트
│   └── reasoning.py             # Reasoning 프롬프트
├── generators/
│   ├── base_generator.py        # 기본 생성기
│   ├── single_doc_generator.py  # Single-hop 생성기
│   └── multi_hop_generator.py   # Multi-hop 생성기
├── selectors/
│   ├── document_selector.py     # Multi-hop 문서 선정
│   └── chain_validator.py       # 체인 유효성 검증
├── filters/
│   ├── passage_dependency.py    # Passage Dependency 필터
│   └── quality_filter.py        # 품질 필터
├── samplers/
│   └── tier_sampler.py          # 브론즈/실버/골든 샘플러
└── output/
    └── golden_pool_180.json     # 골든 풀 결과
```

### 7.2 실행 흐름

```
1. 골든 풀 생성 (180개)
   ┌─────────────────────────────────────────────────────────┐
   │ Simple (단일 문서) - 40%                                 │
   │   ├── Simple Factoid: 36개 (6토픽 × 6개) - 20%           │
   │   ├── Constraint:     18개 (6토픽 × 3개) - 10%           │
   │   └── Reasoning:      18개 (6토픽 × 3개) - 10%           │
   │   합계: 72개                                            │
   └─────────────────────────────────────────────────────────┘

   ┌─────────────────────────────────────────────────────────┐
   │ Advanced (다중 문서) - 60%                               │
   │   ├── Multi-doc 1-hop: 36개 (6토픽 × 6개) - 20%          │
   │   ├── Multi-hop 2-hop: 36개 (6토픽 × 6개) - 20%          │
   │   ├── Multi-hop 3-hop: 18개 (6토픽 × 3개) - 10%          │
   │   └── Multi-hop 5-hop: 18개 (6토픽 × 3개) - 10%          │
   │   합계: 108개                                           │
   └─────────────────────────────────────────────────────────┘

   → 총계: 180개 (Simple 72 + Advanced 108)

2. 자동 필터링
   └── Passage Dependency + Quality Filter

3. Human-in-the-Loop
   └── 180개 → 120개 선정 (골든)

4. 계층적 샘플링
   ├── 골든 (120개) → 실버 (60개) → 브론즈 (30개)
   └── 유형별 균등 샘플링
```

---

## 8. 참고문헌

1. **강의자료**: "RAG 및 LLM 평가 데이터셋 제작부터 평가 및 최적화의 모든 것" Part 2-6
2. **RAGAS**: Es et al. (2023). RAGAS: Automated Evaluation of RAG
3. **Han et al.** (2025). RAG vs. GraphRAG: A Systematic Evaluation
4. **thesis/README.md**: 연구 가설 및 평가 지표 정의

---

**작성일**: 2025-12-02
**버전**: 1.0
