"""
질문 유형별 프롬프트 정의

각 프롬프트는 DESIGN.md의 설계에 기반
"""

# Simple Factoid 프롬프트
SIMPLE_FACTOID_PROMPT = """## 작업
다음 행정문서에서 **단순 사실 질문**을 생성하세요.

## 문서
{context}

## 질문 유형: Simple Factoid
- 날짜, 이름, 숫자, 장소 등 구체적 사실을 묻는 질문
- 단일 문장으로 답변 가능
- "~은 무엇인가?", "~은 언제인가?", "~은 얼마인가?" 형태

## 출력 형식 (JSON)
{{
  "question": "질문 내용",
  "answer": "답변 (문서에서 추출)",
  "evidence": "답변의 근거가 되는 문서 내 문장",
  "question_type": "simple_factoid"
}}

## 예시
문서: "2024년 강남구 청소년 예산은 52억원으로 전년 대비 15% 증가했다."
출력: {{
  "question": "2024년 강남구 청소년 예산은 얼마인가?",
  "answer": "52억원",
  "evidence": "2024년 강남구 청소년 예산은 52억원으로",
  "question_type": "simple_factoid"
}}"""


# Constraint 프롬프트
CONSTRAINT_PROMPT = """## 작업
다음 행정문서에서 **조건부 질문**을 생성하세요.

## 문서
{context}

## 질문 유형: Constraint
- 특정 조건이나 제약이 포함된 질문
- 시점, 범위, 대상 등을 한정하는 조건 필수
- 조건 없이는 답변이 모호해지는 구조

## 출력 형식 (JSON)
{{
  "question": "질문 내용 (조건 포함)",
  "answer": "답변",
  "constraint": "적용된 조건",
  "evidence": "답변의 근거",
  "question_type": "constraint"
}}

## 예시
문서: "2023년 1분기 교통민원 1,234건, 2분기 1,456건 접수"
출력: {{
  "question": "2023년 1분기에 접수된 교통민원 건수는?",
  "answer": "1,234건",
  "constraint": "2023년 1분기",
  "evidence": "2023년 1분기 교통민원 1,234건",
  "question_type": "constraint"
}}"""


# Multi-doc (1-hop) 프롬프트 - 여러 문서가 필요하지만 추론은 1단계
MULTI_DOC_1_PROMPT = """## 작업
다음 여러 문서의 정보를 **종합/비교**하는 질문을 생성하세요.

## 문서들
{contexts}

## 질문 유형: Multi-doc (1-hop)
- 여러 문서의 정보가 필요하지만 추론 단계는 1개
- 비교, 종합, 요약, 공통점/차이점 등
- 순차적 추론이 아닌 병렬적 정보 통합

## 적합한 질문 예시
- "A와 B의 공통점/차이점은?"
- "여러 정책 중 가장 ~한 것은?"
- "A, B, C를 종합하면?"
- "전체적으로 어떤 추세인가?"

## 출력 형식 (JSON)
{{
  "question": "질문 내용",
  "answer": "답변",
  "doc_usage": "각 문서가 어떻게 사용되었는지 설명",
  "evidence_docs": ["문서1 근거", "문서2 근거", ...],
  "question_type": "multi_doc_1"
}}

## 중요
- 모든 문서가 답변에 기여해야 합니다
- 순차적 추론(hop)이 아닌 병렬적 통합입니다
- 한 문서만으로는 완전한 답변이 불가능해야 합니다"""


# Multi-hop 2-hop 프롬프트
MULTI_HOP_2_PROMPT = """## 작업
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
{{
  "question": "질문 내용",
  "answer": "답변",
  "reasoning_steps": [
    "Step 1: 문서 1에서 [정보 A] 확인",
    "Step 2: 문서 2에서 [정보 A]를 통해 [답변] 도출"
  ],
  "evidence_doc1": "문서 1의 근거 문장",
  "evidence_doc2": "문서 2의 근거 문장",
  "question_type": "multi_hop_2"
}}

## 중요
- 두 문서가 모두 필요한 질문을 만드세요
- 한 문서만으로 답변 가능하면 안 됩니다"""


# Multi-hop 3-hop 프롬프트
MULTI_HOP_3_PROMPT = """## 작업
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
{{
  "question": "질문 내용",
  "answer": "답변",
  "reasoning_steps": [
    "Step 1: 문서 1에서 [정보 A] 확인",
    "Step 2: 문서 2에서 [정보 A]와 [정보 B] 연결",
    "Step 3: 문서 3에서 [정보 B]를 통해 [답변] 도출"
  ],
  "evidence_doc1": "문서 1의 근거",
  "evidence_doc2": "문서 2의 근거",
  "evidence_doc3": "문서 3의 근거",
  "question_type": "multi_hop_3"
}}

## 중요
- 어느 한 문서라도 빠지면 답변이 불가능해야 합니다
- 각 hop에서 새로운 정보가 추가되어야 합니다"""


# Multi-hop 5-hop 프롬프트 (선형/비선형 구조 모두 지원)
MULTI_HOP_5_PROMPT = """## 작업
다음 다섯 문서의 정보를 연결하는 **5-hop 질문**을 생성하세요.

## 문서 1
{context_1}

## 문서 2
{context_2}

## 문서 3
{context_3}

## 문서 4
{context_4}

## 문서 5
{context_5}

## 질문 유형: Multi-hop (5-hop)
다섯 문서의 정보를 연결해야 답변 가능합니다.

### 허용되는 구조:

**선형 구조**: 순차적 추론
```
문서1 → 문서2 → 문서3 → 문서4 → 문서5 → 답변
```

**비선형 구조**: 브랜치 + 병합
```
문서1 ─┬→ 문서2 → 문서4 ─┬→ 문서5 → 답변
       └→ 문서3 ─────────┘
```

## 출력 형식 (JSON)
{{
  "question": "질문 내용",
  "answer": "답변",
  "structure_type": "linear | branching",
  "reasoning_steps": [
    "Step 1: 문서 X에서 [정보] 확인",
    "Step 2: ...",
    "Step 3: ...",
    "Step 4: ...",
    "Step 5: [최종 답변] 도출"
  ],
  "evidence_doc1": "문서 1의 근거 문장",
  "evidence_doc2": "문서 2의 근거 문장",
  "evidence_doc3": "문서 3의 근거 문장",
  "evidence_doc4": "문서 4의 근거 문장",
  "evidence_doc5": "문서 5의 근거 문장",
  "question_type": "multi_hop_5"
}}

## 중요
- 5개 문서 모두 답변에 필수적이어야 합니다
- 선형이든 비선형이든 모든 문서가 연결되어야 합니다
- 어느 한 문서라도 빠지면 답변 불가능해야 합니다
- 비선형 구조 시 병합 지점이 명확해야 합니다"""


# Reasoning 프롬프트
REASONING_PROMPT = """## 작업
다음 행정문서에서 **추론/인과 질문**을 생성하세요.

## 문서
{context}

## 질문 유형: Reasoning
- 원인-결과 관계를 묻는 질문
- "왜?", "어떻게?", "무엇 때문에?" 형태
- 단순 사실 나열이 아닌 관계 파악 필요

## 출력 형식 (JSON)
{{
  "question": "질문 내용",
  "answer": "답변",
  "reasoning_type": "causal_antecedent | causal_consequence | comparison",
  "evidence": "답변의 근거",
  "question_type": "reasoning"
}}

## reasoning_type 설명
- causal_antecedent: 원인을 묻는 질문 ("왜 ~했는가?")
- causal_consequence: 결과를 묻는 질문 ("~의 결과는?")
- comparison: 비교 질문 ("~와 ~의 차이는?")

## 예시
문서: "2024년 교통민원 30% 증가는 지하철 공사로 인한 혼잡 때문이다."
출력: {{
  "question": "2024년 교통민원이 증가한 원인은 무엇인가?",
  "answer": "지하철 공사로 인한 교통 혼잡",
  "reasoning_type": "causal_antecedent",
  "evidence": "지하철 공사로 인한 혼잡 때문이다",
  "question_type": "reasoning"
}}"""


# 프롬프트 매핑
PROMPTS = {
    # Simple (단일 문서) - 40%
    "simple_factoid": SIMPLE_FACTOID_PROMPT,  # 20%
    "constraint": CONSTRAINT_PROMPT,           # 10%
    "reasoning": REASONING_PROMPT,             # 10%

    # Advanced (다중 문서) - 60%
    "multi_doc_1": MULTI_DOC_1_PROMPT,         # 20% - 여러 문서, 1-hop (비교/종합)
    "multi_hop_2": MULTI_HOP_2_PROMPT,         # 20% - 2개 문서 순차 연결
    "multi_hop_3": MULTI_HOP_3_PROMPT,         # 10% - 3개 문서 순차 연결
    "multi_hop_5": MULTI_HOP_5_PROMPT,         # 10% - 5개 문서 (선형/비선형)
}
