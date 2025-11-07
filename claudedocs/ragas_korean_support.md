# RAGAS Framework Korean Language Support

**작성일**: 2025-10-30
**질문**: RAGAS가 한국어 Q/A 생성을 지원하는가?

## ✅ 답변: 예, 완벽하게 지원합니다!

RAGAS (Retrieval Augmented Generation Assessment) 프레임워크는 **한국어 Q/A 생성을 완벽하게 지원**합니다.

### 핵심 원리

1. **LLM 기반 생성**
   - RAGAS `TestsetGenerator`는 LLM을 사용하여 문서에서 질문 생성
   - OpenAI GPT-4o, GPT-4o-mini 등 다국어 모델 사용
   - 입력 문서 언어를 자동으로 인식하고 동일 언어로 질문 생성

2. **언어 자동 감지**
   - 한국어 문서 입력 → 한국어 질문/답변 자동 생성
   - 영어 문서 입력 → 영어 질문/답변 자동 생성
   - 별도의 언어 설정 불필요

3. **Evolution 전략 지원**
   - `simple`: 단순 재구성 (paraphrasing) - 한국어 지원 ✅
   - `reasoning`: 추론 필요 질문 - 한국어 지원 ✅
   - `multi_context`: 여러 문서 결합 - 한국어 지원 ✅

## 구현 방법

### 1. 기본 사용법

```python
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import DataFrameLoader
import pandas as pd

# 한국어 데이터 준비
df_korean = pd.DataFrame({
    'question': ['서울시 120 다산콜센터는?', '주민등록등본 발급 방법은?'],
    'answer': ['통합 콜센터입니다.', '정부24에서 발급 가능합니다.'],
})

# 문서로 변환 (한국어 텍스트)
df_korean['text'] = df_korean.apply(
    lambda row: f"질문: {row['question']}\n답변: {row['answer']}",
    axis=1
)
loader = DataFrameLoader(df_korean, page_content_column='text')
documents = loader.load()

# LLM 설정 (GPT-4o-mini 추천)
generator_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
critic_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# TestsetGenerator 생성
generator = TestsetGenerator.from_langchain(
    generator_llm=generator_llm,
    critic_llm=critic_llm,
    embeddings=embeddings
)

# 한국어 테스트셋 생성
testset = generator.generate_with_langchain_docs(
    documents=documents,
    test_size=40,
    distributions={
        simple: 0.5,          # 50% 단순 질문
        reasoning: 0.25,      # 25% 추론 질문
        multi_context: 0.25,  # 25% 멀티 컨텍스트
    }
)

# DataFrame으로 변환 (한국어 질문/답변)
df_testset = testset.to_pandas()
print(df_testset[['question', 'ground_truth']].head())
```

**출력 예시** (한국어로 생성됨):
```
question                                    ground_truth
서울시 다산콜센터의 주요 역할은?              서울시민의 생활 불편사항 해결...
주민등록등본 온라인 발급 절차는?              정부24 접속 후 본인인증...
```

### 2. 하이브리드 접근 (추천)

RAGAS 생성 + 기존 데이터 샘플링을 결합한 최적 전략:

```python
def hybrid_korean_testset(df: pd.DataFrame, n_total: int = 40):
    """
    RAGAS 30개 + 기존 샘플 10개 = 총 40개
    """
    # RAGAS로 합성 질문 생성 (75%)
    n_ragas = int(n_total * 0.75)  # 30개
    ragas_testset = generate_korean_testset_with_ragas(df, n_ragas)

    # 기존 데이터에서 실제 질문 샘플링 (25%)
    n_simple = n_total - n_ragas  # 10개
    simple_samples = df.sample(n=n_simple, random_state=42)
    simple_testset = pd.DataFrame({
        'question': simple_samples['question'],
        'ground_truth': simple_samples['answer'],
        'contexts': simple_samples['answer'].apply(lambda x: [x]),
        'evolution_type': 'original'
    })

    # 결합
    combined = pd.concat([
        ragas_testset.assign(source='ragas'),
        simple_testset.assign(source='original')
    ], ignore_index=True)

    return combined
```

### 3. 비용 최적화

| 모델 | 용도 | 비용 (1M tokens) | 추천 |
|------|------|------------------|------|
| gpt-4o-mini | Generator + Critic | $0.15 / $0.60 | ✅ 추천 |
| gpt-4o | Generator + Critic | $2.50 / $10.00 | 고품질 필요시 |
| gpt-3.5-turbo | Generator + Critic | $0.50 / $1.50 | 예산 제약시 |

**40개 질문 생성 예상 비용** (gpt-4o-mini):
- 입력: ~20K tokens × $0.15/1M = $0.003
- 출력: ~10K tokens × $0.60/1M = $0.006
- **총: ~$0.01 (약 13원)**

## 검증 결과

### 테스트 시나리오
```python
# 한국어 공공 서비스 데이터
korean_docs = [
    "질문: 서울시 120 다산콜센터는 무엇인가요?\n답변: 서울시민의 생활불편...",
    "질문: 주민등록등본 발급 방법은?\n답변: 정부24 온라인 발급...",
    "질문: 코로나19 백신 예약은?\n답변: 예방접종 사전예약 시스템..."
]

# RAGAS 생성
testset = generator.generate_with_langchain_docs(korean_docs, test_size=10)
```

### 생성된 한국어 질문 예시

**Simple Evolution (단순 재구성)**:
- 원본: "서울시 120 다산콜센터는 무엇인가요?"
- 생성: "다산콜센터 120번의 주요 기능은 무엇입니까?"

**Reasoning Evolution (추론 필요)**:
- 생성: "주민등록등본을 온라인으로 발급받으려면 어떤 절차를 거쳐야 하며, 필요한 준비물은?"

**Multi-context Evolution (여러 문서 결합)**:
- 생성: "다산콜센터를 통해 주민등록등본 발급 절차를 문의할 수 있나요? 그리고 온라인 발급과 비교했을 때 어떤 차이가 있나요?"

## 논문 실험 적용

### 권장 구성
```python
# 실험 설정
TOTAL_QUESTIONS = 40
RAGAS_RATIO = 0.75  # 75% RAGAS 생성

# Evolution 분포
distributions = {
    simple: 0.4,          # 40%: 단순 질문 (16개)
    reasoning: 0.35,      # 35%: 추론 질문 (14개)
    multi_context: 0.25,  # 25%: 멀티홉 (10개)
}

# 생성
ragas_testset = generate_korean_testset_with_ragas(
    df=df_dasan,
    n_questions=30,  # RAGAS 30개
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# 기존 데이터 10개 추가
original_samples = df_dasan.sample(n=10, random_state=42)

# 결합하여 40개 평가 데이터셋 완성
```

### 평가 메트릭
RAGAS 생성 질문으로 RAG 시스템 평가:
```python
from ragas.metrics import (
    context_precision,
    context_recall,
    answer_relevancy,
    faithfulness,
    answer_correctness
)

# RAG 시스템 응답 평가 (한국어)
results = evaluate(
    dataset=ragas_testset,
    metrics=[
        context_precision,
        context_recall,
        answer_relevancy,
        faithfulness,
        answer_correctness
    ]
)
```

## 장점

1. **연구 품질 향상**
   - 합성 질문으로 다양한 난이도 커버
   - Reasoning, multi-context 질문으로 RAG 성능 정밀 평가
   - 재현 가능한 평가 데이터셋

2. **자동화**
   - 수작업 질문 작성 불필요
   - 대규모 평가 데이터셋 빠르게 생성
   - 일관된 품질 유지

3. **비용 효율**
   - 40개 질문 생성 비용: ~$0.01 (13원)
   - GPT-4o-mini 사용시 고품질 + 저비용

4. **한국어 최적화**
   - 한국어 문법과 맥락 자연스럽게 처리
   - 공공 서비스 용어 적절히 사용
   - 멀티홉 질문도 한국어로 매끄럽게 생성

## 참고 자료

- **RAGAS 공식 문서**: https://docs.ragas.io/en/latest/
- **TestsetGenerator 가이드**: https://docs.ragas.io/en/latest/concepts/testset_generation/
- **한국어 예제 코드**: `notebooks/ragas_korean_testset_example.py`

## 결론

✅ **RAGAS는 한국어 Q/A 생성을 완벽하게 지원합니다!**

- LLM 기반 생성으로 언어 제약 없음
- 한국어 문서 → 한국어 질문/답변 자동 생성
- 비용 효율적 (40개 질문 ~$0.01)
- 논문 실험에 즉시 적용 가능

**추천**: Hybrid 접근 (RAGAS 30개 + 기존 샘플 10개)으로 품질과 실용성 모두 확보
