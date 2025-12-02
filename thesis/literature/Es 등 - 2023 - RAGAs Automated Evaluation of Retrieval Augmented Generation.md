# RAGAS: Automated Evaluation of Retrieval Augmented Generation

## 1. 논문 정보

- **제목**: RAGAS: Automated Evaluation of Retrieval Augmented Generation
- **저자**: Shahul Es, Jithin James, Luis Espinosa-Anke, Steven Schockaert
- **소속**: Exploding Gradients, CardiffNLP (Cardiff University), AMPLYFI
- **발표**: arXiv:2309.15217v2 [cs.CL] (2023년 9월 제출, 2025년 4월 개정)
- **GitHub**: https://github.com/explodinggradients/ragas

## 2. 핵심 내용 요약

본 논문은 RAG(Retrieval Augmented Generation) 시스템을 **정답(ground truth) 없이 자동으로 평가**할 수 있는 프레임워크인 RAGAS를 제안한다. RAGAS는 세 가지 핵심 품질 차원(Faithfulness, Answer Relevance, Context Relevance)을 LLM 프롬프팅을 통해 측정하며, LlamaIndex 및 Langchain과 통합되어 실무에서 쉽게 사용할 수 있다. 인간 평가와의 일치도를 검증하기 위해 WikiEval 데이터셋을 구축하였으며, Faithfulness 평가에서 95%의 높은 정확도를 달성했다.

## 3. 주요 기여점

### 3.1 Reference-free 평가 프레임워크
- **인간 주석 불필요**: 정답 데이터 없이 RAG 시스템의 품질을 자동으로 평가
- **다차원 평가**: 검색 품질, 생성 품질, 충실성을 독립적으로 측정
- **실무 적용성**: LlamaIndex/Langchain과 즉시 통합 가능

### 3.2 세 가지 핵심 메트릭
1. **Faithfulness (충실성)**: 생성된 답변이 검색된 컨텍스트에 근거하는가?
2. **Answer Relevance (답변 관련성)**: 답변이 질문을 직접적으로 다루는가?
3. **Context Relevance (컨텍스트 관련성)**: 검색된 컨텍스트가 질문에 집중되어 있는가?

### 3.3 WikiEval 데이터셋
- **50개 Wikipedia 페이지**: 2022년 이후 사건 (LLM 학습 cutoff 이후)
- **인간 평가 데이터**: 세 가지 품질 차원에 대한 pairwise comparison
- **높은 평가자 간 일치도**: Faithfulness/Context Relevance 95%, Answer Relevance 90%

## 4. 방법론

### 4.1 Faithfulness 평가

**핵심 아이디어**: 답변의 주장(claim)이 컨텍스트에서 추론 가능한가?

**2단계 프로세스**:
```
1. Statement Extraction (문장 분해)
   - 답변을 atomic statement로 분해
   - 프롬프트: "Given a question and answer, create one or more statements
                from each sentence in the given answer."

2. Verification (검증)
   - 각 statement가 context에서 지원되는지 확인
   - 프롬프트: "Consider the given context and following statements, then
                determine whether they are supported by the information
                present in the context."

3. Scoring (점수 계산)
   - F = |V| / |S|
   - |V|: 지원된 statement 수
   - |S|: 전체 statement 수
```

### 4.2 Answer Relevance 평가

**핵심 아이디어**: 답변이 질문과 얼마나 직접적으로 연관되는가?

**Reverse Question Generation 방식**:
```
1. Question Generation
   - 답변으로부터 n개의 질문 생성
   - 프롬프트: "Generate a question for the given answer."

2. Embedding Similarity
   - text-embedding-ada-002 모델 사용
   - 생성된 질문 qi와 원본 질문 q 간의 cosine similarity 계산

3. Scoring
   - AR = (1/n) * Σ sim(q, qi)
   - 높은 점수 = 답변이 질문에 집중됨
   - 낮은 점수 = 불완전하거나 중복 정보 포함
```

### 4.3 Context Relevance 평가

**핵심 아이디어**: 컨텍스트에 불필요한 정보가 얼마나 포함되었는가?

**Sentence Extraction 방식**:
```
1. Relevant Sentence Extraction
   - 질문에 답하기 위해 필수적인 문장만 추출
   - 프롬프트: "Please extract relevant sentences from the provided context
                that can potentially help answer the following question."

2. Scoring
   - CR = (추출된 문장 수) / (전체 문장 수)
   - 높은 점수 = 컨텍스트가 질문에 집중됨
   - 낮은 점수 = 불필요한 정보 많음
```

### 4.4 구현 세부사항
- **LLM**: gpt-3.5-turbo-16k (OpenAI API)
- **Embedding**: text-embedding-ada-002
- **통합**: LlamaIndex, Langchain 지원
- **Few-shot prompting**: 모든 프롬프트에 demonstration 포함

## 5. 실험 결과

### 5.1 WikiEval 데이터셋 평가 (Pairwise Comparison)

| 메트릭 | RAGAS | GPT Score | GPT Ranking |
|--------|-------|-----------|-------------|
| **Faithfulness** | **0.95** | 0.72 | 0.54 |
| **Answer Relevance** | **0.78** | 0.52 | 0.40 |
| **Context Relevance** | **0.70** | 0.63 | 0.52 |

**주요 성능 지표**:
- **Faithfulness**: 95% 인간 평가 일치도 (매우 높음)
- **Answer Relevance**: 78% 일치도 (미묘한 차이 구분의 어려움)
- **Context Relevance**: 70% 일치도 (가장 어려운 차원)

### 5.2 Baseline 비교

**GPT Score (점수 직접 부여)**:
```
프롬프트: "Faithfulness measures the information consistency of the answer
           against the given context. Given an answer and context, assign
           a score for faithfulness in the range 0-10."
결과: 모든 차원에서 RAGAS보다 낮은 성능
```

**GPT Ranking (직접 순위 비교)**:
```
프롬프트: "Answer Relevancy measures the degree to which a response
           directly addresses... Given a question and answer, rank each
           answer based on Answer Relevancy."
결과: 가장 낮은 성능 (순서 효과에 민감)
```

### 5.3 WikiEval 예시 분석

**High Faithfulness 답변 예시**:
```
Question: Who directed the film Oppenheimer and who stars as J. Robert
          Oppenheimer in the film?

Context: Oppenheimer is a 2023 biographical thriller film written and
         directed by Christopher Nolan... Cillian Murphy stars as
         Oppenheimer...

High Faithfulness: "Christopher Nolan directed the film Oppenheimer.
                    Cillian Murphy stars as J. Robert Oppenheimer in the film."
                    → 컨텍스트에서 완전히 지원됨

Low Faithfulness: "James Cameron directed the film Oppenheimer. Tom Cruise
                   stars as J. Robert Oppenheimer in the film."
                   → 컨텍스트와 모순, 환각(hallucination)
```

## 6. 우리 연구와의 관련성

### 6.1 직접 적용 가능 영역

**평가 프레임워크**:
- 우리의 on-premise 한국어 RAG 시스템 평가에 RAGAS 메트릭 직접 활용
- Faithfulness 메트릭으로 행정문서 기반 답변의 신뢰성 검증
- Context Relevance로 검색 최적화 (긴 컨텍스트 비용 절감)

**한국어 적용 시 고려사항**:
- GPT-3.5-turbo-16k의 한국어 성능 vs 한국어 특화 LLM (EXAONE, Gemini)
- Statement extraction의 한국어 문장 구조 대응
- Embedding 모델의 한국어 적합성 (ada-002 vs multilingual models)

### 6.2 방법론적 참고사항

**Reference-free 평가의 중요성**:
- 행정문서 도메인은 정답 데이터 구축 비용이 매우 높음
- RAGAS의 자동 평가 방식이 우리 실험 환경에 적합

**다차원 평가의 필요성**:
```
우리 시스템: Naive RAG → Advanced RAG → Graph RAG
평가 축:
  - Faithfulness: 행정문서 기반 답변의 정확성
  - Answer Relevance: 사용자 질문에 대한 직접성
  - Context Relevance: 검색 효율성 (on-premise 리소스 제약)
```

**WikiEval 데이터셋 구축 방법론**:
- LLM을 활용한 질문 생성 및 답변 생성
- Pairwise comparison으로 평가자 부담 감소
- 우리 Golden Testset 구축 시 참고 가능

### 6.3 온프레미스 환경 적용 전략

**RAGAS의 OpenAI API 의존성 문제**:
```
원본: gpt-3.5-turbo-16k (평가용 LLM)
대안:
  1. Gemini Pro (Google Vertex AI) - API 사용 가능
  2. EXAONE-3.5 (LG AI Research) - 온프레미스 배포 가능
  3. LLaMA-3.1-70B - 오픈소스, 자체 호스팅

비교 실험 필요:
  - OpenAI API vs Gemini API vs On-premise LLM
  - 평가 일관성 vs 비용 vs 프라이버시
```

**Embedding 모델 선택**:
```
원본: text-embedding-ada-002 (OpenAI)
대안:
  1. multilingual-e5-large (오픈소스, 한국어 성능 우수)
  2. bge-m3 (BAAI, 다국어 지원)
  3. KoSimCSE (한국어 특화)
```

## 7. 인용 가능한 핵심 문장

### 7.1 RAG 시스템의 평가 필요성

> "Evaluating RAG architectures is, however, challenging because there are several dimensions to consider: the ability of the retrieval system to identify relevant and focused context passages, the ability of the LLM to exploit such passages in a faithful way, or the quality of the generation itself."

**한글 번역**: "RAG 아키텍처 평가는 여러 차원을 고려해야 하기 때문에 어려운 과제다: 검색 시스템이 관련성 있고 집중된 컨텍스트 구절을 식별하는 능력, LLM이 그러한 구절을 충실하게 활용하는 능력, 그리고 생성 자체의 품질 등을 고려해야 한다."

**인용 맥락**: RAG 시스템의 다차원적 평가 필요성 강조

---

> "We posit that such a framework can crucially contribute to faster evaluation cycles of RAG architectures, which is especially important given the fast adoption of LLMs."

**한글 번역**: "우리는 이러한 프레임워크가 RAG 아키텍처의 빠른 평가 주기에 결정적으로 기여할 수 있다고 주장하며, 이는 LLM의 빠른 도입을 고려할 때 특히 중요하다."

**인용 맥락**: 자동화된 평가 프레임워크의 실무적 중요성

---

### 7.2 Faithfulness의 중요성

> "Faithfulness refers to the idea that the answer should be grounded in the given context. This is important to avoid hallucinations, and to ensure that the retrieved context can act as a justification for the generated answer."

**한글 번역**: "충실성(Faithfulness)은 답변이 주어진 컨텍스트에 근거해야 한다는 개념을 의미한다. 이는 환각(hallucination)을 방지하고 검색된 컨텍스트가 생성된 답변에 대한 정당화 역할을 할 수 있도록 보장하는 데 중요하다."

**인용 맥락**: 행정문서 기반 RAG에서 충실성 평가의 필수성

---

> "RAG systems are often used in applications where the factual consistency of the generated text w.r.t. the grounded sources is highly important, e.g. in domains such as law, where information is constantly evolving."

**한글 번역**: "RAG 시스템은 생성된 텍스트가 근거 출처와 사실적으로 일치해야 하는 응용 분야에서 자주 사용되며, 예를 들어 정보가 지속적으로 변화하는 법률과 같은 도메인에서 사용된다."

**인용 맥락**: 법률/행정 도메인에서 RAG 신뢰성의 중요성

---

### 7.3 Context Relevance의 실무적 의미

> "Context Relevance refers to the idea that the retrieved context should be focused, containing as little irrelevant information as possible. This is important given the cost associated with feeding long context passages to LLMs."

**한글 번역**: "컨텍스트 관련성(Context Relevance)은 검색된 컨텍스트가 관련 없는 정보를 최소한으로 포함하며 집중되어야 한다는 개념을 의미한다. 이는 LLM에 긴 컨텍스트 구절을 제공하는 데 따른 비용을 고려할 때 중요하다."

**인용 맥락**: 온프레미스 환경에서 컨텍스트 최적화의 필요성

---

> "Moreover, when context passages are too long, LLMs are often less effective in exploiting that context, especially for information that is provided in the middle of the context passage (Liu et al., 2023)."

**한글 번역**: "더욱이, 컨텍스트 구절이 너무 길면 LLM은 종종 해당 컨텍스트를 활용하는 데 효과적이지 못하며, 특히 컨텍스트 구절의 중간에 제공된 정보에 대해 그러하다."

**인용 맥락**: "Lost in the middle" 현상과 검색 최적화의 중요성

---

### 7.4 Reference-free 평가의 장점

> "We focus on settings where reference answers may not be available, and where we want to estimate different proxies for correctness, in addition to the usefulness of the retrieved passages."

**한글 번역**: "우리는 정답이 제공되지 않을 수 있는 설정에 초점을 맞추며, 검색된 구절의 유용성 외에도 정확성에 대한 다양한 대리 지표를 추정하고자 한다."

**인용 맥락**: Ground truth 없이 RAG 시스템을 평가하는 방법론

---

### 7.5 LLM 기반 평가의 효과

> "The results in Table 1 show that our proposed metrics are much closer aligned with the human judgements than the predictions from the two baselines."

**한글 번역**: "표 1의 결과는 우리가 제안한 메트릭이 두 가지 기준선의 예측보다 인간 판단과 훨씬 더 가깝게 일치함을 보여준다."

**인용 맥락**: RAGAS 메트릭의 인간 평가와의 높은 일치도 (Faithfulness 95%)

---

### 7.6 평가의 한계점 인정

> "We found context relevance to be the hardest quality dimension to evaluate. In particular, we observed that ChatGPT often struggles with the task of selecting the sentences from the context that are crucial, especially for longer contexts."

**한글 번역**: "우리는 컨텍스트 관련성이 평가하기 가장 어려운 품질 차원임을 발견했다. 특히, ChatGPT가 특히 긴 컨텍스트의 경우 컨텍스트에서 중요한 문장을 선택하는 작업에서 어려움을 겪는 것을 관찰했다."

**인용 맥락**: RAGAS 한계점 및 향후 개선 방향

---

### 7.7 WikiEval 데이터셋 구축

> "To construct the dataset, we first selected 50 Wikipedia pages covering events that have happened since the start of 2022. In selecting these pages, we prioritised those with recent edits."

**한글 번역**: "데이터셋을 구축하기 위해 우리는 먼저 2022년 초 이후 발생한 사건을 다루는 50개의 Wikipedia 페이지를 선택했다. 이러한 페이지를 선택할 때 최근 편집된 페이지를 우선시했다."

**인용 맥락**: LLM 학습 cutoff 이후 데이터로 평가 데이터셋 구축 방법론

---

## 8. 한계점 및 향후 연구방향

### 8.1 논문에서 언급한 한계점

**Context Relevance 평가의 어려움**:
- ChatGPT가 긴 컨텍스트에서 중요한 문장 선택에 어려움
- 70% 인간 일치도 (다른 메트릭 대비 낮음)
- 더 정교한 sentence extraction 방법 필요

**Answer Relevance의 미묘한 차이 구분**:
- 78% 일치도 (Faithfulness 95% 대비 낮음)
- 불완전한 답변 vs 중복 정보 포함 답변 구분의 어려움
- Reverse question generation 방식의 한계

**WikiEval 데이터셋 규모**:
- 50개 샘플만으로 평가 (통계적 유의성 제한)
- 특정 도메인(Wikipedia 이벤트)에 편향
- 다양한 도메인/태스크로 확장 필요

### 8.2 OpenAI API 의존성

**상업적 LLM 의존**:
```
현재: gpt-3.5-turbo-16k (평가용), text-embedding-ada-002 (임베딩)
문제:
  - API 비용 (대규모 평가 시 부담)
  - 프라이버시 이슈 (민감한 문서 전송)
  - 재현성 문제 (모델 업데이트 시 결과 변동)
```

**향후 연구 방향**:
- 오픈소스 LLM (LLaMA, Mistral)으로 메트릭 재현
- 한국어 특화 모델 (EXAONE, Gemini)과의 비교
- 온프레미스 배포 가능한 경량 평가 모델

### 8.3 한국어 적용 시 추가 과제

**언어적 차이**:
- Statement extraction: 한국어 문장의 복잡한 서술어 구조
- Question generation: 한국어 조사/어미 시스템
- Embedding similarity: 한국어 의미 표현의 특수성

**도메인 차이**:
- Wikipedia (일반 지식) vs 행정문서 (전문 도메인)
- 답변 스타일: 사실 나열 vs 법규 해석
- 컨텍스트 구조: 자유 형식 텍스트 vs 구조화된 문서

### 8.4 우리 연구의 확장 방향

**한국어 RAGAS 벤치마크 구축**:
```
제안: KoreanRAGEval 데이터셋
  - AI Hub 행정문서 기반 50-100개 QA 샘플
  - 인간 평가자 2-3명 (도메인 전문가 포함)
  - 세 가지 메트릭에 대한 pairwise annotation
  - RAGAS vs 한국어 LLM 평가 비교
```

**On-premise 평가 파이프라인**:
```
목표: 완전한 오픈소스 평가 스택
  1. Evaluator LLM: EXAONE-3.5-7.8B / LLaMA-3.1-70B
  2. Embedding: multilingual-e5-large / bge-m3
  3. 비용: $0 (온프레미스), 프라이버시: 보장
  4. 재현성: 모델 버전 고정 가능
```

**메트릭 확장**:
```
RAGAS 기본:
  - Faithfulness, Answer Relevance, Context Relevance

우리 연구 추가:
  - Retrieval Precision/Recall (AutoRAG 기본 메트릭)
  - Context Efficiency (토큰 수 대비 정보량)
  - Administrative Document Compliance (행정 용어/형식 준수)
```

### 8.5 RAGAS 프레임워크의 발전 방향

**공식 저장소 동향** (2025년 기준):
- LangChain/LlamaIndex 통합 강화
- 다국어 지원 확대
- 더 많은 메트릭 추가 (Context Precision, Context Recall)
- 오픈소스 LLM 지원 개선

**학계 후속 연구**:
- 더 큰 규모의 평가 데이터셋 (WikiEval → 수천 샘플)
- 도메인별 특화 메트릭 (의료, 법률, 금융)
- 다국어 RAG 평가 표준화

---

## 9. 실험 재현 가능성

### 9.1 공개된 리소스

**코드**:
- GitHub: https://github.com/explodinggradients/ragas
- PyPI 패키지 설치 가능: `pip install ragas`
- LangChain/LlamaIndex 통합 예제 제공

**데이터셋**:
- WikiEval: https://huggingface.co/datasets/explodinggradients/WikiEval
- 50개 question-context-answer 샘플
- 인간 평가 레이블 포함

### 9.2 우리 프로젝트 적용 계획

**Phase 1: RAGAS 기본 적용**
```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevance, context_relevance

# 우리의 RAG 시스템 평가
results = evaluate(
    dataset=korean_admin_qa_dataset,
    metrics=[faithfulness, answer_relevance, context_relevance],
    llm="gpt-3.5-turbo-16k",  # 또는 Gemini Pro
    embeddings="text-embedding-ada-002"  # 또는 multilingual-e5-large
)
```

**Phase 2: 한국어 최적화**
```python
# 한국어 프롬프트 커스터마이징
from ragas.llms import LangchainLLM
from langchain.chat_models import ChatVertexAI

korean_llm = LangchainLLM(ChatVertexAI(model_name="gemini-pro"))

# 한국어 프롬프트 템플릿 재작성
custom_faithfulness_prompt = """
주어진 질문과 답변에서 각 문장으로부터 하나 이상의 진술을 생성하세요.
질문: [question]
답변: [answer]
"""
```

**Phase 3: On-premise 전환**
```python
# EXAONE-3.5 온프레미스 배포
from ragas.llms import LangchainLLM
from langchain.llms import HuggingFacePipeline

onpremise_llm = HuggingFacePipeline.from_model_id(
    model_id="LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
    task="text-generation",
    device=0  # GPU 0
)

# 비용: $0, 프라이버시: 완전 보장
```

---

## 10. 결론

RAGAS는 **reference-free RAG 평가의 표준 프레임워크**로 자리 잡았으며, 특히 Faithfulness 메트릭은 95%의 인간 일치도를 달성하여 실무에서 신뢰할 수 있는 자동 평가 도구임을 입증했다. 우리의 온프레미스 한국어 행정문서 RAG 시스템 연구에서 RAGAS의 세 가지 메트릭(Faithfulness, Answer Relevance, Context Relevance)을 핵심 평가 축으로 활용할 수 있으며, 특히 **정답 데이터 구축 비용이 높은 행정 도메인**에서 자동 평가의 가치가 크다.

단, OpenAI API 의존성을 고려하여 **한국어 특화 LLM (EXAONE, Gemini)으로의 전환** 및 **온프레미스 배포 가능성** 실험이 필요하며, WikiEval 방법론을 참고하여 **Korean AdminDocRAGEval 데이터셋**을 구축함으로써 우리 시스템의 평가 신뢰성을 높일 수 있다.

---

## 참고문헌 형식

**APA**:
```
Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). RAGAS:
Automated Evaluation of Retrieval Augmented Generation.
arXiv preprint arXiv:2309.15217.
```

**BibTeX**:
```bibtex
@article{es2023ragas,
  title={RAGAS: Automated Evaluation of Retrieval Augmented Generation},
  author={Es, Shahul and James, Jithin and Espinosa-Anke, Luis and Schockaert, Steven},
  journal={arXiv preprint arXiv:2309.15217},
  year={2023}
}
```

---

**작성일**: 2025-11-30
**작성자**: Claude Code (AI Assistant)
**목적**: 석사학위논문 "On-premise Open-source RAG system for Korean public administrative documents" Literature Review
