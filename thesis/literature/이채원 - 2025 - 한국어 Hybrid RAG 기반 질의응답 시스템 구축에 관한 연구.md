# Literature Review: 한국어 Hybrid RAG 기반 질의응답 시스템 구축에 관한 연구

## 1. 논문 정보

- **제목**: 한국어 Hybrid RAG 기반 질의응답 시스템 구축에 관한 연구 (A Study on the Construction of a Korean Hybrid RAG-based Question-Answering System)
- **저자**: 이채원 (CHAEWON LEE)
- **지도교수**: 이강배
- **학위**: 석사학위논문
- **소속**: 동아대학교 대학원 경영정보학과
- **연도**: 2024년 12월 (2025년 출판)
- **학위 유형**: 경영학석사

---

## 2. 핵심 내용 요약

본 연구는 한국어의 언어적 특성을 반영한 Graph RAG와 Vector RAG를 결합한 Hybrid RAG 기반의 기업 맞춤형 질의응답 시스템을 개발하였다. 한국어 특성상 주어와 목적어가 자주 생략되고 중의적 표현이 많아, 생략어 복원과 의존 구문 분석 기법을 LLM 프롬프트에 적용하여 지식 그래프를 구축하였다. Graph RAG는 정보 확장과 잠재적 지식 추론에 강점이 있지만 관련 노드가 없는 질의에 대한 답변이 어려워, Vector RAG를 결합한 Hybrid RAG로 이를 보완하였다. 실험 결과, Hybrid RAG는 Answer Relevance(0.72), Entity Score(0.33), Hallucination(0.26) 지표에서 가장 우수한 성능을 보였다.

---

## 3. 주요 기여점

### 3.1 한국어 특화 지식 그래프 구축 방법론
- **접속어 분리**: 한국어의 자유로운 어미 변형 특성을 고려하여 ['이지만', '불구하고', '이나', '으나', '면서', '지만', '으며'] 기준으로 문장 분리
- **생략어 복원**: Korean Anaphora Resolution 모델(KoCharElectra + LSTM + Bi-LSTM)을 활용하여 생략된 주어/목적어 복원
- **의존 구문 분석**: SpaCy의 ko-core-news-sm 모델을 활용하여 한국어 문장의 단어 간 종속 관계 파악
- **Ko-Triple Extraction**: 3단계 프로세스를 통해 3,237개의 트리플(주어-술어-목적어) 데이터셋 구축

### 3.2 Hybrid RAG 아키텍처 설계
- **Graph RAG 우선 검색**: 질의에 대해 먼저 Graph RAG로 관계 기반 검색 수행 (최대 5개 노드 선택)
- **Vector RAG 보완**: Graph RAG 응답에서 키워드 추출 후 Vector RAG로 참조 문헌 검색
- **Fallback 메커니즘**: Graph RAG 실패 시 Vector RAG가 독립적으로 실행
- **신뢰성 강화**: 참조 문헌(SOURCES) 명시를 통한 응답 검증 가능성 확보

### 3.3 성능 평가 데이터셋 구축
- **자동 Q&A 생성 파이프라인**: Document → Summary → Q&A Data → 검증
- **검증된 140개 Q&A 데이터셋**: 1,020개 생성 후 품질 검증을 통해 선별
- **다층 평가 지표**: Answer Relevance(코사인 유사도), Entity Score(단어 일치율), Hallucination(참조 일치도)

---

## 4. 방법론

### 4.1 지식 그래프 구축 과정

```
Document → 접속어 분리 → 생략어 복원 → 의존 구문 분석 → 트리플 추출 → Knowledge Graph
```

**예시**:
- 원문: "K사는 세계 1위 수준의 에폭시수지 생산능력을 갖추고 있으며, 이를 기반으로 신규법인을 설립하였다."
- 접속어 분리: "으며" 기준으로 2개 문장으로 분리
- 생략어 복원: 두 번째 문장에 "K사는" 추가
- 트리플 추출:
  - (K사, 감소할 것으로 전망된다, 2022년 영업이익률)
  - (2022년 영업이익률, 전망한다, 9.0% 기록할 것)

### 4.2 RAG 시스템 구성

#### Vector RAG
- **청크 분할**: CharacterTextSplitter로 1,500자 단위 분할
- **임베딩**: OpenAI Embeddings API 활용
- **벡터 스토어**: Chroma 벡터 데이터베이스
- **검색**: 질의와 유사도 높은 상위 2개 문서 반환
- **생성**: GPT-3.5-turbo로 답변 생성

#### Graph RAG
- **트리플 텍스트화**: NetworkX 모델로 그래프 관계 정보 포함
- **임베딩**: OpenAIEmbeddings로 벡터화
- **벡터 스토어**: FAISS (Facebook AI Similarity Search)
- **검색**: 최대 5개 관련 노드 선택
- **생성**: OpenAI LLM으로 답변 생성

#### Hybrid RAG
1. Graph RAG 우선 검색 및 응답 생성
2. 응답에서 정규표현식으로 키워드 추출
3. Vector RAG로 키워드 기반 참조 문헌 검색
4. Graph RAG 응답 + Vector RAG 참조 결합
5. Graph RAG 실패 시 Vector RAG 독립 실행

### 4.3 기술 스택
- **LLM**: GPT-3.5-turbo (OpenAI API)
- **임베딩**: OpenAI Embeddings API
- **생략어 복원**: Korean Anaphora Resolution (KoCharElectra + Bi-LSTM)
- **의존 구문 분석**: SpaCy ko-core-news-sm
- **벡터 DB**: Chroma (Vector RAG), FAISS (Graph RAG)
- **그래프**: NetworkX
- **데이터 소스**: 화학 제조업 K사의 기업분석 보고서, 시장 동향 보고서

---

## 5. 실험 결과

### 5.1 성능 지표 비교

| 지표 | Vector RAG | Graph RAG | Hybrid RAG |
|------|------------|-----------|------------|
| **Answer Relevance** | 0.60 | 0.67 | **0.72** ✓ |
| **Entity Score** | 0.27 | 0.29 | **0.33** ✓ |
| **Hallucination** | 0.78 | - | **0.26** ✓ |

### 5.2 주요 발견

#### Answer Relevance (질의-응답 관련성)
- **Hybrid RAG**: 0.72로 최고 성능 → 다양한 정보 소스 결합으로 풍부한 문맥 제공
- **Graph RAG**: 0.67 → 관계 기반 정보로 문맥적 관련성 확보
- **Vector RAG**: 0.60 → 텍스트 유사도 기반으로 제한적 문맥 이해

#### Entity Score (키워드 일치율)
- **Hybrid RAG**: 0.33으로 최고 → 정답 단어 포함률 가장 높음
- **Graph RAG**: 0.29 → Vector RAG보다 약간 우수
- **Vector RAG**: 0.27 → 가장 낮은 키워드 일치도

#### Hallucination (환각 현상)
- **Hybrid RAG**: 0.26으로 최저 → 잘못된 정보 생성 가장 적음
- **Graph RAG**: 측정 불가 (참조 정보 추출 한계)
- **Vector RAG**: 0.78로 최고 → 환각 현상 가장 빈번

### 5.3 질적 분석 예시

**질의**: "미국의 중국에 대한 무역규제가 국도화학에게 미치는 영향이 있을까요?"

- **Vector RAG**: 반덤핑 및 상계관세 조사 언급, 참조 1개 ({Source: 반덤핑, page 1})
- **Graph RAG**: 간접적 영향 추론, 참조 없음
- **Hybrid RAG**: 공급망, 시장 접근성, 생산/판매 감소 등 구체적 영향 제시, 참조 2개 ({Source: 23중국 에폭시, page 3, 1})

---

## 6. 우리 연구와의 관련성

### 6.1 직접적 관련성

#### 한국어 행정문서 처리
- **본 논문**: 기업분석 보고서를 대상으로 한국어 특성 반영
- **우리 연구**: 공공 행정문서를 대상으로 동일한 접근법 적용 가능
- **활용 포인트**: 생략어 복원 + 의존 구문 분석 파이프라인 그대로 차용

#### Hybrid RAG 아키텍처
- **본 논문**: Graph RAG 우선 + Vector RAG 보완 전략
- **우리 연구**: 동일한 Hybrid 전략으로 행정문서 Q&A 시스템 구축
- **차별점**:
  - 본 논문: 기업 도메인 특화
  - 우리 연구: 공공 행정 도메인 특화 (AI Hub 데이터 활용)

#### 성능 평가 프레임워크
- **본 논문**: Answer Relevance, Entity Score, Hallucination 3개 지표
- **우리 연구**: RAGAS 프레임워크 활용 (Faithfulness, Answer Relevancy, Context Precision 등)
- **보완 관계**:
  - 본 논문의 Entity Score → 우리 연구에 추가 가능
  - 우리 연구의 Faithfulness → 본 논문의 Hallucination과 유사 개념

### 6.2 인용 가능한 핵심 포인트

1. **한국어 RAG의 필요성**
   - "한국어는 동사 중심의 언어로 주어나 목적어가 자주 생략되고 중의적 표현이 많아 한국어의 특성상 기존의 영어 중심의 자연어 처리 기술을 적용하는 데 한계가 있어, 한국어의 언어적 특성을 반영한 지식그래프 구축 연구는 부족한 실정이다" (p.2)

2. **Graph RAG의 한계와 Hybrid 접근의 필요성**
   - "Graph RAG는 특정 개체(Entity)를 인식하지 못하는 경우 답변을 생성하기 어려운 한계를 가진다. 이를 보완하기 위해, 본 연구에서는 Vector RAG를 결합하여 Hybrid RAG를 구현하고자 한다" (p.2)

3. **Ko-Triple Extraction 방법론**
   - "본 연구는 기존 선행 연구를 바탕으로, 생략어 복원과 의존 구문 분석을 결합하여 한국어에 특화된 지식베이스 구축 프로세스를 제안한다" (p.5)

4. **성능 우위**
   - "실험 결과, Hybrid RAG는 Graph RAG와 Vector RAG를 결합한 접근법으로, 각 기법의 강점을 효과적으로 통합하여 가장 우수한 성능을 보였다" (p.2)

### 6.3 우리 연구에 적용 가능한 방법론

#### 1. 지식 그래프 구축 파이프라인
```python
# 본 논문의 Ko-Triple Extraction 파이프라인
Document → 접속어 분리 → 생략어 복원 → 의존 구문 분석 → GPT-3.5 트리플 추출

# 우리 연구 적용 예시
AI Hub 행정문서 → 접속어 분리 → Korean Anaphora Resolution → SpaCy 의존 분석 → LLM 트리플 추출
```

#### 2. Hybrid RAG 전략
- **Graph RAG 우선**: 관계 기반 검색으로 맥락적 정확도 확보
- **Vector RAG 보조**: 참조 문헌 및 근거 제공으로 신뢰성 향상
- **Fallback**: Graph RAG 실패 시 Vector RAG 독립 실행

#### 3. Q&A 데이터셋 자동 생성
```
청크 분할 → LLM 요약 → LLM Q&A 생성 (Q1, A1, R1, Q2, A2, R2) → 검증 → 최종 데이터셋
```

---

## 7. 인용 가능한 핵심 문장 (원문)

### 7.1 연구 배경

> "최근 자연어 처리(NLP) 분야에서는 언어 모델이 사용자 질문에 대한 답변을 생성할 때 필요한 정보를 외부 데이터베이스에서 검색하고 이를 통합하여 응답의 정확성과 신뢰성을 높이는 RAG(Retriever Augmented Generation) 기법이 주목받고 있다" (p.1)

> "Graph RAG는 지식 그래프를 검색 엔진으로 활용하여 특정 도메인에서 불필요한 데이터를 효율적으로 필터링하고, 텍스트와 그래프 구조 정보를 통합함으로써 정확하고 일관된 응답을 생성하는 강점을 가진다" (p.1-2)

### 7.2 방법론

> "본 연구는 한국어의 복잡성과 다양성을 비교적 잘 반영할 수 있는 데이터로 기업분석 보고서를 활용하며, 한국어의 언어적 특성을 고려한 생략어 복원과 의존 구문 분석 기법을 연결시켜 기존의 연구와 차별화된 지식베이스 구축 프로세스를 제안한다" (p.3-4)

> "생략어 복원은 문서 내에서 특정 대명사나 개체를 찾는 상호참조 해결과 달리, 생략된 주어나 목적어를 복원해야 하는 문제로, 특히 한국어, 일본어, 중국어와 같은 동아시아 언어 및 이탈리아어에서 빈번하게 발생한다" (p.4)

> "의존 구문 분석은 문장 내에서 단어 간의 의존 관계 태그를 통해 표현하는 자연어 처리 방법으로, 문장의 다양한 해석으로 인해 발생하는 모호성을 해결하는 연구에서 적극적으로 활용되고 있다" (p.4)

### 7.3 Hybrid RAG 설계

> "본 연구에서 구현한 Hybrid RAG는 Graph RAG의 관계 기반 검색과 Vector RAG의 벡터화된 문서 검색을 결합함으로써, 질의응답 시스템에서 높은 정확성과 신뢰성을 보장하고자 하였다" (p.23)

> "Hybrid RAG는 Graph RAG 또는 Vector RAG가 생성한 응답과 함께, 검색된 문서의 출처 및 페이지 정보를 포함한 참고 문헌을 반환하였다. 이렇게 생성된 응답은 질문에 대한 명확한 답변과 근거를 제공하며, 신뢰성있는 시스템을 구현하고자 하였다" (p.23)

### 7.4 실험 결과

> "실험 결과, Hybrid RAG가 Answer Relevance, Entity Score, Hallucination의 3가지 주요 평가 지표에서 가장 우수한 성능을 보였다" (p.31)

> "Hybrid RAG는 0.72로 가장 높은 점수를 기록했으며, 질의와 응답 간의 문맥적 관련성이 가장 우수함을 보여준다. Graph RAG는 0.67로 그 뒤를 이었으며, Vector RAG는 0.60으로 가장 낮은 값을 보였다. 이는 Hybrid RAG가 다양한 정보 소스를 결합하여 더 풍부한 문맥 정보를 제공할 수 있음을 보여주고 있다" (p.30)

> "Hybrid RAG는 0.26으로 가장 낮은 값을 기록하며, 생성된 응답에서 불필요하거나 잘못된 정보가 가장 적게 포함되었음을 보여준다. 반면, Vector RAG는 0.78로 매우 높은 값을 보이며 환각 현상이 가장 빈번하게 발생하였다" (p.30-31)

### 7.5 결론 및 기여

> "본 연구는 한국어의 언어적 특성을 반영한 Hybrid RAG 기반 질의응답 시스템의 구현과 평가를 통해, 한국어 NLP 분야에서의 실질적 활용 가능성을 제시한다" (p.2)

> "본 연구에서 제안한 새로운 Q&A 시스템 설계 방법은 Hybrid RAG의 성능 향상에 기여할 뿐만 아니라, 다양한 도메인에서의 활용 가능성을 제시할 것으로 기대된다" (p.i)

---

## 8. 한계점 및 향후 연구방향

### 8.1 한계점

#### 평가 지표의 다양성 부족
- 현재 3개 지표(Answer Relevance, Entity Score, Hallucination)만 사용
- 사용자 경험, 응답 속도, 시스템 확장성 등 추가 지표 필요

#### Graph RAG의 참조 정보 추출 한계
- Graph RAG에서 Hallucination 지표 측정 불가
- 그래프 구조상 소스 정보를 함께 추출하는 것에 한계 존재

#### 도메인 특화성
- 화학 제조업 K사 데이터로 한정
- 다양한 산업 및 공공 도메인으로의 일반화 필요

#### 한국어 모델의 제한
- 범용 LLM(GPT-3.5-turbo) 사용
- 한국어 특화 LLM 개발 및 활용 필요

### 8.2 향후 연구방향

> "본 연구의 향후과제는 Hybrid RAG를 비롯한 RAG 기반 질의응답 시스템의 성능과 활용성을 개선하고 확장하는 데 중점을 둔다" (p.32)

#### 평가 지표 확장
> "본 연구는 Answer Relevance, Entity Score, Hallucination의 세 가지 성능 지표를 활용하였지만, 이를 넘어 다양한 평가 지표를 추가하여 시스템 성능을 보다 다각적으로 검토할 필요가 있다" (p.32)

#### 한국어 특화 모델 개발
> "한국어의 언어적 특성을 반영한 모델 개발이 필요하다. 이를 위해 대규모 한국어 데이터셋과 최신 언어모델을 활용하여 RAG 시스템에 최적화된 언어모델을 구축해야 한다" (p.32)

#### 도메인 확장
- 금융, 법률, 의료, 공공 행정 등 다양한 도메인으로 확장
- 우리 연구의 공공 행정문서 적용이 이러한 확장의 한 사례

---

## 9. 우리 연구 적용 시 고려사항

### 9.1 데이터 특성 차이

| 구분 | 본 논문 | 우리 연구 |
|------|---------|-----------|
| **도메인** | 화학 제조업 (민간) | 공공 행정 (AI Hub) |
| **문서 유형** | 기업분석 보고서, 시장 동향 보고서 | 행정문서 기계독해 데이터 |
| **언어 특성** | 전문 용어 밀집, 수치 데이터 많음 | 법률/행정 용어, 규정 중심 |
| **트리플 수** | 3,237개 | TBD (우리 연구에서 측정 필요) |
| **Q&A 수** | 140개 (검증 후) | AI Hub 제공 + 자체 생성 |

### 9.2 기술 스택 매핑

| 구성 요소 | 본 논문 | 우리 연구 |
|-----------|---------|-----------|
| **LLM** | GPT-3.5-turbo | Gemini 2.0 Flash (우선), GPT-4o-mini (대체) |
| **임베딩** | OpenAI Embeddings | Gemini Embeddings (우선), OpenAI (대체) |
| **생략어 복원** | Korean Anaphora Resolution | 동일 모델 사용 가능 |
| **의존 구문 분석** | SpaCy ko-core-news-sm | 동일 |
| **Vector DB** | Chroma, FAISS | Chroma (AutoRAG 기본) |
| **평가 프레임워크** | 자체 구현 (3개 지표) | RAGAS + 본 논문 지표 추가 |

### 9.3 적용 전략

#### Phase 1: Ko-Triple Extraction 파이프라인 구축
1. AI Hub 행정문서에 접속어 분리 적용
2. Korean Anaphora Resolution으로 생략어 복원
3. SpaCy로 의존 구문 분석
4. Gemini 2.0 Flash로 트리플 추출 (본 논문은 GPT-3.5 사용)
5. 추출된 트리플로 Knowledge Graph 구축

#### Phase 2: Hybrid RAG 구현
1. **Graph RAG**:
   - FAISS에 트리플 임베딩 저장
   - 질의에 대해 최대 5개 노드 검색
   - Gemini로 관계 기반 답변 생성

2. **Vector RAG**:
   - Chroma에 문서 청크 저장
   - 질의 유사도 기반 검색
   - Gemini로 답변 생성

3. **Hybrid**:
   - Graph RAG 우선 → 키워드 추출 → Vector RAG 참조 추가
   - Graph RAG 실패 시 Vector RAG 독립 실행

#### Phase 3: 평가
1. **RAGAS 지표**: Faithfulness, Answer Relevancy, Context Precision, Context Recall
2. **본 논문 지표 추가**: Entity Score (키워드 일치율)
3. **비교 분석**: Naive RAG vs Graph RAG vs Hybrid RAG

### 9.4 예상 개선 포인트

1. **Gemini의 한국어 성능**: 본 논문은 GPT-3.5 사용, Gemini 2.0 Flash가 한국어 처리에서 더 우수할 가능성
2. **AI Hub 데이터 품질**: 이미 검증된 공공 데이터로 더 일반화된 결과 기대
3. **RAGAS 평가**: 본 논문보다 더 다각적인 평가 가능 (Faithfulness 등)
4. **AutoRAG 통합**: 본 논문은 수동 구현, AutoRAG 프레임워크로 자동화 가능

---

## 10. 참고문헌 (본 논문 인용 시)

### APA 형식
```
이채원. (2024). 한국어 Hybrid RAG 기반 질의응답 시스템 구축에 관한 연구
[석사학위논문, 동아대학교 대학원].
```

### 영문 참고문헌
```
Lee, C. (2024). A Study on the Construction of a Korean Hybrid RAG-based
Question-Answering System [Master's thesis, Dong-A University].
Department of Management Information Systems.
```

### BibTeX
```bibtex
@mastersthesis{lee2024hybrid,
  title={한국어 Hybrid RAG 기반 질의응답 시스템 구축에 관한 연구},
  author={이채원},
  year={2024},
  school={동아대학교 대학원},
  type={석사학위논문},
  address={부산, 대한민국}
}
```

---

## 11. 추가 참고 자료

### 본 논문이 인용한 주요 선행연구

1. **RAG 기본**
   - Lewis et al. (2020). "Retrieval-augmented generation for knowledge-intensive nlp tasks"
   - Gao et al. (2024). "Retrieval-augmented generation for large language models: A survey"

2. **Graph RAG**
   - Hu et al. (2024). "GRAG: Graph Retrieval-Augmented Generation"
   - Sarmah et al. (2024). "HybridRAG: Integrating Knowledge Graphs and Vector Retrieval"

3. **한국어 NLP**
   - Park et al. (2021). "Optimizing ELECTRA-based model for Zero Anaphora Resolution"
   - Cho et al. (2021). "Korean dependency parsing based on sequence labeling"

4. **Knowledge Graph**
   - Ryu & Cha (2022). "Developing a Knowledge Graph based on Ontology Learning"
   - Kim & Gim (2022). "BERT based Relation Extraction Model and Knowledge Graph"

---

## 12. 결론

본 논문은 **한국어 특화 Hybrid RAG 시스템 구축**에 관한 실질적이고 체계적인 연구로, 우리의 공공 행정문서 RAG 프로젝트에 직접적으로 적용 가능한 방법론과 인사이트를 제공한다. 특히 **Ko-Triple Extraction 파이프라인**(접속어 분리 → 생략어 복원 → 의존 구문 분석 → 트리플 추출)과 **Hybrid RAG 아키텍처**(Graph RAG 우선 + Vector RAG 보완)는 그대로 차용할 가치가 있다.

다만, 본 논문은 기업 도메인에 특화되어 있어, 공공 행정문서의 특성(법률 용어, 규정 중심, 계층적 구조)을 추가로 고려해야 하며, 평가 지표도 RAGAS 프레임워크와 결합하여 더 다각적으로 측정할 필요가 있다. Gemini 2.0 Flash와 AutoRAG 프레임워크를 활용하면 본 논문보다 더 발전된 시스템 구축이 가능할 것으로 기대된다.

---

**검토자**: AI Assistant
**검토일**: 2025-11-30
**논문 페이지 수**: 44페이지
**트리플 데이터셋**: 3,237개
**Q&A 데이터셋**: 140개 (검증 후)
**주요 키워드**: Korean NLP, Hybrid RAG, Knowledge Graph, Zero Anaphora Resolution, Dependency Parsing
