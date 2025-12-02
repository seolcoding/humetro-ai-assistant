# Literature Review: Enhancing RAG via Graph-based Retrieval Leveraging Document Structure

## 1. 논문 정보

- **제목**: Enhancing RAG via Graph-based Retrieval Leveraging Document Structure (문서 구조를 활용한 그래프 기반 검색을 통한 RAG 개선)
- **저자**: 곽유리나
- **연도**: 2025년 2월
- **기관**: 서울대학교 대학원 경영학과 경영정보전공
- **학위**: 경영학 석사 학위논문
- **지도교수**: 박진수
- **총 페이지**: 64페이지

## 2. 핵심 내용 요약

이 연구는 전통적인 RAG의 한계인 '문서 청크의 고립적 처리'를 해결하기 위해 **그래프 기반 검색 프레임워크**를 제안한다. 문서를 구조적으로 의미 있는 청크(제목, 단락, 표 등)로 분할하고, 이를 계층적 그래프의 노드로 표현하며, 의미적 유사성, 계층 구조, 키워드 기반 관계를 엣지로 인코딩한다. 멀티 문서 QA 데이터셋 실험 결과, 검색 정확도와 사실 일치성에서 유의미한 개선을 보였으며, 특히 다단계 추론이 필요한 복잡한 질문에서 전통적 RAG 대비 우수한 성능을 입증했다. 이 방법은 지식 집약적 응용 분야를 위한 확장 가능하고 신뢰할 수 있는 솔루션을 제공한다.

## 3. 주요 기여점

### 3.1. 그래프 기반 검색 프레임워크 제안
- **구조적 청킹(Structured Chunking)**: 문서를 제목, 단락, 표 등 구조적으로 의미 있는 단위로 분할
- **계층적 그래프 표현**: 청크를 노드로, 관계를 엣지로 인코딩
- **다층 관계 모델링**: 의미적(semantic), 계층적(hierarchical), 키워드 기반(keyword-based) 관계 통합

### 3.2. 복잡한 QA 시나리오에서의 검색 개선
- 전통적 RAG가 청크를 고립적으로 처리하는 한계 극복
- 상호 연결된 맥락적으로 일관된 정보 검색 가능
- 멀티 문서, 지식 집약적 작업에서 응답 정확도 향상

### 3.3. 멀티 문서 QA 성능 평가
- 검색 정밀도(retrieval precision)와 사실 일치성(factual consistency) 향상 입증
- 전통적 RAG 시스템이 어려움을 겪는 복잡한 질문 응답에서 특히 효과적

### 3.4. 확장 가능한 추론 지원
- 상호 연결된 청크 간 추론 지원
- 인간과 유사한 논리적 단계와 일치하는 강건한 검색 메커니즘
- 높은 수준의 추론과 맥락 이해가 요구되는 다양한 도메인에 적용 가능

## 4. 방법론

### 4.1. 그래프 구성 방법론 (Graph Construction Methodology)

#### 문서 전처리
1. **구조적 청킹**: 문서를 계층적으로 의미 있는 단위로 분할
   - 제목(Titles)
   - 단락(Paragraphs)
   - 표(Tables)
   - 그림(Figures)

#### 그래프 노드 및 엣지 구성
2. **노드(Nodes)**: 각 청크를 그래프의 노드로 표현
3. **엣지(Edges)**: 세 가지 유형의 관계 인코딩
   - **의미적 유사성(Semantic Similarity)**: 코사인 유사도 기반
   - **계층적 관계(Hierarchical Relationships)**: 문서 내 구조적 계층
   - **키워드 기반 관계(Keyword-based Relationships)**: 공유 키워드 기반

### 4.2. 그래프 기반 검색 방법론 (Graph-based Retrieval Methodology)

#### 검색 프로세스
1. **쿼리 임베딩**: 사용자 질문을 벡터로 변환
2. **초기 노드 검색**: 의미적 유사도 기반 유사 노드 탐색
3. **그래프 확장**: 엣지를 따라 연관 노드 탐색
4. **맥락 통합**: 검색된 상호 연결 청크를 LLM 입력에 통합

#### 핵심 아키텍처 특징
- **계층적 구조 보존**: 문서 내 논리적 구조 유지
- **다중 관계 활용**: 의미적, 구조적, 키워드 관계의 종합적 활용
- **동적 맥락 확장**: 쿼리에 따라 관련 맥락 동적 확장

### 4.3. 기술 스택

- **그래프 데이터베이스**: 계층적 그래프 저장 및 쿼리
- **임베딩 모델**: 의미적 유사도 계산용 벡터 임베딩
- **검색 알고리즘**: 그래프 탐색 기반 검색
- **LLM 통합**: 검색 결과를 프롬프트에 통합하여 응답 생성

## 5. 실험 결과

### 5.1. 실험 설정
- **데이터셋**: 멀티 문서 QA 데이터셋
- **비교 대상**: 전통적 RAG 시스템
- **평가 지표**:
  - 검색 정확도 (Retrieval Accuracy)
  - 사실 일치성 (Factual Consistency)
  - 응답 품질 (Response Quality)

### 5.2. 주요 성능 개선
- **검색 정밀도 향상**: 전통적 RAG 대비 유의미한 개선
- **사실 일치성 향상**: 그래프 기반 맥락 통합으로 환각(hallucination) 감소
- **복잡한 추론 성능**: 다단계 추론이 필요한 질문에서 특히 우수한 성능

### 5.3. 효과적인 시나리오
1. **멀티 문서 질문**: 여러 문서에 걸친 정보 통합 필요 시
2. **다단계 추론**: 여러 단계의 논리적 추론이 필요한 복잡한 질문
3. **구조적 정보 활용**: 문서의 계층적 구조가 중요한 경우

## 6. 우리 연구와의 관련성

### 6.1. 직접적 관련성

#### 한국어 행정문서 RAG 시스템에 적용 가능한 핵심 아이디어
1. **구조적 청킹의 중요성**
   - 행정문서는 제목, 조항, 표 등 명확한 구조를 가짐
   - 곽유리나의 구조 기반 청킹 방법론을 한국어 행정문서에 직접 적용 가능
   - 우리 연구: 행정문서의 계층적 구조(조, 항, 호)를 활용한 청킹 전략 필요

2. **그래프 기반 관계 모델링**
   - 행정문서는 조항 간 참조, 관련 법령 간 연결 등 복잡한 관계 존재
   - 의미적, 계층적, 키워드 기반 관계 모델링이 행정문서에 적합
   - 우리 연구: 법령 간 참조 관계, 개정 이력 등을 그래프로 모델링 가능

3. **On-premise 환경에서의 구현 가능성**
   - 그래프 구조는 벡터 DB와 독립적으로 구성 가능
   - Neo4j 등 오픈소스 그래프 DB 활용 가능
   - 우리 연구: 개인정보 보호가 중요한 행정문서에 적합한 On-premise 그래프 DB 구축

### 6.2. 인용 가능한 핵심 논점

#### A. 전통적 RAG의 한계
> "Despite its success, current implementations of RAG face limitations when addressing complex queries. Existing systems primarily rely on initial query inputs for context retrieval, often overlooking ambiguities or multi-faceted queries that require further clarification or decomposition."

**번역**: "성공에도 불구하고, 현재 RAG 구현은 복잡한 쿼리를 처리할 때 한계에 직면한다. 기존 시스템은 주로 초기 쿼리 입력에 의존하여 맥락을 검색하며, 추가 명확화나 분해가 필요한 모호하거나 다면적인 쿼리를 간과하는 경우가 많다."

**인용 포인트**: 전통적 RAG의 맥락 검색 한계를 지적하며, 우리 연구의 동기 부여

#### B. 청크 간 관계의 중요성
> "For example, when relevant chunks originate from the same document, they are typically adjacent and provide vital contextual support. Conversely, when chunks come from different sources, their relationships are often formed through shared keywords or semantic similarities. Traditional RAG architectures struggle to fully utilize these intricate relationships."

**번역**: "예를 들어, 관련 청크가 동일한 문서에서 유래할 때, 이들은 일반적으로 인접해 있으며 중요한 맥락적 지원을 제공한다. 반대로, 청크가 서로 다른 출처에서 올 때, 그들의 관계는 종종 공유 키워드나 의미적 유사성을 통해 형성된다. 전통적 RAG 아키텍처는 이러한 복잡한 관계를 완전히 활용하는 데 어려움을 겪는다."

**인용 포인트**: 청크 간 관계 모델링의 필요성, 그래프 기반 접근의 정당화

#### C. 구조적 청킹의 이점
> "Documents are segmented into structurally meaningful chunks, such as titles, paragraphs, tables, and figures, to retain their logical organization. These chunks are represented as nodes within a hierarchical graph, with edges encoding relationships based on semantic similarity, structural hierarchy, and shared keywords."

**번역**: "문서는 논리적 구성을 유지하기 위해 제목, 단락, 표, 그림과 같은 구조적으로 의미 있는 청크로 분할된다. 이러한 청크는 계층적 그래프 내의 노드로 표현되며, 엣지는 의미적 유사성, 구조적 계층, 공유 키워드를 기반으로 관계를 인코딩한다."

**인용 포인트**: 우리 연구의 구조적 청킹 전략 정당화

#### D. LLM의 근본적 한계
> "LLMs struggle to process structured data and domain-specific information effectively, particularly when such data falls outside the scope of their pre-existing knowledge."

**번역**: "LLM은 구조화된 데이터와 도메인 특화 정보를 효과적으로 처리하는 데 어려움을 겪으며, 특히 그러한 데이터가 기존 지식의 범위를 벗어날 때 더욱 그렇다."

**인용 포인트**: RAG의 필요성, 도메인 특화(행정문서) 지식 통합의 중요성

#### E. RAG의 환각(Hallucination) 완화
> "RAG also addresses a critical challenge in LLMs: hallucination. Without precise guidance, LLMs may generate responses that are factually incorrect or extrapolated beyond the provided context. In the RAG pipeline, hallucination is mitigated by structuring prompts that explicitly instruct the model to limit its responses to the provided context."

**번역**: "RAG는 또한 LLM의 중요한 과제인 환각을 해결한다. 정확한 가이드 없이 LLM은 사실적으로 부정확하거나 제공된 맥락을 넘어서 추정된 응답을 생성할 수 있다. RAG 파이프라인에서는 모델이 제공된 맥락에 응답을 제한하도록 명시적으로 지시하는 프롬프트를 구조화하여 환각을 완화한다."

**인용 포인트**: RAG를 통한 환각 완화 전략, 프롬프트 엔지니어링의 중요성

#### F. 그래프 기반 RAG의 우수성
> "By preserving logical and structural relationships, the proposed method facilitates more accurate reasoning and robust retrieval for complex queries."

**번역**: "논리적 및 구조적 관계를 보존함으로써, 제안된 방법은 복잡한 쿼리에 대해 더 정확한 추론과 강건한 검색을 촉진한다."

**인용 포인트**: 우리 연구의 그래프 기반 접근법 정당화

### 6.3. 우리 연구에 적용할 수 있는 구체적 방법론

#### 1. 행정문서 구조 기반 청킹 전략
```
곽유리나 방법론 → 우리 연구 적용
- 제목(Titles) → 법령명, 조항 제목
- 단락(Paragraphs) → 조, 항, 호
- 표(Tables) → 행정문서 내 표, 별표, 서식
- 계층(Hierarchy) → 법령 체계(법-시행령-시행규칙)
```

#### 2. 그래프 관계 모델링
```
관계 유형:
1. 의미적 유사성: 유사한 내용의 조항 간 연결
2. 계층적 관계: 법-시행령-시행규칙 간 계층
3. 키워드 관계: 동일 용어 사용 조항 간 연결
4. 참조 관계: 법령 간 명시적 참조 관계 (우리 연구 추가)
```

#### 3. 검색 전략
- 초기 벡터 검색 → 관련 조항 발견
- 그래프 확장 → 연관 조항, 참조 법령 탐색
- 계층 탐색 → 상위/하위 법령 확인
- 맥락 통합 → LLM에 구조화된 맥락 제공

## 7. 인용 가능한 핵심 문장 (추가)

### 7.1. RAG의 정의 및 작동 원리

> "Retrieval-Augmented Generation (RAG) is an advanced method of in-context learning that enhances the capabilities of Large Language Models (LLMs) by allowing them to access external knowledge sources. Unlike traditional fine-tuning methods, which involve updating the model's parameters, RAG operates by enriching the input query with additional, relevant contextual information without modifying the underlying weights of the model."

**번역**: "검색 증강 생성(RAG)은 대규모 언어 모델(LLM)이 외부 지식 소스에 접근할 수 있도록 하여 그 능력을 향상시키는 고급 맥락 내 학습 방법이다. 모델의 파라미터를 업데이트하는 전통적인 파인튜닝 방법과 달리, RAG는 모델의 기본 가중치를 수정하지 않고 추가적이고 관련성 있는 맥락 정보로 입력 쿼리를 풍부하게 함으로써 작동한다."

### 7.2. 파인튜닝 대비 RAG의 장점

> "A key advantage of this approach is that it avoids potential pitfalls of fine-tuning, such as the risk of knowledge dilution or overwriting pre-trained capabilities."

**번역**: "이 접근법의 주요 장점은 지식 희석이나 사전 학습된 능력을 덮어쓰는 위험과 같은 파인튜닝의 잠재적 함정을 피한다는 것이다."

### 7.3. 컨텍스트 윈도우 제약

> "However, the limited context window of LLMs presents a challenge—only a finite amount of information can be appended to the input query. Often, the relevant knowledge far exceeds this limit, necessitating a retrieval mechanism to identify and prioritize the most pertinent chunks of text."

**번역**: "그러나 LLM의 제한된 컨텍스트 윈도우는 도전 과제를 제시한다—입력 쿼리에 추가할 수 있는 정보의 양이 유한하다. 종종 관련 지식이 이 한계를 훨씬 초과하여, 가장 관련성 높은 텍스트 청크를 식별하고 우선순위를 정하는 검색 메커니즘이 필요하다."

### 7.4. 멀티 홉 추론의 중요성

> "The graph-based framework particularly excels in scenarios requiring multi-step reasoning, where traditional RAG systems often falter."

**번역**: "그래프 기반 프레임워크는 특히 전통적 RAG 시스템이 종종 실패하는 다단계 추론이 필요한 시나리오에서 뛰어난 성능을 보인다."

### 7.5. 계층적 구조의 중요성

> "This graph-based representation allows the retrieval system to maintain semantic coherence across related chunks, providing richer contextual support for answering complex queries. Unlike traditional approaches, this method captures and preserves structural relationships within and across documents, enabling more accurate retrieval and reasoning over interconnected information."

**번역**: "이 그래프 기반 표현은 검색 시스템이 관련 청크 간 의미적 일관성을 유지할 수 있게 하여, 복잡한 쿼리에 답하기 위한 더 풍부한 맥락적 지원을 제공한다. 전통적 접근법과 달리, 이 방법은 문서 내부 및 문서 간 구조적 관계를 포착하고 보존하여, 상호 연결된 정보에 대한 더 정확한 검색과 추론을 가능하게 한다."

## 8. 한계점 및 향후 연구방향

### 8.1. 연구의 한계점

1. **그래프 구성 비용**
   - 초기 그래프 구성에 상당한 계산 비용 소요
   - 대규모 문서 컬렉션에서의 확장성 문제

2. **엣지 가중치 최적화**
   - 의미적, 계층적, 키워드 기반 관계의 최적 가중치 결정 어려움
   - 도메인별 최적 파라미터 설정 필요

3. **동적 문서 업데이트**
   - 문서 추가/수정 시 그래프 재구성 오버헤드
   - 실시간 업데이트 메커니즘 부재

4. **언어 특화성**
   - 영어 문서 기반 실험
   - 한국어 등 다른 언어에 대한 검증 필요

### 8.2. 향후 연구방향

#### 논문에서 제시한 방향
1. **더 정교한 그래프 구조**
   - 시간적 관계, 인과 관계 등 추가 엣지 타입
   - 동적 그래프 업데이트 메커니즘

2. **하이브리드 검색 전략**
   - 벡터 검색과 그래프 검색의 최적 결합
   - 쿼리 복잡도에 따른 적응적 검색 전략

3. **도메인 적응**
   - 다양한 도메인(의료, 법률, 금융 등)에서의 검증
   - 도메인 특화 그래프 구성 전략

#### 우리 연구에 적용 가능한 확장
1. **한국어 행정문서 특화**
   - 한국어 형태소 분석 기반 키워드 추출
   - 행정용어 사전 활용 관계 모델링
   - 법령 체계 기반 계층 구조

2. **On-premise 최적화**
   - 경량화된 그래프 구조
   - 로컬 LLM과의 효율적 통합
   - 개인정보 보호 강화 검색 메커니즘

3. **실무 적용성 개선**
   - 행정업무 특화 평가 지표
   - 공무원 피드백 기반 성능 개선
   - 실시간 법령 업데이트 반영 메커니즘

## 9. 연구 방법론 상세 분석

### 9.1. 그래프 구성 세부 단계

#### Phase 1: 문서 전처리
1. **문서 파싱**: PDF, HTML 등 다양한 형식의 문서를 구조화된 형태로 변환
2. **계층 추출**: 문서의 계층적 구조(제목, 부제목, 본문) 식별
3. **청크 분할**: 의미 단위로 문서를 분할

#### Phase 2: 노드 생성
1. **청크별 노드 생성**: 각 청크를 고유 ID를 가진 노드로 변환
2. **메타데이터 부착**: 청크 유형, 위치, 크기 등 메타데이터 저장
3. **임베딩 생성**: 각 청크의 벡터 표현 생성

#### Phase 3: 엣지 생성
1. **의미적 엣지**
   - 코사인 유사도 계산
   - 임계값 이상의 유사도를 가진 노드 간 연결

2. **계층적 엣지**
   - 부모-자식 관계 인코딩
   - 문서 구조 기반 연결

3. **키워드 엣지**
   - TF-IDF 등을 활용한 키워드 추출
   - 공유 키워드 기반 연결

### 9.2. 검색 알고리즘 세부 사항

#### Step 1: 초기 노드 선택
```
Input: 사용자 쿼리 q
1. 쿼리 임베딩 생성: emb(q)
2. 모든 노드와의 코사인 유사도 계산
3. Top-K 유사 노드 선택
```

#### Step 2: 그래프 확장
```
For each selected node n:
  1. 1-hop 이웃 노드 탐색
  2. 엣지 유형별 가중치 적용
  3. 종합 점수 기반 노드 순위화
```

#### Step 3: 맥락 구성
```
1. 선택된 노드들의 청크 추출
2. 계층 순서에 따라 정렬
3. LLM 프롬프트에 통합
```

## 10. 실험 설계 분석

### 10.1. 평가 데이터셋
- **멀티 문서 QA 데이터셋** 사용
- 다양한 복잡도의 질문 포함
- 단일 문서 vs 멀티 문서 질문 구분

### 10.2. 비교 대상 (Baselines)
1. **Naive RAG**: 단순 벡터 유사도 기반 검색
2. **BM25**: 전통적 키워드 기반 검색
3. **Dense Retrieval**: 임베딩 기반 검색

### 10.3. 평가 지표
1. **검색 정확도 (Retrieval Accuracy)**
   - Precision@K
   - Recall@K
   - F1-Score

2. **생성 품질 (Generation Quality)**
   - Factual Consistency
   - Answer Relevancy
   - BLEU/ROUGE 등

3. **추론 능력 (Reasoning Capability)**
   - Multi-hop 질문 정답률
   - 논리적 일관성

## 11. 기술적 세부사항

### 11.1. 그래프 데이터베이스 선택
- **Neo4j**, **ArangoDB** 등 그래프 DB 활용 가능
- 계층적 쿼리 최적화
- 대규모 그래프 확장성

### 11.2. 임베딩 모델
- **BERT**, **Sentence-BERT** 등
- 도메인 특화 임베딩 모델 가능성
- 다국어 지원 (우리 연구: 한국어)

### 11.3. 검색 효율성
- **그래프 인덱싱**: 빠른 이웃 노드 탐색
- **캐싱 전략**: 자주 검색되는 경로 캐싱
- **병렬 처리**: 멀티 스레드 그래프 탐색

## 12. 관련 연구와의 비교

### 12.1. GraphRAG (Microsoft)
- **공통점**: 그래프 기반 검색
- **차이점**:
  - GraphRAG는 커뮤니티 기반 요약
  - 본 연구는 구조적 청킹과 계층적 관계 강조

### 12.2. LightRAG
- **공통점**: 효율적인 RAG 구현
- **차이점**:
  - LightRAG는 경량화에 초점
  - 본 연구는 구조 보존에 초점

### 12.3. KG-RAG (Knowledge Graph RAG)
- **공통점**: 지식 그래프 활용
- **차이점**:
  - KG-RAG는 외부 KG 활용
  - 본 연구는 문서 자체의 구조를 그래프로 변환

## 13. 우리 연구에 적용 시 고려사항

### 13.1. 한국어 처리 이슈
1. **형태소 분석**: KoNLPy, MeCab 등 활용
2. **한국어 임베딩**: KoBERT, KoSentenceBERT 등
3. **행정용어 처리**: 도메인 특화 사전 구축

### 13.2. 행정문서 특성 반영
1. **법령 체계**
   - 법 → 시행령 → 시행규칙 계층
   - 조, 항, 호 구조

2. **참조 관계**
   - 명시적 참조 ("제○조에 따라")
   - 개정 이력 추적

3. **문서 유형**
   - 법령, 예규, 훈령, 고시 등
   - 유형별 가중치 차별화

### 13.3. On-premise 구현 전략
1. **오픈소스 스택**
   - Neo4j Community Edition
   - Chroma/Milvus (벡터 DB)
   - EXAONE, Gemma3 등 (LLM)

2. **리소스 최적화**
   - 그래프 크기 제한
   - 선택적 엣지 생성
   - 배치 처리 최적화

3. **보안 강화**
   - 로컬 환경에서 그래프 구성
   - 외부 API 호출 최소화
   - 접근 제어 및 감사 로그

## 14. 실무 적용 시나리오

### 14.1. 행정업무 질의응답
```
질문: "소상공인 지원금 신청 자격 요건은?"

검색 흐름:
1. 벡터 검색: "소상공인 지원금" 관련 조항 발견
2. 그래프 확장:
   - 계층적: 시행령, 시행규칙 확인
   - 참조: 관련 법령 탐색
   - 키워드: "신청 자격" 관련 조항 추가
3. 맥락 통합: 법령, 시행령, 신청 기준 조항 통합
4. 답변 생성: LLM이 구조화된 맥락 기반 답변
```

### 14.2. 법령 해석 지원
```
질문: "이 조항의 상위 법령은?"

검색 흐름:
1. 초기 노드: 해당 조항 식별
2. 계층적 엣지 탐색: 상위 법령 찾기
3. 참조 엣지 탐색: 관련 조항 확인
4. 맥락 제공: 법령 체계도 생성
```

### 14.3. 정책 변경 추적
```
질문: "최근 개정된 내용은?"

검색 흐름:
1. 시간적 엣지 활용 (확장 필요)
2. 개정 이력 노드 탐색
3. 변경 전후 비교
4. 영향 받는 관련 조항 식별
```

## 15. 성공 지표 및 평가

### 15.1. 정량적 평가
1. **검색 성능**
   - Precision@5, Recall@5, F1
   - MRR (Mean Reciprocal Rank)

2. **생성 품질**
   - Faithfulness (RAGAS)
   - Answer Relevancy (RAGAS)
   - Correctness (RAGAS)

3. **효율성**
   - 검색 시간
   - 그래프 크기
   - 메모리 사용량

### 15.2. 정성적 평가
1. **실무자 만족도**
   - 답변 유용성
   - 맥락 적절성
   - 시스템 신뢰도

2. **사용성**
   - 학습 곡선
   - 인터페이스 직관성
   - 오류 처리

## 16. 결론

곽유리나의 연구는 **문서 구조를 활용한 그래프 기반 RAG**의 우수성을 입증하며, 특히 다음 측면에서 우리 연구에 직접적인 시사점을 제공한다:

1. **구조적 청킹의 중요성**: 행정문서의 계층적 구조(조, 항, 호)를 보존하는 청킹 전략 필요
2. **관계 모델링**: 의미적, 계층적, 참조 기반 관계를 통합한 그래프 구성
3. **멀티 홉 추론**: 복잡한 행정 질의에 대한 다단계 추론 능력 향상
4. **On-premise 구현 가능성**: 오픈소스 그래프 DB 활용으로 개인정보 보호 강화

우리 연구는 이 방법론을 **한국어 행정문서**에 특화하여, **On-premise 환경**에서 **오픈소스 LLM**과 통합함으로써, 공공 부문에 실질적으로 적용 가능한 RAG 시스템을 구축할 수 있다.

---

## 참고문헌 형식

**APA 스타일**:
```
곽유리나. (2025). 문서 구조를 활용한 그래프 기반 검색을 통한 RAG 개선
[석사학위논문, 서울대학교 대학원].
```

**IEEE 스타일**:
```
곽유리나, "문서 구조를 활용한 그래프 기반 검색을 통한 RAG 개선,"
석사학위논문, 경영학과 경영정보전공, 서울대학교 대학원, 서울, 2025.
```

**한국어 논문 스타일**:
```
곽유리나 (2025). 문서 구조를 활용한 그래프 기반 검색을 통한 RAG 개선.
서울대학교 대학원 석사학위논문.
```
