# Literature Review: Efficient Knowledge Graph Construction and Retrieval from Unstructured Text for Large-Scale RAG Systems

## 1. 논문 정보

- **제목**: Efficient Knowledge Graph Construction and Retrieval from Unstructured Text for Large-Scale RAG Systems
- **저자**: Congmin Min, Rhea Mathew, Joyce Pan, Sahil Bansal, Abbas Keshavarzi, Amar Viswanathan Kannan (SAP)
- **연도**: 2025
- **출처**: arXiv:2507.03226v2 [cs.AI] (7 Aug 2025)
- **학회/저널**: arXiv preprint

## 2. 핵심 내용 요약

본 논문은 기업 환경에서 대규모 GraphRAG 시스템을 구축하기 위한 확장 가능하고 비용 효율적인 프레임워크를 제안한다. 주요 혁신은 (1) LLM 없이 산업용 NLP 라이브러리(SpaCy)를 활용한 의존성 파싱 기반 지식 그래프 생성 파이프라인과 (2) 하이브리드 쿼리 노드 식별과 효율적인 1-hop 탐색을 결합한 경량 그래프 검색 전략이다. SAP의 레거시 코드 마이그레이션 데이터셋에서 실험한 결과, 전통적인 RAG 대비 최대 15% 성능 향상을 달성했으며, 의존성 파싱 기반 방식이 LLM 기반 방식 성능의 94%(61.87% vs 65.83%)를 달성하면서도 비용과 확장성 측면에서 큰 이점을 보였다. 이는 대규모 기업 환경에서 실용적이고 설명 가능한 GraphRAG 시스템의 실현 가능성을 입증한다.

## 3. 주요 기여점

1. **LLM 없는 지식 그래프 생성**: 산업용 NLP 라이브러리(SpaCy)의 의존성 파싱을 활용하여 LLM 없이 엔티티와 관계를 추출하는 파이프라인 제시
   - GPT-4o 기반 방식 대비 94% 성능 유지하면서 비용 대폭 절감
   - 도메인 독립적(domain-agnostic) 접근법으로 다양한 분야 적용 가능

2. **경량 그래프 검색 전략**: 하이브리드 쿼리 노드 식별 + 효율적 1-hop 탐색을 결합한 cascaded retrieval 아키텍처
   - SpaCy 명사구 추출 + Vector similarity search로 시드 노드 식별
   - 1-hop 그래프 탐색으로 관련 서브그래프 추출 (낮은 지연시간, 높은 재현율)

3. **실제 기업 환경 적용**: SAP 레거시 코드 마이그레이션 태스크에 GraphRAG 적용한 최초 사례
   - CCM Chat: 15% 성능 향상 (LLM-as-Judge 기준)
   - CCM Code Proposal: 77-78.5% winning rate 달성

4. **비용 분석 프레임워크**: 지식 그래프 생성 과정의 API 비용을 정량적으로 추정하는 CostCalculator 제공

## 4. 방법론

### 4.1 지식 그래프 생성 파이프라인

**전처리 단계**:
- **DocumentParser**: Docling 라이브러리로 PDF/HTML/XLSX/CSV를 JSON/Markdown으로 변환
- **HybridChunker**: Markdown 헤더 기반 계층적 청킹 (최대 2048자, 200자 오버랩)
- **SentenceSegmenter**: SpaCy로 문장 분리 및 동사 포함 여부 필터링 (배치 크기 3, 오버랩 1)
- **ContentFilter**: 동사가 없는 문장 제거하여 LLM 호출 최소화

**트리플 추출**:
- **의존성 파싱 기반 추출** (TripleExtractor - Dependency Parsing):
  - SpaCy 의존성 파서로 구문 트리 생성
  - 명사구 추출 및 정제 → 동사 처리 및 관계 추출 → 주어/목적어 식별
  - 특수 패턴 인식 → 트리플 형성 및 후처리
  - 예시: "SAP launched Joule for Consultants" → {SAP, launch, Joule_for_Consultants}

- **LLM 기반 추출** (선택 사항):
  - GPT-4o 또는 Claude Sonnet 활용
  - 비용 계산 기반으로 데이터셋 규모에 따라 선택

**후처리**:
- **EntityRelationNormalizer**: 특수문자 제거, 엔티티/관계 중복 제거 및 표준화
- **RelationEntityFilter**: 스키마 기반 필터링 (선택적)
- **GraphProducer**: Property Graph 포맷으로 변환
- **KGLoader**: 그래프 DB에 적재 (시각화/분석/프로덕션)

**생성된 그래프 규모** (CCM 데이터셋):
- 39,155 노드
- 47,613 엔티티-엔티티 관계
- 63,681 엔티티-청크 관계
- 평균 노드 차수: 1.52 / 최대 차수: 236

### 4.2 그래프 검색 및 생성 파이프라인

**인덱싱**:
- Vector DB (Milvus): 노드/청크/관계 임베딩 저장 (OpenAI ada-002)
- Graph DB (iGraph): 노드/엣지 인메모리 저장 (고속 탐색)

**검색 프로세스**:

1. **Query Entity Identification**:
   - SpaCy 명사구 추출기로 쿼리에서 핵심 개념 추출
   - 쿼리 전체와 노드 임베딩 간 similarity search (top-k=5)
   - 두 방법으로 얻은 엔티티를 시드 노드로 병합

2. **Graph Query Execution**:
   - 시드 노드와 대소문자 구분 없이 정확 매칭
   - 매칭된 노드에서 1-hop 이웃 탐색
   - `random_k_relations` 파라미터로 이웃 수 제어 (중소형: 100, 대형: 200)

3. **Relevance Ranking and Context Selection**:
   - 후보 관계를 엔티티-엔티티 관계와 엔티티-청크 관계로 분리
   - Milvus에서 임베딩 검색 후 쿼리와 코사인 유사도 계산
   - top-k 청크 + top-k*2 관계 선택
   - 선택적으로 Reciprocal Rank Fusion으로 하이브리드 검색 가능

4. **Context Integration with LLM**:
   - Context = {chunks, relations, entity} 형태로 LLM에 전달
   - 전통적 RAG보다 훨씬 풍부한 컨텍스트 제공

### 4.3 평가 방법론

**CCM Chat 평가** (150개 QA 쌍):
1. **Coverage Measurement** (LLM-as-Judge):
   - 0: 정답 미포함 / 0.5: 부분 포함 / 1: 완전 일치
   - Weighted Average = (0.5×%0.5 + 1.0×%1.0) × 100%

2. **RAGAS Scores**:
   - Context Precision: 검색된 청크의 관련성 비율
   - Faithfulness: 생성된 답변이 검색 콘텐츠에 근거한 정도
   - Answer Relevancy: 생성된 답변의 쿼리 직접성

**CCM Code Proposal 평가** (200개 레거시 코드):
1. **Pairwise Comparison**: Dense vs Graph 중 더 정확한 것 선택
2. **Dimensional Scoring** (5가지 기준):
   - Syntax Correctness, Logical Correctness, S/4HANA Compatibility
   - Optimization and Efficiency, Readability

## 5. 실험 결과

### 5.1 CCM Chat 결과

**RAGAS 기반 평가**:

| Method | Context Precision | Faithfulness | Answer Relevancy | Avg. |
|--------|------------------|--------------|------------------|------|
| Dense Vector (ada-002) | 54.35% | 77.18% | 82.92% | 71.48% |
| GraphRAG (GPT-4o) | 63.82% | 74.24% | 89.43% | 75.83% |
| GraphRAG (Dependency) | 61.07% | 72.76% | 90.97% | 74.93% |

**LLM-as-Judge 기반 평가**:

| Method | No Cov. (0) | Partial Cov. (0.5) | Full Cov. (1) | Weighted Avg. |
|--------|-------------|-------------------|---------------|---------------|
| Dense Vector | 40.29% | 15.85% | 42.88% | 50.80% |
| GraphRAG (GPT-4o) | 27.34% | 13.67% | 58.99% | 65.83% |
| GraphRAG (Dependency) | 27.34% | 21.58% | 51.08% | 61.87% |

**주요 성과**:
- Context Precision: 최소 12% 향상
- No Coverage: 32% 감소 (40.29% → 27.34%)
- Full Coverage: 최소 19% 증가 (42.88% → 51.08%)
- 의존성 파싱 방식이 GPT-4o 방식의 94% 성능 달성 (비용은 대폭 절감)

### 5.2 CCM Code Proposal 결과

**GPT-4o 기반 그래프**:
- Winning Rate: 77% (vs Dense 23%)
- Avg. Score: 4.04 (vs Dense 3.48)

**Dependency 기반 그래프**:
- Winning Rate: 78.5% (vs Dense 21.5%)
- Avg. Score: 4.03 (vs Dense 3.43)

→ 의존성 파싱 방식이 LLM 기반과 동등한 성능

### 5.3 비용 분석

**GPT-4o 기반 KG 생성 비용** (800,000 API 호출):
- 순차 처리: ~65.7일
- 2개 워커 병렬: ~33일
→ 대규모 데이터셋에서 실용성 제한

**의존성 파싱 기반**:
- LLM API 호출 불필요
- SpaCy 처리만으로 완료 (수시간 내)
- 비용 절감 효과 극대화

### 5.4 정성적 분석

**강점 사례**:
- 질문: "How do I handle custom code that references VBBS after S/4HANA conversion?"
- Dense RAG: VBBS 관련 내용 검색 실패
- GraphRAG: "If the VBBS is used in customer code... The solution is to create a view on VBBE..." 정확히 추출

**에러 분석**:
- Dense Vector: 환각(hallucination) 및 잘못된 함수 정의 빈번
  - 예: "call function 'sd_vbuk_read_from_doc_multi'" (존재하지 않는 함수)
- GraphRAG: 그래프 기반 검색으로 환각 현상 감소

## 6. 우리 연구와의 관련성

### 6.1 직접적 관련성

1. **On-premise 환경 최적화**:
   - 우리 연구: RTX 3090Ti 24GB 단일 GPU 환경
   - 본 논문: LLM 없는 의존성 파싱 방식 → 온프레미스 배포에 적합
   - **인용 포인트**: 제한된 리소스에서 GraphRAG 구축 가능성 입증

2. **한국어 행정문서 처리**:
   - 본 논문: 도메인 독립적(domain-agnostic) 의존성 파싱 접근
   - SpaCy는 한국어 모델 지원 (`ko_core_news_sm`, `ko_core_news_lg`)
   - **인용 포인트**: 한국어에도 적용 가능한 방법론

3. **비용 효율성**:
   - 우리 연구: 오픈소스 LLM + RAG 시스템
   - 본 논문: LLM API 비용 절감 방법론 제시
   - **인용 포인트**: 상용 API 없이도 GraphRAG 구현 가능

### 6.2 방법론적 시사점

1. **Hybrid Vector + Graph 접근**:
   - 본 논문: Vector similarity + 1-hop graph traversal
   - 우리 KG Cypher RAG fix와 일치하는 설계 원칙
   - **인용 포인트**: "Hybrid approach가 최적 성능" 검증

2. **Cascaded Retrieval 아키텍처**:
   - Recall-oriented stage (1-hop traversal) → Precision-oriented stage (re-ranking)
   - AutoRAG의 passage_reranker와 유사한 구조
   - **인용 포인트**: 다단계 검색의 효과성

3. **평가 방법론**:
   - LLM-as-Judge + RAGAS 조합
   - Coverage Measurement (0/0.5/1 점수)
   - **인용 포인트**: 행정문서 QA 평가 프레임워크 설계 참고

### 6.3 실험 설계 참고사항

1. **데이터셋 구성**:
   - CCM: 550 PDF → 2000 청크 (3000자, 500자 오버랩)
   - 우리 연구: AI Hub 행정문서 (청킹 전략 최적화 필요)

2. **그래프 규모**:
   - 39K 노드, 111K 엣지 (평균 차수 1.52)
   - 우리 연구: 행정문서에서 예상되는 그래프 규모 추정 가능

3. **파라미터 튜닝**:
   - `random_k_relations`: 중소형 100, 대형 200
   - top-k chunks, top-k*2 relations
   - Reciprocal Rank Fusion for hybrid search

## 7. 인용 가능한 핵심 문장

### 7.1 GraphRAG의 필요성

> "While RAG performs well for straightforward fact-based queries, it often fails to deliver coherent results for more complex tasks that require reasoning across multiple documents."

(RAG는 단순한 사실 기반 쿼리에는 잘 작동하지만, 여러 문서에 걸친 추론이 필요한 복잡한 작업에서는 일관된 결과를 제공하지 못하는 경우가 많다.)

> "Traditional RAG systems are ill-suited for this kind of multi-hop, relational reasoning. Graph-based retrieval provides a natural fit for these scenarios, as it captures structured dependencies and enables traversal-based querying across linked entities."

(전통적인 RAG 시스템은 이러한 다중 홉, 관계형 추론에 적합하지 않다. 그래프 기반 검색은 구조화된 의존성을 포착하고 연결된 엔티티 간 탐색 기반 쿼리를 가능하게 하여 이러한 시나리오에 자연스럽게 적합하다.)

### 7.2 LLM 없는 KG 생성의 효과성

> "We present a dependency-based knowledge graph construction pipeline using industrial-grade NLP libraries, eliminating reliance on LLMs and reducing the cost barrier for scalable deployment."

(우리는 산업용 NLP 라이브러리를 사용하는 의존성 기반 지식 그래프 생성 파이프라인을 제시하여, LLM 의존성을 제거하고 확장 가능한 배포의 비용 장벽을 낮춘다.)

> "Our dependency-based construction approach attains 94% of the performance of LLM-generated knowledge graphs (61.87% vs. 65.83%) while significantly reducing cost and improving scalability."

(우리의 의존성 기반 생성 방식은 LLM 생성 지식 그래프 성능의 94%를 달성하면서도(61.87% vs. 65.83%) 비용을 크게 줄이고 확장성을 개선한다.)

### 7.3 도메인 독립성

> "One particularly interesting property of this approach is that it is domain agnostic, meaning it can be applied across a wide range of domains without requiring domain-specific training or customization, making it highly adaptable for diverse text."

(이 접근법의 특히 흥미로운 속성은 도메인 독립적이라는 것으로, 도메인별 학습이나 커스터마이징 없이 광범위한 도메인에 적용할 수 있어 다양한 텍스트에 높은 적응성을 가진다.)

### 7.4 검색 전략

> "Our retrieval approach aligns with the classical cascaded architecture in information retrieval (IR), where an initial recall-oriented stage (e.g., BM25 or dense vector search) is followed by a precision-oriented neural re-ranker."

(우리의 검색 접근법은 정보 검색의 고전적 캐스케이드 아키텍처와 일치하며, 초기 재현율 지향 단계(예: BM25 또는 dense vector search) 다음에 정밀도 지향 신경망 재순위화가 이어진다.)

> "Drawing from small-world connectivity theory, our one-hop traversal effectively retrieves semantically related nodes while keeping the candidate set size tractable—crucial for scaling to large enterprise graphs."

(작은 세계 연결성 이론을 바탕으로, 우리의 1-hop 탐색은 후보 집합 크기를 관리 가능하게 유지하면서 의미적으로 관련된 노드를 효과적으로 검색한다—대규모 기업 그래프로 확장하는 데 중요하다.)

### 7.5 성능 향상

> "Our system achieves up to 15% and 4.35% improvements over traditional RAG baselines based on LLM-as-Judge and RAGAS metrics, respectively."

(우리 시스템은 LLM-as-Judge 및 RAGAS 메트릭을 기준으로 전통적인 RAG 베이스라인 대비 각각 최대 15% 및 4.35%의 성능 향상을 달성한다.)

> "In terms of coverage measurement, the No Coverage rate is reduced by 32% for both variants, while the Full Coverage rate increases by at least 19%."

(커버리지 측정 측면에서, 두 변형 모두 무커버리지 비율이 32% 감소하고 완전 커버리지 비율은 최소 19% 증가한다.)

### 7.6 실용성 및 기업 배포

> "These results validate the feasibility of deploying GraphRAG systems in real-world, large-scale enterprise applications without incurring prohibitive resource requirements paving the way for practical, explainable, and domain-adaptable retrieval-augmented reasoning."

(이러한 결과는 과도한 리소스 요구 없이 실제 대규모 기업 애플리케이션에 GraphRAG 시스템을 배포할 수 있는 타당성을 검증하며, 실용적이고 설명 가능하며 도메인 적응 가능한 검색 증강 추론의 길을 연다.)

## 8. 한계점 및 향후 연구방향

### 8.1 저자들이 인정한 한계점

1. **의존성 파싱의 한계**:
   - 표면 구문에 직접 표현되지 않은 맥락 의존적 또는 암시적 관계를 놓칠 수 있음
   - 복잡한 문장 구조나 은유적 표현 처리 어려움

2. **일반화 가능성**:
   - SAP 특화 도메인에서 강력한 성능 입증
   - HotpotQA 등 공개 벤치마크에서 검증 필요
   - 다양한 기업 환경에서의 적용 가능성 추가 검증 요구

3. **그래프 업데이트 메커니즘**:
   - 동적 콘텐츠의 증분 업데이트(incremental update) 방법 미흡
   - 대규모 그래프의 실시간 유지보수 전략 필요

### 8.2 향후 연구방향 (저자 제안)

1. **공개 벤치마크 평가**:
   - HotpotQA, Natural Questions 등에서 성능 검증
   - SAP 도메인을 넘어선 일반화 가능성 확인

2. **고급 그래프 알고리즘 통합**:
   - 최적화된 Personalized PageRank (PPR) 개발 중
   - Community detection (Leiden 알고리즘) 통합 고려

3. **RDF 트리플 지원**:
   - 현재 Property Graph만 지원
   - RDF 포맷 변환 모듈 추가 예정

### 8.3 우리 연구에서 보완 가능한 부분

1. **한국어 의존성 파싱 최적화**:
   - SpaCy 한국어 모델 vs 맞춤형 한국어 파서 비교
   - 행정문서 특화 용어 및 문장 구조 처리

2. **온프레미스 환경 최적화**:
   - 단일 GPU 환경에서 그래프 구축 및 검색 최적화
   - Milvus 대신 경량 벡터 DB 고려 (FAISS, Qdrant)

3. **한국어 평가 프레임워크**:
   - LLM-as-Judge를 한국어 LLM으로 대체
   - 행정문서 특화 평가 메트릭 설계

4. **하이브리드 접근법 심화**:
   - BM25 (lexical) + Vector (semantic) + Graph (structural) 3-way fusion
   - 한국어 형태소 특성 반영한 BM25 최적화

## 9. 참고문헌 (우리 연구에 유용한 레퍼런스)

본 논문이 인용한 문헌 중 우리 연구에 특히 유용한 참고문헌:

1. **GraphRAG 기초**:
   - [20] Han et al. (2024): Retrieval-augmented generation with graphs (graphrag)
   - [12] Edge et al. (2024): From local to global: A graph rag approach

2. **경량 GraphRAG**:
   - [18, 19] Guo et al. (2024/2025): LightRAG
   - [1] Abane et al. (2024): FastRAG
   - [14] Fan et al. (2025): MiniRAG

3. **평가 프레임워크**:
   - [13] Es et al. (2025): RAGAS
   - [34] Packowski et al. (2024): Enterprise RAG evaluation

4. **의존성 파싱**:
   - [6] Bunescu & Mooney (2005): Shortest path dependency kernel
   - [31] Ningthoujam et al. (2019): Shortest dependency path based LSTM

5. **온프레미스 LLM**:
   - [16] Gao et al. (2023): RAG for large language models survey
   - [4, 5] Barnett et al. (2024) / Bruckhaus (2024): Enterprise RAG challenges

---

**메모**: 이 논문은 우리 연구의 핵심 레퍼런스로, 특히 "LLM 없는 KG 생성", "Hybrid Vector+Graph 검색", "기업 환경 배포" 측면에서 직접적으로 인용 가능하다.
