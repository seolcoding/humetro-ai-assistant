# RAG vs. GraphRAG: A Systematic Evaluation and Key Insights

## 1. 논문 정보

- **제목**: RAG vs. GraphRAG: A Systematic Evaluation and Key Insights
- **저자**: Haoyu Han, Yu Wang, Harry Shomer, Yongjia Lei, Kai Guo, Zhigang Hua, Bo Long, Hui Liu, Jiliang Tang
- **소속**: Michigan State University, University of Oregon, Meta
- **연도**: 2025
- **출판**: arXiv:2502.11371v1 [cs.IR] (2025년 2월 17일)
- **분야**: Information Retrieval, RAG Systems, Graph-based Retrieval

## 2. 핵심 내용 요약

본 논문은 텍스트 기반 태스크에서 전통적인 RAG와 GraphRAG의 성능을 체계적으로 비교 평가한 최초의 연구이다. Question Answering과 Query-based Summarization 태스크에 대해 널리 사용되는 벤치마크 데이터셋(NQ, HotPotQA, MultiHop-RAG, NovelQA, SQuALITY, QMSum 등)으로 평가한 결과, RAG와 GraphRAG는 상호 보완적 강점을 가지는 것으로 나타났다. RAG는 단일 홉 질문과 세부 정보 요구 태스크에서 우수하며, GraphRAG(특히 Community-based Local Search)는 멀티홉 추론 질문에서 뛰어난 성능을 보인다. 이러한 통찰을 바탕으로 Selection과 Integration 전략을 제안하여 전체 성능을 향상시켰다.

## 3. 주요 기여점

### 3.1 최초의 체계적 비교 연구
- RAG와 GraphRAG를 텍스트 기반 태스크에서 동일한 조건으로 체계적으로 평가한 최초의 연구
- 널리 사용되는 벤치마크 데이터셋과 표준 평가 메트릭 활용

### 3.2 태스크별 강점 분석
- **RAG 강점**: 단일 홉 질문(NQ), 세부 정보 요구 질문(NovelQA의 detail-oriented queries)
- **GraphRAG 강점**: 멀티홉 추론 질문(HotPotQA, MultiHop-RAG), 비교/시간 순서 질문

### 3.3 하이브리드 전략 제안
- **Selection 전략**: 질문을 fact-based/reasoning-based로 분류하여 RAG/GraphRAG 선택
- **Integration 전략**: 두 방법의 검색 결과를 결합하여 생성 (최대 6.4% 성능 향상)

### 3.4 평가 방법론의 한계 발견
- LLM-as-a-Judge 평가에서 position bias 존재 확인
- ROUGE/BERTScore 등 ground truth 기반 평가의 중요성 강조

## 4. 방법론

### 4.1 평가 대상 시스템

#### 4.1.1 RAG (Baseline)
- **청킹**: 256 토큰 단위
- **임베딩**: OpenAI text-embedding-ada-002
- **검색**: Top-10 semantic similarity
- **생성 모델**: Llama-3.1-8B-Instruct, Llama-3.1-70B-Instruct

#### 4.1.2 KG-based GraphRAG
- **그래프 구축**: LLM 기반 triplet extraction (entity-relation-entity)
- **검색 방법**: 쿼리 엔티티 매칭 → 멀티홉 이웃 탐색
- **두 가지 변형**:
  - KG-GraphRAG (Triplets): triplet만 검색
  - KG-GraphRAG (Triplets+Text): triplet + 원본 텍스트 검색
- **구현**: LlamaIndex

#### 4.1.3 Community-based GraphRAG
- **그래프 구축**: LLM 기반 KG + 계층적 커뮤니티 탐지
- **커뮤니티 요약**: 각 계층별 커뮤니티 리포트 생성
- **두 가지 검색 방법**:
  - **Local Search**: 엔티티/관계/저레벨 커뮤니티 리포트 검색
  - **Global Search**: 고레벨 커뮤니티 요약만 검색
- **구현**: Microsoft GraphRAG

### 4.2 평가 태스크 및 데이터셋

#### 4.2.1 Question Answering
- **NQ** (1,000개): 단일 홉 QA, single-document
- **HotPotQA** (1,000개): 멀티홉 QA (hard bridging), multi-document
- **MultiHop-RAG** (2,556개): 4가지 쿼리 타입 (Inference, Comparison, Temporal, Null)
- **NovelQA**: 21가지 쿼리 타입 (single-hop, multi-hop, detail-oriented 등)
- **평가 메트릭**: Precision, Recall, F1-score (NQ, HotPotQA), Accuracy (MultiHop-RAG, NovelQA)

#### 4.2.2 Query-based Summarization
- **SQuALITY**: 단일 문서 요약 (4,000-6,000단어 단편 소설)
- **QMSum**: 단일 문서 요약 (회의록)
- **ODSum-story, ODSum-meeting**: 멀티 문서 요약
- **평가 메트릭**: ROUGE-2, BERTScore

## 5. 실험 결과

### 5.1 Question Answering 주요 결과

#### 5.1.1 단일 홉 QA (NQ Dataset, Llama-3.1-70B)
| 방법 | Precision | Recall | F1 |
|------|-----------|--------|-----|
| RAG | 74.55 | 67.82 | **68.18** |
| Community-GraphRAG (Local) | 71.27 | 65.46 | 65.44 |
| KG-GraphRAG (Triplets+Text) | 60.91 | 52.75 | 53.88 |

**결과**: RAG가 단일 홉 질문에서 가장 우수한 성능

#### 5.1.2 멀티홉 QA (HotPotQA, Llama-3.1-70B)
| 방법 | Precision | Recall | F1 |
|------|-----------|--------|-----|
| RAG | 66.34 | 63.99 | 63.88 |
| Community-GraphRAG (Local) | 67.20 | 64.89 | **64.60** |
| KG-GraphRAG (Triplets+Text) | 51.44 | 48.99 | 48.75 |

**결과**: Community-GraphRAG (Local)이 멀티홉 추론에서 우수

#### 5.1.3 MultiHop-RAG Dataset (Llama-3.1-70B)
| 방법 | Inference | Comparison | Null | Temporal | Overall |
|------|-----------|------------|------|----------|---------|
| RAG | 94.85 | 56.31 | 91.36 | 25.73 | 65.77 |
| Community-GraphRAG (Local) | 92.03 | **60.16** | 88.70 | **49.06** | **71.17** |
| Community-GraphRAG (Global) | 89.09 | **66.00** | 13.95 | **59.18** | 65.69 |

**핵심 인사이트**:
- Community-GraphRAG (Global)이 Comparison/Temporal 쿼리에서 강점
- Null 쿼리에서 Community-GraphRAG (Global)이 환각 문제 (13.95%)

#### 5.1.4 상호 보완성 분석 (Confusion Matrix)
- **MultiHop-RAG**: GraphRAG-only 13.6%, RAG-only 11.6%
- **NovelQA**: GraphRAG-only 13.7%, RAG-only 17.1%
- **HotPotQA**: GraphRAG-only 9.8%, RAG-only 9.2%

→ 두 방법이 서로 다른 질문에서 성공하는 상호 보완적 특성 확인

#### 5.1.5 하이브리드 전략 성능 향상
- **Selection 전략**: MultiHop-RAG에서 1.1% 향상 (71.17% → 72.27%)
- **Integration 전략**: MultiHop-RAG에서 6.4% 향상 (71.17% → 77.57%)

### 5.2 Query-based Summarization 결과

#### 5.2.1 단일 문서 요약 (SQuALITY, Llama-3.1-8B)
| 방법 | ROUGE-2 F1 | BERTScore F1 |
|------|-------------|--------------|
| RAG | **10.08** | 77.62 |
| Integration | **10.67** | **77.73** |
| Community-GraphRAG (Local) | 10.10 | 84.66 |

#### 5.2.2 멀티 문서 요약 (ODSum-story, Llama-3.1-8B)
| 방법 | ROUGE-2 F1 | BERTScore F1 |
|------|-------------|--------------|
| RAG | **9.81** | **84.57** |
| Integration | 9.53 | 84.40 |
| Community-GraphRAG (Local) | 8.49 | 83.90 |

**결과**: RAG가 요약 태스크에서 일관되게 우수한 성능 (세부 정보 보존 능력)

### 5.3 LLM-as-a-Judge 평가의 Position Bias

**실험 설계**:
- Order 1: RAG 먼저 제시
- Order 2: GraphRAG 먼저 제시
- 평가 기준: Comprehensiveness (세부성), Diversity (다양성)

**주요 발견**:
- **Position bias 존재**: 순서에 따라 평가 결과가 완전히 반대로 나타남
- **일관된 경향**: RAG는 Comprehensiveness에서, GraphRAG (Global)은 Diversity에서 우수
- **결론**: LLM-as-a-Judge는 position bias로 인해 신뢰성 낮음, ROUGE/BERTScore 등 ground truth 기반 평가 필요

### 5.4 KG-based GraphRAG의 한계

**엔티티 커버리지 분석**:
- HotPotQA: 정답 엔티티의 65.8%만 KG에 존재
- NQ: 정답 엔티티의 65.5%만 KG에 존재

→ LLM 기반 triplet extraction의 불완전성이 성능 저하의 주요 원인

## 6. 우리 연구와의 관련성

### 6.1 한국어 공공행정문서 RAG 시스템 설계에 대한 시사점

#### 6.1.1 태스크 특성 분석 필요
- **우리 연구 적용**: 행정문서 질의는 fact-based가 많지만, 법령 해석/정책 비교는 reasoning-based
- **전략**: 질문 유형 분석 후 RAG/GraphRAG 선택적 적용 고려

#### 6.1.2 멀티홉 추론 필요성
- **행정문서 특성**: 법령-시행령-시행규칙 간 참조 관계, 부서 간 연관 정책
- **GraphRAG 장점**: 문서 간 관계를 그래프로 표현하면 멀티홉 추론에 유리
- **우리 실험과 일치**: KG Cypher RAG에서 vector similarity + graph expansion 조합이 효과적

#### 6.1.3 하이브리드 접근의 중요성
- **본 논문**: Integration 전략이 최대 6.4% 성능 향상
- **우리 checkpoint 결과**: Fixed KG Cypher (vector+graph hybrid)가 0.830 faithfulness 달성
- **적용 방안**: RAG와 GraphRAG를 병행하여 검색 결과 통합

#### 6.1.4 한국어 특화 고려사항
- **엔티티 추출**: 본 논문은 영어 LLM 기반 triplet extraction → 한국어는 NER 성능 제약
- **해결책**: 한국어 특화 NER 모델 또는 사전 기반 엔티티 추출 병행
- **우리 연구**: EXAONE-3.5-7.8B 등 한국어 강화 모델 활용 가능성

### 6.2 평가 방법론 개선

#### 6.2.1 Ground Truth 기반 평가 중요성
- **본 논문**: LLM-as-a-Judge의 position bias 발견
- **우리 연구**: RAGAS (Faithfulness, Answer Relevancy, Answer Correctness) 사용 중
- **개선 방향**: ROUGE/BERTScore 등 객관적 메트릭 병행

#### 6.2.2 평가 데이터셋 구축
- **본 논문**: MultiHop-RAG, NovelQA 등 다양한 쿼리 타입 벤치마크 활용
- **우리 연구**: AI Hub 행정문서 기계독해 데이터 (50 questions)
- **확장 필요**: 단일 홉/멀티홉, fact-based/reasoning-based 쿼리 분리 평가

### 6.3 그래프 구축 전략

#### 6.3.1 KG 불완전성 문제
- **본 논문**: 엔티티 커버리지 65% 수준 → KG-GraphRAG 성능 저하
- **우리 연구에 적용**:
  - 행정문서의 명확한 구조 활용 (조항, 항, 호 등)
  - 법령-시행령 간 명시적 참조 관계 활용
  - LLM 기반 추출과 규칙 기반 추출 병행

#### 6.3.2 Community 계층 구조
- **본 논문**: 계층적 커뮤니티 + 리포트 생성
- **우리 연구 아이디어**:
  - 법령 체계 (법률 → 시행령 → 시행규칙)를 자연스러운 계층으로 활용
  - 부서별/정책 주제별 커뮤니티 구성

## 7. 인용 가능한 핵심 문장

### 7.1 RAG와 GraphRAG의 상호 보완성

> "Our findings reveal that RAG and GraphRAG are complementary, each excelling in different aspects. For the Question Answering task, we observe that RAG performs better on single-hop questions and those requiring detailed information, while GraphRAG is more effective for multi-hop questions."

**번역**: "우리의 연구 결과는 RAG와 GraphRAG가 상호 보완적이며, 각각 다른 측면에서 뛰어나다는 것을 보여준다. 질의응답 태스크에서 RAG는 단일 홉 질문과 세부 정보를 요구하는 질문에서 더 나은 성능을 보이는 반면, GraphRAG는 멀티홉 질문에서 더 효과적이다."

### 7.2 최초의 체계적 비교 연구

> "This is the very first work to systematically evaluate and compare RAG and GraphRAG on text-based tasks using widely adopted datasets and evaluations."

**번역**: "이는 널리 사용되는 데이터셋과 평가 방법을 활용하여 텍스트 기반 태스크에서 RAG와 GraphRAG를 체계적으로 평가하고 비교한 최초의 연구이다."

### 7.3 GraphRAG의 적용 조건

> "What are the advantages and disadvantages of applying GraphRAG to general text-based tasks compared to RAG?"

**번역**: "RAG와 비교하여 일반적인 텍스트 기반 태스크에 GraphRAG를 적용할 때의 장점과 단점은 무엇인가?"

### 7.4 하이브리드 전략의 효과

> "Both strategies generally enhance overall performance. For example, on the MultiHop-RAG dataset with Llama 3.1-70B, Selection and Integration improve the best method by 1.1% and 6.4%, respectively."

**번역**: "두 전략 모두 전반적인 성능을 향상시킨다. 예를 들어, Llama 3.1-70B를 사용한 MultiHop-RAG 데이터셋에서 Selection과 Integration 전략은 최고 성능 방법 대비 각각 1.1%와 6.4%의 향상을 보였다."

### 7.5 LLM-as-a-Judge의 한계

> "Position bias (Shi et al., 2024; Wang et al., 2024) is evident in the LLM-as-a-Judge evaluations for summarization task, as changing the order of the two methods significantly affects the predictions."

**번역**: "요약 태스크에 대한 LLM-as-a-Judge 평가에서 position bias가 명확히 나타나며, 두 방법의 순서를 변경하면 예측 결과에 상당한 영향을 미친다."

### 7.6 KG 불완전성 문제

> "However, the extracted entities and relations may be incomplete, leading to gaps in the retrieved information. To verify this, we calculated the ratio of answer entities present in the constructed KG. We found that only around 65.8% of answer entities exist in the constructed KG for the Hotpot dataset and 65.5% for the NQ dataset."

**번역**: "그러나 추출된 엔티티와 관계는 불완전할 수 있으며, 이는 검색된 정보의 공백을 초래한다. 이를 검증하기 위해 구축된 KG에 존재하는 정답 엔티티의 비율을 계산한 결과, Hotpot 데이터셋의 경우 약 65.8%, NQ 데이터셋의 경우 65.5%의 정답 엔티티만 KG에 존재했다."

### 7.7 Community-GraphRAG의 특성

> "Community-based GraphRAG with Global Search focuses more on the global aspects of whole corpus, whereas RAG captures more detailed information."

**번역**: "Global Search를 사용하는 Community-based GraphRAG는 전체 코퍼스의 전역적 측면에 더 집중하는 반면, RAG는 더 세부적인 정보를 포착한다."

### 7.8 향후 연구 방향

> "Future work can explore improving GraphRAG through better graph construction or developing novel approaches to combine RAG and GraphRAG methods for both effectiveness and efficiency."

**번역**: "향후 연구는 더 나은 그래프 구축을 통해 GraphRAG를 개선하거나, 효과성과 효율성을 모두 고려하여 RAG와 GraphRAG 방법을 결합하는 새로운 접근법을 개발하는 방향으로 진행될 수 있다."

## 8. 한계점 및 향후 연구방향

### 8.1 논문의 한계점

#### 8.1.1 제한된 태스크 범위
- QA와 Query-based Summarization에만 초점
- 다른 태스크(대화 생성, 문서 분류 등)로 확장 필요

#### 8.1.2 LLM 기반 그래프 구축 의존성
- 모든 GraphRAG 방법이 LLM 기반 엔티티/관계 추출에 의존
- 다른 그래프 구축 모델(NER 전문 모델 등) 비교 필요

#### 8.1.3 제한된 생성 모델
- Llama 3.1-8B/70B만 평가
- 다양한 크기와 종류의 LLM으로 확장 필요

#### 8.1.4 영어 중심 평가
- 모든 데이터셋이 영어
- 다국어, 특히 한국어 등 비영어권 언어에 대한 평가 부재

### 8.2 향후 연구방향

#### 8.2.1 그래프 구축 개선
- 더 정확한 엔티티/관계 추출 모델
- 도메인 특화 그래프 구축 전략
- 규칙 기반과 학습 기반 방법의 하이브리드

#### 8.2.2 효율성 최적화
- Integration 전략의 계산 비용 감소
- 동적 그래프 업데이트 메커니즘
- 캐싱 및 인덱싱 최적화

#### 8.2.3 평가 방법론 발전
- Position bias 없는 LLM-based 평가 방법
- 태스크별 특화 평가 메트릭
- 인간 평가와의 상관관계 분석

#### 8.2.4 실무 적용 연구
- 비용-효과 분석 (RAG vs. GraphRAG)
- 온프레미스 환경에서의 배포 전략
- 도메인 특화 시스템 구축 가이드라인

### 8.3 우리 연구에 적용 가능한 향후 방향

#### 8.3.1 한국어 공공행정문서 특화
- 법령 구조를 활용한 계층적 그래프 구축
- 한국어 NER 모델과 LLM 기반 추출의 앙상블
- 행정용어 사전 기반 엔티티 정규화

#### 8.3.2 태스크별 전략 수립
- 단순 사실 조회: RAG
- 법령 해석/비교: GraphRAG (Local)
- 정책 전반 요약: GraphRAG (Global)
- 복합 질의: Integration 전략

#### 8.3.3 평가 체계 확립
- AI Hub 데이터 기반 벤치마크 확장 (현재 50 → 200+ questions)
- 단일 홉/멀티홉 쿼리 분리 평가
- RAGAS + ROUGE/BERTScore 병행 평가
- 실무자 평가 병행 (법무 담당자 등)

## 9. 참고문헌 (주요 인용)

- Edge et al. (2024): From Local to Global: A Graph RAG Approach to Query-Focused Summarization
- Tang and Yang (2024): MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-hop Queries
- Kwiatkowski et al. (2019): Natural Questions: A Benchmark for Question Answering Research
- Yang et al. (2018): HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering

## 10. 핵심 테이블 요약

### 표 1: Question Answering 성능 비교 (Llama-3.1-70B)

| 태스크 | 최고 성능 방법 | F1/Accuracy | 특징 |
|--------|---------------|-------------|------|
| NQ (단일 홉) | RAG | 68.18 | 세부 정보 요구 |
| HotPotQA (멀티홉) | Community-GraphRAG (Local) | 64.60 | 추론 체인 필요 |
| MultiHop-RAG | Community-GraphRAG (Local) | 71.17 | 4가지 쿼리 타입 |
| MultiHop-RAG (Integration) | Integration | **77.57** | 6.4% 향상 |

### 표 2: 상호 보완성 분석

| 데이터셋 | 둘 다 정답 | GraphRAG만 정답 | RAG만 정답 | 둘 다 오답 |
|----------|-----------|----------------|-----------|-----------|
| MultiHop-RAG | 55.4% | 13.6% | 11.6% | 19.4% |
| NovelQA | 40.0% | 13.7% | 17.1% | 29.1% |
| HotPotQA | 45.4% | 9.8% | 9.2% | 35.6% |

---

**메타데이터**:
- 작성일: 2025-11-30
- 논문 읽기 완료: ✅
- 우리 연구와 관련성: ⭐⭐⭐⭐⭐ (5/5)
- 인용 우선순위: 🔴 HIGH (Introduction, Related Works, Methodology 섹션)
