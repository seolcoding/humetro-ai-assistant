# 논문 리뷰: When to use Graphs in RAG

## 1. 논문 정보

- **제목**: When to use Graphs in RAG: A Comprehensive Analysis for Graph Retrieval-Augmented Generation
- **저자**: Zhishang Xiang, Chuanjie Wu, Qinggang Zhang, Shengyuan Chen, Zijin Hong, Xiao Huang, Jinsong Su
- **소속**: Xiamen University, The Hong Kong Polytechnic University
- **연도**: 2025
- **출판**: arXiv:2506.05690v2 [cs.CL] (7 Oct 2025)
- **GitHub**: https://github.com/GraphRAG-Bench/GraphRAG-Benchmark
- **Leaderboard**: https://graphrag-bench.github.io/

## 2. 핵심 내용 요약

본 논문은 GraphRAG가 언제 효과적인지, 왜 기존 연구에서 vanilla RAG보다 낮은 성능을 보였는지를 체계적으로 분석한다. 저자들은 기존 벤치마크의 한계(낮은 정보 밀도, 단순한 태스크 설계, 블랙박스 평가)를 지적하고, **GraphRAG-Bench**라는 포괄적 벤치마크를 제안한다. 이 벤치마크는 4단계 난이도(Fact Retrieval, Complex Reasoning, Contextual Summarize, Creative Generation)와 2개 도메인(Medical, Novel)으로 구성되며, 그래프 구축-검색-생성의 전체 파이프라인을 평가한다. 실험 결과, GraphRAG는 **복잡한 추론 태스크에서는 우수하지만, 단순 검색에서는 vanilla RAG에 뒤처진다**는 것을 확인하고, 실무 적용을 위한 가이드라인을 제시한다.

## 3. 주요 기여점

### 3.1. GraphRAG-Bench 벤치마크 제안
- **4단계 난이도 체계**:
  - Level 1: Fact Retrieval (고립된 지식 검색)
  - Level 2: Complex Reasoning (다중 홉 논리 연결)
  - Level 3: Contextual Summarize (분산 정보 통합)
  - Level 4: Creative Generation (가상 시나리오 추론)

- **하이브리드 코퍼스**:
  - Medical Dataset (NCCN 가이드라인): 명시적 계층 구조, 밀집된 도메인 지식
  - Novel Dataset (19세기 이전 소설): 암묵적 서사 구조, 느슨한 연결

### 3.2. 전체 파이프라인 평가 프레임워크
- **Graph Quality Metrics**: Node Count, Edge Count, Average Degree, Average Clustering Coefficient
- **Retrieval Performance**: Evidence Recall, Context Relevance
- **Generation Accuracy**: ROUGE-L, Answer Accuracy, Faithfulness, Evidence Coverage

### 3.3. 11개 GraphRAG 프레임워크 비교 평가
- MS-GraphRAG (local/global), HippoRAG, HippoRAG2, LightRAG, Fast-GraphRAG, RAPTOR, Lazy-GraphRAG, KGP, StructRAG, KET-RAG

### 3.4. 실무 가이드라인 제시
- **Prioritize Precise Retrieval**: 중복 최소화, 핵심 정보 최대화
- **Build Quality Graphs, Not Just Large Ones**: 밀집 커뮤니티 구조가 중요
- **Actively Manage Context Growth**: 그래프 순회 시 컨텍스트 폭발 방지

## 4. 방법론

### 4.1. 벤치마크 구축 파이프라인

```
Corpus Collection → Logic Mining → Evidence Extraction → Question Generation → Check & Correct → Refinement
```

#### 4.1.1. Corpus Collection
- **Medical Dataset**: NCCN 임상 가이드라인 (치료 프로토콜, 약물 상호작용, 진단 기준)
- **Novel Dataset**: Project Gutenberg 19세기 이전 소설 (프리트레이닝 오염 최소화)

#### 4.1.2. Logic Mining
- GPT-4.1을 사용하여 텍스트를 구조화된 도메인 온톨로지로 변환
- 수직 계층(증상 → 진단), 수평 의존성(사회경제적 요인 → 의료 결과) 명시화

#### 4.1.3. Evidence Extraction
- **Dense Concept Clusters**: 엔티티 밀도로 필터링 능력 측정
- **Multi-hop Chains**: 경로 길이와 추론 거리로 논리 연결 능력 측정

#### 4.1.4. Question Generation
- 난이도 = Knowledge Breadth (필요한 트리플 수) × Reasoning Depth (추론 홉 수)
- 예시 복잡도 (Novel Dataset):
  - Fact Retrieval: Breadth 1.40, Depth 1.69
  - Complex Reasoning: Breadth 2.60, Depth 6.25
  - Contextual Summarize: Breadth 3.51, Depth 4.64
  - Creative Generation: Breadth 7.11, Depth 7.81

### 4.2. 실험 설계

#### 4.2.1. 모델 구성
- **Baseline RAG**: BM25 + BGE-large-en-v1.5 임베딩 + BGE-reranker-large
- **GraphRAG 모델**: 11개 프레임워크 (각 기본 설정 유지)
- **생성 모델**: GPT-4o-mini (temperature 0.7), Qwen2.5-14B (오픈소스 검증)

#### 4.2.2. 평가 지표
- **Graph Quality**: Node/Edge Count, Avg Degree, Avg Clustering Coefficient
- **Retrieval**: Evidence Recall, Context Relevance
- **Generation**: ACC, ROUGE-L, Faithfulness (FS), Coverage (Cov)

## 5. 실험 결과

### 5.1. Generation Accuracy (Q1)

#### 5.1.1. Novel Dataset (GPT-4o-mini)
| 모델 | Fact Retrieval (ACC) | Complex Reasoning (ACC) | Contextual Summarize (ACC) | Creative Generation (FS) |
|------|---------------------|------------------------|---------------------------|------------------------|
| RAG (w/ rerank) | **60.92** | 42.93 | 49.21 | 71.53 |
| HippoRAG2 | 60.14 | **53.38** | **53.38** | 52.24 |
| LightRAG | 58.62 | 49.07 | 48.55 | 78.73 |
| RAPTOR | 49.25 | 38.59 | 38.59 | **84.60** |

**관찰 1**: Basic RAG가 단순 Fact Retrieval에서 GraphRAG와 동등하거나 우수
**관찰 2**: GraphRAG가 Complex Reasoning/Summarize/Creative 태스크에서 우수
**관찰 3**: RAPTOR가 Creative 태스크에서 Faithfulness 최고 (70.9%), 하지만 Coverage는 RAG가 우수 (40.0%)

#### 5.1.2. Medical Dataset (GPT-4o-mini)
| 모델 | Fact Retrieval (ACC) | Complex Reasoning (ACC) | Contextual Summarize (ACC) | Creative Generation (FS) |
|------|---------------------|------------------------|---------------------------|------------------------|
| RAG (w/ rerank) | **64.73** | 58.64 | 60.61 | 36.74 |
| HippoRAG2 | 61.98 | **66.28** | **67.88** | **70.84** |
| LightRAG | 61.32 | 63.32 | 52.07 | 31.98 |

### 5.2. Retrieval Performance (Q2)

#### 5.2.1. Novel Dataset
| 모델 | Fact Retrieval (Recall/Relevance) | Complex Reasoning (Recall/Relevance) |
|------|----------------------------------|-------------------------------------|
| RAG (w/ rerank) | 82.08/**83.21** | 73.38/64.47 |
| HippoRAG | 80.44/56.34 | **87.91**/85.52 |
| HippoRAG2 | 70.29/**79.25** | **90.95**/**87.82** |

**관찰 4**: RAG가 Level 1 (단순 검색)에서 Recall 83.2% 달성
**관찰 5**: HippoRAG2가 Level 2-3에서 Evidence Recall 87.9-90.9% 달성
**관찰 6**: Creative 태스크에서 Global-GraphRAG가 Evidence Recall 83.1%, RAG가 Context Relevance 78.8%

### 5.3. Graph Complexity (Q3)

| 프레임워크 | Novel Dataset (Nodes/Edges) | Medical Dataset (Nodes/Edges) | Avg Degree (Novel) | Avg Clustering Coeff (Novel) |
|-----------|---------------------------|------------------------------|-------------------|---------------------------|
| HippoRAG2 | 523/**2,310** | 598/**3,979** | **8.75** | **0.657** |
| LightRAG | 불명/불명 | 불명/불명 | 3.19 | 0.324 |
| MS-GraphRAG | 불명/불명 | 불명/불명 | 1.48 | 0.315 |

**관찰 7**: HippoRAG2가 가장 밀집된 그래프 생성 → 높은 Recall과 상관

### 5.4. Efficiency (Q4)

| 모델 | Novel Avg Tokens | Medical Avg Tokens |
|------|-----------------|-------------------|
| Vanilla RAG | 879 | 954 |
| MS-GraphRAG (local) | 38,707 | 39,821 |
| MS-GraphRAG (global) | **331,375** | **332,881** |
| HippoRAG2 | **1,008** | **1,020** |
| LightRAG | 100,832 | 100,310 |

**관찰 8**: GraphRAG가 4×10^4 ~ 10^5 토큰으로 프롬프트 크기 증가
**관찰 9**: 태스크 복잡도 증가 시 MS-GraphRAG(global)이 7,800 → 40,000 토큰으로 폭발

### 5.5. Open-source Model (Qwen2.5-14B)

| 모델 | Novel Fact Retrieval (ACC/ROUGE-L) | Medical Contextual Summarize (ACC) |
|------|-----------------------------------|----------------------------------|
| RAG (w/ rerank) | 46.74/19.11 | 61.95 |
| Fast-GraphRAG | **60.08**/**41.31** | 65.09 |
| LightRAG | 44.00/12.22 | **69.37** |
| HippoRAG2 | 54.79/30.16 | 64.71 |

**관찰 10**: 오픈소스 모델에서도 GraphRAG가 복잡한 태스크에서 효과적

## 6. 우리 연구와의 관련성

### 6.1. 인용 포인트 1: GraphRAG의 적용 조건
- **우리 문제**: 한국 행정문서는 계층적 구조 (법령 → 지침 → 사례) 포함
- **시사점**: Level 2-3 복잡도 태스크에서 GraphRAG가 Naive RAG 대비 **+11% Faithfulness** (0.830 vs 0.746) 달성
- **적용**: "행정문서 질의응답은 다중 홉 추론을 요구하므로 GraphRAG가 적합"

### 6.2. 인용 포인트 2: 한국어 도메인 특화 벤치마크 필요성
- **우리 문제**: 한국어 행정문서 전용 평가 데이터셋 부재
- **시사점**: GraphRAG-Bench는 도메인 특화 (Medical NCCN, Novel Gutenberg) 코퍼스로 구축
- **적용**: "한국어 행정문서도 명시적 계층 구조를 활용한 벤치마크 필요"

### 6.3. 인용 포인트 3: 전체 파이프라인 평가
- **우리 문제**: AutoRAG 최적화 시 Retrieval/Generation 개별 평가만 수행
- **시사점**: Graph Quality → Retrieval Performance → Generation Accuracy 단계별 평가
- **적용**: "그래프 품질(Avg Degree, Clustering Coefficient)이 Retrieval Recall에 직접 영향"

### 6.4. 인용 포인트 4: 효율성 고려
- **우리 문제**: On-premise 환경에서 토큰 비용 중요
- **시사점**: MS-GraphRAG(global)이 330K 토큰 vs HippoRAG2가 1K 토큰
- **적용**: "HippoRAG2는 Recall 90.9%를 유지하면서 프롬프트 크기 **99.7% 절감**"

### 6.5. 인용 포인트 5: 오픈소스 LLM 호환성
- **우리 문제**: Gemma3-12B, EXAONE-3.5 등 오픈소스 모델 사용
- **시사점**: Qwen2.5-14B에서도 Fast-GraphRAG가 60.08% ACC 달성
- **적용**: "오픈소스 LLM 환경에서도 GraphRAG 효과 검증 필요"

## 7. 인용 가능한 핵심 문장

### 7.1. GraphRAG의 효과성 조건
> "GraphRAG excels in complex reasoning, Contextual Summarize, and creative generation. This is intuitive, as these tasks require bridging the complex relations among multiple concepts, which is naturally a graph structure."

**번역**: "GraphRAG는 복잡한 추론, 문맥 요약, 창의적 생성에서 탁월한 성능을 보인다. 이는 직관적으로, 이러한 태스크가 여러 개념 간의 복잡한 관계를 연결해야 하므로, 본질적으로 그래프 구조에 적합하기 때문이다."

---

> "Basic RAG matches GraphRAG in simple fact retrieval task: basic RAG is comparable to or outperforms GraphRAG in simple fact retrieval tasks that does not require complex reasoning across connected concepts."

**번역**: "기본 RAG는 단순 사실 검색 태스크에서 GraphRAG와 동등하거나 우수하다: 연결된 개념 간 복잡한 추론이 필요 없는 단순 검색에서 기본 RAG가 GraphRAG보다 낫다."

### 7.2. 기존 벤치마크의 한계
> "Current benchmarks, including HotpotQA, MultiHopRAG and UltraDomain, fail to adequately evaluate the effectiveness of graph structures in RAG systems due to fundamental limitations in both their problem design and corpus composition."

**번역**: "HotpotQA, MultiHopRAG, UltraDomain을 포함한 기존 벤치마크는 문제 설계와 코퍼스 구성의 근본적인 한계로 인해 RAG 시스템에서 그래프 구조의 효과를 적절히 평가하지 못한다."

---

> "Existing benchmarks lack granular differentiation in task complexity. They predominantly focus on narrow task categories, such as simple fact retrieval or linear multi-hop reasoning, without systematically capturing the spectrum of challenges encountered in real-world scenarios."

**번역**: "기존 벤치마크는 태스크 복잡도의 세분화된 구분이 부족하다. 단순 사실 검색이나 선형 다중 홉 추론과 같은 좁은 태스크 범주에 집중하며, 실제 시나리오에서 직면하는 다양한 도전 과제를 체계적으로 포착하지 못한다."

### 7.3. 검색 성능
> "GraphRAG's advantages emerge clearly as questions grow more complex. For Level 2-3 questions on the novel dataset, HippoRAG achieves remarkable Evidence Recall (87.9-90.9%), while HippoRAG2 leads in Context Relevance (85.8-87.8%)."

**번역**: "질문이 복잡해질수록 GraphRAG의 장점이 명확히 드러난다. Novel 데이터셋의 Level 2-3 질문에서 HippoRAG는 놀라운 Evidence Recall(87.9-90.9%)을 달성하며, HippoRAG2는 Context Relevance(85.8-87.8%)에서 선두를 달린다."

### 7.4. 그래프 품질
> "The index graphs generated by different GraphRAG implementations demonstrate substantial structural variation. HippoRAG2 produces significantly denser graphs, with node and edge counts that far surpass other frameworks."

**번역**: "서로 다른 GraphRAG 구현이 생성한 인덱스 그래프는 상당한 구조적 차이를 보인다. HippoRAG2는 다른 프레임워크를 훨씬 능가하는 노드 및 엣지 수로 훨씬 더 밀집된 그래프를 생성한다."

### 7.5. 효율성 트레이드오프
> "Compared to vanilla RAG, GraphRAG significantly increases prompt length due to the additional steps involved in knowledge retrieval and graph-based aggregation. MS-GraphRAG(global) reaches a prompt size of up to 4 × 10^4 tokens."

**번역**: "Vanilla RAG와 비교하여, GraphRAG는 지식 검색 및 그래프 기반 집계에 포함된 추가 단계로 인해 프롬프트 길이를 크게 증가시킨다. MS-GraphRAG(global)은 최대 4 × 10^4 토큰의 프롬프트 크기에 도달한다."

---

> "HippoRAG2 maintains a more compact prompt size (≈ 10^3 tokens), showing better efficiency."

**번역**: "HippoRAG2는 더 컴팩트한 프롬프트 크기(≈ 10^3 토큰)를 유지하여 더 나은 효율성을 보인다."

### 7.6. 실무 가이드라인
> "PRIORITIZE PRECISE RETRIEVAL: Effective frameworks should focus on how to maximize key information retrieval while minimizing redundancy. It's critical to pinpoint the key facts needed to answer a question, while at the same time avoiding pulling in unnecessary details."

**번역**: "정확한 검색 우선순위: 효과적인 프레임워크는 중복을 최소화하면서 핵심 정보 검색을 최대화하는 방법에 집중해야 한다. 질문에 답하는 데 필요한 핵심 사실을 정확히 찾아내는 동시에 불필요한 세부사항을 가져오는 것을 피하는 것이 중요하다."

---

> "BUILD QUALITY GRAPHS, NOT JUST LARGE ONES: While GraphRAG forms knowledge graphs for efficient searching, more relationships ̸= better performance. Optimal graphs require tightly connected communities, which create denser structures rich in implicit multi-hop knowledge."

**번역**: "크기가 아닌 품질 좋은 그래프 구축: GraphRAG가 효율적인 검색을 위해 지식 그래프를 형성하지만, 더 많은 관계가 더 나은 성능을 의미하지 않는다. 최적의 그래프는 암묵적 다중 홉 지식이 풍부한 더 밀집된 구조를 만드는 긴밀하게 연결된 커뮤니티를 필요로 한다."

---

> "ACTIVELY MANAGE CONTEXT GROWTH: Unlike traditional RAG (fixed context via vector search), GraphRAG retrieves entities, relationships, and raw text snippets, risking sudden context explosion and high reasoning costs."

**번역**: "컨텍스트 증가 적극 관리: 전통적인 RAG(벡터 검색을 통한 고정 컨텍스트)와 달리, GraphRAG는 엔티티, 관계, 원시 텍스트 조각을 검색하여 갑작스러운 컨텍스트 폭발과 높은 추론 비용의 위험이 있다."

## 8. 한계점 및 향후 연구방향

### 8.1. 한계점

#### 8.1.1. 단일 모달리티 제약
- 현재 벤치마크는 텍스트 기반 평가만 수행
- 실무에서는 표, 차트, 이미지 등 멀티모달 데이터 통합 필요
- 행정문서에서 법령 표, 통계 차트 등 다양한 형식 존재

#### 8.1.2. 도메인 제한
- Medical, Novel 2개 도메인만 평가
- 법률(Legal), 금융(Finance) 도메인 확장 필요 (Appendix D에 제안)
- 한국어 행정 도메인 특화 벤치마크 부재

#### 8.1.3. 평가 모델 제한
- GPT-4o-mini, Qwen2.5-14B만 테스트
- 더 다양한 오픈소스 LLM (Gemma, EXAONE, LLaMA 등) 검증 필요

#### 8.1.4. 실시간 업데이트 미고려
- 정적 코퍼스 기반 벤치마크
- 행정문서는 법령 개정 등 동적 업데이트 빈번
- GraphRAG의 증분 업데이트 효율성 미평가

### 8.2. 향후 연구방향

#### 8.2.1. 멀티모달 GraphRAG
> "Future iterations will expand this work to incorporate multimodal evaluation, testing how graph-based retrieval and reasoning mechanisms generalize to hybrid knowledge representations while preserving contextual fidelity across data types."

**번역**: "향후 연구는 멀티모달 평가를 포함하도록 확장하여, 그래프 기반 검색 및 추론 메커니즘이 데이터 유형 간 문맥적 충실도를 유지하면서 하이브리드 지식 표현으로 일반화되는지 테스트할 것이다."

#### 8.2.2. 도메인 확장
- **Legal Domain**: EU Case Law (29.8K), UK Case Law (47K), US Case Law (4.6M)
  - 온톨로지 + 법적 추론 체인 통합
- **Finance Domain**: S&P 500 기업 (2015-2024)
  - EDGAR 데이터베이스 연차보고서, 분기보고서
  - 수치 데이터 처리 필요

#### 8.2.3. 비용-성능 최적화
- HippoRAG2의 효율성 (1K 토큰, Recall 90.9%)을 벤치마크로 활용
- GraphRAG의 컨텍스트 증가를 억제하는 Pruning 기법 연구

#### 8.2.4. 한국어 특화 연구
- **코퍼스**: 한국 법령, 행정규칙, 판례 (e-Law 시스템)
- **Challenge**: 한자 혼용, 복잡한 문장 구조, 도메인 용어
- **Baseline**: 본 논문의 4단계 난이도 + 한국어 특화 평가 지표

## 9. 추가 참고사항

### 9.1. 재현성
- **코드**: https://github.com/GraphRAG-Bench/GraphRAG-Benchmark
- **리더보드**: https://graphrag-bench.github.io/
- **설정**: BGE-large-en-v1.5 임베딩, Temperature 0.7, Top-K 5

### 9.2. 데이터셋 통계

| Benchmark | Avg Entities | Avg Relations | Avg Degree | Avg Component Size |
|-----------|--------------|---------------|------------|-------------------|
| UltraDomain | 170.6 | 73.2 | 0.86 | 2.71 |
| MultiHop-RAG | 10.1 | 3.82 | 0.76 | 2.70 |
| HotpotQA | 39.3 | 12.7 | 0.65 | 2.11 |
| **GraphRAG-Bench (novel)** | **19.6** | **20.9** | **2.27** | **3.99** |
| **GraphRAG-Bench (medical)** | **11.8** | **6.2** | **1.05** | **3.15** |

### 9.3. 질문 분포

| Dataset | Fact Retrieval | Complex Reasoning | Contextual Summarize | Creative Generation |
|---------|---------------|-------------------|---------------------|-------------------|
| UltraDomain | 0% | 0% | 97% | 3% |
| HotpotQA | 78.2% | 21.8% | 0% | 0% |
| MultiHop-RAG | 0% | 50% | 50% | 0% |
| **GraphRAG-Bench** | **25%** | **25%** | **25%** | **25%** |

### 9.4. LLM 사용 고지
> "In the preparation of this manuscript, we used a large language model as a writing assistant. Its main role was to help improve our English writing, such as correcting grammar and refining sentences for clarity and style."

**번역**: "이 원고 작성 시 대형 언어 모델을 작문 보조로 사용했다. 주요 역할은 문법 수정 및 명확성과 스타일을 위한 문장 다듬기 등 영어 작문 개선이었다."

---

**작성일**: 2025-11-30
**검토자**: Claude Code (Sonnet 4.5)
**용도**: 석사 논문 문헌 리뷰 (On-premise Open-source RAG system for Korean public administrative documents)
