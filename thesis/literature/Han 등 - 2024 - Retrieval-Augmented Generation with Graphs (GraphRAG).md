# Literature Review: Retrieval-Augmented Generation with Graphs (GraphRAG)

## 1. 논문 정보

- **제목**: Retrieval-Augmented Generation with Graphs (GraphRAG)
- **저자**: Haoyu Han, Yu Wang, Harry Shomer, Kai Guo, Jiayuan Ding, Yongjia Lei, Mahantesh Halappanavar, Ryan A. Rossi, Subhabrata Mukherjee, Xianfeng Tang, Qi He, Zhigang Hua, Bo Long, Tong Zhao, Neil Shah, Amin Javari, Yinglong Xia, Jiliang Tang
- **소속**: Michigan State University, University of Oregon, Pacific Northwest National Laboratory, Adobe Research, Hippocratic AI, Amazon, Meta, Snap Inc., The Home Depot
- **연도**: 2024 (arXiv:2501.00309v2, 2025년 1월 8일 제출)
- **저널/학회**: Preprint (Under review)
- **페이지**: 88페이지
- **GitHub**: https://github.com/Graph-RAG/GraphRAG/

## 2. 핵심 내용 요약

본 논문은 그래프 구조 데이터를 활용한 검색 증강 생성(Retrieval-Augmented Generation, RAG) 시스템인 GraphRAG에 대한 포괄적인 서베이 논문이다. 전통적인 RAG가 텍스트나 이미지와 같은 독립적인 데이터를 다루는 반면, GraphRAG는 노드와 엣지로 구성된 그래프 구조를 통해 이질적이고 관계적인 정보를 인코딩한다. 논문은 GraphRAG의 통합 프레임워크를 제시하며, Query Processor, Retriever, Organizer, Generator, Data Source의 5가지 핵심 컴포넌트로 구성된다. 또한 Knowledge Graph, Document Graph, Scientific Graph, Social Graph 등 10개 도메인별로 GraphRAG의 특화된 설계 방법을 체계적으로 정리한다. 이를 통해 그래프 기반 관계 지식이 semantic/lexical similarity만으로는 불가능한 multi-hop reasoning과 복잡한 관계 추론을 가능하게 함을 보여준다.

## 3. 주요 기여점

### 3.1 GraphRAG의 통합 프레임워크 제시

- **5개 핵심 컴포넌트 정의**:
  - Query Processor (Ω_Processor): 쿼리 전처리 (NER, Relation Extraction, Query Structuration 등)
  - Graph Data Source (G): 그래프 구조로 조직된 정보
  - Retriever (Ω_Retriever): 그래프 기반 검색 (Graph Traversal, GNN-based, Similarity-based 등)
  - Organizer (Ω_Organizer): 검색된 내용의 정제 및 구조화
  - Generator (Ω_Generator): 최종 답변 생성 (LLM-based, GNN-based 등)

### 3.2 도메인별 특화 설계 체계화

- **10개 도메인 분류**: Knowledge Graph, Document Graph, Scientific Graph, Social Graph, Planning & Reasoning Graph, Tabular Graph, Infrastructure Graph, Biological Graph, Scene Graph, Random Graph
- 각 도메인별 task applications, graph construction methods, 특화 기술 정리
- 88페이지에 걸친 방대한 문헌 분석 (500+ 논문 참조)

### 3.3 RAG vs GraphRAG의 차별점 명확화

**Difference 1 - Unified vs Diverse-Formatted Information**:
- RAG: 2D grid (image) 또는 1D sequence (text)로 통일된 포맷
- GraphRAG: 다양한 포맷과 이질적 소스 (triplets, paths, cellular complexes 등)

**Difference 2 - Independent vs Interdependent Information**:
- RAG: 독립적인 chunk 단위 저장 및 활용
- GraphRAG: 노드-엣지로 연결된 상호의존적 정보, multi-hop traversal 가능

**Difference 3 - Domain Invariance vs Domain-specific Information**:
- RAG: 도메인 간 전이 가능한 semantic 공유 (vocabulary, texture 등)
- GraphRAG: 도메인별로 고유한 관계 패턴, 통일된 설계 불가능

## 4. 방법론

### 4.1 GraphRAG 기본 프레임워크

```
Query (Q)
  ↓
Query Processor (Ω_Processor) → Q̂
  ↓
Retriever (Ω_Retriever) + Graph Data Source (G) → Content (C)
  ↓
Organizer (Ω_Organizer) → Ĉ
  ↓
Generator (Ω_Generator) → Answer (A)
```

### 4.2 Query Processor 기술

- **Named Entity Recognition (NER)**: 그래프 노드에 grounding된 엔티티 추출
- **Relational Extraction (RE)**: 그래프 엣지 관계 추출
- **Query Structuration**: 텍스트 쿼리를 GQL(Graph Query Language)로 변환
- **Query Decomposition**: 논리적으로 연결된 서브쿼리로 분해
- **Query Expansion**: 관계 지식 기반 쿼리 확장

### 4.3 Retriever 기술 분류 (Knowledge Graph 기준)

**Path-based Retriever**:
- BFS/DFS로 seed entity부터 k-hop 이내 경로 탐색
- 예: "What drugs treat epithelioid sarcoma and affect EZH2?"
  → Disease→[indication]→Drug←[target]←Gene 경로 탐색

**GNN-based Retriever**:
- Graph Neural Networks로 노드/엣지 임베딩 학습
- Attention mechanism으로 relevance 측정
- 예: GraftNet, REANO, PULLNET

**Similarity-based Retriever**:
- Vector similarity 기반 (textual + relational information)
- Multi-vector embedding으로 entity의 다양한 측면 표현
- 예: STaRK, REALM, EMERGE

**Relation-based Retriever**:
- LLM으로 쿼리에서 relevant relations 추출
- Top-k relations 포함하는 triples 검색
- 예: Kim et al. 2024, GenTKGQA

**Fusion-based Retriever**:
- 여러 retrieval 기법 결합
- 예: Mindmap (k-hop paths + 1-hop subgraph), SubgraphRAG (GNN + textual info)

**Agent-based Retriever**:
- LLM agent로 코드 생성 → 그래프 검색 실행
- 예: KnowledGPT, KG-Agent, KnowAgent

### 4.4 Organizer 기술

**Tuple-based Organizer**:
- Triple 형식으로 조직: "(entity1, relation1, entity2)"
- Path 형식: "(entity1, relation1, entity2, relation2, ..., entity_m)"

**Text Organizer**:
- LLM으로 triple/path를 자연어로 verbalization
- Pre-defined templates 활용
- 예: MindMap, KICGPT, STaRK

**Re-Ranking**:
- Relevancy score, citation impact, recency 기반 재정렬
- Cross-Encoder 활용
- 예: KG-Rank, Yang et al. 2024

**Graph Pruning** (Document Graph):
- Community detection algorithms로 distinct communities 생성
- Local clustering coefficient 기반 pruning
- 예: Microsoft GraphRAG (Edge et al. 2024)

### 4.5 Generator 기술

**LLM-based Generator**:
- ChatGPT, Gemini, Mistral, Gemma 등 활용
- Fine-tuning (LoRA) 적용
- Original query + retrieved context를 template에 삽입

**GNN-based Generator**:
- Language embedding + GNN embedding fusion
- Entity 선택 확률 학습
- 예: Yasunaga et al., Taunk et al., Feng et al.

**Hybrid Generator**:
- LLM + GNN 조합으로 structural nuances 보존
- Graph structure를 verbalization하면 정보 손실 → GNN으로 encoding 후 generation

## 5. 실험 결과

### 5.1 Knowledge Graph QA 벤치마크

본 논문은 서베이 논문으로 직접적인 실험 결과보다는 기존 연구들의 성능을 종합 정리:

**STaRK Benchmark** (Semi-structured Textual and Relational Knowledge):
- Multi-vector similarity + LLM re-ranking 조합이 효과적
- Graph structure 활용 시 semantic search 대비 유의미한 성능 향상

**WebQuestionsSP**:
- GNN-based retriever + LLM generator 조합
- Path-based retrieval이 단순 entity retrieval 대비 우수

### 5.2 Document Graph Summarization

**Microsoft GraphRAG (Edge et al. 2024)**:
- Community detection으로 hierarchical abstraction
- Large-scale document summarization에서 기존 RAG 대비 coherence 향상

### 5.3 Scientific Graph (Molecular Property Prediction)

- 3D molecular structure encoding 필요
- GNN으로 geometric information 보존
- Verbalization만으로는 complex geometry 표현 한계

### 5.4 Planning & Reasoning Graph

**Tool Usage / Embodied Planning**:
- Directional awareness 필수 (resource dependency 순서)
- Monte Carlo Tree Search, A* search 활용
- 예: "Cook a potato and put it into the recycle bin" → action sequence graph 생성

## 6. 우리 연구와의 관련성

### 6.1 한국어 행정문서 RAG에 적용 가능한 인사이트

**Document Graph 구축**:
- 행정문서는 계층적 구조 (법령 → 시행령 → 시행규칙)와 참조 관계가 명확
- **Section 4.2 Document Graph Construction** 활용:
  - Sentence-Sentence edge: semantic similarity로 유사 조항 연결
  - Document-Document edge: 법령 간 인용/참조 관계 활용
  - Hierarchical graph: 법 체계의 계층 구조 반영

**Hybrid Retrieval 전략**:
- 우리의 KG Cypher RAG Fix (CLAUDE.md 참조)와 유사한 접근:
  - Vector similarity로 초기 relevant nodes 탐색
  - Graph traversal로 관련 조항/법령으로 확장
  - Original text 보존 (no LLM summarization)

**Query Decomposition**:
- 복잡한 행정 질의 분해 필요
- 예: "외국인 근로자의 산재보험 가입 절차는?"
  → [외국인 근로자 정의] + [산재보험 가입 요건] + [신청 절차]

### 6.2 On-Premise 환경 최적화

**GNN vs LLM Trade-off**:
- 논문에서 지적한 대로 GNN-based retriever는 LLM보다 경량
- 제한된 GPU 환경(RTX 3090Ti 24GB)에서 GNN으로 graph encoding 후 작은 LLM 활용 가능

**Domain-Specific Design**:
- "Difference 3"의 통찰: 행정문서는 고유한 관계 패턴 (법령 체계, 관할 구조 등)
- 범용 GraphRAG 불가능 → 한국 행정 도메인 특화 설계 필요

### 6.3 평가 방법론

**Graph-based Evaluation**:
- Multi-hop reasoning 평가: path coverage, hop accuracy
- Retrieval precision: retrieved subgraph의 relevance
- 우리 연구의 RAGAS 평가와 결합 가능

## 7. 인용 가능한 핵심 문장

### 7.1 GraphRAG의 필요성

> "Unlike conventional RAG, where semantic information can be uniformly represented as a 2D grid of image patches or a 1D sequence of textual corpora, graph-structured data often encompass diverse formats and are stored in heterogeneous sources."

**번역**: 기존 RAG가 의미 정보를 2D 이미지 패치 그리드나 1D 텍스트 시퀀스로 통일되게 표현할 수 있는 반면, 그래프 구조 데이터는 다양한 포맷을 포함하며 이질적인 소스에 저장된다.

### 7.2 관계 지식의 중요성

> "Considering the query 'What drugs are used to treat epithelioid sarcoma and also affect the EZH2 gene product?', blindly executing the existing BM25 or embedding-based search that relies solely on semantic/lexical similarity ignores relational knowledge encoded in graph structure."

**번역**: "상피양 육종을 치료하고 EZH2 유전자 산물에도 영향을 미치는 약물은?"이라는 질의를 고려할 때, 순수하게 의미/어휘 유사도에만 의존하는 기존 BM25나 임베딩 기반 검색은 그래프 구조에 인코딩된 관계 지식을 무시한다.

### 7.3 독립성 vs 상호의존성

> "In conventional RAG, information is stored and utilized independently. [...] However, GraphRAG stores chunks as interconnected nodes with edges denoting their relations, which can benefit retrieval, organization, and generation."

**번역**: 기존 RAG에서는 정보가 독립적으로 저장되고 활용된다. 그러나 GraphRAG는 청크를 상호 연결된 노드로 저장하며 엣지가 관계를 나타내므로, 검색, 구성, 생성에 도움이 된다.

### 7.4 도메인 특화 설계

> "The relations in graph-structured data are domain-specific. Unlike images and texts, where different domains often share transferable semantics, graph-structured data lacks explicit transferable units. [...] This makes the relational information highly domain-specific, and it is nearly impossible to design a unified GraphRAG applicable to different domains."

**번역**: 그래프 구조 데이터의 관계는 도메인 특화적이다. 서로 다른 도메인이 종종 전이 가능한 의미를 공유하는 이미지와 텍스트와 달리, 그래프 구조 데이터는 명시적으로 전이 가능한 단위가 부족하다. 이는 관계 정보를 매우 도메인 특화적으로 만들며, 서로 다른 도메인에 적용 가능한 통일된 GraphRAG를 설계하는 것은 거의 불가능하다.

### 7.5 Verbalization의 한계

> "When retrieved content includes complex graph structures with textual attributes, simply verbalizing the text of the subgraph and concatenating it into a prompt may obscure critical structural information. In these cases, encoding the graph with graph encoders such as GNNs before integrating it into generation can help preserve structural nuances."

**번역**: 검색된 내용이 텍스트 속성을 가진 복잡한 그래프 구조를 포함할 때, 단순히 서브그래프의 텍스트를 언어화하고 프롬프트에 연결하는 것은 중요한 구조 정보를 가릴 수 있다. 이러한 경우 GNN과 같은 그래프 인코더로 그래프를 인코딩한 후 생성에 통합하면 구조적 뉘앙스를 보존할 수 있다.

### 7.6 Document Graph의 활용

> "Document graphs can help compress the corpus by extracting key components and their relationships, proving highly beneficial for multi-document summarization. These graphs also provide different levels of granularity for summarization through hierarchical clustering."

**번역**: 문서 그래프는 주요 구성 요소와 그 관계를 추출하여 코퍼스를 압축할 수 있으며, 이는 다중 문서 요약에 매우 유익하다. 이러한 그래프는 또한 계층적 클러스터링을 통해 요약을 위한 다양한 세분화 수준을 제공한다.

### 7.7 Multi-hop Reasoning

> "Some multi-hop questions require reasoning across multiple documents, necessitating the use of document-level relationships. Humans often consolidate scattered information into structured knowledge to streamline the reasoning process and make more accurate judgments, in line with cognitive load theory."

**번역**: 일부 다중 홉 질문은 여러 문서에 걸친 추론을 요구하므로 문서 수준의 관계 사용이 필요하다. 인간은 종종 흩어진 정보를 구조화된 지식으로 통합하여 추론 프로세스를 간소화하고 더 정확한 판단을 내리는데, 이는 인지 부하 이론과 일치한다.

### 7.8 Pre-Retrieval 전략

> "Constructing a fine-grained graph for a large volume of documents can be inefficient and unnecessary. The pre-retrieval aims to first retrieve relevant documents based on the query and then construct a graph."

**번역**: 대량의 문서에 대해 세밀한 그래프를 구축하는 것은 비효율적이고 불필요할 수 있다. 사전 검색은 먼저 쿼리를 기반으로 관련 문서를 검색한 다음 그래프를 구축하는 것을 목표로 한다.

## 8. 한계점 및 향후 연구방향

### 8.1 논문에서 제시한 주요 Challenge

**Scalability Challenge**:
- 대규모 그래프에서의 효율적 검색 및 추론
- Graph pruning, community detection의 computational cost

**Multimodal Integration**:
- Text-attributed graphs, scene graphs 등에서 vision + structure 통합
- 행정문서의 경우 표, 차트, 이미지 포함 시 멀티모달 처리 필요

**Evaluation Standardization**:
- 도메인별로 분산된 벤치마크와 평가 지표
- Graph-specific metrics (path coverage, subgraph precision) 표준화 부재

**LLM-GNN Integration**:
- 두 모달리티의 최적 융합 방법 미해결
- Trade-off between interpretability (LLM verbalization) and expressiveness (GNN encoding)

### 8.2 우리 연구에서 다룰 수 있는 방향

**한국어 행정 도메인 특화 GraphRAG**:
- 법령 참조 관계, 조항 계층 구조를 반영한 graph construction
- 행정 용어 특화 NER/RE 모델 개발
- 한국어 multi-hop reasoning 벤치마크 구축

**On-Premise 최적화**:
- GNN-based lightweight retriever 개발
- Hybrid architecture: GNN (structure) + Small LLM (generation)
- Efficient graph indexing 및 caching 전략

**투명성 및 설명 가능성**:
- Retrieval path visualization (어떤 법령 경로로 답변 도출했는지)
- Attention weights로 중요 노드/엣지 강조
- 행정 민원인이 이해 가능한 설명 생성

**Hybrid Vector-Graph Approach**:
- 우리의 KG Cypher Fix 검증 확장
- Vector similarity (초기 탐색) + Graph traversal (확장) 최적 조합
- Domain-specific relation weighting

### 8.3 연구 공백 (Research Gap)

논문에서 지적한 현재 GraphRAG 연구의 불균형:
- Knowledge Graph, Document Graph에 집중 (Figure 2)
- Infrastructure Graph, Tabular Graph는 상대적으로 미개척

우리 연구의 기여 가능 영역:
- **Public Administrative Document Graph**: 새로운 도메인 정립
- 계층적 법령 구조 + 참조 관계 결합한 hybrid graph
- 한국어 특화 query processing 기법

## 9. 참고문헌 정보

논문은 500개 이상의 참조 문헌을 포함하며, 주요 카테고리는:

- **Knowledge Graph RAG**: REALM, EMERGE, STaRK, Mindmap, KnowAgent, KG-Agent
- **Document Graph RAG**: Microsoft GraphRAG (Edge et al. 2024), LangChain
- **GNN 기술**: GCN, GraphSAGE, GAT, R-GCN, Graph Transformer
- **벤치마크**: Freebase, ConceptNet, WikiData, WebQuestionsSP
- **도구**: LangChain, graphrag Python package

## 10. 연구 활용 계획

### 10.1 이론적 기반

- GraphRAG 5-component framework를 우리 시스템 아키텍처 설계에 적용
- Domain-specific design 원칙을 한국어 행정문서에 맞게 구체화

### 10.2 방법론 차용

- **Query Processor**: NER/RE for 행정 용어 및 법령 참조 추출
- **Retriever**: Hybrid (Vector + Graph Traversal) 전략
- **Organizer**: Hierarchical clustering for 법령 계층 구조
- **Generator**: LLM + GNN fusion for structural nuances 보존

### 10.3 평가 설계

- Multi-hop reasoning 평가 추가
- Retrieval path quality 평가
- RAGAS + Graph-specific metrics 결합

### 10.4 논문 인용 포인트

- Literature Review에서 GraphRAG의 필요성 및 장점 설명
- Related Work에서 기존 RAG 대비 GraphRAG의 차별점 강조
- Methodology에서 5-component framework 참조
- Discussion에서 domain-specific design 중요성 논의

---

**검토일**: 2025-11-30
**검토자**: Claude Code
**파일 경로**: /home/wai-3090ti-220/dev/humetro-ai-assistant/thesis/literature/Han 등 - 2024 - Retrieval-Augmented Generation with Graphs (GraphRAG).md
