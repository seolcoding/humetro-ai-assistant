# 문헌 리뷰: A Survey on Knowledge-Oriented Retrieval-Augmented Generation

## 1. 논문 정보

- **제목**: A Survey on Knowledge-Oriented Retrieval-Augmented Generation
- **저자**: Mingyue Cheng, Yucong Luo, Jie Ouyang, Qi Liu, Huijie Liu, Li Li, Shuo Yu, Bohou Zhang, Jiawei Cao, Jie Ma, Daoyu Wang, Enhong Chen
- **소속**: State Key Laboratory of Cognitive Intelligence, University of Science and Technology of China
- **연도**: 2025 (arXiv:2503.10677v2, 2025년 3월 17일)
- **학회/저널**: ACM (제출 중)
- **페이지**: 50 pages
- **GitHub**: https://github.com/USTCAGI/Awesome-Papers-Retrieval-Augmented-Generation

## 2. 핵심 내용 요약

이 논문은 Retrieval-Augmented Generation (RAG)에 대한 포괄적인 서베이로, **지식 중심(knowledge-oriented)** 관점에서 RAG 시스템을 체계적으로 분석한다. RAG는 대규모 검색 시스템과 생성 모델을 결합하여 외부 지식 소스(문서, 데이터베이스, 구조화된 데이터)를 활용함으로써 자연어 이해 및 생성 성능을 향상시키는 기술이다. 본 논문은 RAG의 핵심 구성 요소(검색 메커니즘, 생성 프로세스, 통합 방법)를 상세히 검토하고, Basic RAG부터 Advanced RAG(멀티모달, 메모리, 에이전틱 RAG 등)까지 분류 체계를 제시한다. 또한 평가 벤치마크, 실제 응용 분야(질의응답, 요약, 정보 검색)를 다루며, RAG 시스템 개선을 위한 향후 연구 방향(검색 효율성, 모델 해석가능성, 도메인 특화 적응)을 제시한다.

## 3. 주요 기여점

1. **지식 중심 관점의 통합 프레임워크**: 기존 서베이들이 특정 측면에 집중한 것과 달리, 지식의 활용을 중심으로 RAG의 전체 파이프라인(지식 소싱 → 임베딩 → 인덱싱 → 검색 → 통합 → 생성)을 체계화

2. **포괄적인 분류 체계**:
   - **Basic RAG**: 사용자 의도 이해, 지식 소스/파싱, 임베딩, 인덱싱, 검색, 통합, 생성, 인용
   - **Advanced RAG**: RAG 학습, 멀티모달 RAG, 메모리 RAG, 에이전틱 RAG

3. **5가지 핵심 목표(Fundamentals) 정립**:
   - Precise User Intent Understanding
   - Accurate Knowledge Retrieval
   - Seamless Knowledge Integration
   - Superior Answer Generation
   - Comprehensive RAG Evaluation

4. **최신 연구 동향 반영**: 2020년 RAG 개념 등장 이후 2024년까지의 주요 연구(RETRO, Self-RAG, GraphRAG, MemoRAG 등) 291개 참고문헌 분석

5. **실무 적용 가이드**: 과학, 금융, 교육, 의료, 법률, 산업 등 다양한 도메인에서의 RAG 적용 사례 및 평가 방법론 제공

## 4. 방법론 (사용된 기술 및 아키텍처)

### 4.1 RAG 기본 아키텍처

**Problem Formulation**:
- 기본: y = f(x)
- RAG: y = f(x, z) = f(x, g(x)), 여기서 z = g(x)는 검색된 외부 지식

**3가지 핵심 컴포넌트**:
1. **Retrieval**: 외부 지식 소스에서 관련 정보 검색
2. **Generation**: 내부 지식과 외부 지식을 결합하여 출력 생성
3. **Knowledge Integration**: 검색된 지식과 LLM 내부 지식의 통합

### 4.2 Knowledge Source and Parsing

**지식 유형별 처리**:
- **Structured Knowledge**: Knowledge Graphs (ToG, KG-RAG, GNN-RAG), Tables (TableRAG)
- **Semi-Structured**: HTML (Beautiful Soup, html5ever, htmlparser2)
- **Unstructured**: PDF (OCR, GPTPDF, Marker, MinerU), 일반 텍스트
- **Multimodal**: 이미지(CLIP, ViT), 오디오(Wav2Vec 2.0, CLAP), 비디오(ViViT, VideoPrism)

### 4.3 Knowledge Embedding

**텍스트 임베딩 모델 진화**:
- **Sparse Encoders**: BoW, N-gram, TF-IDF
- **Traditional Dense**: Word2vec, GloVe, fastText
- **BERT-based**: BERT, RoBERTa, ALBERT, DPR
- **LLM-based**: BGE(4096차원), NV-Embed(4096차원), SFR-Embedding(4096차원)

**청킹(Chunking) 전략**:
- Fixed-length, Semantic-based
- **Advanced**: Proposition-level chunking, LumberChunker(LLM 기반), Late chunking(전체 문서 임베딩 후 분할)

### 4.4 Knowledge Indexing

**인덱스 구조**:
- **Structured**: Inverted Index, Table Index
- **Unstructured**: Vector Index (FAISS, Milvus, Chroma)
- **Graph**: Knowledge Graph, RAPTOR(계층적 요약 트리), HNSW
- **Hybrid**: ColBERT (Inverted + Vector)

### 4.5 Knowledge Retrieval

**검색 전략**:
- **Sparse Retrieval**: BM25, TF-IDF, SPLADE
- **Dense Retrieval**: DPR, ANCE, Llama2vec, RepLLaMA
- **Hybrid Retrieval**: RAP-Gen, BlendedRAG, ReACC

**검색 알고리즘**:
- NNS: Brute Force, KD-tree, Ball-tree, M-tree
- ANNS: LSH, Spectral Hashing, Deep Hashing, HNSW, Product Quantization

### 4.6 Knowledge Integration

**통합 레이어**:
- **Input-Layer Integration**: 검색 문서를 쿼리와 직접 연결
- **Intermediate-Layer Integration**: RETRO(Cross-Attention), TOME(Mention Memory), LongMem
- **Output-Layer Integration**: kNN-LM(확률 보간), Calibration-based

### 4.7 Answer Generation

**Denoising 기법**:
- InstructRAG (Rationale Generation)
- Self-RAG (Self-Reflection)
- COMBO (Discriminator-Based)
- Confidence Scoring

**Reasoning 기법**:
- Graph-Based: Think-on-Graph 2.0
- Cross-Attention: RETRO, kNN-based
- Memory-Augmented: EAE, TOME
- Retrieval Calibration

### 4.8 Advanced RAG Approaches

**RAG Training**:
- Static Training: Retriever 또는 Generator 고정
- Unidirectional Guided: Retriever→Generator (RETRO, RALMs) / Generator→Retriever (DKRR, AAR)
- Collaborative: RAG, REALM (MIPS 활용 공동 최적화)

**Multimodal RAG**:
- Vision-Language: CLIP, MuRAG, RA-CM3, VisRAG
- Audio: LA-RAG (Speech Retrieval)
- Video: 시공간 특징 통합

**Memory RAG**:
- **Implicit Memory**: 모델 파라미터에 저장 (재학습 필요)
- **Explicit Memory**: KV Cache (Memory3, MemoRAG)
- **Working Memory**: 프롬프트 내 검색된 텍스트 (일시적)
- CAG: 실시간 검색 제거, 사전 계산된 캐시만 사용

**Agentic RAG**:
- Query Understanding & Strategy Planning (AT-RAG)
- Toolkit Utilization (웹 검색, API, 계산기)
- Reasoning & Decision Optimization (PlanRAG, REAPER)

## 5. 실험 결과 (주요 성능 지표)

### 5.1 Benchmark Datasets

**Question Answering**:
- Single-hop: Natural Questions, TriviaQA, SQuAD, WebQuestions, PopQA, CRAG
- Multi-hop: HotpotQA, 2WikiMultiHopQA, StrategyQA, MuSiQue, MultiHop-RAG
- Long-form: ELI5, WebCPM, NarrativeQA

**Information Extraction**:
- Entity Linking: ZESHEL, CoNLL
- Relation Extraction: T-REx, ZsRE

**Text Understanding**:
- Classification: TREC, SST-2
- Summarization: WikiASP, XSum
- Generation: Biography

### 5.2 Evaluation Metrics

**Query-Context Relevance**:
- Context Relevance (ARES, RAGAS, TruLens)
- R-precision, Recall@k (KILT)

**Context-Answer Coherence**:
- Recall, Precision (CRUD)
- Answer Faithfulness (ARES, RAGAS)
- Accuracy, Misleading Rate (RECALL)
- Groundedness (TruLens)
- Citation 품질: CAS, CRS (RECLAIM), AIS (RARR)

**Query-Answer Accuracy**:
- Accuracy, Hallucination Rate, Missing Rate (CRAG)
- Answer Relevance (RAGAS, ARES, TruLens)
- RAGQuestEval (CRUD)
- Rejection Rate, Error Detection/Correction Rate (RGB)

**Efficiency**:
- Latency (End-to-End Response Time)
- Throughput (Queries per Second)
- Resource Utilization (CPU/GPU, Memory)

### 5.3 주요 모델 성능 비교

논문은 구체적인 수치 비교보다는 방법론과 접근법을 중심으로 서술하고 있으나, 다음과 같은 트렌드를 제시:

1. **Embedding Models**: BGE(4096차원), NV-Embed(4096차원)가 MTEB 벤치마크에서 최고 성능
2. **Chunking**: Late chunking이 전통적 방법 대비 성능 향상
3. **Retrieval**: Hybrid retrieval이 Sparse 또는 Dense 단독보다 우수
4. **Integration**: Intermediate-Layer Integration이 Input/Output보다 문맥 이해 측면에서 우수
5. **Generation**: Self-RAG의 자기 성찰 메커니즘이 환각(hallucination) 감소에 효과적

## 6. 우리 연구와의 관련성

### 6.1 On-premise 환경에서의 RAG 적용

**관련 포인트**:
- EdgeRAG (Section 9.6): 엣지 컴퓨팅 환경에서의 RAG 배포, 낮은 지연시간, 로컬 프라이버시 보호 → **우리의 On-premise 요구사항과 직접 연관**
- 효율성 평가 (Section 7.2): Latency, Throughput, Resource Utilization 측정 → **제한된 하드웨어 자원(RTX 3090Ti) 환경 최적화 필요**

### 6.2 한국어 행정문서 처리

**관련 포인트**:
- Unstructured Knowledge Parsing (Section 5.2.3): PDF 문서 처리 (OCR, MinerU, Marker) → **AI Hub 행정문서 기계독해 데이터셋 전처리**
- Domain-Specific Applications (Section 8.2.5): Legal RAG (CBR-RAG, LegalBench-RAG, HyPA-RAG) → **법률/행정 문서 특화 RAG 적용 가능**
- Multimodal Knowledge: 표, 이미지 포함 문서 처리 → **행정문서의 복잡한 레이아웃 대응**

### 6.3 Open-source LLM 활용

**관련 포인트**:
- LLM-based Encoders: Llama2vec, RepLLaMA → **Llama, EXAONE, Gemma 등 오픈소스 모델 활용 가능**
- Training Strategies (Section 6.1): Static/Unidirectional/Collaborative Training → **제한된 GPU 자원에서 효율적 학습 전략 선택**
- Personalized RAG (Section 9.3): Memory-based mechanisms → **사용자별 문서 히스토리 관리**

### 6.4 평가 방법론

**관련 포인트**:
- RAGAS Framework: Faithfulness, Answer Relevancy, Answer Correctness → **우리의 평가 메트릭과 동일**
- Korean Benchmarks: C-MTEB(중국어), ru-en-RoSBERTa(러시아어) 언급 → **한국어 벤치마크 필요성 인식**
- Citation Generation (Section 5.8): 정확한 출처 추적 → **행정문서 답변의 신뢰성 확보**

### 6.5 Knowledge Graph 활용

**관련 포인트**:
- GraphRAG (Section 9.1): 구조화된 관계 활용 → **우리의 KG Cypher RAG 실험과 직접 연관**
- Structured Knowledge (Section 5.2.1): GRAG, KG-RAG, GNN-RAG, ToG 2.0 → **멀티홉 추론 성능 향상 방법론**
- Hybrid Vector-Graph Architecture → **우리가 발견한 Vector + Graph 결합 필요성 검증**

## 7. 인용 가능한 핵심 문장

### 7.1 RAG의 필요성

> "RAG models augment traditional language models by incorporating external knowledge sources, such as documents, databases, or structured data, during the generation process. Unlike conventional models that rely solely on pre-trained parameters, RAG systems dynamically retrieve relevant information at generation time, allowing them to produce more informed and contextually accurate outputs."

**번역**: RAG 모델은 생성 과정에서 문서, 데이터베이스 또는 구조화된 데이터와 같은 외부 지식 소스를 통합하여 전통적인 언어 모델을 보강한다. 사전 학습된 파라미터에만 의존하는 기존 모델과 달리, RAG 시스템은 생성 시점에 관련 정보를 동적으로 검색하여 더 정보가 풍부하고 문맥적으로 정확한 출력을 생성할 수 있다.

### 7.2 지식 중심 접근의 중요성

> "At the core of RAG lies a knowledge-centric approach, which places external knowledge as a key factor in improving language generation. By incorporating relevant, real-time, and structured information, RAG models can significantly enhance their ability to generate contextually accurate and factually grounded content."

**번역**: RAG의 핵심에는 지식 중심 접근법이 있으며, 이는 외부 지식을 언어 생성 개선의 핵심 요소로 배치한다. 관련성 있고 실시간이며 구조화된 정보를 통합함으로써, RAG 모델은 문맥적으로 정확하고 사실에 근거한 콘텐츠를 생성하는 능력을 크게 향상시킬 수 있다.

### 7.3 주요 과제

> "One of the primary issues is knowledge selection, where the model must effectively identify the most relevant pieces of information from vast external sources. This task is particularly challenging given the large, noisy, and diverse nature of real-world knowledge corpora. Another critical challenge is knowledge retrieval, which involves retrieving the right information at generation time while balancing efficiency and relevance."

**번역**: 주요 문제 중 하나는 지식 선택으로, 모델이 방대한 외부 소스에서 가장 관련성 높은 정보 조각을 효과적으로 식별해야 한다. 이 작업은 실제 지식 코퍼스의 크고, 노이즈가 많으며, 다양한 특성 때문에 특히 어렵다. 또 다른 중요한 과제는 지식 검색으로, 효율성과 관련성을 균형 있게 유지하면서 생성 시점에 올바른 정보를 검색하는 것을 포함한다.

### 7.4 GraphRAG의 가치

> "By coupling retrieved external knowledge with graph structures, GraphRAG can establish explicit connections among different entities and their relations, leading to more interpretable and robust performance, especially for complex factual or reasoning tasks."

**번역**: 검색된 외부 지식을 그래프 구조와 결합함으로써, GraphRAG는 서로 다른 엔티티와 그들의 관계 사이에 명시적인 연결을 설정할 수 있으며, 특히 복잡한 사실 또는 추론 작업에서 더 해석 가능하고 견고한 성능으로 이어진다.

### 7.5 On-premise 배포의 필요성

> "Deploying retrieval and generation capabilities at the edge, rather than relying solely on cloud-based processing, offers key benefits such as lower latency, reduced bandwidth usage, and enhanced local privacy protection. These advantages are particularly crucial for real-time applications and data-sensitive scenarios where offloading to the cloud is impractical or undesirable."

**번역**: 클라우드 기반 처리에만 의존하는 대신 엣지에서 검색 및 생성 기능을 배포하면 낮은 지연시간, 대역폭 사용 감소, 로컬 프라이버시 보호 강화와 같은 주요 이점을 제공한다. 이러한 장점은 클라우드로 오프로딩하는 것이 비실용적이거나 바람직하지 않은 실시간 애플리케이션 및 데이터 민감 시나리오에서 특히 중요하다.

### 7.6 Trustworthy RAG

> "Ensuring that both retrieval pathways and generated outputs are interpretable is essential for validating information accuracy, enhancing user trust, and supporting decision-making in high-stakes domains such as healthcare, finance, and legal analysis."

**번역**: 검색 경로와 생성된 출력 모두가 해석 가능하도록 보장하는 것은 정보 정확성 검증, 사용자 신뢰 향상, 의료, 금융, 법률 분석과 같은 고위험 영역에서 의사결정 지원을 위해 필수적이다.

### 7.7 Evaluation의 복잡성

> "Evaluating the performance of RAG presents a unique set of challenges due to the dual nature of retrieval and generation tasks. Traditional evaluation metrics, such as BLEU and ROUGE, primarily focus on the quality of the generated text by comparing it to reference outputs. However, these metrics may not adequately capture the effectiveness of the retrieval component, which plays a crucial role in determining the relevance and accuracy of the generated content."

**번역**: RAG의 성능 평가는 검색과 생성 작업의 이중적 특성으로 인해 고유한 도전 과제를 제시한다. BLEU 및 ROUGE와 같은 전통적인 평가 메트릭은 주로 참조 출력과 비교하여 생성된 텍스트의 품질에 초점을 맞춘다. 그러나 이러한 메트릭은 생성된 콘텐츠의 관련성과 정확성을 결정하는 데 중요한 역할을 하는 검색 컴포넌트의 효과를 적절히 포착하지 못할 수 있다.

### 7.8 Domain-Specific Applications

> "In the legal sector, RAG models are instrumental in navigating complex legal texts, supporting legal research, document drafting, and client consultations. For example, CBR-RAG combines Case-Based Reasoning with RAG to structure retrieval processes, enhancing legal question-answering quality by ensuring contextually relevant cases inform the LLM's outputs."

**번역**: 법률 부문에서 RAG 모델은 복잡한 법률 텍스트 탐색, 법률 연구, 문서 초안 작성 및 고객 상담 지원에 중요한 역할을 한다. 예를 들어, CBR-RAG는 사례 기반 추론을 RAG와 결합하여 검색 프로세스를 구조화하고, 문맥적으로 관련된 사례가 LLM의 출력에 반영되도록 보장함으로써 법률 질의응답 품질을 향상시킨다.

## 8. 한계점 및 향후 연구방향

### 8.1 논문에서 제시한 한계점

1. **해석가능성 부족**: 검색-생성 프로세스 간 상호작용의 불투명성
2. **환각(Hallucination) 문제**: 검색된 지식과 생성 콘텐츠 간 불일치
3. **확장성 문제**: 대규모 데이터베이스에서의 실시간 검색 효율성
4. **도메인 적응**: 특정 도메인에 대한 fine-tuning 비용
5. **평가 표준 부재**: 통일된 벤치마크 및 메트릭 부족

### 8.2 향후 연구방향 (Section 9)

**GraphRAG**:
- 멀티소스 이종 그래프 데이터 통합
- 동적 그래프 업데이트 및 오류 수정
- 대규모 그래프에서의 실시간 검색 및 확장성

**Multimodal RAG**:
- 다양한 모달리티 간 효과적 표현 및 검색
- 고차원 멀티모달 데이터의 실시간 처리
- 프라이버시, 강건성, 점진적 업데이트

**Personalized RAG**:
- 메모리 추출, 저장, 지식 업데이트, 융합
- Lifelong learning 전략
- 프라이버시 및 보안 규정 준수
- 다국어 및 크로스 도메인 확장

**Agentic RAG**:
- 구조화된 다단계 추론 향상
- 에러 전파 완화 및 해석가능성 유지
- 폐쇄/독점 데이터셋 접근
- 요약보다 깊은 분석적 통찰력 개발

**RAG and Generative Models**:
- Diffusion 모델 등 다양한 생성 모델과의 통합
- 불확실성 정량화, 오류 수정, 적응적 검색 메커니즘

**EdgeRAG**:
- 모델 압축, 검색 효율성, 적응적 캐싱 전략
- 네트워크 불안정 환경에서의 신뢰성
- 적대적 공격 및 무단 접근 방어

**Trustworthy RAG**:
- 신뢰성 평가 프레임워크 개발
- 검증 가능한 인용, 환각 감소, 일관성 보장
- 모델 편향, 프라이버시, 규제 준수

**Benchmark and Evaluation**:
- 멀티모달, 개인화, 온라인 시나리오 포괄
- 해석가능성, 신뢰성, 불확실성 처리 메트릭
- 실시간 제약 및 보안 고려

### 8.3 우리 연구에 적용할 향후 방향

1. **한국어 특화 RAG 벤치마크 구축**: AI Hub 데이터 기반 표준 평가셋 개발
2. **Hybrid Vector-Graph Architecture 최적화**: 우리의 KG Cypher 실험 결과를 이론적으로 뒷받침
3. **On-premise 효율성 극대화**: EdgeRAG 기법 적용, 모델 압축, 양자화
4. **행정문서 Domain Adaptation**: Legal RAG 방법론(CBR-RAG, HyPA-RAG) 참고
5. **Trustworthy RAG for Public Sector**: 인용 검증, 환각 감소, 감사 메커니즘 구축

## 9. 참고문헌 및 추가 자료

- **논문 arXiv**: https://arxiv.org/abs/2503.10677
- **GitHub Repository**: https://github.com/USTCAGI/Awesome-Papers-Retrieval-Augmented-Generation
- **관련 서베이 비교 (Table 1)**: 본 논문이 LLM, Multimodal, Graph, Advanced(All), Evaluation, Knowledge 모든 측면을 커버하는 유일한 서베이
- **핵심 프레임워크**: RAG(Lewis 2020), RETRO(Borgeaud 2022), Self-RAG(Asai 2024), GraphRAG(Edge 2024), MemoRAG(Qian 2024)

---

**작성일**: 2025-11-30
**검토자**: [검토 필요]
**태그**: #RAG #KnowledgeAugmentation #LLM #RetrievalAugmentedGeneration #Survey #GraphRAG #MultimodalRAG #OnPremise #KoreanNLP #AdministrativeDocuments
