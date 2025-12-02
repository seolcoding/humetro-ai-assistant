# Literature Review: Retrieval-Augmented Generation Survey

## 1. 논문 정보

- **제목**: Retrieval-Augmented Generation: A Comprehensive Survey of Architectures, Enhancements, and Robustness Frontiers
- **저자**: Chaitanya Sharma
- **연도**: 2025
- **출판**: arXiv preprint (ACM TOIS 심사 중)
- **DOI**: arXiv:2506.00054v1 [cs.IR]

## 2. 핵심 내용 요약

본 논문은 Retrieval-Augmented Generation (RAG) 시스템의 최신 연구 동향을 포괄적으로 정리한 서베이 논문이다. RAG는 대규모 언어모델(LLM)의 parametric knowledge 한계를 외부 문서 검색을 통해 보완하는 패러다임으로, 사실 일관성(factual consistency)과 도메인 적응성(domain flexibility)을 크게 향상시킨다. 저자는 RAG 아키텍처를 retriever-centric, generator-centric, hybrid, robustness-oriented로 분류하고, 검색 최적화, 컨텍스트 필터링, 디코딩 제어, 효율성 개선 등의 핵심 enhancement 기법들을 체계적으로 분석한다. 또한 short-form QA 및 multi-hop QA 벤치마크에서의 성능 비교 분석을 제시하고, 평가 프레임워크와 향후 연구 방향을 제시한다.

## 3. 주요 기여점

1. **체계적인 분류 체계 (Taxonomy)**: RAG 시스템을 4가지 아키텍처 카테고리로 분류
   - Retriever-based: Query enhancement, retriever adaptation, granularity optimization
   - Generator-based: Faithfulness-aware decoding, context compression, retrieval-guided generation
   - Hybrid: Multi-round retrieval, utility-driven optimization, dynamic triggering
   - Robustness-oriented: Noise mitigation, hallucination control, adversarial defense

2. **5가지 Enhancement 차원 분석**: Retrieval, Filtering, Efficiency, Robustness, Reranking 각 영역의 최신 기법 정리 (Table 1)

3. **Comparative Performance Analysis**: 30개 이상의 RAG 프레임워크에 대한 상대적 성능 향상률 비교 (Table 2, 3)
   - Short-form QA: SELF-RAG가 PopQA에서 270% 향상
   - Multi-hop QA: RQ-RAG가 HotpotQA에서 800% 향상

4. **Benchmark Ecosystem 정리**: RGB, MultiHop-RAG, RAGTruth, MIRAGE, FeB4RAG 등 8개의 주요 벤치마크 비교 (Table 4)

5. **Trade-off 분석**: Retrieval precision vs. generation flexibility, efficiency vs. faithfulness, modularity vs. coordination

## 4. 방법론

### RAG Mathematical Formulation

$$P(y | x) = \sum_{d \in \mathcal{C}} P(y | x, d) \cdot P(d | x)$$

- $P(d|x)$: 검색기의 문서 관련성 점수
- $P(y|x,d)$: 문서를 조건으로 한 생성 확률
- 실제로는 top-k 문서로 근사

### 아키텍처 설계 패턴

**Retriever-Based Enhancements:**
- Query-Driven: RQ-RAG (perplexity-based decomposition), GMR (generative multi-hop), RAG-Fusion (reciprocal rank fusion)
- Retriever-Centric: Re2G (symbolic+neural reranking), SimRAG (self-training), RankRAG (unified reranking)
- Granularity-Aware: LongRAG (compressed long-context), FILCO (context filtering), Sufficient Context analysis

**Generator-Based Enhancements:**
- Faithfulness-Aware: SELF-RAG (critique-generate loop), SelfMem (self-memory), INFO-RAG (denoising)
- Context Compression: FiD-Light (passage compression), xRAG (modality fusion), RAE (answer-aligned semantics)
- Retrieval-Guided: AU-RAG (agent-based), RAG-Ex (perturbation analysis), Confidence-Calibrated RAG

**Hybrid Enhancements:**
- Iterative Retrieval: IM-RAG (inner monologue), GenGround (generate-then-ground), G-Retriever (graph retrieval)
- Utility-Driven: Stochastic RAG (expected utility), M-RAG (multi-agent RL), MedGraphRAG (KG integration)
- Dynamic Triggering: DRAGIN (entropy-based), FLARE (low-confidence based), SELF-ROUTE (difficulty routing), TA-ARE (adaptive), CRAG (corrective)

### Enhancement 기법 분류

**1. Retrieval Enhancement:**
- Adaptive: TA-ARE (14.9% redundant retrieval 감소), DRAGIN (token-level triggering), FLARE (preemptive)
- Multi-source: AU-RAG (agent-based source selection), SimRAG (synthetic QA + round-trip)
- Query refinement: RQ-RAG (perplexity decomposition), R²AG (retrieval metadata injection)
- Hybrid/Structured: M-RAG (semantic partitioning), KRAGEN (KG subgraph), Graph RAG (entity-centric)

**2. Filtering Enhancement:**
- Lexical: FILCO (STRINC + CXMI metrics, hallucination 64% 감소)
- Information-theoretic: IB Filtering (bottleneck framework, EM +3.2)
- Self-supervised: SEER (pseudo-relevance, F1 +13.5%), RAG-Ex (perturbation comparison, 76.9% human alignment)

**3. Efficiency Enhancement:**
- Sparse selection: Sparse RAG (high-signal tokens), R²AG (context-aware injection)
- Inference acceleration: FiD-Light (passage compression), Speculative Pipelining (TTFT 20-50% 감소)
- Caching: RAGCache (hierarchical KV cache), PGDSF (prefix-aware eviction)
- Retrieval quality: RAE (Retriever-as-Answer Classifier + DKS)

**4. Robustness Enhancement:**
- Noise mitigation: RAAT (adversarial training, F1/EM +20-30%), CRAG (inference-time filtering, +12-18%)
- Hallucination reduction: Structured RAG (verified corpora, -30-40%), IM-RAG (iterative refinement, F1 +5.3)
- Security: BadRAG (0.04% poisoning → 98.2% attack success), TrojanRAG (embedding backdoor)

**5. Reranking Enhancement:**
- Adaptive: RLT (ranked list truncation, noise -15%), ToolRerank (familiarity-aware, recall +12%)
- Unified pipeline: RankRAG (joint scoring, MRR@10 +7.8%), uRAG (shared engine, +8% cross-task)
- Fusion-based: RAG-Fusion (reciprocal rank, accuracy +9%), R²AG (recursive reranking, -15% irrelevant)

## 5. 실험 결과

### Short-Form QA 성능 (Table 2)

**최고 성능 프레임워크:**
- SELF-RAG: PopQA 270% 향상 (14.7 → 54.9), ARC-Challenge 200% 향상
- RQ-RAG: PopQA 288% 향상, ARC-Challenge 210% 향상
- Self-CRAG: PopQA 320% 향상, ARC-Challenge 208% 향상
- CRAG: PopQA 303% 향상 (14.7 → 59.3)

**Generator-based 효율성 중심:**
- xRAG: NQ +29% (base 대비), but retrieval baseline 대비 marginal/negative
- FiD-Light: TriviaQA +18.5%, NQ +27%

**Robustness 중심:**
- RAAT: RAG-Bench +116% (base 대비), +27% (retrieval 대비)

### Multi-Hop QA 성능 (Table 6)

**Retriever-based 최고 성능:**
- RQ-RAG: HotpotQA +800% (base 대비), +275% (retrieval 대비)
- LQR: MuSiQue +292% (base 대비), +84% (retrieval 대비)
- LongRAG: HotpotQA +50% (GPT-4o)

**Generator-based:**
- R²AG: HotpotQA +300% (base 대비)
- INFO-RAG: HotpotQA +16-35% (modest)

**Hybrid:**
- FLARE-Direct: 2Wiki +62% (base 대비), +22% (retrieval 대비)
- DRAGIN: HotpotQA +22-44% (base 대비)
- GenGround: HotpotQA +13-36%

### Robustness 성능 (Table 3, 7)

**Faithfulness (FactScore):**
- Self-CRAG: Biography +0.456 (최고)
- Self-RAG: Biography +0.372
- CRAG: Biography +0.252

**Precision/Recall:**
- SELF-RAG: ASQA Precision +29.56%, Recall +18.81%
- Flare-Direct: 2Wiki Precision/Recall +21.6%
- Re2G: TriviaQA Precision +17.8%, Recall +15.9%

### 주요 Trade-off

1. **Retrieval precision vs. Generation flexibility**: 정밀한 검색이 항상 좋은 생성을 보장하지 않음
2. **Efficiency vs. Faithfulness**: xRAG는 효율적이나 사실성에서 타협
3. **Modularity vs. Coordination**: Hybrid 시스템은 성능 좋으나 복잡도 증가

## 6. 우리 연구와의 관련성

### 인용 가능한 포인트

1. **On-premise RAG 시스템 정당성**
   - "RAG addresses critical limitations of parametric knowledge storage—such as factual inconsistency and domain inflexibility"
   - Privacy-preserving retrieval의 중요성 강조 (Section 7.5)

2. **한국어 공공 행정문서 특수성**
   - Domain-specific evaluation 필요성: MIRAGE (의료), Customer Service QA (LinkedIn) 사례
   - Cross-domain generalization 과제 (Section 7.4)

3. **AutoRAG 프레임워크 선택 근거**
   - Modular architecture의 중요성: "high-performing RAG frameworks are modular, with complementary retrieval, filtering, and generation components" (Section 5.4)
   - Hybrid approach의 우수성: retrieval-based와 generator-based의 조합

4. **평가 지표 설계**
   - Core evaluation dimensions: Context Relevance, Answer Faithfulness, Answer Relevance (Section 6.1)
   - Korean-specific metrics 필요: "RAG performance often degrades in the face of domain shifts" (Section 7.4)

5. **Benchmark 설계 참고**
   - RGB (robustness), MultiHop-RAG (multi-hop), RAGTruth (hallucination) 참조
   - Domain-specific benchmark 필요성

6. **Efficiency 최적화 필요성**
   - "retrieval noise and redundancy can degrade output quality" (Section 1)
   - On-premise 환경의 리소스 제약 고려

## 7. 인용 가능한 핵심 문장

### RAG의 필요성

> "Large Language Models (LLMs) have demonstrated impressive generalization across natural language tasks, but their reliance on static, parametric knowledge remains a fundamental limitation. This restricts their ability to handle queries requiring up-to-date, verifiable, or domain-specific information, often resulting in hallucinations or factual inconsistencies."

**번역**: 대규모 언어모델(LLM)은 자연어 과제에서 인상적인 일반화 능력을 보여주지만, 정적인 parametric knowledge에 의존하는 것은 근본적인 한계로 남아있다. 이는 최신의, 검증 가능한, 또는 도메인 특화된 정보를 요구하는 쿼리를 처리하는 능력을 제한하며, 종종 환각(hallucination)이나 사실 불일치를 초래한다.

### RAG의 장점

> "By conditioning generation on retrieved documents, RAG systems offer greater transparency, factual grounding, and adaptability to evolving knowledge bases."

**번역**: 검색된 문서를 조건으로 생성을 수행함으로써, RAG 시스템은 더 큰 투명성, 사실 근거, 그리고 진화하는 지식 베이스에 대한 적응성을 제공한다.

### RAG의 과제

> "However, integrating retrieval with generation introduces unique challenges: retrieval noise and redundancy can degrade output quality; misalignment between retrieved evidence and generated text can lead to hallucinations; and pipeline inefficiencies and latency make deployment costly at scale."

**번역**: 그러나 검색과 생성의 통합은 고유한 과제들을 야기한다: 검색 노이즈와 중복성은 출력 품질을 저하시킬 수 있고; 검색된 증거와 생성된 텍스트 간의 불일치는 환각을 초래할 수 있으며; 파이프라인 비효율성과 지연은 대규모 배포를 비용이 많이 들게 만든다.

### Hybrid Approach의 우수성

> "Hybrid RAG systems tightly couple the retriever and generator, moving beyond modular architectures to treat retrieval and generation as co-adaptive reasoning agents."

**번역**: 하이브리드 RAG 시스템은 검색기와 생성기를 긴밀하게 결합하여, 모듈식 아키텍처를 넘어 검색과 생성을 공동 적응 추론 에이전트로 취급한다.

### Robustness의 중요성

> "Robustness- and security-oriented RAG systems are designed to preserve output quality in the face of noisy, irrelevant, or adversarially manipulated retrieval contexts."

**번역**: 강건성 및 보안 지향 RAG 시스템은 노이즈가 많거나, 관련 없거나, 적대적으로 조작된 검색 컨텍스트에 직면했을 때 출력 품질을 보존하도록 설계된다.

### Multi-hop Reasoning의 필요성

> "Many knowledge-intensive tasks require aggregating evidence across multiple retrieval steps and reasoning over entity or schema-level structures. Current models exhibit limited capacity for compositional inference or procedural synthesis."

**번역**: 많은 지식 집약적 작업들은 여러 검색 단계에 걸친 증거 집계와 엔티티 또는 스키마 수준 구조에 대한 추론을 요구한다. 현재 모델들은 조합적 추론이나 절차적 합성에 대한 제한된 능력을 보인다.

### Domain Adaptation의 과제

> "RAG performance often degrades in the face of domain shifts, novel schema, or temporal drift. Addressing this will require pretraining retrieval modules on diverse proxy tasks, developing meta-retrievers capable of adapting to unseen query distributions."

**번역**: RAG 성능은 도메인 이동, 새로운 스키마, 또는 시간적 변화에 직면했을 때 종종 저하된다. 이를 해결하기 위해서는 다양한 프록시 작업에 대한 검색 모듈 사전학습과, 미지의 쿼리 분포에 적응할 수 있는 메타-검색기 개발이 필요할 것이다.

### Privacy의 중요성

> "As RAG systems are increasingly integrated into user-facing applications, demands for interpretability, personalization, and secure behavior intensify. Future architectures should expose transparent interfaces for explaining retrieval decisions and generation provenance, while supporting privacy-preserving personalization."

**번역**: RAG 시스템이 점점 더 사용자 대면 애플리케이션에 통합됨에 따라, 해석 가능성, 개인화, 그리고 안전한 동작에 대한 요구가 강화된다. 미래의 아키텍처는 검색 결정과 생성 출처를 설명하는 투명한 인터페이스를 제공하는 동시에, 프라이버시 보존 개인화를 지원해야 한다.

### Ablation Study의 중요성

> "Ablation studies consistently reinforce that high-performing RAG frameworks are modular, with complementary retrieval, filtering, and generation components. Performance degradation in ablation settings not only validates novel modules but also guides design toward more interpretable, efficient, and secure RAG pipelines."

**번역**: 어블레이션 연구는 고성능 RAG 프레임워크가 상호 보완적인 검색, 필터링, 생성 구성요소를 가진 모듈식이라는 것을 일관되게 강화한다. 어블레이션 설정에서의 성능 저하는 새로운 모듈을 검증할 뿐만 아니라 더 해석 가능하고, 효율적이며, 안전한 RAG 파이프라인을 향한 설계를 안내한다.

### 평가의 중요성

> "Accurate evaluation demands assessing multiple interdependent components, including retrieval relevance, faithfulness of generated responses, and overall answer utility. These dimensions are interdependent: poor context relevance often cascades into reduced faithfulness and answer relevance."

**번역**: 정확한 평가는 검색 관련성, 생성된 응답의 충실성, 전체 답변 유용성을 포함한 여러 상호 의존적 구성요소를 평가할 것을 요구한다. 이러한 차원들은 상호 의존적이다: 낮은 컨텍스트 관련성은 종종 충실성과 답변 관련성 감소로 이어진다.

## 8. 한계점 및 향후 연구방향

### 논문에서 제시한 Open Challenges (Section 7)

1. **Retrieval Adaptivity and Semantic Alignment**
   - 정적 검색 정책과 고정된 임베딩 변환의 한계
   - 동적으로 조정되는 검색 전략 필요 (depth, modality, source selection)
   - Co-optimized retriever-generator pipelines

2. **Robustness under Noise and Adversarial Conditions**
   - Retrieval perturbations, misleading content, corpus-level poisoning 취약성
   - Retrieval-aware adversarial defenses 필요
   - Noise-aware loss functions, semantic provenance filtering

3. **Multi-Hop Reasoning and Structured Compositionality**
   - 다단계 검색-생성 루프 지원 필요
   - Structured subgoal decomposition
   - Graph-augmented reasoning pipelines

4. **Cross-Domain Generalization and Temporal Adaptivity**
   - Domain shifts, novel schema, temporal drift 시 성능 저하
   - Meta-retrievers, recency-aware document scoring
   - Temporally evolving benchmarks 필요

5. **Explainability, Personalization, and Trust Calibration**
   - 검색 결정과 생성 출처 설명 인터페이스
   - Privacy-preserving personalization
   - Factual salience, source trustworthiness, hallucination risk 신호

### 우리 연구에 대한 시사점

1. **한국어 특화 벤치마크 부재**
   - 모든 실험이 영어 데이터셋(HotpotQA, NQ, TriviaQA 등) 기반
   - 한국어 공공 행정문서에 대한 domain-specific benchmark 필요

2. **On-premise 환경 최적화 연구 부족**
   - 대부분 cloud-based API 모델(GPT-4, Gemini) 또는 대형 모델 중심
   - Resource-constrained 환경에서의 효율성 연구 필요

3. **Legal/Administrative Domain 연구 부족**
   - 의료(MIRAGE, MedGraphRAG), 고객 서비스(LinkedIn) 사례는 있으나
   - 법률/행정 문서의 특수성(formal language, hierarchical structure) 미반영

4. **한국어 평가 메트릭 부재**
   - Context Relevance, Faithfulness, Answer Relevance를 한국어에 맞게 재정의 필요
   - Korean-specific semantic similarity metrics

5. **보안 및 프라이버시 연구 초기 단계**
   - BadRAG, TrojanRAG 등의 공격 사례 제시되었으나 방어 기법은 미흡
   - On-premise 환경의 데이터 프라이버시 보장 메커니즘 연구 필요

### 향후 연구 방향 제안

1. **Korean Administrative RAG Benchmark** 구축
   - Multi-hop reasoning in Korean legal documents
   - Domain-specific faithfulness evaluation
   - Temporal adaptivity (법령 개정 반영)

2. **Efficient On-premise RAG** 최적화
   - Small-scale retriever + generator co-training
   - Quantization-aware RAG pipeline
   - Edge deployment strategies

3. **Hybrid KG-RAG for Administrative Documents**
   - Structured knowledge (법령 체계) + unstructured text 통합
   - Entity-centric graph construction for legal entities

4. **Korean Hallucination Detection**
   - Korean-specific factuality metrics
   - Administrative domain knowledge grounding

---

## 참고문헌 (Selected Key References)

본 논문은 90개의 참고문헌을 인용하며, 주요 RAG 프레임워크는 다음과 같다:

**Retriever-based:** RQ-RAG [6], SimRAG [73], RankRAG [83], LongRAG [31], FILCO [69], SEER [87]

**Generator-based:** SELF-RAG [1], xRAG [10], FiD-Light [24], INFO-RAG [76], R²AG [82]

**Hybrid:** DRAGIN [61], FLARE [32], GenGround [59], Stochastic RAG [84], CRAG [79], TA-ARE [86]

**Robustness:** RAAT [18], BadRAG [78], TrojanRAG [8], RAGTruth [46]

**Benchmarks:** RGB [7], MultiHop-RAG [63], MIRAGE [48], FeB4RAG [68], ARES [55], RAGAS [17]

---

**검토자**: AI Assistant
**검토일**: 2025-11-30
**용도**: On-premise Open-source RAG system for Korean public administrative documents (석사 논문)
