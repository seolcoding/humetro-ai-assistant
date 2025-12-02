# Literature Review: Combining Knowledge Graphs and Large Language Models

## 1. 논문 정보

- **제목**: Combining Knowledge Graphs and Large Language Models
- **저자**: Amanda Kau, Xuzeng He, Aishwarya Nambissan, Aland Astudillo, Hui Yin, Amir Aryani
- **소속**: Australian National University, Swinburne University of Technology
- **연도**: 2024
- **출판**: arXiv:2407.06564v1 [cs.CL] (9 Jul 2024)
- **분류**: Survey Paper

## 2. 핵심 내용 요약

이 논문은 지식 그래프(KG)와 대규모 언어 모델(LLM)의 상호보완적 결합에 대한 체계적 문헌 조사이다. LLM의 주요 약점인 환각(hallucination), 도메인 특화 지식 부족, 해석 가능성 결여 문제를 KG를 통해 완화할 수 있으며, 반대로 KG 구축의 어려움을 LLM이 자동화할 수 있음을 보여준다. 28개의 최신 연구를 분석하여 (1) KG로 강화된 LLM, (2) LLM으로 강화된 KG, (3) 하이브리드 접근법의 세 가지 통합 방향을 체계적으로 분류하고 비교하였다. 특히 "Add-on" 방식과 "Joint" 방식이라는 새로운 분류 체계를 제안하여 두 기술의 결합 방식에 대한 깊이 있는 이해를 제공한다.

## 3. 주요 기여점

### 3.1 체계적 분류 체계 확립

1. **3가지 통합 방향 정립**:
   - LLMs Empowered by KGs: KG를 통한 LLM 성능 향상
   - KGs Empowered by LLMs: LLM을 통한 KG 구축/완성
   - Hybrid Approaches: 양방향 결합을 통한 시너지 창출

2. **새로운 분류 기준 제안**: "Add-on" vs "Joint"
   - **Add-on**: KG와 LLM이 독립적으로 작동하여 상호 보완 (확장성, 비용 절감, 유연성 극대화)
   - **Joint**: KG와 LLM의 임베딩을 통합하여 단일 모델로 작동 (종합적 이해, 최적화된 결과, 정확도 향상)

### 3.2 실무적 통찰 제공

- 28개 논문의 체계적 분석을 통해 기술 동향, 혁신 기법, 공통 과제 도출
- 각 접근법의 강점과 한계를 명확히 제시
- 초보 연구자와 심화 연구자 모두에게 유용한 종합적 개요 제공

## 4. 방법론

### 4.1 LLMs Empowered by KGs

#### 4.1.1 지식 주입 (Knowledge Injection)

**기법 1: Prompt-based Retrieval**
- **KAPING** (Knowledge-Augmented language model PromptING): KG에서 검색한 사실을 질문 앞에 추가하여 zero-shot QA 수행 [20]
- **KICGPT** (Knowledge In Context with GPT): 검색된 KG 사실을 LLM이 재순위화 [22]
- **DRAK** (Domain-specific Retrieval-Augmented Knowledge): 생물분자 도메인에서 구조화된 지식 활용 [23]

**기법 2: Graph Traversal**
- **Knowledge Solver**: LLM에게 KG를 다중 홉(multi-hop) 방식으로 탐색하여 추론하도록 학습 [24]

#### 4.1.2 설명 가능성 향상 (Explainability)

- **QA-GNN**: LLM 인코딩과 KG를 GNN으로 공동 추론, 노드 간 가중치로 추론 경로 시각화 [25]
- **LMExplainer**: KG와 Graph Attention Network로 LLM 의사결정 신호를 자연어 설명으로 변환 [26]

#### 4.1.3 의미적 이해 (Semantic Understanding)

- **LUKE** (Language Understanding with Knowledge-based Embeddings): BERT 확장, 단어와 엔티티를 독립 토큰으로 처리 [27]
- **R3** (Right for Right Reasons): KGQA를 트리 구조 검색으로 전환, 상식 공리를 활용하여 검증 가능한 추론 수행 [28]

### 4.2 KGs Empowered by LLMs

#### 4.2.1 시계열 예측 (Temporal Forecasting)

- **CoH** (Chain-of-History): LLM이 TKG의 엔티티, 관계, 타임스탬프 의미를 이해하여 고차 히스토리 체인 탐색 [29]
- **In-Context Learning for TKG**: LLM에 소수 예제만 제공하여 TKG 패턴 학습, 특수 아키텍처 없이 예측 가능 [30]

#### 4.2.2 지식 그래프 구축 (KG Construction)

**일반 KG 구축**
- **BertNet**: LLM에서 임의 관계의 KG 추출, 프롬프트 패러프레이징 → 엔티티 쌍 랭킹 → KG 생성 [31]
- **TKGCon** (Theme-specific KG Construction): LLM이 온톨로지와 테마별 KG를 비지도 방식으로 구축 [34]

**도메인 특화 KG 구축**
- **Semi-automatic KG Pipeline**: ChatGPT-3.5로 competency questions 생성 → 엔티티/관계 추출 → 온톨로지 매핑 [32]
- **AutoRD**: 희귀질환 정보 추출 및 의료 KG 구축, 비구조화된 의료 텍스트 처리 [33]

### 4.3 Hybrid Approaches

#### 4.3.1 임베딩 융합 (Embedding Fusion)

**외부 융합 (External Fusion)**
- **ERNIE**: T-Encoder (텍스트) + K-Encoder (지식)를 스택하여 통합 feature space 생성 [35]
- **CokeBERT**: LLM으로 단어 토큰 인코딩 + KG에서 엔티티 지식 컨텍스트 추출 후 K-Encoder로 융합 [36]

**내부 융합 (Internal Fusion)**
- **KnowBERT**: BERT 내부에 KAR (Knowledge Attention and Recontextualization) 컴포넌트 추가, 각 transformer block에서 지식 강화 [37]

#### 4.3.2 멀티모달 LLM

- **KRISP**: 이미지+질문 쌍을 multimodal BERT로 처리 (암묵적 지식) + 이미지 심볼로 KG 구축 (명시적 지식) → 비주얼 QA [38]

## 5. 실험 결과

### 5.1 성능 향상 (KG-enhanced LLMs)

| 모델 | 태스크 | 성능 개선 |
|------|--------|-----------|
| **ERNIE** | Entity Typing, Relation Classification | BERT 대비 유의미한 향상 |
| **KnowBERT** | Relation Extraction, Entity Typing | Standard BERT 대비 우수 |
| **QA-GNN** | Question Answering | 추론 경로 시각화로 해석 가능성 향상 |
| **LUKE** | Entity-aware Tasks | 엔티티 임베딩으로 의미 이해 강화 |

### 5.2 비교 분석 (Add-on vs Joint)

| 접근법 | 대표 모델 | 장점 | 성능 특성 |
|--------|-----------|------|-----------|
| **Add-on** | KAPING, BEAR, KnowPhish | 확장성, 비용 절감, 유연성 | KG/LLM 독립 운영 가능 |
| **Joint** | ERNIE, K-BERT, LMExplainer | 종합적 이해, 최적화된 결과 | 의미 이해 태스크에서 우수 |

### 5.3 KG 구축 효율성

- **BertNet**: 수동 구축 대비 자동화된 KG 생성 가능
- **Semi-automatic Pipeline**: 수동 데이터 어노테이션 불필요, 시간/비용 절감
- **AutoRD**: 의료 온톨로지에서 엔티티/관계 자동 추출

### 5.4 실행 시간 분석 (Yang et al. [41])

- **KGPLMs (KG-enhanced PLMs)**: BERT 대비 일관되게 긴 실행 시간
  - Pre-training 단계: 지식 인코더 모듈 추가로 오버헤드 발생
  - Fine-tuning 단계: KG 검색 및 통합으로 추가 연산
  - Inference 단계: 실시간 KG 접근으로 지연 증가
- **Trade-off**: 실행 시간 증가 vs 성능 향상 (knowledge-driven tasks에서 정당화됨)

## 6. 우리 연구와의 관련성

### 6.1 On-premise 환경에서의 적용 가능성

본 연구는 공공 행정문서 처리를 위한 온프레미스 RAG 시스템 구축을 목표로 한다. Kau et al.의 연구에서 제시된 다음 접근법들이 직접 적용 가능하다:

#### 6.1.1 Hybrid Vector-Graph RAG 구현

- **현재 우리 연구**: KG Cypher RAG에서 Vector Similarity Search + Graph Expansion 하이브리드 접근법 검증 완료 (`docs/CHECKPOINT_kg_cypher_fix.md` 참조)
- **Kau et al. 기여**: Knowledge Solver [24]의 multi-hop graph traversal 기법이 우리의 1-hop expansion 접근법의 이론적 근거 제공
- **적용 방안**: QA-GNN [25]의 GNN 기반 경로 추론을 우리의 Cypher query generation에 통합하여 설명 가능성 강화

#### 6.1.2 Add-on 방식의 실용성

- **우리의 설계 철학**: KG와 LLM을 독립적으로 운영하여 온프레미스 환경의 확장성 확보
- **Kau et al. 검증**: BEAR [10], DRAK [23] 등 도메인 특화 KG에서 Add-on 방식의 성공 사례 제시
- **적용 이점**:
  - 한국어 행정문서 KG를 독립적으로 업데이트 가능 (LLM 재학습 불필요)
  - 오픈소스 LLM (EXAONE, GPT-OSS) 교체 시 KG 인프라 재사용 가능
  - 비용 효율적 (Joint 방식 대비 추가 학습 불필요)

#### 6.1.3 한국어 도메인 특화 문제 해결

- **우리 연구의 도전**: "AI Hub 행정문서 기계독해 데이터"의 도메인 특화성
- **Kau et al. 솔루션**:
  - K-BERT [40]: 도메인 지식을 문장에 주입하여 광범위한 사전학습 없이 성능 확보
  - ERNIE [35]: 어휘적, 구문적, 지식 정보를 동시 활용
- **적용 계획**: 행정 용어 온톨로지를 LLM 프롬프트에 주입하여 한국어 행정 도메인 이해도 향상

### 6.2 인용할만한 핵심 포인트

1. **LLM 한계 극복**: "hallucinations and lack of domain-specific knowledge" → KG로 grounding 필요성 정당화
2. **RAG 아키텍처 선택**: Add-on 방식이 "scalability, cost reduction, flexibility" 제공 → 온프레미스 환경 적합성 주장
3. **성능 벤치마크**: KG-enhanced LLMs가 entity typing, relation classification에서 우수 → 우리 실험의 비교 기준
4. **KG 구축 자동화**: LLM 기반 KG 구축이 "eliminate the need for manual data annotation" → 향후 연구 방향

### 6.3 우리 연구와의 차별점

| 항목 | Kau et al. (2024) | 우리 연구 |
|------|-------------------|-----------|
| **초점** | 일반적인 KG-LLM 결합 방법론 | 한국어 행정문서 특화 RAG |
| **언어** | 영어 중심 (BERT, GPT 시리즈) | 한국어 (EXAONE, GPT-OSS) |
| **환경** | 클라우드/API 기반 가정 | 온프레미스 제약 조건 |
| **평가** | 정성적 문헌 조사 | 정량적 평가 (RAGAS, 50개 질문) |
| **KG 구축** | LLM 기반 자동 구축 제안 | 기존 데이터셋 활용 + 검증 |

## 7. 인용 가능한 핵심 문장

### 7.1 LLM의 한계와 KG의 필요성

> "However, they still show some disadvantages, such as hallucinations and lack of domain-specific knowledge, that affect their performance in real-world tasks. These issues can be effectively mitigated by incorporating knowledge graphs (KGs), which organise information in structured formats that capture relationships between entities in a versatile and interpretable fashion."

**번역**: 그러나 LLM은 여전히 환각과 도메인 특화 지식 부족과 같은 단점을 보이며, 이는 실제 작업 성능에 영향을 미친다. 이러한 문제는 엔티티 간의 관계를 다재다능하고 해석 가능한 방식으로 포착하는 구조화된 형식으로 정보를 조직하는 지식 그래프(KG)를 통합함으로써 효과적으로 완화될 수 있다.

**인용 근거**: 우리 연구의 Naive RAG의 한계(Faithfulness 0.746)를 정당화하고, KG 기반 접근법(0.830)의 필요성을 뒷받침

---

### 7.2 KG를 통한 Grounding

> "In this way, KGs could provide facts that LLMs could reason over, grounding them in the process."

**번역**: 이러한 방식으로 KG는 LLM이 추론할 수 있는 사실을 제공하여, 그 과정에서 LLM을 grounding할 수 있다.

**인용 근거**: 우리의 Hybrid Vector-Graph 접근법이 vector similarity search로 starting point를 찾고 graph expansion으로 사실 관계를 확장하는 이론적 근거

---

### 7.3 설명 가능성 향상

> "KGs could, therefore, also allow for better interpretability of LLMs and offer insights into LLMs' reasoning processes, which in turn increase humans' trust in LLMs."

**번역**: 따라서 KG는 LLM의 더 나은 해석 가능성을 제공하고 LLM의 추론 과정에 대한 통찰을 제공하여, 결과적으로 LLM에 대한 인간의 신뢰를 증가시킬 수 있다.

**인용 근거**: 공공 행정문서 처리에서 신뢰성이 중요한 이유, QA-GNN 방식의 graph attention을 통한 추론 경로 시각화 필요성 주장

---

### 7.4 Add-on 방식의 장점

> "The purpose behind employing this approach is so that KGs and LLMs can operate independently to maximizing qualities such as scalability, cost reduction, or flexibility."

**번역**: 이 접근법을 사용하는 목적은 KG와 LLM이 독립적으로 작동하여 확장성, 비용 절감 또는 유연성과 같은 품질을 극대화할 수 있도록 하는 것이다.

**인용 근거**: 온프레미스 환경에서 Add-on 방식 선택의 정당성, 유지보수 용이성 및 비용 효율성 강조

---

### 7.5 Joint 방식의 성능 우위

> "This joint approach ensures that the explanations are human-understandable... Models combining KGs and LLMs typically display a better semantic understanding of knowledge, thus enabling them to perform tasks like entity typing better."

**번역**: 이러한 joint 접근법은 설명이 인간이 이해할 수 있도록 보장한다... KG와 LLM을 결합한 모델은 일반적으로 더 나은 지식의 의미적 이해를 보여주며, 따라서 엔티티 타이핑과 같은 작업을 더 잘 수행할 수 있다.

**인용 근거**: 향후 연구에서 Joint 방식 탐색 시 기대 효과, 한국어 행정 용어 타이핑 정확도 향상 가능성

---

### 7.6 LLM 기반 KG 구축의 효율성

> "Utilising the LLM as an add-on eliminates the need for manual data annotation, saving time and costs."

**번역**: LLM을 add-on으로 활용하면 수동 데이터 어노테이션의 필요성이 제거되어 시간과 비용을 절약할 수 있다.

**인용 근거**: 향후 행정문서 KG 확장 시 LLM 기반 자동 구축 파이프라인 도입 정당성

---

### 7.7 하이브리드 접근법의 의미 이해 강화

> "ERNIE is a language representation model trained on large-scale textual corpora and KGs, allowing it to simultaneously utilise lexical, syntactic, and knowledge information. Incorporating KGs in a joint fashion results in better language understanding."

**번역**: ERNIE는 대규모 텍스트 코퍼스와 KG로 학습된 언어 표현 모델로, 어휘적, 구문적, 지식 정보를 동시에 활용할 수 있다. KG를 joint 방식으로 통합하면 더 나은 언어 이해가 가능하다.

**인용 근거**: 한국어 행정문서의 복잡한 구문 구조(법률 용어, 공문서 형식)를 이해하기 위한 다차원 정보 통합의 필요성

---

### 7.8 실행 시간 Trade-off

> "Due to these knowledge graph-enhanced pre-trained language models (KGPLMs) injecting the knowledge encoder module into pre-trained language models (PLMs), their running times are consistently longer than a vanilla LLM, BERT, across pre-training, fine-tuning, and inference stages. This is although incorporating external knowledge from KGs makes it easier for them to train and enhances their performance."

**번역**: 이러한 지식 그래프 강화 사전학습 언어 모델(KGPLM)은 지식 인코더 모듈을 사전학습 언어 모델(PLM)에 주입하기 때문에, 사전학습, 미세조정, 추론 단계에서 일반 LLM인 BERT보다 일관되게 더 긴 실행 시간을 갖는다. 이는 KG로부터 외부 지식을 통합하는 것이 학습을 더 쉽게 만들고 성능을 향상시킴에도 불구하고 그러하다.

**인용 근거**: 우리의 KG Cypher RAG 실험에서 추론 시간 증가를 정당화하는 근거, 성능 향상(+11% faithfulness)이 시간 비용을 상쇄함을 설명

---

### 7.9 도메인 특화 지식의 중요성

> "BERT is an LLM that lacks domain-specific knowledge since it is pre-trained in general language from large-scale corpora. K-BERT addresses this by injecting domain knowledge from knowledge graphs into sentences. By using the functionalities of LLMs and KGs jointly, good performance in domain-specific tasks can be achieved in K-BERT without requiring extensive pre-training."

**번역**: BERT는 대규모 코퍼스의 일반 언어로 사전학습되어 도메인 특화 지식이 부족한 LLM이다. K-BERT는 지식 그래프로부터 도메인 지식을 문장에 주입하여 이를 해결한다. LLM과 KG의 기능을 jointly 사용함으로써, K-BERT에서 광범위한 사전학습 없이 도메인 특화 작업에서 좋은 성능을 달성할 수 있다.

**인용 근거**: 한국어 행정문서라는 특수 도메인에서 일반 목적 LLM의 한계를 극복하기 위한 지식 주입 방법론의 필요성

---

### 7.10 KG 업데이트의 실용성

> "KGs are significantly easier than LLMs to update, although additional KG completion steps are required. In the case of LLMs, the impracticality of repeating lengthy and costly training processes can significantly affect the time and costs involved."

**번역**: KG는 추가 KG 완성 단계가 필요하지만, LLM보다 훨씬 업데이트하기 쉽다. LLM의 경우, 길고 비용이 많이 드는 학습 과정을 반복하는 것이 비현실적이어서 시간과 비용에 상당한 영향을 미칠 수 있다.

**인용 근거**: 온프레미스 환경에서 행정문서 업데이트가 빈번한 상황에서 KG 기반 접근법의 유지보수 우위성 강조

---

## 8. 한계점 및 향후 연구방향

### 8.1 Kau et al.이 제시한 한계점

#### 8.1.1 KG 가용성 문제

> "One of the major issues is that KGs in some domains may not be widely available, thus limiting the ability to integrate KGs and LLMs."

- **문제**: 특정 도메인에서 KG 부재
- **영향**: KG-LLM 통합의 실용성 제한
- **우리 연구 대응**: AI Hub 데이터셋을 활용한 한국어 행정문서 KG 구축 (corpus.parquet에서 엔티티/관계 추출)

#### 8.1.2 LLM 환각에 의한 KG 품질 저하

> "Even if LLMs were employed to help automate the KG construction process, they could hallucinate or produce incorrect results, compromising KG data's accuracy and validity."

- **문제**: LLM 기반 자동 KG 구축 시 부정확한 정보 생성
- **영향**: KG의 신뢰성 훼손
- **우리 연구 대응**:
  - RAGAS Faithfulness metric으로 환각 측정 (0.398 → 0.830 개선 검증)
  - Human-in-the-loop validation 필요 (향후 연구)

#### 8.1.3 계산 비용 증가

> "Integrating KGs and LLMs can lead to even larger parameter sizes and longer running times... extra time and computational resources would be needed to train these modules as well."

- **문제**: KG-LLM 통합 시 파라미터 증가 및 실행 시간 연장
- **영향**: 온프레미스 환경에서 리소스 부담
- **우리 연구 대응**:
  - RTX 3090Ti 24GB로 실험 진행 중 (`CLAUDE.md` System Spec 참조)
  - Add-on 방식 선택으로 추가 학습 불필요 (Inference-time KG retrieval만 수행)

#### 8.1.4 지식의 신속한 구식화 (Rapid Obsolescence)

> "Another challenge is that KGs and LLMs will likely suffer from becoming outdated due to the rapid evolution of knowledge. One may need to update KGs or LLMs frequently to mitigate this issue."

- **문제**: 지식 진화 속도 > 모델 업데이트 속도
- **영향**: 최신 정보 반영 불가
- **우리 연구 대응**: Add-on 방식으로 KG만 독립적으로 업데이트 가능 (LLM 재학습 불필요)

### 8.2 논문이 다루지 않은 한계점 (우리 연구에서 발견)

#### 8.2.1 한국어 특화 문제

- **문제**: 논문의 모든 사례가 영어 중심 (BERT, GPT, ERNIE 등)
- **Gap**: 한국어 형태소 분석, 조사 처리, 존댓말 등 언어적 특수성 미반영
- **우리 연구 기여**: EXAONE-3.5-7.8B, GPT-OSS-20B 등 한국어 LLM 평가

#### 8.2.2 정량적 벤치마크 부재

- **문제**: 정성적 문헌 조사 중심, 통일된 평가 지표 없음
- **Gap**: Add-on vs Joint 방식의 정량적 비교 불가
- **우리 연구 기여**: RAGAS framework로 Faithfulness, Answer Relevancy, Correctness 정량 측정

#### 8.2.3 실무 배포 시나리오 부족

- **문제**: 온프레미스 환경의 제약 조건 고려 없음 (네트워크 격리, GPU 메모리 제한 등)
- **Gap**: 클라우드 API 기반 가정 (GPT-3, GPT-4 등)
- **우리 연구 기여**: 24GB GPU로 로컬 LLM 실행 가능성 검증

### 8.3 향후 연구방향

#### 8.3.1 Kau et al. 제안 방향

1. **Multimodal KG-LLM Integration**
   - 이미지, 오디오, 비디오를 포함한 멀티모달 KG 구축
   - 예: 행정문서 내 도표, 서식, 도장 이미지 처리

2. **Real-time KG Update Mechanisms**
   - 스트리밍 방식의 KG 업데이트 파이프라인
   - 예: 새로운 법령 공포 시 자동 KG 반영

3. **Efficient Hybrid Architectures**
   - Joint 방식의 계산 비용 최적화
   - 예: Knowledge-aware attention 메커니즘 경량화

#### 8.3.2 우리 연구에서 도출한 향후 방향

1. **한국어 도메인 특화 KG 구축 자동화**
   - **배경**: Kau et al.의 AutoRD [33], TKGCon [34] 기법을 한국어 행정문서에 적용
   - **목표**: "AI Hub 행정문서 기계독해 데이터" 외 타 행정문서 도메인 확장
   - **기대 효과**: 수동 어노테이션 비용 절감

2. **Graph Attention 기반 설명 가능성 강화**
   - **배경**: QA-GNN [25]의 GNN 기반 추론 경로 시각화
   - **목표**: Cypher query 생성 시 각 관계의 중요도 점수 산출
   - **기대 효과**: 공무원 사용자 신뢰도 향상 (답변 근거 제시)

3. **Multi-hop Reasoning 확장**
   - **배경**: 현재 우리 연구는 1-hop expansion만 수행 (`src/kg_agent/kg_cypher_retriever.py`)
   - **목표**: Knowledge Solver [24]의 multi-hop traversal 기법 적용
   - **기대 효과**: 복잡한 다단계 추론 질문 (예: "A 조항을 위반했을 때 B 절차를 거쳐 C 처분을 받는 경우") 성능 향상

4. **Incremental KG Update 메커니즘**
   - **배경**: Kau et al.의 지적 - "KGs will likely suffer from becoming outdated"
   - **목표**: ChromaDB vectordb와 KG 간 동기화 파이프라인 구축 (`autorag_project/resources/vectordb.yaml`)
   - **기대 효과**: 행정문서 개정 시 KG 자동 업데이트

5. **경량화된 Joint 모델 실험**
   - **배경**: Add-on 방식(0.830)과 Joint 방식의 성능 gap 정량 측정 필요
   - **목표**: ERNIE [35] 스타일의 K-Encoder를 EXAONE-3.5-7.8B에 통합
   - **제약**: RTX 3090Ti 24GB 메모리 한계 내에서 구현
   - **기대 효과**: Entity Typing 정확도 향상 (행정 용어 구분)

6. **한국어 특화 Retrieval-Augmented Cypher Generation**
   - **배경**: 현재 Cypher query 생성 시 한국어 엔티티 명칭 불일치 문제 발생 가능
   - **목표**: 한국어 형태소 분석 + KG 스키마 인식 프롬프트 설계
   - **기대 효과**: Cypher query 정확도 향상 → Retrieval Precision 개선

## 9. 메타 정보

### 9.1 리뷰 작성 정보

- **작성일**: 2025-11-30
- **작성자**: Claude Code (AI Assistant)
- **프로젝트**: On-premise Open-source RAG system for Korean public administrative documents
- **관련 문서**:
  - `docs/CHECKPOINT_kg_cypher_fix.md` (KG Cypher RAG 수정 기록)
  - `src/autorag_pilot/config/simple_rag.yaml` (현재 RAG 설정)
  - `CLAUDE.md` (프로젝트 시스템 스펙 및 규칙)

### 9.2 인용 형식

**APA 7th Edition**:
```
Kau, A., He, X., Nambissan, A., Astudillo, A., Yin, H., & Aryani, A. (2024). Combining knowledge graphs and large language models. arXiv preprint arXiv:2407.06564.
```

**BibTeX**:
```bibtex
@article{kau2024combining,
  title={Combining Knowledge Graphs and Large Language Models},
  author={Kau, Amanda and He, Xuzeng and Nambissan, Aishwarya and Astudillo, Aland and Yin, Hui and Aryani, Amir},
  journal={arXiv preprint arXiv:2407.06564},
  year={2024}
}
```

### 9.3 핵심 키워드

- Knowledge Graph (지식 그래프)
- Large Language Model (대규모 언어 모델)
- Retrieval-Augmented Generation (검색 증강 생성)
- Knowledge Injection (지식 주입)
- Explainability (설명 가능성)
- Hybrid Approaches (하이브리드 접근법)
- Graph Neural Network (그래프 신경망)
- Domain-Specific Knowledge (도메인 특화 지식)

### 9.4 관련 논문 (참고문헌에서 발췌)

| 논문 ID | 제목 | 관련성 | 우선순위 |
|---------|------|--------|----------|
| [24] | Knowledge Solver | Multi-hop graph traversal | ⭐⭐⭐ |
| [25] | QA-GNN | Graph 기반 explainability | ⭐⭐⭐ |
| [35] | ERNIE | Joint embedding fusion | ⭐⭐ |
| [29] | Chain-of-History (CoH) | Temporal KG reasoning | ⭐ |
| [33] | AutoRD | Medical domain KG construction | ⭐⭐ |

---

**Note**: 이 문헌 리뷰는 우리 프로젝트의 이론적 기반을 제공하며, 특히 Section 6.1 "On-premise 환경에서의 적용 가능성"과 Section 8.3.2 "우리 연구에서 도출한 향후 방향"이 논문 작성 시 핵심 인용 자료로 활용될 것입니다.
