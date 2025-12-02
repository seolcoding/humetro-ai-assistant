# Literature Review: Retrieval-Augmented Generation with Knowledge Graphs for Customer Service Question Answering

## 1. 논문 정보

- **제목**: Retrieval-Augmented Generation with Knowledge Graphs for Customer Service Question Answering
- **저자**: Zhentao Xu, Mark Jerome Cruz, Matthew Guevara, Tie Wang, Manasi Deshpande, Xiaofeng Wang, Zheng Li
- **소속**: LinkedIn Corporation, Sunnyvale, CA, USA
- **학회**: SIGIR '24 (47th International ACM SIGIR Conference on Research and Development in Information Retrieval)
- **발표일**: July 14-18, 2024, Washington, DC, USA
- **DOI**: https://doi.org/10.1145/3626772.3661370

## 2. 핵심 내용 요약

LinkedIn의 고객 서비스 기술 지원 팀에서 과거 이슈 티켓을 효율적으로 검색하고 답변을 생성하기 위해 RAG와 Knowledge Graph를 결합한 시스템을 제안한다. 기존 RAG는 텍스트를 청크로 분할하여 intra-issue 구조와 inter-issue 관계를 상실하는 문제가 있었으나, 본 논문은 과거 이슈 티켓을 트리 구조로 파싱하고 티켓 간 연결 관계를 그래프로 표현하여 검색 정확도를 77.6% (MRR 기준) 향상시켰다. LinkedIn 고객 서비스 팀에 약 6개월간 배포하여 이슈 해결 시간을 중앙값 기준 28.6% 단축시킨 실제 프로덕션 배포 사례를 제시한다.

## 3. 주요 기여점

### 3.1 이론적 기여

1. **Dual-level Graph Architecture**: Intra-issue tree와 inter-issue graph를 분리한 이중 레벨 아키텍처 제안
   - Intra-issue Tree: 각 티켓 내부 섹션(Summary, Description, Steps to Reproduce 등)을 트리로 표현
   - Inter-issue Graph: 티켓 간 명시적 관계(CLONE_FROM, CLONE_TO)와 암묵적 관계(semantic similarity)를 그래프로 표현

2. **Hybrid Knowledge Graph Construction**: Rule-based parsing + LLM-based parsing을 결합한 하이브리드 접근
   - 사전 정의된 필드(코드 섹션 등)는 규칙 기반 추출
   - 복잡한 텍스트는 YAML 템플릿 기반 LLM 파싱

3. **Intent-based Subgraph Retrieval**: 사용자 쿼리에서 entity와 intent를 추출하여 관련 서브그래프를 검색하는 방법론

### 3.2 실무적 기여

1. **프로덕션 배포 검증**: LinkedIn 고객 서비스 팀에 6개월간 배포하여 실제 효과 입증
   - 이슈 해결 시간 중앙값 28.6% 감소 (7시간 → 5시간)
   - 평균 62.5% 감소 (40시간 → 15시간)

2. **Baseline 대비 성능 향상**:
   - MRR: 77.6% 향상 (0.522 → 0.927)
   - Recall@3: 100% 달성 (0.640 → 1.000)
   - BLEU: 561% 향상 (0.057 → 0.377)

3. **Text Segmentation 문제 해결**: 청크 분할로 인한 맥락 손실을 그래프 구조 보존으로 극복

## 4. 방법론

### 4.1 시스템 아키텍처

전체 시스템은 2단계로 구성:

#### Phase 1: Knowledge Graph Construction

**Graph Structure Definition**
- **Intra-issue Tree** T_i(N, E, R):
  - Node n ∈ N: (i, s) 조합으로 식별 (티켓 i의 섹션 s)
  - Edge e ∈ E, r ∈ R: 섹션 간 계층 관계 및 관계 타입

- **Inter-issue Graph** G(T, E, R):
  - Explicit connections E_exp: Jira에 명시된 관계 (CLONE_FROM, related to, caused by 등)
  - Implicit connections E_imp: 티켓 제목 간 코사인 유사도 기반

**Graph Construction Process**

1. **Intra-Ticket Parsing**:
```
t_i = t_i,rule ∪ t_i,llm
T_i = RuleParse(t_i,rule) + LLMParse(t_i,llm, T_template, prompt)
```
   - Rule-based: 사전 정의 필드 추출 (코드 섹션 등)
   - LLM-based: YAML 템플릿 기반 복잡한 텍스트 파싱

2. **Inter-Ticket Connection**:
```
E_exp = {(T_i, T_j) | T_i explicitly connected to T_j}
E_imp = {(T_i, T_j) | cos(embed(T_i), embed(T_j)) ≥ θ}
```

3. **Embedding Generation**:
   - BERT, E5 등 사전학습 임베딩 모델 사용
   - "issue summary", "issue description", "steps to reproduce" 등 주요 섹션 임베딩 생성
   - Vector Database (QDrant) 저장
   - 긴 텍스트는 동일 섹션 내에서 안전하게 청크 분할 가능

#### Phase 2: Retrieval and Question Answering

**Step 1: Query Entity Identification and Intent Detection**
```
P, I = LLM(q, T_template, prompt)
```
- P: Map(N → V) 형태의 named entity (예: "issue summary" → "login issue")
- I: Set 형태의 query intent (예: {"fix solution"})

**Step 2: Embedding-based Retrieval of Sub-graphs**

1. **EBR-based Ticket Identification**:
```
S_Ti = Σ_(k,v)∈P [ Σ_n∈Ti I{n.sec = k} · cos(embed(v), embed(n.text)) ]
```
   - 각 entity (k, v)에 대해 해당 섹션 k의 모든 노드 n과 유사도 계산
   - 노드 레벨 점수를 티켓 레벨로 집계하여 top K_ticket 선택

2. **LLM-driven Subgraph Extraction**:
   - 원본 쿼리를 검색된 티켓 ID 포함하도록 재작성
   - Cypher query로 변환하여 Neo4j에서 서브그래프 추출
   - 예시:
     - 원본: "How to reproduce the issue where user saw 'csv upload error'..."
     - 변환: "How to reproduce 'ENT-22970'"
     - Cypher: `MATCH (j:Ticket {ticket_ID: 'ENT-22970'}) -[:HAS_DESCRIPTION]-> ... RETURN steps_to_reproduce.value`

**Step 3: Answer Generation**
- LLM (GPT-4)을 decoder로 사용하여 검색된 정보와 쿼리를 결합해 답변 생성
- Fallback mechanism: 쿼리 실행 실패 시 baseline text-based retrieval로 복귀

### 4.2 사용된 기술 스택

- **LLM**: GPT-4
- **Embedding Model**: E5, BERT
- **Vector Database**: QDrant
- **Graph Database**: Neo4j (Cypher query language)
- **Issue Tracking System**: Jira

## 5. 실험 결과

### 5.1 평가 데이터셋

- **Golden Dataset**: 전형적인 쿼리, 지원 티켓, 권위 있는 솔루션으로 구성
- **비교 그룹**:
  - Control Group: 기존 text-based EBR
  - Experimental Group: 제안 방법 (KG-RAG)
- 두 그룹 모두 동일한 LLM (GPT-4)과 임베딩 모델 (E5) 사용

### 5.2 검색 성능 (Retrieval Performance)

| Metric | Baseline | Experiment | Improvement |
|--------|----------|------------|-------------|
| **MRR** | 0.522 | **0.927** | **+77.6%** |
| Recall@1 | 0.400 | **0.860** | +115.0% |
| Recall@3 | 0.640 | **1.000** | +56.3% |
| NDCG@1 | 0.400 | **0.860** | +115.0% |
| NDCG@3 | 0.520 | **0.946** | +81.9% |

**주요 발견**:
- Recall@3에서 100% 달성: 상위 3개 결과 내 모든 관련 문서 포함
- MRR 77.6% 향상: 첫 번째 정확한 답변의 평균 순위 크게 개선

### 5.3 답변 생성 성능 (Question Answering Performance)

| Metric | Baseline | Experiment | Improvement |
|--------|----------|------------|-------------|
| **BLEU** | 0.057 | **0.377** | **+561.4%** |
| **METEOR** | 0.279 | **0.613** | **+119.7%** |
| **ROUGE** | 0.183 | **0.546** | **+198.4%** |

**주요 발견**:
- BLEU 점수 6배 이상 향상: 답변 품질의 극적인 개선
- 모든 생성 지표에서 일관된 성능 향상

### 5.4 프로덕션 배포 성과 (Production Use Case)

LinkedIn 고객 서비스 팀 6개월 배포 결과:

| Group | Mean | P50 (Median) | P90 |
|-------|------|--------------|-----|
| Tool Not Used | 40 hours | 7 hours | 87 hours |
| Tool Used | **15 hours** | **5 hours** | **47 hours** |
| **Improvement** | **-62.5%** | **-28.6%** | **-46.0%** |

**실무 임팩트**:
- 중앙값 해결 시간 28.6% 단축 (7시간 → 5시간)
- 평균 해결 시간 62.5% 단축 (40시간 → 15시간)
- P90 해결 시간 46% 단축 (87시간 → 47시간)

## 6. 우리 연구와의 관련성

### 6.1 직접적 관련성

1. **도메인 유사성**:
   - LinkedIn: 고객 서비스 이슈 티켓
   - 우리 연구: 한국어 행정 문서
   - 공통점: 구조화된 문서, 섹션별 정보, 문서 간 참조 관계

2. **KG-RAG 아키텍처 적용 가능성**:
   - Intra-document tree: 행정 문서의 섹션 구조 (제목, 조항, 단서 등) 표현
   - Inter-document graph: 법령 간 인용/참조 관계, 개정 이력 관계 표현
   - 우리 checkpoint (2025-11-06)에서 Vector+Graph 하이브리드 접근의 중요성 이미 확인

3. **Text Segmentation 문제 해결**:
   - LinkedIn: 이슈 티켓의 문제 설명과 해결책이 청크 분할로 분리되는 문제
   - 우리 연구: 행정 문서의 긴 조항이 청크 분할로 맥락 손실되는 문제
   - **해결책**: 그래프 구조로 논리적 연결성 보존

### 6.2 인용 포인트

1. **Section 1 (Introduction)**:
   - Limitation 1 & 2를 우리 연구의 motivation으로 인용
   - 행정 문서에서도 동일한 문제 발생 (구조 손실, 청크 분할 품질 저하)

2. **Section 3.1.2 (Graph Construction)**:
   - Hybrid parsing (Rule-based + LLM-based) 방법론
   - 한국어 행정 문서에도 규칙 기반 + LLM 파싱 조합 적용 가능

3. **Section 3.2.2 (Retrieval)**:
   - EBR-based starting point + Graph expansion
   - 우리 checkpoint (2025-11-06)의 FIXED KG Cypher와 동일한 원칙 확인

4. **Section 4 & 5 (Results)**:
   - MRR 77.6%, BLEU 561% 향상을 KG-RAG의 효과성 근거로 인용
   - 프로덕션 배포 성과 (28.6% 시간 단축)를 실무 가치 입증으로 인용

### 6.3 차별점

| 항목 | LinkedIn (Xu et al. 2024) | 우리 연구 |
|------|---------------------------|----------|
| **언어** | 영어 | **한국어** (형태소 분석, 복잡한 조사 처리 필요) |
| **도메인** | 고객 서비스 (Jira) | **행정 문서** (법령, 규정, 공문서) |
| **LLM** | GPT-4 (proprietary) | **On-premise Open-source LLM** (EXAONE, GPT-OSS 등) |
| **그래프 관계** | CLONE, SIMILAR | **법령 인용, 개정 이력, 상하위법 관계** |
| **섹션 구조** | Jira 필드 (Summary, Description) | **법조문 구조** (편, 장, 조, 항, 호) |
| **평가 지표** | MRR, BLEU, Resolution Time | **Faithfulness, RAGAS, G-Eval** |

## 7. 인용 가능한 핵심 문장

### 7.1 문제 정의

> **영어 원문**: "Issue tracking documents such as Jira possess inherent structure and are interconnected, with references such as 'issue A is related to/copied from/caused by issue B.' The conventional approach of compressing documents into text chunks leads to the loss of vital information."

> **한글 번역**: "Jira와 같은 이슈 추적 문서는 고유한 구조를 가지며 상호 연결되어 있으나, '이슈 A가 이슈 B와 관련됨/복사됨/원인됨'과 같은 참조 관계를 가진다. 문서를 텍스트 청크로 압축하는 기존 접근법은 이러한 중요한 정보의 손실을 초래한다."

### 7.2 Segmentation 문제

> **영어 원문**: "Segmenting extensive issue tickets into fixed-length segments to accommodate the context length constraints of embedding models can result in the disconnection of related content, leading to incomplete answers. For example, an issue ticket describing an issue at its beginning and its solution at the end may be split during the text segmentation process, resulting in the omission of critical parts of the solution."

> **한글 번역**: "임베딩 모델의 컨텍스트 길이 제약을 수용하기 위해 광범위한 이슈 티켓을 고정 길이 세그먼트로 분할하면 관련 콘텐츠가 단절되어 불완전한 답변이 생성될 수 있다. 예를 들어, 시작 부분에 문제를 설명하고 끝 부분에 해결책을 제시하는 이슈 티켓은 텍스트 분할 과정에서 나뉘어져 해결책의 중요한 부분이 누락될 수 있다."

### 7.3 Dual-level Architecture

> **영어 원문**: "We employ a dual-level architecture that segregates intra-issue and inter-issue relations. The Intra-issue Tree T_i(N, E, R) models each ticket t_i as a tree, where each node n ∈ N corresponds to a distinct section s of ticket t_i. The Inter-issue Graph G(T, E, R) represents the network of connections across different tickets, incorporating both explicit links E_exp, defined in issue tracking tickets, and implicit connections E_imp, derived from semantic similarity between tickets."

> **한글 번역**: "우리는 이슈 내부 관계와 이슈 간 관계를 분리하는 이중 레벨 아키텍처를 사용한다. 이슈 내부 트리 T_i(N, E, R)는 각 티켓 t_i를 트리로 모델링하며, 각 노드 n ∈ N은 티켓 t_i의 고유한 섹션 s에 해당한다. 이슈 간 그래프 G(T, E, R)는 서로 다른 티켓 간의 연결 네트워크를 나타내며, 이슈 추적 티켓에 정의된 명시적 링크 E_exp와 티켓 간 의미적 유사성에서 파생된 암묵적 연결 E_imp를 모두 통합한다."

### 7.4 Hybrid Parsing

> **영어 원문**: "We employ a hybrid methodology, initially utilizing rule-based extraction for predefined fields, such as code sections identified via keywords. Subsequently, for text not amenable to rule-based parsing, we engage an LLM for parsing."

> **한글 번역**: "우리는 하이브리드 방법론을 사용하여, 먼저 키워드로 식별되는 코드 섹션과 같은 사전 정의된 필드에 대해 규칙 기반 추출을 활용한다. 이후 규칙 기반 파싱에 적합하지 않은 텍스트에 대해서는 LLM을 활용하여 파싱한다."

### 7.5 EBR + Graph Expansion

> **영어 원문**: "In the EBR-based ticket identification step, the top K_ticket most relevant historical issue tickets are pinpointed by harnessing the named entity set P derived from user queries. Aggregating these node-level scores to ticket-level by summing contributions from nodes belonging to the same ticket, we rank and select the top K_ticket tickets. This method presupposes that the occurrence of multiple query entities is indicative of pertinent links, thus improving retrieval precision."

> **한글 번역**: "EBR 기반 티켓 식별 단계에서는 사용자 쿼리에서 파생된 named entity 집합 P를 활용하여 가장 관련성 높은 상위 K_ticket개의 과거 이슈 티켓을 찾아낸다. 동일한 티켓에 속한 노드의 기여도를 합산하여 노드 레벨 점수를 티켓 레벨로 집계한 후, 상위 K_ticket개 티켓을 순위화하여 선택한다. 이 방법은 여러 쿼리 엔티티의 출현이 관련 링크를 나타낸다고 가정하여 검색 정밀도를 향상시킨다."

### 7.6 성능 향상

> **영어 원문**: "Empirical assessments on our benchmark datasets, utilizing key retrieval (MRR, Recall@K, NDCG@K) and text generation (BLEU, ROUGE, METEOR) metrics, reveal that our method outperforms the baseline by 77.6% in MRR and by 0.32 in BLEU."

> **한글 번역**: "주요 검색 지표(MRR, Recall@K, NDCG@K)와 텍스트 생성 지표(BLEU, ROUGE, METEOR)를 활용한 벤치마크 데이터셋의 실증 평가 결과, 우리 방법은 MRR에서 77.6%, BLEU에서 0.32 포인트만큼 baseline을 능가한다."

### 7.7 실무 배포 성과

> **영어 원문**: "Our method has been deployed within LinkedIn's customer service team for approximately six months and has reduced the median per-issue resolution time by 28.6%."

> **한글 번역**: "우리 방법은 LinkedIn의 고객 서비스 팀에 약 6개월간 배포되었으며, 이슈당 해결 시간 중앙값을 28.6% 단축시켰다."

### 7.8 Graph의 장점

> **영어 원문**: "This integration of a KG not only improves retrieval accuracy by preserving customer service structure information but also enhances answering quality by mitigating the effects of text segmentation."

> **한글 번역**: "KG의 통합은 고객 서비스 구조 정보를 보존하여 검색 정확도를 향상시킬 뿐만 아니라, 텍스트 분할의 영향을 완화하여 답변 품질을 향상시킨다."

## 8. 한계점 및 향후 연구방향

### 8.1 논문에서 제시한 향후 연구 (Section 6)

1. **Automated Graph Template Extraction**:
   - 현재: 수동으로 YAML 템플릿 (T_template) 설계
   - 향후: 자동화된 그래프 템플릿 추출 메커니즘 개발하여 시스템 적응성 향상

2. **Dynamic Knowledge Graph Updates**:
   - 현재: 정적 KG 구축
   - 향후: 사용자 쿼리 기반 실시간 KG 업데이트로 실시간 응답성 향상

3. **Domain Expansion**:
   - 현재: 고객 서비스 도메인에 국한
   - 향후: 다른 컨텍스트(의료, 법률, 교육 등)로 시스템 적용성 탐색

### 8.2 논문의 한계점 (분석)

1. **Proprietary LLM 의존성**:
   - GPT-4 사용으로 비용 및 데이터 프라이버시 이슈
   - On-premise 배포 시 오픈소스 LLM으로 대체 필요성

2. **평가 데이터셋 규모 미공개**:
   - Golden dataset 크기, 쿼리 수 등 구체적 수치 미제시
   - 재현성 및 벤치마크 비교 어려움

3. **한국어/비영어권 언어 미검증**:
   - 영어 데이터에만 적용
   - 형태소가 복잡한 교착어(한국어, 일본어)에서의 효과 불명확

4. **그래프 구축 비용 분석 부족**:
   - KG 구축 시간, 컴퓨팅 리소스, 유지보수 비용 미언급
   - Baseline 대비 total cost-benefit 분석 부재

5. **복잡한 Multi-hop Reasoning 미검증**:
   - 단순 서브그래프 검색에 초점
   - 3-hop 이상의 복잡한 그래프 추론 성능 미평가

### 8.3 우리 연구에서 보완할 점

1. **On-premise Open-source LLM 적용**:
   - EXAONE-3.5, GPT-OSS-20B 등으로 동일 아키텍처 구현
   - 성능 대비 비용 효율성 분석

2. **한국어 행정 문서 특화**:
   - 법조문 구조 (편/장/조/항/호) 파싱
   - 법령 간 인용 관계 자동 추출
   - 개정 이력 그래프 통합

3. **G-Eval 기반 복합 평가**:
   - BLEU/ROUGE 외에 Faithfulness, Coherence, Fluency 등 다차원 평가
   - Multi-hop reasoning 난이도별 성능 분석

4. **AutoRAG 프레임워크 통합**:
   - KG-RAG를 AutoRAG의 node로 통합
   - Passage augmenter, reranker와 조합 실험

## 9. 연구 방법론 참고사항

### 9.1 재현 가능한 구현 요소

1. **Graph Database**: Neo4j (Cypher query)
2. **Vector Database**: QDrant
3. **Embedding Model**: E5, BERT (우리는 multilingual-e5-large-instruct 사용 가능)
4. **LLM**: GPT-4 → EXAONE-3.5-7.8B, GPT-OSS-20B로 대체
5. **Evaluation Metrics**: MRR, Recall@K, NDCG@K, BLEU, ROUGE, METEOR

### 9.2 구현 시 고려사항

1. **Korean Embedding Model 선택**:
   - multilingual-e5-large-instruct (우리가 이미 사용)
   - KoSimCSE-roberta (한국어 특화)

2. **행정 문서 Tree Structure**:
```yaml
- Summary: 법령명/문서 제목
  - Description: 제1조 (목적)
    - Article_1: 조문 본문
      - Clause_1: 제1항
      - Clause_2: 제2항
  - Steps_to_Reproduce: 제2조 (정의)
    - Article_2: 조문 본문
  - Root_Cause: 제3조 (적용 범위)
```

3. **Inter-document Relations**:
   - CITES: 법령 간 인용 (예: "행정기본법 제5조에 따라...")
   - AMENDS: 개정 관계 (예: "2024.3.1 개정")
   - SUPERSEDES: 대체 관계 (예: "구 법령 폐지")
   - SIMILAR_TO: 의미적 유사도 (cosine similarity ≥ θ)

## 10. 인용 형식

### APA Style
```
Xu, Z., Cruz, M. J., Guevara, M., Wang, T., Deshpande, M., Wang, X., & Li, Z. (2024). Retrieval-Augmented Generation with Knowledge Graphs for Customer Service Question Answering. In Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '24) (pp. 2905-2909). ACM. https://doi.org/10.1145/3626772.3661370
```

### IEEE Style
```
Z. Xu et al., "Retrieval-Augmented Generation with Knowledge Graphs for Customer Service Question Answering," in Proc. 47th Int. ACM SIGIR Conf. Research and Development in Information Retrieval (SIGIR '24), Washington, DC, USA, Jul. 2024, pp. 2905-2909, doi: 10.1145/3626772.3661370.
```

---

**작성일**: 2025-11-30
**작성자**: Claude Code Assistant
**프로젝트**: On-premise Open-source RAG System for Korean Public Administrative Documents
**관련 Checkpoint**: `/home/wai-3090ti-220/dev/humetro-ai-assistant/docs/CHECKPOINT_kg_cypher_fix.md`
