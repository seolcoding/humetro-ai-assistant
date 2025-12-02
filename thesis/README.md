# 석사 논문: On-premise 오픈소스 RAG 시스템 최적화 연구

**주제**: 대한민국 공공행정문서에 최적화된 On-premise 오픈소스 RAG 시스템 구현 조합 탐색

**작성일**: 2025-12-02
**상태**: 연구 진행 중

---

## 1. 연구 개요

### 1.1 연구 배경 (Motivation)

```
공공기관의 데이터 주권/보안 요구 → On-premise 환경 필수
    ↓
On-premise 환경에서 한국어 LLM 성능 검증 부족
    ↓
RAG가 한국어 LLM의 hallucination을 해결하는 핵심 기술
    ↓
그러나 한국어 공공도메인 RAG 최적화 연구 부재
    ↓
∴ 본 연구의 필요성
```

### 1.2 핵심 문제

| 문제 | 근거 |
|------|------|
| LLM Hallucination | 법률/행정 도메인에서 69-88% 발생 (Fan et al., 2024) |
| 한국어 RAG 연구 부족 | 대부분 영어 중심, 한국어 특화 연구 희소 |
| On-premise 검증 부재 | Cloud API 의존 연구가 대다수 |

---

## 2. Research Questions (RQ)

### RQ1: On-premise 오픈소스 LLM의 경쟁력
> On-premise 오픈소스 LLM 기반 RAG 시스템이 상용 API (GPT-4o-mini) 대비 어느 수준의 성능을 달성할 수 있는가?

### RQ2: 검색 전략 비교
> 한국어 공공도메인에서 Lexical(BM25), Semantic(Vector), Hybrid 검색 각각의 성능 차이는 어떠한가?

### RQ3: 한국어 리랭커 효과성
> 한국어 특화 리랭커(KoReranker)는 범용 리랭커 대비 유의미한 성능 향상을 제공하는가?

### RQ4: 비용-성능 최적화
> 24GB VRAM 제약 환경에서 어떤 LLM-RAG 조합이 최적의 비용-성능 균형을 달성하는가?

---

## 3. 가설 (Hypotheses)

### H1: Hybrid Retrieval 우수성
> Hybrid Retrieval (BM25+Vector)이 단일 방식보다 한국어 행정문서에서 높은 Retrieval F1을 보인다.

### H2: KoReranker 효과
> KoReranker는 대규모 corpus (5,000)에서 범용 리랭커 대비 유의미한 성능 향상을 제공한다.

### H3: 오픈소스 LLM 경쟁력
> 12B급 오픈소스 LLM (Gemma3, EXAONE)이 GPT-4o-mini와 유사한 성능을 달성한다.

### H4: GraphRAG 멀티홉 우수성
> GraphRAG는 멀티홉 추론이 필요한 질문에서 Naive RAG보다 우수한 성능을 보인다.

---

## 4. 연구 기여 (Contributions)

### C1. 한국어 공공도메인 RAG 벤치마크 최초 구축
- AI Hub 행정문서 기계독해 데이터 기반
- 100 QA × 5,000 corpus 평가 데이터셋
- 공개 배포를 통한 커뮤니티 기여

### C2. On-premise RAG 최적 파이프라인 도출
- 1,800개 조합 Full Grid Search
- AutoRAG 프레임워크 활용
- 재현 가능한 config 및 코드 공개

### C3. 한국어 리랭커/LLM 효과성 정량 분석
- 6개 리랭커 비교 (KoReranker, ColBERT, MonoT5, BGE, FlashRank, Pass)
- 5개 LLM 비교 (EXAONE-3.5, Gemma3, GPT-OSS, EXAONE-Deep, GPT-4o-mini)
- 비용-성능 트레이드오프 분석

### C4. 공공기관 도입 가능한 청사진 제공
- RTX 3090Ti 24GB 환경 (접근 가능한 하드웨어)
- Cloud API 대비 TCO 분석
- 실무 적용 가이드라인

---

## 5. 실험 설계

### 5.1 데이터셋
| 항목 | 규모 |
|------|------|
| 원본 | AI Hub 행정문서 기계독해 데이터 (63,932 문서, 106,530 QA) |
| 실험용 QA | 100개 (층화 샘플링) |
| 실험용 Corpus | 5,000개 (정답 100 + 노이즈 4,900) |

### 5.2 비교 대상

**Retrieval (3가지)**
- BM25 (Lexical)
- VectorDB (Semantic)
- Hybrid RRF (BM25 + Vector)

**Reranker (6가지)**
- PassReranker (baseline)
- KoReranker (한국어 특화)
- ColBERT
- MonoT5
- BGE-Reranker-Large
- FlashRank

**Generator (5가지)**
- GPT-4o-mini (상용 baseline)
- EXAONE-3.5-7.8B
- Gemma3-12B
- GPT-OSS-20B
- EXAONE-Deep-7.8B

### 5.3 평가 지표
| 카테고리 | 지표 |
|----------|------|
| Retrieval | F1, Recall, Precision |
| Generation | ROUGE, METEOR, BERTScore |
| Faithfulness | G-Eval (Consistency, Coherence, Fluency, Relevance) |

### 5.4 환경
- GPU: RTX 3090Ti 24GB
- OS: Ubuntu Linux
- Framework: AutoRAG v0.3.x
- 총 조합: 1,800개

---

## 6. 선행연구 요약

### 6.1 한국어 RAG 연구

| 연구 | 도메인 | 모델 | 데이터 규모 | 핵심 성과 |
|------|--------|------|------------|----------|
| 정상무 (2024) | 행정문서 요약 | LLaMA 3.1 8B | 500건 | ROUGE-1 58.7%, 파인튜닝+RAG 효과 입증 |
| 이채원 (2025) | 기업분석 Q&A | GPT-3.5 | 3,237 트리플 | Hybrid RAG (Graph+Vector) 우수, Hallucination 0.26 |
| 권혁규 (2025) | 대학 규정 Q&A | GPT-4.1 | 111건 규정 | LangChain 우수, BERT Score 0.746 |

### 6.2 RAG Survey 핵심 인사이트

#### Fan et al. (2024) - KDD 2024
- **아키텍처 분류**: Sparse/Dense Retrieval, Pre/Post-retrieval Enhancement
- **학습 패러다임**: Training-free, Independent, Sequential, Joint Training
- **핵심 발견**:
  - 법률 hallucination 69-88%
  - Noisy retrieval이 hallucination 2배 증가
  - Chunk-level retrieval이 주류

#### Sharma (2025) - RAG Comprehensive Survey
- **Enhancement 분류**: Retrieval, Filtering, Efficiency, Robustness, Reranking
- **최고 성능**: SELF-RAG (PopQA 270% 향상), RQ-RAG (HotpotQA 800% 향상)
- **핵심 발견**:
  - Modular architecture의 중요성
  - Hybrid approach의 우수성

### 6.3 RAG vs GraphRAG 비교

#### Han et al. (2025) - Michigan State & Meta
- **핵심 발견**:
  - RAG: 단일 홉 질문, 세부 정보 요구에 우수
  - GraphRAG: 멀티홉 추론, 비교/시간 순서 질문에 우수
  - Integration 전략으로 6.4% 성능 향상
- **우리 연구 적용**:
  - Vector similarity + Graph expansion 하이브리드
  - FIXED KG Cypher가 0.830 faithfulness 달성

### 6.4 Enterprise RAG

#### B & Purwar (2024) - IIT Madras
- **핵심 발견**:
  - 오픈소스 LLM이 GPT-3.5 대비 67% 비용 절감
  - Llama3-8B가 Mistral-8x7B보다 우수 (파라미터 수 ≠ 성능)
  - 추론 속도 2배 빠름
- **인용 가능**:
  > "Open-source LLMs integrated within RAG framework generate response of similar accuracy and relevance as commercial LLMs."

---

## 7. 인용 가능한 핵심 문장

### RAG 필요성
> "LLMs still suffer from intrinsic limitations, such as the lack of domain-specific knowledge, the problem of 'hallucination'... These problems are particularly notable in domain-specific fields like medicine and law." (Fan et al., 2024)

### 한국어 특화
> "한국어는 동사 중심의 언어로 주어나 목적어가 자주 생략되고 중의적 표현이 많아... 한국어의 언어적 특성을 반영한 지식그래프 구축 연구는 부족한 실정이다." (이채원, 2025)

### RAG vs GraphRAG
> "RAG and GraphRAG are complementary, each excelling in different aspects... Integration strategy improves the best method by 6.4%." (Han et al., 2025)

### 오픈소스 가능성
> "Perplexity LLMs takes only 0.6 USD per million tokens which is less than one-third of the price [of GPT-3.5]." (B & Purwar, 2024)

### Noisy Retrieval 위험
> "LLMs may double the hallucination rate on the non-relevant retrieved passages than on the relevant ones." (Fan et al., 2024)

---

## 8. 논문 구성 (예정)

### Chapter 1: 서론 (10-15p)
- 1.1 연구 배경
- 1.2 연구 필요성
- 1.3 연구 목적 및 범위

### Chapter 2: 이론적 배경 (20-25p)
- 2.1 대규모 언어모델(LLM)의 발전
- 2.2 Retrieval-Augmented Generation (RAG)
- 2.3 Graph RAG
- 2.4 한국어 RAG 연구 동향
- 2.5 평가 프레임워크 (RAGAS, G-Eval)

### Chapter 3: 연구 방법 (25-30p)
- 3.1 데이터셋 구축
- 3.2 시스템 아키텍처
- 3.3 실험 환경
- 3.4 평가 방법

### Chapter 4: 실험 결과 (20-25p)
- 4.1 Baseline 성능 (Naive RAG)
- 4.2 Advanced RAG 최적화 실험
- 4.3 GraphRAG 성능 비교
- 4.4 모델별 성능 분석
- 4.5 비용-성능 Trade-off 분석

### Chapter 5: 결론 (5-10p)
- 5.1 연구 요약
- 5.2 학술적/실무적 기여
- 5.3 한계점 및 향후 연구

---

## 9. 파일 구조

```
thesis/
├── README.md                 # 이 파일 (연구 개요, RQ, 가설, 컨트리뷰션)
├── drafts/                   # 논문 초안
├── literature/               # 선행연구 리뷰 마크다운
│   ├── 정상무 - 2024 - sLLM기반 행정 전자문서 요약.md
│   ├── 이채원 - 2025 - 한국어 Hybrid RAG.md
│   ├── 권혁규 - 2025 - 대학 규정집 RAG 챗봇.md
│   ├── Fan 등 - 2024 - RAG Meeting LLMs Survey.md
│   ├── Sharma - 2025 - RAG Comprehensive Survey.md
│   ├── Han 등 - 2025 - RAG vs GraphRAG.md
│   └── B_Purwar_2024_Enterprise_RAG_Review.md
└── legacy_draft(wrong data).md
```

---

## 10. 체크리스트

### 연구 설계
- [x] Research Questions 정립
- [x] 가설 설정
- [x] 컨트리뷰션 정의
- [x] 선행연구 분석

### 실험
- [ ] 데이터 준비 (100 QA × 5,000 corpus)
- [ ] Full Grid Search 실행
- [ ] 결과 분석
- [ ] 통계적 검정 (paired t-test)
- [ ] Error Analysis

### 논문 작성
- [ ] Chapter 1: 서론
- [ ] Chapter 2: 이론적 배경
- [ ] Chapter 3: 연구 방법
- [ ] Chapter 4: 실험 결과
- [ ] Chapter 5: 결론

---

## 11. 참고문헌 (주요)

1. Fan, W., et al. (2024). A Survey on RAG Meeting LLMs. KDD 2024.
2. Sharma, C. (2025). Retrieval-Augmented Generation: A Comprehensive Survey. arXiv.
3. Han, H., et al. (2025). RAG vs. GraphRAG: A Systematic Evaluation. arXiv.
4. B, G. & Purwar, A. (2024). Evaluating the Efficacy of Open-Source LLMs in Enterprise-Specific RAG Systems. arXiv.
5. 정상무. (2024). sLLM기반 행정 전자문서 요약 생성 RAG 시스템 설계. 국민대학교 석사논문.
6. 이채원. (2025). 한국어 Hybrid RAG 기반 질의응답 시스템 구축. 동아대학교 석사논문.
7. 권혁규. (2025). 검색 증강 생성(RAG) 기반 대학 규정집 질의응답 챗봇 시스템 구축. 한국교통대학교 석사논문.

---

**최종 수정**: 2025-12-02
