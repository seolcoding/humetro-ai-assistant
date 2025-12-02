# 문헌 리뷰: From Local to Global - A Graph RAG Approach to Query-Focused Summarization

## 1. 논문 정보

- **제목**: From Local to Global: A Graph RAG Approach to Query-Focused Summarization
- **저자**: Darren Edge, Ha Trinh, Newman Cheng, Joshua Bradley, Alex Chao, Apurva Mody, Steven Truitt, Dasha Metropolitansky, Robert Osazuwa Ness, Jonathan Larson
- **소속**: Microsoft Research, Microsoft Strategic Missions and Technologies, Microsoft Office of the CTO
- **연도**: 2024 (arXiv:2404.16130v2, 최종 수정 2025년 2월 19일)
- **저널/학회**: arXiv Preprint (Under review)
- **오픈소스**: https://github.com/microsoft/graphrag

## 2. 핵심 내용 요약

본 논문은 대규모 텍스트 코퍼스 전체에 대한 전역적 질문(global sensemaking questions)에 답변하기 위한 GraphRAG 방법론을 제안한다. 기존 벡터 RAG는 특정 문서 집합에서 답을 찾는 국지적 검색(local retrieval)에는 효과적이지만, "데이터셋의 주요 테마는 무엇인가?"와 같은 전역적 이해가 필요한 질의에는 실패한다. GraphRAG는 LLM을 사용하여 소스 문서로부터 엔티티 지식 그래프를 구축하고, Leiden 알고리즘으로 계층적 커뮤니티를 탐지한 후, 각 커뮤니티에 대한 요약을 사전 생성한다. 질의가 주어지면 관련 커뮤니티 요약들을 map-reduce 방식으로 처리하여 최종 답변을 생성한다. 100만 토큰 규모의 데이터셋에서 GraphRAG는 기존 벡터 RAG 대비 comprehensiveness와 diversity 측면에서 72-83%의 승률을 기록하며 우수한 성능을 입증했다.

## 3. 주요 기여점

### 3.1 방법론적 기여
- **전역적 센스메이킹(Global Sensemaking)**: 코퍼스 전체에 대한 이해가 필요한 질의에 답변할 수 있는 최초의 그래프 기반 RAG 접근법
- **계층적 커뮤니티 요약**: Leiden 커뮤니티 탐지 알고리즘을 활용한 그래프 분할 및 계층적 요약 생성
- **Map-Reduce 아키텍처**: 커뮤니티 수준의 부분 답변을 병렬 생성 후 통합하는 확장 가능한 방법론

### 3.2 평가 프레임워크
- **Adaptive Benchmarking**: LLM 기반 페르소나 생성을 통해 도메인 특화된 전역적 질문을 자동 생성
- **LLM-as-a-Judge**: Comprehensiveness, Diversity, Empowerment, Directness 4가지 평가 기준 제시
- **Claim-based Validation**: 추출된 사실적 주장(factual claims)을 기반으로 한 정량적 검증 방법

### 3.3 실용적 기여
- **토큰 효율성**: 루트 레벨 커뮤니티 요약(C0) 사용 시 소스 텍스트 대비 97% 이상 토큰 절감 (9x-43x 감소)
- **오픈소스 생태계**: LangChain, LlamaIndex, NebulaGraph, Neo4J 등 주요 라이브러리에 통합

## 4. 방법론

### 4.1 GraphRAG 워크플로우 (6단계)

```
Source Documents → Text Chunks → Entities & Relationships → Knowledge Graph
→ Graph Communities → Community Summaries → Global Answer
```

#### 4.1.1 텍스트 청킹 (Text Chunking)
- 문서를 600 토큰 단위로 분할 (100 토큰 오버랩)
- 청크 크기는 비용-품질 트레이드오프의 핵심 설계 파라미터

#### 4.1.2 엔티티 및 관계 추출
- **방법**: LLM 프롬프팅을 통한 Named Entity Recognition 및 Relationship Extraction
- **Self-Reflection 기법**: 엔티티 추출 후 LLM에게 누락된 엔티티 재확인 요청 (최대 N회 반복)
- **효과**: 2400 토큰 청크에서 600 토큰 청크 대비 2배 많은 엔티티 추출 성공
- **추출 대상**:
  - Entities: 이름, 타입, 설명
  - Relationships: 소스 엔티티, 타겟 엔티티, 관계 설명, 강도(strength)
  - Claims: 엔티티에 대한 사실적 주장 (날짜, 이벤트, 상호작용 등)

#### 4.1.3 지식 그래프 구축
- Exact string matching 기반 엔티티 매칭 (향후 soft matching으로 개선 가능)
- 관계 중복 횟수를 엣지 가중치로 활용
- 엔티티 설명 집계 및 요약

#### 4.1.4 커뮤니티 탐지
- **알고리즘**: Leiden 커뮤니티 탐지 (Traag et al., 2019)
- **계층 구조**: 재귀적으로 서브 커뮤니티 탐지 (leaf까지)
- **특징**: Mutually exclusive, collectively exhaustive한 파티션 생성

#### 4.1.5 커뮤니티 요약 생성
- **Leaf-level 커뮤니티**: 노드/엣지 degree 기준으로 엘리먼트 우선순위화 후 컨텍스트 윈도우 채우기
- **Higher-level 커뮤니티**: 하위 커뮤니티 요약을 재귀적으로 통합
- **Report 형식**: Title, Summary, Impact Severity Rating, Detailed Findings (5-10 key insights)

#### 4.1.6 질의 응답 (Map-Reduce)
1. **Prepare**: 커뮤니티 요약을 랜덤 셔플 후 청크로 분할
2. **Map**: 각 청크에서 부분 답변 생성 (helpfulness score 0-100 부여)
3. **Reduce**: Score 내림차순 정렬 후 상위 답변들을 통합하여 최종 답변 생성

### 4.2 평가 방법론

#### 4.2.1 질문 생성 (Algorithm 1)
```
K명의 페르소나 → 각 페르소나당 N개 태스크 → 각 태스크당 M개 질문
총 질문 수 = K × N × M (평가에서는 5 × 5 × 5 = 125개)
```

#### 4.2.2 평가 기준
| 기준 | 설명 |
|------|------|
| **Comprehensiveness** | 질문의 모든 측면을 다루는 상세도 |
| **Diversity** | 다양한 관점과 인사이트 제공 정도 |
| **Empowerment** | 독자의 정보에 입각한 판단 지원 정도 |
| **Directness** | 질문에 대한 명확하고 간결한 답변 정도 (Control 기준) |

#### 4.2.3 Claim-based Validation (Experiment 2)
- **Claimify** (Metropolitansky & Larson, 2025): LLM 기반 사실적 주장 추출 도구
- **Comprehensiveness 측정**: 추출된 평균 claim 수
- **Diversity 측정**: Agglomerative clustering (1-ROUGE-L distance) 기반 평균 클러스터 수

## 5. 실험 결과

### 5.1 데이터셋
| 데이터셋 | 설명 | 규모 |
|---------|------|------|
| **Podcast Transcripts** | Behind the Tech with Kevin Scott 팟캐스트 녹취록 | 1,669 청크, ~1M 토큰 |
| **News Articles** | 2013-2023 뉴스 기사 (다양한 카테고리) | 3,197 청크, ~1.7M 토큰 |

### 5.2 비교 조건 (6개)
- **C0-C3**: GraphRAG의 4개 커뮤니티 레벨 (C0=루트, C3=리프)
- **TS**: 소스 텍스트 직접 Map-Reduce (그래프 없음)
- **SS**: 벡터 RAG (Semantic Search baseline)

### 5.3 그래프 구축 결과
| 데이터셋 | 노드 수 | 엣지 수 | 인덱싱 시간 |
|---------|---------|---------|------------|
| Podcast | 8,564 | 20,691 | 281분 (16GB RAM, gpt-4-turbo) |
| News | 15,754 | 19,520 | - |

### 5.4 주요 성능 지표 (Win Rate %)

#### Comprehensiveness
| 조건 | Podcast vs SS | News vs SS |
|------|--------------|------------|
| C0 | 72% (p<.001) | 72% (p<.001) |
| C1 | 75% (p<.001) | 75% (p<.001) |
| C2 | 78% (p<.001) | 79% (p<.001) |
| C3 | 79% (p<.001) | 79% (p<.001) |
| TS | 83% (p<.001) | 80% (p<.001) |

#### Diversity
| 조건 | Podcast vs SS | News vs SS |
|------|--------------|------------|
| C0 | 77% (p<.001) | 62% (p<.01) |
| C1 | 75% (p<.001) | 65% (p<.001) |
| C2 | 81% (p<.001) | 71% (p<.001) |
| C3 | 81% (p<.001) | 69% (p<.001) |
| TS | 82% (p<.001) | 67% (p<.001) |

#### Directness (Control)
- **벡터 RAG(SS)가 모든 비교에서 승리** (44-65% win rate)
- Comprehensiveness와 Diversity는 trade-off 관계 확인

### 5.5 토큰 효율성

| 레벨 | Podcast 토큰 | News 토큰 | 최대 대비 % |
|------|-------------|-----------|------------|
| **C0** | 26,657 | 39,770 | **2.3-2.6%** (9-43배 절감) |
| C1 | 225,756 | 352,641 | 20.7-22.2% |
| C2 | 565,720 | 980,898 | 55.8-57.4% |
| C3 | 746,100 | 1,140,266 | 66.8-73.5% |
| TS | 1,014,611 | 1,707,694 | 100% |

### 5.6 Claim-based Validation 결과

#### Average Claims 수
| 조건 | News | Podcast |
|------|------|---------|
| C0 | **34.18** | 32.21 |
| C3 | 33.14 | 32.28 |
| TS | 32.89 | 31.39 |
| **SS** | **25.23** | **26.50** |

- 모든 전역 방법(C0-C3, TS)이 벡터 RAG(SS)보다 유의미하게 높음 (p<.05)

#### Average Clusters 수 (Distance Threshold 0.7)
| 조건 | News | Podcast |
|------|------|---------|
| **C0** | **20.19** | **20.41** |
| C1 | 19.06 | 20.04 |
| TS | 18.62 | 18.08 |
| **SS** | **15.80** | **16.28** |

### 5.7 핵심 발견
1. **전역적 접근법의 우수성**: GraphRAG 모든 레벨이 벡터 RAG 대비 comprehensiveness와 diversity에서 압도적 우위
2. **최적 레벨**: C2-C3(중간-하위 레벨)이 가장 높은 성능 (57-64% win rate vs TS)
3. **루트 레벨의 효율성**: C0는 성능 소폭 감소하지만 토큰 97% 절감으로 반복적 질의응답에 최적
4. **LLM-Judge 검증**: Claim-based 메트릭과 78% (comprehensiveness), 69-70% (diversity) 일치율

## 6. 우리 연구와의 관련성

### 6.1 직접적 관련성

#### 한국어 행정문서 RAG에 적용 가능한 핵심 아이디어
1. **전역적 질의 처리**: 행정문서에서 "주요 정책 방향", "전체 예산 배분 현황" 등 전역적 이해가 필요한 질문 처리
2. **계층적 요약**: 부서별 → 실국별 → 청단위 계층 구조를 그래프 커뮤니티로 모델링
3. **온프레미스 환경 최적화**: 루트 레벨 커뮤니티 요약으로 토큰 97% 절감 → 오픈소스 LLM 비용 효율성 극대화

#### 벤치마크 방법론 활용
- **Adaptive Benchmarking**: 행정업무 페르소나 기반 평가 질문 생성 (예: 정책입안자, 예산담당자, 민원처리자)
- **LLM-as-a-Judge**: GPT-4/GPT-5 판단자로 EXAONE, Gemma 등 오픈소스 모델 평가
- **Claim-based Validation**: 환각 검증 및 사실성 측정

### 6.2 우리 연구에서 인용할 포인트

#### 1. Graph RAG의 필요성
- 벡터 RAG의 한계: "국지적 검색에는 효과적이지만 전역적 센스메이킹에는 실패"
- 행정문서의 전역적 질의 예시 제시

#### 2. 온프레미스 환경에서의 효율성
- 루트 레벨 커뮤니티 요약(C0)의 토큰 절감 효과 (97%)
- 오픈소스 LLM 사용 시 비용 및 처리 시간 최적화 근거

#### 3. 평가 프레임워크
- LLM-as-a-Judge 방법론의 타당성 검증
- Comprehensiveness, Diversity 기준의 중요성

#### 4. 한국어 적용 시 고려사항
- 엔티티 추출 프롬프트의 도메인 특화 필요성
- Few-shot exemplars의 한국어 행정용어 적응

### 6.3 차별화 포인트

| 항목 | GraphRAG (Microsoft) | 우리 연구 |
|------|---------------------|----------|
| **언어** | 영어 | 한국어 |
| **도메인** | 일반 텍스트 (팟캐스트, 뉴스) | 행정문서 (법령, 공문, 민원) |
| **LLM** | GPT-4-turbo (클라우드) | EXAONE, Gemma 등 (온프레미스) |
| **규모** | 100만 토큰 | AI Hub 데이터셋 기반 |
| **평가** | 일반 센스메이킹 질문 | 행정업무 특화 질문 |

## 7. 인용 가능한 핵심 문장

### 7.1 방법론
> "However, RAG fails on global questions directed at an entire text corpus, such as 'What are the main themes in the dataset?', since this is inherently a query-focused summarization (QFS) task, rather than an explicit retrieval task."

**번역**: 하지만 RAG는 '데이터셋의 주요 테마는 무엇인가?'와 같이 전체 텍스트 코퍼스를 대상으로 하는 전역적 질문에서는 실패하는데, 이는 본질적으로 명시적 검색 작업이 아닌 질의 중심 요약(QFS) 작업이기 때문이다.

### 7.2 핵심 기여
> "GraphRAG first uses an LLM to construct a knowledge graph, where nodes correspond to key entities in the corpus and edges represent relationships between those entities. Next, it partitions the graph into a hierarchy of communities of closely related entities, before using an LLM to generate community-level summaries."

**번역**: GraphRAG는 먼저 LLM을 사용하여 지식 그래프를 구축하는데, 노드는 코퍼스의 핵심 엔티티에 대응하고 엣지는 엔티티 간 관계를 나타낸다. 다음으로 그래프를 밀접하게 관련된 엔티티들의 계층적 커뮤니티로 분할한 후, LLM을 사용하여 커뮤니티 수준의 요약을 생성한다.

### 7.3 성능 개선
> "For a class of global sensemaking questions over datasets in the 1 million token range, we show that GraphRAG leads to substantial improvements over a conventional RAG baseline for both the comprehensiveness and diversity of generated answers."

**번역**: 100만 토큰 범위의 데이터셋에 대한 전역적 센스메이킹 질문 클래스에서, GraphRAG는 생성된 답변의 포괄성과 다양성 모두에서 기존 RAG 베이스라인 대비 상당한 개선을 보인다는 것을 입증한다.

### 7.4 토큰 효율성
> "For low-level community summaries (C3), GraphRAG required 26-33% fewer context tokens, while for root-level community summaries (C0), it required over 97% fewer tokens."

**번역**: 하위 수준 커뮤니티 요약(C3)의 경우 GraphRAG는 26-33% 적은 컨텍스트 토큰을 요구했으며, 루트 수준 커뮤니티 요약(C0)의 경우 97% 이상 적은 토큰을 요구했다.

### 7.5 평가 방법론
> "We developed a novel application of the LLM-as-a-judge technique suitable for questions targeting broad issues and themes where there is no ground-truth answer."

**번역**: 우리는 정답이 없는 광범위한 이슈와 테마를 대상으로 하는 질문에 적합한 LLM-as-a-judge 기법의 새로운 응용을 개발했다.

### 7.6 커뮤니티 탐지
> "GraphRAG contrasts with these approaches by focusing on a previously unexplored quality of graphs in this context: their inherent modularity and the ability to partition graphs into nested modular communities of closely related nodes."

**번역**: GraphRAG는 이 맥락에서 이전에 탐구되지 않았던 그래프의 고유한 특성인 모듈성과 밀접하게 관련된 노드들의 중첩된 모듈형 커뮤니티로 그래프를 분할하는 능력에 집중한다는 점에서 기존 접근법과 차별화된다.

### 7.7 Map-Reduce
> "GraphRAG answers queries through map-reduce processing of community summaries; in the map step, the summaries are used to provide partial answers to the query independently and in parallel, then in the reduce step, the partial answers are combined and used to generate a final global answer."

**번역**: GraphRAG는 커뮤니티 요약의 map-reduce 처리를 통해 질의에 답변한다. map 단계에서는 요약들이 독립적이고 병렬적으로 질의에 대한 부분 답변을 제공하고, reduce 단계에서는 부분 답변들이 결합되어 최종 전역 답변을 생성하는 데 사용된다.

### 7.8 Self-Reflection
> "After entities are extracted from a chunk, we provide the extracted entities back to the LLM, prompting it to 'glean' any entities that it may have missed. This approach allows us to use larger chunk sizes without a drop in quality."

**번역**: 청크에서 엔티티가 추출된 후, 추출된 엔티티를 LLM에 다시 제공하여 놓쳤을 수 있는 엔티티를 '수집'하도록 프롬프트한다. 이 접근법은 품질 저하 없이 더 큰 청크 크기를 사용할 수 있게 한다.

## 8. 한계점 및 향후 연구방향

### 8.1 논문에서 언급한 한계점

#### 평가 범위
1. **제한된 도메인**: 2개 코퍼스(팟캐스트, 뉴스), 각 100만 토큰 규모에만 평가
2. **언어**: 영어 데이터셋에만 실험, 다언어 일반화 검증 필요
3. **환각 검증 부족**: SelfCheckGPT 등 환각률 측정 미실시

#### 기술적 한계
1. **엔티티 매칭**: Exact string matching만 사용 (동음이의어, 표기 변형 처리 미흡)
2. **Empowerment 메트릭**: 혼합된 결과, 구체적 인용/예시 부족 시 낮은 점수
3. **컨텍스트 윈도우**: 8k 토큰 고정 (최적값이지만 더 큰 윈도우 실험 필요)

#### 비용
- GPT-4-turbo 사용 시 인덱싱 비용 (281분, Podcast 데이터셋)
- 클라우드 API 의존성 (온프레미스 오픈소스 LLM 미검증)

### 8.2 논문에서 제안한 향후 연구

#### 하이브리드 RAG
> "We see potential in hybrid RAG schemes that combine embedding-based matching with just-in-time community report generation before employing our map-reduce summarization mechanisms."

**번역**: 우리는 임베딩 기반 매칭을 적시 커뮤니티 리포트 생성과 결합한 후 map-reduce 요약 메커니즘을 사용하는 하이브리드 RAG 방식의 잠재력을 본다.

- **Roll-up**: 여러 계층 수준의 커뮤니티 요약 활용
- **Drill-down**: 상위 수준 요약에서 정보 추적하여 하위로 탐색

#### 로컬 RAG와의 통합
- 임베딩 기반 그래프 어노테이션 매칭
- 전역(global)과 국지적(local) 질의 혼합 처리

### 8.3 우리 연구에서 다룰 추가 연구 주제

#### 한국어 특화 이슈
1. **형태소 분석 기반 엔티티 매칭**: 조사 분리, 복합명사 처리
2. **한국어 행정용어 Few-shot Learning**: 공문서 특유의 용어 및 문체 학습
3. **한자-한글 병기 처리**: "서울특별시(Seoul)", "민원인(民願人)" 등

#### 온프레미스 환경 최적화
1. **오픈소스 LLM 성능 비교**: EXAONE, Gemma, Llama vs GPT-4 (GraphRAG 맥락)
2. **인덱싱 시간 최적화**: Quantization, LoRA 등 경량화 기법 적용
3. **메모리 효율성**: 대규모 그래프 처리를 위한 분산 처리 기법

#### 도메인 적응
1. **행정문서 특화 평가 기준**: Legal compliance, Policy consistency 등
2. **민원 처리 효율성**: 실제 업무 시나리오 기반 평가
3. **보안 및 개인정보**: 온프레미스 환경에서의 데이터 격리 검증

#### 평가 강화
1. **한국어 Claim Extraction**: Claimify의 한국어 적응 또는 대체 도구 개발
2. **Human Evaluation**: LLM-Judge와 실제 행정담당자 평가 비교
3. **환각률 측정**: 한국어 SelfCheckGPT 구현

### 8.4 기술적 개선 방향

| 항목 | 현재 GraphRAG | 개선 방향 |
|------|--------------|----------|
| **엔티티 매칭** | Exact string | Fuzzy matching, 임베딩 유사도 |
| **커뮤니티 탐지** | Leiden | GNN 기반 학습형 클러스터링 |
| **요약 생성** | GPT-4 독립 처리 | Retrieval-augmented summary |
| **평가** | LLM-Judge | Human-in-the-loop validation |
| **배포** | 클라우드 API | 온프레미스 컨테이너화 |

---

## 참고문헌 인용 형식

**APA Style**:
```
Edge, D., Trinh, H., Cheng, N., Bradley, J., Chao, A., Mody, A., Truitt, S.,
Metropolitansky, D., Ness, R. O., & Larson, J. (2024). From local to global:
A graph RAG approach to query-focused summarization. arXiv preprint arXiv:2404.16130v2.
```

**BibTeX**:
```bibtex
@article{edge2024graphrag,
  title={From Local to Global: A Graph RAG Approach to Query-Focused Summarization},
  author={Edge, Darren and Trinh, Ha and Cheng, Newman and Bradley, Joshua and
          Chao, Alex and Mody, Apurva and Truitt, Steven and Metropolitansky, Dasha and
          Ness, Robert Osazuwa and Larson, Jonathan},
  journal={arXiv preprint arXiv:2404.16130},
  year={2024}
}
```

---

**리뷰 작성일**: 2025-11-30
**리뷰어**: Claude Code
**논문 버전**: arXiv:2404.16130v2 (2025-02-19)
