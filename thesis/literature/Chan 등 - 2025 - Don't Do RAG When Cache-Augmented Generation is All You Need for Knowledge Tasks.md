# 문헌 리뷰: Don't Do RAG: When Cache-Augmented Generation is All You Need for Knowledge Tasks

## 1. 논문 정보

- **제목**: Don't Do RAG: When Cache-Augmented Generation is All You Need for Knowledge Tasks
- **저자**: Brian J Chan, Chao-Ting Chen, Jui-Hung Cheng (National Chengchi University), Hen-Hsen Huang (Academia Sinica)
- **게재**: WWW Companion '25 (The Web Conference 2025)
- **연도**: 2025
- **DOI**: 10.1145/3701716.3715490
- **arXiv**: 2412.15605v2 [cs.CL]

## 2. 핵심 내용 요약

본 논문은 전통적인 RAG(Retrieval-Augmented Generation) 시스템의 대안으로 **CAG(Cache-Augmented Generation)**를 제안한다. Long-context LLM의 확장된 컨텍스트 윈도우를 활용하여, 관리 가능한 크기의 지식 베이스를 사전에 모델에 로드하고 KV(Key-Value) 캐시를 미리 계산함으로써 실시간 검색을 완전히 제거한다. SQuAD와 HotPotQA 벤치마크에서 CAG는 RAG 대비 검색 지연을 제거하고 검색 오류를 최소화하면서도 동등하거나 우수한 성능을 달성했다. 특히 제한적이고 관리 가능한 지식 베이스를 다루는 애플리케이션에서 CAG는 RAG보다 간소하고 효율적인 대안이 될 수 있음을 입증했다.

## 3. 주요 기여점

### 3.1 RAG의 효율적 대안 제시
- Long-context LLM의 확장된 컨텍스트 윈도우(Llama 3.1: 32K-64K 유효 컨텍스트)를 활용한 검색 없는 지식 통합 방법론 제시
- 사전 로딩된 문서와 미리 계산된 KV 캐시를 통해 검색 지연, 검색 오류, 시스템 복잡성 제거

### 3.2 정량적 분석
- 관리 가능한 지식 베이스에서 long-context LLM이 전통적 RAG 시스템을 능가하는 시나리오를 광범위한 실험으로 입증
- BERTScore 기준으로 대부분의 경우 CAG가 최고 성능 달성

### 3.3 실용적 통찰력
- 지식 집약적 워크플로우 최적화를 위한 실행 가능한 인사이트 제공
- 특정 애플리케이션(내부 문서, FAQ, 고객 지원 로그 등)에서 검색 없는 방법론의 실행 가능성 입증
- 오픈소스 CAG 프레임워크 공개 (https://github.com/hhhuang/CAG)

## 4. 방법론

### 4.1 CAG 프레임워크 3단계

#### (1) 외부 지식 사전 로딩 (External Knowledge Preloading)
```
C_KV = KV-Encode(D)
```
- 관련 문서 컬렉션 D = {d₁, d₂, ...}를 모델의 확장된 컨텍스트 윈도우에 맞게 전처리
- LLM이 문서를 처리하여 미리 계산된 KV 캐시로 변환
- KV 캐시를 디스크 또는 메모리에 저장 (한 번만 계산)

#### (2) 추론 (Inference)
```
r = M(D ⊕ q) = M(q | C_KV)
```
- 미리 계산된 KV 캐시 C_KV와 사용자 쿼리 q를 함께 로드
- LLM이 캐시된 컨텍스트를 활용하여 응답 생성
- 검색 지연 제거 및 동적 검색으로 인한 오류/누락 위험 감소

#### (3) 캐시 리셋 (Cache Reset)
- KV 캐시는 append-only 방식으로 새 토큰 추가
- 새 토큰을 truncate하여 효율적으로 리셋
- 전체 캐시 재로딩 없이 신속한 재초기화 가능

### 4.2 사용 모델 및 환경
- **모델**: Llama 3.1 8B (128K 토큰 지원, 유효 컨텍스트 32K)
- **하드웨어**: Tesla V100 32G × 8 GPUs
- **프레임워크**: LlamaIndex (RAG 베이스라인용)

### 4.3 베이스라인 시스템
1. **Sparse Retrieval (BM25)**: TF-IDF 기반 키워드 매칭
2. **Dense Retrieval (OpenAI Indexes)**: 의미적 임베딩 기반 검색
3. **In-Context Learning**: 실시간 KV 캐시 계산 (비교 대조군)

## 5. 실험 결과

### 5.1 데이터셋
| 데이터셋 | 크기 | 문서 수 | 토큰 수 | QA 쌍 수 |
|---------|------|---------|---------|----------|
| HotPotQA | Small | 16 | 21k | 1,392 |
| HotPotQA | Medium | 32 | 43k | 1,056 |
| HotPotQA | Large | 64 | 85k | 1,344 |
| SQuAD | Small | 3 | 21k | 500 |
| SQuAD | Medium | 4 | 32k | 500 |
| SQuAD | Large | 7 | 50k | 500 |

### 5.2 BERTScore 성능 비교 (주요 결과)

#### HotPotQA
- **Small**: CAG **0.7951** (최고) vs Sparse RAG Top-5 0.7676 vs Dense RAG Top-3 0.7582
- **Medium**: CAG **0.7821** (최고) vs Sparse RAG Top-5 0.7633 vs Dense RAG Top-3 0.7432
- **Large**: CAG 0.7407 vs Sparse RAG Top-5 **0.7535** vs Dense RAG Top-3 0.7409

#### SQuAD
- **Small**: CAG **0.7695** (최고) vs Sparse RAG Top-3 0.7616 vs Dense RAG Top-10 0.7586
- **Medium**: CAG **0.7383** (최고) vs Dense RAG Top-10 0.7310 vs Sparse RAG Top-3 0.7301
- **Large**: CAG **0.7734** (최고) vs Sparse RAG Top-5 0.7658 vs Dense RAG Top-10 0.7590

**핵심 발견**: CAG는 대부분의 경우 최고 성능을 달성했으며, 특히 Small/Medium 크기에서 우수한 성능을 보임

### 5.3 응답 시간 비교 (HotPotQA, 초 단위)

| 크기 | 시스템 | 검색 시간 | 생성 시간 | 총 시간 |
|------|--------|-----------|-----------|---------|
| Small | CAG | - | 0.85 | **0.85** |
| Small | In-Context | - | 9.32 | 9.32 |
| Small | Dense RAG Top-3 | 0.48 | 1.01 | 1.49 |
| Medium | CAG | - | 1.41 | **1.41** |
| Medium | In-Context | - | 26.37 | 26.37 |
| Medium | Dense RAG Top-3 | 0.41 | 0.96 | 1.37 |
| Large | CAG | - | 2.26 | **2.26** |
| Large | In-Context | - | 92.08 | 92.08 |
| Large | Dense RAG Top-3 | 0.41 | 0.93 | 1.34 |

**핵심 발견**:
- CAG는 검색 시간을 완전히 제거
- In-Context Learning 대비 **10.9배(Small), 18.7배(Medium), 40.7배(Large) 빠름**
- Dense RAG와 유사하거나 더 빠른 총 응답 시간
- 지식 크기 증가에 따라 생성 시간은 증가하지만 검색 오버헤드가 없어 효율적

### 5.4 주요 인사이트
1. **검색 오류 제거**: 전체 문서를 사전 로딩하여 검색 실패 위험 완전 제거
2. **통합 컨텍스트**: 전체 지식 컬렉션에 대한 전체론적이고 일관된 이해 제공
3. **간소화된 아키텍처**: Retriever-Generator 통합 불필요로 복잡성 감소
4. **성능 한계**: 데이터 크기가 매우 커지면 성능 격차 감소 (long-context degradation)

## 6. 우리 연구와의 관련성

### 6.1 행정문서 RAG 시스템에 대한 시사점
1. **제한적 지식 베이스 시나리오**: 서울교통공사의 내부 규정집, 업무 매뉴얼, FAQ 등은 관리 가능한 크기로, CAG 적용 가능성 높음
2. **검색 오류 민감도**: 행정 문서는 정확성이 중요하므로 검색 실패가 치명적 → CAG의 검색 오류 제거가 큰 장점
3. **응답 속도 요구사항**: 내부 챗봇/질의응답 시스템에서 실시간 검색 지연 제거는 사용자 경험 개선에 직접적 기여

### 6.2 On-Premise 환경에서의 적용 가능성
- **메모리 효율성**: KV 캐시 사전 계산으로 추론 시 메모리 사용량 최적화
- **비용 효율성**: 검색 시스템(벡터 DB, BM25 인덱스) 구축 및 유지보수 비용 절감
- **시스템 단순화**: Retriever-Generator 통합 불필요로 아키텍처 복잡도 감소

### 6.3 한국어 환경에서의 고려사항
- 논문은 영어 데이터셋(SQuAD, HotPotQA) 사용 → 한국어 long-context LLM 성능 검증 필요
- 한국어 토크나이저의 토큰 효율성에 따라 유효 컨텍스트 길이 달라질 수 있음

### 6.4 실험 설계에 대한 참고사항
1. **벤치마크 설정**: 다양한 지식 베이스 크기(Small/Medium/Large)에서 CAG vs RAG 성능 비교
2. **평가 지표**: BERTScore 외에 Faithfulness, Answer Relevancy 등 RAGAS 메트릭 활용
3. **응답 시간 측정**: 검색 시간과 생성 시간을 분리하여 측정하여 병목 지점 파악

### 6.5 Hybrid 접근법 가능성
- 논문의 Conclusion에서 언급: "Foundation context 사전 로딩 + 엣지 케이스나 특정 쿼리에만 선택적 검색"
- 우리 연구에서 기본 규정은 사전 로딩, 실시간 업데이트되는 공지사항은 검색 방식 적용 검토 가능

## 7. 인용 가능한 핵심 문장

### 7.1 RAG의 한계
> "RAG introduces challenges such as retrieval latency, potential errors in document selection, and increased system complexity."

**번역**: "RAG는 검색 지연, 문서 선택 시 잠재적 오류, 시스템 복잡성 증가와 같은 문제를 야기한다."

### 7.2 CAG의 핵심 아이디어
> "Instead of relying on a retrieval pipeline, our approach involves preloading the LLM with all relevant documents in advance and precomputing the key-value (KV) cache, which encapsulates the inference state of the LLM."

**번역**: "검색 파이프라인에 의존하는 대신, 우리의 접근법은 모든 관련 문서를 사전에 LLM에 로드하고 LLM의 추론 상태를 캡슐화하는 Key-Value(KV) 캐시를 미리 계산한다."

### 7.3 Long-context LLM의 가능성
> "This 32K to 64K context window is sufficient for storing knowledge sources such as internal company documentation, FAQs, customer support logs, and domain-specific databases, making it practical for many real-world applications."

**번역**: "32K에서 64K의 컨텍스트 윈도우는 내부 회사 문서, FAQ, 고객 지원 로그, 도메인 특화 데이터베이스와 같은 지식 소스를 저장하기에 충분하며, 많은 실제 애플리케이션에 실용적이다."

### 7.4 검색 오류 제거
> "By preloading the entire knowledge collection into the LLM provides a holistic and coherent understanding of the documents, resulting in improved response quality and consistency across a wide range of tasks."

**번역**: "전체 지식 컬렉션을 LLM에 사전 로딩함으로써 문서에 대한 전체론적이고 일관된 이해를 제공하여, 광범위한 작업에서 향상된 응답 품질과 일관성을 달성한다."

### 7.5 성능 우위
> "CAG consistently achieved the highest BERTScore in most cases, outperforming both sparse and dense RAG methods. By preloading the entire reference text from the test set, our method is immune to retrieval errors, ensuring holistic reasoning over all relevant information."

**번역**: "CAG는 대부분의 경우 가장 높은 BERTScore를 일관되게 달성하여 sparse 및 dense RAG 방법 모두를 능가했다. 테스트 세트의 전체 참조 텍스트를 사전 로딩함으로써, 우리의 방법은 검색 오류에 면역이 있으며 모든 관련 정보에 대한 전체론적 추론을 보장한다."

### 7.6 효율성 비교
> "CAG dramatically reduces generation time, particularly as the reference text length increases. This efficiency stems from preloading the KV-cache, which eliminates the need to process the reference text on the fly."

**번역**: "CAG는 특히 참조 텍스트 길이가 증가할 때 생성 시간을 극적으로 감소시킨다. 이 효율성은 KV 캐시 사전 로딩에서 비롯되며, 이는 참조 텍스트를 실시간으로 처리할 필요를 제거한다."

### 7.7 적용 범위
> "Our method requires loading all relevant documents into the model's context, making it well-suited for use cases such as internal knowledge bases of small companies, FAQs, and call centers, where the knowledge source is of a manageable size."

**번역**: "우리의 방법은 모든 관련 문서를 모델의 컨텍스트에 로딩해야 하므로, 중소기업의 내부 지식 베이스, FAQ, 콜센터와 같이 지식 소스가 관리 가능한 크기인 사용 사례에 적합하다."

### 7.8 미래 전망
> "As future models continue to expand their context length, they will be able to process increasingly larger knowledge collections in a single inference step. These two trends will significantly extend the usability of our approach."

**번역**: "미래의 모델이 컨텍스트 길이를 계속 확장함에 따라, 단일 추론 단계에서 점점 더 큰 지식 컬렉션을 처리할 수 있을 것이다. 이러한 두 가지 추세는 우리 접근법의 사용성을 크게 확장할 것이다."

## 8. 한계점 및 향후 연구방향

### 8.1 논문에서 명시한 한계점

#### 8.1.1 지식 베이스 크기 제약
- **한계**: 모든 관련 문서를 모델의 컨텍스트에 로딩해야 하므로 대규모 데이터셋에는 비실용적
- **적용 범위**: 중소기업 내부 문서, FAQ, 콜센터 지식 베이스 등 관리 가능한 크기의 지식 소스에 적합
- **완화 전망**: LLM의 컨텍스트 길이 확장 및 하드웨어 발전으로 한계 감소 예상

#### 8.1.2 Long-context Degradation
- **한계**: 데이터 크기가 매우 커지면 CAG와 RAG의 성능 격차 감소
- **근거**: Li et al. (2024) 연구에서 long-context LLM이 매우 긴 컨텍스트에서 성능 저하 발견
- **실험 결과**: Large 데이터셋에서 CAG의 성능 우위 감소 (HotPotQA Large: CAG 0.7407 vs Sparse RAG 0.7535)

#### 8.1.3 데이터셋 난이도
- **관찰**: Sparse RAG가 Dense RAG를 능가한 결과는 데이터셋이 충분히 도전적이지 않음을 시사
- **영향**: 키워드 매칭만으로도 대부분의 관련 정보를 효과적으로 캡처 가능
- **한국어 적용**: 한국어 행정문서의 특성(전문 용어, 법률 문구 등)에 따라 결과가 달라질 수 있음

### 8.2 추가 고려사항

#### 8.2.1 동적 지식 업데이트
- **한계**: 지식 베이스가 자주 업데이트되는 환경에서 KV 캐시 재계산 비용 발생
- **해결 방안**: Incremental caching 또는 hybrid 접근법 필요

#### 8.2.2 다국어 및 한국어 검증 부재
- **한계**: 영어 데이터셋(SQuAD, HotPotQA)만 사용
- **필요성**: 한국어 long-context LLM(EXAONE, Gemma-Ko 등)에서 CAG 성능 검증 필요
- **토크나이저 영향**: 한국어 토크나이저의 효율성에 따라 유효 컨텍스트 길이 달라질 수 있음

#### 8.2.3 메모리 사용량 분석 부재
- **한계**: KV 캐시의 메모리 사용량 및 디스크 저장 용량에 대한 정량적 분석 미비
- **필요성**: On-premise 환경에서 하드웨어 요구사항 산정을 위한 메모리 프로파일링 필요

### 8.3 향후 연구방향

#### 8.3.1 Hybrid CAG-RAG 접근법
논문의 Conclusion에서 제안한 방향:
> "There is potential for hybrid approaches that combine preloading with selective retrieval. For example, a system could preload a foundation context and use retrieval only to augment edge cases or highly specific queries."

**연구 방향**:
- 기본 규정/매뉴얼은 CAG로 사전 로딩
- 실시간 공지사항, 최신 업데이트는 RAG로 선택적 검색
- 쿼리 분류기를 통해 CAG/RAG 동적 선택

#### 8.3.2 한국어 행정문서 특화 평가
- 한국어 long-context LLM (EXAONE-3.5, Gemma-Ko, etc.)에서 CAG 성능 검증
- 행정문서 특성(법률 용어, 긴 문장, 복잡한 구조)이 CAG 성능에 미치는 영향 분석
- 한국어 토크나이저 효율성에 따른 유효 컨텍스트 길이 측정

#### 8.3.3 Incremental Caching 기법 개발
- 지식 베이스 일부 업데이트 시 전체 KV 캐시 재계산 없이 부분 업데이트 방법론 연구
- 문서 버전 관리 및 캐시 무효화 전략 설계

#### 8.3.4 메모리 최적화 및 압축
- KV 캐시 압축 기법 적용 (양자화, pruning 등)
- 메모리 제약 환경에서의 CAG 최적화 전략

#### 8.3.5 다양한 도메인에서의 CAG 효과 검증
- 의료, 법률, 금융 등 다양한 도메인별 지식 베이스에서 CAG vs RAG 비교
- 도메인별 최적 지식 베이스 크기 임계값 파악

#### 8.3.6 Multi-hop Reasoning 강화
- CAG가 전체 컨텍스트를 보유하는 장점을 활용하여 복잡한 다단계 추론 성능 향상 방법 연구
- HotPotQA와 같은 multi-hop QA에서 CAG의 추론 경로 분석

### 8.4 우리 연구에 적용할 실험 설계

#### Phase 1: CAG vs RAG 기본 성능 비교
- 데이터셋: AI Hub 행정문서 기계독해 데이터 (Small/Medium/Large 분할)
- 모델: EXAONE-3.5-7.8B, Gemma-Ko-12B
- 메트릭: BERTScore, Faithfulness, Answer Relevancy, Response Time

#### Phase 2: Hybrid 접근법 실험
- 시나리오 1: 기본 규정(CAG) + 실시간 공지(RAG)
- 시나리오 2: 쿼리 복잡도 기반 동적 선택 (단순 → CAG, 복잡 → Hybrid)

#### Phase 3: 메모리 및 비용 분석
- KV 캐시 메모리 사용량 측정
- On-premise 환경에서 CAG vs RAG 총 소유 비용(TCO) 비교

## 9. 참고문헌 연결성

### 9.1 본 논문이 인용한 주요 문헌
- **RAG 기초**: Lewis et al. (2020) - Retrieval-augmented generation for knowledge-intensive NLP tasks
- **Long-context 성능**: Li et al. (2024) - Long-context LLMs Struggle with Long In-context Learning
- **RAG vs Long-context**: Leng et al. (2024), Li et al. (2024) - Long Context RAG Performance
- **KV Caching**: Lu et al. (2024) - TurboRAG: Accelerating RAG with Precomputed KV Caches

### 9.2 우리 연구에 함께 인용할 문헌
- **Graph RAG**: Han et al. (2025) - RAG vs. GraphRAG: A Systematic Evaluation
  - CAG, RAG, GraphRAG 3-way 비교 가능성
- **Korean RAG**: 이채원 (2025) - 한국어 Hybrid RAG 기반 질의응답 시스템
  - 한국어 환경에서 CAG 적용 시 참고
- **On-Premise LLM**: Pan & Wang (2025) - Cost-Benefit Analysis of On-Premise LLM Deployment
  - CAG의 비용 효율성 분석 시 참고

## 10. 결론 및 인사이트

본 논문은 long-context LLM의 발전이 RAG 패러다임에 근본적인 변화를 가져올 수 있음을 실증적으로 보여준다. 특히 **제한적이고 관리 가능한 지식 베이스**를 다루는 우리 연구(서울교통공사 행정문서 RAG 시스템)와 매우 높은 관련성을 가진다.

### 우리 연구에 주는 핵심 시사점:
1. **검색 없는 QA 시스템 가능성**: 내부 규정집 크기가 Llama 3.1의 유효 컨텍스트(32K-64K 토큰) 내에 있다면 RAG 없이 CAG만으로 구현 가능
2. **하이브리드 접근법 설계 근거**: 정적 지식(CAG) + 동적 지식(RAG) 조합으로 최적의 효율성과 정확성 달성 가능
3. **평가 실험 설계 참고**: 다양한 지식 베이스 크기에서 응답 시간과 정확도 trade-off 분석 필요
4. **한국어 검증 필요성**: 영어 기반 결과를 한국어 환경에 직접 적용하기 전 검증 실험 필수

이 논문은 "RAG가 항상 정답은 아니다"라는 중요한 메시지를 전달하며, 문제의 특성(지식 베이스 크기, 업데이트 빈도, 쿼리 유형)에 따라 최적의 접근법이 달라질 수 있음을 시사한다. 우리 연구에서는 이를 바탕으로 **CAG vs RAG vs Hybrid** 3-way 비교 실험을 수행하여 한국어 행정문서 환경에 최적화된 솔루션을 제시할 수 있을 것이다.
