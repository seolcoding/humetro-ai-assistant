# RAG 시스템 성능 비교 실험 중간 보고

## 1. 실험 개요

서울 교통 도메인 특화 AI 어시스턴트 개발을 위해 **3가지 RAG(Retrieval-Augmented Generation) 방법론**의 성능을 비교 평가하는 실험을 진행하고 있습니다. 현재 2/3 실험이 완료되었으며, 유의미한 초기 결과를 확인하였습니다.

## 2. 실험 설계

### 2.1 평가 대상 RAG 방법론

1. **Naive RAG** (벡터 유사도 기반)
   - FAISS 벡터 저장소 활용 (5,879 chunks, 3072D embeddings)
   - 질문 임베딩과 문서 임베딩 간 코사인 유사도로 검색

2. **KG RAG Simple** (지식 그래프 + 고정 Cypher)
   - Neo4j 그래프 DB (6,544 nodes, 9,554 relationships)
   - 사전 정의된 Cypher 쿼리로 그래프 순회
   - 노드 임베딩 벡터 유사도 기반 검색

3. **KG RAG Cypher Generation** (LLM 생성 Cypher) - *진행중*
   - GPT-4o-mini가 질문에 맞는 Cypher 쿼리 동적 생성
   - 질문 유형에 따라 최적화된 그래프 탐색

### 2.2 평가 모델 (5개)

| 모델 | 파라미터 | 제공자 | 특징 |
|------|---------|--------|------|
| GPT-4o-mini | - | OpenAI | 범용 성능 기준선 |
| EXAONE-3.5-7.8B | 7.8B | LG AI Research | 한국어 특화 |
| Qwen3-8B | 8B | Alibaba | 다국어 지원 |
| Gemma3-12B | 12B | Google | 오픈소스 대형 모델 |
| GPT-OSS-20B | 20B | - | 대형 오픈소스 모델 |

### 2.3 평가 데이터셋

**골든 테스트셋**: 50개 질문 (균형 잡힌 난이도 분포)
- Single-hop (25개): 단순 정보 검색
  - 예: "여의도역 우회에 대한 중요정보는?"
- Multi-hop (25개): 복합 추론 필요
  - 예: "서울의 장거리 버스 노선 개선이 승객 안전과 운전기사 근무 환경을 어떻게 향상시킬 것으로 예상되나요?"

### 2.4 평가 지표 (RAGAS Framework)

- **Faithfulness**: 검색된 컨텍스트에 기반한 답변의 사실 정확도
- **Answer Relevancy**: 질문과 답변의 관련성
- **Answer Correctness**: 정답(ground truth) 대비 답변 품질

## 3. 중간 결과 (Naive RAG vs KG Simple)

### 3.1 종합 성능 비교

**평균 점수 (전체 모델, 50개 질문)**

| 메트릭 | Naive RAG | KG Simple | 차이 |
|--------|-----------|-----------|------|
| Faithfulness | 0.718 | **0.778** | +8.4% |
| Relevancy | **0.658** | 0.617 | -6.2% |
| Correctness | 0.539 | 0.528 | -2.0% |

### 3.2 주요 발견사항

#### (1) Faithfulness: KG Simple 우세 (+8.4%)
- **KG Simple**이 그래프 구조를 통해 더 정확한 정보 제공
- GPT-4o-mini: 0.688 → 0.778 (+13%)
- EXAONE-3.5: 0.742 → 0.838 (+13%)

#### (2) Relevancy: Naive RAG 우세 (+6.2%)
- **Naive RAG**이 질문과 직접적으로 연관된 답변 생성
- 벡터 유사도 기반 검색이 질문-답변 관련성에 효과적

#### (3) Multi-hop 질문에서 더 나은 성능
- 모든 모델이 Multi-hop 질문에서 Single-hop보다 높은 점수
- Gemma3-12B의 Multi-hop Faithfulness: **0.891** (최고 점수)

### 3.3 모델별 성능

**최고 성능 모델**

| 질문 유형 | Faithfulness | Relevancy | Correctness |
|-----------|-------------|-----------|-------------|
| **Single-hop** | EXAONE-3.5 (KG) | GPT-OSS-20B (Naive) | GPT-4o-mini |
| **Multi-hop** | Gemma3-12B (KG) | GPT-4o-mini (KG) | GPT-4o-mini |

**모델별 특징**
- **GPT-4o-mini**: 균형 잡힌 성능, Correctness 최고
- **Gemma3-12B**: Faithfulness 최고 (Multi-hop 0.891)
- **EXAONE-3.5**: KG Simple에서 Single-hop Faithfulness 최고 (0.838)

## 4. 진행 상황 및 향후 계획

### 4.1 현재 진행 상황

✅ **완료** (2/3)
- Naive RAG: 250 evaluations (5 models × 50 questions)
- KG Simple: 250 evaluations

🔄 **진행 중** (1/3)
- KG Cypher Generation: 예상 완료 시간 ~2시간

### 4.2 예상되는 KG Cypher Generation 특징

**장점**
- 질문 유형에 맞는 동적 쿼리 생성
- 복잡한 관계 추론에 유리할 것으로 예상

**단점**
- LLM Cypher 생성 비용 (GPT-4o-mini 사용)
- Cypher 문법 오류 가능성

### 4.3 최종 분석 계획

KG Cypher Generation 완료 후:
1. **3-way 성능 비교**: Naive vs KG Simple vs KG Cypher
2. **비용-성능 트레이드오프 분석**: API 비용 대비 성능 향상
3. **도메인 특성 분석**: 서울 교통 질문 유형별 최적 방법론
4. **최종 권고안**: 프로덕션 배포를 위한 RAG 전략 수립

## 5. 초기 결론

1. **KG 활용의 효과**: 그래프 구조가 사실 정확도(Faithfulness) 향상에 기여
2. **단순함의 가치**: Naive RAG도 답변 관련성에서 경쟁력 보유
3. **모델 선택의 중요성**: 도메인과 평가 지표에 따라 최적 모델 상이
4. **복합 추론 강점**: Multi-hop 질문에서 전반적으로 더 나은 성능

최종 실험 완료 후 상세 분석 보고서를 제출하겠습니다.
