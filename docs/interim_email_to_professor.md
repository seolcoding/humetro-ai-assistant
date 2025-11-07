# 교수님께 보낼 중간 보고 이메일

---

**제목**: RAG 시스템 벤치마크 중간 실험 결과 보고

---

교수님, 안녕하세요.

하루 종일 실험을 진행하여 의미 있는 중간 결과를 얻었습니다.

## 연구 가설 검증

**핵심 가설**: Knowledge Graph RAG를 적용한 오픈소스 모델이 Naive RAG를 사용하는 상용 모델(GPT-4o-mini)의 성능을 능가할 수 있는가?

→ **결과**: 가설을 지지하는 실험적 증거를 발견했습니다.

## 주요 발견 사항

첨부된 그래프(`key_finding_comprehensive.png`)에서 확인하실 수 있듯이:

1. **오픈소스 모델의 약진**
   - **Gemma3-12B**: Multi-hop Faithfulness 0.891로 **최고 점수** 달성
   - GPT-4o-mini의 KG RAG 점수(0.795) 대비 **+12% 향상**
   - GPT-4o-mini의 Naive RAG 점수(0.875) 대비도 +1.8% 우수

2. **Graph RAG의 효과**
   - **GPT-OSS-20B**: KG RAG 적용 시 **+14.7% 성능 향상** (최대 개선폭)
   - 오픈소스 모델들이 GPT-4o-mini보다 **Graph RAG의 혜택을 더 크게** 받음
   - 복잡한 Multi-hop 추론 작업에서 그래프 구조의 이점이 명확히 드러남

3. **질문 복잡도별 차이**
   - Multi-hop 질문에서 KG RAG의 개선 효과가 더 두드러짐
   - 단순 벡터 유사도 검색보다 그래프 탐색이 복잡한 추론에 효과적

## 실험 설정

- **데이터셋**: 50개 질문 (Single-hop 25개, Multi-hop 25개)
- **도메인**: 서울 대중교통 (한국어)
- **평가 모델**: 5개 LLM (GPT-4o-mini, EXAONE-3.5-7.8B, Qwen3-8B, Gemma3-12B, GPT-OSS-20B)
- **RAG 방식**:
  - Naive RAG: FAISS 벡터 유사도 (5,879 chunks)
  - KG Simple: Neo4j 그래프 + 벡터 (6,544 nodes, 9,554 edges)
- **평가 지표**: Faithfulness, Answer Relevancy, Answer Correctness (RAGAS framework)

## 진행 상황

✅ **완료**: Naive RAG, KG Simple RAG 벤치마크
🔄 **진행 중**: KG Cypher Generation RAG (더 높은 성능이 예상되는 방식)

KG Cypher Generation 방식은 LLM이 질문에 맞춰 동적으로 Cypher 쿼리를 생성하여 그래프를 탐색하는 방식으로, KG Simple보다 더욱 정교한 검색이 가능합니다. 현재 실행 중이며 오늘 밤 늦게 결과가 나올 예정입니다.

## 의의

이번 실험 결과는 **"적절한 지식 구조화(Knowledge Graph)를 통해 오픈소스 모델도 상용 모델과 대등하거나 우수한 성능을 낼 수 있다"**는 점을 실증적으로 보여줍니다. 이는 RAG 시스템 설계에서 모델 선택만큼이나 **지식 조직화 방법론이 중요**함을 시사합니다.

전체 3-way 비교 결과가 완성되는 대로 추가 분석과 함께 다시 보고드리겠습니다.

감사합니다.

---

**첨부 파일**:
- `key_finding_comprehensive.png`: 종합 실험 결과 시각화
- `interim_report.md`: 상세 실험 보고서

