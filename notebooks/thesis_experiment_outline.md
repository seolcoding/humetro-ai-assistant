# 논문 실험 노트북 아웃라인

## 📚 실험 노트북 구조 (논문 연구 방법론 기반)

### 1. 데이터 준비 및 전처리
```
1.1 Deduplicated 데이터 로드
1.2 데이터 품질 검증 및 통계 분석
1.3 Train/Test 분할 (8:2 비율)
1.4 캐시 시스템 구축 (중간 결과물 저장)
```

### 2. 평가 데이터셋 생성 (RAGAS Framework)
```
2.1 RAGAS를 활용한 평가 데이터 생성
    - GPT-4o 활용 (캐시 확인 후 생성)
    - Question Generation
    - Ground Truth Generation
    - Context Generation
2.2 평가 데이터 품질 검증
2.3 평가 메트릭 정의
    - Context Precision/Recall
    - Answer Relevancy
    - Faithfulness
    - Answer Correctness
```

### 3. 지식 그래프 구축 (하이브리드 전략 구현)
```
3.1 SOTA 모델(GPT-4o)을 활용한 고품질 KG 구축
    - Entity Extraction
    - Relation Extraction
    - Triple Generation
    - KG Quality Validation
3.2 Neo4j 데이터베이스 구축
    - Node/Edge 스키마 정의
    - 데이터 Import
    - Index 생성 (Vector + Graph)
3.3 KG 품질 메트릭 측정
    - Coverage
    - Connectivity
    - Semantic Coherence
```

### 4. 계층적 RAG 시스템 구현 (4단계)
```
4.1 Level 1: Baseline (Pure LLM, RAG 없음)
4.2 Level 2: Naive RAG (Vector Search Only)
    - Embedding 모델 선택
    - Vector DB 구축 (ChromaDB/Pinecone)
    - Similarity Search 구현
4.3 Level 3: Structured RAG (Metadata + Hybrid Search)
    - Metadata Filtering
    - BM25 + Vector Search
    - Re-ranking 전략
4.4 Level 4: Graph RAG (KG + Multi-hop Reasoning)
    - Graph Traversal
    - Relation-based Pruning
    - Context Aggregation
```

### 5. 모델별 실험 수행
```
5.1 오픈소스 LLM 설정 (4개 모델)
    - EXAONE-3.5 (7.8B/32B)
    - Qwen 3 (8B)
    - Gemma 3 (12B)
    - GPT-OSS (20B) [필요시 추가]
    - Quantization 설정 (4-bit/8-bit for 24GB VRAM)

5.2 베이스라인 상용 API
    - GPT-4o-mini + Naive RAG
    - Gemini 2.5 Flash + Naive RAG [선택]

5.3 실험 매트릭스 (4 모델 × 4 RAG = 16 시스템)
    - 각 조합별 추론 실행
    - 결과 저장 (JSON/CSV)
    - 실행 시간 측정
```

### 6. 평가 및 분석
```
6.1 RAGAS 자동 평가
    - 각 시스템별 메트릭 계산
    - Statistical Significance Test

6.2 LLM-as-a-Judge (GPT-4o)
    - 정성 평가
    - Human Preference Alignment

6.3 성능 비교 분석
    - 모델별 성능 비교
    - RAG 방식별 성능 비교
    - 교차 분석 (Model × RAG interaction)
```

### 7. Ablation Study
```
7.1 최고 성능 모델 선정
7.2 Component-wise Ablation
    - Vector Only
    - KG Only
    - Vector + KG (Hybrid)
    - Metadata Impact
    - Multi-hop Impact

7.3 Contribution Analysis
    - 각 구성요소의 한계 기여도 측정
    - Statistical Analysis
```

### 8. 비용-효율성 분석
```
8.1 추론 비용 계산
    - 온프레미스: 전력 비용 + 하드웨어 감가상각
    - 상용 API: Token 사용량 × 단가

8.2 TCO (Total Cost of Ownership) 분석
    - 초기 투자 비용
    - 5년 운영 비용 예측
    - Break-even Point 계산

8.3 성능 대비 비용 효율성
    - Cost per Query
    - Performance/Dollar Ratio
```

### 9. 결과 시각화 및 보고서
```
9.1 성능 시각화
    - Radar Chart (Multi-metric comparison)
    - Heatmap (Model × RAG performance)
    - Box Plot (Statistical distribution)

9.2 비용 분석 차트
    - TCO Comparison
    - ROI Timeline

9.3 최종 보고서 생성
    - Executive Summary
    - Technical Details
    - Recommendations
```

## 📊 예상 결과물

1. **16개 시스템 성능 비교표**
2. **Ablation Study 결과**
3. **TCO/ROI 분석 보고서**
4. **최적 구성 추천**
   - 성능 우선
   - 비용 우선
   - 균형 접근

## 🔧 기술 스택

- **LLM Framework**: LangChain, vLLM
- **Vector DB**: ChromaDB, Pinecone
- **Graph DB**: Neo4j
- **Evaluation**: RAGAS, Custom Metrics
- **Visualization**: Plotly, Seaborn
- **Hardware**: RTX 3090Ti (24GB VRAM)

## ⏰ 예상 소요 시간

- 데이터 준비: 2시간
- KG 구축: 4-6시간 (API 호출 포함)
- 실험 수행: 8-10시간
- 평가 및 분석: 4시간
- 총 예상: 18-22시간

## 💾 캐싱 전략

```python
cache_structure = {
    "embeddings/": "벡터 임베딩 캐시",
    "kg_triples/": "지식 그래프 트리플",
    "ragas_eval/": "평가 데이터셋",
    "model_outputs/": "모델별 추론 결과",
    "metrics/": "평가 메트릭"
}
```

이 구조는 논문의 연구 방법론을 충실히 구현하면서도 실제 실험의 효율성을 고려한 설계입니다.