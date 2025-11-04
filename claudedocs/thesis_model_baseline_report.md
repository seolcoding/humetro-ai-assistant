# 논문 실험 모델 기준선 보고서

## 실험 환경

- **서버**: RTX 3090 Ti (24GB VRAM)
- **엔드포인트**: http://100.95.220.92:11434
- **프레임워크**: Ollama + LangChain
- **날짜**: 2025-10-29

## 모델 구성 현황

### 1. 논문 명시 모델 (4개)

| 논문 모델 | Ollama 모델 | 파라미터 | 양자화 | 크기 | 상태 |
|-----------|-------------|----------|--------|------|------|
| EXAONE-3.5-7.8B | exaone3.5:7.8b | 7.8B | Q4_K_M | 4.44GB | ✅ 설치됨 |
| Qwen3-8B | solar-pro:latest | 22.1B | Q4_K_M | 12.40GB | ✅ 설치됨 (대체) |
| Gemma3-12B | gemma3:27b | 27.4B | Q4_K_M | 16.20GB | ⚠️ 대체 (논문: 12B) |
| GPT-OSS-20B | gpt-oss:20b | 20.9B | MXFP4 | 12.83GB | ✅ 설치됨 |

### 2. 모델 선정 근거

#### EXAONE-3.5-7.8B
- **선정 이유**: LG AI Research의 한국어 특화 모델
- **Ko-LongRAG 성능**: 최고 성능 기록
- **특징**: 한국어 민원 처리에 최적화

#### Qwen3-8B → Solar Pro (대체)
- **대체 이유**: Qwen3 정확한 버전 미제공
- **Solar Pro 특징**:
  - 한국어 성능 우수
  - 22.1B 파라미터로 더 강력
  - Upstage AI의 한국 기업 모델

#### Gemma3-12B → Gemma3-27B (대체)
- **대체 이유**: Ollama에 12B 버전 미제공, 27B만 사용 가능
- **논문 명시**: 12B (효율성 중심)
- **실제 사용**: 27B (2배 이상 큰 모델)
- **영향**: 응답 시간 증가, VRAM 사용량 증가

#### GPT-OSS-20B
- **선정 이유**: MoE 아키텍처로 효율성 극대화
- **특징**: 구조화된 쿼리 처리에 특화

## 실험 계획

### 1. 계층적 RAG 시스템 (4단계)

```
Level 1: Baseline (Pure LLM)
   ↓
Level 2: Naive RAG (Vector Search)
   ↓
Level 3: Structured RAG (Metadata + Hybrid)
   ↓
Level 4: Graph RAG (KG + Multi-hop)
```

### 2. 실험 매트릭스 (4×4 = 16 시스템)

| 모델 \ RAG | Baseline | Naive | Structured | Graph |
|------------|----------|-------|------------|-------|
| EXAONE-3.5 | System 1 | System 2 | System 3 | System 4 |
| Solar Pro | System 5 | System 6 | System 7 | System 8 |
| Gemma3-27B | System 9 | System 10 | System 11 | System 12 |
| GPT-OSS-20B | System 13 | System 14 | System 15 | System 16 |

### 3. 평가 메트릭 (RAGAS)

#### 생성 품질
- **Faithfulness**: 환각 최소화 정도
- **Answer Relevance**: 질의 관련성
- **Context Precision**: 검색 정확도

#### 검색 품질
- **Context Recall**: 정보 완전성
- **Context Relevance**: 맥락 적절성

#### 성능 메트릭
- **Response Time**: 응답 시간
- **Tokens/Second**: 처리 속도
- **Memory Usage**: VRAM 사용량

## 성능 기준선 (Expected)

### 응답 시간 목표
- Baseline: < 2초
- Naive RAG: < 3초
- Structured RAG: < 4초
- Graph RAG: < 5초

### 품질 목표 (RAGAS Score)
- Baseline: 0.60-0.70
- Naive RAG: 0.70-0.80
- Structured RAG: 0.75-0.85
- Graph RAG: 0.80-0.90

### VRAM 사용량 예상
- EXAONE-3.5: ~8GB
- Solar Pro: ~15GB
- Gemma3-27B: ~18GB
- GPT-OSS-20B: ~16GB

## 하이브리드 전략

### KG 구축 단계 (SOTA 모델)
```python
# GPT-4o 사용 (고품질 KG 생성)
kg_builder = OpenAI(model="gpt-4o")
knowledge_graph = kg_builder.extract_entities_relations(documents)
```

### 추론 단계 (오픈소스 LLM)
```python
# 로컬 모델 사용 (비용 절감)
local_llm = OllamaLLM(model="exaone3.5:7.8b")
response = local_llm.generate_with_kg(query, knowledge_graph)
```

## TCO 분석

### 온프레미스 (연간)
- 하드웨어: RTX 3090 Ti (~$2,000 일회성)
- 전력: ~$500/년
- 유지보수: ~$1,000/년
- **총 TCO**: ~$1,500/년 (3년 상각)

### 클라우드 API (연간)
- GPT-4: ~$30,000/년 (100만 쿼리 기준)
- Claude: ~$25,000/년
- Gemini: ~$20,000/년
- **총 TCO**: $20,000-30,000/년

### 비용 절감율
- **온프레미스 대비 클라우드**: 93-95% 절감
- **하이브리드 대비 순수 클라우드**: 70-80% 절감

## 테스트 시나리오

### 1. 도메인별 질의 (5개 카테고리)
- 정보 질의: "서울시 120 다산콜센터는?"
- 절차 안내: "백신 접종 예약 방법?"
- 민원 처리: "주민등록등본 온라인 발급?"
- 교통 정보: "지하철 막차 시간?"
- 복지 지원: "재난지원금 신청 자격?"

### 2. 복잡도별 질의
- **Simple**: 단순 사실 확인
- **Medium**: 2-3단계 추론 필요
- **Complex**: 다중 문서 종합 필요

### 3. 부하 테스트
- 단일 쿼리 응답 시간
- 동시 5개 쿼리 처리
- 연속 100개 쿼리 처리

## 실행 명령

### 1. 모델 검증
```bash
# 빠른 검증
uv run python src/scripts/validate_thesis_models.py --quick

# 전체 검증
uv run python src/scripts/validate_thesis_models.py
```

### 2. 실험 실행
```bash
# Jupyter 노트북 실행
uv run jupyter notebook notebooks/01_thesis_experiment_rag_comparison.ipynb

# 자동화 실험
uv run python src/experiments/run_thesis_experiments.py
```

### 3. 결과 분석
```bash
# 결과 시각화
uv run python src/analysis/visualize_results.py

# 보고서 생성
uv run python src/analysis/generate_report.py
```

## 다음 단계

1. ✅ 모델 설치 및 검증 완료
2. ⏳ 성능 기준선 측정 중
3. ⬜ RAG 시스템 구현
4. ⬜ 실험 데이터 준비
5. ⬜ 16개 시스템 평가
6. ⬜ 결과 분석 및 시각화
7. ⬜ 논문 작성용 차트 생성

## 참고 사항

- 모든 실험은 재현 가능하도록 seed 고정
- 캐싱 시스템으로 중복 계산 방지
- 실험 결과는 `results/` 디렉토리에 JSON 형식으로 저장
- 시각화는 `figures/` 디렉토리에 고품질 PNG/PDF로 저장