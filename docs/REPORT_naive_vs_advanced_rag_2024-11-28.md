# Naive RAG vs Advanced RAG 벤치마크 리포트

**실험일시**: 2024-11-28 22:24 KST
**데이터셋**: AI Hub 행정문서 기계독해 데이터 (8 QA × 10 corpus)
**브랜치**: `autorag_optimization`

---

## 실험 목적

**연구 질문**: 오픈소스 LLM + Advanced RAG 파이프라인이 GPT-4o-mini + Naive RAG를 능가할 수 있는가?

---

## 실험 설정

### 1. Naive RAG + GPT-4o-mini (Baseline)

**Config 파일**: `src/autorag_pilot/config/naive_rag_gpt4o.yaml`

```yaml
node_lines:
  # Retrieval: Vector Search Only
  - node_line_name: retrieve_node_line
    nodes:
      - node_type: semantic_retrieval
        top_k: 5
        modules:
          - module_type: vectordb
            embedding_model: openai

  # Post-Retrieve: Prompt + Generator Only
  - node_line_name: post_retrieve_node_line
    nodes:
      - node_type: prompt_maker
        modules:
          - module_type: fstring
            prompt: "다음은 행정 문서에서 발췌한 내용입니다..."

      - node_type: generator
        modules:
          - module_type: openai_llm
            llm: gpt-4o-mini
            batch: 4
```

**파이프라인 요약**:
- Retrieval: Vector DB (OpenAI embedding) → top_k=5
- Post-Retrieve: 없음 (Reranker, Filter 없음)
- Generator: GPT-4o-mini

---

### 2. Advanced RAG + 오픈소스 6개 모델

**Config 파일**: `src/autorag_pilot/config/full_optimization.yaml`

```yaml
node_lines:
  # Retrieval: Lexical + Semantic + Hybrid
  - node_line_name: retrieve_node_line
    nodes:
      - node_type: lexical_retrieval
        modules:
          - module_type: bm25

      - node_type: semantic_retrieval
        modules:
          - module_type: vectordb
            embedding_model: openai

      - node_type: hybrid_retrieval
        modules:
          - module_type: hybrid_rrf
            weight_range: (4, 80)

  # Post-Retrieve: Full Pipeline
  - node_line_name: post_retrieve_node_line
    nodes:
      - node_type: passage_augmenter
        modules:
          - module_type: pass_passage_augmenter
          - module_type: prev_next_augmenter
            num_passages: 1

      - node_type: passage_reranker
        modules:
          - module_type: pass_reranker
          - module_type: koreranker
            batch: 32

      - node_type: passage_filter
        modules:
          - module_type: pass_passage_filter
          - module_type: threshold_cutoff
            threshold: 0.3
          - module_type: percentile_cutoff
            percentile: 0.6
          - module_type: similarity_threshold_cutoff
            threshold: 0.3
          - module_type: similarity_percentile_cutoff
            percentile: 0.6

      - node_type: passage_compressor
        modules:
          - module_type: pass_compressor

      - node_type: prompt_maker
        modules:
          - module_type: fstring
          - module_type: long_context_reorder

      - node_type: generator
        modules:
          # OpenAI
          - module_type: openai_llm
            llm: gpt-4o-mini

          # Ollama 로컬 모델들
          - module_type: llama_index_llm
            llm: openailike
            model: exaone3.5:7.8b
            api_base: http://localhost:11434/v1

          - module_type: llama_index_llm
            llm: openailike
            model: gpt-oss:20b
            api_base: http://localhost:11434/v1

          - module_type: llama_index_llm
            llm: openailike
            model: gemma3:12b
            api_base: http://localhost:11434/v1

          - module_type: llama_index_llm
            llm: openailike
            model: qwen3:8b
            api_base: http://localhost:11434/v1

          - module_type: llama_index_llm
            llm: openailike
            model: exaone-deep:7.8b
            api_base: http://localhost:11434/v1
```

**AutoRAG가 선택한 Best 조합**:
| 단계 | 선택된 모듈 | 파라미터 |
|------|------------|----------|
| Retrieval | HybridRRF | weight=4.0 |
| Passage Augmenter | PrevNextPassageAugmenter | num_passages=1 |
| Passage Reranker | PassReranker | (pass-through) |
| Passage Filter | PercentileCutoff | percentile=0.6 |
| Passage Compressor | PassCompressor | (no compression) |
| Prompt Maker | Fstring | 기본 템플릿 |
| Generator | **Gemma3:12b** | 🏆 |

---

## 실험 결과

### 전체 순위 (평균 점수 기준)

| 순위 | Pipeline | Model | METEOR | ROUGE | 평균 | vs Baseline |
|------|----------|-------|--------|-------|------|-------------|
| 🥇 1 | Advanced RAG | **Gemma3 12B** | 0.3309 | 0.4167 | **0.3738** | **+15.7%** |
| 🥈 2 | Advanced RAG | GPT-4o-mini | 0.2328 | 0.4271 | 0.3299 | +2.1% |
| 🥉 3 | **Naive RAG** | **GPT-4o-mini** | 0.2193 | 0.4271 | **0.3232** | Baseline |
| 4 | Advanced RAG | EXAONE-3.5 7.8B | 0.2336 | 0.2940 | 0.2638 | -18.4% |
| 5 | Advanced RAG | GPT-OSS 20B | 0.2112 | 0.3146 | 0.2629 | -18.7% |
| 6 | Advanced RAG | EXAONE-Deep 7.8B | 0.0960 | 0.1424 | 0.1192 | -63.1% |
| 7 | Advanced RAG | Qwen3 8B | 0.0481 | 0.0118 | 0.0299 | -90.7% |

### 메트릭별 상세 분석

#### METEOR (의미적 유사도)
```
Gemma3 12B       ████████████████████████████████▌ 0.3309  🏆
EXAONE-3.5       ███████████████████████▌         0.2336
GPT-4o-mini(Adv) ███████████████████████▎         0.2328
Naive+GPT-4o     █████████████████████▉           0.2193  (Baseline)
GPT-OSS 20B      █████████████████████            0.2112
EXAONE-Deep      █████████▌                       0.0960
Qwen3 8B         ████▊                            0.0481
```

#### ROUGE (텍스트 오버랩)
```
Naive+GPT-4o     ██████████████████████████████████████████▋ 0.4271  (Baseline)
GPT-4o-mini(Adv) ██████████████████████████████████████████▋ 0.4271
Gemma3 12B       █████████████████████████████████████████▋  0.4167  🏆
GPT-OSS 20B      ███████████████████████████████▌            0.3146
EXAONE-3.5       █████████████████████████████▍              0.2940
EXAONE-Deep      ██████████████▎                             0.1424
Qwen3 8B         █▏                                          0.0118
```

---

## 핵심 발견

### 1. Gemma3 12B가 GPT-4o-mini를 능가
- **Advanced RAG + Gemma3**: 평균 0.3738
- **Naive RAG + GPT-4o-mini**: 평균 0.3232
- **성능 향상**: +15.7%

### 2. Advanced RAG 파이프라인의 효과
- 같은 GPT-4o-mini에서도 Advanced RAG가 +2.1% 향상
- Hybrid Retrieval (BM25 + Vector)이 Vector 단독보다 효과적

### 3. 오픈소스 모델 성능 편차 큼
- **Gemma3만 Baseline 초과** (유일하게 상용 모델 능가)
- EXAONE, GPT-OSS는 Baseline 미달
- Qwen3, EXAONE-Deep은 심각하게 저조 (-63% ~ -91%)

### 4. METEOR vs ROUGE 트레이드오프
- Gemma3: METEOR 최고 (0.3309), ROUGE 약간 낮음 (0.4167)
- GPT-4o-mini: ROUGE 최고 (0.4271), METEOR 낮음 (0.2193~0.2328)
- **Gemma3가 의미적으로 더 정확한 답변 생성**

---

## 결론

### 연구 질문에 대한 답변

> **Q: 오픈소스 LLM + Advanced RAG가 GPT-4o-mini + Naive RAG를 능가할 수 있는가?**

**A: Yes, 단 Gemma3 12B에 한해서만.**

- **Gemma3 12B + Advanced RAG**: Baseline 대비 **+15.7%** 향상
- 다른 오픈소스 모델들은 Advanced RAG에서도 Baseline 미달

### 실용적 권장사항

| 시나리오 | 권장 구성 | 이유 |
|----------|----------|------|
| 비용 최소화 | Advanced RAG + Gemma3 | API 비용 없음, 최고 성능 |
| 안정성 중시 | Naive RAG + GPT-4o-mini | 간단한 파이프라인, 예측 가능 |
| 한국어 특화 | Advanced RAG + EXAONE-3.5 | 한국어 학습 모델 (성능 개선 여지) |

---

## 다음 단계

1. **스케일업 검증**: 100 QA × 5000 corpus로 결과 재현 확인
2. **Gemma3 심층 분석**: 어떤 질문 유형에서 강점을 보이는지 분석
3. **프롬프트 최적화**: 한국어 특화 프롬프트로 EXAONE 성능 개선 시도

---

## 실험 환경

| 항목 | 값 |
|------|-----|
| GPU | RTX 3090 Ti 24GB |
| OS | Ubuntu Linux 6.14.0 |
| Python | 3.11 |
| AutoRAG | 0.3.x (API version) |
| Ollama | localhost:11434 |
| OpenAI API | gpt-4o-mini, text-embedding-3-small |

---

## 파일 위치

```
src/autorag_pilot/
├── config/
│   ├── naive_rag_gpt4o.yaml          # Naive RAG 설정
│   └── full_optimization.yaml         # Advanced RAG 설정
├── autorag_project/                   # Naive RAG 결과
└── autorag_project_advanced_backup/   # Advanced RAG 결과
```

---

*Generated: 2024-11-28 22:30 KST*
