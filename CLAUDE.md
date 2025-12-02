# Current SYSTEM SPEC
- RTX 3090Ti 24GB
- Ubuntu
- CUDA 12.1
- Python 3.11
# CRAWLING FRAMEWORK

- use crawl4ai for every crawling task. do not use any other tool.
- refer crawl4ai doc, skill, examples whenever needed.

# GPT-5 WITH RAGAS EVALUATION

## Issue: GPT-5 Temperature Parameter Incompatibility

GPT-5 only supports `temperature=1` (default value). RAGAS framework internally sets `temperature=1e-8` for deterministic evaluation, causing GPT-5 to reject requests with error:
```
Unsupported value: 'temperature' does not support 1E-8 with this model. Only the default (1) value is supported.
```

## Root Cause

RAGAS evaluation flow:
```
Script → RAGAS evaluate() → LangchainLLMWrapper → ChatOpenAI → OpenAI API
```

- RAGAS uses LangChain's `ChatOpenAI` to call OpenAI API directly
- LiteLLM `drop_params` setting doesn't apply (RAGAS bypasses LiteLLM)
- RAGAS internal code sets `temperature=1e-8` for consistency

## Solution: Patch RAGAS Library

Modify `.venv/lib/python3.12/site-packages/ragas/llms/base.py`:

```python
def get_temperature(self, n: int) -> float:
    """Return the temperature to use for completion based on n."""
    return None  # Force None for GPT-5 compatibility
    # Original: return 0.3 if n > 1 else 1e-8
```

## Test Results

After patching:
- ✅ GPT-5 as judge: Successfully evaluated 5 LLMs across 5 questions
- ✅ Metrics computed: Faithfulness, Answer Relevancy, Answer Correctness
- 🏆 Best performers:
  - Faithfulness: Gemma3-12B (0.900)
  - Answer Relevancy: GPT-4o-mini (0.966)
  - Answer Correctness: EXAONE-3.5-7.8B (0.410)

## Important Notes

- This is a **temporary workaround** requiring library modification
- Future RAGAS updates will overwrite this change
- Consider creating a wrapper or fork for production use
- Alternative: Use GPT-4o as judge (supports temperature parameter)

---

# 🔴 CRITICAL: KG CYPHER RAG FIX (2025-11-06)

## Checkpoint: Experimental Design Ground Truth

**Status**: ✅ FIXED | **Impact**: 🔴 CRITICAL - New baseline for all experiments

### Problem Discovered

KG Cypher Generation had **-47% performance degradation** due to missing vector similarity starting point.

```
OLD Implementation (FAILED):
- Pure LLM Cypher generation
- No vector search → No starting point
- Random graph walk → Empty or irrelevant results
- Result: 0.398 faithfulness (vs 0.746 Naive RAG)

ROOT CAUSE: Missing vector similarity search for initial node selection
```

### Solution Implemented

**Hybrid Vector-Graph Architecture** (`src/kg_agent/kg_cypher_retriever.py`):

```python
1. Vector Similarity Search → Find top-k relevant nodes (STARTING POINT)
2. Graph Expansion → Explore 1-hop neighbors from found nodes
3. Original Text Return → Preserve chunk quality (no LLM summary)
```

### Verified Results

**Quick Test (10Q, 3M):**
```
OLD Cypher:     0.398  (-47% vs Naive)
FIXED Cypher:   0.830  (+11% vs Naive, +6% vs KG Simple)
Improvement:    +108.5% 🚀
```

**Model Performance (Multi-hop):**
```
EXAONE-3.5-7.8B:  0.893 (+179% improvement) 🏆
GPT-4o-mini:      0.802 (+122% improvement)
GPT-OSS-20B:      0.795 (+101% improvement)
```

### New Ground Truth

**RAG Performance Ranking (Corrected):**
```
1. FIXED KG Cypher:  0.830 🥇 (Hybrid Vector+Graph)
2. KG Simple:        0.780 🥈 (Vector+Fixed Cypher)
3. Naive RAG:        0.746 🥉 (Vector Only)
```

### Design Principles Established

**All future KG RAG implementations MUST:**
1. ✅ Start with vector similarity search (non-negotiable)
2. ✅ Expand via graph traversal (from found nodes only)
3. ✅ Preserve original text (no LLM summarization)
4. ✅ Use hybrid approach (Vector + Graph = Optimal)

### Reference Documents

- **Full Analysis**: `docs/CHECKPOINT_kg_cypher_fix.md`
- **Quick Analysis**: `docs/cypher_rag_fix_analysis.md`
- **Implementation**: `src/kg_agent/kg_cypher_retriever.py`
- **Config**: `config/retest_cypher_rag_fixed.json`

### Next Steps

1. **Full Retest Required**: 50 questions, 5 models, FIXED Cypher
2. **Update All Reports**: Use FIXED results as new baseline
3. **Methodology Update**: Reflect Hybrid approach in thesis

**⚠️ IMPORTANT**: All previous KG Cypher results are INVALID. Only use FIXED version going forward.

- 프로젝트에서 사용할 데이터셋은 @data/016.행정_문서_대상_기계독해_데이터/ 의 데이터임. 기존의 데이터는 deprecated. 부적절해서 사용하지 않을 예정. 따라서 관련 결과들도 폐기

---

# 🟡 AutoRAG Config YAML 작성 규칙 (2025-11-29)

## Node Line 구조

AutoRAG v0.3.x에서는 두 개의 node_line으로 구성:

```yaml
node_lines:
  - node_line_name: retrieve_node_line    # 검색 + 후처리
  - node_line_name: post_retrieve_node_line  # 생성
```

## 각 Node Line에 속하는 Node Types

### retrieve_node_line (검색 + 후처리)
| Node Type | 설명 |
|-----------|------|
| `lexical_retrieval` | BM25 기반 검색 |
| `semantic_retrieval` | Vector 기반 검색 |
| `hybrid_retrieval` | BM25 + Vector 혼합 |
| `passage_augmenter` | 청크 확장 |
| `passage_reranker` | 재순위화 |
| `passage_filter` | 필터링 |

### post_retrieve_node_line (생성)
| Node Type | 설명 |
|-----------|------|
| `passage_compressor` | 압축 |
| `prompt_maker` | 프롬프트 생성 |
| `generator` | 답변 생성 |

## 필수 파라미터

### passage_augmenter
```yaml
- node_type: passage_augmenter
  strategy:
    metrics: [retrieval_f1, retrieval_recall, retrieval_precision]
    speed_threshold: 5
  embedding_model: openai  # 필수!
  top_k: 5                 # 필수!
  modules:
    - module_type: pass_passage_augmenter
    - module_type: prev_next_augmenter
      mode: both  # next, prev, both
      num_passages: 1
```

### passage_reranker
```yaml
- node_type: passage_reranker
  strategy:
    metrics: [retrieval_f1, retrieval_recall, retrieval_precision]
    speed_threshold: 30
  top_k: 3  # 필수!
  modules:
    - module_type: pass_reranker
    - module_type: ko_reranker  # koreranker 아님!
      batch: 32
    - module_type: flag_embedding_reranker
      model_name: BAAI/bge-reranker-large
```

### passage_filter (존재하는 모듈만!)
```yaml
- node_type: passage_filter
  modules:
    - module_type: pass_passage_filter
    - module_type: similarity_threshold_cutoff  # threshold_cutoff 아님!
      threshold: 0.3
      embedding_model: openai
    - module_type: similarity_percentile_cutoff  # percentile_cutoff 아님!
      percentile: 0.6
      embedding_model: openai
```

### long_context_reorder (prompt 필수!)
```yaml
- module_type: long_context_reorder
  prompt: "프롬프트 내용..."  # 반드시 prompt 파라미터 필요
```

## 자주 하는 실수

| 잘못된 설정 | 올바른 설정 | 비고 |
|------------|------------|------|
| `node_type: retrieval` | `lexical_retrieval` / `semantic_retrieval` | v0.3.x에서 변경 |
| `ko_reranker` | `koreranker` | 소문자로 사용 |
| `threshold_cutoff` | `similarity_threshold_cutoff` | 존재하지 않는 모듈 |
| `percentile_cutoff` | `similarity_percentile_cutoff` | 존재하지 않는 모듈 |
| `{{variable}}` | `{variable}` | fstring 형식 |
| passage_augmenter에 top_k 누락 | 반드시 `top_k` 지정 | KeyError 발생 |
| bert_score에 model_type 사용 | `lang: ko`만 지원 | TypeError 발생 |

## 검증된 Config 참조

- **Simple**: `src/autorag_pilot/config/simple_rag.yaml`
- **Full**: `src/autorag_pilot/config/full_optimization.yaml`
- **Silver Test**: `src/autorag_silver_test/config/silver_test.yaml`

---

# 📚 석사 논문 연구 참조 (2025-12-02)

## 논문 핵심 문서

**`thesis/README.md`** - 연구의 모든 핵심 정보가 정리된 마스터 문서

### 포함 내용
- Research Questions (RQ1-4)
- 가설 (H1-4)
- 연구 기여 (Contributions C1-4)
- 선행연구 요약 (한국어 RAG, Survey, GraphRAG, Enterprise RAG)
- 인용 가능한 핵심 문장
- 논문 구성 계획
- 실험 설계 요약

### 선행연구 상세 리뷰
`thesis/literature/` 폴더에 개별 논문 리뷰 마크다운 파일 존재:
- 정상무 (2024) - 행정문서 요약 RAG
- 이채원 (2025) - 한국어 Hybrid RAG
- 권혁규 (2025) - 대학 규정 RAG 챗봇
- Fan et al. (2024) - RAG Survey (KDD 2024)
- Sharma (2025) - RAG Comprehensive Survey
- Han et al. (2025) - RAG vs GraphRAG 비교
- B & Purwar (2024) - Enterprise RAG

### 핵심 Research Questions
```
RQ1: On-premise 오픈소스 LLM이 상용 API 대비 어느 수준의 성능을 달성하는가?
RQ2: BM25, Vector, Hybrid 검색 중 한국어 공공도메인에서 어떤 것이 최적인가?
RQ3: KoReranker가 범용 리랭커 대비 유의미한 성능 향상을 제공하는가?
RQ4: 24GB VRAM 제약에서 최적의 비용-성능 조합은 무엇인가?
```

### 핵심 Contributions
```
C1: 한국어 공공도메인 RAG 벤치마크 최초 구축 (100 QA × 5,000 corpus)
C2: On-premise RAG 최적 파이프라인 도출 (1,800개 조합 탐색)
C3: 한국어 리랭커/LLM 효과성 정량 분석 (6 리랭커 × 5 LLM)
C4: 공공기관 도입 가능한 청사진 제공 (RTX 3090Ti 환경)
```

## 참조 방법

논문 관련 작업 시 먼저 `thesis/README.md`를 읽고 컨텍스트 확보
