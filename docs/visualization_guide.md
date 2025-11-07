# Benchmark Results Visualization Guide

## Overview

This document explains the 6 comprehensive visualizations generated from the RAG benchmark comparison (Naive RAG vs KG Simple).

**Location**: `docs/figures/`

**Data**: 5 models × 50 questions (25 Single-hop + 25 Multi-hop)

---

## Figure 1: Overall Performance Comparison

**File**: `fig1_overall_comparison.png`

**Purpose**: High-level summary of average performance across all models and questions.

**Key Insights**:
- **Faithfulness**: KG Simple shows +8.4% improvement over Naive RAG
- **Relevancy**: Naive RAG maintains +6.2% advantage
- **Correctness**: Both methods perform similarly (~2% difference)

**Interpretation**: KG Simple excels at factual accuracy, while Naive RAG better matches question relevance.

---

## Figure 2: Model Performance Comparison

**File**: `fig2_model_performance.png`

**Purpose**: Compare how each model performs with different RAG methods across all three metrics.

**Key Insights**:
- **Faithfulness**: Most models show improvement with KG Simple
  - GPT-4o-mini: +13% with KG
  - EXAONE-3.5: +13% with KG
- **Relevancy**: Naive RAG generally better
  - Consistent advantage across models
- **Correctness**: Mixed results, model-dependent

**Interpretation**: KG benefits are model-agnostic for faithfulness, but relevancy advantage of Naive RAG is universal.

---

## Figure 3: Complexity Analysis (Single-hop vs Multi-hop)

**File**: `fig3_complexity_analysis.png`

**Purpose**: Understand how question complexity affects performance for each RAG method.

**Layout**: 2×3 grid (2 RAG methods × 3 metrics)

**Key Insights**:
- **Multi-hop advantage**: Most models perform BETTER on multi-hop questions
  - Gemma3-12B Multi-hop Faithfulness: 0.891 (highest score overall)
- **Naive RAG**: More consistent between single/multi-hop
- **KG Simple**: Larger variation, stronger on multi-hop

**Interpretation**: Complex reasoning questions benefit from graph structure traversal.

---

## Figure 4: Radar Chart - Multi-dimensional View

**File**: `fig4_radar_comparison.png`

**Purpose**: Holistic performance view of top 3 models across all metrics and complexities.

**Models Featured**:
- GPT-4o-mini (Blue)
- EXAONE-3.5-7.8B (Red)
- Gemma3-12B (Green)

**Key Insights**:
- **Gemma3-12B**: Largest coverage area in KG Simple (especially multi-hop faithfulness)
- **GPT-4o-mini**: Most balanced performance
- **EXAONE-3.5**: Strong faithfulness, weaker relevancy

**Interpretation**: Different models excel in different aspects - no single "best" model.

---

## Figure 5: Score Heatmap

**File**: `fig5_score_heatmap.png`

**Purpose**: Complete score matrix for all models, metrics, and complexities.

**Color Scale**:
- Green: High performance (0.8-1.0)
- Yellow: Medium performance (0.5-0.8)
- Red: Low performance (0.0-0.5)

**Key Insights**:
- **Hotspots** (Green zones):
  - KG Simple: Gemma3-12B Multi-hop Faithfulness (0.891)
  - Naive RAG: GPT-OSS-20B Single-hop Relevancy (0.786)
- **Cold spots** (Yellow/Red zones):
  - Qwen3-8B Relevancy scores consistently lower
  - Some correctness scores showing NaN (evaluation issues)

**Interpretation**: Heatmap reveals specific model-metric-complexity combinations for optimization.

---

## Figure 6: Performance Gap Analysis

**File**: `fig6_performance_gap.png`

**Purpose**: Quantify the difference between single-hop and multi-hop performance.

**Calculation**: Gap = Single-hop Score - Multi-hop Score
- **Positive gap**: Better at single-hop questions
- **Negative gap**: Better at multi-hop questions

**Key Insights**:
- **Most models**: Negative gaps (better at multi-hop)
  - Gemma3-12B: -0.102 faithfulness gap (much better at multi-hop)
- **EXAONE-3.5**: Positive faithfulness gap in KG Simple (+0.103)
  - Better at simple questions
- **Relevancy**: Consistently negative (multi-hop questions get more relevant answers)

**Interpretation**: RAG systems naturally perform better on complex questions requiring reasoning.

---

## Summary of Key Findings

### 1. **RAG Method Selection**

| Priority | Best Choice | Reason |
|----------|-------------|--------|
| Factual Accuracy | **KG Simple** | +8.4% Faithfulness |
| Question Relevance | **Naive RAG** | +6.2% Relevancy |
| Final Answer Quality | **Similar** | ~2% difference |

### 2. **Model Selection**

| Use Case | Best Model | Score |
|----------|-----------|-------|
| Overall Balance | GPT-4o-mini | Consistent across metrics |
| Multi-hop Reasoning | Gemma3-12B | 0.891 Faithfulness |
| Single-hop Accuracy | EXAONE-3.5 (KG) | 0.838 Faithfulness |

### 3. **Question Complexity Impact**

- **Multi-hop questions**: Generally achieve higher scores
- **Graph structure**: Particularly beneficial for complex reasoning
- **Vector similarity**: Sufficient for simple queries

### 4. **Production Recommendations**

1. **Hybrid Approach**: Use KG for factual queries, Naive for exploratory questions
2. **Model Ensemble**: Combine GPT-4o-mini (correctness) + Gemma3-12B (faithfulness)
3. **Question Routing**: Classify complexity and route to appropriate RAG method

---

## Next Steps

After KG Cypher Generation completion:
1. Add third method to all visualizations
2. Perform 3-way comparison analysis
3. Generate cost-performance trade-off charts
4. Create final recommendation matrix

---

**Generated**: 2025-11-06
**Data Source**: `data/evaluation/3way_rag_comparison/`
**Script**: `scripts/visualize_benchmark_results.py`
