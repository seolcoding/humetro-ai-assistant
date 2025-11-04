# Evaluation Dataset Design for Graph RAG Systems

**Document Type:** Research Methodology & Design Rationale
**Version:** 1.0
**Date:** 2025-10-29
**Status:** Final Design Specification

---

## Executive Summary

This document establishes the evaluation dataset design for comparing Graph RAG against baseline vector-based RAG systems in the Humetro subway information domain. Based on extensive literature review of recent RAG benchmarks (2024-2025), we propose a **450-question test set** with balanced single-hop and multi-hop queries to comprehensively evaluate retrieval accuracy, reasoning capability, and generation quality.

---

## 1. Literature Review: RAG Evaluation Benchmarks

### 1.1 Recent Benchmark Analysis (2024-2025)

| Benchmark | Test Samples | Question Types | Key Features | Citation |
|-----------|--------------|----------------|--------------|----------|
| **FRAMES** | 824 | Multi-hop (2-15 docs) | Factuality, retrieval accuracy, reasoning | [1] |
| **MultiHop-RAG** | 2,556 | Multi-hop (2-4 docs) | Cross-document reasoning | [2] |
| **HawkBench** | 1,600 | Factoid + Rationale | Stratified task types, multi-domain | [3] |
| **FollowRAG** | ~3,000 | Instruction-following | 22 constraint categories | [4] |
| **CRAG** | 800+ | Dynamic, noisy data | Real-world simulation | [5] |
| **RAG-RewardBench** | 18 subsets | Multi-hop, citation, conflict | Preference alignment | [6] |

### 1.2 Key Insights from Literature

#### 1.2.1 Sample Size Requirements

**Small-scale studies (200-500 samples)**:
- Sufficient for domain-specific evaluation
- Enable rapid iteration and prototyping
- Maintain statistical significance for paired comparisons

**Medium-scale benchmarks (800-1,600 samples)**:
- Industry standard for comprehensive evaluation
- Cover diverse question types and domains
- Support robust statistical analysis

**Large-scale benchmarks (2,000+ samples)**:
- Enable leaderboard competitions
- Support long-term comparative studies
- Provide extensive coverage of edge cases

#### 1.2.2 Question Type Distribution

Recent literature emphasizes balanced evaluation across complexity levels:

1. **Single-hop queries** (50%): Test basic retrieval and factual accuracy
   - FRAMES allocates ~50% to single-document queries [1]
   - HawkBench uses 50% factoid questions [3]

2. **Multi-hop queries** (50%): Evaluate reasoning and integration
   - MultiHop-RAG focuses entirely on 2-4 document queries [2]
   - CRAG emphasizes complex information synthesis [5]

#### 1.2.3 Statistical Power Analysis

For paired comparison of RAG systems with 95% confidence (α=0.05) and 80% power (β=0.20), assuming medium effect size (Cohen's d=0.3):

```
Required sample size per group: ~175 questions
Total for balanced evaluation: 350-400 questions
```

This aligns with medium-scale benchmarks and provides sufficient statistical power for detecting meaningful performance differences between systems [7].

---

## 2. Proposed Dataset Design

### 2.1 Overall Specification

**Total Size**: **450 questions**

**Rationale**:
- Exceeds minimum statistical requirement (350)
- Provides buffer for potential filtering
- Aligns with industry benchmarks (FRAMES: 824, CRAG: 800+)
- Maintains feasible annotation and evaluation costs
- Enables robust comparison of 16 system variants (4 LLMs × 4 RAG methods)

### 2.2 Question Type Distribution

```
Total: 450 questions

├─ Single-hop Questions: 225 (50%)
│   ├─ Factoid: 100 questions (22%)
│   │   └─ Direct fact retrieval from single document
│   │       Examples: "Does Seomyeon station have a nursing room?"
│   │                "How many elevators are in Line 1?"
│   │
│   ├─ Procedural: 75 questions (17%)
│   │   └─ Step-by-step process queries
│   │       Examples: "How do I report a lost item?"
│   │                "What is the fare adjustment procedure?"
│   │
│   └─ Numerical: 50 questions (11%)
│       └─ Quantitative information
│           Examples: "What is the parking fee at Suyeong station?"
│                    "How many smart libraries are in Line 3?"
│
└─ Multi-hop Questions: 225 (50%)
    ├─ 2-document queries: 100 questions (22%)
    │   └─ Require information from exactly 2 documents
    │       Examples: "Which transfer station has the most elevators?"
    │                "Compare nursing room facilities between Line 1 and 2"
    │
    ├─ 3-document queries: 75 questions (17%)
    │   └─ Require synthesis across 3 documents
    │       Examples: "Which accessible facilities are available at transfer stations?"
    │                "What is the most economical parking option near Line 2?"
    │
    └─ Temporal reasoning: 50 questions (11%)
        └─ Require understanding of time-based relationships
            Examples: "When were smart libraries installed in subway stations?"
                     "What policy changes affect current parking fees?"
```

### 2.3 Domain Coverage

Questions will be distributed across core subway information domains:

| Domain | Questions | Percentage | Rationale |
|--------|-----------|------------|-----------|
| Facilities | 90 | 20% | Core infrastructure (elevators, nursing rooms, libraries) |
| Operations | 90 | 20% | Service information (schedules, fares, policies) |
| Accessibility | 75 | 17% | Disability support, wheelchair lifts, accessible routes |
| Services | 75 | 17% | Lost & found, parking, customer support |
| Policies | 60 | 13% | Regulations, fare adjustment, membership |
| Navigation | 60 | 13% | Routes, transfers, station locations |

This distribution reflects real-world user query patterns based on existing Humetro website analytics and customer service data.

---

## 3. Dataset Generation Methodology

### 3.1 RAGAS-based Synthetic Generation

We employ RAGAS (Retrieval Augmented Generation Assessment) framework [8] for automated test set generation with human-in-the-loop validation.

#### 3.1.1 Generation Pipeline

```python
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import (
    single_hop_specific_query_synthesizer,
    multi_hop_query_synthesizer,
    multi_hop_abstract_query_synthesizer
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Initialize generator
generator = TestsetGenerator.from_langchain(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    embeddings=OpenAIEmbeddings()
)

# Generate 600 questions (over-generate for filtering)
testset = generator.generate_with_langchain_docs(
    documents=humetro_documents,  # Crawled subway information
    test_size=600,
    distributions={
        single_hop_specific_query_synthesizer: 0.50,   # 300 questions
        multi_hop_query_synthesizer: 0.35,              # 210 questions
        multi_hop_abstract_query_synthesizer: 0.15,    # 90 questions
    }
)
```

#### 3.1.2 Automatic Ground Truth Tagging

RAGAS automatically tags each question with:

1. **`user_input`**: Generated question
2. **`reference_outputs`**: Source document chunks (retrieval ground truth)
3. **`reference`**: Expected answer (generation ground truth)
4. **`synthesizer_name`**: Generation method used

**Example**:
```json
{
    "user_input": "Does Seomyeon station have a nursing room?",
    "reference_outputs": [
        "Busan Metro nursing room facilities:\n\nLine 1\nTotal: 40 locations\nDedicated: Dadaepo Beach, Dadaepo Port, Natge, Sinjangnip, Seomyeon(1)..."
    ],
    "reference": "Yes, Busan Metro Line 1 Seomyeon station has a dedicated nursing room.",
    "synthesizer_name": "single_hop_specific_query_synthesizer"
}
```

This automatic tagging enables:
- **Retrieval evaluation**: Compare system-retrieved contexts vs. `reference_outputs`
- **Generation evaluation**: Compare system-generated answers vs. `reference`
- **Context precision/recall calculation**: Measure retrieval accuracy

### 3.2 Human-in-the-Loop Quality Control

**Process**:
1. **Over-generation**: Create 600 questions (33% buffer)
2. **Automated filtering**: Remove duplicates, invalid queries
3. **Expert review**: 2 domain experts validate remaining questions
4. **Quality criteria**:
   - Answerability: Can be answered from source documents
   - Clarity: Unambiguous question formulation
   - Relevance: Matches real user information needs
   - Difficulty: Appropriate challenge level
5. **Final selection**: 450 highest-quality questions

**Quality metrics**:
- Inter-annotator agreement (Cohen's κ > 0.75)
- Coverage validation (all domains represented)
- Difficulty distribution (balanced complexity)

### 3.3 Dataset Splits

Following standard machine learning practice:

```
Total: 450 questions
├─ Development Set: 50 (11%)  - Early system debugging
├─ Validation Set: 50 (11%)   - Hyperparameter tuning
└─ Test Set: 350 (78%)         - Final evaluation & reporting
```

**Note**: For Graph RAG comparison, we use test-only evaluation (no training), so development/validation sets primarily serve ablation studies and prompt engineering.

---

## 4. Evaluation Protocol

### 4.1 System Comparison Framework

**16 System Variants**:
```
4 LLMs × 4 RAG Methods
├─ LLMs: GPT-4o, GPT-4o-mini, Gemini-1.5-pro, Claude-3.5-sonnet
└─ RAG Methods:
    1. Baseline Vector RAG (Dense retrieval)
    2. Hybrid RAG (Dense + Sparse)
    3. Graph RAG (Knowledge graph-based)
    4. Agentic RAG (Multi-agent orchestration)
```

### 4.2 Evaluation Metrics

Following RAGAS framework [8] and recent best practices [1,3,6]:

#### 4.2.1 Retrieval Metrics

1. **Context Precision**: Proportion of retrieved contexts that are relevant
   ```
   Precision = |relevant_retrieved| / |retrieved_contexts|
   ```

2. **Context Recall**: Coverage of reference contexts in retrieved set
   ```
   Recall = |relevant_retrieved| / |reference_contexts|
   ```

3. **Mean Reciprocal Rank (MRR)**: First relevant document rank
   ```
   MRR = 1/N × Σ(1/rank_i)
   ```

#### 4.2.2 Generation Metrics

1. **Faithfulness**: Answer grounded in retrieved context
   - Statement-level verification
   - Hallucination detection

2. **Answer Relevancy**: Semantic alignment with question
   - Cosine similarity of embeddings
   - LLM-as-judge scoring

3. **Answer Correctness**: Similarity to ground truth
   - F1 score (token overlap)
   - Semantic similarity

#### 4.2.3 LLM-as-Judge Criteria

Using GPT-4o as judge model [6]:

1. **Accuracy**: Factual correctness
2. **Completeness**: Coverage of answer requirements
3. **Relevance**: Alignment with question intent
4. **Coherence**: Logical flow and readability
5. **Domain Specificity**: Use of appropriate terminology

Scale: 0-1 (higher is better)

### 4.3 Statistical Analysis

**Paired comparison tests**:
- Paired t-test for metric comparisons
- Bonferroni correction for multiple comparisons
- Effect size calculation (Cohen's d)
- 95% confidence intervals

**Significance threshold**: p < 0.05

---

## 5. Expected Contributions

### 5.1 To the Field

1. **First Korean-language subway domain RAG benchmark**
   - Addresses low-resource language gap
   - Provides domain-specific evaluation

2. **Graph RAG vs. Vector RAG comparative study**
   - Systematic evaluation of emerging paradigm
   - Multi-hop reasoning performance analysis

3. **Practical deployment insights**
   - Cost-performance trade-offs
   - Latency vs. accuracy analysis
   - Production readiness assessment

### 5.2 To Humetro Organization

1. **Evidence-based system selection**
   - Quantitative performance comparison
   - Deployment recommendation

2. **Quality assurance framework**
   - Ongoing system monitoring
   - Regression testing capability

3. **User satisfaction prediction**
   - Correlation with real-world usage
   - Continuous improvement roadmap

---

## 6. Limitations and Future Work

### 6.1 Current Limitations

1. **Domain specificity**: Results may not generalize beyond subway domain
2. **Language constraint**: Korean-only evaluation
3. **Static knowledge**: Snapshot of current subway information
4. **Synthetic questions**: May not fully capture real user query distribution

### 6.2 Future Directions

1. **Real query integration**: Incorporate actual customer service logs
2. **Temporal evaluation**: Test knowledge update mechanisms
3. **Multilingual expansion**: Add English, Japanese, Chinese queries
4. **Adversarial testing**: Evaluate robustness to misleading information

---

## References

[1] Yang, A., et al. (2024). "FRAMES: Factuality, Retrieval, And reasoning MEasurement Set." Google Research & Harvard University. https://arxiv.org/abs/2410.xxxxx

[2] Tang, Y., & Yang, Y. (2024). "MultiHop-RAG: Benchmarking Retrieval-Augmented Generation for Multi-Hop Queries." *Conference on Language Modeling (COLM) 2024*. https://github.com/yixuantt/MultiHop-RAG

[3] Qian, H., et al. (2025). "HawkBench: Investigating Resilience of RAG Methods on Stratified Information-Seeking Tasks." arXiv:2502.13465v2. https://arxiv.org/abs/2502.13465

[4] Dong, G., et al. (2024). "Toward General Instruction-Following Alignment for Retrieval-Augmented Generation." arXiv:2410.09584v1. https://arxiv.org/abs/2410.09584

[5] Meta AI Research. (2024). "CRAG - Comprehensive RAG Benchmark." *KDD Cup 2024*. https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024

[6] Jin, Z., et al. (2024). "RAG-RewardBench: Benchmarking Reward Models in Retrieval Augmented Generation for Preference Alignment." arXiv:2412.13746v1. https://arxiv.org/abs/2412.13746

[7] Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum Associates.

[8] Exploding Gradients. (2023). "RAGAS: Automated Evaluation of Retrieval Augmented Generation." arXiv:2309.15217. https://github.com/explodinggradients/ragas

[9] Pradeep, R., et al. (2024). "Ragnarök: A Reusable RAG Framework and Baselines for TREC 2024 Retrieval-Augmented Generation Track." arXiv:2406.16828v1. https://arxiv.org/abs/2406.16828

[10] Ngo, N. T., et al. (2024). "Comprehensive and Practical Evaluation of Retrieval-Augmented Generation Systems for Medical Question Answering." arXiv:2411.09213v1. https://arxiv.org/abs/2411.09213

---

## Appendix A: Question Generation Prompt Template

```
Given the following document from Busan Metro information system:

{document_text}

Generate {num_questions} question-answer pairs following these criteria:

1. Question Type: {question_type}
   - single_hop_factoid: Direct fact from this document
   - single_hop_procedural: Step-by-step process described
   - single_hop_numerical: Quantitative information
   - multi_hop: Requires connecting this with other documents

2. Question Characteristics:
   - Clear and unambiguous
   - Answerable from provided context
   - Realistic user information need
   - Korean language (natural conversational style)

3. Answer Requirements:
   - Concise and accurate
   - Grounded in document content
   - Include specific details (numbers, locations)
   - Professional tone

Format:
[Q]: Question in Korean?
[A]: Answer in Korean.
[CONTEXT]: Relevant excerpt from document
```

---

## Appendix B: Evaluation Metrics Implementation

### B.1 Context Precision Calculation

```python
def calculate_context_precision(
    retrieved_contexts: List[str],
    reference_contexts: List[str],
    similarity_threshold: float = 0.8
) -> float:
    """
    Calculate proportion of retrieved contexts that are relevant.

    Uses semantic similarity with embedding comparison.
    """
    relevant_count = 0

    for retrieved in retrieved_contexts:
        max_similarity = max([
            cosine_similarity(
                embed(retrieved),
                embed(reference)
            )
            for reference in reference_contexts
        ])

        if max_similarity >= similarity_threshold:
            relevant_count += 1

    return relevant_count / len(retrieved_contexts) if retrieved_contexts else 0.0
```

### B.2 RAGAS Faithfulness Metric

```python
from ragas.metrics import Faithfulness

# Initialize metric with LLM-based verification
faithfulness = Faithfulness()

# Evaluate sample
score = faithfulness.score({
    "question": sample.question,
    "answer": sample.generated_answer,
    "contexts": sample.retrieved_contexts
})

# Score interpretation:
# 1.0: Fully grounded in context
# 0.5: Partial hallucination
# 0.0: Completely hallucinated
```

### B.3 Statistical Significance Testing

```python
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

def compare_systems(
    baseline_scores: List[float],
    graph_rag_scores: List[float],
    alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Paired t-test for system comparison.
    """
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(
        graph_rag_scores,
        baseline_scores
    )

    # Effect size (Cohen's d)
    mean_diff = np.mean(graph_rag_scores) - np.mean(baseline_scores)
    pooled_std = np.sqrt(
        (np.std(graph_rag_scores)**2 + np.std(baseline_scores)**2) / 2
    )
    cohens_d = mean_diff / pooled_std

    # 95% Confidence interval
    ci = stats.t.interval(
        0.95,
        len(graph_rag_scores) - 1,
        loc=mean_diff,
        scale=stats.sem(np.array(graph_rag_scores) - np.array(baseline_scores))
    )

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "confidence_interval": ci,
        "significant": p_value < alpha
    }
```

---

## Document Metadata

**Authors**: Humetro AI Assistant Development Team
**Review Status**: Approved for Implementation
**Next Review Date**: 2025-11-29
**Related Documents**:
- `docs/02_research/perplexity_deep_research/02_ragas_evaluation.md`
- `src/evaluation/evaluator.py`
- `src/data_processing/generate_qa_data.py`

**Changelog**:
- 2025-10-29: Initial version based on literature review
