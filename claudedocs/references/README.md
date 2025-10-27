# Research References for Humetro AI Assistant Project

**Generated:** 2025-10-27
**Purpose:** Comprehensive literature review for Graph RAG research thesis

---

## 📚 Document Structure

### Perplexity Research (Academic Mode)

High-quality, synthesized research from academic sources with detailed analysis and cross-referencing.

1. **[01_graph_rag_research.md](perplexity/01_graph_rag_research.md)**
   - Graph RAG fundamentals and recent advances (2024-2025)
   - Implementation frameworks (LightRAG, KG-Retriever, GFM-RAG, etc.)
   - Domain-specific applications (Medical, Legal, Customer Service)
   - Technical solutions to persistent challenges
   - Evaluation frameworks and hybrid approaches

2. **[02_ragas_evaluation_metrics.md](perplexity/02_ragas_evaluation_metrics.md)**
   - RAGAS framework comprehensive analysis
   - Core evaluation dimensions (Context Relevancy, Faithfulness, Generation Quality)
   - Extended metrics (Context Precision/Recall, Answer Correctness)
   - Implementation guidelines for research and production
   - Integration with LLM-as-Judge

3. **[03_opensource_llm_comparison.md](perplexity/03_opensource_llm_comparison.md)**
   - Gemma, Qwen, EXAONE detailed comparison
   - Architecture innovations and performance benchmarks
   - Domain-specific strengths and deployment considerations
   - Model selection decision matrix
   - Hardware requirements and quantization options

4. **[04_onpremise_llm_deployment_cost.md](perplexity/04_onpremise_llm_deployment_cost.md)**
   - Total Cost of Ownership (TCO) analysis
   - On-premise vs API-based deployment economics
   - Infrastructure optimization strategies (90-95% cost reduction potential)
   - Hybrid and collaborative approaches
   - 5-year TCO projection for Humetro project

### arXiv Papers

Direct academic papers from arXiv focused on Graph RAG implementations.

5. **[graph_rag_papers.md](arxiv/graph_rag_papers.md)**
   - Foundational survey papers (arXiv:2501.00309, arXiv:2408.08921)
   - Implementation frameworks (GRAG, HyperGraphRAG, GraphRAG-R1)
   - Domain-specific applications and efficiency techniques
   - Timeline of major developments (May 2024 - July 2025)
   - Neo4j integration patterns

---

## 🎯 Key Findings Summary

### Graph RAG Performance

- **Multi-hop reasoning:** GraphRAG enables complex query answering across multiple knowledge pieces
- **Hallucination reduction:** 18% reduction in biomedical QA tasks through structured knowledge
- **Domain applications:** Proven success in Medical, Legal, Customer Service domains
- **Implementation maturity:** Production-ready frameworks available (LightRAG, MsGraphRAG-Neo4j)

### Evaluation Frameworks

- **RAGAS:** Reference-free evaluation across 5 core dimensions
- **Cost reduction:** Enables rapid iteration without expensive human annotation
- **Multi-dimensional:** Separates retrieval quality from generation quality for targeted optimization
- **Integration:** Works alongside LLM-as-Judge for comprehensive assessment

### Open-Source LLM Capabilities

- **Gemma:** Best efficiency (1-27B parameters), strong general-purpose performance
- **Qwen:** State-of-the-art specialized performance, excellent multimodal capabilities
- **EXAONE:** Superior reasoning and mathematical capabilities, optimized mid-size models
- **Performance gap:** Open-source models approaching proprietary model performance

### Cost Economics

- **On-premise TCO:** 45-77% cost reduction vs pure API approach over 5 years
- **Domain adaptation:** 90-95% TCO reduction for specialized models
- **Infrastructure optimization:** 30-40% operational cost reduction through MLOps
- **Breakeven point:** ~10,000 queries/day for on-premise deployment viability

---

## 📊 Relevance to Thesis Research Questions

### RQ1: Performance Comparison (4 Models × 4 RAG Methods)

**Supporting Evidence:**

- Gemma/Qwen/EXAONE performance benchmarks → Model selection justified
- GraphRAG architectural patterns → Implementation guidance
- RAGAS metrics → Systematic evaluation methodology
- Expected performance hierarchy documented in literature

### RQ2: Cost-Effectiveness Analysis

**Supporting Evidence:**

- 5-year TCO models with detailed breakdowns
- Domain adaptation cost reduction quantified (90-95%)
- Infrastructure optimization strategies (30-40% reduction)
- Hybrid approach benefits documented

### RQ3: Public Sector Applicability

**Supporting Evidence:**

- Data sovereignty advantages of on-premise deployment
- Privacy and regulatory compliance benefits
- 다산콜센터 use case alignment with Customer Service GraphRAG applications
- Technological independence from foreign BigTech APIs

---

## 🔬 Implementation Recommendations

### Phase 1: Foundation (Months 1-3)

1. **Knowledge Graph Construction:**
   - Use GPT-5 for high-quality one-time graph generation
   - Target 다산콜센터 FAQ domain (3,000 entries)
   - Store in Neo4j for efficient retrieval

2. **Model Deployment:**
   - Gemma 3 12B (efficiency baseline)
   - Qwen 3 8B/14B (performance ceiling)
   - EXAONE 7.8B (reasoning specialist)
   - GPT-OSS 20B (scale comparison)

3. **Evaluation Setup:**
   - Implement RAGAS metrics (5 dimensions)
   - Configure LLM-as-Judge with GPT-4o
   - Prepare test dataset from AI Hub (10,000 Q&A pairs)

### Phase 2: Experimentation (Months 4-6)

1. **RAG Pipeline Development:**
   - Baseline (Pure LLM)
   - Naive RAG (Vector search)
   - Advanced RAG (Hybrid search)
   - Graph RAG (Neo4j + multi-hop)

2. **Systematic Evaluation:**
   - 4 models × 4 methods = 16 combinations
   - Cross-product testing on evaluation dataset
   - Statistical significance testing

3. **Cost Analysis:**
   - GPU utilization metrics
   - Inference latency measurements
   - API cost tracking (for GPT-5 usage)
   - TCO calculation per configuration

### Phase 3: Analysis (Months 7-8)

1. **Performance Analysis:**
   - RAGAS score aggregation and comparison
   - LLM-as-Judge qualitative insights
   - Multi-hop reasoning capability assessment
   - Error analysis and failure mode categorization

2. **Cost-Performance Trade-offs:**
   - Pareto frontier identification
   - Optimal configuration selection per use case
   - Sensitivity analysis on volume/complexity

3. **Thesis Writing:**
   - Literature review (comprehensive, citing 50+ papers)
   - Methodology (reproducible, well-documented)
   - Results (statistically rigorous)
   - Discussion (practical implications for public sector)

---

## 📖 Citation Guidelines

### Primary Survey Papers

```bibtex
@article{han2025graphrag,
  title={Retrieval-Augmented Generation with Graphs (GraphRAG)},
  author={Han, Haoyu and others},
  journal={arXiv preprint arXiv:2501.00309},
  year={2025}
}

@article{ragas2023,
  title={RAGAs: Automated Evaluation of Retrieval Augmented Generation},
  author={Authors},
  journal={arXiv preprint arXiv:2309.15217},
  year={2023}
}
```

### For Methodology Section

- **Graph RAG:** Cite arXiv:2501.00309 (comprehensive survey)
- **RAGAS Evaluation:** Cite arXiv:2309.15217 (original framework)
- **Neo4j Integration:** Cite arXiv:2509.21237 (practical implementation)
- **Cost Analysis:** Cite arXiv:2404.08850 (domain-adapted TCO)

### For Related Work Section

- **Open-source LLMs:** Cite model-specific papers (Gemma, Qwen, EXAONE technical reports)
- **RAG Evolution:** Cite arXiv:2410.12837 (comprehensive RAG survey)
- **Hybrid Approaches:** Cite arXiv:2408.07611 (WeKnow-RAG)

---

## 🔄 Next Steps

### Immediate Actions

1. ✅ **Literature Review Complete:** All major papers identified and summarized
2. ⏭️ **Deep Reading:** Read top 10 most relevant papers in detail
3. ⏭️ **Implementation Planning:** Design detailed experiment pipeline
4. ⏭️ **Hardware Preparation:** Verify GPU availability (2x RTX 4090 or equivalent)

### Research Gaps to Address

1. **Korean Language Adaptation:** Most papers focus on English/Chinese
2. **Public Sector Specificity:** Limited research on government use cases
3. **Long-term Maintenance:** TCO beyond 5 years not well-studied
4. **Temporal Dynamics:** Knowledge graph evolution strategies

### Collaboration Opportunities

- **Seoul 120 Dasan Call Center:** Real-world validation dataset
- **AI Hub:** Pre-existing Q&A datasets for evaluation
- **Open-Source Community:** Contribute GraphRAG-Korean implementations
- **Neo4j Community:** Share multilingual graph construction patterns

---

## 📞 Contact and Contributions

**Project Lead:** [Your Name]
**Institution:** [Your University]
**Repository:** [GitHub Link]

**How to Contribute:**

- Report errors or outdated information in references
- Suggest additional relevant papers
- Share implementation experiences
- Collaborate on Korean language adaptations

---

## 📜 License and Acknowledgments

**Data Sources:**

- Perplexity AI (Academic Search Mode)
- arXiv.org (Open Access Papers)
- Brave Search API

**Acknowledgments:**

- Seoul City 120 Dasan Call Center (Data Provider)
- AI Hub (Dataset Provider)
- Open-Source LLM Communities (Model Development)

**Usage:** These references are compiled for academic research purposes. Please cite original papers when using findings in publications.

---

## 🔖 Quick Reference Table

| Topic | Primary Document | Key Metrics | Implementation Complexity |
|-------|------------------|-------------|---------------------------|
| Graph RAG Architecture | perplexity/01 | 18% hallucination reduction | Medium-High |
| RAGAS Evaluation | perplexity/02 | 5 core dimensions | Low |
| Open-Source LLMs | perplexity/03 | Gemma/Qwen/EXAONE comparison | Medium |
| Cost Analysis | perplexity/04 | 45-77% TCO reduction | Low-Medium |
| arXiv Implementations | arxiv/graph_rag_papers | 20+ frameworks | High |

**Legend:**

- **Low:** Can implement with existing tools/libraries
- **Medium:** Requires custom code but standard patterns
- **High:** Requires novel engineering and research

---

*Last Updated: 2025-10-27*
*Total Papers Reviewed: 70+*
*Total Pages Generated: 5 comprehensive documents*
