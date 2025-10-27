# Open-Source LLM Comparison: Gemma, Qwen, and EXAONE

**Source:** Perplexity Search (Academic Mode)
**Date:** 2025-10-27
**Query:** open source LLM comparison Gemma Qwen EXAONE performance 2024

---

## Executive Summary

This document provides a comprehensive comparison of three leading open-source LLM families: Gemma (Google), Qwen (Alibaba), and EXAONE (LG AI Research). All three model families released significant updates in 2024, representing state-of-the-art capabilities for on-premise deployment scenarios. Each family exhibits distinct strengths across different parameter scales and specialized domains.

---

## Model Family Overview

### Gemma Series (Google DeepMind)

#### Gemma 2 (Released 2024)

**Parameter Scales:** 2B, 7B, 9B, 27B

**Architecture Innovations:**

- Interleaved local-global attention mechanisms
- Modified Transformer architecture for efficiency
- Optimized for lightweight deployment

**Performance Highlights:**

- Outperforms similarly sized open models on **11 out of 18 text-based tasks**
- Strong generalized performance across diverse benchmarks
- Efficient resource utilization at smaller parameter sizes

**Reference:** arXiv:2408.00118

#### Gemma 3 (Released 2024-2025)

**Parameter Scales:** 1B, 7B, 12B, 27B

**Key Innovation:** Multimodal capabilities (vision + language)

**Improvements over Gemma 2:**

- Superior performance in both pretrained and instruction fine-tuned versions
- Vision understanding capabilities
- Enhanced long-context handling through increased local-to-global attention ratio

**Specialized Variants:**

- **CodeGemma:** Optimized for code generation tasks
  - Maintains strong natural language understanding
  - Excellent mathematical reasoning alongside code capabilities
- **PaliGemma:** Vision-Language Model (3B parameters)
  - Based on SigLIP-So400m vision encoder + Gemma-2B language model
  - Versatile transfer learning capabilities

**Reference:** arXiv:2503.19786

---

### Qwen Series (Alibaba)

#### Qwen 2.5 (Released 2024)

**Parameter Scales:** 3B, 7B, 14B, 32B, 72B

**Training Scale:**

- Expanded pre-training datasets from **7 trillion to 18 trillion tokens**
- Substantial improvements in both pre-training and post-training stages

**Performance Highlights:**

**Medical Domain Excellence:**

- **Chinese National Nursing Licensing Examination:**
  - Outperformed GPT-3.5, GPT-4, and GPT-4o
  - Ensemble with other models achieved **90.8% accuracy**
- **Multilingual Medical Reasoning:**
  - Competitive performance against ChatGPT-4
  - Strong results in English and Arabic medical queries

**General Performance:**

- State-of-the-art on Chinese language tasks
- Competitive with proprietary models on English benchmarks

**Reference:** arXiv:2412.15115

#### Qwen2.5-VL (Multimodal Variant)

**Capabilities:**

- State-of-the-art performance comparable to GPT-4o and Claude 3.5 Sonnet
- Excels in document and diagram understanding
- Maintains robust linguistic capabilities of Qwen2.5 LLM

**Reference:** arXiv:2502.13923

#### Qwen Specialized Models

- **Qwen-2.5 3B:** Fine-tuned for drug-drug interaction prediction
- **Code-Qwen:** Coding-specialized variants
- **Math-Qwen-Chat:** Mathematics-focused models

**Reference:** arXiv:2309.16609

---

### EXAONE Series (LG AI Research)

#### EXAONE Deep (Released 2024)

**Parameter Scales:** 2.4B, 7.8B, 32B

**Core Strength:** Reasoning-enhanced architecture

**Performance Characteristics:**

- **EXAONE Deep 2.4B and 7.8B:** Outperform comparable-sized alternatives
- **EXAONE Deep 32B:** Competitive performance against leading open-weight models
- Superior capabilities in reasoning tasks including:
  - Mathematical reasoning
  - Coding benchmarks
  - Logical problem solving

**Availability:** All models openly available for research purposes

**Reference:** arXiv:2503.12524

#### EXAONE 3.0 7.8B Instruction Tuned

**Significance:** First open model released from the EXAONE family

**Evaluation:** Assessed across wide range of public and in-house benchmarks

**Developer:** LG AI Research

**Reference:** arXiv:2408.03541

---

## Comparative Performance Analysis

### Parameter Size Efficiency

| Model Family | Min Size | Max Size | Efficiency Sweet Spot |
|--------------|----------|----------|-----------------------|
| Gemma        | 1B       | 27B      | 7B-9B                 |
| Qwen         | 3B       | 72B      | 7B-14B                |
| EXAONE       | 2.4B     | 32B      | 7.8B                  |

### Domain-Specific Strengths

#### Medical/Healthcare

**Winner:** Qwen 2.5

- Exceptional performance on medical licensing exams
- Outperforms commercial models in specialized medical domains
- Strong multilingual medical reasoning

#### Reasoning/Mathematics

**Winner:** EXAONE Deep

- Purpose-built for enhanced reasoning capabilities
- Superior mathematical problem-solving
- Strong coding benchmark performance

#### General Purpose/Lightweight

**Winner:** Gemma 2/3

- Best performance-per-parameter efficiency
- Broad accessibility with smaller models
- Consistent performance across diverse general tasks

#### Multimodal Understanding

**Winner:** Qwen2.5-VL (with Gemma 3 as strong alternative)

- State-of-the-art document understanding
- Competitive with proprietary models
- Gemma 3 offers vision capabilities at lighter weight

---

## Detailed Benchmark Comparisons

### Multi-Agent Gaming Evaluation

**Benchmark:** Decision-making in multi-agent environments

**Results (Score out of 100):**

- Gemini-1.5-Pro: **69.8** (proprietary baseline)
- LLaMA-3.1-70B: **65.9**
- Mixtral-8x22B: **62.4**
- Gemma models: Strong robustness but limited generalizability
  - Enhanced using Chain-of-Thought techniques

**Reference:** arXiv:2403.11807

### Medical Exam Performance

**Benchmark:** Brazil's Medical Revalidation Exam (Revalida)

**Key Findings:**

- GPT-4o: **63.85%** (best performer)
- Qwen-2.5: Competitive performance in Chinese medical exams
- Open-source models approaching proprietary performance in specialized domains

**Reference:** Applied Sciences 2025

### Bulgarian Language Tasks

**Model:** BgGPT-Gemma-2-27B-Instruct and 9B variants

**Performance:**

- Strong language-specific adaptation capabilities
- Maintains robust capabilities of original Gemma models
- Sets new standard for Bulgarian language AI

**Reference:** arXiv:2412.10893

---

## Specialized Application Performance

### Drug-Drug Interaction Prediction

**Best Model:** Qwen-2.5 3B (fine-tuned)

**Task:** Predicting interactions between pharmaceutical compounds

**Advantage:** Smaller model size with domain specialization outperforms larger general models

**Reference:** arXiv:2502.06890

### Code Generation

**CodeGemma:**

- Resilient natural language understanding
- Strong mathematical reasoning
- Balanced code generation and explanation capabilities

**Code-Qwen:**

- Significantly improved code generation over base models
- Enhanced debugging capabilities

**Reference:** arXiv:2406.11409, arXiv:2309.16609

### Academic Writing

**Comparative Study:** DeepSeek, Qwen, ChatGPT, Gemini, Llama, Mistral, Gemma

**Findings:**

- Qwen 2.5 Max competitive with ChatGPT for academic content generation
- Gemma models provide efficient alternatives for specific writing tasks
- Free and open-source models offer significant potential for academic use

**Reference:** arXiv:2503.04765

---

## Deployment Considerations

### Hardware Requirements

#### Gemma Models

**2B-7B:**

- **Minimum:** 16GB RAM, consumer GPUs (RTX 3060+)
- **Optimal:** 24GB VRAM (RTX 3090/4090)

**9B-27B:**

- **Minimum:** 24GB VRAM (RTX 3090)
- **Optimal:** 48GB+ VRAM (A6000, A100)

#### Qwen Models

**3B-14B:**

- **Minimum:** 16-32GB RAM, mid-tier GPUs
- **Optimal:** 32-48GB VRAM

**32B-72B:**

- **Minimum:** 48-80GB VRAM (multi-GPU or A100)
- **Optimal:** 80GB+ VRAM (A100 80GB)

#### EXAONE Models

**2.4B-7.8B:**

- **Minimum:** 16-24GB VRAM
- **Optimal:** 32GB VRAM

**32B:**

- **Minimum:** 48GB VRAM
- **Optimal:** 80GB VRAM

### Quantization Options

All three model families support:

- **FP16:** Full precision for maximum accuracy
- **INT8:** 2x memory reduction, minimal performance loss
- **INT4:** 4x memory reduction, acceptable for many tasks
- **GGUF/GGML:** Efficient CPU inference with quantization

---

## Licensing and Availability

### Gemma

**License:** Gemma Terms of Use (permissive, allows commercial use)

**Restrictions:** Attribution required, some usage restrictions

**Availability:**

- Hugging Face Hub
- Kaggle Models
- Google AI Studio

### Qwen

**License:** Qwen License Agreement (permissive)

**Restrictions:** Minimal restrictions, allows commercial use

**Availability:**

- Hugging Face Hub
- ModelScope
- Official Qwen GitHub

### EXAONE

**License:** Custom LG AI Research License

**Restrictions:** Research and educational use emphasized

**Availability:**

- Hugging Face Hub
- LG AI Research official channels

---

## Selection Decision Matrix

### Choose **Gemma** if

- ✅ Need lightweight, efficient models (1B-9B sweet spot)
- ✅ Want broad general-purpose capability
- ✅ Deploying on resource-constrained hardware
- ✅ Require strong performance-per-parameter efficiency
- ✅ Building multimodal applications (Gemma 3)

### Choose **Qwen** if

- ✅ Need state-of-the-art performance at all scales
- ✅ Working in specialized domains (especially medical, Chinese language)
- ✅ Require best-in-class multimodal capabilities (Qwen2.5-VL)
- ✅ Want extensive model family with specialized variants
- ✅ Need superior performance justifies larger model size

### Choose **EXAONE** if

- ✅ Prioritize reasoning and mathematical capabilities
- ✅ Focus on coding and technical problem-solving
- ✅ Need optimized mid-size models (7.8B sweet spot)
- ✅ Want models specifically enhanced for logical reasoning
- ✅ Prefer transparent research-focused development

---

## Relevance to Humetro Project

### Recommended Model Selection

#### Primary Candidates for 4-Model Comparison

1. **Gemma 3 12B**
   - Optimal size for on-premise deployment
   - Strong general performance
   - Multimodal capabilities (future-proofing)
   - Efficient resource utilization

2. **Qwen 3 8B** (or Qwen 2.5 14B)
   - State-of-the-art performance in specialized domains
   - Excellent Korean/Asian language support potential
   - Proven medical domain performance (relevant for public service)
   - Strong reasoning capabilities

3. **EXAONE 7.8B**
   - Korean AI research origin (cultural/linguistic relevance)
   - Enhanced reasoning for complex queries
   - Optimal parameter size for efficiency
   - LG AI Research backing ensures continued support

4. **GPT-OSS 20B** (Current Selection)
   - Largest open-source model in comparison
   - Upper bound for on-premise feasibility
   - Performance ceiling reference

### Expected Performance Patterns

Based on literature review, anticipated performance hierarchy:

**Baseline (No RAG):**

- GPT-OSS 20B > Qwen ≈ EXAONE > Gemma

**Naive RAG:**

- Qwen > GPT-OSS ≈ EXAONE > Gemma

**Advanced RAG:**

- Qwen ≈ GPT-OSS > EXAONE > Gemma

**Graph RAG:**

- Qwen > GPT-OSS > EXAONE ≈ Gemma

**Reasoning Tasks:**

- EXAONE > Qwen > GPT-OSS > Gemma

**Resource Efficiency:**

- Gemma > EXAONE > Qwen > GPT-OSS

### Evaluation Focus Areas

1. **Multi-hop Reasoning:** EXAONE expected to excel
2. **Domain Adaptation:** Qwen expected to excel
3. **Efficiency:** Gemma expected to excel
4. **Overall Performance:** GPT-OSS and Qwen expected to lead

---

## Additional Research Papers

### Comparative Studies

1. **Are Small Language Models Ready to Compete with LLMs?** - arXiv:2406.11402
2. **Performance Comparison of LLMs on Clinical Vignette Questions** - PMC10949144
3. **Evaluating LLMs for SDG Mapping** - arXiv:2408.02201

### Benchmarking and Evaluation

1. **OLMES: A Standard for Language Model Evaluations** - arXiv:2406.08446
2. **Benchmarking Benchmark Leakage in LLMs** - arXiv:2404.18824
3. **FuseChat: Knowledge Fusion of Chat Models** - arXiv:2408.07990

### Multimodal Evaluation

1. **Visual Reasoning Evaluation: Grok, Deepseek, Gemini, Qwen, Mistral** - arXiv:2502.16428
2. **Gemini in Reasoning: Unveiling Commonsense in MLLMs** - arXiv:2312.17661

---

## Conclusion

The choice between Gemma, Qwen, and EXAONE for the Humetro project should prioritize:

1. **Performance Requirements:** Qwen offers highest performance ceiling
2. **Resource Constraints:** Gemma provides best efficiency
3. **Reasoning Depth:** EXAONE excels in logical problem-solving
4. **Domain Specificity:** Qwen demonstrates superior specialized capability

**Recommended Configuration:** Include all three model families to comprehensively evaluate the performance-efficiency-capability trade-off space, providing robust evidence for on-premise deployment viability in public sector contexts.
