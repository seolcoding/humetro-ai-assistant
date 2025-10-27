# 🚊 Humetro AI Assistant - Graph RAG Research

> **온프레미스 오픈소스 기반 Graph RAG 시스템의 공공부문 적용 연구**
>
> 다산콜센터 사례를 중심으로 한 실증 연구

## 📌 Research Overview

본 연구는 공공부문에서 데이터 주권과 비용 효율성을 유지하면서도 고성능 AI 서비스를 제공할 수 있는 **온프레미스 Graph RAG 시스템**의 실용성을 실증합니다.

### 🎯 핵심 목표

1. **기술 주권 확보**: 해외 빅테크 API 의존 없이 고성능 달성
2. **비용 효율성**: 5년 TCO 36% 절감
3. **성능 동등성**: 온프레미스 시스템이 상용 API와 동등한 성능 달성

### 🔬 실험 설계

- **4개 오픈소스 LLM** × **4개 RAG 방식** = **16개 시스템 비교**
- **평가 데이터**: 다산콜센터 FAQ 3,000개 + AI Hub Q&A 10,000개
- **평가 지표**: RAGAS + LLM-as-Judge (GPT-4o)

## 🏗️ Project Structure

```
humetro-ai-assistant/
├── data/                        # 데이터 자산
│   ├── raw/                     # 원본 데이터
│   │   ├── dasan_faq/           # 다산콜센터 FAQ
│   │   └── aihub_qa/            # AI Hub 데이터셋
│   ├── processed/               # 전처리 데이터
│   └── knowledge_graphs/        # 지식 그래프
│       ├── gpt5_generated/      # GPT-5 생성 (핵심 자산)
│       └── opensource_kg/       # 비교용
├── experiments/                 # 실험 관리
│   ├── configs/                 # 실험 설정
│   ├── models/                  # 모델별 실험
│   └── rag_methods/             # RAG 방법별
├── results/                     # 실험 결과
├── src/                         # 소스 코드
│   ├── rag_pipeline/            # RAG 파이프라인
│   └── evaluation/              # 평가 시스템
├── scripts/                     # 자동화 스크립트
└── thesis/                      # 논문 자료
```

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.11+
python --version

# CUDA 12.1+ (for GPU)
nvidia-smi

# Neo4j (for Graph RAG)
neo4j --version
```

### Installation

```bash
# Clone repository
git clone https://github.com/your-username/humetro-ai-assistant.git
cd humetro-ai-assistant

# Install dependencies
pip install -r requirements.txt

# Download models (optional - for local execution)
python scripts/download_models.py
```

### Running Experiments

```bash
# Run all experiments (16 combinations)
python scripts/run_experiment.py --config experiments/configs/base_config.yaml

# Run specific model-method combination
python scripts/run_experiment.py \
    --models gemma_3_12b \
    --methods graph_rag

# Analyze results
python scripts/analyze_results.py \
    --results-dir results/20251027_120000_humetro_graph_rag_comparison
```

## 📊 Key Components

### 1. RAG Pipeline Hierarchy

```python
BaseRAG (Abstract)
├── BaselineRAG     # Pure LLM (No retrieval)
├── NaiveRAG        # Vector search only
├── AdvancedRAG     # Hybrid search (BM25 + Semantic)
└── GraphRAG        # Knowledge graph with multi-hop
```

### 2. Evaluation System

- **RAGAS Metrics**:
  - Faithfulness
  - Answer Relevancy
  - Context Precision
  - Context Recall
  - Answer Correctness

- **LLM-as-Judge**:
  - Accuracy
  - Completeness
  - Relevance
  - Coherence
  - Domain Specificity

### 3. Knowledge Graph Construction

```python
# One-time construction with GPT-5
kg_builder = KnowledgeGraphBuilder(model="gpt-5")
kg = kg_builder.build_from_documents(documents)

# Save for permanent use
kg.save("data/knowledge_graphs/gpt5_generated")
```

## 📈 Expected Results

### Performance Comparison

| Model         | Baseline | Naive RAG | Advanced RAG | Graph RAG |
|---------------|----------|-----------|--------------|-----------|
| Gemma 3 12B   | 0.45     | 0.62      | 0.71         | **0.78**  |
| Qwen 3 8B     | 0.42     | 0.59      | 0.68         | **0.75**  |
| EXAONE 7.8B   | 0.43     | 0.60      | 0.69         | **0.76**  |
| GPT-OSS 20B   | 0.47     | 0.64      | 0.73         | **0.80**  |

### Cost Analysis (5-Year TCO)

- **Pure API Approach**: $12,775
- **Hybrid (Our Approach)**: $8,200 (36% savings)
- **Break-even Point**: 18 months

## 🔧 Hardware Requirements

### Minimum (Development)
- GPU: NVIDIA RTX 3090 (24GB VRAM)
- RAM: 32GB
- Storage: 500GB SSD

### Recommended (Production)
- GPU: NVIDIA RTX 4090 or A100
- RAM: 64GB
- Storage: 1TB NVMe SSD

## 📝 Citation

```bibtex
@article{humetro2025,
  title={온프레미스 오픈소스 기반 Graph RAG 시스템의 공공부문 적용 연구},
  author={Your Name},
  journal={Journal Name},
  year={2025}
}
```

## 🤝 Contributing

연구 협력 및 기여를 환영합니다:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/NewFeature`)
3. Commit changes (`git commit -m 'Add NewFeature'`)
4. Push to branch (`git push origin feature/NewFeature`)
5. Open Pull Request

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 🙏 Acknowledgments

- 서울시 120 다산콜센터
- AI Hub 데이터 제공
- 오픈소스 커뮤니티

## 📧 Contact

- **Research Lead**: [your-email@example.com]
- **Project Repository**: [GitHub](https://github.com/your-username/humetro-ai-assistant)
- **Issues**: [GitHub Issues](https://github.com/your-username/humetro-ai-assistant/issues)

---

> **Note**: 이 프로젝트는 학술 연구 목적으로 진행되며, 실제 배포 전 추가적인 보안 검토가 필요합니다.