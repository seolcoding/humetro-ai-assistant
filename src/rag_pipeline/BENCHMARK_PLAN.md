# Production RAG Benchmark Evaluation Plan

## 🎯 Objective
Comprehensive evaluation of RAG system performance across multiple LLMs to determine optimal model for Seoul traffic Q&A service.

## 📊 Evaluation Scope

### Models to Evaluate (5 Total - config/models.yaml 기준)
1. **OpenAI Models**
   - GPT-4o-mini (Cost-effective baseline)

2. **Ollama Local Models (4)**
   - EXAONE-3.5-7.8B (LG AI Research, 한국어 특화)
   - Qwen3-8B (Alibaba, 다국어 지원)
   - Gemma3-12B (Google, 효율적 아키텍처)
   - GPT-OSS-20B (오픈소스 GPT, MoE 아키텍처)

### Question Dataset (50 Total)
- **25 Single-hop questions**: Direct factual queries
- **25 Multi-hop questions**: Complex reasoning queries

## 📝 Question Distribution

### Single-hop Questions (25)
Categories:
- 교통요금 (5): 기본요금, 환승할인, 정기권
- 운행정보 (5): 첫차/막차, 배차간격, 노선
- 교통카드 (5): 종류, 충전, 사용방법
- 시설정보 (5): 역사, 편의시설, 무장애
- 정책/제도 (5): 기후동행카드, 할인제도

### Multi-hop Questions (25)
Categories:
- 경로계획 (5): 최적경로, 시간/비용 계산
- 비교분석 (5): 교통수단 비교, 요금 비교
- 조건부추천 (5): 상황별 최적 선택
- 복합계산 (5): 다단계 요금 계산
- 시나리오기반 (5): 실제 상황 문제해결

## 🔧 Technical Configuration

### RAG Settings
```python
{
    "retriever": {
        "k_documents": 4,              # Top-4 documents per question
        "search_type": "similarity",   # Similarity search
        "vector_store": "FAISS",       # 13,176 vectors
        "embeddings": "text-embedding-3-small"
    },
    "generation": {
        "temperature": 0.0,            # Deterministic for consistency
        "max_tokens": 500,             # Sufficient for answers
        "timeout": 30                  # Per request timeout
    },
    "evaluation": {
        "judge_model": "gpt-4o",       # Consistent judge
        "metrics": [
            "faithfulness",            # Answer grounded in context
            "answer_relevancy",        # Answer addresses question
            "answer_correctness"       # Factual accuracy
        ]
    }
}
```

## 📅 Execution Timeline

### Phase 1: Question Generation (30 min)
1. Generate diverse single-hop questions
2. Create complex multi-hop scenarios
3. Add reference answers for correctness eval
4. Validate question quality

### Phase 2: Model Setup (15 min)
1. Verify Ollama models are running
2. Check API keys (OpenAI)
3. Test connectivity for all models
4. Warm-up runs (1 question each)

### Phase 3: Benchmark Execution (2-3 hours)
```
Estimated time per model:
- GPT-4o-mini: ~12 min (fast API)
- EXAONE-3.5-7.8B: ~24 min (local)
- Qwen3-8B: ~20 min (local)
- Gemma3-12B: ~25 min (local)
- GPT-OSS-20B: ~30 min (local, larger model)

Total: ~111 min + evaluation overhead
```

### Phase 4: Analysis (30 min)
1. Aggregate metrics by model
2. Compare single vs multi-hop performance
3. Cost-performance analysis
4. Generate comparison report

## 💰 Cost Estimation

### OpenAI API Costs
```
Per run (50 questions):
- Input: ~4000 tokens/question × 50 = 200K tokens
- Output: ~300 tokens/question × 50 = 15K tokens
- Judge (GPT-4o): ~150K tokens for evaluation

GPT-4o costs:
- Input: $2.50/1M tokens → $0.50
- Output: $10/1M tokens → $0.15
- Judge: $2.50/1M tokens → $0.38
- Total per run: ~$1.03

GPT-4o-mini costs:
- Input: $0.15/1M tokens → $0.03
- Output: $0.60/1M tokens → $0.01
- Judge (same): $0.38
- Total per run: ~$0.42

Total OpenAI cost: ~$1.45
```

### Local Models
- No API costs (running on local Ollama)
- Only computational resources

## 📈 Expected Outputs

### 1. Detailed Results JSON
```json
{
  "timestamp": "2025-11-05T14:00:00",
  "models": {
    "gpt-4o": {
      "single_hop": {
        "faithfulness": 0.92,
        "answer_relevancy": 0.94,
        "answer_correctness": 0.88
      },
      "multi_hop": { ... }
    },
    ...
  }
}
```

### 2. Comparison Matrix
| Model | Faithfulness | Relevancy | Correctness | Avg Response Time | Cost/1K |
|-------|-------------|-----------|-------------|------------------|---------|
| GPT-4o-mini | 0.87 | 0.91 | 0.83 | 0.8s | $0.75 |
| EXAONE-3.5-7.8B | 0.85 | 0.89 | 0.86 | 2.4s | $0 |
| Qwen3-8B | 0.83 | 0.87 | 0.81 | 2.0s | $0 |
| Gemma3-12B | 0.84 | 0.88 | 0.82 | 2.5s | $0 |
| GPT-OSS-20B | 0.86 | 0.90 | 0.84 | 3.0s | $0 |

### 3. Recommendations
- Best overall performance
- Best cost-performance ratio
- Best for Korean-specific queries
- Best for real-time applications

## 🚀 Execution Script

Create `run_full_benchmark.py`:
```bash
# Unified benchmark runner 사용 (논문용 모델 그룹)
python unified_benchmark.py --questions 50 --models thesis

# 또는 개별 모델 지정
python unified_benchmark.py --questions 50 --models \
    gpt-4o-mini \
    ollama/exaone3.5:7.8b \
    ollama/qwen3:8b \
    ollama/gemma3:12b \
    ollama/gpt-oss:20b

# 설정 확인
# - judge_model: gpt-5 (기본값)
# - retrieval: 4 documents
# - vector_store: seoul_traffic
```

## ✅ Success Criteria

1. **Technical Success**
   - All 6 models evaluated successfully
   - No empty retrieved_contexts
   - Valid metrics for all questions

2. **Quality Metrics**
   - At least one model > 0.85 faithfulness
   - At least one model > 0.90 relevancy
   - At least one model > 0.80 correctness

3. **Practical Insights**
   - Clear winner for production use
   - Cost-performance tradeoff identified
   - Korean language performance validated

## 📋 Pre-flight Checklist

- [ ] Vector store loaded (13,176 vectors)
- [ ] All Ollama models downloaded and running
- [ ] OpenAI API key valid with credits
- [ ] 50 questions prepared with reference answers
- [ ] Output directory created
- [ ] Monitoring setup for long run
- [ ] Backup plan if interruption occurs

## 🔄 Recovery Plan

If benchmark interrupted:
1. Save partial results to checkpoint file
2. Resume from last completed model
3. Merge results at the end
4. Re-run only failed evaluations

## 📞 Next Steps After Benchmark

1. **Immediate Analysis**
   - Validate no empty contexts
   - Check for any NaN metrics
   - Quick winner identification

2. **Detailed Report**
   - Performance breakdown by category
   - Error analysis for failures
   - Cost projection for production

3. **Production Decision**
   - Select primary and fallback models
   - Configure for production RAG
   - Plan A/B testing strategy