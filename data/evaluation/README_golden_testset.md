# Golden Testset: 50 Most Complex Questions

## Quick Start

```bash
# Location
data/evaluation/golden_testset_50q_complex.jsonl

# Format: JSONL (1 question per line)
# Lines: 50
# Size: ~350KB
```

## Data Format

Each line is a JSON object with:

```json
{
  "dialogue_id": "B24034",
  "original_question": "서울시 미세먼지 저감조치인...",
  "original_answer": "문의하신 1천만원 긴급 대출은...",
  "topic_path": "코로나19_관련_상담/금융지원/소상공인_대출",
  "primary_topic": "소상공인_대출",
  "secondary_topics": ["금융지원", "저신용자_지원", "대출_자격조건"],
  "document": "--- (full markdown document with metadata) ---",
  "metadata": {
    "category": "코로나19_관련_상담",
    "doc_length_chars": 1105,
    "entities_count": 7,
    "kb_tags_count": 6,
    ...
  },
  "complexity_score": 760.0
}
```

## Statistics

- **Total Questions**: 50 (top 0.5% from 9,632 consolidated documents)
- **Complexity Range**: 391.3 - 760.0 (mean: 535.9)
- **Answer Length**: 168 - 662 characters (mean: 330.9)
- **Entities per Q**: 4 - 24 (mean: 14.4)
- **Topics per Q**: 2 - 6 (mean: 3.6)
- **KB Tags per Q**: 5 - 9 (mean: 6.6)

## Category Balance

| Category | Count | Percentage |
|----------|-------|------------|
| 일반행정_문의 (General Admin) | 10 | 20.0% |
| 대중교통_안내 (Public Transport) | 10 | 20.0% |
| 생활하수도_관련_문의 (Water/Sewage) | 10 | 20.0% |
| 코로나19_관련_상담 (COVID-19) | 10 | 20.0% |
| Other variations | 10 | 20.0% |

## Complexity Distribution

```
🔴 Extreme    (700-800): ██ (2)      - Multi-part, 20+ entities
🟠 Very High  (600-700): █████ (5)   - Conditional logic, 15+ entities
🟡 High       (500-600): ████ (31)   - Multiple topics, 10+ entities
🟢 Medium-High(400-500): ████ (8)    - Detailed answers, 5+ topics
🔵 Medium     (0-400):   ████ (4)    - Baseline complex questions
```

## Use Cases

### 1. RAG System Evaluation

```bash
# Run evaluation on golden testset
python src/evaluation/parallel_evaluator.py \
  --config config/golden_testset_eval.json \
  --testset data/evaluation/golden_testset_50q_complex.jsonl
```

**Recommended Metrics**:
- Faithfulness (contexts support answer?)
- Answer Relevancy (multi-part addressed?)
- Context Precision (all relevant docs?)
- Answer Correctness (factual accuracy?)

### 2. Model Benchmarking

```python
from datasets import Dataset

# Load golden testset
testset = Dataset.from_json("data/evaluation/golden_testset_50q_complex.jsonl")

# Test your model
results = evaluate_model(model, testset)
```

**Focus Areas**:
- Multi-hop reasoning
- Conditional logic handling
- Korean administrative language
- Entity-rich responses

### 3. Retrieval Strategy Testing

Compare performance across:
- Naive RAG (vector only)
- Knowledge Graph Simple (vector + 1-hop)
- Knowledge Graph Cypher (vector + graph traversal)
- Hybrid approaches

### 4. Error Analysis

Identify failure patterns:
- Which complexity levels fail?
- Category-specific issues?
- Multi-part question handling?

## Quality Assurance

✅ **Validated**:
- No duplicate IDs
- All required fields present
- All answers non-empty
- Category balance achieved
- Complexity scores verified

## Reproduction

To regenerate or customize:

```bash
# Standard 50 questions
uv run python scripts/create_golden_testset.py

# Custom parameters (edit script)
n_questions = 100          # More questions
min_per_category = 20      # Stricter balance
complexity_weights = {...} # Adjust scoring
```

## Example Questions

### Extreme Complexity (760.0)
**Category**: 일반행정_문의

**Q**: "서울시 미세먼지 저감조치인 노후 경유차 운행 제한에 대해 자세히 알려주세요. 제한되는 지역, 기간, 시간은 어떻게 되고 위반 시 과태료는 얼마인가요? 그리고 운행 제한을 피하기 위한 방법과 관련된 정부 지원(저감장치 부착, 조기폐차) 혜택에 대해서도 설명해주세요."

**Why Complex**: 6 sub-questions, 24 entities, multiple topics (environment, traffic, subsidies)

---

### Very High Complexity (701.0)
**Category**: 대중교통_안내

**Q**: "지하철에 자전거를 가지고 탑승할 수 있나요? 일반 자전거와 접이식 자전거의 휴대 규정이 어떻게 다른지, 그리고 평일과 주말에 이용 가능한 노선과 승차 위치, 이용이 불가능한 노선에 대해 자세히 알려주세요."

**Why Complex**: Conditional rules (weekday/weekend), bike type variations, route-specific info

---

### High Complexity (588.2)
**Category**: 코로나19_관련_상담

**Q**: "사회적 거리두기가 1단계로 변경되었다고 들었는데, 기존 2단계와 비교했을 때 구체적으로 어떤 점들이 달라지나요? 특히 스포츠 관람, 공공시설 및 다중이용시설 이용, 모임 규정에 대해 알려주세요."

**Why Complex**: Comparison (Stage 1 vs 2), multiple domains (sports, facilities, gatherings)

## Notes

- **Category Variations**: Some categories have underscore/spacing differences due to data preprocessing
- **Score Interpretation**: Relative measure, not absolute difficulty
- **Bias**: Scoring favors longer, entity-rich questions (by design)
- **Updates**: Testset is static for reproducibility; source dataset may evolve

## Citation

```
Golden Testset: 50 Most Complex Questions from Dasan Call Center QA
Source: AI_HUB_DASAN_QA Dataset (182,719 original → 9,632 consolidated)
Selection: Top 0.5% by complexity score (multi-factor quantitative analysis)
Created: 2025-11-11
Version: 1.0
```

## Support

- **Full Report**: `claudedocs/GOLDEN_TESTSET_REPORT.md`
- **Creation Script**: `scripts/create_golden_testset.py`
- **Source Dataset**: `data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs_full.jsonl`

For questions or issues, see project documentation.
