# Golden Testset: 50 Most Complex Questions

**Date**: 2025-11-11
**Dataset Source**: `data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs_full.jsonl`
**Output File**: `data/evaluation/golden_testset_50q_complex.jsonl`

## Methodology

### Selection Criteria

Questions were selected based on a weighted complexity score calculated from multiple factors:

```python
complexity_score = (
    answer_length * 0.3 +          # Longer answers = more detailed (168-662 chars)
    entity_count * 20 +             # More entities = richer information (4-24 entities)
    topic_count * 15 +              # Multiple topics = broader knowledge (2-6 topics)
    kb_tag_count * 10 +             # More tags = more knowledge domains (5-9 tags)
    question_length * 0.2 +         # Longer questions = more complex (52-254 chars)
    question_parts * 5              # Multiple questions = multi-part inquiry
)
```

### Balance Requirements

- **Total Questions**: 50
- **Minimum per Major Category**: 10 questions
- **Scoring Method**: Sort by complexity score (descending) with category balance

## Results

### Dataset Statistics

**Full Dataset**: 9,632 Q&A pairs from Dasan call center

**Selected Testset**: 50 questions (top 0.5% by complexity)

### Complexity Distribution

| Metric | Min | Max | Mean | Median |
|--------|-----|-----|------|--------|
| **Complexity Score** | 391.3 | 760.0 | 535.9 | 539.3 |
| **Answer Length** | 168 | 662 | 330.9 | 319 |
| **Question Length** | 52 | 254 | 123.0 | 114 |
| **Entity Count** | 4 | 24 | 14.4 | - |
| **Topic Count** | 2 | 6 | 3.6 | - |
| **KB Tag Count** | 5 | 9 | 6.6 | - |

### Category Breakdown

| Category | Count | Percentage |
|----------|-------|------------|
| 일반행정_문의 (General Admin) | 10 | 20.0% |
| 대중교통_안내 (Public Transport) | 10 | 20.0% |
| 생활하수도_관련_문의 (Water/Sewage) | 10 | 20.0% |
| 코로나19_관련_상담 (COVID-19) | 10 | 20.0% |
| 코로나19 관련 상담 | 3 | 6.0% |
| 생활하수도 관련 문의 | 2 | 4.0% |
| Other categories | 5 | 10.0% |

**Note**: Category name variations (with/without underscore) exist due to data preprocessing differences. Total coverage includes all 4 major domains.

## Top 5 Most Complex Questions

### 1. Score: 760.0 - 노후 경유차 운행 제한 (Diesel Vehicle Restrictions)

**Category**: 일반행정_문의 (General Admin)

**Question**: "서울시 미세먼지 저감조치인 노후 경유차 운행 제한에 대해 자세히 알려주세요. 제한되는 지역, 기간, 시간은 어떻게 되고 위반 시 과태료는 얼마인가요? 그리고 운행 제한을 피하기 위한 방법과 관련된 정부 지원(저감장치 부착, 조기폐차) 혜택에 대해서도 설명해주세요."

**Why Complex**:
- Multi-part question (6 distinct sub-questions)
- High entity count (24 entities)
- Long answer (452 characters)
- Multiple topics: 미세먼지, 차량운행제한, 정부지원, 과태료
- Requires knowledge across environmental policy, traffic regulations, and government subsidies

---

### 2. Score: 701.0 - 지하철 자전거 탑승 규정 (Bicycle on Subway Rules)

**Category**: 대중교통_안내 (Public Transport)

**Question**: "지하철에 자전거를 가지고 탑승할 수 있나요? 일반 자전거와 접이식 자전거의 휴대 규정이 어떻게 다른지, 그리고 평일과 주말에 이용 가능한 노선과 승차 위치, 이용이 불가능한 노선에 대해 자세히 알려주세요."

**Why Complex**:
- Multiple conditions (weekday vs weekend, regular vs folding bikes)
- Requires detailed route and station knowledge
- High entity count (18 entities)
- Long answer (344 characters)

---

### 3. Score: 683.2 - 민방위 교육 연차별 차이 (Civil Defense Training by Year)

**Category**: 일반행정_문의 (General Admin)

**Question**: "민방위 교육이 연차별로 어떻게 다른지 궁금합니다. 1년차부터 5년차 이상까지의 교육 방법과, 코로나19로 인해 사이버 교육이나 서면 교육으로 대체되는 경우가 있는지, 있다면 어떤 구에서 가능한지와 서면 교육은 어떻게 진행되는지 알려주세요."

**Why Complex**:
- Requires comparison across multiple years (1-5+)
- Conditional logic (COVID-19 policy changes)
- Geographic variations (different districts)
- High entity count (20 entities)

---

### 4. Score: 639.2 - 수도요금 신용카드 납부 (Water Bill Credit Card Payment)

**Category**: 생활하수도_관련_문의 (Water/Sewage)

**Question**: "수도요금을 신용카드로 납부하는 방법에 대해 궁금합니다. 은행 ATM에서 일회성으로 납부하는 방법과 카드로 자동이체를 신청하는 방법을 각각 상세히 알려주세요. ATM 이용 시 필요 서류, 수수료, 이용 가능 은행과 시간, 그리고 자동이체 신청 방법, 신청 가능한 카드 종류..."

**Why Complex**:
- Two distinct payment methods (one-time vs auto-payment)
- Multiple procedural details (documents, fees, hours, banks)
- High topic count (5 topics)
- Long question (254 characters)

---

### 5. Score: 634.0 - 수도요금 납부 방법 비교 (Water Bill Payment Methods Comparison)

**Category**: 생활하수도_관련_문의 (Water/Sewage)

**Question**: "수도요금 납부 방법에는 어떤 것들이 있는지 궁금합니다. 특히 청구서 납부, 계좌 자동납부, 그리고 입금전용계좌 납부 방법에 대해 상세히 설명해주세요."

**Why Complex**:
- Comparison of 3+ payment methods
- High entity count (17 entities)
- Requires detailed procedural knowledge

## Characteristics of High-Complexity Questions

### 1. Multi-Part Inquiries
- Average question parts: 2-3 sub-questions
- Top questions contain 4-6 distinct information requests

### 2. Rich Entity Coverage
- Mean entity count: 14.4 entities per question
- Top questions: 18-24 entities
- Includes organizations, locations, policies, procedures

### 3. Topic Diversity
- Mean topic count: 3.6 topics per question
- Spans multiple knowledge domains within single question

### 4. Detailed Answers
- Mean answer length: 330.9 characters (vs ~200 for dataset average)
- Indicates comprehensive, multi-faceted responses

### 5. Conditional Logic
- Many questions involve if-then scenarios
- Time-based conditions (weekday/weekend, COVID-19 policies)
- Category-based variations (vehicle types, user types)

## Use Cases

### 1. RAG System Evaluation
- **Faithfulness**: Do retrieved contexts support complex answers?
- **Answer Relevancy**: Can system address multi-part questions?
- **Context Precision**: Are all relevant documents retrieved?

### 2. Model Performance Benchmarking
- Test model capability on complex, multi-hop reasoning
- Evaluate handling of Korean administrative language
- Assess accuracy on policy-related questions

### 3. Retrieval Strategy Testing
- Compare Naive RAG vs Knowledge Graph approaches
- Test hybrid retrieval (vector + graph) effectiveness
- Evaluate chunk size and overlap strategies

### 4. Error Analysis
- Identify failure patterns on complex questions
- Analyze partial answer quality
- Test handling of conditional logic

## Quality Assurance

### Validation Checks
- ✅ All 50 questions have valid metadata
- ✅ Category balance achieved (≥10 per major category)
- ✅ No duplicate dialogue_ids
- ✅ All questions are answerable from provided contexts
- ✅ Complexity scores verified through manual sampling

### Limitations
- Bias towards longer questions (inherent in scoring formula)
- May under-represent simple but critical questions
- Category imbalance in original dataset affects representation
- Some category name variations exist (processing artifacts)

## Reproduction

To regenerate the golden testset:

```bash
uv run python scripts/create_golden_testset.py
```

**Input**: `data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs_full.jsonl`
**Output**: `data/evaluation/golden_testset_50q_complex.jsonl`
**Runtime**: ~10 seconds on 9,632 records

## Next Steps

1. **Baseline Evaluation**: Run all RAG systems on golden testset (50Q × 5 models = 250 evaluations)
2. **Error Analysis**: Identify failure modes on most complex questions
3. **Iterative Improvement**: Use insights to refine retrieval and generation strategies
4. **Expansion**: Consider creating difficulty tiers (easy/medium/hard) for comprehensive evaluation

---

**Generated**: 2025-11-11
**Script**: `scripts/create_golden_testset.py`
**Dataset Version**: AI_HUB_DASAN_QA v1.0 (182,719 original pairs → 9,632 consolidated documents)
