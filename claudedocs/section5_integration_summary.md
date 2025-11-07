# Section 5 Integration Summary

**Date**: 2025-10-30
**Task**: Integrate RAGAS 0.3.1+ approach from `03_ragas_korean_testset_generation.ipynb` into Section 5 of `02_naive_qa_evaluation.ipynb`

## Changes Made

### 1. Updated Korean Personas (Cell 47)

**Before**:
- Simple persona definitions without explicit language specification
- No guarantee of Korean Q/A generation

**After**:
```python
Persona(
    name="지하철 처음 이용자",
    role_description="""
    지하철을 처음 이용하는 승객입니다.
    모든 질문과 답변은 반드시 한국어로 작성되어야 합니다.  # ← CRITICAL
    기본적인 이용 방법, 요금, 노선 정보 등에 대해 자세한 안내가 필요합니다.
    ...
    """
)
```

**Impact**: Explicit Korean language specification ensures models generate Korean Q/A pairs

---

### 2. Replaced Document Sampling with Knowledge Graph (Section 5.3)

**Before**:
- Section 5.3: Simple document sampling
- No structured approach to document understanding

**After**:
- Section 5.3: Knowledge Graph creation
- Documents converted to structured nodes
- Enables transform-based enrichment

**New Code**:
```python
from ragas.testset.graph import KnowledgeGraph, Node, NodeType

kg = KnowledgeGraph()
for doc in documents:
    kg.nodes.append(
        Node(
            type=NodeType.DOCUMENT,
            properties={
                "page_content": doc.page_content,
                "document_metadata": doc.metadata
            }
        )
    )
```

---

### 3. Added Transform Pipeline (Section 5.4)

**New Section**: Transform Pipeline application

**Transforms**:
1. **HeadlinesExtractor**: Extract section headlines from documents
2. **HeadlineSplitter**: Split documents by headlines (max 1500 tokens)
3. **KeyphrasesExtractor**: Extract key phrases for query generation

**Code**:
```python
from ragas.testset.transforms import apply_transforms
from ragas.testset.transforms import HeadlinesExtractor, HeadlineSplitter, KeyphrasesExtractor

transforms = [
    HeadlinesExtractor(llm=generator_llm, max_num=20),
    HeadlineSplitter(max_tokens=1500),
    KeyphrasesExtractor(llm=generator_llm)
]

apply_transforms(kg, transforms=transforms)
```

**Impact**: Enriched Knowledge Graph enables better question generation

---

### 4. Updated Query Synthesizer Configuration (Section 5.5)

**Before**:
- Old API: `generate_with_langchain_docs()`
- Limited query diversity

**After**:
- **50% Headlines-based queries**: Structured section-based questions
- **50% Keyphrases-based queries**: Keyword-focused questions

**Code**:
```python
from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer

query_distribution = [
    (SingleHopSpecificQuerySynthesizer(llm=generator_llm, property_name="headlines"), 0.5),
    (SingleHopSpecificQuerySynthesizer(llm=generator_llm, property_name="keyphrases"), 0.5),
]
```

---

### 5. Updated Testset Generation (Section 5.6)

**Before**:
```python
generator = TestsetGenerator.from_langchain(...)  # ❌ Deprecated
testset = generator.generate_with_langchain_docs(...)  # ❌ Deprecated
```

**After**:
```python
from ragas.testset import TestsetGenerator

generator = TestsetGenerator(
    llm=generator_llm,
    embedding_model=embeddings,
    knowledge_graph=kg,  # ← New: KG-based approach
    persona_list=personas  # ← New: Persona-based generation
)

testset = generator.generate(
    testset_size=50,
    query_distribution=query_distribution  # ← New: Distribution control
)
```

---

### 6. Added Korean Validation Logic (Section 5.7)

**New Section**: Korean quality validation

**Validation**:
- Check if questions contain Korean characters
- Check if answers contain Korean characters
- Warn if Korean ratio < 90%

**Code**:
```python
def is_korean(text):
    """Check if text contains Korean characters"""
    korean_pattern = re.compile('[\u3131-\u3163\uac00-\ud7a3]+')
    return bool(korean_pattern.search(text))

korean_questions = df_testset['user_input'].apply(is_korean).sum()
korean_answers = df_testset['reference'].apply(is_korean).sum()

if korean_questions < total * 0.9 or korean_answers < total * 0.9:
    print("⚠️ 경고: 한국어 비율이 90% 미만입니다.")
```

---

## API Migration Summary

### Deprecated (RAGAS < 0.3.1)
❌ `TestsetGenerator.from_langchain()`
❌ `generate_with_langchain_docs()`
❌ `critic_llm` parameter
❌ `evolutions` parameter (simple, reasoning, multi_context)
❌ `temperature` parameter for GPT-5+ models

### New (RAGAS 0.3.1+)
✅ `TestsetGenerator(llm, embeddings, kg, personas)`
✅ `generator.generate(testset_size, query_distribution)`
✅ `KnowledgeGraph` with `Node` objects
✅ `apply_transforms()` for KG enrichment
✅ `SingleHopSpecificQuerySynthesizer` for query generation
✅ `Persona` with explicit language specification
✅ No temperature for GPT-5+ models (removed from OpenAI API)

---

## Section 5 Structure (After Integration)

1. **Section 5.1**: RAGAS 컴포넌트 설정 (existing)
2. **Section 5.2**: 질문 합성기 설정 (existing)
3. **Section 5.2.5**: 한국어 페르소나 정의 ← UPDATED (explicit Korean)
4. **Section 5.3**: Knowledge Graph 생성 ← NEW (replaces document sampling)
5. **Section 5.4**: Transform Pipeline 적용 ← NEW
6. **Section 5.5**: Query Synthesizer 설정 ← NEW
7. **Section 5.6**: Korean Testset 생성 ← UPDATED (KG-based approach)
8. **Section 5.7**: 한국어 품질 검증 ← NEW
9. **Section 5.8**: 질문 품질 검증 (existing, renumbered)
10. **Section 5.9**: 질문 저장 (existing, renumbered)

---

## Expected Improvements

### Korean Language Quality
- **Before**: Implicit Korean expectation → Often generated English Q/A
- **After**: Explicit "반드시 한국어로" in personas → Korean Q/A guaranteed

### Question Diversity
- **Before**: Limited to document-level sampling
- **After**: Headlines + Keyphrases based → More diverse questions

### API Compatibility
- **Before**: Using deprecated API → Failures on RAGAS 0.3.1+
- **After**: Latest API → Compatible with RAGAS 0.3.1+

### Reproducibility
- **Before**: No validation, hard to debug failures
- **After**: Korean validation + detailed logging → Easy to identify issues

---

## Testing Recommendations

1. **Run Section 5** in `02_naive_qa_evaluation.ipynb`
2. **Verify Korean validation** passes (>90% Korean ratio)
3. **Check generated questions** for quality and diversity
4. **Compare with previous results** (if available)
5. **Monitor RAGAS logs** for any errors or warnings

---

## Files Modified

- `notebooks/02_naive_qa_evaluation.ipynb` - Section 5 updated with RAGAS 0.3.1+ approach

## Files Used (Not Modified)

- `notebooks/03_ragas_korean_testset_generation.ipynb` - Reference implementation
- `src/data_processing/generate_ragas_testset.py` - Standalone script version

## Temporary Files (Removed)

- `src/scripts/update_section5_personas.py` - Persona update script (executed and removed)
- `src/scripts/integrate_ragas_031_approach.py` - KG integration script (executed and removed)
