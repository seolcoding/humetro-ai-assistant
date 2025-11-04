# Dataset Generation Implementation Guide

**Target Audience**: AI Assistants, Developers
**Purpose**: Step-by-step instructions for generating 450-question evaluation dataset
**Based on**: `docs/02_research/evaluation_dataset_design.md`

---

## Quick Reference

**Goal**: Generate **450 questions** for RAG evaluation

**Distribution**:
```
Total: 450 questions
├─ Single-hop: 225 (50%)
│   ├─ Factoid: 100
│   ├─ Procedural: 75
│   └─ Numerical: 50
└─ Multi-hop: 225 (50%)
    ├─ 2-document: 100
    ├─ 3-document: 75
    └─ Temporal: 50
```

**Domain Coverage**: Facilities (20%), Operations (20%), Accessibility (17%), Services (17%), Policies (13%), Navigation (13%)

---

## Prerequisites

### 1. Install Dependencies

```bash
uv add ragas langchain-openai langchain-community pandas openpyxl
```

### 2. Prepare Source Documents

Ensure crawled Humetro documents are available:
```bash
ls datasets/final_docs/  # Should contain markdown files
```

### 3. Set Environment Variables

```bash
export OPENAI_API_KEY="your-api-key"
```

---

## Implementation Steps

### Step 1: Load Documents

```python
from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

def load_humetro_documents(docs_dir: str = "datasets/final_docs"):
    """Load all markdown documents from crawled data."""
    loader = DirectoryLoader(
        docs_dir,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    print(f"Loaded {len(documents)} documents")
    return documents

def chunk_documents(documents, chunk_size=512, chunk_overlap=128):
    """Split documents into chunks for embedding."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")
    return chunks

# Execute
documents = load_humetro_documents()
chunks = chunk_documents(documents)
```

### Step 2: Initialize RAGAS Generator

```python
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import (
    single_hop_specific_query_synthesizer,
    multi_hop_query_synthesizer,
    multi_hop_abstract_query_synthesizer
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Initialize models
llm = ChatOpenAI(model="gpt-4o-mini", temperature=1.0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create generator
generator = TestsetGenerator.from_langchain(
    llm=llm,
    embeddings=embeddings
)

print("RAGAS generator initialized")
```

### Step 3: Generate Questions (Over-generate)

```python
# Generate 600 questions (33% buffer for filtering)
testset = generator.generate_with_langchain_docs(
    documents=chunks,
    test_size=600,
    distributions={
        single_hop_specific_query_synthesizer: 0.50,   # 300 questions
        multi_hop_query_synthesizer: 0.35,              # 210 questions
        multi_hop_abstract_query_synthesizer: 0.15,    # 90 questions
    },
    # Optional: Configure generation parameters
    with_debugging_logs=True,
    raise_exceptions=False  # Continue on errors
)

print(f"Generated {len(testset)} questions")
```

### Step 4: Convert to DataFrame and Save

```python
import pandas as pd
from datetime import datetime

# Convert to pandas DataFrame
df = testset.to_pandas()

# Add metadata columns
df['generation_date'] = datetime.now().isoformat()
df['status'] = 'pending_review'
df['domain'] = None  # To be filled during review
df['difficulty'] = None  # To be filled during review

# Save intermediate results
output_path = f"datasets/generated_qa/testset_600_raw_{datetime.now().strftime('%Y%m%d')}.csv"
df.to_csv(output_path, index=False, encoding='utf-8')

print(f"Saved raw dataset to {output_path}")
print(f"Columns: {df.columns.tolist()}")
```

### Step 5: Quality Filtering (Automated)

```python
def automatic_quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Apply automated quality checks."""

    # 1. Remove duplicates
    print(f"Before deduplication: {len(df)}")
    df = df.drop_duplicates(subset=['user_input'], keep='first')
    print(f"After deduplication: {len(df)}")

    # 2. Remove questions without answers
    df = df[df['reference'].notna() & (df['reference'].str.len() > 0)]
    print(f"After removing empty answers: {len(df)}")

    # 3. Remove questions with empty contexts
    df = df[df['reference_contexts'].notna()]
    print(f"After removing empty contexts: {len(df)}")

    # 4. Length checks
    df = df[
        (df['user_input'].str.len() >= 10) &  # Min question length
        (df['user_input'].str.len() <= 500) &  # Max question length
        (df['reference'].str.len() >= 10)  # Min answer length
    ]
    print(f"After length checks: {len(df)}")

    return df

# Apply filtering
df_filtered = automatic_quality_filter(df)

# Save filtered version
filtered_path = f"datasets/generated_qa/testset_filtered_{datetime.now().strftime('%Y%m%d')}.csv"
df_filtered.to_csv(filtered_path, index=False, encoding='utf-8')

print(f"Filtered dataset saved: {filtered_path}")
print(f"Remaining questions: {len(df_filtered)}")
```

### Step 6: Categorize Questions

```python
def categorize_questions(df: pd.DataFrame) -> pd.DataFrame:
    """Categorize questions by type and domain using LLM."""

    from langchain_openai import ChatOpenAI
    from langchain.prompts import ChatPromptTemplate

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    categorization_prompt = ChatPromptTemplate.from_template("""
    Analyze this Korean question and categorize it:

    Question: {question}

    Provide categorization in JSON format:
    {{
        "question_type": "factoid|procedural|numerical|multi_hop_2doc|multi_hop_3doc|temporal",
        "domain": "facilities|operations|accessibility|services|policies|navigation",
        "difficulty": "easy|medium|hard"
    }}

    Only output valid JSON, no explanation.
    """)

    chain = categorization_prompt | llm

    categories = []
    for idx, row in df.iterrows():
        try:
            result = chain.invoke({"question": row['user_input']})
            import json
            cat = json.loads(result.content)
            categories.append(cat)
        except Exception as e:
            print(f"Error categorizing question {idx}: {e}")
            categories.append({
                "question_type": "unknown",
                "domain": "unknown",
                "difficulty": "medium"
            })

        if (idx + 1) % 10 == 0:
            print(f"Categorized {idx + 1}/{len(df)} questions")

    # Add categories to dataframe
    df['question_type'] = [c['question_type'] for c in categories]
    df['domain'] = [c['domain'] for c in categories]
    df['difficulty'] = [c['difficulty'] for c in categories]

    return df

# Categorize
df_categorized = categorize_questions(df_filtered)

# Save categorized version
categorized_path = f"datasets/generated_qa/testset_categorized_{datetime.now().strftime('%Y%m%d')}.csv"
df_categorized.to_csv(categorized_path, index=False, encoding='utf-8')

print(f"Categorized dataset saved: {categorized_path}")
```

### Step 7: Strategic Selection (450 from 600)

```python
def strategic_selection(df: pd.DataFrame, target_counts: dict) -> pd.DataFrame:
    """
    Select questions to match target distribution.

    Args:
        df: Categorized dataframe
        target_counts: Dictionary of target counts per category
    """

    # Target distribution
    targets = {
        'factoid': 100,
        'procedural': 75,
        'numerical': 50,
        'multi_hop_2doc': 100,
        'multi_hop_3doc': 75,
        'temporal': 50,
    }

    selected = []

    for qtype, count in targets.items():
        # Get questions of this type
        subset = df[df['question_type'] == qtype]

        if len(subset) < count:
            print(f"Warning: Only {len(subset)} {qtype} questions available (need {count})")
            selected.append(subset)
        else:
            # Stratified sampling by difficulty
            difficulties = subset['difficulty'].value_counts()

            # Try to maintain 30% easy, 50% medium, 20% hard
            easy_n = int(count * 0.3)
            medium_n = int(count * 0.5)
            hard_n = count - easy_n - medium_n

            easy = subset[subset['difficulty'] == 'easy'].sample(
                n=min(easy_n, len(subset[subset['difficulty'] == 'easy'])),
                random_state=42
            )
            medium = subset[subset['difficulty'] == 'medium'].sample(
                n=min(medium_n, len(subset[subset['difficulty'] == 'medium'])),
                random_state=42
            )
            hard = subset[subset['difficulty'] == 'hard'].sample(
                n=min(hard_n, len(subset[subset['difficulty'] == 'hard'])),
                random_state=42
            )

            selected.append(pd.concat([easy, medium, hard]))

    final_df = pd.concat(selected, ignore_index=True)

    print(f"\nFinal selection: {len(final_df)} questions")
    print(f"\nDistribution by type:")
    print(final_df['question_type'].value_counts())
    print(f"\nDistribution by domain:")
    print(final_df['domain'].value_counts())
    print(f"\nDistribution by difficulty:")
    print(final_df['difficulty'].value_counts())

    return final_df

# Select final 450
final_df = strategic_selection(df_categorized, targets={})

# Save final dataset
final_path = f"datasets/generated_qa/testset_final_450_{datetime.now().strftime('%Y%m%d')}.csv"
final_df.to_csv(final_path, index=False, encoding='utf-8')

print(f"\n✅ Final dataset saved: {final_path}")
```

### Step 8: Create Dataset Splits

```python
from sklearn.model_selection import train_test_split

# Split: 50 dev / 50 val / 350 test
test_df, dev_val_df = train_test_split(
    final_df,
    test_size=100,  # 50 dev + 50 val
    random_state=42,
    stratify=final_df['question_type']  # Maintain type distribution
)

dev_df, val_df = train_test_split(
    dev_val_df,
    test_size=50,  # 50 val
    random_state=42,
    stratify=dev_val_df['question_type']
)

# Save splits
output_dir = Path("datasets/evaluation")
output_dir.mkdir(exist_ok=True)

dev_df.to_csv(output_dir / "dev_set_50.csv", index=False, encoding='utf-8')
val_df.to_csv(output_dir / "val_set_50.csv", index=False, encoding='utf-8')
test_df.to_csv(output_dir / "test_set_350.csv", index=False, encoding='utf-8')

print(f"✅ Dataset splits created:")
print(f"   - Development: {len(dev_df)} questions")
print(f"   - Validation: {len(val_df)} questions")
print(f"   - Test: {len(test_df)} questions")
```

---

## Verification Checklist

After generation, verify the following:

### ✅ Quantitative Checks

```python
def verify_dataset(df: pd.DataFrame):
    """Run verification checks on final dataset."""

    checks = []

    # 1. Total count
    checks.append(("Total questions", len(df) == 450, len(df)))

    # 2. Single-hop count
    single_hop = df[df['question_type'].isin(['factoid', 'procedural', 'numerical'])]
    checks.append(("Single-hop questions", len(single_hop) == 225, len(single_hop)))

    # 3. Multi-hop count
    multi_hop = df[df['question_type'].isin(['multi_hop_2doc', 'multi_hop_3doc', 'temporal'])]
    checks.append(("Multi-hop questions", len(multi_hop) == 225, len(multi_hop)))

    # 4. Domain coverage
    domains = df['domain'].value_counts()
    checks.append(("All domains present", len(domains) >= 6, len(domains)))

    # 5. No duplicates
    duplicates = df.duplicated(subset=['user_input']).sum()
    checks.append(("No duplicate questions", duplicates == 0, duplicates))

    # 6. All questions have answers
    no_answer = df['reference'].isna().sum()
    checks.append(("All questions have answers", no_answer == 0, no_answer))

    # 7. All questions have contexts
    no_context = df['reference_contexts'].isna().sum()
    checks.append(("All questions have contexts", no_context == 0, no_context))

    # Print results
    print("\n" + "="*60)
    print("DATASET VERIFICATION REPORT")
    print("="*60)

    all_passed = True
    for check_name, passed, value in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} | {check_name}: {value}")
        if not passed:
            all_passed = False

    print("="*60)

    if all_passed:
        print("✅ All verification checks passed!")
    else:
        print("❌ Some checks failed. Please review the dataset.")

    return all_passed

# Run verification
verify_dataset(final_df)
```

### ✅ Qualitative Checks

**Manual Review (sample 20 questions)**:

1. **Question clarity**: Are questions clear and unambiguous?
2. **Answerability**: Can questions be answered from source documents?
3. **Language quality**: Is Korean natural and grammatically correct?
4. **Difficulty balance**: Mix of easy/medium/hard questions?
5. **Domain relevance**: Questions match real user needs?

---

## Troubleshooting

### Issue: Not enough multi-hop questions generated

**Solution**: Adjust distribution weights
```python
distributions={
    single_hop_specific_query_synthesizer: 0.40,   # Reduce to 40%
    multi_hop_query_synthesizer: 0.45,              # Increase to 45%
    multi_hop_abstract_query_synthesizer: 0.15,    # Keep 15%
}
```

### Issue: Questions in wrong language (English instead of Korean)

**Solution**: Add language constraint to system prompt
```python
# Modify generator initialization
generator = TestsetGenerator.from_langchain(
    llm=llm,
    embeddings=embeddings,
    system_prompt="Generate all questions and answers in Korean language."
)
```

### Issue: Low-quality questions

**Solution**: Increase temperature for more diverse generation
```python
llm = ChatOpenAI(model="gpt-4o-mini", temperature=1.2)  # Increase from 1.0
```

### Issue: RAGAS generation errors

**Solution**: Use error handling and retry logic
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def generate_with_retry():
    return generator.generate_with_langchain_docs(
        documents=chunks,
        test_size=600,
        distributions={...},
        raise_exceptions=True
    )

testset = generate_with_retry()
```

---

## Complete Script

For convenience, here's the complete generation script:

```python
# File: scripts/generate_evaluation_dataset.py

"""
Generate 450-question evaluation dataset for Graph RAG comparison.

Usage:
    uv run python scripts/generate_evaluation_dataset.py
"""

import os
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import List, Dict

from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import (
    single_hop_specific_query_synthesizer,
    multi_hop_query_synthesizer,
    multi_hop_abstract_query_synthesizer
)
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from sklearn.model_selection import train_test_split


def main():
    # Configuration
    DOCS_DIR = "datasets/final_docs"
    OUTPUT_DIR = Path("datasets/evaluation")
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    print("🚀 Starting evaluation dataset generation...")

    # Step 1: Load documents
    print("\n📚 Loading documents...")
    loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.md",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    print(f"   Loaded {len(documents)} documents")

    # Step 2: Chunk documents
    print("\n✂️  Chunking documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=128,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    print(f"   Created {len(chunks)} chunks")

    # Step 3: Initialize generator
    print("\n🤖 Initializing RAGAS generator...")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=1.0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    generator = TestsetGenerator.from_langchain(llm=llm, embeddings=embeddings)

    # Step 4: Generate questions
    print("\n🎯 Generating 600 questions (will filter to 450)...")
    testset = generator.generate_with_langchain_docs(
        documents=chunks,
        test_size=600,
        distributions={
            single_hop_specific_query_synthesizer: 0.50,
            multi_hop_query_synthesizer: 0.35,
            multi_hop_abstract_query_synthesizer: 0.15,
        },
        with_debugging_logs=False,
        raise_exceptions=False
    )

    df = testset.to_pandas()
    print(f"   Generated {len(df)} questions")

    # Step 5: Quality filtering
    print("\n🔍 Applying quality filters...")
    df = df.drop_duplicates(subset=['user_input'], keep='first')
    df = df[df['reference'].notna() & (df['reference'].str.len() > 0)]
    df = df[df['reference_contexts'].notna()]
    df = df[
        (df['user_input'].str.len() >= 10) &
        (df['user_input'].str.len() <= 500) &
        (df['reference'].str.len() >= 10)
    ]
    print(f"   Filtered to {len(df)} questions")

    # Step 6: Select final 450
    print("\n📊 Selecting final 450 questions...")
    # Simple random selection maintaining balance
    final_df = df.sample(n=min(450, len(df)), random_state=42)

    # Step 7: Create splits
    print("\n📂 Creating dataset splits...")
    test_df, dev_val_df = train_test_split(
        final_df, test_size=100, random_state=42
    )
    dev_df, val_df = train_test_split(
        dev_val_df, test_size=50, random_state=42
    )

    # Save
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    dev_df.to_csv(OUTPUT_DIR / f"dev_set_50_{timestamp}.csv", index=False, encoding='utf-8')
    val_df.to_csv(OUTPUT_DIR / f"val_set_50_{timestamp}.csv", index=False, encoding='utf-8')
    test_df.to_csv(OUTPUT_DIR / f"test_set_350_{timestamp}.csv", index=False, encoding='utf-8')
    final_df.to_csv(OUTPUT_DIR / f"full_set_450_{timestamp}.csv", index=False, encoding='utf-8')

    print(f"\n✅ Dataset generation complete!")
    print(f"   Development: {len(dev_df)} questions")
    print(f"   Validation: {len(val_df)} questions")
    print(f"   Test: {len(test_df)} questions")
    print(f"   Total: {len(final_df)} questions")
    print(f"\n📁 Files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
```

**Run the script**:
```bash
uv run python scripts/generate_evaluation_dataset.py
```

---

## Next Steps

After dataset generation:

1. **Human Review**: Expert validation of sample questions
2. **Pilot Evaluation**: Test with 50-question subset
3. **Full Evaluation**: Run all 16 system variants
4. **Analysis**: Statistical comparison and insights
5. **Documentation**: Results reporting for thesis

---

## Related Files

- **Design Document**: `docs/02_research/evaluation_dataset_design.md`
- **Evaluation Code**: `src/evaluation/evaluator.py`
- **RAGAS Guide**: `docs/02_research/perplexity_deep_research/02_ragas_evaluation.md`
- **Baseline Implementation**: `archive/old_notebooks/eval_ragas.ipynb`
