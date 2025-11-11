#!/usr/bin/env python3
"""
Create a golden testset of 50 highly complex questions from Dasan QA dataset.

Selection based on quantitative complexity criteria:
- Answer length (detailed responses)
- Entity count (rich information)
- Topic diversity (multiple topics)
- KB tag count (broad knowledge requirements)
- Question length (complex questions)
"""

import json
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List, Any
import re


def calculate_complexity_score(record: Dict[str, Any]) -> float:
    """Calculate complexity score based on multiple factors."""

    # Extract fields
    question = record.get("original_question", "")
    answer = record.get("original_answer", "")
    document = record.get("document", "")
    metadata = record.get("metadata", {})

    # Parse entities and kb_tags from document front matter
    entities = []
    kb_tags = []
    secondary_topics = record.get("secondary_topics", [])

    # Extract from document YAML front matter
    if "---" in document:
        front_matter = document.split("---")[1] if document.count("---") >= 2 else ""

        # Extract entities
        entities_match = re.search(r'entities:\s*\[(.*?)\]', front_matter)
        if entities_match:
            entities_str = entities_match.group(1)
            entities = [e.strip() for e in entities_str.split(',') if e.strip()]

        # Extract kb_tags
        kb_tags_match = re.search(r'kb_tags:\s*\[(.*?)\]', front_matter)
        if kb_tags_match:
            kb_tags_str = kb_tags_match.group(1)
            kb_tags = [k.strip() for k in kb_tags_str.split(',') if k.strip()]

    # Calculate individual metrics
    answer_length = len(answer)
    question_length = len(question)
    entity_count = len(entities)
    topic_count = len(secondary_topics)
    kb_tag_count = len(kb_tags)

    # Multi-part question detection (contains multiple '?')
    question_parts = question.count('?')

    # Calculate weighted complexity score
    complexity_score = (
        answer_length * 0.3 +          # Longer answers = more detailed
        entity_count * 20 +             # More entities = richer information
        topic_count * 15 +              # Multiple topics = broader knowledge
        kb_tag_count * 10 +             # More tags = more knowledge domains
        question_length * 0.2 +         # Longer questions = more complex
        question_parts * 5              # Multiple questions = multi-part inquiry
    )

    return complexity_score


def analyze_and_select(
    input_file: Path,
    output_file: Path,
    n_questions: int = 50,
    min_per_category: int = 10
) -> None:
    """Analyze dataset and select top complex questions."""

    print(f"Loading data from {input_file}...")

    # Load all records
    records = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print(f"Loaded {len(records)} Q&A pairs")

    # Calculate complexity scores
    print("\nCalculating complexity scores...")
    for record in records:
        record['complexity_score'] = calculate_complexity_score(record)

    # Sort by complexity score (descending)
    records.sort(key=lambda x: x['complexity_score'], reverse=True)

    # Get category distribution
    categories = [r.get('metadata', {}).get('category', 'Unknown') for r in records]
    category_counts = Counter(categories)

    print("\nCategory distribution in full dataset:")
    for cat, count in category_counts.most_common():
        print(f"  {cat}: {count}")

    # Select top questions with category balance
    print(f"\nSelecting top {n_questions} questions with category balance (min {min_per_category} per category)...")

    selected = []
    category_selected = Counter()

    # First pass: ensure minimum per category
    for record in records:
        category = record.get('metadata', {}).get('category', 'Unknown')

        if category_selected[category] < min_per_category:
            selected.append(record)
            category_selected[category] += 1

            if len(selected) >= n_questions:
                break

    # Second pass: fill remaining slots with highest scores
    if len(selected) < n_questions:
        remaining = n_questions - len(selected)
        selected_ids = {r['dialogue_id'] for r in selected}

        for record in records:
            if record['dialogue_id'] not in selected_ids:
                selected.append(record)
                category = record.get('metadata', {}).get('category', 'Unknown')
                category_selected[category] += 1

                if len(selected) >= n_questions:
                    break

    # Sort selected by score for better readability
    selected.sort(key=lambda x: x['complexity_score'], reverse=True)

    # Save selected questions
    print(f"\nSaving {len(selected)} questions to {output_file}...")
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for record in selected:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    # Generate summary report
    print("\n" + "="*80)
    print("GOLDEN TESTSET SUMMARY")
    print("="*80)

    print(f"\nTotal questions selected: {len(selected)}")

    print("\nCategory breakdown:")
    for cat, count in sorted(category_selected.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(selected)) * 100
        print(f"  {cat}: {count} ({pct:.1f}%)")

    # Calculate statistics
    scores = [r['complexity_score'] for r in selected]
    answer_lengths = [len(r.get('original_answer', '')) for r in selected]
    question_lengths = [len(r.get('original_question', '')) for r in selected]

    # Extract entity counts
    entity_counts = []
    topic_counts = []
    kb_tag_counts = []

    for r in selected:
        doc = r.get('document', '')
        if "---" in doc:
            front_matter = doc.split("---")[1] if doc.count("---") >= 2 else ""

            # Count entities
            entities_match = re.search(r'entities:\s*\[(.*?)\]', front_matter)
            if entities_match:
                entities = [e.strip() for e in entities_match.group(1).split(',') if e.strip()]
                entity_counts.append(len(entities))

            # Count kb_tags
            kb_tags_match = re.search(r'kb_tags:\s*\[(.*?)\]', front_matter)
            if kb_tags_match:
                kb_tags = [k.strip() for k in kb_tags_match.group(1).split(',') if k.strip()]
                kb_tag_counts.append(len(kb_tags))

        # Count topics
        topics = r.get('secondary_topics', [])
        topic_counts.append(len(topics))

    print("\nComplexity score distribution:")
    print(f"  Min:     {min(scores):.1f}")
    print(f"  Max:     {max(scores):.1f}")
    print(f"  Mean:    {sum(scores)/len(scores):.1f}")
    print(f"  Median:  {sorted(scores)[len(scores)//2]:.1f}")

    print("\nAnswer length (characters):")
    print(f"  Min:     {min(answer_lengths)}")
    print(f"  Max:     {max(answer_lengths)}")
    print(f"  Mean:    {sum(answer_lengths)/len(answer_lengths):.1f}")
    print(f"  Median:  {sorted(answer_lengths)[len(answer_lengths)//2]}")

    print("\nQuestion length (characters):")
    print(f"  Min:     {min(question_lengths)}")
    print(f"  Max:     {max(question_lengths)}")
    print(f"  Mean:    {sum(question_lengths)/len(question_lengths):.1f}")
    print(f"  Median:  {sorted(question_lengths)[len(question_lengths)//2]}")

    if entity_counts:
        print("\nEntity count per question:")
        print(f"  Min:     {min(entity_counts)}")
        print(f"  Max:     {max(entity_counts)}")
        print(f"  Mean:    {sum(entity_counts)/len(entity_counts):.1f}")

    if topic_counts:
        print("\nTopic count per question:")
        print(f"  Min:     {min(topic_counts)}")
        print(f"  Max:     {max(topic_counts)}")
        print(f"  Mean:    {sum(topic_counts)/len(topic_counts):.1f}")

    if kb_tag_counts:
        print("\nKB tag count per question:")
        print(f"  Min:     {min(kb_tag_counts)}")
        print(f"  Max:     {max(kb_tag_counts)}")
        print(f"  Mean:    {sum(kb_tag_counts)/len(kb_tag_counts):.1f}")

    print("\n" + "="*80)
    print(f"Golden testset saved to: {output_file}")
    print("="*80)

    # Show top 5 most complex questions as examples
    print("\nTop 5 most complex questions (preview):")
    print("-"*80)
    for i, record in enumerate(selected[:5], 1):
        q = record['original_question'][:150]
        score = record['complexity_score']
        category = record.get('metadata', {}).get('category', 'Unknown')
        print(f"\n{i}. Score: {score:.1f} | Category: {category}")
        print(f"   Q: {q}{'...' if len(record['original_question']) > 150 else ''}")


def main():
    # Paths
    project_root = Path(__file__).parent.parent
    input_file = project_root / "data/AI_HUB_DASAN_QA/05_consolidated/knowledge_docs_full.jsonl"
    output_file = project_root / "data/evaluation/golden_testset_50q_complex.jsonl"

    # Validate input exists
    if not input_file.exists():
        print(f"Error: Input file not found: {input_file}", file=sys.stderr)
        sys.exit(1)

    # Run analysis and selection
    analyze_and_select(
        input_file=input_file,
        output_file=output_file,
        n_questions=50,
        min_per_category=10
    )


if __name__ == "__main__":
    main()
