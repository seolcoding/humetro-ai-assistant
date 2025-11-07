#!/usr/bin/env python3
"""
Complete 3-Way + FIXED Cypher Comparison
==========================================
Compares all RAG methods including FIXED Cypher
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

# Paths
NAIVE_PATH = Path("data/evaluation/3way_rag_comparison/naive_rag/benchmark_results_2025-11-06T13-52-47.277363.json")
KG_SIMPLE_PATH = Path("data/evaluation/3way_rag_comparison/kg_simple/benchmark_results_2025-11-06T15-32-48.823282.json")
OLD_CYPHER_PATH = Path("data/evaluation/3way_rag_comparison/kg_cypher_generation/benchmark_results_2025-11-06T17-28-23.169289.json")
FIXED_CYPHER_PATH = Path("data/evaluation/full_kg_cypher_fixed/kg_cypher_fixed/benchmark_results_2025-11-06T21-58-22.984286.json")

MODELS = ['GPT-4o-mini', 'EXAONE-3.5-7.8B', 'Qwen3-8B', 'Gemma3-12B', 'GPT-OSS-20B']


def load_all_results():
    """Load all benchmark results"""
    with open(NAIVE_PATH, 'r') as f:
        naive = json.load(f)
    with open(KG_SIMPLE_PATH, 'r') as f:
        kg_simple = json.load(f)
    with open(OLD_CYPHER_PATH, 'r') as f:
        old_cypher = json.load(f)
    with open(FIXED_CYPHER_PATH, 'r') as f:
        fixed_cypher = json.load(f)
    return naive, kg_simple, old_cypher, fixed_cypher


def extract_summary(data: dict, complexity: str, model: str) -> dict:
    """Extract summary metrics"""
    if complexity in data and model in data[complexity]:
        summary = data[complexity][model].get('summary', {})
        return {
            'faithfulness': summary.get('faithfulness', np.nan),
            'relevancy': summary.get('answer_relevancy', np.nan),
            'correctness': summary.get('answer_correctness', np.nan)
        }
    return {'faithfulness': np.nan, 'relevancy': np.nan, 'correctness': np.nan}


def create_full_comparison():
    """Create comprehensive comparison table"""

    naive, kg_simple, old_cypher, fixed_cypher = load_all_results()

    results = []

    for complexity in ['single_hop', 'multi_hop']:
        for model in MODELS:
            naive_scores = extract_summary(naive, complexity, model)
            simple_scores = extract_summary(kg_simple, complexity, model)
            old_scores = extract_summary(old_cypher, complexity, model)
            fixed_scores = extract_summary(fixed_cypher, complexity, model)

            results.append({
                'Complexity': 'Single-hop' if complexity == 'single_hop' else 'Multi-hop',
                'Model': model,
                'Naive_Faith': naive_scores['faithfulness'],
                'Simple_Faith': simple_scores['faithfulness'],
                'Old_Cypher_Faith': old_scores['faithfulness'],
                'Fixed_Cypher_Faith': fixed_scores['faithfulness'],
                'Naive_Relv': naive_scores['relevancy'],
                'Simple_Relv': simple_scores['relevancy'],
                'Old_Cypher_Relv': old_scores['relevancy'],
                'Fixed_Cypher_Relv': fixed_scores['relevancy']
            })

    return pd.DataFrame(results)


def print_analysis(df: pd.DataFrame):
    """Print comprehensive analysis"""

    print("=" * 100)
    print("COMPLETE RAG COMPARISON: Naive vs KG Simple vs OLD Cypher vs FIXED Cypher".center(100))
    print("=" * 100)
    print()

    # Multi-hop focus (most important)
    multi_hop = df[df['Complexity'] == 'Multi-hop'].copy()

    print("📊 MULTI-HOP FAITHFULNESS COMPARISON")
    print("-" * 100)
    print()

    # Average scores
    print("Average Scores:")
    naive_avg = multi_hop['Naive_Faith'].mean()
    simple_avg = multi_hop['Simple_Faith'].mean()
    old_avg = multi_hop['Old_Cypher_Faith'].mean()
    fixed_avg = multi_hop['Fixed_Cypher_Faith'].mean()

    print(f"  Naive RAG:          {naive_avg:.3f}  (baseline)")
    print(f"  KG Simple:          {simple_avg:.3f}  ({((simple_avg - naive_avg) / naive_avg * 100):+.1f}% vs Naive)")
    print(f"  OLD Cypher:         {old_avg:.3f}  ({((old_avg - naive_avg) / naive_avg * 100):+.1f}% vs Naive) ❌")
    print(f"  FIXED Cypher:       {fixed_avg:.3f}  ({((fixed_avg - naive_avg) / naive_avg * 100):+.1f}% vs Naive) ✅")
    print()

    print(f"OLD → FIXED Improvement: {((fixed_avg - old_avg) / old_avg * 100):+.1f}% 🚀")
    print()

    # Model-by-model comparison
    print("\n" + "=" * 100)
    print("MODEL-BY-MODEL MULTI-HOP FAITHFULNESS")
    print("-" * 100)
    print()

    print(f"{'Model':<20} {'Naive':>8} {'KG Simple':>10} {'OLD Cypher':>12} {'FIXED Cypher':>14} {'OLD→FIXED':>12}")
    print("-" * 100)

    for idx, row in multi_hop.iterrows():
        model = row['Model']
        naive = row['Naive_Faith']
        simple = row['Simple_Faith']
        old = row['Old_Cypher_Faith']
        fixed = row['Fixed_Cypher_Faith']

        if not np.isnan(old) and not np.isnan(fixed):
            improvement = ((fixed - old) / old * 100)
            print(f"{model:<20} {naive:>8.3f} {simple:>10.3f} {old:>12.3f} {fixed:>14.3f} {improvement:>11.1f}%")
        else:
            print(f"{model:<20} {naive:>8.3f} {simple:>10.3f} {'N/A':>12} {fixed:>14.3f} {'N/A':>12}")

    print()

    # Ranking
    print("\n" + "=" * 100)
    print("🏆 PERFORMANCE RANKING (Multi-hop Faithfulness Average)")
    print("-" * 100)
    print()

    rankings = [
        ('FIXED Cypher', fixed_avg),
        ('KG Simple', simple_avg),
        ('Naive RAG', naive_avg),
        ('OLD Cypher', old_avg)
    ]
    rankings.sort(key=lambda x: x[1], reverse=True)

    medals = ['🥇', '🥈', '🥉', '❌']
    for i, (method, score) in enumerate(rankings):
        print(f"  {medals[i]} {i+1}. {method:<20} {score:.3f}")

    print()

    # Key insights
    print("\n" + "=" * 100)
    print("💡 KEY INSIGHTS")
    print("-" * 100)
    print()

    print("1. FIXED Cypher vs OLD Cypher:")
    print(f"   - Average improvement: {((fixed_avg - old_avg) / old_avg * 100):+.1f}%")
    print(f"   - OLD was {abs((old_avg - naive_avg) / naive_avg * 100):.1f}% WORSE than Naive")
    print(f"   - FIXED is {((fixed_avg - naive_avg) / naive_avg * 100):+.1f}% BETTER than Naive")
    print()

    print("2. Best Performers (FIXED Cypher, Multi-hop):")
    best_model = multi_hop.loc[multi_hop['Fixed_Cypher_Faith'].idxmax()]
    print(f"   - Best model: {best_model['Model']} ({best_model['Fixed_Cypher_Faith']:.3f})")
    print()

    print("3. Method Comparison:")
    if fixed_avg > simple_avg:
        print(f"   - FIXED Cypher beats KG Simple by {((fixed_avg - simple_avg) / simple_avg * 100):+.1f}%")
    else:
        print(f"   - KG Simple beats FIXED Cypher by {((simple_avg - fixed_avg) / fixed_avg * 100):+.1f}%")
    print()

    print("4. Vector Starting Point Impact:")
    print(f"   - Without vector search (OLD): {old_avg:.3f}")
    print(f"   - With vector search (FIXED): {fixed_avg:.3f}")
    print(f"   - Impact: {((fixed_avg - old_avg) / old_avg * 100):+.1f}% improvement")
    print()

    # Single-hop comparison
    print("\n" + "=" * 100)
    print("📊 SINGLE-HOP COMPARISON (for reference)")
    print("-" * 100)
    print()

    single_hop = df[df['Complexity'] == 'Single-hop'].copy()

    print("Average Scores:")
    print(f"  Naive RAG:          {single_hop['Naive_Faith'].mean():.3f}")
    print(f"  KG Simple:          {single_hop['Simple_Faith'].mean():.3f}")
    print(f"  OLD Cypher:         {single_hop['Old_Cypher_Faith'].mean():.3f}")
    print(f"  FIXED Cypher:       {single_hop['Fixed_Cypher_Faith'].mean():.3f}")

    print("\n" + "=" * 100)


def save_results(df: pd.DataFrame):
    """Save comparison results"""
    output_path = Path("docs/full_4way_comparison.csv")
    df.to_csv(output_path, index=False)
    print(f"\n✅ Full comparison saved to: {output_path}")


def main():
    print("Loading all benchmark results...")
    df = create_full_comparison()

    print_analysis(df)
    save_results(df)

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
