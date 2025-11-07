#!/usr/bin/env python3
"""
3-Way RAG Comparison Analysis
==============================
Comprehensive analysis of Naive RAG, KG Simple, and KG Cypher Generation
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

# Paths to benchmark results
NAIVE_PATH = Path("data/evaluation/3way_rag_comparison/naive_rag/benchmark_results_2025-11-06T13-52-47.277363.json")
KG_SIMPLE_PATH = Path("data/evaluation/3way_rag_comparison/kg_simple/benchmark_results_2025-11-06T15-32-48.823282.json")
KG_CYPHER_PATH = Path("data/evaluation/3way_rag_comparison/kg_cypher_generation/benchmark_results_2025-11-06T17-28-23.169289.json")

MODELS = ['GPT-4o-mini', 'EXAONE-3.5-7.8B', 'Qwen3-8B', 'Gemma3-12B', 'GPT-OSS-20B']
METRICS = ['faithfulness', 'answer_relevancy', 'answer_correctness']


def load_results():
    """Load all three benchmark results"""
    with open(NAIVE_PATH, 'r') as f:
        naive = json.load(f)
    with open(KG_SIMPLE_PATH, 'r') as f:
        kg_simple = json.load(f)
    with open(KG_CYPHER_PATH, 'r') as f:
        kg_cypher = json.load(f)
    return naive, kg_simple, kg_cypher


def extract_summary(data: Dict, complexity: str, model: str) -> Dict:
    """Extract summary metrics for a specific model and complexity"""
    if complexity in data and model in data[complexity]:
        summary = data[complexity][model].get('summary', {})
        return {
            'faithfulness': summary.get('faithfulness', 0),
            'relevancy': summary.get('answer_relevancy', 0),
            'correctness': summary.get('answer_correctness', 0)
        }
    return {'faithfulness': 0, 'relevancy': 0, 'correctness': 0}


def create_comparison_table(naive, kg_simple, kg_cypher):
    """Create comprehensive comparison table"""

    results = []

    for complexity in ['single_hop', 'multi_hop']:
        for model in MODELS:
            naive_scores = extract_summary(naive, complexity, model)
            simple_scores = extract_summary(kg_simple, complexity, model)
            cypher_scores = extract_summary(kg_cypher, complexity, model)

            results.append({
                'Complexity': 'Single-hop' if complexity == 'single_hop' else 'Multi-hop',
                'Model': model,
                'Naive_Faith': naive_scores['faithfulness'],
                'Simple_Faith': simple_scores['faithfulness'],
                'Cypher_Faith': cypher_scores['faithfulness'],
                'Naive_Relv': naive_scores['relevancy'],
                'Simple_Relv': simple_scores['relevancy'],
                'Cypher_Relv': cypher_scores['relevancy'],
                'Naive_Corr': naive_scores['correctness'],
                'Simple_Corr': simple_scores['correctness'],
                'Cypher_Corr': cypher_scores['correctness']
            })

    return pd.DataFrame(results)


def calculate_improvements(df: pd.DataFrame):
    """Calculate improvement rates"""

    df['Simple_vs_Naive_Faith'] = ((df['Simple_Faith'] - df['Naive_Faith']) / df['Naive_Faith'] * 100)
    df['Cypher_vs_Naive_Faith'] = ((df['Cypher_Faith'] - df['Naive_Faith']) / df['Naive_Faith'] * 100)
    df['Cypher_vs_Simple_Faith'] = ((df['Cypher_Faith'] - df['Simple_Faith']) / df['Simple_Faith'] * 100)

    return df


def print_summary_report(df: pd.DataFrame):
    """Print comprehensive summary report"""

    print("=" * 80)
    print("3-WAY RAG COMPARISON: COMPREHENSIVE RESULTS".center(80))
    print("=" * 80)
    print()

    # Overall averages
    print("📊 OVERALL AVERAGES (All Models)")
    print("-" * 80)

    for complexity in ['Single-hop', 'Multi-hop']:
        subset = df[df['Complexity'] == complexity]

        print(f"\n{complexity}:")
        print(f"  Naive RAG:")
        print(f"    Faithfulness:  {subset['Naive_Faith'].mean():.3f}")
        print(f"    Relevancy:     {subset['Naive_Relv'].mean():.3f}")
        print(f"    Correctness:   {subset['Naive_Corr'].mean():.3f}")

        print(f"  KG Simple:")
        print(f"    Faithfulness:  {subset['Simple_Faith'].mean():.3f}")
        print(f"    Relevancy:     {subset['Simple_Relv'].mean():.3f}")
        print(f"    Correctness:   {subset['Simple_Corr'].mean():.3f}")

        print(f"  KG Cypher Generation:")
        print(f"    Faithfulness:  {subset['Cypher_Faith'].mean():.3f}")
        print(f"    Relevancy:     {subset['Cypher_Relv'].mean():.3f}")
        print(f"    Correctness:   {subset['Cypher_Corr'].mean():.3f}")

    print("\n" + "=" * 80)
    print("🏆 TOP PERFORMERS (Multi-hop Faithfulness)")
    print("-" * 80)

    multi_hop = df[df['Complexity'] == 'Multi-hop'].copy()

    # Best Naive RAG
    best_naive = multi_hop.loc[multi_hop['Naive_Faith'].idxmax()]
    print(f"\nBest Naive RAG:")
    print(f"  {best_naive['Model']}: {best_naive['Naive_Faith']:.3f}")

    # Best KG Simple
    best_simple = multi_hop.loc[multi_hop['Simple_Faith'].idxmax()]
    print(f"\nBest KG Simple:")
    print(f"  {best_simple['Model']}: {best_simple['Simple_Faith']:.3f}")

    # Best KG Cypher
    best_cypher = multi_hop.loc[multi_hop['Cypher_Faith'].idxmax()]
    print(f"\nBest KG Cypher Generation:")
    print(f"  {best_cypher['Model']}: {best_cypher['Cypher_Faith']:.3f}")

    # Overall best
    print(f"\n{'*' * 80}")
    print(f"🥇 OVERALL BEST PERFORMANCE (Multi-hop):")

    all_scores = []
    for idx, row in multi_hop.iterrows():
        all_scores.append(('Naive', row['Model'], row['Naive_Faith']))
        all_scores.append(('KG Simple', row['Model'], row['Simple_Faith']))
        all_scores.append(('KG Cypher', row['Model'], row['Cypher_Faith']))

    all_scores.sort(key=lambda x: x[2], reverse=True)

    for i, (method, model, score) in enumerate(all_scores[:3], 1):
        print(f"  {i}. {method} + {model}: {score:.3f}")
    print(f"{'*' * 80}")

    print("\n" + "=" * 80)
    print("📈 IMPROVEMENT ANALYSIS (Multi-hop Faithfulness)")
    print("-" * 80)

    multi_hop_imp = calculate_improvements(multi_hop)

    print("\nKG Simple vs Naive RAG:")
    for idx, row in multi_hop_imp.iterrows():
        improvement = row['Simple_vs_Naive_Faith']
        symbol = "🟢" if improvement > 0 else "🔴"
        print(f"  {symbol} {row['Model']}: {improvement:+.1f}%")

    print("\nKG Cypher Generation vs Naive RAG:")
    for idx, row in multi_hop_imp.iterrows():
        improvement = row['Cypher_vs_Naive_Faith']
        symbol = "🟢" if improvement > 0 else "🔴"
        print(f"  {symbol} {row['Model']}: {improvement:+.1f}%")

    print("\nKG Cypher vs KG Simple:")
    for idx, row in multi_hop_imp.iterrows():
        improvement = row['Cypher_vs_Simple_Faith']
        symbol = "🟢" if improvement > 0 else "🔴"
        print(f"  {symbol} {row['Model']}: {improvement:+.1f}%")

    print("\n" + "=" * 80)
    print("💡 KEY INSIGHTS")
    print("-" * 80)

    # Calculate average improvements
    avg_simple_vs_naive = multi_hop_imp['Simple_vs_Naive_Faith'].mean()
    avg_cypher_vs_naive = multi_hop_imp['Cypher_vs_Naive_Faith'].mean()
    avg_cypher_vs_simple = multi_hop_imp['Cypher_vs_Simple_Faith'].mean()

    print(f"\n1. Average Improvement (Multi-hop Faithfulness):")
    print(f"   - KG Simple vs Naive:           {avg_simple_vs_naive:+.1f}%")
    print(f"   - KG Cypher vs Naive:           {avg_cypher_vs_naive:+.1f}%")
    print(f"   - KG Cypher vs KG Simple:       {avg_cypher_vs_simple:+.1f}%")

    # Count positive improvements
    simple_wins = (multi_hop_imp['Simple_vs_Naive_Faith'] > 0).sum()
    cypher_wins = (multi_hop_imp['Cypher_vs_Naive_Faith'] > 0).sum()
    cypher_vs_simple_wins = (multi_hop_imp['Cypher_vs_Simple_Faith'] > 0).sum()

    print(f"\n2. Win Rate (Models showing improvement):")
    print(f"   - KG Simple beats Naive:        {simple_wins}/5 models ({simple_wins/5*100:.0f}%)")
    print(f"   - KG Cypher beats Naive:        {cypher_wins}/5 models ({cypher_wins/5*100:.0f}%)")
    print(f"   - KG Cypher beats KG Simple:    {cypher_vs_simple_wins}/5 models ({cypher_vs_simple_wins/5*100:.0f}%)")

    # Find best method
    print(f"\n3. Best RAG Method by Metric:")

    for metric_short, metric_full in [('Faith', 'Faithfulness'), ('Relv', 'Relevancy'), ('Corr', 'Correctness')]:
        naive_col = f'Naive_{metric_short}'
        simple_col = f'Simple_{metric_short}'
        cypher_col = f'Cypher_{metric_short}'

        naive_avg = multi_hop[naive_col].mean()
        simple_avg = multi_hop[simple_col].mean()
        cypher_avg = multi_hop[cypher_col].mean()

        best_method = max([('Naive', naive_avg), ('KG Simple', simple_avg), ('KG Cypher', cypher_avg)], key=lambda x: x[1])

        print(f"   - {metric_full}: {best_method[0]} ({best_method[1]:.3f})")

    print("\n" + "=" * 80)


def save_detailed_csv(df: pd.DataFrame):
    """Save detailed comparison to CSV"""
    output_path = Path("docs/3way_comparison_detailed.csv")
    df_with_improvements = calculate_improvements(df)
    df_with_improvements.to_csv(output_path, index=False)
    print(f"\n✅ Detailed results saved to: {output_path}")


def main():
    print("Loading benchmark results...")
    naive, kg_simple, kg_cypher = load_results()

    print("Creating comparison table...")
    df = create_comparison_table(naive, kg_simple, kg_cypher)

    print_summary_report(df)
    save_detailed_csv(df)

    print("\n✅ Analysis complete!")


if __name__ == "__main__":
    main()
