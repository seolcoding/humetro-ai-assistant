#!/usr/bin/env python3
"""
Benchmark Results Visualization
================================

Generate comprehensive visualizations for RAG benchmark comparison.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (14, 9)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

# Output directory
OUTPUT_DIR = Path("docs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Metric descriptions
METRIC_DESCRIPTIONS = {
    'Faithfulness': 'Factual accuracy based on retrieved context',
    'Relevancy': 'Question-answer relevance alignment',
    'Correctness': 'Overall answer quality vs ground truth'
}


def load_benchmark_data(naive_path: str, kg_path: str) -> Tuple[Dict, Dict]:
    """Load benchmark results from JSON files"""
    with open(naive_path, 'r') as f:
        naive_data = json.load(f)
    with open(kg_path, 'r') as f:
        kg_data = json.load(f)
    return naive_data, kg_data


def extract_scores(data: Dict, complexity: str) -> pd.DataFrame:
    """Extract scores from benchmark data using summary field"""
    records = []

    for model, model_data in data[complexity].items():
        if model == 'metadata':
            continue

        if 'summary' in model_data:
            summary = model_data['summary']
            records.append({
                'Model': model,
                'Faithfulness': summary.get('faithfulness', np.nan),
                'Relevancy': summary.get('answer_relevancy', np.nan),
                'Correctness': summary.get('answer_correctness', np.nan)
            })

    return pd.DataFrame(records)


def add_metric_annotations(ax, y_position=0.02):
    """Add metric description annotations to plot"""
    descriptions = [
        'Faithfulness: Factual accuracy based on context',
        'Relevancy: Question-answer alignment',
        'Correctness: Overall quality vs ground truth'
    ]
    annotation_text = ' | '.join(descriptions)

    ax.text(0.5, y_position, annotation_text,
           transform=ax.transAxes, ha='center', va='bottom',
           fontsize=8, style='italic', color='gray',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.3, edgecolor='gray'))


def plot_overall_comparison(naive_data: Dict, kg_data: Dict):
    """Plot 1: Overall performance comparison across all metrics"""

    # Extract scores
    metrics = ['Faithfulness', 'Relevancy', 'Correctness']

    naive_single = extract_scores(naive_data, 'single_hop')
    naive_multi = extract_scores(naive_data, 'multi_hop')
    kg_single = extract_scores(kg_data, 'single_hop')
    kg_multi = extract_scores(kg_data, 'multi_hop')

    # Calculate averages
    naive_avg = pd.concat([naive_single[metrics], naive_multi[metrics]]).mean()
    kg_avg = pd.concat([kg_single[metrics], kg_multi[metrics]]).mean()

    # Create plot
    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(metrics))
    width = 0.35

    bars1 = ax.bar(x - width/2, naive_avg, width, label='Naive RAG (Vector Similarity)',
                   color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, kg_avg, width, label='KG Simple (Graph + Vector)',
                   color='#e74c3c', alpha=0.8, edgecolor='black', linewidth=1.2)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_xlabel('Evaluation Metrics', fontweight='bold', fontsize=13)
    ax.set_ylabel('Average Score (0-1)', fontweight='bold', fontsize=13)
    ax.set_title('Overall RAG Performance Comparison\n(Averaged across 5 models × 50 questions)',
                fontweight='bold', fontsize=15, pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right', frameon=True, shadow=True, fontsize=11)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, 1.0)

    # Add metric descriptions
    add_metric_annotations(ax, y_position=-0.12)

    # Add improvement annotation
    improvement = ((kg_avg['Faithfulness'] - naive_avg['Faithfulness']) / naive_avg['Faithfulness'] * 100)
    ax.text(0.5, 0.95, f'KG Simple shows +{improvement:.1f}% Faithfulness improvement over Naive RAG',
           transform=ax.transAxes, ha='center', va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6),
           fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_overall_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {OUTPUT_DIR / 'fig1_overall_comparison.png'}")
    plt.close()


def plot_model_performance(naive_data: Dict, kg_data: Dict):
    """Plot 2: Performance by model across both RAG methods"""

    naive_single = extract_scores(naive_data, 'single_hop')
    naive_multi = extract_scores(naive_data, 'multi_hop')
    kg_single = extract_scores(kg_data, 'single_hop')
    kg_multi = extract_scores(kg_data, 'multi_hop')

    # Combine single and multi for each method
    naive_combined = pd.concat([naive_single, naive_multi]).groupby('Model').mean().reset_index()
    kg_combined = pd.concat([kg_single, kg_multi]).groupby('Model').mean().reset_index()

    # Create subplots for each metric
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    metrics = ['Faithfulness', 'Relevancy', 'Correctness']
    colors = ['#3498db', '#e74c3c']

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Prepare data
        models = naive_combined['Model'].tolist()
        naive_scores = naive_combined[metric].tolist()
        kg_scores = kg_combined[metric].tolist()

        x = np.arange(len(models))
        width = 0.35

        bars1 = ax.bar(x - width/2, naive_scores, width, label='Naive RAG',
                      color=colors[0], alpha=0.8, edgecolor='black', linewidth=1)
        bars2 = ax.bar(x + width/2, kg_scores, width, label='KG Simple',
                      color=colors[1], alpha=0.8, edgecolor='black', linewidth=1)

        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if not np.isnan(height):
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{height:.2f}',
                           ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xlabel('Models', fontweight='bold', fontsize=12)
        ax.set_ylabel('Score', fontweight='bold', fontsize=12)
        ax.set_title(f'{metric}\n{METRIC_DESCRIPTIONS[metric]}',
                    fontweight='bold', pad=15, fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
        ax.legend(loc='upper right', frameon=True, fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1.0)

    fig.suptitle('Model Performance Comparison across RAG Methods\n(Averaged: Single-hop + Multi-hop)',
                fontweight='bold', fontsize=16, y=1.0)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig2_model_performance.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {OUTPUT_DIR / 'fig2_model_performance.png'}")
    plt.close()


def plot_complexity_analysis(naive_data: Dict, kg_data: Dict):
    """Plot 3: Single-hop vs Multi-hop performance"""

    fig, axes = plt.subplots(2, 3, figsize=(20, 13))

    methods = [
        ('Naive RAG', naive_data, '#3498db'),
        ('KG Simple', kg_data, '#e74c3c')
    ]

    metrics = ['Faithfulness', 'Relevancy', 'Correctness']

    for row_idx, (method_name, data, color) in enumerate(methods):
        single = extract_scores(data, 'single_hop')
        multi = extract_scores(data, 'multi_hop')

        for col_idx, metric in enumerate(metrics):
            ax = axes[row_idx, col_idx]

            models = single['Model'].tolist()
            single_scores = single[metric].tolist()
            multi_scores = multi[metric].tolist()

            x = np.arange(len(models))
            width = 0.35

            bars1 = ax.bar(x - width/2, single_scores, width, label='Single-hop (Simple)',
                          color=color, alpha=0.6, edgecolor='black', linewidth=1)
            bars2 = ax.bar(x + width/2, multi_scores, width, label='Multi-hop (Complex)',
                          color=color, alpha=0.9, edgecolor='black', linewidth=1)

            # Add value labels
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    if not np.isnan(height):
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{height:.2f}',
                               ha='center', va='bottom', fontsize=8)

            ax.set_ylabel('Score', fontweight='bold', fontsize=11)
            ax.set_title(f'{method_name}: {metric}\n{METRIC_DESCRIPTIONS[metric]}',
                        fontweight='bold', fontsize=10)
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
            ax.legend(loc='upper right', frameon=True, fontsize=9)
            ax.grid(axis='y', alpha=0.3, linestyle='--')
            ax.set_ylim(0, 1.0)

    fig.suptitle('Question Complexity Impact: Single-hop vs Multi-hop\n(Performance by Question Type)',
                fontweight='bold', fontsize=16, y=0.995)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig3_complexity_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {OUTPUT_DIR / 'fig3_complexity_analysis.png'}")
    plt.close()


def plot_radar_chart(naive_data: Dict, kg_data: Dict):
    """Plot 4: Radar chart for comprehensive performance view"""

    # Extract average scores
    naive_single = extract_scores(naive_data, 'single_hop')
    naive_multi = extract_scores(naive_data, 'multi_hop')
    kg_single = extract_scores(kg_data, 'single_hop')
    kg_multi = extract_scores(kg_data, 'multi_hop')

    # Get top 3 models
    top_models = ['GPT-4o-mini', 'EXAONE-3.5-7.8B', 'Gemma3-12B']

    fig, axes = plt.subplots(1, 2, figsize=(18, 8), subplot_kw=dict(projection='polar'))

    methods = [
        ('Naive RAG', [naive_single, naive_multi], axes[0]),
        ('KG Simple', [kg_single, kg_multi], axes[1])
    ]

    categories = ['Faith\n(Single)', 'Faith\n(Multi)',
                 'Relev\n(Single)', 'Relev\n(Multi)',
                 'Correct\n(Single)', 'Correct\n(Multi)']

    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    colors = ['#3498db', '#e74c3c', '#2ecc71']

    for method_name, data_list, ax in methods:
        single_df, multi_df = data_list

        for idx, model in enumerate(top_models):
            single_row = single_df[single_df['Model'] == model]
            multi_row = multi_df[multi_df['Model'] == model]

            if single_row.empty or multi_row.empty:
                continue

            values = [
                single_row['Faithfulness'].values[0],
                multi_row['Faithfulness'].values[0],
                single_row['Relevancy'].values[0],
                multi_row['Relevancy'].values[0],
                single_row['Correctness'].values[0],
                multi_row['Correctness'].values[0]
            ]

            # Handle NaN values
            values = [v if not np.isnan(v) else 0 for v in values]
            values += values[:1]

            ax.plot(angles, values, 'o-', linewidth=2.5, label=model, color=colors[idx], markersize=6)
            ax.fill(angles, values, alpha=0.15, color=colors[idx])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.set_title(f'{method_name}\nMulti-dimensional Performance Profile',
                    fontweight='bold', fontsize=13, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), frameon=True, fontsize=10)
        ax.grid(True, alpha=0.3)

    # Add explanation
    fig.text(0.5, 0.02,
            'Larger coverage area = Better overall performance | Faith=Faithfulness | Relev=Relevancy | Correct=Correctness',
            ha='center', fontsize=10, style='italic', color='gray')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig4_radar_comparison.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {OUTPUT_DIR / 'fig4_radar_comparison.png'}")
    plt.close()


def plot_heatmap(naive_data: Dict, kg_data: Dict):
    """Plot 5: Heatmap showing all scores"""

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    methods = [
        ('Naive RAG', naive_data, axes[0]),
        ('KG Simple', kg_data, axes[1])
    ]

    for method_name, data, ax in methods:
        # Combine single and multi
        single = extract_scores(data, 'single_hop')
        multi = extract_scores(data, 'multi_hop')

        # Create matrix
        models = single['Model'].tolist()
        metrics = ['Faith\n(S)', 'Faith\n(M)',
                  'Relev\n(S)', 'Relev\n(M)',
                  'Correct\n(S)', 'Correct\n(M)']

        matrix = []
        for model in models:
            s_row = single[single['Model'] == model]
            m_row = multi[multi['Model'] == model]

            row = [
                s_row['Faithfulness'].values[0] if not s_row.empty else np.nan,
                m_row['Faithfulness'].values[0] if not m_row.empty else np.nan,
                s_row['Relevancy'].values[0] if not s_row.empty else np.nan,
                m_row['Relevancy'].values[0] if not m_row.empty else np.nan,
                s_row['Correctness'].values[0] if not s_row.empty else np.nan,
                m_row['Correctness'].values[0] if not m_row.empty else np.nan
            ]
            matrix.append(row)

        df = pd.DataFrame(matrix, index=models, columns=metrics)

        sns.heatmap(df, annot=True, fmt='.3f', cmap='RdYlGn', center=0.5,
                   vmin=0, vmax=1.0, ax=ax, cbar_kws={'label': 'Score'},
                   linewidths=0.5, linecolor='gray', annot_kws={'fontsize': 10, 'fontweight': 'bold'})

        ax.set_title(f'{method_name}\nComplete Score Matrix\n(S=Single-hop, M=Multi-hop)',
                    fontweight='bold', fontsize=13, pad=15)
        ax.set_xlabel('Metrics', fontweight='bold', fontsize=12)
        ax.set_ylabel('Models', fontweight='bold', fontsize=12)

    # Add metric explanation
    fig.text(0.5, 0.01,
            'Faith=Faithfulness (context accuracy) | Relev=Relevancy (Q-A alignment) | Correct=Correctness (quality vs GT)',
            ha='center', fontsize=10, style='italic', color='gray')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig5_score_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {OUTPUT_DIR / 'fig5_score_heatmap.png'}")
    plt.close()


def plot_performance_gap(naive_data: Dict, kg_data: Dict):
    """Plot 6: Performance gap between Single and Multi-hop"""

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    methods = [
        ('Naive RAG', naive_data, axes[0]),
        ('KG Simple', kg_data, axes[1])
    ]

    for method_name, data, ax in methods:
        single = extract_scores(data, 'single_hop')
        multi = extract_scores(data, 'multi_hop')

        models = single['Model'].tolist()
        metrics = ['Faithfulness', 'Relevancy', 'Correctness']

        gaps = {metric: [] for metric in metrics}

        for model in models:
            s_row = single[single['Model'] == model]
            m_row = multi[multi['Model'] == model]

            for metric in metrics:
                if not s_row.empty and not m_row.empty:
                    s_val = s_row[metric].values[0]
                    m_val = m_row[metric].values[0]
                    if not np.isnan(s_val) and not np.isnan(m_val):
                        gap = s_val - m_val
                        gaps[metric].append(gap)
                    else:
                        gaps[metric].append(0)
                else:
                    gaps[metric].append(0)

        x = np.arange(len(models))
        width = 0.25

        colors = ['#3498db', '#e74c3c', '#2ecc71']

        for idx, metric in enumerate(metrics):
            offset = width * (idx - 1)
            bars = ax.bar(x + offset, gaps[metric], width, label=metric,
                         color=colors[idx], alpha=0.8, edgecolor='black', linewidth=1)

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if abs(height) > 0.01:
                    ax.text(bar.get_x() + bar.get_width()/2., height,
                           f'{height:+.2f}',
                           ha='center', va='bottom' if height > 0 else 'top',
                           fontsize=8, fontweight='bold')

        ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
        ax.set_xlabel('Models', fontweight='bold', fontsize=12)
        ax.set_ylabel('Performance Gap (Single - Multi)', fontweight='bold', fontsize=12)
        ax.set_title(f'{method_name}\nComplexity Performance Gap\n(+ Better at Simple | - Better at Complex)',
                    fontweight='bold', fontsize=12, pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=10)
        ax.legend(loc='best', frameon=True, fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig6_performance_gap.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {OUTPUT_DIR / 'fig6_performance_gap.png'}")
    plt.close()


def main():
    print("=" * 70)
    print("Benchmark Results Visualization".center(70))
    print("=" * 70)
    print()

    # Load data
    naive_path = "data/evaluation/3way_rag_comparison/naive_rag/benchmark_results_2025-11-06T13-52-47.277363.json"
    kg_path = "data/evaluation/3way_rag_comparison/kg_simple/benchmark_results_2025-11-06T15-32-48.823282.json"

    print("Loading benchmark data...")
    naive_data, kg_data = load_benchmark_data(naive_path, kg_path)
    print(f"✅ Loaded data for {len(naive_data['metadata']['models'])} models")
    print()

    # Generate plots
    print("Generating visualizations...")
    print()

    plot_overall_comparison(naive_data, kg_data)
    plot_model_performance(naive_data, kg_data)
    plot_complexity_analysis(naive_data, kg_data)
    plot_radar_chart(naive_data, kg_data)
    plot_heatmap(naive_data, kg_data)
    plot_performance_gap(naive_data, kg_data)

    print()
    print("=" * 70)
    print("✅ All visualizations completed!".center(70))
    print("=" * 70)
    print(f"\n📁 Output directory: {OUTPUT_DIR}")
    print(f"📊 Generated 6 figures:")
    print(f"   1. fig1_overall_comparison.png - Overall RAG performance")
    print(f"   2. fig2_model_performance.png - Performance by model")
    print(f"   3. fig3_complexity_analysis.png - Single vs Multi-hop")
    print(f"   4. fig4_radar_comparison.png - Multi-dimensional view")
    print(f"   5. fig5_score_heatmap.png - Complete score matrix")
    print(f"   6. fig6_performance_gap.png - Complexity gap analysis")


if __name__ == "__main__":
    main()
