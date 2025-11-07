#!/usr/bin/env python3
"""
Key Finding: Multi-hop Performance with Graph RAG
==================================================

Single comprehensive figure showing open-source models surpassing GPT-4o-mini
in multi-hop reasoning with Knowledge Graph RAG.
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.figsize'] = (16, 10)

OUTPUT_DIR = Path("docs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load benchmark results"""
    with open('data/evaluation/3way_rag_comparison/naive_rag/benchmark_results_2025-11-06T13-52-47.277363.json', 'r') as f:
        naive = json.load(f)
    with open('data/evaluation/3way_rag_comparison/kg_simple/benchmark_results_2025-11-06T15-32-48.823282.json', 'r') as f:
        kg = json.load(f)
    return naive, kg


def create_comprehensive_figure():
    """Create single-page comprehensive figure"""

    naive_data, kg_data = load_data()

    # Create figure with custom layout
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.4, wspace=0.3)

    # Main title
    fig.suptitle('Knowledge Graph RAG Enables Open-Source Models to Surpass GPT-4o-mini\nin Complex Multi-hop Reasoning Tasks',
                fontsize=18, fontweight='bold', y=0.98)

    models = ['GPT-4o-mini', 'EXAONE-3.5-7.8B', 'Qwen3-8B', 'Gemma3-12B', 'GPT-OSS-20B']

    # === PANEL 1a: Single-hop Faithfulness (Top Left, Smaller) ===
    ax1a = fig.add_subplot(gs[0:2, 0])

    naive_single = [naive_data['single_hop'][m]['summary']['faithfulness'] for m in models]
    kg_single = [kg_data['single_hop'][m]['summary']['faithfulness'] for m in models]

    x = np.arange(len(models))
    width = 0.35

    colors_naive = ['#FF6B6B' if m == 'GPT-4o-mini' else '#4ECDC4' for m in models]
    colors_kg = ['#FF6B6B' if m == 'GPT-4o-mini' else '#45B7D1' for m in models]

    bars1a = ax1a.bar(x - width/2, naive_single, width, label='Naive RAG',
                     color=colors_naive, alpha=0.7, edgecolor='black', linewidth=1.2)
    bars2a = ax1a.bar(x + width/2, kg_single, width, label='KG RAG',
                     color=colors_kg, alpha=0.9, edgecolor='black', linewidth=1.2)

    # Add value labels for single-hop
    for bar1, bar2 in zip(bars1a, bars2a):
        h1, h2 = bar1.get_height(), bar2.get_height()
        ax1a.text(bar1.get_x() + bar1.get_width()/2., h1 + 0.01,
                 f'{h1:.3f}', ha='center', va='bottom', fontsize=8)
        ax1a.text(bar2.get_x() + bar2.get_width()/2., h2 + 0.01,
                 f'{h2:.3f}', ha='center', va='bottom', fontsize=8)

    ax1a.set_ylabel('Faithfulness Score', fontweight='bold', fontsize=11)
    ax1a.set_title('Single-hop\n(Simple Questions)', fontweight='bold', fontsize=12, pad=10)
    ax1a.set_xticks(x)
    ax1a.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax1a.legend(loc='upper left', frameon=True, fontsize=9)
    ax1a.set_ylim(0, 1.0)
    ax1a.grid(axis='y', alpha=0.3, linestyle='--')

    # === PANEL 1b: Multi-hop Faithfulness (Top Center-Right, LARGER & EMPHASIZED) ===
    ax1b = fig.add_subplot(gs[0:2, 1:3])

    naive_multi = [naive_data['multi_hop'][m]['summary']['faithfulness'] for m in models]
    kg_multi = [kg_data['multi_hop'][m]['summary']['faithfulness'] for m in models]

    bars1b = ax1b.bar(x - width/2, naive_multi, width, label='Naive RAG (Vector Only)',
                     color=colors_naive, alpha=0.7, edgecolor='black', linewidth=1.5)
    bars2b = ax1b.bar(x + width/2, kg_multi, width, label='KG RAG (Graph + Vector)',
                     color=colors_kg, alpha=0.9, edgecolor='black', linewidth=1.5)

    # Add value labels and improvement arrows for multi-hop
    for i, (bar1, bar2) in enumerate(zip(bars1b, bars2b)):
        h1, h2 = bar1.get_height(), bar2.get_height()
        ax1b.text(bar1.get_x() + bar1.get_width()/2., h1 + 0.01,
                 f'{h1:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        ax1b.text(bar2.get_x() + bar2.get_width()/2., h2 + 0.01,
                 f'{h2:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Add improvement arrows and percentages
        if h2 > h1:
            improvement = ((h2 - h1) / h1) * 100
            ax1b.annotate('', xy=(bar2.get_x() + bar2.get_width()/2., h2 - 0.02),
                         xytext=(bar1.get_x() + bar1.get_width()/2., h1 + 0.02),
                         arrowprops=dict(arrowstyle='->', lw=2, color='green'))
            ax1b.text(x[i], max(h1, h2) + 0.05, f'+{improvement:.1f}%',
                     ha='center', fontsize=9, color='green', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))

    ax1b.set_ylabel('Faithfulness Score', fontweight='bold', fontsize=13)
    ax1b.set_title('⭐ Multi-hop (CRITICAL: Complex Reasoning) ⭐\n(Multiple Information Hops Required)',
                  fontweight='bold', fontsize=14, pad=10, color='#D32F2F')
    ax1b.set_xticks(x)
    ax1b.set_xticklabels(models, rotation=30, ha='right', fontsize=11)
    ax1b.legend(loc='upper left', frameon=True, shadow=True, fontsize=11)
    ax1b.set_ylim(0, 1.0)
    ax1b.grid(axis='y', alpha=0.3, linestyle='--')

    # Highlight GPT-4o-mini KG baseline
    ax1b.axhline(y=kg_multi[0], color='red', linestyle='--', linewidth=2, alpha=0.5)

    # Add annotation for Gemma3-12B achievement
    gemma_idx = models.index('Gemma3-12B')
    ax1b.annotate('🏆 Gemma3-12B\nHighest: 0.891\n(+12% vs GPT-4o-mini KG)',
                 xy=(gemma_idx, kg_multi[gemma_idx]),
                 xytext=(gemma_idx + 0.5, 0.95),
                 fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8, edgecolor='red', linewidth=2),
                 arrowprops=dict(arrowstyle='->', lw=2.5, color='red'))

    # === PANEL 2: Single-hop vs Multi-hop Comparison (Top Right) ===
    ax2 = fig.add_subplot(gs[0, 3])

    complexities = ['Single-hop', 'Multi-hop']
    naive_avg = [
        np.mean([naive_data['single_hop'][m]['summary']['faithfulness'] for m in models]),
        np.mean([naive_data['multi_hop'][m]['summary']['faithfulness'] for m in models])
    ]
    kg_avg = [
        np.mean([kg_data['single_hop'][m]['summary']['faithfulness'] for m in models]),
        np.mean([kg_data['multi_hop'][m]['summary']['faithfulness'] for m in models])
    ]

    x2 = np.arange(len(complexities))
    width2 = 0.35

    bars_n = ax2.bar(x2 - width2/2, naive_avg, width2, label='Naive RAG',
                    color='#4ECDC4', alpha=0.7, edgecolor='black', linewidth=1.2)
    bars_k = ax2.bar(x2 + width2/2, kg_avg, width2, label='KG RAG',
                    color='#45B7D1', alpha=0.9, edgecolor='black', linewidth=1.2)

    for bars in [bars_n, bars_k]:
        for bar in bars:
            h = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., h + 0.01,
                    f'{h:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax2.set_ylabel('Avg Faithfulness', fontweight='bold', fontsize=11)
    ax2.set_title('Question Complexity\nImpact', fontweight='bold', fontsize=12)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(complexities, fontsize=10)
    ax2.legend(loc='upper left', frameon=True, fontsize=9)
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', alpha=0.3)

    # === PANEL 3: Improvement Rates (Middle Right) ===
    ax3 = fig.add_subplot(gs[1, 3])

    improvements = [((kg_multi[i] - naive_multi[i]) / naive_multi[i] * 100)
                   for i in range(len(models))]

    colors_imp = ['#FF6B6B' if m == 'GPT-4o-mini' else '#45B7D1' for m in models]
    bars_imp = ax3.barh(models, improvements, color=colors_imp, alpha=0.8,
                       edgecolor='black', linewidth=1.2)

    for i, (bar, imp) in enumerate(zip(bars_imp, improvements)):
        width_val = bar.get_width()
        ax3.text(width_val + 0.5, bar.get_y() + bar.get_height()/2.,
                f'{imp:+.1f}%', ha='left', va='center', fontsize=10, fontweight='bold')

    ax3.set_xlabel('Improvement (%)', fontweight='bold', fontsize=11)
    ax3.set_title('KG RAG\nImprovement Rate', fontweight='bold', fontsize=12)
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=1)
    ax3.grid(axis='x', alpha=0.3)

    # === PANEL 4: Metric Explanation (Bottom Left) ===
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis('off')

    metric_text = """
    EVALUATION METRICS EXPLAINED

    • Faithfulness (Focus of this analysis)
      → Factual accuracy based on retrieved context
      → Measures if answer statements are supported by evidence
      → Range: 0.0 (no support) to 1.0 (fully supported)
      → Critical for trustworthy AI systems

    • Answer Relevancy
      → Alignment between question and answer
      → Measures if answer addresses the question

    • Answer Correctness
      → Overall quality vs ground truth
      → Combines semantic + factual accuracy
    """

    ax4.text(0.05, 0.95, metric_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8, edgecolor='gray', linewidth=2))

    # === PANEL 5: Test Configuration (Bottom Middle) ===
    ax5 = fig.add_subplot(gs[2, 1:3])
    ax5.axis('off')

    config_text = """
    EXPERIMENT CONFIGURATION

    Dataset:
    • 50 questions (25 single-hop + 25 multi-hop)
    • Seoul metropolitan transportation domain
    • Korean language questions & answers

    RAG Methods:
    • Naive RAG: FAISS vector similarity (5,879 chunks)
    • KG RAG: Neo4j graph + vector (6,544 nodes, 9,554 edges)

    Models Tested:
    • GPT-4o-mini (OpenAI, proprietary)
    • 4 Open-source models (EXAONE, Qwen, Gemma, GPT-OSS)

    Evaluation:
    • Judge: GPT-4o-mini (RAGAS framework)
    • k=4 documents retrieved per query
    """

    ax5.text(0.05, 0.95, config_text, transform=ax5.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, edgecolor='gray', linewidth=2))

    # === PANEL 6: Key Findings (Bottom Right) ===
    ax6 = fig.add_subplot(gs[2, 3])
    ax6.axis('off')

    findings_text = """
    KEY FINDINGS

    ⭐ MULTI-HOP is CRITICAL:
      Complex reasoning shows biggest
      gains with Graph RAG

    ✓ Gemma3-12B achieves HIGHEST score
      (0.891) in multi-hop, surpassing
      GPT-4o-mini KG by 12%

    ✓ GPT-OSS-20B shows LARGEST gain
      (+14.7%) with graph structure
      in multi-hop tasks

    ✓ Single-hop: Minimal difference
      Multi-hop: Significant advantage
      → Graph structure crucial for
      complex reasoning

    ✓ Graph RAG democratizes AI:
      Open-source can match or exceed
      proprietary models with proper
      knowledge organization
    """

    ax6.text(0.05, 0.95, findings_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', family='monospace', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8, edgecolor='green', linewidth=2))

    # Add legend for model types
    proprietary_patch = mpatches.Patch(color='#FF6B6B', label='Proprietary (OpenAI)', alpha=0.7)
    opensource_patch = mpatches.Patch(color='#45B7D1', label='Open-source', alpha=0.7)
    fig.legend(handles=[proprietary_patch, opensource_patch],
              loc='upper right', bbox_to_anchor=(0.98, 0.96),
              frameon=True, shadow=True, fontsize=11, title='Model Type')

    # Footer
    fig.text(0.5, 0.01,
            'Data: 50-question benchmark (Seoul transportation domain) | Judge: GPT-4o-mini (RAGAS) | Graph: Neo4j (6.5K nodes)',
            ha='center', fontsize=9, style='italic', color='gray')

    plt.savefig(OUTPUT_DIR / 'key_finding_comprehensive.png', dpi=300, bbox_inches='tight')
    print(f"✅ Saved: {OUTPUT_DIR / 'key_finding_comprehensive.png'}")
    plt.close()


def main():
    print("=" * 70)
    print("Creating Key Finding Comprehensive Figure".center(70))
    print("=" * 70)
    print()

    create_comprehensive_figure()

    print()
    print("=" * 70)
    print("✅ Key finding figure completed!".center(70))
    print("=" * 70)
    print(f"\n📁 Saved to: {OUTPUT_DIR / 'key_finding_comprehensive.png'}")
    print("\n🎯 Key message:")
    print("   Open-source models + Graph RAG → Surpass GPT-4o-mini")
    print("   Gemma3-12B: 0.891 (highest score, +12% vs GPT-4o-mini)")
    print("   GPT-OSS-20B: +14.7% improvement (largest gain)")


if __name__ == "__main__":
    main()
