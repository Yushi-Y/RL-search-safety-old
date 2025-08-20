#!/usr/bin/env python3
"""
Analysis script for comparing refusal performance between IT and Search models.
Generates various plots and statistics based on refusal evaluation scores.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from scipy import stats

def load_refusal_data(file_path):
    """Load and parse the refusal evaluation data."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract relevant information
    records = []
    for item in data:
        records.append({
            'question_index': item['question_index'],
            'question': item['question'][:100] + '...' if len(item['question']) > 100 else item['question'],
            'it_score': item['it_score'],
            'search_score': item['search_score'],
            'score_difference': item['it_score'] - item['search_score']  # Positive means IT better
        })
    
    return pd.DataFrame(records)

def create_score_distribution_plot(df):
    """Create distribution plots for IT and Search scores."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Score distributions
    axes[0, 0].hist(df['it_score'], bins=5, alpha=0.7, label='IT Model', color='blue', edgecolor='black')
    axes[0, 0].hist(df['search_score'], bins=5, alpha=0.7, label='Search Model', color='red', edgecolor='black')
    axes[0, 0].set_xlabel('Refusal Score (1=no refusal, 5=complete refusal)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Score Distribution Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Box plot comparison
    box_data = [df['it_score'], df['search_score']]
    axes[0, 1].boxplot(box_data, labels=['IT Model', 'Search Model'])
    axes[0, 1].set_ylabel('Refusal Score')
    axes[0, 1].set_title('Score Distribution Box Plot')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Score difference distribution
    axes[1, 0].hist(df['score_difference'], bins=9, alpha=0.7, color='green', edgecolor='black')
    axes[1, 0].axvline(x=0, color='red', linestyle='--', label='No difference')
    axes[1, 0].set_xlabel('Score Difference (IT - Search)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Score Difference Distribution\n(Positive = IT better)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Scatter plot
    axes[1, 1].scatter(df['it_score'], df['search_score'], alpha=0.6, s=50)
    axes[1, 1].plot([1, 5], [1, 5], 'r--', label='Perfect correlation')
    axes[1, 1].set_xlabel('IT Model Score')
    axes[1, 1].set_ylabel('Search Model Score')
    axes[1, 1].set_title('Score Correlation')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('refusal_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_detailed_comparison_plot(df):
    """Create detailed comparison plots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Score frequency heatmap
    score_combinations = df.groupby(['it_score', 'search_score']).size().unstack(fill_value=0)
    sns.heatmap(score_combinations, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
    axes[0, 0].set_title('Score Combination Frequency')
    axes[0, 0].set_xlabel('Search Model Score')
    axes[0, 0].set_ylabel('IT Model Score')
    
    # Performance categories
    df['performance_category'] = df['score_difference'].apply(
        lambda x: 'IT Much Better' if x >= 2 
        else 'IT Better' if x == 1 
        else 'Similar' if x == 0 
        else 'Search Better' if x == -1 
        else 'Search Much Better'
    )
    
    category_counts = df['performance_category'].value_counts()
    colors = ['darkgreen', 'lightgreen', 'yellow', 'lightcoral', 'darkred']
    axes[0, 1].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%', 
                   colors=colors[:len(category_counts)])
    axes[0, 1].set_title('Performance Comparison Categories')
    
    # Average scores by score range
    score_ranges = ['1', '2', '3', '4', '5']
    it_counts = [sum(df['it_score'] == i) for i in range(1, 6)]
    search_counts = [sum(df['search_score'] == i) for i in range(1, 6)]
    
    x = np.arange(len(score_ranges))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, it_counts, width, label='IT Model', color='blue', alpha=0.7)
    axes[1, 0].bar(x + width/2, search_counts, width, label='Search Model', color='red', alpha=0.7)
    axes[1, 0].set_xlabel('Refusal Score')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Score Distribution by Model')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(score_ranges)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Cumulative score comparison
    it_cumulative = np.cumsum(sorted(df['it_score']))
    search_cumulative = np.cumsum(sorted(df['search_score']))
    
    axes[1, 1].plot(range(len(it_cumulative)), it_cumulative, label='IT Model', linewidth=2)
    axes[1, 1].plot(range(len(search_cumulative)), search_cumulative, label='Search Model', linewidth=2)
    axes[1, 1].set_xlabel('Question Index (sorted by score)')
    axes[1, 1].set_ylabel('Cumulative Score')
    axes[1, 1].set_title('Cumulative Score Comparison')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('refusal_detailed_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_statistical_analysis(df):
    """Print comprehensive statistical analysis."""
    print("=" * 60)
    print("REFUSAL PERFORMANCE STATISTICAL ANALYSIS")
    print("=" * 60)
    
    # Basic statistics
    print("\n📊 BASIC STATISTICS:")
    print(f"Total questions evaluated: {len(df)}")
    print(f"IT Model - Mean: {df['it_score'].mean():.3f}, Std: {df['it_score'].std():.3f}")
    print(f"Search Model - Mean: {df['search_score'].mean():.3f}, Std: {df['search_score'].std():.3f}")
    
    # Score distributions
    print(f"\n📈 SCORE DISTRIBUTIONS:")
    print("IT Model:")
    it_dist = Counter(df['it_score'])
    for score in range(1, 6):
        count = it_dist[score]
        pct = (count / len(df)) * 100
        print(f"  Score {score}: {count} ({pct:.1f}%)")
    
    print("Search Model:")
    search_dist = Counter(df['search_score'])
    for score in range(1, 6):
        count = search_dist[score]
        pct = (count / len(df)) * 100
        print(f"  Score {score}: {count} ({pct:.1f}%)")
    
    # Comparison analysis
    print(f"\n🔍 PERFORMANCE COMPARISON:")
    better_it = sum(df['score_difference'] > 0)
    better_search = sum(df['score_difference'] < 0)
    equal = sum(df['score_difference'] == 0)
    
    print(f"IT performs better: {better_it} ({(better_it/len(df)*100):.1f}%)")
    print(f"Search performs better: {better_search} ({(better_search/len(df)*100):.1f}%)")
    print(f"Equal performance: {equal} ({(equal/len(df)*100):.1f}%)")
    
    # Statistical tests
    print(f"\n🧮 STATISTICAL TESTS:")
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(df['it_score'], df['search_score'])
    print(f"Paired t-test: t-statistic = {t_stat:.4f}, p-value = {p_value:.4f}")
    if p_value < 0.05:
        winner = "IT" if df['it_score'].mean() > df['search_score'].mean() else "Search"
        print(f"  → Significant difference detected! {winner} model performs better.")
    else:
        print(f"  → No significant difference detected.")
    
    # Wilcoxon signed-rank test (non-parametric)
    w_stat, w_p_value = stats.wilcoxon(df['it_score'], df['search_score'])
    print(f"Wilcoxon signed-rank test: W = {w_stat:.4f}, p-value = {w_p_value:.4f}")
    
    # Correlation
    correlation, corr_p = stats.pearsonr(df['it_score'], df['search_score'])
    print(f"Pearson correlation: r = {correlation:.4f}, p-value = {corr_p:.4f}")
    
    # Cohen's d (effect size)
    pooled_std = np.sqrt(((len(df)-1)*df['it_score'].var() + (len(df)-1)*df['search_score'].var()) / (2*len(df)-2))
    cohens_d = (df['it_score'].mean() - df['search_score'].mean()) / pooled_std
    print(f"Cohen's d (effect size): {cohens_d:.4f}")
    
    effect_interpretation = "negligible" if abs(cohens_d) < 0.2 else "small" if abs(cohens_d) < 0.5 else "medium" if abs(cohens_d) < 0.8 else "large"
    print(f"  → Effect size is {effect_interpretation}")
    
    # Perfect refusal analysis
    print(f"\n🎯 PERFECT REFUSAL ANALYSIS (Score = 5):")
    it_perfect = sum(df['it_score'] == 5)
    search_perfect = sum(df['search_score'] == 5)
    both_perfect = sum((df['it_score'] == 5) & (df['search_score'] == 5))
    
    print(f"IT perfect refusals: {it_perfect} ({(it_perfect/len(df)*100):.1f}%)")
    print(f"Search perfect refusals: {search_perfect} ({(search_perfect/len(df)*100):.1f}%)")
    print(f"Both perfect: {both_perfect} ({(both_perfect/len(df)*100):.1f}%)")
    
    # Weak refusal analysis
    print(f"\n⚠️ WEAK REFUSAL ANALYSIS (Score ≤ 2):")
    it_weak = sum(df['it_score'] <= 2)
    search_weak = sum(df['search_score'] <= 2)
    either_weak = sum((df['it_score'] <= 2) | (df['search_score'] <= 2))
    
    print(f"IT weak refusals: {it_weak} ({(it_weak/len(df)*100):.1f}%)")
    print(f"Search weak refusals: {search_weak} ({(search_weak/len(df)*100):.1f}%)")
    print(f"Either model weak: {either_weak} ({(either_weak/len(df)*100):.1f}%)")

def main():
    """Main analysis function."""
    print("Loading refusal evaluation data...")
    
    # Load data
    df = load_refusal_data('refusal_evals/arditi_train_comparison_refusal_eval.json')
    
    # Print statistical analysis
    print_statistical_analysis(df)
    
    # Create visualizations
    print(f"\n📊 Creating visualizations...")
    create_score_distribution_plot(df)
    create_detailed_comparison_plot(df)
    
    print(f"\n✅ Analysis complete! Plots saved as:")
    print(f"  - refusal_performance_analysis.png")
    print(f"  - refusal_detailed_comparison.png")
    
    # Save detailed data for further analysis
    df.to_csv('refusal_performance_data.csv', index=False)
    print(f"  - refusal_performance_data.csv (raw data)")

if __name__ == "__main__":
    main()