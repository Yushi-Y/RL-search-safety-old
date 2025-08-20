#!/usr/bin/env python3
"""
Analyze correlations between different model refusal scores and create visualizations.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def load_data(filepath):
    """Load the evaluation results JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_scores(data):
    """Extract scores for all four model types."""
    scores_data = []
    
    for item in data:
        # Extract scores, handling potential string/error values
        question_it_score = item.get('question_it_score', None)
        it_score = item.get('it_score', None)
        question_search_score = item.get('question_search_score', None)
        search_score = item.get('search_score', None)
        
        # Convert to numeric, skip if any are non-numeric
        try:
            question_it_score = float(question_it_score) if question_it_score != "ERROR" else None
            it_score = float(it_score) if it_score != "ERROR" else None
            question_search_score = float(question_search_score) if question_search_score != "ERROR" else None
            search_score = float(search_score) if search_score != "ERROR" else None
            
            # Only include if all scores are valid
            if all(score is not None for score in [question_it_score, it_score, question_search_score, search_score]):
                scores_data.append({
                    'question_index': item.get('question_index', len(scores_data)),
                    'question_it': question_it_score,
                    'it': it_score,
                    'question_search': question_search_score,
                    'search': search_score,
                    'question': item.get('question', '')[:100] + '...'  # Truncated question for reference
                })
        except (ValueError, TypeError):
            continue  # Skip invalid entries
    
    return pd.DataFrame(scores_data)

def compute_correlations(df):
    """Compute correlations between different model pairs."""
    correlations = {}
    
    # Define the pairs to analyze
    pairs = [
        ('it', 'search', 'IT vs Search'),
        ('question_it', 'question_search', 'Question-IT vs Question-Search'), 
        ('it', 'question_it', 'IT vs Question-IT'),
        ('search', 'question_search', 'Search vs Question-Search')
    ]
    
    for col1, col2, label in pairs:
        # Pearson correlation
        pearson_r, pearson_p = pearsonr(df[col1], df[col2])
        
        # Spearman correlation (rank-based, good for non-linear relationships)
        spearman_r, spearman_p = spearmanr(df[col1], df[col2])
        
        correlations[label] = {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'col1': col1,
            'col2': col2
        }
    
    return correlations

def create_visualizations(df, correlations, output_dir='.'):
    """Create correlation plots and visualizations."""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create a figure with subplots for all correlation pairs
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Refusal Score Correlations Between Different Models\n(Score 1=No Refusal, Score 5=Complete Refusal)', 
                 fontsize=16, fontweight='bold')
    
    # Define colors for each comparison
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    pairs = [
        ('it', 'search', 'IT vs Search'),
        ('question_it', 'question_search', 'Question-IT vs Question-Search'),
        ('it', 'question_it', 'IT vs Question-IT'),
        ('search', 'question_search', 'Search vs Question-Search')
    ]
    
    for idx, (col1, col2, label) in enumerate(pairs):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Create scatter plot
        ax.scatter(df[col1], df[col2], alpha=0.7, color=colors[idx], s=50)
        
        # Add correlation line
        z = np.polyfit(df[col1], df[col2], 1)
        p = np.poly1d(z)
        ax.plot(df[col1], p(df[col1]), color='red', linestyle='--', alpha=0.8, linewidth=2)
        
        # Get correlation stats
        corr_data = correlations[label]
        
        # Set labels and title
        ax.set_xlabel(f'{col1.replace("_", "-").title()} Score', fontsize=12)
        ax.set_ylabel(f'{col2.replace("_", "-").title()} Score', fontsize=12)
        ax.set_title(f'{label}\nPearson r={corr_data["pearson_r"]:.3f} (p={corr_data["pearson_p"]:.3f})\n'
                    f'Spearman ρ={corr_data["spearman_r"]:.3f} (p={corr_data["spearman_p"]:.3f})', 
                    fontsize=11, fontweight='bold')
        
        # Set axis limits and ticks
        ax.set_xlim(0.5, 5.5)
        ax.set_ylim(0.5, 5.5)
        ax.set_xticks(range(1, 6))
        ax.set_yticks(range(1, 6))
        ax.grid(True, alpha=0.3)
        
        # Add diagonal line for perfect correlation
        ax.plot([1, 5], [1, 5], color='gray', linestyle=':', alpha=0.5, linewidth=1)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/refusal_correlations_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a correlation matrix heatmap
    plt.figure(figsize=(10, 8))
    
    # Prepare correlation matrix
    corr_matrix = df[['question_it', 'it', 'question_search', 'search']].corr()
    
    # Create heatmap
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # Mask upper triangle
    sns.heatmap(corr_matrix, 
                annot=True, 
                cmap='RdBu_r', 
                center=0,
                square=True,
                fmt='.3f',
                cbar_kws={"shrink": 0.8},
                mask=mask)
    
    plt.title('Correlation Matrix: Refusal Scores Between All Models\n(Score 1=No Refusal, Score 5=Complete Refusal)', 
              fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('')
    plt.ylabel('')
    
    # Rename labels for better readability
    labels = ['Question-IT', 'IT', 'Question-Search', 'Search']
    plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
    plt.yticks(range(len(labels)), labels, rotation=0)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/refusal_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create box plots to show score distributions
    plt.figure(figsize=(12, 8))
    
    # Melt the dataframe for easier plotting
    df_melted = df[['question_it', 'it', 'question_search', 'search']].melt()
    df_melted['variable'] = df_melted['variable'].replace({
        'question_it': 'Question-IT',
        'it': 'IT', 
        'question_search': 'Question-Search',
        'search': 'Search'
    })
    
    sns.boxplot(data=df_melted, x='variable', y='value', palette='Set2')
    plt.title('Distribution of Refusal Scores by Model Type\n(Score 1=No Refusal, Score 5=Complete Refusal)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Model Type', fontsize=12)
    plt.ylabel('Refusal Score', fontsize=12)
    plt.xticks(rotation=45)
    plt.ylim(0.5, 5.5)
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/refusal_score_distributions.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_correlation_summary(correlations, df):
    """Print a detailed summary of correlation analysis."""
    print("="*80)
    print("REFUSAL SCORE CORRELATION ANALYSIS")
    print("="*80)
    print(f"Analysis based on {len(df)} questions")
    print(f"Score Scale: 1 = No Refusal (Unsafe), 5 = Complete Refusal (Safe)")
    print()
    
    # Print basic statistics
    print("BASIC STATISTICS:")
    print("-" * 40)
    stats_df = df[['question_it', 'it', 'question_search', 'search']].describe()
    stats_df.index = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    stats_df.columns = ['Question-IT', 'IT', 'Question-Search', 'Search']
    print(stats_df.round(3))
    print()
    
    # Print correlations
    print("CORRELATION ANALYSIS:")
    print("-" * 40)
    
    for label, corr_data in correlations.items():
        print(f"\n{label}:")
        print(f"  Pearson r = {corr_data['pearson_r']:.3f} (p = {corr_data['pearson_p']:.3f})")
        
        # Interpret Pearson correlation strength
        r = abs(corr_data['pearson_r'])
        if r >= 0.8:
            strength = "Very Strong"
        elif r >= 0.6:
            strength = "Strong"
        elif r >= 0.4:
            strength = "Moderate"
        elif r >= 0.2:
            strength = "Weak"
        else:
            strength = "Very Weak"
        
        direction = "Positive" if corr_data['pearson_r'] > 0 else "Negative"
        print(f"  Interpretation: {strength} {direction} correlation")
        
        print(f"  Spearman ρ = {corr_data['spearman_r']:.3f} (p = {corr_data['spearman_p']:.3f})")
        
        # Statistical significance
        if corr_data['pearson_p'] < 0.001:
            sig_level = "highly significant (p < 0.001)"
        elif corr_data['pearson_p'] < 0.01:
            sig_level = "very significant (p < 0.01)"
        elif corr_data['pearson_p'] < 0.05:
            sig_level = "significant (p < 0.05)"
        else:
            sig_level = "not significant (p ≥ 0.05)"
        
        print(f"  Statistical significance: {sig_level}")
    
    print("\n" + "="*80)
    print("KEY INSIGHTS:")
    print("="*80)
    
    # Find highest and lowest correlations
    pearson_correlations = [(label, corr['pearson_r']) for label, corr in correlations.items()]
    pearson_correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    print(f"• Strongest correlation: {pearson_correlations[0][0]} (r = {pearson_correlations[0][1]:.3f})")
    print(f"• Weakest correlation: {pearson_correlations[-1][0]} (r = {pearson_correlations[-1][1]:.3f})")
    
    # Compare prompting strategies
    question_vs_standard_it = correlations['IT vs Question-IT']['pearson_r']
    question_vs_standard_search = correlations['Search vs Question-Search']['pearson_r']
    
    print(f"\n• Question-based vs Standard prompting consistency:")
    print(f"  - IT models: r = {question_vs_standard_it:.3f}")
    print(f"  - Search models: r = {question_vs_standard_search:.3f}")
    
    if abs(question_vs_standard_it) > abs(question_vs_standard_search):
        print(f"  → IT models show more consistent behavior across prompting strategies")
    else:
        print(f"  → Search models show more consistent behavior across prompting strategies")
    
    # Compare model types
    it_vs_search = correlations['IT vs Search']['pearson_r']
    question_it_vs_question_search = correlations['Question-IT vs Question-Search']['pearson_r']
    
    print(f"\n• IT vs Search model consistency:")
    print(f"  - Standard prompting: r = {it_vs_search:.3f}")
    print(f"  - Question-based prompting: r = {question_it_vs_question_search:.3f}")
    
    if abs(it_vs_search) > abs(question_it_vs_question_search):
        print(f"  → Standard prompting shows more consistent behavior across model types")
    else:
        print(f"  → Question-based prompting shows more consistent behavior across model types")

def main():
    """Main function to run the correlation analysis."""
    # Load data
    data_file = "refusal_evals/arditi_val_all_comparison_refusal_eval.json"
    print(f"Loading data from {data_file}...")
    
    try:
        data = load_data(data_file)
        print(f"Loaded {len(data)} evaluation results")
    except FileNotFoundError:
        print(f"Error: Could not find {data_file}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {data_file}")
        return
    
    # Extract scores
    df = extract_scores(data)
    print(f"Extracted valid scores for {len(df)} questions")
    
    if len(df) == 0:
        print("Error: No valid scores found in the data")
        return
    
    # Compute correlations
    correlations = compute_correlations(df)
    
    # Print summary
    print_correlation_summary(correlations, df)
    
    # Create visualizations
    print("\nCreating visualizations...")
    create_visualizations(df, correlations)
    
    print("\nAnalysis complete! Check the generated plots:")
    print("- refusal_correlations_scatter.png")
    print("- refusal_correlation_matrix.png") 
    print("- refusal_score_distributions.png")

if __name__ == "__main__":
    main()