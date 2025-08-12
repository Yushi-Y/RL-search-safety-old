#!/usr/bin/env python3
"""
Weight Difference Analysis between Original Qwen2.5-Instruct-7B and RL-trained Search-R1 Model
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import os
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import argparse
from pathlib import Path

@dataclass
class ModelConfig:
    """Configuration for model comparison"""
    original_model_id: str = "Qwen/Qwen2.5-7B-Instruct"
    rl_model_id: str = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-ppo"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16
    save_dir: str = "./model_diffing_results"

class WeightAnalyzer:
    """Analyze weight differences between two models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.original_model = None
        self.rl_model = None
        self.original_state_dict = None
        self.rl_state_dict = None
        self.differences = {}
        self.summary_stats = {}
        
        # Create output directory
        os.makedirs(config.save_dir, exist_ok=True)
        
    def load_models(self):
        """Load both models"""
        print("🔄 Loading original model...")
        self.original_model = AutoModelForCausalLM.from_pretrained(
            self.config.original_model_id,
            torch_dtype=self.config.dtype,
            device_map="auto",
            trust_remote_code=True,
            use_flash_attention_2=False
        )
        
        print("🔄 Loading RL-trained model...")
        self.rl_model = AutoModelForCausalLM.from_pretrained(
            self.config.rl_model_id,
            torch_dtype=self.config.dtype,
            device_map="auto",
            trust_remote_code=True,
            use_flash_attention_2=False
        )
        
        # Get state dictionaries
        self.original_state_dict = self.original_model.state_dict()
        self.rl_state_dict = self.rl_model.state_dict()
        
        print(f"✅ Models loaded successfully")
        print(f"   Original model: {self.config.original_model_id}")
        print(f"   RL model: {self.config.rl_model_id}")
        
    def analyze_weight_differences(self):
        """Analyze weight differences between models"""
        print("\n🔍 Analyzing weight differences...")
        
        all_keys = set(self.original_state_dict.keys()) & set(self.rl_state_dict.keys())
        print(f"   Found {len(all_keys)} common parameter keys")
        
        for key in all_keys:
            original_weight = self.original_state_dict[key]
            rl_weight = self.rl_state_dict[key]
            
            # Ensure same shape
            if original_weight.shape != rl_weight.shape:
                print(f"⚠️  Shape mismatch for {key}: {original_weight.shape} vs {rl_weight.shape}")
                continue
                
            # Calculate differences
            diff = rl_weight - original_weight
            abs_diff = torch.abs(diff)
            
            # Statistics
            stats = {
                'mean_diff': diff.mean().item(),
                'std_diff': diff.std().item(),
                'max_diff': diff.max().item(),
                'min_diff': diff.min().item(),
                'mean_abs_diff': abs_diff.mean().item(),
                'std_abs_diff': abs_diff.std().item(),
                'max_abs_diff': abs_diff.max().item(),
                'l2_norm_diff': torch.norm(diff).item(),
                'relative_change': (torch.norm(diff) / torch.norm(original_weight)).item(),
                'shape': list(original_weight.shape),
                'num_params': original_weight.numel()
            }
            
            self.differences[key] = {
                'diff_tensor': diff.cpu().numpy(),
                'stats': stats
            }
            
        print(f"✅ Weight analysis completed for {len(self.differences)} parameters")
        
    def generate_summary_statistics(self):
        """Generate comprehensive summary statistics"""
        print("\n📊 Generating summary statistics...")
        
        # Overall statistics
        total_params = sum(diff['stats']['num_params'] for diff in self.differences.values())
        total_l2_diff = sum(diff['stats']['l2_norm_diff'] for diff in self.differences.values())
        
        # Parameter type analysis
        param_types = {}
        for key in self.differences.keys():
            param_type = self._get_parameter_type(key)
            if param_type not in param_types:
                param_types[param_type] = []
            param_types[param_type].append(key)
        
        # Layer-wise analysis
        layer_stats = self._analyze_by_layer()
        
        self.summary_stats = {
            'total_parameters': total_params,
            'total_l2_difference': total_l2_diff,
            'average_relative_change': np.mean([diff['stats']['relative_change'] for diff in self.differences.values()]),
            'max_relative_change': max([diff['stats']['relative_change'] for diff in self.differences.values()]),
            'parameter_types': param_types,
            'layer_statistics': layer_stats,
            'top_changed_layers': self._get_top_changed_layers(10),
            'model_info': {
                'original_model': self.config.original_model_id,
                'rl_model': self.config.rl_model_id,
                'analysis_date': pd.Timestamp.now().isoformat()
            }
        }
        
        print(f"✅ Summary statistics generated")
        
    def _get_parameter_type(self, key: str) -> str:
        """Categorize parameter by type"""
        if 'attention' in key.lower():
            return 'attention'
        elif 'mlp' in key.lower() or 'feed_forward' in key.lower():
            return 'mlp'
        elif 'embed' in key.lower():
            return 'embedding'
        elif 'norm' in key.lower() or 'ln' in key.lower():
            return 'normalization'
        elif 'lm_head' in key.lower() or 'output' in key.lower():
            return 'output'
        else:
            return 'other'
            
    def _analyze_by_layer(self) -> Dict[str, Any]:
        """Analyze differences by layer"""
        layer_stats = {}
        
        for key in self.differences.keys():
            # Extract layer number if present
            layer_num = self._extract_layer_number(key)
            if layer_num is not None:
                if layer_num not in layer_stats:
                    layer_stats[layer_num] = {
                        'parameters': [],
                        'total_l2_diff': 0,
                        'avg_relative_change': 0
                    }
                layer_stats[layer_num]['parameters'].append(key)
                layer_stats[layer_num]['total_l2_diff'] += self.differences[key]['stats']['l2_norm_diff']
        
        # Calculate averages
        for layer_num in layer_stats:
            params = layer_stats[layer_num]['parameters']
            relative_changes = [self.differences[param]['stats']['relative_change'] for param in params]
            layer_stats[layer_num]['avg_relative_change'] = np.mean(relative_changes)
            
        return layer_stats
        
    def _extract_layer_number(self, key: str) -> int:
        """Extract layer number from parameter key"""
        import re
        match = re.search(r'layers\.(\d+)', key)
        if match:
            return int(match.group(1))
        return None
        
    def _get_top_changed_layers(self, top_k: int = 10) -> List[Tuple[str, float]]:
        """Get top k most changed parameters"""
        changes = [(key, diff['stats']['relative_change']) 
                  for key, diff in self.differences.items()]
        changes.sort(key=lambda x: x[1], reverse=True)
        return changes[:top_k]
        
    def create_visualizations(self):
        """Create comprehensive visualizations"""
        print("\n📈 Creating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # 1. Overall distribution of weight differences
        self._plot_weight_difference_distribution()
        
        # 2. Layer-wise analysis
        self._plot_layer_analysis()
        
        # 3. Parameter type analysis
        self._plot_parameter_type_analysis()
        
        # 4. Top changed parameters
        self._plot_top_changed_parameters()
        
        # 5. Heatmap of layer differences
        self._plot_layer_heatmap()
        
        print(f"✅ Visualizations saved to {self.config.save_dir}")
        
    def _plot_weight_difference_distribution(self):
        """Plot distribution of weight differences"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Weight Difference Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Collect all differences
        all_diffs = []
        all_abs_diffs = []
        all_relative_changes = []
        
        for diff in self.differences.values():
            all_diffs.extend(diff['diff_tensor'].flatten())
            all_abs_diffs.append(diff['stats']['mean_abs_diff'])
            all_relative_changes.append(diff['stats']['relative_change'])
        
        # Plot 1: Distribution of weight differences
        axes[0, 0].hist(all_diffs, bins=100, alpha=0.7, density=True)
        axes[0, 0].set_title('Distribution of Weight Differences')
        axes[0, 0].set_xlabel('Weight Difference')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].axvline(0, color='red', linestyle='--', alpha=0.7)
        
        # Plot 2: Distribution of absolute differences
        axes[0, 1].hist(all_abs_diffs, bins=50, alpha=0.7)
        axes[0, 1].set_title('Distribution of Mean Absolute Differences')
        axes[0, 1].set_xlabel('Mean Absolute Difference')
        axes[0, 1].set_ylabel('Count')
        
        # Plot 3: Distribution of relative changes
        axes[1, 0].hist(all_relative_changes, bins=50, alpha=0.7)
        axes[1, 0].set_title('Distribution of Relative Changes')
        axes[1, 0].set_xlabel('Relative Change (L2 norm ratio)')
        axes[1, 0].set_ylabel('Count')
        
        # Plot 4: Box plot of relative changes by parameter type
        param_types = {}
        for key, diff in self.differences.items():
            param_type = self._get_parameter_type(key)
            if param_type not in param_types:
                param_types[param_type] = []
            param_types[param_type].append(diff['stats']['relative_change'])
        
        if param_types:
            axes[1, 1].boxplot(param_types.values(), labels=param_types.keys())
            axes[1, 1].set_title('Relative Changes by Parameter Type')
            axes[1, 1].set_ylabel('Relative Change')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{self.config.save_dir}/weight_difference_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_layer_analysis(self):
        """Plot layer-wise analysis"""
        if not self.summary_stats['layer_statistics']:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Layer-wise Analysis', fontsize=16, fontweight='bold')
        
        layers = sorted(self.summary_stats['layer_statistics'].keys())
        l2_diffs = [self.summary_stats['layer_statistics'][layer]['total_l2_diff'] for layer in layers]
        relative_changes = [self.summary_stats['layer_statistics'][layer]['avg_relative_change'] for layer in layers]
        param_counts = [len(self.summary_stats['layer_statistics'][layer]['parameters']) for layer in layers]
        
        # Plot 1: L2 difference by layer
        axes[0, 0].plot(layers, l2_diffs, marker='o', linewidth=2, markersize=6)
        axes[0, 0].set_title('Total L2 Difference by Layer')
        axes[0, 0].set_xlabel('Layer Number')
        axes[0, 0].set_ylabel('Total L2 Difference')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Average relative change by layer
        axes[0, 1].plot(layers, relative_changes, marker='s', linewidth=2, markersize=6, color='orange')
        axes[0, 1].set_title('Average Relative Change by Layer')
        axes[0, 1].set_xlabel('Layer Number')
        axes[0, 1].set_ylabel('Average Relative Change')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Parameter count by layer
        axes[1, 0].bar(layers, param_counts, alpha=0.7)
        axes[1, 0].set_title('Number of Parameters by Layer')
        axes[1, 0].set_xlabel('Layer Number')
        axes[1, 0].set_ylabel('Parameter Count')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Scatter plot of L2 diff vs relative change
        axes[1, 1].scatter(l2_diffs, relative_changes, alpha=0.6, s=50)
        axes[1, 1].set_title('L2 Difference vs Relative Change')
        axes[1, 1].set_xlabel('Total L2 Difference')
        axes[1, 1].set_ylabel('Average Relative Change')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.config.save_dir}/layer_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_parameter_type_analysis(self):
        """Plot parameter type analysis"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('Parameter Type Analysis', fontsize=16, fontweight='bold')
        
        param_types = self.summary_stats['parameter_types']
        
        # Calculate statistics for each parameter type
        type_stats = {}
        for param_type, keys in param_types.items():
            relative_changes = [self.differences[key]['stats']['relative_change'] for key in keys]
            type_stats[param_type] = {
                'count': len(keys),
                'avg_relative_change': np.mean(relative_changes),
                'std_relative_change': np.std(relative_changes),
                'total_params': sum(self.differences[key]['stats']['num_params'] for key in keys)
            }
        
        # Plot 1: Parameter count by type
        types = list(type_stats.keys())
        counts = [type_stats[t]['count'] for t in types]
        axes[0].bar(types, counts, alpha=0.7)
        axes[0].set_title('Parameter Count by Type')
        axes[0].set_ylabel('Number of Parameters')
        axes[0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Average relative change by type
        avg_changes = [type_stats[t]['avg_relative_change'] for t in types]
        std_changes = [type_stats[t]['std_relative_change'] for t in types]
        axes[1].bar(types, avg_changes, yerr=std_changes, alpha=0.7, capsize=5)
        axes[1].set_title('Average Relative Change by Type')
        axes[1].set_ylabel('Average Relative Change')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{self.config.save_dir}/parameter_type_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_top_changed_parameters(self):
        """Plot top changed parameters"""
        top_changed = self.summary_stats['top_changed_layers']
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        params = [param for param, _ in top_changed]
        changes = [change for _, change in top_changed]
        
        # Truncate long parameter names for display
        display_params = []
        for param in params:
            if len(param) > 50:
                display_params.append(param[:47] + "...")
            else:
                display_params.append(param)
        
        bars = ax.barh(range(len(display_params)), changes, alpha=0.7)
        ax.set_yticks(range(len(display_params)))
        ax.set_yticklabels(display_params)
        ax.set_xlabel('Relative Change')
        ax.set_title('Top 10 Most Changed Parameters')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, change) in enumerate(zip(bars, changes)):
            ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2, 
                   f'{change:.4f}', va='center', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"{self.config.save_dir}/top_changed_parameters.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_layer_heatmap(self):
        """Plot heatmap of layer differences"""
        if not self.summary_stats['layer_statistics']:
            return
            
        # Create matrix for heatmap
        layers = sorted(self.summary_stats['layer_statistics'].keys())
        param_types = ['attention', 'mlp', 'normalization', 'other']
        
        heatmap_data = np.zeros((len(layers), len(param_types)))
        
        for i, layer in enumerate(layers):
            layer_params = self.summary_stats['layer_statistics'][layer]['parameters']
            for j, param_type in enumerate(param_types):
                type_params = [p for p in layer_params if self._get_parameter_type(p) == param_type]
                if type_params:
                    avg_change = np.mean([self.differences[p]['stats']['relative_change'] for p in type_params])
                    heatmap_data[i, j] = avg_change
        
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto')
        ax.set_xticks(range(len(param_types)))
        ax.set_xticklabels(param_types, rotation=45)
        ax.set_yticks(range(len(layers)))
        ax.set_yticklabels([f'Layer {layer}' for layer in layers])
        ax.set_title('Layer-wise Parameter Type Heatmap\n(Average Relative Change)')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Average Relative Change')
        
        # Add text annotations
        for i in range(len(layers)):
            for j in range(len(param_types)):
                if heatmap_data[i, j] > 0:
                    ax.text(j, i, f'{heatmap_data[i, j]:.3f}', 
                           ha='center', va='center', color='white', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f"{self.config.save_dir}/layer_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
    def save_results(self):
        """Save all results to files"""
        print("\n💾 Saving results...")
        
        # Save summary statistics
        with open(f"{self.config.save_dir}/summary_statistics.json", 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            summary_copy = self.summary_stats.copy()
            summary_copy['model_info']['analysis_date'] = str(summary_copy['model_info']['analysis_date'])
            json.dump(summary_copy, f, indent=2, default=str)
        
        # Save detailed differences
        detailed_results = {}
        for key, diff in self.differences.items():
            detailed_results[key] = {
                'stats': diff['stats'],
                'shape': diff['stats']['shape'],
                'num_params': diff['stats']['num_params']
            }
        
        with open(f"{self.config.save_dir}/detailed_differences.json", 'w') as f:
            json.dump(detailed_results, f, indent=2, default=str)
        
        # Create summary report
        self._create_summary_report()
        
        print(f"✅ Results saved to {self.config.save_dir}")
        
    def _create_summary_report(self):
        """Create a human-readable summary report"""
        report = f"""
# Model Weight Difference Analysis Report

## Model Information
- **Original Model**: {self.config.original_model_id}
- **RL-Trained Model**: {self.config.rl_model_id}
- **Analysis Date**: {self.summary_stats['model_info']['analysis_date']}

## Overall Statistics
- **Total Parameters**: {self.summary_stats['total_parameters']:,}
- **Total L2 Difference**: {self.summary_stats['total_l2_difference']:.6f}
- **Average Relative Change**: {self.summary_stats['average_relative_change']:.6f}
- **Maximum Relative Change**: {self.summary_stats['max_relative_change']:.6f}

## Parameter Type Analysis
"""
        
        for param_type, keys in self.summary_stats['parameter_types'].items():
            relative_changes = [self.differences[key]['stats']['relative_change'] for key in keys]
            report += f"""
### {param_type.title()} Parameters
- **Count**: {len(keys)}
- **Average Relative Change**: {np.mean(relative_changes):.6f}
- **Std Relative Change**: {np.std(relative_changes):.6f}
"""
        
        report += f"""
## Top 10 Most Changed Parameters
"""
        
        for i, (param, change) in enumerate(self.summary_stats['top_changed_layers'], 1):
            report += f"{i}. **{param}**: {change:.6f}\n"
        
        report += f"""
## Layer Analysis
- **Total Layers**: {len(self.summary_stats['layer_statistics'])}
- **Layer with Most Changes**: Layer {max(self.summary_stats['layer_statistics'].keys(), key=lambda x: self.summary_stats['layer_statistics'][x]['avg_relative_change'])}
- **Layer with Least Changes**: Layer {min(self.summary_stats['layer_statistics'].keys(), key=lambda x: self.summary_stats['layer_statistics'][x]['avg_relative_change'])}

## Key Insights
1. **Attention Mechanisms**: {'High' if np.mean([self.differences[key]['stats']['relative_change'] for key in self.summary_stats['parameter_types'].get('attention', [])]) > 0.01 else 'Low'} change observed
2. **MLP Layers**: {'High' if np.mean([self.differences[key]['stats']['relative_change'] for key in self.summary_stats['parameter_types'].get('mlp', [])]) > 0.01 else 'Low'} change observed
3. **Embedding Layer**: {'High' if np.mean([self.differences[key]['stats']['relative_change'] for key in self.summary_stats['parameter_types'].get('embedding', [])]) > 0.01 else 'Low'} change observed

## Files Generated
- `summary_statistics.json`: Detailed numerical statistics
- `detailed_differences.json`: Parameter-by-parameter differences
- `weight_difference_distribution.png`: Distribution plots
- `layer_analysis.png`: Layer-wise analysis
- `parameter_type_analysis.png`: Parameter type comparison
- `top_changed_parameters.png`: Most changed parameters
- `layer_heatmap.png`: Layer heatmap visualization
"""
        
        with open(f"{self.config.save_dir}/analysis_report.md", 'w') as f:
            f.write(report)
            
    def run_full_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚀 Starting Model Weight Difference Analysis")
        print("=" * 60)
        
        self.load_models()
        self.analyze_weight_differences()
        self.generate_summary_statistics()
        self.create_visualizations()
        self.save_results()
        
        print("\n🎉 Analysis completed successfully!")
        print(f"📁 Results saved to: {self.config.save_dir}")
        print("📊 Check the generated visualizations and reports for detailed insights.")

def main():
    parser = argparse.ArgumentParser(description="Analyze weight differences between original and RL-trained models")
    parser.add_argument("--original_model", default="Qwen/Qwen2.5-7B-Instruct", 
                       help="Original model ID")
    parser.add_argument("--rl_model", default="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-ppo", 
                       help="RL-trained model ID")
    parser.add_argument("--save_dir", default="./model_diffing_results", 
                       help="Directory to save results")
    parser.add_argument("--device", default="auto", 
                       help="Device to use (auto, cuda, cpu)")
    
    args = parser.parse_args()
    
    config = ModelConfig(
        original_model_id=args.original_model,
        rl_model_id=args.rl_model,
        device=args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"),
        save_dir=args.save_dir
    )
    
    analyzer = WeightAnalyzer(config)
    analyzer.run_full_analysis()

if __name__ == "__main__":
    main()
