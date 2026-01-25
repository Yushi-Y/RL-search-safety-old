"""
Analyze weight changes from base Qwen2.5-7B-Instruct to SearchR1 model.

This helps understand:
1. Which layers change most during RL training
2. Whether changes are localized or global
3. Magnitude and distribution of weight changes
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import defaultdict

# Configuration
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEARCHR1_MODEL = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.2"
OUTPUT_DIR = Path("/VData/kebl6672/ARL/weight_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)


def compute_weight_diff_stats(base_weights: dict, searchr1_weights: dict) -> dict:
    """
    Compute statistics about weight changes.

    Returns:
        Dictionary with per-layer and per-module statistics
    """
    stats = {
        "per_layer": defaultdict(dict),
        "per_module_type": defaultdict(dict),
        "global": {},
    }

    all_diffs = []
    all_relative_changes = []

    # Match keys and compute differences
    common_keys = set(base_weights.keys()) & set(searchr1_weights.keys())

    print(f"\n  Analyzing {len(common_keys)} matched parameters...")

    for key in tqdm(sorted(common_keys), desc="  Computing differences"):
        base_weight = base_weights[key]
        searchr1_weight = searchr1_weights[key]

        if base_weight.shape != searchr1_weight.shape:
            print(f"  Warning: Shape mismatch for {key}")
            continue

        # Compute absolute difference
        diff = (searchr1_weight - base_weight).float()
        abs_diff = diff.abs()

        # Compute relative change (avoid division by zero)
        base_norm = base_weight.norm().item()
        diff_norm = diff.norm().item()
        relative_change = diff_norm / (base_norm + 1e-8)

        all_diffs.append(diff_norm)
        all_relative_changes.append(relative_change)

        # Extract layer number and module type from key
        # Example keys: "model.layers.0.self_attn.q_proj.weight"
        parts = key.split(".")

        layer_idx = None
        module_type = "other"

        if "layers" in parts:
            try:
                layer_idx = int(parts[parts.index("layers") + 1])
            except (ValueError, IndexError):
                pass

        # Determine module type
        if "self_attn" in key:
            module_type = "attention"
        elif "mlp" in key:
            module_type = "mlp"
        elif "embed" in key:
            module_type = "embedding"
        elif "lm_head" in key or "output" in key:
            module_type = "output"

        # Store per-layer stats
        if layer_idx is not None:
            if layer_idx not in stats["per_layer"]:
                stats["per_layer"][layer_idx] = {
                    "total_norm": 0,
                    "diff_norm": 0,
                    "relative_change": 0,
                    "n_params": 0,
                }

            stats["per_layer"][layer_idx]["total_norm"] += base_norm
            stats["per_layer"][layer_idx]["diff_norm"] += diff_norm
            stats["per_layer"][layer_idx]["relative_change"] += relative_change
            stats["per_layer"][layer_idx]["n_params"] += 1

        # Store per-module-type stats
        if module_type not in stats["per_module_type"]:
            stats["per_module_type"][module_type] = {
                "total_norm": 0,
                "diff_norm": 0,
                "relative_change": 0,
                "n_params": 0,
            }

        stats["per_module_type"][module_type]["total_norm"] += base_norm
        stats["per_module_type"][module_type]["diff_norm"] += diff_norm
        stats["per_module_type"][module_type]["relative_change"] += relative_change
        stats["per_module_type"][module_type]["n_params"] += 1

    # Compute global stats
    stats["global"] = {
        "mean_diff": np.mean(all_diffs),
        "median_diff": np.median(all_diffs),
        "std_diff": np.std(all_diffs),
        "mean_relative_change": np.mean(all_relative_changes),
        "median_relative_change": np.median(all_relative_changes),
        "max_relative_change": np.max(all_relative_changes),
    }

    # Average per-layer stats
    for layer_idx in stats["per_layer"]:
        n = stats["per_layer"][layer_idx]["n_params"]
        stats["per_layer"][layer_idx]["relative_change"] /= n

    # Average per-module-type stats
    for module_type in stats["per_module_type"]:
        n = stats["per_module_type"][module_type]["n_params"]
        stats["per_module_type"][module_type]["relative_change"] /= n

    return stats


def analyze_weight_changes():
    """Main analysis pipeline - compare SearchR1 with base Qwen2.5-7B-Instruct."""
    print("="*80)
    print("Weight Change Analysis: Base Qwen2.5-7B-Instruct → SearchR1")
    print("="*80)

    # Load base model
    print(f"\nLoading base model from {BASE_MODEL}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    base_weights = base_model.state_dict()
    print(f"  Loaded {len(base_weights)} parameters")

    # Clean up base model to save memory
    del base_model
    torch.cuda.empty_cache()

    # Load SearchR1 model
    print(f"\nLoading SearchR1 model from {SEARCHR1_MODEL}...")
    searchr1_model = AutoModelForCausalLM.from_pretrained(
        SEARCHR1_MODEL,
        torch_dtype=torch.float32,
        device_map="cpu",
        trust_remote_code=True,
    )
    searchr1_weights = searchr1_model.state_dict()
    print(f"  Loaded {len(searchr1_weights)} parameters")

    # Compute statistics
    print(f"\n{'='*80}")
    print("Computing Weight Differences")
    print(f"{'='*80}")

    stats = compute_weight_diff_stats(base_weights, searchr1_weights)

    results = {
        "global": stats["global"],
        "per_layer": {k: v for k, v in sorted(stats["per_layer"].items())},
        "per_module_type": dict(stats["per_module_type"]),
    }

    # Print summary
    print(f"\n  Global Statistics:")
    print(f"    Mean relative change: {stats['global']['mean_relative_change']*100:.4f}%")
    print(f"    Median relative change: {stats['global']['median_relative_change']*100:.4f}%")
    print(f"    Max relative change: {stats['global']['max_relative_change']*100:.4f}%")

    print(f"\n  Per Module Type:")
    for module_type, module_stats in sorted(
        stats["per_module_type"].items(),
        key=lambda x: x[1]["relative_change"],
        reverse=True
    ):
        print(f"    {module_type:15s}: {module_stats['relative_change']*100:.4f}%")

    # Clean up
    del base_weights, searchr1_weights, searchr1_model
    torch.cuda.empty_cache()

    # Save results
    print(f"\n{'='*80}")
    print("Saving results...")

    with open(OUTPUT_DIR / "weight_changes_searchr1.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {OUTPUT_DIR / 'weight_changes_searchr1.json'}")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_weight_changes(results)

    return results


def plot_weight_changes(results):
    """Plot weight change analysis for SearchR1 vs base model."""

    # Plot 1: Per-layer weight changes
    per_layer_stats = results["per_layer"]

    if per_layer_stats:
        layer_indices = sorted(per_layer_stats.keys())
        relative_changes = [per_layer_stats[idx]["relative_change"] for idx in layer_indices]

        fig, ax = plt.subplots(figsize=(14, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(layer_indices)))
        relative_changes_pct = [x * 100 for x in relative_changes]
        ax.bar(layer_indices, relative_changes_pct, alpha=0.8, color=colors)
        ax.set_xlabel('Layer Index', fontsize=12)
        ax.set_ylabel('Relative Weight Change (%)', fontsize=12)
        ax.set_title('Per-Layer Weight Changes: Base Qwen2.5-7B-Instruct → SearchR1', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "per_layer_changes_searchr1.png", dpi=300, bbox_inches='tight')
        print(f"  Saved: {OUTPUT_DIR / 'per_layer_changes_searchr1.png'}")
        plt.close()

    # Plot 2: Per module type comparison
    per_module_stats = results["per_module_type"]

    if per_module_stats:
        module_types = sorted(per_module_stats.keys())
        relative_changes = [per_module_stats[mt]["relative_change"] for mt in module_types]
        relative_changes_pct = [x * 100 for x in relative_changes]

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
        ax.barh(module_types, relative_changes_pct, alpha=0.8, color=colors[:len(module_types)])
        ax.set_xlabel('Relative Weight Change (%)', fontsize=12)
        ax.set_ylabel('Module Type', fontsize=12)
        ax.set_title('Weight Changes by Module Type: Base → SearchR1', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')

        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "per_module_changes_searchr1.png", dpi=300, bbox_inches='tight')
        print(f"  Saved: {OUTPUT_DIR / 'per_module_changes_searchr1.png'}")
        plt.close()

    # Plot 3: Global statistics summary
    global_stats = results["global"]

    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['Mean', 'Median', 'Max']
    values = [
        global_stats['mean_relative_change'] * 100,
        global_stats['median_relative_change'] * 100,
        global_stats['max_relative_change'] * 100
    ]

    colors = ['#3498db', '#2ecc71', '#e74c3c']
    bars = ax.bar(metrics, values, alpha=0.8, color=colors)
    ax.set_ylabel('Relative Weight Change (%)', fontsize=12)
    ax.set_title('Global Weight Change Statistics: Base → SearchR1', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}%',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "global_stats_searchr1.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / 'global_stats_searchr1.png'}")
    plt.close()

    plt.close('all')


if __name__ == "__main__":
    results = analyze_weight_changes()

    print("\n" + "="*80)
    print("Weight Analysis Complete!")
    print("="*80)
    print("\nGenerated files:")
    print("1. per_layer_changes_searchr1.png - Per-layer weight changes visualization")
    print("2. per_module_changes_searchr1.png - Module type comparison")
    print("3. global_stats_searchr1.png - Global statistics summary")
    print("4. weight_changes_searchr1.json - Detailed statistics in JSON format")
    print("\nInterpretation:")
    print("- Large changes in early layers may affect representation formation")
    print("- Large changes in late layers may affect decision-making")
    print("- Module type differences show which components are most affected by RL training")
    print("- Compare these changes with harmful direction emergence to understand the relationship")
