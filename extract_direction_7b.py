"""
Extract harmful search direction from base Qwen2.5-7B-Instruct and SearchR1 models.

This script:
1. Loads base model and SearchR1 model
2. Extracts hidden representations for harmful vs safe conditions
3. Computes harmful search direction via mean difference
4. Compares directions between base and SearchR1
5. Saves directions for steering/projection analysis
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
from tqdm import tqdm
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
import random

# Configuration
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SEARCHR1_MODEL = "PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-it-em-grpo-v0.2"
OUTPUT_DIR = Path("/VData/kebl6672/ARL/direction_analysis_7b")
OUTPUT_DIR.mkdir(exist_ok=True)

# Dataset paths
HARMFUL_INSTRUCTIONS_PATH = Path("/VData/kebl6672/ARL/refusal_datasets/harmful_combined.json")
HARMLESS_INSTRUCTIONS_PATH = Path("/VData/kebl6672/ARL/harmless_datasets/harmless_val.json")

# Dataset sizes (set to None to use all)
MAX_HARMFUL = None  # Use all 871 harmful examples
MAX_HARMLESS = 1000  # Randomly sample 1000 harmless examples

# Layer to extract from (middle to late layers)
LAYER_INDICES = [14, 18, 22, 26, 28]  # For Qwen 7B (28 layers total, 28 is last layer)

# Random seed for reproducibility
RANDOM_SEED = 42


def load_instructions() -> Tuple[List[str], List[str]]:
    """Load harmful and harmless instructions from datasets."""
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)

    # Load harmful instructions
    with open(HARMFUL_INSTRUCTIONS_PATH, 'r') as f:
        harmful_data = json.load(f)
    harmful_instructions = [item['instruction'] for item in harmful_data]

    # Apply harmful limit if specified
    if MAX_HARMFUL is not None and len(harmful_instructions) > MAX_HARMFUL:
        harmful_instructions = random.sample(harmful_instructions, MAX_HARMFUL)
        print(f"Randomly sampled {MAX_HARMFUL} harmful instructions from {len(harmful_data)}")
    else:
        print(f"Using all {len(harmful_instructions)} harmful instructions")

    # Load harmless instructions
    with open(HARMLESS_INSTRUCTIONS_PATH, 'r') as f:
        harmless_data = json.load(f)
    harmless_instructions = [item['instruction'] for item in harmless_data]

    # Apply harmless limit if specified
    if MAX_HARMLESS is not None and len(harmless_instructions) > MAX_HARMLESS:
        harmless_instructions = random.sample(harmless_instructions, MAX_HARMLESS)
        print(f"Randomly sampled {MAX_HARMLESS} harmless instructions from {len(harmless_data)}")
    else:
        print(f"Using all {len(harmless_instructions)} harmless instructions")

    return harmful_instructions, harmless_instructions


def load_model_and_tokenizer(model_path: str, device: str = "cuda"):
    """Load model and tokenizer."""
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def extract_hidden_states(
    model,
    tokenizer,
    instruction: str,
    layer_idx: int,
    force_search: bool = False,
) -> torch.Tensor:
    """
    Extract hidden states at specified layer.

    Args:
        model: The language model
        tokenizer: The tokenizer
        instruction: The harmful or harmless instruction
        layer_idx: Which layer to extract from
        force_search: If True, append <search> token after instruction (attack condition)

    Returns:
        Hidden states tensor [hidden_dim]
    """
    # Prepare the instruction with system prompt (similar to attack code)
    question = instruction.strip()
    if question and question[-1] != '?':
        question += '?'

    # System prompt that teaches the model how to use search
    system_prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. Question: {question}\n"""

    # Apply chat template if available
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": system_prompt}],
            add_generation_prompt=True,
            tokenize=False
        )
    else:
        prompt = system_prompt

    # Attack: append <search> token AFTER the full prompt
    if force_search:
        prompt = prompt + "<search>"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    # Extract hidden states from specified layer
    hidden_states = outputs.hidden_states[layer_idx]  # [batch, seq_len, hidden_dim]

    # Return last token's hidden state (where decision is made)
    return hidden_states[0, -1, :].cpu()  # [hidden_dim]


def extract_reps_for_checkpoint(
    model,
    tokenizer,
    layer_idx: int,
    harmful_instructions: List[str],
    harmless_instructions: List[str],
) -> Dict[str, torch.Tensor]:
    """
    Extract representations for a given checkpoint (no direction computation).

    Returns:
        Dictionary with unsafe_reps, safe_refusal_reps, benign_reps tensors
    """
    print(f"  Extracting representations from layer {layer_idx}...")

    # Collect representations
    unsafe_reps = []  # Harmful + forced search
    safe_refusal_reps = []  # Harmful + no forced search (refusal)
    benign_reps = []  # Safe instructions for reference

    # Process harmful instructions
    for instruction in tqdm(harmful_instructions, desc="  Harmful instructions"):
        # Unsafe: forced search before refusal
        unsafe_rep = extract_hidden_states(
            model, tokenizer, instruction, layer_idx, force_search=True
        )
        unsafe_reps.append(unsafe_rep)

        # Safe: normal refusal
        safe_rep = extract_hidden_states(
            model, tokenizer, instruction, layer_idx, force_search=False
        )
        safe_refusal_reps.append(safe_rep)

    # Process safe instructions for reference
    for instruction in tqdm(harmless_instructions, desc="  Harmless instructions"):
        benign_rep = extract_hidden_states(
            model, tokenizer, instruction, layer_idx, force_search=False
        )
        benign_reps.append(benign_rep)

    # Stack into tensors
    unsafe_reps = torch.stack(unsafe_reps)  # [n_harmful, hidden_dim]
    safe_refusal_reps = torch.stack(safe_refusal_reps)
    benign_reps = torch.stack(benign_reps)

    return {
        "unsafe": unsafe_reps,
        "safe_refusal": safe_refusal_reps,
        "benign": benign_reps,
    }


def compute_directions_and_stats(
    reps: Dict[str, torch.Tensor],
    layer_idx: int,
    reference_directions: Dict[str, torch.Tensor] = None,
) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """
    Compute directions and statistics from representations.

    Args:
        reps: Dictionary with unsafe, safe_refusal, benign tensors
        layer_idx: Layer index for stats
        reference_directions: If provided, use these directions for projection instead of computing new ones.
                             This enables Option A (project onto IT's directions).

    Returns:
        directions: Dictionary with jailbreak and harmfulness direction tensors
        stats: Dictionary with analysis statistics
    """
    unsafe_reps = reps["unsafe"]
    safe_refusal_reps = reps["safe_refusal"]
    benign_reps = reps["benign"]

    # Compute directions from THIS model's representations
    jailbreak_direction = (unsafe_reps.mean(dim=0) - safe_refusal_reps.mean(dim=0))
    jailbreak_direction_norm = jailbreak_direction.norm().item()
    jailbreak_direction = jailbreak_direction / jailbreak_direction.norm()  # Normalize

    harmfulness_direction = (safe_refusal_reps.mean(dim=0) - benign_reps.mean(dim=0))
    harmfulness_direction_norm = harmfulness_direction.norm().item()
    harmfulness_direction = harmfulness_direction / harmfulness_direction.norm()  # Normalize

    # Use reference directions for projection if provided (Option A)
    if reference_directions is not None:
        proj_jailbreak_dir = reference_directions["jailbreak"]
        proj_harmfulness_dir = reference_directions["harmfulness"]
    else:
        proj_jailbreak_dir = jailbreak_direction
        proj_harmfulness_dir = harmfulness_direction

    # Compute statistics using projection directions
    stats = {
        "layer_idx": layer_idx,
        # Mean norms
        "unsafe_mean_norm": unsafe_reps.mean(dim=0).norm().item(),
        "safe_refusal_mean_norm": safe_refusal_reps.mean(dim=0).norm().item(),
        "benign_mean_norm": benign_reps.mean(dim=0).norm().item(),

        # Jailbreak direction statistics (projected onto reference direction)
        "jailbreak_direction_norm": jailbreak_direction_norm,
        "jailbreak_unsafe_projection": (unsafe_reps @ proj_jailbreak_dir).mean().item(),
        "jailbreak_safe_projection": (safe_refusal_reps @ proj_jailbreak_dir).mean().item(),
        "jailbreak_benign_projection": (benign_reps @ proj_jailbreak_dir).mean().item(),
        "jailbreak_separation": (
            (unsafe_reps @ proj_jailbreak_dir).mean() - (safe_refusal_reps @ proj_jailbreak_dir).mean()
        ).item(),

        # Harmfulness direction statistics (projected onto reference direction)
        "harmfulness_direction_norm": harmfulness_direction_norm,
        "harmfulness_unsafe_projection": (unsafe_reps @ proj_harmfulness_dir).mean().item(),
        "harmfulness_safe_projection": (safe_refusal_reps @ proj_harmfulness_dir).mean().item(),
        "harmfulness_benign_projection": (benign_reps @ proj_harmfulness_dir).mean().item(),
        "harmfulness_separation": (
            (safe_refusal_reps @ proj_harmfulness_dir).mean() - (benign_reps @ proj_harmfulness_dir).mean()
        ).item(),

        # Direction similarity (this model's own directions)
        "direction_similarity": (jailbreak_direction @ harmfulness_direction).item(),
    }

    # Return this model's own directions (for saving)
    directions = {
        "jailbreak": jailbreak_direction,
        "harmfulness": harmfulness_direction,
    }

    return directions, stats


def analyze_models():
    """Main analysis pipeline - compare base and SearchR1 models.

    Option A: Extract IT directions first, then project both models onto IT's directions.
    """
    print("="*80)
    print("Harmful Search Direction Extraction: IT vs SearchR1")
    print("="*80)
    print("Using Option A: All projections onto IT's directions")

    # Load instruction datasets
    print("\nLoading instruction datasets...")
    harmful_instructions, harmless_instructions = load_instructions()

    # Models to analyze (IT first, then SearchR1)
    models_to_analyze = [
        ("base", BASE_MODEL),
        ("searchr1", SEARCHR1_MODEL),
    ]

    print(f"\nAnalyzing {len(models_to_analyze)} models:")
    for name, path in models_to_analyze:
        print(f"  - {name}: {path}")

    # Results storage
    results = {
        "models": [],
        "directions": {},  # model -> layer -> direction
        "reps": {},  # model -> layer -> representations (mean only to save space)
        "stats": {},  # model -> layer -> stats
    }

    # Storage for IT's directions (reference for Option A)
    it_directions = {}  # layer -> {jailbreak, harmfulness}

    # Analyze each model
    for model_name, model_path in models_to_analyze:
        print(f"\n{'='*80}")
        print(f"Analyzing: {model_name}")
        print(f"{'='*80}")

        # Load model
        try:
            model, tokenizer = load_model_and_tokenizer(model_path)
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            continue

        results["models"].append(model_name)
        results["directions"][model_name] = {}
        results["reps"][model_name] = {}
        results["stats"][model_name] = {}

        # Extract for each layer
        for layer_idx in LAYER_INDICES:
            # Step 1: Extract representations
            reps = extract_reps_for_checkpoint(
                model, tokenizer, layer_idx, harmful_instructions, harmless_instructions
            )

            # Step 2: Compute directions and stats
            # For IT (base): compute directions, use own directions for projection
            # For SearchR1: compute directions, but use IT's directions for projection
            if model_name == "base":
                directions, stats = compute_directions_and_stats(reps, layer_idx, reference_directions=None)
                # Store IT's directions for SearchR1 to use
                it_directions[layer_idx] = {
                    "jailbreak": directions["jailbreak"],
                    "harmfulness": directions["harmfulness"],
                }
            else:
                # Use IT's directions for projection (Option A)
                directions, stats = compute_directions_and_stats(reps, layer_idx, reference_directions=it_directions[layer_idx])

            # Store directions (convert bfloat16 to float32 first)
            results["directions"][model_name][f"layer_{layer_idx}"] = {
                "jailbreak": directions["jailbreak"].float().numpy(),
                "harmfulness": directions["harmfulness"].float().numpy(),
            }

            # Store mean representations (for potential reanalysis)
            results["reps"][model_name][f"layer_{layer_idx}"] = {
                "unsafe_mean": reps["unsafe"].mean(dim=0).float().numpy(),
                "safe_refusal_mean": reps["safe_refusal"].mean(dim=0).float().numpy(),
                "benign_mean": reps["benign"].mean(dim=0).float().numpy(),
            }

            results["stats"][model_name][f"layer_{layer_idx}"] = stats

            print(f"\n  Layer {layer_idx} Statistics (projected onto IT's directions):")
            print(f"    Jailbreak separation: {stats['jailbreak_separation']:.4f}")
            print(f"    Harmfulness separation: {stats['harmfulness_separation']:.4f}")
            print(f"    Direction similarity (own): {stats['direction_similarity']:.4f}")
            print(f"    Jailbreak - Unsafe proj: {stats['jailbreak_unsafe_projection']:.4f}")
            print(f"    Harmfulness - Unsafe proj: {stats['harmfulness_unsafe_projection']:.4f}")

        # Clean up GPU memory
        del model
        torch.cuda.empty_cache()

    # Save results
    print(f"\n{'='*80}")
    print("Saving results...")

    # Save directions (as numpy arrays)
    directions_to_save = {}
    for model, layers in results["directions"].items():
        directions_to_save[model] = {}
        for layer, direction_dict in layers.items():
            directions_to_save[model][layer] = {
                "jailbreak": direction_dict["jailbreak"].tolist(),
                "harmfulness": direction_dict["harmfulness"].tolist(),
            }

    with open(OUTPUT_DIR / "directions_it_vs_searchr1.json", "w") as f:
        json.dump(directions_to_save, f, indent=2)

    # Save mean representations
    reps_to_save = {}
    for model, layers in results["reps"].items():
        reps_to_save[model] = {}
        for layer, rep_dict in layers.items():
            reps_to_save[model][layer] = {
                "unsafe_mean": rep_dict["unsafe_mean"].tolist(),
                "safe_refusal_mean": rep_dict["safe_refusal_mean"].tolist(),
                "benign_mean": rep_dict["benign_mean"].tolist(),
            }

    with open(OUTPUT_DIR / "reps_it_vs_searchr1.json", "w") as f:
        json.dump(reps_to_save, f, indent=2)

    # Save statistics
    with open(OUTPUT_DIR / "stats_it_vs_searchr1.json", "w") as f:
        json.dump(results["stats"], f, indent=2)

    print(f"Results saved to {OUTPUT_DIR}")
    print(f"  - directions_it_vs_searchr1.json (both IT and SearchR1 directions)")
    print(f"  - reps_it_vs_searchr1.json (mean representations for both models)")
    print(f"  - stats_it_vs_searchr1.json (statistics, projected onto IT's directions)")

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_comparison(results)

    return results


def plot_comparison(results):
    """Plot comparison between IT and SearchR1.

    All projections are onto IT's directions (Option A).
    """
    models = results["models"]

    # Plot 1: Jailbreak separation scores comparison across layers
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(LAYER_INDICES))
    width = 0.35

    base_scores = [results["stats"]["base"][f"layer_{layer}"]["jailbreak_separation"] for layer in LAYER_INDICES]
    searchr1_scores = [results["stats"]["searchr1"][f"layer_{layer}"]["jailbreak_separation"] for layer in LAYER_INDICES]

    ax.bar(x - width/2, base_scores, width, label='IT', alpha=0.8, color='#3498db')
    ax.bar(x + width/2, searchr1_scores, width, label='SearchR1', alpha=0.8, color='#e74c3c')

    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Jailbreak Separation (on IT direction)', fontsize=12)
    ax.set_title('Jailbreak Separation: IT vs SearchR1 (projected onto IT direction)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Layer {l}' for l in LAYER_INDICES])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "jailbreak_separation_comparison.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / 'jailbreak_separation_comparison.png'}")
    plt.close()

    # Plot 2: Harmfulness separation scores comparison across layers
    fig, ax = plt.subplots(figsize=(12, 6))

    base_scores = [results["stats"]["base"][f"layer_{layer}"]["harmfulness_separation"] for layer in LAYER_INDICES]
    searchr1_scores = [results["stats"]["searchr1"][f"layer_{layer}"]["harmfulness_separation"] for layer in LAYER_INDICES]

    ax.bar(x - width/2, base_scores, width, label='IT', alpha=0.8, color='#9b59b6')
    ax.bar(x + width/2, searchr1_scores, width, label='SearchR1', alpha=0.8, color='#e67e22')

    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Harmfulness Separation (on IT direction)', fontsize=12)
    ax.set_title('Harmfulness Separation: IT vs SearchR1 (projected onto IT direction)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Layer {l}' for l in LAYER_INDICES])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "harmfulness_separation_comparison.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / 'harmfulness_separation_comparison.png'}")
    plt.close()

    # Plot 3: Projection values for jailbreak direction for each layer
    n_layers = len(LAYER_INDICES)
    n_cols = 3
    n_rows = (n_layers + n_cols - 1) // n_cols  # Ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    for idx, layer_idx in enumerate(LAYER_INDICES):
        ax = axes[idx]

        categories = ['Unsafe', 'Safe Refusal', 'Benign']
        base_values = [
            results["stats"]["base"][f"layer_{layer_idx}"]["jailbreak_unsafe_projection"],
            results["stats"]["base"][f"layer_{layer_idx}"]["jailbreak_safe_projection"],
            results["stats"]["base"][f"layer_{layer_idx}"]["jailbreak_benign_projection"],
        ]
        searchr1_values = [
            results["stats"]["searchr1"][f"layer_{layer_idx}"]["jailbreak_unsafe_projection"],
            results["stats"]["searchr1"][f"layer_{layer_idx}"]["jailbreak_safe_projection"],
            results["stats"]["searchr1"][f"layer_{layer_idx}"]["jailbreak_benign_projection"],
        ]

        x = np.arange(len(categories))
        width = 0.35

        ax.bar(x - width/2, base_values, width, label='IT', alpha=0.8, color='#3498db')
        ax.bar(x + width/2, searchr1_values, width, label='SearchR1', alpha=0.8, color='#e74c3c')

        ax.set_ylabel('Projection (on IT direction)', fontsize=10)
        ax.set_title(f'Layer {layer_idx}: Jailbreak Projections (IT dir)', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    # Hide unused subplots
    for idx in range(n_layers, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "jailbreak_projection_comparison.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / 'jailbreak_projection_comparison.png'}")
    plt.close()

    # Plot 4: Projection values for harmfulness direction for each layer
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten()

    for idx, layer_idx in enumerate(LAYER_INDICES):
        ax = axes[idx]

        categories = ['Unsafe', 'Safe Refusal', 'Benign']
        base_values = [
            results["stats"]["base"][f"layer_{layer_idx}"]["harmfulness_unsafe_projection"],
            results["stats"]["base"][f"layer_{layer_idx}"]["harmfulness_safe_projection"],
            results["stats"]["base"][f"layer_{layer_idx}"]["harmfulness_benign_projection"],
        ]
        searchr1_values = [
            results["stats"]["searchr1"][f"layer_{layer_idx}"]["harmfulness_unsafe_projection"],
            results["stats"]["searchr1"][f"layer_{layer_idx}"]["harmfulness_safe_projection"],
            results["stats"]["searchr1"][f"layer_{layer_idx}"]["harmfulness_benign_projection"],
        ]

        x = np.arange(len(categories))
        width = 0.35

        ax.bar(x - width/2, base_values, width, label='IT', alpha=0.8, color='#9b59b6')
        ax.bar(x + width/2, searchr1_values, width, label='SearchR1', alpha=0.8, color='#e67e22')

        ax.set_ylabel('Projection (on IT direction)', fontsize=10)
        ax.set_title(f'Layer {layer_idx}: Harmfulness Projections (IT dir)', fontsize=11, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')

    # Hide unused subplots
    for idx in range(n_layers, len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "harmfulness_projection_comparison.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / 'harmfulness_projection_comparison.png'}")
    plt.close()

    # Plot 5: Direction similarity between jailbreak and harmfulness (within each model's own directions)
    fig, ax = plt.subplots(figsize=(10, 6))

    base_sim = [results["stats"]["base"][f"layer_{layer}"]["direction_similarity"] for layer in LAYER_INDICES]
    searchr1_sim = [results["stats"]["searchr1"][f"layer_{layer}"]["direction_similarity"] for layer in LAYER_INDICES]

    ax.plot(LAYER_INDICES, base_sim, marker='o', linewidth=2, markersize=8, label='IT', color='#3498db')
    ax.plot(LAYER_INDICES, searchr1_sim, marker='s', linewidth=2, markersize=8, label='SearchR1', color='#e74c3c')

    ax.set_xlabel('Layer Index', fontsize=12)
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title('Direction Similarity (Jailbreak vs Harmfulness) - Each Model\'s Own Directions', fontsize=14, fontweight='bold')
    ax.set_ylim([-1, 1])  # Allow negative values (orthogonal/opposite directions)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)  # Add zero line
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "direction_similarity_within_model.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {OUTPUT_DIR / 'direction_similarity_within_model.png'}")
    plt.close()

    plt.close('all')


if __name__ == "__main__":
    results = analyze_models()

    print("\n" + "="*80)
    print("Analysis Complete!")
    print("="*80)
    print("\nGenerated files:")
    print("1. jailbreak_separation_comparison.png - Jailbreak separation (on IT direction)")
    print("2. harmfulness_separation_comparison.png - Harmfulness separation (on IT direction)")
    print("3. jailbreak_projection_comparison.png - Jailbreak projections for unsafe/safe/benign")
    print("4. harmfulness_projection_comparison.png - Harmfulness projections for unsafe/safe/benign")
    print("5. direction_similarity_within_model.png - Each model's own direction similarity")
    print("6. directions_it_vs_searchr1.json - Each model's own directions")
    print("7. reps_it_vs_searchr1.json - Mean representations for both models")
    print("8. stats_it_vs_searchr1.json - All statistics (projected onto IT's directions)")
    print("\nInterpretation:")
    print("- All projections are onto IT's directions (Option A) for fair comparison")
    print("- Separation score: Higher means better distinction between categories")
    print("- Direction similarity: Each model's jailbreak vs harmfulness direction correlation")
