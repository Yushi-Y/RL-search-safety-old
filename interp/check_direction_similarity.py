"""
Quick check: How similar are refusal direction vs search direction?
Compare across all models.

Compares:
- Refusal direction (simple prompts, no <search>)
- Search clean direction (simple prompts + <search>, NO system prompt)
- [Optional] Search harmfulness direction (with system prompt - old version)
"""
import json
import numpy as np
from pathlib import Path

MODELS = {
    "llama3b": {
        "layers": [14, 18, 22, 26, 28],
    },
    "qwen7b": {
        "layers": [14, 18, 22, 26, 28],
    },
    "qwen14b": {
        "layers": [24, 32, 40, 44, 48],
    },
    "qwen32b": {
        "layers": [32, 42, 52, 60, 64],
    },
}

# Choose which search direction to compare
SEARCH_DIR_TYPE = "clean"  # Options: "clean" (new, no system prompt) or "harmfulness" (old, with system prompt)

results_dir = Path("/VData/kebl6672/ARL/interp_results")

print(f"Direction Similarity Analysis: Refusal vs Search {SEARCH_DIR_TYPE.title()}")
print("=" * 80)

for model_name, model_cfg in MODELS.items():
    refusal_file = results_dir / f"refusal_direction_{model_name}" / "refusal_direction.json"

    # Choose search direction file based on type
    if SEARCH_DIR_TYPE == "clean":
        # New version: search_direction_clean or search_safety_direction (with search_clean_direction.json)
        search_file = results_dir / f"search_direction_clean_{model_name}" / "search_direction_clean.json"
        # Fallback to alternate naming
        if not search_file.exists():
            search_file = results_dir / f"search_safety_direction_{model_name}" / "search_clean_direction.json"
    else:
        # Old version: search_harmfulness_direction
        search_file = results_dir / f"search_harmfulness_direction_{model_name}" / "search_harmfulness_direction.json"

    if not refusal_file.exists():
        print(f"\n{model_name.upper()}: Refusal file missing, skipping...")
        continue

    if not search_file.exists():
        print(f"\n{model_name.upper()}: Search {SEARCH_DIR_TYPE} file missing, skipping...")
        continue

    with open(refusal_file) as f:
        refusal_dirs = json.load(f)
    with open(search_file) as f:
        search_dirs = json.load(f)

    print(f"\n{model_name.upper()}: (Refusal vs Search {SEARCH_DIR_TYPE.title()})")
    print("-" * 80)

    for layer_idx in model_cfg["layers"]:
        layer_key = f"layer_{layer_idx}"
        if layer_key in refusal_dirs and layer_key in search_dirs:
            ref_vec = np.array(refusal_dirs[layer_key])
            search_vec = np.array(search_dirs[layer_key])

            # Cosine similarity
            cos_sim = np.dot(ref_vec, search_vec)

            # L2 distance
            l2_dist = np.linalg.norm(ref_vec - search_vec)

            print(f"  Layer {layer_idx:2d}: cos_sim={cos_sim:+.4f} ({abs(cos_sim)*100:5.1f}%), L2_dist={l2_dist:.4f}")

print("\n" + "=" * 80)
