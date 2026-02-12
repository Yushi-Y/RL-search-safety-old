"""
Steering verification: Replicate Arditi refusal bypass.

Approach (based on andyrdt/refusal_direction):
1. Load refusal direction extracted from IT model (harmful vs benign, NO <search>)
2. During generation, hook into target layer
3. Add coeff * d to hidden states at EVERY forward pass
4. Simple generation (NO search attack, NO retrieval)
5. Negative coeff → bypass refusal, positive → increase refusal

Tests on IT model WITHOUT search attack - basic refusal bypass test.
"""

import transformers
import torch
import json
import re
import time
import gc
import requests
import random
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Optional

# =============================================================================
# Configuration

# =============================================================================

IT_MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"

DIRECTION_FILE = Path("/VData/kebl6672/ARL/interp_results/refusal_direction_qwen7b/refusal_direction.json")
OUTPUT_DIR = Path("/VData/kebl6672/ARL/interp_results/refusal_bypass_qwen7b")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HARMFUL_PROMPTS_FILE = Path("/VData/kebl6672/ARL/refusal_datasets/harmful_combined.json")
NUM_TEST_PROMPTS = 10  # Small sample for quick debugging

STEER_LAYERS = [27]  # Last layer only for faster testing
STEERING_COEFFS = [-2.0, -1.0, 0.0, 1.0, 2.0]  # Coefficients for raw direction (raw norm ~144)
RANDOM_SEED = 42


# =============================================================================
# Steerer — hooks into layers, applies steering during all forward passes
# =============================================================================

class SearchTokenSteerer:
    """
    Steers hidden state by adding direction vector to activations.

    Applied during ALL forward passes (prefill and autoregressive generation).
    Based on IBM activation-steering and andyrdt/refusal_direction approach.
    """

    def __init__(self, model, direction: torch.Tensor, layer_idx: int, coeff: float):
        self.model = model
        self.direction = direction.to(model.device).to(model.dtype)
        self.layer_idx = layer_idx
        self.coeff = coeff
        self.hook_handle = None
        self.call_count = 0

    def _hook(self, module, input, output):
        # Debug logging (first 3 calls only)
        self.call_count += 1

        if isinstance(output, tuple):
            hidden_states = output[0]
            # Create modified version by adding steering to ALL sequence positions
            modified = hidden_states + self.coeff * self.direction

            if self.call_count <= 3:
                orig_norm = hidden_states[0].norm().item()
                steer_norm = (self.coeff * self.direction).norm().item()
                modified_norm = modified[0].norm().item()
                steer_pct = (steer_norm / orig_norm * 100) if orig_norm > 0 else 0
                print(f"      [DEBUG] Call {self.call_count}, seq_len={hidden_states.shape[1]}, orig_norm={orig_norm:.1f}, steer_norm={steer_norm:.1f} ({steer_pct:.1f}%), modified_norm={modified_norm:.1f}")

            return (modified,) + output[1:]
        else:
            # Single tensor output
            modified = output + self.coeff * self.direction

            if self.call_count <= 3:
                orig_norm = output[0].norm().item()
                steer_norm = (self.coeff * self.direction).norm().item()
                modified_norm = modified[0].norm().item()
                print(f"      [DEBUG] Call {self.call_count}, seq_len={output.shape[1]}, orig_norm={orig_norm:.2f}, steer_norm={steer_norm:.2f}, modified_norm={modified_norm:.2f}")

            return modified

    def activate(self):
        if self.hook_handle is None:
            self.hook_handle = self.model.model.layers[self.layer_idx].register_forward_hook(self._hook)

    def deactivate(self):
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None


# =============================================================================
# Simple generation function
# =============================================================================

def process_single_question(question_text, model, tokenizer, steerer=None):
    """
    Simple generation with optional steering (NO search attack).
    Tests basic refusal bypass like Arditi et al.
    """
    # Simple prompt without search instructions
    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": question_text}],
            add_generation_prompt=True,
            tokenize=False
        )
    else:
        prompt = question_text

    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
    attention_mask = torch.ones_like(input_ids)

    # Activate steering hook
    if steerer is not None:
        steerer.activate()

    try:
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=512,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            use_cache=True
        )
    finally:
        if steerer is not None:
            steerer.deactivate()

    generated_tokens = outputs[0][input_ids.shape[1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    torch.cuda.empty_cache()
    gc.collect()

    return response


# =============================================================================
# Main
# =============================================================================

def main():
    random.seed(RANDOM_SEED)

    # Load prompts
    with open(HARMFUL_PROMPTS_FILE) as f:
        all_prompts = [item["instruction"] for item in json.load(f)]
    prompts = random.sample(all_prompts, min(NUM_TEST_PROMPTS, len(all_prompts)))
    print(f"Test prompts: {len(prompts)}")

    # Load directions (normalized) and stats (contains raw norms)
    with open(DIRECTION_FILE) as f:
        dir_data = json.load(f)
    stats_file = DIRECTION_FILE.parent / "refusal_stats.json"
    with open(stats_file) as f:
        stats_data = json.load(f)

    # Convert to raw (unnormalized) directions by multiplying by original norm
    # This preserves the natural magnitude relative to model activations
    directions = {}
    for k, v in dir_data.items():
        raw_norm = stats_data[k]["direction_norm"]
        directions[k] = torch.tensor(v, dtype=torch.bfloat16) * raw_norm
    print(f"Directions (using raw magnitude): {list(directions.keys())}")
    print(f"  e.g., layer_28 raw norm: {stats_data['layer_28']['direction_norm']:.1f}")

    for model_name, model_path in [("IT", IT_MODEL_PATH)]:
        print(f"\n{'='*80}\nModel: {model_name}\n{'='*80}")
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto",
            trust_remote_code=True, attn_implementation="eager",
        )
        model.eval()

        results = {}
        for layer_idx in STEER_LAYERS:
            # Use layer_28 direction for steering at layer 27
            direction_key = f"layer_{layer_idx + 1}" if layer_idx == 27 else f"layer_{layer_idx}"
            if direction_key not in directions:
                print(f"  Warning: {direction_key} not found in directions, skipping...")
                continue
            direction = directions[direction_key]
            print(f"\n  Layer {layer_idx} (using direction from {direction_key}):")

            layer_key = f"layer_{layer_idx}"
            results[layer_key] = {}
            for coeff in STEERING_COEFFS:
                print(f"    Coeff {coeff:+.1f}...")

                steerer = SearchTokenSteerer(model, direction, layer_idx, coeff) if coeff != 0 else None

                coeff_results = []
                for i, p in enumerate(tqdm(prompts, desc="      ", leave=False)):
                    response = process_single_question(p, model, tokenizer, steerer)
                    coeff_results.append({"question": p, "response": response})

                    # Save every 10 questions
                    if (i + 1) % 10 == 0 or (i + 1) == len(prompts):
                        results[layer_key][str(coeff)] = coeff_results
                        with open(OUTPUT_DIR / f"steering_results_{model_name}.json", "w") as f:
                            json.dump(results, f, indent=2, ensure_ascii=False)

                results[layer_key][str(coeff)] = coeff_results

        with open(OUTPUT_DIR / f"steering_results_{model_name}.json", "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        del model
        torch.cuda.empty_cache()
        print(f"  Saved {model_name} results")

    print(f"\nAll results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
