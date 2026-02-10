"""
Steering experiment: Test jailbreak and harmfulness directions on Qwen-7B IT model.

Uses IT model's own directions (jailbreak and harmfulness) to steer itself.
Collects full responses for manual analysis.

This follows Arditi et al. "Refusal in LLMs is mediated by a single direction" methodology.
"""

import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List
from tqdm import tqdm
import random

# Configuration
MODEL_PATH = "Qwen/Qwen2.5-7B-Instruct"
DIRECTIONS_FILE = Path("/VData/kebl6672/ARL/direction_analysis_7b/directions_it_vs_searchr1.json")
OUTPUT_DIR = Path("/VData/kebl6672/ARL/steering_results_7b")
OUTPUT_DIR.mkdir(exist_ok=True)

# Harmful prompts from repo dataset
HARMFUL_PROMPTS_FILE = Path("/VData/kebl6672/ARL/refusal_datasets/harmful_combined.json")
NUM_TEST_PROMPTS = 100  # Sample size for testing

# Layers to test (middle to late layers where refusal is typically mediated)
STEER_LAYERS = [18, 22, 27]  # 7B model has layers 0-27

# Steering coefficients to test
STEERING_COEFFS = [-2.0, 0.0, 2.0]

# Direction types to test
DIRECTION_TYPES = ["jailbreak", "harmfulness"]

# Random seed
RANDOM_SEED = 42


def load_harmful_prompts() -> List[str]:
    """Load harmful prompts from repo dataset."""
    print(f"Loading harmful prompts from {HARMFUL_PROMPTS_FILE}")
    with open(HARMFUL_PROMPTS_FILE, "r") as f:
        data = json.load(f)

    all_prompts = [item["instruction"] for item in data]
    print(f"  Total prompts available: {len(all_prompts)}")

    # Sample if needed
    random.seed(RANDOM_SEED)
    if len(all_prompts) > NUM_TEST_PROMPTS:
        prompts = random.sample(all_prompts, NUM_TEST_PROMPTS)
        print(f"  Sampled {NUM_TEST_PROMPTS} prompts for testing")
    else:
        prompts = all_prompts
        print(f"  Using all {len(prompts)} prompts")

    return prompts


class JailbreakSteerer:
    """
    Applies activation steering using the jailbreak direction.

    Steering is applied during generation at every forward pass,
    following Arditi et al.'s methodology.
    """

    def __init__(
        self,
        model,
        tokenizer,
        direction: torch.Tensor,
        layer_idx: int,
        coeff: float = 1.0,
        steer_all_positions: bool = True,
    ):
        """
        Args:
            model: The language model
            tokenizer: The tokenizer
            direction: [hidden_dim] jailbreak direction vector (normalized)
            layer_idx: Which layer to apply steering
            coeff: Steering coefficient (positive = more jailbreak-like)
            steer_all_positions: If True, steer all positions. If False, only last.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.direction = direction.to(model.device).to(model.dtype)
        self.layer_idx = layer_idx
        self.coeff = coeff
        self.steer_all_positions = steer_all_positions
        self.hook_handle = None

    def _get_steering_hook(self):
        """Create steering hook for forward pass."""
        def hook(module, input, output):
            # output is typically (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden_states = output[0]
                rest = output[1:]
            else:
                hidden_states = output
                rest = None

            # Apply steering to all positions (prompt + each generated token)
            # Shape: [batch, seq_len, hidden_dim] where seq_len=prompt_len for first call, then 1 per token
            if self.steer_all_positions:
                hidden_states = hidden_states + self.coeff * self.direction
            else:
                hidden_states[:, -1, :] = hidden_states[:, -1, :] + self.coeff * self.direction

            if rest is not None:
                return (hidden_states,) + rest
            return hidden_states

        return hook

    def activate(self):
        """Register the steering hook."""
        if self.hook_handle is not None:
            return  # Already active

        layer = self.model.model.layers[self.layer_idx]
        self.hook_handle = layer.register_forward_hook(self._get_steering_hook())

    def deactivate(self):
        """Remove the steering hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

    def __enter__(self):
        self.activate()
        return self

    def __exit__(self, *args):
        self.deactivate()


def load_model_and_tokenizer(model_path: str, device: str = "cuda"):
    """Load model and tokenizer."""
    print(f"Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer


def load_directions() -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Load IT model's directions for steering.

    We use IT's jailbreak direction to steer both IT and SearchR1 models.
    This tests whether the direction learned from IT captures the
    transformation that RL training induces.
    """
    print(f"Loading IT model's directions from {DIRECTIONS_FILE}")
    with open(DIRECTIONS_FILE, "r") as f:
        data = json.load(f)

    # Use IT (base) model's directions for steering both models
    directions = {}
    for layer_key, layer_dirs in data["base"].items():
        directions[layer_key] = {
            "jailbreak": torch.tensor(layer_dirs["jailbreak"], dtype=torch.bfloat16),
            "harmfulness": torch.tensor(layer_dirs["harmfulness"], dtype=torch.bfloat16),
        }

    print(f"  Using IT's jailbreak direction to steer both IT and SearchR1")
    return directions


def format_prompt(instruction: str, tokenizer) -> str:
    """Format instruction with system prompt and chat template."""
    question = instruction.strip()
    if question and question[-1] != '?':
        question += '?'

    system_prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. Question: {question}\n"""

    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": system_prompt}],
            add_generation_prompt=True,
            tokenize=False
        )
    else:
        prompt = system_prompt

    return prompt


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    steerer: Optional[JailbreakSteerer] = None,
) -> str:
    """Generate response, optionally with steering."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Activate steering if provided
    if steerer is not None:
        steerer.activate()

    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy for reproducibility
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode only the generated part
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    finally:
        if steerer is not None:
            steerer.deactivate()

    return response


def run_steering_experiment(
    model,
    tokenizer,
    model_name: str,
    directions: Dict[str, Dict[str, torch.Tensor]],
    prompts: List[str],
) -> Dict:
    """
    Run steering experiment on a model.

    Tests both jailbreak and harmfulness directions across multiple layers and coefficients.
    Stores full responses for manual analysis.
    """
    print(f"\n{'='*80}")
    print(f"Running steering experiment: {model_name}")
    print(f"{'='*80}")

    results = {
        "model_name": model_name,
        "directions": {},  # direction_type -> layer -> coefficients -> responses
    }

    for direction_type in DIRECTION_TYPES:
        print(f"\n  Direction: {direction_type}")
        results["directions"][direction_type] = {"layers": {}}

        for layer_idx in STEER_LAYERS:
            layer_key = f"layer_{layer_idx}"
            if layer_key not in directions:
                print(f"    Skipping layer {layer_idx} (no direction found)")
                continue

            print(f"\n    Layer {layer_idx}:")
            direction = directions[layer_key][direction_type]

            layer_results = {
                "coefficients": {},
            }

            for coeff in STEERING_COEFFS:
                print(f"      Coefficient {coeff:+.1f}...")

                coeff_results = {
                    "responses": [],
                }

                # Create steerer (coeff=0 means no steering)
                steerer = None
                if coeff != 0:
                    steerer = JailbreakSteerer(
                        model, tokenizer, direction, layer_idx, coeff=coeff
                    )

                for prompt_text in tqdm(prompts, desc=f"        Prompts", leave=False):
                    prompt = format_prompt(prompt_text, tokenizer)
                    response = generate_response(model, tokenizer, prompt, steerer=steerer)

                    coeff_results["responses"].append({
                        "prompt": prompt_text,
                        "response": response,
                    })

                print(f"        Done ({len(coeff_results['responses'])} responses)")

                layer_results["coefficients"][str(coeff)] = coeff_results

            results["directions"][direction_type]["layers"][layer_key] = layer_results

    return results


def main():
    """Main experiment pipeline."""
    print("="*80)
    print("Activation Steering Experiment")
    print("Qwen-7B IT model with jailbreak & harmfulness directions")
    print("="*80)

    random.seed(RANDOM_SEED)

    # Load harmful prompts from repo dataset
    harmful_prompts = load_harmful_prompts()

    # Load directions
    directions = load_directions()
    print(f"Loaded directions for layers: {list(directions.keys())}")
    print(f"Direction types: {DIRECTION_TYPES}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)

    # Run experiment
    results = run_steering_experiment(model, tokenizer, "IT", directions, harmful_prompts)

    # Cleanup
    del model
    torch.cuda.empty_cache()

    # Save raw results
    print(f"\n{'='*80}")
    print("Saving results...")

    with open(OUTPUT_DIR / "steering_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Create comparison-friendly format (organized by prompt)
    first_dir = list(results["directions"].keys())[0]
    first_layer = list(results["directions"][first_dir]["layers"].keys())[0]
    n_prompts = len(results["directions"][first_dir]["layers"][first_layer]["coefficients"]["0.0"]["responses"])

    comparison_results = []
    for prompt_idx in range(n_prompts):
        prompt_comparison = {
            "prompt_idx": prompt_idx,
            "prompt": None,
            "comparisons": {}
        }

        for direction_type, dir_results in results["directions"].items():
            prompt_comparison["comparisons"][direction_type] = {}

            for layer_key, layer_results in dir_results["layers"].items():
                layer_comparison = {}

                for coeff, coeff_results in layer_results["coefficients"].items():
                    response_data = coeff_results["responses"][prompt_idx]
                    if prompt_comparison["prompt"] is None:
                        prompt_comparison["prompt"] = response_data["prompt"]
                    layer_comparison[f"coeff_{coeff}"] = response_data["response"]

                prompt_comparison["comparisons"][direction_type][layer_key] = layer_comparison

        comparison_results.append(prompt_comparison)

    with open(OUTPUT_DIR / "steering_comparison.json", "w") as f:
        json.dump(comparison_results, f, indent=2)

    # Save as readable text file
    with open(OUTPUT_DIR / "steering_comparison.txt", "w") as f:
        for item in comparison_results:
            f.write("="*100 + "\n")
            f.write(f"PROMPT {item['prompt_idx']}: {item['prompt']}\n")
            f.write("="*100 + "\n\n")

            for direction_type, layers in item["comparisons"].items():
                for layer_key, coeffs in layers.items():
                    f.write(f"--- {direction_type.upper()} direction | {layer_key} ---\n\n")

                    for coeff_key in sorted(coeffs.keys(), key=lambda x: float(x.split("_")[1])):
                        coeff_val = coeff_key.split("_")[1]
                        f.write(f"[coeff {float(coeff_val):+.1f}]\n")
                        f.write(f"{coeffs[coeff_key]}\n\n")

                    f.write("\n")
            f.write("\n\n")

    print(f"Results saved to:")
    print(f"  - {OUTPUT_DIR / 'steering_results.json'} (raw)")
    print(f"  - {OUTPUT_DIR / 'steering_comparison.json'} (by prompt)")
    print(f"  - {OUTPUT_DIR / 'steering_comparison.txt'} (readable)")

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print("="*80)
    for direction_type, dir_results in results["directions"].items():
        print(f"\n{direction_type} direction:")
        for layer_key, layer_results in dir_results["layers"].items():
            print(f"  {layer_key}:")
            for coeff, coeff_results in layer_results["coefficients"].items():
                print(f"    coeff {float(coeff):+.1f}: {len(coeff_results['responses'])} responses")


if __name__ == "__main__":
    main()
