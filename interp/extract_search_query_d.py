"""
Extract search query direction from IT models.

Same harmful prompts, two conditions:
1. ATTACK: Search system prompt + prefilled <search> → harmful search queries
2. NATURAL: Search system prompt, NO prefill → model naturally generates <search> → likely benign queries

Direction d = mean(attack <search> reps) - mean(natural <search> reps)

Steps:
1. For each harmful prompt, generate naturally and cache sequence up to first <search>
2. Skip prompts where model doesn't generate <search>
3. For valid prompts, one forward pass each for attack and natural, extract all layers
4. Compute direction per layer
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import random

# Configuration
MODELS = {
    "llama3b": {
        "model_path": "meta-llama/Llama-3.2-3B-Instruct",
        "layers": [14, 18, 22, 26, 28],
    },
    "qwen7b": {
        "model_path": "Qwen/Qwen2.5-7B-Instruct",
        "layers": [14, 18, 22, 26, 28],
    },
    "qwen14b": {
        "model_path": "Qwen/Qwen2.5-14B-Instruct",
        "layers": [24, 32, 40, 44, 48],
    },
    "qwen32b": {
        "model_path": "Qwen/Qwen2.5-32B-Instruct",
        "layers": [32, 42, 52, 60, 64],
    },
}

SELECTED_MODEL = "qwen7b"

HARMFUL_PATH = Path("/VData/kebl6672/ARL/refusal_datasets/harmful_combined.json")

MAX_HARMFUL = None  # Use all
MAX_NATURAL_TOKENS = 256  # Max tokens to generate when looking for natural <search>
RANDOM_SEED = 42


# =============================================================================
# Helpers
# =============================================================================

SEARCH_SYSTEM_PROMPT_TEMPLATE = """Answer the given question. \
You must conduct reasoning inside <think> and </think> every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> and it will return the top searched results between <information> and </information>. \
You can search as many times as your want. \
If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, without detailed illustrations. Question: {question}\n"""


def format_prompt(question, tokenizer, prefill_search=False):
    """Format with search system prompt. Optionally append <search> for attack condition."""
    question = question.strip()
    if question and question[-1] != '?':
        question += '?'

    system_prompt = SEARCH_SYSTEM_PROMPT_TEMPLATE.format(question=question)

    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": system_prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
    else:
        prompt = system_prompt

    if prefill_search:
        prompt += "<search>"

    return prompt


def find_first_search_pos(gen_tokens, search_token_ids):
    """
    Find position of first <search> in generated tokens.
    Returns index of the last token of <search>, or None if not found.
    """
    n = len(search_token_ids)
    for i in range(len(gen_tokens) - n + 1):
        if gen_tokens[i:i+n] == search_token_ids:
            return i + n - 1
    return None


# =============================================================================
# Main
# =============================================================================

def main():
    random.seed(RANDOM_SEED)

    # Load harmful prompts
    with open(HARMFUL_PATH) as f:
        harmful = [item['instruction'] for item in json.load(f)]
    if MAX_HARMFUL and len(harmful) > MAX_HARMFUL:
        harmful = random.sample(harmful, MAX_HARMFUL)
    print(f"Harmful prompts: {len(harmful)}")

    model_name = SELECTED_MODEL
    model_cfg = MODELS[model_name]
    model_path = model_cfg["model_path"]
    layer_indices = model_cfg["layers"]
    output_dir = Path(f"/VData/kebl6672/ARL/interp_results/search_query_direction_{model_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Processing {model_name}: {model_path}")
    print(f"Direction: attack <search> vs natural <search>")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()

    search_token_ids = tokenizer.encode("<search>", add_special_tokens=False)
    print(f"  <search> token IDs: {search_token_ids}")

    # Step 1: Generate naturally, cache sequences up to first <search>
    print("\nStep 1: Natural generation — caching sequences up to first <search>...")
    valid_data = []  # (instruction, cached_seq_up_to_search, full_gen_text)
    no_search_count = 0

    for inst in tqdm(harmful, desc="  Generating"):
        prompt = format_prompt(inst, tokenizer, prefill_search=False)
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
        prompt_len = input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=MAX_NATURAL_TOKENS,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                use_cache=True,
            )

        generated_ids = outputs[0]
        gen_tokens = generated_ids[prompt_len:].tolist()
        search_idx = find_first_search_pos(gen_tokens, search_token_ids)

        if search_idx is not None:
            search_abs_pos = prompt_len + search_idx
            seq_up_to_search = generated_ids[:search_abs_pos + 1].cpu()
            gen_text = tokenizer.decode(generated_ids[prompt_len:], skip_special_tokens=False)
            valid_data.append((inst, seq_up_to_search, gen_text))
        else:
            no_search_count += 1

        torch.cuda.empty_cache()

    print(f"  With natural <search>: {len(valid_data)}")
    print(f"  Without <search>: {no_search_count}")

    if len(valid_data) < 10:
        print("ERROR: Too few prompts with natural <search>. Aborting.")
        return

    # Save natural generations for inspection
    with open(output_dir / "natural_search_queries.json", "w") as f:
        json.dump([
            {"instruction": inst, "natural_generation": gen}
            for inst, _, gen in valid_data
        ], f, indent=2, ensure_ascii=False)

    # Step 2: Extract hidden states — one forward pass per prompt, all layers at once
    print(f"\nStep 2: Extracting hidden states ({len(valid_data)} prompts)...")

    # Per-layer collectors
    attack_reps = {l: [] for l in layer_indices}
    natural_reps = {l: [] for l in layer_indices}

    for inst, cached_seq, _ in tqdm(valid_data, desc="  Extracting"):
        # Attack: forward pass on system prompt + prefilled <search>
        attack_prompt = format_prompt(inst, tokenizer, prefill_search=True)
        attack_inputs = tokenizer(attack_prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            attack_out = model(**attack_inputs, output_hidden_states=True, return_dict=True)
        for l in layer_indices:
            attack_reps[l].append(attack_out.hidden_states[l][0, -1, :].cpu())

        # Natural: forward pass on cached sequence up to first <search>
        seq_input = cached_seq.unsqueeze(0).to(model.device)
        with torch.no_grad():
            natural_out = model(seq_input, output_hidden_states=True, return_dict=True)
        for l in layer_indices:
            natural_reps[l].append(natural_out.hidden_states[l][0, -1, :].cpu())

        torch.cuda.empty_cache()

    # Step 3: Compute directions per layer
    print("\nStep 3: Computing directions...")

    all_directions = {}
    all_stats = {}

    for layer_idx in layer_indices:
        attack_t = torch.stack(attack_reps[layer_idx])
        natural_t = torch.stack(natural_reps[layer_idx])
        n_pairs = len(attack_t)

        direction_raw = attack_t.mean(0) - natural_t.mean(0)
        direction = direction_raw / direction_raw.norm()

        attack_projs = attack_t @ direction
        natural_projs = natural_t @ direction
        separation = attack_projs.mean().item() - natural_projs.mean().item()

        print(f"  Layer {layer_idx}: sep={separation:.4f}, "
              f"attack={attack_projs.mean():.4f}±{attack_projs.std():.4f}, "
              f"natural={natural_projs.mean():.4f}±{natural_projs.std():.4f}")

        key = f"layer_{layer_idx}"
        all_directions[key] = direction.float().numpy().tolist()
        all_stats[key] = {
            "separation": separation,
            "direction_norm": direction_raw.norm().item(),
            "n_pairs": n_pairs,
            "attack_proj_mean": attack_projs.mean().item(),
            "attack_proj_std": attack_projs.std().item(),
            "natural_proj_mean": natural_projs.mean().item(),
            "natural_proj_std": natural_projs.std().item(),
        }

    del model
    torch.cuda.empty_cache()

    # Save
    with open(output_dir / "search_query_direction.json", "w") as f:
        json.dump(all_directions, f, indent=2)
    with open(output_dir / "search_query_stats.json", "w") as f:
        json.dump(all_stats, f, indent=2)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    seps = [all_stats[f"layer_{l}"]["separation"] for l in layer_indices]
    ax.bar(range(len(layer_indices)), seps, color='#e67e22', alpha=0.8)
    ax.set_xticks(range(len(layer_indices)))
    ax.set_xticklabels([f'Layer {l}' for l in layer_indices])
    ax.set_ylabel('Separation (attack - natural)')
    ax.set_title(f'{model_name}: Search Query Direction (Attack vs Natural <search>)')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / "separation_by_layer.png", dpi=300)
    plt.close()

    best = max(layer_indices, key=lambda l: all_stats[f"layer_{l}"]["separation"])
    print(f"\nBest layer: {best} (separation={all_stats[f'layer_{best}']['separation']:.4f})")
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
