import matplotlib.pyplot as plt
import numpy as np

# Data (0-100 scale, first query only)
models = ['Llama-3.2-3B', 'Qwen2.5-7B', 'Qwen2.5-14B', 'Qwen2.5-32B']
it_scores = [25.8, 38.7, 47.6, 58.5]
ppo_scores = [12.3, 23.7, 17.4, 31.5]
grpo_scores = [10.2, 8.8, 5.6, 22.5]

x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(11, 5))

# Bars
bars_it = ax.bar(x - width, it_scores, width, label='IT (Instruction-Tuned)', color='#4C9BE8', edgecolor='black', linewidth=0.8)
bars_ppo = ax.bar(x, ppo_scores, width, label='RL (PPO)', color='#E8694C', edgecolor='black', linewidth=0.8)

# GRPO bars
bars_grpo = ax.bar(x + width, grpo_scores, width, label='RL (GRPO)', color='#8B5CF6', edgecolor='black', linewidth=0.8)

# Value labels on IT bars
for bar in bars_it:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Value labels on PPO bars
for bar in bars_ppo:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Value labels on GRPO bars
for bar in bars_grpo:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')



ax.set_ylabel('Search Safety Score', fontsize=15)
ax.tick_params(axis='y', labelsize=13)
ax.set_title('Search Safety Under Prefill Attack: IT vs RL Models', fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=14)
ax.set_ylim(0, 72)
ax.legend(fontsize=13, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/VData/kebl6672/ARL/figures/search_safety_it_vs_rl.pdf', bbox_inches='tight', dpi=300)
plt.savefig('/VData/kebl6672/ARL/figures/search_safety_it_vs_rl.png', bbox_inches='tight', dpi=300)
print("Saved to figures/search_safety_it_vs_rl.pdf and .png")
