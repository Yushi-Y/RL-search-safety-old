import matplotlib.pyplot as plt
import numpy as np

# Data (0-100 scale, first query only)
models = ['Llama-3.2-3B', 'Qwen2.5-7B', 'Qwen2.5-14B']
it_scores = [25.8, 38.7, 47.6]
ppo_scores = [12.3, 23.7, 17.4]
grpo_scores = [None, None, 5.6]  # Only Qwen14B has GRPO results

x = np.arange(len(models))
width = 0.25

fig, ax = plt.subplots(figsize=(9, 5))

# Bars
bars_it = ax.bar(x - width, it_scores, width, label='IT (Instruction-Tuned)', color='#4C9BE8', edgecolor='black', linewidth=0.8)
bars_ppo = ax.bar(x, ppo_scores, width, label='RL (PPO)', color='#E8694C', edgecolor='black', linewidth=0.8)

# GRPO bars (only where available)
grpo_x = []
grpo_y = []
for i, score in enumerate(grpo_scores):
    if score is not None:
        grpo_x.append(x[i] + width)
        grpo_y.append(score)
bars_grpo = ax.bar(grpo_x, grpo_y, width, label='RL (GRPO)', color='#8B5CF6', edgecolor='black', linewidth=0.8)

# Value labels on IT bars
for bar in bars_it:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Value labels on PPO bars
for bar in bars_ppo:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Value labels on GRPO bars
for bar in bars_grpo:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
            f'{height:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Drop annotations (IT → PPO)
for i in range(len(models)):
    drop_ppo = it_scores[i] - ppo_scores[i]
    mid_y = (it_scores[i] + ppo_scores[i]) / 2
    ax.text(x[i] + 0.02, mid_y, f'−{drop_ppo:.1f}',
            ha='center', va='center', fontsize=10, fontweight='bold', color='#CC3333')

# Drop annotation for GRPO (Qwen14B only)
drop_grpo = it_scores[2] - grpo_scores[2]
mid_y = (it_scores[2] + grpo_scores[2]) / 2
ax.text(x[2] + width + 0.02, mid_y, f'−{drop_grpo:.1f}',
        ha='center', va='center', fontsize=10, fontweight='bold', color='#7C3AED')

ax.set_ylabel('Search Safety Score (0–100)', fontsize=13)
ax.set_title('Search Safety Under Prefill Attack: IT vs RL Models', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12)
ax.set_ylim(0, 62)
ax.legend(fontsize=11, loc='upper left')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('/VData/kebl6672/ARL/figures/search_safety_it_vs_rl.pdf', bbox_inches='tight', dpi=300)
plt.savefig('/VData/kebl6672/ARL/figures/search_safety_it_vs_rl.png', bbox_inches='tight', dpi=300)
print("Saved to figures/search_safety_it_vs_rl.pdf and .png")
