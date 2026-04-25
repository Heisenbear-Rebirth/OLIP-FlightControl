"""L3 极端实验结果绘图"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = '/mnt/e/study/AeroCat/dualTask/code/logs/l3_extreme'

def load(name):
    with open(f'{BASE}/{name}', 'r') as f:
        return json.load(f)

def smooth(y, w=5):
    if len(y) < w: return y
    return np.convolve(y, np.ones(w)/w, mode='valid')

data = {}
for mode in ['full', 'no_l3diag', 'baseline']:
    d = load(f'l3_extreme_{mode}_metrics.json')
    steps = np.array([m['total_steps'] for m in d]) / 1e6
    rewards = np.array([m['mean_reward'] for m in d])
    final = np.mean(rewards[-5:])
    data[mode] = {'steps': steps, 'rewards': rewards, 'final': final}

print("="*60)
print("  L3 极端验证实验结果")
print("="*60)
for mode in ['full', 'no_l3diag', 'baseline']:
    f = data[mode]['final']
    delta = data['full']['final'] - f
    print(f"  {mode:12s}: {f:.4f}  (vs Full: {delta:+.4f}, {delta/data['full']['final']*100:+.1f}%)")

print(f"\n  L3 诊断贡献 (Full - No-L3Diag): {data['full']['final'] - data['no_l3diag']['final']:+.4f}")
print(f"  完整方案 vs Baseline:           {data['full']['final'] - data['baseline']['final']:+.4f}")

# 绘图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

cfgs = {
    'full':      {'color': '#2196F3', 'ls': '-',  'label': 'Full (S+L3+OLIP)', 'lw': 2.5},
    'no_l3diag': {'color': '#4CAF50', 'ls': '-.', 'label': 'No-L3Diag (S+OLIP)', 'lw': 2.5},
    'baseline':  {'color': '#9E9E9E', 'ls': ':',  'label': 'Baseline (S only)',   'lw': 2.0},
}

for mode, cfg in cfgs.items():
    s, r = data[mode]['steps'], data[mode]['rewards']
    ax1.plot(s, r, alpha=0.15, color=cfg['color'])
    sr = smooth(r)
    ax1.plot(s[:len(sr)], sr, color=cfg['color'], linewidth=cfg['lw'], linestyle=cfg['ls'], label=cfg['label'])
    fv = data[mode]['final']
    ax1.annotate(f'{fv:.3f}', xy=(s[-1]+0.5, fv), fontsize=10, fontweight='bold', color=cfg['color'])

ax1.set_xlabel('Training Steps (M)', fontsize=12)
ax1.set_ylabel('Mean Reward', fontsize=12)
ax1.set_title('Extreme Conditions: Training Curves\n(Heavy load, bias torque, mid-episode switch)', fontsize=12, fontweight='bold')
ax1.legend(fontsize=10, loc='lower right')
ax1.grid(True, alpha=0.3)

# 柱状图
labels = ['Full\n(S+L3+OLIP)', 'No-L3Diag\n(S+OLIP)', 'Baseline\n(S only)']
values = [data['full']['final'], data['no_l3diag']['final'], data['baseline']['final']]
colors = ['#2196F3', '#4CAF50', '#9E9E9E']

bars = ax2.bar(labels, values, color=colors, width=0.5, edgecolor='white', linewidth=2)
for bar, val in zip(bars, values):
    ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003, f'{val:.4f}',
             ha='center', fontsize=12, fontweight='bold')

full_v = values[0]
for i in [1, 2]:
    d = full_v - values[i]
    color = '#C62828' if d > 0 else '#1B5E20'
    ax2.text(i, values[i]-0.012, f'Δ = {d:+.4f}\n({d/full_v*100:+.1f}%)', 
             ha='center', fontsize=10, color=color)

ax2.set_ylabel('Final Mean Reward', fontsize=12)
ax2.set_title('L3 Diagnostic Value Under Extreme Conditions\n(Heavy + Bias Torque + Motor Degradation)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.2, axis='y')
ax2.axhline(y=full_v, color='#2196F3', linestyle=':', alpha=0.5)
mn = min(values) - 0.03; mx = max(values) + 0.03
ax2.set_ylim([mn, mx])

plt.tight_layout()
plt.savefig(f'{BASE}/l3_extreme_results.png', dpi=200, bbox_inches='tight')
print("\n图已保存")
