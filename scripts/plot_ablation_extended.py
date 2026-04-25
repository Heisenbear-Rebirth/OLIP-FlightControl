"""绘制扩展消融实验结果"""
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = '/mnt/e/study/AeroCat/dualTask/code/logs/ablation_extended'

def load(name):
    with open(f'{BASE}/{name}', 'r') as f:
        return json.load(f)

def smooth(y, w=3):
    if len(y) < w: return y
    return np.convolve(y, np.ones(w)/w, mode='valid')

# ========= 实验 A: OLIP 缩放 =========
dr_labels = ['DR20', 'DR40', 'DR60', 'DR80']
dr_pcts = ['±20%', '±40%', '±60%', '±80%']

full_finals = []
nolip_finals = []
deltas = []

print("="*60)
print("  实验 A: OLIP 缩放实验结果")
print("="*60)

for dr in dr_labels:
    fd = load(f'{dr}_full_metrics.json')
    nd = load(f'{dr}_no_olip_metrics.json')
    ff = np.mean([m['mean_reward'] for m in fd[-3:]])
    nf = np.mean([m['mean_reward'] for m in nd[-3:]])
    delta = ff - nf
    full_finals.append(ff)
    nolip_finals.append(nf)
    deltas.append(delta)
    print(f"  {dr}: Full={ff:.4f}, No-OLIP={nf:.4f}, Δ={delta:+.4f} ({delta/ff*100:+.1f}%)")

# 图 A1: 缩放趋势图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

x = np.arange(len(dr_labels))
w = 0.35
bars1 = ax1.bar(x - w/2, full_finals, w, color='#2196F3', label='Full (with OLIP)', edgecolor='white', linewidth=1.5)
bars2 = ax1.bar(x + w/2, nolip_finals, w, color='#FF5722', label='No-OLIP', edgecolor='white', linewidth=1.5)

for bar, val in zip(bars1, full_finals):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003, f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')
for bar, val in zip(bars2, nolip_finals):
    ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003, f'{val:.3f}', ha='center', fontsize=9, fontweight='bold')

ax1.set_xlabel('Domain Randomization Range (Mass)', fontsize=12)
ax1.set_ylabel('Final Mean Reward', fontsize=12)
ax1.set_title('OLIP Contribution vs. DR Range', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(dr_pcts, fontsize=11)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.2, axis='y')
ax1.set_ylim([min(min(full_finals), min(nolip_finals))-0.03, max(max(full_finals), max(nolip_finals))+0.03])

# 图 A2: Delta 趋势
ax2.bar(x, deltas, 0.5, color=['#66BB6A' if d >= 0 else '#EF5350' for d in deltas], edgecolor='white', linewidth=1.5)
for i, d in enumerate(deltas):
    ax2.text(i, d + 0.001 * (1 if d >= 0 else -1), f'{d:+.4f}\n({d/full_finals[i]*100:+.1f}%)', 
             ha='center', fontsize=10, fontweight='bold', va='bottom' if d >= 0 else 'top')

ax2.axhline(y=0, color='black', linewidth=0.8)
ax2.set_xlabel('Domain Randomization Range (Mass)', fontsize=12)
ax2.set_ylabel('Δ Reward (Full − No-OLIP)', fontsize=12)
ax2.set_title('OLIP Advantage Scaling', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(dr_pcts, fontsize=11)
ax2.grid(True, alpha=0.2, axis='y')

plt.suptitle('Experiment A: OLIP Feature Contribution Scales with Domain Randomization',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{BASE}/expA_olip_scaling.png', dpi=200, bbox_inches='tight')
print("\n实验A图已保存")


# ========= 实验 B: 电机退化 =========
print(f"\n{'='*60}")
print("  实验 B: 电机退化实验结果")
print("="*60)

b_data = {}
for mode in ['full', 'no_l3diag', 'baseline']:
    d = load(f'motor_degrade_{mode}_metrics.json')
    steps = np.array([m['total_steps'] for m in d]) / 1e6
    rewards = np.array([m['mean_reward'] for m in d])
    final = np.mean(rewards[-3:])
    b_data[mode] = {'steps': steps, 'rewards': rewards, 'final': final}
    print(f"  {mode:12s}: {final:.4f}")

fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5.5))

# 训练曲线
cfgs = {
    'full':      {'color': '#2196F3', 'ls': '-',  'label': 'Full (S+L3+OLIP)', 'lw': 2.5},
    'no_l3diag': {'color': '#4CAF50', 'ls': '-.', 'label': 'No-L3Diag (S+OLIP)', 'lw': 2.5},
    'baseline':  {'color': '#9E9E9E', 'ls': ':',  'label': 'Baseline (S only)',   'lw': 2.0},
}

for mode, cfg in cfgs.items():
    s, r = b_data[mode]['steps'], b_data[mode]['rewards']
    ax3.plot(s, r, alpha=0.2, color=cfg['color'])
    sr = smooth(r)
    ax3.plot(s[:len(sr)], sr, color=cfg['color'], linewidth=cfg['lw'], linestyle=cfg['ls'], label=cfg['label'])

ax3.set_xlabel('Training Steps (M)', fontsize=12)
ax3.set_ylabel('Mean Reward', fontsize=12)
ax3.set_title('Motor Degradation: Training Curves', fontsize=13, fontweight='bold')
ax3.legend(fontsize=10, loc='lower right')
ax3.grid(True, alpha=0.3)

# 柱状图
labels = ['Full\n(S+L3+OLIP)', 'No-L3Diag\n(S+OLIP)', 'Baseline\n(S only)']
values = [b_data['full']['final'], b_data['no_l3diag']['final'], b_data['baseline']['final']]
colors = ['#2196F3', '#4CAF50', '#9E9E9E']

bars = ax4.bar(labels, values, color=colors, width=0.5, edgecolor='white', linewidth=2)
for bar, val in zip(bars, values):
    ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003, f'{val:.4f}',
             ha='center', fontsize=12, fontweight='bold')

full_v = values[0]
for i in [1, 2]:
    d = full_v - values[i]
    ax4.text(i, values[i]-0.008, f'Δ = {d:+.4f}', ha='center', fontsize=10,
             color='#C62828' if d > 0 else '#1B5E20')

ax4.set_ylabel('Final Mean Reward', fontsize=12)
ax4.set_title('Motor Degradation: Final Performance', fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.2, axis='y')
ax4.axhline(y=full_v, color='#2196F3', linestyle=':', alpha=0.5)
ax4.set_ylim([min(values)-0.03, max(values)+0.03])

plt.suptitle('Experiment B: L3 Diagnostic Value Under Motor Degradation (1-2 motors at 50-80%)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{BASE}/expB_motor_degrade.png', dpi=200, bbox_inches='tight')
print("实验B图已保存")

# 汇总
print(f"\n{'='*60}")
print("  汇总")
print("="*60)
print(f"  实验A - OLIP 贡献随 DR 缩放:")
for i, dr in enumerate(dr_pcts):
    print(f"    {dr}: Δ = {deltas[i]:+.4f} ({deltas[i]/full_finals[i]*100:+.1f}%)")
print(f"\n  实验B - L3 诊断在电机退化场景:")
fd = b_data['full']['final']
nd = b_data['no_l3diag']['final']
bd = b_data['baseline']['final']
print(f"    Full vs No-L3Diag: {fd-nd:+.4f} ({(fd-nd)/fd*100:+.1f}%)")
print(f"    Full vs Baseline:  {fd-bd:+.4f} ({(fd-bd)/fd*100:+.1f}%)")
