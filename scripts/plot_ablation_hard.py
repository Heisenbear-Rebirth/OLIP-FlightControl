"""绘制困难消融实验对比图"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = '/mnt/e/study/AeroCat/dualTask/code/logs/ablation_hard'

# 读取 4 组数据
data = {}
for mode in ['full', 'no_olip', 'no_l3diag', 'baseline']:
    with open(f'{BASE}/{mode}_metrics.json', 'r') as f:
        data[mode] = json.load(f)

def extract(d):
    steps = np.array([m['total_steps'] for m in d]) / 1e6
    rewards = np.array([m['mean_reward'] for m in d])
    return steps, rewards

def smooth(y, w=5):
    if len(y) < w: return y
    return np.convolve(y, np.ones(w)/w, mode='valid')

# ========= 图1: 四线对比 =========
fig, ax = plt.subplots(figsize=(11, 6.5))

configs = {
    'full':      {'color': '#2196F3', 'ls': '-',  'label': 'Full (Sensing + L3Diag + OLIP)', 'lw': 2.5},
    'no_olip':   {'color': '#FF5722', 'ls': '--', 'label': 'No-OLIP (Sensing + L3Diag)',     'lw': 2.5},
    'no_l3diag': {'color': '#4CAF50', 'ls': '-.', 'label': 'No-L3Diag (Sensing + OLIP)',     'lw': 2.5},
    'baseline':  {'color': '#9E9E9E', 'ls': ':',  'label': 'Baseline (Sensing Only)',        'lw': 2.0},
}

finals = {}
for mode, cfg in configs.items():
    s, r = extract(data[mode])
    ax.plot(s, r, alpha=0.15, color=cfg['color'])
    sr = smooth(r)
    ax.plot(s[:len(sr)], sr, color=cfg['color'], linewidth=cfg['lw'], linestyle=cfg['ls'], label=cfg['label'])
    finals[mode] = np.mean(r[-5:])

ax.set_xlabel('Training Steps (M)', fontsize=13)
ax.set_ylabel('Mean Reward', fontsize=13)
ax.set_title('Ablation Study: Hard Conditions (Mass ±60%, Wind Gusts)', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 50])
ax.set_ylim([0.25, 0.56])

# 标注最终值
y_offsets = {'full': 0, 'no_olip': -0.005, 'no_l3diag': 0.005, 'baseline': -0.005}
for mode, cfg in configs.items():
    s, r = extract(data[mode])
    fv = finals[mode]
    ax.annotate(f'{fv:.3f}', xy=(s[-1]+0.5, fv + y_offsets[mode]),
                fontsize=11, fontweight='bold', color=cfg['color'], ha='left', va='center')

plt.tight_layout()
plt.savefig(f'{BASE}/hard_ablation_comparison.png', dpi=200, bbox_inches='tight')
print("困难消融对比图已保存")

# ========= 图2: 柱状图 =========
fig2, ax2 = plt.subplots(figsize=(9, 5.5))

labels = ['Full\n(S+L3+OLIP)', 'No-OLIP\n(S+L3)', 'No-L3Diag\n(S+OLIP)', 'Baseline\n(S only)']
values = [finals['full'], finals['no_olip'], finals['no_l3diag'], finals['baseline']]
colors = ['#2196F3', '#FF5722', '#4CAF50', '#9E9E9E']

bars = ax2.bar(labels, values, color=colors, width=0.55, edgecolor='white', linewidth=2)

for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f'{val:.4f}', ha='center', fontsize=12, fontweight='bold')

# 差异标注
full_v = finals['full']
for i, (mode, label) in enumerate([(1, 'no_olip'), (2, 'no_l3diag'), (3, 'baseline')]):
    delta = full_v - finals[label]
    ax2.text(i+1, finals[label] - 0.012, f'Δ = {delta:+.4f}', ha='center', fontsize=10,
             color='#C62828' if delta > 0 else '#1B5E20')

ax2.set_ylabel('Final Mean Reward (avg last 5)', fontsize=12)
ax2.set_title('Hard Ablation: Final Performance Comparison\n(Mass ±60%, Thrust ±50%, Wind Gusts)', 
              fontsize=13, fontweight='bold')
ax2.set_ylim([0.45, 0.56])
ax2.grid(True, alpha=0.2, axis='y')
ax2.axhline(y=full_v, color='#2196F3', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig(f'{BASE}/hard_ablation_bar.png', dpi=200, bbox_inches='tight')
print("柱状图已保存")

# ========= 统计 =========
print(f"\n{'='*60}")
print(f"  困难消融实验结果统计 (质量±60%, 推力±50%, 阵风)")
print(f"{'='*60}")
for mode in ['full', 'no_olip', 'no_l3diag', 'baseline']:
    delta = finals['full'] - finals[mode]
    pct = delta / finals['full'] * 100 if finals['full'] != 0 else 0
    print(f"  {mode:12s}: {finals[mode]:.4f}  (Δ vs Full = {delta:+.4f}, {pct:+.1f}%)")

print(f"\n  OLIP 贡献 (Full vs No-OLIP):     {finals['full'] - finals['no_olip']:+.4f} ({(finals['full'] - finals['no_olip'])/finals['full']*100:+.1f}%)")
print(f"  L3Diag 贡献 (Full vs No-L3Diag): {finals['full'] - finals['no_l3diag']:+.4f} ({(finals['full'] - finals['no_l3diag'])/finals['full']*100:+.1f}%)")
print(f"  组合贡献 (Full vs Baseline):     {finals['full'] - finals['baseline']:+.4f} ({(finals['full'] - finals['baseline'])/finals['full']*100:+.1f}%)")

# 学习速度对比 (到达 0.48 的步数)
print(f"\n  学习速度对比 (到达 reward=0.48 的步数):")
for mode in ['full', 'no_olip', 'no_l3diag', 'baseline']:
    s, r = extract(data[mode])
    idx = np.where(smooth(r, 3) >= 0.48)[0]
    if len(idx) > 0:
        print(f"    {mode:12s}: {s[idx[0]]:.1f}M steps")
    else:
        print(f"    {mode:12s}: 未达到")
