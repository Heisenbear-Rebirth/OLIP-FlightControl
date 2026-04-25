"""绘制消融实验对比图"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = '/mnt/e/study/AeroCat/dualTask/code/logs'

# 读取 3 组数据
with open(f'{BASE}/hover_v2_full/metrics.json', 'r') as f:
    full_data = json.load(f)
with open(f'{BASE}/ablation/no_olip_metrics.json', 'r') as f:
    no_olip_data = json.load(f)
with open(f'{BASE}/ablation/no_l3diag_metrics.json', 'r') as f:
    no_l3diag_data = json.load(f)

# 提取
def extract(data):
    steps = np.array([m['total_steps'] for m in data]) / 1e6
    rewards = np.array([m['mean_reward'] for m in data])
    return steps, rewards

def smooth(y, w=5):
    if len(y) < w: return y
    return np.convolve(y, np.ones(w)/w, mode='valid')

full_s, full_r = extract(full_data)
nolip_s, nolip_r = extract(no_olip_data)
nol3_s, nol3_r = extract(no_l3diag_data)

# ========= 图1: 三线对比 =========
fig, ax = plt.subplots(figsize=(10, 6))

# 原始 + 平滑
ax.plot(full_s, full_r, alpha=0.15, color='#2196F3')
ax.plot(full_s[:len(smooth(full_r))], smooth(full_r), 
        color='#2196F3', linewidth=2.5, label='Full (Sensing + L3Diag + OLIP)')

ax.plot(nolip_s, nolip_r, alpha=0.15, color='#FF5722')
ax.plot(nolip_s[:len(smooth(nolip_r))], smooth(nolip_r),
        color='#FF5722', linewidth=2.5, linestyle='--', label='No-OLIP (Sensing + L3Diag)')

ax.plot(nol3_s, nol3_r, alpha=0.15, color='#4CAF50')
ax.plot(nol3_s[:len(smooth(nol3_r))], smooth(nol3_r),
        color='#4CAF50', linewidth=2.5, linestyle='-.', label='No-L3Diag (Sensing + OLIP)')

ax.set_xlabel('Training Steps (M)', fontsize=13)
ax.set_ylabel('Mean Reward', fontsize=13)
ax.set_title('Ablation Study: Observation Component Contribution', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 50])
ax.set_ylim([0.3, 0.6])

# 标注最终值
for label, s, r, color, y_off in [
    ('Full', full_s, full_r, '#2196F3', 0),
    ('No-OLIP', nolip_s, nolip_r, '#FF5722', -0.008),
    ('No-L3Diag', nol3_s, nol3_r, '#4CAF50', 0.008),
]:
    final_r = np.mean(r[-3:])
    ax.annotate(f'{final_r:.3f}', xy=(s[-1], final_r + y_off),
                fontsize=11, fontweight='bold', color=color,
                ha='left', va='center')

plt.tight_layout()
plt.savefig(f'{BASE}/ablation/ablation_comparison.png', dpi=200, bbox_inches='tight')
print("消融对比图已保存")

# ========= 图2: 最终性能柱状图 =========
fig2, ax2 = plt.subplots(figsize=(8, 5))

# 最后 5 个点取均值
full_final = np.mean(full_r[-5:])
nolip_final = np.mean(nolip_r[-5:])
nol3_final = np.mean(nol3_r[-5:])

configs = ['Full\n(S+L3+OLIP)', 'No-OLIP\n(S+L3)', 'No-L3Diag\n(S+OLIP)']
values = [full_final, nolip_final, nol3_final]
colors = ['#2196F3', '#FF5722', '#4CAF50']

bars = ax2.bar(configs, values, color=colors, width=0.5, edgecolor='white', linewidth=2)

# 标注数值
for bar, val in zip(bars, values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
             f'{val:.4f}', ha='center', fontsize=12, fontweight='bold')

# 差异标注
delta_olip = full_final - nolip_final
delta_l3 = full_final - nol3_final
ax2.text(1, nolip_final - 0.015, f'Δ = {delta_olip:+.4f}', ha='center', fontsize=10, color='#C62828')
ax2.text(2, nol3_final - 0.015, f'Δ = {delta_l3:+.4f}', ha='center', fontsize=10, color='#1B5E20')

ax2.set_ylabel('Final Mean Reward (avg last 5)', fontsize=12)
ax2.set_title('Ablation: Final Performance Comparison', fontsize=13, fontweight='bold')
ax2.set_ylim([0.45, 0.58])
ax2.grid(True, alpha=0.2, axis='y')
ax2.axhline(y=full_final, color='#2196F3', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig(f'{BASE}/ablation/ablation_bar.png', dpi=200, bbox_inches='tight')
print("柱状图已保存")

# ========= 统计输出 =========
print(f"\n{'='*50}")
print(f"  消融实验结果统计")
print(f"{'='*50}")
print(f"  Full (S+L3+OLIP):   {full_final:.4f}")
print(f"  No-OLIP (S+L3):     {nolip_final:.4f}  (Δ = {delta_olip:+.4f}, {delta_olip/full_final*100:+.1f}%)")
print(f"  No-L3Diag (S+OLIP): {nol3_final:.4f}  (Δ = {delta_l3:+.4f}, {delta_l3/full_final*100:+.1f}%)")
print(f"\n  OLIP 贡献: {delta_olip:.4f} ({delta_olip/full_final*100:.1f}%)")
print(f"  L3Diag 贡献: {delta_l3:.4f} ({delta_l3/full_final*100:.1f}%)")
