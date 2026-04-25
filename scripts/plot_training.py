"""绘制 Hover PPO v2 训练曲线"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 读取指标
with open('/mnt/e/study/AeroCat/dualTask/code/logs/hover_v2_full/metrics.json', 'r') as f:
    metrics = json.load(f)

steps = np.array([m['total_steps'] for m in metrics]) / 1e6
rewards = np.array([m['mean_reward'] for m in metrics])
vf_losses = np.array([m['vf_loss'] for m in metrics])
entropies = np.array([m['entropy'] for m in metrics])

# 平滑函数
def smooth(y, window=5):
    if len(y) < window:
        return y
    return np.convolve(y, np.ones(window)/window, mode='valid')

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1. 奖励曲线
axes[0].plot(steps, rewards, 'b-', alpha=0.3, linewidth=0.8)
s_steps = steps[:len(smooth(rewards))]
axes[0].plot(s_steps, smooth(rewards), 'b-', linewidth=2, label='Smoothed')
axes[0].axhline(y=0.55, color='r', linestyle='--', alpha=0.6, label='Target (0.55)')
axes[0].axhline(y=0.85, color='g', linestyle='--', alpha=0.4, label='Theoretical Max')
axes[0].set_xlabel('Steps (M)', fontsize=12)
axes[0].set_ylabel('Mean Reward', fontsize=12)
axes[0].set_title('Training Reward Curve', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3)
axes[0].legend(fontsize=10)
axes[0].set_ylim([0.3, 0.6])

# 2. 价值函数损失
axes[1].plot(steps, vf_losses, 'r-', alpha=0.3, linewidth=0.8)
axes[1].plot(s_steps, smooth(vf_losses), 'r-', linewidth=2)
axes[1].set_xlabel('Steps (M)', fontsize=12)
axes[1].set_ylabel('Value Loss', fontsize=12)
axes[1].set_title('Value Function Loss', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3)

# 3. 策略熵
axes[2].plot(steps, entropies, 'g-', linewidth=1.5)
axes[2].set_xlabel('Steps (M)', fontsize=12)
axes[2].set_ylabel('Entropy', fontsize=12)
axes[2].set_title('Policy Entropy', fontsize=13, fontweight='bold')
axes[2].grid(True, alpha=0.3)

plt.suptitle('AeroCat Hover PPO Training (OLIP-Conditioned, 20D Obs, 512 Envs)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/mnt/e/study/AeroCat/dualTask/code/logs/hover_v2_full/training_curves.png',
            dpi=200, bbox_inches='tight')
print("训练曲线已保存")

# 统计
print(f"\n========== 训练统计 ==========")
print(f"  总步数: {steps[-1]*1e6:,.0f}")
print(f"  训练耗时: {metrics[-1]['elapsed_time']:.1f}s")
print(f"  FPS: {metrics[-1]['total_steps'] / metrics[-1]['elapsed_time']:,.0f}")
print(f"  初始奖励: {rewards[0]:.4f}")
print(f"  最终奖励: {rewards[-1]:.4f}")
print(f"  最高奖励: {np.max(rewards):.4f} (@ {steps[np.argmax(rewards)]:.1f}M)")
print(f"  奖励提升: {(rewards[-1]-rewards[0])/abs(rewards[0])*100:.1f}%")
print(f"  最终 VF Loss: {vf_losses[-1]:.2f}")
print(f"  最终 Entropy: {entropies[-1]:.4f}")
print(f"  理论最大奖励: ~0.85")
print(f"  达标率: {rewards[-1]/0.85*100:.1f}%")
