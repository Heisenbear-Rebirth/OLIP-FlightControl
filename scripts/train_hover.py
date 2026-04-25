"""
AeroCat Hover PPO 训练脚本

极简版 PPO 训练, 专为交底书验证设计:
- 使用 hover_env.py 的 18D 观测空间
- MLP 策略网络 [18] → 128 → 64 → [4]
- 512 并行环境, ~1000万步
- 预计 30-60 分钟完成 (RTX 4060)
"""

import os
import sys
import time
import json
from pathlib import Path
from functools import partial
from typing import NamedTuple, Tuple, Dict, Any

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState
from flax import struct

# 添加 src 到路径
ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from aerocat.envs.hover_env import (
    HoverEnvConfig, reset_hover_env, step_hover_env,
    generate_olip_features
)
from aerocat.core.state import L3Config


# =============================================================================
# PPO 网络
# =============================================================================
class ActorCritic(nn.Module):
    """
    Actor-Critic MLP 网络
    
    结构: [20] → 256 → 128 → [4] (actor) / [1] (critic)
    """
    action_dim: int = 4
    
    @nn.compact
    def __call__(self, x):
        # 共享特征提取
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        
        # Actor: 输出动作均值
        action_mean = nn.Dense(self.action_dim)(x)
        action_mean = nn.tanh(action_mean)  # 限制到 [-1, 1]
        
        # 可学习的对数标准差 (初始化为 -1.0, 即 std≈0.37, 低探索)
        log_std = self.param(
            'log_std',
            nn.initializers.constant(-1.0),
            (self.action_dim,)
        )
        
        # Critic: 独立头
        v = nn.Dense(128)(x)
        v = nn.relu(v)
        value = nn.Dense(1)(v)
        value = jnp.squeeze(value, axis=-1)
        
        return action_mean, log_std, value


# =============================================================================
# PPO 数据结构
# =============================================================================
class Transition(NamedTuple):
    """单步转换数据"""
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    value: jnp.ndarray
    log_prob: jnp.ndarray


# =============================================================================
# PPO 核心函数
# =============================================================================
def sample_action(params, apply_fn, obs, key):
    """从策略中采样动作"""
    action_mean, log_std, value = apply_fn(params, obs)
    std = jnp.exp(log_std)
    
    noise = jax.random.normal(key, action_mean.shape)
    action = action_mean + noise * std
    action = jnp.clip(action, -1.0, 1.0)
    
    # 计算对数概率
    log_prob = -0.5 * jnp.sum(
        ((action - action_mean) / (std + 1e-8)) ** 2 + 
        2 * log_std + jnp.log(2 * jnp.pi),
        axis=-1
    )
    
    return action, log_prob, value


def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    """计算 GAE 优势估计"""
    T = rewards.shape[0]
    advantages = jnp.zeros_like(rewards)
    gae = jnp.zeros(rewards.shape[1])
    
    def scan_fn(gae, t):
        idx = T - 1 - t
        delta = rewards[idx] + gamma * values[idx + 1] * (1 - dones[idx]) - values[idx]
        gae = delta + gamma * gae_lambda * (1 - dones[idx]) * gae
        return gae, gae
    
    # 使用 lax.scan 反向计算
    _, advantages_reversed = jax.lax.scan(
        scan_fn, gae, jnp.arange(T)
    )
    advantages = jnp.flip(advantages_reversed, axis=0)
    
    returns = advantages + values[:-1]
    return advantages, returns


def ppo_loss(params, apply_fn, batch, clip_eps=0.2, vf_coeff=0.5, ent_coeff=0.01):
    """PPO 损失函数"""
    obs, actions, old_log_probs, advantages, returns = batch
    
    action_mean, log_std, values = apply_fn(params, obs)
    std = jnp.exp(log_std)
    
    # 新的对数概率
    log_probs = -0.5 * jnp.sum(
        ((actions - action_mean) / (std + 1e-8)) ** 2 + 
        2 * log_std + jnp.log(2 * jnp.pi),
        axis=-1
    )
    
    # 策略损失 (PPO Clip)
    ratio = jnp.exp(log_probs - old_log_probs)
    advantages_normalized = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
    
    pg_loss1 = ratio * advantages_normalized
    pg_loss2 = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages_normalized
    pg_loss = -jnp.mean(jnp.minimum(pg_loss1, pg_loss2))
    
    # 价值损失
    vf_loss = jnp.mean((values - returns) ** 2)
    
    # 熵奖励
    entropy = 0.5 * jnp.sum(1 + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1)
    entropy_loss = -jnp.mean(entropy)
    
    total_loss = pg_loss + vf_coeff * vf_loss + ent_coeff * entropy_loss
    
    return total_loss, {
        'pg_loss': pg_loss,
        'vf_loss': vf_loss,
        'entropy': jnp.mean(entropy),
        'approx_kl': jnp.mean((ratio - 1) - jnp.log(ratio + 1e-8)),
    }


# =============================================================================
# 训练主循环
# =============================================================================
def train_hover(
    num_envs: int = 512,
    total_timesteps: int = 10_000_000,
    rollout_length: int = 64,       # 每次 rollout 步数
    num_epochs: int = 4,            # PPO epoch 数
    minibatch_size: int = 2048,     # minibatch 大小
    learning_rate: float = 3e-4,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    clip_eps: float = 0.2,
    seed: int = 42,
    log_interval: int = 10,
    save_dir: str = None,
):
    """
    PPO 悬停训练主函数
    """
    print("=" * 60)
    print("  AeroCat Hover PPO Training")
    print("=" * 60)
    
    # 设置输出目录
    if save_dir is None:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        save_dir = str(ROOT_DIR / "logs" / f"hover_ppo_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    
    # 环境配置
    env_config = HoverEnvConfig()
    l3_config = L3Config()
    
    # 计算训练迭代数
    batch_size = num_envs * rollout_length
    num_updates = total_timesteps // batch_size
    num_minibatches = batch_size // minibatch_size
    
    print(f"\n配置:")
    print(f"  并行环境数: {num_envs}")
    print(f"  总步数: {total_timesteps:,}")
    print(f"  Rollout 长度: {rollout_length}")
    print(f"  更新次数: {num_updates}")
    print(f"  Batch 大小: {batch_size:,}")
    print(f"  Minibatch: {num_minibatches} x {minibatch_size}")
    print(f"  保存目录: {save_dir}")
    
    # 初始化 RNG
    rng = jax.random.PRNGKey(seed)
    
    # 初始化网络
    network = ActorCritic(action_dim=4)
    dummy_obs = jnp.zeros((1, env_config.obs_dim))
    rng, init_key = jax.random.split(rng)
    params = network.init(init_key, dummy_obs)
    
    # 统计参数量
    param_count = sum(x.size for x in jax.tree_util.tree_leaves(params))
    print(f"  网络参数量: {param_count:,}")
    
    # 优化器
    tx = optax.adam(learning_rate)
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=tx,
    )
    
    # 初始环境
    rng, reset_key = jax.random.split(rng)
    env_out = reset_hover_env(reset_key, num_envs, env_config)
    phys_state, l3_state, phys_params, mixer_matrix, olip_features, prev_action, step_count, obs = env_out
    
    print(f"\n设备: {jax.devices()[0]}")
    print(f"\n开始训练...\n")
    
    # JIT 编译核心函数
    @jax.jit
    def collect_rollout(rng, train_state, phys_state, l3_state, prev_action, step_count, obs,
                        pp, mx, ol):
        """收集一段 rollout 数据 (参数同步传递)"""
        
        def rollout_step(carry, _):
            rng, phys_st, l3_st, prev_act, sc, ob, c_pp, c_mx, c_ol = carry
            
            rng, action_key, step_key = jax.random.split(rng, 3)
            
            # 采样动作
            action, log_prob, value = sample_action(
                train_state.params, train_state.apply_fn, ob, action_key
            )
            
            # 环境步进 (使用 carry 中的物理参数)
            phys_st_new, l3_st_new, obs_new, reward, done, info = step_hover_env(
                step_key, phys_st, l3_st, action, prev_act, sc,
                c_pp, l3_config, c_mx, c_ol, env_config
            )
            
            # 自动重置 (生成新参数)
            rng, reset_key = jax.random.split(rng)
            reset_out = reset_hover_env(reset_key, num_envs, env_config)
            r_phys, r_l3, r_pp, r_mx, r_ol, r_prev, r_sc, r_obs = reset_out
            
            # done 掩码选择
            def select(new, reset):
                if hasattr(new, 'shape') and new.ndim > 0:
                    done_exp = done.reshape(done.shape + (1,) * (new.ndim - 1))
                    return jnp.where(done_exp, reset, new)
                return jnp.where(done, reset, new)
            
            phys_st_final = jax.tree_util.tree_map(select, phys_st_new, r_phys)
            l3_st_final = jax.tree_util.tree_map(select, l3_st_new, r_l3)
            # 同步更新物理参数和 OLIP 特征
            pp_final = jax.tree_util.tree_map(select, c_pp, r_pp)
            mx_final = jax.tree_util.tree_map(select, c_mx, r_mx)
            ol_final = jnp.where(done[:, None], r_ol, c_ol)
            obs_final = jnp.where(done[:, None], r_obs, obs_new)
            prev_act_final = jnp.where(done[:, None], r_prev, action)
            sc_final = jnp.where(done, r_sc, sc + 1)
            
            transition = Transition(
                obs=ob,
                action=action,
                reward=reward,
                done=done,
                value=value,
                log_prob=log_prob,
            )
            
            new_carry = (rng, phys_st_final, l3_st_final, prev_act_final, sc_final, 
                         obs_final, pp_final, mx_final, ol_final)
            return new_carry, transition
        
        init_carry = (rng, phys_state, l3_state, prev_action, step_count, obs, pp, mx, ol)
        final_carry, rollout = jax.lax.scan(rollout_step, init_carry, None, length=rollout_length)
        
        # 计算最后一步的 value (用于 GAE)
        _, _, _, _, _, last_obs, _, _, _ = final_carry
        _, _, last_value = train_state.apply_fn(train_state.params, last_obs)
        
        return final_carry, rollout, last_value
    
    @jax.jit
    def update_step(train_state, rollout, last_value):
        """PPO 更新"""
        # 计算 GAE
        values_with_last = jnp.concatenate([rollout.value, last_value[None, :]], axis=0)
        advantages, returns = compute_gae(
            rollout.reward, values_with_last, rollout.done, gamma, gae_lambda
        )
        
        # 展平
        batch_obs = rollout.obs.reshape(-1, env_config.obs_dim)
        batch_actions = rollout.action.reshape(-1, 4)
        batch_log_probs = rollout.log_prob.reshape(-1)
        batch_advantages = advantages.reshape(-1)
        batch_returns = returns.reshape(-1)
        
        # 多 epoch 更新
        def epoch_step(train_state, _):
            # Minibatch (简化: 全 batch)
            batch = (batch_obs, batch_actions, batch_log_probs, batch_advantages, batch_returns)
            
            grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)
            (loss, loss_info), grads = grad_fn(
                train_state.params, train_state.apply_fn, batch, clip_eps
            )
            train_state = train_state.apply_gradients(grads=grads)
            
            return train_state, loss_info
        
        train_state, loss_infos = jax.lax.scan(epoch_step, train_state, None, length=num_epochs)
        
        # 返回最后一个 epoch 的 info
        last_info = jax.tree_util.tree_map(lambda x: x[-1], loss_infos)
        
        # 添加奖励统计
        last_info['mean_reward'] = jnp.mean(rollout.reward)
        last_info['mean_episode_reward'] = jnp.mean(jnp.sum(rollout.reward, axis=0))
        
        return train_state, last_info
    
    # 主训练循环
    metrics_history = []
    start_time = time.time()
    
    for update in range(num_updates):
        rng, rollout_key = jax.random.split(rng)
        
        # 收集数据 (参数同步传递)
        carry_out, rollout, last_value = collect_rollout(
            rollout_key, train_state, phys_state, l3_state, prev_action, step_count, obs,
            phys_params, mixer_matrix, olip_features
        )
        rng_new, phys_state, l3_state, prev_action, step_count, obs, \
            phys_params, mixer_matrix, olip_features = carry_out
        rng = rng_new
        
        # PPO 更新
        train_state, loss_info = update_step(train_state, rollout, last_value)
        
        # 日志
        if update % log_interval == 0:
            elapsed = time.time() - start_time
            total_steps = (update + 1) * batch_size
            fps = total_steps / elapsed if elapsed > 0 else 0
            
            mean_reward = float(loss_info['mean_reward'])
            pg_loss = float(loss_info['pg_loss'])
            vf_loss = float(loss_info['vf_loss'])
            entropy = float(loss_info['entropy'])
            
            print(f"Update {update:5d}/{num_updates} | "
                  f"Steps: {total_steps:>10,} | "
                  f"Reward: {mean_reward:+.4f} | "
                  f"PG: {pg_loss:.4f} | "
                  f"VF: {vf_loss:.4f} | "
                  f"Ent: {entropy:.3f} | "
                  f"FPS: {fps:,.0f}")
            
            metrics_history.append({
                'update': update,
                'total_steps': total_steps,
                'mean_reward': mean_reward,
                'pg_loss': pg_loss,
                'vf_loss': vf_loss,
                'entropy': entropy,
                'elapsed_time': elapsed,
            })
    
    # 保存最终模型和训练曲线
    print(f"\n训练完成! 总耗时: {time.time() - start_time:.1f}s")
    
    # 保存指标
    metrics_path = os.path.join(save_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_history, f, indent=2)
    print(f"指标已保存: {metrics_path}")
    
    # 保存模型参数
    import pickle
    params_path = os.path.join(save_dir, "final_params.pkl")
    with open(params_path, 'wb') as f:
        pickle.dump(jax.device_get(train_state.params), f)
    print(f"模型已保存: {params_path}")
    
    return train_state, metrics_history


# =============================================================================
# 入口
# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AeroCat Hover PPO Training")
    parser.add_argument("--envs", type=int, default=512, help="并行环境数")
    parser.add_argument("--steps", type=int, default=10_000_000, help="总步数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--lr", type=float, default=3e-4, help="学习率")
    parser.add_argument("--save-dir", type=str, default=None, help="保存目录")
    
    args = parser.parse_args()
    
    train_hover(
        num_envs=args.envs,
        total_timesteps=args.steps,
        seed=args.seed,
        learning_rate=args.lr,
        save_dir=args.save_dir,
    )
