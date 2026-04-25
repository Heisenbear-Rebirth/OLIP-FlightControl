"""
AeroCat 消融实验脚本

3 组对照实验, 验证 OLIP 和 L3 诊断的贡献:
1. Full:     11D 感知 + 4D L3诊断 + 5D OLIP (已完成)
2. No-OLIP:  11D 感知 + 4D L3诊断 + 0D (OLIP 置零)
3. No-L3Diag: 11D 感知 + 0D (L3诊断置零) + 5D OLIP

所有组网络结构一致 (20D 输入), 仅零化对应观测通道。
"""

import os
import sys
import json
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from aerocat.envs.hover_env import (
    HoverEnvConfig, reset_hover_env, step_hover_env,
    build_hover_observation, generate_olip_features
)
from aerocat.core.state import L3Config


# =============================================================================
# 网络 (和 train_hover.py 一致)
# =============================================================================
class ActorCritic(nn.Module):
    action_dim: int = 4
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        
        action_mean = nn.Dense(self.action_dim)(x)
        action_mean = nn.tanh(action_mean)
        
        log_std = self.param('log_std', nn.initializers.constant(-1.0), (self.action_dim,))
        
        v = nn.Dense(128)(x)
        v = nn.relu(v)
        value = nn.Dense(1)(v)
        value = jnp.squeeze(value, axis=-1)
        
        return action_mean, log_std, value


# =============================================================================
# 观测掩码
# =============================================================================
def apply_obs_mask(obs, mode="full"):
    """
    对观测施加消融掩码
    
    观测布局 (20D):
    [0:4]   姿态四元数       <- 始终保留
    [4:7]   角速度           <- 始终保留
    [7:10]  加速度           <- 始终保留
    [10]    高度误差         <- 始终保留
    [11:15] L3 诊断 (4D)     <- No-L3Diag 时置零
    [15:20] OLIP 特征 (5D)   <- No-OLIP 时置零
    """
    if mode == "full":
        return obs
    elif mode == "no_olip":
        # 零化 OLIP 特征 [15:20]
        return obs.at[..., 15:20].set(0.0)
    elif mode == "no_l3diag":
        # 零化 L3 诊断 [11:15]
        return obs.at[..., 11:15].set(0.0)
    else:
        raise ValueError(f"Unknown mode: {mode}")


# =============================================================================
# PPO 核心 (复用 train_hover.py 逻辑)
# =============================================================================
from typing import NamedTuple

class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    value: jnp.ndarray
    log_prob: jnp.ndarray


def sample_action(params, apply_fn, obs, key):
    action_mean, log_std, value = apply_fn(params, obs)
    std = jnp.exp(log_std)
    noise = jax.random.normal(key, action_mean.shape)
    action = jnp.clip(action_mean + noise * std, -1.0, 1.0)
    log_prob = -0.5 * jnp.sum(
        ((action - action_mean) / (std + 1e-8)) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi),
        axis=-1
    )
    return action, log_prob, value


def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    T = rewards.shape[0]
    gae = jnp.zeros(rewards.shape[1])
    def scan_fn(gae, t):
        idx = T - 1 - t
        delta = rewards[idx] + gamma * values[idx + 1] * (1 - dones[idx]) - values[idx]
        gae = delta + gamma * gae_lambda * (1 - dones[idx]) * gae
        return gae, gae
    _, advantages_reversed = jax.lax.scan(scan_fn, gae, jnp.arange(T))
    advantages = jnp.flip(advantages_reversed, axis=0)
    returns = advantages + values[:-1]
    return advantages, returns


def ppo_loss(params, apply_fn, batch, clip_eps=0.2):
    obs, actions, old_log_probs, advantages, returns = batch
    action_mean, log_std, values = apply_fn(params, obs)
    std = jnp.exp(log_std)
    log_probs = -0.5 * jnp.sum(
        ((actions - action_mean) / (std + 1e-8)) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi),
        axis=-1
    )
    ratio = jnp.exp(log_probs - old_log_probs)
    adv_norm = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
    pg_loss = -jnp.mean(jnp.minimum(ratio * adv_norm,
                                      jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * adv_norm))
    vf_loss = jnp.mean((values - returns) ** 2)
    entropy = jnp.mean(0.5 * jnp.sum(1 + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1))
    total_loss = pg_loss + 0.5 * vf_loss - 0.01 * entropy
    return total_loss, {'pg_loss': pg_loss, 'vf_loss': vf_loss, 'entropy': entropy}


# =============================================================================
# 单次实验
# =============================================================================
def run_experiment(mode, num_envs=512, total_steps=50_000_000, seed=42, save_dir=None):
    """运行单个消融实验"""
    env_config = HoverEnvConfig()
    l3_config = L3Config()
    rollout_length = 64
    num_epochs = 4
    batch_size = num_envs * rollout_length
    num_updates = total_steps // batch_size
    
    print(f"\n{'='*50}")
    print(f"  实验: {mode.upper()}")
    print(f"  总步数: {total_steps:,}  |  更新次数: {num_updates}")
    print(f"{'='*50}")
    
    rng = jax.random.PRNGKey(seed)
    
    # 初始化网络
    network = ActorCritic(action_dim=4)
    rng, init_key = jax.random.split(rng)
    params = network.init(init_key, jnp.zeros((1, env_config.obs_dim)))
    tx = optax.adam(3e-4)
    train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)
    
    # 初始环境
    rng, reset_key = jax.random.split(rng)
    phys_state, l3_state, phys_params, mixer_matrix, olip_features, prev_action, step_count, obs = \
        reset_hover_env(reset_key, num_envs, env_config)
    obs = apply_obs_mask(obs, mode)
    
    # JIT 核心
    @jax.jit
    def collect_and_update(rng, train_state, phys_state, l3_state, prev_action, step_count, obs,
                           pp, mx, ol):
        # --- Rollout ---
        def rollout_step(carry, _):
            rng, phys_st, l3_st, prev_act, sc, ob, c_pp, c_mx, c_ol = carry
            rng, ak, sk = jax.random.split(rng, 3)
            
            action, log_prob, value = sample_action(train_state.params, train_state.apply_fn, ob, ak)
            
            phys_st_new, l3_st_new, obs_new, reward, done, info = step_hover_env(
                sk, phys_st, l3_st, action, prev_act, sc,
                c_pp, l3_config, c_mx, c_ol, env_config
            )
            # 消融掩码
            obs_new = apply_obs_mask(obs_new, mode)
            
            rng, rk = jax.random.split(rng)
            r_out = reset_hover_env(rk, num_envs, env_config)
            r_phys, r_l3, r_pp, r_mx, r_ol, r_prev, r_sc, r_obs = r_out
            r_obs = apply_obs_mask(r_obs, mode)
            
            def sel(new, reset):
                if hasattr(new, 'shape') and new.ndim > 0:
                    return jnp.where(done.reshape(done.shape + (1,)*(new.ndim-1)), reset, new)
                return jnp.where(done, reset, new)
            
            pf = jax.tree_util.tree_map(sel, phys_st_new, r_phys)
            lf = jax.tree_util.tree_map(sel, l3_st_new, r_l3)
            ppf = jax.tree_util.tree_map(sel, c_pp, r_pp)
            mxf = jax.tree_util.tree_map(sel, c_mx, r_mx)
            olf = jnp.where(done[:, None], r_ol, c_ol)
            of = jnp.where(done[:, None], r_obs, obs_new)
            paf = jnp.where(done[:, None], r_prev, action)
            scf = jnp.where(done, r_sc, sc + 1)
            
            trans = Transition(obs=ob, action=action, reward=reward, done=done, value=value, log_prob=log_prob)
            return (rng, pf, lf, paf, scf, of, ppf, mxf, olf), trans
        
        carry0 = (rng, phys_state, l3_state, prev_action, step_count, obs, pp, mx, ol)
        final_carry, rollout = jax.lax.scan(rollout_step, carry0, None, length=rollout_length)
        _, _, _, _, _, last_obs, _, _, _ = final_carry
        _, _, last_value = train_state.apply_fn(train_state.params, last_obs)
        
        # --- Update ---
        vals = jnp.concatenate([rollout.value, last_value[None, :]], axis=0)
        advantages, returns = compute_gae(rollout.reward, vals, rollout.done)
        
        b_obs = rollout.obs.reshape(-1, env_config.obs_dim)
        b_act = rollout.action.reshape(-1, 4)
        b_lp = rollout.log_prob.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)
        
        def epoch_step(ts, _):
            batch = (b_obs, b_act, b_lp, b_adv, b_ret)
            (loss, info), grads = jax.value_and_grad(ppo_loss, has_aux=True)(
                ts.params, ts.apply_fn, batch)
            ts = ts.apply_gradients(grads=grads)
            return ts, info
        
        train_state_new, _ = jax.lax.scan(epoch_step, train_state, None, length=num_epochs)
        
        mean_reward = jnp.mean(rollout.reward)
        return final_carry, train_state_new, mean_reward
    
    # --- 主循环 ---
    metrics = []
    t0 = time.time()
    
    for update in range(num_updates):
        rng, rk = jax.random.split(rng)
        carry_out, train_state, mean_reward = collect_and_update(
            rk, train_state, phys_state, l3_state, prev_action, step_count, obs,
            phys_params, mixer_matrix, olip_features
        )
        rng_new, phys_state, l3_state, prev_action, step_count, obs, \
            phys_params, mixer_matrix, olip_features = carry_out
        rng = rng_new
        
        total_steps_done = (update + 1) * batch_size
        if update % 50 == 0:
            r = float(mean_reward)
            elapsed = time.time() - t0
            fps = total_steps_done / elapsed if elapsed > 0 else 0
            print(f"  [{mode:12s}] Update {update:5d}/{num_updates} | "
                  f"Steps: {total_steps_done:>10,} | Reward: {r:+.4f} | FPS: {fps:,.0f}")
            metrics.append({'update': update, 'total_steps': total_steps_done,
                            'mean_reward': r, 'elapsed': elapsed})
    
    # 保存
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f'{mode}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    import pickle
    with open(os.path.join(save_dir, f'{mode}_params.pkl'), 'wb') as f:
        pickle.dump(jax.device_get(train_state.params), f)
    
    final_r = float(mean_reward)
    print(f"  [{mode:12s}] 完成! 最终奖励: {final_r:.4f}, 耗时: {time.time()-t0:.1f}s")
    return metrics


# =============================================================================
# 主函数
# =============================================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AeroCat 消融实验")
    parser.add_argument("--steps", type=int, default=50_000_000)
    parser.add_argument("--envs", type=int, default=512)
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()
    
    if args.save_dir is None:
        args.save_dir = str(ROOT_DIR / "logs" / "ablation")
    
    print("=" * 60)
    print("  AeroCat 消融实验 (3 组)")
    print("=" * 60)
    
    all_results = {}
    
    # 实验 1: No-OLIP
    all_results['no_olip'] = run_experiment('no_olip', args.envs, args.steps, seed=42, save_dir=args.save_dir)
    
    # 实验 2: No-L3Diag
    all_results['no_l3diag'] = run_experiment('no_l3diag', args.envs, args.steps, seed=42, save_dir=args.save_dir)
    
    print("\n" + "=" * 60)
    print("  所有实验完成!")
    print("=" * 60)
