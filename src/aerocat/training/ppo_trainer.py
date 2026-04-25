"""
AeroCat v18.0 PPO Trainer with APDC Curriculum

PPO 训练循环实现:
- make_train() 工厂函数
- APDC 课程化学习 (curriculum_lambda)
- RNN 状态管理
- GAE 优势估计
- 完全 JIT 编译的训练循环

参考: 06_训练环境_APDC课程化学习_v18.0.md
"""

import jax
import jax.numpy as jnp
from flax import struct, serialization, linen as nn
from flax.training import train_state
import optax
from jaxtyping import Array, Float, Int, PRNGKeyArray
from typing import Tuple, Dict, NamedTuple, Callable, Any
from functools import partial
import time
from tqdm.auto import tqdm

# 导入项目模块
from ..networks.actor_critic import (
    StochasticActorCritic, LSTMState, 
    create_stochastic_actor_critic, map_action_to_physical,
    compute_v_physics,
    OBS_DIM, ACTION_DIM, LSTM_DIM
)
from ..envs.uav_env import reset_env, step_env, EnvConfig, EnvState, TimeStep
from ..generators.param_generator import init_params as generate_params
from ..control.l3_controller import map_action_to_rates_and_thrust
from ..training.plotter import MetricsPlotter
from ..core.state import EnvParams


from ..config import TrainConfig, PPOConfig

# Checkpoint & Visualization imports
from ..utils.checkpoint_manager import CheckpointManager
from ..utils.visualizer import VideoRecorder, EnhancedVideoRecorder, compute_heading_yaw_rate
import os
import numpy as np
import json


# =============================================================================
# Rollout 数据结构
# =============================================================================
class Transition(NamedTuple):
    """单步转移数据"""
    obs: Float[Array, "num_envs obs_dim"]
    raw_action: Float[Array, "num_envs action_dim"] # Pre-Tanh (u)
    action: Float[Array, "num_envs action_dim"]     # Post-Tanh (a)
    reward: Float[Array, "num_envs"]
    done: Float[Array, "num_envs"]
    value: Float[Array, "num_envs"]
    log_prob: Float[Array, "num_envs"]
    # P0 Fix: 记录每步进入网络前的 LSTM state，用于 Update 阶段时序重建
    lstm_h: Float[Array, "num_envs lstm_dim"]   # 调用网络前的 h
    lstm_c: Float[Array, "num_envs lstm_dim"]   # 调用网络前的 c


class RolloutBuffer(NamedTuple):
    """完整 rollout 数据"""
    transitions: Transition  # [num_steps, num_envs, ...]
    last_value: Float[Array, "num_envs"]


# =============================================================================
# GAE 计算
# =============================================================================
def compute_gae(
    rewards: Float[Array, "num_steps num_envs"],
    values: Float[Array, "num_steps num_envs"],
    dones: Float[Array, "num_steps num_envs"],
    last_value: Float[Array, "num_envs"],
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[Float[Array, "num_steps num_envs"], Float[Array, "num_steps num_envs"]]:
    """
    计算 Generalized Advantage Estimation (GAE)
    
    Returns:
        advantages: GAE 优势估计
        returns: 目标回报
    """
    num_steps = rewards.shape[0]
    
    # 反向遍历计算 GAE
    def _gae_step(carry, t):
        gae, next_value = carry
        
        done = dones[t]
        value = values[t]
        reward = rewards[t]
        
        # TD error: δ_t = r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)
        delta = reward + gamma * next_value * (1.0 - done) - value
        
        # GAE: A_t = δ_t + γ * λ * (1 - done) * A_{t+1}
        gae = delta + gamma * gae_lambda * (1.0 - done) * gae
        
        return (gae, value), gae
    
    # 从最后一步开始反向扫描
    init_carry = (jnp.zeros_like(last_value), last_value)
    _, advantages = jax.lax.scan(
        _gae_step,
        init_carry,
        jnp.arange(num_steps - 1, -1, -1)
    )
    
    # 反转回正向顺序
    advantages = advantages[::-1]
    
    # 计算回报
    returns = advantages + values
    
    return advantages, returns


def compute_gae_pda(
    rewards: Float[Array, "num_steps num_envs"],
    values: Float[Array, "num_steps num_envs"],
    v_phys: Float[Array, "num_steps num_envs"],
    dones: Float[Array, "num_steps num_envs"],
    last_value: Float[Array, "num_envs"],
    last_v_phys: Float[Array, "num_envs"],
    gamma: float = 0.99,
    gae_lambda: float = 0.95
) -> Tuple[Float[Array, "num_steps num_envs"], Float[Array, "num_steps num_envs"]]:
    """
    Physics-Decomposed Advantage (PDA) — C4 创新点
    
    将 TD error 分解为解析项和残差项:
      δ_t = [r_t + γ·V_phys(s_{t+1}) - V_phys(s_t)]   ← δ_phys: 零方差
           + [γ·V_res(s_{t+1})  - V_res(s_t)]            ← δ_res: 方差缩减
    
    核心洞察: δ_phys 是解析可计算的（无估计误差），
             δ_res 中 V_res 的值域远小于 V_total → GAE 方差缩减
    
    实现: 和标准 GAE 数学等价（因为 V=V_phys+V_res），但梯度流经
          V_res 网络的信号噪声更低，帮助 Critic 更快收敛。
    """
    num_steps = rewards.shape[0]
    
    # V_res = V_total - V_phys
    v_res = values - v_phys
    last_v_res = last_value - last_v_phys
    
    def _gae_step(carry, t):
        gae, next_v_phys, next_v_res = carry
        
        done = dones[t]
        cur_v_phys = v_phys[t]
        cur_v_res = v_res[t]
        reward = rewards[t]
        
        # 分解 TD error
        delta_phys = reward + gamma * next_v_phys * (1.0 - done) - cur_v_phys
        delta_res = gamma * next_v_res * (1.0 - done) - cur_v_res
        delta = delta_phys + delta_res
        
        # GAE 递推（与标准 GAE 等价）
        gae = delta + gamma * gae_lambda * (1.0 - done) * gae
        
        return (gae, cur_v_phys, cur_v_res), gae
    
    init_carry = (jnp.zeros_like(last_value), last_v_phys, last_v_res)
    _, advantages = jax.lax.scan(
        _gae_step,
        init_carry,
        jnp.arange(num_steps - 1, -1, -1)
    )
    
    advantages = advantages[::-1]
    returns = advantages + values
    
    return advantages, returns


# =============================================================================
# PPO 损失函数
# =============================================================================
def ppo_loss_from_precomputed(
    log_probs_new: Float[Array, "batch"],
    old_log_probs: Float[Array, "batch"],
    advantages: Float[Array, "batch"],
    returns: Float[Array, "batch"],
    values: Float[Array, "batch"],
    action_mean: Float[Array, "batch action_dim"],
    action_std: Float[Array, "batch action_dim"],
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
    pg_loss_coef: float = 1.0,
    kl_coef: float = 1.0,
    bc_coef: float = 0.0,
    expert_actions: Any = None,
) -> Tuple[Float[Array, ""], Dict]:
    """
    PPO 损失计算（基于预计算的 log_probs/values，支持 BC 辅助损失）

    P0 Fix: 从外部 scan 传入已按时序重建的 log_probs_new / values，
             避免在此处做零状态前向传播导致 ratio 虚假偏离 1.0。

    Args:
        log_probs_new: 新策略对 rollout 动作的对数概率（时序 scan 重建）
        old_log_probs: rollout 时计算的对数概率（做为 ratio 分母）
        advantages:    GAE 优势估计
        returns:       GAE 目标回报
        values:        新策略的状态价值估计（时序 scan 重建）
        action_mean:   新策略的动作均值（用于 BC Loss）
        action_std:    新策略的动作标准差（用于 entropy）
        pg_loss_coef:  策略梯度系数（0.0=Critic 预热，1.0=完整 PPO）
        kl_coef:       KL 分散惩罚系数
        bc_coef:       BC 辅助损失系数（>0 时需提供 expert_actions）
        expert_actions: 专家演示动作（已经过 tanh，与网络输出空间一致）
    """
    from ..networks.actor_critic import compute_log_prob

    # PPO ratio
    ratio = jnp.exp(log_probs_new - old_log_probs)

    # 归一化优势
    advantages_normalized = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)

    # Clipped surrogate loss
    pg_loss1 = ratio * advantages_normalized
    pg_loss2 = jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * advantages_normalized
    pg_loss = -jnp.mean(jnp.minimum(pg_loss1, pg_loss2))

    # Value loss
    vf_loss = jnp.mean((values - returns) ** 2)

    # Entropy bonus
    var = action_std ** 2
    entropy = 0.5 * jnp.sum(jnp.log(2 * jnp.pi * jnp.e * var), axis=-1)
    entropy_loss = -jnp.mean(entropy)

    # KL 惩罚（Schulman 近似）
    approx_kl = jnp.mean((ratio - 1) - jnp.log(ratio))

    # P1: BC 辅助损失（防止遗忘 BC 参数，bc_coef 余弦退场）
    # expert_actions 已经过 tanh，与 tanh(action_mean) 空间一致
    bc_loss = jax.lax.cond(
        bc_coef > 0.0,
        lambda: jnp.mean(jnp.square(jnp.tanh(action_mean) - expert_actions)),
        lambda: jnp.array(0.0),
    ) if expert_actions is not None else jnp.array(0.0)

    # 彻底隔离：当 pg_loss_coef=0 (Warmup) 时，所有 Actor 相关的梯度必须绝对为 0
    actor_loss = pg_loss + ent_coef * entropy_loss + bc_coef * bc_loss
    
    total_loss = (
        pg_loss_coef * actor_loss  # 此时只包含被截断的 PPO 损失、熵和 BC 损失
        + vf_coef * vf_loss        # Critic 的独立更新
    )

    metrics = {
        "pg_loss": pg_loss,
        "vf_loss": vf_loss,
        "entropy": jnp.mean(entropy),
        "approx_kl": approx_kl,
        "clip_frac": jnp.mean(jnp.abs(ratio - 1) > clip_eps),
        "pg_loss_coef": jnp.array(pg_loss_coef),
        "kl_coef": jnp.array(kl_coef),
        "bc_loss": bc_loss,
    }

    return total_loss, metrics


def ppo_loss(
    params: dict,
    apply_fn: Callable,
    obs: Float[Array, "batch obs_dim"],
    raw_actions: Float[Array, "batch action_dim"],
    old_log_probs: Float[Array, "batch"],
    advantages: Float[Array, "batch"],
    returns: Float[Array, "batch"],
    lstm_state: LSTMState,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
    pg_loss_coef: float = 1.0,
    kl_coef: float = 1.0,
) -> Tuple[Float[Array, ""], Dict]:
    """
    PPO 损失（传统接口，保留用于兼容性）

    注意：此版本直接做整批前向传播，LSTM state 须由调用方正确提供。
    P0 修复后的主训练循环已改用 ppo_loss_from_precomputed + scan 方式。
    """
    from ..networks.actor_critic import compute_log_prob

    action_mean, action_std, values, _ = apply_fn(params, obs, lstm_state, None)
    log_probs = compute_log_prob(raw_actions, action_mean, action_std)

    return ppo_loss_from_precomputed(
        log_probs_new=log_probs,
        old_log_probs=old_log_probs,
        advantages=advantages,
        returns=returns,
        values=values,
        action_mean=action_mean,
        action_std=action_std,
        clip_eps=clip_eps,
        vf_coef=vf_coef,
        ent_coef=ent_coef,
        pg_loss_coef=pg_loss_coef,
        kl_coef=kl_coef,
    )


# =============================================================================
# 训练状态
# =============================================================================
class TrainState(train_state.TrainState):
    """扩展的训练状态 (包含课程 lambda)"""
    curriculum_lambda: Float[Array, ""]
    success_rate: Float[Array, ""]
    total_steps: Int[Array, ""]

# =============================================================================
# create_train_functions 工厂函数 (Refactored for Python Loop)
# =============================================================================
class TrainFunctions(NamedTuple):
    init_fn: Callable[[PRNGKeyArray], TrainState]
    step_fn: Callable[[TrainState], Tuple[TrainState, Dict]]
    reset_env_fn: Callable[[PRNGKeyArray], Tuple[EnvState, EnvParams, Any]]

def create_train_functions(config: TrainConfig, pg_loss_coef: float = 1.0) -> TrainFunctions:
    """
    创建训练所需的 JIT 函数集合
    
    Args:
        config: 完整训练配置
        pg_loss_coef: 策略梯度系数 (Python 层静态常量)。
                      0.0=只更新 Critic (warmup)。1.0=完整 PPO。
                      JIT 编译时会内联此常量。
    
    Returns:
        TrainFunctions: 包含 init_fn, step_fn 等
    """
    
    # 获取运行时环境配置
    env_config = config.get_runtime_env_config()
    
    # -------------------------------------------------------------------------
    # 1. init_fn: 初始化训练状态
    # -------------------------------------------------------------------------
    def init_fn(rng: PRNGKeyArray) -> Tuple[TrainState, EnvState, EnvParams, LSTMState, Float[Array, "batch obs"], PRNGKeyArray]:
        """初始化"""
        rng, init_rng, env_rng = jax.random.split(rng, 3)
        
        # 创建网络
        model, params, _ = create_stochastic_actor_critic(init_rng, config.num_envs)
        
        # 创建优化器
        tx = optax.chain(
            optax.clip_by_global_norm(config.ppo.max_grad_norm),
            optax.adam(config.ppo.learning_rate)
        )
        
        # 创建训练状态
        opt_state = tx.init(params)
        state = TrainState(
            step=jnp.array(0, dtype=jnp.int32),
            apply_fn=model.apply,
            params=params,
            tx=tx,
            opt_state=opt_state,
            curriculum_lambda=jnp.array(config.curriculum.lambda_start, dtype=jnp.float32),
            success_rate=jnp.array(0.0, dtype=jnp.float32),
            total_steps=jnp.array(0, dtype=jnp.int32),
        )
        
        # 初始化环境
        rng, reset_rng = jax.random.split(rng)
        # 修复: 传入 lambda_start，否则默认 1.0 (最难) 影响第一次 rollout
        env_params = generate_params(reset_rng, config.num_envs, config.physics, config.curriculum.lambda_start)
        
        rng, env_init_rng = jax.random.split(rng)
        env_state, _, obs = reset_env(
            env_init_rng, 
            config.num_envs, 
            env_config,
            config.curriculum.lambda_start,
            params=env_params
        )
        
        # 初始化 LSTM 状态
        lstm_state = model.init_lstm_state(config.num_envs)
        
        # Packing everything into a large tuple for Python loop handling is messy.
        # Ideally, we should put everything into TrainState or a RunnerState class.
        # For minimal change, we return the tuple as used in scan.
        
        # We need a container for the 'Runner State' that persists between steps
        # But TrainState is frozen dataclass.
        
        return state, env_state, env_params, lstm_state, obs, rng

    # -------------------------------------------------------------------------
    # 2. step_fn: 执行一次 Update (collect + train)
    # -------------------------------------------------------------------------
    def step_fn(runner_state):
        """
        执行一个完整的 Update 周期:
        1. Rollout (num_steps)
        2. GAE
        3. PPO Update (epochs)
        4. Curriculum Update
        """
        state, env_state, env_params, lstm_state, obs, rng = runner_state
        
        # Re-create model reference (closure)
        model, _, _ = create_stochastic_actor_critic(jax.random.PRNGKey(0), config.num_envs)
        
        # --- Internal Rollout Step ---
        def _env_step(carry, _):
            state, env_state, env_params, lstm_state, obs, rng = carry
            rng, action_rng, step_rng = jax.random.split(rng, 3)
            
            raw_action, action, log_prob, value, new_lstm_state = model.apply(
                state.params, obs, lstm_state, None,
                method=model.sample_action, rngs={"noise": action_rng}
            )
            # physical_action = map_action_to_physical(action) # REMOVED: Double mapping fix
            # Fix Ghost Params (v18.0.5): Capture updated params from auto-reset
            new_env_state, timestep, new_env_params = step_env(
                step_rng, env_state, action, env_params, env_config, state.curriculum_lambda
            )
            
            # P0 Fix: 存储调用网络**前**的 LSTM state（而非调用后），
            # 用于 Update 阶段按时序重建 log_prob_new，避免 state 归零导致虚假 ratio
            transition = Transition(
                obs=obs, raw_action=raw_action, action=action, reward=timestep.reward,
                done=timestep.done.astype(jnp.float32), value=value, log_prob=log_prob,
                lstm_h=lstm_state.h,   # 进入网络前的 h
                lstm_c=lstm_state.c,   # 进入网络前的 c
            )
            
            # Handle Done for LSTM: episode 结束时重置 LSTM 的隐状态
            done_mask = timestep.done[:, None].astype(jnp.float32)
            new_lstm_state = LSTMState(
                h=new_lstm_state.h * (1.0 - done_mask),
                c=new_lstm_state.c * (1.0 - done_mask),
            )
            
            # Use new_env_params for next step
            new_carry = (state, new_env_state, new_env_params, new_lstm_state, timestep.obs, rng)
            return new_carry, transition

        # --- Execute Rollout ---
        # Note: We need to pass state into scan, but scan expects constant carry structure
        rollout_carry = (state, env_state, env_params, lstm_state, obs, rng)
        final_rollout_carry, transitions = jax.lax.scan(
            _env_step, rollout_carry, None, length=config.num_steps
        )
        _, new_env_state, _, final_lstm_state, last_obs, rng = final_rollout_carry
        
        # --- GAE (PDA: Physics-Decomposed Advantage) ---
        _, _, last_value, _ = model.apply(
            state.params, last_obs, final_lstm_state, None
        )
        
        # 获取 V_phys 分量用于 PDA
        # 每步的 obs 已在 transitions 中，可 batch 计算 V_phys
        all_obs_flat = transitions.obs.reshape(-1, transitions.obs.shape[-1])  # [T*E, obs]
        psc_w = state.params['params']['psc_weights']
        psc_b = state.params['params']['psc_bias']
        all_v_phys_flat = compute_v_physics(all_obs_flat, psc_w, psc_b)  # [T*E]
        all_v_phys = all_v_phys_flat.reshape(transitions.obs.shape[0], transitions.obs.shape[1])  # [T, E]
        last_v_phys = compute_v_physics(last_obs, psc_w, psc_b)  # [E]
        
        advantages, returns = compute_gae_pda(
            transitions.reward, transitions.value, all_v_phys, transitions.done, 
            last_value, last_v_phys, config.ppo.gamma, config.ppo.gae_lambda
        )
        
        # --- PPO Update (P0 Fix: 按 env 维度切 minibatch，保留时序连续性) ---
        # transitions 保持 [num_steps, num_envs, ...] 形状，不做全局 reshape
        # 只在 env 维度做 permutation，每个 minibatch 包含 env_per_mb 个环境的完整序列
        num_envs = config.num_envs
        num_steps = config.num_steps
        env_per_mb = num_envs // config.ppo.num_minibatches

        # advantages / returns 保持 [num_steps, num_envs] 形状
        advantages_2d = advantages    # [num_steps, num_envs]
        returns_2d    = returns       # [num_steps, num_envs]

        def _update_epoch(carry, _):
            state, rng = carry
            rng, perm_rng = jax.random.split(rng)
            # 只对 env 维度做 permutation（保留每个 env 的时序结构）
            perm_envs = jax.random.permutation(perm_rng, num_envs)

            def _update_minibatch(state, mb_idx):
                # 取 env_per_mb 个连续 env 的索引
                env_idx = jax.lax.dynamic_slice(perm_envs, (mb_idx * env_per_mb,), (env_per_mb,))

                # 提取 minibatch 数据：shape [num_steps, env_per_mb, ...]
                mb_obs       = transitions.obs[:, env_idx, :]        # [T, E, obs]
                mb_raw_act   = transitions.raw_action[:, env_idx, :] # [T, E, act]
                mb_log_probs = transitions.log_prob[:, env_idx]      # [T, E]
                mb_adv       = advantages_2d[:, env_idx]             # [T, E]
                mb_ret       = returns_2d[:, env_idx]                # [T, E]

                # P0 Fix: 使用第 0 步记录的真实 LSTM state 作为 scan 的初始 carry
                # transitions.lstm_h/c 存储了每步进入网络前的状态，第 0 步即 episode 首步
                mb_lstm_h0 = transitions.lstm_h[0, env_idx, :]  # [E, lstm_dim]
                mb_lstm_c0 = transitions.lstm_c[0, env_idx, :]  # [E, lstm_dim]
                init_lstm  = LSTMState(h=mb_lstm_h0, c=mb_lstm_c0)

                def loss_fn(params):
                    # 用 lax.scan 沿时间轴重建 log_prob_new / values，正确传递 LSTM carry
                    def scan_step(lstm_carry, step_inputs):
                        step_obs, step_raw_act, step_done = step_inputs
                        action_mean_t, action_std_t, values_t, new_lstm = model.apply(
                            params, step_obs, lstm_carry, None
                        )
                        from ..networks.actor_critic import compute_log_prob
                        log_probs_t = compute_log_prob(step_raw_act, action_mean_t, action_std_t)
                        # done-mask: episode 结束后重置 LSTM carry（与 rollout 阶段一致）
                        done_mask = step_done[:, None].astype(jnp.float32)
                        new_lstm = LSTMState(
                            h=new_lstm.h * (1.0 - done_mask),
                            c=new_lstm.c * (1.0 - done_mask),
                        )
                        return new_lstm, (log_probs_t, values_t, action_mean_t, action_std_t)

                    _, (log_probs_new, values_all, means_all, stds_all) = jax.lax.scan(
                        scan_step,
                        init_lstm,
                        (mb_obs, mb_raw_act, transitions.done[:, env_idx])
                    )
                    # 展平 [num_steps, env_per_mb] -> [num_steps * env_per_mb]
                    flat = lambda x: x.reshape((-1,) + x.shape[2:])
                    return ppo_loss_from_precomputed(
                        log_probs_new=flat(log_probs_new),
                        old_log_probs=flat(mb_log_probs),
                        advantages=flat(mb_adv),
                        returns=flat(mb_ret),
                        values=flat(values_all),
                        action_mean=flat(means_all),
                        action_std=flat(stds_all),
                        clip_eps=config.ppo.clip_eps,
                        vf_coef=config.ppo.vf_coef,
                        ent_coef=config.ppo.ent_coef,
                        pg_loss_coef=pg_loss_coef,
                        kl_coef=config.ppo.kl_coef,
                    )

                (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
                return state.apply_gradients(grads=grads), metrics

            state, metrics = jax.lax.scan(
                _update_minibatch, state, jnp.arange(config.ppo.num_minibatches)
            )
            return (state, rng), metrics

        rng, update_rng = jax.random.split(rng)
        (state, _), all_metrics = jax.lax.scan(
            _update_epoch, (state, update_rng), None, length=config.ppo.update_epochs
        )
        
        # --- Curriculum (v18.7 成功率方案) ---
        # 
        # 公式: R = clip(0.45×R_vel + 0.35×R_dir + 0.05×R_spin + 0.05×R_act + 0.025, [-1,1])
        # R_max = 0.45 + 0.35 + 0 + 0 + 0.025 = 0.825
        #
        # 场景:                  v_err  cos_sim  R_total
        # ① 完美                 0      1.0      0.825
        # ② 良好 (v_err=1)       1      0.9      0.724
        # ③ 一般 (v_err=2)       2      0.7      0.589
        # ④ 较差 (v_err=5)       5      0.0      0.173
        # ⑤ 极端 (v_err=8)       8     -0.5     -0.237
        #
        # stable_threshold=0.50: v_err≈2.5, cos≈0.6 以上算稳定帧
        # success_threshold=0.55: 55% 帧稳定即推进 lambda
        
        mean_reward = jnp.mean(transitions.reward)
        
        # 稳定帧判定
        stable_ratio = jnp.mean((transitions.reward > config.curriculum.stable_threshold).astype(jnp.float32))
        
        success_rate = jnp.clip(stable_ratio, 0.0, 1.0).astype(jnp.float32)
        
        # 课程进阶
        new_lambda = jax.lax.cond(
            success_rate > config.curriculum.success_threshold,
            lambda l: jnp.minimum(l + config.curriculum.lambda_increase_rate, config.curriculum.lambda_end),
            lambda l: l,
            state.curriculum_lambda
        ).astype(jnp.float32)
        
        state = state.replace(
            curriculum_lambda=new_lambda,
            success_rate=success_rate,
            total_steps=state.total_steps + config.num_envs * config.num_steps
        )
        
        # --- New Params ---
        rng, param_rng = jax.random.split(rng)
        new_env_params = generate_params(param_rng, config.num_envs, config.physics, state.curriculum_lambda)
        
        new_runner_state = (state, new_env_state, new_env_params, final_lstm_state, last_obs, rng)
        
        # --- PSC Metrics (v19.0) ---
        v_phys_mean = jnp.mean(all_v_phys)
        v_res_mean = jnp.mean(transitions.value - all_v_phys)
        
        metrics_summary = {
            "mean_reward": mean_reward,
            "success_rate": success_rate,
            "curriculum_lambda": new_lambda,
            "pg_loss": jnp.mean(all_metrics["pg_loss"]),
            "vf_loss": jnp.mean(all_metrics["vf_loss"]),
            "entropy": jnp.mean(all_metrics["entropy"]),
            "approx_kl": jnp.mean(all_metrics["approx_kl"]),
            "clip_frac": jnp.mean(all_metrics["clip_frac"]),
            "bc_loss": jnp.mean(all_metrics["bc_loss"]),
            "sps": 0.0,  # Placeholder, computed in python loop
            # v19.0 PSC 指标
            "v_phys_mean": v_phys_mean,
            "v_res_mean": v_res_mean,
            "psc_w0_vel": psc_w[0],
            "psc_w1_ang": psc_w[1],
            "psc_w2_tilt": psc_w[2],
            "psc_w3_int": psc_w[3],
            "psc_w4_sat": psc_w[4],
            "psc_bias": psc_b,
        }
        
        return new_runner_state, metrics_summary

    return TrainFunctions(init_fn=init_fn, step_fn=step_fn, reset_env_fn=None)


# =============================================================================
# make_train (Python Loop with Checkpointing & Eval)
# =============================================================================
def make_train(config: TrainConfig) -> Callable[[PRNGKeyArray], Tuple[TrainState, Dict]]:
    """
    创建一个带有 Checkpoint 和 Evaluation 的训练函数
    
    采用 Python 循环外层 + JIT 编译的内层 step_fn。
    
    Args:
        config: 训练配置
    
    Returns:
        train: Callable(rng) -> (final_state, metrics)
    """
    funcs = create_train_functions(config)
    
    # 给 log_dir 自动追加时间戳后缀（避免覆盖旧日志）
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir_timestamped = os.path.join(
        config.logging.log_dir, 
        f"{config.logging.experiment_name}_{timestamp}"
    )
    
    # 初始化 CheckpointManager
    # 如果 save_dir 未显式指定（默认值 "checkpoints"），则放在 log 目录下
    ckpt_save_dir = config.checkpoint.save_dir
    if ckpt_save_dir in (None, "", "checkpoints"):
        ckpt_save_dir = os.path.join(log_dir_timestamped, "checkpoints")
    ckpt_manager = None
    if ckpt_save_dir:
        ckpt_manager = CheckpointManager(
            ckpt_save_dir, 
            keep_last_n=config.checkpoint.keep_last_n
        )
    
    # Initialize Matplotlib Plotter
    plotter = MetricsPlotter(os.path.join(log_dir_timestamped, config.logging.experiment_name))
    plotter.load() # Restore history if exists (for resume)
        
    def train(rng: PRNGKeyArray) -> Tuple[TrainState, Dict]:
        # 0. Instantiate model structure for evaluation usage
        # We need a dummy model instance to call apply/sample_action.
        # Params are passed dynamically from state.
        model, _, _ = create_stochastic_actor_critic(jax.random.PRNGKey(0), config.num_envs)
        
        # 1. 初始化
        # init_fn returns (state, env_state, env_params, lstm_state, obs, rng)
        runner_state = funcs.init_fn(rng)
        
        # Unpack state to check for resume
        state, env_state, env_params, lstm_state, obs, rng = runner_state
        start_step = 0
        
        # 尝试恢复 Checkpoint
        # 确定使用的 Checkpoint Manager
        resume_manager_to_use = None
        is_auto_resume = False

        if config.checkpoint.resume_from:
            resume_path = os.path.abspath(config.checkpoint.resume_from)
            if os.path.exists(resume_path):
                tqdm.write(f"[*] Resume: Attempting to restore from: {resume_path}")
                try:
                    resume_manager_to_use = CheckpointManager(resume_path)
                except Exception as e:
                    print(f"[!] Resume: Failed to init manager: {e}")
            else:
                print(f"[-] Resume: Path not found: {resume_path}")
        # elif ckpt_manager:
        #     # Auto-resume from save directory
        #     resume_manager_to_use = ckpt_manager
        #     is_auto_resume = True

        # 执行恢复流程
        if resume_manager_to_use:
            # Define restoration structures
            basic_structure = {
                "params": state.params,
                "opt_state": state.opt_state,
                "step": state.step
            }
            full_structure = basic_structure.copy()
            full_structure.update({
                "curr_lambda": state.curriculum_lambda,
                "total_steps": state.total_steps,
                "env_state": env_state,
                "env_params": env_params,
                "lstm_state": lstm_state,
                "obs": obs,
                "rng": rng
            })

            try:
                # Try Full Resume First
                restored_params = None
                try:
                    tqdm.write("[*] Resume: Attempting full state restore (perfect resume)...")
                    restored_params, step = resume_manager_to_use.restore_latest(full_structure)
                except Exception as e:
                    tqdm.write(f"[-] Resume: Full restore failed: {e}. Falling back to basic restore.")
                
                # Fallback to Basic
                if restored_params is None:
                    tqdm.write("[*] Resume: Attempting basic state restore (legacy)...")
                    restored_params, step = resume_manager_to_use.restore_latest(basic_structure)

                if restored_params:
                    tqdm.write(f"[+] Resume: Successfully restored from step {step}")
                    # Calculate steps_per_iter locally for legacy restore
                    _steps_per_iter = config.num_steps * config.num_envs
                    
                    # 1. Restore TrainState
                    state = state.replace(
                        params=restored_params["params"], 
                        opt_state=restored_params["opt_state"],
                        step=restored_params.get("step", step // _steps_per_iter),
                        curriculum_lambda=restored_params.get("curr_lambda", state.curriculum_lambda),
                        total_steps=restored_params.get("total_steps", state.total_steps)
                    )
                    
                    # 2. Restore Runner State (Perfect Resume)
                    if "env_state" in restored_params:
                        tqdm.write("[+] Resume: Perfect restore (Env, LSTM, Obs state restored)")
                        env_state = restored_params["env_state"]
                        env_params = restored_params["env_params"]
                        lstm_state = restored_params["lstm_state"]
                        obs = restored_params["obs"]
                        rng = restored_params["rng"]
                    else:
                        tqdm.write("[-] Resume: Partial restore (Env/LSTM state reset, legacy checkpoint)")
                    
                    runner_state = (state, env_state, env_params, lstm_state, obs, rng)
                    start_step = step
                    
                    # 清理 checkpoint 之后的 metrics 记录，避免 resume 后曲线重复
                    plotter.truncate(start_step)
            except Exception as e:
                print(f"[!] Resume: Failed: {e}")

        # Behavior Cloning Hot-Start
        # Only load if we are starting fresh (start_step == 0) and path is provided
        if start_step == 0 and config.ppo.pretrained_path:
            pt_path = config.ppo.pretrained_path
            if os.path.exists(pt_path):
                tqdm.write(f"[*] Hot-Start: Loading pretrained policy from {pt_path}")
                try:
                    with open(pt_path, "rb") as f:
                        params_bytes = f.read()

                    # 尝试严格加载 (完全兼容的 checkpoint)
                    try:
                        new_params = serialization.from_bytes(state.params, params_bytes)
                        
                        # v18.8: 重置 log_std, BC 训练的 log_std 太低(≈-3, std≈0.05)
                        # PPO 需要更大的 std 才能稳定训练 (ratio 不会爆炸)
                        RESET_LOG_STD = -2.0  # std ≈ 0.135, 合理的探索宽度
                        def _reset_log_std(params_dict):
                            """递归查找并重置 log_std 参数"""
                            if isinstance(params_dict, dict):
                                result = {}
                                for k, v in params_dict.items():
                                    if k == 'log_std':
                                        old_val = v
                                        result[k] = jnp.full_like(v, RESET_LOG_STD)
                                        tqdm.write(f"    log_std: {old_val} → {RESET_LOG_STD} (std: {float(jnp.exp(old_val[0])):.4f} → {float(jnp.exp(RESET_LOG_STD)):.4f})")
                                    else:
                                        result[k] = _reset_log_std(v)
                                return result
                            return params_dict
                        new_params = _reset_log_std(new_params)
                        
                        state = state.replace(params=new_params)
                        runner_state = (state, env_state, env_params, lstm_state, obs, rng)
                        tqdm.write("[+] Hot-Start: Pretrained parameters loaded successfully (full match)")
                    except Exception as strict_err:
                        # 参数树不完全匹配 (例如 BC 参数缺少新增的 log_std)
                        # 使用 partial load: 只加载旧参数中存在的部分，忽略新增参数
                        tqdm.write(f"[-] Hot-Start: Strict load failed ({strict_err}), trying partial load...")
                        try:
                            # BC 参数用 flax.serialization.to_bytes 存储，用 msgpack 解码为原始 dict
                            from flax.serialization import msgpack_restore
                            bc_params = msgpack_restore(params_bytes)
                        except Exception as decode_err:
                            tqdm.write(f"[!] Hot-Start: Cannot decode params: {decode_err}")
                            raise

                        def partial_merge(target, source):
                            """递归合并: source 有的 key 覆盖 target, target 独有的 key 保留"""
                            if not isinstance(target, dict):
                                return source
                            merged = dict(target)
                            for k, v in source.items():
                                if k in target:
                                    merged[k] = partial_merge(target[k], v)
                                # source 中 target 没有的 key 直接忽略
                            return merged

                        merged = partial_merge(state.params, bc_params)
                        new_params = jax.tree_util.tree_map(lambda x: x, merged)
                        state = state.replace(params=new_params)
                        runner_state = (state, env_state, env_params, lstm_state, obs, rng)
                        tqdm.write("[+] Hot-Start: Pretrained parameters loaded (partial merge, new params use default init)")

                except Exception as e:
                    tqdm.write(f"[!] Hot-Start: Failed to load pretrained params: {e}")
            else:
                tqdm.write(f"[-] Hot-Start: Pretrained path not found: {pt_path}")

                
        # 2. 训练循环
        tqdm.write(f"[*] Training: Starting from step {start_step}")

        # 2. 计算迭代次数
        # total_steps = num_iterations * num_steps * num_envs
        steps_per_iter = config.num_steps * config.num_envs
        remaining_steps = config.total_timesteps - start_step
        num_iterations = max(0, remaining_steps // steps_per_iter)
        
        tqdm.write(f"[*] Training: {num_iterations} iterations ({remaining_steps} steps remaining)...")
        
        # Critic 预热: 为 warmup/normal 分别创建独立的 JIT 副本
        # pg_loss_coef 作为 Python 常量传入闭包，JIT 会内联
        critic_warmup_iters = getattr(config.ppo, 'critic_warmup_steps', 0)
        
        funcs_warmup = create_train_functions(config, pg_loss_coef=0.0)  # 只更新 Critic
        funcs_normal = create_train_functions(config, pg_loss_coef=1.0)  # 完整 PPO
        step_fn_warmup_jit = jax.jit(funcs_warmup.step_fn)
        step_fn_normal_jit = jax.jit(funcs_normal.step_fn)
        
        # 内循环 Scan: K 次 step_fn 包进 lax.scan，减少 GPU↔Host 同步
        scan_iters = getattr(config.ppo, 'scan_iters', 1)
        
        def _make_multi_step(single_step_fn, K):
            """将单步 step_fn 包装为 K 步 lax.scan 版本"""
            @jax.jit
            def multi_step(runner_state):
                def scan_body(rs, _):
                    new_rs, m = single_step_fn(rs)
                    return new_rs, m
                final_rs, all_metrics = jax.lax.scan(scan_body, runner_state, None, length=K)
                # 返回最后一步的 metrics（用于日志）
                last_metrics = jax.tree_util.tree_map(lambda x: x[-1], all_metrics)
                return final_rs, last_metrics
            return multi_step
        
        if scan_iters > 1:
            multi_step_warmup = _make_multi_step(funcs_warmup.step_fn, scan_iters)
            multi_step_normal = _make_multi_step(funcs_normal.step_fn, scan_iters)
            tqdm.write(f"[*] Inner Scan: {scan_iters} iters/sync (GPU stays on-device between syncs)")
        else:
            multi_step_warmup = step_fn_warmup_jit
            multi_step_normal = step_fn_normal_jit
        
        tqdm.write(f"[*] Critic Warmup: {critic_warmup_iters} iters (only Critic updated, Actor frozen)")
        tqdm.write(f"[*] Normal Training: starts at iter {critic_warmup_iters}")
        
        # Entropy 动态控制器状态
        ENT_COEF_LEVELS = [5e-4, 1e-3, 1.5e-3, 2e-3, 3e-3]  # 恢复旧版成功训练档位
        # 找到当前 ent_coef 最近的档位
        current_ent_level = min(range(len(ENT_COEF_LEVELS)), 
                                key=lambda i: abs(ENT_COEF_LEVELS[i] - config.ppo.ent_coef))
        ent_cooldown_counter = 0
        ent_history_buffer = []  # 滑动窗口
        ent_smooth_window = getattr(config.ppo, 'entropy_smooth_window', 20)
        ent_cooldown_max = getattr(config.ppo, 'entropy_cooldown', 100)
        ent_target_min = getattr(config.ppo, 'entropy_target_min', -4.5)
        ent_target_max = getattr(config.ppo, 'entropy_target_max', -3.5)
        tqdm.write(f"[*] Entropy Controller: levels={ENT_COEF_LEVELS}, current=L{current_ent_level}({ENT_COEF_LEVELS[current_ent_level]:.1e}), target=[{ent_target_min}, {ent_target_max}]")
        
        def _rebuild_jit_fns():
            """当 ent_coef 变化时重建 JIT 函数"""
            nonlocal multi_step_warmup, multi_step_normal
            funcs_w = create_train_functions(config, pg_loss_coef=0.0)
            funcs_n = create_train_functions(config, pg_loss_coef=1.0)
            if scan_iters > 1:
                multi_step_warmup = _make_multi_step(funcs_w.step_fn, scan_iters)
                multi_step_normal = _make_multi_step(funcs_n.step_fn, scan_iters)
            else:
                multi_step_warmup = jax.jit(funcs_w.step_fn)
                multi_step_normal = jax.jit(funcs_n.step_fn)
        
        # 3. 训练循环 (Python Loop)
        # 步幅 = scan_iters，每次 Python 循环执行 K 次 JIT 迭代
        history = {
            "mean_reward": [],
            "pg_loss": [],
            "vf_loss": [],
            "entropy": [],
            "sps": []
        }
        
        t0 = time.time()
        
        # Python 循环的步幅 = scan_iters
        outer_iters = num_iterations // scan_iters
        pbar = tqdm(range(outer_iters), desc="Training", unit=f"x{scan_iters}iter")
        interrupted = False
        try:
            for outer_i in pbar:
                iter_start = time.time()
                
                # 当前逻辑迭代范围: [i_start, i_end)
                i_start = outer_i * scan_iters
                i_end = i_start + scan_iters
                
                # Critic Warmup 过渡处理
                if i_start < critic_warmup_iters <= i_end:
                    # 跨越 warmup 边界：逐步执行以精确切换
                    for i in range(i_start, i_end):
                        if i == critic_warmup_iters:
                            tqdm.write(f"[+] Critic Warmup DONE at iter {i}. Switching to full PPO update.")
                        fn = step_fn_warmup_jit if i < critic_warmup_iters else step_fn_normal_jit
                        runner_state, metrics = fn(runner_state)
                elif i_end <= critic_warmup_iters:
                    # 完全在 warmup 期内
                    runner_state, metrics = multi_step_warmup(runner_state)
                else:
                    # 完全在正常训练期
                    runner_state, metrics = multi_step_normal(runner_state)

                
                # Unpack for utils & saving
                state, env_state, env_params, lstm_state, obs, rng = runner_state
                
                # 当前步数（以 i_end 为基准）
                current_step = start_step + i_end * steps_per_iter
                
                # Logging
                iter_time = time.time() - iter_start
                sps = (steps_per_iter * scan_iters) / iter_time
                
                pbar.set_postfix({
                    "step": current_step,
                    "rew": f"{metrics['mean_reward']:.2f}",
                    "lam": f"{metrics['curriculum_lambda']:.2f}",
                    "sps": int(sps)
                })
                    
                # History update
                history["mean_reward"].append(metrics['mean_reward'])
                
                # Plotter Update
                plot_metrics = {}
                for k, v in metrics.items():
                    if hasattr(v, 'item'):
                        plot_metrics[k] = float(v)
                    elif isinstance(v, (int, float, jnp.ndarray)):
                        plot_metrics[k] = float(v)
                plot_metrics['sps'] = float(sps)
                plotter.update(plot_metrics, step=int(current_step))
                
                # Save artifacts (Plot + JSON) periodically
                if outer_i % (config.logging.save_interval // max(scan_iters, 1)) == 0:
                    plotter.plot()
                    plotter.save()
                
                # =============================================================
                # Entropy 动态控制器
                # =============================================================
                ent_val = float(plot_metrics.get('entropy', -999.0))
                ent_history_buffer.append(ent_val)
                if len(ent_history_buffer) > ent_smooth_window:
                    ent_history_buffer.pop(0)
                
                if outer_i > 50 and len(ent_history_buffer) >= ent_smooth_window:
                    ent_avg = sum(ent_history_buffer) / len(ent_history_buffer)
                    
                    # 动态调档
                    if ent_cooldown_counter > 0:
                        ent_cooldown_counter -= 1
                    elif ent_avg > ent_target_max and current_ent_level > 0:
                        # entropy 太高（太随机）→ 降档（减小 ent_coef）
                        current_ent_level -= 1
                        config.ppo.ent_coef = ENT_COEF_LEVELS[current_ent_level]
                        ent_cooldown_counter = ent_cooldown_max
                        tqdm.write(f"[ENT-CTRL] ↓ entropy={ent_avg:.3f} > {ent_target_max} → L{current_ent_level}(ent_coef={config.ppo.ent_coef:.1e}), cooldown={ent_cooldown_max}")
                        _rebuild_jit_fns()
                    elif ent_avg < ent_target_min and current_ent_level < len(ENT_COEF_LEVELS) - 1:
                        # entropy 太低（太窄）→ 升档（增大 ent_coef）
                        current_ent_level += 1
                        config.ppo.ent_coef = ENT_COEF_LEVELS[current_ent_level]
                        ent_cooldown_counter = ent_cooldown_max
                        tqdm.write(f"[ENT-CTRL] ↑ entropy={ent_avg:.3f} < {ent_target_min} → L{current_ent_level}(ent_coef={config.ppo.ent_coef:.1e}), cooldown={ent_cooldown_max}")
                        _rebuild_jit_fns()
                    
                    # 极端兜底: 即使在最低档，entropy 仍超过硬停止阈值
                    entropy_stop = getattr(config.ppo, 'entropy_stop_threshold', -3.0)
                    if ent_avg > entropy_stop and current_ent_level == 0:
                        if not hasattr(pbar, '_ent_over_count'):
                            pbar._ent_over_count = 0
                        pbar._ent_over_count += 1
                        if pbar._ent_over_count >= 30:
                            tqdm.write(f"[!] ENTROPY SAFETY STOP: avg={ent_avg:.3f} > {entropy_stop} at lowest level for 30 iters")
                            if ckpt_manager:
                                save_item = {
                                    "params": state.params, "opt_state": state.opt_state,
                                    "step": state.step, "curr_lambda": state.curriculum_lambda,
                                    "total_steps": state.total_steps,
                                    "env_state": env_state, "env_params": env_params,
                                    "lstm_state": lstm_state, "obs": obs, "rng": rng
                                }
                                ckpt_manager.save(current_step, save_item)
                            plotter.plot()
                            plotter.save()
                            return runner_state, metrics
                    else:
                        if hasattr(pbar, '_ent_over_count'):
                            pbar._ent_over_count = 0
                
                # Checkpoint
                if ckpt_manager and (current_step % config.checkpoint.save_every_steps < steps_per_iter * scan_iters):
                    tqdm.write(f"[*] Checkpoint: Saving at step {current_step}...")
                    
                    # Perfect Resume: Save everything needed to reconstruct runner_state
                    save_item = {
                        "params": state.params,
                        "opt_state": state.opt_state,
                        "step": state.step,
                        "curr_lambda": state.curriculum_lambda,
                        "total_steps": state.total_steps,
                        # Extra state for perfect resume
                        "env_state": env_state,
                        "env_params": env_params,
                        "lstm_state": lstm_state,
                        "obs": obs,
                        "rng": rng
                    }
                    ckpt_manager.save(current_step, save_item, metrics={"reward": float(metrics['mean_reward'])})
                    
                # Evaluation & Video
                if outer_i % (config.logging.eval_interval // max(scan_iters, 1)) == 0 and config.logging.record_video:
                    # Run Evaluation Episode (One env)
                    # Use the helper function `train_step_debug` logic but just for eval
                    # Actually we can reuse `step_env` with a separate eval key
                    
                    # Setup eval config (single env)
                    eval_env_config = config.env.to_runtime()
                    
                    # Eval Rollout Loop
                    # We reuse the `step_env` but operate on a single instance manually extracted or just use index 0 of the batch
                    eval_obs_list = []
                    
                    # We need fresh init for eval to be clean
                    # But creating a new env state is expensive if JIT needs recompliation?
                    # step_env is JITted. We can reuse it.
                    # Just use the current state[0] for visualization!
                    # Or better: Run a short rollout using the first env in parallel with training? No, that messes up state.
                    
                    # 2. Force Reset for clean evaluation (Generate FRESH params for batch=1)
                    # Note: We use a separate key to ensure independence from training trajectory
                    eval_reset_key = jax.random.fold_in(rng, current_step + 1000)
                    
                    def eval_reset_fn(k, l):
                        # params=None triggers init_params(batch_size=1) inside reset_env
                        return reset_env(k, 1, eval_env_config, l, params=None)
                    
                    eval_reset_jit = jax.jit(eval_reset_fn)
                    
                    # Perform Reset (Generates new params)
                    # Perform Reset (Generates new params)
                    # print(f"Eval Lambda: {metrics['curriculum_lambda']:.4f}") # REMOVED: Redundant
                    eval_state, eval_params, eval_obs = eval_reset_jit(
                         eval_reset_key, metrics['curriculum_lambda']
                    )
                    
                    # Reset LSTM
                    eval_lstm = model.init_lstm_state(1)
                    
                    # print(f"Eval Start: Height={-eval_state.phys_state.position[0, 2]:.2f}m") # REMOVED: Duplicate
                    
                    video_dir = os.path.join(config.logging.log_dir, "videos")
                    os.makedirs(video_dir, exist_ok=True)
                    video_path = os.path.join(video_dir, f"eval_step_{current_step}.mp4")
                    recorder = EnhancedVideoRecorder(video_path, fps=50)
                    eval_len = int(config.env.max_episode_time / 0.02)
                    
                    tqdm.write(f"[*] Video: Recording eval (Lambda={metrics['curriculum_lambda']:.2f}, Height={-eval_state.phys_state.position[0, 2]:.1f}m) -> {os.path.basename(video_path)}")
                    # print(f"Recording {eval_len} frames...") # REMOVED: Redundant
                    
                    # Eval JIT function (to speed up loop)
                    # We need a step function that takes (state, lstm, obs, rng) -> (state, lstm, obs, info)
                    # And returns visualization info
                    
                    def eval_step(s, l, o, r, p): # added params argument
                        r, ar, sr = jax.random.split(r, 3)
                        # Use sample_action to get raw action
                        ra, a, lp, v, nl = model.apply(state.params, o, l, None, method=model.sample_action, rngs={"noise": ar})
                        # pa = map_action_to_physical(a) # REMOVED: Double mapping fix
                        
                        # Step Env
                        ns, ts, np = step_env(sr, s, a, 
                                          p, # Use explicitly passed params
                                          eval_env_config, 
                                          metrics['curriculum_lambda'])
                        
                        # Handle LSTM done
                        dm = ts.done[:, None].astype(jnp.float32)
                        nl = LSTMState(h=nl.h * (1.0 - dm), c=nl.c * (1.0 - dm))
                        
                        return ns, nl, ts.obs, ts.reward, ts.info, ts.done, np

                    eval_step_jit = jax.jit(eval_step)
                    
                    eval_rng = jax.random.PRNGKey(current_step) # Deterministic seed for this step
                    
                    for _ in range(eval_len):
                        eval_state, eval_lstm, eval_obs, reward, info, done, eval_params = eval_step_jit(
                            eval_state, eval_lstm, eval_obs, eval_rng, eval_params
                        )
                        
                        # Offload to CPU for rendering
                        state_cpu = jax.device_get(eval_state)
                        info_cpu = jax.device_get(info)
                        # Inject reward for visualizer which expects it in info
                        info_cpu['reward'] = jax.device_get(reward)
                        
                        # 计算增强渲染所需数据
                        q_np = np.array(state_cpu.phys_state.quaternion[0])
                        omega_np = np.array(state_cpu.phys_state.angular_velocity[0])
                        yaw_rate_act = compute_heading_yaw_rate(q_np, omega_np)
                        yaw_rate_cmd = float(state_cpu.l1_state.prev_yaw_rate_cmd[0])
                        
                        recorder.render_enhanced_frame(
                            state_cpu, info_cpu,
                            yaw_rate_actual=yaw_rate_act,
                            yaw_rate_cmd=yaw_rate_cmd,
                            extra_label="PPO Agent",
                        )
                        
                        # Update RNG
                        eval_rng, _ = jax.random.split(eval_rng)
                        
                        if done[0]:
                            break
                            
                    recorder.save()
                
        except KeyboardInterrupt:
            interrupted = True
            pbar.close()
            tqdm.write("")
            tqdm.write("[*] Interrupted: Saving checkpoint and metrics before exit...")
            
            # 解包当前状态
            state, env_state, env_params, lstm_state, obs, rng = runner_state
            current_step = start_step + (pbar.n) * scan_iters * steps_per_iter
            
            # 保存 Checkpoint
            if ckpt_manager:
                tqdm.write(f"[*] Interrupted: Saving checkpoint at step {current_step}...")
                save_item = {
                    "params": state.params,
                    "opt_state": state.opt_state,
                    "step": state.step,
                    "curr_lambda": state.curriculum_lambda,
                    "total_steps": state.total_steps,
                    "env_state": env_state,
                    "env_params": env_params,
                    "lstm_state": lstm_state,
                    "obs": obs,
                    "rng": rng
                }
                ckpt_manager.save(current_step, save_item, metrics={"reward": 0.0})
            
            # 保存 Metrics 和 Plot
            tqdm.write("[*] Interrupted: Saving metrics and plot...")
            plotter.plot()
            plotter.save()
            
            tqdm.write(f"[+] Interrupted: State saved. Resume with: --resume {config.checkpoint.save_dir}")
        
        return state, history
        
    return train


# =============================================================================
# 辅助函数: 非 JIT 训练 (便于调试)
# =============================================================================
def train_step_debug(
    state: TrainState,
    model: StochasticActorCritic,
    env_state: EnvState,
    env_params: Any,
    lstm_state: LSTMState,
    obs: Float[Array, "batch obs_dim"],
    config: PPOConfig,
    rng: PRNGKeyArray,
    env_config: EnvConfig = None
) -> Tuple[TrainState, EnvState, LSTMState, Float[Array, "batch obs_dim"], Dict]:
    """
    单步训练 (非 JIT, 用于调试)
    
    Args:
        env_config: 环境配置。若为 None 则使用默认 EnvConfig()，
                    建议通过 TrainConfig.get_runtime_env_config() 获取以保持一致性。
    """
    if env_config is None:
        env_config = EnvConfig()
    
    # 采样动作
    rng, action_rng = jax.random.split(rng)
    raw_action, action, log_prob, value, new_lstm_state = model.apply(
        state.params, obs, lstm_state, None,
        method=model.sample_action,
        rngs={"noise": action_rng}
    )
    
    # 执行环境步骤
    rng, step_rng = jax.random.split(rng)
    new_env_state, timestep = step_env(
        step_rng, env_state, action, 
        env_params, env_config, state.curriculum_lambda
    )
    
    metrics = {
        "reward": jnp.mean(timestep.reward),
        "done": jnp.mean(timestep.done.astype(jnp.float32)),
    }
    
    return state, new_env_state, new_lstm_state, timestep.obs, metrics


# =============================================================================
# 入口点
# =============================================================================
def run_training(
    seed: int = 42,
    total_timesteps: int = 10_000_000,
    num_envs: int = 4096
) -> Tuple[TrainState, Dict]:
    """
    运行完整训练
    
    Args:
        seed: 随机种子
        total_timesteps: 总训练步数
        num_envs: 并行环境数
    
    Returns:
        final_state: 最终训练状态
        metrics: 训练指标历史
    """
    train_config = TrainConfig()
    train_config.seed = seed
    train_config.total_timesteps = total_timesteps
    train_config.num_envs = num_envs
    
    # 确保其它必要参数存在 (使用默认值)
    # TrainConfig should be initialized with defaults if not set.
    
    train_fn = make_train(train_config)
    # DO NOT JIT the outer loop! It contains IO and Python control flow.
    # train_fn_jit = jax.jit(train_fn) 
    
    rng = jax.random.PRNGKey(seed)
    
    print(f"Starting training with {num_envs} envs, {total_timesteps} steps...")
    start_time = time.time()
    
    final_state, metrics = train_fn(rng)
    
    # 等待完成
    if hasattr(final_state, 'params'):
        jax.block_until_ready(final_state.params)
    
    elapsed = time.time() - start_time
    sps = total_timesteps / elapsed
    
    print(f"Training completed in {elapsed:.1f}s ({sps:,.0f} SPS)")
    
    return final_state, metrics
