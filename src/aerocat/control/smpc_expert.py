"""
AeroCat v18.0 SMPC Expert Controller (CEM-MPC) v3

核心变化: 放弃简化前向模型, 直接用 step_env 作为 MPC 的真实前向模型。
利用 JAX 的 vmap + lax.scan 实现批量 rollout, 在 GPU 上并行评估候选序列。

这确保了 MPC 预测与真实物理完全一致, 彻底消除 model mismatch 问题。

动作空间: CEM 在 RL action [-1,1]^4 空间直接优化 (不需物理空间转换)
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
from typing import Tuple
from functools import partial

from ..envs.uav_env import step_env, EnvConfig
from ..core.state import EnvState

# =============================================================================
# 常量
# =============================================================================
GRAVITY = 9.81

# CEM 默认参数
DEFAULT_HORIZON = 6           # 预测步数 (× 0.02s = 0.12s) — 短 horizon 减少计算量
DEFAULT_NUM_SAMPLES = 256     # 候选序列数 (256 够用, 因为模型完全准确)
DEFAULT_NUM_ELITES = 32       # 精英数
DEFAULT_NUM_ITERATIONS = 3    # CEM 迭代轮数

# =============================================================================
# 真实前向 Rollout (用 step_env)
# =============================================================================
def _make_rollout_fn(env_config: EnvConfig):
    """
    创建一个基于 step_env 的 rollout 函数

    返回的函数签名: rollout(state, params, action_seq, key, lam) -> rewards
    """

    def rollout_single(state, params, action_seq, key, lam):
        """
        用真实 step_env 展开一整条动作序列, 返回累计 reward

        Args:
            state:      EnvState (batch=1)
            params:     EnvParams
            action_seq: [H, 4] rl action 序列
            key:        PRNGKey
            lam:        curriculum lambda

        Returns:
            total_reward: scalar 累计 reward
        """
        def step_fn(carry, action_t):
            state, params, key = carry
            key, step_key = jax.random.split(key)
            action_batch = action_t[None, :]  # [1, 4]
            new_state, ts, new_params = step_env(
                step_key, state, action_batch, params, env_config, lam
            )
            reward = ts.reward[0]
            done = ts.done[0]
            # 如果已 done, 后续 reward 为 0
            return (new_state, new_params, key), (reward, done)

        init_carry = (state, params, key)
        _, (rewards, dones) = jax.lax.scan(step_fn, init_carry, action_seq)

        # 终止后 reward 清零 (用 alive mask)
        alive = jnp.cumprod(1.0 - dones.astype(jnp.float32))
        # 第一个 done 之后的 reward 都清零
        masked_rewards = rewards * jnp.concatenate([jnp.array([1.0]), alive[:-1]])

        # 折扣因子
        gamma = 0.97
        discount = gamma ** jnp.arange(len(rewards))
        return jnp.sum(masked_rewards * discount)

    return rollout_single


# =============================================================================
# CEM 优化器 (真实模型)
# =============================================================================
def create_cem_optimizer(env_config: EnvConfig):
    """
    创建使用真实 step_env 的 CEM 优化器

    返回一个 JIT 编译的优化函数
    """
    rollout_single = _make_rollout_fn(env_config)
    # vmap over samples: 每个 sample 用不同的 action_seq 和 rng
    rollout_batch = jax.vmap(
        rollout_single,
        in_axes=(None, None, 0, 0, None)  # state, params 共享; action_seq, key 逐样本
    )

    @partial(jax.jit, static_argnums=(5, 6, 7, 8))
    def cem_optimize(
        state, params,
        prev_action_seq,  # [H, 4] warm-start
        key,
        lam,              # curriculum lambda (scalar)
        horizon: int = DEFAULT_HORIZON,
        num_samples: int = DEFAULT_NUM_SAMPLES,
        num_elites: int = DEFAULT_NUM_ELITES,
        num_iterations: int = DEFAULT_NUM_ITERATIONS,
    ):
        """
        CEM 优化器 (真实环境 rollout)

        Args:
            state:           EnvState (batch=1)
            params:          EnvParams
            prev_action_seq: [H, 4] 上一步优化的 RL action 序列
            key:             PRNGKey
            lam:             curriculum lambda
            horizon:         预测步数
            num_samples:     采样数
            num_elites:      精英数
            num_iterations:  CEM 迭代数

        Returns:
            best_action:     [4] 当前步最优 RL action
            best_action_seq: [H, 4] 最优序列 (供下一步 warm-start)
        """
        # Warm-start: 右移
        warm_mean = jnp.concatenate([prev_action_seq[1:], prev_action_seq[-1:]], axis=0)

        # 初始标准差
        init_std = jnp.full((horizon, 4), 0.5)
        # 油门轴 std 更小
        init_std = init_std.at[:, 3].set(0.2)

        def cem_iteration(carry, _):
            mean, std, key = carry
            key, sample_key, eval_key = jax.random.split(key, 3)

            # 1. 采样
            noise = jax.random.normal(sample_key, (num_samples, horizon, 4))
            samples = mean[None, :, :] + noise * std[None, :, :]
            samples = jnp.clip(samples, -1.0, 1.0)

            # 2. 批量评估 (用真实 step_env!)
            eval_keys = jax.random.split(eval_key, num_samples)
            rewards = rollout_batch(state, params, samples, eval_keys, lam)

            # 3. 选择精英 (最高 reward)
            elite_indices = jnp.argsort(-rewards)[:num_elites]  # 负号: 取最大 reward
            elites = samples[elite_indices]

            # 4. 更新分布
            new_mean = jnp.mean(elites, axis=0)
            new_std = jnp.std(elites, axis=0) + 1e-4

            alpha = 0.7
            mean_updated = alpha * new_mean + (1 - alpha) * mean
            std_updated = alpha * new_std + (1 - alpha) * std
            std_updated = jnp.maximum(std_updated, 0.03)

            return (mean_updated, std_updated, key), rewards[elite_indices[0]]

        init_carry = (warm_mean, init_std, key)
        (final_mean, _, _), _ = jax.lax.scan(
            cem_iteration, init_carry, None, length=num_iterations
        )

        best_action = jnp.clip(final_mean[0], -1.0, 1.0)
        return best_action, final_mean

    return cem_optimize


# =============================================================================
# SMPC 专家状态
# =============================================================================
class SMPCExpertState:
    """SMPC 专家的持续状态 (跨步维护)"""

    def __init__(self, horizon=DEFAULT_HORIZON):
        self.prev_action_seq = jnp.zeros((horizon, 4))
        self.horizon = horizon
        # 兼容 v2 属性名
        self.prev_phys_action = jnp.zeros(4)
        self.prev_phys_action_seq = jnp.zeros((horizon, 4))

    def reset(self):
        self.prev_action_seq = jnp.zeros((self.horizon, 4))


# =============================================================================
# 高层接口
# =============================================================================
_cem_optimizer = None
_cem_env_config = None


def smpc_expert_step(
    phys_state,
    l1_state,
    env_params,
    expert_state: SMPCExpertState,
    key: PRNGKeyArray,
    horizon: int = DEFAULT_HORIZON,
    num_samples: int = DEFAULT_NUM_SAMPLES,
    num_elites: int = DEFAULT_NUM_ELITES,
    num_iterations: int = DEFAULT_NUM_ITERATIONS,
    env_config: EnvConfig = None,
    curriculum_lambda: float = 0.0,
    env_state: "EnvState" = None,
    env_params_full = None,
):
    """
    SMPC 专家单步 v3

    v3 变化: 需要传递完整的 env_state 和 env_params (而非仅 phys_state/l1_state)
    以便用 step_env 做真实 rollout.

    如果 env_state 为 None, 使用 phys_state/l1_state 的兼容模式 (功能受限)。
    """
    global _cem_optimizer, _cem_env_config

    if env_config is None:
        env_config = EnvConfig()

    # 懒初始化 CEM 优化器 (JIT 编译)
    if _cem_optimizer is None or _cem_env_config is not env_config:
        _cem_optimizer = create_cem_optimizer(env_config)
        _cem_env_config = env_config

    # 使用完整 state
    if env_state is None:
        raise ValueError("v3 SMPC requires env_state parameter for real-env rollout")

    if env_params_full is None:
        env_params_full = env_params

    # CEM 优化 (直接在 RL action 空间)
    best_action, best_seq = _cem_optimizer(
        env_state, env_params_full,
        expert_state.prev_action_seq,
        key,
        jnp.float32(curriculum_lambda),
        horizon, num_samples, num_elites, num_iterations,
    )

    expert_state.prev_action_seq = best_seq
    # v2 兼容
    expert_state.prev_phys_action_seq = best_seq
    expert_state.prev_phys_action = best_action

    rl_action = best_action[None, :]  # [1, 4]
    return rl_action, expert_state


# =============================================================================
# 工具函数
# =============================================================================
def create_default_action_seq(mass, motor_max_thrust, horizon=DEFAULT_HORIZON):
    """创建默认 (悬停) RL 动作序列: rates=0, thrust≈hover"""
    hover_throttle = jnp.clip(
        (mass * GRAVITY) / (4.0 * motor_max_thrust + 1e-6),
        0.05, 0.95
    )
    # RL action thrust = 2 * throttle - 1
    hover_thrust_action = 2.0 * hover_throttle - 1.0
    hover_action = jnp.array([0.0, 0.0, 0.0, hover_thrust_action])
    return jnp.tile(hover_action[None, :], (horizon, 1))
