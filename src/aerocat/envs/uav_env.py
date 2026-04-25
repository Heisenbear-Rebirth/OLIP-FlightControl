"""
AeroCat v18.0 Main UAV Environment

完整实现 Neuro-Cascaded Adaptive Architecture 的训练环境:
- 50Hz RL 控制循环
- 500Hz L3 内环 (10x substeps via jax.lax.scan)
- Dual-Track L1 制导层
- PF-GDR 领域随机化
- APDC 平滑课程学习
- 稀疏脉冲碰撞注入

所有代码为纯 JAX，100% GPU 执行，支持 jit/vmap。
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Bool, PRNGKeyArray
from typing import Tuple, Dict, Any, NamedTuple, Optional
import chex
from flax import struct

from ..core.state import (
    EnvState, EnvParams, PhysState, L1State, L3State, DelayBuffer,
    PhysParams, L1Config, L3Config, compute_mixer_matrix
)
from ..control.l1_guidance import (
    l1_process, generate_virtual_sticks, euler_to_quaternion, quaternion_to_yaw
)
from ..control.l3_controller import (
    l3_substep, map_action_to_rates_and_thrust, compute_diagnostics
)
# v19.0: OLIP 已移除，不再导入
from .reward import compute_reward_v18, check_termination, RewardConfig

# 导入优化的物理引擎
from ..physics.dynamics import (
    physics_dynamics_step as physics_step_optimized,
    compute_motor_power_current,
    motor_dynamics_step,
    physics_dynamics_step,
)
from ..physics.battery import compute_ocv
from ..physics.turbulence import compute_wind_field


# =============================================================================
# 环境配置
# =============================================================================
@struct.dataclass
class EnvConfig:
    """环境配置参数"""
    
    # 时间配置
    dt_rl: float = 0.02           # RL 步长 (50Hz)
    dt_sub: float = 0.002         # 内环步长 (500Hz)
    num_substeps: int = 10        # 内环子步数
    max_episode_time: float = 10.0 # 最大 episode 时间 (s), 延长以支持 mid-episode 故障注入
    
    # 观测空间
    obs_dim: int = 31             # 31 维观测 (v19.0: 移除 OLIP 11D, 新增 L3 诊断 5D)
    action_dim: int = 4           # 4 维动作
    
    # 延迟配置
    buffer_size: int = 25         # 环形缓冲区大小 (50ms @ 500Hz)
    
    # 课程学习配置
    curriculum_lambda: float = 0.0  # 课程进度 [0, 1]
    collision_prob_max: float = 0.2 # 最大碰撞概率 (次/秒)
    collision_mag_vel_max: float = 3.0  # 最大速度冲击 (m/s)
    collision_mag_rate_max: float = 15.0  # 最大角速度冲击 (rad/s)
    
    # 初始状态配置
    init_height_range: Tuple[float, float] = (0.5, 5.0)
    init_max_tilt: float = 3.14        # 最大初始倾角 (rad)
    init_max_rate: float = 15.0        # 最大初始角速度 (rad/s)
    
    # 奖励配置
    reward_config: RewardConfig = struct.field(default_factory=RewardConfig)


# =============================================================================
# 时间步结果
# =============================================================================
class TimeStep(NamedTuple):
    """环境步进返回值"""
    obs: Float[Array, "batch 37"]
    reward: Float[Array, "batch"]
    done: Bool[Array, "batch"]
    truncated: Bool[Array, "batch"]
    info: Dict[str, Any]


# =============================================================================
# 物理动力学 (问题 #6: 移除简化版，使用 physics/ 模块的完整实现)
# =============================================================================
# 注意: 实际环境使用 physics_step_optimized() 来自 physics.dynamics 模块
# 完整物理实现位于: src/aerocat/physics/dynamics.py
# 
# physics_step_optimized() 包含:
# - 电机动力学 (非对称一阶滞后)
# - 推力/力矩计算 (含电机倾斜角)
# - 刚体 Newton-Euler 动力学
# - 二次阻力模型 (平移 + 旋转)
# - 四元数积分 (一阶欧拉)
# - 风场扰动
# - 传感器噪声注入
#
# 以下仅保留四元数乘法辅助函数 (用于延迟缓冲区等)


def quaternion_multiply(q1: Float[Array, "batch 4"], q2: Float[Array, "batch 4"]) -> Float[Array, "batch 4"]:
    """
    四元数乘法 Hamilton 约定 [w, x, y, z]
    
    用途: 四元数更新、姿态误差计算等
    公式: q1 ⊗ q2
    """
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    return jnp.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


# =============================================================================
# 延迟缓冲区操作
# =============================================================================
def update_delay_buffer(
    buffer: DelayBuffer,
    gyro: Float[Array, "batch 3"],
    accel: Float[Array, "batch 3"],
    action: Float[Array, "batch 4"]
) -> DelayBuffer:
    """更新延迟缓冲区 (环形写入)"""
    batch_size = gyro.shape[0]
    
    # 新的写入索引
    new_idx = (buffer.write_idx + 1) % 25
    
    # 更新缓冲区 (使用 scatter)
    # 为每个 batch 写入对应位置
    batch_indices = jnp.arange(batch_size)
    
    # 更新 gyro buffer
    new_gyro_buffer = buffer.imu_gyro_buffer.at[batch_indices, new_idx].set(gyro)
    new_accel_buffer = buffer.imu_accel_buffer.at[batch_indices, new_idx].set(accel)
    new_action_buffer = buffer.action_buffer.at[batch_indices, new_idx].set(action)
    
    return buffer.replace(
        imu_gyro_buffer=new_gyro_buffer,
        imu_accel_buffer=new_accel_buffer,
        action_buffer=new_action_buffer,
        write_idx=new_idx,
    )


def read_delayed_sensor(
    buffer: DelayBuffer
) -> Tuple[Float[Array, "batch 3"], Float[Array, "batch 3"], Float[Array, "batch 4"]]:
    """读取延迟后的传感器数据"""
    batch_size = buffer.write_idx.shape[0]
    batch_indices = jnp.arange(batch_size)
    
    # 计算读取索引
    read_idx = (buffer.write_idx - buffer.imu_delay_steps) % 25
    action_read_idx = (buffer.write_idx - buffer.action_delay_steps) % 25
    
    # 读取数据
    delayed_gyro = buffer.imu_gyro_buffer[batch_indices, read_idx]
    delayed_accel = buffer.imu_accel_buffer[batch_indices, read_idx]
    delayed_action = buffer.action_buffer[batch_indices, action_read_idx]
    
    return delayed_gyro, delayed_accel, delayed_action


# =============================================================================
# 观测空间构建 (37 维, v18.3)
# =============================================================================
def build_observation(
    phys_state: PhysState,
    gyro: Float[Array, "batch 3"],
    accel: Float[Array, "batch 3"],
    l1_state: L1State,
    l3_state: L3State,

    prev_action: Float[Array, "batch 4"],
    episode_time: Float[Array, "batch"],
    l3_config: L3Config,
    sticks: Float[Array, "batch 4"] = None,
    v_cmd_heading: Float[Array, "batch 3"] = None,
    max_time: float = 10.0
) -> Float[Array, "batch 31"]:
    """
    构建 31 维观测向量 (v19.0)
    
    v19.0 变更:
    - 移除 OLIP 11D（不再是研究范围，降级为消融实验对比）
    - 新增 L3 诊断 5D（tracking_rms, integral_rate, motor_spread）
    - 净变化: -11 + 5 = -6D (37→31)
    
    组成:
    - [0:4]   q_curr            (4)  ← 本体状态: 姿态四元数
    - [4:7]   v_actual_heading  (3)  ← 本体状态: 实际速度 (Heading Frame)
    - [7:10]  omega_norm        (3)  ← 本体状态: 角速度
    - [10:13] accel_norm        (3)  ← 本体状态: 加速度
    - [13:22] L3 diagnostics    (9)  ← 本体状态: L3 诊断 (v19.0: 4D→9D)
    - [22:25] v_cmd_heading     (3)  ← 飞手指令: 目标速度
    - [25:26] yaw_rate_cmd      (1)  ← 飞手指令: 偏航角速度命令
    - [26:30] prev_action       (4)  ← 附加信息
    - [30]    base_throttle     (1)  ← 附加信息: 悬停油门前馈
    ----------------------------------------
    总计 31 维
    """
    batch_size = phys_state.quaternion.shape[0]
    
    # 1. 当前机体姿态四元数 (Absolute)
    q_curr = phys_state.quaternion  # [batch, 4]
    
    # 2. 实际速度 (Heading Frame 下投影)
    # NED → Heading Frame: v_heading = R_z(ψ)^T · v_NED
    #   v_fwd  = cos(ψ)·v_N + sin(ψ)·v_E
    #   v_right = -sin(ψ)·v_N + cos(ψ)·v_E
    # 验证: ψ=90°(朝东), v_E=1 → v_fwd = sin(90°)=1 ✓
    # 修复(v18.3.1): 原代码用 sin(-ψ)=-sin(ψ) 符号错误
    v_world = phys_state.velocity  # [batch, 3], NED
    yaw_curr = quaternion_to_yaw(q_curr)  # [batch], ψ
    cos_yaw = jnp.cos(yaw_curr)   # cos(ψ)
    sin_yaw = jnp.sin(yaw_curr)   # sin(ψ)  [原为 sin(-ψ)=-sin(ψ), 已修复]
    v_actual_heading = jnp.stack([
        cos_yaw * v_world[..., 0] + sin_yaw * v_world[..., 1],   # 前向: cos(ψ)vN + sin(ψ)vE
        -sin_yaw * v_world[..., 0] + cos_yaw * v_world[..., 1],  # 右向: -sin(ψ)vN + cos(ψ)vE
        v_world[..., 2]                                           # Down(Z)
    ], axis=-1)
    # 归一化: max_speed=15.0 m/s
    v_actual_heading_norm = v_actual_heading / 15.0  # [batch, 3]
    
    # 3. 角速度 (归一化)
    # Tech Spec 02: 归一化因子 35.0 rad/s (~2000 deg/s)
    # v18.3.1: obs[7:8] 保留 body-frame roll/pitch rate (PID 控制直接需要)
    #          obs[9] 替换为 heading-frame 航向角速率 ψ̇ (与 yaw_rate_cmd 同坐标系)
    # 计算: ψ̇ = R(q)[2,:] · ω_body (旋转矩阵第三行与 body 角速度的点积)
    # 无奇点，纯 JAX 运算
    _GYRO_SCALE = 35.0
    omega_rp_norm = gyro[..., :2] / _GYRO_SCALE  # body roll/pitch rate [batch, 2]
    
    qw, qx, qy, qz = q_curr[..., 0], q_curr[..., 1], q_curr[..., 2], q_curr[..., 3]
    yaw_rate_world = (
        2.0 * (qx * qz - qw * qy) * gyro[..., 0] +
        2.0 * (qy * qz + qw * qx) * gyro[..., 1] +
        (1.0 - 2.0 * (qx ** 2 + qy ** 2)) * gyro[..., 2]
    )  # heading-frame 航向角速率 ψ̇ (rad/s)
    yaw_rate_actual_norm = yaw_rate_world / _GYRO_SCALE  # [batch]
    
    omega_norm = jnp.concatenate([
        omega_rp_norm,                          # [batch, 2] body p, q
        yaw_rate_actual_norm[..., None],         # [batch, 1] heading ψ̇
    ], axis=-1)  # [batch, 3]
    
    # 4. 加速度计 (归一化)
    # Tech Spec 02: 归一化因子 5G = 5 * 9.81 = 49.05 m/s^2
    # 注意: imu_accel 存储 IMU 比力 (Specific Force = 合力/质量)，Body Frame FRD
    # 悬停时 ≈ [0, 0, +g_body] (FRD: 推力向上=-Z方向, 比力≈+g沿Body-Z)
    accel_norm = accel / 49.05
    
    # 5. L3 诊断 (v19.0: 4D → 9D)
    l3_diag = compute_diagnostics(l3_state, phys_state)  # [batch, 9]
    # [0:3] PID 积分项归一化: 归一化到 [-1, 1] (i_limit 限幅保证)
    l3_diag = l3_diag.at[..., 0:3].set(l3_diag[..., 0:3] / l3_config.i_limit[..., None])
    # [3] 混控器饱和度: 已在 compute_diagnostics 中 clip
    l3_diag = l3_diag.at[..., 3].set(jnp.clip(l3_diag[..., 3], 0.0, 1.0))
    # [4:9] tracking_rms, integral_rate, motor_spread 已在 compute_diagnostics 归一化
    
    # 6. 速度命令 (Heading Frame)
    # 归一化: max_speed=15.0 m/s (与 v_actual_heading 保持一致)
    if v_cmd_heading is None:
        v_cmd_heading_obs = jnp.zeros((batch_size, 3))
    else:
        v_cmd_heading_obs = v_cmd_heading / 15.0  # [batch, 3]
    
    # 7. 偏航角速度命令 (yaw_rate_cmd)
    # v18.3.1: 将摇杆值转换为物理航向角速率 (rad/s)，再用与 omega/ψ̇ 相同的
    # 归一化因子 (/35.0) 处理。obs[9] (ψ̇_actual) 与 obs[20] (ψ̇_cmd) 现在
    # 均为 heading-frame 航向角速率，量纲、尺度、坐标系完全统一。
    _MAX_YAW_RATE = 100.0 * jnp.pi / 180.0  # 1.745 rad/s (v18.4: 200→100°/s, 与 L1Config.max_yaw_rate 一致)
    if sticks is None:
        yaw_rate_cmd = jnp.zeros((batch_size, 1))
    else:
        yaw_rate_cmd = sticks[..., 2:3] * _MAX_YAW_RATE / _GYRO_SCALE  # [batch, 1]
    
    # 8. (v19.0: OLIP 已彻底移除)
    
    # 9. 上一帧动作
    # prev_action: [batch, 4]
    
    # 10. 悬停油门前馈 (base_throttle)
    base_throttle = l1_state.base_throttle[..., None]  # [batch, 1]
    
    # 组装 31D 观测 (v19.0: 移除 OLIP 11D, 新增 L3 诊断 5D)
    obs = jnp.concatenate([
        q_curr,                # [0:4]   姿态四元数
        v_actual_heading_norm, # [4:7]   实际速度 (Heading Frame)
        omega_norm,            # [7:10]  角速度
        accel_norm,            # [10:13] 加速度
        l3_diag,               # [13:22] L3 诊断 (9D)
        v_cmd_heading_obs,     # [22:25] 速度命令 (Heading Frame)
        yaw_rate_cmd,          # [25:26] 偏航角速度命令
        prev_action,           # [26:30] 上一帧动作
        base_throttle,         # [30:31] 悬停油门前馈
    ], axis=-1)
    
    # 安全防护: 防止 NaN/Inf 传入网络导致梯度崩溃
    obs = jnp.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)
    obs = jnp.clip(obs, -5.0, 5.0)
    
    return obs


def quaternion_conjugate(q: Float[Array, "batch 4"]) -> Float[Array, "batch 4"]:
    """四元数共轭"""
    return jnp.concatenate([q[..., 0:1], -q[..., 1:4]], axis=-1)


# =============================================================================
# 稀疏脉冲碰撞注入 (Sparse Impulse Injection)
# =============================================================================
def apply_impulse_injection(
    phys_state: PhysState,
    key: PRNGKeyArray,
    episode_time: Float[Array, "batch"],
    config: EnvConfig,
    curriculum_lambda: float
) -> PhysState:
    """
    稀疏脉冲碰撞注入
    
    参考: 04_训练环境_仿真构建指南_v18.0.md 第2.3.5.1节
    """
    batch_size = phys_state.position.shape[0]
    k1, k2, k3 = jax.random.split(key, 3)
    
    # 1. 计算碰撞概率 (每步)
    prob_per_second = config.collision_prob_max * jnp.maximum(curriculum_lambda - 0.3, 0.0) / 0.7
    prob_per_step = prob_per_second * config.dt_rl
    
    # 2. 伯努利判定 (前 1s 不触发)
    trigger = jax.random.bernoulli(k1, prob_per_step, shape=(batch_size,))
    is_collision = trigger & (episode_time > 1.0)
    
    # 3. 计算冲击强度 (随课程难度)
    # 使用 state 中的有效难度 (Experience Replay)
    # scale_vel = config.collision_mag_vel_max * jnp.maximum(curriculum_lambda - 0.3, 0.0) / 0.7
    # Note: calculate locally to avoid signature change of apply_impulse_injection if unnecessary, 
    # but here we passed curriculum_lambda. We should pass state.effective_lambda from step_env.
    scale_vel = config.collision_mag_vel_max * jnp.maximum(curriculum_lambda - 0.3, 0.0) / 0.7
    scale_rate = config.collision_mag_rate_max * jnp.maximum(curriculum_lambda - 0.3, 0.0) / 0.7
    
    # 4. 随机冲击向量
    delta_vel = jax.random.uniform(k2, (batch_size, 3), minval=-1.0, maxval=1.0) * scale_vel[..., None]
    delta_rate = jax.random.uniform(k3, (batch_size, 3), minval=-1.0, maxval=1.0) * scale_rate[..., None]
    
    # 5. 应用冲击
    collision_mask = is_collision[..., None].astype(jnp.float32)
    new_velocity = phys_state.velocity + delta_vel * collision_mask
    new_omega = phys_state.angular_velocity + delta_rate * collision_mask
    
    return phys_state.replace(
        velocity=new_velocity,
        angular_velocity=new_omega
    )


# =============================================================================
# 环境重置 (reset_env)
# =============================================================================
def reset_env(
    key: PRNGKeyArray,
    batch_size: int,
    config: EnvConfig,
    curriculum_lambda: float = 0.0,
    params: Optional[EnvParams] = None
) -> Tuple[EnvState, EnvParams, Float[Array, "batch 37"]]:
    """
    环境重置
    
    参考: 04_训练环境_仿真构建指南_v18.0.md 第2.2节
    
    Args:
        key: JAX 随机密钥
        batch_size: 批量大小
        config: 环境配置
        curriculum_lambda: 课程进度 [0, 1]
        params: (Optional) 预先生成的环境参数 (PF-GDR)
    
    Returns:
        state: 初始环境状态
        params: 环境参数 (如果传入则返回更新后的副本)
        obs: 初始观测
    """
    keys = jax.random.split(key, 15)
    
    # 1. 确定基础参数
    if params is None:
        # 如果未提供，生成新参数
        # 注意: 这里使用 curriculum_lambda 来初始化难度
        from ..generators.param_generator import init_params
        params = init_params(keys[0], batch_size, None, curriculum_lambda)
        phys_params = params.phys_params
        l1_config = params.l1_config
        l3_config = params.l3_config
        mixer_matrix = params.mixer_matrix
    else:
        # 使用传入的 PF-GDR 参数
        phys_params = params.phys_params
        l1_config = params.l1_config
        l3_config = params.l3_config
        mixer_matrix = params.mixer_matrix
    
    # 1.5. 经验回放 (Experience Replay) 采样
    # P(replay) = 0.2
    # lambda_eff = where(replay, Uniform(0, lambda), lambda)
    
    # 确保 curriculum_lambda 广播到 batch
    curr_lambda_batch = jnp.broadcast_to(curriculum_lambda, (batch_size,))
    
    should_replay = jax.random.bernoulli(keys[11], p=0.2, shape=(batch_size,))
    replay_lambda = jax.random.uniform(keys[10], shape=(batch_size,), minval=0.0, maxval=curr_lambda_batch + 1e-6)
    
    # 有效难度
    effective_lambda = jnp.where(should_replay, replay_lambda, curr_lambda_batch)
    
    # 2. Turbulence Curriculum Update
    t_min, t_max = config.turbulence_intensity_range if hasattr(config, 'turbulence_intensity_range') else (0.05, 0.25)
    t_max_scaled = t_min + (t_max - t_min) * effective_lambda
    new_turbulence_intensity = jax.random.uniform(keys[14], (batch_size,), minval=t_min, maxval=t_max_scaled)
    
    phys_params = phys_params.replace(
        turbulence_intensity=new_turbulence_intensity,
    )
    
    # 3. 初始化物理状态
    # 位置: 随机高度
    height_min, height_max = config.init_height_range
    height = jax.random.uniform(keys[2], (batch_size,), minval=height_min, maxval=height_max)
    position = jnp.zeros((batch_size, 3))
    position = position.at[:, 2].set(-height)  # NED: 高度为负 Z
    
    # 姿态: 随机倾角 (基于有效难度)
    max_tilt = config.init_max_tilt * effective_lambda
    roll = jax.random.uniform(keys[3], (batch_size,), minval=-max_tilt, maxval=max_tilt)
    pitch = jax.random.uniform(keys[4], (batch_size,), minval=-max_tilt, maxval=max_tilt)
    yaw = jax.random.uniform(keys[5], (batch_size,), minval=-jnp.pi, maxval=jnp.pi)
    quaternion = euler_to_quaternion(roll, pitch, yaw)
    
    # 角速度: 随机 (基于有效难度)
    max_rate = config.init_max_rate * effective_lambda
    angular_velocity = jax.random.uniform(keys[6], (batch_size, 3), 
                                          minval=-max_rate[..., None], maxval=max_rate[..., None])
    # 计算精确悬停油门 (用于初始化)
    # 
    # v18.0.2 更新: 由于添加了推力线性化 (sqrt 补偿)
    # 混控器输出: th_linear
    # 经过 sqrt 后: th_actual = sqrt(th_linear)
    # 推力: F = th_actual² × F_max = th_linear × F_max
    # 
    # 悬停条件: 4 × F_hover = mg => th_linear = mg / (4 × F_max)
    # 
    # 注意: 不再需要 sqrt
    hover_throttle = (phys_params.mass * 9.81) / (4.0 * phys_params.motor_max_thrust)
    hover_throttle = jnp.clip(hover_throttle, 0.1, 0.9)
    
    # 初始 motor_throttle 需要是 sqrt 后的实际值
    # 因为 phys_state.motor_throttle 存储的是实际油门（经过 sqrt 后的值）
    init_motor_throttle_actual = jnp.sqrt(hover_throttle[:, None])
    init_motor_throttle = jnp.repeat(init_motor_throttle_actual, 4, axis=1)

    phys_state = PhysState(
        position=position,
        velocity=jnp.zeros((batch_size, 3)),
        quaternion=quaternion,
        angular_velocity=angular_velocity,
        motor_throttle=init_motor_throttle,  # 动态悬停油门
        battery_soc=jnp.ones((batch_size,)),
        battery_voltage=compute_ocv(jnp.ones((batch_size,)), phys_params.battery_ocv_coeff),
        battery_V1=jnp.zeros((batch_size,)),
        battery_V2=jnp.zeros((batch_size,)),
        imu_gyro=jnp.zeros((batch_size, 3)),
        imu_accel=jnp.tile(jnp.array([0.0, 0.0, 9.81]), (batch_size, 1)),
        wind_velocity=jnp.zeros((batch_size, 3)),
        turbulence_state=jnp.zeros((batch_size, 3)),
    )
    
    # 4. 初始化 L1 状态
    l1_state = L1State(
        target_yaw=yaw,  # 锁定初始航向
        headless_ref_yaw=yaw,
        vz_integral=jnp.zeros((batch_size,)),
        vz_error_prev=jnp.zeros((batch_size,)),
        mode=jnp.zeros((batch_size,), dtype=jnp.int32),  # 初始为 Headed，故障触发后在 step_env 中切换
        virtual_stick_state=jnp.zeros((batch_size, 4)),
        velocity_error_heading=jnp.zeros((batch_size, 3)),  # v18.2: 初始速度误差为零
        prev_v_cmd_heading=jnp.zeros((batch_size, 3)),     # Slew Rate Limiter 初始状态
        prev_yaw_rate_cmd=jnp.zeros((batch_size,)),        # v18.4: Yaw Slew Rate Limiter 初始状态
        base_throttle=hover_throttle, # 动态悬停油门
        target_velocity_body=jnp.zeros((batch_size, 3)),
    )
    
    # 5. 初始化 L3 状态
    l3_state = L3State.create_default(batch_size)
    
    # 6. (v19.0: OLIP 已移除)
    
    # 7. 初始化延迟缓冲区
    delay_buffer = DelayBuffer.create_default(batch_size)
    # 随机延迟步数 (5-50ms @ 500Hz = 2-25 steps)
    imu_delay_steps = jax.random.randint(keys[8], (batch_size,), 2, 10)
    action_delay_steps = jax.random.randint(keys[9], (batch_size,), 1, 5)
    delay_buffer = delay_buffer.replace(
        imu_delay_steps=imu_delay_steps,
        action_delay_steps=action_delay_steps
    )
    
    # 计算初始总倾角
    # tilt approx sqrt(roll^2 + pitch^2) for small angles, but exact is ACos(Z_body_z)
    # quaternion is euler_to_quaternion(roll, pitch, yaw)
    # Z-axis of body in world frame from euler:
    # z_body_z = cos(roll) * cos(pitch)
    # tilt = acos(z_body_z)
    # But wait, roll/pitch are generated directly.
    # cos(tilt) = cos(R) * cos(P)
    cos_tilt_init = jnp.cos(roll) * jnp.cos(pitch)
    init_tilt_angle = jnp.arccos(jnp.clip(cos_tilt_init, -1.0, 1.0))
    
    # 动态设定终止阈值
    # 如果初始倾角 > 60度 (1.047 rad)，则允许完全翻转 (3.14)
    # 否则限制为 90度 (1.57 rad)
    # 注意: 即使设为 3.14, 撞地仍然会终止
    termination_threshold = jnp.where(
        init_tilt_angle > 1.047, 
        jnp.full((batch_size,), 3.14), 
        jnp.full((batch_size,), 1.57) # Standard 90 deg limit
    )
    
    # 8. 组装环境状态
    state = EnvState(
        phys_state=phys_state,
        l1_state=l1_state,
        l3_state=l3_state,
        obs_buffer=delay_buffer,
        step_count=jnp.zeros((batch_size,), dtype=jnp.int32),
        episode_time=jnp.zeros((batch_size,)),
        prev_action=jnp.zeros((batch_size, 4)),
        effective_lambda=effective_lambda,
        termination_tilt_threshold=termination_threshold,
        # v18.1: 参考轨迹初始化为飞机出生位置 (spawn pos)
        ref_position=position,  # NED, shape [batch, 3]，与 phys_state.position 相同
    )
    
    # 9. 环境参数 (返回)
    # 必须从新的 phys_params 重构
    if params is None:
        params_out = EnvParams(
            phys_params=phys_params,
            l1_config=l1_config,
            l3_config=l3_config,
            mixer_matrix=mixer_matrix,
        )
    else:
        params_out = params.replace(phys_params=phys_params)
    
    # 10. 初始观测
    # Reset 时 Buffer 为空/零，直接使用当前传感器数据 (Zero Delay implies "perfect" at t=0 or just latency=0)
    # 或者读取初始 Buffer (其中全是 0)。
    # 物理上 t=0 时应该读数为 0 还是当前状态?
    # 通常 IMU 初始化即有读数 (Gravity 9.81).
    # 为了一致性，我们手动传入初始 IMU 数据。
    obs = build_observation(
        phys_state, 
        phys_state.imu_gyro,  # Initial: No delay
        phys_state.imu_accel, # Initial: No delay
        l1_state, l3_state,
        state.prev_action, state.episode_time, 
        params_out.l3_config,
        sticks=None,          # v18.3: reset 时无摇杆信号
        v_cmd_heading=None,   # v18.3: reset 时无速度命令
    )
    
    return state, params_out, obs


# =============================================================================
# 环境步进 (step_env)
# =============================================================================
def step_env(
    key: PRNGKeyArray,
    state: EnvState,
    action: Float[Array, "batch 4"],
    params: EnvParams,
    config: EnvConfig,
    curriculum_lambda: float = 0.0
) -> Tuple[EnvState, TimeStep, EnvParams]:
    """
    环境单步更新 (50Hz)
    
    返回:
        state: 新状态
        timestep: 时间步数据
        params: 更新后的环境参数 (用于处理 auto-reset 时的参数重置)
    """
    keys = jax.random.split(key, 5)
    batch_size = action.shape[0]
    
    # =====================================================================
    # 1. 虚拟摇杆生成 (AR(1))
    # =====================================================================
    sticks, l1_state_sticks = generate_virtual_sticks(
        state.l1_state, keys[0], params.l1_config
    )
    
    # -----------------------------------------------------------------
    # Curriculum Scaling: Dampen sticks at low lambda (Level 0 = Hover)
    # -----------------------------------------------------------------
    # 摇杆幅度: Lambda=0 时保留 30% 的命令范围 (足以产生 ~1m/s 的速度命令)
    # Lambda=1 时为完整范围。防止低 Lambda 下摇杆被衰减到接近零导致模型学不到追踪行为。
    scale_factor = jnp.clip(curriculum_lambda, 0.0, 1.0) * 0.7 + 0.3
    
    # Roll/Pitch/Yaw (Neutral = 0.0)
    sticks_rpy = sticks[..., :3] * scale_factor
    
    # Throttle (Neutral = 0.5)
    sticks_thr = (sticks[..., 3] - 0.5) * scale_factor + 0.5
    
    # Reassemble (Action: Need to use .at[...].set or concatenate)
    sticks = jnp.concatenate([sticks_rpy, sticks_thr[..., None]], axis=-1)
    
    # Also force Level 0 to be strictly Headed Mode (Mode 0)
    # Because Headless mode logic might be confusing at start? 
    # Actually L1 state handles mode. But scale_factor=0 ensures near-hover commands.
    
    # =====================================================================
    # 2. L1 层处理 (v18.2: 坐标变换层，不再做 PID)
    # =====================================================================
    # 课程化 Slew Rate: max_v_cmd_rate 随 λ 线性增长
    # λ=0 → 0.4 m/s² (平缓悬停), λ=1 → 2.0 m/s² (全速机动)
    effective_v_cmd_rate = (
        params.l1_config.max_v_cmd_rate +
        (params.l1_config.max_v_cmd_rate_full - params.l1_config.max_v_cmd_rate)
        * jnp.clip(curriculum_lambda, 0.0, 1.0)
    )
    l1_config_step = params.l1_config.replace(max_v_cmd_rate=effective_v_cmd_rate)
    
    v_cmd_heading, v_error_heading, t_hover, l1_state_new = l1_process(
        sticks, state.phys_state, l1_state_sticks, l1_config_step, config.dt_rl
    )
    
    # =====================================================================
    # 3. 动作映射 (RL 输出 -> 物理指令)
    # =====================================================================
    target_rates, thrust = map_action_to_rates_and_thrust(action)
    
    # 问题 #5 修复: RL 完全控制油门
    # 原实现: thrust = thrust * 0.5 + t_base * 0.5 (50/50 混合)
    # 新实现: RL 完全控制，t_base 仅作为观测中的参考信息
    # 注意: t_base (悬停油门估计) 已包含在观测向量中 (obs[4])
    # RL 网络可以学习利用这个信息来输出合适的油门
    # thrust 取值 [0, 1]，由 map_action_to_rates_and_thrust 从 action[3] 映射
    # 这允许 RL 在研究目的下完全探索油门控制空间
    
    
    # =====================================================================
    # 5. L3 内环 + 物理仿真 (jax.lax.scan, 10x 500Hz)
    # =====================================================================
    # 为每一步物理生成随机密钥
    loop_keys = jax.random.split(keys[3], config.num_substeps)

    def inner_loop_body(carry, loop_key):
        """内环单步 (500Hz)"""
        phys_state, l3_state = carry
        
        # L3 子步
        phys_state, l3_state, motor_cmds = l3_substep(
            phys_state, l3_state,
            target_rates, thrust,
            params.phys_params, params.l3_config, params.mixer_matrix,
            config.dt_sub
        )
        
        # 物理动力学
        phys_state = physics_step_optimized(
            phys_state, motor_cmds, params.phys_params, loop_key, config.dt_sub
        )
        
        return (phys_state, l3_state), None
    
    init_carry = (state.phys_state, state.l3_state)
    (phys_state_new, l3_state_new), _ = jax.lax.scan(
        inner_loop_body,
        init_carry,
        xs=loop_keys,
        length=config.num_substeps
    )
    
    # =====================================================================
    # 5. 稀疏碰撞注入
    # =====================================================================
    episode_time_new = state.episode_time + config.dt_rl
    phys_state_new = apply_impulse_injection(
        phys_state_new, keys[1], episode_time_new, config, state.effective_lambda
    )
    
    # =====================================================================
    # 5.5 风场更新 (50Hz)
    # =====================================================================
    # 提取阵风参数
    gust_params = {
        'active': params.phys_params.gust_active,
        'start_time': params.phys_params.gust_start_time,
        'duration': params.phys_params.gust_duration,
        'magnitude': params.phys_params.gust_magnitude,
        'direction': params.phys_params.gust_direction,
    }
    
    # 计算风场
    # roughness_length 已在 PhysParams 中
    roughness_length = params.phys_params.roughness_length
    
    wind_velocity, turbulence_state_new = compute_wind_field(
        phys_state_new.position,
        episode_time_new,
        params.phys_params.wind_speed_mean,
        params.phys_params.wind_direction,
        params.phys_params.turbulence_intensity,
        phys_state_new.turbulence_state,
        gust_params,
        roughness_length,
        keys[2]
    )
    
    phys_state_new = phys_state_new.replace(
        wind_velocity=wind_velocity,
        turbulence_state=turbulence_state_new
    )
    
    # =====================================================================
    # 6. 更新延迟缓冲区
    # =====================================================================
    obs_buffer_new = update_delay_buffer(
        state.obs_buffer,
        phys_state_new.imu_gyro,
        phys_state_new.imu_accel,
        action
    )
    
    # =====================================================================
    # 7. 组装新状态
    # =====================================================================
    # [DEPRECATED v18.5] ref_position 积分已禁用 (R_pos 已移除)
    # 部署时位置保持由外环 PID 处理，此处保持 ref_position 不变
    state_new = state.replace(
        phys_state=phys_state_new,
        l1_state=l1_state_new,
        l3_state=l3_state_new,
        obs_buffer=obs_buffer_new,
        step_count=state.step_count + 1,
        episode_time=episode_time_new,
        prev_action=action,
        effective_lambda=state.effective_lambda,
        ref_position=state.ref_position,  # 不再积分，保持不变
    )
    
    # =====================================================================
    # 8. 构建观测 (使用延迟数据)
    # =====================================================================
    delayed_gyro, delayed_accel, delayed_action = read_delayed_sensor(obs_buffer_new)
    
    obs = build_observation(
        state_new.phys_state, 
        delayed_gyro,
        delayed_accel,
        state_new.l1_state, 
        state_new.l3_state, 
        action,
        state_new.episode_time,
        params.l3_config,
        sticks=sticks,            # v18.3: 传入摇杆信号以获取 yaw_rate_cmd
        v_cmd_heading=v_cmd_heading,  # v18.3: 传入 Heading Frame 速度命令
    )
    
    # =====================================================================
    # 9. 计算奖励
    # =====================================================================
    # 获取 v_actual_heading 以便传入 reward (Heading Frame 下投影)
    # NED→Heading: v_fwd=cos(ψ)vN+sin(ψ)vE, v_right=-sin(ψ)vN+cos(ψ)vE
    v_world = phys_state_new.velocity  # [batch, 3], NED
    yaw_curr_for_rew = quaternion_to_yaw(phys_state_new.quaternion)
    cos_yaw_rew = jnp.cos(yaw_curr_for_rew)   # cos(ψ)
    sin_yaw_rew = jnp.sin(yaw_curr_for_rew)   # sin(ψ) [原为 -sin(ψ), 已修复]
    v_actual_heading = jnp.stack([
        cos_yaw_rew * v_world[..., 0] + sin_yaw_rew * v_world[..., 1],
        -sin_yaw_rew * v_world[..., 0] + cos_yaw_rew * v_world[..., 1],
        v_world[..., 2]
    ], axis=-1)

    # 9. 计算奖励
    reward, reward_info = compute_reward_v18(
        phys_state_new, l1_state_new, action, state.prev_action,
        ref_position=state_new.ref_position,
        v_actual_heading=v_actual_heading,
        v_cmd_heading=v_cmd_heading,
        config=config.reward_config
    )
    
    # =====================================================================
    # 10. 终止判断
    # =====================================================================
    done, truncated, term_info = check_termination(
        phys_state_new, episode_time_new, config.max_episode_time, 
        max_tilt_angle=state.termination_tilt_threshold # Use dynamic threshold
    )
    
    # 组装信息字典
    info = {
        **reward_info,
        **term_info,
        'episode_time': episode_time_new,
        'step_count': state_new.step_count,
    }
    
    timestep = TimeStep(
        obs=obs,
        reward=reward,
        done=done,
        truncated=truncated,
        info=info
    )
    
    # =====================================================================
    # 11. 自动重置 (Auto-Reset)
    # =====================================================================
    # 生成全量重置状态
    # 注意: 使用传入的 params 保持物理参数一致 (但故障会重新随机化)
    # 关键修复 (v18.0.5): reset_env 会生成 NEW params (新的故障、风场等)
    # 我们必须捕获并返回这些新参数，否则会出现 Ghost Parameter Bug
    reset_state, reset_params, reset_obs = reset_env(
        keys[2], 
        batch_size, 
        config, 
        curriculum_lambda, 
        params=params
    )
    
    # 根据 done 掩码选择状态
    def where_done(x, y):
        # x: new_state (continuing), y: reset_state
        # done: [batch], broadcast to matches x
        if hasattr(x, 'shape') and x.shape == (): # scalar array
             return jnp.where(done, y, x)
        
        # Leaf might be a float/int (not array) in some Pytrees? 
        # JAX trees are usually arrays.
        # Handle structural mismatch or scalar
        is_array = hasattr(x, 'ndim')
        if not is_array: return jnp.where(done, y, x) # Fallback for scalar
        
        ndim = x.ndim
        done_shape = done.shape + (1,) * (ndim - 1)
        done_expanded = jnp.reshape(done, done_shape)
        return jnp.where(done_expanded, y, x)

    # 混合状态: If done, use reset_state; else use state_new
    final_state = jax.tree_util.tree_map(where_done, state_new, reset_state)
    
    # 混合参数: If done, use reset_params; else use old params
    # 这是修复 Sim-to-Real 的核心
    final_params = jax.tree_util.tree_map(where_done, params, reset_params)
    
    # 如果发生重置，观测也需要更新为 reset_obs
    # PPO 通常使用 info['final_observation'] 存储结束前的观测用于价值计算
    # 但在这里我们直接返回混合后的观测给 Agent 下一步使用
    final_obs = jax.tree_util.tree_map(where_done, obs, reset_obs)
    
    # 更新 TimeStep 中的 obs
    timestep = timestep._replace(obs=final_obs)
    
    return final_state, timestep, final_params


# =============================================================================
# 高层接口
# =============================================================================
class UAVEnv:
    """
    AeroCat v18.0 UAV 环境封装类
    
    提供 Gymnax 风格的接口。
    """
    
    def __init__(self, config: EnvConfig = None):
        self.config = config or EnvConfig()
        self._reset_fn = jax.jit(lambda k, b, c: reset_env(k, b, self.config, c), static_argnums=(1,))
        self._step_fn = jax.jit(lambda k, s, a, p, c: step_env(k, s, a, p, self.config, c))
    
    def reset(
        self,
        key: PRNGKeyArray,
        batch_size: int,
        curriculum_lambda: float = 0.0
    ) -> Tuple[EnvState, EnvParams, Float[Array, "batch 37"]]:
        """重置环境"""
        return self._reset_fn(key, batch_size, curriculum_lambda)
    
    def step(
        self,
        key: PRNGKeyArray,
        state: EnvState,
        action: Float[Array, "batch 4"],
        params: EnvParams,
        curriculum_lambda: float = 0.0
    ) -> Tuple[EnvState, TimeStep, EnvParams]:
        """执行一步"""
        return self._step_fn(key, state, action, params, curriculum_lambda)
    
    @property
    def observation_space(self) -> int:
        return self.config.obs_dim
    
    @property
    def action_space(self) -> int:
        return self.config.action_dim


# =============================================================================
# 批量 vmap 包装器
# =============================================================================
def make_vmap_env(config: EnvConfig = None) -> Tuple:
    """
    创建 vmap 兼容的环境函数
    
    Returns:
        reset_fn: vmapped reset
        step_fn: vmapped step
    """
    cfg = config or EnvConfig()
    
    # 这些函数已经是批处理的，无需额外 vmap
    reset_fn = jax.jit(lambda k, b, c: reset_env(k, b, cfg, c))
    step_fn = jax.jit(lambda k, s, a, p, c: step_env(k, s, a, p, cfg, c))
    
    return reset_fn, step_fn
