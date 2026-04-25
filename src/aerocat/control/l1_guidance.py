"""
AeroCat v18.2 L1 Guidance Layer (坐标变换层)

v18.2 架构重构: L1 层从速度 PID 控制器简化为纯坐标变换层。

核心职责:
- Headed/Headless 双模式切换
- 虚拟摇杆 AR(1) 生成器
- 摇杆 → Heading Frame 速度命令映射
- 速度误差计算 (v_error = v_cmd - v_actual)
- 偏航角管理

v18.2 关键变更:
- 移除水平速度 PID (不再计算 q_target)
- 新增 compute_heading_frame_velocities() → 输出 3D 速度误差
- l1_process() 返回 (v_cmd_heading, v_error_heading, t_hover, l1_state_new)

所有函数为纯函数，使用 jnp.where 进行分支，支持 jit/vmap。
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
from typing import Tuple
import chex

from ..core.state import L1State, L1Config, PhysState


# =============================================================================
# 四元数工具函数
# =============================================================================
def euler_to_quaternion(
    roll: Float[Array, "batch"],
    pitch: Float[Array, "batch"],
    yaw: Float[Array, "batch"]
) -> Float[Array, "batch 4"]:
    """
    欧拉角转四元数 (ZYX 顺序)
    
    Args:
        roll: 横滚角 (rad)
        pitch: 俯仰角 (rad)
        yaw: 偏航角 (rad)
    
    Returns:
        quaternion: [w, x, y, z]
    """
    cr = jnp.cos(roll / 2)
    sr = jnp.sin(roll / 2)
    cp = jnp.cos(pitch / 2)
    sp = jnp.sin(pitch / 2)
    cy = jnp.cos(yaw / 2)
    sy = jnp.sin(yaw / 2)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return jnp.stack([w, x, y, z], axis=-1)


def quaternion_to_yaw(q: Float[Array, "batch 4"]) -> Float[Array, "batch"]:
    """
    从四元数提取偏航角
    
    Args:
        q: 四元数 [w, x, y, z]
    
    Returns:
        yaw: 偏航角 (rad)
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = jnp.arctan2(siny_cosp, cosy_cosp)
    
    return yaw


def quaternion_normalize(q: Float[Array, "batch 4"]) -> Float[Array, "batch 4"]:
    """归一化四元数"""
    norm = jnp.linalg.norm(q, axis=-1, keepdims=True)
    return q / (norm + 1e-8)


# =============================================================================
# 虚拟摇杆生成器 (OU 过程 + 泊松跳跃)
# =============================================================================
def generate_virtual_sticks(
    l1_state: L1State,
    key: jax.random.PRNGKey,
    l1_config: L1Config,
) -> Tuple[Float[Array, "batch 4"], L1State]:
    """
    生成虚拟摇杆信号 (Ornstein-Uhlenbeck 过程 + 泊松跳跃)
    
    模拟真实飞手的摇杆输入:
    - OU 过程产生平滑、均值回归的连续运动
    - 泊松跳跃模拟急刹/松杆 (瞬间归中)
    
    v18.4: Yaw 轴独立参数 (更平缓的航向控制)
    - Roll/Pitch/Throttle: ou_theta=1.5, ou_sigma=0.7, jump_rate=0.2
    - Yaw:                 ou_theta=0.5, ou_sigma=0.3, jump_rate=0.05
    
    OU 离散公式:
        x_{t+1} = x_t + θ(μ - x_t)·dt + σ·√dt·ε_t,  ε~N(0,1)
    
    泊松跳跃:
        以概率 λ_jump·dt 触发, x_{t+1} = μ (松杆归中)
    
    稳态分布: N(μ, σ²/(2θ))
    
    Args:
        l1_state: 当前 L1 状态
        key: JAX 随机密钥
        l1_config: L1 配置 (含 OU 参数 + yaw 独立参数)
    
    Returns:
        sticks: [batch, 4] 虚拟摇杆 [roll, pitch, yaw, throttle]
        l1_state_new: 更新后的 L1 状态
    """
    batch_size = l1_state.virtual_stick_state.shape[0]
    k_noise, k_jump = jax.random.split(key)
    
    x = l1_state.virtual_stick_state  # [batch, 4]
    dt = 0.02  # 50Hz RL 频率
    
    # 构建 per-axis OU 参数: [batch, 4]
    # Roll(0), Pitch(1), Yaw(2), Throttle(3)
    # Yaw 轴使用独立的更平缓参数, 其他三轴共用原参数
    mu = l1_config.ou_mu  # [batch] scalar, 所有轴共用均值 0
    
    theta_rpt = l1_config.ou_theta      # [batch] Roll/Pitch/Throttle
    theta_yaw = l1_config.ou_theta_yaw  # [batch] Yaw 独立
    theta = jnp.stack([theta_rpt, theta_rpt, theta_yaw, theta_rpt], axis=-1)  # [batch, 4]
    
    sigma_rpt = l1_config.ou_sigma
    sigma_yaw = l1_config.ou_sigma_yaw
    sigma = jnp.stack([sigma_rpt, sigma_rpt, sigma_yaw, sigma_rpt], axis=-1)  # [batch, 4]
    
    jump_rpt = l1_config.jump_rate
    jump_yaw = l1_config.jump_rate_yaw
    jump = jnp.stack([jump_rpt, jump_rpt, jump_yaw, jump_rpt], axis=-1)  # [batch, 4]
    
    # 1. OU 更新: dx = θ(μ-x)dt + σ√dt·ε
    noise = jax.random.normal(k_noise, shape=(batch_size, 4))
    x_new = x + theta * (mu[..., None] - x) * dt + sigma * jnp.sqrt(dt) * noise
    
    # 2. 泊松跳跃 (急刹/松杆归中)
    # 每轴独立触发: 概率 λ·dt / 轴 / 步
    jump_prob = jump * dt  # [batch, 4]
    jump_mask = jax.random.bernoulli(k_jump, jump_prob, shape=(batch_size, 4))
    x_new = jnp.where(jump_mask, mu[..., None], x_new)
    
    # 3. 边界截断 [-1, 1]
    x_new = jnp.clip(x_new, -1.0, 1.0)
    sticks = x_new
    
    # 4. Throttle 映射: [-1, 1] → [0, 1]
    sticks = sticks.at[..., 3].set((sticks[..., 3] + 1.0) / 2.0)
    
    # 更新状态
    l1_state_new = l1_state.replace(virtual_stick_state=x_new)
    
    return sticks, l1_state_new


# =============================================================================
# Heading Frame 速度计算 (v18.2 核心)
# =============================================================================
def compute_heading_frame_velocities(
    sticks: Float[Array, "batch 4"],
    phys_state: PhysState,
    l1_state: L1State,
    l1_config: L1Config,
    dt: float = 0.02
) -> Tuple[Float[Array, "batch 3"], Float[Array, "batch 3"], Float[Array, "batch 3"], L1State]:
    """
    计算 Heading Frame 下的速度命令、实际速度和速度误差
    
    v18.2 架构核心: 替代旧的 PID 速度控制，L1 层仅做坐标变换。
    
    Heading Frame 定义:
    - X 轴: 飞机前进方向 (由 yaw 决定)
    - Y 轴: 飞机右侧
    - Z 轴: 向下 (FRD)
    
    两种模式:
    - Headed: 摇杆直接映射为 heading 系速度命令 (max_speed=15.0 m/s)
    - Headless: 摇杆→NED 速度 (max_speed=3.0 m/s) → R_yaw^T 旋转→heading 系
    
    垂直分量: vz_cmd = (throttle - 0.5) * 2 * max_vz
    
    Args:
        sticks: [roll, pitch, yaw, throttle] 虚拟摇杆信号
        phys_state: 当前物理状态
        l1_state: 当前 L1 状态
        l1_config: L1 配置
        dt: 时间步长
        
    Returns:
        v_cmd_heading: Heading Frame 速度命令 [batch, 3]
        v_actual_heading: Heading Frame 实际速度 [batch, 3]
        v_error_heading: Heading Frame 速度误差 [batch, 3]
        l1_state_new: 更新后的 L1 状态
    """
    roll_stick = sticks[..., 0]
    pitch_stick = sticks[..., 1]
    yaw_stick = sticks[..., 2]
    
    # 获取当前四元数和偏航角
    q_curr = phys_state.quaternion
    yaw_curr = quaternion_to_yaw(q_curr)
    v_world = phys_state.velocity  # NED 坐标系
    
    # NED → Heading Frame 旋转: v_heading = R_z(ψ)^T · v_NED
    # R_z(ψ)^T = [[cos(ψ), sin(ψ), 0], [-sin(ψ), cos(ψ), 0], [0,0,1]]
    # v_fwd  = cos(ψ)·v_N + sin(ψ)·v_E  (正确)
    # 修复(v18.3.1): 原注释写 "R_z(-yaw)" 且用 sin(-ψ)=-sin(ψ)，符号错误
    cos_yaw = jnp.cos(yaw_curr)   # cos(ψ)
    sin_yaw = jnp.sin(yaw_curr)   # sin(ψ)  [原为 sin(-ψ)=-sin(ψ), 已修复]
    
    # =====================================================================
    # Headed 模式: 摇杆直接映射为 Heading Frame 速度命令
    # =====================================================================
    max_speed_headed = 15.0  # m/s, 对标消费级无人机中等飞行速度
    
    # 摇杆 → Heading Frame 速度命令
    v_cmd_x_headed = pitch_stick * max_speed_headed   # 前向
    v_cmd_y_headed = roll_stick * max_speed_headed    # 右向
    
    # 实际速度: 世界系 → Heading Frame (R_z(-yaw) * v_world)
    v_actual_x_headed = cos_yaw * v_world[..., 0] + sin_yaw * v_world[..., 1]
    v_actual_y_headed = -sin_yaw * v_world[..., 0] + cos_yaw * v_world[..., 1]
    
    # Yaw 积分 (仅 Headed 模式)
    # v18.4: Yaw Slew Rate Limiter — 防止 OU/泊松跳跃产生的 yaw 命令阶跃突变
    yaw_rate_raw = yaw_stick * l1_config.max_yaw_rate  # 原始 yaw rate cmd
    max_yaw_delta = l1_config.max_yaw_cmd_rate * dt    # 每步最大变化量 (默认 ~0.035 rad/s)
    prev_yaw_rate = l1_state.prev_yaw_rate_cmd         # 上一步的 yaw rate cmd
    yaw_rate_delta = jnp.clip(yaw_rate_raw - prev_yaw_rate, -max_yaw_delta, max_yaw_delta)
    yaw_rate_headed = prev_yaw_rate + yaw_rate_delta    # 平滑后的 yaw rate cmd
    yaw_target_headed = l1_state.target_yaw + yaw_rate_headed * dt
    yaw_target_headed = jnp.arctan2(jnp.sin(yaw_target_headed), jnp.cos(yaw_target_headed))
    
    # =====================================================================
    # Headless 模式: 摇杆→NED 速度→R_yaw^T 旋转→Heading Frame
    # =====================================================================
    max_speed_headless = 3.0  # m/s, Headless 模式限速
    
    # 摇杆 → NED 速度命令
    v_north_cmd = pitch_stick * max_speed_headless
    v_east_cmd = roll_stick * max_speed_headless
    
    # NED 速度命令 → Heading Frame (R_z(-yaw) * v_cmd_NED)
    v_cmd_x_headless = cos_yaw * v_north_cmd + sin_yaw * v_east_cmd
    v_cmd_y_headless = -sin_yaw * v_north_cmd + cos_yaw * v_east_cmd
    
    # 实际速度: 世界系 → Heading Frame (与 Headed 相同的旋转)
    v_actual_x_headless = v_actual_x_headed  # 旋转公式相同
    v_actual_y_headless = v_actual_y_headed
    
    # Headless 模式: Yaw 锁定当前航向
    yaw_target_headless = yaw_curr
    
    # =====================================================================
    # 根据模式选择 (批量 jnp.where)
    # =====================================================================
    is_headed = (l1_state.mode == 0)  # [batch]
    
    v_cmd_x = jnp.where(is_headed, v_cmd_x_headed, v_cmd_x_headless)
    v_cmd_y = jnp.where(is_headed, v_cmd_y_headed, v_cmd_y_headless)
    v_actual_x = jnp.where(is_headed, v_actual_x_headed, v_actual_x_headless)
    v_actual_y = jnp.where(is_headed, v_actual_y_headed, v_actual_y_headless)
    yaw_target_new = jnp.where(is_headed, yaw_target_headed, yaw_target_headless)
    
    # =====================================================================
    # 垂直通道 (独立于模式)
    # =====================================================================
    # 油门摇杆映射: [0, 1] -> [-max_vz, max_vz]，正值=向上
    vz_cmd = (sticks[..., 3] - 0.5) * 2 * l1_config.max_vz
    
    # 实际垂直速度: NED z 向下为正，取负得向上速度
    vz_actual = -v_world[..., 2]
    
    # =====================================================================
    # 组装 3D 速度向量 (Heading Frame: X=前, Y=右, Z=下)
    # =====================================================================
    # FRD 坐标系: z 正向下，vz_cmd 正向上 → v_cmd_z = -vz_cmd
    v_cmd_raw = jnp.stack([v_cmd_x, v_cmd_y, -vz_cmd], axis=-1)
    v_actual_heading = jnp.stack([v_actual_x, v_actual_y, -vz_actual], axis=-1)
    
    # =====================================================================
    # Slew Rate Limiter (速度命令速率限制)
    # =====================================================================
    # 限制每步速度命令变化量 ≤ max_v_cmd_rate * dt (默认 8.0*0.02=0.16 m/s)
    # 确保 OU 噪声和泊松跳跃不会产生超过物理极限的速度命令突变
    max_delta = l1_config.max_v_cmd_rate * dt  # [batch] scalar
    prev_v_cmd = l1_state.prev_v_cmd_heading   # [batch, 3]
    
    delta_v = v_cmd_raw - prev_v_cmd
    delta_v_clipped = jnp.clip(delta_v, -max_delta[..., None], max_delta[..., None])
    v_cmd_heading = prev_v_cmd + delta_v_clipped
    
    v_error_heading = v_cmd_heading - v_actual_heading
    
    # =====================================================================
    # 更新 L1 状态
    # =====================================================================
    # =====================================================================
    # v18.4: 保存限速后的 yaw rate cmd (用于下一步 slew rate limiter)
    # Headed: 使用限速后的 yaw_rate_headed; Headless: yaw 锁定,rate=0
    yaw_rate_effective = jnp.where(is_headed, yaw_rate_headed, jnp.zeros_like(yaw_rate_headed))
    
    l1_state_new = l1_state.replace(
        target_yaw=yaw_target_new,
        velocity_error_heading=v_error_heading,
        prev_v_cmd_heading=v_cmd_heading,
        prev_yaw_rate_cmd=yaw_rate_effective,
    )
    
    return v_cmd_heading, v_actual_heading, v_error_heading, l1_state_new


# =============================================================================
# 高度控制 - 悬停油门前馈
# =============================================================================
def altitude_feedforward(
    l1_config: L1Config,
) -> Float[Array, "batch"]:
    """
    高度控制 - 仅前馈

    L2 观测空间已包含 v_error_heading[2] (垂直速度误差)，
    L1 仅提供悬停油门前馈值 t_hover_guess。
    L2 专家策略/神经网络负责油门闭环控制。
    
    Args:
        l1_config: L1 配置
    
    Returns:
        t_hover: 悬停油门前馈值 [0, 1]
    """
    t_hover = jnp.clip(l1_config.t_hover_guess, 0.1, 0.9)
    return t_hover


# =============================================================================
# L1 层主处理函数 (v18.2)
# =============================================================================
def l1_process(
    sticks: Float[Array, "batch 4"],
    phys_state: PhysState,
    l1_state: L1State,
    l1_config: L1Config,
    dt: float = 0.02
) -> Tuple[Float[Array, "batch 3"], Float[Array, "batch 3"], Float[Array, "batch"], L1State]:
    """
    L1 层主处理函数 (v18.2 坐标变换层)
    
    v18.2 架构: L1 不再进行 PID 控制，仅做坐标变换和速度误差计算。
    RL 网络（L2）直接从速度误差学习控制策略。
    
    Args:
        sticks: 虚拟摇杆 [roll, pitch, yaw, throttle]
        phys_state: 当前物理状态
        l1_state: 当前 L1 状态
        l1_config: L1 配置
        dt: 时间步长
    
    Returns:
        v_cmd_heading: Heading Frame 速度命令 [batch, 3]
        v_error_heading: Heading Frame 速度误差 [batch, 3] (核心驱动信号)
        t_hover: 悬停油门前馈 [batch]
        l1_state_new: 更新后的 L1 状态
    """
    # 1. 计算 Heading Frame 速度命令、实际速度和误差
    v_cmd_heading, v_actual_heading, v_error_heading, l1_state_new = \
        compute_heading_frame_velocities(
            sticks, phys_state, l1_state, l1_config, dt
        )
    
    # 2. 悬停油门前馈 (独立于模式)
    t_hover = altitude_feedforward(l1_config)
    
    # 3. 存储速度命令到 L1 状态（供 reward 等使用）
    l1_state_new = l1_state_new.replace(
        base_throttle=t_hover,
        target_velocity_body=v_cmd_heading,
    )
    
    return v_cmd_heading, v_error_heading, t_hover, l1_state_new


# =============================================================================
# 模式切换逻辑
# =============================================================================
def check_mode_switch(
    l3_saturation: Float[Array, "batch"],
    l3_yaw_integral: Float[Array, "batch"],
    l1_state: L1State,
    saturation_threshold: float = 0.95,
    yaw_integral_threshold: float = 0.25,
) -> L1State:
    """
    检查是否需要切换到 Headless 模式
    
    触发条件: 饱和度 > 0.95 且 Yaw 积分项过大
    
    Args:
        l3_saturation: L3 层饱和度
        l3_yaw_integral: L3 层 Yaw 积分项
        l1_state: 当前 L1 状态
        saturation_threshold: 饱和度阈值
        yaw_integral_threshold: Yaw 积分阈值
    
    Returns:
        l1_state_new: 更新后的 L1 状态 (可能切换模式)
    """
    # 故障检测条件
    fault_detected = (l3_saturation > saturation_threshold) & \
                     (jnp.abs(l3_yaw_integral) > yaw_integral_threshold)
    
    # 切换到 Headless 模式
    new_mode = jnp.where(fault_detected, 1, l1_state.mode)
    
    l1_state_new = l1_state.replace(mode=new_mode)
    
    return l1_state_new
