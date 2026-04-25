"""
AeroCat v18.0 L3 Control Layer

实现内环控制层 (Inner Loop)，包含:
- Rate PID 控制器 (P + I + D-on-Measurement)
- 动态混控器矩阵
- 电机指令输出

所有函数为纯函数，无状态类方法，支持 jit/vmap。
此模块设计用于 jax.lax.scan 内的 500Hz 子步循环。
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int
from typing import Tuple
import chex

from ..core.state import L3State, L3Config, PhysState, PhysParams


# =============================================================================
# Rate PID 控制器
# =============================================================================
def pid_controller(
    target_rates: Float[Array, "batch 3"],
    gyro: Float[Array, "batch 3"],
    l3_state: L3State,
    l3_config: L3Config,
    dt: float = 0.002
) -> Tuple[Float[Array, "batch 3"], L3State]:
    """
    Rate PID 控制器 (D-on-Measurement)
    
    采用 D-on-Measurement 避免设定值跳变引起的微分尖峰。
    
    Args:
        target_rates: 目标角速度 [roll_rate, pitch_rate, yaw_rate] (rad/s)
        gyro: 当前陀螺仪读数 [p, q, r] (rad/s)
        l3_state: 当前 L3 状态
        l3_config: L3 配置 (PID 增益)
        dt: 时间步长 (s)
    
    Returns:
        pid_output: PID 输出力矩 [τ_roll, τ_pitch, τ_yaw]
        l3_state_new: 更新后的 L3 状态
    """
    # 误差计算
    error = target_rates - gyro  # [batch, 3]
    
    # P 项
    p_out = l3_config.kp * error  # [batch, 3]
    
    # I 项 (带抗饱和限幅)
    i_new = l3_state.pid_integral + l3_config.ki * error * dt
    i_limit = jnp.asarray(l3_config.i_limit)
    i_new = jnp.clip(i_new, -i_limit, i_limit)
    
    # D 项 (D-on-Measurement: 对测量值求导而非误差)
    # 避免设定值突变引起的微分尖峰
    gyro_deriv = (gyro - l3_state.gyro_prev) / dt
    d_out = -l3_config.kd * gyro_deriv  # 负号: 阻尼当前运动
    
    # 总输出
    pid_output = p_out + i_new + d_out  # [batch, 3]
    
    # 更新 L3 状态
    l3_state_new = l3_state.replace(
        pid_integral=i_new,
        gyro_prev=gyro,
        pid_output_prev=pid_output
    )
    
    return pid_output, l3_state_new


# =============================================================================
# 动态混控器
# =============================================================================
def compute_motor_commands(
    thrust: Float[Array, "batch"],
    pid_output: Float[Array, "batch 3"],
    mixer_matrix: Float[Array, "batch 4 4"],
    motor_loss: Float[Array, "batch 4"]
) -> Tuple[Float[Array, "batch 4"], Float[Array, "batch"]]:
    """
    动态混控器: 将推力和力矩分配给各电机
    
    混控器矩阵基于机架几何动态计算。
    
    电机布局 (X 型, NED 坐标系):
        FL(2)   FR(0)
            X
        RL(1)   RR(3)
    
    Args:
        thrust: 总推力指令 [0, 1]
        pid_output: PID 输出力矩 [τ_roll, τ_pitch, τ_yaw]
        mixer_matrix: 混控器矩阵 [batch, 4, 4]
        motor_loss: 电机效率差异 [batch, 4], 0=正常, 1=完全失效
    
    Returns:
        motor_cmds: 各电机指令 [0, 1]
        saturation: 饱和度指标 (max(cmd) - 1.0 的正部分)
    """
    # 组装控制输入向量 [Thrust, Roll, Pitch, Yaw]
    control_input = jnp.stack([
        thrust,
        pid_output[..., 0],  # Roll
        pid_output[..., 1],  # Pitch
        pid_output[..., 2],  # Yaw
    ], axis=-1)  # [batch, 4]
    
    # 混控: motor_cmds = mixer_matrix @ control_input
    # [batch, 4, 4] @ [batch, 4, 1] -> [batch, 4]
    motor_cmds = jnp.einsum('bij,bj->bi', mixer_matrix, control_input)
    
    # 应用电机效率差异 (连续值 [0, 1])
    motor_cmds = motor_cmds * (1.0 - motor_loss)
    
    # 计算饱和度 (用于反馈给 L2)
    max_cmd = jnp.max(motor_cmds, axis=-1)
    saturation = jnp.maximum(max_cmd - 1.0, 0.0)
    
    # 限幅到 [0, 1]
    motor_cmds = jnp.clip(motor_cmds, 0.0, 1.0)
    
    # =========================================================================
    # 推力线性化 (Thrust Linearization)
    # =========================================================================
    # 参考: Betaflight thrust_linear 功能
    # 
    # 问题: 物理推力 F = th² × F_max (非线性)
    #       当 PID 输出使电机分配不均匀时，Jensen 不等式导致总推力增加
    # 
    # 解决: 对混控器输出进行 sqrt 补偿
    #       th_output = sqrt(th_linear)
    #       则 F = th_output² × F_max = th_linear × F_max (线性)
    # 
    # 这与 Betaflight 的 thrust_linear 功能等效
    motor_cmds = jnp.sqrt(motor_cmds)
    
    return motor_cmds, saturation


# =============================================================================
# 电机动力学 (一阶非对称滞后)
# =============================================================================
def motor_dynamics(
    current_throttle: Float[Array, "batch 4"],
    target_throttle: Float[Array, "batch 4"],
    tau_up: Float[Array, "batch 4"],
    tau_down: Float[Array, "batch 4"],
    deadband: Float[Array, "batch 4"],
    dt: float = 0.002
) -> Float[Array, "batch 4"]:
    """
    电机一阶非对称滞后动力学 + 死区
    
    模拟 ESC 加速快、减速慢的特性，并应用电机启动阈值。
    
    参考:
        04_训练环境_仿真构建指南_v18.0.md 1.3.2节 (一阶滞后)
        05_训练环境_PF-GDR领域随机化_v18.0.md 3.5节 (死区)
    
    公式:
        τ = τ_up   if target > current
        τ = τ_down if target <= current
        α = dt / (τ + dt)
        next = (1 - α) * current + α * target
        next = 0  if next < deadband  (死区截断)
    
    Args:
        current_throttle: 当前电机油门 [0, 1]
        target_throttle: 目标电机油门 [0, 1]
        tau_up: 加速时间常数 (s)
        tau_down: 减速时间常数 (s)
        deadband: 电机死区阈值 [0, 0.1] (PF-GDR §3.5)
        dt: 时间步长 (s)
    
    Returns:
        next_throttle: 下一步电机油门
    """
    # 判断方向
    is_accelerating = target_throttle > current_throttle
    
    # 选择时间常数
    tau = jnp.where(is_accelerating, tau_up, tau_down)
    
    # 计算平滑系数 alpha
    alpha = dt / (tau + dt)
    
    # 一阶滞后更新
    next_throttle = (1.0 - alpha) * current_throttle + alpha * target_throttle
    
    # 应用死区: 低于阈值的油门截断为 0 (模拟电机启动阈值)
    next_throttle = jnp.where(next_throttle < deadband, 0.0, next_throttle)
    
    return next_throttle


# =============================================================================
# L3 完整子步 (用于 jax.lax.scan)
# =============================================================================
def l3_substep(
    phys_state: PhysState,
    l3_state: L3State,
    target_rates: Float[Array, "batch 3"],
    thrust: Float[Array, "batch"],
    phys_params: PhysParams,
    l3_config: L3Config,
    mixer_matrix: Float[Array, "batch 4 4"],
    dt: float = 0.002
) -> Tuple[PhysState, L3State, Float[Array, "batch 4"]]:
    """
    L3 内环单个子步 (500Hz)
    
    包含 PID、Mixer、电机动力学。
    物理动力学在此函数外部调用。
    
    Args:
        phys_state: 当前物理状态
        l3_state: 当前 L3 状态
        target_rates: L2 下发的目标角速度 (rad/s)
        thrust: L2 下发的总推力指令 [0, 1]
        phys_params: 物理参数
        l3_config: L3 配置
        mixer_matrix: 预计算的混控器矩阵
        dt: 时间步长
    
    Returns:
        phys_state_new: 更新后的物理状态 (电机油门已更新)
        l3_state_new: 更新后的 L3 状态
        motor_cmds: 电机指令 (用于物理仿真)
    """
    # 1. 获取传感器读数
    gyro = phys_state.imu_gyro
    
    # 2. PID 计算
    pid_output, l3_state_new = pid_controller(
        target_rates, gyro, l3_state, l3_config, dt
    )
    
    # 3. 混控器
    motor_cmds, saturation = compute_motor_commands(
        thrust, pid_output, mixer_matrix, phys_params.motor_loss
    )
    
    # 4. 电机动力学
    new_throttle = motor_dynamics(
        phys_state.motor_throttle,
        motor_cmds,
        phys_params.motor_tau_up,
        phys_params.motor_tau_down,
        phys_params.motor_deadband,
        dt
    )
    
    # 5. 更新状态
    phys_state_new = phys_state.replace(motor_throttle=new_throttle)
    l3_state_new = l3_state_new.replace(mixer_saturation=saturation)
    
    return phys_state_new, l3_state_new, motor_cmds


# =============================================================================
# L3 完整循环 (10 次子步)
# =============================================================================
def l3_inner_loop(
    phys_state: PhysState,
    l3_state: L3State,
    target_rates: Float[Array, "batch 3"],
    thrust: Float[Array, "batch"],
    phys_params: PhysParams,
    l3_config: L3Config,
    mixer_matrix: Float[Array, "batch 4 4"],
    physics_step_fn,  # 物理步进函数
    num_substeps: int = 10,
    dt_sub: float = 0.002
) -> Tuple[PhysState, L3State]:
    """
    L3 内环完整循环 (使用 jax.lax.scan)
    
    执行 num_substeps 次 500Hz 子步。
    
    Args:
        phys_state: 初始物理状态
        l3_state: 初始 L3 状态
        target_rates: L2 下发的目标角速度 (固定不变)
        thrust: L2 下发的总推力 (固定不变)
        phys_params: 物理参数
        l3_config: L3 配置
        mixer_matrix: 混控器矩阵
        physics_step_fn: 物理步进函数 (state, action, params, dt) -> state
        num_substeps: 子步数量 (默认 10)
        dt_sub: 子步时间步长 (默认 0.002s = 500Hz)
    
    Returns:
        phys_state_final: 最终物理状态
        l3_state_final: 最终 L3 状态
    """
    def scan_fn(carry, _):
        """jax.lax.scan 的循环体"""
        phys_state, l3_state = carry
        
        # L3 子步 (PID + Mixer + Motor Dynamics)
        phys_state, l3_state, motor_cmds = l3_substep(
            phys_state, l3_state,
            target_rates, thrust,
            phys_params, l3_config, mixer_matrix,
            dt_sub
        )
        
        # 物理动力学步进
        phys_state = physics_step_fn(phys_state, motor_cmds, phys_params, dt_sub)
        
        return (phys_state, l3_state), None
    
    # 初始 carry
    init_carry = (phys_state, l3_state)
    
    # 执行 scan (编译为单个 XLA 循环)
    (phys_state_final, l3_state_final), _ = jax.lax.scan(
        scan_fn,
        init_carry,
        xs=None,
        length=num_substeps
    )
    
    return phys_state_final, l3_state_final


# =============================================================================
# 动作映射 (v19.0: tanh·scale 替换 Signed Power)
# =============================================================================
def map_action_to_rates_and_thrust(
    raw_action: Float[Array, "batch 4"],
    max_rp_rate: float = 1080.0 * jnp.pi / 180.0,   # ~18.8 rad/s
    max_yaw_rate: float = 2000.0 * jnp.pi / 180.0,  # ~35 rad/s
) -> Tuple[Float[Array, "batch 3"], Float[Array, "batch"]]:
    """
    将 RL 网络输出映射为物理指令 (v19.0: tanh·scale)
    
    v19.0 变更: 去除 Signed Power (|x|^alpha)，改用纯线性 scale
    网络输出已经过 tanh ∈ [-1, 1]，直接 × max_rate 即可
    
    Args:
        raw_action: 网络原始输出 [roll, pitch, yaw, thrust] ∈ [-1, 1]^4
        max_rp_rate: Roll/Pitch 最大角速度 (rad/s)
        max_yaw_rate: Yaw 最大角速度 (rad/s)
    
    Returns:
        target_rates: 目标角速度 [roll_rate, pitch_rate, yaw_rate] (rad/s)
        thrust: 推力指令 [0, 1]
    """
    # Roll/Pitch 线性映射
    rp_rates = raw_action[..., :2] * max_rp_rate
    
    # Yaw 线性映射 (更大范围)
    yaw_rate = raw_action[..., 2:3] * max_yaw_rate
    
    # 合并角速度
    target_rates = jnp.concatenate([rp_rates, yaw_rate], axis=-1)
    
    # 油门映射: [-1, 1] -> [0, 1]
    thrust = (raw_action[..., 3] + 1.0) / 2.0
    
    return target_rates, thrust


# =============================================================================
# 诊断指标计算 (v19.0 扩展: 4D → 9D)
# =============================================================================
def compute_diagnostics(
    l3_state: L3State,
    phys_state = None,
) -> Float[Array, "batch 9"]:
    """
    计算 L3 层诊断指标 (反馈给 L2 观测)
    
    v19.0 扩展: 从 4D 增至 9D
    - [0:3]  pid_integral (Roll, Pitch, Yaw)  — PID 积分器值
    - [3]    saturation                       — 混控器饱和度
    - [4:7]  tracking_rms (Roll, Pitch, Yaw)  — 跟踪误差 RMS（反映控制品质）
    - [7]    integral_rate                    — 积分器变化速率（反映偏差持续性）
    - [8]    motor_spread                     — 电机指令离散度（反映不对称负载）
    
    设计理由 (C2 创新):
    这些信号携带了 PID 控制器的"解题过程"信息，而非仅仅是系统输出。
    RL Agent 看到这些信号后能"感知"到 PID 是否在挣扎、执行器是否饱和。
    
    Args:
        l3_state: L3 状态
        phys_state: 物理状态（用于提取电机油门计算 motor_spread）
    
    Returns:
        diagnostics: [batch, 9]
    """
    # 原始 4D: integral + saturation
    pid_integral = l3_state.pid_integral         # [batch, 3]
    saturation = l3_state.mixer_saturation        # [batch]
    
    # --- v19.0 新增 5D ---
    
    # tracking_rms: |pid_output_prev|，近似跟踪误差 RMS
    # 越大说明 PID 越"用力"纠偏，跟踪品质越差
    tracking_rms = jnp.abs(l3_state.pid_output_prev)  # [batch, 3]
    # 归一化到 ~[0, 1] 范围（典型 PID 输出范围 [-1, 1]）
    tracking_rms = jnp.clip(tracking_rms, 0.0, 2.0) / 2.0
    
    # integral_rate: 积分器变化速率的 L2 范数
    # 快速变化说明持续偏差（稳态时趋近零）
    # 使用 pid_integral 自身的 L2 norm 近似（每步 dt 内积分值的幅度）
    integral_rate = jnp.sqrt(jnp.sum(l3_state.pid_integral**2, axis=-1) + 1e-8)
    integral_rate = jnp.clip(integral_rate, 0.0, 1.0)  # [batch]
    
    # motor_spread: 电机指令的标准差（归一化）
    # 高 spread 意味着不对称补偿（例如某电机损坏后其他电机"拉偏"）
    if phys_state is not None:
        throttle = phys_state.motor_throttle           # [batch, 4]
        motor_spread = jnp.std(throttle, axis=-1)      # [batch]
        motor_spread = jnp.clip(motor_spread, 0.0, 0.5) / 0.5  # 归一化到 [0,1]
    else:
        motor_spread = jnp.zeros_like(saturation)
    
    return jnp.concatenate([
        pid_integral,                       # [batch, 3]  idx 0:3
        saturation[..., None],              # [batch, 1]  idx 3
        tracking_rms,                       # [batch, 3]  idx 4:7
        integral_rate[..., None],           # [batch, 1]  idx 7
        motor_spread[..., None],            # [batch, 1]  idx 8
    ], axis=-1)

