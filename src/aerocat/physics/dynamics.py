"""
AeroCat v18.0 Physics Dynamics Engine

6-DOF 刚体动力学实现，针对 v18.0 PhysState 结构优化:
- 电机动力学 (非对称一阶滞后)
- 推力/力矩计算 (含倾斜角)
- 刚体动力学 (Newton-Euler 方程)
- 二次阻力模型 (平移 + 旋转)
- 四元数积分

所有函数为纯 JAX，支持 jit/vmap，针对 4096+ 并行环境优化。
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray, Int
from typing import Tuple
import chex
from functools import partial

from ..core.state import PhysState, PhysParams
from .battery import battery_ecm_step, compute_motor_power_current
from .sensors import imu_noise_model


# =============================================================================
# 四元数工具函数
# =============================================================================
def quaternion_multiply(q1: Float[Array, "batch 4"], q2: Float[Array, "batch 4"]) -> Float[Array, "batch 4"]:
    """四元数乘法 [w, x, y, z]"""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    
    return jnp.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def quaternion_conjugate(q: Float[Array, "batch 4"]) -> Float[Array, "batch 4"]:
    """四元数共轭 [w, x, y, z] -> [w, -x, -y, -z]"""
    return jnp.stack([q[..., 0], -q[..., 1], -q[..., 2], -q[..., 3]], axis=-1)


def quaternion_normalize(q: Float[Array, "batch 4"]) -> Float[Array, "batch 4"]:
    """四元数归一化"""
    norm = jnp.linalg.norm(q, axis=-1, keepdims=True)
    return q / (norm + 1e-8)


def rotate_vector(v: Float[Array, "batch 3"], q: Float[Array, "batch 4"]) -> Float[Array, "batch 3"]:
    """
    用四元数旋转向量: v' = q * v * q^-1
    
    Args:
        v: 待旋转向量
        q: 旋转四元数 [w, x, y, z]
    
    Returns:
        旋转后的向量
    """
    # 将向量扩展为四元数 [0, x, y, z]
    v_quat = jnp.concatenate([jnp.zeros_like(v[..., :1]), v], axis=-1)
    
    # q * v * q^-1
    q_conj = quaternion_conjugate(q)
    rotated = quaternion_multiply(quaternion_multiply(q, v_quat), q_conj)
    
    return rotated[..., 1:4]


def quaternion_integrate(q: Float[Array, "batch 4"], omega: Float[Array, "batch 3"], dt: float) -> Float[Array, "batch 4"]:
    """
    四元数一阶积分: q_new = q + 0.5 * q * [0, omega] * dt
    
    Args:
        q: 当前四元数
        omega: 角速度 (rad/s)
        dt: 时间步长
    
    Returns:
        更新后的四元数 (已归一化)
    """
    omega_quat = jnp.concatenate([jnp.zeros_like(omega[..., :1]), omega], axis=-1)
    q_dot = 0.5 * quaternion_multiply(q, omega_quat)
    q_new = q + q_dot * dt
    return quaternion_normalize(q_new)


# =============================================================================
# 电机动力学 [DEPRECATED - 未被调用]
# 注意: 实际电机动力学在 l3_controller.motor_dynamics() 中执行
# 此函数保留仅供参考，不参与仿真主循环
# =============================================================================
def motor_dynamics_step(
    current_throttle: Float[Array, "batch 4"],
    target_throttle: Float[Array, "batch 4"],
    tau_up: Float[Array, "batch 4"],
    tau_down: Float[Array, "batch 4"],
    deadband: Float[Array, "batch 4"] = None,
    dt: float = 0.002
) -> Float[Array, "batch 4"]:
    """
    电机一阶非对称滞后动力学 + 死区
    
    参考: 04_训练环境_仿真构建指南_v18.0.md 1.3.2节
    
    公式:
        τ = τ_up   if target > current
        τ = τ_down if target <= current
        α = dt / (τ + dt)
        next = (1 - α) * current + α * target
    """
    # 判断加速/减速
    is_accelerating = target_throttle > current_throttle
    
    # 选择时间常数
    tau = jnp.where(is_accelerating, tau_up, tau_down)
    
    # 计算平滑系数
    alpha = dt / (tau + dt + 1e-8)
    
    # 一阶滞后
    new_throttle = (1.0 - alpha) * current_throttle + alpha * target_throttle
    
    # 应用死区 (低于阈值截断为 0)
    if deadband is not None:
        new_throttle = jnp.where(new_throttle < deadband, 0.0, new_throttle)
    
    
    
    # 限幅
    new_throttle = jnp.clip(new_throttle, 0.0, 1.0)
    
    return new_throttle


# =============================================================================
# 推力和力矩计算
# =============================================================================
def compute_thrust_and_torque(
    motor_throttle: Float[Array, "batch 4"],
    phys_params: PhysParams
) -> Tuple[Float[Array, "batch 3"], Float[Array, "batch 3"]]:
    """
    计算推力和力矩
    
    电机布局 (X 型, NED 坐标系):
        FL(2)   FR(0)
            X
        RL(1)   RR(3)
    
    Args:
        motor_throttle: 电机油门 [0, 1]
        phys_params: 物理参数
    
    Returns:
        thrust_body: 机体系推力 (N)
        torque_body: 机体系力矩 (N·m)
    """
    batch_size = motor_throttle.shape[0]
    
    # 1. 计算各电机推力 (推力与油门平方成正比)
    motor_thrust = motor_throttle ** 2 * phys_params.motor_max_thrust[..., None]  # [batch, 4]
    
    # 注意: motor_loss 已在 l3_controller.compute_motor_commands 中应用
    # 不在此处重复应用，避免双重损耗 Bug
    
    # 2. 总推力 (机体 Z 轴向上)
    total_thrust = jnp.sum(motor_thrust, axis=-1)  # [batch]
    
    # 推力向量 (机体系, FRD: Z 向下，推力向上取负)
    thrust_body = jnp.stack([
        jnp.zeros((batch_size,)),
        jnp.zeros((batch_size,)),
        -total_thrust  # FRD: 推力向上为 -Z
    ], axis=-1)
    
    # 3. 计算力矩
    arm_length = phys_params.arm_length
    frame_angle = phys_params.frame_angle
    
    # 力臂系数
    A = jnp.sin(frame_angle)  # Roll 力臂
    B = jnp.cos(frame_angle)  # Pitch 力臂
    
    # Roll 力矩: (FL + RL) - (FR + RR) * A * arm_length
    roll_torque = (
        (motor_thrust[..., 2] + motor_thrust[..., 1]) -
        (motor_thrust[..., 0] + motor_thrust[..., 3])
    ) * A * arm_length
    
    # Pitch 力矩: (FL + FR) - (RL + RR) * B * arm_length
    pitch_torque = (
        (motor_thrust[..., 2] + motor_thrust[..., 0]) -
        (motor_thrust[..., 1] + motor_thrust[..., 3])
    ) * B * arm_length
    
    # Yaw 力矩: 反扭矩 (CCW - CW)
    # FR(0): CCW, RL(1): CCW, FL(2): CW, RR(3): CW
    # Yaw 力矩: 反扭矩 (CW - CCW)
    # FR(0): CCW (+Z spin -> -Z torque)
    # RL(1): CCW (+Z spin -> -Z torque)
    # FL(2): CW  (-Z spin -> +Z torque)
    # RR(3): CW  (-Z spin -> +Z torque)
    # 使用 VAC sigma (force_to_torque_ratio) - per motor
    yaw_torque_per_motor = motor_thrust * phys_params.force_to_torque_ratio
    yaw_torque = (
        yaw_torque_per_motor[..., 2] + yaw_torque_per_motor[..., 3] -
        yaw_torque_per_motor[..., 0] - yaw_torque_per_motor[..., 1]
    )
    
    # 重心偏移产生的额外力矩
    cog_torque = jnp.cross(phys_params.cog_offset, thrust_body)
    
    torque_body = jnp.stack([roll_torque, pitch_torque, yaw_torque], axis=-1) + cog_torque
    
    return thrust_body, torque_body


# =============================================================================
# 二次阻力模型
# =============================================================================
def compute_quadratic_drag(
    velocity: Float[Array, "batch 3"],
    angular_velocity: Float[Array, "batch 3"],
    drag_coeff_xyz: Float[Array, "batch 3"],
    rot_drag_coeff: Float[Array, "batch 3"]
) -> Tuple[Float[Array, "batch 3"], Float[Array, "batch 3"]]:
    """
    二次阻力模型 (平移 + 旋转)
    
    参考: 04_训练环境_仿真构建指南_v18.0.md 1.3.1节
    
    公式:
        F_drag = -Cd * |v| * v  (二次)
        τ_drag = -Cr * |ω| * ω  (二次)
    """
    # 平移阻力
    v_norm = jnp.linalg.norm(velocity, axis=-1, keepdims=True)
    drag_force = -drag_coeff_xyz * v_norm * velocity
    
    # 旋转阻力 (对自旋恢复至关重要)
    omega_norm = jnp.linalg.norm(angular_velocity, axis=-1, keepdims=True)
    drag_torque = -rot_drag_coeff * omega_norm * angular_velocity
    
    return drag_force, drag_torque


# =============================================================================
# 完整物理步进
# =============================================================================
def physics_dynamics_step(
    phys_state: PhysState,
    motor_cmds: Float[Array, "batch 4"],
    phys_params: PhysParams,
    key: PRNGKeyArray,
    dt: float = 0.002
) -> PhysState:
    """
    完整 6-DOF 刚体物理动力学步进
    
    这是 AeroCat v18.0 的核心物理仿真函数,在 500Hz L3 内环中被调用。
    参考: 04_训练环境_仿真构建指南_v18.0.md
    
    ========== 物理模型架构 ==========
    
    坐标系统: NED (North-East-Down)
    - X: 北 (前)
    - Y: 东 (右)  
    - Z: 下 (机体重力方向)
    
    四元数约定: Hamilton [w, x, y, z]
    - q = [cos(θ/2), sin(θ/2) * axis]
    - 旋转方向遵循右手定则
    
    电机布局 (X 型):
        FL(2)   FR(0)
            ╲ / 
             X
            / ╲
        RL(1)   RR(3)
    
    ========== 数学模型 ==========
    
    1. 电机动力学 (非对称一阶滞后):
       τ = τ_up   if cmd > current   (加速)
       τ = τ_down if cmd <= current  (减速)
       
       θ_new = θ + (cmd - θ) * (1 - exp(-dt/τ))
       
    2. 推力计算:
       F_motor[i] = ρ_KN * θ[i]^2 * F_max * V/V_nom
       
       其中:
       - ρ_KN: 油门-推力系数 (~1.0 for linear)
       - F_max: 最大推力 (N)
       - V/V_nom: 电压归一化因子
       
    3. 力矩计算:
       τ_roll  = arm_length * (F[1] + F[2] - F[0] - F[3]) * cos(45°)
       τ_pitch = arm_length * (F[0] + F[2] - F[1] - F[3]) * sin(45°)
       τ_yaw   = Kq * (F[0] + F[1] - F[2] - F[3])
       
    4. 二次阻力模型:
       F_drag = -C_d * |V_air| * V_air   (机体系)
       τ_drag = -C_r * |ω| * ω           (机体系)
       
    5. 刚体动力学 (Newton-Euler):
       线性: m * a = F_thrust + F_drag + m * g
       角向: I * α = τ_motor - ω × (I * ω) - τ_drag
       
    6. 数值积分 (半隐式欧拉):
       v_new = v + a * dt
       x_new = x + v_new * dt
       
       角速度使用半隐式方案处理阻力:
       ω_new = ω_pred / (1 + dt * C_r * |ω| / I)
       
       这比显式 Euler 稳定得多,特别是在大角速度时。
    
    7. 四元数积分 (一阶):
       q_dot = 0.5 * q ⊗ [0, ω]
       q_new = normalize(q + q_dot * dt)
    
    ========== 传感器模型 ==========
    
    IMU 噪声模型:
       gyro  = ω_true + b_gyro + N(0, σ_gyro)
       accel = a_body/m + b_accel + N(0, σ_accel)
    
    ========== 性能优化 ==========
    
    - 所有计算在机体系进行,减少坐标变换
    - 使用 JAX 的 JIT 编译
    - 批量化设计,支持 vmap
    - 无 Python 循环,纯 JAX 操作
    
    Args:
        phys_state: 当前物理状态 (PhysState)
        motor_cmds: 电机指令 [batch, 4], 取值 [0, 1]
        phys_params: 物理参数 (PhysParams)
        key: JAX 随机数密钥 (用于传感器噪声)
        dt: 时间步长 (s), 默认 0.002 (500Hz)
    
    Returns:
        PhysState: 更新后的物理状态
        
    Note:
        此函数在 uav_env.py 中被导入为 physics_step_optimized()
    """
    batch_size = phys_state.position.shape[0]
    
    # =================================================================
    # 1. 电机油门 (已由 l3_substep 更新，此处直接使用)
    # =================================================================
    # 注意: motor_throttle 已经在 l3_substep 中通过 motor_dynamics() 更新
    # 此处不再重复调用电机动力学，避免重复应用滤波器
    new_throttle = phys_state.motor_throttle

    # =================================================================
    # 1.5. 电池动力学 (ECM)
    # =================================================================
    # 计算总电流
    current = compute_motor_power_current(
        new_throttle,
        phys_state.battery_voltage,
        phys_params.motor_max_thrust
    )
    
    # 步进 ECM 模型
    new_soc, new_voltage, new_v1, new_v2 = battery_ecm_step(
        phys_state.battery_soc,
        current,
        phys_state.battery_V1,
        phys_state.battery_V2,
        phys_params.battery_r0,
        phys_params.battery_r1,
        phys_params.battery_r2,
        phys_params.battery_c1,
        phys_params.battery_c2,
        phys_params.battery_ocv_coeff,
        # Capacity (mAh -> Ah)
        phys_params.battery_capacity / 1000.0, 
        dt
    )
    
    # =================================================================
    # 2. 推力/力矩计算
    # =================================================================
    thrust_body, torque_body = compute_thrust_and_torque(new_throttle, phys_params)
    
    # =================================================================
    # 3. 二次阻力 (机体系, 基于空速)
    # =================================================================
    # 计算世界系空速: V_air = V_ground - V_wind
    airspeed_world = phys_state.velocity - phys_state.wind_velocity
    
    # 将世界系空速转换到机体系
    q_conj = quaternion_conjugate(phys_state.quaternion)
    airspeed_body = rotate_vector(airspeed_world, q_conj)
    
    drag_force_body, drag_torque = compute_quadratic_drag(
        airspeed_body,
        phys_state.angular_velocity,
        phys_params.drag_coeff_xyz,
        phys_params.rot_drag_coeff
    )
    
    # =================================================================
    # 4. 刚体线性运动
    # =================================================================
    # 总机体力 = 推力 + 阻力
    total_force_body = thrust_body + drag_force_body
    
    # 转换到世界系
    total_force_world = rotate_vector(total_force_body, phys_state.quaternion)
    
    # 重力 (NED: +Z 向下)
    gravity = jnp.array([0.0, 0.0, 9.81])
    gravity_force = phys_params.mass[..., None] * gravity
    
    # 总力 + 重力
    total_force_world = total_force_world + gravity_force
    
    # 线加速度
    linear_accel = total_force_world / phys_params.mass[..., None]
    
    
    # Semi-implicit Euler
    new_velocity = phys_state.velocity + linear_accel * dt
    new_position = phys_state.position + new_velocity * dt
    
    # 速度限幅 (物理极限)
    new_velocity = jnp.clip(new_velocity, -100.0, 100.0)
    
    # =================================================================
    # 5. 刚体角运动 (Euler 方程)
    # =================================================================
    # =================================================================
    # 5. 刚体角运动 (Euler 方程)
    # =================================================================
    # 改用半隐式积分 (Semi-Implicit) 处理二次阻力，防止数值不稳定
    # Explicit Euler 对二次阻力极其不稳定，特别是当 omega > 2 rad/s 时
    
    # 1. 驱动力矩 (电机 - 陀螺效应)
    # Euler: I * α = τ_motor - ω × (I * ω) - τ_drag
    inertia = jnp.stack([
        phys_params.inertia_xx,
        phys_params.inertia_yy,
        phys_params.inertia_zz
    ], axis=-1)
    
    omega = phys_state.angular_velocity
    I_omega = inertia * omega
    gyroscopic = jnp.cross(omega, I_omega)
    
    driving_torque = torque_body - gyroscopic
    alpha_drive = driving_torque / (inertia + 1e-8)
    
    # 2. 预测无阻力角速度
    omega_pred = omega + alpha_drive * dt
    
    # 3. 应用阻力 (半隐式)
    # 阻力形式: τ_drag = -C * |ω| * ω
    # 离散化: (ω_new - ω) / dt = ... - (C * |ω| / I) * ω_new
    # ω_new * (1 + dt * C * |ω| / I) = ω_pred
    # ω_new = ω_pred / (1 + dt * C * |ω| / I)
    
    omega_norm = jnp.linalg.norm(omega, axis=-1, keepdims=True)
    damping_factor = phys_params.rot_drag_coeff * omega_norm
    damping_rate = damping_factor / (inertia + 1e-8)
    
    new_omega = omega_pred / (1.0 + damping_rate * dt)
    
    # 角速度限幅 (物理极限)
    new_omega = jnp.clip(new_omega, -100.0, 100.0)
    
    # =================================================================
    # 6. 四元数积分
    # =================================================================
    new_quaternion = quaternion_integrate(phys_state.quaternion, new_omega, dt)
    
    # =================================================================
    # 7. 传感器更新 (带噪声)
    # =================================================================
    true_gyro = new_omega
    # 加速度计真值 (比力) = 机体系总力 / 质量
    # Accel = F_total / m (Specific Force)
    true_accel = total_force_body / phys_params.mass[..., None]
    
    new_gyro, new_accel = imu_noise_model(
        true_gyro,
        true_accel,
        phys_params.gyro_noise_std,
        phys_params.accel_noise_std,
        phys_params.gyro_bias,
        phys_params.accel_bias,
        key
    )
    
    # =================================================================
    # 8. 组装新状态
    # =================================================================
    return phys_state.replace(
        position=new_position,
        velocity=new_velocity,
        quaternion=new_quaternion,
        angular_velocity=new_omega,
        motor_throttle=new_throttle,
        battery_soc=new_soc,
        battery_voltage=new_voltage,
        battery_V1=new_v1,
        battery_V2=new_v2,
        imu_gyro=new_gyro,
        imu_accel=new_accel,
    )


# =============================================================================
# 子步物理更新 (为 L3 内环优化)
# =============================================================================
def physics_substep(
    phys_state: PhysState,
    motor_cmds: Float[Array, "batch 4"],
    phys_params: PhysParams,
    key: PRNGKeyArray,
    dt: float = 0.002
) -> PhysState:
    """
    单个物理子步 (500Hz)
    
    为 jax.lax.scan 内循环优化，避免不必要的计算。
    """
    return physics_dynamics_step(phys_state, motor_cmds, phys_params, key, dt)


# =============================================================================
# 批量 vmap 包装器
# =============================================================================
# 如果需要对非批量状态进行批处理
# physics_step_batch = jax.vmap(physics_dynamics_step, in_axes=(0, 0, 0, None))
