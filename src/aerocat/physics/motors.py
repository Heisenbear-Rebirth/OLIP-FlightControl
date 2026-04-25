"""
AeroCat v18.0 Motor Dynamics Model (Standalone)

电机动力学独立模块:
- 非对称一阶滞后 (加速/减速不同时间常数)
- ESC 特性模拟
- 推力/反扭矩计算
- 电机效率差异

参考: 04_训练环境_仿真构建指南_v18.0.md 1.3.2节
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing import Tuple


# =============================================================================
# 非对称一阶滞后电机模型
# =============================================================================
def motor_first_order_lag(
    current_throttle: Float[Array, "batch 4"],
    target_throttle: Float[Array, "batch 4"],
    tau_up: Float[Array, "batch 4"],
    tau_down: Float[Array, "batch 4"],
    dt: float = 0.002
) -> Float[Array, "batch 4"]:
    """
    非对称一阶滞后电机动力学
    
    参考: 04_训练环境_仿真构建指南_v18.0.md 1.3.2节
    
    公式:
        τ = τ_up   if target > current (加速)
        τ = τ_down if target <= current (减速)
        α = dt / (τ + dt)
        next = (1 - α) * current + α * target
    
    ESC 特性:
    - 加速响应快 (tau_up ≈ 0.01-0.03s)
    - 减速响应慢 (tau_down ≈ 0.02-0.10s)，因主动刹车能力有限
    
    Args:
        current_throttle: 当前电机油门 [0, 1]
        target_throttle: 目标电机油门 [0, 1]
        tau_up: 加速时间常数 (s)
        tau_down: 减速时间常数 (s)
        dt: 时间步长 (s)
    
    Returns:
        new_throttle: 更新后的电机油门
    """
    # 1. 判断加速 vs 减速
    is_accelerating = target_throttle > current_throttle
    
    # 2. 选择对应的时间常数
    tau = jnp.where(is_accelerating, tau_up, tau_down)
    
    # 3. 计算平滑系数 α
    tau_safe = jnp.maximum(tau, 1e-6)  # 防止除零
    alpha = dt / (tau_safe + dt)
    
    # 4. 一阶滞后更新
    new_throttle = (1.0 - alpha) * current_throttle + alpha * target_throttle
    
    # 5. 限幅到 [0, 1]
    new_throttle = jnp.clip(new_throttle, 0.0, 1.0)
    
    return new_throttle


def motor_with_deadband(
    throttle: Float[Array, "batch 4"],
    deadband: Float[Array, "batch 4"]
) -> Float[Array, "batch 4"]:
    """
    应用电机死区
    
    低于死区阈值的输入映射为 0。
    
    Args:
        throttle: 电机油门 [0, 1]
        deadband: 死区阈值 [0, 0.1] 典型
    
    Returns:
        effective_throttle: 有效油门
    """
    # 低于死区直接截断
    effective = jnp.where(throttle < deadband, 0.0, throttle)
    return effective


def motor_with_efficiency_loss(
    throttle: Float[Array, "batch 4"],
    motor_loss: Float[Array, "batch 4"]
) -> Float[Array, "batch 4"]:
    """
    应用电机效率差异
    
    Args:
        throttle: 电机油门 [0, 1]
        motor_loss: 电机效率差异 [0, 1] (工艺公差)
    
    Returns:
        effective_throttle: 考虑效率差异后的有效油门
    """
    return throttle * (1.0 - motor_loss)


# =============================================================================
# 推力计算
# =============================================================================
def motor_throttle_to_thrust(
    throttle: Float[Array, "batch 4"],
    max_thrust: Float[Array, "batch"]
) -> Float[Array, "batch 4"]:
    """
    油门到推力转换
    
    推力与转速平方成正比，而转速与油门线性相关:
        F = k * (RPM)² ∝ throttle²
    
    Args:
        throttle: 电机油门 [0, 1]
        max_thrust: 单电机最大推力 (N)
    
    Returns:
        thrust: 各电机推力 [batch, 4]
    """
    # F = throttle² * F_max
    thrust = throttle ** 2 * max_thrust[..., None]
    return thrust


def motor_thrust_to_torque(
    thrust: Float[Array, "batch 4"],
    torque_coefficient: float = 0.01
) -> Float[Array, "batch 4"]:
    """
    推力到反扭矩转换
    
    反扭矩与推力成正比:
        τ = k_m * F
    
    电机旋转方向 (X 型):
        FR(0): CCW (+)
        RL(1): CCW (+)
        FL(2): CW  (-)
        RR(3): CW  (-)
    
    Args:
        thrust: 各电机推力 [batch, 4]
        torque_coefficient: 扭矩系数 (典型 0.01-0.02)
    
    Returns:
        yaw_torque: 各电机产生的偏航力矩
    """
    # 旋转方向符号
    spin_signs = jnp.array([1.0, 1.0, -1.0, -1.0])  # CCW, CCW, CW, CW
    
    # 反扭矩 = 推力 * 系数 * 旋转方向
    yaw_torque = thrust * torque_coefficient * spin_signs
    
    return yaw_torque


# =============================================================================
# 完整电机模型步进
# =============================================================================
def motor_dynamics_complete(
    current_throttle: Float[Array, "batch 4"],
    target_throttle: Float[Array, "batch 4"],
    tau_up: Float[Array, "batch 4"],
    tau_down: Float[Array, "batch 4"],
    deadband: Float[Array, "batch 4"],
    motor_loss: Float[Array, "batch 4"],
    max_thrust: Float[Array, "batch"],
    dt: float = 0.002
) -> Tuple[Float[Array, "batch 4"], Float[Array, "batch 4"], Float[Array, "batch 4"]]:
    """
    完整电机动力学模型
    
    包含:
    1. 非对称一阶滞后
    2. 死区
    3. 效率差异
    4. 推力计算
    5. 反扭矩计算
    
    Args:
        current_throttle: 当前油门
        target_throttle: 目标油门
        tau_up, tau_down: 时间常数
        deadband: 死区阈值
        motor_loss: 效率差异
        max_thrust: 最大推力
        dt: 时间步长
    
    Returns:
        new_throttle: 更新后的油门
        thrust: 各电机推力
        yaw_torque: 各电机偏航力矩
    """
    # 1. 一阶滞后
    new_throttle = motor_first_order_lag(
        current_throttle, target_throttle, tau_up, tau_down, dt
    )
    
    # 2. 应用死区
    effective_throttle = motor_with_deadband(new_throttle, deadband)
    
    # 3. 效率差异
    effective_throttle = motor_with_efficiency_loss(
        effective_throttle, motor_loss
    )
    
    # 4. 计算推力
    thrust = motor_throttle_to_thrust(effective_throttle, max_thrust)
    
    # 5. 计算反扭矩
    yaw_torque = motor_thrust_to_torque(thrust)
    
    return new_throttle, thrust, yaw_torque


# =============================================================================
# 响应时间估算 (用于 PF-GDR)
# =============================================================================
def estimate_motor_bandwidth(tau_up: float, tau_down: float) -> float:
    """
    估算电机-3dB 带宽
    
    对于一阶系统: f_3dB = 1 / (2π * τ)
    
    Returns:
        bandwidth: 近似带宽 (Hz)
    """
    tau_avg = (tau_up + tau_down) / 2.0
    bandwidth = 1.0 / (2.0 * jnp.pi * tau_avg)
    return bandwidth
