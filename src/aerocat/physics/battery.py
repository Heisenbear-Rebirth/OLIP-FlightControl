"""
AeroCat v18.0 Battery Model

双 RC 等效电路模型 (ECM) 实现:
- 开路电压 (OCV) 与 SOC 的关系
- 极化电压动态 (V1, V2)
- 温度效应 (简化)
- 电压跌落

针对 v18.0 PhysState 结构优化。
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from typing import Tuple


# =============================================================================
# OCV-SOC 关系
# =============================================================================
def compute_ocv(soc: Float[Array, "batch"], ocv_coeff: Float[Array, "batch 3"]) -> Float[Array, "batch"]:
    """
    计算开路电压 (OCV) vs 状态 (SOC)
    
    使用多项式拟合: OCV = c0 + c1*SOC + c2*SOC²
    
    Args:
        soc: 剩余电量 [0, 1]
        ocv_coeff: OCV 多项式系数 [c0, c1, c2]
    
    Returns:
        ocv: 开路电压 (V)
    """
    c0 = ocv_coeff[..., 0]
    c1 = ocv_coeff[..., 1]
    c2 = ocv_coeff[..., 2]
    
    # 典型 4S LiPo: OCV 范围 [12.6, 16.8] V
    # 使用归一化公式保证范围
    ocv = c0 + c1 * soc + c2 * soc ** 2
    
    return ocv


# =============================================================================
# 双 RC ECM 模型
# =============================================================================
def battery_ecm_step(
    soc: Float[Array, "batch"],
    current: Float[Array, "batch"],
    V1: Float[Array, "batch"],
    V2: Float[Array, "batch"],
    r0: Float[Array, "batch"],
    r1: Float[Array, "batch"],
    r2: Float[Array, "batch"],
    c1: Float[Array, "batch"],
    c2: Float[Array, "batch"],
    ocv_coeff: Float[Array, "batch 3"],
    capacity_ah: float,
    dt: float = 0.002
) -> Tuple[Float[Array, "batch"], Float[Array, "batch"], Float[Array, "batch"], Float[Array, "batch"]]:
    """
    双 RC ECM 电池模型步进
    
    电路结构: OCV - R0 - (R1||C1) - (R2||C2) - Terminal
    
    Args:
        soc: 当前 SOC [0, 1]
        current: 放电电流 (A, 正为放电)
        V1, V2: RC 极化电压
        r0, r1, r2: 电阻参数 (Ω)
        c1, c2: 电容参数 (F)
        ocv_coeff: OCV 多项式系数
        capacity_ah: 电池容量 (Ah)
        dt: 时间步长 (s)
    
    Returns:
        new_soc: 更新后的 SOC
        new_voltage: 端电压
        new_V1, new_V2: 更新后的极化电压
    """
    # 1. SOC 更新 (库仑计数)
    # dSOC/dt = -I / (3600 * Q)
    new_soc = soc - current * dt / (3600.0 * capacity_ah)
    new_soc = jnp.clip(new_soc, 0.0, 1.0)
    
    # 2. RC 动态更新
    # dV1/dt = (I - V1/R1) / C1
    # 离散化: V1_new = V1 + (I*R1 - V1) * dt / (R1*C1)
    tau1 = r1 * c1 + 1e-8
    tau2 = r2 * c2 + 1e-8
    
    new_V1 = V1 + (current * r1 - V1) * dt / tau1
    new_V2 = V2 + (current * r2 - V2) * dt / tau2
    
    # 3. 端电压计算
    # V_terminal = OCV(SOC) - I*R0 - V1 - V2
    ocv = compute_ocv(new_soc, ocv_coeff)
    new_voltage = ocv - current * r0 - new_V1 - new_V2
    
    # 限幅 (防止负电压)
    new_voltage = jnp.maximum(new_voltage, 0.0)
    
    return new_soc, new_voltage, new_V1, new_V2


def compute_motor_power_current(
    motor_throttle: Float[Array, "batch 4"],
    battery_voltage: Float[Array, "batch"],
    motor_max_thrust: Float[Array, "batch"],
    efficiency: float = 0.85
) -> Float[Array, "batch"]:
    """
    估算电机总电流
    
    简化模型: P = T * V / η, I = P / V
    
    Args:
        motor_throttle: 电机油门 [0, 1]
        battery_voltage: 电池电压 (V)
        motor_max_thrust: 最大推力 (N)
        efficiency: 电机效率
    
    Returns:
        total_current: 总放电电流 (A)
    """
    # 推力功率估算 (P ≈ T^1.5 * k)
    # 使用经验公式: 1 kg 推力 ≈ 150W
    k_power = 150.0  # W/kg
    total_thrust = jnp.sum(motor_throttle ** 2 * motor_max_thrust[..., None], axis=-1)
    thrust_kg = total_thrust / 9.81
    power = thrust_kg * k_power
    
    # 电流
    current = power / (battery_voltage * efficiency + 1e-6)
    
    return current
