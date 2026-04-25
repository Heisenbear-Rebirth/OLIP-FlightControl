"""
AeroCat v18.0 PF-GDR Parameter Generator (The "Drone Factory")

基于物理约束的生成式领域随机化 (Physics-First Generative Domain Randomization)。
使用 5-Step 生成链确保所有生成的无人机物理有效。

参考: 05_训练环境_PF-GDR领域随机化_v18.0.md

所有函数为纯 JAX，支持 jit/vmap，GPU 执行。
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PRNGKeyArray
from typing import Tuple
from functools import partial

from ..core.state import (
    PhysParams, L1Config, L3Config, EnvParams,
    compute_mixer_matrix
)
from ..config import PhysicsConfig
from ..physics.turbulence import randomize_gust_params


# =============================================================================
# 物理常量 (PF-GDR 核心)
# =============================================================================
C_MACH = 0.65        # 桨尖目标马赫数
V_SOUND = 340.0      # 声速 (m/s)
D_DRAG = 3.0e-6      # 阻力系数经验值
GRAVITY = 9.81       # 重力加速度


# =============================================================================
# 分布工具函数
# =============================================================================
def log_uniform(key: PRNGKeyArray, shape: Tuple, low: float, high: float) -> Float[Array, "..."]:
    """对数均匀分布采样"""
    u = jax.random.uniform(key, shape)
    return jnp.exp(jnp.log(low) + u * (jnp.log(high) - jnp.log(low)))


def uniform(key: PRNGKeyArray, shape: Tuple, low: float, high: float) -> Float[Array, "..."]:
    """均匀分布采样"""
    return jax.random.uniform(key, shape, minval=low, maxval=high)


def normal_clipped(key: PRNGKeyArray, shape: Tuple, mean: float, std: float, 
                   clip_min: float = None, clip_max: float = None) -> Float[Array, "..."]:
    """截断正态分布"""
    x = jax.random.normal(key, shape) * std + mean
    if clip_min is not None or clip_max is not None:
        x = jnp.clip(x, clip_min, clip_max)
    return x


# =============================================================================
# Step 1: 几何骨架 (Geometry Base)
# =============================================================================
def generate_geometry(key: PRNGKeyArray, batch_size: int, config: PhysicsConfig) -> Tuple[dict, PRNGKeyArray]:
    """
    Step 1: 几何骨架生成
    
    参考: PF-GDR Step 1
    
    Returns:
        geometry: dict with prop_diameter, arm_length, frame_angle
    """
    keys = jax.random.split(key, 6)
    
    # 1. 螺旋桨直径 (对数均匀)
    d_min, d_max = config.prop_diameter_range
    prop_diameter = log_uniform(keys[0], (batch_size,), d_min, d_max)
    
    # 2. 机臂长度: L = D/2 × Uniform(1.05, 1.5)
    arm_scale = uniform(keys[1], (batch_size,), 1.05, 1.5)
    arm_length = prop_diameter / 2.0 * arm_scale
    
    # 3. 机架夹角: [30°, 60°]
    frame_angle_deg = uniform(keys[2], (batch_size,), 30.0, 60.0)
    frame_angle = jnp.deg2rad(frame_angle_deg)
    
    # 4. 机臂长度缩放 (v17 保留): [0.75, 1.25] per arm
    arm_length_scale = uniform(keys[3], (batch_size, 4), 0.85, 1.15)
    
    # 5. 机臂角度抖动: N(0, 5°)
    arm_angle_jitter = normal_clipped(keys[4], (batch_size, 4), 0.0, 5.0, -15.0, 15.0)
    arm_angle_jitter = jnp.deg2rad(arm_angle_jitter)
    
    geometry = {
        'prop_diameter': prop_diameter,
        'arm_length': arm_length,
        'frame_angle': frame_angle,
        'arm_length_scale': arm_length_scale,
        'arm_angle_jitter': arm_angle_jitter,
    }
    
    return geometry, keys[5]


# =============================================================================
# Step 2: 动力系统匹配 (Power Matching)
# =============================================================================
def generate_power_system(
    key: PRNGKeyArray,
    batch_size: int,
    geometry: dict,
    config: PhysicsConfig
) -> Tuple[dict, PRNGKeyArray]:
    """
    Step 2: 动力系统生成
    
    参考: PF-GDR Step 2 (已修正单位问题)
    
    关键物理约束:
    1. 桨尖速度不超过 0.7 马赫 (约 240 m/s)
    2. 推力与 D³×n² 成正比 (静态推力理论)
    3. 大桨配低 KV，小桨配高 KV
    """
    keys = jax.random.split(key, 8)
    
    D = geometry['prop_diameter']  # 单位: 米
    
    # =================================================================
    # 1. 电池系统
    # =================================================================
    # 电池串数与尺寸关联 (经验规则)
    # 2寸(0.05m) -> 2S, 5寸(0.127m) -> 4S, 10寸(0.254m) -> 6S, 15寸(0.38m) -> 12S
    # 使用分段线性: cells = 20 * D + 1
    cells_base = 20.0 * D + 1.0
    cells_noise = uniform(keys[0], (batch_size,), -0.5, 0.5)
    battery_cells = jnp.clip(jnp.round(cells_base + cells_noise), 2, 12).astype(jnp.int32)
    battery_cells_float = battery_cells.astype(jnp.float32)
    
    # 电压
    nominal_voltage = battery_cells_float * 4.2  # 满电
    v_min, v_max = config.voltage_per_cell_range
    voltage_per_cell = uniform(keys[1], (batch_size,), v_min, v_max)
    initial_voltage = battery_cells_float * voltage_per_cell
    
    # =================================================================
    # 2. 电机 RPM (从桨尖速度约束推导)
    # =================================================================
    # 桨尖速度: V_tip = π × D × n = π × D × (RPM/60)
    # 马赫限制: V_tip_max = 0.65 × 340 ≈ 221 m/s
    # 推导: RPM_max = (V_tip_max × 60) / (π × D)
    
    V_TIP_MAX = C_MACH * V_SOUND  # 221 m/s
    
    # 采样实际工作桨尖速度 (允许一定余量)
    v_tip_actual = uniform(keys[2], (batch_size,), 0.5 * V_TIP_MAX, 0.95 * V_TIP_MAX)
    
    max_rpm = (v_tip_actual * 60.0) / (jnp.pi * D)
    
    # KV = RPM / V
    motor_kv = max_rpm / nominal_voltage
    motor_kv = jnp.clip(motor_kv, 500, 8000)  # 物理范围
    
    # 自检: 小桨高 RPM, 大桨低 RPM
    # D=0.05m -> RPM ≈ 40000-80000
    # D=0.40m -> RPM ≈ 5000-10000
    
    # =================================================================
    # 3. 最大推力 (Standard Propeller Theory - SI 单位)
    # =================================================================
    # 标准公式: F = Ct × ρ × n² × D⁴
    # 
    # 其中:
    #   Ct = 推力系数 (无量纲) ∈ [0.08, 0.12]
    #   ρ  = 空气密度 = 1.225 kg/m³
    #   n  = 转速 (revs/second) = RPM / 60
    #   D  = 直径 (meters)
    #
    # 自检: 5寸桨 (0.127m) @ 25000 RPM:
    #   n = 25000/60 = 416.7 rps
    #   F = 0.1 × 1.225 × 416.7² × 0.127⁴
    #   F = 0.1 × 1.225 × 173611 × 0.00026
    #   F ≈ 5.5 N (约 560g)  ✓ 物理合理
    
    RHO = config.air_density  # 空气密度 kg/m³
    
    # 采样推力系数 (不同桨叶设计)
    ct_min, ct_max = config.thrust_coeff_range
    Ct = uniform(keys[3], (batch_size,), ct_min, ct_max)
    
    n = max_rpm / 60.0  # 转速 (rps)
    
    max_motor_thrust = Ct * RHO * (n ** 2) * (D ** 4)
    
    # 合理范围限幅 (防止极端值)
    max_motor_thrust = jnp.clip(max_motor_thrust, 0.2, 80.0)  # 0.2N ~ 80N per motor
    total_max_thrust = max_motor_thrust * 4.0
    
    # =================================================================
    # 4. 电机响应时间
    # =================================================================
    # 大桨惯量大，响应慢
    tau_base = uniform(keys[6], (batch_size,), 0.015, 0.04)
    size_factor = jnp.sqrt(D / 0.127)  # 以 5 寸为基准
    motor_tau_up = tau_base * size_factor
    motor_tau_up = jnp.clip(motor_tau_up, 0.01, 0.15)
    
    # 减速比加速慢 (无主动刹车)
    tau_ratio = uniform(keys[4], (batch_size,), 1.5, 4.0)
    motor_tau_down = motor_tau_up * tau_ratio
    
    # 扩展到 4 电机
    motor_tau_up_4 = jnp.tile(motor_tau_up[:, None], (1, 4))
    motor_tau_down_4 = jnp.tile(motor_tau_down[:, None], (1, 4))
    
    # 5. VAC: Force-to-Torque Ratio (sigma)
    # sigma_base = 0.015 * sqrt(D / 0.127)
    # sigma = sigma_base * Uniform(0.7, 1.4)
    sigma_base = 0.015 * jnp.sqrt(D / 0.127)
    sigma_noise = uniform(keys[5], (batch_size,), 0.7, 1.4)
    sigma = sigma_base * sigma_noise
    force_to_torque_ratio = jnp.tile(sigma[:, None], (1, 4))

    power = {
        'battery_cells': battery_cells,
        'nominal_voltage': nominal_voltage,
        'initial_voltage': initial_voltage,
        'motor_kv': motor_kv,
        'max_rpm': max_rpm,
        'motor_max_thrust': max_motor_thrust,
        'total_max_thrust': total_max_thrust,
        'motor_tau_up': motor_tau_up_4,
        'motor_tau_down': motor_tau_down_4,
        'force_to_torque_ratio': force_to_torque_ratio,
    }
    
    return power, keys[7]


# =============================================================================
# Step 3: 质量生成 (Mass Generation)
# =============================================================================
def generate_mass(
    key: PRNGKeyArray,
    batch_size: int,
    geometry: dict,
    power: dict,
    config: PhysicsConfig,
    curriculum_lambda: float = 0.0
) -> Tuple[dict, PRNGKeyArray]:
    """
    Step 3: 质量生成 + 约束校验
    
    Args:
        curriculum_lambda: 课程进度 (0.0~1.0), 用于缩放 TWR 难度
        
    参考: PF-GDR Step 3
    
    公式:
        TWR ∈ [1.5, 12.0] (Full)
        At lambda=0: TWR ∈ [2.0, 3.0] (Easy)
        M = F_max / (9.81 × TWR)
        
    约束:
        ρ_min = 150 × L^(-0.5)
        ρ_max = 1500 × L^(-0.2)
        M_min = 2.0 × D^3.0
    """
    keys = jax.random.split(key, 5)
    
    D = geometry['prop_diameter']
    L = geometry['arm_length']
    F_max = power['total_max_thrust']
    
    # 1. 采样 TWR (Curriculum Scaled - 分布范围扩大模式)
    # 参考: 06_训练环境_APDC课程化学习_v18.0.md
    # 设计理念: λ=0 时 TWR 集中在中心值附近，λ=1 时 TWR 覆盖完整范围
    twr_min_final, twr_max_final = config.twr_range  # typically [1.5, 12.0]
    
    # 中心值: 完整范围的中点 (对数空间更合理，但简化为线性)
    twr_center = (twr_min_final + twr_max_final) / 2.0  # 约 6.75
    
    # λ=0 时的半宽度 (窄分布，易于学习)
    twr_halfwidth_easy = 0.5
    
    # λ=1 时的半宽度 (完整分布)
    twr_halfwidth_full = (twr_max_final - twr_min_final) / 2.0
    
    # 随 λ 扩大分布范围
    twr_halfwidth = twr_halfwidth_easy + (twr_halfwidth_full - twr_halfwidth_easy) * curriculum_lambda
    
    # 计算当前 min/max
    current_twr_min = twr_center - twr_halfwidth
    current_twr_max = twr_center + twr_halfwidth
    
    # 确保不超出物理边界
    current_twr_min = jnp.maximum(current_twr_min, twr_min_final)
    current_twr_max = jnp.minimum(current_twr_max, twr_max_final)

    twr = uniform(keys[0], (batch_size,), current_twr_min, current_twr_max)
    
    # 2. 计算质量 (从推力和 TWR 直接推导)
    mass = F_max / (config.gravity * twr)
    
    # 3. 宽松约束 (只防止极端值)
    # 最小质量: 20g (微型无人机)
    # 最大质量: 30kg (大型载机)
    mass = jnp.clip(mass, 0.02, 30.0)
    
    # 4. 重算 TWR 确保合理
    twr_actual = F_max / (config.gravity * mass)
    twr_actual = jnp.clip(twr_actual, 1.0, 25.0)
    
    # 4. 重心偏移: 与课程进度关联
    # λ=0 时 cog_offset=0 (完美重心，易于悬停)
    # λ=1 时 cog_offset∈[-0.05, 0.05]m (最大不对称)
    # 这对于 Level 0 悬停阶段至关重要，避免初期翻车
    cog_offset_max = 0.05 * curriculum_lambda
    cog_offset = uniform(keys[1], (batch_size, 3), -cog_offset_max, cog_offset_max)
    
    mass_params = {
        'mass': mass,
        'twr': twr_actual,
        'cog_offset': cog_offset,
    }
    
    return mass_params, keys[4]


# =============================================================================
# Step 4: 转动惯量 (Inertia Physics)
# =============================================================================
def generate_inertia(
    key: PRNGKeyArray,
    batch_size: int,
    geometry: dict,
    mass_params: dict
) -> Tuple[dict, PRNGKeyArray]:
    """
    Step 4: 转动惯量计算
    
    参考: PF-GDR Step 4
    
    公式:
        γ ∈ [0.4, 1.5] (惯量分布因子)
        I_base = M × L²
        I_xx = I_base × γ × sin(θ)
        I_yy = I_base × γ × cos(θ)
        I_zz ≈ (I_xx + I_yy) × Uniform(1.0, 1.8)
    """
    keys = jax.random.split(key, 4)
    
    M = mass_params['mass']
    L = geometry['arm_length']
    theta = geometry['frame_angle']
    
    # 1. 惯量分布因子
    gamma = uniform(keys[0], (batch_size,), 0.4, 1.5)
    
    # 2. 基础惯量: I_base = M × L²
    I_base = M * (L ** 2)
    
    # 3. 三轴惯量
    I_xx = I_base * gamma * jnp.sin(theta)
    I_yy = I_base * gamma * jnp.cos(theta)
    
    # 垂直轴定理约束
    izz_factor = uniform(keys[1], (batch_size,), 1.0, 1.8)
    I_zz = (I_xx + I_yy) * izz_factor
    
    # 确保正值
    I_xx = jnp.maximum(I_xx, 1e-6)
    I_yy = jnp.maximum(I_yy, 1e-6)
    I_zz = jnp.maximum(I_zz, 1e-6)
    
    inertia = {
        'gamma': gamma,
        'inertia_xx': I_xx,
        'inertia_yy': I_yy,
        'inertia_zz': I_zz,
    }
    
    return inertia, keys[3]


# =============================================================================
# Step 5: 内环参数与极限 (Inner Loop & Limits)
# =============================================================================
def generate_control_params(
    key: PRNGKeyArray,
    batch_size: int,
    geometry: dict,
    power: dict,
    inertia: dict
) -> Tuple[dict, dict, PRNGKeyArray]:
    """
    Step 5: 控制参数生成
    
    参考: PF-GDR Step 5
    
    L3 PID:
        τ_max = 0.5 × F_max × L × sin(45°)
        I_norm = I_xx / τ_max
        K_p_base ≈ 20.0 × I_norm
        K_d_base ≈ 0.05 × K_p_base
        
    L1 PID:
        K_p_z ∈ [1.0, 4.0]
        K_i_z ∈ [0.2, 1.5]
    """
    keys = jax.random.split(key, 12)
    
    F_max = power['total_max_thrust']
    L = geometry['arm_length']
    I_xx = inertia['inertia_xx']
    
    # =================================================================
    # L3 Rate Loop PID
    # =================================================================
    # 修正增益计算公式 (v18.0.2 - 归一化修复)
    # 
    # 问题: 之前的实现 Kp = bandwidth * I 产生力矩单位输出 (N·m)
    # 但混控器期望归一化指令 u ∈ [-1, 1]
    # 
    # 修复: 将 PID 输出除以 τ_max 归一化
    # Kp_normalized = (bandwidth * I) / τ_max
    #
    # 这样 PID 输出 = Kp_normalized * ω_err 将在合理范围内
    # 当 ω_err = τ_max / (bandwidth * I) 时，输出 = 1.0
    
    # 目标带宽: 50 rad/s² per rad/s (保守值)
    target_bandwidth = 50.0
    
    # =========================================================================
    # v18.4: 三轴独立归一化 (匹配 BetaFlight-style mixer [±1,±1,±1,±1])
    #
    # 原理: PID 输出 = 1.0 时，对应的物理力矩为 tau_per_unit
    #   - Roll:  tau = F_total * sin(angle) * arm_length  (力臂力矩)
    #   - Pitch: tau = F_total * cos(angle) * arm_length  (力臂力矩)
    #   - Yaw:   tau = F_total * sigma                    (反扭矩)
    #
    # Kp = (bandwidth * I) / tau_per_unit
    # =========================================================================
    frame_angle = geometry['frame_angle']
    A = jnp.sin(frame_angle)  # Roll 力臂系数
    B = jnp.cos(frame_angle)  # Pitch 力臂系数
    I_yy = inertia['inertia_yy']
    I_zz = inertia['inertia_zz']
    sigma = power['force_to_torque_ratio'][:, 0]  # [batch]
    
    # 各轴 tau_per_unit_PID (PID=1.0 时的等效物理力矩)
    tau_roll  = jnp.maximum(F_max * A * L, 1e-6)
    tau_pitch = jnp.maximum(F_max * B * L, 1e-6)
    tau_yaw   = jnp.maximum(F_max * sigma, 1e-6)
    
    # 基准增益: Kp = bandwidth * I / tau
    Kp_base_roll  = jnp.maximum((target_bandwidth * I_xx)  / tau_roll,  0.2)
    Kp_base_pitch = jnp.maximum((target_bandwidth * I_yy)  / tau_pitch, 0.2)
    Kp_base_yaw   = jnp.maximum((target_bandwidth * I_zz)  / tau_yaw,   0.2)
    
    Kd_base_roll  = 0.02 * Kp_base_roll
    Kd_base_pitch = 0.02 * Kp_base_pitch
    Kd_base_yaw   = 0.02 * Kp_base_yaw
    
    # 随机化 (±50%, 对数均匀)
    kp_scale = log_uniform(keys[0], (batch_size,), 0.5, 2.0)
    kd_scale = log_uniform(keys[1], (batch_size,), 0.5, 2.0)
    ki_scale = uniform(keys[2], (batch_size,), 0.1, 0.3)  # I/P 比例 10~30%
    
    # 按轴构建增益 [roll, pitch, yaw]
    l3_gains_p = jnp.stack([
        Kp_base_roll  * kp_scale,
        Kp_base_pitch * kp_scale,
        Kp_base_yaw   * kp_scale,
    ], axis=-1)  # [batch, 3]
    
    l3_gains_d = jnp.stack([
        Kd_base_roll  * kd_scale,
        Kd_base_pitch * kd_scale,
        Kd_base_yaw   * kd_scale,
    ], axis=-1)
    
    l3_gains_i = l3_gains_p * ki_scale[:, None]  # I = P * ki_ratio
    
    # 轴间差异 (±20%)
    axis_variation = uniform(keys[3], (batch_size, 3), 0.8, 1.2)
    l3_gains_p = l3_gains_p * axis_variation
    l3_gains_i = l3_gains_i * axis_variation
    l3_gains_d = l3_gains_d * axis_variation
    
    # 向后兼容: tau_max 用于后续 omega_max 计算
    tau_max = tau_roll
    
    # =================================================================
    # L1 Altitude/Velocity PID
    # =================================================================
    l1_kp_z = uniform(keys[4], (batch_size,), 1.0, 4.0)
    l1_ki_z = uniform(keys[5], (batch_size,), 0.2, 1.5)
    l1_kd_z = log_uniform(keys[6], (batch_size,), 0.01, 0.2)
    l1_k_map = uniform(keys[7], (batch_size,), 2.0, 6.0)
    
    # =================================================================
    # 物理最大角速度 (用于归一化)
    # =================================================================
    # V_max ≈ sqrt(τ_max / D_drag)
    omega_max = jnp.sqrt(tau_max / (D_DRAG + 1e-10))
    omega_max = jnp.clip(omega_max, 10.0, 100.0)  # 物理极限
    
    # =================================================================
    # 动态混控器矩阵
    # =================================================================
    # 注意: mixer_matrix 在 init_params() 中使用正确的 sigma 计算，此处无需生成
    
    l3_config = {
        'gains_p': l3_gains_p,
        'gains_i': l3_gains_i,
        'gains_d': l3_gains_d,
        'tau_max': tau_max,
        'omega_max': omega_max,
    }
    
    l1_config = {
        'kp_z': l1_kp_z,
        'ki_z': l1_ki_z,
        'kd_z': l1_kd_z,
        'k_map': l1_k_map,
    }
    
    return l3_config, l1_config, keys[11]


# =============================================================================
# 生成次要参数 (Populate Secondary)
# =============================================================================
def generate_secondary_params(
    key: PRNGKeyArray,
    batch_size: int,
    geometry: dict,
    power: dict,
    mass_params: dict,
    config: PhysicsConfig,
    curriculum_lambda: float = 1.0
) -> Tuple[dict, PRNGKeyArray]:
    """
    生成次要物理参数 (噪声、阻力、湍流等)
    
    参考: PF-GDR Doc Section 3.x
    """
    keys = jax.random.split(key, 25)
    
    # =================================================================
    # 空气动力学
    # =================================================================
    w_min, w_max = config.wind_speed_range
    # Scale wind speed strictly by curriculum
    w_max_scaled = w_min + (w_max - w_min) * curriculum_lambda
    wind_speed_mean = uniform(keys[0], (batch_size,), w_min, w_max_scaled)
    wind_direction = uniform(keys[1], (batch_size,), 0.0, 2 * jnp.pi)
    
    # Turbulence Curriculum: Scale max intensity by lambda
    # Range [t_min * lambda, t_max * lambda]? No, better: [t_min, t_max * lambda]
    # User plan: "Scale turbulence_intensity sampling range using curriculum_lambda"
    t_min, t_max = config.turbulence_intensity_range
    # Ensure min < max even when lambda is small
    t_max_scaled = t_min + (t_max - t_min) * curriculum_lambda
    turbulence_intensity = uniform(keys[2], (batch_size,), t_min, t_max_scaled)
    
    
    gust_intensity = uniform(keys[3], (batch_size,), 1.5, 3.5)
    
    # Generate Gust Params (Episode-constant)
    # Assume episode starts at t=0
    gust_params = randomize_gust_params(
        batch_size, jnp.zeros((batch_size,)), curriculum_lambda, keys[20]
    )
    
    air_density = uniform(keys[4], (batch_size,), 0.9, 1.3)
    
    # 机身阻力 (各向异性)
    drag_xy = log_uniform(keys[5], (batch_size,), 0.1, 1.5)
    drag_z_factor = uniform(keys[6], (batch_size,), 1.5, 3.0)
    drag_z = drag_xy * drag_z_factor
    drag_coeff_xyz = jnp.stack([drag_xy, drag_xy, drag_z], axis=-1)
    
    # 旋转阻力 (对自旋恢复关键)
    # v18.4: 参考真实四旋翼数据调整范围 (旧: 0.1~1.5, 高出真实值 10-100×)
    #   - 0.005: 清洁微型机架 (whoop/micro)
    #   - 0.15:  大型机架 + 云台载荷 + 螺旋桨保护罩
    #   - 30× log-uniform 范围，保持域随机化宽度
    rot_drag_coeff = log_uniform(keys[7], (batch_size,), 0.005, 0.15)
    rot_drag_coeff_xyz = jnp.tile(rot_drag_coeff[:, None], (1, 3))
    
    roughness_length = uniform(keys[8], (batch_size,), 0.01, 0.1)
    
    # =================================================================
    # 传感器
    # =================================================================
    sensor_latency = log_uniform(keys[9], (batch_size,), 0.005, 0.050)
    gyro_noise_std = uniform(keys[10], (batch_size,), 0.005, 0.03)
    accel_noise_std = uniform(keys[11], (batch_size,), 0.01, 0.06)
    
    # 偏置 (与课程进度关联)
    # λ=0 时偏置=0 (理想传感器，便于 L3 PID 稳定)
    # λ=1 时恢复正常随机化
    gyro_bias_std = 0.02 * curriculum_lambda
    accel_bias_std = 0.1 * curriculum_lambda
    gyro_bias = normal_clipped(keys[12], (batch_size, 3), 0.0, gyro_bias_std, -0.1, 0.1)
    accel_bias = normal_clipped(keys[13], (batch_size, 3), 0.0, accel_bias_std, -0.5, 0.5)
    
    # =================================================================
    # 电机效率差异 (工艺公差)
    # =================================================================
    motor_loss_max = 0.3 * curriculum_lambda
    motor_loss = uniform(keys[14], (batch_size, 4), 0.0, motor_loss_max)
    motor_deadband = uniform(keys[15], (batch_size, 4), 0.02, 0.10)
    
    # =================================================================
    # 碰撞参数
    # =================================================================
    collision_mag_vel_max = uniform(keys[16], (batch_size,), 2.0, 5.0)
    collision_mag_rate_max = uniform(keys[17], (batch_size,), 10.0, 20.0)
    
    # =================================================================
    # 电池
    # =================================================================
    # 容量与质量关联 (经验: 1kg ≈ 3000mAh)
    capacity_base = mass_params['mass'] * 3000.0
    capacity_noise = uniform(keys[18], (batch_size,), 0.5, 1.5)
    battery_capacity = capacity_base * capacity_noise
    battery_capacity = jnp.clip(battery_capacity, 300.0, 20000.0)
    
    c_rating = uniform(keys[19], (batch_size,), 20.0, 150.0)
    
    # 内阻: R ∝ 1 / (capacity × c_rating)
    internal_resistance = 0.1 / (battery_capacity / 1000.0 * c_rating / 50.0 + 0.1)
    
    # OCV 系数 (4S LiPo 典型)
    S = power['battery_cells'].astype(jnp.float32)
    ocv_c0 = S * 3.0   # 空电压
    ocv_c1 = S * 1.0   # 线性系数
    ocv_c2 = S * 0.2   # 二次系数
    ocv_coeff = jnp.stack([ocv_c0, ocv_c1, ocv_c2], axis=-1)
    
    # RC 参数 (简化)
    r0 = internal_resistance
    r1 = internal_resistance * 0.5
    r2 = internal_resistance * 0.3
    c1 = jnp.full((batch_size,), 1000.0)  # Farads
    c2 = jnp.full((batch_size,), 5000.0)
    
    secondary = {
        # 空气动力学
        'wind_speed_mean': wind_speed_mean,
        'wind_direction': wind_direction,
        'turbulence_intensity': turbulence_intensity,
        'gust_intensity': gust_intensity,
        'air_density': air_density,
        'drag_coeff_xyz': drag_coeff_xyz,
        'rot_drag_coeff': rot_drag_coeff_xyz,
        'roughness_length': roughness_length,
        
        # 传感器
        'sensor_latency': sensor_latency,
        'gyro_noise_std': gyro_noise_std,
        'accel_noise_std': accel_noise_std,
        'gyro_bias': gyro_bias,
        'accel_bias': accel_bias,
        
        # 电机
        'motor_loss': motor_loss,
        'motor_deadband': motor_deadband,
        
        # 碰撞
        'collision_mag_vel_max': collision_mag_vel_max,
        'collision_mag_rate_max': collision_mag_rate_max,
        
        # 电池
        'battery_capacity': battery_capacity,
        'c_rating': c_rating,
        'internal_resistance': internal_resistance,
        'ocv_coeff': ocv_coeff,
        'r0': r0,
        'r1': r1,
        'r2': r2,
        'c1': c1,
        'c2': c2,
        
        # Gust
        'gust_active': gust_params['active'],
        'gust_start_time': gust_params['start_time'],
        'gust_duration': gust_params['duration'],
        'gust_magnitude': gust_params['magnitude'],
        'gust_direction': gust_params['direction'],
    }
    
    return secondary, keys[24]


# =============================================================================
# 完整参数生成 (5-Step Chain)
# =============================================================================
def sample_params(key: PRNGKeyArray, batch_size: int, config: PhysicsConfig, curriculum_lambda: float = 1.0) -> Tuple[PhysParams, L1Config, L3Config]:
    """
    执行 PF-GDR 5-Step 生成链
    
    Args:
        key: JAX 随机密钥
        batch_size: 批量大小
        config: 物理配置
        curriculum_lambda: 课程进度 (default 1.0)
    
    Returns:
        phys_params: 物理参数
        l1_config: L1 层配置
        l3_config: L3 层配置
    """
    
    # Step 1: Geometry
    geometry, key = generate_geometry(key, batch_size, config)
    
    # Step 2: Power
    power, key = generate_power_system(key, batch_size, geometry, config)
    
    # Step 3: Mass
    # Pass curriculum_lambda to scale TWR difficulty
    mass_params, key = generate_mass(key, batch_size, geometry, power, config, curriculum_lambda)
    
    # Step 4: Inertia
    inertia, key = generate_inertia(key, batch_size, geometry, mass_params)
    
    # Step 5: Control
    l3_config_dict, l1_config_dict, key = generate_control_params(
        key, batch_size, geometry, power, inertia  # power 含 force_to_torque_ratio
    )
    
    # Secondary Parameters
    secondary, key = generate_secondary_params(key, batch_size, geometry, power, mass_params, config, curriculum_lambda)
    
    # =================================================================
    # 组装 PhysParams (匹配 core/state.py 定义)
    # =================================================================
    
    # 将 max_thrust 转换为 thrust_coeff (简化)
    # thrust = Ct * omega^2, 假设 omega_max^2 = max_thrust / Ct
    motor_thrust_coeff = jnp.full((batch_size, 4), 0.001)  # 简化系数
    motor_torque_coeff = jnp.full((batch_size, 4), 0.0001)  # 力矩系数 (Legacy)
    force_to_torque_ratio = power['force_to_torque_ratio'] # VAC sigma
    
    phys_params = PhysParams(
        # Geometry
        arm_length=geometry['arm_length'],
        frame_angle=geometry['frame_angle'],
        mass=mass_params['mass'],
        
        # Inertia
        inertia_xx=inertia['inertia_xx'],
        inertia_yy=inertia['inertia_yy'],
        inertia_zz=inertia['inertia_zz'],
        cog_offset=mass_params['cog_offset'],
        
        # Motor
        motor_thrust_coeff=motor_thrust_coeff,
        motor_torque_coeff=motor_torque_coeff,
        force_to_torque_ratio=force_to_torque_ratio,
        motor_tau_up=power['motor_tau_up'],
        motor_tau_down=power['motor_tau_down'],
        motor_max_thrust=power['motor_max_thrust'],
        motor_loss=secondary['motor_loss'],
        motor_deadband=secondary['motor_deadband'],
        
        # Battery (ECM)
        battery_r0=secondary['r0'],
        battery_r1=secondary['r1'],
        battery_r2=secondary['r2'],
        battery_c1=secondary['c1'],
        battery_c2=secondary['c2'],
        battery_ocv_coeff=secondary['ocv_coeff'],
        
        # Aerodynamics
        drag_coeff_xyz=secondary['drag_coeff_xyz'],
        rot_drag_coeff=secondary['rot_drag_coeff'],
        
        # Sensors
        gyro_noise_std=secondary['gyro_noise_std'],
        accel_noise_std=secondary['accel_noise_std'],
        gyro_bias=secondary['gyro_bias'],
        accel_bias=secondary['accel_bias'],
        
        # Latency
        rc_latency=secondary['sensor_latency'],
        imu_latency=secondary['sensor_latency'] * 0.5,
        
        # Environment
        wind_speed_mean=secondary['wind_speed_mean'],
        wind_direction=secondary['wind_direction'],
        turbulence_intensity=secondary['turbulence_intensity'],
        roughness_length=secondary['roughness_length'],
        
        # Gust
        gust_active=secondary['gust_active'],
        gust_start_time=secondary['gust_start_time'],
        gust_duration=secondary['gust_duration'],
        gust_magnitude=secondary['gust_magnitude'],
        gust_direction=secondary['gust_direction'],
        
        # Other
        battery_capacity=secondary['battery_capacity'],
    )
    
    # =================================================================
    # 组装 L1Config (使用生成的批量参数)
    # =================================================================
    # 计算动态悬停油门 (用于 L1 前馈)
    # 
    # v18.0.2 更新: 由于添加了推力线性化 (sqrt 补偿)
    # 混控器输出: th_linear
    # 经过 sqrt 后: th_actual = sqrt(th_linear)
    # 推力: F = th_actual² × F_max = th_linear × F_max
    # 
    # 悬停条件: 4 × F_hover = mg
    #          4 × th_linear × F_max = mg
    #          th_linear = mg / (4 × F_max)
    # 
    # 注意: 不再需要 sqrt
    t_hover_guess = (mass_params['mass'] * 9.81) / (4.0 * power['motor_max_thrust'])
    t_hover_guess = jnp.clip(t_hover_guess, 0.1, 0.9)

    l1_config = L1Config(
        # 显式设置的批量参数
        vz_kp=l1_config_dict['kp_z'],  # [batch]
        vz_ki=l1_config_dict['ki_z'],  # [batch]
        vz_kd=l1_config_dict['kd_z'],  # [batch]
        t_hover_guess=t_hover_guess,   # [batch]
        # 所有默认值必须广播到 [batch] 以保证 JAX scan carry shape 一致
        max_yaw_rate=jnp.full((batch_size,), 100.0 * jnp.pi / 180.0),  # v18.4: 200→100°/s
        max_tilt=jnp.full((batch_size,), 45.0 * jnp.pi / 180.0),
        max_vz=jnp.full((batch_size,), 3.0),
        vz_i_limit=jnp.full((batch_size,), 0.3),
        # v18.2: 移除 vxy_kp/ki/kd/i_limit (L1 不再做水平速度 PID)
        # OU + Poisson Jump 摇杆生成参数 (Roll/Pitch/Throttle)
        ou_theta=jnp.full((batch_size,), 1.5),
        ou_sigma=jnp.full((batch_size,), 0.7),
        ou_mu=jnp.full((batch_size,), 0.0),
        jump_rate=jnp.full((batch_size,), 0.2),
        # v18.4: Yaw 轴独立 OU 参数 (更平缓的航向控制)
        ou_theta_yaw=jnp.full((batch_size,), 0.5),
        ou_sigma_yaw=jnp.full((batch_size,), 0.3),
        jump_rate_yaw=jnp.full((batch_size,), 0.05),
        max_v_cmd_rate=jnp.full((batch_size,), 0.4),
        max_v_cmd_rate_full=jnp.full((batch_size,), 2.0),
        # v18.4: Yaw 命令速率限制 (100°/s/s)
        max_yaw_cmd_rate=jnp.full((batch_size,), 100.0 * jnp.pi / 180.0),
    )
    
    # =================================================================
    # 组装 L3Config (使用生成的批量参数)
    # =================================================================
    l3_config = L3Config(
        kp=l3_config_dict['gains_p'],  # [batch, 3]
        ki=l3_config_dict['gains_i'],  # [batch, 3]
        kd=l3_config_dict['gains_d'],  # [batch, 3]
        # 所有默认值必须广播到 [batch] 以保证 JAX scan carry shape 一致
        i_limit=jnp.full((batch_size,), 0.3),
        dt_sub=jnp.full((batch_size,), 0.002),
    )
    
    return phys_params, l1_config, l3_config


# =============================================================================
# 集成助手函数
# =============================================================================
def init_params(key: PRNGKeyArray, batch_size: int, config: PhysicsConfig = None, curriculum_lambda: float = 1.0) -> EnvParams:
    """
    初始化完整的 EnvParams (包括混控矩阵)
    
    这是提供给 reset_env() 使用的主要接口。
    
    Args:
        key: JAX 随机密钥
        batch_size: 批量大小
        config: 物理配置 (如果为 None，使用默认值)
        curriculum_lambda: 课程进度
    
    Returns:
        EnvParams: 完整的环境参数
    """
    if config is None:
        config = PhysicsConfig()

    # 生成参数
    phys_params, l1_config, l3_config = sample_params(key, batch_size, config, curriculum_lambda)
    
    # 计算混控矩阵 (VAC Update)
    # 使用生成的 force_to_torque_ratio 的平均值? 或者直接传数组?
    # compute_mixer_matrix 支持 batch 数组.
    # 取 sigma = force_to_torque_ratio[:, 0] (假设4电机一致或取平均)
    # 其实 param_generator 产生的是 tile 过的, 所以取 [:, 0] 没问题
    sigma = phys_params.force_to_torque_ratio[..., 0]
    mixer_matrix = compute_mixer_matrix(
        phys_params.frame_angle,
        sigma  # 力矩比 sigma
    )
    
    # 组装 EnvParams
    env_params = EnvParams(
        phys_params=phys_params,
        l1_config=l1_config,
        l3_config=l3_config,
        mixer_matrix=mixer_matrix,
    )
    
    return env_params


# JIT 编译版本
init_params_jit = jax.jit(init_params, static_argnums=(1,))


# =============================================================================
# 调试/验证工具
# =============================================================================
def validate_params(phys_params: PhysParams) -> dict:
    """
    验证生成的参数是否在物理合理范围内
    
    Returns:
        dict: 验证结果
    """
    checks = {}
    
    # TWR 检查
    twr = (phys_params.motor_max_thrust * 4.0) / (GRAVITY * phys_params.mass)
    checks['twr_in_range'] = jnp.all((twr >= 1.2) & (twr <= 20.0))
    checks['twr_mean'] = jnp.mean(twr)
    
    # 惯量正值
    checks['inertia_positive'] = jnp.all(
        (phys_params.inertia_xx > 0) & 
        (phys_params.inertia_yy > 0) & 
        (phys_params.inertia_zz > 0)
    )
    
    # 质量合理
    checks['mass_in_range'] = jnp.all((phys_params.mass >= 0.02) & (phys_params.mass <= 50.0))
    checks['mass_mean'] = jnp.mean(phys_params.mass)
    
    return checks
