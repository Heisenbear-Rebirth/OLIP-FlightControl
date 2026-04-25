"""
AeroCat v18.0 Wind & Turbulence Model

大气扰动模型实现:
- Dryden 连续湍流谱 (MIL-F-8785C)
- 阵风模型 (1-cos Discrete Gust)
- 风剪切 (对数风廓线 Wind Shear)

参考: 05_训练环境_PF-GDR领域随机化_v18.0.md 3.6节
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
from typing import Tuple, Dict


# =============================================================================
# 常量
# =============================================================================
GRAVITY = 9.81
AIR_DENSITY = 1.225  # kg/m³ at sea level


# =============================================================================
# Dryden 湍流模型 (连续功率谱)
# =============================================================================
def dryden_turbulence_step(
    state: Float[Array, "batch 3"],
    velocity: Float[Array, "batch 3"],
    turbulence_intensity: Float[Array, "batch"],
    wind_speed: Float[Array, "batch"],
    altitude: Float[Array, "batch"],
    key: PRNGKeyArray,
    dt: float = 0.002
) -> Tuple[Float[Array, "batch 3"], Float[Array, "batch 3"]]:
    """
    Dryden 连续湍流模型步进
    
    基于 MIL-F-8785C 规范的低空湍流模型。
    用一阶马尔可夫过程逼近 Dryden 功率谱。
    
    公式 (简化为一阶滤波器):
        dq/dt = -q/τ + σ*√(2/τ)*ξ
        τ = L / V (时间常数)
        σ = turbulence_intensity * V (湍流强度)
    
    Args:
        state: 湍流状态 [u, v, w]
        velocity: 飞行速度 [batch, 3]
        turbulence_intensity: 湍流强度 (0.05-0.25)
        wind_speed: 平均风速 (m/s)
        altitude: 飞行高度 (m)
        key: JAX 随机密钥
        dt: 时间步长 (s)
    
    Returns:
        new_state: 更新后的湍流状态
        turbulence: 湍流速度扰动 [u, v, w]
    """
    batch_size = state.shape[0]
    
    # 计算空速
    airspeed = jnp.linalg.norm(velocity, axis=-1) + 1e-6  # 避免除零
    
    # Dryden 尺度长度 (低空近似)
    # L_u = L_v = h / (0.177 + 0.000823*h)^1.2
    # L_w = h
    h_safe = jnp.maximum(altitude, 1.0)  # 最小 1m
    L_u = h_safe / jnp.power(0.177 + 0.000823 * h_safe, 1.2)
    L_v = L_u
    L_w = h_safe
    
    # 时间常数 τ = L / V
    tau_u = L_u / airspeed
    tau_v = L_v / airspeed
    tau_w = L_w / airspeed
    
    # 湍流强度 σ (标准差)
    # 低空: σ_u = σ_v = σ_w = turbulence_intensity * V_wind
    sigma = turbulence_intensity * (wind_speed + 1.0)  # 至少 1 m/s 基准
    
    # 生成白噪声驱动
    noise = jax.random.normal(key, (batch_size, 3))
    
    # 一阶马尔可夫更新
    # q_new = q * exp(-dt/τ) + σ * sqrt(1 - exp(-2*dt/τ)) * ξ
    tau = jnp.stack([tau_u, tau_v, tau_w], axis=-1)
    tau_safe = jnp.maximum(tau, 1e-6)
    
    alpha = jnp.exp(-dt / tau_safe)
    beta = sigma[..., None] * jnp.sqrt(1.0 - alpha ** 2)
    
    new_state = alpha * state + beta * noise
    
    # 输出湍流速度
    turbulence = new_state
    
    return new_state, turbulence





# =============================================================================
# 阵风模型 (Discrete Gust)
# =============================================================================
def gust_model(
    time: Float[Array, "batch"],
    gust_start_time: Float[Array, "batch"],
    gust_duration: Float[Array, "batch"],
    gust_magnitude: Float[Array, "batch"],
    gust_direction: Float[Array, "batch 3"],
    active: Float[Array, "batch"]
) -> Float[Array, "batch 3"]:
    """
    离散阵风模型 (1-cos 形状)
    
    参考: MIL-F-8785C 阵风规范
    
    公式:
        V_gust = 0.5 * V_max * (1 - cos(π * t / T))  if 0 < t < T
        V_gust = 0                                   otherwise
    
    Args:
        time: 当前时间
        gust_start_time: 阵风开始时间
        gust_duration: 阵风持续时间
        gust_magnitude: 阵风最大速度
        gust_direction: 阵风方向 (单位向量)
        active: 阵风激活标志
    
    Returns:
        gust_velocity: 阵风速度 [batch, 3]
    """
    # 相对时间
    t_rel = time - gust_start_time
    
    # 归一化时间 [0, 1]
    t_normalized = t_rel / (gust_duration + 1e-6)
    
    # 1-cos 形状
    shape = 0.5 * (1.0 - jnp.cos(jnp.pi * t_normalized))
    
    # 在有效范围内
    in_range = (t_normalized >= 0.0) & (t_normalized <= 1.0)
    
    # 应用形状
    magnitude = gust_magnitude * shape * in_range.astype(jnp.float32) * active
    
    # 阵风速度
    gust_velocity = gust_direction * magnitude[..., None]
    
    return gust_velocity


# =============================================================================
# 风剪切模型 (Wind Shear)
# =============================================================================
def wind_shear_model(
    altitude: Float[Array, "batch"],
    base_wind_speed: Float[Array, "batch"],
    wind_direction: Float[Array, "batch"],
    roughness_length: Float[Array, "batch"],
    reference_height: float = 10.0
) -> Float[Array, "batch 3"]:
    """
    大气边界层风剪切模型 (对数风廓线)
    
    公式:
        V(h) = V_ref * ln(h / z0) / ln(h_ref / z0)
    
    Args:
        altitude: 飞行高度 (m)
        base_wind_speed: 参考高度风速 (m/s)
        wind_direction: 风向 (rad, 0=北)
        roughness_length: 粗糙度长度 (m)
        reference_height: 参考高度 (m)
    
    Returns:
        wind_velocity: 风速向量 [batch, 3]
    """
    # 安全高度 (避免对数奇点)
    h_safe = jnp.maximum(altitude, roughness_length + 0.1)
    z0_safe = jnp.maximum(roughness_length, 0.001)
    
    # 对数风廓线
    log_profile = jnp.log(h_safe / z0_safe) / jnp.log(reference_height / z0_safe)
    log_profile = jnp.clip(log_profile, 0.0, 3.0)  # 限制范围
    
    # 实际风速
    wind_speed = base_wind_speed * log_profile
    
    # 转换为 NED 向量
    # wind_direction = 0 表示从北向吹 (向南)
    wind_x = -wind_speed * jnp.cos(wind_direction)  # North 分量
    wind_y = -wind_speed * jnp.sin(wind_direction)  # East 分量
    wind_z = jnp.zeros_like(wind_speed)              # Down 分量
    
    wind_velocity = jnp.stack([wind_x, wind_y, wind_z], axis=-1)
    
    return wind_velocity


# =============================================================================
# 完整风场计算 (组合所有模型)
# =============================================================================
def compute_wind_field(
    position: Float[Array, "batch 3"],
    time: Float[Array, "batch"],
    mean_wind_speed: Float[Array, "batch"],
    wind_direction: Float[Array, "batch"],
    turbulence_intensity: Float[Array, "batch"],
    turbulence_state: Float[Array, "batch 3"],
    gust_params: Dict,
    roughness_length: Float[Array, "batch"],
    key: PRNGKeyArray,
    dt: float = 0.002
) -> Tuple[Float[Array, "batch 3"], Float[Array, "batch 3"]]:
    """
    计算完整风场 (平均风 + 湍流 + 阵风)
    
    Args:
        position: 飞机位置 [batch, 3]
        time: 当前时间
        mean_wind_speed: 平均风速 (m/s)
        wind_direction: 风向 (rad)
        turbulence_intensity: 湍流强度
        turbulence_state: 当前湍流状态
        gust_params: 阵风参数字典
        roughness_length: 粗糙度长度
        key: 随机密钥
        dt: 时间步长
    
    Returns:
        wind_velocity: 总风速 [batch, 3]
        new_turbulence_state: 更新后的湍流状态
    """
    batch_size = position.shape[0]
    k1, k2 = jax.random.split(key)
    
    # 高度 (NED: Z 向下为正，取负)
    altitude = -position[..., 2]
    altitude = jnp.maximum(altitude, 0.1)  # 安全下限
    
    # 1. 平均风 (含风剪切)
    mean_wind = wind_shear_model(
        altitude, mean_wind_speed, wind_direction, roughness_length
    )
    
    # 2. Dryden 湍流
    # 使用平均风速近似空速 (假设无人机相对于地面静止/悬停)
    # Airspeed = V_drone - V_wind. If V_drone=0, Airspeed = -V_wind.
    velocity_approx = -mean_wind 
    new_turbulence_state, dryden_turbulence = dryden_turbulence_step(
        turbulence_state,
        velocity_approx,
        turbulence_intensity,
        mean_wind_speed,
        altitude,
        k1,
        dt
    )
    
    # 3. 阵风 (如果参数有效)
    gust_wind = jnp.zeros((batch_size, 3))
    if gust_params is not None and 'active' in gust_params:
        gust_wind = gust_model(
            time,
            gust_params.get('start_time', jnp.zeros(batch_size)),
            gust_params.get('duration', jnp.full((batch_size,), 1.0)),
            gust_params.get('magnitude', jnp.zeros(batch_size)),
            gust_params.get('direction', jnp.tile(jnp.array([1.0, 0.0, 0.0]), (batch_size, 1))),
            gust_params.get('active', jnp.zeros(batch_size))
        )
    
    # 合成总风速
    total_wind = mean_wind + dryden_turbulence + gust_wind
    
    return total_wind, new_turbulence_state


# =============================================================================
# 默认参数生成
# =============================================================================
def create_default_gust_params(batch_size: int) -> Dict:
    """创建默认阵风参数"""
    return {
        'active': jnp.zeros((batch_size,)),
        'start_time': jnp.zeros((batch_size,)),
        'duration': jnp.full((batch_size,), 1.0),
        'magnitude': jnp.zeros((batch_size,)),
        'direction': jnp.tile(jnp.array([1.0, 0.0, 0.0]), (batch_size, 1)),
    }


def randomize_gust_params(
    batch_size: int,
    time: Float[Array, "batch"],
    curriculum_lambda: float,
    key: PRNGKeyArray
) -> Dict:
    """随机化阵风参数"""
    k1, k2, k3, k4, k5, k6 = jax.random.split(key, 6)
    
    # 阵风激活概率 (随课程增加)
    prob = 0.1 * curriculum_lambda  # 最大 10%
    active = jax.random.bernoulli(k1, prob, (batch_size,)).astype(jnp.float32)
    
    # 开始时间
    start_time = time + jax.random.uniform(k2, (batch_size,), minval=0.5, maxval=2.0)
    
    # 持续时间
    duration = jax.random.uniform(k3, (batch_size,), minval=0.5, maxval=1.5)
    
    # 幅度 (随课程增加)
    max_mag = 5.0 * curriculum_lambda
    magnitude = jax.random.uniform(k4, (batch_size,), minval=0.0, maxval=max_mag)
    
    # 随机方向
    # k5, k6 already split
    direction = jax.random.normal(k5, (batch_size, 3))
    direction = direction / (jnp.linalg.norm(direction, axis=-1, keepdims=True) + 1e-6)
    
    return {
        'active': active,
        'start_time': start_time,
        'duration': duration,
        'magnitude': magnitude,
        'direction': direction,
    }
