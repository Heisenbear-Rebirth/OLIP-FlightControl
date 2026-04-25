"""
AeroCat v18.0 Sensor Models

IMU 传感器模型:
- 白噪声 (ARW/VRW)
- 偏置不稳定性
- 温度效应 (简化)
- 振动耦合 (简化)

针对 v18.0 PhysState 结构优化。
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
from typing import Tuple


# =============================================================================
# IMU 噪声模型
# =============================================================================
def imu_noise_model(
    true_gyro: Float[Array, "batch 3"],
    true_accel: Float[Array, "batch 3"],
    gyro_noise_std: Float[Array, "batch"],
    accel_noise_std: Float[Array, "batch"],
    gyro_bias: Float[Array, "batch 3"],
    accel_bias: Float[Array, "batch 3"],
    key: PRNGKeyArray
) -> Tuple[Float[Array, "batch 3"], Float[Array, "batch 3"]]:
    """
    IMU 噪声模型
    
    包含:
    - 白噪声 (高斯)
    - 固定偏置
    
    Args:
        true_gyro: 真实角速度 (rad/s)
        true_accel: 真实加速度 (m/s²)
        gyro_noise_std: 陀螺仪噪声标准差 (rad/s)
        accel_noise_std: 加速度计噪声标准差 (m/s²)
        gyro_bias: 陀螺仪偏置 (rad/s)
        accel_bias: 加速度计偏置 (m/s²)
        key: JAX 随机密钥
    
    Returns:
        measured_gyro: 测量角速度
        measured_accel: 测量加速度
    """
    batch_size = true_gyro.shape[0]
    k1, k2 = jax.random.split(key)
    
    # 生成噪声
    gyro_noise = jax.random.normal(k1, (batch_size, 3)) * gyro_noise_std[..., None]
    accel_noise = jax.random.normal(k2, (batch_size, 3)) * accel_noise_std[..., None]
    
    # 测量值 = 真值 + 偏置 + 噪声
    measured_gyro = true_gyro + gyro_bias + gyro_noise
    measured_accel = true_accel + accel_bias + accel_noise
    
    return measured_gyro, measured_accel


def imu_simple_update(
    true_gyro: Float[Array, "batch 3"],
    true_accel: Float[Array, "batch 3"],
    gyro_bias: Float[Array, "batch 3"],
    accel_bias: Float[Array, "batch 3"]
) -> Tuple[Float[Array, "batch 3"], Float[Array, "batch 3"]]:
    """
    简化 IMU 更新 (无噪声，只有偏置)
    
    用于确定性测试。
    """
    measured_gyro = true_gyro + gyro_bias
    measured_accel = true_accel + accel_bias
    
    return measured_gyro, measured_accel


# =============================================================================
# 延迟模型 (用于 Ring Buffer)
# =============================================================================
def apply_sensor_delay(
    buffer: Float[Array, "batch buffer_size 3"],
    current_value: Float[Array, "batch 3"],
    write_idx: int,
    delay_steps: int,
    buffer_size: int = 25
) -> Tuple[Float[Array, "batch buffer_size 3"], Float[Array, "batch 3"]]:
    """
    应用传感器延迟 (环形缓冲区)
    
    Args:
        buffer: 环形缓冲区
        current_value: 当前值
        write_idx: 写入索引
        delay_steps: 延迟步数
        buffer_size: 缓冲区大小
    
    Returns:
        new_buffer: 更新后的缓冲区
        delayed_value: 延迟后的值
    """
    batch_size = current_value.shape[0]
    
    # 写入当前值
    new_idx = write_idx % buffer_size
    new_buffer = buffer.at[:, new_idx].set(current_value)
    
    # 读取延迟值
    read_idx = (write_idx - delay_steps) % buffer_size
    delayed_value = new_buffer[:, read_idx]
    
    return new_buffer, delayed_value
