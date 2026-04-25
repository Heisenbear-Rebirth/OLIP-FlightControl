"""
AeroCat Hover 简化环境

专为交底书验证设计的极简悬停环境:
- 观测: 18维 (本体11D + L3诊断4D + OLIP特征5D，OLIP在训练中模拟生成)
- 动作: 4维 (目标角速度3D + 推力1D)
- 奖励: 3项 (直立 + 高度保持 + 动作平滑)
- 无 L1 制导层、无课程学习、无延迟缓冲、无碰撞注入

复用 v18 底层: physics/dynamics.py, control/l3_controller.py, core/state.py
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Bool, PRNGKeyArray
from typing import Tuple, Dict, Any, NamedTuple
from flax import struct

from ..core.state import (
    PhysState, L3State, PhysParams, L3Config, compute_mixer_matrix
)
from ..control.l3_controller import (
    l3_substep, map_action_to_rates_and_thrust, compute_diagnostics
)
from ..physics.dynamics import physics_dynamics_step
from ..physics.battery import compute_ocv


# =============================================================================
# 简化环境配置
# =============================================================================
@struct.dataclass
class HoverEnvConfig:
    """极简悬停环境配置"""
    
    # 时间配置
    dt_rl: float = 0.02           # RL 步长 (50Hz)
    dt_sub: float = 0.002         # 内环步长 (500Hz)
    num_substeps: int = 10        # 内环子步数
    max_episode_steps: int = 200  # 最大步数 (4秒)
    
    # 观测/动作空间
    obs_dim: int = 20             # 20维观测 (本体11D + L3诊断4D + OLIP特征5D)
    action_dim: int = 4           # 4维动作
    
    # 初始状态 (简化: 小范围随机)
    init_max_tilt: float = 0.3    # 最大初始倾角 (rad, ~17°)
    init_max_rate: float = 1.0    # 最大初始角速度 (rad/s)
    init_height: float = 10.0     # 初始高度 (m)
    
    # 终止条件
    max_tilt_angle: float = 1.0   # 最大允许倾角 (rad, ~57°)
    
    # 域随机化范围
    mass_range: Tuple[float, float] = (0.7, 1.3)       # 质量范围 (kg)
    thrust_coeff_range: Tuple[float, float] = (0.8, 1.2) # 推力系数范围


# =============================================================================
# 时间步结果
# =============================================================================
class HoverTimeStep(NamedTuple):
    """环境步进返回值"""
    obs: Float[Array, "batch 18"]
    reward: Float[Array, "batch"]
    done: Bool[Array, "batch"]
    info: Dict[str, Any]


# =============================================================================
# OLIP 模拟特征生成 (训练时用域随机化参数模拟)
# =============================================================================
def generate_olip_features(
    phys_params: PhysParams,
    batch_size: int
) -> Float[Array, "batch 5"]:
    """
    生成 OLIP 物理特征向量 (5D)
    
    训练时从域随机化参数计算, 模拟 OLIP 标定结果:
    - φ0: 悬停油门 = mg / (4 * F_max), 归一化到 [0, 1]
    - φ1: 推力灵敏度 = F_max / m, 归一化到 [0, 1]
    - φ2: 系统延迟 = (tau_up + tau_down) / 2, 归一化
    - φ3: 阻尼比 = rot_drag_coeff 均值, 归一化
    - φ4: 非对称性 = motor_loss 标准差, 归一化
    """
    # φ0: 悬停油门
    hover_throttle = (phys_params.mass * 9.81) / (4.0 * phys_params.motor_max_thrust)
    phi_0 = jnp.clip(hover_throttle, 0.0, 1.0)
    
    # φ1: 推力灵敏度 (归一化: 典型值 5.0/1.0=5, 范围 ~2-10)
    thrust_sensitivity = phys_params.motor_max_thrust / phys_params.mass
    phi_1 = jnp.clip(thrust_sensitivity / 10.0, 0.0, 1.0)
    
    # φ2: 系统延迟 (归一化: 典型 tau_up=0.03, tau_down=0.08)
    avg_tau = (jnp.mean(phys_params.motor_tau_up, axis=-1) + 
               jnp.mean(phys_params.motor_tau_down, axis=-1)) / 2.0
    phi_2 = jnp.clip(avg_tau / 0.1, 0.0, 1.0)
    
    # φ3: 阻尼比 (归一化)
    avg_rot_drag = jnp.mean(phys_params.rot_drag_coeff, axis=-1)
    phi_3 = jnp.clip(avg_rot_drag / 0.01, 0.0, 1.0)
    
    # φ4: 非对称性 (电机效率差异标准差)
    phi_4 = jnp.std(phys_params.motor_loss, axis=-1)
    phi_4 = jnp.clip(phi_4 / 0.1, 0.0, 1.0)
    
    return jnp.stack([phi_0, phi_1, phi_2, phi_3, phi_4], axis=-1)


# =============================================================================
# 简化域随机化
# =============================================================================
def randomize_params(
    key: PRNGKeyArray,
    batch_size: int,
    config: HoverEnvConfig
) -> Tuple[PhysParams, Float[Array, "batch 4 4"]]:
    """
    简化的域随机化参数生成
    
    仅随机化: 质量、推力系数
    其余使用默认值
    """
    keys = jax.random.split(key, 5)
    
    # 基础参数
    base_params = PhysParams.create_default(batch_size)
    
    # 质量随机化
    mass = jax.random.uniform(
        keys[0], (batch_size,), 
        minval=config.mass_range[0], 
        maxval=config.mass_range[1]
    )
    
    # 推力系数随机化 (乘以基础最大推力)
    thrust_scale = jax.random.uniform(
        keys[1], (batch_size,),
        minval=config.thrust_coeff_range[0],
        maxval=config.thrust_coeff_range[1]
    )
    motor_max_thrust = base_params.motor_max_thrust * thrust_scale
    
    # 小范围电机不对称 (±5%)
    motor_loss = jax.random.uniform(keys[2], (batch_size, 4), minval=0.0, maxval=0.05)
    
    # 更新参数
    params = base_params.replace(
        mass=mass,
        motor_max_thrust=motor_max_thrust,
        motor_loss=motor_loss,
    )
    
    # 计算混控器矩阵
    mixer_matrix = compute_mixer_matrix(
        params.frame_angle,
        jnp.mean(params.force_to_torque_ratio, axis=-1)
    )
    
    return params, mixer_matrix


# =============================================================================
# 观测空间构建 (18维)
# =============================================================================
def build_hover_observation(
    phys_state: PhysState,
    l3_state: L3State,
    olip_features: Float[Array, "batch 5"],
    height_target: float = 10.0
) -> Float[Array, "batch 18"]:
    """
    构建 18 维观测向量
    
    本体感知 (11D):
    - [0:4]   姿态四元数 [qw, qx, qy, qz]
    - [4:7]   机体角速度 [ωx, ωy, ωz] / 35.0
    - [7:10]  机体加速度 [ax, ay, az] / 49.05
    - [10]    高度误差 Δh / 10.0
    
    L3 诊断 (4D):
    - [11:14] PID 积分器归一化 [Ix, Iy, Iz] / i_limit
    - [14]    混控器饱和度
    
    OLIP 物理特征 (5D, episode 内常量):
    - [15:20] [φ0, φ1, φ2, φ3, φ4]
    """
    # 本体感知
    quat = phys_state.quaternion                      # [batch, 4]
    omega_norm = phys_state.angular_velocity / 35.0   # [batch, 3]
    accel_norm = phys_state.imu_accel / 49.05         # [batch, 3]
    
    # 高度误差 (NED: Z向下, 目标高度 = -height_target)
    height_error = (-height_target - phys_state.position[..., 2]) / 10.0  # [batch]
    
    # L3 诊断 (4D: 积分器3D + 饱和度1D)
    pid_integral_norm = l3_state.pid_integral / 0.3   # [batch, 3], i_limit=0.3
    saturation = jnp.clip(l3_state.mixer_saturation, 0.0, 1.0)  # [batch]
    
    obs = jnp.concatenate([
        quat,                                 # [0:4]   姿态四元数
        omega_norm,                           # [4:7]   角速度
        accel_norm,                           # [7:10]  加速度
        height_error[..., None],              # [10]    高度误差
        pid_integral_norm,                    # [11:14] PID 积分器
        saturation[..., None],                # [14]    饱和度
        olip_features,                        # [15:20] OLIP 特征
    ], axis=-1)
    
    # 安全: NaN/Inf 保护
    obs = jnp.nan_to_num(obs, nan=0.0, posinf=5.0, neginf=-5.0)
    obs = jnp.clip(obs, -5.0, 5.0)
    
    return obs


# =============================================================================
# 简化奖励函数 (3 项)
# =============================================================================
def compute_hover_reward(
    phys_state: PhysState,
    action: Float[Array, "batch 4"],
    prev_action: Float[Array, "batch 4"],
    height_target: float = 10.0
) -> Tuple[Float[Array, "batch"], Dict[str, Float[Array, "batch"]]]:
    """
    3 项悬停奖励
    
    R = 0.5 * R_upright + 0.3 * R_height + 0.2 * R_smooth + C_alive
    """
    quat = phys_state.quaternion
    qx, qy = quat[..., 1], quat[..., 2]
    
    # R_upright: 鼓励水平 (1 - 2*(qx² + qy²))
    r_upright = 1.0 - 2.0 * (qx**2 + qy**2)
    r_upright = jnp.clip(r_upright, -1.0, 1.0)
    
    # R_height: 高度保持 (exp(-5 * Δh²))
    height_error = -height_target - phys_state.position[..., 2]  # NED
    r_height = jnp.exp(-5.0 * height_error**2)
    
    # R_smooth: 动作平滑 (-||a_t - a_{t-1}||²)
    action_diff = action - prev_action
    r_smooth = -jnp.sum(action_diff**2, axis=-1)
    
    # 存活偏置
    c_alive = 0.05
    
    # 总奖励
    reward = 0.5 * r_upright + 0.3 * r_height + 0.2 * r_smooth + c_alive
    
    info = {
        'r_upright': r_upright,
        'r_height': r_height,
        'r_smooth': r_smooth,
    }
    
    return reward, info


# =============================================================================
# 终止判断
# =============================================================================
def check_hover_termination(
    phys_state: PhysState,
    step_count: Float[Array, "batch"],
    config: HoverEnvConfig
) -> Bool[Array, "batch"]:
    """
    终止条件:
    1. 倾角超限
    2. 坠地 (高度 < 0.3m, NED: Z > -0.3)
    3. 达到最大步数
    """
    quat = phys_state.quaternion
    qx, qy = quat[..., 1], quat[..., 2]
    
    # 近似倾角: tilt ≈ 2*sqrt(qx² + qy²) (小角度近似)
    tilt = 2.0 * jnp.sqrt(qx**2 + qy**2 + 1e-8)
    tilt_exceeded = tilt > config.max_tilt_angle
    
    # 坠地
    crashed = phys_state.position[..., 2] > -0.3  # NED: Z > -0.3 即高度 < 0.3m
    
    # 超时
    timeout = step_count >= config.max_episode_steps
    
    return tilt_exceeded | crashed | timeout


# =============================================================================
# 四元数辅助
# =============================================================================
def euler_to_quaternion_simple(
    roll: Float[Array, "batch"],
    pitch: Float[Array, "batch"],
    yaw: Float[Array, "batch"]
) -> Float[Array, "batch 4"]:
    """欧拉角→四元数 [w, x, y, z]"""
    cr, sr = jnp.cos(roll/2), jnp.sin(roll/2)
    cp, sp = jnp.cos(pitch/2), jnp.sin(pitch/2)
    cy, sy = jnp.cos(yaw/2), jnp.sin(yaw/2)
    
    w = cr*cp*cy + sr*sp*sy
    x = sr*cp*cy - cr*sp*sy
    y = cr*sp*cy + sr*cp*sy
    z = cr*cp*sy - sr*sp*cy
    
    return jnp.stack([w, x, y, z], axis=-1)


# =============================================================================
# 环境重置
# =============================================================================
def reset_hover_env(
    key: PRNGKeyArray,
    batch_size: int,
    config: HoverEnvConfig
) -> Tuple[PhysState, L3State, PhysParams, Float[Array, "batch 4 4"],
           Float[Array, "batch 5"], Float[Array, "batch 4"], 
           Float[Array, "batch"], Float[Array, "batch 18"]]:
    """
    环境重置
    
    返回: phys_state, l3_state, phys_params, mixer_matrix, 
          olip_features, prev_action, step_count, obs
    """
    keys = jax.random.split(key, 6)
    
    # 1. 域随机化参数
    phys_params, mixer_matrix = randomize_params(keys[0], batch_size, config)
    
    # 2. 初始物理状态
    roll = jax.random.uniform(keys[1], (batch_size,), 
                               minval=-config.init_max_tilt, maxval=config.init_max_tilt)
    pitch = jax.random.uniform(keys[2], (batch_size,),
                                minval=-config.init_max_tilt, maxval=config.init_max_tilt)
    yaw = jax.random.uniform(keys[3], (batch_size,), minval=-jnp.pi, maxval=jnp.pi)
    quaternion = euler_to_quaternion_simple(roll, pitch, yaw)
    
    angular_velocity = jax.random.uniform(
        keys[4], (batch_size, 3),
        minval=-config.init_max_rate, maxval=config.init_max_rate
    )
    
    # 悬停油门
    hover_throttle = (phys_params.mass * 9.81) / (4.0 * phys_params.motor_max_thrust)
    hover_throttle = jnp.clip(hover_throttle, 0.1, 0.9)
    init_motor_throttle = jnp.repeat(jnp.sqrt(hover_throttle[:, None]), 4, axis=1)
    
    phys_state = PhysState(
        position=jnp.zeros((batch_size, 3)).at[:, 2].set(-config.init_height),
        velocity=jnp.zeros((batch_size, 3)),
        quaternion=quaternion,
        angular_velocity=angular_velocity,
        motor_throttle=init_motor_throttle,
        battery_soc=jnp.ones((batch_size,)),
        battery_voltage=jnp.full((batch_size,), 16.8),
        battery_V1=jnp.zeros((batch_size,)),
        battery_V2=jnp.zeros((batch_size,)),
        imu_gyro=jnp.zeros((batch_size, 3)),
        imu_accel=jnp.tile(jnp.array([0.0, 0.0, 9.81]), (batch_size, 1)),
        wind_velocity=jnp.zeros((batch_size, 3)),
        turbulence_state=jnp.zeros((batch_size, 3)),
    )
    
    # 3. L3 状态
    l3_state = L3State.create_default(batch_size)
    
    # 4. OLIP 特征 (整个 episode 不变)
    olip_features = generate_olip_features(phys_params, batch_size)
    
    # 5. 初始观测
    prev_action = jnp.zeros((batch_size, 4))
    step_count = jnp.zeros((batch_size,), dtype=jnp.int32)
    obs = build_hover_observation(phys_state, l3_state, olip_features, config.init_height)
    
    return (phys_state, l3_state, phys_params, mixer_matrix, 
            olip_features, prev_action, step_count, obs)


# =============================================================================
# 环境步进
# =============================================================================
def step_hover_env(
    key: PRNGKeyArray,
    phys_state: PhysState,
    l3_state: L3State,
    action: Float[Array, "batch 4"],
    prev_action: Float[Array, "batch 4"],
    step_count: Float[Array, "batch"],
    phys_params: PhysParams,
    l3_config: L3Config,
    mixer_matrix: Float[Array, "batch 4 4"],
    olip_features: Float[Array, "batch 5"],
    config: HoverEnvConfig
) -> Tuple[PhysState, L3State, Float[Array, "batch 18"], 
           Float[Array, "batch"], Bool[Array, "batch"], Dict]:
    """
    环境单步更新 (50Hz)
    
    返回: phys_state_new, l3_state_new, obs, reward, done, info
    """
    # 1. 动作映射
    target_rates, thrust = map_action_to_rates_and_thrust(action)
    
    # 2. L3 内环 + 物理仿真 (10x 500Hz)
    loop_keys = jax.random.split(key, config.num_substeps)
    
    def inner_loop_body(carry, loop_key):
        phys_st, l3_st = carry
        
        phys_st, l3_st, motor_cmds = l3_substep(
            phys_st, l3_st,
            target_rates, thrust,
            phys_params, l3_config, mixer_matrix,
            config.dt_sub
        )
        
        phys_st = physics_dynamics_step(
            phys_st, motor_cmds, phys_params, loop_key, config.dt_sub
        )
        
        return (phys_st, l3_st), None
    
    (phys_state_new, l3_state_new), _ = jax.lax.scan(
        inner_loop_body,
        (phys_state, l3_state),
        xs=loop_keys,
        length=config.num_substeps
    )
    
    # 3. 观测
    new_step_count = step_count + 1
    obs = build_hover_observation(
        phys_state_new, l3_state_new, olip_features, config.init_height
    )
    
    # 4. 奖励
    reward, reward_info = compute_hover_reward(
        phys_state_new, action, prev_action, config.init_height
    )
    
    # 5. 终止
    done = check_hover_termination(phys_state_new, new_step_count, config)
    
    # 坠毁惩罚
    quat = phys_state_new.quaternion
    tilt = 2.0 * jnp.sqrt(quat[..., 1]**2 + quat[..., 2]**2 + 1e-8)
    crash_penalty = jnp.where(
        tilt > config.max_tilt_angle,
        -5.0,
        jnp.where(phys_state_new.position[..., 2] > -0.3, -5.0, 0.0)
    )
    reward = reward + crash_penalty * done.astype(jnp.float32)
    
    info = {
        **reward_info,
        'step_count': new_step_count,
        'tilt': tilt,
    }
    
    return phys_state_new, l3_state_new, obs, reward, done, info
