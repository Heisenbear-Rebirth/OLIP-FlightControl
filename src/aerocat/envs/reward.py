"""
AeroCat v18.5 Reward Function

v18.5 奖励函数 (5 核心项, 零冲突):
- R_velocity:  速度幅值追踪 (w=0.40) - 高斯核, 含垂直分量
- R_direction: 速度方向追踪 (w=0.30) - 余弦相似度, Heading Frame
- R_yaw:       航向追踪     (w=0.15) - 机头指向 vs target_yaw
- R_spin:      自旋控制     (w=0.05) - 故障时豁免
- R_act:       动作平滑     (w=0.05)
- C_alive:     存活保底     (0.025)

[REMOVED v18.5] R_upright(局部最优陷阱), R_pos(部署时外环 PID 处理)
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Bool
from typing import Tuple, Dict, NamedTuple
import chex
from flax import struct

from ..core.state import PhysState, L1State, EnvParams
from ..physics.dynamics import rotate_vector, quaternion_conjugate


# =============================================================================
# 奖励权重配置 (来自 v18.0 规范)
# =============================================================================
@struct.dataclass
class RewardConfig:
    """
    奖励函数权重配置 (v18.7)

    v18.7 变更:
    - 移除 R_yaw: 积分角度对齐交给部署时外环 PID
    - Huber 核提供恒定梯度已间接解决对齐问题
    - 4 核心项: R_velocity + R_direction + R_spin + R_act

    设计哲学: RL 只做 "速率→电机" 控制映射，所有积分量对齐 (xyz位置/yaw角度) 交给外环 PID
    """
    # v18.7 权重表 (sum = 0.925)
    w_velocity: float = 0.45   # 速度幅值追踪 (Huber 核, 含垂直)
    w_direction: float = 0.35  # 速度方向追踪 (余弦相似度)
    w_spin: float = 0.05       # 自旋控制 (Huber, 故障时豁免)
    w_act: float = 0.05        # 动作平滑 (Huber)
    c_alive: float = 0.025     # 存活保底

    # Huber 参数 (v18.6)
    vel_huber_delta: float = 0.2    # 速度 Huber δ: <0.2 m/s 二次, >0.2 m/s 线性
    vel_huber_max: float = 5.0      # 速度归一化参考点 (R=0 时的误差)
    spin_huber_delta: float = 1.0   # 角速度 Huber δ (rad/s)
    spin_huber_max: float = 15.0    # 角速度归一化参考点
    act_huber_delta: float = 0.1    # 动作差异 Huber δ
    act_huber_max: float = 2.0      # 动作差异归一化参考点 (最大变化 ~2.0)

    # [DISABLED] 保留字段用于接口兼容
    w_yaw: float = 0.0         # v18.7: 移除, 部署时外环 PID
    w_upright: float = 0.0
    w_pos: float = 0.0
    w_align: float = 0.0
    k_xy: float = 0.05
    k_z: float = 0.10
    lambda_vel: float = 0.15   # [DEPRECATED] 保留向后兼容
    k_pos: float = 0.5         # [DEPRECATED] 保留向后兼容


# =============================================================================
# 四元数工具函数
# =============================================================================
def quaternion_to_z_axis(q: Float[Array, "batch 4"]) -> Float[Array, "batch 3"]:
    """
    从四元数提取机体 Z 轴在世界坐标系中的方向
    
    用于计算推力矢量方向 (机体 Z 轴即推力方向)
    
    Args:
        q: 四元数 [w, x, y, z]
    
    Returns:
        z_axis: Z 轴方向向量 (已归一化)
    """
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    
    # 旋转矩阵第三列: Z 轴经旋转后的方向
    z_x = 2.0 * (x * z + w * y)
    z_y = 2.0 * (y * z - w * x)
    z_z = 1.0 - 2.0 * (x * x + y * y)
    
    return jnp.stack([z_x, z_y, z_z], axis=-1)


# =============================================================================
# 推力矢量对齐奖励 (R_align)
# 公式: R_align = z_body · z_target
# =============================================================================
def compute_align_reward(
    q_curr: Float[Array, "batch 4"],
    q_target: Float[Array, "batch 4"] = None
) -> Float[Array, "batch"]:
    """
    [DEPRECATED] 推力矢量对齐奖励 - 仅保留用于接口兼容
    
    v18.2: 已被 compute_upright_reward() 替代
    """
    return compute_upright_reward(q_curr)


def compute_upright_reward(
    q_curr: Float[Array, "batch 4"],
) -> Float[Array, "batch"]:
    """
    [DEPRECATED v18.5] 机体竖直奖励 - 不再在主奖励函数中使用

    移除原因: 局部最优陷阱，惩罚正常倾斜飞行。
    保留函数体以兼容 compute_align_reward 调用。
    """
    q_x = q_curr[..., 1]
    q_y = q_curr[..., 2]
    z_body_z = 1.0 - 2.0 * (q_x ** 2 + q_y ** 2)
    return z_body_z


# =============================================================================
# Huber 工具函数 (v18.6 新增)
# =============================================================================
def _huber(x: Float[Array, "..."], delta: float) -> Float[Array, "..."]:
    """
    Huber 函数: 小误差二次，大误差线性。
    H(x, δ) = 0.5x²           if x ≤ δ
            = δ(x - 0.5δ)     if x > δ
    """
    return jnp.where(x <= delta, 0.5 * x ** 2, delta * (x - 0.5 * delta))


def _huber_reward(error_norm: Float[Array, "..."],
                  delta: float, max_error: float) -> Float[Array, "..."]:
    """
    Huber 归一化奖励: R = 1 - H(e, δ) / H(e_max, δ)

    e=0 → R=1.0;  e=e_max → R=0.0;  e>e_max → R<0 (线性下降，恒定梯度)
    """
    normalizer = _huber(jnp.array(max_error), delta)
    return 1.0 - _huber(error_norm, delta) / normalizer


def _huber_penalty(error_norm: Float[Array, "..."],
                   delta: float, max_error: float) -> Float[Array, "..."]:
    """
    Huber 归一化惩罚: R = -H(e, δ) / H(e_max, δ)  ∈ [-1, 0]

    e=0 → R=0;  e=e_max → R=-1.0
    """
    normalizer = _huber(jnp.array(max_error), delta)
    return -_huber(error_norm, delta) / normalizer


# =============================================================================
# 速度追踪奖励 (R_velocity) - v18.6: Huber 核
# =============================================================================
def compute_velocity_reward(
    v_error: Float[Array, "batch 3"],
    v_cmd: Float[Array, "batch 3"] = None,
    lambda_vel: float = 0.15,
    dir_scale_k: float = 2.0,
    delta: float = 0.2,
    max_error: float = 5.0,
) -> Float[Array, "batch"]:
    """
    速度追踪奖励 (v18.6: Huber 核替代高斯核)

    公式:
        R = 1 - H(||v_error||, δ) / H(e_max, δ)

    梯度特性 (δ=0.2):
    - e < 0.2 m/s: dR/de ∝ e   (二次精细调整)
    - e > 0.2 m/s: dR/de = 常数 (永不消失！)

    取值范围: R=1 @ e=0, R=0 @ e=e_max, R<0 @ e>e_max
    """
    error_norm = jnp.sqrt(jnp.sum(v_error ** 2, axis=-1) + 1e-8)
    return _huber_reward(error_norm, delta, max_error)


# =============================================================================
# 速度方向追踪奖励 (R_direction) - v18.3 新增
# 公式: R_dir = (1 + cos_sim(v_actual_heading, v_cmd_heading)) / 2
# 说明: 从 NED 到 Heading Frame 仅为二维正交旋转，不改变向量夹角，
#       因此在 Heading Frame 计算 cos_sim 完全等价于在 NED 世界系计算。
# =============================================================================
def compute_direction_reward(
    v_actual_heading: Float[Array, "batch 3"],
    v_cmd_heading: Float[Array, "batch 3"],
) -> Float[Array, "batch"]:
    """
    速度方向追踪奖励 (余弦相似度, Heading Frame 极简计算)
    
    v18.3: 考察无人机的真实飞行轨迹与指令飞行轨迹的相似度。
    直接在 Heading Frame 进行计算可省去坐标转换，数学上 100% 等价于世界坐标系。
    
    当 v_cmd 或 v_actual 幅值很小时，方向无意义，平滑退化为 1.0。
    
    公式:
        cos_sim = dot(v_actual_heading, v_cmd_heading) / (|v_actual| × |v_cmd|)
        R_dir = (1 + cos_sim) / 2    ∈ [0, 1]
    
    Args:
        v_actual_heading: 实际速度向量 [batch, 3] (Heading Frame)
        v_cmd_heading: 速度命令向量 [batch, 3] (Heading Frame)
    
    Returns:
        r_direction: 方向追踪奖励 [0, 1]
    """
    cmd_norm = jnp.sqrt(jnp.sum(v_cmd_heading ** 2, axis=-1) + 1e-8)
    actual_norm = jnp.sqrt(jnp.sum(v_actual_heading ** 2, axis=-1) + 1e-8)
    
    # 余弦相似度
    cos_sim = jnp.sum(v_cmd_heading * v_actual_heading, axis=-1) / (cmd_norm * actual_norm + 1e-8)
    cos_sim = jnp.clip(cos_sim, -1.0, 1.0)
    
    # 映射到 [0, 1]
    r_dir = (1.0 + cos_sim) / 2.0
    
    # 当命令幅值或实际速度幅值很小时，方向无意义 → 平滑过渡到 1.0
    cmd_gate = jnp.clip(cmd_norm / 0.5, 0.0, 1.0)
    actual_gate = jnp.clip(actual_norm / 0.3, 0.0, 1.0)
    gate = cmd_gate * actual_gate
    r_dir = r_dir * gate + (1.0 - gate)  # 低速时 r_dir → 1.0（不惩罚）
    
    return r_dir


# =============================================================================
# 航向追踪奖励 (R_yaw) - v18.3 新增
# 公式: R_yaw = (1 + cos_sim(heading_dir, target_dir)) / 2
# 独立考察机头指向与 target_yaw 的偏差，与飞行方向 R_direction 正交解耦
# =============================================================================
def compute_yaw_reward(
    q_curr: Float[Array, "batch 4"],
    target_yaw: Float[Array, "batch"],
) -> Float[Array, "batch"]:
    """
    航向追踪奖励 (v18.3 新增)
    
    专门考察机头方向（NED 水平面）与积分出的 target_yaw 之间的偏差。
    与 R_direction (飞行速度方向) 完全正交解耦。
    
    公式:
        yaw_curr = atan2(2(wz+xy), 1-2(y²+z²))
        heading_dir = [cos(yaw_curr), sin(yaw_curr)]  (NED 水平面)
        target_dir  = [cos(target_yaw), sin(target_yaw)]
        cos_sim = dot(heading_dir, target_dir)
        R_yaw = (1 + cos_sim) / 2    ∈ [0, 1]
    
    取值范围:
        - 1.0: 机头完全对准目标航向
        - 0.0: 机头指向与目标完全相反 (180°)
    
    Args:
        q_curr: 当前姿态四元数 [w, x, y, z]
        target_yaw: 目标偏航角 (rad)
    
    Returns:
        r_yaw: 航向追踪奖励 [0, 1]
    """
    # 1. 从四元数提取当前 yaw
    qw = q_curr[..., 0]
    qx = q_curr[..., 1]
    qy = q_curr[..., 2]
    qz = q_curr[..., 3]
    yaw_curr = jnp.arctan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz)
    )
    
    # 2. 水平面机头单位向量
    heading_cos = jnp.cos(yaw_curr)
    heading_sin = jnp.sin(yaw_curr)
    
    # 3. 目标机头单位向量
    target_cos = jnp.cos(target_yaw)
    target_sin = jnp.sin(target_yaw)
    
    # 4. 余弦相似度 (两个单位向量的点积)
    cos_sim = heading_cos * target_cos + heading_sin * target_sin
    cos_sim = jnp.clip(cos_sim, -1.0, 1.0)
    
    # 5. 映射到 [0, 1]
    r_yaw = (1.0 + cos_sim) / 2.0
    
    return r_yaw


# =============================================================================
# 自旋惩罚 (R_spin) - v18.6: Huber 归一化
# =============================================================================
def compute_spin_reward(
    omega: Float[Array, "batch 3"],
    k_xy: float = 0.05,
    k_z: float = 0.10,
    delta: float = 1.0,
    max_error: float = 15.0,
) -> Float[Array, "batch"]:
    """
    自旋惩罚 (v18.6: Huber 归一化到 [-1, 0])

    公式: R_spin = -H(||ω||, δ) / H(ω_max, δ)

    取值范围: [-1, 0]
    """
    omega_norm = jnp.sqrt(jnp.sum(omega ** 2, axis=-1) + 1e-8)
    return _huber_penalty(omega_norm, delta, max_error)


# =============================================================================
# 水平位置漂移惩罚 (R_pos) - v18.1 新增
# 公式: R_pos = exp(-k_pos * (x² + y²))
# 取值范围: (0, 1]，漂移为0时=1，漂移越大越接近0
# =============================================================================
def compute_position_reward(
    position: Float[Array, "batch 3"],
    ref_position: Float[Array, "batch 3"],
    k_pos: float = 0.5
) -> Float[Array, "batch"]:
    """
    [DEPRECATED v18.5] 位置跟踪奖励 - 不再在主奖励函数中使用

    移除原因: 在极端初始条件下产生追击震荡，与恐怡恢复任务冲突。
    位置保持交给部署时外环 PID 处理。
    """
    dx = position[..., 0] - ref_position[..., 0]
    dy = position[..., 1] - ref_position[..., 1]
    dz = position[..., 2] - ref_position[..., 2]
    d_sq = dx ** 2 + dy ** 2 + dz ** 2
    return jnp.exp(-k_pos * d_sq)



def compute_action_reward(
    action: Float[Array, "batch 4"],
    prev_action: Float[Array, "batch 4"],
    delta: float = 0.1,
    max_error: float = 2.0,
) -> Float[Array, "batch"]:
    """
    动作平滑惩罚 (v18.6: Huber 归一化到 [-1, 0])

    公式: R_act = -H(||Δa||, δ) / H(Δa_max, δ)
    取值范围: [-1, 0] (不再无界)
    """
    action_diff = action - prev_action
    diff_norm = jnp.sqrt(jnp.sum(action_diff ** 2, axis=-1) + 1e-8)
    return _huber_penalty(diff_norm, delta, max_error)


# =============================================================================
# 总奖励计算 (v18.0 规范)
# =============================================================================
def compute_reward_v18(
    phys_state: PhysState,
    l1_state: L1State,
    action: Float[Array, "batch 4"],
    prev_action: Float[Array, "batch 4"],
    ref_position: Float[Array, "batch 3"],
    v_actual_heading: Float[Array, "batch 3"] = None,
    v_cmd_heading: Float[Array, "batch 3"] = None,
    config: RewardConfig = None
) -> Tuple[Float[Array, "batch"], Dict[str, Float[Array, "batch"]]]:  # noqa: E501

    """
    计算 v18.5 总奖励
    
    v18.5 公式 (5 核心项):
        R_total = 0.40×R_velocity + 0.30×R_direction +
                  0.15×R_yaw + 0.05×R_spin + 0.05×R_act + 0.025
    
    Args:
        phys_state: 当前物理状态
        l1_state: L1 状态
        action: 当前动作
        prev_action: 上一步动作
        ref_position: 参考位置
        v_actual_heading: 实际速度 (Heading Frame)
        v_cmd_heading: 速度命令 (Heading Frame)
        config: 奖励配置
    
    Returns:
        reward: 总奖励标量
        info: 各分项奖励字典
    """
    if config is None:
        config = RewardConfig()
    
    # 提取状态
    q_curr = phys_state.quaternion
    omega = phys_state.angular_velocity
    
    # 速度误差 (Heading Frame)
    v_error = l1_state.velocity_error_heading
    
    # 计算各分项
    r_velocity = compute_velocity_reward(
        v_error, delta=config.vel_huber_delta, max_error=config.vel_huber_max)
    r_spin = compute_spin_reward(
        omega, delta=config.spin_huber_delta, max_error=config.spin_huber_max)
    r_act = compute_action_reward(
        action, prev_action, delta=config.act_huber_delta, max_error=config.act_huber_max)

    # 速度方向奖励 (Heading Frame)
    if v_actual_heading is None or v_cmd_heading is None:
        v_cmd_heading = l1_state.target_velocity_body
        v_actual_heading = l1_state.target_velocity_body - l1_state.velocity_error_heading

    r_direction = compute_direction_reward(v_actual_heading, v_cmd_heading)

    # 加权求和 + clip [-1, 1]
    reward_raw = (
        config.w_velocity  * r_velocity  +
        config.w_direction * r_direction +
        config.w_spin      * r_spin      +
        config.w_act       * r_act       +
        config.c_alive
    )
    reward = jnp.clip(reward_raw, -1.0, 1.0)

    # 信息字典
    info = {
        'r_velocity': r_velocity,
        'r_direction': r_direction,
        'r_spin': r_spin,
        'r_act': r_act,
        'c_alive': jnp.full_like(reward, config.c_alive),
        'reward_raw': reward_raw,
    }

    return reward, info



# =============================================================================
# 终止判断
# =============================================================================
def check_termination(
    phys_state: PhysState,
    episode_time: Float[Array, "batch"],
    max_episode_time: float = 10.0,   # 最大 10 秒
    min_height: float = 0.05,         # [DEPRECATED v18.4] 已移除地面碰撞，保留参数向后兼容
    battery_threshold: float = 0.01,  # 电池耗尽阈值
    max_tilt_angle: Float[Array, "batch"] | float = 2.5  # 最大倾角 (rad)
) -> Tuple[Bool[Array, "batch"], Bool[Array, "batch"], Dict[str, Bool[Array, "batch"]]]:
    """
    检查终止条件

    终止条件 (v18.4: 仅保留超时和电池耗尽):
    1. 超时: time >= max_time
    2. 电池耗尽: SOC < threshold

    说明: v18.4 移除了 crashed 和 flipped 条件。
    - crashed: lambda=0.4 时大倾角+低高度频繁触地，产生无意义短 episode
    - flipped: BC 预训练模型不会出现此情况，无需此保护
    """
    # 终止条件 (仅超时和电池)
    timeout = episode_time >= max_episode_time
    battery_dead = phys_state.battery_soc < battery_threshold

    # 任意条件满足则终止
    done = timeout | battery_dead

    # 截断 (正常超时结束)
    truncated = timeout & (~battery_dead)

    # 终止原因 (保留 key 向后兼容)
    info = {
        'crashed': jnp.zeros_like(timeout),
        'timeout': timeout,
        'battery_dead': battery_dead,
        'flipped': jnp.zeros_like(timeout),
    }

    return done, truncated, info

