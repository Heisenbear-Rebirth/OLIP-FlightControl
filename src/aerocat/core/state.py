"""
AeroCat v18.2 Core State Definitions

定义完整的环境状态 PyTree 结构，支持 4096+ 并行环境的 JAX vmap/jit 操作。
遵循 Neuro-Cascaded Adaptive Architecture 的分层设计。

v18.2 架构变更:
- L1State: 移除 vxy_integral/vxy_error_prev/target_quaternion (L1 不再做 PID)
- L1State: 新增 velocity_error_heading (3D 速度误差，核心 RL 驱动信号)
- L1Config: 移除 vxy_kp/ki/kd/i_limit (水平速度 PID 参数)

坐标系约定:
- 地理系: NED (North-East-Down)
- 机体系: FRD (Front-Right-Down)
- 四元数: [w, x, y, z] (实部在前)
"""

from typing import Tuple
import jax
import jax.numpy as jnp
import chex
from flax import struct
from jaxtyping import Array, Float, Int, Bool, PRNGKeyArray


# =============================================================================
# 类型别名
# =============================================================================
Vector3 = Float[Array, "3"]
Vector4 = Float[Array, "4"]
Quaternion = Float[Array, "4"]
BatchScalar = Float[Array, "batch"]
BatchVector3 = Float[Array, "batch 3"]
BatchVector4 = Float[Array, "batch 4"]
BatchQuaternion = Float[Array, "batch 4"]


# =============================================================================
# 配置常量
# =============================================================================
@struct.dataclass
class L1Config:
    """L1 层配置常量 (意图解析 / 坐标变换层, v18.2)"""
    
    # 摇杆映射
    max_yaw_rate: float = 100.0 * jnp.pi / 180.0    # 1.745 rad/s (v18.4: 200→100°/s, 对标大疆航拍机 Normal 模式)
    max_tilt: float = 45.0 * jnp.pi / 180.0         # 0.78 rad
    max_vz: float = 3.0                              # m/s
    
    # 高度 PID (v18.2: 仅保留参数定义，实际仅使用 t_hover_guess 前馈)
    vz_kp: float = 0.5
    vz_ki: float = 0.1
    vz_kd: float = 0.05
    vz_i_limit: float = 0.3
    
    # v18.2: 移除水平速度 PID 参数 (vxy_kp/ki/kd/i_limit)
    # L1 层不再做 PID 控制，仅做坐标变换
    
    # OU + Poisson Jump 摇杆生成 (Roll/Pitch 轴)
    # OU 过程: dx = θ(μ-x)dt + σ√dt·ε, 产生平滑、均值回归的摇杆轨迹
    # 泊松跳跃: 以 λ_jump 频率触发松杆归中 (急刹)
    ou_theta: float = 1.5                            # 回归速率 (1/s), τ=1/θ≈0.67s
    ou_sigma: float = 0.7                            # 波动率, 稳态 std = σ/√(2θ) ≈ 0.40
    ou_mu: float = 0.0                               # 均值 (摇杆中位)
    jump_rate: float = 0.2                           # 泊松跳跃率 (1/s), 平均 5s 一次急刹
    
    # v18.4: Yaw 轴独立 OU 参数 (航拍机 yaw 操控更平缓)
    # 设计依据: 大疆航拍机 yaw 操控特点 — 缓慢推杆转向, 极少急刹
    ou_theta_yaw: float = 0.5                        # 慢回归 (1/s), τ=2.0s (对比 RP 轴 0.67s)
    ou_sigma_yaw: float = 0.3                        # 小波动, 稳态 std = 0.3/√1 = 0.30 (对比 RP 轴 0.40)
    jump_rate_yaw: float = 0.05                      # 极少跳跃 (1/s), 平均 20s 一次 (对比 RP 轴 5s)
    
    # 速度命令速率限制 (Slew Rate Limiter, 课程化线性增长)
    # effective_rate = max_v_cmd_rate + (max_v_cmd_rate_full - max_v_cmd_rate) * λ
    max_v_cmd_rate: float = 0.4                      # λ=0 时最大速度命令变化率 (m/s²)
    max_v_cmd_rate_full: float = 2.0                  # λ=1 时最大速度命令变化率 (m/s²)
    
    # v18.4: Yaw 命令速率限制 (防止 OU/泊松跳跃产生的阶跃突变)
    # 限制 yaw_rate_cmd 每步最大变化量 = max_yaw_cmd_rate * dt
    # 100°/s/s × 0.02s = 2°/s 每步，从 0 到满杆 (100°/s) 需 1.0s
    max_yaw_cmd_rate: float = 100.0 * jnp.pi / 180.0  # 1.745 rad/s/s (Yaw 命令最大变化率)
    
    # 默认悬停油门
    t_hover_guess: float = 0.64


@struct.dataclass
class L3Config:
    """L3 层配置 (内环控制层)"""
    
    # Rate PID 增益 [Roll, Pitch, Yaw]
    kp: Float[Array, "3"] = struct.field(
        default_factory=lambda: jnp.array([4.0, 4.0, 2.0])
    )
    ki: Float[Array, "3"] = struct.field(
        default_factory=lambda: jnp.array([0.5, 0.5, 0.3])
    )
    kd: Float[Array, "3"] = struct.field(
        default_factory=lambda: jnp.array([0.02, 0.02, 0.01])
    )
    
    # 积分限幅 (占最大力矩的 30%)
    i_limit: float = 0.3
    
    # 子步时间步长 (500Hz)
    dt_sub: float = 0.002


# =============================================================================
# 物理状态 (PhysState)
# =============================================================================
@struct.dataclass
class PhysState:
    """
    物理状态 (刚体运动学 + 电源 + 传感器)
    
    所有向量遵循:
    - 位置/速度: NED 地理坐标系 (m, m/s)
    - 角速度: FRD 机体坐标系 (rad/s)
    - 四元数: [w, x, y, z] 从地理系到机体系的旋转
    """
    
    # ----- 刚体运动学 (13 dim) -----
    position: Float[Array, "batch 3"]           # NED 位置 (m)
    velocity: Float[Array, "batch 3"]           # NED 速度 (m/s)
    quaternion: Float[Array, "batch 4"]         # 姿态四元数 [w,x,y,z]
    angular_velocity: Float[Array, "batch 3"]   # 机体角速度 (rad/s)
    
    # ----- 电机状态 (4 dim) -----
    motor_throttle: Float[Array, "batch 4"]     # 归一化油门 [0, 1]
    
    # ----- 电池状态 (ECM 模型, 4 dim) -----
    battery_soc: Float[Array, "batch"]          # 剩余电量 [0, 1]
    battery_voltage: Float[Array, "batch"]      # 端电压 (V)
    battery_V1: Float[Array, "batch"]           # 极化电压 1
    battery_V2: Float[Array, "batch"]           # 极化电压 2
    
    # ----- 传感器读数 (带噪声, 6 dim) -----
    imu_gyro: Float[Array, "batch 3"]           # 陀螺仪 (rad/s)
    imu_accel: Float[Array, "batch 3"]          # 加速度计 (m/s²)
    
    # ----- 环境状态 (6 dim) -----
    wind_velocity: Float[Array, "batch 3"]      # 总风速 NED (m/s)
    turbulence_state: Float[Array, "batch 3"]   # Dryden 湍流内部状态
    
    @classmethod
    def create_default(cls, batch_size: int) -> "PhysState":
        """创建默认初始状态"""
        return cls(
            position=jnp.zeros((batch_size, 3)).at[:, 2].set(-10.0),  # 10m 高度 (NED: Z=Down, so Height=-Z -> Z=-10)
            velocity=jnp.zeros((batch_size, 3)),
            quaternion=jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (batch_size, 1)),
            angular_velocity=jnp.zeros((batch_size, 3)),
            motor_throttle=jnp.full((batch_size, 4), 0.64),  # 悬停油门
            battery_soc=jnp.ones((batch_size,)),
            battery_voltage=jnp.full((batch_size,), 16.8),  # 4S 满电
            battery_V1=jnp.zeros((batch_size,)),
            battery_V2=jnp.zeros((batch_size,)),
            imu_gyro=jnp.zeros((batch_size, 3)),
            imu_accel=jnp.tile(jnp.array([0.0, 0.0, 9.81]), (batch_size, 1)),  # NED 重力
            wind_velocity=jnp.zeros((batch_size, 3)),
            turbulence_state=jnp.zeros((batch_size, 3)),
        )


# =============================================================================
# L1 层状态 (制导层)
# =============================================================================
@struct.dataclass
class L1State:
    """
    L1 层状态 (意图解析 / 坐标变换层, v18.2)
    
    v18.2 变更:
    - 移除: vxy_integral, vxy_error_prev (水平速度 PID 积分器)
    - 移除: target_quaternion (L1 不再计算目标姿态)
    - 新增: velocity_error_heading (3D 速度误差，RL 核心驱动信号)
    """
    
    # ----- Headed 模式: Yaw 积分器 -----
    target_yaw: Float[Array, "batch"]           # 目标航向 (rad)
    
    # ----- Headless 模式: 参考航向锁定 -----
    headless_ref_yaw: Float[Array, "batch"]     # 参考航向 (rad)
    
    # ----- 高度控制积分器 (保留用于前馈兼容) -----
    vz_integral: Float[Array, "batch"]          # Vz 误差积分
    vz_error_prev: Float[Array, "batch"]        # 上一步 Vz 误差
    
    # ----- 当前模式 -----
    mode: Int[Array, "batch"]                   # 0=Headed, 1=Headless
    
    # ----- 虚拟摇杆状态 (OU + Poisson Jump 过程) -----
    virtual_stick_state: Float[Array, "batch 4"]  # [roll, pitch, yaw, throttle]
    
    # ----- v18.2 新增: 速度误差 (核心 RL 驱动信号) -----
    velocity_error_heading: Float[Array, "batch 3"]  # Heading Frame 速度误差 [vx_err, vy_err, vz_err]
    
    # ----- 速度命令速率限制状态 -----
    prev_v_cmd_heading: Float[Array, "batch 3"]   # 上一步速度命令 (用于 Slew Rate Limiter)
    
    # ----- v18.4: Yaw 命令速率限制状态 -----
    prev_yaw_rate_cmd: Float[Array, "batch"]   # 上一步 yaw 角速率命令 (用于 Yaw Slew Rate Limiter)
    
    # ----- L1 输出缓存 -----
    base_throttle: Float[Array, "batch"]        # 悬停油门前馈
    target_velocity_body: Float[Array, "batch 3"]  # 期望 Heading Frame 速度 (用于 Reward)
    
    @classmethod
    def create_default(cls, batch_size: int) -> "L1State":
        """创建默认初始状态"""
        return cls(
            target_yaw=jnp.zeros((batch_size,)),
            headless_ref_yaw=jnp.zeros((batch_size,)),
            vz_integral=jnp.zeros((batch_size,)),
            vz_error_prev=jnp.zeros((batch_size,)),
            mode=jnp.zeros((batch_size,), dtype=jnp.int32),  # 默认 Headed
            virtual_stick_state=jnp.zeros((batch_size, 4)),
            velocity_error_heading=jnp.zeros((batch_size, 3)),
            prev_v_cmd_heading=jnp.zeros((batch_size, 3)),
            prev_yaw_rate_cmd=jnp.zeros((batch_size,)),
            base_throttle=jnp.full((batch_size,), 0.64),
            target_velocity_body=jnp.zeros((batch_size, 3)),
        )


# =============================================================================
# L3 层状态 (内环控制层)
# =============================================================================
@struct.dataclass
class L3State:
    """
    L3 层状态 (内环 PID + Mixer)
    
    维护 Rate PID 积分器和饱和度反馈。
    """
    
    # ----- Rate PID 积分器 [Roll, Pitch, Yaw] -----
    pid_integral: Float[Array, "batch 3"]       # 积分项累积值
    
    # ----- 上一帧陀螺仪读数 (D-on-Measurement) -----
    gyro_prev: Float[Array, "batch 3"]          # 用于微分项
    
    # ----- 饱和度指标 (反馈给 L2) -----
    mixer_saturation: Float[Array, "batch"]     # 混控器饱和度 [0, ∞)
    
    # ----- 上一帧 PID 输出 (平滑) -----
    pid_output_prev: Float[Array, "batch 3"]    # 用于动作平滑
    
    @classmethod
    def create_default(cls, batch_size: int) -> "L3State":
        """创建默认初始状态"""
        return cls(
            pid_integral=jnp.zeros((batch_size, 3)),
            gyro_prev=jnp.zeros((batch_size, 3)),
            mixer_saturation=jnp.zeros((batch_size,)),
            pid_output_prev=jnp.zeros((batch_size, 3)),
        )


# =============================================================================
# 延迟仿真缓冲区
# =============================================================================
@struct.dataclass
class DelayBuffer:
    """
    环形缓冲区用于延迟仿真
    
    Buffer 大小 = 25 slots (50ms @ 500Hz 或 25 slots @ 50Hz for 500ms max delay)
    """
    
    # ----- IMU 延迟缓冲 -----
    imu_gyro_buffer: Float[Array, "batch 25 3"]
    imu_accel_buffer: Float[Array, "batch 25 3"]
    
    # ----- 动作延迟缓冲 (RC latency) -----
    action_buffer: Float[Array, "batch 25 4"]
    
    # ----- 当前写入索引 -----
    write_idx: Int[Array, "batch"]
    
    # ----- 延迟步数 (每 episode 随机固定) -----
    imu_delay_steps: Int[Array, "batch"]
    action_delay_steps: Int[Array, "batch"]
    
    @classmethod
    def create_default(cls, batch_size: int) -> "DelayBuffer":
        """创建默认缓冲区 (零延迟)"""
        return cls(
            imu_gyro_buffer=jnp.zeros((batch_size, 25, 3)),
            imu_accel_buffer=jnp.tile(
                jnp.array([0.0, 0.0, 9.81])[None, None, :], 
                (batch_size, 25, 1)
            ),
            action_buffer=jnp.zeros((batch_size, 25, 4)),
            write_idx=jnp.zeros((batch_size,), dtype=jnp.int32),
            imu_delay_steps=jnp.zeros((batch_size,), dtype=jnp.int32),
            action_delay_steps=jnp.zeros((batch_size,), dtype=jnp.int32),
        )


# =============================================================================
# 完整环境状态 (EnvState)
# =============================================================================
@struct.dataclass
class EnvState:
    """
    v18.0 完整环境状态
    
    包含所有层级的动态状态，支持 vmap 并行和 jit 编译。
    """
    
    # ----- 物理状态 -----
    phys_state: PhysState
    
    # ----- L1 层状态 (制导层) -----
    l1_state: L1State
    
    # ----- L3 层状态 (内环控制) -----
    l3_state: L3State
    
    # ----- (v19.0: OLIP 已移除) -----
    
    # ----- 延迟仿真缓冲区 -----
    obs_buffer: DelayBuffer
    
    # ----- 环境元状态 -----
    step_count: Int[Array, "batch"]             # 当前步数
    episode_time: Float[Array, "batch"]           # 当前时间 (s)
    prev_action: Float[Array, "batch 4"]          # 上一帧动作 (用于平滑奖励)
    
    # 辅助状态
    effective_lambda: Float[Array, "batch"]       # 有效课程难度 (Experience Replay)
    termination_tilt_threshold: Float[Array, "batch"] # 动态终止倾角阈值
    
    # ----- 参考轨迹位置 (v18.1 新增) -----
    # 每步按 v_cmd_world * dt 积分，代表"按指令飞行"应到达的位置
    # R_pos 惩罚 actual_pos 与 ref_position 的水平偏差，而非原点偏差
    ref_position: Float[Array, "batch 3"]         # NED 坐标系，参考轨迹当前位置 (m)
    
    @classmethod
    def create_default(cls, batch_size: int) -> "EnvState":
        """创建默认初始状态"""
        return cls(
            phys_state=PhysState.create_default(batch_size),
            l1_state=L1State.create_default(batch_size),
            l3_state=L3State.create_default(batch_size),
            obs_buffer=DelayBuffer.create_default(batch_size),
            step_count=jnp.zeros((batch_size,), dtype=jnp.int32),
            episode_time=jnp.zeros((batch_size,)),
            prev_action=jnp.zeros((batch_size, 4)),
            effective_lambda=jnp.zeros((batch_size,)),
            termination_tilt_threshold=jnp.full((batch_size,), 2.5),
            # 参考轨迹初始化为原点 (reset_env 会设置为实际 spawn 位置)
            ref_position=jnp.zeros((batch_size, 3)),
        )


# =============================================================================
# 物理参数 (PhysParams) - 继承自 v17.2 并扩展
# =============================================================================
@struct.dataclass
class PhysParams:
    """
    物理参数 (88+ 参数, episode 内不变)
    
    包含几何、质量、动力、气动、传感器等所有可随机化参数。
    """
    
    # ----- 1. 几何参数 (6) -----
    arm_length: Float[Array, "batch"]           # 机臂长度 (m)
    frame_angle: Float[Array, "batch"]          # 机架角度 (rad), X型为 π/4
    mass: Float[Array, "batch"]                 # 总质量 (kg)
    
    # ----- 2. 惯性矩阵 (对角元素) -----
    inertia_xx: Float[Array, "batch"]
    inertia_yy: Float[Array, "batch"]
    inertia_zz: Float[Array, "batch"]
    cog_offset: Float[Array, "batch 3"]         # 重心偏移 (m)
    
    # ----- 3. 电机参数 -----
    motor_thrust_coeff: Float[Array, "batch 4"]   # 推力系数 Ct
    motor_torque_coeff: Float[Array, "batch 4"]   # 力矩系数 Cm (Legacy)
    force_to_torque_ratio: Float[Array, "batch 4"] # 推力-扭矩系数比 (VAC: sigma)
    motor_tau_up: Float[Array, "batch 4"]         # 加速时间常数 (s)
    motor_tau_down: Float[Array, "batch 4"]       # 减速时间常数 (s)
    motor_max_thrust: Float[Array, "batch"]       # 单电机最大推力 (N)
    motor_loss: Float[Array, "batch 4"]           # 电机效率差异 [0, 1] (工艺公差)
    motor_deadband: Float[Array, "batch 4"]        # 电机死区阈值 (PF-GDR §3.5)
    
    # ----- 4. 电池参数 (ECM) -----
    battery_r0: Float[Array, "batch"]             # 内阻 (Ω)
    battery_r1: Float[Array, "batch"]             # 极化电阻 1
    battery_r2: Float[Array, "batch"]             # 极化电阻 2
    battery_c1: Float[Array, "batch"]             # 极化电容 1
    battery_c2: Float[Array, "batch"]             # 极化电容 2
    battery_ocv_coeff: Float[Array, "batch 3"]    # OCV 多项式系数
    
    # ----- 5. 气动参数 -----
    drag_coeff_xyz: Float[Array, "batch 3"]       # 平移阻力系数
    rot_drag_coeff: Float[Array, "batch 3"]       # 旋转阻力系数
    
    # ----- 6. 传感器噪声 -----
    gyro_noise_std: Float[Array, "batch"]         # 陀螺仪噪声 (rad/s)
    accel_noise_std: Float[Array, "batch"]        # 加速度计噪声 (m/s²)
    gyro_bias: Float[Array, "batch 3"]            # 陀螺仪偏置
    accel_bias: Float[Array, "batch 3"]           # 加速度计偏置
    
    # ----- 7. 延迟参数 -----
    rc_latency: Float[Array, "batch"]             # RC 延迟 (s)
    imu_latency: Float[Array, "batch"]            # IMU 延迟 (s)
    
    # ----- 8. 环境参数 -----
    wind_speed_mean: Float[Array, "batch"]        # 平均风速 (m/s)
    wind_direction: Float[Array, "batch"]         # 风向 (rad)
    turbulence_intensity: Float[Array, "batch"]   # 湍流强度
    roughness_length: Float[Array, "batch"]       # 粗糙度长度 (m)
    
    # ----- 10. 阵风参数 (Gust) -----
    gust_active: Float[Array, "batch"]            # 激活标志 (0/1)
    gust_start_time: Float[Array, "batch"]        # 开始时间 (s)
    gust_duration: Float[Array, "batch"]          # 持续时间 (s)
    gust_magnitude: Float[Array, "batch"]         # 峰值强度 (m/s)
    gust_direction: Float[Array, "batch 3"]       # 阵风方向
    
    # ----- 11. 其它物理参数 -----
    battery_capacity: Float[Array, "batch"]       # 电池容量 (mAh)

    
    @classmethod
    def create_default(cls, batch_size: int) -> "PhysParams":
        """创建默认参数 (理想四旋翼)"""
        return cls(
            arm_length=jnp.full((batch_size,), 0.15),
            frame_angle=jnp.full((batch_size,), jnp.pi / 4),  # 45° X型
            mass=jnp.full((batch_size,), 1.0),
            inertia_xx=jnp.full((batch_size,), 0.01),
            inertia_yy=jnp.full((batch_size,), 0.01),
            inertia_zz=jnp.full((batch_size,), 0.02),
            cog_offset=jnp.zeros((batch_size, 3)),
            motor_thrust_coeff=jnp.full((batch_size, 4), 0.001),
            motor_torque_coeff=jnp.full((batch_size, 4), 0.0001),
            force_to_torque_ratio=jnp.full((batch_size, 4), 0.01),  # Default sigma
            motor_tau_up=jnp.full((batch_size, 4), 0.03),
            motor_tau_down=jnp.full((batch_size, 4), 0.08),
            motor_max_thrust=jnp.full((batch_size,), 5.0),
            motor_loss=jnp.zeros((batch_size, 4)),
            motor_deadband=jnp.full((batch_size, 4), 0.03),  # 默认 3% 死区
            battery_r0=jnp.full((batch_size,), 0.02),
            battery_r1=jnp.full((batch_size,), 0.01),
            battery_r2=jnp.full((batch_size,), 0.005),
            battery_c1=jnp.full((batch_size,), 1000.0),
            battery_c2=jnp.full((batch_size,), 5000.0),
            battery_ocv_coeff=jnp.tile(jnp.array([3.0, 1.2, 0.0]), (batch_size, 1)),
            drag_coeff_xyz=jnp.full((batch_size, 3), 0.01),
            rot_drag_coeff=jnp.full((batch_size, 3), 0.001),
            gyro_noise_std=jnp.full((batch_size,), 0.01),
            accel_noise_std=jnp.full((batch_size,), 0.1),
            gyro_bias=jnp.zeros((batch_size, 3)),
            accel_bias=jnp.zeros((batch_size, 3)),
            rc_latency=jnp.full((batch_size,), 0.02),
            imu_latency=jnp.full((batch_size,), 0.01),
            wind_speed_mean=jnp.zeros((batch_size,)),
            wind_direction=jnp.zeros((batch_size,)),
            turbulence_intensity=jnp.full((batch_size,), 0.1),
            roughness_length=jnp.full((batch_size,), 0.05),
            gust_active=jnp.zeros((batch_size,)),
            gust_start_time=jnp.zeros((batch_size,)),
            gust_duration=jnp.ones((batch_size,)),
            gust_magnitude=jnp.zeros((batch_size,)),
            gust_direction=jnp.tile(jnp.array([1.0, 0.0, 0.0]), (batch_size, 1)),
            battery_capacity=jnp.full((batch_size,), 3000.0),  # mAh

        )


# =============================================================================
# 完整环境参数 (EnvParams)
# =============================================================================
@struct.dataclass
class EnvParams:
    """
    环境参数 (每 episode 采样一次，episode 内不变)
    """
    
    # ----- 物理参数 -----
    phys_params: PhysParams
    
    # ----- L1 层配置 -----
    l1_config: L1Config
    
    # ----- L3 层配置 -----
    l3_config: L3Config
    
    # ----- 预计算的混控器矩阵 (依赖几何参数) -----
    mixer_matrix: Float[Array, "batch 4 4"]
    
    @classmethod
    def create_default(cls, batch_size: int) -> "EnvParams":
        """创建默认参数"""
        # 计算默认 X 型混控器矩阵
        angle = jnp.pi / 4  # 45°
        A = jnp.sin(angle)
        B = jnp.cos(angle)
        C = 0.1  # 力矩系数比
        
        # 混控器矩阵: [Thrust, Roll, Pitch, Yaw] -> [FR, RL, FL, RR]
        # Correct X-Config Yaw: +Yaw (Right) -> Increase CW (FL/RR), Decrease CCW (FR/RL)
        mixer_base = jnp.array([
            [1.0, -A,  B, -C],   # FR (CCW)
            [1.0,  A, -B, -C],   # RL (CCW)
            [1.0,  A,  B,  C],   # FL (CW)
            [1.0, -A, -B,  C],   # RR (CW)
        ])
        mixer_matrix = jnp.tile(mixer_base[None, :, :], (batch_size, 1, 1))
        
        return cls(
            phys_params=PhysParams.create_default(batch_size),
            l1_config=L1Config(),
            l3_config=L3Config(),
            mixer_matrix=mixer_matrix,
        )


# =============================================================================
# 辅助函数
# =============================================================================
def compute_mixer_matrix(
    frame_angle: Float[Array, "batch"],
    torque_ratio: Float[Array, "batch"]
) -> Float[Array, "batch 4 4"]:
    """
    BetaFlight-style 归一化混控器矩阵 (v18.4)
    
    所有轴使用 ±1 系数，几何/反扭矩因子只在 physics 层出现。
    消除了旧版本中 A²/B²/sigma² 的 double-counting 问题。
    
    参考: BetaFlight X-frame mixer (motorMix_QUAD_X)
    
    Args:
        frame_angle: 机架角度 (rad) — 保留参数，不再使用
        torque_ratio: 力矩/推力比 — 保留参数，不再使用
    
    Returns:
        mixer_matrix: [batch, 4, 4] 混控器矩阵
    """
    ones = jnp.ones_like(frame_angle)
    
    # BetaFlight X-frame 标准: [Thrust, Roll, Pitch, Yaw]
    # 几何因子 (A, B) 和反扭矩系数 (sigma) 只在 dynamics.py 中出现
    row0 = jnp.stack([ ones, -ones,  ones, -ones], axis=-1)  # FR (CCW)
    row1 = jnp.stack([ ones,  ones, -ones, -ones], axis=-1)  # RL (CCW)
    row2 = jnp.stack([ ones,  ones,  ones,  ones], axis=-1)  # FL (CW)
    row3 = jnp.stack([ ones, -ones, -ones,  ones], axis=-1)  # RR (CW)
    
    mixer_matrix = jnp.stack([row0, row1, row2, row3], axis=1)
    
    return mixer_matrix
