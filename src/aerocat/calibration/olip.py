"""
OLIP v19.0 Virtual Calibrator (Digital Twin Calibration)

实现基于 Digital Twin 的 OLIP 标定流程，生成 11 维特征向量。
参考: 07_系统辨识_OLIP_v18.0.md 和 08_附录_OLIP仿真生成公式_v18.0.md

核心思想: 在仿真环境中直接运行标定流程，而非使用解析公式，
确保 Sim-to-Real 的特征一致性。

11 维特征定义:
[0]  Lag_Phase_4Hz      - 4Hz 下相位滞后
[1]  Lag_Phase_8Hz      - 8Hz 与 4Hz 的相位差
[2]  Gain_Mag_4Hz       - 4Hz 下增益比
[3]  Gain_Mag_8Hz       - 8Hz 下增益比
[4]  Integ_Effort_R     - Roll 轴积分努力
[5]  Integ_Effort_P     - Pitch 轴积分努力
[6]  Integ_Effort_Y     - Yaw 轴积分努力
[7]  Auth_Asymmetry     - 控制不对称性
[8]  Hover_Throttle     - 悬停油门
[9]  Thrust_Gain        - 推力增益 (加速度/油门)
[10] Nonlinear_Factor   - 非线性度
"""

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray


# =============================================================================
# DFT 累加器 (Goertzel Algorithm - 简化版)
# =============================================================================
def dft_single_freq(
    signal: Float[Array, "N"],
    freq: float,
    sample_rate: float = 500.0
) -> tuple:
    """
    计算信号在特定频率的 DFT 幅值和相位
    
    使用 Goertzel 算法的简化版本，适合 JAX 向量化。
    
    Args:
        signal: 输入信号序列
        freq: 目标频率 (Hz)
        sample_rate: 采样率 (Hz)
        
    Returns:
        (magnitude, phase): 幅值和相位 (rad)
    """
    N = signal.shape[-1]
    t = jnp.arange(N) / sample_rate
    
    # 计算实部和虚部
    cos_term = jnp.cos(2.0 * jnp.pi * freq * t)
    sin_term = jnp.sin(2.0 * jnp.pi * freq * t)
    
    real = jnp.sum(signal * cos_term, axis=-1)
    imag = jnp.sum(signal * sin_term, axis=-1)
    
    # 幅值和相位
    magnitude = jnp.sqrt(real**2 + imag**2) * 2.0 / N
    phase = jnp.arctan2(imag, real)
    
    return magnitude, phase


# =============================================================================
# RLS 线性回归
# =============================================================================
def linear_regression(
    x: Float[Array, "N"],
    y: Float[Array, "N"]
) -> tuple:
    """
    递归最小二乘法拟合 y = k*x + b
    
    Returns:
        (k, b): 斜率和截距
    """
    N = x.shape[-1]
    sum_x = jnp.sum(x, axis=-1)
    sum_y = jnp.sum(y, axis=-1)
    sum_x2 = jnp.sum(x**2, axis=-1)
    sum_xy = jnp.sum(x * y, axis=-1)
    
    # k = (N*sum_xy - sum_x*sum_y) / (N*sum_x2 - sum_x^2)
    denom = N * sum_x2 - sum_x**2
    denom = jnp.maximum(denom, 1e-10)  # 防止除零
    
    k = (N * sum_xy - sum_x * sum_y) / denom
    b = (sum_y - k * sum_x) / N
    
    return k, b


# =============================================================================
# 简化物理步进 (用于标定)
# =============================================================================
def simple_dynamics_step(
    state: dict,
    throttle: Float[Array, "batch"],
    rate_cmd: Float[Array, "batch 3"],
    phys_params,
    l3_config,
    dt: float = 0.002
) -> dict:
    """
    简化的动力学步进，用于 OLIP 标定
    
    与 dynamics.py + l3_controller.py 保持物理一致性:
    - 推力模型: throttle² × motor_max_thrust (与 compute_thrust_and_torque 一致)
    - 电机混控: 简化 X 型混控 + clip [0, 1] (与 compute_motor_commands 一致)
    - 旋转阻力: τ_drag = -Cr × |ω| × ω (与 compute_quadratic_drag 一致)
    
    不包含: 风场、碰撞、传感器噪声、四元数旋转、电池模型
    """
    batch_size = throttle.shape[0]
    
    # =========================================================================
    # 1. 推力与垂直加速度 (与 dynamics.py 一致: F = throttle² × F_max)
    # =========================================================================
    thrust_per_motor = throttle ** 2 * phys_params.motor_max_thrust
    total_thrust = thrust_per_motor * 4.0
    acc_z = (total_thrust - phys_params.mass * 9.81) / phys_params.mass
    
    # =========================================================================
    # 2. PID 角速率控制
    # =========================================================================
    rate_err = rate_cmd - state['omega']
    
    # P 项
    p_out = l3_config.kp * rate_err
    
    # I 项更新
    i_integral = state['i_integral'] + l3_config.ki * rate_err * dt
    i_integral = jnp.clip(i_integral, -l3_config.i_limit[..., None], l3_config.i_limit[..., None])
    
    # PID 输出 (归一化力矩需求)
    pid_output = p_out + i_integral  # [batch, 3]
    
    # =========================================================================
    # 3. 简化混控器 + 电机限幅 (与 compute_motor_commands 一致)
    # =========================================================================
    # X 型四旋翼简化混控:
    #   motor[0] (FR) = throttle - roll - pitch + yaw
    #   motor[1] (RL) = throttle + roll - pitch - yaw
    #   motor[2] (FL) = throttle + roll + pitch + yaw
    #   motor[3] (RR) = throttle - roll + pitch - yaw
    #
    # 这里只关心 Roll 轴 (谐波测试轴), 简化为:
    #   motor_high = throttle + |pid_roll|/4   (增推电机)
    #   motor_low  = throttle - |pid_roll|/4   (减推电机)
    # 限幅后反算实际能施加的差动推力
    
    pid_roll = pid_output[..., 0:1]    # [batch, 1]
    pid_pitch = pid_output[..., 1:2]   # [batch, 1]
    pid_yaw = pid_output[..., 2:3]     # [batch, 1]
    
    # 4 个电机指令 (含 roll, pitch, yaw 混控)
    th = throttle[..., None]  # [batch, 1]
    motor_cmds = jnp.concatenate([
        th - pid_roll - pid_pitch + pid_yaw,   # FR
        th + pid_roll - pid_pitch - pid_yaw,   # RL
        th + pid_roll + pid_pitch + pid_yaw,   # FL
        th - pid_roll + pid_pitch - pid_yaw,   # RR
    ], axis=-1)  # [batch, 4]
    
    # 电机限幅 [0, 1] — 这是物理上限
    motor_cmds = jnp.clip(motor_cmds, 0.0, 1.0)
    
    # =========================================================================
    # 4. 从限幅后的电机指令计算实际物理力矩
    # =========================================================================
    # 推力线性化 (与 compute_motor_commands 中 sqrt 对应):
    # dynamics.py: motor_thrust = cmd² × max_thrust
    # l3_controller: cmd_out = sqrt(cmd_linear) (推力线性化)
    # 组合效果: thrust = sqrt(cmd)² × max = cmd × max (线性)
    # 这里 motor_cmds 是线性空间的, 所以直接乘即可
    motor_thrust = motor_cmds * phys_params.motor_max_thrust[..., None]  # [batch, 4]
    
    # 力臂参数
    A = jnp.sin(phys_params.frame_angle)  # Roll 力臂系数
    B = jnp.cos(phys_params.frame_angle)  # Pitch 力臂系数
    arm = phys_params.arm_length
    
    # 实际物理力矩 (与 compute_thrust_and_torque 一致)
    # Roll:  (FL + RL) - (FR + RR)
    roll_torque = (
        (motor_thrust[..., 2] + motor_thrust[..., 1]) -
        (motor_thrust[..., 0] + motor_thrust[..., 3])
    ) * A * arm
    
    # Pitch: (FL + FR) - (RL + RR)
    pitch_torque = (
        (motor_thrust[..., 2] + motor_thrust[..., 0]) -
        (motor_thrust[..., 1] + motor_thrust[..., 3])
    ) * B * arm
    
    # Yaw:   (CW - CCW) × force_to_torque_ratio
    yaw_torque_per_motor = motor_thrust * phys_params.force_to_torque_ratio
    yaw_torque = (
        yaw_torque_per_motor[..., 2] + yaw_torque_per_motor[..., 3] -
        yaw_torque_per_motor[..., 0] - yaw_torque_per_motor[..., 1]
    )
    
    physical_torque = jnp.stack([roll_torque, pitch_torque, yaw_torque], axis=-1)
    
    # =========================================================================
    # 5. 旋转阻力 (与 compute_quadratic_drag 一致)
    # =========================================================================
    # τ_drag = -Cr × |ω| × ω  (二次阻力, 自然限制角速度增长)
    omega_norm = jnp.linalg.norm(state['omega'], axis=-1, keepdims=True)
    drag_torque = -phys_params.rot_drag_coeff * omega_norm * state['omega']
    
    # =========================================================================
    # 6. 角速度更新 (Newton-Euler, 简化为对角惯性)
    # =========================================================================
    inertia = jnp.stack([
        phys_params.inertia_xx,
        phys_params.inertia_yy,
        phys_params.inertia_zz
    ], axis=-1)
    
    # 电机响应滞后 (简化一阶)
    alpha = dt / (phys_params.motor_tau_up[:, 0] + dt)
    
    # 净力矩 = 电机力矩 + 阻力力矩
    net_torque = physical_torque + drag_torque
    omega_dot = net_torque / jnp.maximum(inertia, 1e-6)
    omega_new = state['omega'] + omega_dot * dt * alpha[:, None]
    
    # Euler 积分器安全防护:
    # 极端参数组合 (inertia ~ 1e-5, rot_drag ~ 0.1) 下, 显式 Euler 积分
    # 的数值稳定性不完备。100 rad/s (≈ 5730 deg/s) 远超物理合理范围,
    # 仅作为积分器的数值保护, 不替代任何物理机制。
    omega_new = jnp.clip(omega_new, -100.0, 100.0)
    
    # =========================================================================
    # 7. 位置更新 (NED 坐标系: Z 向下为正)
    # =========================================================================
    # acc_z > 0 表示推力 > 重力 (净力向上), 在 NED 中应减小 vel_z
    vel_new = state['vel'].at[:, 2].set(state['vel'][:, 2] - acc_z * dt)
    pos_new = state['pos'] + vel_new * dt
    
    return {
        'pos': pos_new,
        'vel': vel_new,
        'omega': omega_new,
        'i_integral': i_integral,
        'acc_z': acc_z,
        'throttle': throttle,
    }


# =============================================================================
# Phase 0 & 1: 悬停寻优 + Ramp 测试
# =============================================================================
def run_hover_calibration(
    phys_params,
    l3_config,
    key: PRNGKeyArray,
    num_hover_steps: int = 500,  # 1.0s @ 500Hz
    num_ramp_steps: int = 50
) -> tuple:
    """
    Phase 0 + 1: 悬停油门寻优 + 推力灵敏度测试
    
    返回: (hover_throttle, thrust_gain)
    """
    batch_size = phys_params.mass.shape[0]
    
    # 初始状态
    state = {
        'pos': jnp.zeros((batch_size, 3)),
        'vel': jnp.zeros((batch_size, 3)),
        'omega': jnp.zeros((batch_size, 3)),
        'i_integral': jnp.zeros((batch_size, 3)),
        'acc_z': jnp.zeros((batch_size,)),
        'throttle': jnp.full((batch_size,), 0.5),
    }
    
    # 目标高度
    target_z = 0.0
    
    # Phase 0: 简单高度 PID 找悬停油门
    kp_alt = 0.5
    ki_alt = 0.1
    
    alt_integral = jnp.zeros((batch_size,))
    throttle_history = []
    
    def hover_step(carry, _):
        state, alt_integral = carry
        
        # 高度误差 (NED: Z向下，所以取负)
        alt_err = target_z - (-state['pos'][:, 2])
        
        # PI 控制
        alt_integral_new = alt_integral + ki_alt * alt_err * 0.002
        alt_integral_new = jnp.clip(alt_integral_new, -0.3, 0.3)
        
        throttle = 0.5 + kp_alt * alt_err + alt_integral_new
        throttle = jnp.clip(throttle, 0.1, 0.9)
        
        # 执行一步
        state_new = simple_dynamics_step(
            state, throttle, jnp.zeros((batch_size, 3)),
            phys_params, l3_config, dt=0.002
        )
        
        return (state_new, alt_integral_new), throttle
    
    (state_final, _), throttle_seq = jax.lax.scan(
        hover_step,
        (state, alt_integral),
        None,
        length=num_hover_steps
    )
    
    # 取最后 100 步的平均作为悬停油门
    hover_throttle = jnp.mean(throttle_seq[-100:], axis=0)
    
    # Phase 1: Ramp 测试 - 在悬停点附近线性扫描
    ramp_range = 0.1  # ±10%
    ramp_throttles = jnp.linspace(
        hover_throttle - ramp_range,
        hover_throttle + ramp_range,
        num_ramp_steps
    ).T  # [batch, num_steps]
    
    # 收集加速度响应
    acc_z_samples = []
    
    def ramp_step(state, throttle):
        state_new = simple_dynamics_step(
            state, throttle, jnp.zeros((batch_size, 3)),
            phys_params, l3_config, dt=0.002
        )
        return state_new, state_new['acc_z']
    
    _, acc_z_seq = jax.lax.scan(
        ramp_step,
        state_final,
        ramp_throttles.T  # 转置以便 scan 遍历
    )
    
    # RLS 拟合: acc_z = k * throttle + b
    # 对每个 batch 独立拟合
    def fit_single_batch(idx):
        x = ramp_throttles[idx]
        y = acc_z_seq[:, idx]
        k, b = linear_regression(x, y)
        return k
    
    # 使用 vmap 进行批量拟合
    thrust_gain = jax.vmap(
        lambda i: linear_regression(ramp_throttles[i], acc_z_seq[:, i])[0]
    )(jnp.arange(batch_size))
    
    return hover_throttle, thrust_gain, state_final


# =============================================================================
# Phase 2 & 3: 谐波响应测试
# =============================================================================
def run_harmonic_calibration(
    phys_params,
    l3_config,
    init_state: dict,
    freq: float,
    key: PRNGKeyArray,
    duration: float = 0.75,  # 0.75s 的正弦波
    sample_rate: float = 500.0
) -> tuple:
    """
    谐波响应测试 (4Hz 或 8Hz)
    
    返回: (mag_gain, phase_lag, i_term_effort)
    """
    batch_size = phys_params.mass.shape[0]
    num_steps = int(duration * sample_rate)
    
    # 生成正弦波指令 (20 deg/s 振幅)
    amplitude = 20.0 * jnp.pi / 180.0  # 转换为 rad/s
    t = jnp.arange(num_steps) / sample_rate
    sine_cmd = amplitude * jnp.sin(2.0 * jnp.pi * freq * t)  # [num_steps]
    
    # 扩展到 batch
    sine_cmd_batch = jnp.broadcast_to(sine_cmd[None, :], (batch_size, num_steps))
    
    # 收集响应
    gyro_responses = []
    i_term_responses = []
    
    def harmonic_step(state, cmd):
        # 只对 Roll 轴注入正弦波
        rate_cmd = jnp.zeros((batch_size, 3))
        rate_cmd = rate_cmd.at[:, 0].set(cmd)
        
        state_new = simple_dynamics_step(
            state, state['throttle'], rate_cmd,
            phys_params, l3_config, dt=1.0/sample_rate
        )
        
        return state_new, (state_new['omega'][:, 0], state_new['i_integral'][:, 0])
    
    _, (gyro_seq, i_term_seq) = jax.lax.scan(
        harmonic_step,
        init_state,
        sine_cmd_batch.T  # [num_steps, batch]
    )
    
    # 转置得到 [batch, num_steps]
    gyro_seq = gyro_seq.T
    i_term_seq = i_term_seq.T
    
    # DFT 分析
    # 对每个 batch 计算 DFT
    def analyze_batch(idx):
        cmd_mag, cmd_phase = dft_single_freq(sine_cmd, freq, sample_rate)
        gyro_mag, gyro_phase = dft_single_freq(gyro_seq[idx], freq, sample_rate)
        
        # 增益比
        gain = gyro_mag / (cmd_mag + 1e-10)
        
        # 相位滞后
        phase_lag = gyro_phase - cmd_phase
        
        # 积分项努力
        i_term_mag, _ = dft_single_freq(i_term_seq[idx], freq, sample_rate)
        i_effort = i_term_mag / (cmd_mag + 1e-10)
        
        return gain, phase_lag, i_effort
    
    gains, phase_lags, i_efforts = jax.vmap(analyze_batch)(jnp.arange(batch_size))
    
    return gains, phase_lags, i_efforts, gyro_seq, i_term_seq


# =============================================================================
# 主标定函数
# =============================================================================
def virtual_calibrator(
    phys_params,
    l3_config,
    key: PRNGKeyArray
) -> Float[Array, "batch 11"]:
    """
    OLIP v19.0 Virtual Calibrator
    
    在仿真中运行完整标定流程，生成 11 维特征向量。
    
    Args:
        phys_params: 物理参数 (PhysParams)
        l3_config: L3 控制配置 (L3Config)
        key: JAX 随机密钥
        
    Returns:
        olip_vec: [batch, 11] 特征向量
    """
    batch_size = phys_params.mass.shape[0]
    keys = jax.random.split(key, 5)
    
    # =========================================================================
    # Phase 0 & 1: 悬停 + Ramp
    # =========================================================================
    hover_throttle, thrust_gain, hover_state = run_hover_calibration(
        phys_params, l3_config, keys[0]
    )
    
    # =========================================================================
    # Phase 2: 4Hz 谐波响应
    # =========================================================================
    gain_4hz, phase_4hz, i_effort_4hz, gyro_4hz, i_term_4hz = run_harmonic_calibration(
        phys_params, l3_config, hover_state, freq=4.0, key=keys[1]
    )
    
    # =========================================================================
    # Phase 3: 8Hz 谐波响应
    # =========================================================================
    gain_8hz, phase_8hz, _, _, _ = run_harmonic_calibration(
        phys_params, l3_config, hover_state, freq=8.0, key=keys[2]
    )
    
    # =========================================================================
    # 特征提取
    # =========================================================================
    
    # [0] Lag_Phase_4Hz: 4Hz 相位滞后
    feat_lag_4hz = phase_4hz
    
    # [1] Lag_Phase_8Hz: 8Hz 与 4Hz 的相位差 (惯性滞后)
    feat_lag_8hz_diff = phase_8hz - phase_4hz
    
    # [2] Gain_Mag_4Hz: 4Hz 增益
    feat_gain_4hz = gain_4hz
    
    # [3] Gain_Mag_8Hz: 8Hz 增益
    feat_gain_8hz = gain_8hz
    
    # [4-6] Integ_Effort: 积分努力 (Roll, Pitch, Yaw)
    # 这里简化为只测 Roll 轴，然后复制到其他轴 (实际应分别测)
    feat_integ_r = i_effort_4hz
    feat_integ_p = i_effort_4hz * jax.random.uniform(keys[3], (batch_size,), minval=0.9, maxval=1.1)
    feat_integ_y = i_effort_4hz * jax.random.uniform(keys[4], (batch_size,), minval=0.8, maxval=1.2)
    
    # [7] Auth_Asymmetry: 控制不对称性
    # 计算积分项正向和负向累积的差异
    i_positive = jnp.sum(jnp.maximum(i_term_4hz, 0), axis=-1)
    i_negative = jnp.sum(jnp.minimum(i_term_4hz, 0), axis=-1)
    feat_asymmetry = jnp.abs(i_positive + i_negative) / (jnp.abs(i_positive) + jnp.abs(i_negative) + 1e-10)
    
    # [8] Hover_Throttle: 悬停油门
    feat_hover_throttle = hover_throttle
    
    # [9] Thrust_Gain: 推力灵敏度
    feat_thrust_gain = thrust_gain
    
    # [10] Nonlinear_Factor: 非线性度
    # 计算信号中非谐波成分的比例
    # 简化: 使用 1 - (基频能量 / 总能量)
    total_energy = jnp.sum(gyro_4hz**2, axis=-1)
    # 基频能量近似为 mag^2 * N / 2
    N = gyro_4hz.shape[-1]
    fundamental_energy = gain_4hz**2 * N / 2.0
    spectral_purity = fundamental_energy / (total_energy + 1e-10)
    feat_nonlinear = 1.0 - jnp.clip(spectral_purity, 0.0, 1.0)
    
    # =========================================================================
    # 归一化 (Z-Score 近似)
    # =========================================================================
    # 使用经验统计值进行归一化
    def normalize(x, mean, std):
        return (x - mean) / (std + 1e-10)
    
    feat_lag_4hz_norm = normalize(feat_lag_4hz, -0.5, 0.3)
    feat_lag_8hz_diff_norm = normalize(feat_lag_8hz_diff, -0.3, 0.2)
    feat_gain_4hz_norm = normalize(feat_gain_4hz, 1.0, 0.3)
    feat_gain_8hz_norm = normalize(feat_gain_8hz, 0.7, 0.3)
    feat_integ_r_norm = normalize(feat_integ_r, 0.2, 0.1)
    feat_integ_p_norm = normalize(feat_integ_p, 0.2, 0.1)
    feat_integ_y_norm = normalize(feat_integ_y, 0.2, 0.1)
    feat_asymmetry_norm = feat_asymmetry  # 已在 [0, 1]
    feat_hover_norm = normalize(feat_hover_throttle, 0.5, 0.15)
    feat_thrust_norm = normalize(feat_thrust_gain, 10.0, 5.0)
    feat_nonlinear_norm = feat_nonlinear  # 已在 [0, 1]
    
    # =========================================================================
    # 组装 OLIP 向量
    # =========================================================================
    olip_vec = jnp.stack([
        feat_lag_4hz_norm,       # [0]
        feat_lag_8hz_diff_norm,  # [1]
        feat_gain_4hz_norm,      # [2]
        feat_gain_8hz_norm,      # [3]
        feat_integ_r_norm,       # [4]
        feat_integ_p_norm,       # [5]
        feat_integ_y_norm,       # [6]
        feat_asymmetry_norm,     # [7]
        feat_hover_norm,         # [8]
        feat_thrust_norm,        # [9]
        feat_nonlinear_norm,     # [10]
    ], axis=-1)
    
    # =========================================================================
    # 注入 Sim2Real 噪声 (±5%)
    # =========================================================================
    noise = jax.random.uniform(keys[4], (batch_size, 11), minval=0.95, maxval=1.05)
    olip_vec = olip_vec * noise
    
    # 安全防护: 防止标定过程中的数值溢出传播到 obs
    olip_vec = jnp.nan_to_num(olip_vec, nan=0.0, posinf=3.0, neginf=-3.0)
    olip_vec = jnp.clip(olip_vec, -3.0, 3.0)
    
    return olip_vec


# JIT 编译版本
virtual_calibrator_jit = jax.jit(virtual_calibrator)
