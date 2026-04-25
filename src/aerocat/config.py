"""
AeroCat v18.0 Training Configuration

Centralized configuration management for the entire training pipeline.
Bridging user-friendly JSON config to JAX-compatible efficient structs.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
import json
from pathlib import Path
import math

# Import JAX-compatible configs for conversion
from .envs.uav_env import EnvConfig as RuntimeEnvConfig, RewardConfig as RuntimeRewardConfig


@dataclass
class PhysicsConfig:
    """PF-GDR Domain Randomization Configuration"""
    # Core Constraints
    mach_limit: float = 0.65
    v_sound: float = 340.0
    air_density: float = 1.225
    gravity: float = 9.81
    
    # Randomization Ranges (Min, Max)
    prop_diameter_range: List[float] = field(default_factory=lambda: [0.05, 0.40])
    thrust_coeff_range: List[float] = field(default_factory=lambda: [0.08, 0.12])
    twr_range: List[float] = field(default_factory=lambda: [2.5, 12.0])
    
    # Battery
    voltage_per_cell_range: List[float] = field(default_factory=lambda: [3.5, 4.2])
    
    # Wind & Atmosphere
    wind_speed_range: List[float] = field(default_factory=lambda: [0.0, 15.0])
    turbulence_intensity_range: List[float] = field(default_factory=lambda: [0.05, 0.25])


@dataclass
class RewardConfig:
    """Reward Function Weights (v18.7: Huber 核, 4 核心项)"""
    # v18.7 权重表
    w_velocity: float = 0.45
    w_direction: float = 0.35
    w_spin: float = 0.05
    w_act: float = 0.05
    c_alive: float = 0.025
    
    # Huber 参数
    vel_huber_delta: float = 0.2
    vel_huber_max: float = 5.0
    spin_huber_delta: float = 1.0
    spin_huber_max: float = 15.0
    act_huber_delta: float = 0.1
    act_huber_max: float = 2.0
    
    # [DISABLED] 保留字段向后兼容
    w_yaw: float = 0.0
    w_upright: float = 0.0
    w_pos: float = 0.0
    k_xy: float = 0.05
    k_z: float = 0.10
    lambda_vel: float = 0.15
    k_pos: float = 0.5
    
    def to_runtime(self) -> RuntimeRewardConfig:
        return RuntimeRewardConfig(
            w_velocity=self.w_velocity,
            w_direction=self.w_direction,
            w_yaw=self.w_yaw,
            w_upright=self.w_upright,
            w_spin=self.w_spin,
            w_act=self.w_act,
            c_alive=self.c_alive,
            w_pos=self.w_pos,
            w_align=0.0,
            k_xy=self.k_xy,
            k_z=self.k_z,
            lambda_vel=self.lambda_vel,
            k_pos=self.k_pos,
            vel_huber_delta=self.vel_huber_delta,
            vel_huber_max=self.vel_huber_max,
            spin_huber_delta=self.spin_huber_delta,
            spin_huber_max=self.spin_huber_max,
            act_huber_delta=self.act_huber_delta,
            act_huber_max=self.act_huber_max,
        )


@dataclass
class ObsNormalizationConfig:
    """Observation Normalization Constants"""
    gyro_scale: float = 35.0          # rad/s (~2000 deg/s)
    accel_scale: float = 49.05        # m/s^2 (~5G)
    latency_offset: float = 20.0
    latency_scale: float = 40.0


@dataclass
class EnvConfig:
    """Environment Setup"""
    # Frequencies (Master Source)
    physics_freq: int = 500
    control_freq: int = 50
    
    # Episode
    max_episode_time: float = 10.0    # seconds (延长以支持 mid-episode 故障注入)
    
    # Delay Buffer
    buffer_size: int = 25
    
    # Initialization
    init_height_range: List[float] = field(default_factory=lambda: [0.5, 5.0])
    init_max_tilt: float = 3.14
    init_max_rate: float = 15.0
    
    def to_runtime(self) -> RuntimeEnvConfig:
        dt_rl = 1.0 / self.control_freq
        dt_sub = 1.0 / self.physics_freq
        num_substeps = int(self.physics_freq / self.control_freq)
        
        return RuntimeEnvConfig(
            dt_rl=dt_rl,
            dt_sub=dt_sub,
            num_substeps=num_substeps,
            max_episode_time=self.max_episode_time,
            buffer_size=self.buffer_size,
            init_height_range=tuple(self.init_height_range),
            init_max_tilt=self.init_max_tilt,
            init_max_rate=self.init_max_rate
        )


@dataclass
class NetworkConfig:
    """Neural Network Architecture"""
    encoder_dims: List[int] = field(default_factory=lambda: [64, 48])
    lstm_dim: int = 48
    decoder_dims: List[int] = field(default_factory=lambda: [48, 32])
    actor_dim: int = 4
    critic_dim: int = 1
    activation: str = "relu"
    use_layer_norm: bool = True


@dataclass
class PPOConfig:
    """PPO Algorithm Configuration"""
    # 保守参数 (实验 #0/#5 验证: BC→PPO 平稳过渡, lambda→0.5)
    learning_rate: float = 5e-5
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_eps: float = 0.2             # v19.0.1: 0.1→0.2 (PPO 标准值，降低 clip_frac 拥堵)
    vf_coef: float = 0.5             # v19.0.1: 1.0→0.5 (标准 PPO，防 vf 梯度主导)
    ent_coef: float = 2e-3
    # Entropy 动态控制
    entropy_target_min: float = -4.5
    entropy_target_max: float = -3.5
    entropy_stop_threshold: float = -3.0
    entropy_smooth_window: int = 20
    entropy_cooldown: int = 100
    max_grad_norm: float = 0.5
    
    num_minibatches: int = 8
    update_epochs: int = 4
    minibatch_size: int = 32768
    
    lr_schedule: str = "linear"
    warmup_frac: float = 0.01
    
    # Critic 预热: 延长冻结期让 Critic 先适应 BC 策略的 value landscape
    critic_warmup_steps: int = 100    # v19.0.1: 50→100 (延长 warmup 让 Critic 先稳定)
    
    # 内循环 Scan: 将 K 次 step_fn 包进 jax.lax.scan，减少 GPU↔Host 同步
    scan_iters: int = 10             # 每次 Python 循环执行 K 次 JIT 迭代

    # KL 惩罚系数 (已废弃): PPO 采用 Surrogate Objective Clipping 机制
    # 在 clipped loss 上附加常数系数的 KL 惩罚会导致震荡，已从 loss 中移除
    kl_coef: float = 0.0
    
    pretrained_path: str = None


@dataclass
class CurriculumConfig:
    """APDC Curriculum Configuration"""
    lambda_start: float = 0.0
    lambda_end: float = 1.0
    
    # 课程进阶阈值设定 (基于理论计算)
    # stable_threshold: 奖励高于此值的帧视为"稳定帧" (对应 θ ≤ 20° 姿态误差)
    # success_threshold: 稳定帧比例达到此值才进阶课程
    stable_threshold: float = 0.50
    success_threshold: float = 0.55
    
    lambda_increase_rate: float = 0.01
    lambda_decrease_rate: float = 0.005
    
    # Weights for curriculum components
    wind_weight: float = 1.0
    turbulence_weight: float = 1.0
    sensor_noise_weight: float = 0.8


@dataclass
class LoggingConfig:
    """Logging Configuration"""
    log_dir: str = "logs"
    experiment_name: str = "aerocat_ppo"
    log_interval: int = 10
    eval_interval: int = 500       # 每 500 迭代评估一次（Video 录制很慢）
    save_interval: int = 200       # 每 200 迭代保存曲线/JSON
    use_tensorboard: bool = True
    use_json_log: bool = True
    record_video: bool = False     # 训练期间不录制 Video，减少 GPU↔Host 同步


@dataclass
class CheckpointConfig:
    """Checkpoint Configuration"""
    save_dir: str = "checkpoints"
    save_every_steps: int = 10_000_000
    keep_last_n: int = -1          # -1 = 保留所有 checkpoint，不自动删除
    resume_from: Optional[str] = None


@dataclass
class TrainConfig:
    """Root Training Configuration"""
    # Core
    seed: int = 42
    total_timesteps: int = 500_000_000  # v19.0.2: 22B→500M (合理默认值)
    num_envs: int = 4096
    num_steps: int = 128
    
    # Sub-configs
    env: EnvConfig = field(default_factory=EnvConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    obs_normalization: ObsNormalizationConfig = field(default_factory=ObsNormalizationConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    
    # Hardware
    jax_enable_x64: bool = False
    jax_platform: str = "gpu"

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.num_steps
    
    def get_runtime_env_config(self) -> RuntimeEnvConfig:
        """Convert to JAX-compatible runtime config"""
        # Base Env Config
        runtime_config = self.env.to_runtime()
        
        # Inject Reward Config
        runtime_config = runtime_config.replace(
            reward_config=self.reward.to_runtime()
        )
        
        return runtime_config

    @property
    def num_updates(self) -> int:
        return self.total_timesteps // self.batch_size
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        import dataclasses
        return dataclasses.asdict(self)
    
    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "TrainConfig":
        with open(path, "r") as f:
            data = json.load(f)
        
        # Helper to recursively load dataclasses
        # Note: This is a simple implementation. For production, use libraries like dacite.
        # Here we manually map known sub-configs
        
        def load_sub(cls_type, data_dict):
            if data_dict is None: return cls_type()
            # Filter keys that exist in the dataclass
            valid_keys = cls_type.__dataclass_fields__.keys()
            filtered = {k: v for k, v in data_dict.items() if k in valid_keys}
            return cls_type(**filtered)

        config = cls()
        
        # Load Top-Level
        for k in ["seed", "total_timesteps", "num_envs", "num_steps", "jax_enable_x64", "jax_platform"]:
            if k in data:
                setattr(config, k, data[k])

        # Load Sections
        config.env = load_sub(EnvConfig, data.get("env"))
        config.physics = load_sub(PhysicsConfig, data.get("physics"))
        config.reward = load_sub(RewardConfig, data.get("reward"))
        config.obs_normalization = load_sub(ObsNormalizationConfig, data.get("obs_normalization"))
        config.network = load_sub(NetworkConfig, data.get("network"))
        config.ppo = load_sub(PPOConfig, data.get("ppo"))
        config.curriculum = load_sub(CurriculumConfig, data.get("curriculum"))
        config.logging = load_sub(LoggingConfig, data.get("logging"))
        config.checkpoint = load_sub(CheckpointConfig, data.get("checkpoint"))
        
        return config

# Helpers
def get_default_config() -> TrainConfig:
    return TrainConfig()

def get_debug_config() -> TrainConfig:
    c = TrainConfig()
    c.total_timesteps = 100_000
    c.num_envs = 64
    c.num_steps = 32
    c.logging.log_interval = 1
    return c

def get_fast_config() -> TrainConfig:
    c = TrainConfig()
    c.total_timesteps = 1_000_000
    c.num_envs = 1024
    return c
