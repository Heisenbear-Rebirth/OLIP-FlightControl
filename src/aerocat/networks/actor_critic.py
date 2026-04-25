"""
AeroCat v19.0 L2 Neural Network - Physics-Structured Actor-Critic

核心变更 (v19.0):
  1. Physics-Structured Critic (PSC): V = V_phys(s;w) + V_res(s;θ)
     - V_phys: 5 个解析物理基函数 × 5 个可学习权重（零网络参数）
     - V_res: 标准 MLP Critic 网络（学习物理无法解释的残差）
     理论基础: Ng 1999 (策略不变性), PINN (Raissi 2019)
  2. tanh·scale 替换 Signed Power Mapping（修复原点梯度消失）

网络架构 (约 178k 参数):
- Encoder MLP: 42 -> 128 -> 128 (LayerNorm + Tanh)
- LSTM Core: 128 units (OptimizedLSTMCell)
- Decoder MLP: 128 -> 128 -> 64 (Tanh)
- Actor Head: 64 -> 4 (Tanh)
- Critic: V_phys(5 weights) + V_res(Encoder -> Dense -> 1)

参考: NCA_FCS_v19.0_Design.md
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from jaxtyping import Array, Float, PRNGKeyArray
from typing import Tuple, Optional, NamedTuple


# =============================================================================
# 常量定义
# =============================================================================
OBS_DIM = 31            # 观测维度 (v19.0: 移除 OLIP 11D, 新增 L3 诊断 5D)
ACTION_DIM = 4          # 动作维度 (Roll rate, Pitch rate, Yaw rate, Thrust)
LSTM_DIM = 128          # LSTM 隐藏维度

# 动作映射参数 (v19.0: tanh·scale 替换 Signed Power)
MAX_RP_RATE = 18.85     # Roll/Pitch 最大角速率 (rad/s) ≈ 1080°/s
MAX_YAW_RATE = 34.91    # Yaw 最大角速率 (rad/s) ≈ 2000°/s

# PSC 物理基函数索引 (31D obs 中的位置)
# obs[0:4]   = quaternion → tilt = 2*(qx²+qy²)
# obs[4:7]   = v_actual (heading frame)
# obs[7:10]  = angular_velocity (body frame)
# obs[13:16] = pid_integral (3D, 已归一化)
# obs[16]    = saturation (clipped [0,1])
# obs[22:25] = v_cmd (heading frame) → v_error = v_cmd - v_actual
PSC_NUM_BASIS = 5       # V_phys 基函数数量


# =============================================================================
# 隐状态结构
# =============================================================================
class LSTMState(NamedTuple):
    """LSTM 隐状态 (h, c)"""
    h: Float[Array, "batch lstm_dim"]
    c: Float[Array, "batch lstm_dim"]


# =============================================================================
# 动作映射 (v19.0: tanh·scale 替换 Signed Power)
# =============================================================================
def map_action_to_physical(raw_action: Float[Array, "batch 4"]) -> Float[Array, "batch 4"]:
    """
    将网络原始输出 (tanh, [-1,1]) 线性映射到物理值
    
    v19.0 变更: 移除 Signed Power (|x|^2.5) → 使用纯 tanh·scale
    原因: Signed Power 在原点梯度为 0 → 策略梯度消失 → PPO 无法学习细微调整
          tanh 在原点梯度 = 1 → 细微残差可学
    
    映射:
    - Action 0-1: Roll/Pitch rate → tanh(x) * MAX_RP_RATE
    - Action 2:   Yaw rate → tanh(x) * MAX_YAW_RATE
    - Action 3:   Thrust → (tanh(x) + 1) / 2 → [0, 1]
    """
    roll_rate = raw_action[..., 0] * MAX_RP_RATE
    pitch_rate = raw_action[..., 1] * MAX_RP_RATE
    yaw_rate = raw_action[..., 2] * MAX_YAW_RATE
    thrust = (raw_action[..., 3] + 1.0) / 2.0  # [-1, 1] -> [0, 1]
    
    return jnp.stack([roll_rate, pitch_rate, yaw_rate, thrust], axis=-1)


# =============================================================================
# Physics-Structured Critic (PSC) — V_phys 解析项
# =============================================================================
def compute_v_physics(
    obs: Float[Array, "batch obs_dim"],
    w: Float[Array, "5"],
    b: Float[Array, ""] = None
) -> Float[Array, "batch"]:
    """
    Physics-Structured Critic 的解析物理价值函数 V_phys(s; w, b)。
    
    V_phys = b + w·φ(s)
    
    5 个可学习权重 × 5 个解析基函数 + 1 个可学习偏置，无网络参数。
    
    v19.0.1 修复: 加偏置项 b。
    根因: 所有 φ 都是 -x² 形式 → V_phys 恒 ≤ 0
          但真实 V(s) ≈ mean_reward/(1-γ) ≈ 30（正值）
          → V_res 被迫独立学全部正值 → 暴涨到 28 → 丧失 PSC 优势
    修复: b_init ≈ 30（典型正值回报），让 V_phys 从正确量级起步
    
    基函数选取依据:
    - φ_vel:  速度误差代价 → 越偏离目标越差
    - φ_ang:  角速度代价 → 旋转越快越不安全
    - φ_tilt: 倾斜代价 → 越倾斜越接近翻转
    - φ_int:  PID 积分器负担 → 系统越挣扎越不健康
    - φ_sat:  饱和代价 → 执行器饱和意味着失控风险
    
    理论保证: V_phys(s) 是 s 的确定函数 → 不影响 A(s,a) 的相对排序
              → 不改变最优策略 (Ng et al. 1999)
    
    Args:
        obs: 观测向量 [batch, obs_dim]（已归一化）
        w: PSC 可学习权重 [5]（标量）
        b: PSC 可学习偏置 [1]（标量），默认 0.0
    
    Returns:
        v_phys: 物理价值估计 [batch]
    """
    if b is None:
        b = jnp.float32(0.0)
    
    # φ_1: 速度误差代价
    # obs[22:25] = v_cmd, obs[4:7] = v_actual → 均已归一化到 /15.0
    v_error = obs[..., 22:25] - obs[..., 4:7]   # heading frame 速度误差
    phi_vel = -jnp.sum(v_error**2, axis=-1)       # 越大越差
    
    # φ_2: 角速度代价
    # obs[7:10] = angular_velocity（已归一化到 /GYRO_SCALE）
    omega = obs[..., 7:10]
    phi_ang = -jnp.sum(omega**2, axis=-1)
    
    # φ_3: 倾斜代价
    # obs[0:4] = quaternion [w, x, y, z]
    # tilt_metric = 2*(qx²+qy²) ≈ 1-cos(tilt) for small angles
    qx = obs[..., 1]
    qy = obs[..., 2]
    phi_tilt = -2.0 * (qx**2 + qy**2)   # 0=水平, -2=完全翻转
    
    # φ_4: 积分器负担
    # obs[13:16] = pid_integral（已归一化到 ±1）
    i_term = obs[..., 13:16]
    phi_int = -jnp.sum(i_term**2, axis=-1)
    
    # φ_5: 饱和代价
    # obs[16] = saturation (clipped [0,1])
    sat = obs[..., 16]
    phi_sat = -sat**2
    
    # 加权求和 + 偏置
    v_phys = b + (w[0] * phi_vel + w[1] * phi_ang + w[2] * phi_tilt 
                  + w[3] * phi_int + w[4] * phi_sat)
    
    return v_phys


# =============================================================================
# Encoder MLP
# =============================================================================
class Encoder(nn.Module):
    """
    编码器 MLP: 37 -> 128 -> 128
    
    使用 LayerNorm + Tanh 激活。
    """
    hidden_dim: int = 128
    output_dim: int = 128
    
    @nn.compact
    def __call__(self, x: Float[Array, "batch obs_dim"]) -> Float[Array, "batch output_dim"]:
        # Layer 1: 37 -> 128
        x = nn.Dense(self.hidden_dim, name="fc1")(x)
        x = nn.LayerNorm(name="ln1")(x)
        x = nn.tanh(x)
        
        # Layer 2: 128 -> 128
        x = nn.Dense(self.output_dim, name="fc2")(x)
        x = nn.LayerNorm(name="ln2")(x)
        x = nn.tanh(x)
        
        return x


# =============================================================================
# Decoder MLP
# =============================================================================
class Decoder(nn.Module):
    """
    解码器 MLP: 128 -> 128 -> 64
    
    使用 Tanh 激活。
    """
    hidden_dim: int = 128
    output_dim: int = 64
    
    @nn.compact
    def __call__(self, x: Float[Array, "batch lstm_dim"]) -> Float[Array, "batch output_dim"]:
        # Layer 1: 128 -> 128
        x = nn.Dense(self.hidden_dim, name="fc1")(x)
        x = nn.tanh(x)
        
        # Layer 2: 128 -> 64
        x = nn.Dense(self.output_dim, name="fc2")(x)
        x = nn.tanh(x)
        
        return x


# =============================================================================
# Actor-Critic Network (Sandwich Architecture)
# =============================================================================
class ActorCritic(nn.Module):
    """
    Sandwich Architecture Actor-Critic 网络
    
    结构:
        Obs -> Encoder -> LSTM -> Decoder -> (Actor, Critic)
    
    特点:
    - 使用 OptimizedLSTMCell 处理时序信息
    - Actor 输出 Tanh 激活 (后续使用 Signed Power 映射)
    - Critic 输出单一值
    """
    lstm_dim: int = LSTM_DIM
    
    def setup(self):
        self.encoder = Encoder()
        self.lstm = nn.OptimizedLSTMCell(features=self.lstm_dim, name="lstm")
        self.decoder = Decoder()
        
        # Actor head: 64 -> 4 (Tanh)
        self.actor_head = nn.Dense(ACTION_DIM, name="actor")
        
        # Critic head: 64 -> 1
        self.critic_head = nn.Dense(1, name="critic")
    
    def __call__(
        self,
        obs: Float[Array, "batch obs_dim"],
        lstm_state: LSTMState,
        done: Optional[Float[Array, "batch"]] = None
    ) -> Tuple[Float[Array, "batch action_dim"], Float[Array, "batch"], LSTMState]:
        """
        前向传播
        
        Args:
            obs: 观测 [batch, 32]
            lstm_state: LSTM 隐状态 (h, c)
            done: episode 结束标志 (用于重置隐状态)
        
        Returns:
            action: 原始动作输出 (Tanh) [batch, 4]
            value: 状态价值 [batch]
            new_lstm_state: 更新后的 LSTM 状态
        """
        batch_size = obs.shape[0]
        
        # 1. 如果 episode 结束，重置 LSTM 状态
        if done is not None:
            # done 为 True 时重置隐状态
            reset_mask = done[:, None]  # [batch, 1]
            h = lstm_state.h * (1.0 - reset_mask)
            c = lstm_state.c * (1.0 - reset_mask)
            lstm_state = LSTMState(h=h, c=c)
        
        # 2. Encoder
        encoded = self.encoder(obs)  # [batch, 128]
        
        # 3. LSTM
        # OptimizedLSTMCell 期望 carry = (c, h)
        carry = (lstm_state.c, lstm_state.h)
        new_carry, lstm_out = self.lstm(carry, encoded)
        new_c, new_h = new_carry
        new_lstm_state = LSTMState(h=new_h, c=new_c)
        
        # 4. Decoder
        decoded = self.decoder(lstm_out)  # [batch, 64]
        
        # 5. Actor Head (Tanh activation)
        raw_action = nn.tanh(self.actor_head(decoded))  # [batch, 4]
        
        # 6. Critic Head
        value = self.critic_head(decoded).squeeze(-1)  # [batch]
        
        return raw_action, value, new_lstm_state
    
    def forward_mlp_only(
        self,
        obs: Float[Array, "batch obs_dim"]
    ) -> Tuple[Float[Array, "batch action_dim"], Float[Array, "batch"]]:
        """
        仅 MLP 前向传播 (用于 BC 预训练)
        
        跳过 LSTM，直接将 Encoder 输出传给 Decoder。
        这允许 BC 阶段随机打乱数据，避免 LSTM 过拟合专家策略。
        
        设计理由:
        1. LSTM 主要用于估计不可观测状态（风速、电机老化）
        2. 专家策略拥有仿真器"上帝视角"，不需要 LSTM 估计
        3. BC 训练 LSTM 会让它"依赖"完美的专家知识
        4. PPO 阶段 LSTM 从零学习，学到的是探索性记忆
        
        Args:
            obs: 观测 [batch, 32]
        
        Returns:
            raw_action: 原始动作输出 (Tanh) [batch, 4]
            value: 状态价值 [batch]
        """
        # 1. Encoder
        encoded = self.encoder(obs)  # [batch, 128]
        
        # 2. 跳过 LSTM，直接使用 Encoder 输出作为 Decoder 输入
        # 注意：Encoder 输出维度 == LSTM 输出维度 == 128
        
        # 3. Decoder
        decoded = self.decoder(encoded)  # [batch, 64]
        
        # 4. Actor Head (Tanh activation)
        raw_action = nn.tanh(self.actor_head(decoded))  # [batch, 4]
        
        # 5. Critic Head
        value = self.critic_head(decoded).squeeze(-1)  # [batch]
        
        return raw_action, value
    
    def init_lstm_state(self, batch_size: int) -> LSTMState:
        """初始化 LSTM 隐状态为零"""
        return LSTMState(
            h=jnp.zeros((batch_size, self.lstm_dim)),
            c=jnp.zeros((batch_size, self.lstm_dim))
        )


# =============================================================================
# 推理辅助函数
# =============================================================================
def create_actor_critic(
    rng: PRNGKeyArray,
    batch_size: int = 1
) -> Tuple[ActorCritic, dict, LSTMState]:
    """
    创建 Actor-Critic 网络并初始化参数
    
    Args:
        rng: JAX 随机密钥
        batch_size: 批量大小 (用于初始化)
    
    Returns:
        model: ActorCritic 模块
        params: 初始化的参数
        init_state: 初始 LSTM 状态
    """
    model = ActorCritic()
    
    # 创建虚拟输入
    dummy_obs = jnp.zeros((batch_size, OBS_DIM))
    init_state = model.init_lstm_state(batch_size)
    
    # 初始化参数
    params = model.init(rng, dummy_obs, init_state)
    
    return model, params, init_state


def inference_step(
    model: ActorCritic,
    params: dict,
    obs: Float[Array, "batch obs_dim"],
    lstm_state: LSTMState,
    done: Optional[Float[Array, "batch"]] = None
) -> Tuple[Float[Array, "batch action_dim"], Float[Array, "batch"], LSTMState]:
    """
    推理步骤 (封装 apply)
    
    Returns:
        physical_action: 物理动作 (已映射)
        value: 状态价值
        new_lstm_state: 新的 LSTM 状态
    """
    raw_action, value, new_state = model.apply(params, obs, lstm_state, done)
    physical_action = map_action_to_physical(raw_action)
    return physical_action, value, new_state


# =============================================================================
# 用于 PPO 的辅助结构
# =============================================================================
class Transition(NamedTuple):
    """PPO 训练数据结构"""
    obs: Float[Array, "batch obs_dim"]
    action: Float[Array, "batch action_dim"]
    reward: Float[Array, "batch"]
    done: Float[Array, "batch"]
    value: Float[Array, "batch"]
    log_prob: Float[Array, "batch"]


def compute_log_prob(
    raw_action: Float[Array, "batch action_dim"],
    action_mean: Float[Array, "batch action_dim"],
    action_std: Float[Array, "batch action_dim"]
) -> Float[Array, "batch"]:
    """
    计算动作的对数概率 (Tanh Normal)
    
    Args:
        raw_action: Pre-Tanh 动作 (u) ~ N(mean, std)
        action_mean: 高斯均值
        action_std: 高斯标准差
    
    Returns:
        log_prob: Post-Tanh 动作 (a = tanh(u)) 的对数概率
    """
    # 1. 高斯分布对数概率 log p(u)
    var = action_std ** 2
    gaussian_log_prob = -0.5 * jnp.sum(
        (raw_action - action_mean) ** 2 / var + 
        jnp.log(2 * jnp.pi * var),
        axis=-1
    )
    
    # 2. Tanh 变换修正 (Jacobian)
    # a = tanh(u)
    # p(a) = p(u) / |det(da/du)|
    # log p(a) = log p(u) - sum(log(1 - tanh(u)^2))
    # 使用 softplus 技巧增强数值稳定性: 
    # log(1 - tanh(u)^2) = 2 * (log(2) - u - softplus(-2u))
    
    # 简单实现 (配合 1e-6 偏移):
    # adjustment = jnp.sum(jnp.log(1.0 - jnp.tanh(raw_action)**2 + 1e-6), axis=-1)
    
    # 使用稳定公式:
    adjustment = jnp.sum(
        2.0 * (jnp.log(2.0) - raw_action - jax.nn.softplus(-2.0 * raw_action)),
        axis=-1
    )
    
    log_prob = gaussian_log_prob - adjustment
    
    return log_prob


# =============================================================================
# 随机策略 (用于 PPO 探索)
# =============================================================================
class StochasticActorCritic(nn.Module):
    """
    随机策略版本的 Actor-Critic
    
    Actor 输出均值和 log_std，采样时使用重参数化技巧。
    """
    lstm_dim: int = LSTM_DIM
    log_std_init: float = -3.0  # 固定 log_std (std≈0.05)。sweep数据：stoch SR=76.4%，接近80%进阶閘值
    
    def setup(self):
        # === Actor 路径 (接受 BC 预训练参数) ===
        self.actor_encoder = Encoder()
        self.lstm = nn.OptimizedLSTMCell(features=self.lstm_dim, name="lstm")
        self.decoder = Decoder()
        # v19.0.2: actor_mean 零初始化
        # 初始策略输出 action ≈ 0 → target_rates ≈ 0, thrust ≈ 0.5
        # → L3 PID 维持悬停。RL 从安全基线开始探索小幅修正。
        # 之前 lecun_normal 初始化导致初始输出 ±100 deg/s → PID 被干扰
        self.actor_mean = nn.Dense(
            ACTION_DIM, name="actor_mean",
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros
        )

        # log_std: 可训练参数（v18.1: 从固定常数改为 nn.Parameter）
        # 初始化为 -3.0 (保持与 BC 预训练一致)，裁剪到 [-5, 0] 防止探索爆炸
        # std 范围: exp(-5)=0.007 ~ exp(0)=1.0
        self.log_std = self.param(
            "log_std",
            lambda rng, shape: jnp.full(shape, self.log_std_init),
            (ACTION_DIM,)
        )

        # === Critic 路径 (v19.0 PSC: V = V_phys + V_res) ===
        # V_phys: 解析物理项（5 个可学习权重，无 hidden layer）
        # V_res:  独立网络学习残差（梯度不触及 Actor 参数）
        
        # PSC 权重: 初始化为 reward 权重的折扣近似 w_init ≈ r_weight/(1-γ)
        # 具体: vel=0.45/0.01=45, ang=0.02/0.01=2, tilt=0.02/0.01=2, int=0.005/0.01=0.5, sat=0.01/0.01=1
        self.psc_weights = self.param(
            "psc_weights",
            lambda rng, shape: jnp.array([45.0, 2.0, 2.0, 0.5, 1.0]),
            (PSC_NUM_BASIS,)
        )
        # PSC 偏置: V_phys = b + w·φ(s)
        # v19.0.1: 30→20。实际训练中 ~75% episode 提前终止，
        # 真实折扣回报 ≈ 0.28 × 80步(有效) ≈ 22，bias=20 更接近真实量级
        self.psc_bias = self.param(
            "psc_bias",
            lambda rng, shape: jnp.float32(20.0),
            ()
        )
        
        # V_res 网络（独立 Encoder + Dense → 1）
        self.critic_encoder = Encoder()
        self.critic_dense = nn.Dense(64, name="critic_dense")
        self.critic_head = nn.Dense(1, name="critic")

    def __call__(
        self,
        obs: Float[Array, "batch obs_dim"],
        lstm_state: LSTMState,
        done: Optional[Float[Array, "batch"]] = None
    ) -> Tuple[Float[Array, "batch action_dim"], Float[Array, "batch action_dim"],
               Float[Array, "batch"], LSTMState]:
        """
        前向传播（方案D：Actor/Critic 双路径分离）
        """
        # 重置 LSTM 状态
        if done is not None:
            reset_mask = done[:, None]
            h = lstm_state.h * (1.0 - reset_mask)
            c = lstm_state.c * (1.0 - reset_mask)
            lstm_state = LSTMState(h=h, c=c)

        # === Actor 路径（pg_loss 梯度在此路径内）===
        encoded = self.actor_encoder(obs)
        carry = (lstm_state.c, lstm_state.h)
        new_carry, lstm_out = self.lstm(carry, encoded)
        new_c, new_h = new_carry
        new_lstm_state = LSTMState(h=new_h, c=new_c)
        decoded = self.decoder(lstm_out)
        action_mean = self.actor_mean(decoded)
        # v18.1: log_std 现为可训练参数，裁剪到 [-5, 0] 防止 std 过大(探索爆炸)
        log_std_clipped = jnp.clip(self.log_std, -5.0, 0.0)
        action_std = jnp.exp(log_std_clipped) * jnp.ones_like(action_mean)

        # === Critic 路径 (PSC: V = V_phys + V_res) ===
        v_phys = compute_v_physics(obs, self.psc_weights, self.psc_bias)
        critic_encoded = self.critic_encoder(obs)
        critic_decoded = nn.tanh(self.critic_dense(critic_encoded))
        v_res = self.critic_head(critic_decoded).squeeze(-1)
        value = v_phys + v_res

        return action_mean, action_std, value, new_lstm_state

    def actor_forward(
        self,
        obs: Float[Array, "batch obs_dim"],
        lstm_state: LSTMState,
        done: Optional[Float[Array, "batch"]] = None
    ) -> Tuple[Float[Array, "batch action_dim"], Float[Array, "batch action_dim"], LSTMState]:
        """
        仅 Actor 路径前向传播（跳过 Critic，用于 PPO Update scan 加速）
        
        Critic 路径不依赖 LSTM 状态，无需在时序 scan 内计算。
        将其提取到 scan 外部可大幅减少 scan 内的计算量。
        
        Returns:
            action_mean: 动作均值 [batch, 4]
            action_std: 动作标准差 [batch, 4]
            new_lstm_state: 更新后的 LSTM 状态
        """
        # 重置 LSTM 状态
        if done is not None:
            reset_mask = done[:, None]
            h = lstm_state.h * (1.0 - reset_mask)
            c = lstm_state.c * (1.0 - reset_mask)
            lstm_state = LSTMState(h=h, c=c)

        # Actor 路径
        encoded = self.actor_encoder(obs)
        carry = (lstm_state.c, lstm_state.h)
        new_carry, lstm_out = self.lstm(carry, encoded)
        new_c, new_h = new_carry
        new_lstm_state = LSTMState(h=new_h, c=new_c)
        decoded = self.decoder(lstm_out)
        action_mean = self.actor_mean(decoded)
        log_std_clipped = jnp.clip(self.log_std, -5.0, 0.0)
        action_std = jnp.exp(log_std_clipped) * jnp.ones_like(action_mean)

        return action_mean, action_std, new_lstm_state

    def critic_forward(
        self,
        obs: Float[Array, "batch obs_dim"]
    ) -> Tuple[Float[Array, "batch"], Float[Array, "batch"], Float[Array, "batch"]]:
        """
        仅 Critic 路径前向传播 (PSC: V = V_phys + V_res)
        
        无 LSTM 依赖，可 batch 并行计算。
        返回三个分量用于 PDA (Physics-Decomposed Advantage)。
        
        Returns:
            value: V_total = V_phys + V_res [batch]
            v_phys: 解析物理项 [batch]
            v_res: 学习残差项 [batch]
        """
        v_phys = compute_v_physics(obs, self.psc_weights, self.psc_bias)
        critic_encoded = self.critic_encoder(obs)
        critic_decoded = nn.tanh(self.critic_dense(critic_encoded))
        v_res = self.critic_head(critic_decoded).squeeze(-1)
        value = v_phys + v_res
        return value, v_phys, v_res

    def sample_action(
        self,
        obs: Float[Array, "batch obs_dim"],
        lstm_state: LSTMState,
        done: Optional[Float[Array, "batch"]] = None
    ) -> Tuple[Float[Array, "batch action_dim"], Float[Array, "batch action_dim"], 
               Float[Array, "batch"], Float[Array, "batch"], LSTMState]:
        """
        采样动作 (重参数化)
        
        Returns:
            raw_action: Pre-Tanh 动作 (u)
            action: Post-Tanh 动作 (a)
            log_prob: 对数概率
            value: 状态价值
            new_lstm_state: 新的 LSTM 状态
        """
        action_mean, action_std, value, new_state = self(obs, lstm_state, done)
        
        # 获取 RNG
        rng = self.make_rng("noise")
        
        # 重参数化技巧: u = mean + std * noise
        noise = jax.random.normal(rng, action_mean.shape)
        raw_action = action_mean + action_std * noise
        
        # Tanh 变换: a = tanh(u)
        action = nn.tanh(raw_action)
        
        # 计算对数概率 (使用 raw_action)
        log_prob = compute_log_prob(raw_action, action_mean, action_std)
        
        return raw_action, action, log_prob, value, new_state
    
    def forward_mlp_only(
        self,
        obs: Float[Array, "batch obs_dim"]
    ) -> Tuple[Float[Array, "batch action_dim"], Float[Array, "batch"]]:
        """
        仅 MLP 前向传播 (用于 BC 预训练)
        
        跳过 LSTM，直接将 actor_encoder 输出传给 Decoder。
        返回确定性动作（使用均值，无采样）。
        """
        # Actor path: actor_encoder -> Decoder (跳过 LSTM)
        encoded = self.actor_encoder(obs)
        decoded = self.decoder(encoded)
        action_mean = self.actor_mean(decoded)
        action = nn.tanh(action_mean)

        # Critic path (PSC)
        v_phys = compute_v_physics(obs, self.psc_weights, self.psc_bias)
        critic_encoded = self.critic_encoder(obs)
        critic_decoded = nn.tanh(self.critic_dense(critic_encoded))
        v_res = self.critic_head(critic_decoded).squeeze(-1)
        value = v_phys + v_res

        return action, value


    def init_lstm_state(self, batch_size: int) -> LSTMState:
        """初始化 LSTM 隐状态"""
        return LSTMState(
            h=jnp.zeros((batch_size, self.lstm_dim)),
            c=jnp.zeros((batch_size, self.lstm_dim))
        )


def create_stochastic_actor_critic(
    rng: PRNGKeyArray,
    batch_size: int = 1
) -> Tuple[StochasticActorCritic, dict, LSTMState]:
    """创建随机策略 Actor-Critic"""
    model = StochasticActorCritic()
    
    dummy_obs = jnp.zeros((batch_size, OBS_DIM))
    init_state = model.init_lstm_state(batch_size)
    
    params = model.init(rng, dummy_obs, init_state)
    
    return model, params, init_state
