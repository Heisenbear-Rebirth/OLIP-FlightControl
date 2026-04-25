"""
AeroCat 消融实验 v2 —— 困难条件

修改：
1. 域随机化加大: 质量 ±60%, 推力 ±50%
2. 加入随机阵风扰动 (每步随机脉冲)
3. 初始扰动更大: 倾角 ±0.5rad, 角速度 ±3rad/s
"""

import os, sys, json, time
from pathlib import Path

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from flax.training.train_state import TrainState

ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from aerocat.envs.hover_env import (
    HoverEnvConfig, reset_hover_env, step_hover_env,
    build_hover_observation, generate_olip_features,
    randomize_params, euler_to_quaternion_simple,
    check_hover_termination, compute_hover_reward
)
from aerocat.core.state import (
    PhysState, L3State, PhysParams, L3Config, compute_mixer_matrix
)
from aerocat.control.l3_controller import (
    l3_substep, map_action_to_rates_and_thrust
)
from aerocat.physics.dynamics import physics_dynamics_step
from typing import NamedTuple


# =============================================================================
# 困难环境配置
# =============================================================================
class HardHoverConfig:
    """困难悬停配置"""
    dt_rl = 0.02
    dt_sub = 0.002
    num_substeps = 10
    max_episode_steps = 200
    obs_dim = 20
    action_dim = 4
    
    # 更大初始扰动
    init_max_tilt = 0.5       # ±28.6°
    init_max_rate = 3.0       # ±3 rad/s
    init_height = 10.0
    max_tilt_angle = 1.0
    
    # 大范围域随机化
    mass_range = (0.4, 1.6)         # ±60%
    thrust_coeff_range = (0.5, 1.5) # ±50%
    
    # 阵风扰动
    wind_gust_std = 2.0       # 阵风标准差 (m/s)
    wind_gust_prob = 0.3      # 每步触发概率


def hard_randomize_params(key, batch_size, config):
    """大范围域随机化"""
    keys = jax.random.split(key, 5)
    base_params = PhysParams.create_default(batch_size)
    
    mass = jax.random.uniform(keys[0], (batch_size,), 
                               minval=config.mass_range[0], maxval=config.mass_range[1])
    thrust_scale = jax.random.uniform(keys[1], (batch_size,),
                                       minval=config.thrust_coeff_range[0], 
                                       maxval=config.thrust_coeff_range[1])
    motor_max_thrust = base_params.motor_max_thrust * thrust_scale
    motor_loss = jax.random.uniform(keys[2], (batch_size, 4), minval=0.0, maxval=0.1)
    
    params = base_params.replace(mass=mass, motor_max_thrust=motor_max_thrust, motor_loss=motor_loss)
    mixer_matrix = compute_mixer_matrix(
        params.frame_angle, jnp.mean(params.force_to_torque_ratio, axis=-1))
    return params, mixer_matrix


def hard_reset(key, batch_size, config):
    """困难环境重置"""
    keys = jax.random.split(key, 6)
    
    phys_params, mixer_matrix = hard_randomize_params(keys[0], batch_size, config)
    
    roll = jax.random.uniform(keys[1], (batch_size,), minval=-config.init_max_tilt, maxval=config.init_max_tilt)
    pitch = jax.random.uniform(keys[2], (batch_size,), minval=-config.init_max_tilt, maxval=config.init_max_tilt)
    yaw = jax.random.uniform(keys[3], (batch_size,), minval=-jnp.pi, maxval=jnp.pi)
    quaternion = euler_to_quaternion_simple(roll, pitch, yaw)
    
    angular_velocity = jax.random.uniform(keys[4], (batch_size, 3),
                                           minval=-config.init_max_rate, maxval=config.init_max_rate)
    
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
    
    l3_state = L3State.create_default(batch_size)
    olip_features = generate_olip_features(phys_params, batch_size)
    prev_action = jnp.zeros((batch_size, 4))
    step_count = jnp.zeros((batch_size,), dtype=jnp.int32)
    obs = build_hover_observation(phys_state, l3_state, olip_features, config.init_height)
    
    return (phys_state, l3_state, phys_params, mixer_matrix,
            olip_features, prev_action, step_count, obs)


def hard_step(key, phys_state, l3_state, action, prev_action, step_count,
              phys_params, l3_config, mixer_matrix, olip_features, config):
    """带阵风扰动的环境步进"""
    keys = jax.random.split(key, 3)
    
    target_rates, thrust = map_action_to_rates_and_thrust(action)
    loop_keys = jax.random.split(keys[0], config.num_substeps)
    
    def inner_loop_body(carry, loop_key):
        phys_st, l3_st = carry
        phys_st, l3_st, motor_cmds = l3_substep(
            phys_st, l3_st, target_rates, thrust,
            phys_params, l3_config, mixer_matrix, config.dt_sub)
        phys_st = physics_dynamics_step(phys_st, motor_cmds, phys_params, loop_key, config.dt_sub)
        return (phys_st, l3_st), None
    
    (phys_state_new, l3_state_new), _ = jax.lax.scan(
        inner_loop_body, (phys_state, l3_state), xs=loop_keys, length=config.num_substeps)
    
    # 阵风扰动: 随机角速度脉冲
    gust_mask = jax.random.uniform(keys[1], (action.shape[0],)) < config.wind_gust_prob
    gust_impulse = jax.random.normal(keys[2], (action.shape[0], 3)) * config.wind_gust_std
    gust_impulse = gust_impulse * gust_mask[:, None]
    
    new_angular_vel = phys_state_new.angular_velocity + gust_impulse * config.dt_rl
    phys_state_new = phys_state_new.replace(angular_velocity=new_angular_vel)
    
    new_step_count = step_count + 1
    obs = build_hover_observation(phys_state_new, l3_state_new, olip_features, config.init_height)
    reward, reward_info = compute_hover_reward(phys_state_new, action, prev_action, config.init_height)
    done = check_hover_termination(phys_state_new, new_step_count, 
                                    type('C', (), {'max_tilt_angle': config.max_tilt_angle, 
                                                   'max_episode_steps': config.max_episode_steps})())
    
    quat = phys_state_new.quaternion
    tilt = 2.0 * jnp.sqrt(quat[..., 1]**2 + quat[..., 2]**2 + 1e-8)
    crash_penalty = jnp.where(tilt > config.max_tilt_angle, -5.0,
                               jnp.where(phys_state_new.position[..., 2] > -0.3, -5.0, 0.0))
    reward = reward + crash_penalty * done.astype(jnp.float32)
    
    info = {**reward_info, 'step_count': new_step_count, 'tilt': tilt}
    return phys_state_new, l3_state_new, obs, reward, done, info


# =============================================================================
# 网络 + PPO (同 run_ablation.py)
# =============================================================================
class ActorCritic(nn.Module):
    action_dim: int = 4
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        action_mean = nn.Dense(self.action_dim)(x)
        action_mean = nn.tanh(action_mean)
        log_std = self.param('log_std', nn.initializers.constant(-1.0), (self.action_dim,))
        v = nn.Dense(128)(x)
        v = nn.relu(v)
        value = nn.Dense(1)(v)
        value = jnp.squeeze(value, axis=-1)
        return action_mean, log_std, value


class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    value: jnp.ndarray
    log_prob: jnp.ndarray


def sample_action(params, apply_fn, obs, key):
    action_mean, log_std, value = apply_fn(params, obs)
    std = jnp.exp(log_std)
    noise = jax.random.normal(key, action_mean.shape)
    action = jnp.clip(action_mean + noise * std, -1.0, 1.0)
    log_prob = -0.5 * jnp.sum(
        ((action - action_mean) / (std + 1e-8)) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1)
    return action, log_prob, value


def compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95):
    T = rewards.shape[0]
    gae = jnp.zeros(rewards.shape[1])
    def scan_fn(gae, t):
        idx = T - 1 - t
        delta = rewards[idx] + gamma * values[idx + 1] * (1 - dones[idx]) - values[idx]
        gae = delta + gamma * gae_lambda * (1 - dones[idx]) * gae
        return gae, gae
    _, adv_rev = jax.lax.scan(scan_fn, gae, jnp.arange(T))
    advantages = jnp.flip(adv_rev, axis=0)
    return advantages, advantages + values[:-1]


def ppo_loss(params, apply_fn, batch, clip_eps=0.2):
    obs, actions, old_lp, advantages, returns = batch
    action_mean, log_std, values = apply_fn(params, obs)
    std = jnp.exp(log_std)
    lp = -0.5 * jnp.sum(((actions - action_mean) / (std + 1e-8)) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1)
    ratio = jnp.exp(lp - old_lp)
    an = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)
    pg = -jnp.mean(jnp.minimum(ratio * an, jnp.clip(ratio, 1 - clip_eps, 1 + clip_eps) * an))
    vf = jnp.mean((values - returns) ** 2)
    ent = jnp.mean(0.5 * jnp.sum(1 + 2 * log_std + jnp.log(2 * jnp.pi), axis=-1))
    return pg + 0.5 * vf - 0.01 * ent, {'pg_loss': pg, 'vf_loss': vf, 'entropy': ent}


def apply_obs_mask(obs, mode="full"):
    if mode == "full": return obs
    elif mode == "no_olip": return obs.at[..., 15:20].set(0.0)
    elif mode == "no_l3diag": return obs.at[..., 11:15].set(0.0)
    elif mode == "baseline": return obs.at[..., 11:20].set(0.0)  # 仅本体感知
    else: raise ValueError(f"Unknown mode: {mode}")


# =============================================================================
# 训练
# =============================================================================
def run_hard_experiment(mode, num_envs=512, total_steps=50_000_000, seed=42, save_dir=None):
    config = HardHoverConfig()
    l3_config = L3Config()
    rollout_length = 64
    num_epochs = 4
    batch_size = num_envs * rollout_length
    num_updates = total_steps // batch_size
    
    print(f"\n{'='*50}")
    print(f"  困难消融: {mode.upper()} (mass ±60%, wind gust)")
    print(f"  总步数: {total_steps:,}  |  更新: {num_updates}")
    print(f"{'='*50}")
    
    rng = jax.random.PRNGKey(seed)
    network = ActorCritic(action_dim=4)
    rng, ik = jax.random.split(rng)
    params = network.init(ik, jnp.zeros((1, config.obs_dim)))
    tx = optax.adam(3e-4)
    train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)
    
    rng, rk = jax.random.split(rng)
    phys_state, l3_state, phys_params, mixer_matrix, olip_features, prev_action, step_count, obs = \
        hard_reset(rk, num_envs, config)
    obs = apply_obs_mask(obs, mode)
    
    @jax.jit
    def collect_and_update(rng, train_state, phys_state, l3_state, prev_action, step_count, obs,
                           pp, mx, ol):
        def rollout_step(carry, _):
            rng, ps, ls, pa, sc, ob, cpp, cmx, col = carry
            rng, ak, sk = jax.random.split(rng, 3)
            
            action, log_prob, value = sample_action(train_state.params, train_state.apply_fn, ob, ak)
            
            ps_new, ls_new, obs_new, reward, done, info = hard_step(
                sk, ps, ls, action, pa, sc, cpp, l3_config, cmx, col, config)
            obs_new = apply_obs_mask(obs_new, mode)
            
            rng, rkk = jax.random.split(rng)
            r_out = hard_reset(rkk, num_envs, config)
            r_ps, r_ls, r_pp, r_mx, r_ol, r_pa, r_sc, r_obs = r_out
            r_obs = apply_obs_mask(r_obs, mode)
            
            def sel(new, reset):
                if hasattr(new, 'shape') and new.ndim > 0:
                    return jnp.where(done.reshape(done.shape + (1,)*(new.ndim-1)), reset, new)
                return jnp.where(done, reset, new)
            
            pf = jax.tree_util.tree_map(sel, ps_new, r_ps)
            lf = jax.tree_util.tree_map(sel, ls_new, r_ls)
            ppf = jax.tree_util.tree_map(sel, cpp, r_pp)
            mxf = jax.tree_util.tree_map(sel, cmx, r_mx)
            olf = jnp.where(done[:, None], r_ol, col)
            of = jnp.where(done[:, None], r_obs, obs_new)
            paf = jnp.where(done[:, None], r_pa, action)
            scf = jnp.where(done, r_sc, sc + 1)
            
            trans = Transition(obs=ob, action=action, reward=reward, done=done, value=value, log_prob=log_prob)
            return (rng, pf, lf, paf, scf, of, ppf, mxf, olf), trans
        
        carry0 = (rng, phys_state, l3_state, prev_action, step_count, obs, pp, mx, ol)
        final_carry, rollout = jax.lax.scan(rollout_step, carry0, None, length=rollout_length)
        _, _, _, _, _, last_obs, _, _, _ = final_carry
        _, _, last_value = train_state.apply_fn(train_state.params, last_obs)
        
        vals = jnp.concatenate([rollout.value, last_value[None, :]], axis=0)
        advantages, returns = compute_gae(rollout.reward, vals, rollout.done)
        
        b_obs = rollout.obs.reshape(-1, config.obs_dim)
        b_act = rollout.action.reshape(-1, 4)
        b_lp = rollout.log_prob.reshape(-1)
        b_adv = advantages.reshape(-1)
        b_ret = returns.reshape(-1)
        
        def epoch_step(ts, _):
            batch = (b_obs, b_act, b_lp, b_adv, b_ret)
            (loss, info), grads = jax.value_and_grad(ppo_loss, has_aux=True)(ts.params, ts.apply_fn, batch)
            ts = ts.apply_gradients(grads=grads)
            return ts, info
        
        train_state_new, _ = jax.lax.scan(epoch_step, train_state, None, length=num_epochs)
        mean_reward = jnp.mean(rollout.reward)
        return final_carry, train_state_new, mean_reward
    
    metrics = []
    t0 = time.time()
    
    for update in range(num_updates):
        rng, rk = jax.random.split(rng)
        carry_out, train_state, mean_reward = collect_and_update(
            rk, train_state, phys_state, l3_state, prev_action, step_count, obs,
            phys_params, mixer_matrix, olip_features)
        rng_new, phys_state, l3_state, prev_action, step_count, obs, \
            phys_params, mixer_matrix, olip_features = carry_out
        rng = rng_new
        
        total_steps_done = (update + 1) * batch_size
        if update % 50 == 0:
            r = float(mean_reward)
            elapsed = time.time() - t0
            fps = total_steps_done / elapsed if elapsed > 0 else 0
            print(f"  [{mode:12s}] Update {update:5d}/{num_updates} | "
                  f"Steps: {total_steps_done:>10,} | Reward: {r:+.4f} | FPS: {fps:,.0f}")
            metrics.append({'update': update, 'total_steps': total_steps_done,
                            'mean_reward': r, 'elapsed': elapsed})
    
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f'{mode}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    import pickle
    with open(os.path.join(save_dir, f'{mode}_params.pkl'), 'wb') as f:
        pickle.dump(jax.device_get(train_state.params), f)
    
    final_r = float(mean_reward)
    print(f"  [{mode:12s}] 完成! 最终奖励: {final_r:.4f}, 耗时: {time.time()-t0:.1f}s")
    return metrics


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="AeroCat 困难消融实验")
    parser.add_argument("--steps", type=int, default=50_000_000)
    parser.add_argument("--envs", type=int, default=512)
    parser.add_argument("--save-dir", type=str, default=None)
    args = parser.parse_args()
    
    if args.save_dir is None:
        args.save_dir = str(ROOT_DIR / "logs" / "ablation_hard")
    
    print("=" * 60)
    print("  AeroCat 困难消融实验 (4 组)")
    print("  质量 ±60%, 推力 ±50%, 阵风扰动")
    print("=" * 60)
    
    for mode in ['full', 'no_olip', 'no_l3diag', 'baseline']:
        run_hard_experiment(mode, args.envs, args.steps, seed=42, save_dir=args.save_dir)
    
    print("\n" + "=" * 60)
    print("  所有实验完成!")
    print("=" * 60)
