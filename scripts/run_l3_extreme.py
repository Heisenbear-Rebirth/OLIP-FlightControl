"""
AeroCat L3 诊断验证实验 —— 极端条件

设计思路:
1. 重载飞行器 (质量 1.5-2.5kg)，电机接近饱和
2. 持续偏置力矩 (模拟重心偏移/侧风)，迫使 PID 积分器持续蓄积
3. Episode 中段力矩突变，策略必须实时感知 PID 状态变化
4. 更严格的坠毁惩罚 (积分器饱和 → 失控 → 大坠毁惩罚)

对比: Full vs No-L3Diag vs Baseline
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
    build_hover_observation, generate_olip_features,
    euler_to_quaternion_simple, compute_hover_reward
)
from aerocat.core.state import (
    PhysState, L3State, PhysParams, L3Config, compute_mixer_matrix
)
from aerocat.control.l3_controller import l3_substep, map_action_to_rates_and_thrust
from aerocat.physics.dynamics import physics_dynamics_step
from typing import NamedTuple


# =============================================================================
# 网络
# =============================================================================
class ActorCritic(nn.Module):
    action_dim: int = 4
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x); x = nn.relu(x)
        x = nn.Dense(128)(x); x = nn.relu(x)
        am = nn.Dense(self.action_dim)(x); am = nn.tanh(am)
        ls = self.param('log_std', nn.initializers.constant(-1.0), (self.action_dim,))
        v = nn.Dense(128)(x); v = nn.relu(v)
        val = nn.Dense(1)(v); val = jnp.squeeze(val, axis=-1)
        return am, ls, val

class Transition(NamedTuple):
    obs: jnp.ndarray; action: jnp.ndarray; reward: jnp.ndarray
    done: jnp.ndarray; value: jnp.ndarray; log_prob: jnp.ndarray


def sample_action(params, apply_fn, obs, key):
    am, ls, val = apply_fn(params, obs)
    std = jnp.exp(ls)
    noise = jax.random.normal(key, am.shape)
    action = jnp.clip(am + noise * std, -1.0, 1.0)
    lp = -0.5 * jnp.sum(((action - am)/(std+1e-8))**2 + 2*ls + jnp.log(2*jnp.pi), axis=-1)
    return action, lp, val

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    T = rewards.shape[0]; gae = jnp.zeros(rewards.shape[1])
    def sf(gae, t):
        i = T-1-t
        d = rewards[i] + gamma*values[i+1]*(1-dones[i]) - values[i]
        gae = d + gamma*lam*(1-dones[i])*gae
        return gae, gae
    _, ar = jax.lax.scan(sf, gae, jnp.arange(T))
    return jnp.flip(ar, axis=0), jnp.flip(ar, axis=0) + values[:-1]

def ppo_loss(params, apply_fn, batch, eps=0.2):
    obs, act, olp, adv, ret = batch
    am, ls, vals = apply_fn(params, obs)
    std = jnp.exp(ls)
    lp = -0.5*jnp.sum(((act-am)/(std+1e-8))**2 + 2*ls + jnp.log(2*jnp.pi), axis=-1)
    ratio = jnp.exp(lp - olp)
    an = (adv - jnp.mean(adv))/(jnp.std(adv)+1e-8)
    pg = -jnp.mean(jnp.minimum(ratio*an, jnp.clip(ratio,1-eps,1+eps)*an))
    vf = jnp.mean((vals-ret)**2)
    ent = jnp.mean(0.5*jnp.sum(1+2*ls+jnp.log(2*jnp.pi), axis=-1))
    return pg + 0.5*vf - 0.01*ent, {'pg': pg, 'vf': vf, 'ent': ent}

def apply_mask(obs, mode):
    if mode == "full": return obs
    elif mode == "no_l3diag": return obs.at[..., 11:15].set(0.0)
    elif mode == "baseline": return obs.at[..., 11:20].set(0.0)
    else: raise ValueError(mode)


# =============================================================================
# 极端 L3 环境
# =============================================================================
MAX_STEPS = 300  # 更长 episode (6秒)，给力矩突变留时间

def extreme_reset(key, batch_size):
    """
    极端条件: 重载 + 持续偏置力矩 + 中段突变
    返回额外状态: bias_torque, bias_torque_phase2, switch_step
    """
    keys = jax.random.split(key, 10)
    base = PhysParams.create_default(batch_size)
    
    # 重载: 质量 1.5-2.5x 标称 (标称1.0kg)
    mass = jax.random.uniform(keys[0], (batch_size,), minval=1.5, maxval=2.5)
    
    # 弱动力: 推力系数 0.7-1.0 (削弱推力)
    ts = jax.random.uniform(keys[1], (batch_size,), minval=0.7, maxval=1.0)
    mmt = base.motor_max_thrust * ts
    
    # 电机退化: 1个电机效率降到 60-80%
    motor_loss = jnp.zeros((batch_size, 4))
    degrade_idx = jax.random.randint(keys[8], (batch_size,), 0, 4)  # 哪个电机
    degrade_amt = jax.random.uniform(keys[9], (batch_size,), minval=0.2, maxval=0.4)
    for i in range(4):
        motor_loss = motor_loss.at[:, i].set(
            jnp.where(degrade_idx == i, degrade_amt, 0.0))
    
    params = base.replace(mass=mass, motor_max_thrust=mmt, motor_loss=motor_loss)
    mx = compute_mixer_matrix(params.frame_angle, jnp.mean(params.force_to_torque_ratio, axis=-1))
    
    # 小初始倾角 (留给偏置力矩发挥空间)
    roll = jax.random.uniform(keys[2], (batch_size,), minval=-0.1, maxval=0.1)
    pitch = jax.random.uniform(keys[3], (batch_size,), minval=-0.1, maxval=0.1)
    yaw = jax.random.uniform(keys[4], (batch_size,), minval=-jnp.pi, maxval=jnp.pi)
    quat = euler_to_quaternion_simple(roll, pitch, yaw)
    
    hover_th = (params.mass * 9.81) / (4.0 * params.motor_max_thrust)
    hover_th = jnp.clip(hover_th, 0.1, 0.9)
    init_mt = jnp.repeat(jnp.sqrt(hover_th[:, None]), 4, axis=1)
    
    ps = PhysState(
        position=jnp.zeros((batch_size, 3)).at[:, 2].set(-10.0),
        velocity=jnp.zeros((batch_size, 3)),
        quaternion=quat,
        angular_velocity=jnp.zeros((batch_size, 3)),
        motor_throttle=init_mt,
        battery_soc=jnp.ones((batch_size,)),
        battery_voltage=jnp.full((batch_size,), 16.8),
        battery_V1=jnp.zeros((batch_size,)),
        battery_V2=jnp.zeros((batch_size,)),
        imu_gyro=jnp.zeros((batch_size, 3)),
        imu_accel=jnp.tile(jnp.array([0.0, 0.0, 9.81]), (batch_size, 1)),
        wind_velocity=jnp.zeros((batch_size, 3)),
        turbulence_state=jnp.zeros((batch_size, 3)),
    )
    
    ls = L3State.create_default(batch_size)
    olip = generate_olip_features(params, batch_size)
    pa = jnp.zeros((batch_size, 4))
    sc = jnp.zeros((batch_size,), dtype=jnp.int32)
    obs = build_hover_observation(ps, ls, olip, 10.0)
    
    # 持续偏置力矩 (Phase 1: 0-150步, Phase 2: 150-300步)
    # 幅度大: 0.3-0.8 Nm 在 roll/pitch 上
    bias_torque_1 = jax.random.uniform(keys[5], (batch_size, 3), minval=-0.8, maxval=0.8)
    bias_torque_1 = bias_torque_1.at[:, 2].set(0.0)  # yaw 不加偏置
    
    # Phase 2: 突变 — 方向反转 + 幅度变化
    bias_torque_2 = jax.random.uniform(keys[6], (batch_size, 3), minval=-0.8, maxval=0.8)
    bias_torque_2 = bias_torque_2.at[:, 2].set(0.0)
    
    # 突变时刻: 100-200步之间随机
    switch_step = jax.random.randint(keys[7], (batch_size,), 100, 200)
    
    return (ps, ls, params, mx, olip, pa, sc, obs,
            bias_torque_1, bias_torque_2, switch_step)


def extreme_step(key, ps, ls, action, pa, sc, pp, mx, ol,
                  bias_torque_1, bias_torque_2, switch_step):
    """
    带持续偏置力矩的环境步进
    关键: 偏置力矩直接注入角速度导数，迫使 PID 积分器蓄积
    """
    l3_config = L3Config()
    tr, th = map_action_to_rates_and_thrust(action)
    loop_keys = jax.random.split(key, 10)
    
    # 选择当前阶段的偏置力矩
    current_bias = jnp.where(
        (sc < switch_step)[:, None], bias_torque_1, bias_torque_2)
    
    def inner(carry, lk):
        ps_c, ls_c = carry
        ps_c, ls_c, mc = l3_substep(ps_c, ls_c, tr, th, pp, l3_config, mx, 0.002)
        ps_c = physics_dynamics_step(ps_c, mc, pp, lk, 0.002)
        
        # 注入持续偏置力矩: 直接叠加到角速度
        # bias_torque / inertia ≈ bias_torque / 0.01 * dt = 大角加速度
        # 使用较温和的注入: bias * dt_sub
        new_omega = ps_c.angular_velocity + current_bias * 0.002
        ps_c = ps_c.replace(angular_velocity=new_omega)
        
        return (ps_c, ls_c), None
    
    (ps_new, ls_new), _ = jax.lax.scan(inner, (ps, ls), xs=loop_keys, length=10)
    
    nsc = sc + 1
    obs = build_hover_observation(ps_new, ls_new, ol, 10.0)
    reward, ri = compute_hover_reward(ps_new, action, pa, 10.0)
    
    # 终止条件
    q = ps_new.quaternion
    tilt = 2.0 * jnp.sqrt(q[..., 1]**2 + q[..., 2]**2 + 1e-8)
    done = (tilt > 1.0) | (ps_new.position[..., 2] > -0.3) | (nsc >= MAX_STEPS)
    
    # 坠毁惩罚 (L3 诊断的价值: 避免这些坠毁)
    crash = jnp.where(tilt > 1.0, -10.0,
                       jnp.where(ps_new.position[..., 2] > -0.3, -10.0, 0.0))
    reward = reward + crash * done.astype(jnp.float32)
    
    # 额外奖励: PID 积分器未饱和 → 正向激励
    integral_mag = jnp.sqrt(jnp.sum(ls_new.pid_integral**2, axis=-1))
    integral_penalty = -0.05 * jnp.clip(integral_mag - 0.2, 0.0, 1.0)
    reward = reward + integral_penalty
    
    return ps_new, ls_new, obs, reward, done, nsc


# =============================================================================
# 训练
# =============================================================================
def train_extreme(mode, num_envs=512, total_steps=80_000_000, seed=42, save_dir="."):
    rollout_len = 64; batch_sz = num_envs * rollout_len
    num_updates = total_steps // batch_sz
    
    print(f"\n{'='*55}")
    print(f"  L3 极端验证: {mode.upper()}")
    print(f"  重载(1.5-2.5kg) + 偏置力矩(±0.8Nm) + 中段突变")
    print(f"  总步数: {total_steps:,}  |  更新: {num_updates}")
    print(f"{'='*55}")
    
    rng = jax.random.PRNGKey(seed)
    net = ActorCritic(action_dim=4)
    rng, ik = jax.random.split(rng)
    params = net.init(ik, jnp.zeros((1, 20)))
    ts = TrainState.create(apply_fn=net.apply, params=params, tx=optax.adam(3e-4))
    
    rng, rk = jax.random.split(rng)
    reset_out = extreme_reset(rk, num_envs)
    ps, ls, pp, mx, ol, pa, sc, obs, bt1, bt2, ss = reset_out
    obs = apply_mask(obs, mode)
    
    @jax.jit
    def collect_update(rng, ts, ps, ls, pa, sc, obs, pp, mx, ol, bt1, bt2, ss):
        def rstep(carry, _):
            rng, ps, ls, pa, sc, ob, pp, mx, ol, bt1, bt2, ss = carry
            rng, ak, sk, rkk = jax.random.split(rng, 4)
            
            a, lp, v = sample_action(ts.params, ts.apply_fn, ob, ak)
            psn, lsn, obsn, r, d, nsc = extreme_step(
                sk, ps, ls, a, pa, sc, pp, mx, ol, bt1, bt2, ss)
            obsn = apply_mask(obsn, mode)
            
            # Auto-reset
            ro = extreme_reset(rkk, num_envs)
            rps, rls, rpp, rmx, rol, rpa, rsc, robs, rbt1, rbt2, rss = ro
            robs = apply_mask(robs, mode)
            
            def sel(n, r):
                if hasattr(n,'shape') and n.ndim>0:
                    return jnp.where(d.reshape(d.shape+(1,)*(n.ndim-1)), r, n)
                return jnp.where(d, r, n)
            
            pf = jax.tree_util.tree_map(sel, psn, rps)
            lf = jax.tree_util.tree_map(sel, lsn, rls)
            ppf = jax.tree_util.tree_map(sel, pp, rpp)
            mxf = jax.tree_util.tree_map(sel, mx, rmx)
            olf = jnp.where(d[:,None], rol, ol)
            of = jnp.where(d[:,None], robs, obsn)
            paf = jnp.where(d[:,None], rpa, a)
            scf = jnp.where(d, rsc, nsc)
            bt1f = jnp.where(d[:,None], rbt1, bt1)
            bt2f = jnp.where(d[:,None], rbt2, bt2)
            ssf = jnp.where(d, rss, ss)
            
            t = Transition(obs=ob, action=a, reward=r, done=d, value=v, log_prob=lp)
            return (rng, pf, lf, paf, scf, of, ppf, mxf, olf, bt1f, bt2f, ssf), t
        
        c0 = (rng, ps, ls, pa, sc, obs, pp, mx, ol, bt1, bt2, ss)
        fc, roll = jax.lax.scan(rstep, c0, None, length=rollout_len)
        rng_n, _, _, _, _, lo, _, _, _, _, _, _ = fc
        _, _, lv = ts.apply_fn(ts.params, lo)
        vals = jnp.concatenate([roll.value, lv[None,:]], axis=0)
        adv, ret = compute_gae(roll.reward, vals, roll.done)
        bo = roll.obs.reshape(-1,20); ba = roll.action.reshape(-1,4)
        blp = roll.log_prob.reshape(-1); badv = adv.reshape(-1); bret = ret.reshape(-1)
        
        def es(ts, _):
            (l,i),g = jax.value_and_grad(ppo_loss, has_aux=True)(
                ts.params, ts.apply_fn, (bo,ba,blp,badv,bret))
            return ts.apply_gradients(grads=g), i
        tsn, _ = jax.lax.scan(es, ts, None, length=4)
        return fc, tsn, jnp.mean(roll.reward)
    
    metrics = []; t0 = time.time()
    for u in range(num_updates):
        rng, rk = jax.random.split(rng)
        co, ts, mr = collect_update(
            rk, ts, ps, ls, pa, sc, obs, pp, mx, ol, bt1, bt2, ss)
        rng_n, ps, ls, pa, sc, obs, pp, mx, ol, bt1, bt2, ss = co
        rng = rng_n
        tsd = (u+1)*batch_sz
        if u % 50 == 0:
            r = float(mr); el = time.time()-t0
            fps = tsd/el if el > 0 else 0
            print(f"  [{mode:12s}] {u:5d}/{num_updates} | {tsd:>10,} | R: {r:+.4f} | FPS: {fps:,.0f}")
            metrics.append({'update': u, 'total_steps': tsd, 'mean_reward': r, 'elapsed': el})
    
    final_r = float(mr)
    print(f"  [{mode:12s}] Done! Final: {final_r:.4f}, Time: {time.time()-t0:.1f}s")
    
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, f'l3_extreme_{mode}_metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics, final_r


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=80_000_000)
    p.add_argument("--envs", type=int, default=512)
    p.add_argument("--save-dir", default=None)
    args = p.parse_args()
    
    save_dir = args.save_dir or str(ROOT_DIR / "logs" / "l3_extreme")
    
    print("="*60)
    print("  AeroCat L3 诊断极端验证实验")
    print("  重载 + 偏置力矩 + 中段突变 + 电机退化")
    print("="*60)
    
    results = {}
    for mode in ['full', 'no_l3diag', 'baseline']:
        m, f = train_extreme(mode, args.envs, args.steps, seed=42, save_dir=save_dir)
        results[mode] = f
    
    print(f"\n{'='*60}")
    print("  L3 极端验证结果")
    print("="*60)
    for mode in ['full', 'no_l3diag', 'baseline']:
        r = results[mode]
        delta = results['full'] - r
        print(f"  {mode:12s}: {r:.4f}  (vs Full: {delta:+.4f})")
    
    print(f"\n  L3 诊断贡献: {results['full'] - results['no_l3diag']:+.4f} "
          f"({(results['full'] - results['no_l3diag'])/results['full']*100:+.1f}%)")
    print(f"  完整方案 vs Baseline: {results['full'] - results['baseline']:+.4f} "
          f"({(results['full'] - results['baseline'])/results['full']*100:+.1f}%)")
