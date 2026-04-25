"""
AeroCat 扩展消融实验

实验A: OLIP 缩放实验 (4 个 DR 级别 × Full/No-OLIP)
  证明: 域随机化范围越大 → OLIP 贡献越大

实验B: 电机退化实验 (Full/No-L3Diag/Baseline)  
  证明: 在执行器受损场景下 L3 诊断提供安全优势
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
    HoverEnvConfig, build_hover_observation, generate_olip_features,
    euler_to_quaternion_simple, check_hover_termination, compute_hover_reward
)
from aerocat.core.state import (
    PhysState, L3State, PhysParams, L3Config, compute_mixer_matrix
)
from aerocat.control.l3_controller import l3_substep, map_action_to_rates_and_thrust
from aerocat.physics.dynamics import physics_dynamics_step
from typing import NamedTuple


# =============================================================================
# 网络 + PPO (同之前)
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
    lp = -0.5 * jnp.sum(((action - am) / (std + 1e-8))**2 + 2*ls + jnp.log(2*jnp.pi), axis=-1)
    return action, lp, val

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    T = rewards.shape[0]; gae = jnp.zeros(rewards.shape[1])
    def sf(gae, t):
        i = T-1-t
        d = rewards[i] + gamma*values[i+1]*(1-dones[i]) - values[i]
        gae = d + gamma*lam*(1-dones[i])*gae
        return gae, gae
    _, ar = jax.lax.scan(sf, gae, jnp.arange(T))
    adv = jnp.flip(ar, axis=0)
    return adv, adv + values[:-1]

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
    elif mode == "no_olip": return obs.at[..., 15:20].set(0.0)
    elif mode == "no_l3diag": return obs.at[..., 11:15].set(0.0)
    elif mode == "baseline": return obs.at[..., 11:20].set(0.0)
    else: raise ValueError(mode)


# =============================================================================
# 参数化环境 (支持电机退化)
# =============================================================================
def make_reset(mass_range, thrust_range, motor_degrade=False):
    """生成 reset 函数"""
    def reset_fn(key, batch_size):
        keys = jax.random.split(key, 8)
        base = PhysParams.create_default(batch_size)
        
        mass = jax.random.uniform(keys[0], (batch_size,), minval=mass_range[0], maxval=mass_range[1])
        ts = jax.random.uniform(keys[1], (batch_size,), minval=thrust_range[0], maxval=thrust_range[1])
        mmt = base.motor_max_thrust * ts
        
        # 电机退化: 随机 1-2 个电机效率降到 50-80%
        if motor_degrade:
            degradation = jnp.ones((batch_size, 4))
            # 每个环境随机选1-2个电机退化
            degrade_mask = jax.random.uniform(keys[6], (batch_size, 4)) < 0.35  # ~1.4个电机
            degrade_factor = jax.random.uniform(keys[7], (batch_size, 4), minval=0.5, maxval=0.8)
            degradation = jnp.where(degrade_mask, degrade_factor, degradation)
            motor_loss = 1.0 - degradation  # loss = 1 - efficiency
        else:
            motor_loss = jax.random.uniform(keys[2], (batch_size, 4), minval=0.0, maxval=0.05)
        
        params = base.replace(mass=mass, motor_max_thrust=mmt, motor_loss=motor_loss)
        mx = compute_mixer_matrix(params.frame_angle, jnp.mean(params.force_to_torque_ratio, axis=-1))
        
        roll = jax.random.uniform(keys[3], (batch_size,), minval=-0.3, maxval=0.3)
        pitch = jax.random.uniform(keys[4], (batch_size,), minval=-0.3, maxval=0.3)
        yaw = jax.random.uniform(keys[5], (batch_size,), minval=-jnp.pi, maxval=jnp.pi)
        quat = euler_to_quaternion_simple(roll, pitch, yaw)
        
        hover_th = (params.mass * 9.81) / (4.0 * params.motor_max_thrust)
        hover_th = jnp.clip(hover_th, 0.1, 0.9)
        init_mt = jnp.repeat(jnp.sqrt(hover_th[:, None]), 4, axis=1)
        
        ps = PhysState(
            position=jnp.zeros((batch_size, 3)).at[:, 2].set(-10.0),
            velocity=jnp.zeros((batch_size, 3)),
            quaternion=quat,
            angular_velocity=jax.random.uniform(keys[2], (batch_size, 3), minval=-1.0, maxval=1.0),
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
        return ps, ls, params, mx, olip, pa, sc, obs
    return reset_fn

def make_step():
    """生成 step 函数"""
    l3_config = L3Config()
    def step_fn(key, ps, ls, action, pa, sc, pp, mx, ol):
        tr, th = map_action_to_rates_and_thrust(action)
        loop_keys = jax.random.split(key, 10)
        def inner(carry, lk):
            ps, ls = carry
            ps, ls, mc = l3_substep(ps, ls, tr, th, pp, l3_config, mx, 0.002)
            ps = physics_dynamics_step(ps, mc, pp, lk, 0.002)
            return (ps, ls), None
        (ps_new, ls_new), _ = jax.lax.scan(inner, (ps, ls), xs=loop_keys, length=10)
        nsc = sc + 1
        obs = build_hover_observation(ps_new, ls_new, ol, 10.0)
        reward, ri = compute_hover_reward(ps_new, action, pa, 10.0)
        done = check_hover_termination(ps_new, nsc, type('C',(),{'max_tilt_angle':1.0,'max_episode_steps':200})())
        q = ps_new.quaternion
        tilt = 2.0*jnp.sqrt(q[...,1]**2+q[...,2]**2+1e-8)
        crash = jnp.where(tilt > 1.0, -5.0, jnp.where(ps_new.position[...,2] > -0.3, -5.0, 0.0))
        reward = reward + crash * done.astype(jnp.float32)
        return ps_new, ls_new, obs, reward, done
    return step_fn


# =============================================================================
# 通用训练器
# =============================================================================
def train_one(mode, reset_fn, step_fn, num_envs=512, total_steps=50_000_000, seed=42, label=""):
    rollout_len = 64
    batch_sz = num_envs * rollout_len
    num_updates = total_steps // batch_sz
    
    rng = jax.random.PRNGKey(seed)
    net = ActorCritic(action_dim=4)
    rng, ik = jax.random.split(rng)
    params = net.init(ik, jnp.zeros((1, 20)))
    ts = TrainState.create(apply_fn=net.apply, params=params, tx=optax.adam(3e-4))
    
    rng, rk = jax.random.split(rng)
    ps, ls, pp, mx, ol, pa, sc, obs = reset_fn(rk, num_envs)
    obs = apply_mask(obs, mode)
    
    @jax.jit
    def collect_update(rng, ts, ps, ls, pa, sc, obs, pp, mx, ol):
        def rstep(carry, _):
            rng, ps, ls, pa, sc, ob, cpp, cmx, col = carry
            rng, ak, sk = jax.random.split(rng, 3)
            a, lp, v = sample_action(ts.params, ts.apply_fn, ob, ak)
            psn, lsn, obsn, r, d = step_fn(sk, ps, ls, a, pa, sc, cpp, cmx, col)
            obsn = apply_mask(obsn, mode)
            rng, rkk = jax.random.split(rng)
            ro = reset_fn(rkk, num_envs)
            rps, rls, rpp, rmx, rol, rpa, rsc, robs = ro
            robs = apply_mask(robs, mode)
            def sel(n, r):
                if hasattr(n,'shape') and n.ndim>0:
                    return jnp.where(d.reshape(d.shape+(1,)*(n.ndim-1)), r, n)
                return jnp.where(d, r, n)
            pf = jax.tree_util.tree_map(sel, psn, rps)
            lf = jax.tree_util.tree_map(sel, lsn, rls)
            ppf = jax.tree_util.tree_map(sel, cpp, rpp)
            mxf = jax.tree_util.tree_map(sel, cmx, rmx)
            olf = jnp.where(d[:,None], rol, col)
            of = jnp.where(d[:,None], robs, obsn)
            paf = jnp.where(d[:,None], rpa, a)
            scf = jnp.where(d, rsc, sc+1)
            t = Transition(obs=ob, action=a, reward=r, done=d, value=v, log_prob=lp)
            return (rng, pf, lf, paf, scf, of, ppf, mxf, olf), t
        
        c0 = (rng, ps, ls, pa, sc, obs, pp, mx, ol)
        fc, roll = jax.lax.scan(rstep, c0, None, length=rollout_len)
        _, _, _, _, _, lo, _, _, _ = fc
        _, _, lv = ts.apply_fn(ts.params, lo)
        vals = jnp.concatenate([roll.value, lv[None,:]], axis=0)
        adv, ret = compute_gae(roll.reward, vals, roll.done)
        bo = roll.obs.reshape(-1,20); ba = roll.action.reshape(-1,4)
        blp = roll.log_prob.reshape(-1); badv = adv.reshape(-1); bret = ret.reshape(-1)
        def es(ts, _):
            (l,i),g = jax.value_and_grad(ppo_loss, has_aux=True)(ts.params, ts.apply_fn, (bo,ba,blp,badv,bret))
            return ts.apply_gradients(grads=g), i
        tsn, _ = jax.lax.scan(es, ts, None, length=4)
        return fc, tsn, jnp.mean(roll.reward)
    
    metrics = []; t0 = time.time()
    for u in range(num_updates):
        rng, rk = jax.random.split(rng)
        co, ts, mr = collect_update(rk, ts, ps, ls, pa, sc, obs, pp, mx, ol)
        rng_n, ps, ls, pa, sc, obs, pp, mx, ol = co; rng = rng_n
        tsd = (u+1)*batch_sz
        if u % 100 == 0:
            r = float(mr); el = time.time()-t0
            print(f"  [{label:20s}] {u:5d}/{num_updates} | {tsd:>10,} | R: {r:+.4f} | FPS: {tsd/el:,.0f}")
            metrics.append({'update': u, 'total_steps': tsd, 'mean_reward': r, 'elapsed': el})
    
    final_r = float(mr)
    print(f"  [{label:20s}] Done! Final: {final_r:.4f}, Time: {time.time()-t0:.1f}s")
    return metrics, final_r


# =============================================================================
# 实验 A: OLIP 缩放实验
# =============================================================================
def run_experiment_A(save_dir, num_envs=512, total_steps=30_000_000):
    """4个 DR 级别 × Full/No-OLIP"""
    print("\n" + "="*60)
    print("  实验 A: OLIP 缩放实验 (域随机化范围 vs OLIP 贡献)")
    print("="*60)
    
    dr_levels = [
        ("DR20", (0.8, 1.2), (0.9, 1.1)),
        ("DR40", (0.6, 1.4), (0.7, 1.3)),
        ("DR60", (0.4, 1.6), (0.5, 1.5)),
        ("DR80", (0.2, 1.8), (0.3, 1.7)),
    ]
    
    step_fn = make_step()
    results_A = {}
    
    for dr_name, mass_r, thrust_r in dr_levels:
        reset_fn = make_reset(mass_r, thrust_r, motor_degrade=False)
        
        for mode in ['full', 'no_olip']:
            label = f"{dr_name}_{mode}"
            metrics, final_r = train_one(mode, reset_fn, step_fn, num_envs, total_steps, seed=42, label=label)
            results_A[label] = {'metrics': metrics, 'final_reward': final_r, 'dr': dr_name, 'mode': mode}
            
            with open(os.path.join(save_dir, f'{label}_metrics.json'), 'w') as f:
                json.dump(metrics, f, indent=2)
    
    return results_A


# =============================================================================
# 实验 B: 电机退化实验
# =============================================================================
def run_experiment_B(save_dir, num_envs=512, total_steps=50_000_000):
    """电机退化场景下 Full/No-L3Diag/Baseline 对比"""
    print("\n" + "="*60)
    print("  实验 B: 电机退化实验 (1-2个电机降至50-80%效率)")
    print("="*60)
    
    reset_fn = make_reset((0.7, 1.3), (0.8, 1.2), motor_degrade=True)
    step_fn = make_step()
    results_B = {}
    
    for mode in ['full', 'no_l3diag', 'baseline']:
        label = f"motor_degrade_{mode}"
        metrics, final_r = train_one(mode, reset_fn, step_fn, num_envs, total_steps, seed=42, label=label)
        results_B[label] = {'metrics': metrics, 'final_reward': final_r, 'mode': mode}
        
        with open(os.path.join(save_dir, f'{label}_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
    
    return results_B


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--envs", type=int, default=512)
    args = parser.parse_args()
    
    save_dir = args.save_dir or str(ROOT_DIR / "logs" / "ablation_extended")
    os.makedirs(save_dir, exist_ok=True)
    
    print("="*60)
    print("  AeroCat 扩展消融实验套件")
    print("="*60)
    
    # 实验 A: 8 runs × 30M steps ≈ 4 min
    results_A = run_experiment_A(save_dir, args.envs, total_steps=30_000_000)
    
    # 实验 B: 3 runs × 50M steps ≈ 3 min
    results_B = run_experiment_B(save_dir, args.envs, total_steps=50_000_000)
    
    # 汇总
    print("\n" + "="*60)
    print("  实验 A 结果: OLIP 缩放")
    print("="*60)
    for dr in ["DR20", "DR40", "DR60", "DR80"]:
        f = results_A[f"{dr}_full"]['final_reward']
        n = results_A[f"{dr}_no_olip"]['final_reward']
        delta = f - n
        print(f"  {dr}: Full={f:.4f}, No-OLIP={n:.4f}, Δ={delta:+.4f} ({delta/f*100:+.1f}%)")
    
    print("\n" + "="*60)
    print("  实验 B 结果: 电机退化场景")
    print("="*60)
    for mode in ['full', 'no_l3diag', 'baseline']:
        r = results_B[f"motor_degrade_{mode}"]['final_reward']
        print(f"  {mode:12s}: {r:.4f}")
    
    print("\n所有实验完成!")
