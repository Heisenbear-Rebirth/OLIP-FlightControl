"""
PID Expert v3 — 从 Optuna JSON 文件加载分段 PID 增益

支持 10 段 lambda 分区，每段独立参数。
所有下游脚本（render/collect/eval）统一调用此模块。
"""

import json
import os
import numpy as np
import jax.numpy as jnp
from typing import Dict, List, Optional

# 默认 PID 参数（当没有 JSON 文件时使用的 fallback，来自 eval_pid_expert_v2）
DEFAULT_PARAMS = {
    'k_att_rp': 37.5620,
    'ki_att_rp': 0.9554,
    'k_rate_rp': 0.4426,
    'k_att_yaw': 12.0030,
    'ki_att_yaw': 0.2196,
    'k_rate_yaw': 3.1724,
    'kp_vz': 2.9028,
    'ki_vz': 0.5506,
    'kd_vz': 0.1856,
    'kp_vel': 0.0145,
    'ki_vel': 0.0172,
    'kd_vel': 0.0034,
    'i_limit_vel': 0.3219,
}

# Optuna 搜索空间定义（供 tune_pid_optuna.py 使用）
SEARCH_SPACE = {
    'k_att_rp':    (5.0,   80.0),
    'ki_att_rp':   (0.01,   3.0),
    'k_rate_rp':   (0.05,   5.0),
    'k_att_yaw':   (1.0,   30.0),
    'ki_att_yaw':  (0.01,   1.0),
    'k_rate_yaw':  (0.1,   10.0),
    'kp_vz':       (0.5,   10.0),
    'ki_vz':       (0.05,   3.0),
    'kd_vz':       (0.01,   1.0),
    'kp_vel':      (0.001,  0.1),
    'ki_vel':      (0.001,  0.1),
    'kd_vel':      (0.0005, 0.05),
    'i_limit_vel': (0.05,   1.0),
}

# Lambda bins
NUM_BINS = 10
LAMBDA_BINS = [(i / NUM_BINS, (i + 1) / NUM_BINS) for i in range(NUM_BINS)]


def load_pid_gains(json_path: str) -> Dict:
    """
    加载 Optuna 调参输出的 JSON 文件

    Args:
        json_path: JSON 文件路径

    Returns:
        gains: 完整 JSON dict，包含 'bins' 列表
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


def get_params_for_lambda(gains: Dict, lam: float) -> Dict:
    """
    根据 lambda 值选择对应 bin 的 PID 参数

    Args:
        gains: 从 JSON 加载的完整字典
        lam:   当前 curriculum lambda [0, 1]

    Returns:
        params: PID 参数字典
    """
    for b in gains['bins']:
        if b['lambda_min'] <= lam < b['lambda_max']:
            return b['params']
    # lambda=1.0 匹配最后一个 bin
    return gains['bins'][-1]['params']


def get_default_gains() -> Dict:
    """
    生成默认的全 lambda 增益（所有 bin 使用相同参数）

    Returns:
        gains: 与 JSON 格式一致的字典
    """
    bins = []
    for lo, hi in LAMBDA_BINS:
        bins.append({
            'lambda_min': lo,
            'lambda_max': hi,
            'params': dict(DEFAULT_PARAMS),
            'optuna_score': -1.0,
            'n_trials': 0,
        })
    return {'version': 'default', 'bins': bins}


def save_gains(gains: Dict, json_path: str):
    """保存增益到 JSON 文件"""
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(gains, f, indent=2)
    print(f"[+] 已保存 PID 增益到 {json_path}")
