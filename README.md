# AeroCat: 基于在线物理标定与深度强化学习的无人机自适应飞行控制系统

## 项目简介

本项目提出一种 **OLIP（开环物理标定）+ RL + PID 级联** 的多旋翼无人机自适应飞行控制方法。核心创新包括：

1. **OLIP 开环阶跃标定**：飞行前 6 秒阶跃激励提取 5 维物理特征向量
2. **物理特征条件化 RL**：将 OLIP 特征注入 PPO 策略观测空间，实现免调参自适应
3. **L3 内环诊断反馈**：PID 积分器和混控饱和度回传 RL，感知控制边界

## 目录结构

```
code/
├── scripts/                    # 训练和实验脚本
│   ├── train_hover.py          # PPO 悬停训练 (主训练脚本)
│   ├── run_ablation.py         # 基础消融实验 (Full/No-OLIP/No-L3Diag)
│   ├── run_ablation_hard.py    # 困难条件消融 (+阵风扰动)
│   ├── run_ablation_extended.py# 扩展消融: OLIP缩放 + 电机退化
│   ├── run_l3_extreme.py       # L3 极端验证 (重载+偏置力矩)
│   ├── plot_training.py        # 训练曲线可视化
│   ├── plot_ablation.py        # 消融对比图
│   ├── plot_ablation_hard.py   # 困难消融图
│   ├── plot_ablation_extended.py # 扩展消融图 (OLIP缩放/电机退化)
│   └── plot_l3_extreme.py      # L3 极端实验图
│
├── src/aerocat/                # 核心代码库
│   ├── core/state.py           # 物理状态和参数定义
│   ├── physics/                # 物理引擎 (四元数动力学/电机/电池/传感器)
│   ├── control/l3_controller.py# L3 PID 控制器 (Rate PID + 混控器)
│   ├── calibration/olip.py     # OLIP 在线物理标定模块
│   ├── envs/hover_env.py       # 悬停环境 (20维观测/3项奖励/域随机化)
│   ├── networks/               # Actor-Critic 网络
│   └── training/               # PPO 训练器
│
└── logs/                       # 实验结果
    ├── hover_v2_full/          # 主训练结果 (50M步, 奖励0.537)
    ├── ablation/               # 基础消融
    ├── ablation_hard/          # 困难条件消融
    ├── ablation_extended/      # OLIP缩放 + 电机退化实验
    └── l3_extreme/             # L3 极端验证实验
```

## 关键实验结果

### OLIP 缩放实验
| 域随机化 | OLIP 增益 |
|:-------:|:---------:|
| ±20% | −0.7% |
| ±40% | **+1.3%** |
| ±80% | **+2.0%** |

### 极端条件下整体增益
| 实验组 | 奖励 | vs Baseline |
|:------:|:----:|:-----------:|
| Full (S+L3+OLIP) | 0.488 | **+7.8%** |
| No-L3Diag (S+OLIP) | 0.487 | +7.6% |
| Baseline (S only) | 0.450 | — |

## 技术栈

- **框架**: JAX + Flax + Optax
- **算法**: PPO (Proximal Policy Optimization)
- **并行**: 512 环境 GPU 全并行
- **硬件**: NVIDIA RTX 4060, 训练速度 ~787k FPS
