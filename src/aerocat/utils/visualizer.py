"""
AeroCat v18.0 Visualization Utils (Ported from v17.2)
"""

import os
import matplotlib
matplotlib.use('Agg') # Force headless backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import imageio
from typing import Dict, Any, Optional
import io
from PIL import Image

from ..envs.uav_env import EnvState

class VideoRecorder:
    """
    Renders 3D drone flight visualization from environment state.
    Uses v17.2 wireframe style for better attitude visibility.
    """
    def __init__(self, filename: str = "test_flight.mp4", fps: int = 50):
        self.filename = filename
        self.fps = fps
        self.frames = []
        
    def reset(self):
        """Clear frames buffer"""
        self.frames = []
        
    def render_frame(self, state: EnvState, info: Dict[str, Any] = None, step: int = None):
        """
        渲染单帧 3D 飞行可视化（第一个环境）。

        Args:
            state: 环境状态（需已 jax.device_get 到 CPU）
            info: 包含奖励分项等信息的字典
            step: 当前步数（若 None 则从 info['step_count'] 提取）
        """
        pos = np.array(state.phys_state.position[0])    # NED
        quat = np.array(state.phys_state.quaternion[0])  # [w, x, y, z]
        throttle = np.array(state.phys_state.motor_throttle[0])
        rpms = throttle * 30000.0

        # NED -> ENU 显示坐标
        pos_vis = np.array([pos[1], pos[0], -pos[2]])

        # --- 画布 ---
        # 增加分辨率，背景色改为纯白色，凸显科研严谨风格
        fig = plt.figure(figsize=(10, 10), dpi=150, facecolor='white')
        # 减小外边距，最大化 3D 绘图区域
        plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('white')
        ax.tick_params(colors='black', labelsize=8)

        view_range = 2.0  # 缩小视野范围，让无人机显示更大
        ax.set_xlim(pos_vis[0] - view_range, pos_vis[0] + view_range)
        ax.set_ylim(pos_vis[1] - view_range, pos_vis[1] + view_range)
        # 强制 Z 轴比例与其他轴一致，避免拉伸畸变
        ax.set_zlim(pos_vis[2] - view_range, pos_vis[2] + view_range)
        ax.set_box_aspect([1, 1, 1])

        # 隐藏轴标签文字，因为留白已经取消，文字会被切掉
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # --- 无人机 ---
        self._draw_drone(ax, pos_vis, quat, rpms)

        # --- 速度矢量箭头 ---
        vel_ned = np.array(state.phys_state.velocity[0])
        vel_enu = np.array([vel_ned[1], vel_ned[0], -vel_ned[2]])

        v_target_body = np.array(state.l1_state.target_velocity_body[0])
        w, x, y, z = quat
        R_ned = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
        ])
        v_target_ned = R_ned @ v_target_body
        v_target_enu = np.array([v_target_ned[1], v_target_ned[0], -v_target_ned[2]])

        self._draw_velocity_vectors(ax, pos_vis, vel_enu, v_target_enu)

        # 图例
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(handles, labels, loc='upper right', fontsize=8, facecolor='white',
                      labelcolor='black', framealpha=0.8, edgecolor='silver')

        # --- 信息叠加 ---
        if step is None:
            step = info.get('step_count', [0])[0] if info else 0
        self._add_annotations(ax, pos_vis, state, info, step)

        # --- 捕获帧 ---
        fig.canvas.draw()
        try:
            image = np.array(fig.canvas.buffer_rgba())[:, :, :3]
            self.frames.append(image)
        except Exception as e:
            print(f"Frame capture failed: {e}")

        plt.close(fig)

    def _draw_velocity_vectors(self, ax, pos, vel_current, vel_target):
        """
        绘制速度矢量箭头
        
        Args:
            ax: matplotlib 3D 坐标轴
            pos: 无人机位置 (ENU)
            vel_current: 当前速度矢量 (ENU) [m/s]
            vel_target: 目标速度矢量 (ENU) [m/s]
        """
        # 缩放因子：1 m/s -> 0.2m 箭头长度
        scale = 0.2
        
        # 当前速度 (蓝色实线箭头)
        vel_mag = np.linalg.norm(vel_current)
        if vel_mag > 0.1:  # 仅当速度大于 0.1 m/s 时绘制
            ax.quiver(
                pos[0], pos[1], pos[2],  # 起点
                vel_current[0] * scale, vel_current[1] * scale, vel_current[2] * scale,  # 方向
                color='dodgerblue', arrow_length_ratio=0.2, linewidth=2.5,
                label=f'Current: {vel_mag:.1f} m/s'
            )
        
        # 目标速度 (黄色虚线箭头)
        target_mag = np.linalg.norm(vel_target)
        if target_mag > 0.1:  # 仅当目标速度大于 0.1 m/s 时绘制
            ax.quiver(
                pos[0], pos[1], pos[2],  # 起点
                vel_target[0] * scale, vel_target[1] * scale, vel_target[2] * scale,  # 方向
                color='gold', arrow_length_ratio=0.2, linewidth=2.0, linestyle='dashed',
                label=f'Target: {target_mag:.1f} m/s'
            )

    def _draw_drone(self, ax, pos, quat, rpms):
        """Draw X-config quadcopter"""
        # Rotation matrix from quaternion (w, x, y, z)
        # Note: This R converts Body -> World (NED)
        w, x, y, z = quat
        R_ned = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
            [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
        ])
        
        # To convert vector v_body to v_vis (ENU):
        # v_world_ned = R_ned @ v_body
        # v_world_enu = M @ v_world_ned
        # Where M maps [N, E, D] -> [E, N, U]
        # M = [[0, 1, 0], [1, 0, 0], [0, 0, -1]]
        
        M = np.array([
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, -1]
        ])
        
        R_vis = M @ R_ned
        
        arm_length = 0.25
        # X-Config angles (45, 135, 225, 315)
        angles = np.deg2rad([45, 135, 225, 315])
        body_points = []
        for ang in angles:
            body_points.append([arm_length * np.cos(ang), arm_length * np.sin(ang), 0])
        body_points = np.array(body_points)
        
        # Rotate and Translate
        # P_world = R @ P_body + Pos
        # We need to apply R_vis to P_body and add pos_vis
        
        world_points = (R_vis @ body_points.T).T + pos
        
        # Draw Arms
        for p in world_points:
            ax.plot([pos[0], p[0]], [pos[1], p[1]], [pos[2], p[2]], 'cyan', linewidth=2)
            
        # Draw Props
        for i, p in enumerate(world_points):
            rpm_norm = np.clip(rpms[i] / 30000.0, 0.1, 1.0)
            prop_rad = 0.1
            theta = np.linspace(0, 2*np.pi, 10)
            circle = np.array([np.cos(theta)*prop_rad, np.sin(theta)*prop_rad, np.zeros_like(theta)])
            
            # Rotate circle to visual frame
            world_circle = (R_vis @ circle).T + p
            
            color = 'red' if i < 2 else 'green' # Front (arms 0,1) vs Back (arms 2,3) visualization
            ax.plot(world_circle[:,0], world_circle[:,1], world_circle[:,2], color=color, alpha=0.5 + 0.5*rpm_norm)

    def _add_annotations(self, ax, pos, state, info, step):
        """
        信息叠加层（增强版）

        显示内容：步数/时间、高度/倾角、速度追踪、奖励分项、电池状态。
        info 字典中的可选键会自动显示（缺失时跳过）。
        """
        def gs(x):
            """安全提取标量"""
            if x is None:
                return 0.0
            x = np.asarray(x)
            return float(x.flat[0])

        # --- 基础物理量 ---
        height = pos[2]  # ENU: Z = Up
        q = np.array(state.phys_state.quaternion[0])
        z_body_z = 1.0 - 2.0 * (q[1]**2 + q[2]**2)
        tilt_deg = np.degrees(np.arccos(np.clip(z_body_z, -1, 1)))

        vel_world = np.array(state.phys_state.velocity[0])  # NED
        vel_mag = np.linalg.norm(vel_world)
        v_target_body = np.array(state.l1_state.target_velocity_body[0])
        vel_target_mag = np.linalg.norm(v_target_body)

        # vz (向上为正)
        vz_curr = -vel_world[2]  # NED Down -> Up
        vz_cmd = -v_target_body[2] if len(v_target_body) > 2 else 0.0

        # 电池
        volt = gs(state.phys_state.battery_voltage[0])
        soc = gs(state.phys_state.battery_soc[0])

        # --- 可选 info 项 ---
        reward   = gs(info.get('reward', 0)) if info else 0.0
        r_align  = gs(info.get('r_align', None)) if info else None
        r_vel    = gs(info.get('r_velocity', None)) if info else None
        r_spin   = gs(info.get('r_spin', None)) if info else None
        r_act    = gs(info.get('r_act', None)) if info else None

        t_sec = step * 0.02  # 50Hz

        # --- 构建文本 ---
        # 奖励颜色
        reward_color = 'green' if reward > 0.70 else 'red'

        lines = [
            f"Step {step:4d}  t={t_sec:.2f}s",
            f"─────────────────────",
            f"Height : {height:6.2f} m",
            f"Tilt   : {tilt_deg:6.1f}°",
            f"─────────────────────",
            f"V_act  : {vel_mag:5.2f} m/s",
            f"V_cmd  : {vel_target_mag:5.2f} m/s",
            f"Vz_act : {vz_curr:+5.2f} m/s",
            f"Vz_cmd : {vz_cmd:+5.2f} m/s",
            f"─────────────────────",
        ]

        # 奖励分项（仅当 info 提供时显示）
        if r_align is not None:
            lines.append(f"R_align: {r_align:6.3f}")
        if r_vel is not None:
            lines.append(f"R_vel  : {r_vel:6.3f}")
        if r_spin is not None:
            lines.append(f"R_spin : {r_spin:+6.3f}")
        if r_act is not None:
            lines.append(f"R_act  : {r_act:+6.3f}")
        lines.append(f"Reward : {reward:6.3f}")
        lines.append(f"─────────────────────")
        lines.append(f"Bat: {volt:.1f}V  SOC:{soc*100:.0f}%")

        text_str = "\n".join(lines)

        # 文本改为黑色，背景改为带轻微透明度的白色框
        ax.text2D(0.02, 0.98, text_str, transform=ax.transAxes, color='black',
                  fontsize=9, family='monospace', verticalalignment='top',
                  bbox=dict(facecolor='white', alpha=0.8, edgecolor='silver', boxstyle='round,pad=0.4'))

    def save(self):
        """Save video to file (Preserved v18 Implementation)"""
        if not self.frames:
            print("No frames to save.")
            return

        # Ensure directory exists
        import os
        # Resolve to absolute path for clear logging
        abs_path = os.path.abspath(self.filename)
        os.makedirs(os.path.dirname(abs_path) or '.', exist_ok=True)
        
        # print(f"Saving visualization to {abs_path}...")
        
        # Determine format based on extension
        if self.filename.endswith('.mp4'):
            try:
                import imageio
                imageio.mimsave(abs_path, self.frames, fps=self.fps, format='FFMPEG')
                # print(f"Saved MP4 to {abs_path}")
            except ImportError:
                print("imageio[ffmpeg] not found. Falling back to GIF.")
                self.filename = self.filename.replace('.mp4', '.gif')
                abs_path = abs_path.replace('.mp4', '.gif')
                self._save_gif(abs_path)
            except Exception as e:
                print(f"Failed to save MP4: {e}. Falling back to GIF.")
                self.filename = self.filename.replace('.mp4', '.gif')
                abs_path = abs_path.replace('.mp4', '.gif')
                self._save_gif(abs_path)
        else:
            self._save_gif(abs_path)
            
        # Note: fig is per-frame in this impl, no global fig to close
        
    def _save_gif(self, abs_path=None):
        path = abs_path or self.filename
        try: 
            import imageio
            imageio.mimsave(path, self.frames, fps=self.fps)
            print(f"Saved GIF to {path}")
        except Exception as e:
            print(f"Failed to save GIF: {e}")


class EnhancedVideoRecorder(VideoRecorder):
    """
    增强 VideoRecorder: 无地面 + yaw rate bars + wind arrow + 角速度信息

    用于 ppo_trainer.py 训练过程中的 eval 视频和独立渲染脚本。
    """

    def render_enhanced_frame(self, state: EnvState, info: Dict[str, Any] = None,
                              step: int = None,
                              yaw_rate_actual: float = 0.0,
                              yaw_rate_cmd: float = 0.0,
                              wind_ned: Optional[np.ndarray] = None,
                              extra_label: str = ""):
        """
        渲染增强帧 (无地面 + yaw rate + wind + angular velocity)

        Args:
            state:            EnvState (CPU)
            info:             奖励分项等信息字典
            step:             当前步数
            yaw_rate_actual:  heading-frame 实际偏航角速率 (rad/s)
            yaw_rate_cmd:     heading-frame 目标偏航角速率 (rad/s)
            wind_ned:         风速向量 [3], NED (m/s), None 则自动提取
            extra_label:      额外标题文本 (如 "SMPC Expert", "PPO Agent")
        """
        pos = np.array(state.phys_state.position[0])
        quat = np.array(state.phys_state.quaternion[0])
        throttle = np.array(state.phys_state.motor_throttle[0])
        rpms = throttle * 30000.0

        # NED → ENU 显示坐标
        pos_vis = np.array([pos[1], pos[0], -pos[2]])

        # --- 画布 (白色背景, 无地面) ---
        fig = plt.figure(figsize=(10, 10), dpi=150, facecolor='white')
        plt.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('white')
        ax.tick_params(colors='black', labelsize=8)

        view_range = 2.0
        ax.set_xlim(pos_vis[0] - view_range, pos_vis[0] + view_range)
        ax.set_ylim(pos_vis[1] - view_range, pos_vis[1] + view_range)
        ax.set_zlim(pos_vis[2] - view_range, pos_vis[2] + view_range)
        ax.set_box_aspect([1, 1, 1])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        # --- 无人机 ---
        self._draw_drone(ax, pos_vis, quat, rpms)

        # --- 速度矢量 (蓝色=实际, 金色=目标) ---
        vel_ned = np.array(state.phys_state.velocity[0])
        vel_enu = np.array([vel_ned[1], vel_ned[0], -vel_ned[2]])

        v_target_body = np.array(state.l1_state.target_velocity_body[0])
        w, x, y, z = quat
        R_ned = np.array([
            [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
            [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
            [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y]
        ])
        v_target_ned = R_ned @ v_target_body
        v_target_enu = np.array([v_target_ned[1], v_target_ned[0], -v_target_ned[2]])

        self._draw_velocity_vectors(ax, pos_vis, vel_enu, v_target_enu)

        # --- 偏航角速率指示器 ---
        self._draw_yaw_rate_bars(ax, pos_vis, yaw_rate_actual, yaw_rate_cmd)

        # --- 风力箭头 ---
        if wind_ned is None:
            wind_ned = np.array(state.phys_state.wind_velocity[0])
        self._draw_wind_arrow(ax, pos_vis, wind_ned)

        # --- 图例 ---
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(handles, labels, loc='upper right', fontsize=7,
                      facecolor='white', labelcolor='black',
                      framealpha=0.8, edgecolor='silver')

        # --- 基础信息叠加 ---
        if step is None:
            step = info.get('step_count', [0])[0] if info else 0
        self._add_annotations(ax, pos_vis, state, info, step)

        # --- 附加信息面板 (yaw rate + 角速度 + 风) ---
        wind_enu = np.array([wind_ned[1], wind_ned[0], -wind_ned[2]])
        wind_mag = np.linalg.norm(wind_enu)
        omega = np.array(state.phys_state.angular_velocity[0])
        omega_deg = np.degrees(omega)

        extra_lines = []
        if extra_label:
            extra_lines.append(f"─── {extra_label} ───")
        extra_lines.extend([
            f"─── Yaw Rate ──────",
            f"ψ̇_cmd: {float(yaw_rate_cmd)*180/np.pi:+7.1f} °/s",
            f"ψ̇_act: {float(yaw_rate_actual)*180/np.pi:+7.1f} °/s",
            f"─── Angular Vel ───",
            f"ω_x : {omega_deg[0]:+7.1f} °/s",
            f"ω_y : {omega_deg[1]:+7.1f} °/s",
            f"ω_z : {omega_deg[2]:+7.1f} °/s",
            f"─── Wind ──────────",
            f"Wind : {wind_mag:5.1f} m/s",
        ])
        ax.text2D(0.02, 0.42, "\n".join(extra_lines),
                  transform=ax.transAxes, color='black',
                  fontsize=8, family='monospace', verticalalignment='top',
                  bbox=dict(facecolor='lightyellow', alpha=0.8,
                            edgecolor='silver', boxstyle='round,pad=0.4'))

        # --- 捕获帧 ---
        fig.canvas.draw()
        try:
            image = np.array(fig.canvas.buffer_rgba())[:, :, :3]
            self.frames.append(image)
        except Exception as e:
            print(f"Frame capture failed: {e}")
        plt.close(fig)

    def _draw_yaw_rate_bars(self, ax, pos, yaw_rate_actual, yaw_rate_cmd):
        """偏航角速率水平条形指示器"""
        MAX_RATE = 100.0 * np.pi / 180.0
        MAX_LEN = 0.8

        def _draw_bar(rate, z_offset, color, lw):
            rate_f = float(rate)
            if abs(rate_f) < 0.02:
                return
            length = np.clip(abs(rate_f) / MAX_RATE, 0.0, 1.0) * MAX_LEN
            sign = np.sign(rate_f)
            z = pos[2] + z_offset
            ax.scatter([pos[0]], [pos[1]], [z],
                       color=color, s=lw * 6, alpha=0.6, zorder=5)
            ax.quiver(pos[0], pos[1], z,
                      sign * length, 0, 0,
                      color=color, arrow_length_ratio=0.15,
                      linewidth=lw, alpha=0.85)

        _draw_bar(yaw_rate_cmd,    0.55, 'crimson',    3.5)
        _draw_bar(yaw_rate_actual, 0.45, 'darkorange', 2.0)

    def _draw_wind_arrow(self, ax, pos, wind_ned):
        """风力合成箭头 (绿色)"""
        wind_enu = np.array([wind_ned[1], wind_ned[0], -wind_ned[2]])
        wind_mag = np.linalg.norm(wind_enu)
        scale = 0.12
        if wind_mag > 0.1:
            wind_origin = pos + np.array([0.0, 0.0, -0.35])
            ax.quiver(wind_origin[0], wind_origin[1], wind_origin[2],
                      wind_enu[0] * scale, wind_enu[1] * scale, wind_enu[2] * scale,
                      color='limegreen', arrow_length_ratio=0.18, linewidth=2.5,
                      alpha=0.85, label=f'Wind: {wind_mag:.1f} m/s')


def compute_heading_yaw_rate(quaternion, angular_velocity):
    """
    计算 heading-frame 偏航角速率 ψ̇

    与 uav_env.build_observation 中的 yaw_rate 计算完全一致。

    Args:
        quaternion:       [4] 四元数 [w, x, y, z]
        angular_velocity: [3] body 角速度 [p, q, r]

    Returns:
        yaw_rate: float heading-frame 偏航角速率 (rad/s)
    """
    qw, qx, qy, qz = quaternion[0], quaternion[1], quaternion[2], quaternion[3]
    p, q, r = angular_velocity[0], angular_velocity[1], angular_velocity[2]
    yaw_rate = (
        2.0 * (qx * qz - qw * qy) * p +
        2.0 * (qy * qz + qw * qx) * q +
        (1.0 - 2.0 * (qx**2 + qy**2)) * r
    )
    return float(yaw_rate)

