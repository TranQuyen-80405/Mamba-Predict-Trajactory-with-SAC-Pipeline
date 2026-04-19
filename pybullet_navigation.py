import pybullet as p
import pybullet_data
import numpy as np
import time
import random
import math
import os
import csv
from datetime import datetime

# Subprocess from Jupyter may inherit MPLBACKEND=matplotlib_inline; matplotlib
# validates env on `import matplotlib` *before* matplotlib.use() runs.
_mpl_be = os.environ.get("MPLBACKEND", "")
if "inline" in _mpl_be.lower() or _mpl_be.startswith("module://"):
    os.environ["MPLBACKEND"] = "Agg"

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

MAP_SIZE = 12.0
SIM_FREQ = 200
CTRL_FREQ = 50

N_STATIC_OBS    = 12      
N_DYNAMIC_OBS   = 5       
DYN_SPEED       = 1.8

CAMERA_DIST = 4.0
CAMERA_PITCH = -35.0

WHEEL_RADIUS = 0.06
WHEEL_BASE = 0.23
MAX_V_MPS = 1.5
MAX_W_RADPS = 2.5
MIN_OBS_POINTS = 40
COVER_CELL_SIZE = 0.5

class RL_Env:
    def __init__(self, gui=True):
        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1.0 / SIM_FREQ)

        self.plane_id = p.loadURDF("plane.urdf")
        self.create_walls()
        self.static_obs = []
        self.dynamic_obs = []
        self.step_idx = 0
        self.episode_idx = 1
        self.current_episode_rewards = []
        self.current_episode_logs = []
        self.episode_history = []
        self.visited_cells = set()
        self.totalCover = 0
        self.coverable_area = (2 * MAP_SIZE) * (2 * MAP_SIZE)
        self.total_coverable_cells = max(1, int(self.coverable_area / (COVER_CELL_SIZE ** 2)))
        self.spawn_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.path_positions = []

        self._last_pc_time = 0.0          
        self._last_pc_summary = None      

        self._init_logging_files()
        self.spawn()
        self.create_obstacles()
        self._reset_episode_metrics()

        print("W/S/A/D = Control robot | R = Reset | Q = Quit")

    def spawn(self):
        if hasattr(self, 'robot'):
            p.removeBody(self.robot)
        self.robot = p.loadURDF("urdf/diff_drive_2wheel.urdf", [0, 0, 0.12])
        self.spawn_pos = np.array([0.0, 0.0], dtype=np.float32)
        self.wheels = []
        self.steers = []
        self.left_wheels = []
        self.right_wheels = []
        for i in range(p.getNumJoints(self.robot)):
            name = p.getJointInfo(self.robot, i)[1].decode().lower()
            if "wheel" in name: self.wheels.append(i)
            if "steer" in name: self.steers.append(i)
            if "left_wheel" in name:
                self.left_wheels.append(i)
            if "right_wheel" in name:
                self.right_wheels.append(i)

    def _init_logging_files(self):
        self.logs_root_dir = os.path.join("experiment")
        os.makedirs(self.logs_root_dir, exist_ok=True)

        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logs_dir = os.path.join(self.logs_root_dir, self.run_id)
        self.episodes_dir = os.path.join(self.logs_dir, "episodes")
        os.makedirs(self.episodes_dir, exist_ok=True)

        self.epoch_csv_path = os.path.join(self.logs_dir, "epoch.csv")
        self.position_csv_path = os.path.join(self.logs_dir, "position.csv")
        self.base_map_path = os.path.join(self.logs_dir, "map_base.png")
        self.obstacles_csv_path = os.path.join(self.logs_dir, "obstacles_map.csv")

        self.current_episode_dir = None
        self.current_episode_log_csv_path = None
        self.current_episode_position_csv_path = None
        self.current_episode_pc_csv_path = None

        with open(self.epoch_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode", "avg_reward", "avg_cover_rate", "final_total_cover",
                "final_cover_rate", "steps", "reason"
            ])

        with open(self.position_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])

    def _reset_episode_metrics(self):
        self.step_idx = 0
        self.current_episode_rewards = []
        self.current_episode_logs = []
        self.visited_cells = set()
        self.totalCover = 0
        self.path_positions = []

        self._last_pc_time = 0.0
        self._last_pc_summary = None

        x, y = self.getPosition()
        self.path_positions.append((x, y))
        self._mark_position_coverage(x, y)
        self._init_episode_logging()
        self.draw_episode_initial_map()

    def _init_episode_logging(self):
        episode_folder = f"episode_{self.episode_idx}"
        self.current_episode_dir = os.path.join(self.episodes_dir, episode_folder)
        os.makedirs(self.current_episode_dir, exist_ok=True)

        self.current_episode_log_csv_path = os.path.join(self.current_episode_dir, "experiment_log.csv")
        self.current_episode_position_csv_path = os.path.join(self.current_episode_dir, "position.csv")
        self.current_episode_pc_csv_path = os.path.join(self.current_episode_dir, "pointcloud.csv")

        with open(self.current_episode_log_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "episode", "step", "x", "y", "reward", "loss", "coverage_rate",
                "total_cover", "collision"
            ])

        with open(self.current_episode_position_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["x", "y"])
            
        with open(self.current_episode_pc_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "point_idx", "x", "y", "z", "intensity"])

    def create_walls(self):
        h, t = MAP_SIZE, 0.3
        for pos, size in [([0,h,0.5],[h,t,1]), ([0,-h,0.5],[h,t,1]),
                          ([h,0,0.5],[t,h,1]), ([-h,0,0.5],[t,h,1])]:
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=size)
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=size, rgbaColor=[0.4,0.4,0.5,1])
            p.createMultiBody(0, col, vis, pos)

    def create_obstacles(self):
        for bid in self.static_obs + self.dynamic_obs:
            p.removeBody(bid)
        self.static_obs.clear()
        self.dynamic_obs.clear()

        for _ in range(N_STATIC_OBS):
            x = random.uniform(-MAP_SIZE+2, MAP_SIZE-2)
            y = random.uniform(-MAP_SIZE+2, MAP_SIZE-2)
            if abs(x) < 3 and abs(y) < 3: 
                continue  
            size = random.uniform(0.2,0.9)
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size, size, size])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[size, size, size], rgbaColor=[0.2, 0.5, 0.8, 1])
            bid = p.createMultiBody(0, col, vis, [x, y, size])
            self.static_obs.append(bid)

        for _ in range(N_DYNAMIC_OBS):
            x = random.uniform(-MAP_SIZE+3, MAP_SIZE-3)
            y = random.uniform(-MAP_SIZE+3, MAP_SIZE-3)
            if abs(x) < 4 and abs(y) < 4: 
                continue
            size = 0.35
            col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size, size, size])
            vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[size, size, size], rgbaColor=[0.9, 0.3, 0.2, 1])
            bid = p.createMultiBody(1.0, col, vis, [x, y, size])
            angle = random.uniform(0, 2*np.pi)
            vx = DYN_SPEED * np.cos(angle)
            vy = DYN_SPEED * np.sin(angle)
            p.resetBaseVelocity(bid, [vx, vy, 0])
            self.dynamic_obs.append(bid)

    def reset(self):
        if self.current_episode_rewards:
            self._close_episode(reason="reset")

        self._reset_world_for_next_episode()
        self._reset_episode_metrics()
        return self.get_obs()

    def _reset_world_for_next_episode(self):
        self.spawn()
        p.resetBasePositionAndOrientation(self.robot, [0, 0, 0.12], [0, 0, 0, 1])
        p.resetBaseVelocity(self.robot, [0, 0, 0])
        self.create_obstacles()

    def drive(self, linear_x, angular_z):
        v = linear_x
        w = angular_z
        omega_l = (v - 0.5 * WHEEL_BASE * w) / WHEEL_RADIUS
        omega_r = (v + 0.5 * WHEEL_BASE * w) / WHEEL_RADIUS

        for j in self.left_wheels:
            p.setJointMotorControl2(self.robot, j, p.VELOCITY_CONTROL,
                                    targetVelocity=omega_l, force=80)
        for j in self.right_wheels:
            p.setJointMotorControl2(self.robot, j, p.VELOCITY_CONTROL,
                                    targetVelocity=omega_r, force=80)

    def step_cmd_vel(self, linear_x, angular_z):
        return self.step((linear_x, angular_z))

    def _camera_pose(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        rot = p.getMatrixFromQuaternion(orn)
        forward = np.array([rot[0], rot[3], rot[6]], dtype=np.float32)
        up = np.array([rot[2], rot[5], rot[8]], dtype=np.float32)
        right = np.cross(forward, up)
        right /= (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, forward)
        up /= (np.linalg.norm(up) + 1e-8)

        cam_pos = np.array(pos, dtype=np.float32) + forward * 0.2
        cam_pos[2] += 0.2
        return cam_pos, forward, right, up

    def get_camera_data(self, width=128, height=128, fov=60, near=0.1, far=10.0):
        cam_pos, forward_vec, right_vec, up_vec = self._camera_pose()
        target_pos = cam_pos + forward_vec * 1.0

        view_matrix = p.computeViewMatrix(cam_pos.tolist(), target_pos.tolist(), up_vec.tolist())
        proj_matrix = p.computeProjectionMatrixFOV(fov, width/height, near, far)
        
        cam_data = p.getCameraImage(width, height, viewMatrix=view_matrix, projectionMatrix=proj_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)

        # RGB
        rgba_buffer = np.reshape(cam_data[2], (height, width, 4))
        rgb = rgba_buffer[:, :, :3]
        
        # Depth map & seg mask
        depth_buffer = np.reshape(cam_data[3], (height, width))
        depth_m = far * near / (far - (far - near) * depth_buffer)
        seg_mask = np.reshape(cam_data[4], (height, width))
        
        # Point cloud 3D
        cx, cy = width / 2, height / 2
        fx = fy = (width / 2) / np.tan(np.deg2rad(fov / 2))
        x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
        z = depth_m
        x = (x_grid - cx) * z / fx
        y = (y_grid - cy) * z / fy
        point_cloud_cam = np.stack((x, y, z), axis=-1)

        world_points = (
            cam_pos[None, None, :]
            + right_vec[None, None, :] * point_cloud_cam[:, :, 0:1]
            - up_vec[None, None, :] * point_cloud_cam[:, :, 1:2]
            + forward_vec[None, None, :] * point_cloud_cam[:, :, 2:3]
        )

        return rgb, depth_m, seg_mask, point_cloud_cam.reshape(-1, 3), world_points.reshape(-1, 3)

    @staticmethod
    def _clip_cmd_vel(linear_x, angular_z):
        linear_x = float(np.clip(linear_x, -MAX_V_MPS, MAX_V_MPS))
        angular_z = float(np.clip(angular_z, -MAX_W_RADPS, MAX_W_RADPS))
        return linear_x, angular_z

    def get_pointcloud(self, world_frame=True, width=128, height=128, fov=60, near=0.1, far=10.0):
        _, depth_m, _, pc_cam, pc_world = self.get_camera_data(width=width, height=height, fov=fov, near=near, far=far)
        points = pc_world if world_frame else pc_cam
        flat_depth = depth_m.reshape(-1)
        intensity = 1.0 - np.clip((flat_depth - near) / (far - near + 1e-8), 0.0, 1.0)
        return np.concatenate([points, intensity[:, None]], axis=1)

    @staticmethod
    def _pc_summary(pc_xyzi: np.ndarray) -> dict:
        xyz = pc_xyzi[:, :3]
        return {
            "n":        len(pc_xyzi),
            "x_mean":   round(float(xyz[:, 0].mean()), 3),
            "y_mean":   round(float(xyz[:, 1].mean()), 3),
            "z_mean":   round(float(xyz[:, 2].mean()), 3),
            "x_min":    round(float(xyz[:, 0].min()),  3),
            "x_max":    round(float(xyz[:, 0].max()),  3),
            "z_min":    round(float(xyz[:, 2].min()),  3),
            "z_max":    round(float(xyz[:, 2].max()),  3),
            "i_mean":   round(float(pc_xyzi[:, 3].mean()), 3),
        }

    def _try_print_pc(self, step: int):
        now = time.time()
        if now - self._last_pc_time < 1.0:
            return                          

        self._last_pc_time = now
        pc = self.get_pointcloud()          
        self._append_pointcloud_csv(step, pc)
        summary = self._pc_summary(pc)

        if summary == self._last_pc_summary:
            return                          

        self._last_pc_summary = summary
        print(
            f"  [PC : step {step}] "
            f"points={summary['n']} | "
            f"centroid=({summary['x_mean']}, {summary['y_mean']}, {summary['z_mean']}) | "
            f"x:[{summary['x_min']}, {summary['x_max']}] "
            f"z:[{summary['z_min']}, {summary['z_max']}] | "
            f"intensity={summary['i_mean']}"
        )

    def _append_pointcloud_csv(self, step: int, pc_xyzi: np.ndarray):
        if self.current_episode_pc_csv_path is None:
            return

        with open(self.current_episode_pc_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            for point_idx, point in enumerate(pc_xyzi):
                writer.writerow([
                    step,
                    point_idx,
                    float(point[0]),
                    float(point[1]),
                    float(point[2]),
                    float(point[3]),
                ])

    def getPosition(self):
        pos, _ = p.getBasePositionAndOrientation(self.robot)
        return float(pos[0]), float(pos[1])

    def _position_to_cell(self, x, y):
        return int(np.floor(x / COVER_CELL_SIZE)), int(np.floor(y / COVER_CELL_SIZE))

    def _mark_position_coverage(self, x, y):
        old = self.totalCover
        cell = self._position_to_cell(x, y)
        self.visited_cells.add(cell)
        self.totalCover = len(self.visited_cells)
        return self.totalCover - old

    def getNewCoverage(self):
        x, y = self.getPosition()
        return self._mark_position_coverage(x, y)

    def totalCoverage(self):
        return self.totalCover

    def get_coverage_rate(self):
        return float(self.totalCover / self.total_coverable_cells)

    def _compute_reward(self, new_cover, collision):
        reward = float(new_cover)
        if new_cover <= 0:
            reward -= 100
        else:
            reward += 10.0 * new_cover
        if collision:
            reward -= 200
        return reward

    def _append_position_log(self, x, y):
        with open(self.position_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([x, y])

        if self.current_episode_position_csv_path:
            with open(self.current_episode_position_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([x, y])

    def _append_experiment_log(self, row):
        if self.current_episode_log_csv_path is None:
            return

        with open(self.current_episode_log_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                row["episode"], row["step"], row["x"], row["y"], row["reward"], row["loss"],
                row["cover_rate"], row["total_cover"], row["collision"]
            ])

    def getLog(self):
        return list(self.current_episode_logs)

    def _collect_obstacle_boxes(self):
        boxes = []
        for bid in self.static_obs + self.dynamic_obs:
            aabb_min, aabb_max = p.getAABB(bid)
            boxes.append({
                "type": "dynamic" if bid in self.dynamic_obs else "static",
                "x0": float(aabb_min[0]),
                "y0": float(aabb_min[1]),
                "w": float(aabb_max[0] - aabb_min[0]),
                "h": float(aabb_max[1] - aabb_min[1]),
            })
        return boxes

    def draw_episode_initial_map(self, save_path=None):
        if plt is None or self.current_episode_dir is None:
            return None
        from matplotlib.patches import Rectangle

        if save_path is None:
            save_path = os.path.join(self.current_episode_dir, "map.png")

        boxes = self._collect_obstacle_boxes()
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_title(f"Episode {self.episode_idx} Initial Map")
        ax.set_xlim(-MAP_SIZE, MAP_SIZE)
        ax.set_ylim(-MAP_SIZE, MAP_SIZE)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

        for b in boxes:
            color = "tab:red" if b["type"] == "dynamic" else "tab:blue"
            from matplotlib.patches import Rectangle as Rect
            rect = Rect((b["x0"], b["y0"]), b["w"], b["h"], color=color, alpha=0.35)
            ax.add_patch(rect)

        ax.scatter([self.spawn_pos[0]], [self.spawn_pos[1]], c="gold", s=80, marker="*", label="Spawn")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(save_path, dpi=140)
        plt.close(fig)
        return save_path

    def draw_episode_path(self, episode=None, save_path=None):
        if plt is None:
            return None
        from matplotlib.patches import Rectangle

        target_episode = self.episode_idx if episode is None else int(episode)
        episode_dir = os.path.join(self.episodes_dir, f"episode_{target_episode}")
        position_csv_path = os.path.join(episode_dir, "position.csv")

        if not os.path.exists(position_csv_path):
            return None

        points = []
        with open(position_csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    x_val = float(row.get("x", 0.0))
                    y_val = float(row.get("y", 0.0))
                except (TypeError, ValueError):
                    continue
                points.append((x_val, y_val))

        if not points:
            return None

        if save_path is None:
            save_path = os.path.join(episode_dir, "path.png")

        boxes = self._collect_obstacle_boxes()
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.set_title(f"Episode {target_episode} Path")
        ax.set_xlim(-MAP_SIZE, MAP_SIZE)
        ax.set_ylim(-MAP_SIZE, MAP_SIZE)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)

        for b in boxes:
            color = "tab:red" if b["type"] == "dynamic" else "tab:blue"
            rect = Rectangle((b["x0"], b["y0"]), b["w"], b["h"], color=color, alpha=0.25)
            ax.add_patch(rect)

        path = np.array(points, dtype=np.float32)
        ax.plot(path[:, 0], path[:, 1], "k-", linewidth=1.2, alpha=0.9, label="Path")
        ax.scatter(path[:, 0], path[:, 1], s=10, c=np.arange(path.shape[0]), cmap="viridis", label="Steps")
        ax.scatter([self.spawn_pos[0]], [self.spawn_pos[1]], c="gold", s=80, marker="*", label="Spawn")
        ax.legend(loc="upper right")
        fig.tight_layout()
        fig.savefig(save_path, dpi=140)
        plt.close(fig)
        return save_path

    def _close_episode(self, reason="end"):
        if not self.current_episode_rewards:
            return
        avg_reward = float(np.mean(self.current_episode_rewards))
        avg_cover_rate = float(np.mean([r["cover_rate"] for r in self.current_episode_logs])) if self.current_episode_logs else 0.0
        final_cover_rate = self.get_coverage_rate()
        print(
            f"\n[Episode {self.episode_idx} closed: {reason}] "
            f"avg_reward={avg_reward:.4f} | avg_cover_rate={avg_cover_rate*100:.2f}% | "
            f"final_total_cover={self.totalCoverage()} | final_cover_rate={final_cover_rate*100:.2f}%"
        )
        if self.current_episode_dir is not None:
            path_img = self.draw_episode_path(self.episode_idx)
            if path_img is not None:
                print(f"Saved path map: {path_img}")

        with open(self.epoch_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                self.episode_idx, avg_reward, avg_cover_rate,
                self.totalCoverage(), final_cover_rate, self.step_idx, reason,
            ])

        self.episode_history.append({
            "episode": self.episode_idx,
            "avg_reward": avg_reward,
            "avg_cover_rate": avg_cover_rate,
            "final_total_cover": self.totalCoverage(),
            "final_cover_rate": final_cover_rate,
            "reason": reason,
        })
        self.episode_idx += 1

    def _action_from_keyboard(self):
        events = p.getKeyboardEvents()

        def is_down(*codes):
            return any(events.get(code, 0) & p.KEY_IS_DOWN for code in codes)

        def was_triggered(*codes):
            return any(events.get(code, 0) & p.KEY_WAS_TRIGGERED for code in codes)

        fwd = is_down(ord('w'), p.B3G_UP_ARROW)
        bwd = is_down(ord('s'), p.B3G_DOWN_ARROW)
        lft = is_down(ord('a'), p.B3G_LEFT_ARROW)
        rgt = is_down(ord('d'), p.B3G_RIGHT_ARROW)
        rst = was_triggered(ord('r'))
        q   = was_triggered(ord('q'))

        if q:
            return None, None, rst, q, False

        linear_x  = MAX_V_MPS if fwd else (-0.7 * MAX_V_MPS if bwd else 0.0)
        angular_z = MAX_W_RADPS if lft else (-MAX_W_RADPS if rgt else 0.0)
        has_motion = bool(fwd or bwd or lft or rgt)
        return linear_x, angular_z, rst, q, has_motion

    def step(self, action=None, loss=None):
        if action is None:
            linear_x, angular_z, rst, q, has_motion = self._action_from_keyboard()
            if q:
                self._close_episode(reason="quit")
                return None
            if rst:
                return self.reset()
            if not has_motion:
                return self.get_obs()
        else:
            linear_x, angular_z = action

        linear_x, angular_z = self._clip_cmd_vel(linear_x, angular_z)
        self.drive(linear_x, angular_z)

        for _ in range(SIM_FREQ // CTRL_FREQ):
            p.stepSimulation()

        self._bounce_dynamic_obs()
        self.step_idx += 1

        x, y = self.getPosition()
        self.path_positions.append((x, y))
        new_cover = self.getNewCoverage()

        obs = self.get_obs()
        collision  = bool(obs["collision"])
        reward     = self._compute_reward(new_cover=new_cover, collision=collision)
        self.current_episode_rewards.append(reward)

        dx = x - float(self.spawn_pos[0])
        dy = y - float(self.spawn_pos[1])
        dist = math.hypot(dx, dy)
        cover_rate = self.get_coverage_rate()
        loss_value = float(loss) if loss is not None else np.nan

        row = {
            "step": self.step_idx, "episode": self.episode_idx,
            "x": x, "y": y, "dx": dx, "dy": dy, "dist": dist,
            "reward": reward, "loss": loss_value,
            "new_cover": int(new_cover), "total_cover": int(self.totalCoverage()),
            "cover_rate": cover_rate, "collision": collision,
        }
        self.current_episode_logs.append(row)
        self._append_position_log(x, y)
        self._append_experiment_log(row)

        loss_txt = f"{loss_value:.4f}" if not np.isnan(loss_value) else "nan"
        print(
            f"episode: {self.episode_idx} | step: {self.step_idx} | x: ({x:.3f}, {y:.3f}) | "
            f"reward: {reward:.4f} | loss: {loss_txt} | coverage rate: {cover_rate*100:.2f}%"
        )

        self._try_print_pc(self.step_idx)

        obs["reward"]       = reward
        obs["new_coverage"] = new_cover
        obs["total_coverage"] = self.totalCoverage()
        obs["coverage_rate"]  = cover_rate
        obs["spawn_delta"]    = np.array([dx, dy], dtype=np.float32)

        if collision:
            self._close_episode(reason="collision")
            self._reset_world_for_next_episode()
            self._reset_episode_metrics()
            print("[COLLISION] collision detected")
            return self.get_obs()

        return obs

    def _bounce_dynamic_obs(self):
        h = MAP_SIZE - 1.0
        for bid in self.dynamic_obs:
            pos, _ = p.getBasePositionAndOrientation(bid)
            vel, _ = p.getBaseVelocity(bid)
            vx, vy = vel[0], vel[1]
            if abs(pos[0]) > h or abs(pos[1]) > h:
                p.resetBaseVelocity(bid, [-vx, -vy, 0])

    def get_obs(self):
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        vel, _   = p.getBaseVelocity(self.robot)
        yaw      = p.getEulerFromQuaternion(orn)[2]
        speed    = np.hypot(vel[0], vel[1])
        rgb, depth_map, seg_mask, _, _ = self.get_camera_data()
        return {
            "pos":       np.array(pos[:2]),
            "yaw":       yaw,
            "speed":     speed,
            "collision": self.check_collision(),
            "rgb":       rgb,
            "depth":     depth_map,
            "seg":       seg_mask,
        }

    def check_collision(self):
        contacts = p.getContactPoints(self.robot)
        for c in contacts:
            if c[2] != 0:
                return True
        return False

    def run_keyboard(self):
        while True:
            obs = self.step()
            if obs is None:
                break

            robot_pos = obs['pos']
            robot_yaw = obs['yaw']
            p.resetDebugVisualizerCamera(
                cameraDistance=CAMERA_DIST,
                cameraYaw=math.degrees(robot_yaw) + 180,
                cameraPitch=CAMERA_PITCH,
                cameraTargetPosition=[robot_pos[0], robot_pos[1], 0.2]
            )

            status = "COLLISION!" if obs["collision"] else f"Speed: {obs['speed']:.2f}"
            print(status + " " * 20, end="\r")

            time.sleep(1/CTRL_FREQ)
        p.disconnect()

if __name__ == "__main__":
    env = RL_Env(gui=True)
    env.run_keyboard()