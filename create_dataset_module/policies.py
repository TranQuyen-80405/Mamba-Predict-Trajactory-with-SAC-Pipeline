"""
Rollout policies for dataset generation.

All policies share the same signature:
    act(obs) -> np.ndarray(2,)  # (linear_x, angular_z)

Mix ratios come from DataGenConfig (must sum to 1.0):
    random       default 0.50  (broad coverage, ~weak positives)
    scripted     default 0.30  (goal-oriented, realistic trajectories)
    adversarial  default 0.20  (pushes positives; tune if positive(1s) too low)
    stationary   default 0.00  (v=w=0; dynamic obstacles can still cause contact)

The observation dict has the following keys (produced by the generator
from DatasetEnv): "ego_xy", "yaw", "obstacles_xy", "goal_xy". Optional
keys may be ignored.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


MAX_V_MPS = 1.5
MAX_W_RADPS = 2.5


class _BasePolicy:
    def reset(self, rng: Optional[np.random.Generator] = None) -> None:
        self.rng = rng if rng is not None else np.random.default_rng()

    def act(self, obs: Dict) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class RandomPolicy(_BasePolicy):
    """
    Ornstein-Uhlenbeck noise on (v, w). Smoother than pure IID uniform
    which wastes most rollouts on pathological jerk; matches
    docs/strategy_full_pipeline.md § 5.1 description.
    """

    def __init__(
        self,
        theta: float = 0.15,
        sigma_v: float = 0.6,
        sigma_w: float = 0.8,
        dt: float = 0.05,
    ) -> None:
        self.theta = float(theta)
        self.sigma_v = float(sigma_v)
        self.sigma_w = float(sigma_w)
        self.dt = float(dt)
        self.state = np.zeros(2, dtype=np.float32)
        self.rng = np.random.default_rng()

    def reset(self, rng: Optional[np.random.Generator] = None) -> None:
        super().reset(rng)
        self.state = np.zeros(2, dtype=np.float32)

    def act(self, obs: Dict) -> np.ndarray:
        dW_v = self.rng.normal(0.0, self.sigma_v) * np.sqrt(self.dt)
        dW_w = self.rng.normal(0.0, self.sigma_w) * np.sqrt(self.dt)
        self.state[0] += -self.theta * self.state[0] * self.dt + dW_v
        self.state[1] += -self.theta * self.state[1] * self.dt + dW_w
        v = float(np.clip(self.state[0], -MAX_V_MPS, MAX_V_MPS))
        w = float(np.clip(self.state[1], -MAX_W_RADPS, MAX_W_RADPS))
        return np.array([v, w], dtype=np.float32)


class ScriptedPolicy(_BasePolicy):
    """
    PD controller that drives the robot toward a randomly-sampled waypoint.
    Re-samples the waypoint whenever the robot comes within ``goal_radius``.
    """

    def __init__(
        self,
        goal_radius: float = 0.5,
        map_half_size: float = 10.0,
        kp_lin: float = 0.8,
        kp_ang: float = 2.0,
    ) -> None:
        self.goal_radius = float(goal_radius)
        self.map_half_size = float(map_half_size)
        self.kp_lin = float(kp_lin)
        self.kp_ang = float(kp_ang)
        self.rng = np.random.default_rng()
        self.goal = np.zeros(2, dtype=np.float32)
        self._need_new_goal = True

    def reset(self, rng: Optional[np.random.Generator] = None) -> None:
        super().reset(rng)
        self._need_new_goal = True

    def _sample_goal(self) -> np.ndarray:
        R = self.map_half_size
        return self.rng.uniform(-R, R, size=2).astype(np.float32)

    def act(self, obs: Dict) -> np.ndarray:
        ego_xy = np.asarray(obs["ego_xy"], dtype=np.float32).reshape(2)
        yaw = float(obs["yaw"])
        if self._need_new_goal:
            self.goal = self._sample_goal()
            self._need_new_goal = False

        delta = self.goal - ego_xy
        dist = float(np.linalg.norm(delta))
        if dist < self.goal_radius:
            self._need_new_goal = True
            return np.zeros(2, dtype=np.float32)

        desired_yaw = float(np.arctan2(delta[1], delta[0]))
        err = _wrap_angle(desired_yaw - yaw)
        w = float(np.clip(self.kp_ang * err, -MAX_W_RADPS, MAX_W_RADPS))
        # Slow down when mis-pointed to avoid overshoot.
        v_gain = max(0.0, float(np.cos(err)))
        v = float(np.clip(self.kp_lin * dist * v_gain, 0.0, MAX_V_MPS))
        return np.array([v, w], dtype=np.float32)


class AdversarialPolicy(_BasePolicy):
    """
    Heads straight for the nearest obstacle centroid. The generator is
    responsible for bailing out as soon as ``contact_flag`` trips, so the
    rollout terminates in the first collision frame and the risk labels
    are dominated by genuine positives.
    """

    def __init__(self, kp_ang: float = 2.0, v_target: float = 1.0) -> None:
        self.kp_ang = float(kp_ang)
        self.v_target = float(v_target)
        self.rng = np.random.default_rng()

    def act(self, obs: Dict) -> np.ndarray:
        ego_xy = np.asarray(obs["ego_xy"], dtype=np.float32).reshape(2)
        yaw = float(obs["yaw"])
        obstacles = np.asarray(
            obs.get("obstacles_xy", np.zeros((0, 2), dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1, 2)

        if obstacles.shape[0] == 0:
            # No obstacles known -> just barrel forward with a bit of jitter.
            w = float(self.rng.uniform(-0.5, 0.5))
            return np.array([self.v_target, w], dtype=np.float32)

        d = np.linalg.norm(obstacles - ego_xy[None, :], axis=1)
        i = int(np.argmin(d))
        target = obstacles[i]
        delta = target - ego_xy
        desired_yaw = float(np.arctan2(delta[1], delta[0]))
        err = _wrap_angle(desired_yaw - yaw)
        w = float(np.clip(self.kp_ang * err, -MAX_W_RADPS, MAX_W_RADPS))
        v = float(self.v_target)
        return np.array([v, w], dtype=np.float32)


class StationaryPolicy(_BasePolicy):
    """
    Zero linear / angular command so the base stays put unless pushed by
    physics. Useful with **dynamic obstacles** in the scene: contact can
    still occur when an obstacle moves into the robot.
    """

    def act(self, obs: Dict) -> np.ndarray:  # noqa: ARG002 - API parity
        return np.array([0.0, 0.0], dtype=np.float32)


def _wrap_angle(a: float) -> float:
    """Wrap an angle to (-pi, pi]."""
    return float(np.arctan2(np.sin(a), np.cos(a)))
