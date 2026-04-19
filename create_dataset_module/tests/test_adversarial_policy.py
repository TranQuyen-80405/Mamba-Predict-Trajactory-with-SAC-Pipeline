"""Unit tests for AdversarialPolicy obstacle selection."""

from __future__ import annotations

import numpy as np

from create_dataset_module.policies import AdversarialPolicy


def test_prefers_closest_obstacle_in_forward_hemisphere() -> None:
    """Nearest Euclidean can lie behind the robot; policy should target ahead first."""
    p = AdversarialPolicy()
    p.reset(np.random.default_rng(0))
    # Face +x; far obstacle ahead (4,0), closer behind (-2, 0)
    obs = {
        "ego_xy": np.array([0.0, 0.0], dtype=np.float32),
        "yaw": 0.0,
        "obstacles_xy": np.array([[-2.0, 0.0], [4.0, 0.0]], dtype=np.float32),
    }
    a = p.act(obs)
    # Target should be (4,0): small turn, forward v > 0
    assert a[0] > 0.5
    assert abs(a[1]) < 0.3


def test_falls_back_to_global_nearest_when_none_ahead() -> None:
    p = AdversarialPolicy()
    p.reset(np.random.default_rng(1))
    # Face +x; both obstacles behind (negative x) — pick global nearest.
    obs = {
        "ego_xy": np.array([0.0, 0.0], dtype=np.float32),
        "yaw": 0.0,
        "obstacles_xy": np.array([[-1.0, 0.0], [-3.0, 0.1]], dtype=np.float32),
    }
    a = p.act(obs)
    assert a[0] > 0.5
    assert np.isfinite(a[1])
    assert abs(a[1]) <= 2.5 + 1e-3
