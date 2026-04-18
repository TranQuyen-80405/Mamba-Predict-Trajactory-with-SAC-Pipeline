"""StationaryPolicy outputs zero commands."""
from __future__ import annotations

import unittest

from create_dataset_module.policies import StationaryPolicy


class TestStationaryPolicy(unittest.TestCase):
    def test_act_is_zero(self) -> None:
        p = StationaryPolicy()
        p.reset()
        a = p.act({"ego_xy": [0.0, 0.0], "yaw": 0.0, "obstacles_xy": []})
        self.assertEqual(a.shape, (2,))
        self.assertAlmostEqual(float(a[0]), 0.0)
        self.assertAlmostEqual(float(a[1]), 0.0)


if __name__ == "__main__":
    unittest.main()
