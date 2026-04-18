"""
Unit-test the lookahead risk-label generation against the exact
semantics stated in docs/strategy_full_pipeline.md § 5.1.

Pure numpy, no PyBullet.
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from create_dataset_module.generator import lookahead_any  # noqa: E402


class TestLookaheadAny(unittest.TestCase):
    def test_single_contact_propagates_backward(self):
        T = 110
        contact = np.zeros(T, dtype=np.bool_)
        contact[100] = True

        r05 = lookahead_any(contact, 10)   # 0.5s @ 20 Hz
        r10 = lookahead_any(contact, 20)
        r20 = lookahead_any(contact, 40)

        # r05[t]==1 iff t in [91, 100]
        for t in range(91, 101):
            self.assertEqual(r05[t], 1.0, f"expect r05[{t}]=1")
        for t in list(range(0, 91)) + list(range(101, T)):
            self.assertEqual(r05[t], 0.0, f"expect r05[{t}]=0")

        for t in range(81, 101):
            self.assertEqual(r10[t], 1.0)
        for t in list(range(0, 81)) + list(range(101, T)):
            self.assertEqual(r10[t], 0.0)

        for t in range(61, 101):
            self.assertEqual(r20[t], 1.0)
        for t in list(range(0, 61)) + list(range(101, T)):
            self.assertEqual(r20[t], 0.0)

    def test_no_contact_gives_all_zero(self):
        r = lookahead_any(np.zeros(50, dtype=np.bool_), 20)
        self.assertEqual(float(r.sum()), 0.0)

    def test_contact_at_start(self):
        contact = np.zeros(20, dtype=np.bool_)
        contact[0] = True
        r = lookahead_any(contact, 10)
        self.assertEqual(r[0], 1.0)
        # Lookahead window is forward-only - nothing before the contact
        # gets flagged here because there IS nothing before index 0.
        # Index 1..10 should NOT be flagged unless they themselves contact.
        self.assertEqual(float(r[1:].sum()), 0.0)

    def test_near_end_window_shrinks(self):
        # Contact at the very last index -> only that index is 1 for
        # any horizon (window shrinks as it goes past T).
        T = 50
        contact = np.zeros(T, dtype=np.bool_)
        contact[-1] = True
        r = lookahead_any(contact, 40)
        # r[t]=1 for t in [10, 49].
        for t in range(10, T):
            self.assertEqual(r[t], 1.0)
        for t in range(0, 10):
            self.assertEqual(r[t], 0.0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
