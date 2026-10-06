"""
Unit tests for discrete parameter grid snapping.
"""

import unittest
from circuits.two_stage_opamp_paper import snap_parameters, snap_value, PARAM_GRID


class TestGridSnapping(unittest.TestCase):
    def test_wdiff_snapping(self):
        # Wdiff in [0.4, 6.4], step 0.2
        self.assertEqual(snap_value(0.41, 0.4, 6.4, 0.2), 0.4)
        self.assertEqual(snap_value(0.51, 0.4, 6.4, 0.2), 0.6)
        self.assertEqual(snap_value(3.39, 0.4, 6.4, 0.2), 3.4)
        self.assertEqual(snap_value(0.1, 0.4, 6.4, 0.2), 0.4)   # clamped low
        self.assertEqual(snap_value(7.0, 0.4, 6.4, 0.2), 6.4)   # clamped high

    def test_wnb_snapping(self):
        # Wnb in [145, 175], step 1.0
        self.assertEqual(snap_value(145.4, 145.0, 175.0, 1.0), 145.0)
        self.assertEqual(snap_value(159.7, 145.0, 175.0, 1.0), 160.0)
        self.assertEqual(snap_value(140.0, 145.0, 175.0, 1.0), 145.0)  # clamped low
        self.assertEqual(snap_value(180.0, 145.0, 175.0, 1.0), 175.0)  # clamped high

    def test_wpo_snapping(self):
        # Wpo in [0.5, 10.5], step 0.5
        self.assertEqual(snap_value(0.6, 0.5, 10.5, 0.5), 0.5)
        self.assertEqual(snap_value(0.8, 0.5, 10.5, 0.5), 1.0)
        self.assertEqual(snap_value(5.3, 0.5, 10.5, 0.5), 5.5)

    def test_wbp_snapping(self):
        # Wbp in [20, 40], step 1.0
        self.assertEqual(snap_value(20.3, 20.0, 40.0, 1.0), 20.0)
        self.assertEqual(snap_value(29.8, 20.0, 40.0, 1.0), 30.0)

    def test_cc_snapping(self):
        # Cc in [0.1, 3.0], step 0.1
        self.assertEqual(snap_value(0.12, 0.1, 3.0, 0.1), 0.1)
        self.assertEqual(snap_value(1.54, 0.1, 3.0, 0.1), 1.5)
        self.assertEqual(snap_value(1.56, 0.1, 3.0, 0.1), 1.6)

    def test_full_dict_snapping(self):
        raw = {
            "Wdiff": 3.42,
            "Wnb": 159.9,
            "Wpo": 5.48,
            "Wbp": 29.9,
            "Cc": 1.54,
        }
        expected = {
            "Wdiff": 3.4,
            "Wnb": 160.0,
            "Wpo": 5.5,
            "Wbp": 30.0,
            "Cc": 1.5,
        }
        self.assertEqual(snap_parameters(raw), expected)


if __name__ == "__main__":
    unittest.main()
