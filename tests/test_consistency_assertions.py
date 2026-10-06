"""
Unit tests for physical consistency assertions.
Requirement 1d.(ii):
  PM in (0, 180]
  gain_db in [20, 100]
  UGBW > f3dB
  UGBW within one decade of f3dB*10^(gain_db/20)
  every saturation boolean equal to its own Vds > Vdsat comparison
  Pdc > 0 and within 50% of VDD*|I(Vdd)|
  a violated assertion marks the result invalid (maximally penalized)
"""

import unittest
from cadence.psf_parser import check_physical_consistency


class TestConsistencyAssertions(unittest.TestCase):
    def setUp(self):
        # Valid baseline result
        self.valid_sat_dict = {
            f"M{i}": (0.50, 0.15, True) for i in range(1, 9)
        }
        self.base_params = {
            "gain_db": 50.0,
            "ugbw": 10.0e6,      # 10 MHz
            "pm": 60.0,          # 60 deg
            "f3db": 30.0e3,      # 30 kHz
            "pdc": 1.8 * 60e-6,  # 108 uW
            "i_vdd": 60e-6,
            "sat_dict": self.valid_sat_dict,
        }

    def test_valid_baseline_passes(self):
        is_valid, reasons = check_physical_consistency(**self.base_params)
        self.assertTrue(is_valid)
        self.assertEqual(len(reasons), 0)

    def test_pm_out_of_bounds(self):
        # PM <= 0
        p = dict(self.base_params, pm=-5.0)
        is_valid, reasons = check_physical_consistency(**p)
        self.assertFalse(is_valid)
        self.assertTrue(any("PM" in r for r in reasons))

        # PM > 180
        p = dict(self.base_params, pm=185.0)
        is_valid, reasons = check_physical_consistency(**p)
        self.assertFalse(is_valid)
        self.assertTrue(any("PM" in r for r in reasons))

    def test_gain_out_of_bounds(self):
        # gain < 20 dB
        p = dict(self.base_params, gain_db=15.0)
        is_valid, reasons = check_physical_consistency(**p)
        self.assertFalse(is_valid)
        self.assertTrue(any("gain_db" in r for r in reasons))

        # gain > 100 dB
        p = dict(self.base_params, gain_db=105.0)
        is_valid, reasons = check_physical_consistency(**p)
        self.assertFalse(is_valid)
        self.assertTrue(any("gain_db" in r for r in reasons))

    def test_ugbw_less_than_f3db(self):
        p = dict(self.base_params, ugbw=10e3, f3db=50e3)
        is_valid, reasons = check_physical_consistency(**p)
        self.assertFalse(is_valid)
        self.assertTrue(any("UGBW" in r for r in reasons))

    def test_ugbw_outside_one_decade_prediction(self):
        # predicted = f3db * 10^(gain_db/20) = 30e3 * 10^2.5 = 30e3 * 316.2 = 9.486 MHz
        # 1 decade range: [0.9486 MHz, 94.86 MHz]
        # Test 100 kHz (< 0.1x predicted)
        p = dict(self.base_params, ugbw=100.0e3)
        is_valid, reasons = check_physical_consistency(**p)
        self.assertFalse(is_valid)
        self.assertTrue(any("decade" in r for r in reasons))

        # Test 200 MHz (> 10x predicted)
        p = dict(self.base_params, ugbw=200.0e6)
        is_valid, reasons = check_physical_consistency(**p)
        self.assertFalse(is_valid)
        self.assertTrue(any("decade" in r for r in reasons))

    def test_saturation_boolean_mismatch(self):
        # Inconsistent boolean (reported as True but Vds < Vdsat)
        bad_sat = dict(self.valid_sat_dict)
        bad_sat["M7"] = (0.02, 0.15, True)  # Vds < Vdsat, but boolean is True
        p = dict(self.base_params, sat_dict=bad_sat)
        is_valid, reasons = check_physical_consistency(**p)
        self.assertFalse(is_valid)
        self.assertTrue(any("Saturation boolean" in r for r in reasons))

    def test_pdc_out_of_bounds(self):
        # Pdc <= 0
        p = dict(self.base_params, pdc=-1e-4)
        is_valid, reasons = check_physical_consistency(**p)
        self.assertFalse(is_valid)
        self.assertTrue(any("Pdc" in r for r in reasons))

        # Pdc differs by > 50% from VDD * |I(Vdd)|
        p = dict(self.base_params, pdc=1.8 * 60e-6 * 2.0)  # 100% higher
        is_valid, reasons = check_physical_consistency(**p)
        self.assertFalse(is_valid)
        self.assertTrue(any("50%" in r for r in reasons))


if __name__ == "__main__":
    unittest.main()
