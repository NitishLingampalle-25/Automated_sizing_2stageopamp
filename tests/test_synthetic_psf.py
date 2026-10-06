"""
Synthetic ground-truth tests for PSFASCII parser.
Validates extraction of single-pole response parameters against analytical values.
Requirement 1d.(i): assert agreement to 1% on frequencies and 0.5 deg on PM.
"""

import os
import math
import shutil
import tempfile
import unittest
import numpy as np
from cadence.psf_parser import parse_ac_sweep, parse_opinfo, parse_dc_op, parse_simulation_results


class TestSyntheticPSF(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="synth_psf_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def generate_synthetic_ac_file(self, filename: str, a0: float, fp: float):
        """
        Generates synthetic acSwp.ac for H(s) = a0 / (1 + j * f / fp).
        Frequency swept from 1 Hz to 100 MHz with 20 points per decade.
        """
        freqs = np.logspace(0, 8, 161)
        with open(filename, "w") as f:
            f.write('HEADER\n"PSFversion" "1.00"\n"analysis name" "acSwp"\n')
            f.write('TYPE\n"sweep" FLOAT DOUBLE PROP("key" "sweep")\n')
            f.write('"V" COMPLEX DOUBLE PROP("key" "node")\n')
            f.write('SWEEP\n"freq" "sweep"\n')
            f.write('TRACE\n"vout" "V"\n')
            f.write('VALUE\n')
            for f_val in freqs:
                # Analytical single-pole response
                denom = 1.0 + 1j * (f_val / fp)
                h = a0 / denom
                f.write(f'"freq" {f_val:.16e}\n')
                f.write(f'"vout" ({h.real:.16e} {h.imag:.16e})\n')
            f.write('END\n')

    def generate_synthetic_dc_files(self, opinfo_file: str, dcop_file: str):
        """
        Generates synthetic opInfo.info and dcOp.dc with known voltages and currents.
        """
        with open(opinfo_file, "w") as f:
            f.write('HEADER\n"PSFversion" "1.00"\n')
            f.write('TYPE\n"bsim3v3" STRUCT(\n"trise" FLOAT DOUBLE\n"ids" FLOAT DOUBLE\n"isub" FLOAT DOUBLE\n')
            f.write('"vgs" FLOAT DOUBLE\n"vds" FLOAT DOUBLE\n"vbs" FLOAT DOUBLE\n"vgb" FLOAT DOUBLE\n')
            f.write('"vdb" FLOAT DOUBLE\n"vgd" FLOAT DOUBLE\n"vth" FLOAT DOUBLE\n"vdsat" FLOAT DOUBLE\n')
            f.write('"vfbeff" FLOAT DOUBLE\n"gm" FLOAT DOUBLE\n) PROP("key" "inst")\nVALUE\n')
            
            # Devices: M1-M7 in saturation, M8 in saturation
            for i in range(1, 9):
                vds = 0.50
                vdsat = 0.15
                f.write(f'"{f"M{i}"}" "bsim3v3" (\n')
                f.write('0.0\n1.5e-5\n0.0\n0.6\n')
                f.write(f'{vds}\n0.0\n0.6\n0.5\n0.1\n0.45\n{vdsat}\n0.0\n2e-4\n)\n')
            f.write('END\n')

        with open(dcop_file, "w") as f:
            f.write('HEADER\n"PSFversion" "1.00"\n')
            f.write('VALUE\n')
            f.write('"vdd" "V" 9.000000000000000e-01\n')
            f.write('"vss" "V" -9.000000000000000e-01\n')
            f.write('"vout" "V" 0.000000000000000e+00\n')
            f.write('"Vdd:p" "I" -6.000000000000000e-05\n')
            f.write('"Vss:p" "I" 6.000000000000000e-05\n')
            f.write('END\n')

    def test_single_pole_frequency_and_pm_accuracy(self):
        """
        Tests single pole response with known parameters:
          A0 = 100.0 (40.0 dB)
          fp = 10,000.0 Hz (10 kHz)
        Theoretical values:
          Gain = 40.0 dB
          f3dB = 10,000.0 Hz
          UGBW = fp * sqrt(A0^2 - 1) = 10000 * sqrt(9999) = 999,949.998 Hz
          PM = 180 - arctan(sqrt(A0^2 - 1)) = 180 - 89.427 = 90.573 deg
        """
        a0 = 100.0
        fp = 10000.0
        expected_gain_db = 20.0 * math.log10(a0)  # 40.0 dB
        expected_f3db = fp                        # 10,000.0 Hz
        expected_ugbw = fp * math.sqrt(a0**2 - 1.0)
        expected_pm = 180.0 - math.degrees(math.atan(math.sqrt(a0**2 - 1.0)))

        ac_file = os.path.join(self.test_dir, "acSwp.ac")
        self.generate_synthetic_ac_file(ac_file, a0, fp)

        gain_db, ugbw, pm, f3db = parse_ac_sweep(ac_file)

        # Gain agreement: within 0.1 dB
        self.assertAlmostEqual(gain_db, expected_gain_db, delta=0.1)

        # f3dB agreement: within 1%
        f3db_err_pct = abs(f3db - expected_f3db) / expected_f3db * 100.0
        self.assertLess(f3db_err_pct, 1.0, f"f3dB error {f3db_err_pct:.2f}% exceeds 1% limit")

        # UGBW agreement: within 1%
        ugbw_err_pct = abs(ugbw - expected_ugbw) / expected_ugbw * 100.0
        self.assertLess(ugbw_err_pct, 1.0, f"UGBW error {ugbw_err_pct:.2f}% exceeds 1% limit")

        # PM agreement: within 0.5 degrees
        pm_err = abs(pm - expected_pm)
        self.assertLess(pm_err, 0.5, f"PM error {pm_err:.3f} deg exceeds 0.5 deg limit")

    def test_synthetic_dc_operating_point(self):
        """
        Tests parsing of known synthetic DC operating point file.
        """
        opinfo_file = os.path.join(self.test_dir, "opInfo.info")
        dcop_file = os.path.join(self.test_dir, "dcOp.dc")
        self.generate_synthetic_dc_files(opinfo_file, dcop_file)

        sat_dict = parse_opinfo(opinfo_file)
        self.assertEqual(len(sat_dict), 8)
        for m, (vds, vdsat, is_sat) in sat_dict.items():
            self.assertAlmostEqual(vds, 0.50, places=4)
            self.assertAlmostEqual(vdsat, 0.15, places=4)
            self.assertTrue(is_sat)

        dc_data = parse_dc_op(dcop_file)
        self.assertAlmostEqual(abs(dc_data["Vdd:p"]), 6.0e-5, delta=1e-8)
        self.assertAlmostEqual(dc_data["vdd"], 0.9, places=2)
        self.assertAlmostEqual(dc_data["vss"], -0.9, places=2)


if __name__ == "__main__":
    unittest.main()
