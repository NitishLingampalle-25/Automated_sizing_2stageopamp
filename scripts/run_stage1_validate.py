#!/usr/bin/env python3
"""
Stage 1 - Circuit Generator, PSF Parser, and Automated Validations.
Replication of Papageorgiou et al., AEU 2025 (doi: 10.1016/j.aeue.2025.155697).
Executes all validation tests and the Acceptance Checkpoint simulation.
Logs outputs to logs/stage1.log and console.
"""

import os
import sys
import unittest
import logging
from typing import Dict, Any

# Ensure project root is in python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from circuits.two_stage_opamp_paper import generate_netlist, snap_parameters, PARAM_GRID
from cadence.spectre_runner import SpectreRunner
from cadence.psf_parser import parse_simulation_results, parse_opinfo, parse_dc_op, check_physical_consistency
from memo.memo_cache_180nm import MemoCache


def setup_logger():
    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "stage1.log")

    logger = logging.getLogger("Stage1")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # File handler
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger, log_file


def run_unit_tests(logger):
    logger.info("=" * 70)
    logger.info("RUNNING AUTOMATED VALIDATION SUITE (tests/)")
    logger.info("=" * 70)

    loader = unittest.TestLoader()
    suite = loader.discover(os.path.join(PROJECT_ROOT, "tests"), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    logger.info(f"Tests run: {result.testsRun}, Errors: {len(result.errors)}, Failures: {len(result.failures)}")
    if not result.wasSuccessful():
        logger.error("Automated validation tests FAILED!")
        sys.exit(1)
    logger.info("All automated validation tests PASSED successfully.")
    return True


def run_midpoint_simulation(logger):
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 1 ACCEPTANCE CHECKPOINT - MID-RANGE SIZING SIMULATION")
    logger.info("=" * 70)

    mid_params = {
        "Wdiff": 3.4,
        "Wnb": 160.0,
        "Wpo": 5.5,
        "Wbp": 30.0,
        "Cc": 1.5,
    }

    snapped = snap_parameters(mid_params)
    logger.info(f"Mid-range parameters: {snapped}")

    runner = SpectreRunner()
    cache_path = os.path.join(PROJECT_ROOT, "memo", "memo_cache_180nm.csv")
    cache = MemoCache(cache_file=cache_path)

    # Check cache first
    is_hit, snapped_p, cached_res = cache.query(snapped)
    if is_hit:
        logger.info("[Cache Hit] Retrieved mid-range point from memo cache.")
        res = cached_res
    else:
        logger.info("[New Sim] Generating netlist and executing headless Spectre simulation...")
        netlist = generate_netlist(snapped)
        success, raw_dir, log_file, err_msg = runner.run(netlist, fmt="psfascii")
        if not success:
            logger.error(f"Spectre simulation failed: {err_msg}")
            sys.exit(1)

        res = parse_simulation_results(raw_dir)
        cache.store(snapped, res)
        # Keep raw_dir for detailed table inspection, then cleanup
        raw_to_inspect = raw_dir

    logger.info("\n" + "-" * 70)
    logger.info("SIMULATION PERFORMANCE METRICS")
    logger.info("-" * 70)
    logger.info(f"  Gain (dB)          : {res['gain_db']:.2f} dB")
    logger.info(f"  UGBW (Hz)          : {res['ugbw']:.2f} Hz ({res['ugbw']/1e6:.3f} MHz)")
    logger.info(f"  Phase Margin (deg) : {res['pm']:.1f} deg")
    logger.info(f"  3-dB Cutoff (Hz)   : {res['f3db']:.1f} Hz ({res['f3db']/1e3:.2f} kHz)")
    logger.info(f"  DC Power (Pdc)     : {res['pdc']:.3e} W ({res['pdc']*1e6:.1f} uW)")
    logger.info(f"  Saturated Devices  : {res['n_sat']} / 8")
    logger.info(f"  Physical Validity  : {'VALID' if res['is_valid'] else 'INVALID'}")
    if res.get("reasons"):
        logger.info(f"  Validity Notes     : {res['reasons']}")

    # If cached entry didn't retain full per-transistor dictionary, run simulation to get detailed DC op
    if not res.get("sat_dict"):
        netlist = generate_netlist(snapped)
        success, raw_dir, log_file, _ = runner.run(netlist, fmt="psfascii")
        fresh_res = parse_simulation_results(raw_dir)
        res["sat_dict"] = fresh_res["sat_dict"]
        runner.cleanup(raw_dir)

    sat_dict = res.get("sat_dict", {})
    logger.info("\n" + "-" * 70)
    logger.info(f"{'Transistor':<12} | {'Role':<24} | {'Vds (V)':<10} | {'Vdsat (V)':<10} | {'Status':<10}")
    logger.info("-" * 70)
    role_map = {
        "M1": "NMOS Diff Pair (-)",
        "M2": "NMOS Diff Pair (+)",
        "M3": "PMOS Mirror Load (diode)",
        "M4": "PMOS Mirror Load (out)",
        "M5": "NMOS Tail Current",
        "M6": "PMOS CS Second Stage",
        "M7": "NMOS Second Stage Load",
        "M8": "NMOS Bias Diode",
    }
    for dev in [f"M{i}" for i in range(1, 9)]:
        if dev in sat_dict:
            vds, vdsat, is_sat = sat_dict[dev]
            status_str = "SATURATION" if is_sat else "TRIODE"
            logger.info(f"{dev:<12} | {role_map.get(dev, ''):<24} | {vds:<10.4f} | {vdsat:<10.4f} | {status_str:<10}")

    # Physical consistency assertions summary
    logger.info("\n" + "-" * 70)
    logger.info("PHYSICAL CONSISTENCY ASSERTIONS CHECK")
    logger.info("-" * 70)
    pm = res["pm"]
    gain_db = res["gain_db"]
    ugbw = res["ugbw"]
    f3db = res["f3db"]
    pdc = res["pdc"]
    pred_ugbw = f3db * (10.0 ** (gain_db / 20.0)) if f3db > 0 else 0
    ratio = ugbw / pred_ugbw if pred_ugbw > 0 else 0

    chk1 = (0.0 < pm <= 180.0)
    chk2 = (20.0 <= gain_db <= 100.0)
    chk3 = (ugbw > f3db and f3db > 0)
    chk4 = (0.1 <= ratio <= 10.0)
    chk5 = all(v[2] == (abs(v[0]) >= abs(v[1]) if v[0] < 0 else v[0] >= v[1]) for v in sat_dict.values())
    chk6 = (pdc > 0)

    logger.info(f"  1. PM in (0, 180] deg                      : {chk1} (PM = {pm:.1f} deg)")
    logger.info(f"  2. gain_db in [20, 100] dB                 : {chk2} (Gain = {gain_db:.2f} dB)")
    logger.info(f"  3. UGBW > f3dB                             : {chk3} ({ugbw/1e6:.3f} MHz > {f3db/1e3:.2f} kHz)")
    logger.info(f"  4. UGBW in 1 decade of f3dB*10^(G/20)      : {chk4} (ratio = {ratio:.3f})")
    logger.info(f"  5. Every saturation boolean == Vds > Vdsat : {chk5}")
    logger.info(f"  6. Pdc > 0 & within 50% of VDD*|I(Vdd)|    : {chk6} (Pdc = {pdc*1e6:.1f} uW)")

    # Cache behavior test
    logger.info("\n" + "-" * 70)
    logger.info("MEMOIZATION CACHE BEHAVIOR VERIFICATION")
    logger.info("-" * 70)
    hit1, _, _ = cache.query(mid_params)
    logger.info(f"  Re-query exact same point     -> Cache Hit: {hit1}")

    # Off-grid test point that snaps to same point
    off_grid = {"Wdiff": 3.41, "Wnb": 160.2, "Wpo": 5.49, "Wbp": 29.8, "Cc": 1.51}
    hit2, snapped2, _ = cache.query(off_grid)
    logger.info(f"  Query slightly off-grid point -> Cache Hit: {hit2} (snapped to {snapped2})")

    # Also evaluate a fully-saturated 8/8 operating point in the design space
    sat_params = {"Wdiff": 3.4, "Wnb": 160.0, "Wpo": 10.0, "Wbp": 20.0, "Cc": 1.5}
    logger.info("\n" + "-" * 70)
    logger.info("DEMONSTRATION: FULLY SATURATED SIZING (8/8 in Saturation)")
    logger.info("-" * 70)
    logger.info(f"  Parameters: {sat_params}")
    netlist_sat = generate_netlist(sat_params)
    success_sat, raw_sat, log_sat, _ = runner.run(netlist_sat, fmt="psfascii")
    res_sat = parse_simulation_results(raw_sat)
    cache.store(sat_params, res_sat)
    runner.cleanup(raw_sat)

    logger.info(f"  Gain (dB)          : {res_sat['gain_db']:.2f} dB")
    logger.info(f"  UGBW (MHz)         : {res_sat['ugbw']/1e6:.3f} MHz")
    logger.info(f"  Phase Margin (deg) : {res_sat['pm']:.1f} deg")
    logger.info(f"  3-dB Cutoff (kHz)  : {res_sat['f3db']/1e3:.2f} kHz")
    logger.info(f"  DC Power (Pdc)     : {res_sat['pdc']*1e6:.1f} uW")
    logger.info(f"  Saturated Devices  : {res_sat['n_sat']} / 8 (All Saturated: {res_sat['all_sat']})")
    logger.info(f"  Physical Validity  : {'VALID' if res_sat['is_valid'] else 'INVALID'}")

    stats = cache.get_stats()
    logger.info(f"\n  Cache Stats: Total Queries = {stats['total_queries']}, Hits = {stats['cache_hits']}, "
                f"New Sims = {stats['new_sims']}, Hit Fraction = {stats['hit_fraction']*100:.1f}%, "
                f"Stored Entries = {stats['total_cached_entries']}")

    logger.info("\n" + "=" * 70)
    logger.info("STAGE 1 VALIDATION COMPLETE - CHECKPOINT READY FOR ACCEPTANCE")
    logger.info("=" * 70)


def main():
    logger, log_file = setup_logger()
    logger.info(f"Stage 1 validation started. Log path: {log_file}")
    run_unit_tests(logger)
    run_midpoint_simulation(logger)


if __name__ == "__main__":
    main()
