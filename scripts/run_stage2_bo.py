#!/usr/bin/env python3
"""
Stage 2 - Bayesian Optimization for Design Space Narrowing.
Replication of Papageorgiou et al., AEU 2025 (doi: 10.1016/j.aeue.2025.155697).
Executes 100-call GP-LCB optimization, minimizes non-saturated devices,
derives narrowed parameter bounds, and outputs all required artifacts.
Logs to logs/stage2.log and outputs to results/180nm/<timestamp>/.
"""

import os
import sys
import time
import logging
from typing import Dict, Any

# Ensure project root is in python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bo.bo_paper import execute_stage2_bo


def setup_logger():
    log_dir = os.path.join(PROJECT_ROOT, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "stage2.log")

    logger = logging.getLogger("BOPaper")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    return logger, log_file


def main():
    logger, log_file = setup_logger()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(PROJECT_ROOT, "results", "180nm", timestamp)

    logger.info("=" * 70)
    logger.info("STAGE 2 - BAYESIAN OPTIMIZATION DESIGN SPACE NARROWING")
    logger.info("=" * 70)
    logger.info(f"Start timestamp: {timestamp}")
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Log file: {log_file}")

    start_time = time.time()
    result = execute_stage2_bo(
        results_dir=results_dir,
        fixed_cc=1.5,
        n_calls=100,
        random_state=42,
    )
    elapsed = time.time() - start_time

    logger.info("\n" + "=" * 70)
    logger.info(f"STAGE 2 EXECUTION COMPLETE (Elapsed: {elapsed:.1f} s)")
    logger.info("=" * 70)
    logger.info(f"Status: {result.get('status')}")
    logger.info(f"Total Evaluations: {result.get('total_evaluations')}")
    logger.info(f"Feasible Designs (8/8 Saturated): {result.get('feasible_count')}")
    logger.info(f"Narrowed Bounds: {result.get('narrowed_bounds')}")

    # Check generated files
    expected_files = [
        "all_evaluations.csv",
        "narrowed_bounds_180nm.json",
        "fig5_parameter_distributions.png",
        "feasibility_scatter_matrix.png",
    ]
    logger.info("\nGenerated Artifacts:")
    for fn in expected_files:
        fp = os.path.join(results_dir, fn)
        if os.path.exists(fp):
            logger.info(f"  [OK] {fn} ({os.path.getsize(fp)} bytes)")
        else:
            logger.error(f"  [MISSING] {fn}")

    # Save a symlink or pointer 'latest' in results/180nm/
    latest_link = os.path.join(PROJECT_ROOT, "results", "180nm", "latest")
    try:
        if os.path.islink(latest_link) or os.path.exists(latest_link):
            os.remove(latest_link)
        os.symlink(results_dir, latest_link)
        logger.info(f"Updated symlink {latest_link} -> {results_dir}")
    except Exception as e:
        logger.warning(f"Could not update latest symlink: {e}")

    logger.info("=" * 70)


if __name__ == "__main__":
    main()
