#!/usr/bin/env python3
"""
Stage 3 Execution Script: PPO Training.
Replication of Papageorgiou et al., AEU 2025 (doi: 10.1016/j.aeue.2025.155697).

Runs PPO training with the paper's exact setup:
  - 16-dim observation state
  - MultiDiscrete([3,3,3,3,3]) action space
  - Paper's gated reward function with overshoot incentive
  - SB3 PPO with MlpPolicy (two tanh 64x64 layers, 15 action outputs)
  - 20,000 steps initial budget; if success rate < 80%, 10k increments up to 50k
  - Outputs saved to results/180nm/<timestamp>/
"""

import os
import sys
import glob
import logging
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl.train_ppo import train_ppo_agent


def setup_logging(log_file: str):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    file_h = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_h.setFormatter(formatter)
    stream_h = logging.StreamHandler(sys.stdout)
    stream_h.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = [file_h, stream_h]


def find_latest_bounds_file() -> str:
    # First check symlink or newest directory in results/180nm/
    results_base = os.path.join(PROJECT_ROOT, "results", "180nm")
    latest_symlink = os.path.join(results_base, "latest", "narrowed_bounds_180nm.json")
    if os.path.exists(latest_symlink):
        return latest_symlink

    # Search pattern for narrowed_bounds_180nm.json
    pattern = os.path.join(results_base, "*", "narrowed_bounds_180nm.json")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if matches:
        return matches[0]

    raise FileNotFoundError("Could not find narrowed_bounds_180nm.json from Stage 2!")


import argparse


def main():
    parser = argparse.ArgumentParser(description="Stage 3: PPO Training")
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--finetune", type=int, default=0, help="Number of fine-tuning steps from latest checkpoint")
    args = parser.parse_args()

    log_file = os.path.join(PROJECT_ROOT, "logs", "stage3.log")
    setup_logging(log_file)
    logger = logging.getLogger("Stage3")

    logger.info("=" * 70)
    logger.info("STAGE 3: PPO REINFORCEMENT LEARNING TRAINING (180nm CMOS OpAmp)")
    logger.info("=" * 70)
    if args.finetune > 0:
        logger.info(f"Mode: FINE-TUNE ({args.finetune} steps)")
    elif args.resume:
        logger.info("Mode: RESUME enabled")

    bounds_file = find_latest_bounds_file()
    logger.info(f"Using Stage 2 narrowed bounds from: {bounds_file}")

    # Use the same run directory as Stage 2 (or create a unified timestamp directory)
    stage2_dir = os.path.dirname(bounds_file)
    results_dir = stage2_dir
    logger.info(f"Writing Stage 3 outputs to: {results_dir}")

    # Train or fine-tune PPO agent
    results = train_ppo_agent(
        results_dir=results_dir,
        bounds_file=bounds_file,
        total_timesteps=20000,
        seed=42,
        resume=(args.resume or args.finetune > 0),
        finetune_steps=args.finetune,
    )

    logger.info("=" * 70)
    logger.info("STAGE 3 TRAINING COMPLETE")
    logger.info(f"Model saved to: {results['model_path']}")
    logger.info(f"Total timesteps: {results['total_timesteps']}")
    logger.info(f"Final success rate: {results['final_success_rate']*100:.1f}%")
    logger.info(f"Training time: {results['training_time']:.1f} s")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
