"""
PPO Training script for 180nm OpAmp automated sizing.
Replication of Papageorgiou et al., AEU 2025 (doi: 10.1016/j.aeue.2025.155697).

Agent: Stable-Baselines3 PPO with MlpPolicy:
  Two tanh dense layers, shared feature extractor, actor head (15 action outputs), critic value head.
Hyperparameters:
  gamma = 0.91, gae_lambda = 0.98, ent_coef = 0.09, vf_coef = 0.5
  learning_rate = 3e-4, n_steps = 256, seed = 42
Budget:
  20,000 steps initial budget; if success rate < 80%, increments of 10,000 up to 50,000 steps.
"""

import os
import sys
import csv
import json
import time
import logging
from typing import Dict, Any, List, Optional
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rl.env_paper import OpAmpPaperEnv
from cadence.spectre_runner import SpectreRunner
from memo.memo_cache_180nm import MemoCache

logger = logging.getLogger("TrainPPO")


class MetricsCallback(BaseCallback):
    """
    Tracks and logs per-episode statistics during PPO training:
    - Episode reward
    - Episode length
    - Cumulative / per-episode simulations vs cache hits
    - Cache-hit fraction
    - Success rate against training gate specs
    - Periodic saving every 500 steps and on exit
    """
    def __init__(
        self,
        log_csv_path: str,
        checkpoint_dir: Optional[str] = None,
        initial_episode: int = 0,
        initial_steps: int = 0,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.log_csv_path = log_csv_path
        self.checkpoint_dir = checkpoint_dir
        self.episode_offset = initial_episode
        self.last_checkpoint_step = initial_steps
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_successes: List[bool] = []
        self.episode_data: List[Dict[str, Any]] = []

        # Load existing episode records if resuming
        if os.path.exists(log_csv_path):
            try:
                with open(log_csv_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for r in reader:
                        self.episode_data.append({
                            "episode": int(r["episode"]),
                            "reward": float(r["reward"]),
                            "length": int(r["length"]),
                            "success": r["success"].lower() == "true",
                            "simulations": int(r.get("simulations", 0)),
                            "cache_hits": int(r.get("cache_hits", 0)),
                            "hit_fraction": float(r.get("hit_fraction", 0.0)),
                            "total_steps": int(r.get("total_steps", 0)),
                        })
                        self.episode_rewards.append(float(r["reward"]))
                        self.episode_lengths.append(int(r["length"]))
                        self.episode_successes.append(r["success"].lower() == "true")
            except Exception as e:
                logger.warning(f"Error loading existing episodes from {log_csv_path}: {e}")

        self.num_envs = None
        self.current_rewards: List[float] = []
        self.current_lens: List[int] = []
        self.current_successes: List[bool] = []

    def _save_checkpoint(self, step: int):
        if self.checkpoint_dir and self.model is not None:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(self.checkpoint_dir, "ppo_model.zip")
            self.model.save(ckpt_path)
            logger.info(f"Checkpoint saved at step {step} to {ckpt_path}")

    def _flush_csv(self):
        if self.log_csv_path and self.episode_data:
            os.makedirs(os.path.dirname(self.log_csv_path), exist_ok=True)
            with open(self.log_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.episode_data[0].keys()))
                writer.writeheader()
                writer.writerows(self.episode_data)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [0.0])
        dones = self.locals.get("dones", [False])

        if self.num_envs is None:
            self.num_envs = len(dones)
            self.current_rewards = [0.0] * self.num_envs
            self.current_lens = [0] * self.num_envs
            self.current_successes = [False] * self.num_envs

        # Periodic checkpoint every 500 steps
        if self.num_timesteps - self.last_checkpoint_step >= 500:
            self.last_checkpoint_step = self.num_timesteps
            self._save_checkpoint(self.num_timesteps)

        for i in range(self.num_envs):
            self.current_rewards[i] += float(rewards[i])
            self.current_lens[i] += 1

            if infos and i < len(infos):
                if infos[i].get("specs_met", False):
                    self.current_successes[i] = True

            if dones[i]:
                ep_idx = len(self.episode_data) + 1
                self.episode_rewards.append(self.current_rewards[i])
                self.episode_lengths.append(self.current_lens[i])
                self.episode_successes.append(self.current_successes[i])

                # Wire the real per-episode counters from environment worker
                ep_sims = 0
                ep_hits = 0
                if infos and i < len(infos):
                    ep_sims = int(infos[i].get("episode_sims", 0))
                    ep_hits = int(infos[i].get("episode_hits", 0))
                
                tot = ep_sims + ep_hits
                hit_frac = round(ep_hits / tot, 3) if tot > 0 else 0.0

                record = {
                    "episode": ep_idx,
                    "reward": round(self.current_rewards[i], 3),
                    "length": self.current_lens[i],
                    "success": self.current_successes[i],
                    "simulations": ep_sims,
                    "cache_hits": ep_hits,
                    "hit_fraction": hit_frac,
                    "total_steps": self.num_timesteps,
                }
                self.episode_data.append(record)

                if ep_idx % 10 == 0 or ep_idx == 1:
                    recent_succ = np.mean(self.episode_successes[-50:]) if self.episode_successes else 0.0
                    recent_rew = np.mean(self.episode_rewards[-20:])
                    recent_len = np.mean(self.episode_lengths[-20:])
                    logger.info(
                        f"[Episode {ep_idx:4d} | Step {self.num_timesteps:6d}] "
                        f"Rew: {recent_rew:7.2f} | Len: {recent_len:4.1f} | "
                        f"Recent Success: {recent_succ*100:5.1f}% | "
                        f"Sims: {record['simulations']} | Hits: {record['cache_hits']} ({record['hit_fraction']*100:.1f}%)"
                    )

                    # Periodically flush CSV so live data is always visible
                    self._flush_csv()

                # Reset episode accumulators for this env
                self.current_rewards[i] = 0.0
                self.current_lens[i] = 0
                self.current_successes[i] = False

        return True

    def _on_training_end(self):
        self._save_checkpoint(self.num_timesteps)
        self._flush_csv()
        logger.info(f"Saved training episode logs ({len(self.episode_data)} episodes) to {self.log_csv_path}")


def plot_fig6_training_progress(csv_path: str, save_path: str):
    """
    Plots two panels: mean reward per episode and mean episode length per episode (paper Fig. 6).
    """
    if not os.path.exists(csv_path):
        logger.warning(f"Cannot plot Fig. 6: {csv_path} not found.")
        return

    episodes, rewards, lengths = [], [], []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            episodes.append(int(r["episode"]))
            rewards.append(float(r["reward"]))
            lengths.append(float(r["length"]))

    if not episodes:
        return

    # Calculate rolling averages (window = 25 episodes)
    w = min(25, max(3, len(episodes) // 5))
    rolling_rew = np.convolve(rewards, np.ones(w)/w, mode="valid")
    rolling_len = np.convolve(lengths, np.ones(w)/w, mode="valid")
    valid_eps = episodes[w - 1:]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    fig.suptitle("Fig. 6 - PPO Agent Training Progress (180nm CMOS OpAmp)", fontsize=13, weight="bold")

    # Panel 1: Mean reward per episode
    ax1.plot(episodes, rewards, color="#4A90E2", alpha=0.3, label="Raw Episode Reward")
    ax1.plot(valid_eps, rolling_rew, color="#003366", linewidth=2.0, label=f"Rolling Mean (w={w})")
    ax1.set_ylabel("Episode Reward", fontsize=11, weight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right")

    # Panel 2: Mean episode length per episode
    ax2.plot(episodes, lengths, color="#E67E22", alpha=0.3, label="Raw Episode Length")
    ax2.plot(valid_eps, rolling_len, color="#B95C00", linewidth=2.0, label=f"Rolling Mean (w={w})")
    ax2.set_xlabel("Episode", fontsize=11, weight="bold")
    ax2.set_ylabel("Episode Length (Steps)", fontsize=11, weight="bold")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved Fig. 6 training progress plot to {save_path}")


def make_single_env(rank: int, b_file: Optional[str], s: int):
    def _init():
        r = SpectreRunner()
        c = MemoCache()
        e = OpAmpPaperEnv(
            bounds_file=b_file,
            max_steps=20,
            runner=r,
            cache=c,
        )
        return e
    return _init


def train_ppo_agent(
    results_dir: str,
    bounds_file: Optional[str] = None,
    total_timesteps: int = 20000,
    seed: int = 42,
    num_envs: Optional[int] = None,
    resume: bool = False,
) -> Dict[str, Any]:
    """
    Instantiates environment, sets up SB3 PPO model, and runs training with checkpoints.
    Uses SubprocVecEnv for multi-process simulation parallelism.
    """
    os.makedirs(results_dir, exist_ok=True)
    from stable_baselines3.common.vec_env import SubprocVecEnv
    import atexit

    if num_envs is None:
        num_envs = min(os.cpu_count() or 4, 8)

    logger.info(f"Creating SubprocVecEnv with {num_envs} parallel workers (bounds={bounds_file})...")
    env = SubprocVecEnv([make_single_env(i, bounds_file, seed) for i in range(num_envs)])

    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=[64, 64],
    )
    n_steps_per_env = max(16, 256 // num_envs)

    ep_log_path = os.path.join(results_dir, "training_episodes.csv")
    model_path = os.path.join(results_dir, "ppo_model.zip")

    initial_step = 0
    initial_episode = 0
    model = None

    if resume and os.path.exists(model_path):
        logger.info(f"Resuming training: loading model from checkpoint {model_path}...")
        model = PPO.load(
            model_path,
            env=env,
            seed=seed,
            policy_kwargs=policy_kwargs,
            gamma=0.91,
            gae_lambda=0.98,
            ent_coef=0.09,
            vf_coef=0.5,
            learning_rate=3e-4,
            n_steps=n_steps_per_env,
        )
        if os.path.exists(ep_log_path):
            try:
                with open(ep_log_path, "r", encoding="utf-8") as f:
                    rows = list(csv.DictReader(f))
                    if rows:
                        initial_step = int(rows[-1].get("total_steps", 0))
                        initial_episode = int(rows[-1].get("episode", len(rows)))
            except Exception as e:
                logger.warning(f"Error reading initial step from {ep_log_path}: {e}")
        logger.info(f"Resumed state: initial_step={initial_step}, initial_episode={initial_episode}")
    else:
        if resume:
            logger.warning(f"Resume requested but checkpoint {model_path} not found. Starting fresh from step 0.")
        logger.info(f"Initializing fresh PPO model with paper hyperparameters (n_steps_per_env={n_steps_per_env}, total_batch={n_steps_per_env*num_envs})...")
        model = PPO(
            policy="MlpPolicy",
            env=env,
            policy_kwargs=policy_kwargs,
            gamma=0.91,
            gae_lambda=0.98,
            ent_coef=0.09,
            vf_coef=0.5,
            learning_rate=3e-4,
            n_steps=n_steps_per_env,
            seed=seed,
            verbose=0,
        )

    callback = MetricsCallback(
        log_csv_path=ep_log_path,
        checkpoint_dir=results_dir,
        initial_episode=initial_episode,
        initial_steps=initial_step,
    )

    # atexit saving handler
    def save_on_exit():
        try:
            if model is not None:
                model.save(model_path)
                logger.info(f"[atexit] Saved checkpoint to {model_path}")
        except Exception:
            pass
    atexit.register(save_on_exit)

    remaining_steps = max(0, total_timesteps - initial_step)
    logger.info(f"Beginning PPO training: target={total_timesteps}, remaining={remaining_steps} timesteps...")
    start_time = time.time()
    if remaining_steps > 0:
        model.learn(total_timesteps=remaining_steps, callback=callback, reset_num_timesteps=False)
    training_time = time.time() - start_time
    logger.info(f"PPO training finished in {training_time:.1f} s.")

    last_50_successes = callback.episode_successes[-50:] if len(callback.episode_successes) >= 50 else callback.episode_successes
    final_success_rate = float(np.mean(last_50_successes)) if last_50_successes else 0.0
    logger.info(f"Training budget complete. Recent success rate: {final_success_rate*100:.1f}%")

    current_budget = total_timesteps
    while total_timesteps >= 20000 and final_success_rate < 0.80 and current_budget < 50000:
        increment = 10000
        logger.info(f"Success rate {final_success_rate*100:.1f}% < 80%. Extending budget by {increment} steps (current={current_budget})...")
        model.learn(total_timesteps=increment, callback=callback, reset_num_timesteps=False)
        current_budget += increment
        last_50_successes = callback.episode_successes[-50:] if len(callback.episode_successes) >= 50 else callback.episode_successes
        final_success_rate = float(np.mean(last_50_successes)) if last_50_successes else 0.0
        logger.info(f"Budget now {current_budget} steps. Updated success rate: {final_success_rate*100:.1f}%")

    model.save(model_path)
    logger.info(f"Saved trained PPO model to {model_path}")

    fig6_path = os.path.join(results_dir, "fig6_training_progress.png")
    plot_fig6_training_progress(ep_log_path, fig6_path)

    try:
        env.close()
    except Exception as e:
        logger.warning(f"Error closing environment: {e}")

    return {
        "model_path": model_path,
        "total_timesteps": current_budget,
        "final_success_rate": final_success_rate,
        "training_time": training_time,
        "results_dir": results_dir,
    }
