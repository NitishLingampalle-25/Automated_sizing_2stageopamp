"""
Gymnasium Environment for OpAmp Reinforcement Learning.
Replication of Papageorgiou et al., AEU 2025 (doi: 10.1016/j.aeue.2025.155697).

State: 5 normalized parameter values + 8 transistor saturation booleans + 3 normalized metrics (Gain, UGBW, PM).
Action: MultiDiscrete([3, 3, 3, 3, 3]) -> {-1, 0, +1} step per parameter.
Episode: Max 20 steps.
Reward: Paper's gated reward with overshoot bonus.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Tuple, Optional
import numpy as np

import gymnasium as gym
from gymnasium import spaces

import fcntl
from circuits.two_stage_opamp_paper import generate_netlist, snap_parameters, snap_value, PARAM_GRID
from cadence.spectre_runner import SpectreRunner
from cadence.psf_parser import parse_simulation_results
from memo.memo_cache_180nm import MemoCache, compute_param_md5

logger = logging.getLogger("OpAmpEnv")

# Paper's training-gate target specifications
TARGET_SPECS = {
    "gain_db": 30.0,       # >= 30 dB
    "ugbw": 1.0e6,         # >= 1 MHz
    "pm": 45.0,            # >= 45 deg
    "n_unsat": 0,          # all 8 transistors in saturation
}

PARAM_NAMES = ["Wdiff", "Wnb", "Wpo", "Wbp", "Cc"]


class OpAmpPaperEnv(gym.Env):
    """
    Gymnasium environment representing the 180nm two-stage OpAmp sizing MDP.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        bounds_file: Optional[str] = None,
        max_steps: int = 20,
        runner: Optional[SpectreRunner] = None,
        cache: Optional[MemoCache] = None,
    ):
        super().__init__()
        self.max_steps = max_steps
        self.runner = runner if runner is not None else SpectreRunner()
        self.cache = cache if cache is not None else MemoCache()

        # Load narrowed bounds if file provided, otherwise use default ranges
        self.bounds = self._load_bounds(bounds_file)

        # Action space: MultiDiscrete([3, 3, 3, 3, 3]) -> 0: -step, 1: stay, 2: +step
        self.action_space = spaces.MultiDiscrete([3, 3, 3, 3, 3])

        # Observation space: 16 dimensions
        # [5 normalized params, 8 saturation booleans (0 or 1), 3 normalized metrics (gain, ugbw, pm)]
        self.observation_space = spaces.Box(
            low=-2.0,
            high=5.0,
            shape=(16,),
            dtype=np.float32,
        )

        # State tracking
        self.current_params: Dict[str, float] = {}
        self.current_metrics: Dict[str, Any] = {}
        self.step_count = 0
        self.episode_count = 0

        # Performance tracking stats
        self.total_env_steps = 0
        self.total_simulations = 0
        self.total_cache_hits = 0
        self.episode_simulations = 0
        self.episode_cache_hits = 0

    def _load_bounds(self, bounds_file: Optional[str]) -> Dict[str, Tuple[float, float]]:
        """Loads narrowed bounds from JSON or falls back to full PARAM_GRID."""
        if bounds_file and os.path.exists(bounds_file):
            try:
                with open(bounds_file, "r") as f:
                    data = json.load(f)
                bounds = {}
                for k in PARAM_NAMES:
                    if k in data and len(data[k]) == 2:
                        bounds[k] = (float(data[k][0]), float(data[k][1]))
                    else:
                        bounds[k] = (PARAM_GRID[k][0], PARAM_GRID[k][1])
                logger.info(f"Loaded narrowed bounds from {bounds_file}: {bounds}")
                return bounds
            except Exception as e:
                logger.warning(f"Error reading bounds file {bounds_file}: {e}")

        # Fallback to default paper grid ranges
        return {k: (PARAM_GRID[k][0], PARAM_GRID[k][1]) for k in PARAM_NAMES}

    def _get_obs(self) -> np.ndarray:
        """Constructs the 16-dimensional observation vector."""
        obs = np.zeros(16, dtype=np.float32)

        # 1. 5 Normalized parameters in [0, 1] relative to current bounds
        for i, k in enumerate(PARAM_NAMES):
            low, high = self.bounds[k]
            span = max(high - low, 1e-6)
            obs[i] = float(np.clip((self.current_params[k] - low) / span, 0.0, 1.0))

        # 2. 8 Saturation booleans for M1..M8
        sat_dict = self.current_metrics.get("sat_dict", {})
        for i, dev in enumerate([f"M{d}" for d in range(1, 9)]):
            is_sat = sat_dict.get(dev, (0, 0, False))[2]
            obs[5 + i] = 1.0 if is_sat else 0.0

        # 3. 3 Normalized metrics: Gain, UGBW, PM
        gain_db = self.current_metrics.get("gain_db", 0.0)
        ugbw = self.current_metrics.get("ugbw", 0.0)
        pm = self.current_metrics.get("pm", 0.0)

        obs[13] = float(gain_db / 100.0)       # 0..1 range for 0..100 dB
        obs[14] = float(ugbw / 20.0e6)         # 0..1 range for 0..20 MHz
        obs[15] = float(pm / 180.0)            # 0..1 range for 0..180 deg

        return obs

    def _evaluate(self, params: Dict[str, float]) -> Dict[str, Any]:
        """Evaluates circuit through memo cache + Spectre runner with atomic deduplication."""
        is_hit, snapped, cached_metrics = self.cache.query(params)
        if is_hit and cached_metrics is not None:
            self.total_cache_hits += 1
            self.episode_cache_hits += 1
            return cached_metrics

        # Point was not in local cache; acquire atomic lock on this design point
        key = compute_param_md5(snapped)
        lock_dir = os.path.join(os.path.dirname(self.cache.cache_file), "locks")
        os.makedirs(lock_dir, exist_ok=True)
        lock_file = os.path.join(lock_dir, f"{key}.lock")
        with open(lock_file, "w") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                # Re-check persistent cache under lock
                self.cache._load_cache()
                if key in self.cache.cache:
                    self.total_cache_hits += 1
                    self.episode_cache_hits += 1
                    metrics = self.cache.cache[key]
                else:
                    self.total_simulations += 1
                    self.episode_simulations += 1
                    netlist = generate_netlist(snapped)
                    success, raw_dir, log_file, err_msg = self.runner.run(netlist, fmt="psfascii")
                    if success:
                        metrics = parse_simulation_results(raw_dir)
                        self.runner.cleanup(raw_dir)
                    else:
                        metrics = {
                            "gain_db": 0.0,
                            "ugbw": 0.0,
                            "pm": 0.0,
                            "f3db": 0.0,
                            "pdc": 1e-3,
                            "n_sat": 0,
                            "all_sat": False,
                            "sat_dict": {},
                            "is_valid": False,
                            "reasons": [err_msg],
                        }
                    self.cache.store(snapped, metrics)
            finally:
                fcntl.flock(lf.fileno(), fcntl.LOCK_UN)
                try:
                    os.remove(lock_file)
                except OSError:
                    pass

        return metrics

    def _compute_reward(self, metrics: Dict[str, Any]) -> Tuple[float, bool]:
        """
        Computes the paper's exact gated reward function:
          Training-gate targets: Gain >= 30 dB, UGB >= 1 MHz, PM >= 45 deg, n_unsat == 0
          specs unmet: r = - sum_i max(0, target_i - achieved_i) - 5 * n_unsat - 100
          specs met:   r = sum_i |target_i - achieved_i| + 0.01 (rewards overshoot)
        """
        gain_db = metrics.get("gain_db", 0.0)
        ugbw = metrics.get("ugbw", 0.0)
        pm = metrics.get("pm", 0.0)
        n_sat = metrics.get("n_sat", 0)
        n_unsat = max(0, 8 - n_sat)

        # Normalization divisors for dimensionless sum
        g_t, g_scale = TARGET_SPECS["gain_db"], 30.0
        u_t, u_scale = TARGET_SPECS["ugbw"], 1.0e6
        p_t, p_scale = TARGET_SPECS["pm"], 45.0

        specs_met = (
            gain_db >= g_t and
            ugbw >= u_t and
            pm >= p_t and
            n_unsat == 0
        )

        if not specs_met:
            # Normalized deficits
            def_gain = max(0.0, g_t - gain_db) / g_scale
            def_ugbw = max(0.0, u_t - ubw if (ubw := ugbw) is not None else 0.0) / u_scale
            def_pm   = max(0.0, p_t - pm) / p_scale
            sum_deficit = def_gain + def_ugbw + def_pm

            # r = - sum_deficits - 5 * n_unsat - 100
            reward = float(-sum_deficit - 5.0 * n_unsat - 100.0)
        else:
            # Overshoot reward (positive incentive for exceeding targets)
            os_gain = abs(gain_db - g_t) / g_scale
            os_ugbw = abs(ugbw - u_t) / u_scale
            os_pm   = abs(pm - p_t) / p_scale
            sum_overshoot = os_gain + os_ugbw + os_pm

            # r = sum_overshoot + 0.01
            reward = float(sum_overshoot + 0.01)

        return reward, specs_met

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self.step_count = 0
        self.episode_count += 1
        self.episode_simulations = 0
        self.episode_cache_hits = 0

        # Sample random point inside narrowed bounds on discrete grid
        self.current_params = {}
        for k in PARAM_NAMES:
            low, high = self.bounds[k]
            step = PARAM_GRID[k][2]
            n_steps = int(round((high - low) / step))
            if n_steps > 0:
                rand_step = self.np_random.integers(0, n_steps + 1)
                val = low + rand_step * step
            else:
                val = low
            self.current_params[k] = snap_value(val, low, high, step)

        self.current_params = snap_parameters(self.current_params)
        self.current_metrics = self._evaluate(self.current_params)

        obs = self._get_obs()
        info = {
            "params": dict(self.current_params),
            "metrics": dict(self.current_metrics),
            "step": self.step_count,
            "episode_sims": self.episode_simulations,
            "episode_hits": self.episode_cache_hits,
            "total_sims": self.total_simulations,
            "total_hits": self.total_cache_hits,
        }
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.step_count += 1
        self.total_env_steps += 1

        # Action: array of 5 integers in {0, 1, 2} -> {-1, 0, +1}
        action_deltas = np.array(action, dtype=int) - 1

        # Apply deltas with boundary clamping and snapping
        for i, k in enumerate(PARAM_NAMES):
            step_size = PARAM_GRID[k][2]
            delta = action_deltas[i] * step_size
            new_val = self.current_params[k] + delta
            low, high = self.bounds[k]
            self.current_params[k] = snap_value(new_val, low, high, step_size)

        self.current_params = snap_parameters(self.current_params)
        self.current_metrics = self._evaluate(self.current_params)

        reward, specs_met = self._compute_reward(self.current_metrics)

        # Terminate if training gate specs met; truncate at max_steps
        terminated = bool(specs_met)
        truncated = bool(self.step_count >= self.max_steps)

        obs = self._get_obs()
        info = {
            "params": dict(self.current_params),
            "metrics": dict(self.current_metrics),
            "specs_met": specs_met,
            "step": self.step_count,
            "reward": reward,
            "episode_sims": self.episode_simulations,
            "episode_hits": self.episode_cache_hits,
            "total_sims": self.total_simulations,
            "total_hits": self.total_cache_hits,
        }
        return obs, reward, terminated, truncated, info
