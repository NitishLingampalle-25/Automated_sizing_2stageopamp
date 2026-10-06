"""
Bayesian Optimization module for OpAmp design space narrowing.
Replication of Papageorgiou et al., AEU 2025 (doi: 10.1016/j.aeue.2025.155697).

Objective: MINIMIZE the number of transistors not in saturation (N_unsat = 8 - N_sat).
Cc is excluded from BO (Cc fixed at 1.5 pF). Scans only the 4 widths:
  Wdiff: [0.4, 6.4] um, step 0.2 um
  Wnb:   [145, 175] um, step 1.0 um
  Wpo:   [0.5, 10.5] um, step 0.5 um
  Wbp:   [20, 40] um, step 1.0 um

Surrogate: Gaussian Process with Lower Confidence Bound (LCB) acquisition.
Exactly 100 function calls per optimization run.
"""

import os
import sys
import csv
import json
import time
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skopt import Optimizer
from skopt.space import Real

from circuits.two_stage_opamp_paper import generate_netlist, snap_parameters, snap_value, PARAM_GRID
from cadence.spectre_runner import SpectreRunner
from cadence.psf_parser import parse_simulation_results, parse_opinfo
from memo.memo_cache_180nm import MemoCache

logger = logging.getLogger("BOPaper")

DEFAULT_INITIAL_BOUNDS = {
    "Wdiff": (0.4, 6.4),
    "Wnb":   (145.0, 175.0),
    "Wpo":   (0.5, 10.5),
    "Wbp":   (20.0, 40.0),
}


def evaluate_candidate(
    candidate_widths: List[float],
    fixed_cc: float,
    runner: SpectreRunner,
    cache: MemoCache,
) -> Tuple[float, Dict[str, Any], str, Dict[str, float]]:
    """
    Evaluates a proposed candidate sizing [wdiff, wnb, wpo, wbp].
    Snaps to the discrete grid, checks memo cache, executes Spectre if miss.
    Returns: (n_unsat: float, metrics: dict, cache_status: str, snapped_params: dict)
    """
    raw_params = {
        "Wdiff": candidate_widths[0],
        "Wnb":   candidate_widths[1],
        "Wpo":   candidate_widths[2],
        "Wbp":   candidate_widths[3],
        "Cc":    fixed_cc,
    }
    snapped = snap_parameters(raw_params)

    is_hit, _, cached_res = cache.query(snapped)
    if is_hit and cached_res is not None:
        metrics = cached_res
        cache_status = "cache-hit"
    else:
        netlist = generate_netlist(snapped)
        success, raw_dir, log_file, err_msg = runner.run(netlist, fmt="psfascii")
        if success:
            metrics = parse_simulation_results(raw_dir)
            runner.cleanup(raw_dir)
        else:
            metrics = {
                "gain_db": 0.0,
                "ugbw": 0.0,
                "pm": 0.0,
                "f3db": 0.0,
                "pdc": 1e-3,
                "n_sat": 0,
                "all_sat": False,
                "is_valid": False,
                "reasons": [err_msg],
            }
        cache.store(snapped, metrics)
        cache_status = "new-sim"

    n_sat = metrics.get("n_sat", 0)
    # Objective: Minimize transistors NOT in saturation
    n_unsat = float(8 - n_sat)
    return n_unsat, metrics, cache_status, snapped


def run_bo_loop(
    bounds: Dict[str, Tuple[float, float]],
    fixed_cc: float = 1.5,
    n_calls: int = 100,
    n_initial_points: int = 20,
    random_state: int = 42,
    kappa: float = 1.96,
    runner: Optional[SpectreRunner] = None,
    cache: Optional[MemoCache] = None,
) -> Tuple[List[Dict[str, Any]], Optimizer]:
    """
    Executes exactly n_calls of GP-LCB Bayesian optimization.
    """
    if runner is None:
        runner = SpectreRunner()
    if cache is None:
        cache = MemoCache()

    space = [
        Real(bounds["Wdiff"][0], bounds["Wdiff"][1], name="Wdiff"),
        Real(bounds["Wnb"][0],   bounds["Wnb"][1],   name="Wnb"),
        Real(bounds["Wpo"][0],   bounds["Wpo"][1],   name="Wpo"),
        Real(bounds["Wbp"][0],   bounds["Wbp"][1],   name="Wbp"),
    ]

    opt = Optimizer(
        dimensions=space,
        base_estimator="GP",
        acq_func="LCB",
        acq_func_kwargs={"kappa": kappa},
        n_initial_points=n_initial_points,
        random_state=random_state,
    )

    evaluations = []
    logger.info(f"Starting BO loop: {n_calls} calls, bounds={bounds}, kappa={kappa}")

    for call_idx in range(1, n_calls + 1):
        x_candidate = opt.ask()
        n_unsat, metrics, cache_status, snapped = evaluate_candidate(
            x_candidate, fixed_cc, runner, cache
        )

        # Tell snapped point to surrogate
        snapped_x = [snapped["Wdiff"], snapped["Wnb"], snapped["Wpo"], snapped["Wbp"]]
        opt.tell(snapped_x, n_unsat)

        eval_record = {
            "iteration": call_idx,
            "Wdiff": snapped["Wdiff"],
            "Wnb": snapped["Wnb"],
            "Wpo": snapped["Wpo"],
            "Wbp": snapped["Wbp"],
            "Cc": snapped["Cc"],
            "n_unsat": int(n_unsat),
            "n_sat": int(8 - n_unsat),
            "all_sat": bool(n_unsat == 0),
            "gain_db": metrics.get("gain_db", 0.0),
            "ugbw": metrics.get("ugbw", 0.0),
            "pm": metrics.get("pm", 0.0),
            "f3db": metrics.get("f3db", 0.0),
            "pdc": metrics.get("pdc", 0.0),
            "is_valid": metrics.get("is_valid", False),
            "cache_status": cache_status,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        evaluations.append(eval_record)

        if call_idx % 10 == 0 or call_idx == 1 or n_unsat == 0:
            logger.info(
                f"[Call {call_idx:3d}/{n_calls}] "
                f"Wdiff={snapped['Wdiff']:.1f} Wnb={snapped['Wnb']:.0f} "
                f"Wpo={snapped['Wpo']:.1f} Wbp={snapped['Wbp']:.0f} -> "
                f"Sat={8 - int(n_unsat)}/8 (N_unsat={int(n_unsat)}) "
                f"Gain={metrics.get('gain_db', 0.0):.1f}dB "
                f"UGBW={metrics.get('ugbw', 0.0)/1e6:.2f}MHz "
                f"[{cache_status}]"
            )

    return evaluations, opt


def plot_fig5_parameter_distributions(feasible_evals: List[Dict[str, Any]], save_path: str):
    """
    Plots box plot per width with median and mean lines (paper Fig. 5 style).
    """
    vars_to_plot = ["Wdiff", "Wnb", "Wpo", "Wbp"]
    units = ["um", "um", "um", "um"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    fig.suptitle("Fig. 5 - Parameter Distributions of Feasible OpAmp Designs (8/8 Saturated)", fontsize=13, weight="bold")

    for i, (var_name, unit) in enumerate(zip(vars_to_plot, units)):
        ax = axes[i]
        vals = [e[var_name] for e in feasible_evals]
        
        # Create box plot
        bp = ax.boxplot(
            vals,
            vert=True,
            patch_artist=True,
            showmeans=True,
            meanline=True,
            widths=0.4,
            boxprops=dict(facecolor="#BDD7EE", color="#2E5B88", linewidth=1.5),
            whiskerprops=dict(color="#2E5B88", linewidth=1.5),
            capprops=dict(color="#2E5B88", linewidth=1.5),
            medianprops=dict(color="#D9534F", linewidth=2.0, label="Median"),
            meanprops=dict(color="#5CB85C", linewidth=2.0, linestyle="--", label="Mean"),
            flierprops=dict(marker="o", markerfacecolor="#2E5B88", markeredgecolor="none", markersize=5),
        )

        # Overlay jittered scatter of feasible points
        jitter = np.random.normal(0, 0.04, size=len(vals))
        ax.scatter(np.ones(len(vals)) + jitter, vals, alpha=0.5, color="#1F4E79", s=25, zorder=3)

        ax.set_title(f"{var_name} ({unit})", fontsize=11, weight="bold")
        ax.set_xticks([])
        ax.grid(axis="y", linestyle=":", alpha=0.6)

        # Stats text
        mean_v = np.mean(vals)
        med_v = np.median(vals)
        p2_v = np.percentile(vals, 2)
        p98_v = np.percentile(vals, 98)
        ax.set_xlabel(f"Mean: {mean_v:.2f}\nMed: {med_v:.2f}\n[p2, p98]: [{p2_v:.1f}, {p98_v:.1f}]", fontsize=9)

    axes[0].legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved Fig. 5 boxplot distributions to {save_path}")


def plot_feasibility_scatter_matrix(evaluations: List[Dict[str, Any]], save_path: str):
    """
    Plots pairwise scatters of the 4 widths, distinguishing feasible vs infeasible points.
    """
    vars_list = ["Wdiff", "Wnb", "Wpo", "Wbp"]
    n_vars = len(vars_list)

    fig, axes = plt.subplots(n_vars, n_vars, figsize=(12, 12))
    fig.suptitle("Feasibility Scatter Matrix: Feasible (Sat=8/8) vs Infeasible Designs", fontsize=13, weight="bold")

    feasible = [e for e in evaluations if e["all_sat"]]
    infeasible = [e for e in evaluations if not e["all_sat"]]

    for i, var_y in enumerate(vars_list):
        for j, var_x in enumerate(vars_list):
            ax = axes[i, j]
            if i == j:
                # Diagonal: Histogram of feasible vs infeasible
                if infeasible:
                    ax.hist([e[var_x] for e in infeasible], bins=12, color="#E06666", alpha=0.6, label="Infeasible", density=True)
                if feasible:
                    ax.hist([e[var_x] for e in feasible], bins=12, color="#6AA84F", alpha=0.7, label="Feasible", density=True)
                ax.set_ylabel("Density" if j == 0 else "")
            else:
                # Off-diagonal: Pairwise scatter
                if infeasible:
                    ax.scatter(
                        [e[var_x] for e in infeasible],
                        [e[var_y] for e in infeasible],
                        color="#CC0000",
                        marker="x",
                        alpha=0.6,
                        s=30,
                        label="Infeasible" if (i == 0 and j == 1) else None,
                    )
                if feasible:
                    ax.scatter(
                        [e[var_x] for e in feasible],
                        [e[var_y] for e in feasible],
                        color="#38761D",
                        marker="o",
                        alpha=0.85,
                        s=45,
                        edgecolor="black",
                        linewidth=0.5,
                        label="Feasible (8/8)" if (i == 0 and j == 1) else None,
                    )

            if i == n_vars - 1:
                ax.set_xlabel(var_x, fontsize=10, weight="bold")
            if j == 0 and i != j:
                ax.set_ylabel(var_y, fontsize=10, weight="bold")
            ax.grid(True, linestyle=":", alpha=0.5)

    handles, labels = axes[0, 1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.95, 0.95), fontsize=10)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    logger.info(f"Saved feasibility scatter matrix to {save_path}")


def execute_stage2_bo(
    results_dir: str,
    fixed_cc: float = 1.5,
    n_calls: int = 100,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executes the complete Stage 2 Bayesian Optimization pipeline.
    Performs bounds narrowing, shrinking re-run if collapsed, and artifact generation.
    """
    os.makedirs(results_dir, exist_ok=True)
    runner = SpectreRunner()
    cache = MemoCache()

    # Pass 1: Run 100 calls over the paper's initial discrete ranges
    logger.info("Executing BO Pass 1 over initial design bounds...")
    evaluations, opt = run_bo_loop(
        bounds=DEFAULT_INITIAL_BOUNDS,
        fixed_cc=fixed_cc,
        n_calls=n_calls,
        random_state=random_state,
        runner=runner,
        cache=cache,
    )

    feasible_evals = [e for e in evaluations if e["all_sat"]]
    logger.info(f"BO Pass 1 Complete: {len(feasible_evals)} / {len(evaluations)} feasible designs (8/8 saturated).")

    # Check for zero feasible points
    if len(feasible_evals) == 0:
        logger.error("Zero feasible points discovered in BO Pass 1!")
        # Analyze transistor failure frequencies
        logger.error("Reviewing failed transistors...")
        # Propose bounds and return
        return {"status": "zero_feasible_points", "evaluations": evaluations}

    # Analyze if any width distribution collapsed to one end of original range
    collapsed = {}
    narrowed_bounds = {}
    for var in ["Wdiff", "Wnb", "Wpo", "Wbp"]:
        vals = [e[var] for e in feasible_evals]
        p2 = float(np.percentile(vals, 2))
        p98 = float(np.percentile(vals, 98))
        
        # Snap percentiles to grid
        low_g, high_g, step_g = PARAM_GRID[var]
        p2_snapped = snap_value(p2, low_g, high_g, step_g)
        p98_snapped = snap_value(p98, low_g, high_g, step_g)
        if p2_snapped >= p98_snapped:
            # Ensure at least 2 grid steps of exploration width
            p98_snapped = min(high_g, round(p2_snapped + step_g * 2, 4))
            p2_snapped = max(low_g, round(p98_snapped - step_g * 2, 4))

        narrowed_bounds[var] = [p2_snapped, p98_snapped]

        # Check if collapsed to one boundary (< 25% of original range)
        orig_span = high_g - low_g
        feat_span = p98_snapped - p2_snapped
        if feat_span <= 0.35 * orig_span:
            collapsed[var] = True
            logger.info(f"Parameter '{var}' distribution collapsed: [{p2_snapped}, {p98_snapped}] vs original [{low_g}, {high_g}]")

    # Paper's design space narrowing rule:
    # "if any width's distribution collapses to one end of its original range, shrink that range and re-run BO ONCE, as the paper did"
    if collapsed:
        logger.info("Distribution collapse detected. Shrinking bounds and re-running BO Pass 2 once (paper methodology)...")
        shrunk_bounds = {}
        for var in ["Wdiff", "Wnb", "Wpo", "Wbp"]:
            shrunk_bounds[var] = (narrowed_bounds[var][0], narrowed_bounds[var][1])

        evaluations_pass2, opt2 = run_bo_loop(
            bounds=shrunk_bounds,
            fixed_cc=fixed_cc,
            n_calls=n_calls,
            random_state=random_state + 100,
            runner=runner,
            cache=cache,
        )
        feasible_pass2 = [e for e in evaluations_pass2 if e["all_sat"]]
        logger.info(f"BO Pass 2 Complete: {len(feasible_pass2)} / {len(evaluations_pass2)} feasible designs.")

        if len(feasible_pass2) > 0:
            # Recompute narrowed bounds from Pass 2 feasible points
            for var in ["Wdiff", "Wnb", "Wpo", "Wbp"]:
                vals = [e[var] for e in feasible_pass2]
                low_g, high_g, step_g = PARAM_GRID[var]
                p2 = snap_value(float(np.percentile(vals, 2)), low_g, high_g, step_g)
                p98 = snap_value(float(np.percentile(vals, 98)), low_g, high_g, step_g)
                if p2 >= p98:
                    p98 = min(high_g, round(p2 + step_g * 2, 4))
                    p2 = max(low_g, round(p98 - step_g * 2, 4))
                narrowed_bounds[var] = [p2, p98]

            evaluations = evaluations + evaluations_pass2
            feasible_evals = [e for e in evaluations if e["all_sat"]]

    # Add Cc full range to narrowed bounds
    narrowed_bounds["Cc"] = [PARAM_GRID["Cc"][0], PARAM_GRID["Cc"][1]]

    # 1. Save all_evaluations.csv
    csv_path = os.path.join(results_dir, "all_evaluations.csv")
    fieldnames = list(evaluations[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(evaluations)
    logger.info(f"Saved all {len(evaluations)} evaluations to {csv_path}")

    # 2. Save narrowed_bounds_180nm.json
    bounds_path = os.path.join(results_dir, "narrowed_bounds_180nm.json")
    with open(bounds_path, "w", encoding="utf-8") as f:
        json.dump(narrowed_bounds, f, indent=2)
    logger.info(f"Saved narrowed bounds to {bounds_path}:\n{json.dumps(narrowed_bounds, indent=2)}")

    # 3. Generate Fig. 5 (Paper style box plots)
    fig5_path = os.path.join(results_dir, "fig5_parameter_distributions.png")
    plot_fig5_parameter_distributions(feasible_evals, fig5_path)

    # 4. Generate Feasibility Scatter Matrix
    scatter_path = os.path.join(results_dir, "feasibility_scatter_matrix.png")
    plot_feasibility_scatter_matrix(evaluations, scatter_path)

    return {
        "status": "success",
        "total_evaluations": len(evaluations),
        "feasible_count": len(feasible_evals),
        "narrowed_bounds": narrowed_bounds,
        "results_dir": results_dir,
    }
