"""
Two-stage Miller-compensated Operational Amplifier netlist generator.
Replication of Papageorgiou et al., AEU 2025 (doi: 10.1016/j.aeue.2025.155697).
180nm CMOS technology (gpdk180).

Design variables and paper's discrete grid ranges:
  Wdiff: [0.4, 6.4] um, step 0.2 um (NMOS differential pair M1, M2)
  Wnb:   [145, 175] um, step 1.0 um (NMOS bias branch M8, tail M5, active load M7)
  Wpo:   [0.5, 10.5] um, step 0.5 um (PMOS active load mirror M3, M4)
  Wbp:   [20, 40] um, step 1.0 um (PMOS common-source second stage driver M6)
  Cc:    [0.1, 3.0] pF, step 0.1 pF (Miller compensation capacitor)
Fixed channel lengths: L = 0.50 um for all transistors (documented parameter to achieve physical gain >= 70 dB).
Operating conditions: VDD = +0.9 V, VSS = -0.9 V, Io = 30 uA, CL = 10 pF.
"""

from typing import Dict, Any, Tuple
import numpy as np

# Paper's discrete design space specification: (min, max, step)
PARAM_GRID: Dict[str, Tuple[float, float, float]] = {
    "Wdiff": (0.4, 6.4, 0.2),
    "Wnb":   (145.0, 175.0, 1.0),
    "Wpo":   (0.5, 10.5, 0.5),
    "Wbp":   (20.0, 40.0, 1.0),
    "Cc":    (0.1, 3.0, 0.1),
}

MODEL_PATH = "/home/install/FOUNDRY/analog/180nm/models/spectre/gpdk.scs"
MODEL_SECTION = "NN"


def snap_value(val: float, low: float, high: float, step: float) -> float:
    """Snap a floating-point value to the nearest discrete grid point."""
    clamped = max(low, min(high, float(val)))
    snapped = low + round((clamped - low) / step) * step
    # Avoid floating-point representation artifacts
    return round(float(snapped), 4)


def snap_parameters(params: Dict[str, float]) -> Dict[str, float]:
    """Snap all proposed parameters to the paper's discrete grid."""
    snapped = {}
    for key, (low, high, step) in PARAM_GRID.items():
        if key in params:
            snapped[key] = snap_value(params[key], low, high, step)
        else:
            raise KeyError(f"Missing required parameter '{key}' in proposed params.")
    return snapped


def generate_netlist(params: Dict[str, float]) -> str:
    """
    Generate the Spectre netlist for the two-stage Miller-compensated OpAmp.
    Automatically snaps proposed parameters to the discrete grid.
    Computes accurate physical diffusion parameters (as, ad, ps, pd) based on
    transistor widths and process minimum diffusion length (0.36 um) to avoid
    the default 1 um^2 PDK subcircuit parasitics bug.
    """
    p = snap_parameters(params)
    wdiff = p["Wdiff"]
    wnb   = p["Wnb"]
    wpo   = p["Wpo"]
    wbp   = p["Wbp"]
    cc    = p["Cc"]

    netlist = f"""// Two-Stage Miller-Compensated OpAmp (Papageorgiou et al., AEU 2025)
// Discrete parameters: Wdiff={wdiff:.2f}u, Wnb={wnb:.1f}u, Wpo={wpo:.2f}u, Wbp={wbp:.1f}u, Cc={cc:.2f}p
simulator lang=spectre
include "{MODEL_PATH}" section={MODEL_SECTION}
opt options soft_bin=allmodels save=allpub subcktprobelvl=2

// DC Supply and Bias Sources
Vdd (vdd 0) vsource dc=0.9
Vss (vss 0) vsource dc=-0.9
Vin_p (inp 0) vsource dc=0 mag=1
Vin_m (inm 0) vsource dc=0 mag=0
Ibias (vdd nbias) isource dc=30u

// Bias branch & tail current source (NMOS, W=Wnb, L=0.50u)
M8 (nbias nbias vss vss) nmos1 w={wnb:.4f}u l=0.50u as={wnb:.4f}u*0.36u ad={wnb:.4f}u*0.36u ps=2*({wnb:.4f}u+0.36u) pd=2*({wnb:.4f}u+0.36u)
M5 (tail nbias vss vss) nmos1 w={wnb:.4f}u l=0.50u as={wnb:.4f}u*0.36u ad={wnb:.4f}u*0.36u ps=2*({wnb:.4f}u+0.36u) pd=2*({wnb:.4f}u+0.36u)

// First Stage: NMOS differential pair (W=Wdiff, L=0.50u)
M1 (net_m3 inm tail vss) nmos1 w={wdiff:.4f}u l=0.50u as={wdiff:.4f}u*0.36u ad={wdiff:.4f}u*0.36u ps=2*({wdiff:.4f}u+0.36u) pd=2*({wdiff:.4f}u+0.36u)
M2 (net_m4 inp tail vss) nmos1 w={wdiff:.4f}u l=0.50u as={wdiff:.4f}u*0.36u ad={wdiff:.4f}u*0.36u ps=2*({wdiff:.4f}u+0.36u) pd=2*({wdiff:.4f}u+0.36u)

// First Stage: PMOS active load mirror (W=Wpo, L=0.50u)
M3 (net_m3 net_m3 vdd vdd) pmos1 w={wpo:.4f}u l=0.50u as={wpo:.4f}u*0.36u ad={wpo:.4f}u*0.36u ps=2*({wpo:.4f}u+0.36u) pd=2*({wpo:.4f}u+0.36u)
M4 (net_m4 net_m3 vdd vdd) pmos1 w={wpo:.4f}u l=0.50u as={wpo:.4f}u*0.36u ad={wpo:.4f}u*0.36u ps=2*({wpo:.4f}u+0.36u) pd=2*({wpo:.4f}u+0.36u)

// Second Stage: PMOS common-source driver (W=Wbp, L=0.50u) & NMOS active load (W=Wnb, L=0.50u)
M6 (vout net_m4 vdd vdd) pmos1 w={wbp:.4f}u l=0.50u as={wbp:.4f}u*0.36u ad={wbp:.4f}u*0.36u ps=2*({wbp:.4f}u+0.36u) pd=2*({wbp:.4f}u+0.36u)
M7 (vout nbias vss vss) nmos1 w={wnb:.4f}u l=0.50u as={wnb:.4f}u*0.36u ad={wnb:.4f}u*0.36u ps=2*({wnb:.4f}u+0.36u) pd=2*({wnb:.4f}u+0.36u)

// Compensation & Load
Cc (net_m4 vout) capacitor c={cc:.4f}p
CL (vout 0) capacitor c=10p

// Analyses
dcOp dc
opInfo info what=oppoint where=rawfile
acSwp ac start=1 stop=100M dec=20
"""
    return netlist
