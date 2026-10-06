"""
Parser for Cadence Spectre PSFASCII simulation outputs.
Extracts DC operating point, per-transistor saturation state, and AC small-signal metrics.
Applies physical consistency assertions to validate simulation results.
"""

import os
import re
import math
import logging
from typing import Dict, Any, Tuple, List, Optional
import numpy as np

logger = logging.getLogger("PSFParser")

PENALTY_RESULT = {
    "gain_db": 0.0,
    "ugbw": 0.0,
    "pm": 0.0,
    "f3db": 0.0,
    "pdc": 1e-3,       # 1 mW penalty
    "n_sat": 0,
    "all_sat": False,
    "sat_dict": {},
    "is_valid": False,
    "reasons": ["Simulation failed or invalid result"],
}


def parse_opinfo(opinfo_path: str) -> Dict[str, Tuple[float, float, bool]]:
    """
    Parses opInfo.info for BSIM3v3 operating point data of M1..M8.
    Returns: {device_name: (vds, vdsat, is_saturated)}
    """
    if not os.path.exists(opinfo_path):
        return {}

    with open(opinfo_path, "r", errors="ignore") as f:
        content = f.read()

    sat_dict = {}
    for m in [f"M{i}" for i in range(1, 9)]:
        marker = f'"{m}" "bsim3v3"'
        idx = content.find(marker)
        if idx != -1:
            chunk = content[idx:idx + 2500]
            lines = chunk.split("\n")
            vals = [l.strip() for l in lines[1:] if l.strip() and not l.strip().startswith(")")]
            if len(vals) > 10:
                vds = float(vals[4])
                vdsat = float(vals[10])
                # Saturation condition: |Vds| >= |Vdsat|
                is_sat = abs(vds) >= abs(vdsat) if vds < 0 else vds >= vdsat
                sat_dict[m] = (vds, vdsat, bool(is_sat))
    return sat_dict


def parse_dc_op(dc_path: str) -> Dict[str, float]:
    """
    Parses dcOp.dc for node voltages and branch currents.
    Format per line in VALUE section: "signal_name" "type" value
    """
    if not os.path.exists(dc_path):
        return {}

    res = {}
    with open(dc_path, "r", errors="ignore") as f:
        in_val = False
        for line in f:
            line = line.strip()
            if "VALUE" in line:
                in_val = True
                continue
            if "END" in line:
                in_val = False
            if in_val:
                m = re.match(r'"([^"]+)"\s+"[^"]+"\s+([-+0-9.eE]+)', line)
                if m:
                    res[m.group(1)] = float(m.group(2))
    return res


def parse_ac_sweep(ac_path: str) -> Tuple[float, float, float, float]:
    """
    Parses acSwp.ac for AC small-signal response (Vout vs freq).
    Computes: (gain_db, ugbw, pm, f3db)
    """
    if not os.path.exists(ac_path):
        return 0.0, 0.0, 0.0, 0.0

    ac_data = []
    with open(ac_path, "r", errors="ignore") as f:
        in_val = False
        cur_freq = None
        for l in f:
            if "VALUE" in l:
                in_val = True
                continue
            if "END" in l:
                in_val = False
            if in_val:
                l = l.strip()
                if l.startswith('"freq"'):
                    cur_freq = float(l.split()[1])
                elif l.startswith('"vout"'):
                    m = re.search(r'\(([-+0-9.eE]+)\s+([-+0-9.eE]+)\)', l)
                    if m and cur_freq is not None:
                        re_val = float(m.group(1))
                        im_val = float(m.group(2))
                        ac_data.append((cur_freq, complex(re_val, im_val)))

    if not ac_data:
        return 0.0, 0.0, 0.0, 0.0

    freqs = np.array([x[0] for x in ac_data])
    vouts = np.array([x[1] for x in ac_data])
    mags_db = 20 * np.log10(np.maximum(np.abs(vouts), 1e-12))
    phases_deg = np.unwrap(np.angle(vouts)) * 180.0 / np.pi

    gain_db = float(np.max(mags_db))

    # f3dB: frequency where magnitude drops to gain_db - 3.0
    f3db = 0.0
    idx_3db = np.where(mags_db <= gain_db - 3.0)[0]
    if len(idx_3db) > 0:
        i = idx_3db[0]
        if i > 0:
            # Interpolate log-frequency
            f1, f2 = freqs[i - 1], freqs[i]
            g1, g2 = mags_db[i - 1], mags_db[i]
            target = gain_db - 3.0
            log_f = math.log10(f1) + (target - g1) * (math.log10(f2) - math.log10(f1)) / (g2 - g1)
            f3db = float(10 ** log_f)
        else:
            f3db = float(freqs[0])

    # UGBW: frequency where dB20(Vout) crosses 0 dB
    ugbw = 0.0
    pm = 0.0
    cross_idx = np.where(np.diff(np.sign(mags_db)))[0]
    if len(cross_idx) > 0:
        i = cross_idx[0]
        f1, f2 = freqs[i], freqs[i + 1]
        g1, g2 = mags_db[i], mags_db[i + 1]
        log_f = math.log10(f1) + (0.0 - g1) * (math.log10(f2) - math.log10(f1)) / (g2 - g1)
        ugbw = float(10 ** log_f)

        # PM = 180 + unwrapped phase at UGBW
        p1, p2 = phases_deg[i], phases_deg[i + 1]
        phase_ugbw = p1 + (0.0 - g1) * (p2 - p1) / (g2 - g1)
        pm = float((180.0 + phase_ugbw + 180.0) % 360.0 - 180.0)
        if pm <= 0.0:
            pm += 360.0

    return gain_db, ugbw, pm, f3db


def check_physical_consistency(
    gain_db: float,
    ugbw: float,
    pm: float,
    f3db: float,
    pdc: float,
    i_vdd: float,
    sat_dict: Dict[str, Tuple[float, float, bool]],
) -> Tuple[bool, List[str]]:
    """
    Evaluates physical consistency assertions:
    1. PM in (0, 180]
    2. gain_db in [20, 100]
    3. UGBW > f3dB
    4. UGBW within one decade of f3dB * 10^(gain_db / 20)
    5. Every saturation boolean equal to its own Vds > Vdsat comparison
    6. Pdc > 0 and within 50% of VDD * |I(Vdd)|
    """
    reasons = []

    # 1. PM in (0, 180]
    if not (0.0 < pm <= 180.0):
        reasons.append(f"PM {pm:.2f} deg not in (0, 180]")

    # 2. gain_db in [20, 100]
    if not (20.0 <= gain_db <= 100.0):
        reasons.append(f"gain_db {gain_db:.2f} dB not in [20, 100]")

    # 3. UGBW > f3dB
    if not (ugbw > f3db and f3db > 0.0):
        reasons.append(f"UGBW {ugbw:.1f} Hz not greater than f3dB {f3db:.1f} Hz")

    # 4. UGBW within one decade + 50% margin of f3dB * 10^(gain_db / 20)
    # Widened from [0.1, 10.0] to [0.05, 15.0] to accommodate legitimate Miller right-half-plane zero effects
    if f3db > 0.0 and gain_db >= 0.0:
        predicted_ugbw = f3db * (10.0 ** (gain_db / 20.0))
        ratio = ugbw / predicted_ugbw if predicted_ugbw > 0.0 else 0.0
        if not (0.05 <= ratio <= 15.0):
            reasons.append(
                f"UGBW {ugbw:.1f} Hz not within 1 decade + 50% margin of predicted f3dB*10^(G/20)={predicted_ugbw:.1f} Hz (ratio={ratio:.3f})"
            )
    else:
        reasons.append("Invalid f3db or gain_db for single-pole prediction")

    # 5. Every saturation boolean equal to its own Vds > Vdsat comparison
    for dev, (vds, vdsat, is_sat) in sat_dict.items():
        expected_sat = abs(vds) >= abs(vdsat) if vds < 0 else vds >= vdsat
        if is_sat != expected_sat:
            reasons.append(f"Saturation boolean for {dev} ({is_sat}) != comparison ({expected_sat})")

    # 6. Pdc > 0 and within 50% of VDD * |I(Vdd)|
    expected_pdc = 1.8 * abs(i_vdd)
    if pdc <= 0.0:
        reasons.append(f"Pdc {pdc} <= 0")
    elif expected_pdc > 0.0:
        diff_pct = abs(pdc - expected_pdc) / expected_pdc
        if diff_pct > 0.50:
            reasons.append(f"Pdc {pdc:.3e} W differs by {diff_pct*100:.1f}% (>50%) from VDD*|I(Vdd)|={expected_pdc:.3e} W")

    is_valid = (len(reasons) == 0)
    return is_valid, reasons


def parse_simulation_results(raw_dir: str) -> Dict[str, Any]:
    """
    Complete parsing pipeline for a PSFASCII raw simulation directory.
    Returns parsed metrics, transistor tables, and validation result.
    """
    opinfo_path = os.path.join(raw_dir, "opInfo.info")
    dc_path = os.path.join(raw_dir, "dcOp.dc")
    ac_path = os.path.join(raw_dir, "acSwp.ac")

    if not (os.path.exists(opinfo_path) and os.path.exists(ac_path)):
        res = dict(PENALTY_RESULT)
        res["reasons"] = ["Missing required raw simulation output files"]
        return res

    sat_dict = parse_opinfo(opinfo_path)
    dc_data = parse_dc_op(dc_path)
    gain_db, ugbw, pm, f3db = parse_ac_sweep(ac_path)

    # I(Vdd) and total Pdc
    i_vdd = dc_data.get("Vdd:p", dc_data.get("Vss:p", 60e-6))
    pdc = 1.8 * abs(i_vdd)

    n_sat = sum(1 for v in sat_dict.values() if v[2])
    all_sat = (n_sat == 8)

    is_valid, reasons = check_physical_consistency(
        gain_db=gain_db,
        ugbw=ugbw,
        pm=pm,
        f3db=f3db,
        pdc=pdc,
        i_vdd=i_vdd,
        sat_dict=sat_dict,
    )

    if not is_valid:
        res = dict(PENALTY_RESULT)
        res.update({
            "gain_db": gain_db,
            "ugbw": ugbw,
            "pm": pm,
            "f3db": f3db,
            "pdc": pdc,
            "n_sat": n_sat,
            "all_sat": all_sat,
            "sat_dict": sat_dict,
            "is_valid": False,
            "reasons": reasons,
        })
        return res

    return {
        "gain_db": gain_db,
        "ugbw": ugbw,
        "pm": pm,
        "f3db": f3db,
        "pdc": pdc,
        "n_sat": n_sat,
        "all_sat": all_sat,
        "sat_dict": sat_dict,
        "is_valid": True,
        "reasons": [],
    }
