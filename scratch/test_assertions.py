import subprocess, os, re, numpy as np

# Let's check physical consistency assertions on a few points
# PM in (0, 180]
# gain_db in [20, 100]
# UGBW > f3dB
# UGBW within one decade of f3dB*10^(gain_db/20) (i.e. 0.1 <= UGBW / (f3dB * 10^(gain_db/20)) <= 10.0)

def check_consistency(gain_db, ugbw, pm, f3db):
    reasons = []
    if not (0 < pm <= 180):
        reasons.append(f"PM {pm:.1f} not in (0, 180]")
    if not (20 <= gain_db <= 100):
        reasons.append(f"gain_db {gain_db:.1f} not in [20, 100]")
    if not (ugbw > f3db):
        reasons.append(f"UGBW {ugbw} <= f3dB {f3db}")
    predicted_ugbw = f3db * (10 ** (gain_db / 20.0))
    ratio = ugbw / predicted_ugbw if predicted_ugbw > 0 else 0
    if not (0.1 <= ratio <= 10.0):
        reasons.append(f"UGBW {ugbw:.1f} not within 1 decade of f3dB*10^(G/20)={predicted_ugbw:.1f} (ratio={ratio:.3f})")
    return len(reasons) == 0, reasons

print("Consistency checker ready")
