import subprocess
import os
import re
import numpy as np

os.environ["PATH"] = "/home/install/SPECTRE211/bin:" + os.environ.get("PATH", "")
os.environ["CDS_LIC_FILE"] = "5280@cadence"
os.environ["LM_LICENSE_FILE"] = "5280@cadence"

def run_test(w_load_name, w_tail_name, w_p2_name, w_n2_name, w_bias_name):
    # Fixed test values
    vals = {
        "Wdiff": 3.4,
        "Wnb": 160.0,
        "Wpo": 5.5,
        "Wbp": 30.0,
        "Cc": 1.5
    }
    
    w_diff = vals["Wdiff"]
    w_load = vals[w_load_name]
    w_tail = vals[w_tail_name]
    w_p2 = vals[w_p2_name]
    w_n2 = vals[w_n2_name]
    w_bias = vals[w_bias_name]
    cc = vals["Cc"]
    
    netlist = f"""
simulator lang=spectre
include "/home/install/FOUNDRY/analog/180nm/models/spectre/gpdk.scs" section=NN
opt options soft_bin=allmodels save=allpub subcktprobelvl=2
Vdd (vdd 0) vsource dc=0.9
Vss (vss 0) vsource dc=-0.9
Vin_p (inp 0) vsource dc=0 mag=1
Vin_m (inm 0) vsource dc=0 mag=0

Ibias (vdd nbias) isource dc=30u
M8 (nbias nbias vss vss) nmos1 w={w_bias}u l=0.18u
M5 (tail nbias vss vss) nmos1 w={w_tail}u l=0.18u
M1 (net_m3 inm tail vss) nmos1 w={w_diff}u l=0.18u
M2 (net_m4 inp tail vss) nmos1 w={w_diff}u l=0.18u
M3 (net_m3 net_m3 vdd vdd) pmos1 w={w_load}u l=0.18u
M4 (net_m4 net_m3 vdd vdd) pmos1 w={w_load}u l=0.18u

M6 (vout net_m4 vdd vdd) pmos1 w={w_p2}u l=0.18u
M7 (vout nbias vss vss) nmos1 w={w_n2}u l=0.18u
Cc (net_m4 vout) capacitor c={cc}p
CL (vout 0) capacitor c=10p

dcOp dc
opInfo info what=oppoint where=rawfile
acSwp ac start=1 stop=100M dec=20
"""
    raw_dir = "scratch/raw_assign"
    os.makedirs(raw_dir, exist_ok=True)
    scs_file = "scratch/assign.scs"
    with open(scs_file, "w") as f:
        f.write(netlist)
    cmd = ["spectre", "-64", scs_file, "-format", "psfascii", "-raw", raw_dir, "+log", "scratch/assign.log"]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0: return None
    
    # Read opInfo.info
    info_file = os.path.join(raw_dir, "opInfo.info")
    sat_dict = {}
    with open(info_file) as f:
        content = f.read()
    for m in [f"M{i}" for i in range(1, 9)]:
        idx = content.find(f'"{m}" "bsim3v3"')
        if idx != -1:
            chunk = content[idx:idx+2500]
            lines = chunk.split("\n")
            vals_l = [l.strip() for l in lines[1:] if not l.strip().startswith(")")]
            vds = float(vals_l[4])
            vdsat = float(vals_l[10])
            is_sat = abs(vds) >= abs(vdsat) and abs(vds) > 0.05
            sat_dict[m] = (vds, vdsat, is_sat)
            
    # Read acSwp.ac
    ac_file = os.path.join(raw_dir, "acSwp.ac")
    ac_data = []
    with open(ac_file) as f:
        cur_freq = None
        in_val = False
        for l in f:
            if "VALUE" in l: in_val = True; continue
            if "END" in l: in_val = False
            if in_val:
                l = l.strip()
                if l.startswith('"freq"'):
                    cur_freq = float(l.split()[1])
                elif l.startswith('"vout"'):
                    m = re.search(r'\(([-+0-9.eE]+)\s+([-+0-9.eE]+)\)', l)
                    if m and cur_freq is not None:
                        ac_data.append((cur_freq, complex(float(m.group(1)), float(m.group(2)))))
    
    gain_db, ugbw, pm = 0, 0, 0
    if ac_data:
        freqs = np.array([x[0] for x in ac_data])
        vouts = np.array([x[1] for x in ac_data])
        mags_db = 20 * np.log10(np.maximum(np.abs(vouts), 1e-12))
        gain_db = np.max(mags_db)
        phases_deg = np.unwrap(np.angle(vouts)) * 180 / np.pi
        cross_idx = np.where(np.diff(np.sign(mags_db)))[0]
        if len(cross_idx) > 0:
            i = cross_idx[0]
            f1, f2 = freqs[i], freqs[i+1]
            g1, g2 = mags_db[i], mags_db[i+1]
            ugbw = f1 + (0 - g1) * (f2 - f1) / (g2 - g1)
            p1, p2 = phases_deg[i], phases_deg[i+1]
            phase_ugbw = p1 + (0 - g1) * (p2 - p1) / (g2 - g1)
            pm = (180 + phase_ugbw + 180) % 360 - 180
            if pm < 0: pm += 360

    n_sat = sum(1 for v in sat_dict.values() if v[2])
    return n_sat, sat_dict, gain_db, ugbw, pm

# Candidate assignments:
# Which variable represents which transistor?
# Wdiff: M1, M2
# Variables for M34, M5, M6, M7, M8:
# We know Wnb is NMOS, Wpo is PMOS, Wbp is PMOS (or bias).
candidates = [
    # (w_load, w_tail, w_p2, w_n2, w_bias)
    ("Wbp", "Wnb", "Wpo", "Wnb", "Wnb"), # T1
    ("Wbp", "Wnb", "Wpo", "Wpo", "Wnb"),
    ("Wpo", "Wnb", "Wbp", "Wnb", "Wnb"),
    ("Wpo", "Wnb", "Wbp", "Wpo", "Wnb"),
    ("Wbp", "Wnb", "Wbp", "Wpo", "Wnb"),
    ("Wpo", "Wnb", "Wpo", "Wpo", "Wnb"),
    ("Wbp", "Wbp", "Wpo", "Wpo", "Wbp"),
]

for c in candidates:
    res = run_test(*c)
    if res:
        print(f"Assign {c} -> Sat={res[0]}/8, Gain={res[2]:.1f} dB, UGBW={res[3]/1e6:.3f} MHz, PM={res[4]:.1f} deg")
