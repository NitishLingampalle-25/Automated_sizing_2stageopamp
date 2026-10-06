import subprocess, os, re, numpy as np

os.environ["PATH"] = "/home/install/SPECTRE211/bin:" + os.environ.get("PATH", "")
os.environ["CDS_LIC_FILE"] = "5280@cadence"
os.environ["LM_LICENSE_FILE"] = "5280@cadence"

def run_cc(cc_val):
    nl = f"""
simulator lang=spectre
include "/home/install/FOUNDRY/analog/180nm/models/spectre/gpdk.scs" section=NN
opt options soft_bin=allmodels save=allpub subcktprobelvl=2
Vdd (vdd 0) vsource dc=0.9
Vss (vss 0) vsource dc=-0.9
Vin_p (inp 0) vsource dc=0 mag=1
Vin_m (inm 0) vsource dc=0 mag=0

Ibias (vdd nbias) isource dc=30u
M8 (nbias nbias vss vss) nmos1 w=160u l=0.18u
M5 (tail nbias vss vss) nmos1 w=160u l=0.18u
M1 (net_m3 inm tail vss) nmos1 w=3.4u l=0.18u
M2 (net_m4 inp tail vss) nmos1 w=3.4u l=0.18u
M3 (net_m3 net_m3 vdd vdd) pmos1 w=10u l=0.18u
M4 (net_m4 net_m3 vdd vdd) pmos1 w=10u l=0.18u

M6 (vout net_m4 vdd vdd) pmos1 w=20u l=0.18u
M7 (vout nbias vss vss) nmos1 w=160u l=0.18u
Cc (net_m4 vout) capacitor c={cc_val}
CL (vout 0) capacitor c=10p

dcOp dc
opInfo info what=oppoint where=rawfile
acSwp ac start=1 stop=100M dec=10
"""
    raw_dir = f"scratch/raw_cc_{cc_val}"
    os.makedirs(raw_dir, exist_ok=True)
    scs_f = f"scratch/cc_{cc_val}.scs"
    with open(scs_f, "w") as f: f.write(nl)
    subprocess.run(["spectre", "-64", scs_f, "-format", "psfascii", "-raw", raw_dir, "+log", f"scratch/cc_{cc_val}.log"])
    
    with open(os.path.join(raw_dir, "acSwp.ac")) as f: text = f.read()
    data = []
    in_val, cur_freq = False, None
    for l in text.splitlines():
        if "VALUE" in l: in_val = True; continue
        if "END" in l: in_val = False
        if in_val:
            l = l.strip()
            if l.startswith('"freq"'): cur_freq = float(l.split()[1])
            elif l.startswith('"vout"'):
                m = re.search(r'\(([-+0-9.eE]+)\s+([-+0-9.eE]+)\)', l)
                if m and cur_freq is not None:
                    data.append((cur_freq, float(m.group(1)), float(m.group(2))))
    return data

d1 = run_cc("0.1p")
d2 = run_cc("3.0p")

print("Freq (Hz)  | Mag(0.1p) dB | Mag(3.0p) dB")
for (f1, r1, i1), (f2, r2, i2) in zip(d1[::3], d2[::3]):
    m1 = 20 * np.log10(np.abs(complex(r1, i1)))
    m2 = 20 * np.log10(np.abs(complex(r2, i2)))
    print(f"{f1:10.1e} | {m1:12.2f} | {m2:12.2f}")
