import subprocess, os, re, numpy as np

os.environ["PATH"] = "/home/install/SPECTRE211/bin:" + os.environ.get("PATH", "")
os.environ["CDS_LIC_FILE"] = "5280@cadence"
os.environ["LM_LICENSE_FILE"] = "5280@cadence"

netlist = """
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
Cc (net_m4 vout) capacitor c=1.5p
CL (vout 0) capacitor c=10p

dcOp dc
opInfo info what=oppoint where=rawfile
acSwp ac start=1 stop=100M dec=10
"""
raw_dir = "scratch/raw_debug"
os.makedirs(raw_dir, exist_ok=True)
with open("scratch/debug.scs", "w") as f: f.write(netlist)
cmd = ["spectre", "-64", "scratch/debug.scs", "-format", "psfascii", "-raw", raw_dir, "+log", "scratch/debug.log"]
subprocess.run(cmd)

# Let us parse both vout and net_m4
with open(os.path.join(raw_dir, "acSwp.ac")) as f:
    text = f.read()

signals = {}
in_val = False
cur_freq = None
for l in text.splitlines():
    if "VALUE" in l: in_val = True; continue
    if "END" in l: in_val = False
    if in_val:
        l = l.strip()
        if l.startswith('"freq"'):
            cur_freq = float(l.split()[1])
            signals[cur_freq] = {}
        else:
            m = re.match(r'"([^"]+)"\s+\(([-+0-9.eE]+)\s+([-+0-9.eE]+)\)', l)
            if m and cur_freq is not None:
                sig_name = m.group(1)
                signals[cur_freq][sig_name] = complex(float(m.group(2)), float(m.group(3)))

print(f"{'Freq (Hz)':10s} | {'Gain net_m4 (dB)':16s} | {'Gain vout (dB)':16s} | {'Stage 2 Gain (dB)':18s}")
for f in [1, 10, 100, 1000, 10000, 100000, 1e6, 10e6, 100e6]:
    if f in signals:
        vm4 = signals[f].get("net_m4", 0)
        vout = signals[f].get("vout", 0)
        g_m4 = 20 * np.log10(np.abs(vm4)) if np.abs(vm4) > 0 else -999
        g_out = 20 * np.log10(np.abs(vout)) if np.abs(vout) > 0 else -999
        g_s2 = g_out - g_m4
        print(f"{f:10.0f} | {g_m4:16.2f} | {g_out:16.2f} | {g_s2:18.2f}")
