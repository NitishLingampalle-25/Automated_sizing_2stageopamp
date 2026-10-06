import subprocess
import os
import re
import numpy as np

os.environ["PATH"] = "/home/install/SPECTRE211/bin:" + os.environ.get("PATH", "")
os.environ["CDS_LIC_FILE"] = "5280@cadence"
os.environ["LM_LICENSE_FILE"] = "5280@cadence"

def simulate(netlist, name):
    raw_dir = f"scratch/raw_{name}"
    os.makedirs(raw_dir, exist_ok=True)
    scs_file = f"scratch/{name}.scs"
    with open(scs_file, "w") as f:
        f.write(netlist)
    cmd = [
        "spectre", "-64", scs_file,
        "-format", "psfascii",
        "-raw", raw_dir,
        "+log", f"scratch/{name}.log"
    ]
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if p.returncode != 0:
        return None, "Spectre error"
    
    # Read dcOp.dc for node voltages
    dc_file = os.path.join(raw_dir, "dcOp.dc")
    voltages = {}
    if os.path.exists(dc_file):
        with open(dc_file) as f:
            lines = f.readlines()
        in_val = False
        for l in lines:
            if "VALUE" in l: in_val = True; continue
            if "END" in l: in_val = False
            if in_val:
                parts = l.strip().split()
                if len(parts) >= 3 and parts[1] == '"V"':
                    node = parts[0].strip('"')
                    voltages[node] = float(parts[2])
    
    # Read opInfo.info for transistor regions
    info_file = os.path.join(raw_dir, "opInfo.info")
    sat_count = 0
    sat_info = {}
    if os.path.exists(info_file):
        with open(info_file) as f:
            content = f.read()
        # Find M1..M8 entries
        for m in [f"M{i}" for i in range(1, 9)]:
            # Look for "M..." "bsim3v3" (
            idx = content.find(f'"{m}" "bsim3v3"')
            if idx != -1:
                chunk = content[idx:idx+2500]
                # Extract vds, vdsat, region
                lines = chunk.split("\n")
                # region is usually near the end of the tuple before PROP
                vals = []
                for cl in lines[1:]:
                    cl = cl.strip()
                    if cl.startswith(")"): break
                    vals.append(cl)
                # Parse vds, vdsat, region
                # From earlier inspection:
                # index 4: vds
                # index 10: vdsat
                try:
                    vds = float(vals[4])
                    vdsat = float(vals[10])
                    # region is usually around index 100 or so, let's find integer 2
                    is_sat = abs(vds) >= abs(vdsat) and abs(vds) > 0.05
                    sat_info[m] = (vds, vdsat, is_sat)
                    if is_sat: sat_count += 1
                except:
                    pass
    
    # Read acSwp.ac for gain and PM
    ac_file = os.path.join(raw_dir, "acSwp.ac")
    ac_data = []
    if os.path.exists(ac_file):
        with open(ac_file) as f:
            lines = f.readlines()
        in_val = False
        cur_freq = None
        for l in lines:
            if "VALUE" in l: in_val = True; continue
            if "END" in l: in_val = False
            if in_val:
                l = l.strip()
                if l.startswith('"freq"'):
                    cur_freq = float(l.split()[1])
                elif l.startswith('"vout"'):
                    m = re.search(r'\(([-+0-9.eE]+)\s+([-+0-9.eE]+)\)', l)
                    if m and cur_freq is not None:
                        re_v = float(m.group(1))
                        im_v = float(m.group(2))
                        ac_data.append((cur_freq, complex(re_v, im_v)))
    
    gain_db, ugbw, pm = None, None, None
    if ac_data:
        freqs = np.array([x[0] for x in ac_data])
        vouts = np.array([x[1] for x in ac_data])
        mags_db = 20 * np.log10(np.abs(vouts))
        phases_deg = np.unwrap(np.angle(vouts)) * 180 / np.pi
        gain_db = np.max(mags_db)
        
        # Find 0 dB crossing
        cross_idx = np.where(np.diff(np.sign(mags_db)))[0]
        if len(cross_idx) > 0:
            i = cross_idx[0]
            # Linear interpolate
            f1, f2 = freqs[i], freqs[i+1]
            g1, g2 = mags_db[i], mags_db[i+1]
            ugbw = f1 + (0 - g1) * (f2 - f1) / (g2 - g1)
            p1, p2 = phases_deg[i], phases_deg[i+1]
            phase_ugbw = p1 + (0 - g1) * (p2 - p1) / (g2 - g1)
            pm = 180 + phase_ugbw
            # Normalize PM to [0, 180]
            pm = (pm + 180) % 360 - 180
            if pm < 0: pm += 360
    
    return {
        "voltages": voltages,
        "sat_count": sat_count,
        "sat_info": sat_info,
        "gain_db": gain_db,
        "ugbw": ugbw,
        "pm": pm
    }, None

print("Simulate function ready")

# Candidate 1: Standard Razavi 2-stage opamp with NMOS bias
# But what if M7 is sized by Wnb or Wpo or Wbp?
# Let's test what happens if M8 is PMOS vs NMOS

# Let's test Candidate 1 with balanced widths:
# In Candidate 1: M8 is NMOS (Wnb), M5 is NMOS (Wnb), M7 is NMOS (Wnb)
# But why was it unbalanced?
# Let's check what Io and voltages are.

# Let's try Candidate 2: PMOS bias mirror branch
# Io = 30u from pbias to vss. M8 (pbias pbias vdd vdd) pmos1 w=Wbp.
# M6 is PMOS (w=Wpo) biased by pbias.
# M7 is NMOS (w=Wnb) driven by stage 1.
# But what is M5?

# Let's try Candidate 3:
# What if M8 is NMOS bias (w=Wnb), M5 is NMOS (w=Wnb),
# M3, M4 are PMOS (w=Wbp).
# What if M6 is PMOS (w=Wpo), but M7 is NMOS with width scaled or...
# Wait! What if M7 is NOT w=Wnb?
# What if the 4 widths are:
# Wdiff: M1, M2
# Wnb: NMOS tail (M5) and bias (M8)? What is M7?
# Could M7 be Wpo or Wbp?

netlists = {}

# Test 1: M8 is NMOS, M7 is Wpo, Wbp, etc.
netlists["T1_standard"] = """
simulator lang=spectre
include "/home/install/FOUNDRY/analog/180nm/models/spectre/gpdk.scs" section=NN
opt options soft_bin=allmodels save=allpub subcktprobelvl=2
Vdd (vdd 0) vsource dc=0.9
Vss (vss 0) vsource dc=-0.9
Vin_p (inp 0) vsource dc=0 mag=1
Vin_m (inm 0) vsource dc=0 mag=0

// Io = 30uA into diode-connected bias branch
Ibias (vdd nbias) isource dc=30u
M8 (nbias nbias vss vss) nmos1 w=160u l=0.18u
M5 (tail nbias vss vss) nmos1 w=160u l=0.18u
M1 (net_m3 inm tail vss) nmos1 w=3.4u l=0.18u
M2 (net_m4 inp tail vss) nmos1 w=3.4u l=0.18u
M3 (net_m3 net_m3 vdd vdd) pmos1 w=30u l=0.18u
M4 (net_m4 net_m3 vdd vdd) pmos1 w=30u l=0.18u

M6 (vout net_m4 vdd vdd) pmos1 w=5.5u l=0.18u
M7 (vout nbias vss vss) nmos1 w=160u l=0.18u
Cc (net_m4 vout) capacitor c=1.5p
CL (vout 0) capacitor c=10p

dcOp dc
opInfo info what=oppoint where=rawfile
acSwp ac start=1 stop=100M dec=20
"""

# Test 2: What if M6 is Wbp (30u) and M3, M4 is Wpo (5.5u)?
netlists["T2_M6_Wbp_M34_Wpo"] = """
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
M3 (net_m3 net_m3 vdd vdd) pmos1 w=5.5u l=0.18u
M4 (net_m4 net_m3 vdd vdd) pmos1 w=5.5u l=0.18u

M6 (vout net_m4 vdd vdd) pmos1 w=30u l=0.18u
M7 (vout nbias vss vss) nmos1 w=160u l=0.18u
Cc (net_m4 vout) capacitor c=1.5p
CL (vout 0) capacitor c=10p

dcOp dc
opInfo info what=oppoint where=rawfile
acSwp ac start=1 stop=100M dec=20
"""

# Test 3: What if M7 is Wpo?
netlists["T3_M7_Wpo"] = """
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
M3 (net_m3 net_m3 vdd vdd) pmos1 w=30u l=0.18u
M4 (net_m4 net_m3 vdd vdd) pmos1 w=30u l=0.18u

M6 (vout net_m4 vdd vdd) pmos1 w=30u l=0.18u
M7 (vout nbias vss vss) nmos1 w=160u l=0.18u
Cc (net_m4 vout) capacitor c=1.5p
CL (vout 0) capacitor c=10p

dcOp dc
opInfo info what=oppoint where=rawfile
acSwp ac start=1 stop=100M dec=20
"""

for name, nl in netlists.items():
    res, err = simulate(nl, name)
    print(f"=== {name} ===")
    if err:
        print("Error:", err)
    else:
        print(f"Vout: {res['voltages'].get('vout')}, Sat count: {res['sat_count']}")
        print(f"Gain: {res['gain_db']} dB, UGBW: {res['ugbw']} Hz, PM: {res['pm']} deg")
        for m, s in res['sat_info'].items():
            print(f"  {m}: Vds={s[0]:.4f}, Vdsat={s[1]:.4f}, Sat={s[2]}")
